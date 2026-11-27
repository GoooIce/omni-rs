use crate::engine_harness::{Adapter, Context, Event, Lifecycle};
use crate::gpu::{self, GpuBackend as Backend};
use omni_hub::providers::{
    ChatCompletionRequest, ChatMessage, ChatRole, ExternalProviderTarget, ProviderError,
};
use omni_hy_mt::{Error, Model, Request, Translator};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub enum Command {
    Load(PathBuf),
    Translate {
        job_id: u64,
        request: Request,
    },
    TranslateExternal {
        job_id: u64,
        request: Request,
        target: ExternalProviderTarget,
    },
}
#[derive(serde::Serialize)]
pub struct Update {
    pub kind: &'static str,
    pub message: String,
    pub text: String,
    pub job_id: u64,
}
impl Update {
    fn status(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            text: String::new(),
            job_id: 0,
        }
    }
}
impl Event for Update {
    fn lifecycle(kind: Lifecycle, message: String, _progress: f64) -> Self {
        Self::status(kind.as_str(), message)
    }
    fn trace(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
pub struct Worker {
    translator: Option<Translator<Backend>>,
}
pub type Engine = crate::engine_harness::Engine<Worker>;
impl Adapter for Worker {
    type Command = Command;
    type Update = Update;
    type Error = Error;
    const NAME: &'static str = "translation";
    fn new() -> Self {
        Self { translator: None }
    }
    fn unload(&mut self) {
        self.translator = None;
    }
    fn load_path(command: &Command) -> Option<&Path> {
        match command {
            Command::Load(path) => Some(path),
            _ => None,
        }
    }
    fn check_files(path: &Path) -> Result<(), String> {
        crate::model_location::check_files(path)
    }
    fn is_cancelled(error: &Error) -> bool {
        matches!(error, Error::Interrupted)
    }
    fn load(&mut self, path: &Path, context: &Context<Update>) -> omni_hy_mt::Result<Update> {
        let model = Model::from_directory(
            path,
            gpu::default_precision(),
            &gpu::gpu_device(),
            &context.stop,
            |n, total| {
                context.loading(
                    n as f64 / total as f64,
                    format!("正在加载 Hy-MT2-7B · {n}/{total}"),
                );
            },
        )?;
        self.translator = Some(Translator::new(model, 4096));
        Ok(Update::status("ready", "Hy-MT2-7B 已就绪"))
    }
    fn run(&mut self, command: Command, context: &Context<Update>) -> omni_hy_mt::Result<Update> {
        match command {
            Command::Load(_) => unreachable!("load handled by harness"),
            Command::Translate { job_id, request } => {
                let translator = self
                    .translator
                    .as_mut()
                    .ok_or_else(|| Error::Config("请先载入翻译模型".into()))?;
                let output =
                    pollster::block_on(translator.translate(&request, &context.stop, |event| {
                        context.updates.publish(Update {
                            kind: "translation",
                            job_id,
                            text: event.text,
                            message: "正在翻译…".into(),
                        });
                    }))?;
                Ok(Update {
                    kind: if output.finish_reason == "cancelled" {
                        "cancelled"
                    } else {
                        "done"
                    },
                    job_id,
                    text: output.text,
                    message: format!(
                        "{} · {:.2} 秒 · 复用 {} 个前缀 token",
                        match output.finish_reason.as_str() {
                            "cancelled" => "已停止翻译",
                            "length" => "已达输出上限，译文未完整结束",
                            _ => "翻译完成",
                        },
                        output.compute_ms / 1000.0,
                        output.cached_tokens
                    ),
                })
            }
            Command::TranslateExternal {
                job_id,
                request,
                target,
            } => run_external_translation(job_id, &request, &target, &context.stop),
        }
    }
}

/// Build the exact Hy-MT target-language prompt and dispatch via OpenAI
/// `/v1/chat/completions`, preserving target-language semantics and validation.
pub fn run_external_translation(
    job_id: u64,
    request: &Request,
    target: &ExternalProviderTarget,
    cancel: &std::sync::atomic::AtomicBool,
) -> omni_hy_mt::Result<Update> {
    request.validate()?;
    let start = Instant::now();
    let prompt = format!(
        "Translate the following text into {}. Note that you should only output the translated result without any additional explanation:\n\n{}",
        request.target_language.trim(),
        request.text
    );
    let client = target.client().map_err(|e| Error::Config(e.to_string()))?;
    let resp = client
        .chat_completions(
            &ChatCompletionRequest {
                model: target.model.clone(),
                messages: vec![ChatMessage::text(ChatRole::User, prompt)],
                temperature: Some(0.2),
                top_p: Some(0.95),
                presence_penalty: None,
                max_tokens: Some(request.options.max_new_tokens.max(512)),
                seed: None,
            },
            cancel,
        )
        .map_err(|err| match err {
            ProviderError::Cancelled => Error::Interrupted,
            other => Error::Config(other.to_string()),
        })?;
    let translated = omni_breeze_tts::enhancement::without_thinking(&resp.content);
    Ok(Update {
        kind: "done",
        job_id,
        text: translated,
        message: format!(
            "外部翻译完成 · {:.2} 秒 · {}",
            start.elapsed().as_secs_f64(),
            target.display_badge()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_hub::providers::ProviderProfile;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn external_translation_preserves_target_language_prompt_and_validation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let captured = Arc::new(Mutex::new(String::new()));
        let captured_clone = Arc::clone(&captured);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let handle = std::thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let mut reader = BufReader::new(stream.try_clone().unwrap());
                        let mut line = String::new();
                        let _ = reader.read_line(&mut line);
                        let mut len = 0usize;
                        loop {
                            let mut h = String::new();
                            let _ = reader.read_line(&mut h);
                            let t = h.trim();
                            if t.is_empty() {
                                break;
                            }
                            if let Some((k, v)) = t.split_once(':') {
                                if k.eq_ignore_ascii_case("content-length") {
                                    len = v.trim().parse().unwrap_or(0);
                                }
                            }
                        }
                        let mut body = vec![0u8; len];
                        if len > 0 {
                            let _ = reader.read_exact(&mut body);
                        }
                        *captured_clone.lock().unwrap() =
                            String::from_utf8_lossy(&body).into_owned();
                        let payload = serde_json::json!({
                            "choices": [{
                                "message": {"role": "assistant", "content": "こんにちは世界"},
                                "finish_reason": "stop"
                            }]
                        })
                        .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        let _ = stream.flush();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        let mut profile = ProviderProfile::magpie_local();
        profile.base_url = format!("http://{addr}/v1");
        let target = ExternalProviderTarget {
            profile,
            api_key: None,
            proxy_url: None,
            model: "qwen-mt".to_owned(),
            voice: None,
            speed: None,
            response_format: None,
            known_secrets: vec![],
        };
        let cancel = AtomicBool::new(false);
        let update = run_external_translation(
            7,
            &Request {
                text: "你好世界".to_owned(),
                target_language: "Japanese".to_owned(),
                options: omni_hy_mt::Options::default(),
            },
            &target,
            &cancel,
        )
        .unwrap();
        shutdown.store(true, Ordering::Release);
        let _ = handle.join();

        assert_eq!(update.kind, "done");
        assert_eq!(update.job_id, 7);
        assert_eq!(update.text, "こんにちは世界");
        assert!(captured
            .lock()
            .unwrap()
            .contains("Translate the following text into Japanese."));
    }
}
