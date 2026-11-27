//! OpenAI-compatible external [`Rewriter`] backend for Qwen-Image-2.1 prompt
//! enhancement (`t2i` text-only and `edit` multimodal with ordered images).
//!
//! Feeds the exact upstream system prompts, message ordering (`<image1>`…
//! `<imageN>` before text), sampling parameters, strict JSON parser, and
//! ratio-resolution contract owned by [`super`].

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use image::ImageFormat;
use omni_hub::providers::{
    ChatCompletionRequest, ChatContentPart, ChatMessage, ChatRole, ExternalProviderTarget,
    OpenAiCompatibleClient, ProviderError,
};

use super::{Content, RewriteOutcome, Rewriter, Role, TaskProfile, build_messages};
use crate::{Error, Result, reference};

/// Maximum edge for reference images sent to the OpenAI-compatible vision
/// enhancer, matching upstream `prompt_rewrite`'s 1024×1024 cap.
const MAX_REWRITE_IMAGE_EDGE: usize = 1024;

pub struct OpenAiRewriter {
    client: OpenAiCompatibleClient,
    model: String,
}

impl OpenAiRewriter {
    pub fn new(client: OpenAiCompatibleClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn from_target(target: &ExternalProviderTarget) -> Result<Self> {
        let client = target.client().map_err(|e| Error::Config(e.to_string()))?;
        Ok(Self::new(client, target.model.clone()))
    }
}

impl Rewriter for OpenAiRewriter {
    fn rewrite(
        &mut self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
    ) -> Result<RewriteOutcome> {
        let cancel = AtomicBool::new(false);
        self.rewrite_cancellable(profile, prompt, images, &cancel)
    }

    fn rewrite_cancellable(
        &mut self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
        cancel: &AtomicBool,
    ) -> Result<RewriteOutcome> {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let upstream_messages = build_messages(profile, prompt, images)?;
        let mut api_messages = Vec::with_capacity(upstream_messages.len());

        for msg in upstream_messages {
            let role = match msg.role {
                Role::System => ChatRole::System,
                Role::User => ChatRole::User,
            };
            let has_images = msg.content.iter().any(|c| matches!(c, Content::Image(_)));
            if !has_images {
                let text = msg
                    .content
                    .into_iter()
                    .filter_map(|c| match c {
                        Content::Text(s) => Some(s),
                        Content::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                api_messages.push(ChatMessage::text(role, text));
            } else {
                // Preserve exact upstream part order: <image1>..<imageN> before text.
                let mut parts = Vec::with_capacity(msg.content.len());
                for item in msg.content {
                    if cancel.load(Ordering::Acquire) {
                        return Err(Error::Cancelled);
                    }
                    match item {
                        Content::Image(path) => {
                            let url = encode_reference_image_data_url(&path, cancel)?;
                            parts.push(ChatContentPart::ImageUrl { url });
                        }
                        Content::Text(text) => {
                            parts.push(ChatContentPart::Text(text));
                        }
                    }
                }
                api_messages.push(ChatMessage::parts(role, parts));
            }
        }

        let sampling = profile.sampling();
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: api_messages,
            temperature: Some(sampling.temperature),
            top_p: Some(sampling.top_p),
            presence_penalty: Some(sampling.presence_penalty),
            max_tokens: Some(sampling.max_new_tokens),
            seed: Some(sampling.seed),
        };

        let response = self
            .client
            .chat_completions(&request, cancel)
            .map_err(|err| match err {
                ProviderError::Cancelled => Error::Cancelled,
                other => Error::Config(other.to_string()),
            })?;

        Ok(RewriteOutcome::from_decoded(profile, &response.content))
    }
}

fn encode_reference_image_data_url(path: &PathBuf, cancel: &AtomicBool) -> Result<String> {
    let img = reference::open(path)?;
    let (target_w, target_h) =
        reference::dimensions(img.width(), img.height(), MAX_REWRITE_IMAGE_EDGE)?;
    let resized = if target_w != img.width() as usize || target_h != img.height() as usize {
        reference::resize(&img, target_w, target_h, cancel)?
    } else {
        img
    };
    let mut png_bytes = Cursor::new(Vec::new());
    resized
        .write_to(&mut png_bytes, ImageFormat::Png)
        .map_err(|e| Error::Config(format!("rewrite：编码参考图 PNG 失败：{e}")))?;
    let b64 = BASE64_STANDARD.encode(png_bytes.into_inner());
    Ok(format!("data:image/png;base64,{b64}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite::{RatioDecision, resolve_canvas};
    use image::{Rgba, RgbaImage};
    use omni_hub::credentials::SecretString;
    use omni_hub::providers::ProviderProfile;
    use serde_json::Value;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn start_mock_chat_server(
        response_content: String,
    ) -> (
        String,
        Arc<Mutex<Vec<Value>>>,
        Arc<AtomicBool>,
        std::thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let bodies_clone = Arc::clone(&bodies);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let handle = std::thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let mut reader = BufReader::new(stream.try_clone().unwrap());
                        let mut req_line = String::new();
                        let _ = reader.read_line(&mut req_line);
                        let mut content_length = 0usize;
                        loop {
                            let mut line = String::new();
                            let _ = reader.read_line(&mut line);
                            let t = line.trim();
                            if t.is_empty() {
                                break;
                            }
                            if let Some((k, v)) = t.split_once(':')
                                && k.eq_ignore_ascii_case("content-length")
                            {
                                content_length = v.trim().parse().unwrap_or(0);
                            }
                        }
                        let mut body = vec![0u8; content_length];
                        if content_length > 0 {
                            let _ = reader.read_exact(&mut body);
                        }
                        if let Ok(json) = serde_json::from_slice::<Value>(&body) {
                            bodies_clone.lock().unwrap().push(json);
                        }
                        let payload = serde_json::json!({
                            "choices": [{
                                "message": { "role": "assistant", "content": response_content },
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
        (format!("http://{addr}/v1"), bodies, shutdown, handle)
    }

    #[test]
    fn openai_rewriter_t2i_uses_upstream_system_prompt_and_resolves_ratio() {
        let (base_url, bodies, shutdown, handle) = start_mock_chat_server(
            "<think>plan wide</think>{\"rewritten_prompt\":\"A cinematic 16:9 glass hall\",\"wh_ratio\":\"16:9\"}".to_owned(),
        );
        let mut profile = ProviderProfile::magpie_local();
        profile.base_url = base_url;
        let client = OpenAiCompatibleClient::new(profile, None, None, vec![]).unwrap();
        let mut rewriter = OpenAiRewriter::new(client, "qwen-pe");

        let outcome = rewriter
            .rewrite(TaskProfile::T2i, "玻璃音乐厅", &[])
            .unwrap();
        shutdown.store(true, Ordering::Release);
        let _ = handle.join();

        assert!(outcome.parse_ok);
        assert_eq!(outcome.positive_prompt, "A cinematic 16:9 glass hall");
        assert_eq!(outcome.ratio, RatioDecision::WhRatio(16, 9));
        assert_eq!(
            resolve_canvas(&outcome, &[], false).unwrap(),
            Some((2752, 1536))
        );
        assert_eq!(resolve_canvas(&outcome, &[], true).unwrap(), None);

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0]["model"], "qwen-pe");
        assert_eq!(
            recorded[0]["messages"][0]["content"],
            TaskProfile::T2i.system_prompt()
        );
        assert_eq!(recorded[0]["messages"][1]["content"], "玻璃音乐厅");
        assert_eq!(recorded[0]["presence_penalty"], 1.5);
    }

    #[test]
    fn openai_rewriter_edit_sends_ordered_images_before_text_and_follows_image_ratio() {
        let dir = tempfile::tempdir().unwrap();
        let img1_path = dir.path().join("first.png");
        let img2_path = dir.path().join("second.png");
        let img1 = RgbaImage::from_pixel(64, 64, Rgba([255, 0, 0, 255]));
        let img2 = RgbaImage::from_pixel(128, 64, Rgba([0, 0, 255, 255]));
        img1.save(&img1_path).unwrap();
        img2.save(&img2_path).unwrap();

        let (base_url, bodies, shutdown, handle) = start_mock_chat_server(
            r#"{"rewritten_prompt":"将<image1>放入<image2>背景","wh_ratio":"","ratio_follow":"<image2>"}"#.to_owned(),
        );
        let mut profile = ProviderProfile::custom("cloud", "Cloud", &base_url);
        profile.timeout_secs = 10;
        let client = OpenAiCompatibleClient::new(
            profile,
            Some(SecretString::new("sk-edit-test")),
            None,
            vec![],
        )
        .unwrap();
        let mut rewriter = OpenAiRewriter::new(client, "gpt-4o");

        let images = vec![img1_path.clone(), img2_path.clone()];
        let outcome = rewriter
            .rewrite(TaskProfile::Edit, "把图1放进图2", &images)
            .unwrap();
        shutdown.store(true, Ordering::Release);
        let _ = handle.join();

        assert!(outcome.parse_ok);
        assert_eq!(outcome.ratio, RatioDecision::FollowImage(2));
        let (w, h) = resolve_canvas(&outcome, &images, false).unwrap().unwrap();
        assert_eq!(w, h * 2, "must follow 2:1 ratio of <image2>");

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(
            recorded[0]["messages"][0]["content"],
            TaskProfile::Edit.system_prompt()
        );
        let parts = recorded[0]["messages"][1]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[2]["type"], "text");
        assert_eq!(parts[2]["text"], "把图1放进图2");
        let url1 = parts[0]["image_url"]["url"].as_str().unwrap();
        let url2 = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url1.starts_with("data:image/png;base64,"));
        assert!(url2.starts_with("data:image/png;base64,"));
        assert_ne!(url1, url2, "distinct images must preserve exact order");
    }
}
