//! HTTP Omni requests: uploaded bytes in, ordered events and a published WAV out.
use super::{Failure, ModelPaths, OmniBackend};
use crate::jobs::{Context, Output};
use base64::{Engine, engine::general_purpose::STANDARD};
use omni_lfm3::{
    generate::{EndReason, OmniEvent, OmniOptions, OutputMode},
    model::{Conversation, InputPart, OmniModel},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::HashMap, sync::atomic::Ordering};

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    mode: String,
    max_slots: usize,
    seed: u64,
    audio_temperature: Option<f32>,
    audio_top_k: Option<usize>,
    session_id: Option<String>,
    reset_session: bool,
    finish_session: bool,
}
impl Default for Request {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: None,
            mode: "auto".into(),
            max_slots: 256,
            seed: 0,
            audio_temperature: None,
            audio_top_k: None,
            session_id: None,
            reset_session: false,
            finish_session: false,
        }
    }
}
pub struct Job {
    messages: Vec<Value>,
    media: Vec<InputPart>,
    tools: Option<Vec<Value>>,
    options: OmniOptions,
    session_id: Option<String>,
    reset_session: bool,
    finish_session: bool,
}
pub struct LivePacket {
    pub samples: Vec<f32>,
    pub is_final: bool,
}
pub struct LiveJob {
    pub receiver: std::sync::mpsc::Receiver<LivePacket>,
    pub language: Option<String>,
    pub route: LiveRoute,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveRoute {
    Preview,
    Stable,
}
pub fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}
impl Request {
    pub fn into_job(self) -> Result<Job, String> {
        if self.messages.is_empty()
            || self.messages.len() > 256
            || !(1..=1500).contains(&self.max_slots)
        {
            return Err("invalid Omni messages or generation limit".into());
        }
        if self
            .audio_temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("speech temperature must be finite and non-negative".into());
        }
        if self
            .session_id
            .as_ref()
            .is_some_and(|id| !valid_session_id(id))
            || (self.session_id.is_none() && (self.reset_session || self.finish_session))
        {
            return Err("invalid Omni session ID or session options".into());
        }
        let mode = match self.mode.as_str() {
            "auto" => OutputMode::Auto,
            "text" => OutputMode::Text,
            "audio" => OutputMode::Audio,
            "interleaved" => OutputMode::Interleaved,
            _ => return Err("invalid Omni output mode".into()),
        };
        let mut messages = self.messages;
        let mut media = Vec::new();
        for message in &mut messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or("message lacks role")?
                .to_owned();
            if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
                return Err("invalid message role".into());
            }
            let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                let kind = part
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or("content part lacks type")?
                    .to_owned();
                if kind == "audio_out"
                    && (role != "assistant"
                        || part.get("codes").and_then(Value::as_array).is_none())
                {
                    return Err("assistant audio_out parts require speech codebook frames".into());
                }
                if matches!(kind.as_str(), "image" | "audio" | "video") {
                    if role != "user" {
                        return Err("uploaded media must belong to a user turn".into());
                    }
                    let object = part.as_object().ok_or("invalid media part")?;
                    if object.keys().any(|key| key != "type" && key != "data") {
                        return Err("media parts accept uploaded data only".into());
                    }
                    let encoded = object
                        .get("data")
                        .and_then(Value::as_str)
                        .ok_or("media part lacks base64 data")?;
                    let bytes = STANDARD
                        .decode(encoded)
                        .map_err(|_| "invalid base64 media data")?;
                    if bytes.is_empty() {
                        return Err("media data is empty".into());
                    }
                    media.push(match kind.as_str() {
                        "image" => InputPart::Image(bytes),
                        "audio" => InputPart::Audio(bytes),
                        _ => InputPart::Video(bytes),
                    });
                    *part = json!({"type":kind});
                }
            }
        }
        if media.len() > 64 {
            return Err("too many media parts".into());
        }
        Ok(Job {
            messages,
            media,
            tools: self.tools,
            options: OmniOptions {
                mode,
                max_slots: self.max_slots,
                seed: self.seed,
                audio_temperature: self.audio_temperature,
                audio_top_k: self.audio_top_k,
                ..Default::default()
            },
            session_id: self.session_id,
            reset_session: self.reset_session,
            finish_session: self.finish_session,
        })
    }
}

pub fn run(
    model: &OmniModel<OmniBackend>,
    job: Job,
    context: &Context,
    sessions: &mut HashMap<String, Conversation<OmniBackend>>,
) -> Result<Output, Failure> {
    let mut media = Vec::with_capacity(job.media.len());
    for input in job.media {
        if context.cancel().load(Ordering::Acquire) {
            return Err(Failure::Cancelled);
        }
        let (_, mut prepared) = model.prepare_parts(&[input], context.cancel())?;
        media.append(&mut prepared);
    }
    let mut temporary = Conversation::new(model);
    let session_id = job.session_id.clone();
    if session_id
        .as_ref()
        .is_some_and(|id| !sessions.contains_key(id))
        && sessions.len() >= 16
    {
        return Err(Failure::Error(
            "too many active Omni sessions; finish an existing session first".into(),
        ));
    }
    let conversation = match &session_id {
        Some(id) => sessions
            .entry(id.clone())
            .or_insert_with(|| Conversation::new(model)),
        None => &mut temporary,
    };
    if job.reset_session {
        conversation.clear();
    }
    let mut text_tokens = Vec::new();
    let mut event_error = None;
    let result=conversation.generate(model,&job.messages,job.tools.as_deref(),&media,&job.options,context.cancel(),|event| {
        let value=match event {
            OmniEvent::TextToken(token) => {
                text_tokens.push(*token);
                match model.tokenizer.decode(&text_tokens,true) {
                    Ok(text)=>json!({"type":"text","token":token,"text":text}),
                    Err(error)=>{ event_error=Some(error); return false; }
                }
            }
            OmniEvent::SpeechCodes(codes)=>json!({"type":"speech_codes","codes":codes}),
            OmniEvent::Pcm { samples,sample_rate }=>{
                let mut bytes=Vec::with_capacity(samples.len()*4);
                for sample in samples { bytes.extend_from_slice(&sample.to_le_bytes()); }
                json!({"type":"audio","encoding":"f32le-base64","sample_rate":sample_rate,"data":STANDARD.encode(bytes)})
            }
        };
        if !context.event(value) {
            event_error=Some(omni_r2t2::Error::Inference("Omni event buffer limit exceeded".into()));
            return false;
        }
        !context.cancel().load(Ordering::Acquire)
    });
    if job.finish_session
        && let Some(id) = &session_id
    {
        sessions.remove(id);
    }
    let result = result?;
    if let Some(error) = event_error {
        return Err(Failure::from(error));
    }
    let text = model.tokenizer.decode(&text_tokens, true)?;
    let cancelled = result.reason == EndReason::Cancelled;
    let reason = match result.reason {
        EndReason::Stop(_) => "stop",
        EndReason::Limit => "limit",
        EndReason::Cancelled => "cancelled",
    };
    let mut value = json!({"text":text,"tokens":result.tokens,"audio_frames":result.audio_frames,"sample_rate":model.codec.sample_rate(),
        "reason":reason,"partial":reason!="stop","session_id":session_id});
    let file = if result.samples.is_empty() {
        None
    } else {
        let name = format!("{}.wav", context.id());
        let file = context.file("wav", |path| {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: model.codec.sample_rate(),
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer =
                hound::WavWriter::create(path, spec).map_err(|e| Failure::Error(e.to_string()))?;
            for sample in result.samples {
                writer
                    .write_sample(sample)
                    .map_err(|e| Failure::Error(e.to_string()))?;
            }
            writer
                .finalize()
                .map_err(|e| Failure::Error(e.to_string()))?;
            Ok(())
        })?;
        value["audio_url"] = json!(format!("/v1/files/{name}"));
        Some(file)
    };
    let mut output = Output::new(value, cancelled);
    output.file = file;
    Ok(output)
}

pub fn load(paths: &ModelPaths, context: &Context) -> Result<OmniModel<OmniBackend>, Failure> {
    if context.cancel().load(Ordering::Acquire) {
        return Err(Failure::Cancelled);
    }
    context.progress(json!({"stage":"loading","message":"Loading LFM3 Omni"}));
    let model = OmniModel::load(&paths.lfm3, paths.precision, &Default::default())?;
    if context.cancel().load(Ordering::Acquire) {
        return Err(Failure::Cancelled);
    }
    Ok(model)
}

pub fn run_live(
    model: &OmniModel<OmniBackend>,
    job: LiveJob,
    context: &Context,
) -> Result<Output, Failure> {
    if job.route == LiveRoute::Stable {
        return run_stable_live(model, job, context);
    }
    use omni_lfm3::live::{LiveOptions, LiveTranscription};
    let mut live = LiveTranscription::new(model, &model.live_transcription_prompt()?)?;
    let options = LiveOptions::default();
    let mut tokens = Vec::new();
    let mut last_input = std::time::Instant::now();
    let mut reason = loop {
        if context.cancel().load(Ordering::Acquire) {
            break "cancelled";
        }
        let packet = match job
            .receiver
            .recv_timeout(std::time::Duration::from_millis(100))
        {
            Ok(packet) => packet,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                if last_input.elapsed() < std::time::Duration::from_secs(60) =>
            {
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(Failure::Error(
                    "live audio input timed out after 60 seconds".into(),
                ));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Failure::Error(
                    "live audio input closed without a final chunk".into(),
                ));
            }
        };
        last_input = std::time::Instant::now();
        let mut event_error = None;
        let output = match live.push_pcm(
            model,
            &packet.samples,
            16_000,
            packet.is_final,
            &options,
            context.cancel(),
            |token| {
                tokens.push(token);
                match model.tokenizer.decode(&tokens, true) {
                    Ok(text) => {
                        if !context.event(json!({"type":"text","token":token,"text":text})) {
                            event_error = Some(omni_r2t2::Error::Inference(
                                "Omni event buffer limit exceeded".into(),
                            ));
                            return false;
                        }
                    }
                    Err(error) => {
                        event_error = Some(error);
                        return false;
                    }
                }
                !context.cancel().load(Ordering::Acquire)
            },
        ) {
            Ok(output) => output,
            Err(omni_r2t2::Error::Interrupted) if context.cancel().load(Ordering::Acquire) => {
                break "cancelled";
            }
            Err(error) => return Err(error.into()),
        };
        if let Some(error) = event_error {
            return Err(error.into());
        }
        if output.cancelled {
            break "cancelled";
        }
        if packet.is_final {
            break if output.closed { "stop" } else { "limit" };
        }
    };
    let preview = model.tokenizer.decode(&tokens, true)?;
    let mut text = preview.clone();
    if reason != "cancelled"
        && let Some(language) = job.language
    {
        let mut final_tokens = Vec::new();
        let mut event_error = None;
        let result = model.transcribe_audio(
            live.into_audio()?,
            &language,
            512,
            context.cancel(),
            |token| {
                final_tokens.push(token);
                match model.tokenizer.decode(&final_tokens, true) {
                    Ok(text) => {
                        if !context.event(json!({"type":"final_text","token":token,"text":text})) {
                            event_error = Some(omni_r2t2::Error::Inference(
                                "Omni event buffer limit exceeded".into(),
                            ));
                            return false;
                        }
                    }
                    Err(error) => {
                        event_error = Some(error);
                        return false;
                    }
                }
                !context.cancel().load(Ordering::Acquire)
            },
        )?;
        if let Some(error) = event_error {
            return Err(error.into());
        }
        reason = match result.reason {
            EndReason::Stop(_) => "stop",
            EndReason::Limit => "limit",
            EndReason::Cancelled => "cancelled",
        };
        if !final_tokens.is_empty() {
            tokens = final_tokens;
            text = model.tokenizer.decode(&tokens, true)?;
        }
    }
    Ok(Output::new(
        json!({"text":text,"preview_text":preview,"tokens":tokens,"reason":reason,"partial":reason!="stop"}),
        reason == "cancelled",
    ))
}

fn run_stable_live(
    model: &OmniModel<OmniBackend>,
    job: LiveJob,
    context: &Context,
) -> Result<Output, Failure> {
    use omni_lfm3::stable::StableTranscription;
    let language = job.language.as_deref().unwrap_or("Chinese");
    let mut live = StableTranscription::new(model, language, 512)?;
    let mut last_input = std::time::Instant::now();
    let mut conflict = false;
    let reason = loop {
        if context.cancel().load(Ordering::Acquire) {
            break "cancelled";
        }
        let packet = match job
            .receiver
            .recv_timeout(std::time::Duration::from_millis(100))
        {
            Ok(packet) => packet,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                if last_input.elapsed() < std::time::Duration::from_secs(60) =>
            {
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(Failure::Error(
                    "live audio input timed out after 60 seconds".into(),
                ));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Failure::Error(
                    "live audio input closed without a final chunk".into(),
                ));
            }
        };
        last_input = std::time::Instant::now();
        let mut event_error = None;
        let output = live.push_pcm(
            model,
            &packet.samples,
            16_000,
            packet.is_final,
            context.cancel(),
            |event| {
                if !context.event(json!({"type":"text","delta":event.delta,"text":event.text,
                "is_final":event.is_final,"conflict":event.conflict}))
                {
                    event_error = Some(omni_r2t2::Error::Inference(
                        "Omni event buffer limit exceeded".into(),
                    ));
                    return false;
                }
                !context.cancel().load(Ordering::Acquire)
            },
        )?;
        if let Some(error) = event_error {
            return Err(error.into());
        }
        conflict |= output.conflict;
        if let Some(reason) = output.reason {
            break match reason {
                EndReason::Stop(_) => "stop",
                EndReason::Limit => "limit",
                EndReason::Cancelled => "cancelled",
            };
        }
    };
    Ok(Output::new(
        json!({"text":live.text(),"reason":reason,"partial":reason!="stop",
        "route":"stable","conflict":conflict}),
        reason == "cancelled",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sessions_validate_ids_and_speech_history_requires_codes() {
        let request = |value| serde_json::from_value::<Request>(value).unwrap().into_job();
        assert!(
            request(
                json!({"messages":[{"role":"user","content":"hi"}],"session_id":"conversation_1"})
            )
            .is_ok()
        );
        assert!(
            request(json!({"messages":[{"role":"user","content":"hi"}],"session_id":"../bad"}))
                .is_err()
        );
        assert!(
            request(json!({"messages":[{"role":"user","content":"hi"}],"reset_session":true}))
                .is_err()
        );
        assert!(
            request(json!({"messages":[{"role":"assistant","content":[{"type":"audio_out"}]}]}))
                .is_err()
        );
        assert!(request(json!({"messages":[{"role":"assistant","content":[{"type":"audio_out","codes":[1,2]}]}]})).is_ok());
    }

    #[test]
    fn rejects_invalid_speech_temperature_before_job_admission() {
        let request = |value| serde_json::from_value::<Request>(value).unwrap().into_job();
        assert!(
            request(json!({"messages":[{"role":"user","content":"hi"}],"audio_temperature":-0.5}))
                .is_err()
        );
        assert!(request(json!({"messages":[{"role":"user","content":"hi"}],"audio_temperature":0.0,"audio_top_k":0})).is_ok());
        let job = request(json!({"messages":[{"role":"user","content":"hi"}],"audio_temperature":0.4,"audio_top_k":12})).unwrap();
        assert_eq!(job.options.audio_temperature, Some(0.4));
        assert_eq!(job.options.audio_top_k, Some(12));
    }
}
