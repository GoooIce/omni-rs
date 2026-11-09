use burn::tensor::backend::Backend;
use clap::{Parser, ValueEnum};
use omni_lfm3::{
    Precision,
    generate::{EndReason, OmniEvent, OmniOptions, OutputMode},
    model::{InputPart, OmniModel},
};
use serde_json::{Value, json};
use std::{
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Auto,
    Text,
    Audio,
    Interleaved,
}
impl From<Mode> for OutputMode {
    fn from(value: Mode) -> Self {
        match value {
            Mode::Auto => Self::Auto,
            Mode::Text => Self::Text,
            Mode::Audio => Self::Audio,
            Mode::Interleaved => Self::Interleaved,
        }
    }
}
#[derive(Clone, Copy, ValueEnum)]
enum BackendChoice {
    Metal,
    Cpu,
}
#[derive(Clone, Copy, ValueEnum)]
enum LiveRoute {
    Preview,
    Stable,
}
#[derive(Parser)]
#[command(version, about = "Local LFM3-Omni inference using Rust and Burn")]
struct Args {
    /// Directory containing the official checkpoint and audio_tokenizer.
    #[arg(long)]
    model: PathBuf,
    /// Ordered part: text:TEXT, image:PATH, audio:PATH or video:PATH. Repeat for mixed input.
    #[arg(long = "part")]
    parts: Vec<String>,
    /// Previous messages as JSON; media content parts may contain a local `path`.
    #[arg(long)]
    conversation: Option<PathBuf>,
    /// Tool definitions as a JSON array.
    #[arg(long)]
    tools: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "auto")]
    mode: Mode,
    #[arg(long, default_value_t = 256)]
    max_slots: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Speech sampling temperature; defaults to the checkpoint setting, 0 is greedy.
    #[arg(long, allow_hyphen_values = true, value_parser = parse_audio_temperature, conflicts_with = "live_pcm")]
    audio_temperature: Option<f32>,
    /// Restrict speech sampling to the top K codes; 0 disables the restriction.
    #[arg(long, conflicts_with = "live_pcm")]
    audio_top_k: Option<usize>,
    /// Save generated speech as a new 24 kHz WAV file.
    #[arg(long)]
    audio_output: Option<PathBuf>,
    /// Select a speech adapter that passed the frozen development gate.
    #[arg(long)]
    speech_adapter: Option<PathBuf>,
    #[arg(long, value_enum)]
    backend: Option<BackendChoice>,
    #[arg(long, default_value = "f32")]
    precision: String,
    /// Return text, token IDs, audio codes and result status as JSON.
    #[arg(long)]
    json: bool,
    /// Transcribe a stream of little-endian f32 mono PCM at 16 kHz from stdin.
    #[arg(long)]
    live_pcm: bool,
    /// Explicitly queue normally completed live speech for local learning.
    #[arg(long, requires = "live_pcm")]
    learn_state: Option<PathBuf>,
    /// Live transcription route: preview with optional final correction, or stable committed text.
    #[arg(long, value_enum, requires = "live_pcm")]
    live_route: Option<LiveRoute>,
    /// Maximum text tokens emitted after each live audio chunk.
    #[arg(long, default_value_t = 48)]
    live_block_tokens: usize,
    /// Refine the completed live recording using an English language name, e.g. Chinese.
    #[arg(long, requires = "live_pcm")]
    live_language: Option<String>,
}
fn parse_audio_temperature(value: &str) -> Result<f32, String> {
    let temperature: f32 = value
        .parse()
        .map_err(|_| "speech temperature must be a number")?;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("speech temperature must be finite and non-negative".into());
    }
    Ok(temperature)
}
fn main() {
    if let Err(error) = execute() {
        eprintln!("lfm3-omni: {error}");
        std::process::exit(1);
    }
}
fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let precision = match args.precision.as_str() {
        "f32" => Precision::F32,
        "f16" => Precision::F16,
        _ => return Err("precision must be f32 or f16".into()),
    };
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Relaxed))?;
    let _foreground = args
        .learn_state
        .as_deref()
        .map(|state| {
            omni_lfm3::speech_inbox::ForegroundLease::acquire(
                state,
                &stop,
                std::time::Duration::from_secs(30),
            )
        })
        .transpose()?;
    let backend = args.backend.unwrap_or(if cfg!(feature = "metal") {
        BackendChoice::Metal
    } else {
        BackendChoice::Cpu
    });
    match backend {
        BackendChoice::Metal => {
            #[cfg(feature = "metal")]
            {
                run::<omni_lfm3::NativeGpu>(
                    &args,
                    precision,
                    &burn::backend::wgpu::WgpuDevice::default(),
                    &stop,
                )
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("binary was built without the metal feature".into())
            }
        }
        BackendChoice::Cpu => {
            #[cfg(feature = "cpu")]
            {
                run::<burn::backend::NdArray<f32>>(&args, precision, &Default::default(), &stop)
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err("binary was built without the cpu feature".into())
            }
        }
    }
}
fn parse_part(value: &str) -> Result<InputPart, Box<dyn std::error::Error>> {
    let (kind, value) = value
        .split_once(':')
        .ok_or("parts must have the form TYPE:VALUE")?;
    Ok(match kind {
        "text" => InputPart::Text(value.into()),
        "image" => InputPart::Image(std::fs::read(value)?),
        "audio" => InputPart::Audio(std::fs::read(value)?),
        "video" => InputPart::Video(std::fs::read(value)?),
        _ => return Err(format!("unsupported part type {kind:?}").into()),
    })
}
fn prepare_history<B: Backend>(
    model: &OmniModel<B>,
    messages: &mut [Value],
    stop: &AtomicBool,
) -> Result<Vec<omni_lfm3::prepared::PreparedMedia<B>>, Box<dyn std::error::Error>> {
    let mut all = Vec::new();
    for message in messages {
        let Some(parts) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        if !parts.iter().any(|p| {
            matches!(
                p.get("type").and_then(Value::as_str),
                Some("image" | "audio" | "video")
            )
        }) {
            continue;
        }
        if message.get("role").and_then(Value::as_str) != Some("user") {
            return Err("media in conversation history must belong to a user message".into());
        }
        let mut input = Vec::new();
        for part in parts {
            let kind = part
                .get("type")
                .and_then(Value::as_str)
                .ok_or("conversation part lacks type")?;
            match kind {
                "text" => input.push(InputPart::Text(
                    part.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                )),
                "image" | "audio" | "video" => {
                    let path = part
                        .get("path")
                        .and_then(Value::as_str)
                        .ok_or("conversation media part lacks path")?;
                    input.push(parse_part(&format!("{kind}:{path}"))?);
                }
                _ => return Err(format!("unsupported conversation part {kind:?}").into()),
            }
        }
        let (user, mut media) = model.prepare_parts(&input, stop)?;
        message["content"] = user["content"].clone();
        all.append(&mut media);
    }
    Ok(all)
}
fn save_wav(path: &Path, samples: &[f32], rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::new(BufWriter::new(temp.as_file_mut()), spec)?;
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }
    temp.persist_noclobber(path).map_err(|e| e.error)?;
    Ok(())
}
fn run<B: Backend>(
    args: &Args,
    precision: Precision,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut messages: Vec<Value> = match &args.conversation {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Vec::new(),
    };
    let parts = args
        .parts
        .iter()
        .map(|part| parse_part(part))
        .collect::<Result<Vec<_>, _>>()?;
    if !args.live_pcm && messages.is_empty() && parts.is_empty() {
        return Err("provide --part or --conversation".into());
    }
    if args.live_pcm
        && (!messages.is_empty()
            || !parts.is_empty()
            || args.tools.is_some()
            || args.audio_output.is_some())
    {
        return Err(
            "--live-pcm cannot be combined with conversation, parts, tools or audio output".into(),
        );
    }
    let tools: Option<Vec<Value>> = args
        .tools
        .as_ref()
        .map(|path| -> Result<_, Box<dyn std::error::Error>> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let mut model = OmniModel::<B>::load(&args.model, precision, device)?;
    if let Some(directory) = &args.speech_adapter {
        let apply = || -> omni_lfm3::Result<_> {
            let fingerprint = omni_lfm3::speech_adapter::source_fingerprint(&args.model)?;
            omni_lfm3::speech_adapter::VerifiedAdapter::load(
                directory,
                &fingerprint,
                model.text.hidden_size(),
            )
        };
        match apply().and_then(|adapter| {
            model
                .text
                .install_conv_lora(adapter.layer, adapter.rank, &adapter.a, &adapter.b)
        }) {
            Ok(()) => eprintln!("speech adapter enabled for this session"),
            Err(error) => {
                eprintln!("speech adapter rejected; continuing with frozen base: {error}")
            }
        }
    }
    if args.live_pcm {
        return run_live(&model, args, stop);
    }
    let mut media = prepare_history(&model, &mut messages, stop)?;
    if !parts.is_empty() {
        let (message, mut new_media) = model.prepare_parts(&parts, stop)?;
        messages.push(message);
        media.append(&mut new_media);
    }
    let options = OmniOptions {
        mode: args.mode.into(),
        max_slots: args.max_slots,
        seed: args.seed,
        audio_temperature: args.audio_temperature,
        audio_top_k: args.audio_top_k,
        ..Default::default()
    };
    let mut cache = model.new_text_cache();
    let mut text_tokens = Vec::new();
    let mut printed = String::new();
    let mut output_error = None;
    let result = model.generate_omni(
        &messages,
        tools.as_deref(),
        &media,
        &mut cache,
        &options,
        stop,
        |event| {
            if let OmniEvent::TextToken(token) = event {
                text_tokens.push(*token);
                if !args.json {
                    match model.tokenizer.decode(&text_tokens, true) {
                        Ok(text) if !text.contains('\u{fffd}') && text.starts_with(&printed) => {
                            let delta = &text[printed.len()..];
                            if let Err(e) = std::io::stdout()
                                .write_all(delta.as_bytes())
                                .and_then(|_| std::io::stdout().flush())
                            {
                                output_error = Some(e);
                                return false;
                            }
                            printed = text;
                        }
                        Ok(_) => (),
                        Err(e) => {
                            output_error = Some(std::io::Error::other(e.to_string()));
                            return false;
                        }
                    }
                }
            }
            true
        },
    )?;
    if let Some(error) = output_error {
        return Err(error.into());
    }
    let text = model.tokenizer.decode(&text_tokens, true)?;
    let audio_path = if result.samples.is_empty() {
        None
    } else {
        let path = args
            .audio_output
            .clone()
            .unwrap_or_else(|| PathBuf::from("lfm3-output.wav"));
        save_wav(&path, &result.samples, model.codec.sample_rate())?;
        Some(path)
    };
    let reason = match result.reason {
        EndReason::Stop(_) => "stop",
        EndReason::Limit => "limit",
        EndReason::Cancelled => "cancelled",
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string(
                &json!({"text":text,"tokens":result.tokens,"audio_frames":result.audio_frames,
            "audio_path":audio_path,"sample_rate":model.codec.sample_rate(),"reason":reason})
            )?
        );
    } else {
        if text.starts_with(&printed) {
            print!("{}", &text[printed.len()..]);
        }
        println!();
        if let Some(path) = audio_path {
            eprintln!("Audio: {}", path.display());
        }
        if reason != "stop" {
            eprintln!("Finished: {reason}");
        }
    }
    Ok(())
}

enum PcmInput {
    Samples(Vec<f32>),
    End,
    Error(String),
}
fn read_pcm(mut input: impl Read, sender: std::sync::mpsc::SyncSender<PcmInput>) {
    let mut bytes = vec![0_u8; 16_000 * 4];
    loop {
        let mut filled = 0;
        while filled < bytes.len() {
            match input.read(&mut bytes[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    let _ = sender.send(PcmInput::Error(error.to_string()));
                    return;
                }
            }
        }
        if filled % 4 != 0 {
            let _ = sender.send(PcmInput::Error(
                "PCM byte length must be a multiple of four".into(),
            ));
            return;
        }
        if filled > 0 {
            let samples = bytes[..filled]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            if sender.send(PcmInput::Samples(samples)).is_err() {
                return;
            }
        }
        if filled < bytes.len() {
            let _ = sender.send(PcmInput::End);
            return;
        }
    }
}
fn run_live<B: Backend>(
    model: &OmniModel<B>,
    args: &Args,
    stop: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    use omni_lfm3::live::{LiveOptions, LiveTranscription};
    if !matches!(args.mode, Mode::Auto | Mode::Text) {
        return Err("--live-pcm produces transcription text; use auto or text mode".into());
    }
    if matches!(args.live_route, Some(LiveRoute::Stable)) {
        return run_stable_live(model, args, stop);
    }
    let mut live = LiveTranscription::new(model, &model.live_transcription_prompt()?)?;
    let mut learning = args
        .learn_state
        .as_ref()
        .map(|_| omni_lfm3::speech_inbox::SpeechCapture::new());
    let options = LiveOptions {
        max_new_tokens: args.live_block_tokens,
        ..Default::default()
    };
    let (sender, receiver) = std::sync::mpsc::sync_channel(2);
    std::thread::spawn(move || read_pcm(std::io::stdin().lock(), sender));
    let mut tokens = Vec::new();
    let mut printed = String::new();
    let mut output_error = None;
    let mut reason = loop {
        if stop.load(Ordering::Relaxed) {
            break "cancelled";
        }
        let chunk = match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(chunk) => chunk,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("PCM input reader stopped unexpectedly".into());
            }
        };
        let (samples, is_final) = match chunk {
            PcmInput::Samples(samples) => {
                if let Some(capture) = &mut learning {
                    capture.push(&samples);
                }
                (samples, false)
            }
            PcmInput::End => (Vec::new(), true),
            PcmInput::Error(error) => return Err(error.into()),
        };
        let output = live.push_pcm(model, &samples, 16_000, is_final, &options, stop, |token| {
            tokens.push(token);
            if !args.json {
                match model.tokenizer.decode(&tokens, true) {
                    Ok(text) if !text.contains('\u{fffd}') && text.starts_with(&printed) => {
                        if let Err(error) = std::io::stdout()
                            .write_all(text[printed.len()..].as_bytes())
                            .and_then(|_| std::io::stdout().flush())
                        {
                            output_error = Some(error);
                            return false;
                        }
                        printed = text;
                    }
                    Ok(_) => (),
                    Err(error) => {
                        output_error = Some(std::io::Error::other(error.to_string()));
                        return false;
                    }
                }
            }
            true
        });
        if let Some(error) = output_error.take() {
            return Err(error.into());
        }
        let output = match output {
            Ok(output) => output,
            Err(omni_lfm3::Error::Interrupted) if stop.load(Ordering::Relaxed) => break "cancelled",
            Err(error) => return Err(error.into()),
        };
        if output.cancelled {
            break "cancelled";
        }
        if is_final {
            break if output.closed { "stop" } else { "limit" };
        }
    };
    let preview = model.tokenizer.decode(&tokens, true)?;
    if !args.json && preview.starts_with(&printed) {
        print!("{}", &preview[printed.len()..]);
    }
    let mut text = preview.clone();
    if reason != "cancelled"
        && let Some(language) = &args.live_language
    {
        let result =
            model.transcribe_audio(live.into_audio()?, language, args.max_slots, stop, |_| true)?;
        reason = match result.reason {
            EndReason::Stop(_) => "stop",
            EndReason::Limit => "limit",
            EndReason::Cancelled => "cancelled",
        };
        if !result.tokens.is_empty() {
            tokens = result.tokens;
            text = model.tokenizer.decode(&tokens, true)?;
        }
        if !args.json {
            println!("\n[final] {text}");
        }
    } else if !args.json {
        println!();
    }
    let learning = finish_learning(learning, args.learn_state.as_deref(), reason, stop);
    if args.json {
        let mut result =
            json!({"text":text,"preview_text":preview,"tokens":tokens,"reason":reason});
        if let Some(learning) = learning {
            result["learning"] = learning;
        }
        println!("{}", serde_json::to_string(&result)?);
    } else if let Some(learning) = learning {
        eprintln!("learning: {learning}");
    }
    Ok(())
}

fn run_stable_live<B: Backend>(
    model: &OmniModel<B>,
    args: &Args,
    stop: &AtomicBool,
) -> Result<(), Box<dyn std::error::Error>> {
    use omni_lfm3::stable::StableTranscription;
    let language = args.live_language.as_deref().unwrap_or("Chinese");
    let mut live = StableTranscription::new(model, language, args.max_slots)?;
    let mut learning = args
        .learn_state
        .as_ref()
        .map(|_| omni_lfm3::speech_inbox::SpeechCapture::new());
    let (sender, receiver) = std::sync::mpsc::sync_channel(2);
    std::thread::spawn(move || read_pcm(std::io::stdin().lock(), sender));
    let mut output_error = None;
    let mut conflict = false;
    let reason = loop {
        if stop.load(Ordering::Relaxed) {
            break "cancelled";
        }
        let input = match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(input) => input,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("PCM input reader stopped unexpectedly".into());
            }
        };
        let (samples, is_final) = match input {
            PcmInput::Samples(samples) => {
                if let Some(capture) = &mut learning {
                    capture.push(&samples);
                }
                (samples, false)
            }
            PcmInput::End => (Vec::new(), true),
            PcmInput::Error(error) => return Err(error.into()),
        };
        let output = live.push_pcm(model, &samples, 16_000, is_final, stop, |event| {
            if !args.json && !event.delta.is_empty() {
                if let Err(error) = std::io::stdout()
                    .write_all(event.delta.as_bytes())
                    .and_then(|_| std::io::stdout().flush())
                {
                    output_error = Some(error);
                    return false;
                }
            }
            !stop.load(Ordering::Acquire)
        })?;
        if let Some(error) = output_error.take() {
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
    let learning = finish_learning(learning, args.learn_state.as_deref(), reason, stop);
    if args.json {
        let mut result =
            json!({"text":live.text(),"route":"stable","reason":reason,"conflict":conflict});
        if let Some(learning) = learning {
            result["learning"] = learning;
        }
        println!("{}", serde_json::to_string(&result)?);
    } else {
        println!();
        if conflict {
            eprintln!("已保留先前确认文字，最终识别存在差异");
        }
        if reason != "stop" {
            eprintln!("Finished: {reason}");
        }
        if let Some(learning) = learning {
            eprintln!("learning: {learning}");
        }
    }
    Ok(())
}

fn finish_learning(
    capture: Option<omni_lfm3::speech_inbox::SpeechCapture>,
    state: Option<&Path>,
    reason: &str,
    stop: &AtomicBool,
) -> Option<Value> {
    use omni_lfm3::speech_inbox::{Completion, SpeechInbox};
    let (capture, state) = (capture?, state?.to_owned());
    if reason == "limit" {
        return Some(json!({"state":"rejected", "reason":"transcription reached its token limit"}));
    }
    let completion = if reason == "stop" && !stop.load(Ordering::Acquire) {
        Completion::Normal
    } else {
        Completion::Cancelled
    };
    let result = std::thread::spawn(move || {
        let inbox = SpeechInbox::open(state)?;
        capture.finish(&inbox, completion)
    })
    .join();
    Some(match result {
        Ok(Ok(receipt)) => serde_json::to_value(receipt)
            .unwrap_or_else(|error| json!({"state":"error", "reason":error.to_string()})),
        Ok(Err(error)) => json!({"state":"error", "reason":error.to_string()}),
        Err(_) => json!({"state":"error", "reason":"speech inbox worker panicked"}),
    })
}

#[cfg(test)]
mod live_tests {
    use super::*;
    #[test]
    fn live_route_is_an_explicit_opt_in() {
        let parse = |extra: &[&str]| {
            let mut args = vec!["lfm3-omni", "--model", "unused"];
            args.extend_from_slice(extra);
            Args::try_parse_from(args)
        };
        assert!(parse(&[]).unwrap().live_route.is_none());
        assert!(matches!(
            parse(&["--live-pcm", "--live-route", "stable"])
                .unwrap()
                .live_route,
            Some(LiveRoute::Stable)
        ));
        assert!(parse(&["--live-route", "stable"]).is_err());
        assert!(parse(&["--live-pcm", "--live-route", "unknown"]).is_err());
    }
    #[test]
    fn speech_sampling_flags_validate_and_conflict_with_live_transcription() {
        let parse = |extra: &[&str]| {
            let mut args = vec!["lfm3-omni", "--model", "unused"];
            args.extend_from_slice(extra);
            Args::try_parse_from(args)
        };
        let args = parse(&["--audio-temperature", "0.4", "--audio-top-k", "32"]).unwrap();
        assert_eq!(args.audio_temperature, Some(0.4));
        assert_eq!(args.audio_top_k, Some(32));
        assert!(parse(&["--audio-temperature", "0", "--audio-top-k", "0"]).is_ok());
        assert!(parse(&["--audio-temperature", "-0.5"]).is_err());
        assert!(parse(&["--audio-temperature", "NaN"]).is_err());
        assert!(parse(&["--live-pcm", "--audio-top-k", "3"]).is_err());
    }
    #[test]
    fn reads_complete_pcm_chunks_and_rejects_partial_samples() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(3);
        let bytes = [0.25_f32.to_le_bytes(), (-0.5_f32).to_le_bytes()].concat();
        read_pcm(std::io::Cursor::new(bytes), sender);
        match receiver.recv().unwrap() {
            PcmInput::Samples(samples) => assert_eq!(samples, [0.25, -0.5]),
            _ => panic!("expected samples"),
        }
        assert!(matches!(receiver.recv().unwrap(), PcmInput::End));
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        read_pcm(std::io::Cursor::new(vec![0_u8; 3]), sender);
        assert!(matches!(receiver.recv().unwrap(), PcmInput::Error(_)));
    }
}
