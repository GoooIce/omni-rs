//! Ordered expansion of media placeholders, including a video's soundtrack clock.
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// Prepared media counts in the same order as placeholders in the conversation.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MediaSpec {
    Image {
        tokens: usize,
    },
    Audio {
        tokens: usize,
    },
    Video {
        tokens_per_frame: usize,
        frame_indices: Vec<usize>,
        fps: f64,
        temporal_patch_size: usize,
        audio_tokens: Option<usize>,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Image,
    Audio,
    VideoFrame { frame: usize },
    VideoAudio { offset: usize },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub media_index: usize,
    pub kind: SegmentKind,
    pub tokens: usize,
}
pub struct ExpandedMedia {
    pub text: String,
    pub segments: Vec<Segment>,
}
#[derive(Clone, Debug)]
pub struct MediaTokens {
    pub image: String,
    pub audio: String,
    pub video: String,
    pub start: String,
    pub end: String,
    pub vision_markers: bool,
    pub audio_markers: bool,
    pub seconds_per_audio_token: f64,
    pub max_tokens: usize,
}
impl Default for MediaTokens {
    fn default() -> Self {
        Self {
            image: "<|image|>".into(),
            audio: "<|audio|>".into(),
            video: "<|video|>".into(),
            start: "<|vision_start|>".into(),
            end: "<|vision_end|>".into(),
            vision_markers: true,
            audio_markers: true,
            seconds_per_audio_token: 0.08,
            max_tokens: 128_000,
        }
    }
}
#[derive(Deserialize)]
struct Processor {
    image_token: String,
    audio_token: String,
    video_token: String,
    vision_start_token: String,
    vision_end_token: String,
    vision_markers: bool,
    audio_markers: bool,
    feature_extractor: FeatureConfig,
}
#[derive(Deserialize)]
struct FeatureConfig {
    hop_length: usize,
    encoder_stride: usize,
    sampling_rate: usize,
}
#[derive(Deserialize)]
struct ModelConfig {
    text_config: TextLimit,
}
#[derive(Deserialize)]
struct TextLimit {
    max_position_embeddings: usize,
}
impl MediaTokens {
    pub fn load(directory: &Path) -> Result<Self> {
        let p: Processor =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        let c: ModelConfig =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if p.feature_extractor.sampling_rate == 0 {
            return Err(Error::Config("invalid audio sample rate".into()));
        }
        let value = Self {
            image: p.image_token,
            audio: p.audio_token,
            video: p.video_token,
            start: p.vision_start_token,
            end: p.vision_end_token,
            vision_markers: p.vision_markers,
            audio_markers: p.audio_markers,
            seconds_per_audio_token: p.feature_extractor.hop_length as f64
                * p.feature_extractor.encoder_stride as f64
                / p.feature_extractor.sampling_rate as f64,
            max_tokens: c.text_config.max_position_embeddings,
        };
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<()> {
        if [
            &self.image,
            &self.audio,
            &self.video,
            &self.start,
            &self.end,
        ]
        .iter()
        .any(|s| s.is_empty())
            || self.image == self.audio
            || self.image == self.video
            || self.audio == self.video
            || !self.seconds_per_audio_token.is_finite()
            || self.seconds_per_audio_token <= 0.0
            || self.max_tokens == 0
        {
            return Err(Error::Config("invalid media token configuration".into()));
        }
        Ok(())
    }
    fn wrap(&self, text: &str, enabled: bool) -> String {
        if enabled {
            format!("{}{text}{}", self.start, self.end)
        } else {
            text.into()
        }
    }
}

/// Expand rendered chat-template placeholders and retain their exact media order.
pub fn expand(prompt: &str, media: &[MediaSpec], t: &MediaTokens) -> Result<ExpandedMedia> {
    t.validate()?;
    let mut prompt = prompt.to_owned();
    for token in [&t.image, &t.audio, &t.video] {
        prompt = prompt.replace(&format!("{}{token}{}", t.start, t.end), token);
    }
    let mut text = String::new();
    let mut segments = Vec::new();
    let mut remaining = prompt.as_str();
    let mut index = 0;
    let mut total = 0usize;
    loop {
        let found = [(&t.image, 0), (&t.audio, 1), (&t.video, 2)]
            .into_iter()
            .filter_map(|(token, kind)| remaining.find(token).map(|pos| (pos, token, kind)))
            .min_by_key(|v| v.0);
        let Some((position, token, kind)) = found else {
            text.push_str(remaining);
            break;
        };
        text.push_str(&remaining[..position]);
        remaining = &remaining[position + token.len()..];
        let spec = media.get(index).ok_or_else(|| {
            Error::Inference("media placeholder has no corresponding input".into())
        })?;
        let mut add = |kind: SegmentKind, count: usize| -> Result<()> {
            if count == 0 {
                return Ok(());
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| Error::Inference("media token count overflow".into()))?;
            if total > t.max_tokens {
                return Err(Error::Inference("media exceeds context limit".into()));
            }
            segments.push(Segment {
                media_index: index,
                kind,
                tokens: count,
            });
            Ok(())
        };
        let replacement = match (kind, spec) {
            (0, MediaSpec::Image { tokens }) if *tokens > 0 => {
                add(SegmentKind::Image, *tokens)?;
                t.wrap(&t.image.repeat(*tokens), t.vision_markers)
            }
            (1, MediaSpec::Audio { tokens }) if *tokens > 0 => {
                add(SegmentKind::Audio, *tokens)?;
                t.wrap(&t.audio.repeat(*tokens), t.audio_markers)
            }
            (
                2,
                MediaSpec::Video {
                    tokens_per_frame,
                    frame_indices,
                    fps,
                    temporal_patch_size,
                    audio_tokens,
                },
            ) => {
                if *tokens_per_frame == 0
                    || *temporal_patch_size == 0
                    || frame_indices.is_empty()
                    || !fps.is_finite()
                    || *fps <= 0.0
                    || frame_indices.windows(2).any(|w| w[0] > w[1])
                    || frame_indices.iter().any(|i| *i > (1usize << 53))
                {
                    return Err(Error::Inference("invalid video frame clock".into()));
                }
                let fps = (fps * 1_000_000.0).round_ties_even() / 1_000_000.0;
                if fps <= 0.0 {
                    return Err(Error::Inference("invalid video frame rate".into()));
                }
                let groups = frame_indices.len().div_ceil(*temporal_patch_size);
                let mut bounds = vec![0usize];
                if let Some(count) = audio_tokens {
                    for group in 1..groups {
                        let start = frame_indices[group * temporal_patch_size] as f64 / fps;
                        let boundary = (start / t.seconds_per_audio_token)
                            .round_ties_even()
                            .max(0.0) as usize;
                        bounds.push(
                            boundary
                                .min(*count)
                                .max(*bounds.last().expect("initial boundary")),
                        );
                    }
                    bounds.push(*count);
                }
                let mut body = String::new();
                for group in 0..groups {
                    let first = frame_indices[group * temporal_patch_size];
                    let last = frame_indices
                        [((group + 1) * temporal_patch_size - 1).min(frame_indices.len() - 1)];
                    let timestamp = (first as f64 + last as f64) / (2.0 * fps);
                    body.push_str(&format!("<{timestamp:.1} seconds>"));
                    add(SegmentKind::VideoFrame { frame: group }, *tokens_per_frame)?;
                    body.push_str(&t.wrap(&t.video.repeat(*tokens_per_frame), t.vision_markers));
                    if audio_tokens.is_some() {
                        let offset = bounds[group];
                        let count = bounds[group + 1] - offset;
                        add(SegmentKind::VideoAudio { offset }, count)?;
                        body.push_str(&t.audio.repeat(count));
                    }
                }
                t.wrap(&body, t.vision_markers)
            }
            _ => {
                return Err(Error::Inference(
                    "media input order or placeholder type does not match".into(),
                ));
            }
        };
        text.push_str(&replacement);
        index += 1;
    }
    if index != media.len() {
        return Err(Error::Inference(
            "media input has no corresponding placeholder".into(),
        ));
    }
    Ok(ExpandedMedia { text, segments })
}
