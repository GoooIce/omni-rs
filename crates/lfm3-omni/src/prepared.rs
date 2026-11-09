//! Bind encoded media to exactly the placeholder tokens that name it.
use crate::media::{MediaSpec, Segment, SegmentKind};
use burn::tensor::{Tensor, backend::Backend};
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use std::path::Path;

pub enum PreparedMedia<B: Backend> {
    Image(Tensor<B, 2>),
    Audio(Tensor<B, 2>),
    Video {
        frames: Vec<Tensor<B, 2>>,
        soundtrack: Option<Tensor<B, 2>>,
        frame_indices: Vec<usize>,
        fps: f64,
        temporal_patch_size: usize,
    },
}
impl<B: Backend> PreparedMedia<B> {
    pub fn spec(&self) -> Result<MediaSpec> {
        match self {
            Self::Image(x) => Ok(MediaSpec::Image {
                tokens: x.dims()[0],
            }),
            Self::Audio(x) => Ok(MediaSpec::Audio {
                tokens: x.dims()[0],
            }),
            Self::Video {
                frames,
                soundtrack,
                frame_indices,
                fps,
                temporal_patch_size,
            } => {
                if *temporal_patch_size == 0
                    || frames.is_empty()
                    || frame_indices.len().div_ceil(*temporal_patch_size) != frames.len()
                    || frames.iter().any(|f| f.dims() != frames[0].dims())
                {
                    return Err(Error::Inference(
                        "video frames do not match their clock or shapes".into(),
                    ));
                }
                Ok(MediaSpec::Video {
                    tokens_per_frame: frames[0].dims()[0],
                    frame_indices: frame_indices.clone(),
                    fps: *fps,
                    temporal_patch_size: *temporal_patch_size,
                    audio_tokens: soundtrack.as_ref().map(|x| x.dims()[0]),
                })
            }
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct MediaIds {
    #[serde(rename = "image_token_id")]
    pub image: u32,
    #[serde(rename = "audio_token_id")]
    pub audio: u32,
    #[serde(rename = "video_token_id")]
    pub video: u32,
}
impl MediaIds {
    pub fn load(directory: &Path) -> Result<Self> {
        let ids: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if ids.image == ids.audio || ids.image == ids.video || ids.audio == ids.video {
            return Err(Error::Config("media token IDs are not distinct".into()));
        }
        Ok(ids)
    }
}
/// Replace only placeholder rows, preserving every text embedding and media ordering.
pub fn bind_embeddings<B: Backend>(
    ids: &[u32],
    mut embeddings: Tensor<B, 3>,
    segments: &[Segment],
    media: &[PreparedMedia<B>],
    tokens: MediaIds,
) -> Result<Tensor<B, 3>> {
    let [batch, n, width] = embeddings.dims();
    if batch != 1
        || n != ids.len()
        || tokens.image == tokens.audio
        || tokens.image == tokens.video
        || tokens.audio == tokens.video
    {
        return Err(Error::Inference(
            "invalid prompt embedding shape or media token IDs".into(),
        ));
    }
    let mut pos = 0;
    let mut next = 0;
    while pos < ids.len() {
        let id = ids[pos];
        if id != tokens.image && id != tokens.audio && id != tokens.video {
            pos += 1;
            continue;
        }
        let segment = segments
            .get(next)
            .ok_or_else(|| Error::Inference("unbound media placeholder".into()))?;
        let item = media
            .get(segment.media_index)
            .ok_or_else(|| Error::Inference("missing media embedding".into()))?;
        let (wanted, value) = match (&segment.kind, item) {
            (SegmentKind::Image, PreparedMedia::Image(x)) => (tokens.image, x.clone()),
            (SegmentKind::Audio, PreparedMedia::Audio(x)) => (tokens.audio, x.clone()),
            (SegmentKind::VideoFrame { frame }, PreparedMedia::Video { frames, .. }) => (
                tokens.video,
                frames
                    .get(*frame)
                    .ok_or_else(|| Error::Inference("missing video frame embedding".into()))?
                    .clone(),
            ),
            (
                SegmentKind::VideoAudio { offset },
                PreparedMedia::Video {
                    soundtrack: Some(x),
                    ..
                },
            ) => {
                let [count, channels] = x.dims();
                let end = offset
                    .checked_add(segment.tokens)
                    .ok_or_else(|| Error::Inference("audio slice overflow".into()))?;
                if end > count {
                    return Err(Error::Inference("video soundtrack is too short".into()));
                }
                (tokens.audio, x.clone().slice([*offset..end, 0..channels]))
            }
            _ => {
                return Err(Error::Inference(
                    "media embedding type does not match its placeholder".into(),
                ));
            }
        };
        let end = pos
            .checked_add(segment.tokens)
            .ok_or_else(|| Error::Inference("placeholder span overflow".into()))?;
        if segment.tokens == 0
            || wanted != id
            || end > n
            || ids[pos..end].iter().any(|v| *v != wanted)
            || value.dims() != [segment.tokens, width]
        {
            return Err(Error::Inference(
                "media embedding and placeholder count do not match".into(),
            ));
        }
        embeddings = embeddings.slice_assign(
            [0..1, pos..end, 0..width],
            value.reshape([1, segment.tokens, width]),
        );
        next += 1;
        pos = end;
    }
    if next != segments.len() {
        return Err(Error::Inference(
            "media embedding has no placeholder".into(),
        ));
    }
    Ok(embeddings)
}

/// Restore generated assistant speech from its original codebook embeddings.
pub fn bind_speech_frames<B: Backend>(
    ids: &[u32],
    mut embeddings: Tensor<B, 3>,
    token: u32,
    frames: &[Tensor<B, 3>],
) -> Result<Tensor<B, 3>> {
    let [batch, length, width] = embeddings.dims();
    let positions: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(index, &id)| (id == token).then_some(index))
        .collect();
    if batch != 1
        || length != ids.len()
        || positions.len() != frames.len()
        || frames.iter().any(|frame| frame.dims() != [1, 1, width])
    {
        return Err(Error::Inference(
            "assistant speech placeholders and codebook frames do not match".into(),
        ));
    }
    for (position, frame) in positions.into_iter().zip(frames) {
        embeddings =
            embeddings.slice_assign([0..1, position..position + 1, 0..width], frame.clone());
    }
    Ok(embeddings)
}
