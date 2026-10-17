//! One image-prompt assembly request.
//!
//! Callers hand over a conversation, its images and the towers that already
//! know how to embed tokens and pixels. They get back a [`Prompt`]: token
//! ids, spliced embeddings, canvas positions and the generation-time
//! `rope_delta`. Slice grids, placeholder expansion, M-RoPE and the pad-run
//! splice stay inside this module, so the pad divisor and the M-RoPE merge
//! factor cannot drift apart at a call site.

use crate::{
    chat::{self, Grid, Message},
    config::MiniCpmConfig,
    image,
    model::TextModel,
    positions::{self, Markers},
    tokenizer::MiniCpmTokenizer,
};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::Precision;
use std::path::Path;

/// Assembled decoder input. Text-only prompts leave the canvas fields empty;
/// generation then uses ordinary 1-D positions.
pub struct Prompt<B: Backend> {
    pub ids: Vec<u32>,
    pub embeddings: Option<Tensor<B, 3>>,
    pub positions: Option<Vec<[i64; 3]>>,
    pub rope_delta: i64,
}

impl<B: Backend> Prompt<B> {
    pub fn text(ids: Vec<u32>) -> Self {
        Self {
            ids,
            embeddings: None,
            positions: None,
            rope_delta: 0,
        }
    }
}

/// Encodes the rendered ChatML string. The checkpoint tokenizer is the
/// production adapter.
pub trait Tokens {
    fn encode(&self, text: &str) -> Vec<u32>;
}

impl Tokens for MiniCpmTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MiniCpmTokenizer::encode(self, text)
    }
}

/// Token-embedding rows `(1, seq, hidden)` the splice writes image features into.
pub trait TokenEmbed<B: Backend> {
    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String>;
}

impl<B: Backend> TokenEmbed<B> for TextModel<B> {
    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String> {
        self.embeddings(ids).map_err(|e| e.to_string())
    }
}

/// One assembly request. `vision` is present when `images` is non-empty; the
/// tower's numeric implementation is not part of this request.
pub struct Request<'a, B: Backend, T: Tokens = MiniCpmTokenizer, E: TokenEmbed<B> = TextModel<B>> {
    pub vision: Option<&'a dyn VisionEmbed<B>>,
    pub tokenizer: &'a T,
    pub model: &'a E,
    pub config: &'a MiniCpmConfig,
    pub model_dir: &'a Path,
    pub messages: &'a [Message],
    pub images: &'a [std::path::PathBuf],
    pub enable_thinking: Option<bool>,
    pub device: &'a B::Device,
    pub precision: Precision,
}

/// Internal seam for image features. Production passes the loaded vision tower;
/// tests pass a stand-in that returns a known row per language token.
pub trait VisionEmbed<B: Backend> {
    fn embed_image(
        &self,
        image: &crate::image::Preprocessed,
        output: burn::tensor::FloatDType,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String>;
}

/// Assemble one prompt. An empty image list is the text-only ChatML path:
/// the template is unchanged, and generation keeps 1-D positions. Images
/// expand placeholders, lay out the canvas and splice vision rows together.
pub fn assemble<B: Backend, T: Tokens, E: TokenEmbed<B>>(
    request: Request<'_, B, T, E>,
) -> Result<Prompt<B>, String> {
    if request.images.is_empty() {
        let rendered = chat::render(request.messages, true, request.enable_thinking)
            .map_err(|e| e.to_string())?;
        return Ok(Prompt::text(request.tokenizer.encode(&rendered)));
    }
    assemble_images(request)
}

fn assemble_images<B: Backend, T: Tokens, E: TokenEmbed<B>>(
    request: Request<'_, B, T, E>,
) -> Result<Prompt<B>, String> {
    let Request {
        vision,
        tokenizer,
        model,
        config,
        model_dir,
        messages,
        images,
        enable_thinking,
        device,
        precision,
    } = request;
    let vision = vision.ok_or_else(|| "image prompt needs a vision tower".to_string())?;
    let image_parts = messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter(|part| matches!(part, chat::Part::Image))
        .count();
    if image_parts != images.len() {
        return Err(format!(
            "{image_parts} image parts in messages but {} images given",
            images.len()
        ));
    }

    let preprocess = image::PreprocessConfig::from_directory(model_dir);
    let mut prepared = Vec::with_capacity(images.len());
    for path in images {
        let rgba = image::open(path).map_err(|e| e.to_string())?;
        prepared.push(image::preprocess(&rgba, &preprocess).map_err(|e| e.to_string())?);
    }
    assemble_prepared(
        vision,
        tokenizer,
        model,
        config,
        messages,
        &prepared,
        enable_thinking,
        device,
        precision,
    )
}

fn assemble_prepared<B: Backend, T: Tokens, E: TokenEmbed<B>>(
    vision: &dyn VisionEmbed<B>,
    tokenizer: &T,
    model: &E,
    config: &MiniCpmConfig,
    messages: &[Message],
    prepared: &[image::Preprocessed],
    enable_thinking: Option<bool>,
    device: &B::Device,
    precision: Precision,
) -> Result<Prompt<B>, String> {
    let (divisor, merge_factor) = canvas_factors(config);
    let output = match precision {
        Precision::F16 => FloatDType::F16,
        Precision::F32 => FloatDType::F32,
    };
    let mut per_image = Vec::with_capacity(prepared.len());
    for prep in prepared {
        per_image.push(vision.embed_image(prep, output, device)?);
    }

    let replacements = image_replacements(&prepared, divisor);
    let text = render_with_images(messages, &replacements, enable_thinking)?;
    let ids = tokenizer.encode(&text);
    let types: Vec<u8> = ids
        .iter()
        .map(|&id| u8::from(id == config.image_token_id))
        .collect();
    let targets: Vec<Grid> = prepared
        .iter()
        .flat_map(|prep| prep.target_sizes.iter().copied())
        .collect();
    let (table, delta) =
        positions::rope_index(&ids, &types, &targets, merge_factor, &markers(config));
    let features = Tensor::cat(per_image, 0);
    let hidden = splice(model, &ids, features, config, device)?;
    Ok(Prompt {
        ids,
        embeddings: Some(hidden),
        positions: Some(table),
        rope_delta: delta,
    })
}

/// Window merge and the language merger share one factor. The pad divisor is
/// that factor's area (2×2 then 2×2 → merge 4, divisor 16).
fn canvas_factors(config: &MiniCpmConfig) -> (usize, usize) {
    let (window_h, window_w) = config.vision_config.window_kernel_size;
    let (merge_h, merge_w) = config.merge_kernel_size;
    let down_h = window_h * merge_h;
    let down_w = window_w * merge_w;
    (down_h * down_w, down_h.max(down_w))
}

fn markers(config: &MiniCpmConfig) -> Markers {
    Markers {
        image_start: config.image_start_id,
        image_end: config.image_end_id,
        slice_start: config.slice_start_id,
        slice_end: config.slice_end_id,
        newline: config.newline_id,
    }
}

fn image_replacements(prepared: &[image::Preprocessed], divisor: usize) -> Vec<String> {
    prepared
        .iter()
        .enumerate()
        .map(|(index, prep)| {
            format!(
                "{}{}",
                chat::image_id_prefix(index),
                chat::image_replacement(
                    &prep.target_sizes,
                    prep.slice_grid.0,
                    prep.slice_grid.1,
                    divisor,
                    chat::IMAGE_PAD,
                )
            )
        })
        .collect()
}

fn render_with_images(
    messages: &[Message],
    replacements: &[String],
    enable_thinking: Option<bool>,
) -> Result<String, String> {
    let text = chat::render(messages, true, enable_thinking).map_err(|e| e.to_string())?;
    let marker_count = text.matches(chat::IMAGE_PAD).count();
    if marker_count != replacements.len() {
        return Err(format!(
            "rendered prompt has {marker_count} {} markers but {} images were given",
            chat::IMAGE_PAD,
            replacements.len()
        ));
    }
    let mut out = String::with_capacity(text.len());
    let mut tails = text.splitn(replacements.len() + 1, chat::IMAGE_PAD);
    out.push_str(tails.next().unwrap_or_default());
    for (replacement, tail) in replacements.iter().zip(tails) {
        out.push_str(replacement);
        out.push_str(tail);
    }
    Ok(out)
}

fn splice<B: Backend, E: TokenEmbed<B>>(
    model: &E,
    ids: &[u32],
    features: Tensor<B, 2>,
    config: &MiniCpmConfig,
    device: &B::Device,
) -> Result<Tensor<B, 3>, String> {
    let mut hidden = model.embed_tokens(ids)?;
    let mut cursor = 0usize;
    let mut index = 0usize;
    let feature_values = features
        .clone()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .map_err(|e| e.to_string())?
        .to_vec();
    let language_width = config.text_config.hidden_size;
    let rows = features.dims()[0];
    while index < ids.len() {
        if ids[index] != config.image_token_id {
            index += 1;
            continue;
        }
        let start = index;
        while index < ids.len() && ids[index] == config.image_token_id {
            index += 1;
        }
        let count = index - start;
        let begin = cursor * language_width;
        let end = (cursor + count) * language_width;
        if end > feature_values.len() {
            return Err(format!(
                "image placeholders cover more than {rows} vision tokens"
            ));
        }
        let row_values = feature_values[begin..end].to_vec();
        cursor += count;
        let patch_embeddings = Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(row_values, [1, count, language_width]),
            device,
        );
        let dtype = omni_r2t2::tensor::dtype(&hidden);
        hidden = hidden.slice_assign(
            [0..1, start..index, 0..language_width],
            patch_embeddings.cast(dtype),
        );
    }
    if cursor != rows {
        return Err(format!(
            "image placeholders cover {cursor} of {rows} vision tokens"
        ));
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, Part, Role};
    use burn::backend::NdArray;

    type B = NdArray;

    struct Lexicon {
        specials: Vec<(&'static str, u32)>,
    }

    impl Tokens for Lexicon {
        fn encode(&self, text: &str) -> Vec<u32> {
            let mut specials = self.specials.clone();
            specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
            let mut ids = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some((token, id)) =
                    specials.iter().find(|(token, _)| rest.starts_with(token))
                {
                    ids.push(*id);
                    rest = &rest[token.len()..];
                } else {
                    let ch = rest.chars().next().unwrap();
                    ids.push(u32::from(ch));
                    rest = &rest[ch.len_utf8()..];
                }
            }
            ids
        }
    }

    struct ZeroEmbed;

    impl TokenEmbed<B> for ZeroEmbed {
        fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String> {
            let width = 4;
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(vec![0f32; ids.len() * width], [1, ids.len(), width]),
                &Default::default(),
            ))
        }
    }

    fn config() -> MiniCpmConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "minicpmv4_7",
            "vision_config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "image_size": 224,
                "num_attention_heads": 1,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "rope_parameters": { "rope_theta": 10000.0 }
            },
            "image_token_id": 248056,
            "video_token_id": 248057,
            "image_start_id": 248078,
            "image_end_id": 248079,
            "slice_start_id": 248088,
            "slice_end_id": 248089,
            "newline_id": 10,
            "insert_layer_id": 0,
            "downsample_mode": "16x",
            "mrope_mode": "canvas",
            "merge_kernel_size": [2, 2]
        }))
        .unwrap()
    }

    fn lexicon() -> Lexicon {
        Lexicon {
            specials: vec![
                ("<|im_start|>", 1001),
                ("<|im_end|>", 1002),
                ("<think>", 1003),
                ("<image_id>", 1004),
                ("</image_id>", 1005),
                ("</think>", 1006),
                ("<|image_pad|>", 248056),
                ("</image>", 248079),
                ("<image>", 248078),
                ("</slice>", 248089),
                ("<slice>", 248088),
            ],
        }
    }

    #[test]
    fn text_only_prompt_is_chat_tokens_without_a_canvas() {
        let config = config();
        let tokenizer = lexicon();
        let model = ZeroEmbed;
        let device = Default::default();
        let messages = vec![Message::text(Role::User, "Hi")];
        let prompt = assemble(Request {
            vision: None,
            tokenizer: &tokenizer,
            model: &model,
            config: &config,
            model_dir: Path::new("."),
            messages: &messages,
            images: &[],
            enable_thinking: None,
            device: &device,
            precision: Precision::F32,
        })
        .unwrap();

        assert_eq!(
            prompt.ids,
            vec![
                1001, 117, 115, 101, 114, 10, 72, 105, 1002, 10, 1001, 97, 115, 115, 105, 115, 116,
                97, 110, 116, 10, 1003, 10,
            ]
        );
        assert!(prompt.embeddings.is_none());
        assert!(prompt.positions.is_none());
        assert_eq!(prompt.rope_delta, 0);
    }

    struct OnesEmbed;

    impl TokenEmbed<B> for OnesEmbed {
        fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String> {
            let width = 4;
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(
                    vec![1.0f32; ids.len() * width],
                    [1, ids.len(), width],
                ),
                &Default::default(),
            ))
        }
    }

    /// Stand-in tower: 64 language rows, the unsliced thumbnail's official
    /// token count, each filled with 3 so the splice is visible.
    struct ThumbnailFeatures;

    impl VisionEmbed<burn::backend::NdArray> for ThumbnailFeatures {
        fn embed_image(
            &self,
            _image: &crate::image::Preprocessed,
            _output: burn::tensor::FloatDType,
            device: &burn::backend::ndarray::NdArrayDevice,
        ) -> Result<Tensor<burn::backend::NdArray, 2>, String> {
            let hidden = 4;
            let rows = 64;
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(vec![3.0f32; rows * hidden], [rows, hidden]),
                device,
            ))
        }
    }

    #[test]
    fn unsliced_image_pads_follow_the_official_canvas() {
        let config = config();
        let tokenizer = lexicon();
        let model = OnesEmbed;
        let vision = ThumbnailFeatures;
        let device = Default::default();
        let dir = std::env::temp_dir().join("omni-rs-prompt-unsliced");
        std::fs::create_dir_all(&dir).unwrap();
        let image_path = dir.join("square.png");
        ::image::RgbImage::from_pixel(32, 32, ::image::Rgb([10, 20, 30]))
            .save(&image_path)
            .unwrap();
        let messages = vec![Message {
            role: Role::User,
            content: vec![Part::Image, Part::Text("Q".to_string())],
        }];
        let prompt = assemble(Request {
            vision: Some(&vision),
            tokenizer: &tokenizer,
            model: &model,
            config: &config,
            model_dir: &dir,
            messages: &messages,
            images: &[image_path],
            enable_thinking: None,
            device: &device,
            precision: Precision::F32,
        })
        .unwrap();

        // ChatML around one unsliced thumbnail: 64 pads, `<image>` at index 9.
        let mut expected_ids = vec![1001, 117, 115, 101, 114, 10, 1004, 48, 1005, 248078];
        expected_ids.extend(std::iter::repeat(248056).take(64));
        expected_ids.extend([
            248079, 10, 81, 1002, 10, 1001, 97, 115, 115, 105, 115, 116, 97, 110, 116, 10, 1003, 10,
        ]);
        assert_eq!(prompt.ids, expected_ids);
        let positions = prompt.positions.expect("canvas table");
        assert_eq!(positions.len(), expected_ids.len());
        // Official text-image-text fixture: pads are a row-major language grid
        // whose side is the patch grid divided by the merge factor. An unsliced
        // thumbnail is 448/14 = 32 patches, merge 4, so an 8×8 canvas. `<image>`
        // sits on the halo; `</image>` sits on the far corner.
        let origin = 9;
        assert_eq!(positions[9], [9, 8, 8]);
        for (i, slot) in positions[10..74].iter().enumerate() {
            let row = (i / 8) as i64;
            let col = (i % 8) as i64;
            assert_eq!(*slot, [origin, origin + row, origin + col], "pad {i}");
        }
        assert_eq!(positions[74], [9, 17, 17]);
        assert_eq!(positions[75], [9, 9, 9]);
        assert_eq!(positions[76], [18, 18, 18]);
        assert_eq!(prompt.rope_delta, -58);

        let embedded = prompt.embeddings.expect("spliced embeddings").into_data();
        let values = embedded
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert_eq!(values.len(), expected_ids.len() * 4);
        for (index, id) in expected_ids.iter().enumerate() {
            let row = &values[index * 4..(index + 1) * 4];
            let expected = if *id == 248056 { 3.0 } else { 1.0 };
            assert!(
                row.iter().all(|value| *value == expected),
                "row {index} id {id} = {row:?}"
            );
        }
    }

    struct FilledRows {
        rows: usize,
        value: f32,
    }

    impl VisionEmbed<burn::backend::NdArray> for FilledRows {
        fn embed_image(
            &self,
            _image: &crate::image::Preprocessed,
            _output: burn::tensor::FloatDType,
            device: &burn::backend::ndarray::NdArrayDevice,
        ) -> Result<Tensor<burn::backend::NdArray, 2>, String> {
            let hidden = 4;
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(
                    vec![self.value; self.rows * hidden],
                    [self.rows, hidden],
                ),
                device,
            ))
        }
    }

    #[test]
    fn sliced_image_canvas_matches_the_official_table() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/canvas-positions.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let case = &fixture["sliced-image"];
        let official_ids: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        let official_channels: Vec<Vec<i64>> = case["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_i64().unwrap())
                    .collect()
            })
            .collect();
        let official_positions: Vec<[i64; 3]> = (0..official_ids.len())
            .map(|index| {
                [
                    official_channels[0][index],
                    official_channels[1][index],
                    official_channels[2][index],
                ]
            })
            .collect();
        let target_sizes: Vec<crate::chat::Grid> = case["target_sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let pair = pair.as_array().unwrap();
                crate::chat::Grid {
                    height: pair[0].as_u64().unwrap() as usize,
                    width: pair[1].as_u64().unwrap() as usize,
                }
            })
            .collect();
        let pad_rows = official_ids.iter().filter(|id| **id == 248056).count();
        let mut config = config();
        // The fixture's newline token is 198, not the test lexicon's raw '\n'.
        config.newline_id = 198;
        let mut tokenizer = lexicon();
        tokenizer.specials.push(("\n", 198));
        let model = OnesEmbed;
        let vision = FilledRows {
            rows: pad_rows,
            value: 3.0,
        };
        let device = Default::default();
        // A text part after the image is joined with a newline, and that token
        // is the canvas newline, so it stays in the frame and moves rope_delta
        // off the fixture. The image alone is the official sliced sequence.
        let messages = vec![Message {
            role: Role::User,
            content: vec![Part::Image],
        }];
        let prepared = vec![image::Preprocessed {
            crops: vec![Vec::new(); target_sizes.len()],
            target_sizes,
            // The fixture lays the four detail crops as two rows of two.
            slice_grid: (2, 2),
        }];
        let prompt = assemble_prepared(
            &vision,
            &tokenizer,
            &model,
            &config,
            &messages,
            &prepared,
            None,
            &device,
            Precision::F32,
        )
        .unwrap();

        let official_delta = case["rope_deltas"][0][0].as_i64().unwrap();
        assert_eq!(prompt.rope_delta, official_delta);
        let positions = prompt.positions.expect("canvas table");
        let image_start = 248_078u32;
        let our_start = prompt.ids.iter().position(|id| *id == image_start).unwrap();
        let their_start = official_ids
            .iter()
            .position(|id| *id == image_start)
            .unwrap();
        let mut matched = 0;
        while our_start + matched < prompt.ids.len()
            && their_start + matched < official_ids.len()
            && prompt.ids[our_start + matched] == official_ids[their_start + matched]
        {
            matched += 1;
        }
        // `<image>` through the last `</slice>`: thumbnail, four detail crops, row break.
        assert_eq!(matched, 19, "official image span");
        for step in 0..matched {
            for channel in 0..3 {
                assert_eq!(
                    positions[our_start + step][channel] - positions[our_start][channel],
                    official_positions[their_start + step][channel]
                        - official_positions[their_start][channel],
                    "token {step} channel {channel}"
                );
            }
        }
        let embedded = prompt.embeddings.expect("spliced embeddings").into_data();
        let values = embedded
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut pads = 0;
        for (index, id) in prompt.ids.iter().enumerate() {
            if *id != 248056 {
                continue;
            }
            pads += 1;
            let row = &values[index * 4..(index + 1) * 4];
            assert!(row.iter().all(|value| *value == 3.0), "pad row {index}");
        }
        assert_eq!(pads, pad_rows);
    }

    struct GridRows;

    impl VisionEmbed<burn::backend::NdArray> for GridRows {
        fn embed_image(
            &self,
            image: &crate::image::Preprocessed,
            _output: burn::tensor::FloatDType,
            device: &burn::backend::ndarray::NdArrayDevice,
        ) -> Result<Tensor<burn::backend::NdArray, 2>, String> {
            let (rows, value) = match image.target_sizes[0].height {
                16 => (16, 3.0f32),
                8 => (4, 5.0),
                _ => return Err("unexpected crop".into()),
            };
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(vec![value; rows * 4], [rows, 4]),
                device,
            ))
        }
    }

    #[test]
    fn two_images_keep_order_in_the_prompt() {
        let config = config();
        let tokenizer = lexicon();
        let model = OnesEmbed;
        let vision = GridRows;
        let device = Default::default();
        let grid = |h, w| crate::chat::Grid {
            height: h,
            width: w,
        };
        let prepared = vec![
            image::Preprocessed {
                crops: vec![Vec::new()],
                target_sizes: vec![grid(16, 16)],
                slice_grid: (0, 0),
            },
            image::Preprocessed {
                crops: vec![Vec::new()],
                target_sizes: vec![grid(8, 8)],
                slice_grid: (0, 0),
            },
        ];
        let messages = vec![
            Message::text(Role::System, "SYS"),
            Message {
                role: Role::User,
                content: vec![Part::Image, Part::Image, Part::Text("Q".to_string())],
            },
        ];
        let prompt = assemble_prepared(
            &vision,
            &tokenizer,
            &model,
            &config,
            &messages,
            &prepared,
            Some(false),
            &device,
            Precision::F32,
        )
        .unwrap();
        let first = prompt
            .ids
            .windows(3)
            .position(|w| w == [1004, 48, 1005])
            .unwrap();
        let second = prompt
            .ids
            .windows(3)
            .position(|w| w == [1004, 49, 1005])
            .unwrap();
        assert!(first < second, "image 0 is expanded before image 1");
        assert_eq!(
            &prompt.ids[..13],
            &[1001, 115, 121, 115, 116, 101, 109, 10, 83, 89, 83, 1002, 10]
        );
        assert!(prompt.ids.ends_with(&[1003, 10, 10, 1006, 10, 10]));
        let positions = prompt.positions.expect("canvas");
        assert_eq!(positions.len(), prompt.ids.len());
        let values = prompt
            .embeddings
            .expect("spliced")
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let pads: Vec<f32> = prompt
            .ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == 248056)
            .map(|(index, _)| values[index * 4])
            .collect();
        assert_eq!(pads.len(), 20);
        assert!(pads[..16].iter().all(|value| *value == 3.0));
        assert!(pads[16..].iter().all(|value| *value == 5.0));
    }

    #[test]
    fn pad_count_must_match_vision_rows() {
        let config = config();
        let tokenizer = lexicon();
        let model = OnesEmbed;
        let vision = FilledRows {
            rows: 1,
            value: 3.0,
        };
        let device = Default::default();
        let prepared = vec![image::Preprocessed {
            crops: vec![Vec::new()],
            target_sizes: vec![crate::chat::Grid {
                height: 16,
                width: 16,
            }],
            slice_grid: (0, 0),
        }];
        let messages = vec![Message {
            role: Role::User,
            content: vec![Part::Image, Part::Text("Q".to_string())],
        }];
        let error = match assemble_prepared(
            &vision,
            &tokenizer,
            &model,
            &config,
            &messages,
            &prepared,
            None,
            &device,
            Precision::F32,
        ) {
            Err(error) => error,
            Ok(_) => panic!("pad/feature mismatch should fail"),
        };
        assert!(
            error.contains("vision tokens"),
            "pad/feature mismatch should fail, got {error}"
        );
    }
}
