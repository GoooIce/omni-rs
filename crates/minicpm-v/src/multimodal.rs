//! Multimodal prompt assembly: preprocess an image, run the vision tower,
//! expand the placeholder text and splice the embeddings into the token
//! stream with canvas M-RoPE positions.
use crate::{
    chat::{self, Grid, Message, Role},
    config::MiniCpmConfig,
    image,
    model::TextModel,
    positions::{self, Markers},
    vision::VisionTower,
};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{Precision, weights::WeightSource};

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

/// Marker token ids the canvas layout consumes, from the model config.
pub fn markers(config: &MiniCpmConfig) -> Markers {
    Markers {
        image_start: config.image_start_id,
        image_end: config.image_end_id,
        slice_start: config.slice_start_id,
        slice_end: config.slice_end_id,
        newline: config.newline_id,
    }
}

/// Inputs to single-image prompt assembly.
pub struct Build<'a, B: Backend> {
    pub tokenizer: &'a crate::tokenizer::MiniCpmTokenizer,
    pub model: &'a TextModel<B>,
    pub config: &'a MiniCpmConfig,
    pub model_dir: &'a std::path::Path,
    pub image: &'a std::path::Path,
    pub prompt: &'a str,
    pub device: &'a B::Device,
    pub precision: Precision,
}

pub struct Multimodal<B: Backend> {
    pub vision: VisionTower<B>,
    model: std::marker::PhantomData<B>,
}

impl<B: Backend> Multimodal<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        config: &MiniCpmConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
        Ok(Self {
            vision: VisionTower::load(w, config, p, d)?,
            model: std::marker::PhantomData,
        })
    }

    /// Build one single-image prompt: user text plus the image.
    pub fn build_prompt(&self, request: Build<'_, B>) -> Result<Prompt<B>, String> {
        let Build {
            tokenizer,
            model,
            config,
            model_dir,
            image: image_path,
            prompt,
            device,
            precision,
        } = request;
        let rgba = image::open(image_path).map_err(|e| e.to_string())?;
        let preprocessed =
            image::preprocess(&rgba, &image::PreprocessConfig::from_directory(model_dir))
                .map_err(|e| e.to_string())?;
        let patch = config.vision_config.patch_size;
        let width = config.vision_config.num_channels * patch * patch;
        let output = match precision {
            Precision::F16 => FloatDType::F16,
            Precision::F32 => FloatDType::F32,
        };
        let crops: Vec<Tensor<B, 2>> = preprocessed
            .crops
            .iter()
            .map(|values| {
                Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(values.clone(), [values.len() / width, width]),
                    device,
                )
                .cast(output)
            })
            .collect();
        let features = self
            .vision
            .forward(&crops, &preprocessed.target_sizes, output)
            .map_err(|e| e.to_string())?;

        // One image: local id 0, thumbnail plus slice rows.
        let grids: Vec<Grid> = preprocessed.target_sizes.clone();
        let replacement = format!(
            "{}{}",
            chat::image_id_prefix(0),
            chat::image_replacement(
                &grids,
                preprocessed.slice_grid.0,
                preprocessed.slice_grid.1,
                16,
                chat::IMAGE_PAD,
            )
        );
        let messages = vec![Message {
            role: Role::User,
            content: vec![chat::Part::Image, chat::Part::Text(prompt.to_string())],
        }];
        let rendered = chat::render(&messages, true, None).map_err(|e| e.to_string())?;
        let text_input = rendered.replace(chat::IMAGE_PAD, &replacement);
        let ids = tokenizer.encode(&text_input);

        // Canvas positions over pad tokens.
        let types: Vec<u8> = ids
            .iter()
            .map(|&id| u8::from(id == config.image_token_id))
            .collect();
        let (table, delta) = positions::rope_index(
            &ids,
            &types,
            &preprocessed.target_sizes,
            4,
            &markers(config),
        );

        // Splice vision features into the token embeddings at pad runs.
        let mut hidden = model.embeddings(&ids).map_err(|e| e.to_string())?;
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
            let rows: Vec<f32> =
                feature_values[cursor * language_width..(cursor + count) * language_width].to_vec();
            cursor += count;
            let patch_embeddings = Tensor::<B, 3>::from_data(
                burn::tensor::TensorData::new(rows, [1, count, language_width]),
                device,
            );
            let dtype = omni_r2t2::tensor::dtype(&hidden);
            hidden = hidden.slice_assign(
                [0..1, start..index, 0..language_width],
                patch_embeddings.cast(dtype),
            );
        }
        if cursor != features.dims()[0] {
            return Err(format!(
                "image placeholders cover {cursor} of {} vision tokens",
                features.dims()[0]
            ));
        }
        Ok(Prompt {
            ids,
            embeddings: Some(hidden),
            positions: Some(table),
            rope_delta: delta,
        })
    }

    /// Build a prompt from a full message list with N ordered images: each
    /// `Part::Image` in `messages` consumes one entry of `images`, in order,
    /// so the first image part becomes `<image_id>0</image_id>` (the upstream
    /// `<image1>`), the second `<image_id>1</image_id>`, and so on. A leading
    /// system message is supported; `enable_thinking` is passed through to
    /// `chat::render` for the assistant generation prompt.
    pub fn build_prompt_multi(&self, request: MultiBuild<'_, B>) -> Result<Prompt<B>, String> {
        let MultiBuild {
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
        if images.is_empty() {
            return Err(
                "build_prompt_multi needs at least one image; use Prompt::text for text-only"
                    .to_string(),
            );
        }
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
        let patch = config.vision_config.patch_size;
        let width = config.vision_config.num_channels * patch * patch;
        let output = match precision {
            Precision::F16 => FloatDType::F16,
            Precision::F32 => FloatDType::F32,
        };
        let mut preprocessed = Vec::with_capacity(images.len());
        let mut per_image_features = Vec::with_capacity(images.len());
        for path in images {
            let rgba = image::open(path).map_err(|e| e.to_string())?;
            let prep = image::preprocess(&rgba, &preprocess).map_err(|e| e.to_string())?;
            let crops: Vec<Tensor<B, 2>> = prep
                .crops
                .iter()
                .map(|values| {
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(
                            values.clone(),
                            [values.len() / width, width],
                        ),
                        device,
                    )
                    .cast(output)
                })
                .collect();
            let features = self
                .vision
                .forward(&crops, &prep.target_sizes, output)
                .map_err(|e| e.to_string())?;
            preprocessed.push(prep);
            per_image_features.push(features);
        }

        let replacements = image_replacements(&preprocessed);
        let text_input = render_with_images(messages, &replacements, enable_thinking)?;
        let ids = tokenizer.encode(&text_input);

        // Canvas positions over pad tokens; crop grids concatenate in image
        // order (each image's thumbnail first, then its slice crops).
        let types: Vec<u8> = ids
            .iter()
            .map(|&id| u8::from(id == config.image_token_id))
            .collect();
        let (table, delta) = positions::rope_index(
            &ids,
            &types,
            &all_target_sizes(&preprocessed),
            4,
            &markers(config),
        );

        // Splice vision features into the token embeddings at pad runs; the
        // concatenated feature rows align with the pad runs in image order.
        let features = Tensor::cat(per_image_features, 0);
        let mut hidden = model.embeddings(&ids).map_err(|e| e.to_string())?;
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
            let rows: Vec<f32> =
                feature_values[cursor * language_width..(cursor + count) * language_width].to_vec();
            cursor += count;
            let patch_embeddings = Tensor::<B, 3>::from_data(
                burn::tensor::TensorData::new(rows, [1, count, language_width]),
                device,
            );
            let dtype = omni_r2t2::tensor::dtype(&hidden);
            hidden = hidden.slice_assign(
                [0..1, start..index, 0..language_width],
                patch_embeddings.cast(dtype),
            );
        }
        if cursor != features.dims()[0] {
            return Err(format!(
                "image placeholders cover {cursor} of {} vision tokens",
                features.dims()[0]
            ));
        }
        Ok(Prompt {
            ids,
            embeddings: Some(hidden),
            positions: Some(table),
            rope_delta: delta,
        })
    }
}

/// Inputs to multi-image prompt assembly with an arbitrary message list.
pub struct MultiBuild<'a, B: Backend> {
    pub tokenizer: &'a crate::tokenizer::MiniCpmTokenizer,
    pub model: &'a TextModel<B>,
    pub config: &'a MiniCpmConfig,
    pub model_dir: &'a std::path::Path,
    /// Full message list; a leading system message is supported. Each
    /// `Part::Image` consumes one entry of `images`, in order.
    pub messages: &'a [Message],
    /// Ordered image paths; `images[i]` pairs with the i-th `Part::Image`.
    pub images: &'a [std::path::PathBuf],
    /// Thinking behavior of the assistant generation prompt: `None` leaves a
    /// `<think>` block open for the model, `Some(false)` closes it empty.
    pub enable_thinking: Option<bool>,
    pub device: &'a B::Device,
    pub precision: Precision,
}

/// Per-image placeholder replacements in image order: `<image_id>i</image_id>`
/// followed by the thumbnail and slice pad runs of image i (`16x` downsample).
fn image_replacements(preprocessed: &[image::Preprocessed]) -> Vec<String> {
    preprocessed
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
                    16,
                    chat::IMAGE_PAD,
                )
            )
        })
        .collect()
}

/// Crop grids of every image concatenated in image order (each image's
/// thumbnail first, then its slice crops) — the layout `rope_index` consumes.
fn all_target_sizes(preprocessed: &[image::Preprocessed]) -> Vec<Grid> {
    preprocessed
        .iter()
        .flat_map(|prep| prep.target_sizes.iter().copied())
        .collect()
}

/// Render the ChatML text and substitute the i-th `Part::Image` placeholder
/// with the i-th replacement, preserving the image order of the messages.
/// Each replacement itself contains IMAGE_PAD once per placeholder token, so
/// the count contract is checked BEFORE any substitution — scanning the
/// substituted text would count the placeholders just inserted.
fn render_with_images(
    messages: &[Message],
    replacements: &[String],
    enable_thinking: Option<bool>,
) -> Result<String, String> {
    let text = chat::render(messages, true, enable_thinking).map_err(|e| e.to_string())?;
    let markers = text.matches(chat::IMAGE_PAD).count();
    if markers != replacements.len() {
        return Err(format!(
            "rendered prompt has {markers} {} markers but {} images were given",
            chat::IMAGE_PAD,
            replacements.len()
        ));
    }
    // Split-and-join, not replacen: a replacement itself contains IMAGE_PAD
    // once per placeholder token, so naive sequential `replacen(.., 1)` would
    // target pads inside replacements inserted earlier.
    let mut out = String::with_capacity(text.len());
    let mut tails = text.splitn(replacements.len() + 1, chat::IMAGE_PAD);
    out.push_str(tails.next().unwrap_or_default());
    for (replacement, tail) in replacements.iter().zip(tails) {
        out.push_str(replacement);
        out.push_str(tail);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Part, Role};

    fn preprocessed(target_sizes: &[Grid], slice_grid: (usize, usize)) -> image::Preprocessed {
        image::Preprocessed {
            crops: Vec::new(),
            target_sizes: target_sizes.to_vec(),
            slice_grid,
        }
    }

    #[test]
    fn replacements_and_target_sizes_follow_image_order() {
        // Image 0: thumbnail + a 2x1 slice grid; image 1: thumbnail only.
        let grid = |h, w| Grid {
            height: h,
            width: w,
        };
        let prep = vec![
            preprocessed(&[grid(28, 28), grid(28, 28), grid(28, 28)], (1, 2)),
            preprocessed(&[grid(28, 56)], (0, 0)),
        ];
        let replacements = image_replacements(&prep);
        assert_eq!(replacements.len(), 2);
        assert!(replacements[0].starts_with(&chat::image_id_prefix(0)));
        assert!(replacements[1].starts_with(&chat::image_id_prefix(1)));
        // Image 0: thumbnail 28*28/16 = 49 pads plus two slice crops of 49
        // pads each; image 1: 28*56/16 = 98 thumbnail pads, no slices.
        let pads = |text: &str| text.matches(chat::IMAGE_PAD).count();
        assert_eq!(pads(&replacements[0]), 49 + 2 * 49);
        assert_eq!(pads(&replacements[1]), 98);
        assert_eq!(
            all_target_sizes(&prep),
            vec![grid(28, 28), grid(28, 28), grid(28, 28), grid(28, 56),],
            "crop grids concatenate in image order, thumbnail first"
        );
    }

    #[test]
    fn render_with_images_keeps_message_order_and_system() {
        let messages = vec![
            Message::text(Role::System, "SYS"),
            Message {
                role: Role::User,
                content: vec![Part::Image, Part::Image, Part::Text("PROMPT".to_string())],
            },
        ];
        // Real replacements contain IMAGE_PAD once per placeholder token; the
        // substitution must leave those pads untouched and in order.
        let pad = chat::IMAGE_PAD;
        let r0 = format!("<image_id>0</image_id>{pad}{pad}");
        let r1 = format!("<image_id>1</image_id>{pad}{pad}{pad}");
        let text = render_with_images(&messages, &[r0, r1], Some(false)).unwrap();
        assert!(text.contains("<|im_start|>system\nSYS<|im_end|>"));
        let (i0, i1, prompt) = (
            text.find("<image_id>0</image_id>").unwrap(),
            text.find("<image_id>1</image_id>").unwrap(),
            text.find("PROMPT").unwrap(),
        );
        assert!(i0 < i1 && i1 < prompt, "images expand in message order");
        // Exactly the replacement pads survive: 2 + 3, none dropped or eaten.
        assert_eq!(text.matches(chat::IMAGE_PAD).count(), 5);
        let first_block = &text[i0..i1];
        assert_eq!(first_block.matches(chat::IMAGE_PAD).count(), 2);
        assert!(
            text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "thinking disabled closes the block empty"
        );
    }

    #[test]
    fn render_with_images_rejects_count_mismatch() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![Part::Image, Part::Text("PROMPT".to_string())],
        }];
        assert!(render_with_images(&messages, &[], None).is_err());
        assert!(render_with_images(&messages, &["R0".into(), "R1".into()], None).is_err());
    }
}
