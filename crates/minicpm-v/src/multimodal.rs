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
}
