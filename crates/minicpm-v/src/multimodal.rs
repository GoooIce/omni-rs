//! Loaded vision tower. Image prompts are assembled by [`crate::prompt::assemble`];
//! this module only turns a preprocessed image into language-token rows.

pub use crate::prompt::Prompt;

use crate::{config::MiniCpmConfig, image, prompt::VisionEmbed, vision::VisionTower};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{Precision, weights::WeightSource};

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
}

impl<B: Backend> VisionEmbed<B> for Multimodal<B> {
    fn embed_image(
        &self,
        image: &image::Preprocessed,
        output: FloatDType,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String> {
        let crops: Vec<Tensor<B, 2>> = image
            .crops
            .iter()
            .zip(&image.target_sizes)
            .map(|(values, grid)| {
                let patches = grid.height * grid.width;
                if patches == 0 || !values.len().is_multiple_of(patches) {
                    return Err("vision crop does not match its patch grid".to_string());
                }
                let width = values.len() / patches;
                Ok(Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(values.clone(), [patches, width]),
                    device,
                )
                .cast(output))
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.vision
            .forward(&crops, &image.target_sizes, output)
            .map_err(|e| e.to_string())
    }
}
