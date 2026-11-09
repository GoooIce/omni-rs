//! Frozen LFM3 program writer and image-conditioned reader features.
use crate::{
    chart::{ChartProgram, ChartSpec, Question},
    reader::Example,
};
use burn::tensor::backend::Backend;
use image::{DynamicImage, ImageFormat, RgbImage};
use omni_lfm3::{
    Precision,
    generate::EndReason,
    model::{InputPart, OmniModel},
};
use std::{
    io::Cursor,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub struct FrozenReader<B: Backend> {
    model: OmniModel<B>,
    yes: usize,
    no: usize,
    directory: PathBuf,
}
impl<B: Backend> FrozenReader<B> {
    pub fn load(
        directory: &Path,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self, String> {
        let model =
            OmniModel::<B>::load(directory, precision, device).map_err(|e| e.to_string())?;
        let yes = model.tokenizer.encode("Yes").map_err(|e| e.to_string())?;
        let no = model.tokenizer.encode("No").map_err(|e| e.to_string())?;
        if yes.len() != 1 || no.len() != 1 || yes == no {
            return Err("reader labels must be distinct single tokens".into());
        }
        Ok(Self {
            model,
            yes: yes[0] as usize,
            no: no[0] as usize,
            directory: directory.to_owned(),
        })
    }
    pub fn verify_source(&self, fingerprint: &str, stop: &AtomicBool) -> Result<(), String> {
        if crate::fingerprint::model_fingerprint(&self.directory, stop)? != fingerprint {
            return Err("frozen model files changed during the evolution round".into());
        }
        Ok(())
    }
    pub fn write_program(
        &self,
        spec: ChartSpec,
        stop: &AtomicBool,
    ) -> Result<Option<ChartProgram>, String> {
        let (message, media) = self
            .model
            .prepare_parts(&[InputPart::Text(spec.request())], stop)
            .map_err(|e| e.to_string())?;
        let mut cache = self.model.new_text_cache();
        let result = self
            .model
            .generate_text(&[message], None, &media, &mut cache, 64, stop, |_| {
                !stop.load(Ordering::Acquire)
            })
            .map_err(|e| e.to_string())?;
        match result.reason {
            EndReason::Cancelled => return Err("interrupted".into()),
            EndReason::Limit => return Ok(None),
            EndReason::Stop(_) => (),
        }
        let text = self
            .model
            .tokenizer
            .decode(&result.tokens, true)
            .map_err(|e| e.to_string())?;
        Ok(ChartProgram::parse(&text).ok())
    }
    pub fn features(
        &self,
        image: &RgbImage,
        question: Question,
        label: bool,
        stop: &AtomicBool,
    ) -> Result<Example, String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let mut png = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image.clone())
            .write_to(&mut png, ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        let parts = [
            InputPart::Image(png.into_inner()),
            InputPart::Text(question.prompt().into()),
        ];
        let (message, media) = self
            .model
            .prepare_parts(&parts, stop)
            .map_err(|e| e.to_string())?;
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let prompt = self
            .model
            .prepare_prompt(&[message], None, &media)
            .map_err(|e| e.to_string())?;
        let mut cache = self.model.new_text_cache();
        let hidden = self
            .model
            .text
            .forward_embeddings(prompt.embeddings, &mut cache)
            .map_err(|e| e.to_string())?;
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let [_, n, width] = hidden.dims();
        let last = hidden.slice([0..1, n - 1..n, 0..width]);
        let logits = self
            .model
            .text
            .logits(last.clone())
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| e.to_string())?;
        let features = last
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| e.to_string())?;
        if self.yes >= logits.len()
            || self.no >= logits.len()
            || !logits[self.yes].is_finite()
            || !logits[self.no].is_finite()
            || features.iter().any(|x| !x.is_finite())
        {
            return Err("non-finite or missing reader activations".into());
        }
        let base_logit = logits[self.yes] - logits[self.no];
        if !base_logit.is_finite() {
            return Err("non-finite reader logit difference".into());
        }
        Ok(Example {
            features,
            base_logit,
            label,
        })
    }
}
