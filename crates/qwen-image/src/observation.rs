use super::{Options, gaussian_noise};
use crate::{Error, Result};

/// Default Qwen-Image-2.1-Turbo schedule length. The checkpoint sigmas stay as loaded.
pub const DEFAULT_TURBO_STEPS: usize = 8;
const LATENT_CHANNELS: usize = 64;

pub struct ObservationCell {
    pub step: usize,
    pub image: image::RgbaImage,
}

/// Official-text row: one retained canvas decode per Turbo step, plus the shareable noise.
pub struct OfficialObservation {
    cells: Vec<ObservationCell>,
    initial_noise: Vec<f32>,
    width: usize,
    height: usize,
}

impl OfficialObservation {
    pub fn begin(request: ObservationRequest, steps: usize) -> Result<Self> {
        if steps != DEFAULT_TURBO_STEPS {
            return Err(Error::Config(format!(
                "去噪观察要求默认 Turbo 的 {DEFAULT_TURBO_STEPS} 个去噪步，当前调度是 {steps} 步"
            )));
        }
        let (width, height) = request.canvas();
        Ok(Self {
            cells: Vec::new(),
            initial_noise: latent_noise(LATENT_CHANNELS, height, width, request.seed()),
            width,
            height,
        })
    }

    pub fn trajectories(&self) -> usize {
        1
    }

    pub fn uses_negative_guidance(&self) -> bool {
        false
    }

    pub fn is_official_text(&self) -> bool {
        true
    }

    pub fn initial_noise(&self) -> &[f32] {
        &self.initial_noise
    }

    pub fn canvas(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn cells(&self) -> &[ObservationCell] {
        &self.cells
    }

    /// Keep this step's decode. An earlier cell is left in place.
    pub fn retain(&mut self, step: usize, image: image::RgbaImage) -> Result<()> {
        if step != self.cells.len() + 1 || step > DEFAULT_TURBO_STEPS {
            return Err(Error::Config(
                "去噪观察按步留下格子，不能替换已经出现的画面".into(),
            ));
        }
        if image.width() as usize != self.width || image.height() as usize != self.height {
            return Err(Error::Config(format!(
                "观察格须按请求画布 {}×{} 解码",
                self.width, self.height
            )));
        }
        self.cells.push(ObservationCell { step, image });
        Ok(())
    }
}

#[derive(Debug)]
pub enum RequestKind {
    Ordinary { negative_guidance: bool },
    OfficialObservation(ObservationRequest),
}

impl RequestKind {
    pub fn trajectories(&self) -> usize {
        1
    }

    pub fn retained_decodes(&self) -> usize {
        match self {
            Self::Ordinary { .. } => 1,
            Self::OfficialObservation(_) => DEFAULT_TURBO_STEPS,
        }
    }

    pub fn negative_guidance(&self) -> bool {
        match self {
            Self::Ordinary { negative_guidance } => *negative_guidance,
            Self::OfficialObservation(_) => false,
        }
    }

    pub fn loads_minicpm_v(&self) -> bool {
        false
    }

    pub fn reads_projection_weights(&self) -> bool {
        false
    }

    pub fn decodes_after_each_step(&self) -> bool {
        matches!(self, Self::OfficialObservation(_))
    }
}

impl Options {
    pub fn request_kind(&self) -> Result<RequestKind> {
        if !self.observation {
            self.validate()?;
            return Ok(RequestKind::Ordinary {
                negative_guidance: self.guidance_enabled(),
            });
        }
        Ok(RequestKind::OfficialObservation(
            ObservationRequest::accept(self)?,
        ))
    }
}

pub(super) fn latent_noise(channels: usize, height: usize, width: usize, seed: u64) -> Vec<f32> {
    let h = height / 16;
    let w = width / 16;
    gaussian_noise(channels * h * w, seed)
}

/// A pure-text observation request: guidance off, no negative prompt, no references.
#[derive(Debug)]
pub struct ObservationRequest {
    prompt: String,
    width: usize,
    height: usize,
    seed: u64,
}

impl ObservationRequest {
    pub fn accept(options: &Options) -> Result<Self> {
        if !options.observation {
            return Err(Error::Config("去噪观察尚未开启".into()));
        }
        options.validate()?;
        if !options.images.is_empty() {
            return Err(Error::Config(
                "去噪观察不接受参考图；清除参考图，或关闭观察".into(),
            ));
        }
        if options.negative_prompt.is_some() {
            return Err(Error::Config(
                "去噪观察不接受负向提示词；清空负向提示词，或关闭观察".into(),
            ));
        }
        if options.true_cfg_scale > 1.0 {
            return Err(Error::Config(
                "去噪观察要求引导关闭；关闭负向引导，或关闭观察".into(),
            ));
        }
        let (width, height) = options.output_dimensions()?;
        Ok(Self {
            prompt: options.prompt.clone(),
            width,
            height,
            seed: options.seed,
        })
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn canvas(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn guidance_enabled(&self) -> bool {
        false
    }
}
