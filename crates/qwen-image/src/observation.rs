use super::{Options, gaussian_noise};
use crate::projection::{ConditionProjection, MiniCpm35bProjection, ProjectionSource};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationRow {
    Official,
    MiniCpm1b,
    MiniCpm35b,
}

/// One trajectory inside a denoising observation. Histories are not shared.
pub struct Trajectory {
    row: ObservationRow,
    prompt: String,
    seed: u64,
    width: usize,
    height: usize,
    initial_noise: Vec<f32>,
    attention_history: u64,
}

impl Trajectory {
    pub fn row(&self) -> ObservationRow {
        self.row
    }

    pub fn is_official_text(&self) -> bool {
        self.row == ObservationRow::Official
    }

    pub fn is_minicpm_v_4_7_1b(&self) -> bool {
        self.row == ObservationRow::MiniCpm1b
    }

    pub fn is_minicpm_v_4_7_35b(&self) -> bool {
        self.row == ObservationRow::MiniCpm35b
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn canvas(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn initial_noise(&self) -> &[f32] {
        &self.initial_noise
    }

    pub fn attention_history(&self) -> u64 {
        self.attention_history
    }

    pub fn guidance_enabled(&self) -> bool {
        false
    }
}

/// Official text row, then MiniCPM-V-4.7-1B, then MiniCPM-V-4.7-35B-A3B-int4.
pub struct DenoisingObservation {
    trajectories: [Trajectory; 3],
    cells: [Vec<ObservationCell>; 3],
    projection: Option<ConditionProjection>,
    projection_35b: Option<MiniCpm35bProjection>,
}

impl DenoisingObservation {
    pub fn begin(request: ObservationRequest, steps: usize) -> Result<Self> {
        if steps != DEFAULT_TURBO_STEPS {
            return Err(Error::Config(format!(
                "去噪观察要求默认 Turbo 的 {DEFAULT_TURBO_STEPS} 个去噪步，当前调度是 {steps} 步"
            )));
        }
        let (width, height) = request.canvas();
        let noise = latent_noise(LATENT_CHANNELS, height, width, request.seed());
        let prompt = request.prompt().to_string();
        let seed = request.seed();
        Ok(Self {
            trajectories: [
                Trajectory {
                    row: ObservationRow::Official,
                    prompt: prompt.clone(),
                    seed,
                    width,
                    height,
                    initial_noise: noise.clone(),
                    attention_history: 1,
                },
                Trajectory {
                    row: ObservationRow::MiniCpm1b,
                    prompt: prompt.clone(),
                    seed,
                    width,
                    height,
                    initial_noise: noise.clone(),
                    attention_history: 2,
                },
                Trajectory {
                    row: ObservationRow::MiniCpm35b,
                    prompt,
                    seed,
                    width,
                    height,
                    initial_noise: noise,
                    attention_history: 3,
                },
            ],
            cells: [Vec::new(), Vec::new(), Vec::new()],
            projection: None,
            projection_35b: None,
        })
    }

    /// Refuse a run that would draw with an untrained map or a JSON rewrite.
    pub fn open(
        request: ObservationRequest,
        steps: usize,
        minicpm_1b: ProjectionSource<'_>,
        minicpm_35b: ProjectionSource<'_>,
    ) -> Result<Self> {
        let projection = match minicpm_1b {
            ProjectionSource::Weights(path) => ConditionProjection::load(path)?,
            ProjectionSource::Untrained => {
                return Err(Error::Config(
                    "去噪观察不能用未训练的 MiniCPM-V-4.7-1B 投影出图".into(),
                ));
            }
            ProjectionSource::JsonRewrite => {
                return Err(Error::Config("去噪观察不会退回 JSON 改写".into()));
            }
        };
        let projection_35b = match minicpm_35b {
            ProjectionSource::Weights(path) => MiniCpm35bProjection::load(path)?,
            ProjectionSource::Untrained => {
                return Err(Error::Config(
                    "去噪观察不能用未训练的 MiniCPM-V-4.7-35B-A3B-int4 投影出图".into(),
                ));
            }
            ProjectionSource::JsonRewrite => {
                return Err(Error::Config("去噪观察不会退回 JSON 改写".into()));
            }
        };
        let mut observation = Self::begin(request, steps)?;
        observation.projection = Some(projection);
        observation.projection_35b = Some(projection_35b);
        Ok(observation)
    }

    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    pub fn cells(&self, row: ObservationRow) -> &[ObservationCell] {
        match row {
            ObservationRow::Official => &self.cells[0],
            ObservationRow::MiniCpm1b => &self.cells[1],
            ObservationRow::MiniCpm35b => &self.cells[2],
        }
    }

    pub fn retain(
        &mut self,
        row: ObservationRow,
        step: usize,
        image: image::RgbaImage,
    ) -> Result<()> {
        let (width, height) = self.trajectories[0].canvas();
        let cells = match row {
            ObservationRow::Official => &mut self.cells[0],
            ObservationRow::MiniCpm1b => &mut self.cells[1],
            ObservationRow::MiniCpm35b => &mut self.cells[2],
        };
        if step != cells.len() + 1 || step > DEFAULT_TURBO_STEPS {
            return Err(Error::Config(
                "去噪观察按步留下格子，不能替换已经出现的画面".into(),
            ));
        }
        if image.width() as usize != width || image.height() as usize != height {
            return Err(Error::Config(format!(
                "观察格须按请求画布 {width}×{height} 解码"
            )));
        }
        cells.push(ObservationCell { step, image });
        Ok(())
    }

    pub fn projection(&self) -> Option<&ConditionProjection> {
        self.projection.as_ref()
    }

    pub fn projection_35b(&self) -> Option<&MiniCpm35bProjection> {
        self.projection_35b.as_ref()
    }

    pub fn uses_negative_guidance(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub enum RequestKind {
    Ordinary { negative_guidance: bool },
    OfficialObservation(ObservationRequest),
}

impl RequestKind {
    pub fn trajectories(&self) -> usize {
        match self {
            Self::Ordinary { .. } => 1,
            Self::OfficialObservation(_) => 3,
        }
    }

    pub fn retained_decodes(&self) -> usize {
        match self {
            Self::Ordinary { .. } => 1,
            Self::OfficialObservation(_) => DEFAULT_TURBO_STEPS * 3,
        }
    }

    pub fn negative_guidance(&self) -> bool {
        match self {
            Self::Ordinary { negative_guidance } => *negative_guidance,
            Self::OfficialObservation(_) => false,
        }
    }

    pub fn loads_minicpm_v(&self) -> bool {
        matches!(self, Self::OfficialObservation(_))
    }

    pub fn loads_minicpm_35b(&self) -> bool {
        matches!(self, Self::OfficialObservation(_))
    }

    pub fn reads_projection_weights(&self) -> bool {
        matches!(self, Self::OfficialObservation(_))
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
#[derive(Debug, Clone)]
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
