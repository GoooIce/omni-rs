use super::{Options, gaussian_noise};
use crate::projection::{ConditionProjection, MiniCpm35bProjection, ProjectionSource};
use crate::{Error, Result};
use std::path::Path;

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

impl ObservationRow {
    /// The only row sequence: official text, fitted 1B, then fitted 35B.
    pub fn order() -> [Self; 3] {
        [Self::Official, Self::MiniCpm1b, Self::MiniCpm35b]
    }

    fn slot(self) -> usize {
        Self::order()
            .iter()
            .position(|row| *row == self)
            .expect("observation row is missing from order()")
    }
}

/// How one observation row gets its condition. The official row uses the text
/// tower `Model` encodes; a fitted row is that row's projection of its last layer.
pub(super) enum RowCondition {
    OfficialText,
    Projected {
        values: Vec<f32>,
        label: &'static str,
    },
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

/// Three trajectories in [`ObservationRow::order`], sharing one initial noise.
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
        let seed = request.seed();
        let trajectories = std::array::from_fn(|index| {
            let row = ObservationRow::order()[index];
            Trajectory {
                row,
                prompt: request.prompt().to_string(),
                seed,
                width,
                height,
                initial_noise: noise.clone(),
                attention_history: (index as u64) + 1,
            }
        });
        Ok(Self {
            trajectories,
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
        &self.cells[row.slot()]
    }

    pub fn retain(
        &mut self,
        row: ObservationRow,
        step: usize,
        image: image::RgbaImage,
    ) -> Result<()> {
        let (width, height) = self.trajectories[0].canvas();
        let cells = &mut self.cells[row.slot()];
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

    /// Fail before denoising when either fitted map is missing.
    pub(super) fn require_fitted_projections(&self) -> Result<()> {
        self.fitted_1b()?;
        self.fitted_35b()?;
        Ok(())
    }

    /// Condition for `row`. A fitted row reads its last layer and drops that checkpoint.
    pub(super) fn row_condition(
        &self,
        row: ObservationRow,
        prompt: &str,
        official_length: Option<usize>,
        model_root: &Path,
    ) -> Result<RowCondition> {
        match row {
            ObservationRow::Official => Ok(RowCondition::OfficialText),
            ObservationRow::MiniCpm1b => Ok(RowCondition::Projected {
                values: self.project_1b(prompt, aligned_length(official_length)?, model_root)?,
                label: "MiniCPM-V-4.7-1B",
            }),
            ObservationRow::MiniCpm35b => Ok(RowCondition::Projected {
                values: self.project_35b(prompt, aligned_length(official_length)?, model_root)?,
                label: "MiniCPM-V-4.7-35B-A3B-int4",
            }),
        }
    }

    fn fitted_1b(&self) -> Result<&ConditionProjection> {
        self.projection.as_ref().ok_or_else(|| {
            Error::Config(
                "去噪观察缺少 MiniCPM-V-4.7-1B 投影权重，不能用未训练投影出图，也不会退回 JSON 改写"
                    .into(),
            )
        })
    }

    fn fitted_35b(&self) -> Result<&MiniCpm35bProjection> {
        self.projection_35b.as_ref().ok_or_else(|| {
            Error::Config(
                "去噪观察缺少 MiniCPM-V-4.7-35B-A3B-int4 投影权重，不能用未训练投影出图，也不会退回 JSON 改写"
                    .into(),
            )
        })
    }

    fn project_1b(
        &self,
        prompt: &str,
        official_length: usize,
        model_root: &Path,
    ) -> Result<Vec<f32>> {
        let dir = crate::minicpm_condition::default_minicpm_1b_dir(model_root);
        let checkpoint = crate::minicpm_condition::MiniCpm1bText::load(&dir)?;
        let layer = checkpoint.last_layer(prompt)?;
        drop(checkpoint);
        let projected = self
            .fitted_1b()?
            .project_last_layer(&layer, official_length)?;
        Ok(projected.values().to_vec())
    }

    fn project_35b(
        &self,
        prompt: &str,
        official_length: usize,
        model_root: &Path,
    ) -> Result<Vec<f32>> {
        let dir = crate::minicpm_condition::default_minicpm_35b_dir(model_root);
        let checkpoint = crate::minicpm_condition::MiniCpm35bText::load(&dir)?;
        let layer = checkpoint.last_layer(prompt)?;
        drop(checkpoint);
        let projected = self
            .fitted_35b()?
            .project_last_layer(&layer, official_length)?;
        Ok(projected.values().to_vec())
    }

    pub fn uses_negative_guidance(&self) -> bool {
        false
    }
}

fn aligned_length(official_length: Option<usize>) -> Result<usize> {
    official_length
        .ok_or_else(|| Error::Config("官方文本条件的宽度不是 4096，无法对齐投影条件".into()))
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
