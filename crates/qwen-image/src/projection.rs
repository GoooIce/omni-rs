//! Map a MiniCPM-V-4.7-1B last-layer sequence onto an official text condition.
//!
//! The official condition is 4096 wide. Positions line up with that sequence,
//! so a shorter or longer last layer is resampled before the linear map.
//! This path never reads an image and never calls JSON prompt rewrite.

use crate::{Error, Result};
use std::path::{Path, PathBuf};

pub const CONDITION_WIDTH: usize = 4096;
const TRAIN_PROMPTS: usize = 16;
/// Diagonal penalty, as a fraction of the mean feature energy.
/// 10_000 left weights near zero, so every prompt became the training-set mean
/// and denoised to the same picture. 1 keeps the input able to move the condition.
const RIDGE_SCALE: f64 = 1.0;

/// Twenty-four pure text-to-image prompts. The first sixteen update the projection.
const TEXT_TO_IMAGE_PROMPTS: [&str; 24] = [
    "一只白瓷茶壶放在木桌上，窗边有柔和的晨光",
    "雪地里的红色狐狸，背景是模糊的松林",
    "海边栈道上的黄色雨衣，远处有灯塔",
    "竹林间的石灯，夜里有薄雾",
    "一碗拉面的特写，热气上升，木筷搁在碗边",
    "老式自行车靠在青砖墙边，墙上有爬山虎",
    "宇宙中的小型空间站，舷窗透出暖光",
    "雨后的城市十字路口，地面反光，行人撑着透明伞",
    "陶罐里的一枝白梅，背景是素色宣纸",
    "山间木屋的烟囱冒着白烟，屋顶有薄雪",
    "玻璃温室里的热带植物，阳光穿过水雾",
    "深夜书房的台灯，桌上摊开一本线装书",
    "沙漠中的骆驼商队，远处是沙丘和落日",
    "儿童风筝在春天的草地上，风筝是鲤鱼形状",
    "黑色钢琴放在空旷的白色展厅中央",
    "渔港清晨的木船，筐里是银色的鱼",
    "紫藤花覆盖的拱门，石板路上有光斑",
    "热气球漂在金色麦田上空",
    "陶瓷杯里的拿铁拉花，旁边是一张旧报纸",
    "雪夜的红色邮筒，周围是安静的住宅街",
    "悬崖边的观景台，海面是深蓝色",
    "工笔风格的锦鲤在莲叶间游动",
    "旧火车站的挂钟，月台上只有一只行李箱",
    "月亮升上稻田，田里有一条窄窄的田埂",
];

pub fn text_to_image_split() -> (&'static [&'static str], &'static [&'static str]) {
    (
        &TEXT_TO_IMAGE_PROMPTS[..TRAIN_PROMPTS],
        &TEXT_TO_IMAGE_PROMPTS[TRAIN_PROMPTS..],
    )
}

/// Official text-condition cache inside the caller's `artifacts/` tree.
pub fn default_official_condition_cache() -> PathBuf {
    PathBuf::from("artifacts/minicpm-v-4.7-1b/cache")
}

const PROMPT_MANIFEST: &str = "prompt-manifest.txt";

/// Records `prompts` in order beside numbered condition-cache files.
pub fn write_prompt_manifest(cache: &Path, prompts: &[&str]) -> Result<()> {
    if prompts
        .iter()
        .any(|prompt| prompt.contains('\n') || prompt.contains('\r'))
    {
        return Err(Error::Config("提示词清单的每一行只能是一条提示词".into()));
    }
    std::fs::create_dir_all(cache)?;
    let mut text = String::new();
    for prompt in prompts {
        text.push_str(prompt);
        text.push('\n');
    }
    std::fs::write(cache.join(PROMPT_MANIFEST), text)?;
    Ok(())
}

fn prompt_manifest_matches(cache: &Path, prompts: &[&str]) -> bool {
    let Ok(text) = std::fs::read_to_string(cache.join(PROMPT_MANIFEST)) else {
        return false;
    };
    let mut listed: Vec<&str> = text.split('\n').collect();
    if listed.last() == Some(&"") {
        listed.pop();
    }
    listed == prompts
}

/// `official-NN.bin` or `last-layer-NN.bin` when the manifest lists `prompts` in order.
pub fn matching_cache_file(
    cache: &Path,
    kind: &str,
    index: usize,
    prompts: &[&str],
) -> Option<PathBuf> {
    if !prompt_manifest_matches(cache, prompts) {
        return None;
    }
    let path = cache.join(format!("{kind}-{index:02}.bin"));
    path.is_file().then_some(path)
}

pub fn default_projection_weights_path() -> PathBuf {
    std::env::var_os("OMNI_1B_PROJECTION")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("artifacts/minicpm-v-4.7-1b/text-condition-projection.bin")
        })
}

/// Where an observation looks for the fitted 1B map.
pub enum ProjectionSource<'a> {
    Weights(&'a Path),
    Untrained,
    JsonRewrite,
}

/// One sequence laid out row-major as `length * width`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    length: usize,
    width: usize,
    values: Vec<f32>,
}

impl Sequence {
    pub fn last_layer(length: usize, width: usize, values: Vec<f32>) -> Result<Self> {
        if length == 0 || width == 0 || values.len() != length * width {
            return Err(Error::Config(
                "最后一层序列的长度、宽度和数值数量不一致".into(),
            ));
        }
        Ok(Self {
            length,
            width,
            values,
        })
    }

    pub fn length(&self) -> usize {
        self.length
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn get(&self, position: usize, dim: usize) -> f32 {
        self.values[position * self.width + dim]
    }
}

/// Linear map from a model's last-layer width onto the official 4096-d condition.
#[derive(Clone, Debug)]
pub struct ConditionProjection {
    in_width: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    fitted: bool,
}

impl ConditionProjection {
    pub fn from_linear(weight: &[f32], bias: &[f32]) -> Result<Self> {
        if bias.len() != CONDITION_WIDTH || weight.is_empty() || weight.len() % CONDITION_WIDTH != 0
        {
            return Err(Error::Config(
                "投影须把最后一层映射到 4096 维的官方文本条件".into(),
            ));
        }
        Ok(Self {
            in_width: weight.len() / CONDITION_WIDTH,
            weight: weight.to_vec(),
            bias: bias.to_vec(),
            fitted: false,
        })
    }

    pub fn in_width(&self) -> usize {
        self.in_width
    }

    /// Project a text-prompt last-layer sequence. `official_length` is that
    /// prompt's official text-condition length.
    pub fn project_last_layer(
        &self,
        sequence: &Sequence,
        official_length: usize,
    ) -> Result<Sequence> {
        if sequence.width() != self.in_width {
            return Err(Error::Config(
                "最后一层序列的宽度与 MiniCPM-V-4.7-1B 投影不一致".into(),
            ));
        }
        if official_length == 0 {
            return Err(Error::Config("官方文本条件的长度不能为 0".into()));
        }
        let aligned = resample(sequence, official_length);
        let mut values = vec![0.0; official_length * CONDITION_WIDTH];
        for position in 0..official_length {
            for dim in 0..CONDITION_WIDTH {
                let mut value = self.bias[dim];
                for input in 0..self.in_width {
                    value += aligned[position * self.in_width + input]
                        * self.weight[input * CONDITION_WIDTH + dim];
                }
                values[position * CONDITION_WIDTH + dim] = value;
            }
        }
        Sequence::last_layer(official_length, CONDITION_WIDTH, values)
    }

    pub fn project_image(&self, _image: &[u8], _official_length: usize) -> Result<Sequence> {
        Err(Error::Config("MiniCPM-V-4.7-1B 的投影条件不读图像".into()))
    }

    pub fn project_json_rewrite(
        &self,
        _rewritten: &str,
        _official_length: usize,
    ) -> Result<Sequence> {
        Err(Error::Config(
            "MiniCPM-V-4.7-1B 的投影条件不调用 JSON 改写".into(),
        ))
    }

    /// All-zero map. The same vector every time, before any prompt updates it.
    pub fn deterministic_init(in_width: usize) -> Result<Self> {
        if in_width == 0 {
            return Err(Error::Config("投影输入宽度不能为 0".into()));
        }
        Self::from_linear(
            &vec![0.0; in_width * CONDITION_WIDTH],
            &vec![0.0; CONDITION_WIDTH],
        )
    }

    /// Update the map from training pairs only. Each pair is a last-layer
    /// sequence and the official text condition for the same prompt.
    pub fn fit(train: &[(Sequence, Sequence)]) -> Result<Self> {
        if train.is_empty() {
            return Err(Error::Config("投影拟合至少需要一条训练提示词".into()));
        }
        let in_width = train[0].0.width();
        if in_width == 0 {
            return Err(Error::Config("投影输入宽度不能为 0".into()));
        }
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for (input, target) in train {
            if input.width() != in_width
                || target.width() != CONDITION_WIDTH
                || target.length() == 0
            {
                return Err(Error::Config(
                    "拟合样本须是最后一层序列，目标是同提示词的 4096 维官方文本条件".into(),
                ));
            }
            inputs.extend(resample(input, target.length()));
            targets.extend_from_slice(target.values());
        }
        let count = targets.len() / CONDITION_WIDTH;
        let (weight, bias) = ridge_fit(&inputs, &targets, count, in_width, CONDITION_WIDTH);
        let mut fitted = Self::from_linear(&weight, &bias)?;
        fitted.fitted = true;
        Ok(fitted)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.fitted {
            return Err(Error::Config(
                "去噪观察不能用未训练的 MiniCPM-V-4.7-1B 投影出图".into(),
            ));
        }
        reject_nonfinite_weights(&self.weight, &self.bias)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::with_capacity(20 + (self.weight.len() + self.bias.len()) * 4);
        bytes.extend_from_slice(b"OM1B");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(self.in_width as u32).to_le_bytes());
        bytes.extend_from_slice(&(CONDITION_WIDTH as u32).to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for value in self.weight.iter().chain(self.bias.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(missing_projection());
        }
        let bytes = std::fs::read(path)?;
        let mut cursor = 0usize;
        let mut take = |len: usize| -> Result<&[u8]> {
            let end = cursor + len;
            let slice = bytes
                .get(cursor..end)
                .ok_or_else(|| Error::Config("MiniCPM-V-4.7-1B 投影权重无法读取".into()))?;
            cursor = end;
            Ok(slice)
        };
        if take(4)? != b"OM1B" {
            return Err(Error::Config("MiniCPM-V-4.7-1B 投影权重无法读取".into()));
        }
        let version = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let in_width = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let out_width = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let fitted = u32::from_le_bytes(take(4)?.try_into().unwrap());
        if version != 1 || out_width != CONDITION_WIDTH || in_width == 0 {
            return Err(Error::Config("MiniCPM-V-4.7-1B 投影权重无法读取".into()));
        }
        if fitted != 1 {
            return Err(Error::Config(
                "去噪观察不能用未训练的 MiniCPM-V-4.7-1B 投影出图".into(),
            ));
        }
        let mut weight = vec![0.0; in_width * CONDITION_WIDTH];
        let mut bias = vec![0.0; CONDITION_WIDTH];
        for value in weight.iter_mut().chain(bias.iter_mut()) {
            *value = f32::from_le_bytes(take(4)?.try_into().unwrap());
        }
        if cursor != bytes.len() {
            return Err(Error::Config("MiniCPM-V-4.7-1B 投影权重无法读取".into()));
        }
        reject_nonfinite_weights(&weight, &bias)?;
        Ok(Self {
            in_width,
            weight,
            bias,
            fitted: true,
        })
    }

    pub fn mean_squared_error(&self, pairs: &[(Sequence, Sequence)]) -> Result<f32> {
        if pairs.is_empty() {
            return Err(Error::Config("均方误差至少需要一条条件".into()));
        }
        let mut total = 0.0f64;
        let mut count = 0usize;
        for (input, target) in pairs {
            if target.width() != CONDITION_WIDTH {
                return Err(Error::Config(
                    "条件均方误差只在 4096 维官方文本条件上计算".into(),
                ));
            }
            let predicted = self.project_last_layer(input, target.length())?;
            for (pred, truth) in predicted.values().iter().zip(target.values()) {
                let delta = f64::from(*pred) - f64::from(*truth);
                total += delta * delta;
                count += 1;
            }
        }
        Ok((total / count as f64) as f32)
    }
}

fn reject_nonfinite_weights(weight: &[f32], bias: &[f32]) -> Result<()> {
    if weight
        .iter()
        .chain(bias.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Error::Config("投影权重含有非有限数值".into()));
    }
    Ok(())
}

fn missing_projection() -> Error {
    Error::Config(
        "去噪观察缺少 MiniCPM-V-4.7-1B 投影权重，不能用未训练投影出图，也不会退回 JSON 改写".into(),
    )
}

/// Linear map from a MiniCPM-V-4.7-35B-A3B-int4 last layer onto the official condition.
///
/// The local checkpoint's text hidden size is 2048. Fitting reuses
/// [`ConditionProjection`]; this type only fixes that width and its own weight file.
#[derive(Clone)]
pub struct MiniCpm35bProjection {
    inner: ConditionProjection,
}

impl MiniCpm35bProjection {
    /// Hidden size of the local MiniCPM-V-4.7-35B-A3B-int4 text tower.
    pub const HIDDEN: usize = 2048;

    pub fn from_linear(weight: &[f32], bias: &[f32]) -> Result<Self> {
        let inner = ConditionProjection::from_linear(weight, bias)?;
        if inner.in_width != Self::HIDDEN {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影的输入宽度必须是 2048".into(),
            ));
        }
        Ok(Self { inner })
    }

    pub fn project_last_layer(
        &self,
        sequence: &Sequence,
        official_length: usize,
    ) -> Result<Sequence> {
        if sequence.width() != Self::HIDDEN {
            return Err(Error::Config(
                "最后一层序列的宽度与 MiniCPM-V-4.7-35B-A3B-int4 投影不一致，应为 2048".into(),
            ));
        }
        self.inner.project_last_layer(sequence, official_length)
    }

    pub fn project_image(&self, _image: &[u8], _official_length: usize) -> Result<Sequence> {
        Err(Error::Config(
            "MiniCPM-V-4.7-35B-A3B-int4 的投影条件不读图像".into(),
        ))
    }

    pub fn project_json_rewrite(
        &self,
        _rewritten: &str,
        _official_length: usize,
    ) -> Result<Sequence> {
        Err(Error::Config(
            "MiniCPM-V-4.7-35B-A3B-int4 的投影条件不调用 JSON 改写".into(),
        ))
    }

    /// All-zero 2048→4096 map, before any of the shared training prompts update it.
    pub fn deterministic_init() -> Result<Self> {
        Ok(Self {
            inner: ConditionProjection::deterministic_init(Self::HIDDEN)?,
        })
    }

    /// Fit on the shared text-to-image pairs. Inputs must already be width 2048.
    pub fn fit(train: &[(Sequence, Sequence)]) -> Result<Self> {
        if train.iter().any(|(input, _)| input.width() != Self::HIDDEN) {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影拟合的最后一层宽度必须是 2048".into(),
            ));
        }
        Ok(Self {
            inner: ConditionProjection::fit(train)?,
        })
    }

    pub fn mean_squared_error(&self, pairs: &[(Sequence, Sequence)]) -> Result<f32> {
        self.inner.mean_squared_error(pairs)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if !self.inner.fitted {
            return Err(Error::Config(
                "去噪观察不能用未训练的 MiniCPM-V-4.7-35B-A3B-int4 投影出图".into(),
            ));
        }
        if self.inner.in_width != Self::HIDDEN {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影的输入宽度必须是 2048".into(),
            ));
        }
        reject_nonfinite_weights(&self.inner.weight, &self.inner.bias)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut bytes =
            Vec::with_capacity(20 + (self.inner.weight.len() + self.inner.bias.len()) * 4);
        bytes.extend_from_slice(b"OM35");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(self.inner.in_width as u32).to_le_bytes());
        bytes.extend_from_slice(&(CONDITION_WIDTH as u32).to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for value in self.inner.weight.iter().chain(self.inner.bias.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(missing_35b_projection());
        }
        let bytes = std::fs::read(path)?;
        let mut cursor = 0usize;
        let mut take = |len: usize| -> Result<&[u8]> {
            let end = cursor + len;
            let slice = bytes.get(cursor..end).ok_or_else(|| {
                Error::Config("MiniCPM-V-4.7-35B-A3B-int4 投影权重无法读取".into())
            })?;
            cursor = end;
            Ok(slice)
        };
        if take(4)? != b"OM35" {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影权重无法读取".into(),
            ));
        }
        let version = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let in_width = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let out_width = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let fitted = u32::from_le_bytes(take(4)?.try_into().unwrap());
        if version != 1 || out_width != CONDITION_WIDTH || in_width != Self::HIDDEN {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影权重无法读取".into(),
            ));
        }
        if fitted != 1 {
            return Err(Error::Config(
                "去噪观察不能用未训练的 MiniCPM-V-4.7-35B-A3B-int4 投影出图".into(),
            ));
        }
        let mut weight = vec![0.0; in_width * CONDITION_WIDTH];
        let mut bias = vec![0.0; CONDITION_WIDTH];
        for value in weight.iter_mut().chain(bias.iter_mut()) {
            *value = f32::from_le_bytes(take(4)?.try_into().unwrap());
        }
        if cursor != bytes.len() {
            return Err(Error::Config(
                "MiniCPM-V-4.7-35B-A3B-int4 投影权重无法读取".into(),
            ));
        }
        reject_nonfinite_weights(&weight, &bias)?;
        Self::from_linear(&weight, &bias).map(|mut projection| {
            projection.inner.fitted = true;
            projection
        })
    }
}

pub fn default_35b_projection_weights_path() -> PathBuf {
    std::env::var_os("OMNI_35B_PROJECTION")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("artifacts/minicpm-v-4.7-35b-a3b/text-condition-projection.bin")
        })
}

fn missing_35b_projection() -> Error {
    Error::Config(
        "去噪观察缺少 MiniCPM-V-4.7-35B-A3B-int4 投影权重，不能用未训练投影出图，也不会退回 JSON 改写"
            .into(),
    )
}

fn ridge_fit(
    inputs: &[f32],
    targets: &[f32],
    count: usize,
    in_width: usize,
    outputs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut x_mean = vec![0.0f64; in_width];
    let mut y_mean = vec![0.0f64; outputs];
    for row in 0..count {
        for input in 0..in_width {
            x_mean[input] += f64::from(inputs[row * in_width + input]);
        }
        for dim in 0..outputs {
            y_mean[dim] += f64::from(targets[row * outputs + dim]);
        }
    }
    let rows = count as f64;
    for value in &mut x_mean {
        *value /= rows;
    }
    for value in &mut y_mean {
        *value /= rows;
    }

    let mut gram = vec![0.0f64; in_width * in_width];
    let mut cross = vec![0.0f64; outputs * in_width];
    for row in 0..count {
        let mut centered = vec![0.0f64; in_width];
        for input in 0..in_width {
            centered[input] = f64::from(inputs[row * in_width + input]) - x_mean[input];
        }
        for left in 0..in_width {
            for right in 0..in_width {
                gram[left * in_width + right] += centered[left] * centered[right];
            }
            for dim in 0..outputs {
                let centered_target = f64::from(targets[row * outputs + dim]) - y_mean[dim];
                cross[dim * in_width + left] += centered[left] * centered_target;
            }
        }
    }
    let trace: f64 = (0..in_width)
        .map(|input| gram[input * in_width + input])
        .sum();
    let lambda = RIDGE_SCALE * (trace / in_width as f64).max(1e-6);
    for input in 0..in_width {
        gram[input * in_width + input] += lambda;
    }
    cholesky(&mut gram, in_width);

    let mut weight = vec![0.0f32; in_width * outputs];
    let mut bias = vec![0.0f32; outputs];
    let mut solved = vec![0.0f64; in_width];
    for dim in 0..outputs {
        for input in 0..in_width {
            solved[input] = cross[dim * in_width + input];
        }
        solve_cholesky(&gram, &mut solved, in_width);
        let mut shift = y_mean[dim];
        for input in 0..in_width {
            weight[input * outputs + dim] = solved[input] as f32;
            shift -= x_mean[input] * solved[input];
        }
        bias[dim] = shift as f32;
    }
    (weight, bias)
}

fn cholesky(matrix: &mut [f64], width: usize) {
    for row in 0..width {
        for col in 0..=row {
            let mut sum = matrix[row * width + col];
            for k in 0..col {
                sum -= matrix[row * width + k] * matrix[col * width + k];
            }
            if row == col {
                matrix[row * width + col] = sum.max(1e-12).sqrt();
            } else {
                matrix[row * width + col] = sum / matrix[col * width + col];
            }
        }
    }
}

fn solve_cholesky(factor: &[f64], values: &mut [f64], width: usize) {
    for row in 0..width {
        let mut sum = values[row];
        for col in 0..row {
            sum -= factor[row * width + col] * values[col];
        }
        values[row] = sum / factor[row * width + row];
    }
    for row in (0..width).rev() {
        let mut sum = values[row];
        for col in (row + 1)..width {
            sum -= factor[col * width + row] * values[col];
        }
        values[row] = sum / factor[row * width + row];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOLDOUT: [&str; 8] = [
        "紫藤花覆盖的拱门，石板路上有光斑",
        "热气球漂在金色麦田上空",
        "陶瓷杯里的拿铁拉花，旁边是一张旧报纸",
        "雪夜的红色邮筒，周围是安静的住宅街",
        "悬崖边的观景台，海面是深蓝色",
        "工笔风格的锦鲤在莲叶间游动",
        "旧火车站的挂钟，月台上只有一只行李箱",
        "月亮升上稻田，田里有一条窄窄的田埂",
    ];

    /// Nonzero entries of `output = weight * input + bias`. Every other entry is 0.
    const KNOWN_WEIGHTS: [(usize, usize, f32); 2] = [(0, 0, 0.5), (1, 2, -0.25)];
    const KNOWN_BIAS: [(usize, f32); 2] = [(0, 0.75), (2, -0.5)];

    fn known_linear_pair(index: usize, in_width: usize) -> (Sequence, Sequence) {
        let length = 2;
        let mut input_values = vec![0.0; length * in_width];
        for position in 0..length {
            for dim in 0..in_width {
                input_values[position * in_width + dim] = match (index + position + dim) % 5 {
                    0 => -1.0,
                    1 => -0.5,
                    2 => 0.0,
                    3 => 0.5,
                    _ => 1.0,
                };
            }
        }
        let input = Sequence::last_layer(length, in_width, input_values).unwrap();
        let mut expected = vec![0.0; length * CONDITION_WIDTH];
        for position in 0..length {
            for &(dim, bias) in &KNOWN_BIAS {
                expected[position * CONDITION_WIDTH + dim] = bias;
            }
            for &(in_dim, out_dim, weight) in &KNOWN_WEIGHTS {
                expected[position * CONDITION_WIDTH + out_dim] +=
                    input.get(position, in_dim) * weight;
            }
        }
        let target = Sequence::last_layer(length, CONDITION_WIDTH, expected).unwrap();
        (input, target)
    }

    fn assert_known_map_anchor(pair: &(Sequence, Sequence)) {
        assert_eq!(pair.0.length(), 2);
        assert_eq!(pair.1.width(), CONDITION_WIDTH);
        assert_eq!(pair.0.get(0, 0), -1.0);
        assert_eq!(pair.0.get(0, 1), -0.5);
        assert_eq!(pair.1.get(0, 0), 0.25);
        assert_eq!(pair.1.get(0, 1), 0.0);
        assert_eq!(pair.1.get(0, 2), -0.375);
    }

    fn assert_holdout_beats_init_within(init_mse: f32, fitted_mse: f32) {
        assert!(
            fitted_mse < init_mse,
            "fitted {fitted_mse} vs init {init_mse}"
        );
        assert!(
            fitted_mse < 1.0e-4,
            "fitted holdout mse {fitted_mse} is not below 1e-4"
        );
    }

    #[test]
    fn minicpm_35b_projection_matches_the_official_condition_shape() {
        const HIDDEN: usize = 2048;
        assert_eq!(MiniCpm35bProjection::HIDDEN, HIDDEN);

        let mut weight = vec![0.0; HIDDEN * CONDITION_WIDTH];
        weight[0] = 2.0;
        let mut bias = vec![0.0; CONDITION_WIDTH];
        bias[0] = 1.0;
        let projection = MiniCpm35bProjection::from_linear(&weight, &bias).unwrap();
        assert!(
            MiniCpm35bProjection::from_linear(
                &vec![0.0; 1024 * CONDITION_WIDTH],
                &vec![0.0; CONDITION_WIDTH],
            )
            .is_err()
        );

        let mut aligned_values = vec![0.0; 2 * HIDDEN];
        aligned_values[0] = 3.0;
        aligned_values[HIDDEN] = 4.0;
        let aligned = Sequence::last_layer(2, HIDDEN, aligned_values).unwrap();
        let same_length = projection.project_last_layer(&aligned, 2).unwrap();
        assert_eq!(same_length.length(), 2);
        assert_eq!(same_length.width(), CONDITION_WIDTH);
        assert_eq!(same_length.get(0, 0), 7.0);
        assert_eq!(same_length.get(1, 0), 9.0);
        assert_eq!(same_length.get(0, 1), 0.0);

        let mut stretched_values = vec![0.0; 2 * HIDDEN];
        stretched_values[HIDDEN] = 10.0;
        let stretched = Sequence::last_layer(2, HIDDEN, stretched_values).unwrap();
        let output = projection.project_last_layer(&stretched, 3).unwrap();
        assert_eq!(output.length(), 3);
        assert_eq!(output.width(), CONDITION_WIDTH);
        assert_eq!(output.get(0, 0), 1.0);
        assert_eq!(output.get(1, 0), 11.0);
        assert_eq!(output.get(2, 0), 21.0);
        assert_eq!(output.get(2, 4095), 0.0);

        let narrow = Sequence::last_layer(1, 1024, vec![0.0; 1024]).unwrap();
        let mismatch = projection
            .project_last_layer(&narrow, 1)
            .unwrap_err()
            .to_string();
        assert!(mismatch.contains("2048"), "{mismatch}");

        let image = projection
            .project_image(&[9, 9, 9], 3)
            .unwrap_err()
            .to_string();
        assert!(image.contains("图像"), "{image}");
        let rewritten = projection
            .project_json_rewrite("{\"rewritten_prompt\":\"一只茶壶\"}", 3)
            .unwrap_err()
            .to_string();
        assert!(rewritten.contains("改写"), "{rewritten}");
    }

    #[test]
    fn minicpm_35b_holdout_condition_error_beats_its_untrained_init() {
        let (train_prompts, holdout_prompts) = text_to_image_split();
        assert_eq!(train_prompts.len(), 16);
        assert_eq!(holdout_prompts, HOLDOUT);
        assert_eq!(train_prompts[0], "一只白瓷茶壶放在木桌上，窗边有柔和的晨光");
        assert_eq!(train_prompts[15], "渔港清晨的木船，筐里是银色的鱼");
        assert!(train_prompts.iter().all(|prompt| !HOLDOUT.contains(prompt)));

        let hidden = MiniCpm35bProjection::HIDDEN;
        let train: Vec<_> = (0..train_prompts.len())
            .map(|index| known_linear_pair(index, hidden))
            .collect();
        let holdout: Vec<_> = (0..holdout_prompts.len())
            .map(|index| known_linear_pair(16 + index, hidden))
            .collect();
        assert_eq!(train[0].0.width(), hidden);
        assert_known_map_anchor(&train[0]);
        let init = MiniCpm35bProjection::deterministic_init().unwrap();
        let fitted = MiniCpm35bProjection::fit(&train).unwrap();
        let init_mse = init.mean_squared_error(&holdout).unwrap();
        let fitted_mse = fitted.mean_squared_error(&holdout).unwrap();
        assert_holdout_beats_init_within(init_mse, fitted_mse);
        let pixels = Sequence::last_layer(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert!(
            fitted
                .mean_squared_error(&[(holdout[0].0.clone(), pixels)])
                .is_err()
        );
        let narrow = known_linear_pair(0, 4);
        assert!(MiniCpm35bProjection::fit(&[narrow]).is_err());
    }

    #[test]
    fn holdout_condition_error_beats_the_untrained_projection() {
        let (train_prompts, holdout_prompts) = text_to_image_split();
        assert_eq!(train_prompts.len(), 16);
        assert_eq!(holdout_prompts, HOLDOUT);
        assert_eq!(train_prompts[0], "一只白瓷茶壶放在木桌上，窗边有柔和的晨光");
        assert_eq!(train_prompts[15], "渔港清晨的木船，筐里是银色的鱼");
        assert!(train_prompts.iter().all(|prompt| !HOLDOUT.contains(prompt)));

        let in_width = 4;
        let train: Vec<_> = (0..train_prompts.len())
            .map(|index| known_linear_pair(index, in_width))
            .collect();
        let holdout: Vec<_> = (0..holdout_prompts.len())
            .map(|index| known_linear_pair(16 + index, in_width))
            .collect();
        assert_eq!(train[0].0.width(), in_width);
        assert_known_map_anchor(&train[0]);
        let init = ConditionProjection::deterministic_init(in_width).unwrap();
        let fitted = ConditionProjection::fit(&train).unwrap();
        let init_mse = init.mean_squared_error(&holdout).unwrap();
        let fitted_mse = fitted.mean_squared_error(&holdout).unwrap();
        assert_holdout_beats_init_within(init_mse, fitted_mse);
    }

    #[test]
    fn numbered_cache_files_require_the_current_prompt_manifest() {
        let (train, holdout) = text_to_image_split();
        let prompts: Vec<&str> = train
            .iter()
            .copied()
            .chain(holdout.iter().copied())
            .collect();
        assert_eq!(prompts.len(), 24);

        let cache = std::env::temp_dir().join(format!(
            "omni-prompt-cache-{}-{}",
            std::process::id(),
            "manifest"
        ));
        let _ = std::fs::remove_dir_all(&cache);
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(cache.join("official-00.bin"), b"stale").unwrap();
        std::fs::write(cache.join("last-layer-00.bin"), b"stale").unwrap();

        assert!(
            matching_cache_file(&cache, "official", 0, &prompts).is_none(),
            "a missing manifest must not expose official cache files"
        );
        assert!(
            matching_cache_file(&cache, "last-layer", 0, &prompts).is_none(),
            "a missing manifest must not expose last-layer cache files"
        );

        let mut reordered = prompts.clone();
        reordered.swap(0, 1);
        write_prompt_manifest(&cache, &reordered).unwrap();
        assert!(matching_cache_file(&cache, "official", 0, &prompts).is_none());
        assert!(matching_cache_file(&cache, "last-layer", 0, &prompts).is_none());

        let mut shortened = prompts.clone();
        shortened.pop();
        write_prompt_manifest(&cache, &shortened).unwrap();
        assert!(matching_cache_file(&cache, "official", 0, &prompts).is_none());

        write_prompt_manifest(&cache, &prompts).unwrap();
        assert_eq!(
            matching_cache_file(&cache, "official", 0, &prompts).as_deref(),
            Some(cache.join("official-00.bin").as_path())
        );
        assert_eq!(
            matching_cache_file(&cache, "last-layer", 0, &prompts).as_deref(),
            Some(cache.join("last-layer-00.bin").as_path())
        );
        assert!(matching_cache_file(&cache, "official", 1, &prompts).is_none());
        assert!(matching_cache_file(&cache, "last-layer", 23, &prompts).is_none());

        let official_cache = default_official_condition_cache();
        assert!(official_cache.is_relative());
        assert_eq!(
            official_cache,
            PathBuf::from("artifacts/minicpm-v-4.7-1b/cache")
        );

        let _ = std::fs::remove_dir_all(&cache);
    }

    #[test]
    fn save_and_load_reject_nonfinite_projection_weights() {
        let dir =
            std::env::temp_dir().join(format!("omni-projection-finite-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut projection = ConditionProjection {
            in_width: 2,
            weight: vec![0.25; 2 * CONDITION_WIDTH],
            bias: vec![0.1; CONDITION_WIDTH],
            fitted: true,
        };
        let path = dir.join("1b.bin");
        projection.save(&path).unwrap();
        let loaded = ConditionProjection::load(&path).unwrap();
        assert_eq!(loaded.in_width, 2);
        assert_eq!(loaded.weight[0], 0.25);
        assert_eq!(loaded.bias[0], 0.1);

        projection.weight[0] = f32::NAN;
        let saved = projection.save(&path).unwrap_err().to_string();
        assert!(saved.contains("非有限"), "{saved}");
        projection.weight[0] = 0.25;
        projection.bias[1] = f32::INFINITY;
        let saved = projection.save(&path).unwrap_err().to_string();
        assert!(saved.contains("非有限"), "{saved}");
        projection.bias[1] = 0.1;

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
        let nan_path = dir.join("1b-nan.bin");
        std::fs::write(&nan_path, &bytes).unwrap();
        let loaded = ConditionProjection::load(&nan_path)
            .unwrap_err()
            .to_string();
        assert!(loaded.contains("非有限"), "{loaded}");

        bytes[20..24].copy_from_slice(&0.25f32.to_le_bytes());
        let bias_at = 20 + 2 * CONDITION_WIDTH * 4;
        bytes[bias_at..bias_at + 4].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
        let inf_path = dir.join("1b-inf.bin");
        std::fs::write(&inf_path, &bytes).unwrap();
        let loaded = ConditionProjection::load(&inf_path)
            .unwrap_err()
            .to_string();
        assert!(loaded.contains("非有限"), "{loaded}");

        let mut wide = MiniCpm35bProjection {
            inner: ConditionProjection {
                in_width: MiniCpm35bProjection::HIDDEN,
                weight: vec![0.0; MiniCpm35bProjection::HIDDEN * CONDITION_WIDTH],
                bias: vec![0.0; CONDITION_WIDTH],
                fitted: true,
            },
        };
        let wide_path = dir.join("35b.bin");
        wide.inner.weight[0] = f32::NAN;
        let saved = match wide.save(&wide_path) {
            Err(error) => error.to_string(),
            Ok(()) => panic!("nonfinite 35B weights must not be saved"),
        };
        assert!(saved.contains("非有限"), "{saved}");
        wide.inner.weight[0] = 0.0;
        wide.inner.bias[4] = f32::INFINITY;
        let saved = match wide.save(&wide_path) {
            Err(error) => error.to_string(),
            Ok(()) => panic!("nonfinite 35B weights must not be saved"),
        };
        assert!(saved.contains("非有限"), "{saved}");
        wide.inner.bias[4] = 0.0;
        wide.save(&wide_path).unwrap();
        let mut wide_bytes = std::fs::read(&wide_path).unwrap();
        wide_bytes[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&wide_path, &wide_bytes).unwrap();
        let loaded = match MiniCpm35bProjection::load(&wide_path) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("nonfinite 35B weights must not load"),
        };
        assert!(loaded.contains("非有限"), "{loaded}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn resample(sequence: &Sequence, target_length: usize) -> Vec<f32> {
    let source_length = sequence.length();
    let width = sequence.width();
    if source_length == target_length {
        return sequence.values().to_vec();
    }
    let mut aligned = vec![0.0; target_length * width];
    for position in 0..target_length {
        let cursor = if target_length == 1 {
            0.0
        } else {
            position as f32 * (source_length - 1) as f32 / (target_length - 1) as f32
        };
        let left = cursor.floor() as usize;
        let right = (left + 1).min(source_length - 1);
        let frac = cursor - left as f32;
        for dim in 0..width {
            let start = sequence.get(left, dim);
            let end = sequence.get(right, dim);
            aligned[position * width + dim] = start * (1.0 - frac) + end * frac;
        }
    }
    aligned
}
