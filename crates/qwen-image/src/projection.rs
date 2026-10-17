//! Map a MiniCPM-V-4.7-1B last-layer sequence onto an official text condition.
//!
//! The official condition is 4096 wide. Positions line up with that sequence,
//! so a shorter or longer last layer is resampled before the linear map.
//! This path never reads an image and never calls JSON prompt rewrite.

use crate::{Error, Result};
use std::path::{Path, PathBuf};

pub const CONDITION_WIDTH: usize = 4096;
const TRAIN_PROMPTS: usize = 16;
/// Shrinks the linear map toward the deterministic zero init. The bias stays free.
const RIDGE_SCALE: f64 = 10_000.0;

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

fn missing_projection() -> Error {
    Error::Config(
        "去噪观察缺少 MiniCPM-V-4.7-1B 投影权重，不能用未训练投影出图，也不会退回 JSON 改写".into(),
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

    fn synthetic_pair(index: usize, in_width: usize) -> (Sequence, Sequence) {
        let length = 2;
        let mut input = vec![0.0; length * in_width];
        for (offset, value) in input.iter_mut().enumerate() {
            *value = ((index * 13 + offset * 5) % 9) as f32 / 4.0 - 0.5;
        }
        let input = Sequence::last_layer(length, in_width, input).unwrap();
        let mut target = vec![0.0; length * CONDITION_WIDTH];
        for position in 0..length {
            for dim in 0..CONDITION_WIDTH {
                let source = input.get(position, dim % in_width);
                target[position * CONDITION_WIDTH + dim] =
                    0.75 + 0.1 * source + 0.01 * (dim % 5) as f32;
            }
        }
        let target = Sequence::last_layer(length, CONDITION_WIDTH, target).unwrap();
        (input, target)
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
            .map(|index| synthetic_pair(index, in_width))
            .collect();
        let holdout: Vec<_> = (0..holdout_prompts.len())
            .map(|index| synthetic_pair(16 + index, in_width))
            .collect();
        let init = ConditionProjection::deterministic_init(in_width).unwrap();
        let fitted = ConditionProjection::fit(&train).unwrap();
        let init_mse = init.mean_squared_error(&holdout).unwrap();
        let fitted_mse = fitted.mean_squared_error(&holdout).unwrap();
        assert!(
            fitted_mse < init_mse,
            "fitted {fitted_mse} vs init {init_mse}"
        );
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
