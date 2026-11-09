//! Bounded chart programs, a trusted renderer, and an independent pixel check.
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};

pub const WIDTH: u32 = 192;
pub const HEIGHT: u32 = 192;
const BASELINE: u32 = 170;
const UNIT: u32 = 14;
const LEFT_X: std::ops::Range<u32> = 40..70;
const RIGHT_X: std::ops::Range<u32> = 122..152;
const PAPER: Rgb<u8> = Rgb([255, 255, 255]);
const LEFT: Rgb<u8> = Rgb([210, 40, 40]);
const RIGHT: Rgb<u8> = Rgb([40, 80, 210]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChartSpec {
    pub left: u8,
    pub right: u8,
    pub question: Question,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Question {
    Taller,
    Twice,
}
impl Question {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Taller => {
                "Look at the two bars. Is the left bar taller than the right bar? Answer only Yes or No."
            }
            Self::Twice => {
                "Look at the two bars. Is the left bar at least twice as tall as the right bar? Answer only Yes or No."
            }
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChartProgram {
    pub left: u8,
    pub right: u8,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Split {
    Train,
    Dev,
    Test,
}

impl ChartSpec {
    pub fn new(left: u8, right: u8) -> Result<Self, String> {
        if !(1..=9).contains(&left) || !(1..=9).contains(&right) || left == right {
            return Err("chart heights must differ and be between 1 and 9".into());
        }
        Ok(Self {
            left,
            right,
            question: Question::Taller,
        })
    }
    pub fn with_question(mut self, question: Question) -> Self {
        self.question = question;
        self
    }
    /// Both orientations of a pair belong to the same family and split.
    pub fn family(self) -> usize {
        let a = usize::from(self.left.min(self.right));
        let b = usize::from(self.left.max(self.right));
        (1..a).map(|i| 9 - i).sum::<usize>() + b - a - 1
    }
    pub fn split(self) -> Split {
        match self.family() % 6 {
            4 => Split::Dev,
            5 => Split::Test,
            _ => Split::Train,
        }
    }
    /// Whether the left bar is taller.
    pub fn label(self) -> bool {
        match self.question {
            Question::Taller => self.left > self.right,
            Question::Twice => u16::from(self.left) >= 2 * u16::from(self.right),
        }
    }
    pub fn request(self) -> String {
        format!(
            "Return only compact JSON with integer fields left and right. A chart has a left bar of height {} and a right bar of height {}.",
            self.left, self.right
        )
    }
}
impl ChartProgram {
    /// Treat the model output as data, never as executable source code.
    pub fn parse(value: &str) -> Result<Self, String> {
        let program: Self =
            serde_json::from_str(value.trim()).map_err(|error| error.to_string())?;
        ChartSpec::new(program.left, program.right)?;
        Ok(program)
    }
}

pub fn render(program: &ChartProgram) -> Result<RgbImage, String> {
    ChartSpec::new(program.left, program.right)?;
    let mut image = RgbImage::from_pixel(WIDTH, HEIGHT, PAPER);
    for (x_range, value, color) in [
        (LEFT_X, program.left, LEFT),
        (RIGHT_X, program.right, RIGHT),
    ] {
        for x in x_range {
            for y in BASELINE - u32::from(value) * UNIT..BASELINE {
                image.put_pixel(x, y, color);
            }
        }
    }
    Ok(image)
}

/// Verify the intended values and every pixel before assigning a label.
pub fn verify(spec: &ChartSpec, program: &ChartProgram, image: &RgbImage) -> Result<(), String> {
    ChartSpec::new(spec.left, spec.right)?;
    if program.left != spec.left
        || program.right != spec.right
        || image.dimensions() != (WIDTH, HEIGHT)
    {
        return Err("program or canvas does not match the requested chart".into());
    }
    for (x, y, pixel) in image.enumerate_pixels() {
        let expected =
            if LEFT_X.contains(&x) && y < BASELINE && y >= BASELINE - u32::from(spec.left) * UNIT {
                LEFT
            } else if RIGHT_X.contains(&x)
                && y < BASELINE
                && y >= BASELINE - u32::from(spec.right) * UNIT
            {
                RIGHT
            } else {
                PAPER
            };
        if *pixel != expected {
            return Err(format!("rendered pixel differs at {x},{y}"));
        }
    }
    Ok(())
}
