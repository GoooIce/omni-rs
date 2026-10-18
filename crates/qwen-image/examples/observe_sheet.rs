//! Run five text-to-image prompts through denoising observation, one at a time.
//!
//! Each prompt keeps the official row, the 1B row, and the 35B row. Cells are
//! written as they arrive, and index.html is refreshed after every cell.

use omni_qwen_image::{Model, ObservationRow, Options, Precision, text_to_image_split};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const PROMPTS: usize = 5;
const SIDE: usize = 512;
const SEED: u64 = 42;

fn main() {
    let output = arg_dir(1, "artifacts/denoising-observation");
    let model_dir = arg_dir(2, "/Users/goooice/code/omni-rs/models/Qwen-Image-2.1-Turbo");
    std::fs::create_dir_all(&output).expect("output directory");
    let prompts: Vec<&str> = text_to_image_split()
        .0
        .iter()
        .copied()
        .take(PROMPTS)
        .collect();
    let mut finished = Vec::new();
    write_html(&output, &prompts, &finished);

    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    ctrlc::set_handler(move || flag.store(true, Ordering::Release)).expect("ctrl-c");
    let device = Default::default();
    eprintln!("loading Qwen-Image-2.1-Turbo from {}", model_dir.display());
    let mut model = Model::<burn::backend::wgpu::Metal>::load(
        &model_dir,
        Precision::F16,
        &device,
        &cancel,
        |progress| {
            if progress.stage == "step"
                || progress.current == 1
                || progress.current == progress.total
            {
                eprintln!(
                    "load {} {}/{}",
                    progress.stage, progress.current, progress.total
                );
            }
        },
    )
    .expect("image model");

    for (index, prompt) in prompts.iter().enumerate() {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let started = Instant::now();
        eprintln!("prompt {index}/{PROMPTS}: {prompt}");
        let options = Options {
            prompt: (*prompt).to_string(),
            width: SIDE,
            height: SIDE,
            seed: SEED,
            observation: true,
            ..Default::default()
        };
        let sheet = output.clone();
        let page_prompts = prompts.clone();
        model
            .observe(
                &options,
                &cancel,
                |_| {},
                move |row, step, image| {
                    let name = file_name(index, row, step);
                    let path = sheet.join(&name);
                    image.save(&path).expect("save cell");
                    eprintln!("saved {}", path.display());
                    write_html(&sheet, &page_prompts, &[]);
                },
            )
            .expect("observation");
        finished.push(*prompt);
        write_html(&output, &prompts, &finished);
        eprintln!(
            "prompt {index} finished in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
    eprintln!("sheet {}", output.join("index.html").display());
}

fn arg_dir(index: usize, default: &str) -> PathBuf {
    std::env::args()
        .nth(index)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn file_name(index: usize, row: ObservationRow, step: usize) -> String {
    let row = match row {
        ObservationRow::Official => "official",
        ObservationRow::MiniCpm1b => "1b",
        ObservationRow::MiniCpm35b => "35b",
    };
    format!("p{index:02}-{row}-{step:02}.png")
}

fn write_html(output: &Path, prompts: &[&str], _finished: &[&str]) {
    let rows = [
        (ObservationRow::Official, "官方文本条件"),
        (ObservationRow::MiniCpm1b, "MiniCPM-V-4.7-1B"),
        (ObservationRow::MiniCpm35b, "MiniCPM-V-4.7-35B-A3B-int4"),
    ];
    let mut body = String::new();
    for (index, prompt) in prompts.iter().enumerate() {
        body.push_str(&format!(
            "<section><h2>{}</h2><p class=\"prompt\">{}</p><table><thead><tr><th></th>",
            index + 1,
            html_escape(prompt)
        ));
        for step in 1..=8 {
            body.push_str(&format!("<th>步 {step}</th>"));
        }
        body.push_str("</tr></thead><tbody>");
        for (row, label) in rows {
            body.push_str(&format!("<tr><th>{}</th>", label));
            for step in 1..=8 {
                let name = file_name(index, row, step);
                let ready = output.join(&name).is_file();
                if ready {
                    body.push_str(&format!(
                        "<td><img src=\"{name}\" alt=\"{label} 第 {step} 步\"></td>"
                    ));
                } else {
                    body.push_str("<td class=\"pending\">…</td>");
                }
            }
            body.push_str("</tr>");
        }
        body.push_str("</tbody></table></section>");
    }
    let html = format!(
        r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>去噪观察 · 五条提示词</title>
<style>
  body {{ margin: 0; background: #101612; color: #e7f0ea; font: 16px/1.5 "Iowan Old Style", Palatino, serif; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-weight: 500; font-size: 28px; margin: 0 0 8px; }}
  .note {{ color: #9db0a4; margin: 0 0 28px; }}
  section {{ margin: 0 0 36px; }}
  h2 {{ font-size: 18px; font-weight: 500; margin: 0; }}
  .prompt {{ margin: 4px 0 12px; color: #d5e4da; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #2c3a32; padding: 6px; vertical-align: middle; text-align: center; }}
  th {{ font-weight: 500; color: #b7c9be; font-size: 13px; }}
  td img {{ width: 128px; height: 128px; object-fit: contain; background:
    repeating-conic-gradient(#1a2420 0% 25%, #24302a 0% 50%) 50% / 16px 16px; }}
  .pending {{ color: #6d7f74; height: 128px; }}
</style>
</head>
<body>
<main>
  <h1>去噪观察</h1>
  <p class="note">五条提示词，种子 {SEED}，画布 {SIDE}×{SIDE}。每一行单独去噪。页面每 20 秒刷新。</p>
  {body}
</main>
</body>
</html>
"#
    );
    std::fs::write(output.join("index.html"), html).expect("html");
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
