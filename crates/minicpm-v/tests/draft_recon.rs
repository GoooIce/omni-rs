//! #51 draft-replacement recon: offline ground truth plus 1B teacher-forced
//! match. `ground_truth` runs plain greedy on the 35B int4 (pinned prose
//! prompt plus this recon's code prompt) and dumps token ids under
//! `artifacts/draft-recon/`. `one_b_match` teacher-forces the 1B over those
//! sequences: per-position argmax match, top-2 coverage, and the empirical
//! accepted-length curve per block size. Both need the local checkpoints and
//! are `#[ignore]d`.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    generate,
    model::TextModel,
    multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use serde_json::json;
use std::sync::atomic::AtomicBool;

const PROSE: &str = "用几句话说明分页 KV 缓存是什么。";
// The pinned prose prompt EOSes after ~70 tokens ("in a few sentences"); this
// variant asks for the same explanatory register with more length so the
// match statistics have positions to count.
const PROSE_LONG: &str = "详细说明分页 KV 缓存是什么，包括它解决的问题、数据结构和工作流程。";
// The historical code-harness prompt was never pinned in VALIDATION; this
// recon defines one and records it next to the artifact.
const CODE: &str = "用 Python 写一个函数判断输入字符串是否为回文，并给出示例。";
const NEW_TOKENS: usize = 420;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_35b_greedy(root: &std::path::Path, prompt: &str) -> serde_json::Value {
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<Metal>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();
    let rendered = chat::render(&[Message::text(Role::User, prompt)], true, Some(false)).unwrap();
    let ids = tokenizer.encode(&rendered);
    let cancel = AtomicBool::new(false);
    let outcome = generate::generate(
        &model,
        &Prompt::text(ids.clone()),
        generate::Options {
            max_new_tokens: NEW_TOKENS,
            temperature: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            ..generate::Options::default()
        },
        &ModelDirectory::new(&model_dir).eos_ids(),
        &cancel,
        |_| Ok(()),
        |tokens| tokenizer.decode(tokens),
    )
    .unwrap();
    json!({
        "prompt": prompt,
        "input_ids": ids,
        "generated": outcome.tokens,
        "text": tokenizer.decode(&outcome.tokens),
        "finish_reason": format!("{:?}", outcome.finish_reason),
    })
}

#[test]
#[ignore]
fn ground_truth() {
    let root = repo_root();
    let out = root.join("artifacts/draft-recon");
    if !root
        .join("models/MiniCPM-V-4.7-35B-A3B-int4/config.json")
        .is_file()
    {
        eprintln!("skipping: 35B int4 checkpoint not present");
        return;
    }
    std::fs::create_dir_all(&out).unwrap();
    for (name, prompt) in [("prose", PROSE), ("prose_long", PROSE_LONG), ("code", CODE)] {
        let start = std::time::Instant::now();
        let data = run_35b_greedy(&root, prompt);
        let path = out.join(format!("{name}-35b-greedy.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&data).unwrap()).unwrap();
        eprintln!(
            "{name}: {} tokens ({:?}) in {:.1}s -> {}",
            data["generated"].as_array().unwrap().len(),
            data["finish_reason"],
            start.elapsed().as_secs_f32(),
            path.display()
        );
    }
}

/// Cycle simulation over the match vector, mirroring `dspark` accounting:
/// each cycle commits one bonus plus the accepted prefix, then the first
/// mismatch (or the block end) supplies the next bonus.
fn tau_hat(matches: &[bool], block: usize) -> f64 {
    let n = matches.len();
    let mut position = 0usize;
    let mut cycles = 0usize;
    let mut tokens = 0usize;
    while position + 1 < n {
        let mut accepted = 0usize;
        while accepted < block && position + 1 + accepted < n && matches[position + 1 + accepted] {
            accepted += 1;
        }
        tokens += 1 + accepted;
        position += 1 + accepted;
        cycles += 1;
    }
    if cycles == 0 {
        0.0
    } else {
        tokens as f64 / cycles as f64
    }
}

#[test]
#[ignore]
fn one_b_match() {
    let root = repo_root();
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 1B checkpoint not present");
        return;
    }
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<Metal>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();

    for name in ["prose", "prose_long", "code"] {
        let path = root.join(format!("artifacts/draft-recon/{name}-35b-greedy.json"));
        let Some(data) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        else {
            eprintln!(
                "skipping {name}: run `ground_truth` first ({})",
                path.display()
            );
            continue;
        };
        let input: Vec<u32> = data["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let generated: Vec<u32> = data["generated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        // seq[i] logits predict seq[i + 1]; drop the last generated token,
        // which has no successor to score.
        let mut seq = input.clone();
        seq.extend_from_slice(&generated[..generated.len() - 1]);
        let prompt: Prompt<Metal> = Prompt::text(seq.clone());
        let mut state = model.new_state(&device);
        state.rope_delta = prompt.rope_delta;
        // `forward` returns only the last row's logits; the verify path
        // returns one row per fed token, which is what teacher forcing over
        // the whole sequence needs.
        let (logits, _, _) = model
            .forward_verifying(&seq, &mut state, &[])
            .map_err(|error| error.to_string())
            .unwrap();
        let argmax = logits.clone().argmax(1).into_data().convert::<i32>();
        let picks: Vec<i32> = argmax.as_slice::<i32>().unwrap().to_vec();
        let (_, top2) = logits.topk_with_indices(2, 1);
        let top2 = top2.into_data().convert::<i32>();
        let top2: Vec<i32> = top2.as_slice::<i32>().unwrap().to_vec();

        // matches[j]: the 1B predicts generated[j] from the ground-truth
        // prefix, exactly what a standalone chain draft conditions on.
        let mut matches = Vec::with_capacity(generated.len() - 1);
        let mut in_top2 = 0usize;
        for j in 0..generated.len() - 1 {
            let row = input.len() + j - 1;
            let want = generated[j] as i32;
            matches.push(picks[row] == want);
            if top2[row * 2] == want || top2[row * 2 + 1] == want {
                in_top2 += 1;
            }
        }
        let hit = matches.iter().filter(|m| **m).count();
        let p = hit as f64 / matches.len() as f64;
        let curve: Vec<serde_json::Value> = [2usize, 3, 4, 7, 8, 15, 16]
            .iter()
            .map(|block| {
                json!({
                    "block": block,
                    "tau_hat": tau_hat(&matches, *block),
                })
            })
            .collect();
        let result = json!({
            "name": name,
            "generated_tokens": generated.len(),
            "positions": matches.len(),
            "match_rate": p,
            "top2_coverage": in_top2 as f64 / matches.len() as f64,
            "tau_hat_curve": curve,
            "match_prefix": format!(
                "{}",
                matches.iter().take(64).map(|m| if *m { '1' } else { '0' }).collect::<String>()
            ),
        });
        let out = root.join(format!("artifacts/draft-recon/{name}-1b-match.json"));
        std::fs::write(&out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
        eprintln!(
            "{name}: p={p:.3} top2={:.3} tau_hat(block)={}",
            in_top2 as f64 / matches.len() as f64,
            curve
                .iter()
                .map(|c| format!("{}:{:.2}", c["block"], c["tau_hat"].as_f64().unwrap()))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}
