use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

pub(crate) fn source(a: char, b: char, c: char, d: char) -> Value {
    json!({"speaker_sha256":a.to_string().repeat(64), "session_sha256":b.to_string().repeat(64),
        "recording_sha256":c.to_string().repeat(64), "family_sha256":d.to_string().repeat(64)})
}

pub(crate) fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

pub(crate) fn wav(path: &Path, rate: u32, amplitude: i16) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [0_i16, amplitude, -amplitude, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

pub(crate) fn case(
    root: &Path,
    split: &str,
    id: &str,
    number: u32,
    src: Value,
    amplitude: i16,
) -> Value {
    let name = format!("{id}.wav");
    let path = root.join(format!("{split}-media")).join(&name);
    wav(&path, 16_000, amplitude);
    json!({"id":id,"audio_file":name,"audio_sha256":format!("{:x}",Sha256::digest(fs::read(path).unwrap())),
        "spoken_number":number,"answer":number+1,"source":src,
        "declaration":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"}})
}

pub(crate) fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let path = root.path();
    let model = path.join("model");
    let codec = model.join("audio_tokenizer");
    fs::create_dir_all(&codec).unwrap();
    fs::write(model.join("config.json"), br#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016,"text_config":{"hidden_size":16}}"#).unwrap();
    for name in [
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
    ] {
        fs::write(model.join(name), b"synthetic fixture").unwrap();
    }
    fs::write(
        codec.join("config.json"),
        br#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
    )
    .unwrap();
    fs::write(codec.join("model.safetensors"), b"synthetic codec").unwrap();
    fs::write(path.join("layout.v3"), b"omni-rsi-number-train-dev-v3\n").unwrap();
    for name in ["train-media", "dev-media", "train-target-media"] {
        fs::create_dir(path.join(name)).unwrap();
    }
    let target_wav = path.join("train-target-media/answer.wav");
    wav(&target_wav, 24_000, 4096);
    let encoded = omni_rsi::speech_audio_target::encode_target_with(
        &model,
        &target_wav,
        &AtomicBool::new(false),
        |_, _| Ok(vec![vec![1; 12], vec![2; 12], vec![3; 12], vec![4; 12]]),
    )
    .unwrap();
    let mut target = serde_json::to_value(encoded).unwrap();
    target["audio_file"] = "answer.wav".into();
    target["source"] =
        json!({"kind":"licensed-synthetic","source":"target fixture","license":"CC0-1.0"});
    target["source_group"] = source('d', 'e', '0', 'f');
    let mut first = case(
        path,
        "train",
        "train-a",
        12,
        source('1', '2', '3', '4'),
        4096,
    );
    first["speech_target"] = target;
    let second = case(
        path,
        "train",
        "train-b",
        14,
        source('1', '2', '3', '4'),
        8192,
    );
    let dev = case(path, "dev", "dev-a", 15, source('5', '6', '7', '8'), 12_288);
    for (split, records) in [("train", vec![first, second]), ("dev", vec![dev])] {
        let groups: Vec<_> = records
            .iter()
            .map(|record| {
                let mut group = json!({"id":record["id"],"source":record["source"]});
                if !record["speech_target"].is_null() {
                    group["target_source"] = record["speech_target"]["source_group"].clone();
                }
                group
            })
            .collect();
        write_json(
            &path.join(format!("{split}.groups.v3.json")),
            &json!({"version":3,"partition":split,"records":groups}),
        );
        write_json(
            &path.join(format!("{split}.v3.json")),
            &json!({"version":3,"partition":split,"rule":"number-plus-one-v1","records":records}),
        );
    }
    write_json(
        &path.join("holdout-public.json"),
        &json!({"version":2,"content_sha256":"f".repeat(64),
        "sources":[source('9','a','b','c')]}),
    );
    root
}

#[cfg(all(unix, feature = "cpu"))]
pub(crate) fn repeated_target_budget_fixture(
    samples: usize,
    padded_bytes: Option<u64>,
    copies: usize,
) -> tempfile::TempDir {
    let root = fixture();
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    let reference =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures/codec/config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(reference).unwrap()).unwrap();
    let (hop, stack) = if samples > 1_000 {
        (400_usize, 8_usize)
    } else {
        (8, 2)
    };
    if samples > 1_000 {
        config["stft_n_fft"] = 1024.into();
        config["stft_win_length"] = 1024.into();
        config["stft_hop_length"] = hop.into();
        config["stft_frame_stack"] = stack.into();
    }
    write_json(&codec.join("config.json"), &config);
    write_json(
        &model.join("config.json"),
        &json!({"num_codebooks":3,"codebook_size":13,"audio_eos_code":12,"text_config":{"hidden_size":16}}),
    );

    let wav = root.path().join("train-target-media/answer.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for i in 0..samples {
        writer
            .write_sample(if i % 2 == 0 { 4096_i16 } else { -4096_i16 })
            .unwrap();
    }
    writer.finalize().unwrap();
    if let Some(size) = padded_bytes {
        fs::OpenOptions::new()
            .write(true)
            .open(&wav)
            .unwrap()
            .set_len(size)
            .unwrap();
    }
    let frame_count = samples.div_ceil(hop * stack);
    let encoded = omni_rsi::speech_audio_target::encode_target_with(
        &model,
        &wav,
        &AtomicBool::new(false),
        |_, _| Ok(vec![vec![1, 2, 3]; frame_count]),
    )
    .unwrap();
    let mut target = serde_json::to_value(encoded).unwrap();
    target["audio_file"] = "answer.wav".into();
    target["source"] =
        json!({"kind":"licensed-synthetic","source":"budget fixture","license":"CC0-1.0"});
    target["source_group"] = source('d', 'e', '0', 'f');
    target["encoding"] = "local_liquidcodec_f32".into();
    target["runtime_backend"] = "untrusted-fixture".into();

    let manifest_path = root.path().join("train.v3.json");
    let groups_path = root.path().join("train.groups.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let mut groups: Value = serde_json::from_slice(&fs::read(&groups_path).unwrap()).unwrap();
    let base = manifest["records"][0].clone();
    let mut records = Vec::new();
    let mut identities = Vec::new();
    for i in 0..copies {
        let mut row = base.clone();
        let name = format!("budget-{i}.wav");
        fs::hard_link(
            root.path().join("train-media/train-a.wav"),
            root.path().join("train-media").join(&name),
        )
        .unwrap();
        row["id"] = format!("budget-{i}").into();
        row["audio_file"] = name.into();
        row["speech_target"] = target.clone();
        identities.push(
            json!({"id":row["id"],"source":row["source"],"target_source":target["source_group"]}),
        );
        records.push(row);
    }
    manifest["records"] = records.into();
    groups["records"] = identities.into();
    write_json(&manifest_path, &manifest);
    write_json(&groups_path, &groups);
    root
}
