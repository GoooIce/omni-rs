//! Compose existing small synthetic fixtures into a 16-wide Omni model.
//! The codec fixture is only loaded to construct the model; no codec quality
//! or relationship to these artificial target codes is asserted.
use std::{collections::BTreeMap, fs, path::Path};

use burn::backend::{Autodiff, NdArray};
use omni_lfm3::{Precision, codec::CodecConfig, config::TextConfig, model::OmniModel};
use omni_r2t2::weights::Weights;
use serde_json::{Value, json};

pub type Cpu = Autodiff<NdArray<f32>>;

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn cropped(
    shape: &[usize],
    data: &[u8],
    rows: usize,
    cols: Option<usize>,
) -> (Vec<usize>, Vec<u8>) {
    if let Some(cols) = cols {
        let mut result = Vec::with_capacity(rows * cols * 4);
        for row in 0..rows {
            let start = row * shape[1] * 4;
            result.extend_from_slice(&data[start..start + cols * 4]);
        }
        (vec![rows, cols], result)
    } else {
        (vec![rows], data[..rows * 4].to_vec())
    }
}

fn merge_fixture_weights(fixtures: &Path, output: &Path, answer_layer: bool) {
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for fixture in [
        "text",
        "audio_encoder_128",
        "depth_decoder",
        "vision_encoder",
    ] {
        let file = fs::read(fixtures.join(fixture).join("synthetic.bin")).unwrap();
        let header_len = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&file[8..8 + header_len]).unwrap();
        let payload = &file[8 + header_len..];
        for (name, metadata) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            assert_eq!(metadata["dtype"], "F32");
            let shape: Vec<usize> = serde_json::from_value(metadata["shape"].clone()).unwrap();
            let offsets: Vec<usize> =
                serde_json::from_value(metadata["data_offsets"].clone()).unwrap();
            let bytes = &payload[offsets[0]..offsets[1]];
            let (shape, bytes) = match name.as_str() {
                "model.audio_projector.linear_1.bias"
                | "model.audio_projector.linear_2.bias"
                | "model.multi_modal_projector.linear_2.bias" => cropped(&shape, bytes, 16, None),
                "model.audio_projector.linear_1.weight"
                | "model.audio_projector.linear_2.weight" => cropped(&shape, bytes, 16, Some(16)),
                "model.multi_modal_projector.linear_2.weight" => {
                    cropped(&shape, bytes, 16, Some(32))
                }
                "model.depth_decoder.depth_linear.weight" => cropped(&shape, bytes, 64, Some(16)),
                _ => (shape, bytes.to_vec()),
            };
            assert!(tensors.insert(name.clone(), (shape, bytes)).is_none());
        }
    }
    if answer_layer {
        let source: Vec<_> = tensors
            .iter()
            .filter(|(name, _)| name.starts_with("model.language_model.layers.0."))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        assert!(!source.is_empty());
        for layer in 4..24 {
            for (name, value) in &source {
                let name = name.replacen("layers.0.", &format!("layers.{layer}."), 1);
                assert!(tensors.insert(name, value.clone()).is_none());
            }
        }
    }
    let codebook: Vec<u8> = (0..44 * 16)
        .flat_map(|i| (((i as f32) * 0.17).sin() * 0.2).to_le_bytes())
        .collect();
    tensors.insert(
        "model.codebook_embedding.weight".into(),
        (vec![44, 16], codebook),
    );
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, bytes)) in tensors {
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(
            name,
            json!({"dtype":"F32", "shape":shape, "data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    fs::write(output, file).unwrap();
}

pub fn model() -> (tempfile::TempDir, OmniModel<Cpu>) {
    fixture(false)
}

pub fn answer_model() -> (tempfile::TempDir, OmniModel<Cpu>) {
    fixture(true)
}

fn fixture(answer_layer: bool) -> (tempfile::TempDir, OmniModel<Cpu>) {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures");
    let root = tempfile::tempdir().unwrap();
    let mut config = read_json(&fixtures.join("text/config.json"));
    let audio = read_json(&fixtures.join("audio_encoder_128/config.json"));
    let vision = read_json(&fixtures.join("vision_encoder/config.json"));
    let depth = read_json(&fixtures.join("depth_decoder/config.json"));
    if answer_layer {
        config["text_config"]["num_hidden_layers"] = json!(24);
        config["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .extend((4..24).map(|_| json!("conv")));
    }
    config["audio_encoder_config"] = audio["audio_encoder_config"].clone();
    for key in [
        "vision_config",
        "downsample_factor",
        "projector_layer_norm_eps",
        "projector_hidden_size",
        "projector_bias",
        "projector_hidden_act",
    ] {
        config[key] = vision[key].clone();
    }
    for key in ["depth_config", "num_codebooks", "codebook_size"] {
        config[key] = depth[key].clone();
    }
    for (key, value) in [
        ("audio_out_token_id", json!(4)),
        ("audio_start_token_id", json!(3)),
        ("mixed_start_token_id", json!(5)),
        ("text_end_token_id", json!(6)),
        ("audio_eos_code", json!(10)),
        ("interleaved_n_text", json!(1)),
        ("interleaved_n_audio", json!(1)),
        ("maskgit_schedule", json!([1, 1, 1, 1])),
        ("image_token_id", json!(20)),
        ("video_token_id", json!(21)),
        ("audio_token_id", json!(22)),
    ] {
        config[key] = value;
    }
    fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let image = json!({"size":{"shortest_edge":4,"longest_edge":64},"patch_size":2,"temporal_patch_size":2,"merge_size":2,
        "rescale_factor":0.0039215686,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
        "do_resize":true,"do_rescale":true,"do_normalize":true,"do_convert_rgb":true,"resample":3});
    let processor = json!({"image_token":"<|image|>","audio_token":"<|audio|>","video_token":"<|video|>",
        "vision_start_token":"<|start|>","vision_end_token":"<|end|>","vision_markers":false,"audio_markers":false,
        "feature_extractor":{"feature_size":128,"sampling_rate":16000,"n_fft":400,"win_length":400,"hop_length":160,
            "log_floor":-12.0,"normalization_offset":4.0,"normalization_divisor":4.0,"encoder_stride":8},
        "image_processor":image});
    fs::write(
        root.path().join("processor_config.json"),
        serde_json::to_vec(&processor).unwrap(),
    )
    .unwrap();
    let mut video = image;
    video["min_frames"] = json!(1);
    video["max_frames"] = json!(4);
    video["fps"] = json!(2.0);
    video["cap_pixels_per_frame"] = json!(false);
    fs::write(
        root.path().join("video_preprocessor_config.json"),
        serde_json::to_vec(&video).unwrap(),
    )
    .unwrap();
    fs::write(
        root.path().join("generation_config.json"),
        br#"{"eos_token_id":[6]}"#,
    )
    .unwrap();
    fs::write(
        root.path().join("tokenizer_config.json"),
        br#"{"bos_token":"<bos>"}"#,
    )
    .unwrap();
    fs::write(root.path().join("chat_template.jinja"), "{{ bos_token }}{% for m in messages %}{% for p in m.content %}{% if p.type == 'text' %} {{ p.text }} {% elif p.type == 'audio' %}<|audio|>{% endif %}{% endfor %}{% endfor %}").unwrap();
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for id in 0..23 {
        let token = match id {
            0 => "<unk>".into(),
            1 => "<bos>".into(),
            2 => "hum".into(),
            3 => "<astart>".into(),
            4 => "<aout>".into(),
            5 => "<mixed>".into(),
            6 => "<|im_end|>".into(),
            20 => "<|image|>".into(),
            21 => "<|video|>".into(),
            22 => "<|audio|>".into(),
            _ => format!("t{id}"),
        };
        vocab.insert(token.clone(), json!(id));
        if id != 2 {
            added.push(json!({"id":id,"content":token,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}));
        }
    }
    let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":added,
        "normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":vocab,"unk_token":"<unk>"}});
    fs::write(
        root.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    merge_fixture_weights(
        &fixtures,
        &root.path().join("model.safetensors"),
        answer_layer,
    );
    let codec = fixtures.join("codec");
    let backbone: TextConfig =
        serde_json::from_slice(&fs::read(codec.join("backbone.json")).unwrap()).unwrap();
    let codec_config = CodecConfig::load(&codec)
        .unwrap()
        .with_backbone(backbone)
        .unwrap();
    let codec_weights = Weights::open_file(&codec.join("synthetic.bin")).unwrap();
    let model = OmniModel::<Cpu>::load_with_codec_weights(
        root.path(),
        Precision::F32,
        &Default::default(),
        &codec_weights,
        codec_config,
    )
    .unwrap();
    (root, model)
}
