//! Pinned checkpoint catalog, source availability metadata, and structured
//! local file-readiness inspection shared by the downloader and desktop UI.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::DownloadSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckpointId {
    Confucius4R2t2,
    BreezeTts2,
    HyMt2_7b,
    QwenImage21Turbo,
    MertV2FullSong,
    SheetSage2,
    Yue2_3b,
    Yue2Vae,
    MiniCpmV47_1b,
    MiniCpmV47_35bInt4,
    Lfm3Omni2b,
}

impl CheckpointId {
    pub const ALL: [Self; 11] = [
        Self::Confucius4R2t2,
        Self::BreezeTts2,
        Self::HyMt2_7b,
        Self::QwenImage21Turbo,
        Self::MertV2FullSong,
        Self::SheetSage2,
        Self::Yue2_3b,
        Self::Yue2Vae,
        Self::MiniCpmV47_1b,
        Self::MiniCpmV47_35bInt4,
        Self::Lfm3Omni2b,
    ];

    pub fn slug(self) -> &'static str {
        self.descriptor().slug
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|id| id.slug() == slug)
    }

    pub fn descriptor(self) -> &'static CheckpointDescriptor {
        CheckpointCatalog::get(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedSource {
    pub repository: &'static str,
    pub revision: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileFilter {
    /// Only download files whose path appears in the fixed list.
    ExactList(&'static [&'static str]),
    /// Download every blob except hidden entries (`.`) and the skipped prefix.
    ExcludePrefix(&'static str),
    /// Download allowlisted filenames plus any `*.safetensors`.
    AllowListOrSafetensors(&'static [&'static str]),
    /// Music checkpoints: json/safetensors/tiktoken/py/txt/md + license files.
    MusicArtifacts,
}

impl FileFilter {
    pub fn includes(self, path: &str) -> bool {
        if path.starts_with('.') {
            return false;
        }
        match self {
            Self::ExactList(names) => names.contains(&path),
            Self::ExcludePrefix(prefix) => !path.starts_with(prefix),
            Self::AllowListOrSafetensors(names) => {
                names.contains(&path) || path.ends_with(".safetensors")
            }
            Self::MusicArtifacts => {
                path.ends_with(".json")
                    || path.ends_with(".safetensors")
                    || path.ends_with(".tiktoken")
                    || path.ends_with(".py")
                    || path.ends_with(".txt")
                    || path.ends_with(".md")
                    || matches!(path, "LICENSE" | "MODEL_LICENSE")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointDescriptor {
    pub id: CheckpointId,
    pub slug: &'static str,
    pub display_name: &'static str,
    pub panel_label: &'static str,
    pub default_rel_dir: &'static str,
    pub estimated_bytes: u64,
    pub modelscope: Option<PinnedSource>,
    pub huggingface: Option<PinnedSource>,
    pub manual_url: Option<&'static str>,
    pub file_filter: FileFilter,
    pub expected_model_type: Option<&'static str>,
    pub pinned_file_sha256: &'static [(&'static str, &'static str)],
}

impl CheckpointDescriptor {
    pub fn source(&self, source: DownloadSource) -> Option<PinnedSource> {
        match source {
            DownloadSource::ModelScope => self.modelscope,
            DownloadSource::HuggingFace => self.huggingface,
        }
    }

    pub fn supports(&self, source: DownloadSource) -> bool {
        self.source(source).is_some()
    }

    pub fn available_sources(&self) -> Vec<DownloadSource> {
        let mut out = Vec::new();
        if self.modelscope.is_some() {
            out.push(DownloadSource::ModelScope);
        }
        if self.huggingface.is_some() {
            out.push(DownloadSource::HuggingFace);
        }
        out
    }

    pub fn is_auto_downloadable(&self) -> bool {
        self.modelscope.is_some() || self.huggingface.is_some()
    }

    pub fn is_hf_only(&self) -> bool {
        self.modelscope.is_none() && self.huggingface.is_some()
    }

    pub fn source_badge(&self) -> &'static str {
        match (self.modelscope.is_some(), self.huggingface.is_some()) {
            (true, true) => "魔搭 / Hugging Face",
            (false, true) => "仅 Hugging Face",
            (true, false) => "仅魔搭 ModelScope",
            (false, false) => "仅本地目录（未锁定版本）",
        }
    }

    pub fn formatted_size(&self) -> String {
        format_bytes(self.estimated_bytes)
    }
}

const CONFUCIUS_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
    "generation_config.json",
    "README.md",
    "model.safetensors",
];

const HY_MT_ALLOWLIST: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "LICENSE.txt",
    "README.md",
];

const MERT_PINNED_SHA256: &[(&str, &str)] = &[(
    "model.safetensors",
    "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d",
)];

static CATALOG: [CheckpointDescriptor; 11] = [
    CheckpointDescriptor {
        id: CheckpointId::Confucius4R2t2,
        slug: "confucius4-r2t2",
        display_name: "Confucius4-R2T2",
        panel_label: "语音转写 (ASR)",
        default_rel_dir: "models/Confucius4-R2T2",
        estimated_bytes: 5_000_000_000,
        modelscope: Some(PinnedSource {
            repository: "netease-youdao/Confucius4-R2T2",
            revision: "master",
        }),
        huggingface: Some(PinnedSource {
            repository: "netease-youdao/Confucius4-R2T2",
            revision: "main",
        }),
        manual_url: None,
        file_filter: FileFilter::ExactList(CONFUCIUS_FILES),
        expected_model_type: Some("qwen3_asr"),
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::BreezeTts2,
        slug: "breeze-tts-2",
        display_name: "Breeze-TTS-2",
        panel_label: "语音合成 (TTS)",
        default_rel_dir: "models/Breeze-TTS-2",
        estimated_bytes: 7_200_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "BreezeBlue/Breeze-TTS-2",
            revision: "3e28c5151381a722f1d8661b4118c298caa77aa4",
        }),
        manual_url: None,
        file_filter: FileFilter::ExcludePrefix("assets/"),
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::HyMt2_7b,
        slug: "hy-mt2-7b",
        display_name: "Hy-MT2-7B",
        panel_label: "文本翻译",
        default_rel_dir: "models/Hy-MT2-7B",
        estimated_bytes: 15_000_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "tencent/Hy-MT2-7B",
            revision: "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223",
        }),
        manual_url: None,
        file_filter: FileFilter::AllowListOrSafetensors(HY_MT_ALLOWLIST),
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::QwenImage21Turbo,
        slug: "qwen-image-2.1-turbo",
        display_name: "Qwen-Image-2.1-Turbo",
        panel_label: "图像生成与编辑",
        default_rel_dir: "models/Qwen-Image-2.1-Turbo",
        estimated_bytes: 37_000_000_000,
        modelscope: Some(PinnedSource {
            repository: "Qwen/Qwen-Image-2.1-Turbo",
            revision: "7015dbe91fd9c8efcb95fdf5b784b59349eca6d7",
        }),
        huggingface: Some(PinnedSource {
            repository: "Qwen/Qwen-Image-2.1-Turbo",
            revision: "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
        }),
        manual_url: None,
        file_filter: FileFilter::ExcludePrefix("assets/"),
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::MertV2FullSong,
        slug: "mert-v2-fullsong",
        display_name: "MERT-v2-FullSong",
        panel_label: "音乐分析与转谱",
        default_rel_dir: "models/MERT-v2-FullSong",
        estimated_bytes: 2_400_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "m-a-p/MERT-v2-FullSong",
            revision: "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
        }),
        manual_url: None,
        file_filter: FileFilter::MusicArtifacts,
        expected_model_type: None,
        pinned_file_sha256: MERT_PINNED_SHA256,
    },
    CheckpointDescriptor {
        id: CheckpointId::SheetSage2,
        slug: "sheetsage2",
        display_name: "SheetSage2",
        panel_label: "音乐转谱",
        default_rel_dir: "models/SheetSage2",
        estimated_bytes: 230_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "m-a-p/SheetSage2",
            revision: "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc",
        }),
        manual_url: None,
        file_filter: FileFilter::MusicArtifacts,
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::Yue2_3b,
        slug: "yue2-3b",
        display_name: "YuE2-3B",
        panel_label: "音乐生成",
        default_rel_dir: "models/YuE2-3B",
        estimated_bytes: 6_800_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "m-a-p/YuE2-3B",
            revision: "8f24312f187763b9854adeec87e437e2b03bbff1",
        }),
        manual_url: None,
        file_filter: FileFilter::MusicArtifacts,
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::Yue2Vae,
        slug: "yue2-vae",
        display_name: "YuE2-Vae",
        panel_label: "音乐解码",
        default_rel_dir: "models/YuE2-Vae",
        estimated_bytes: 530_000_000,
        modelscope: None,
        huggingface: Some(PinnedSource {
            repository: "m-a-p/YuE2-Vae",
            revision: "152733a19ad43aa67e367f9b5503ef8075bb5126",
        }),
        manual_url: None,
        file_filter: FileFilter::MusicArtifacts,
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::MiniCpmV47_1b,
        slug: "minicpm-v-4.7-1b",
        display_name: "MiniCPM-V-4.7-1B",
        panel_label: "视觉观察 (VLM)",
        default_rel_dir: "models/MiniCPM-V-4.7-1B",
        estimated_bytes: 2_900_000_000,
        modelscope: Some(PinnedSource {
            repository: "OpenBMB/MiniCPM-V-4.7-1B",
            revision: "19a6e190f7056249ab058a89a2bb3e6ef88341cf",
        }),
        huggingface: Some(PinnedSource {
            repository: "openbmb/MiniCPM-V-4.7-1B",
            revision: "b0918c741c57aeec539bfa277db764088cfc18e3",
        }),
        manual_url: None,
        file_filter: FileFilter::ExcludePrefix("minicpm_v4dot7_assets/"),
        expected_model_type: Some("minicpmv4_7"),
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::MiniCpmV47_35bInt4,
        slug: "minicpm-v-4.7-35b-int4",
        display_name: "MiniCPM-V-4.7-35B-A3B-int4",
        panel_label: "提示词增强",
        default_rel_dir: "models/MiniCPM-V-4.7-35B-A3B-int4",
        estimated_bytes: 21_000_000_000,
        modelscope: Some(PinnedSource {
            repository: "OpenBMB/MiniCPM-V-4.7-35B-A3B-int4",
            revision: "53ebb85146c2c693a27de284547aec38522657ed",
        }),
        huggingface: Some(PinnedSource {
            repository: "openbmb/MiniCPM-V-4.7-35B-A3B-int4",
            revision: "efabb002f29d275e190d2dae623a6cd5b0989a0e",
        }),
        manual_url: None,
        file_filter: FileFilter::ExcludePrefix("assets/"),
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
    CheckpointDescriptor {
        id: CheckpointId::Lfm3Omni2b,
        slug: "lfm3-omni-2b",
        display_name: "LFM3-Omni-2B",
        panel_label: "全能对话 (Omni)",
        default_rel_dir: "models/LFM3-Omni-2B",
        estimated_bytes: 4_500_000_000,
        modelscope: None,
        huggingface: None,
        manual_url: Some("https://huggingface.co/LiquidAI/LFM3-Omni-2B"),
        file_filter: FileFilter::ExcludePrefix("assets/"),
        expected_model_type: None,
        pinned_file_sha256: &[],
    },
];

pub struct CheckpointCatalog;

impl CheckpointCatalog {
    pub fn all() -> &'static [CheckpointDescriptor] {
        &CATALOG
    }

    pub fn get(id: CheckpointId) -> &'static CheckpointDescriptor {
        CATALOG
            .iter()
            .find(|desc| desc.id == id)
            .expect("all CheckpointId variants must exist in CATALOG")
    }

    pub fn find_by_slug(slug: &str) -> Option<&'static CheckpointDescriptor> {
        CATALOG.iter().find(|desc| desc.slug == slug)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadinessReport {
    pub checkpoint: CheckpointId,
    pub directory: PathBuf,
    pub ready: bool,
    pub missing_files: Vec<String>,
    pub invalid_reasons: Vec<String>,
}

impl ReadinessReport {
    pub fn summary(&self) -> String {
        if self.ready {
            return format!("{} 已就绪", self.checkpoint.descriptor().display_name);
        }
        let mut parts = Vec::new();
        if !self.missing_files.is_empty() {
            let shown = if self.missing_files.len() <= 3 {
                self.missing_files.join("、")
            } else {
                format!(
                    "{} 等 {} 个文件",
                    self.missing_files[..2].join("、"),
                    self.missing_files.len()
                )
            };
            parts.push(format!("缺少 {shown}"));
        }
        if !self.invalid_reasons.is_empty() {
            parts.push(self.invalid_reasons.join("；"));
        }
        parts.join("；")
    }
}

/// Inspect a local checkpoint directory and return every missing file or
/// structural validation error without allocating GPU tensors.
pub fn inspect_readiness(id: CheckpointId, directory: &Path) -> ReadinessReport {
    let mut missing = Vec::new();
    let mut invalid = Vec::new();

    match id {
        CheckpointId::Confucius4R2t2
        | CheckpointId::HyMt2_7b
        | CheckpointId::MiniCpmV47_1b
        | CheckpointId::MiniCpmV47_35bInt4 => {
            check_required(
                directory,
                "",
                &["config.json", "tokenizer.json"],
                &mut missing,
            );
            check_weights(directory, "", "model", &mut missing, &mut invalid);
        }
        CheckpointId::BreezeTts2 => {
            check_required(
                directory,
                "",
                &[
                    "config.json",
                    "tokenizer.json",
                    "audio_tokenizer/config.json",
                    "audio_tokenizer/model.safetensors",
                ],
                &mut missing,
            );
            check_weights(directory, "", "model", &mut missing, &mut invalid);
        }
        CheckpointId::QwenImage21Turbo => {
            check_required(
                directory,
                "",
                &[
                    "processor/tokenizer.json",
                    "text_encoder/config.json",
                    "transformer/config.json",
                    "vae/config.json",
                ],
                &mut missing,
            );
            check_weights(
                &directory.join("text_encoder"),
                "text_encoder/",
                "model",
                &mut missing,
                &mut invalid,
            );
            check_weights(
                &directory.join("transformer"),
                "transformer/",
                "diffusion_pytorch_model",
                &mut missing,
                &mut invalid,
            );
            check_weights(
                &directory.join("vae"),
                "vae/",
                "diffusion_pytorch_model",
                &mut missing,
                &mut invalid,
            );
        }
        CheckpointId::MertV2FullSong | CheckpointId::SheetSage2 | CheckpointId::Yue2Vae => {
            check_required(directory, "", &["config.json"], &mut missing);
            check_weights(directory, "", "model", &mut missing, &mut invalid);
        }
        CheckpointId::Yue2_3b => {
            check_required(
                directory,
                "",
                &["config.json", "qwen.tiktoken"],
                &mut missing,
            );
            check_weights(directory, "", "model", &mut missing, &mut invalid);
        }
        CheckpointId::Lfm3Omni2b => {
            check_required(
                directory,
                "",
                &[
                    "config.json",
                    "processor_config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "audio_tokenizer/config.json",
                ],
                &mut missing,
            );
            check_weights(directory, "", "model", &mut missing, &mut invalid);
            check_weights(
                &directory.join("audio_tokenizer"),
                "audio_tokenizer/",
                "model",
                &mut missing,
                &mut invalid,
            );
        }
    }

    let ready = missing.is_empty() && invalid.is_empty();
    ReadinessReport {
        checkpoint: id,
        directory: directory.to_path_buf(),
        ready,
        missing_files: missing,
        invalid_reasons: invalid,
    }
}

fn check_required(directory: &Path, prefix: &str, names: &[&str], missing: &mut Vec<String>) {
    for name in names {
        if !directory.join(name).is_file() {
            missing.push(format!("{prefix}{name}"));
        }
    }
}

fn check_weights(
    directory: &Path,
    prefix: &str,
    stem: &str,
    missing: &mut Vec<String>,
    invalid: &mut Vec<String>,
) {
    let index_name = format!("{stem}.safetensors.index.json");
    let index_path = directory.join(&index_name);
    if index_path.is_file() {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let Ok(bytes) = fs::read(&index_path) else {
            invalid.push(format!("无法读取 {prefix}{index_name}"));
            return;
        };
        let Ok(index) = serde_json::from_slice::<Index>(&bytes) else {
            invalid.push(format!("模型分片索引无效：{prefix}{index_name}"));
            return;
        };
        if index.weight_map.is_empty() {
            invalid.push(format!("模型分片索引为空：{prefix}{index_name}"));
            return;
        }
        let shards: BTreeSet<&String> = index.weight_map.values().collect();
        for shard in shards {
            if Path::new(shard)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                invalid.push(format!("模型分片路径无效：{prefix}{shard}"));
                continue;
            }
            if !directory.join(shard).is_file() {
                missing.push(format!("{prefix}{shard}"));
            }
        }
    } else {
        let single = format!("{stem}.safetensors");
        if !directory.join(&single).is_file() {
            missing.push(format!("{prefix}{single}"));
        }
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_all_checkpoints_and_encodes_source_asymmetry() {
        assert_eq!(CheckpointCatalog::all().len(), CheckpointId::ALL.len());
        let mut slugs = BTreeSet::new();
        for desc in CheckpointCatalog::all() {
            assert!(slugs.insert(desc.slug), "duplicate slug {}", desc.slug);
            assert_eq!(CheckpointId::from_slug(desc.slug), Some(desc.id));
        }

        // Dual-source checkpoints.
        for id in [
            CheckpointId::Confucius4R2t2,
            CheckpointId::QwenImage21Turbo,
            CheckpointId::MiniCpmV47_1b,
            CheckpointId::MiniCpmV47_35bInt4,
        ] {
            let d = CheckpointCatalog::get(id);
            assert!(d.supports(DownloadSource::ModelScope));
            assert!(d.supports(DownloadSource::HuggingFace));
            assert!(!d.is_hf_only());
        }

        // HF-only checkpoints must never advertise ModelScope.
        for id in [
            CheckpointId::BreezeTts2,
            CheckpointId::HyMt2_7b,
            CheckpointId::MertV2FullSong,
            CheckpointId::SheetSage2,
            CheckpointId::Yue2_3b,
            CheckpointId::Yue2Vae,
        ] {
            let d = CheckpointCatalog::get(id);
            assert!(!d.supports(DownloadSource::ModelScope));
            assert!(d.supports(DownloadSource::HuggingFace));
            assert!(d.is_hf_only());
        }

        // LFM3-Omni-2B remains unpinned (manual local directory only).
        let omni = CheckpointCatalog::get(CheckpointId::Lfm3Omni2b);
        assert!(!omni.is_auto_downloadable());
        assert!(omni.manual_url.is_some());
    }

    #[test]
    fn readiness_inspection_reports_exact_missing_files_and_bad_shards() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("omni-hub-ready-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let report = inspect_readiness(CheckpointId::BreezeTts2, &dir);
        assert!(!report.ready);
        assert!(report.missing_files.contains(&"config.json".to_owned()));
        assert!(
            report
                .missing_files
                .contains(&"audio_tokenizer/model.safetensors".to_owned())
        );
        assert!(report.summary().contains("缺少"));

        fs::create_dir_all(dir.join("audio_tokenizer")).unwrap();
        fs::write(dir.join("config.json"), "{}").unwrap();
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("audio_tokenizer/config.json"), "{}").unwrap();
        fs::write(dir.join("audio_tokenizer/model.safetensors"), b"ok").unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"shard-1.safetensors"}}"#,
        )
        .unwrap();

        let missing_shard = inspect_readiness(CheckpointId::BreezeTts2, &dir);
        assert!(!missing_shard.ready);
        assert_eq!(missing_shard.missing_files, vec!["shard-1.safetensors"]);

        fs::write(dir.join("shard-1.safetensors"), b"ok").unwrap();
        let ready = inspect_readiness(CheckpointId::BreezeTts2, &dir);
        assert!(ready.ready);

        fs::remove_dir_all(dir).unwrap();
    }
}
