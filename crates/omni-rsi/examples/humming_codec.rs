//! Offline, score-backed LiquidCodec measurement. No microphone or network.

#[cfg(any(feature = "cpu", feature = "metal"))]
mod runner {
    use omni_lfm3::{
        Precision,
        codec::{Codec, CodecEncoder},
    };
    use omni_rsi::{
        fingerprint::codec_fingerprint,
        humming::Clock,
        humming_audio::{PitchConfig, measure_codec, synthetic_melody},
    };
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        env,
        error::Error,
        fs,
        path::{Path, PathBuf},
        sync::atomic::AtomicBool,
    };

    #[cfg(feature = "metal")]
    type RunBackend = omni_lfm3::NativeGpu;
    #[cfg(all(feature = "cpu", not(feature = "metal")))]
    type RunBackend = burn::backend::NdArray<f32>;

    fn hash_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn hash_pcm(samples: &[f32]) -> String {
        let mut hash = Sha256::new();
        for sample in samples {
            hash.update(sample.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn source_hash() -> String {
        let sources: &[(&str, &[u8])] = &[
            (
                "humming_audio.rs",
                include_bytes!("../src/humming_audio.rs"),
            ),
            ("humming.rs", include_bytes!("../src/humming.rs")),
            ("fingerprint.rs", include_bytes!("../src/fingerprint.rs")),
            ("codec.rs", include_bytes!("../../lfm3-omni/src/codec.rs")),
            ("text.rs", include_bytes!("../../lfm3-omni/src/text.rs")),
            ("humming_codec.rs", include_bytes!("humming_codec.rs")),
            ("Cargo.lock", include_bytes!("../Cargo.lock")),
        ];
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi-humming-codec-sources-v1\0");
        for (name, contents) in sources {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update((contents.len() as u64).to_le_bytes());
            hash.update(contents);
        }
        format!("{:x}", hash.finalize())
    }

    fn write_wav(path: &Path, samples: &[f32], rate: u32) -> Result<(), Box<dyn Error>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut wav = hound::WavWriter::create(path, spec)?;
        for &sample in samples {
            wav.write_sample(sample)?;
        }
        wav.finalize()?;
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut args = env::args_os().skip(1);
        let model = PathBuf::from(
            args.next()
                .ok_or("expected MODEL_DIRECTORY OUTPUT_DIRECTORY")?,
        );
        let destination = PathBuf::from(
            args.next()
                .ok_or("expected MODEL_DIRECTORY OUTPUT_DIRECTORY")?,
        );
        if args.next().is_some() {
            return Err("expected exactly MODEL_DIRECTORY OUTPUT_DIRECTORY".into());
        }
        if destination.exists() {
            return Err("output directory already exists; choose a new path".into());
        }
        let directory = if model.join("audio_tokenizer").is_dir() {
            model.join("audio_tokenizer")
        } else {
            model
        };
        let stop = AtomicBool::new(false);
        let model_hash = codec_fingerprint(&directory, &stop)?;
        let device = Default::default();
        let encoder = CodecEncoder::<RunBackend>::load(&directory, Precision::F32, &device)?;
        let decoder = Codec::<RunBackend>::load(&directory, Precision::F32, &device)?;
        let melody = synthetic_melody();
        let config = PitchConfig::default();
        let clock = Clock {
            sample_rate_hz: melody.sample_rate_hz,
            origin_ns: 0,
        };
        let input_samples = melody.samples.len() as u64;
        let measurement = measure_codec(
            &encoder,
            &decoder,
            melody.sample_rate_hz,
            &melody.samples,
            &melody.notes,
            &config,
            0.1,
        )?;
        let reconstruction = measurement.reconstruction;
        if codec_fingerprint(&directory, &stop)? != model_hash {
            return Err("codec files changed during the run".into());
        }

        let parameters: Value = json!({
            "sample_rate_hz": melody.sample_rate_hz,
            "input_samples": input_samples,
            "input_clock": clock,
            "output_clock": clock,
            "notes": melody.notes,
            "recipe": "equal temperament; sine harmonics 0.6/0.3/0.1; gain 0.3; linear 10 ms fades; 24 kHz mono; zero-valued rests",
            "pitch_estimator": config,
            "onset_tolerance_seconds": 0.1,
            "input_prefix_end": input_samples,
            "input_mode": "Offline"
        });
        let parameters_hash = hash_bytes(&serde_json::to_vec(&parameters)?);
        let report = json!({
            "schema": "omni-rsi-liquidcodec-fidelity-v1",
            "scope": "synthetic_scored_melody_only",
            "source": "generated by this repository; no third-party audio",
            "license": "Apache-2.0 (repository code and generated synthetic fixture)",
            "natural_humming_evidence": "insufficient",
            "following_capability_evidence": "not_evaluated",
            "model_directory": directory,
            "model_fingerprint_sha256": model_hash,
            "precision": "f32",
            "backend": if cfg!(feature = "metal") { "wgpu_default_with_metal_feature" } else { "cpu_ndarray" },
            "target_os": env::consts::OS,
            "source_fingerprint_sha256": source_hash(),
            "source_fingerprint_scope": ["humming_audio.rs", "humming.rs", "fingerprint.rs", "codec.rs", "text.rs", "humming_codec.rs", "Cargo.lock"],
            "parameters_sha256": parameters_hash,
            "parameters": parameters,
            "input_pcm_f32le_sha256": hash_pcm(&melody.samples),
            "decoded_pcm_f32le_sha256": hash_pcm(&reconstruction.samples),
            "encoded_frame_count": reconstruction.frames.len(),
            "decoded_samples_including_padding": reconstruction.samples.len(),
            "input_measurement": measurement.input,
            "decoded_measurement": measurement.decoded,
            "artifacts": {
                "input_pcm": "input-f32.wav",
                "encoded_frames": "codec-frames.json",
                "decoded_pcm": "decoded-f32.wav"
            }
        });
        fs::create_dir(&destination)?;
        write_wav(
            &destination.join("input-f32.wav"),
            &melody.samples,
            melody.sample_rate_hz,
        )?;
        fs::write(
            destination.join("codec-frames.json"),
            serde_json::to_vec(&reconstruction.frames)?,
        )?;
        write_wav(
            &destination.join("decoded-f32.wav"),
            &reconstruction.samples,
            melody.sample_rate_hz,
        )?;
        fs::write(
            destination.join("report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!(
            "Synthetic codec report written to {}",
            destination.display()
        );
        Ok(())
    }
}

#[cfg(any(feature = "cpu", feature = "metal"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}

#[cfg(not(any(feature = "cpu", feature = "metal")))]
fn main() {
    eprintln!("enable the cpu or metal feature");
    std::process::exit(1);
}
