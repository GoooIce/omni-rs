//! Shared pure checks for model and codec configuration snapshots.
use serde::Deserialize;

pub(crate) const TARGET_SAMPLE_RATE: u32 = 24_000;
pub(crate) const MAX_TARGET_CODEBOOKS: usize = 128;

#[derive(Deserialize)]
struct ModelFormat {
    num_codebooks: usize,
    codebook_size: usize,
    audio_eos_code: u32,
}

#[derive(Deserialize)]
struct CodecFormat {
    n_q: usize,
    fsq_levels: Vec<u32>,
    sampling_rate: u32,
}

pub(crate) struct TargetCodeFormat {
    pub(crate) num_codebooks: usize,
    pub(crate) code_count: u32,
    pub(crate) eos_code: u32,
}

impl TargetCodeFormat {
    /// Check the caller's own configuration bytes, without opening any paths.
    pub(crate) fn from_bytes(model: &[u8], codec: &[u8]) -> Result<Self, String> {
        let model: ModelFormat = serde_json::from_slice(model).map_err(|e| e.to_string())?;
        let codec: CodecFormat = serde_json::from_slice(codec).map_err(|e| e.to_string())?;
        let code_count = codec
            .fsq_levels
            .iter()
            .try_fold(1_u32, |product, &level| {
                (level >= 2).then(|| product.checked_mul(level)).flatten()
            })
            .ok_or("invalid codec levels")?;
        if codec.fsq_levels.is_empty()
            || codec.sampling_rate != TARGET_SAMPLE_RATE
            || !(1..=MAX_TARGET_CODEBOOKS).contains(&codec.n_q)
            || codec.n_q != model.num_codebooks
            || model.audio_eos_code < code_count
            || model.audio_eos_code as usize >= model.codebook_size
        {
            return Err("model and codec codebook formats do not match".into());
        }
        Ok(Self {
            num_codebooks: model.num_codebooks,
            code_count,
            eos_code: model.audio_eos_code,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configuration_requires_matching_bounded_codebooks_and_separate_eos() {
        let model = json!({"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016});
        let codec = json!({"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000});
        let parse = |model: &serde_json::Value, codec: &serde_json::Value| {
            TargetCodeFormat::from_bytes(
                &serde_json::to_vec(model).unwrap(),
                &serde_json::to_vec(codec).unwrap(),
            )
        };
        let valid = parse(&model, &codec).unwrap();
        assert_eq!(
            (valid.num_codebooks, valid.code_count, valid.eos_code),
            (12, 2016, 2016)
        );
        for (key, value) in [
            ("n_q", json!(0)),
            ("n_q", json!(129)),
            ("sampling_rate", json!(16000)),
            ("fsq_levels", json!([])),
            ("fsq_levels", json!([8, 1])),
            ("fsq_levels", json!([65536, 65536])),
        ] {
            let mut invalid = codec.clone();
            invalid[key] = value;
            assert!(parse(&model, &invalid).is_err(), "{key}");
        }
        for (key, value) in [
            ("num_codebooks", json!(11)),
            ("audio_eos_code", json!(2015)),
            ("audio_eos_code", json!(2017)),
        ] {
            let mut invalid = model.clone();
            invalid[key] = value;
            assert!(parse(&invalid, &codec).is_err(), "{key}");
        }
    }
}
