//! Bind the policy used by this implementation to the claimed protocol.
//! This is not a source-verification or qualification mechanism.

use sha2::{Digest, Sha256};

pub(super) struct FormalIdentity {
    pub stop_tokens: Vec<u32>,
    pub stop_policy_sha256: String,
    pub implementation_sha256: String,
    pub comparison_sha256: String,
}

fn part(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
}

pub(super) fn compute(protocol: &str, tokens: &[u32]) -> Result<FormalIdentity, String> {
    if tokens.is_empty() || tokens.len() > 1024 {
        return Err("invalid frozen stop policy".into());
    }
    let mut stops = tokens.to_vec();
    stops.sort_unstable();
    stops.dedup();
    let mut policy = Sha256::new();
    policy.update(b"omni-rsi/formal-stop-policy/v1\0");
    policy.update((stops.len() as u64).to_be_bytes());
    for token in &stops {
        policy.update(token.to_be_bytes());
    }
    let stop_policy_sha256 = format!("{:x}", policy.finalize());

    let sources: &[(&str, &[u8])] = &[
        ("formal", include_bytes!("number_formal_holdout.rs")),
        (
            "identity",
            include_bytes!("number_formal_holdout_identity.rs"),
        ),
        (
            "journal",
            include_bytes!("number_formal_holdout_journal.rs"),
        ),
        (
            "scoring",
            include_bytes!("number_formal_holdout_scoring.rs"),
        ),
        ("blind", include_bytes!("number_blind_holdout.rs")),
        ("pcm-identity", include_bytes!("number_development.rs")),
        ("ledger", include_bytes!("number_holdout_ledger.rs")),
        ("statistics", include_bytes!("number_gate.rs")),
        (
            "number-scoring",
            include_bytes!("speech_number_evaluation.rs"),
        ),
        (
            "wav-decoding",
            include_bytes!("../../lfm3-omni/src/media_io.rs"),
        ),
        ("private-acl", include_bytes!("private_acl.rs")),
    ];
    let mut implementation = Sha256::new();
    implementation.update(b"omni-rsi/formal-implementation/v1\0");
    for (name, bytes) in sources {
        part(&mut implementation, name.as_bytes());
        part(&mut implementation, bytes);
    }
    let implementation_sha256 = format!("{:x}", implementation.finalize());
    let mut comparison = Sha256::new();
    comparison.update(b"omni-rsi/formal-comparison/v1\0");
    part(&mut comparison, protocol.as_bytes());
    part(&mut comparison, stop_policy_sha256.as_bytes());
    part(&mut comparison, implementation_sha256.as_bytes());
    Ok(FormalIdentity {
        stop_tokens: stops,
        stop_policy_sha256,
        implementation_sha256,
        comparison_sha256: format!("{:x}", comparison.finalize()),
    })
}
