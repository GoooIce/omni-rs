use omni_rsi::fingerprint::model_fingerprint;
use std::sync::atomic::AtomicBool;

#[test]
fn a_reader_package_is_bound_to_actual_model_and_preprocessor_bytes() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
    ] {
        std::fs::write(dir.path().join(name), name.as_bytes()).unwrap();
    }
    let stop = AtomicBool::new(false);
    let first = model_fingerprint(dir.path(), &stop).unwrap();
    assert_eq!(first, model_fingerprint(dir.path(), &stop).unwrap());
    std::fs::write(dir.path().join("processor_config.json"), b"different").unwrap();
    assert_ne!(first, model_fingerprint(dir.path(), &stop).unwrap());
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    assert!(model_fingerprint(dir.path(), &stop).is_err());
}
