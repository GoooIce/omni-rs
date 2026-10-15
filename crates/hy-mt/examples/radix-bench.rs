//! Warm-kernel on/off comparison through the production translation interface.
use omni_hy_mt::{Model, Options, Precision, Request, Translator};
use std::sync::atomic::AtomicBool;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    type B = burn::backend::wgpu::Metal;
    let cancel = AtomicBool::new(false);
    let path = std::env::args().nth(1).unwrap_or("models/Hy-MT2-7B".into());
    let model = Model::<B>::from_directory(
        path,
        Precision::F16,
        &Default::default(),
        &cancel,
        |_, _| {},
    )?;
    let mut translator = Translator::new(model, 4096);
    let mut request = Request {
        text: "今天天气很好，我们一起去公园散步吧。".into(),
        target_language: "English".into(),
        options: Options {
            temperature: 0.0,
            max_new_tokens: 64,
            ..Default::default()
        },
    };
    request.options.use_cache = false;
    let expected = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?.text;
    // Warm every path before measuring either side.
    let _ = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    request.options.use_cache = true;
    let _ = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    let _ = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    for round in 0..3 {
        for enabled in [false, true] {
            request.options.use_cache = enabled;
            let result = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
            assert_eq!(result.text, expected);
            assert_eq!(result.finish_reason, "stop");
            assert_eq!(
                result.cached_tokens,
                if enabled { result.prompt_tokens - 1 } else { 0 }
            );
            println!(
                "{}",
                serde_json::json!({"round":round,"cache_enabled":enabled,"result":result})
            );
        }
    }
    request.text = "今天天气很好，我们一起去图书馆看书吧。".into();
    let branch = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    assert!(branch.cached_tokens > 0 && branch.cached_tokens < branch.prompt_tokens - 1);
    request.options.use_cache = false;
    let uncached = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    assert_eq!(branch.text, uncached.text);
    assert_eq!(branch.finish_reason, uncached.finish_reason);
    println!(
        "{}",
        serde_json::json!({"branch":"passed","cached":branch,"uncached":uncached})
    );
    Ok(())
}
