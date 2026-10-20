//! How one native transcription starts. Mode, microphone, language and context
//! become the `StreamConfig` the desktop engine runs.
use omni_r2t2::{StreamConfig, TextPostprocess};

/// `language: None` lets the model detect the language.
pub fn stream_config(
    mode: usize,
    microphone: bool,
    language: Option<String>,
    context: String,
) -> StreamConfig {
    let (chunk_ms, lookahead_ms, max_new_tokens) = match mode {
        1 => (160, 160, 4),
        2 => (960, 0, 8),
        3 => (1280, 0, 12),
        _ => (320, 160, 8),
    };
    StreamConfig {
        language,
        context,
        chunk_ms,
        lookahead_ms,
        max_new_tokens,
        text_postprocess: TextPostprocess::Official,
        stable_tail_ms: if microphone { 800 } else { 0 },
        final_padding_ms: 640,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::stream_config;
    use omni_r2t2::{StreamConfig, TextPostprocess};

    #[test]
    fn desktop_modes_start_with_official_postprocess_padding_and_tail() {
        // chunk_ms, lookahead_ms, max_new_tokens from the native product table.
        let modes = [
            (0, 320, 160, 8),
            (1, 160, 160, 4),
            (2, 960, 0, 8),
            (3, 1280, 0, 12),
        ];
        let defaults = StreamConfig::default();
        for (mode, chunk_ms, lookahead_ms, max_new_tokens) in modes {
            let file = stream_config(mode, false, None, "prior".into());
            let microphone = stream_config(mode, true, None, "prior".into());
            for config in [&file, &microphone] {
                assert_eq!(config.chunk_ms, chunk_ms, "mode {mode}");
                assert_eq!(config.lookahead_ms, lookahead_ms, "mode {mode}");
                assert_eq!(config.max_new_tokens, max_new_tokens, "mode {mode}");
                assert_eq!(config.text_postprocess, TextPostprocess::Official);
                assert_eq!(config.final_padding_ms, 640);
                assert_eq!(config.language, None);
                assert_eq!(config.context, "prior");
                assert_eq!(config.unfixed_tokens, defaults.unfixed_tokens);
                assert_eq!(config.final_max_tokens, defaults.final_max_tokens);
                assert_eq!(config.window_seconds, defaults.window_seconds);
                assert_eq!(config.max_context_tokens, defaults.max_context_tokens);
                assert_eq!(config.reuse_cache, defaults.reuse_cache);
                assert_eq!(config.paged_attention, defaults.paged_attention);
            }
            assert_eq!(file.stable_tail_ms, 0);
            assert_eq!(microphone.stable_tail_ms, 800);
        }
    }
}
