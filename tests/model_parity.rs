//! Numerical invariants tested on a small, fully populated Qwen3-ASR architecture.
//! These tests check caching and weight layout without downloading a real model.
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use omni_r2t2::{
    Precision,
    audio::{MelFeatures, MelFrontend},
    model::{
        decoder::DecoderCache,
        encoder::{EncoderCache, audio_token_count},
    },
    weights::{CheckpointLayout, CheckpointStage, DeviceWeights, WeightSource},
};
mod support;
use support::{MemoryWeights, tiny_config, tiny_model, tiny_tokenizer};

type B = NdArray<f32>;

fn values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}
fn close(actual: Vec<f32>, expected: Vec<f32>) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(error < 2e-5, "maximum error: {error}");
}

#[test]
fn model_loading_validates_checkpoint_and_tokenizer_through_the_public_interface() {
    let config = tiny_config();
    let tokenizer = tiny_tokenizer();
    let device = Default::default();
    let layout = CheckpointLayout::new(&config).unwrap();
    let first = &layout.tensors()[0].name;

    let mut missing = MemoryWeights::new(&config);
    missing.remove(first);
    assert!(matches!(
        omni_r2t2::Model::<B>::load(config.clone(), &tokenizer, &missing, Precision::F32, &device),
        Err(omni_r2t2::Error::Checkpoint(message)) if message.contains(first)
    ));

    let mut malformed = MemoryWeights::new(&config);
    malformed.corrupt_shape(first);
    assert!(matches!(
        omni_r2t2::Model::<B>::load(config.clone(), &tokenizer, &malformed, Precision::F32, &device),
        Err(omni_r2t2::Error::Checkpoint(message)) if message.contains(first)
    ));

    let weights = MemoryWeights::new(&config);
    let mut incompatible = config.clone();
    incompatible.thinker_config.audio_token_id = 7;
    assert!(matches!(
        omni_r2t2::Model::<B>::load(incompatible, &tokenizer, &weights, Precision::F32, &device),
        Err(omni_r2t2::Error::Tokenizer(_))
    ));

    let model =
        omni_r2t2::Model::<B>::load(config, &tokenizer, &weights, Precision::F32, &device).unwrap();
    assert_eq!(model.precision(), Precision::F32);
    assert_eq!(model.config().thinker_config.text_config.hidden_size, 16);
}

#[test]
fn loading_progress_classifies_every_requested_tensor_in_model_load_order() {
    struct ObservedWeights {
        inner: MemoryWeights,
        layout: CheckpointLayout,
        stages: std::cell::RefCell<Vec<CheckpointStage>>,
    }
    impl ObservedWeights {
        fn observe(&self, name: &str) {
            let stage = self
                .layout
                .stage(name)
                .expect("loaded tensor must have a stage");
            let mut stages = self.stages.borrow_mut();
            if stages.last() != Some(&stage) {
                stages.push(stage);
            }
        }
    }
    impl WeightSource<B> for ObservedWeights {
        fn contains(&self, name: &str) -> bool {
            <MemoryWeights as WeightSource<B>>::contains(&self.inner, name)
        }
        fn tensor<const D: usize>(
            &self,
            name: &str,
            shape: [usize; D],
            precision: Precision,
            device: &burn::backend::ndarray::NdArrayDevice,
        ) -> omni_r2t2::Result<Tensor<B, D>> {
            self.observe(name);
            self.inner.tensor(name, shape, precision, device)
        }
        fn rows(
            &self,
            name: &str,
            shape: [usize; 2],
            start: usize,
            end: usize,
            precision: Precision,
            device: &burn::backend::ndarray::NdArrayDevice,
        ) -> omni_r2t2::Result<Tensor<B, 2>> {
            self.observe(name);
            self.inner.rows(name, shape, start, end, precision, device)
        }
    }
    let config = tiny_config();
    let weights = ObservedWeights {
        inner: MemoryWeights::new(&config),
        layout: CheckpointLayout::new(&config).unwrap(),
        stages: Default::default(),
    };
    assert_eq!(weights.layout.stage("unknown"), None);
    omni_r2t2::Model::<B>::load(
        config,
        &tiny_tokenizer(),
        &weights,
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        weights.stages.into_inner(),
        [
            CheckpointStage::EncoderLayer(0),
            CheckpointStage::EncoderLayer(1),
            CheckpointStage::AudioFeatures,
            CheckpointStage::Vocabulary,
            CheckpointStage::DecoderLayer(0),
            CheckpointStage::DecoderLayer(1),
            CheckpointStage::TextOutput,
        ]
    );
}

#[test]
fn decoder_chunked_prefill_and_kv_rollback_match_full_forward() {
    let model = tiny_model();
    let decoder = model.diagnostics().decoder;
    let tokens = [1, 4, 2, 7, 3, 5, 9];
    let mut cache = DecoderCache::new(2);
    let full = decoder
        .forward(decoder.embed(&tokens).unwrap(), &mut cache)
        .unwrap();
    let mut cache = DecoderCache::new(2);
    let _ = decoder
        .forward(decoder.embed(&tokens[..3]).unwrap(), &mut cache)
        .unwrap();
    let chunked = decoder
        .forward(decoder.embed(&tokens[3..]).unwrap(), &mut cache)
        .unwrap();
    close(values(chunked), values(full));
    cache.truncate(3);
    let revised = [1, 4, 2, 11, 12];
    let actual = decoder
        .forward(decoder.embed(&revised[3..]).unwrap(), &mut cache)
        .unwrap();
    let expected = decoder
        .forward(decoder.embed(&revised).unwrap(), &mut DecoderCache::new(2))
        .unwrap();
    close(values(actual), values(expected));
}

#[test]
fn encoder_closed_windows_match_recomputation_and_loudness_invalidates_cache() {
    let model = tiny_model();
    let encoder = model.diagnostics().encoder;
    let mut mel = MelFrontend::new();
    let pcm: Vec<f32> = (0..16000 * 9)
        .map(|i| (i as f32 * 0.09).sin() * 0.01)
        .collect();
    // Once 8.02 seconds have arrived, the first 8 second window is stable.
    mel.push(&pcm[..128320]).unwrap();
    let first = mel.features();
    let mut cache = EncoderCache::default();
    let encoded = encoder.forward(&first, &mut cache, true).unwrap();
    assert_eq!(
        encoded.embeddings.dims()[0],
        audio_token_count(first.frames)
    );
    // Identical features must reuse 104 tokens, preserving every embedding.
    let reused = encoder.forward(&first, &mut cache, true).unwrap();
    assert_eq!(reused.reused_tokens, 104);
    close(values(reused.embeddings), values(encoded.embeddings));
    let loud: Vec<_> = (0..2560).map(|i| (i as f32 * 0.14).sin() * 0.8).collect();
    mel.push(&loud).unwrap();
    let features = mel.features();
    let cached = encoder.forward(&features, &mut cache, true).unwrap();
    assert_eq!(cached.reused_tokens, 0);
    assert_eq!(cached.reused_convolution_blocks, 0);
    let fresh = encoder
        .forward(&features, &mut EncoderCache::default(), false)
        .unwrap();
    close(values(cached.embeddings), values(fresh.embeddings));
}

#[test]
fn stable_convolution_prefixes_match_full_recomputation_across_boundaries() {
    // The provisional tail changes at each call; stable prefix values depend
    // only on their mel/time position. This detects caching a partial block or
    // storing a full block before its last centered STFT frames are stable.
    fn features(frames: usize) -> MelFeatures {
        let stable_frames = frames.saturating_sub(2);
        let values = (0..128)
            .flat_map(|m| {
                (0..frames).map(move |t| {
                    let stable_value = ((m * 73 + t * 17) as f32 * 0.01).sin();
                    stable_value
                        + if t >= stable_frames {
                            frames as f32 * 0.001
                        } else {
                            0.
                        }
                })
            })
            .collect();
        MelFeatures {
            values,
            frames,
            stable_frames,
            log_max: 1.,
        }
    }
    let model = tiny_model();
    let encoder = model.diagnostics().encoder;
    let mut cache = EncoderCache::default();
    for frames in [
        1, 8, 32, 99, 100, 101, 102, 160, 199, 200, 201, 202, 317, 799, 800, 801, 802, 850, 1001,
    ] {
        let mel = features(frames);
        let cached = encoder.forward(&mel, &mut cache, true).unwrap();
        let fresh = encoder
            .forward(&mel, &mut EncoderCache::default(), false)
            .unwrap();
        assert_eq!(cached.embeddings.dims()[0], audio_token_count(frames));
        if frames == 160 {
            assert_eq!(cached.reused_convolution_blocks, 1);
            assert_eq!(cached.reused_tokens, 0);
        }
        if frames == 317 {
            assert_eq!(cached.reused_convolution_blocks, 2);
        }
        if frames == 850 {
            assert_eq!(cached.reused_tokens, 104);
            assert_eq!(cached.reused_convolution_blocks, 0);
        }
        close(values(cached.embeddings), values(fresh.embeddings));
    }

    // A shorter input invalidates all saved prefixes even at the same log max.
    let mel = features(160);
    let shrunk = encoder.forward(&mel, &mut cache, true).unwrap();
    assert_eq!(shrunk.reused_tokens, 0);
    assert_eq!(shrunk.reused_convolution_blocks, 0);
    close(
        values(shrunk.embeddings),
        values(
            encoder
                .forward(&mel, &mut EncoderCache::default(), false)
                .unwrap()
                .embeddings,
        ),
    );
    let uncached = encoder.forward(&mel, &mut cache, false).unwrap();
    assert_eq!(uncached.reused_tokens, 0);
    assert_eq!(uncached.reused_convolution_blocks, 0);
    cache.clear();
    let cleared = encoder.forward(&mel, &mut cache, true).unwrap();
    assert_eq!(cleared.reused_convolution_blocks, 0);
    close(values(cleared.embeddings), values(uncached.embeddings));
}

#[test]
fn convolution_lengths_restart_at_every_hundred_frames() {
    assert_eq!(audio_token_count(1), 1);
    assert_eq!(audio_token_count(8), 1);
    assert_eq!(audio_token_count(100), 13);
    assert_eq!(audio_token_count(101), 14);
    assert_eq!(audio_token_count(800), 104);
    assert_eq!(audio_token_count(801), 105);
    let model = tiny_model();
    let encoder = model.diagnostics().encoder;
    for frames in [1, 8, 99, 100, 101] {
        let mut frontend = MelFrontend::new();
        frontend.push(&vec![0.0; frames * 160]).unwrap();
        let output = encoder
            .forward(&frontend.features(), &mut EncoderCache::default(), false)
            .unwrap();
        assert_eq!(output.embeddings.dims(), [audio_token_count(frames), 16]);
        assert!(values(output.embeddings).iter().all(|v| v.is_finite()));
    }
}

#[test]
fn browser_weight_chunks_preserve_layout_across_slice_boundaries() {
    let device = Default::default();
    let mut weights = DeviceWeights::<B>::new(Precision::F32, &device);
    let original: Vec<f32> = (0..24).map(|x| x as f32 * 0.25).collect();
    let bytes: Vec<u8> = original.iter().flat_map(|x| x.to_le_bytes()).collect();
    weights
        .add_chunk(
            "x".into(),
            vec![6, 4],
            safetensors::Dtype::F32,
            0,
            &bytes[..40],
        )
        .unwrap();
    assert!(
        weights
            .rows("x", [6, 4], 0, 2, Precision::F32, &device)
            .is_err()
    );
    assert!(
        weights
            .add_chunk(
                "x".into(),
                vec![6, 4],
                safetensors::Dtype::F32,
                11,
                &bytes[40..]
            )
            .is_err()
    );
    weights
        .add_chunk(
            "x".into(),
            vec![6, 4],
            safetensors::Dtype::F32,
            10,
            &bytes[40..],
        )
        .unwrap();
    close(
        values(
            weights
                .rows("x", [6, 4], 1, 5, Precision::F32, &device)
                .unwrap(),
        ),
        original[4..20].to_vec(),
    );
    let expected = Tensor::<B, 2>::from_data(TensorData::new(original, [6, 4]), &device);
    close(
        values(
            weights
                .tensor("x", [6, 4], Precision::F32, &device)
                .unwrap(),
        ),
        values(expected),
    );
}

#[test]
fn streaming_packet_boundaries_do_not_change_committed_output_or_lose_the_tail() {
    let model = tiny_model();
    for text_postprocess in [
        omni_r2t2::TextPostprocess::Legacy,
        omni_r2t2::TextPostprocess::Official,
    ] {
        let config = omni_r2t2::StreamConfig {
            text_postprocess,
            chunk_ms: 80,
            lookahead_ms: 80,
            max_new_tokens: 2,
            final_max_tokens: 4,
            max_context_tokens: 1024,
            final_padding_ms: 640,
            ..Default::default()
        };
        let pcm: Vec<_> = (0..7699).map(|i| (i as f32 * 0.07).sin() * 0.2).collect();
        let mut one = omni_r2t2::Session::new(model.clone(), config.clone()).unwrap();
        let mut many = omni_r2t2::Session::new(model.clone(), config).unwrap();
        let mut expected = pollster::block_on(one.push(&pcm)).unwrap();
        expected.push(pollster::block_on(one.finish()).unwrap());
        let mut actual = Vec::new();
        for packet in pcm.chunks(137) {
            actual.extend(pollster::block_on(many.push(packet)).unwrap());
        }
        actual.push(pollster::block_on(many.finish()).unwrap());
        assert_eq!(actual.len(), expected.len());
        let mut committed = String::new();
        for (a, b) in actual.iter().zip(&expected) {
            committed.push_str(&a.delta);
            assert_eq!(committed, a.text);
            assert_eq!(
                (&a.delta, &a.text, a.audio_seconds, a.is_final),
                (&b.delta, &b.text, b.audio_seconds, b.is_final)
            );
        }
        assert_eq!(actual.last().unwrap().audio_seconds, 7699.0 / 16000.0);
        assert!(pollster::block_on(many.push(&[0.0])).is_err());
        assert!(pollster::block_on(many.finish()).is_err());
    }
}

#[test]
fn session_cache_reuse_matches_recomputation_across_audio_window_rollover() {
    let model = tiny_model();
    let config = omni_r2t2::StreamConfig {
        chunk_ms: 2000,
        lookahead_ms: 0,
        window_seconds: 8,
        max_new_tokens: 2,
        final_max_tokens: 4,
        max_context_tokens: 1024,
        ..Default::default()
    };
    let mut cached = omni_r2t2::Session::new(model.clone(), config.clone()).unwrap();
    let mut fresh = omni_r2t2::Session::new(
        model,
        omni_r2t2::StreamConfig {
            reuse_cache: false,
            ..config
        },
    )
    .unwrap();
    let pcm: Vec<_> = (0..16000 * 10 + 319)
        .map(|i| (i as f32 * 0.07).sin() * 0.2)
        .collect();
    let mut actual = pollster::block_on(cached.push(&pcm)).unwrap();
    actual.push(pollster::block_on(cached.finish()).unwrap());
    let mut expected = pollster::block_on(fresh.push(&pcm)).unwrap();
    expected.push(pollster::block_on(fresh.finish()).unwrap());
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(&expected) {
        assert_eq!(
            (&a.delta, &a.text, &a.language, a.audio_seconds, a.is_final),
            (&b.delta, &b.text, &b.language, b.audio_seconds, b.is_final),
        );
    }
    assert_eq!(
        actual.last().unwrap().audio_seconds,
        pcm.len() as f64 / 16000.0
    );
}

#[test]
fn finalization_decodes_on_exact_chunk_boundaries_and_handles_empty_streams() {
    let model = tiny_model();
    for padding in [0, 640] {
        for samples in [0, 1, 1279, 1280, 1281, 2560] {
            let config = omni_r2t2::StreamConfig {
                chunk_ms: 80,
                lookahead_ms: 0,
                max_new_tokens: 2,
                final_max_tokens: 4,
                max_context_tokens: 1024,
                final_padding_ms: padding,
                ..Default::default()
            };
            let mut session = omni_r2t2::Session::new(model.clone(), config).unwrap();
            assert!(pollster::block_on(session.push(&[f32::NAN])).is_err());
            let mut events = pollster::block_on(session.push(&vec![0.0; samples])).unwrap();
            assert_eq!(events.len(), samples / 1280);
            let event = pollster::block_on(session.finish()).unwrap();
            assert!(event.is_final);
            assert_eq!(event.audio_seconds, samples as f64 / 16000.0);
            if samples == 0 {
                assert!(event.text.is_empty() && event.compute_ms == 0.0);
            } else {
                assert!(event.compute_ms > 0.0);
            }
            events.push(event);
            assert_eq!(events.iter().filter(|e| e.is_final).count(), 1);
            let deltas: String = events.iter().map(|e| e.delta.as_str()).collect();
            assert_eq!(deltas, events.last().unwrap().text);
        }
    }
}

#[test]
fn interruption_drops_pending_audio_and_preserves_committed_text_without_decoding() {
    use std::sync::atomic::Ordering;
    let mut session = omni_r2t2::Session::new(
        tiny_model(),
        omni_r2t2::StreamConfig {
            chunk_ms: 80,
            lookahead_ms: 0,
            max_new_tokens: 2,
            final_max_tokens: 4,
            max_context_tokens: 1024,
            final_padding_ms: 640,
            ..Default::default()
        },
    )
    .unwrap();
    let events = pollster::block_on(session.push(&[0.1; 1400])).unwrap();
    let text: String = events.iter().map(|e| e.delta.as_str()).collect();
    session.stop_handle().store(true, Ordering::Release);
    assert!(matches!(
        pollster::block_on(session.push(&[0.2; 16000])),
        Err(omni_r2t2::Error::Interrupted)
    ));
    assert!(matches!(
        pollster::block_on(session.finish()),
        Err(omni_r2t2::Error::Interrupted)
    ));
    let final_event = session.interrupt();
    assert_eq!(final_event.text, text);
    assert_eq!(final_event.audio_seconds, 0.08);
    assert_eq!(final_event.compute_ms, 0.0);
    assert!(final_event.delta.is_empty() && final_event.is_final);
    assert_eq!(session.interrupt().text, text);
    assert!(pollster::block_on(session.push(&[0.1])).is_err());
}

#[test]
fn loading_interruption_stops_weight_reads_and_allows_a_fresh_retry() {
    use std::{
        cell::{Cell, RefCell},
        sync::atomic::{AtomicBool, Ordering},
    };
    struct Observed {
        inner: MemoryWeights,
        reads: RefCell<Vec<String>>,
    }
    impl WeightSource<B> for Observed {
        fn contains(&self, name: &str) -> bool {
            <MemoryWeights as WeightSource<B>>::contains(&self.inner, name)
        }
        fn tensor<const D: usize>(
            &self,
            name: &str,
            shape: [usize; D],
            p: Precision,
            d: &burn::backend::ndarray::NdArrayDevice,
        ) -> omni_r2t2::Result<Tensor<B, D>> {
            self.reads.borrow_mut().push(name.into());
            self.inner.tensor(name, shape, p, d)
        }
        fn rows(
            &self,
            name: &str,
            shape: [usize; 2],
            start: usize,
            end: usize,
            p: Precision,
            d: &burn::backend::ndarray::NdArrayDevice,
        ) -> omni_r2t2::Result<Tensor<B, 2>> {
            self.reads.borrow_mut().push(name.into());
            self.inner.rows(name, shape, start, end, p, d)
        }
    }
    let config = tiny_config();
    let weights = Observed {
        inner: MemoryWeights::new(&config),
        reads: Default::default(),
    };
    let cancel = AtomicBool::new(false);
    let at_stop = Cell::new(None);
    let result = omni_r2t2::Model::<B>::load_cancellable(
        config.clone(),
        &tiny_tokenizer(),
        &weights,
        Precision::F32,
        &Default::default(),
        &cancel,
        |_, stage| {
            if stage.starts_with("文本解码器") {
                at_stop.set(Some(weights.reads.borrow().len()));
                cancel.store(true, Ordering::Release);
            }
        },
    );
    assert!(matches!(result, Err(omni_r2t2::Error::Interrupted)));
    assert_eq!(Some(weights.reads.borrow().len()), at_stop.get());
    assert!(at_stop.get().unwrap() > 0);
    cancel.store(false, Ordering::Release);
    let retry = omni_r2t2::Model::<B>::load_cancellable(
        config,
        &tiny_tokenizer(),
        &weights,
        Precision::F32,
        &Default::default(),
        &cancel,
        |_, _| {},
    )
    .unwrap();
    assert_eq!(retry.config().thinker_config.text_config.hidden_size, 16);
    assert!(weights.reads.borrow().len() > at_stop.get().unwrap());
}
