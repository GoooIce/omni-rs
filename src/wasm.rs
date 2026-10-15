use crate::{Model, ModelConfig, Precision, StreamConfig, Transcription, weights::DeviceWeights};
use burn::backend::wgpu::{WebGpu, WgpuDevice, graphics, init_setup_async};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

fn js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

/// The same bounded-history sinc resampler as native capture. Flush on stop.
#[wasm_bindgen]
pub struct WebResampler {
    inner: crate::audio::Resampler,
}

#[wasm_bindgen]
impl WebResampler {
    #[wasm_bindgen(constructor)]
    pub fn new(input_rate: u32) -> Result<WebResampler, JsValue> {
        Ok(Self {
            inner: crate::audio::Resampler::new(input_rate, 16000).map_err(js_error)?,
        })
    }

    pub fn push(&mut self, pcm: Vec<f32>, final_chunk: bool) -> Result<Vec<f32>, JsValue> {
        self.inner.push(&pcm, final_chunk).map_err(js_error)
    }
}

#[wasm_bindgen]
pub struct WebModelBuilder {
    config: ModelConfig,
    tokenizer: Vec<u8>,
    weights: DeviceWeights<WebGpu>,
    device: WgpuDevice,
    precision: Precision,
}

#[wasm_bindgen]
impl WebModelBuilder {
    pub async fn create(
        config_json: String,
        tokenizer: Vec<u8>,
        precision: String,
    ) -> Result<WebModelBuilder, JsValue> {
        let config = ModelConfig::from_json(config_json.as_bytes()).map_err(js_error)?;
        let precision = match precision.as_str() {
            "f16" => Precision::F16,
            "f32" => Precision::F32,
            _ => return Err(js_error("precision must be f16 or f32")),
        };
        let device = WgpuDevice::default();
        init_setup_async::<graphics::WebGpu>(&device, Default::default()).await;
        precision
            .validate_device::<WebGpu>(&device)
            .map_err(js_error)?;
        Ok(Self {
            config,
            tokenizer,
            weights: DeviceWeights::new(precision, &device),
            device,
            precision,
        })
    }

    /// `offset` counts elements, not bytes. Pieces of a tensor must arrive in order.
    #[wasm_bindgen(js_name=addTensorChunk)]
    pub fn add_tensor_chunk(
        &mut self,
        name: String,
        shape: Vec<u32>,
        dtype: String,
        offset: u32,
        bytes: Vec<u8>,
    ) -> Result<(), JsValue> {
        let dtype = match dtype.as_str() {
            "BF16" => safetensors::Dtype::BF16,
            "F16" => safetensors::Dtype::F16,
            "F32" => safetensors::Dtype::F32,
            _ => return Err(js_error("unsupported tensor dtype")),
        };
        self.weights
            .add_chunk(
                name,
                shape.into_iter().map(|v| v as usize).collect(),
                dtype,
                offset as usize,
                &bytes,
            )
            .map_err(js_error)
    }

    pub fn finish(self) -> Result<WebModel, JsValue> {
        let model = Model::load(
            self.config,
            &self.tokenizer,
            &self.weights,
            self.precision,
            &self.device,
        )
        .map_err(js_error)?;
        Ok(WebModel {
            model: Arc::new(model),
        })
    }
}

#[wasm_bindgen]
pub struct WebModel {
    model: Arc<Model<WebGpu>>,
}
#[wasm_bindgen]
impl WebModel {
    #[wasm_bindgen(js_name=createSession)]
    pub fn create_session(&self, config_json: String) -> Result<WebSession, JsValue> {
        self.create_transcription(config_json, 16000)
    }

    /// Mono packets may use the capture device's rate; EOF flushes rate conversion.
    #[wasm_bindgen(js_name=createTranscription)]
    pub fn create_transcription(
        &self,
        config_json: String,
        input_rate: u32,
    ) -> Result<WebSession, JsValue> {
        let config: StreamConfig = serde_json::from_str(&config_json).map_err(js_error)?;
        Ok(WebSession {
            transcription: Transcription::new(self.model.clone(), config, input_rate)
                .map_err(js_error)?,
        })
    }
}

#[wasm_bindgen]
pub struct WebSession {
    transcription: Transcription<WebGpu>,
}

/// Independent from WebSession's mutable Wasm borrow during an async push.
#[wasm_bindgen]
pub struct WebStopHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
}
#[wasm_bindgen]
impl WebStopHandle {
    pub fn stop(&self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
    }
}
#[wasm_bindgen]
impl WebSession {
    #[wasm_bindgen(js_name=stopHandle)]
    pub fn stop_handle(&self) -> WebStopHandle {
        WebStopHandle {
            stop: self.transcription.stop_handle(),
        }
    }

    pub fn interrupt(&mut self) -> Result<JsValue, JsValue> {
        let event = self.transcription.interrupt().map_err(session_error)?;
        event_value(&event)
    }
    /// Input is mono Float32 PCM at the configured rate. Await every push.
    pub async fn push(&mut self, pcm: Vec<f32>) -> Result<JsValue, JsValue> {
        let events = self.transcription.push(&pcm).await.map_err(session_error)?;
        event_value(&events)
    }

    /// All EOF events, including a decode triggered by the resampler tail.
    #[wasm_bindgen(js_name=finishEvents)]
    pub async fn finish_events(&mut self) -> Result<JsValue, JsValue> {
        let events = self.transcription.finish().await.map_err(session_error)?;
        event_value(&events)
    }

    /// Compatibility for callers expecting one terminal event. Deltas and work
    /// from any resampler-tail events are folded into that event.
    pub async fn finish(&mut self) -> Result<JsValue, JsValue> {
        let mut events = self.transcription.finish().await.map_err(session_error)?;
        let mut event = events.pop().expect("EOF always produces a terminal event");
        if !events.is_empty() {
            let mut delta = String::new();
            for previous in events {
                delta.push_str(&previous.event.delta);
                event.event.compute_ms += previous.event.compute_ms;
                event.event.generated_tokens += previous.event.generated_tokens;
                event.event.reused_audio_tokens += previous.event.reused_audio_tokens;
            }
            delta.push_str(&event.event.delta);
            event.event.delta = delta;
        }
        event_value(&event)
    }
}

// Flattened serde structs serialize as maps. Workers consume plain objects via
// event.is_final and object spread, so do not use the default JS Map serializer.
fn event_value(event: &impl serde::Serialize) -> Result<JsValue, JsValue> {
    event
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
        .map_err(js_error)
}

fn session_error(error: crate::Error) -> JsValue {
    let value = js_sys::Error::new(&error.to_string());
    if matches!(error, crate::Error::Interrupted) {
        value.set_name("Interrupted");
    }
    value.into()
}
