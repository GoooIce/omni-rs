mod jobs;
mod results;
mod worker;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
pub use jobs::Jobs;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, mpsc::SyncSender},
};
pub use worker::ModelPaths;

#[derive(Clone)]
struct AppState {
    jobs: Jobs,
    live_inputs: Arc<Mutex<HashMap<String, SyncSender<worker::omni::LivePacket>>>>,
}
type ApiResult<T> = Result<T, (StatusCode, Json<Value>)>;
fn error(code: StatusCode, message: impl ToString) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error":message.to_string()})))
}

pub fn router(jobs: Jobs) -> Router {
    Router::new()
        .route("/",get(||async {Html(include_str!("../assets/api.html"))}))
        .route("/health",get(||async {Json(json!({"status":"ok"}))}))
        .route("/v1/models",get(||async {Json(json!({"models":["netease-youdao/Confucius4-R2T2","BreezeBlue/Breeze-TTS-2","tencent/Hy-MT2-7B","Qwen/Qwen-Image-2.1-Turbo", "m-a-p/MERT-v2-FullSong", "m-a-p/SheetSage2", "m-a-p/YuE2-3B", "LiquidAI/LFM3-Omni-2B"]}))}))
        .route("/v1/omni/generations",post(omni))
        .route("/v1/omni/live",post(omni_live_start))
        .route("/v1/omni/live/{id}",post(omni_live_push))
        .route("/v1/omni/sessions/{id}",axum::routing::delete(omni_forget))
        .route("/v1/translations",post(translate))
        .route("/v1/audio/transcriptions",post(transcribe))
        .route("/v1/audio/speech",post(speech))
        .route("/v1/images/generations",post(image))
        .route("/v1/music/analyses",post(music_analysis))
        .route("/v1/music/scores",post(music_score))
        .route("/v1/music/generations",post(music_generation))
        .route("/v1/jobs/{id}",get(job).delete(cancel))
        .route("/v1/jobs/{id}/events",get(events))
        .route("/v1/files/{name}",get(file))
        .layer(DefaultBodyLimit::max(64*1024*1024))
        .layer(middleware::from_fn(same_origin))
        .with_state(AppState {jobs,live_inputs:Arc::new(Mutex::new(HashMap::new()))})
}
// The bundled page is same-origin. Foreign browser pages cannot control local
// inference through simple requests; CLI clients need no Origin header.
async fn same_origin(headers: HeaderMap, request: axum::extract::Request, next: Next) -> Response {
    if let Some(origin) = headers.get(header::ORIGIN) {
        let host = headers
            .get(header::HOST)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        let valid = origin
            .to_str()
            .ok()
            .and_then(|o| o.parse::<axum::http::Uri>().ok())
            .is_some_and(|uri| {
                matches!(uri.scheme_str(), Some("http") | Some("https"))
                    && uri.authority().is_some_and(|a| a.as_str() == host)
            });
        if !valid {
            return error(
                StatusCode::FORBIDDEN,
                "use the API page on this server's origin",
            )
            .into_response();
        }
    }
    next.run(request).await
}
fn submit(state: &AppState, command: worker::Command) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    state
        .jobs
        .submit(command)
        .map(|view| (StatusCode::ACCEPTED, Json(view)))
        .map_err(|message| error(StatusCode::SERVICE_UNAVAILABLE, message))
}
async fn omni(
    State(state): State<AppState>,
    Json(request): Json<worker::omni::Request>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    let job = request
        .into_job()
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    submit(&state, worker::Command::Omni(job))
}
async fn omni_forget(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    if !worker::omni::valid_session_id(&id) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid Omni session ID"));
    }
    submit(&state, worker::Command::OmniForget(id))
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LiveStartQuery {
    language: Option<String>,
    route: Option<String>,
}
async fn omni_live_start(
    State(state): State<AppState>,
    Query(query): Query<LiveStartQuery>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    let route = match query.route.as_deref().unwrap_or("preview") {
        "preview" => worker::omni::LiveRoute::Preview,
        "stable" => worker::omni::LiveRoute::Stable,
        _ => {
            return Err(error(
                StatusCode::BAD_REQUEST,
                "invalid live transcription route",
            ));
        }
    };
    if query.language.as_ref().is_some_and(|name| {
        name.trim().is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphabetic() || c == b' ' || c == b'-')
    }) {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "language must be an English language name",
        ));
    }
    let mut inputs = state.live_inputs.lock().unwrap();
    inputs.retain(|id, _| {
        state
            .jobs
            .get(id)
            .is_some_and(|job| matches!(job.status, "queued" | "running"))
    });
    if inputs.len() >= 16 {
        return Err(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "too many live audio streams",
        ));
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(8);
    let (status, view) = submit(
        &state,
        worker::Command::OmniLive(worker::omni::LiveJob {
            receiver,
            language: query.language,
            route,
        }),
    )?;
    inputs.insert(view.0.id.clone(), sender);
    Ok((status, view))
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LiveInputQuery {
    #[serde(rename = "final")]
    is_final: bool,
}
async fn omni_live_push(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<LiveInputQuery>,
    bytes: Bytes,
) -> ApiResult<StatusCode> {
    if bytes.len() > 1024 * 1024
        || !bytes.len().is_multiple_of(4)
        || (bytes.is_empty() && !query.is_final)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "expected at most 1 MiB of little-endian f32 PCM",
        ));
    }
    let samples: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect();
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(error(StatusCode::BAD_REQUEST, "PCM must be finite"));
    }
    let mut inputs = state.live_inputs.lock().unwrap();
    let sender = inputs.get(&id).ok_or_else(|| {
        error(
            StatusCode::NOT_FOUND,
            "unknown or finished live audio stream",
        )
    })?;
    match sender.try_send(worker::omni::LivePacket {
        samples,
        is_final: query.is_final,
    }) {
        Ok(()) => {
            if query.is_final {
                inputs.remove(&id);
            }
            Ok(StatusCode::ACCEPTED)
        }
        Err(std::sync::mpsc::TrySendError::Full(_)) => Err(error(
            StatusCode::TOO_MANY_REQUESTS,
            "live audio input is full; retry this chunk",
        )),
        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
            inputs.remove(&id);
            Err(error(
                StatusCode::GONE,
                "live audio job is no longer running",
            ))
        }
    }
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct EventQuery {
    after: usize,
}
async fn events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<EventQuery>,
) -> ApiResult<Json<Value>> {
    let events = state
        .jobs
        .events(&id, query.after)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown or expired job"))?;
    let next = query.after.saturating_add(events.len());
    Ok(Json(json!({"events":events,"next":next})))
}
async fn translate(
    State(state): State<AppState>,
    Json(request): Json<omni_hy_mt::Request>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    request
        .validate()
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    submit(&state, worker::Command::Translate(request))
}
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AsrQuery {
    language: Option<String>,
    context: Option<String>,
}
async fn transcribe(
    State(state): State<AppState>,
    Query(query): Query<AsrQuery>,
    bytes: Bytes,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    // Parse before admission/model allocation. Only a supplied WAV is read;
    // HTTP callers cannot ask the server to open arbitrary host file paths.
    omni_r2t2::input::WavSource::new(std::io::Cursor::new(bytes.as_ref()))
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let config = omni_r2t2::StreamConfig {
        language: query.language.filter(|s| !s.is_empty()),
        context: query.context.unwrap_or_default(),
        ..Default::default()
    };
    if config.context.len() > 32768 || config.language.as_ref().is_some_and(|s| s.len() > 64) {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "ASR context or language is too long",
        ));
    }
    submit(
        &state,
        worker::Command::Transcribe {
            audio: bytes.to_vec(),
            config,
        },
    )
}
async fn speech(
    State(state): State<AppState>,
    Json(request): Json<worker::Speech>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    request
        .validate()
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    submit(&state, worker::Command::Speech(request))
}
async fn image(
    State(state): State<AppState>,
    Json(request): Json<worker::Image>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    request
        .options()
        .validate()
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    submit(&state, worker::Command::Image(request))
}
async fn job(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<jobs::View>> {
    let job = state
        .jobs
        .get(&id)
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "unknown or expired job"))?;
    Ok(Json(job))
}

async fn music_audio(bytes: Bytes, max_seconds: usize) -> ApiResult<Vec<f32>> {
    tokio::task::spawn_blocking(move || {
        omni_music::audio::read_wav(std::io::Cursor::new(bytes), max_seconds)
    })
    .await
    .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
    .map_err(|e| error(StatusCode::BAD_REQUEST, e))
}
async fn music_analysis(
    State(state): State<AppState>,
    bytes: Bytes,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    let audio = music_audio(bytes, 360).await?;
    submit(
        &state,
        worker::Command::Music(omni_music::pipeline::Request::Analyze(audio)),
    )
}
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScoreQuery {
    melody_only: bool,
}
impl Default for ScoreQuery {
    fn default() -> Self {
        Self { melody_only: true }
    }
}
async fn music_score(
    State(state): State<AppState>,
    Query(query): Query<ScoreQuery>,
    bytes: Bytes,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    let audio = music_audio(bytes, 300).await?;
    submit(
        &state,
        worker::Command::Music(omni_music::pipeline::Request::Transcribe {
            audio,
            melody_only: query.melody_only,
        }),
    )
}
async fn music_generation(
    State(state): State<AppState>,
    Json(request): Json<omni_music::SongRequest>,
) -> ApiResult<(StatusCode, Json<jobs::View>)> {
    request
        .validate()
        .map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    submit(
        &state,
        worker::Command::Music(omni_music::pipeline::Request::Generate(request)),
    )
}
async fn cancel(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<StatusCode> {
    if !state.jobs.cancel(&id) {
        return Err(error(StatusCode::NOT_FOUND, "unknown or expired job"));
    }
    Ok(StatusCode::ACCEPTED)
}
async fn file(State(state): State<AppState>, Path(name): Path<String>) -> ApiResult<Response> {
    let (bytes, content_type) = tokio::task::spawn_blocking(move || state.jobs.read_file(&name))
        .await
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .map_err(|_| error(StatusCode::NOT_FOUND, "unknown published result"))?;
    Ok(([(header::CONTENT_TYPE, content_type)], bytes).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;
    fn app() -> Router {
        let paths = ModelPaths {
            lfm3: "missing".into(),
            asr: "missing".into(),
            tts: "missing".into(),
            translation: "missing".into(),
            image: "missing".into(),
            music: "missing".into(),
            output: std::env::temp_dir(),
            precision: omni_r2t2::Precision::F32,
            cache_tokens: 16,
            translation_batch: Default::default(),
            asr_batch_size: 4,
            tts_batch_size: 4,
            image_batch_size: 1,
            asr_paged_attention: false,
            tts_paged_attention: false,
        };
        router(Jobs::spawn(paths))
    }
    struct FileExecutor;
    impl jobs::Executor for FileExecutor {
        type Command = worker::Command;
        fn recover(&mut self) {}
        fn run(
            &mut self,
            _: Self::Command,
            cx: &jobs::Context,
        ) -> Result<jobs::Output, worker::Failure> {
            let file = cx.file("wav", |path| {
                let mut writer = hound::WavWriter::create(
                    path,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 24000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )
                .unwrap();
                writer.write_sample(321i16).unwrap();
                writer.finalize().unwrap();
                Ok(())
            })?;
            let mut output = jobs::Output::new(
                json!({"audio_url":format!("/v1/files/{}.wav", cx.id())}),
                true,
            );
            output.file = Some(file);
            Ok(output)
        }
    }
    async fn get(app: &Router, path: &str) -> Response {
        app.clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }
    #[tokio::test]
    async fn downloads_survive_history_eviction_and_restart_but_hide_unpublished_files() {
        let dir = results::tests::Directory::new();
        let files = results::Results::new(dir.0.clone());
        let jobs = Jobs::with_executor(files.clone(), 4, || FileExecutor);
        let app = router(jobs.clone());
        let staged = files
            .prepare::<std::io::Error>("abcd-1.png", |path| std::fs::write(path, b"unfinished"))
            .unwrap();
        assert_eq!(
            get(&app, "/v1/files/abcd-1.png").await.status(),
            StatusCode::NOT_FOUND
        );
        drop(staged);
        let mut first = None;
        for _ in 0..34 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/audio/speech")
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"text":"hello"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::ACCEPTED);
            let view: Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let id = view["id"].as_str().unwrap().to_owned();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let view = jobs.get(&id).unwrap();
                if view.status == "cancelled" {
                    assert!(view.result.is_some());
                    break;
                }
                assert!(matches!(view.status, "queued" | "running"), "{view:?}");
                assert!(std::time::Instant::now() < deadline);
                tokio::task::yield_now().await;
            }
            first.get_or_insert(id);
        }
        let id = first.unwrap();
        let url = format!("/v1/files/{id}.wav");
        assert_eq!(
            get(&app, &format!("/v1/jobs/{id}")).await.status(),
            StatusCode::NOT_FOUND
        );
        let response = get(&app, &url).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
        let before = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let mut decoded = hound::WavReader::new(std::io::Cursor::new(&before)).unwrap();
        assert_eq!(decoded.samples::<i16>().next().unwrap().unwrap(), 321);
        jobs.shutdown();
        drop(app);
        drop(jobs);
        let restarted = router(Jobs::with_executor(files, 4, || FileExecutor));
        let response = get(&restarted, &url).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/wav");
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
            before
        );
        std::fs::remove_file(dir.0.join("published").join(&id).join(format!("{id}.wav"))).unwrap();
        assert_eq!(get(&restarted, &url).await.status(), StatusCode::NOT_FOUND);
    }
    #[tokio::test]
    async fn bad_inputs_are_rejected_before_model_loading() {
        for (path, body, content_type) in [
            (
                "/v1/translations",
                r#"{"text":"","target_language":"English"}"#,
                "application/json",
            ),
            ("/v1/audio/speech", r#"{"text":"   "}"#, "application/json"),
            (
                "/v1/images/generations",
                r#"{"prompt":""}"#,
                "application/json",
            ),
            ("/v1/audio/transcriptions", "bad WAV", "audio/wav"),
            ("/v1/music/analyses", "bad WAV", "audio/wav"),
            ("/v1/music/scores", "bad WAV", "audio/wav"),
            (
                "/v1/music/generations",
                r#"{"style":""}"#,
                "application/json",
            ),
            (
                "/v1/music/generations",
                r#"{"style":"piano","cot":"off","abc":"X:1"}"#,
                "application/json",
            ),
        ] {
            let response = app()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", content_type)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
    }
    #[tokio::test]
    async fn unknown_jobs_and_cross_origin_mutations_are_rejected() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/jobs/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let response = app()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/v1/jobs/missing")
                    .header("origin", "https://elsewhere.example")
                    .header("host", "127.0.0.1:8848")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
    struct OmniEventExecutor;
    impl jobs::Executor for OmniEventExecutor {
        type Command = worker::Command;
        fn recover(&mut self) {}
        fn run(
            &mut self,
            command: Self::Command,
            cx: &jobs::Context,
        ) -> Result<jobs::Output, worker::Failure> {
            assert!(matches!(command, worker::Command::Omni(_)));
            cx.event(json!({"type":"text","token":17}));
            while !cx.cancel().load(std::sync::atomic::Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            cx.event(json!({"type":"stopped"}));
            Ok(jobs::Output::new(json!({"partial":true}), true))
        }
    }
    struct OmniLiveExecutor;
    struct OmniRouteExecutor;
    impl jobs::Executor for OmniRouteExecutor {
        type Command = worker::Command;
        fn recover(&mut self) {}
        fn run(
            &mut self,
            command: Self::Command,
            cx: &jobs::Context,
        ) -> Result<jobs::Output, worker::Failure> {
            let worker::Command::OmniLive(job) = command else {
                panic!("expected live audio");
            };
            let route = match job.route {
                worker::omni::LiveRoute::Preview => "preview",
                worker::omni::LiveRoute::Stable => "stable",
            };
            cx.event(json!({"route":route}));
            Ok(jobs::Output::new(json!({"route":route}), false))
        }
    }
    #[tokio::test]
    async fn live_route_defaults_to_preview_and_can_select_stable() {
        let dir = results::tests::Directory::new();
        let jobs = Jobs::with_executor(results::Results::new(dir.0.clone()), 4, || {
            OmniRouteExecutor
        });
        let app = router(jobs.clone());
        for (uri, expected) in [
            ("/v1/omni/live", "preview"),
            ("/v1/omni/live?route=stable&language=Chinese", "stable"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::ACCEPTED);
            let view: Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap(),
            )
            .unwrap();
            let id = view["id"].as_str().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while jobs.events(id, 0).unwrap().is_empty() {
                assert!(std::time::Instant::now() < deadline);
                tokio::task::yield_now().await;
            }
            assert_eq!(jobs.events(id, 0).unwrap()[0]["route"], expected);
        }
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/omni/live?route=unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    impl jobs::Executor for OmniLiveExecutor {
        type Command = worker::Command;
        fn recover(&mut self) {}
        fn run(
            &mut self,
            command: Self::Command,
            cx: &jobs::Context,
        ) -> Result<jobs::Output, worker::Failure> {
            let worker::Command::OmniLive(job) = command else {
                panic!("expected live audio");
            };
            loop {
                if cx.cancel().load(std::sync::atomic::Ordering::Acquire) {
                    return Ok(jobs::Output::new(json!({"partial":true}), true));
                }
                match job
                    .receiver
                    .recv_timeout(std::time::Duration::from_millis(5))
                {
                    Ok(packet) => {
                        cx.event(json!({"type":"samples","samples":packet.samples}));
                        if packet.is_final {
                            return Ok(jobs::Output::new(json!({"partial":false}), false));
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return Err(worker::Failure::Error("input disconnected".into())),
                }
            }
        }
    }
    #[tokio::test]
    async fn live_audio_accepts_ordered_chunks_delivers_before_eof_and_can_cancel() {
        let dir = results::tests::Directory::new();
        let jobs =
            Jobs::with_executor(results::Results::new(dir.0.clone()), 4, || OmniLiveExecutor);
        let app = router(jobs.clone());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/omni/live")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let view: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap();
        let id = view["id"].as_str().unwrap();
        let uri = format!("/v1/omni/live/{id}");
        let bad = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .body(Body::from(vec![1_u8, 2, 3]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let pcm = [0.25_f32.to_le_bytes(), (-0.5_f32).to_le_bytes()].concat();
        let pushed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .body(Body::from(pcm))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(pushed.status(), StatusCode::ACCEPTED);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while jobs.events(id, 0).unwrap().is_empty() {
            assert!(std::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        assert_eq!(jobs.get(id).unwrap().status, "running");
        assert_eq!(
            jobs.events(id, 0).unwrap()[0]["samples"],
            json!([0.25, -0.5])
        );
        let final_uri = format!("{uri}?final=true");
        let ended = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&final_uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ended.status(), StatusCode::ACCEPTED);
        while jobs.get(id).unwrap().status != "done" {
            assert!(std::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/omni/live")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let view: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap();
        let id = view["id"].as_str().unwrap();
        assert!(jobs.cancel(id));
        while jobs.get(id).unwrap().status != "cancelled" {
            assert!(std::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        jobs.shutdown();
    }
    #[tokio::test]
    async fn omni_upload_streams_ordered_events_and_cancels_without_host_paths() {
        let dir = results::tests::Directory::new();
        let jobs = Jobs::with_executor(results::Results::new(dir.0.clone()), 4, || {
            OmniEventExecutor
        });
        let app = router(jobs.clone());
        let body=json!({"messages":[{"role":"user","content":[{"type":"text","text":"inspect"},{"type":"image","data":"AQID"}]}],"mode":"text","max_slots":4}).to_string();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/omni/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let view: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap();
        let id = view["id"].as_str().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !jobs.events(id, 0).unwrap().is_empty() {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/jobs/{id}/events?after=0"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let events: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(events["events"][0]["token"], 17);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/jobs/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        loop {
            if jobs.get(id).unwrap().status == "cancelled" {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        assert_eq!(jobs.events(id, 1).unwrap()[0]["type"], "stopped");
        let bad =
            json!({"messages":[{"role":"user","content":[{"type":"image","path":"/etc/passwd"}]}]})
                .to_string();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/omni/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(bad))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        jobs.shutdown();
    }
}
