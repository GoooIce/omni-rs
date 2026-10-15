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
pub use worker::ModelPaths;

#[derive(Clone)]
struct AppState {
    jobs: Jobs,
}
type ApiResult<T> = Result<T, (StatusCode, Json<Value>)>;
fn error(code: StatusCode, message: impl ToString) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error":message.to_string()})))
}

pub fn router(jobs: Jobs) -> Router {
    Router::new()
        .route("/",get(||async {Html(include_str!("../../../web/api.html"))}))
        .route("/health",get(||async {Json(json!({"status":"ok"}))}))
        .route("/v1/models",get(||async {Json(json!({"models":["netease-youdao/Confucius4-R2T2","BreezeBlue/Breeze-TTS-2","tencent/Hy-MT2-7B","Qwen/Qwen-Image-2.1-Turbo", "m-a-p/MERT-v2-FullSong", "m-a-p/SheetSage2", "m-a-p/YuE2-3B"]}))}))
        .route("/v1/translations",post(translate))
        .route("/v1/audio/transcriptions",post(transcribe))
        .route("/v1/audio/speech",post(speech))
        .route("/v1/images/generations",post(image))
        .route("/v1/music/analyses",post(music_analysis))
        .route("/v1/music/scores",post(music_score))
        .route("/v1/music/generations",post(music_generation))
        .route("/v1/jobs/{id}",get(job).delete(cancel))
        .route("/v1/files/{name}",get(file))
        .layer(DefaultBodyLimit::max(64*1024*1024))
        .layer(middleware::from_fn(same_origin))
        .with_state(AppState {jobs})
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
            (
                "/v1/audio/speech",
                r#"{"text":"hello","max_frames":0}"#,
                "application/json",
            ),
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
}
