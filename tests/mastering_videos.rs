use anyhow::{anyhow, bail, Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use asr_api::decoder::DecoderState;
use asr_api::ingress::ListenIngress;
use asr_api::worker::WorkerState;
use av_ingest_proxy::{TranscribeAudioResolver, TranscribeAudioStream};
use bytes::Bytes;
use futures_util::StreamExt;
use http::{header::CONTENT_TYPE, Request};
use serde_json::{json, Value};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use upload_response::{ResponseWatcher, UploadResponseService};
use web_service::{BodyStream, HandlerResponse, ServerError};

const DEFAULT_MASTERING_URLS: &[&str] = &[
    "https://www.youtube.com/watch?v=Szv32PCJfs0",
    "https://www.youtube.com/watch?v=d8hA8eOMxCY",
    "https://www.youtube.com/watch?v=XXCPQe4qzpc",
    "https://www.youtube.com/watch?v=M88T8jFL2uU",
    "https://www.youtube.com/watch?v=ZHXD-BlKyL8",
    "https://www.youtube.com/watch?v=9VmjOIid2C4",
];
const DEFAULT_UPLOAD_RESPONSE_SLOT_SIZE_KB: usize = 32;
const DEFAULT_UPLOAD_RESPONSE_RING_BYTES: usize = 1024 * 1024 * 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcribes_audio_mastering_videos() -> Result<()> {
    if !env_flag("MEDIA_RESEARCH_STACK_MASTERING_BENCH") {
        eprintln!("skipping long mastering benchmark; set MEDIA_RESEARCH_STACK_MASTERING_BENCH=1");
        return Ok(());
    }

    env::set_var("ASR_COHERE_BACKEND", "mlx");
    env::set_var("AV_INGEST_PROXY_RESOLVE_MODE", "transcribe");
    ensure_mlx_metallib_for_current_exe()?;

    let model_dir = PathBuf::from(env::var("ASR_MODEL_DIR").context("ASR_MODEL_DIR is required")?);
    let urls = mastering_urls()?;
    let startup_grace =
        Duration::from_secs(env_u64("MEDIA_RESEARCH_STACK_STARTUP_GRACE_SECS", 20)?);

    let resolver = TranscribeAudioResolver::from_env()?;
    let asr = LocalAsrHarness::new(model_dir)?;
    tokio::time::sleep(startup_grace).await;

    let report_path = report_path()?;
    eprintln!("writing benchmark report to {}", report_path.display());

    let mut completed = 0usize;
    let mut aggregate_audio_seconds = 0.0f64;
    let mut aggregate_wall_seconds = 0.0f64;

    for (index, source_url) in urls.iter().enumerate() {
        let started_at = Instant::now();
        let audio = resolver.open_youtube_audio(source_url).await?;
        let audio_seconds = audio
            .duration_seconds
            .with_context(|| format!("av-ingest did not return duration for {source_url}"))?
            as f64;
        let content_length = header_u64(&audio, "content-length");
        let content_type =
            header_string(&audio, CONTENT_TYPE.as_str()).or_else(|| audio.mime_type.clone());
        let resolver_name = audio.resolver.clone();
        let itag = audio.itag;
        let source_mime_type = audio.mime_type.clone();
        eprintln!(
            "[{}/{}] opened {}s source via {} itag {:?} bytes {:?}",
            index + 1,
            urls.len(),
            audio_seconds,
            resolver_name,
            itag,
            content_length
        );
        let transcript = asr.transcribe(audio).await?;
        let wall_seconds = started_at.elapsed().as_secs_f64();
        let rtfx = audio_seconds / wall_seconds.max(0.001);

        let transcript_words = transcript.split_whitespace().count();
        assert!(
            transcript_words >= 5,
            "short transcript for {source_url}: {transcript:?}"
        );

        completed += 1;
        aggregate_audio_seconds += audio_seconds;
        aggregate_wall_seconds += wall_seconds;

        let record = json!({
            "index": index,
            "source_url": source_url,
            "audio_seconds": audio_seconds,
            "wall_seconds": wall_seconds,
            "rtfx": rtfx,
            "content_length": content_length,
            "content_type": content_type,
            "source_mime_type": source_mime_type,
            "resolver": resolver_name,
            "itag": itag,
            "transport": "in-process",
            "transcript_chars": transcript.chars().count(),
            "transcript_words": transcript_words,
        });
        append_json_line(&report_path, &record)?;

        eprintln!(
            "[{}/{}] {:.1}s audio in {:.1}s wall = {:.2} RTFx :: {}",
            completed,
            urls.len(),
            audio_seconds,
            wall_seconds,
            rtfx,
            source_url
        );
    }

    assert_eq!(completed, urls.len());
    let aggregate_rtfx = aggregate_audio_seconds / aggregate_wall_seconds.max(0.001);
    eprintln!(
        "mastering benchmark complete: {:.1}s audio in {:.1}s wall = {:.2} aggregate RTFx",
        aggregate_audio_seconds, aggregate_wall_seconds, aggregate_rtfx
    );
    Ok(())
}

struct LocalAsrHarness {
    ingress: ListenIngress,
    _service: Arc<UploadResponseService>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl LocalAsrHarness {
    fn new(model_dir: PathBuf) -> Result<Self> {
        let ingress_config = asr_config(AppRole::Ingress, "media-research-ingress", None)?;
        let decoder_config = asr_config(AppRole::Decoder, "media-research-decoder", None)?;
        let worker_config =
            asr_config(AppRole::Worker, "media-research-worker-0", Some(model_dir))?;

        ingress_config.validate()?;

        let service = Arc::new(UploadResponseService::new(
            ingress_config.upload_response_config(),
        ));
        let watcher = ResponseWatcher::new(service.clone())
            .with_poll_interval_ms(ingress_config.upload_response_watch_poll_ms)
            .spawn();
        let decoder =
            Arc::new(DecoderState::new(decoder_config)).spawn_cache_worker(service.clone());
        let backend = Arc::new(AsrBackend::new(
            worker_config.model_dir()?,
            worker_config.resolved_model_provider()?,
            &worker_config.device_ids,
            worker_config.onnx_sessions,
            worker_config.cohere_max_new_tokens,
        )?);
        let worker =
            Arc::new(WorkerState::new(worker_config, backend)).spawn_cache_worker(service.clone());
        let ingress = ListenIngress::new(ingress_config, service.clone());

        Ok(Self {
            ingress,
            _service: service,
            handles: vec![watcher, decoder, worker],
        })
    }

    async fn transcribe(&self, audio: TranscribeAudioStream) -> Result<String> {
        let content_type = header_string(&audio, CONTENT_TYPE.as_str())
            .or_else(|| audio.mime_type.clone())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/listen?utterances=true&paragraphs=true&timestamps=true&language=en_US")
            .header(CONTENT_TYPE, content_type)
            .body(())
            .context("failed to build listen request")?;
        let response = self
            .ingress
            .handle_listen(req, body_stream_from_audio(audio))
            .await;
        transcript_from_response(response)
    }
}

impl Drop for LocalAsrHarness {
    fn drop(&mut self) {
        for handle in &self.handles {
            handle.abort();
        }
    }
}

fn body_stream_from_audio(audio: TranscribeAudioStream) -> BodyStream {
    Box::pin(audio.into_response().bytes_stream().map(|chunk| {
        chunk.map_err(|error| ServerError::Config(format!("source media stream failed: {error}")))
    }))
}

fn transcript_from_response(response: HandlerResponse) -> Result<String> {
    let body = response.body.unwrap_or_else(Bytes::new);
    if !response.status.is_success() {
        bail!(
            "ASR failed with HTTP {}: {}",
            response.status,
            String::from_utf8_lossy(&body)
        );
    }
    let value = serde_json::from_slice::<Value>(&body).context("failed to parse ASR JSON")?;
    value
        .pointer("/results/channels/0/alternatives/0/transcript")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("ASR response did not contain a transcript"))
}

fn header_string(audio: &TranscribeAudioStream, name: &str) -> Option<String> {
    audio
        .response()
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

fn header_u64(audio: &TranscribeAudioStream, name: &str) -> Option<u64> {
    header_string(audio, name)?.parse().ok()
}

fn mastering_urls() -> Result<Vec<String>> {
    let urls = env::var("MEDIA_RESEARCH_STACK_MASTERING_URLS")
        .ok()
        .map(|value| {
            value
                .split(|ch: char| ch == ',' || ch.is_whitespace())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            DEFAULT_MASTERING_URLS
                .iter()
                .map(|url| url.to_string())
                .collect()
        });

    if urls.len() < 2 {
        bail!("mastering benchmark requires at least two source URLs");
    }
    Ok(urls)
}

fn report_path() -> Result<PathBuf> {
    let path = env::var("MEDIA_RESEARCH_STACK_MASTERING_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/mastering-videos/report.jsonl"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::remove_file(&path);
    Ok(path)
}

fn append_json_line(path: &Path, value: &Value) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn env_u64(name: &str, default: u64) -> Result<u64> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("invalid {name}={value:?}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn ensure_mlx_metallib_for_current_exe() -> Result<()> {
    let exe = env::current_exe().context("failed to locate current test executable")?;
    let exe_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("test executable has no parent directory"))?;
    let target_dir = exe_dir
        .parent()
        .filter(|_| exe_dir.file_name().is_some_and(|name| name == "deps"))
        .unwrap_or(exe_dir);
    let source = target_dir.join("mlx.metallib");
    let dest = exe_dir.join("mlx.metallib");
    if dest.is_file() {
        return Ok(());
    }
    anyhow::ensure!(
        source.is_file(),
        "MLX metallib not found at {}; build the MLX target first",
        source.display()
    );
    fs::copy(&source, &dest).with_context(|| {
        format!(
            "failed to copy MLX metallib from {} to {}",
            source.display(),
            dest.display()
        )
    })?;
    Ok(())
}

fn asr_config(role: AppRole, worker_id: &str, model_dir: Option<PathBuf>) -> Result<AppConfig> {
    let upload_response_slot_size_kb = env_usize(
        "UPLOAD_RESPONSE_SLOT_SIZE_KB",
        DEFAULT_UPLOAD_RESPONSE_SLOT_SIZE_KB,
    )?;
    let upload_response_slot_bytes = upload_response_slot_size_kb.saturating_mul(1024).max(1);
    let upload_response_ring_bytes = env_usize(
        "UPLOAD_RESPONSE_RING_BYTES",
        DEFAULT_UPLOAD_RESPONSE_RING_BYTES,
    )?;
    let upload_response_slots_per_stream = env_optional_usize("UPLOAD_RESPONSE_SLOTS_PER_STREAM")?
        .unwrap_or_else(|| {
            slots_for_ring_bytes(upload_response_ring_bytes, upload_response_slot_bytes)
        })
        .max(3);

    Ok(AppConfig {
        role,
        rust_log: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        log_format: LogFormat::Compact,
        port: 0,
        enable_h3: false,
        tls_cert_path: None,
        tls_key_path: None,
        model_dir,
        model_provider: AsrModelProvider::Cohere,
        device_ids: Vec::new(),
        onnx_sessions: 1,
        cohere_max_new_tokens: env::var("MEDIA_RESEARCH_STACK_MASTERING_MAX_NEW_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(384),
        chunk_seconds: 30.0,
        overlap_seconds: 2.0,
        final_min_seconds: 0.5,
        utt_split_seconds: 0.8,
        upload_response_num_streams: 16,
        upload_response_slot_size_kb,
        upload_response_slots_per_stream,
        upload_response_timeout_ms: env_u64(
            "MEDIA_RESEARCH_STACK_MASTERING_REQUEST_TIMEOUT_MS",
            6 * 60 * 60 * 1_000,
        )
        .unwrap_or(6 * 60 * 60 * 1_000),
        upload_response_watch_poll_ms: 1,
        upload_response_worker_poll_ms: 2,
        upload_response_max_inflight: 2,
        upload_response_worker_id: worker_id.to_string(),
        upload_response_ingress_urls: Vec::new(),
        upload_response_discovery_dns: None,
        upload_response_discovery_interval_ms: 2_000,
        upload_response_insecure_tls: true,
        upload_response_worker_heartbeat_interval_ms: 1_000,
        upload_response_worker_ttl_ms: 5_000,
    })
}

fn slots_for_ring_bytes(ring_bytes: usize, slot_bytes: usize) -> usize {
    let slot_bytes = slot_bytes.max(1);
    ring_bytes.max(slot_bytes).saturating_add(slot_bytes - 1) / slot_bytes
}

fn env_optional_usize(name: &str) -> Result<Option<usize>> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid {name}={value:?}"))
        })
        .transpose()
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    env_optional_usize(name).map(|value| value.unwrap_or(default))
}
