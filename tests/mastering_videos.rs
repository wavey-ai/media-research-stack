use anyhow::{anyhow, bail, Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use asr_api::decoder::DecoderState;
use asr_api::ingress::ListenIngress;
use asr_api::worker::WorkerState;
use async_trait::async_trait;
use av_ingest_proxy::{TranscribeAudioResolver, TranscribeAudioStream};
use bytes::Bytes;
use futures_util::StreamExt;
use http::{header::CONTENT_TYPE, Request, Response, StatusCode};
use serde_json::{json, Value};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use upload_response::{ResponseWatcher, UploadResponseService};
use web_service::{BodyStream, ServerError, StreamWriter};

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
    init_tracing();
    ensure_mlx_metallib_for_current_exe()?;

    let model_dir = PathBuf::from(env::var("ASR_MODEL_DIR").context("ASR_MODEL_DIR is required")?);
    let urls = mastering_urls()?;
    let startup_grace =
        Duration::from_secs(env_u64("MEDIA_RESEARCH_STACK_STARTUP_GRACE_SECS", 20)?);

    let resolver = TranscribeAudioResolver::from_env()?;
    let asr = LocalAsrHarness::new(model_dir)?;
    tokio::time::sleep(startup_grace).await;

    let report_path = report_path()?;
    let progress_path = progress_path()?;
    eprintln!("writing benchmark report to {}", report_path.display());
    eprintln!("writing streaming progress to {}", progress_path.display());

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
        let transcript = asr
            .transcribe(audio, &progress_path, index, source_url)
            .await?;
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

    async fn transcribe(
        &self,
        audio: TranscribeAudioStream,
        progress_path: &Path,
        source_index: usize,
        source_url: &str,
    ) -> Result<String> {
        let content_type = header_string(&audio, CONTENT_TYPE.as_str())
            .or_else(|| audio.mime_type.clone())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let listen_uri = if env_flag("MEDIA_RESEARCH_STACK_MASTERING_INTERIM") {
            "/v1/listen?utterances=true&paragraphs=true&timestamps=true&interim_results=true&language=en_US"
        } else {
            "/v1/listen?utterances=true&paragraphs=true&timestamps=true&language=en_US"
        };
        let req = Request::builder()
            .method("POST")
            .uri(listen_uri)
            .header(CONTENT_TYPE, content_type)
            .body(())
            .context("failed to build listen request")?;
        let writer =
            ProgressStreamWriter::new(progress_path, source_index, source_url.to_string())?;
        let collector = writer.collector();
        self.ingress
            .handle_listen_stream(req, body_stream_from_audio(audio), Box::new(writer))
            .await
            .map_err(|error| anyhow!("streaming listen failed: {error}"))?;
        collector.transcript()
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

struct ProgressStreamWriter {
    file: File,
    pending: Vec<u8>,
    state: Arc<Mutex<ProgressState>>,
    source_index: usize,
    source_url: String,
    status: Option<StatusCode>,
}

#[derive(Default)]
struct ProgressState {
    status: Option<StatusCode>,
    events: usize,
    final_segments: Vec<String>,
    last_interim: Option<String>,
    error_body: Vec<u8>,
}

#[derive(Clone)]
struct ProgressCollector {
    state: Arc<Mutex<ProgressState>>,
}

impl ProgressStreamWriter {
    fn new(path: &Path, source_index: usize, source_url: String) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open progress file {}", path.display()))?;
        Ok(Self {
            file,
            pending: Vec::new(),
            state: Arc::new(Mutex::new(ProgressState::default())),
            source_index,
            source_url,
            status: None,
        })
    }

    fn collector(&self) -> ProgressCollector {
        ProgressCollector {
            state: self.state.clone(),
        }
    }

    fn write_progress_value(&mut self, value: &Value) -> Result<(), ServerError> {
        let line =
            serde_json::to_string(value).map_err(|error| ServerError::Config(error.to_string()))?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        Ok(())
    }

    fn process_line(&mut self, line: &[u8]) -> Result<(), ServerError> {
        let line = trim_ascii(line);
        if line.is_empty() {
            return Ok(());
        }

        let event = serde_json::from_slice::<Value>(line).unwrap_or_else(|error| {
            json!({
                "type": "ParseError",
                "error": error.to_string(),
                "raw": String::from_utf8_lossy(line),
            })
        });
        let wrapped = json!({
            "source_index": self.source_index,
            "source_url": self.source_url,
            "event": event.clone(),
        });
        self.write_progress_value(&wrapped)?;

        if self.status.is_some_and(|status| !status.is_success()) {
            let mut state = self.lock_state()?;
            if !state.error_body.is_empty() {
                state.error_body.push(b'\n');
            }
            state.error_body.extend_from_slice(line);
            return Ok(());
        }

        self.record_event(&event)
    }

    fn record_event(&self, event: &Value) -> Result<(), ServerError> {
        let mut state = self.lock_state()?;
        state.events += 1;

        match event.get("type").and_then(Value::as_str) {
            Some("Metadata") => {
                if let Some(request_id) = event.get("request_id").and_then(Value::as_str) {
                    eprintln!(
                        "[{}] ASR metadata request_id={request_id}",
                        self.source_index + 1
                    );
                }
            }
            Some("SpeechStarted") => {
                if let Some(timestamp) = event.get("timestamp").and_then(Value::as_f64) {
                    eprintln!(
                        "[{}] speech started at {:.1}s",
                        self.source_index + 1,
                        timestamp
                    );
                }
            }
            Some("Results") => {
                let transcript = event
                    .pointer("/channel/alternatives/0/transcript")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                if transcript.is_empty() {
                    return Ok(());
                }

                let is_final = event
                    .get("is_final")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if is_final {
                    state.final_segments.push(transcript.to_string());
                } else {
                    state.last_interim = Some(transcript.to_string());
                }

                let label = if is_final { "final" } else { "interim" };
                let start = event.get("start").and_then(Value::as_f64).unwrap_or(0.0);
                let duration = event
                    .get("duration")
                    .and_then(Value::as_f64)
                    .unwrap_or(start);
                eprintln!(
                    "[{}] ASR {label} {:.1}-{:.1}s: {}",
                    self.source_index + 1,
                    start,
                    duration,
                    transcript_preview(transcript)
                );
            }
            _ => {}
        }

        Ok(())
    }

    fn drain_complete_lines(&mut self) -> Result<(), ServerError> {
        while let Some(line_end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let mut line = self.pending.drain(..=line_end).collect::<Vec<_>>();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(&line)?;
        }
        Ok(())
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, ProgressState>, ServerError> {
        self.state
            .lock()
            .map_err(|_| ServerError::Config("progress state mutex poisoned".into()))
    }
}

#[async_trait]
impl StreamWriter for ProgressStreamWriter {
    async fn send_response(&mut self, response: Response<()>) -> Result<(), ServerError> {
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        self.status = Some(status);
        self.lock_state()?.status = Some(status);
        self.write_progress_value(&json!({
            "source_index": self.source_index,
            "source_url": self.source_url,
            "event": {
                "type": "ResponseHead",
                "status": status.as_u16(),
                "content_type": content_type,
            },
        }))?;
        Ok(())
    }

    async fn send_data(&mut self, data: Bytes) -> Result<(), ServerError> {
        self.pending.extend_from_slice(&data);
        self.drain_complete_lines()
    }

    async fn finish(&mut self) -> Result<(), ServerError> {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            self.process_line(&line)?;
        }
        self.file.flush()?;
        Ok(())
    }
}

impl ProgressCollector {
    fn transcript(&self) -> Result<String> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow!("progress state mutex poisoned"))?;

        match state.status {
            Some(status) if !status.is_success() => {
                let body = String::from_utf8_lossy(&state.error_body);
                bail!("ASR failed with HTTP {status}: {body}");
            }
            Some(_) => {}
            None => bail!("ASR stream did not send a response head"),
        }

        let transcript = state
            .final_segments
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        if !transcript.trim().is_empty() {
            return Ok(transcript);
        }
        if let Some(interim) = state
            .last_interim
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            return Ok(interim.to_string());
        }

        bail!(
            "ASR stream did not contain a transcript; received {} events",
            state.events
        );
    }
}

fn trim_ascii(line: &[u8]) -> &[u8] {
    let start = line
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(line.len());
    let end = line
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(start);
    &line[start..end]
}

fn transcript_preview(transcript: &str) -> String {
    const LIMIT: usize = 220;
    let mut preview = String::new();
    for (index, ch) in transcript.chars().enumerate() {
        if index >= LIMIT {
            preview.push_str("...");
            return preview;
        }
        preview.push(ch);
    }
    preview
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

fn progress_path() -> Result<PathBuf> {
    let path = env::var("MEDIA_RESEARCH_STACK_MASTERING_PROGRESS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/mastering-videos/progress.ndjson"));
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

fn init_tracing() {
    let filter = env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .compact()
        .try_init();
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
