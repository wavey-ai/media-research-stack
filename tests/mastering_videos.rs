use anyhow::{anyhow, bail, Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use asr_api::decoder::DecoderState;
use asr_api::ingress::ListenIngress;
use asr_api::worker::WorkerState;
use async_trait::async_trait;
use av_ingest_proxy::{TranscribeAudioResolver, TranscribeAudioStream};
use bytes::Bytes;
use futures_util::future;
use futures_util::StreamExt;
use http::{header::CONTENT_TYPE, Request, Response, StatusCode};
use serde_json::{json, Value};
use std::collections::HashSet;
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
const DEFAULT_UPLOAD_RESPONSE_RING_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_UPLOAD_RESPONSE_NUM_STREAMS: usize = 2;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcribes_audio_mastering_videos() -> Result<()> {
    if !env_flag_any(&[
        "MEDIA_RESEARCH_STACK_BENCH",
        "MEDIA_RESEARCH_STACK_MASTERING_BENCH",
    ]) {
        eprintln!("skipping long research benchmark; set MEDIA_RESEARCH_STACK_BENCH=1");
        return Ok(());
    }

    env::set_var("ASR_COHERE_BACKEND", "mlx");
    env::set_var("AV_INGEST_PROXY_RESOLVE_MODE", "transcribe");
    ensure_mlx_runtime_env()?;
    init_tracing();

    let model_dir = PathBuf::from(env::var("ASR_MODEL_DIR").context("ASR_MODEL_DIR is required")?);
    let urls = research_urls()?;
    let startup_grace =
        Duration::from_secs(env_u64("MEDIA_RESEARCH_STACK_STARTUP_GRACE_SECS", 20)?);

    let setup_started_at = Instant::now();
    eprintln!(
        "benchmark setup: creating av-ingest resolver for {} source(s)",
        urls.len()
    );
    let resolver = TranscribeAudioResolver::from_env()?;
    eprintln!(
        "benchmark setup: av-ingest resolver ready in {:.2}s",
        setup_started_at.elapsed().as_secs_f64()
    );
    eprintln!(
        "benchmark setup: loading ASR stack from {}",
        model_dir.display()
    );
    let asr_started_at = Instant::now();
    let asr = LocalAsrHarness::new(model_dir)?;
    eprintln!(
        "benchmark setup: ASR stack ready in {:.2}s",
        asr_started_at.elapsed().as_secs_f64()
    );
    eprintln!(
        "benchmark setup: startup grace {}s",
        startup_grace.as_secs()
    );
    tokio::time::sleep(startup_grace).await;

    let report_path = report_path()?;
    let progress_path = progress_path()?;
    let resume = env_flag("MEDIA_RESEARCH_STACK_RESUME");
    let completed_sources = if resume {
        completed_sources(&report_path)?
    } else {
        HashSet::new()
    };
    eprintln!("writing benchmark report to {}", report_path.display());
    eprintln!("writing streaming progress to {}", progress_path.display());

    let mut completed = 0usize;
    let mut skipped = 0usize;
    let mut aggregate_audio_seconds = 0.0f64;
    let mut aggregate_wall_seconds = 0.0f64;

    for (index, source_url) in urls.iter().enumerate() {
        if completed_sources.contains(source_url) {
            skipped += 1;
            eprintln!(
                "[{}/{}] already completed; skipping {}",
                index + 1,
                urls.len(),
                source_url
            );
            continue;
        }
        let started_at = Instant::now();
        eprintln!(
            "[{}/{}] opening source {}",
            index + 1,
            urls.len(),
            source_url
        );
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
        let transcribe_started_at = Instant::now();
        let transcript = asr
            .transcribe(audio, &progress_path, index, source_url)
            .await?;
        eprintln!(
            "[{}/{}] ASR stream returned in {:.2}s",
            index + 1,
            urls.len(),
            transcribe_started_at.elapsed().as_secs_f64()
        );
        let wall_seconds = started_at.elapsed().as_secs_f64();
        let rtfx = audio_seconds / wall_seconds.max(0.001);

        let transcript_words = transcript.split_whitespace().count();
        assert!(
            transcript_words >= 5,
            "short transcript for {source_url}: {transcript_words} words"
        );

        completed += 1;
        aggregate_audio_seconds += audio_seconds;
        aggregate_wall_seconds += wall_seconds;

        let record = json!({
            "status": "ok",
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

    assert_eq!(completed + skipped, urls.len());
    let aggregate_rtfx = aggregate_audio_seconds / aggregate_wall_seconds.max(0.001);
    eprintln!(
        "research benchmark complete: {} processed, {} resumed, {:.1}s audio in {:.1}s wall = {:.2} aggregate RTFx",
        completed, skipped, aggregate_audio_seconds, aggregate_wall_seconds, aggregate_rtfx
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
        let started_at = Instant::now();
        eprintln!("ASR harness: building configs");
        let ingress_config = asr_config(AppRole::Ingress, "media-research-ingress", None)?;
        let decoder_config = asr_config(AppRole::Decoder, "media-research-decoder", None)?;
        let worker_config =
            asr_config(AppRole::Worker, "media-research-worker-0", Some(model_dir))?;

        ingress_config.validate()?;
        eprintln!(
            "ASR harness: configs ready in {:.2}s",
            started_at.elapsed().as_secs_f64()
        );

        let service_started_at = Instant::now();
        let service = Arc::new(UploadResponseService::new(
            ingress_config.upload_response_config(),
        ));
        let watcher = ResponseWatcher::new(service.clone())
            .with_poll_interval_ms(ingress_config.upload_response_watch_poll_ms)
            .spawn();
        let decoder =
            Arc::new(DecoderState::new(decoder_config)).spawn_cache_worker(service.clone());
        eprintln!(
            "ASR harness: upload-response and decoder ready in {:.2}s",
            service_started_at.elapsed().as_secs_f64()
        );

        let backend_started_at = Instant::now();
        eprintln!("ASR harness: constructing ASR backend");
        let backend = Arc::new(AsrBackend::new(
            worker_config.model_dir()?,
            worker_config.resolved_model_provider()?,
            &worker_config.device_ids,
            worker_config.onnx_sessions,
            worker_config.cohere_max_new_tokens,
        )?);
        eprintln!(
            "ASR harness: ASR backend ready in {:.2}s",
            backend_started_at.elapsed().as_secs_f64()
        );

        let worker_started_at = Instant::now();
        let worker =
            Arc::new(WorkerState::new(worker_config, backend)).spawn_cache_worker(service.clone());
        let ingress = ListenIngress::new(ingress_config, service.clone());
        eprintln!(
            "ASR harness: worker/ingress ready in {:.2}s",
            worker_started_at.elapsed().as_secs_f64()
        );

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
        let listen_uri = if env_flag_any(&[
            "MEDIA_RESEARCH_STACK_INTERIM",
            "MEDIA_RESEARCH_STACK_MASTERING_INTERIM",
        ]) {
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
        eprintln!("[{}] ASR stream starting", source_index + 1);
        self.ingress
            .handle_listen_stream(
                req,
                body_stream_from_audio(audio, source_index, source_url),
                Box::new(writer),
            )
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

fn body_stream_from_audio(
    audio: TranscribeAudioStream,
    source_index: usize,
    source_url: &str,
) -> BodyStream {
    let content_length = header_u64(&audio, "content-length").unwrap_or(0);
    let source_url = source_url.to_string();
    Box::pin(audio.into_response().bytes_stream().scan(
        (0usize, 1024 * 1024usize),
        move |(total_bytes, next_log_at), chunk| {
            let result = match chunk {
                Ok(bytes) => {
                    *total_bytes += bytes.len();
                    while *total_bytes >= *next_log_at {
                        eprintln!(
                            "[{}] source body read {} / {} bytes from {}",
                            source_index + 1,
                            total_bytes,
                            content_length,
                            source_url
                        );
                        *next_log_at += 1024 * 1024;
                    }
                    Ok(bytes)
                }
                Err(error) => {
                    eprintln!(
                        "[{}] source body error after {} / {} bytes from {}: {}",
                        source_index + 1,
                        total_bytes,
                        content_length,
                        source_url,
                        error
                    );
                    Err(ServerError::Config(format!(
                        "source media stream failed after {total_bytes}/{content_length} bytes: {error}"
                    )))
                }
            };
            future::ready(Some(result))
        },
    ))
}

struct ProgressStreamWriter {
    file: File,
    pending: Vec<u8>,
    state: Arc<Mutex<ProgressState>>,
    source_index: usize,
    source_url: String,
    status: Option<StatusCode>,
    store_transcripts: bool,
    log_transcript_previews: bool,
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
            store_transcripts: env_flag("MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS"),
            log_transcript_previews: env_flag_any(&[
                "MEDIA_RESEARCH_STACK_LOG_TRANSCRIPT_PREVIEWS",
                "MEDIA_RESEARCH_STACK_LOG_TRANSCRIPTS",
            ]),
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
                "raw_bytes": line.len(),
            })
        });
        let persisted_event = progress_event(&event, self.store_transcripts);
        let wrapped = json!({
            "source_index": self.source_index,
            "source_url": self.source_url,
            "event": persisted_event,
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
                if self.log_transcript_previews {
                    eprintln!(
                        "[{}] ASR {label} {:.1}-{:.1}s: {}",
                        self.source_index + 1,
                        start,
                        duration,
                        transcript_preview(transcript)
                    );
                } else {
                    eprintln!(
                        "[{}] ASR {label} {:.1}-{:.1}s: {} chars, {} words",
                        self.source_index + 1,
                        start,
                        duration,
                        transcript.chars().count(),
                        transcript.split_whitespace().count()
                    );
                }
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

fn progress_event(event: &Value, store_transcripts: bool) -> Value {
    if store_transcripts {
        return event.clone();
    }

    let mut redacted = event.clone();
    if let Some(alternative) = redacted
        .pointer_mut("/channel/alternatives/0")
        .and_then(Value::as_object_mut)
    {
        let transcript = alternative
            .remove("transcript")
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_default();
        let word_count = alternative
            .remove("words")
            .and_then(|value| value.as_array().map(Vec::len))
            .unwrap_or_else(|| transcript.split_whitespace().count());
        alternative.remove("paragraphs");
        alternative.insert(
            "transcript_chars".to_string(),
            Value::from(transcript.chars().count()),
        );
        alternative.insert("transcript_words".to_string(), Value::from(word_count));
    }
    if let Some(object) = redacted.as_object_mut() {
        object.remove("utterances");
    }
    redacted
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

fn research_urls() -> Result<Vec<String>> {
    let urls = if let Some(path) = first_env(&["MEDIA_RESEARCH_STACK_URLS_FILE"]) {
        urls_from_file(Path::new(&path))?
    } else if let Some(value) = first_env(&[
        "MEDIA_RESEARCH_STACK_URLS",
        "MEDIA_RESEARCH_STACK_MASTERING_URLS",
    ]) {
        split_urls(&value)
    } else {
        DEFAULT_MASTERING_URLS
            .iter()
            .map(|url| url.to_string())
            .collect()
    };

    anyhow::ensure!(
        !urls.is_empty(),
        "research run requires at least one source URL"
    );
    Ok(urls)
}

fn urls_from_file(path: &Path) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read source URL file {}", path.display()))?;
    let trimmed = contents.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let value: Value = serde_json::from_str(trimmed)
            .with_context(|| format!("failed to parse source URL file {}", path.display()))?;
        let entries = value
            .get("videos")
            .and_then(Value::as_array)
            .or_else(|| value.as_array())
            .ok_or_else(|| anyhow!("JSON URL file must be an array or contain a videos array"))?;
        return entries
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .or_else(|| entry.get("url").and_then(Value::as_str))
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| anyhow!("JSON URL entry must be a string or contain a url"))
            })
            .collect();
    }

    Ok(contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .flat_map(split_urls)
        .collect())
}

fn split_urls(value: &str) -> Vec<String> {
    value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn report_path() -> Result<PathBuf> {
    let path = first_env(&[
        "MEDIA_RESEARCH_STACK_REPORT",
        "MEDIA_RESEARCH_STACK_MASTERING_REPORT",
    ])
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("target/research/report.jsonl"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if !env_flag("MEDIA_RESEARCH_STACK_RESUME") {
        let _ = fs::remove_file(&path);
    }
    Ok(path)
}

fn progress_path() -> Result<PathBuf> {
    let path = first_env(&[
        "MEDIA_RESEARCH_STACK_PROGRESS",
        "MEDIA_RESEARCH_STACK_MASTERING_PROGRESS",
    ])
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("target/research/progress.ndjson"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if !env_flag("MEDIA_RESEARCH_STACK_RESUME") {
        let _ = fs::remove_file(&path);
    }
    Ok(path)
}

fn completed_sources(path: &Path) -> Result<HashSet<String>> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(error) => return Err(error.into()),
    };
    let mut completed = HashSet::new();
    for (index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid report JSON on line {}", index + 1))?;
        if value
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| status != "ok")
        {
            continue;
        }
        if let Some(source_url) = value.get("source_url").and_then(Value::as_str) {
            completed.insert(source_url.to_string());
        }
    }
    Ok(completed)
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

fn env_flag_any(names: &[&str]) -> bool {
    names.iter().any(|name| env_flag(name))
}

fn first_env(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()))
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

fn ensure_mlx_runtime_env() -> Result<()> {
    let runtime = if let Some(path) = first_env(&["ASR_MLX_TRANSCRIBE_BIN"]) {
        PathBuf::from(path)
    } else {
        env::current_dir()
            .context("failed to resolve current directory")?
            .join("../asr-api/apple/.build/release/asr-mlx-transcribe")
    };
    anyhow::ensure!(
        runtime.is_file(),
        "Cohere MLX runtime was not found. Build it with `swift build -c release --package-path ../asr-api/apple`, then set ASR_MLX_TRANSCRIBE_BIN."
    );
    let runtime_dir = runtime
        .parent()
        .ok_or_else(|| anyhow!("Cohere MLX runtime has no parent directory"))?;
    let destination = runtime_dir.join("mlx.metallib");
    if !destination.is_file() {
        let build_dir = runtime_dir
            .parent()
            .ok_or_else(|| anyhow!("Cohere MLX runtime is not inside a Swift build directory"))?;
        let source = [
            build_dir.join("mlx-metal/mlx.metallib"),
            build_dir.join("arm64-apple-macosx/debug/mlx.metallib"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| anyhow!("MLX Metal library was not produced by the Swift build"))?;
        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to copy MLX metallib from {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }
    env::set_var("ASR_MLX_TRANSCRIBE_BIN", runtime);
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
        cohere_max_new_tokens: first_env(&[
            "ASR_COHERE_MAX_NEW_TOKENS",
            "MEDIA_RESEARCH_STACK_MASTERING_MAX_NEW_TOKENS",
        ])
        .and_then(|value| value.parse().ok())
        .unwrap_or(128),
        chunk_seconds: 30.0,
        overlap_seconds: 2.0,
        final_min_seconds: 0.5,
        utt_split_seconds: 0.8,
        upload_response_num_streams: env_usize(
            "UPLOAD_RESPONSE_NUM_STREAMS",
            DEFAULT_UPLOAD_RESPONSE_NUM_STREAMS,
        )?,
        upload_response_slot_size_kb,
        upload_response_slots_per_stream,
        upload_response_timeout_ms: first_env(&[
            "UPLOAD_RESPONSE_TIMEOUT_MS",
            "MEDIA_RESEARCH_STACK_MASTERING_REQUEST_TIMEOUT_MS",
        ])
        .map(|value| {
            value
                .parse::<u64>()
                .context("invalid upload response timeout")
        })
        .transpose()?
        .unwrap_or(6 * 60 * 60 * 1_000),
        upload_response_watch_poll_ms: 1,
        upload_response_worker_poll_ms: 2,
        upload_response_max_inflight: env_usize("UPLOAD_RESPONSE_MAX_INFLIGHT", 1)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_url_lists_on_commas_and_whitespace() {
        assert_eq!(
            split_urls("https://example.com/a, https://example.com/b\nhttps://example.com/c"),
            vec![
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c"
            ]
        );
    }

    #[test]
    fn redacts_transcript_text_from_progress_by_default() {
        let event = json!({
            "type": "Results",
            "channel": {
                "alternatives": [{
                    "transcript": "one two three",
                    "words": [{"word": "one"}, {"word": "two"}, {"word": "three"}],
                    "paragraphs": {"transcript": "one two three"}
                }]
            },
            "utterances": [{"transcript": "one two three"}]
        });

        let redacted = progress_event(&event, false);
        let alternative = redacted
            .pointer("/channel/alternatives/0")
            .and_then(Value::as_object)
            .unwrap();
        assert!(!alternative.contains_key("transcript"));
        assert!(!alternative.contains_key("words"));
        assert!(!alternative.contains_key("paragraphs"));
        assert_eq!(alternative.get("transcript_chars"), Some(&Value::from(13)));
        assert_eq!(alternative.get("transcript_words"), Some(&Value::from(3)));
        assert!(redacted.get("utterances").is_none());
    }

    #[test]
    fn preserves_transcript_text_when_explicitly_enabled() {
        let event = json!({
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "owned recording"}]}
        });
        assert_eq!(progress_event(&event, true), event);
    }

    #[test]
    fn ring_size_rounds_up_to_whole_slots() {
        assert_eq!(slots_for_ring_bytes(65_537, 32_768), 3);
    }
}
