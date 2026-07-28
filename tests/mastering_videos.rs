use anyhow::{anyhow, bail, Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use asr_api::decoder::DecoderState;
use asr_api::ingress::ListenIngress;
use asr_api::worker::WorkerState;
use async_trait::async_trait;
use av_ingest_proxy::{TranscribeAudioResolver, TranscribeAudioStream};
use bytes::Bytes;
use futures_util::{future, stream, StreamExt};
use http::{header::CONTENT_TYPE, Request, Response, StatusCode};
use media_research_stack::research_cache::{
    ensure_cached_source, open_cached_source, source_file_stem, source_urls_from_file,
    SourceMetadata,
};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
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

struct OpenedSource {
    metadata: SourceMetadata,
    body: BodyStream,
}

#[derive(Clone, Debug)]
struct BenchmarkSettings {
    backend: String,
    device_ids: Vec<usize>,
    onnx_sessions: usize,
    asr_concurrency: usize,
    worker_instances: usize,
    minimum_transcript_words: usize,
}

struct SourceSuccess {
    metadata: SourceMetadata,
    audio_seconds: f64,
    transcript: String,
    transcript_words: usize,
    transcript_path: Option<PathBuf>,
    transcribe_seconds: f64,
}

struct SourceAttempt {
    index: usize,
    source_url: String,
    wall_seconds: f64,
    result: Result<SourceSuccess>,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcribes_audio_mastering_videos() -> Result<()> {
    if !env_flag_any(&[
        "MEDIA_RESEARCH_STACK_BENCH",
        "MEDIA_RESEARCH_STACK_MASTERING_BENCH",
    ]) {
        eprintln!("skipping long research benchmark; set MEDIA_RESEARCH_STACK_BENCH=1");
        return Ok(());
    }

    set_default_asr_backend();
    env::set_var("AV_INGEST_PROXY_RESOLVE_MODE", "transcribe");
    let settings = benchmark_settings()?;
    if settings.backend == "mlx" {
        ensure_mlx_runtime_env()?;
    }
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
    eprintln!(
        "benchmark setup: backend={}, device_ids={:?}, ONNX sessions={}, request concurrency={}, worker instances={}",
        settings.backend,
        settings.device_ids,
        settings.onnx_sessions,
        settings.asr_concurrency,
        settings.worker_instances,
    );
    let asr_started_at = Instant::now();
    let asr = Arc::new(LocalAsrHarness::new(model_dir, &settings)?);
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
    let transcripts_dir = transcripts_dir()?;
    let media_dir = media_dir()?;
    let require_cached = env_flag("MEDIA_RESEARCH_STACK_REQUIRE_CACHE");
    let cache_wait = Duration::from_secs(env_u64(
        "MEDIA_RESEARCH_STACK_CACHE_WAIT_SECS",
        if require_cached { 6 * 60 * 60 } else { 0 },
    )?);
    anyhow::ensure!(
        !require_cached || media_dir.is_some(),
        "MEDIA_RESEARCH_STACK_REQUIRE_CACHE requires MEDIA_RESEARCH_STACK_MEDIA_DIR"
    );
    let resume = env_flag("MEDIA_RESEARCH_STACK_RESUME");
    let continue_on_error = env_flag("MEDIA_RESEARCH_STACK_CONTINUE_ON_ERROR");
    let completed_sources = if resume {
        completed_sources(&report_path)?
    } else {
        HashSet::new()
    };
    eprintln!("writing benchmark report to {}", report_path.display());
    eprintln!("writing streaming progress to {}", progress_path.display());
    if let Some(path) = &transcripts_dir {
        eprintln!("writing completed transcripts to {}", path.display());
    }
    if let Some(path) = &media_dir {
        if require_cached {
            eprintln!(
                "reading cache-only audio from {} with a {}s wait",
                path.display(),
                cache_wait.as_secs()
            );
        } else {
            eprintln!("caching compressed source audio in {}", path.display());
        }
    }

    let mut completed = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut aggregate_audio_seconds = 0.0f64;
    let mut aggregate_wall_seconds = 0.0f64;
    let mut aggregate_asr_seconds = 0.0f64;
    let mut aggregate_transcript_words = 0usize;
    let mut pending_sources = Vec::new();
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
        pending_sources.push((index, source_url.clone()));
    }

    let run_started_at = Instant::now();
    let total_sources = urls.len();
    let attempts = stream::iter(pending_sources.into_iter().map(|(index, source_url)| {
        let asr = Arc::clone(&asr);
        let resolver = &resolver;
        let media_dir = media_dir.clone();
        let progress_path = progress_path.clone();
        let transcripts_dir = transcripts_dir.clone();
        async move {
            transcribe_source(
                asr.as_ref(),
                resolver,
                media_dir.as_deref(),
                &progress_path,
                transcripts_dir.as_deref(),
                index,
                source_url,
                total_sources,
                require_cached,
                cache_wait,
                settings.minimum_transcript_words,
            )
            .await
        }
    }))
    .buffer_unordered(settings.asr_concurrency);
    tokio::pin!(attempts);

    while let Some(attempt) = attempts.next().await {
        let SourceAttempt {
            index,
            source_url,
            wall_seconds,
            result,
        } = attempt;
        let SourceSuccess {
            metadata,
            audio_seconds,
            transcript,
            transcript_words,
            transcript_path,
            transcribe_seconds,
            ..
        } = match result {
            Ok(result) => result,
            Err(error) if continue_on_error => {
                failed += 1;
                let error_message = format!("{error:#}");
                append_json_line(
                    &report_path,
                    &json!({
                        "status": "error",
                        "index": index,
                        "source_url": source_url,
                        "wall_seconds": wall_seconds,
                        "asr_backend": settings.backend,
                        "device_ids": settings.device_ids,
                        "onnx_sessions": settings.onnx_sessions,
                        "asr_concurrency": settings.asr_concurrency,
                        "worker_instances": settings.worker_instances,
                        "error": error_message,
                    }),
                )?;
                if is_systemic_youtube_auth_error(&error_message) {
                    bail!(
                            "YouTube rejected the audio-download session; stopping the sweep instead of retrying every source. Set AV_INGEST_PROXY_YTDLP_COOKIES to a readable Netscape cookies file or AV_INGEST_PROXY_YTDLP_COOKIES_FROM_BROWSER to a logged-in browser, verify one URL with yt-dlp, and resume the existing report. Original error: {error_message}"
                        );
                }
                eprintln!(
                    "[{}/{}] source failed after {:.1}s; continuing: {:#}",
                    index + 1,
                    urls.len(),
                    wall_seconds,
                    error
                );
                continue;
            }
            Err(error) => return Err(error),
        };
        let rtfx = audio_seconds / wall_seconds.max(0.001);
        let asr_rtfx = audio_seconds / transcribe_seconds.max(0.001);
        let asr_input_mib_per_second = metadata
            .content_length
            .map(|bytes| bytes as f64 / (1024.0 * 1024.0) / transcribe_seconds.max(0.001));
        let asr_transcript_words_per_second =
            transcript_words as f64 / transcribe_seconds.max(0.001);

        completed += 1;
        aggregate_audio_seconds += audio_seconds;
        aggregate_wall_seconds += wall_seconds;
        aggregate_asr_seconds += transcribe_seconds;
        aggregate_transcript_words += transcript_words;
        let run_asr_rtfx = aggregate_audio_seconds / aggregate_asr_seconds.max(0.001);
        let run_pipeline_rtfx = aggregate_audio_seconds / aggregate_wall_seconds.max(0.001);
        let run_transcript_words_per_second =
            aggregate_transcript_words as f64 / aggregate_asr_seconds.max(0.001);
        let run_elapsed_seconds = run_started_at.elapsed().as_secs_f64();
        let run_effective_rtfx = aggregate_audio_seconds / run_elapsed_seconds.max(0.001);

        let record = json!({
            "status": "ok",
            "metrics_schema": 2,
            "index": index,
            "source_url": source_url,
            "asr_backend": settings.backend,
            "device_ids": settings.device_ids,
            "onnx_sessions": settings.onnx_sessions,
            "asr_concurrency": settings.asr_concurrency,
            "worker_instances": settings.worker_instances,
            "audio_seconds": audio_seconds,
            "wall_seconds": wall_seconds,
            "rtfx": rtfx,
            "asr_wall_seconds": transcribe_seconds,
            "asr_rtfx": asr_rtfx,
            "asr_input_mib_per_second": asr_input_mib_per_second,
            "asr_transcript_words_per_second": asr_transcript_words_per_second,
            "run_processed_sources": completed,
            "run_audio_seconds": aggregate_audio_seconds,
            "run_asr_wall_seconds": aggregate_asr_seconds,
            "run_asr_rtfx": run_asr_rtfx,
            "run_pipeline_wall_seconds": aggregate_wall_seconds,
            "run_pipeline_rtfx": run_pipeline_rtfx,
            "run_elapsed_seconds": run_elapsed_seconds,
            "run_effective_rtfx": run_effective_rtfx,
            "run_transcript_words_per_second": run_transcript_words_per_second,
            "content_length": metadata.content_length,
            "content_type": metadata.content_type,
            "source_mime_type": metadata.source_mime_type,
            "resolver": metadata.resolver,
            "itag": metadata.itag,
            "cached_audio": metadata.cached,
            "transport": "in-process",
            "transcript_chars": transcript.chars().count(),
            "transcript_words": transcript_words,
            "transcript_path": transcript_path,
        });
        append_json_line(&report_path, &record)?;

        eprintln!(
            "[{}/{}] {:.1}s audio: ASR {:.1}s = {:.2}x; pipeline {:.1}s = {:.2}x; {:.1} transcript words/s :: {}",
            completed,
            urls.len(),
            audio_seconds,
            transcribe_seconds,
            asr_rtfx,
            wall_seconds,
            rtfx,
            asr_transcript_words_per_second,
            source_url
        );
        eprintln!(
            "run throughput: {} source(s), {:.1}s audio, ASR service {:.2}x, effective {:.2}x",
            completed, aggregate_audio_seconds, run_asr_rtfx, run_effective_rtfx
        );
    }

    assert_eq!(completed + skipped + failed, urls.len());
    let aggregate_asr_rtfx = aggregate_audio_seconds / aggregate_asr_seconds.max(0.001);
    let aggregate_words_per_second =
        aggregate_transcript_words as f64 / aggregate_asr_seconds.max(0.001);
    let run_elapsed_seconds = run_started_at.elapsed().as_secs_f64();
    let run_effective_rtfx = aggregate_audio_seconds / run_elapsed_seconds.max(0.001);
    eprintln!(
        "research benchmark complete: {} processed, {} resumed, {} failed, {:.1}s audio; ASR service {:.1}s = {:.2}x; elapsed {:.1}s = {:.2}x effective; {:.1} transcript words/s",
        completed,
        skipped,
        failed,
        aggregate_audio_seconds,
        aggregate_asr_seconds,
        aggregate_asr_rtfx,
        run_elapsed_seconds,
        run_effective_rtfx,
        aggregate_words_per_second,
    );
    asr.shutdown().await;
    anyhow::ensure!(
        failed == 0,
        "research sweep completed with {failed} failed source(s)"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn transcribe_source(
    asr: &LocalAsrHarness,
    resolver: &TranscribeAudioResolver,
    media_dir: Option<&Path>,
    progress_path: &Path,
    transcripts_dir: Option<&Path>,
    index: usize,
    source_url: String,
    total_sources: usize,
    require_cached: bool,
    cache_wait: Duration,
    minimum_transcript_words: usize,
) -> SourceAttempt {
    let started_at = Instant::now();
    let result = async {
        eprintln!(
            "[{}/{}] opening source {}",
            index + 1,
            total_sources,
            source_url
        );
        let audio = open_research_audio(
            resolver,
            media_dir,
            index,
            &source_url,
            require_cached,
            cache_wait,
        )
        .await?;
        let metadata = audio.metadata.clone();
        let audio_seconds = metadata.duration_seconds as f64;
        eprintln!(
            "[{}/{}] opened {}s source via {} itag {:?} bytes {:?} cached={}",
            index + 1,
            total_sources,
            audio_seconds,
            metadata.resolver,
            metadata.itag,
            metadata.content_length,
            metadata.cached,
        );
        let transcribe_started_at = Instant::now();
        let transcript = asr
            .transcribe(
                audio,
                progress_path,
                index,
                &source_url,
                minimum_transcript_words == 0,
            )
            .await?;
        let transcribe_seconds = transcribe_started_at.elapsed().as_secs_f64();
        eprintln!(
            "[{}/{}] ASR stream returned in {:.2}s",
            index + 1,
            total_sources,
            transcribe_seconds
        );
        let transcript_words = transcript.split_whitespace().count();
        anyhow::ensure!(
            transcript_words >= minimum_transcript_words,
            "short transcript for {source_url}: {transcript_words} words; minimum is {minimum_transcript_words}"
        );
        let transcript_path = transcripts_dir
            .map(|directory| write_transcript(directory, index, &source_url, &transcript))
            .transpose()?;
        Ok(SourceSuccess {
            metadata,
            audio_seconds,
            transcript,
            transcript_words,
            transcript_path,
            transcribe_seconds,
        })
    }
    .await;
    let wall_seconds = started_at.elapsed().as_secs_f64();
    SourceAttempt {
        index,
        source_url,
        wall_seconds,
        result,
    }
}

struct LocalAsrHarness {
    ingress: ListenIngress,
    _service: Arc<UploadResponseService>,
    handles: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl LocalAsrHarness {
    fn new(model_dir: PathBuf, settings: &BenchmarkSettings) -> Result<Self> {
        let started_at = Instant::now();
        eprintln!("ASR harness: building configs");
        let ingress_config =
            asr_config(AppRole::Ingress, "media-research-ingress", None, settings)?;
        let decoder_config =
            asr_config(AppRole::Decoder, "media-research-decoder", None, settings)?;
        let worker_configs = (0..settings.worker_instances)
            .map(|index| {
                asr_config(
                    AppRole::Worker,
                    &format!("media-research-worker-{index}"),
                    Some(model_dir.clone()),
                    settings,
                )
            })
            .collect::<Result<Vec<_>>>()?;

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
        let worker_config = worker_configs
            .first()
            .context("ASR harness requires one worker configuration")?;
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
        let workers = worker_configs
            .into_iter()
            .map(|config| {
                Arc::new(WorkerState::new(config, Arc::clone(&backend)))
                    .spawn_cache_worker(service.clone())
            })
            .collect::<Vec<_>>();
        let ingress = ListenIngress::new(ingress_config, service.clone());
        eprintln!(
            "ASR harness: worker/ingress ready in {:.2}s",
            worker_started_at.elapsed().as_secs_f64()
        );

        let mut handles = vec![watcher, decoder];
        handles.extend(workers);
        Ok(Self {
            ingress,
            _service: service,
            handles: Mutex::new(handles),
        })
    }

    async fn transcribe(
        &self,
        audio: OpenedSource,
        progress_path: &Path,
        source_index: usize,
        source_url: &str,
        allow_empty_transcript: bool,
    ) -> Result<String> {
        let content_type = audio
            .metadata
            .content_type
            .clone()
            .or_else(|| audio.metadata.source_mime_type.clone())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let mut listen_uri = if env_flag_any(&[
            "MEDIA_RESEARCH_STACK_INTERIM",
            "MEDIA_RESEARCH_STACK_MASTERING_INTERIM",
        ]) {
            "/v1/listen?utterances=true&paragraphs=true&timestamps=true&interim_results=true&language=en_US".to_string()
        } else {
            "/v1/listen?utterances=true&paragraphs=true&timestamps=true&language=en_US".to_string()
        };
        if is_s16le_16khz_mono(&content_type) {
            listen_uri.push_str("&encoding=linear16&sample_rate=16000&channels=1");
        }
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
            .handle_listen_stream(req, audio.body, Box::new(writer))
            .await
            .map_err(|error| anyhow!("streaming listen failed: {error}"))?;
        collector.transcript(allow_empty_transcript)
    }

    async fn shutdown(&self) {
        let handles = {
            let mut handles = self
                .handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for handle in handles.iter() {
                handle.abort();
            }
            std::mem::take(&mut *handles)
        };
        for handle in handles {
            let _ = handle.await;
        }
    }
}

async fn open_research_audio(
    resolver: &TranscribeAudioResolver,
    media_dir: Option<&Path>,
    source_index: usize,
    source_url: &str,
    require_cached: bool,
    cache_wait: Duration,
) -> Result<OpenedSource> {
    if let Some(directory) = media_dir {
        if let Some(source) = open_cached_audio(directory, source_index, source_url).await? {
            return Ok(source);
        }
        if require_cached {
            return wait_for_cached_audio(directory, source_index, source_url, cache_wait).await;
        }
        return cache_audio(resolver, directory, source_index, source_url).await;
    }

    let audio = resolver.open_youtube_audio(source_url).await?;
    let metadata = source_metadata(&audio, source_url)?;
    let body = body_stream_from_audio(audio, source_index, source_url);
    Ok(OpenedSource { metadata, body })
}

fn is_s16le_16khz_mono(content_type: &str) -> bool {
    let mut fields = content_type
        .split(';')
        .map(|field| field.trim().to_ascii_lowercase());
    if fields.next().as_deref() != Some("audio/x-raw") {
        return false;
    }

    let fields = fields.collect::<HashSet<_>>();
    fields.contains("format=s16le")
        && fields.contains("rate=16000")
        && fields.contains("channels=1")
}

fn source_metadata(audio: &TranscribeAudioStream, source_url: &str) -> Result<SourceMetadata> {
    Ok(SourceMetadata {
        source_url: source_url.to_string(),
        duration_seconds: audio
            .duration_seconds
            .with_context(|| format!("av-ingest did not return duration for {source_url}"))?,
        content_length: header_u64(audio, "content-length"),
        content_type: header_string(audio, CONTENT_TYPE.as_str())
            .or_else(|| audio.mime_type.clone()),
        source_mime_type: audio.mime_type.clone(),
        resolver: audio.resolver.clone(),
        itag: audio.itag,
        cached: false,
    })
}

async fn open_cached_audio(
    directory: &Path,
    source_index: usize,
    source_url: &str,
) -> Result<Option<OpenedSource>> {
    let Some(source) = open_cached_source(directory, source_index, source_url)? else {
        return Ok(None);
    };
    let body = body_stream_from_file(&source.media_path, source_index, source_url).await?;
    Ok(Some(OpenedSource {
        metadata: source.metadata,
        body,
    }))
}

async fn wait_for_cached_audio(
    directory: &Path,
    source_index: usize,
    source_url: &str,
    timeout: Duration,
) -> Result<OpenedSource> {
    let started_at = Instant::now();
    eprintln!(
        "[{}] waiting up to {}s for the cache process",
        source_index + 1,
        timeout.as_secs()
    );
    loop {
        if let Some(source) = open_cached_audio(directory, source_index, source_url).await? {
            eprintln!(
                "[{}] cache became ready after {:.2}s",
                source_index + 1,
                started_at.elapsed().as_secs_f64()
            );
            return Ok(source);
        }
        anyhow::ensure!(
            started_at.elapsed() < timeout,
            "cache process did not publish source {} within {}s",
            source_url,
            timeout.as_secs()
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn cache_audio(
    resolver: &TranscribeAudioResolver,
    directory: &Path,
    source_index: usize,
    source_url: &str,
) -> Result<OpenedSource> {
    eprintln!(
        "[{}] downloading compressed source audio with av-ingest",
        source_index + 1
    );
    let outcome = ensure_cached_source(resolver, directory, source_index, source_url).await?;
    let media_rtfx = outcome.media_rtfx();
    let source = outcome.source;
    let metadata = source.metadata;
    eprintln!(
        "[{}] cached {} compressed bytes at {:.1}x real time from {}",
        source_index + 1,
        metadata.content_length.unwrap_or(0),
        media_rtfx,
        source_url
    );
    let body = body_stream_from_file(&source.media_path, source_index, source_url).await?;
    Ok(OpenedSource { metadata, body })
}

impl Drop for LocalAsrHarness {
    fn drop(&mut self) {
        let handles = self
            .handles
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for handle in handles {
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

async fn body_stream_from_file(
    path: &Path,
    source_index: usize,
    source_url: &str,
) -> Result<BodyStream> {
    let file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("failed to open cached audio {}", path.display()))?;
    let content_length = file.metadata().await?.len();
    let source_url = source_url.to_string();
    Ok(Box::pin(stream::try_unfold(
        (file, 0u64, 1024 * 1024u64),
        move |(mut file, mut total_bytes, mut next_log_at)| {
            let source_url = source_url.clone();
            async move {
                let mut buffer = vec![0u8; 64 * 1024];
                let read = file.read(&mut buffer).await.map_err(|error| {
                    ServerError::Config(format!("cached audio read failed: {error}"))
                })?;
                if read == 0 {
                    return Ok(None);
                }
                buffer.truncate(read);
                total_bytes += read as u64;
                while total_bytes >= next_log_at {
                    eprintln!(
                        "[{}] cached body read {} / {} bytes from {}",
                        source_index + 1,
                        total_bytes,
                        content_length,
                        source_url
                    );
                    next_log_at += 1024 * 1024;
                }
                Ok(Some((
                    Bytes::from(buffer),
                    (file, total_bytes, next_log_at),
                )))
            }
        },
    )))
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
    log_segment_metrics: bool,
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
            log_segment_metrics: env_flag("MEDIA_RESEARCH_STACK_LOG_SEGMENT_METRICS"),
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
                } else if self.log_segment_metrics {
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
    fn transcript(&self, allow_empty: bool) -> Result<String> {
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
        if allow_empty {
            return Ok(String::new());
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
        source_urls_from_file(Path::new(&path))?
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

fn transcripts_dir() -> Result<Option<PathBuf>> {
    let Some(path) = first_env(&["MEDIA_RESEARCH_STACK_TRANSCRIPTS_DIR"]).map(PathBuf::from) else {
        return Ok(None);
    };
    fs::create_dir_all(&path)
        .with_context(|| format!("failed to create transcript directory {}", path.display()))?;
    Ok(Some(path))
}

fn media_dir() -> Result<Option<PathBuf>> {
    let Some(path) = first_env(&["MEDIA_RESEARCH_STACK_MEDIA_DIR"]).map(PathBuf::from) else {
        return Ok(None);
    };
    fs::create_dir_all(&path)
        .with_context(|| format!("failed to create media cache directory {}", path.display()))?;
    Ok(Some(path))
}

fn write_transcript(
    directory: &Path,
    source_index: usize,
    source_url: &str,
    transcript: &str,
) -> Result<PathBuf> {
    let path = directory.join(format!(
        "{}.txt",
        source_file_stem(source_index, source_url)
    ));
    let temporary_path = path.with_extension("txt.part");
    let normalized = format!("{}\n", transcript.trim());
    fs::write(&temporary_path, normalized)
        .with_context(|| format!("failed to write transcript {}", temporary_path.display()))?;
    fs::rename(&temporary_path, &path)
        .with_context(|| format!("failed to publish transcript {}", path.display()))?;
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

fn is_systemic_youtube_auth_error(error: &str) -> bool {
    error.contains("Sign in to confirm you’re not a bot")
        || error.contains("Sign in to confirm you're not a bot")
        || error.contains("cannot decrypt v10 cookies")
        || error.contains("find-generic-password failed")
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

fn set_default_asr_backend() {
    if first_env(&["ASR_COHERE_BACKEND"]).is_none() {
        env::set_var(
            "ASR_COHERE_BACKEND",
            if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "onnx"
            },
        );
    }
}

fn benchmark_settings() -> Result<BenchmarkSettings> {
    let backend = env::var("ASR_COHERE_BACKEND")
        .unwrap_or_else(|_| {
            if cfg!(target_os = "macos") {
                "mlx".to_string()
            } else {
                "onnx".to_string()
            }
        })
        .trim()
        .to_ascii_lowercase();
    let asr_concurrency = positive_env_usize("MEDIA_RESEARCH_STACK_ASR_CONCURRENCY", 1)?;
    let worker_instances = positive_env_usize("MEDIA_RESEARCH_STACK_WORKER_INSTANCES", 1)?;
    let onnx_sessions = positive_env_usize("ASR_ONNX_SESSIONS", 1)?;
    let minimum_transcript_words = env_usize("MEDIA_RESEARCH_STACK_MIN_TRANSCRIPT_WORDS", 5)?;
    let device_ids = parse_device_ids(
        first_env(&["ASR_DEVICE_IDS"])
            .as_deref()
            .unwrap_or(if backend == "mlx" { "" } else { "0" }),
    )?;
    Ok(BenchmarkSettings {
        backend,
        device_ids,
        onnx_sessions,
        asr_concurrency,
        worker_instances,
        minimum_transcript_words,
    })
}

fn parse_device_ids(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid ASR_DEVICE_IDS entry {value:?}"))
        })
        .collect()
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

fn asr_config(
    role: AppRole,
    worker_id: &str,
    model_dir: Option<PathBuf>,
    settings: &BenchmarkSettings,
) -> Result<AppConfig> {
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
        device_ids: settings.device_ids.clone(),
        onnx_sessions: settings.onnx_sessions,
        cohere_max_new_tokens: first_env(&[
            "ASR_COHERE_MAX_NEW_TOKENS",
            "MEDIA_RESEARCH_STACK_MASTERING_MAX_NEW_TOKENS",
        ])
        .and_then(|value| value.parse().ok())
        .unwrap_or(384),
        chunk_seconds: 30.0,
        overlap_seconds: 2.0,
        final_min_seconds: 0.5,
        utt_split_seconds: 0.8,
        upload_response_num_streams: env_usize(
            "UPLOAD_RESPONSE_NUM_STREAMS",
            DEFAULT_UPLOAD_RESPONSE_NUM_STREAMS.max(settings.asr_concurrency),
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
        upload_response_max_inflight: env_usize(
            "UPLOAD_RESPONSE_MAX_INFLIGHT",
            settings.asr_concurrency,
        )?,
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

fn positive_env_usize(name: &str, default: usize) -> Result<usize> {
    let value = env_usize(name, default)?;
    anyhow::ensure!(value > 0, "{name} must be greater than 0");
    Ok(value)
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
    fn retains_an_empty_successful_transcript_when_allowed() {
        let collector = ProgressCollector {
            state: Arc::new(Mutex::new(ProgressState {
                status: Some(StatusCode::OK),
                ..ProgressState::default()
            })),
        };
        assert!(collector.transcript(false).is_err());
        assert_eq!(collector.transcript(true).unwrap(), "");
    }

    #[test]
    fn ring_size_rounds_up_to_whole_slots() {
        assert_eq!(slots_for_ring_bytes(65_537, 32_768), 3);
    }

    #[test]
    fn parses_empty_and_multiple_device_ids() {
        assert_eq!(parse_device_ids("").unwrap(), Vec::<usize>::new());
        assert_eq!(parse_device_ids("0, 2").unwrap(), vec![0, 2]);
        assert!(parse_device_ids("gpu").is_err());
    }

    #[test]
    fn recognizes_the_benchmark_pcm_format() {
        assert!(is_s16le_16khz_mono(
            "audio/x-raw; format=S16LE; rate=16000; channels=1"
        ));
        assert!(!is_s16le_16khz_mono("audio/webm"));
        assert!(!is_s16le_16khz_mono(
            "audio/x-raw; format=S16LE; rate=48000; channels=1"
        ));
    }

    #[test]
    fn recognizes_systemic_youtube_auth_failures() {
        assert!(is_systemic_youtube_auth_error(
            "ERROR: [youtube] id: Sign in to confirm you’re not a bot"
        ));
        assert!(is_systemic_youtube_auth_error(
            "WARNING: cannot decrypt v10 cookies: no key found"
        ));
        assert!(!is_systemic_youtube_auth_error(
            "ERROR: [youtube] id: Video unavailable"
        ));
    }

    #[test]
    fn writes_a_clean_transcript_atomically() {
        let directory = env::temp_dir().join(format!(
            "media-research-stack-transcript-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();

        let path = write_transcript(
            &directory,
            6,
            "https://www.youtube.com/watch?v=video-id&feature=test",
            "  exact transcript text  ",
        )
        .unwrap();

        assert_eq!(path.file_name().unwrap(), "0007-video-id.txt");
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "exact transcript text\n"
        );
        assert!(!path.with_extension("txt.part").exists());
        fs::remove_dir_all(directory).unwrap();
    }
}
