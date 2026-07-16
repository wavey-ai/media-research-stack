use anyhow::{Context, Result};
use asr_api::asr::AsrBackend;
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use asr_api::decoder::DecoderState;
use asr_api::ingress::{ListenIngress, ListenIngressWebSocketHandler};
use asr_api::router::AppRouter;
use asr_api::worker::WorkerState;
use clap::Parser;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use upload_response::{ResponseWatcher, UploadResponseRouter, UploadResponseService};
use web_service::{H2H3Server, Server, ServerBuilder};

const DEFAULT_UPLOAD_RESPONSE_RING_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_UPLOAD_RESPONSE_NUM_STREAMS: usize = 2;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "media-research-stack",
    about = "Run local av-ingest and asr-api services in one process"
)]
struct Config {
    #[arg(long, env = "ASR_MODEL_DIR")]
    model_dir: PathBuf,

    #[arg(long, env = "ASR_MLX_TRANSCRIBE_BIN")]
    mlx_transcribe_bin: Option<PathBuf>,

    #[arg(long, env = "AV_INGEST_PROXY_PORT", default_value_t = 8444)]
    av_port: u16,

    #[arg(
        long,
        env = "AV_INGEST_PROXY_RESOLVE_MODE",
        default_value = "transcribe"
    )]
    av_resolve_mode: String,

    #[arg(long, default_value_t = 8443)]
    asr_ingress_port: u16,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    rust_log: String,

    #[arg(long, env = "ASR_COHERE_MAX_NEW_TOKENS", default_value_t = 128)]
    cohere_max_new_tokens: usize,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    final_min_seconds: f32,

    #[arg(long, env = "UTT_SPLIT_SECONDS", default_value_t = 0.8)]
    utt_split_seconds: f64,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_NUM_STREAMS",
        default_value_t = DEFAULT_UPLOAD_RESPONSE_NUM_STREAMS
    )]
    upload_response_num_streams: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOT_SIZE_KB", default_value_t = 32)]
    upload_response_slot_size_kb: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_RING_BYTES",
        default_value_t = DEFAULT_UPLOAD_RESPONSE_RING_BYTES
    )]
    upload_response_ring_bytes: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
        help = "Explicit slot-count override; otherwise derived from UPLOAD_RESPONSE_RING_BYTES"
    )]
    upload_response_slots_per_stream: Option<usize>,

    #[arg(long, env = "UPLOAD_RESPONSE_TIMEOUT_MS", default_value_t = 300_000)]
    upload_response_timeout_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_MAX_INFLIGHT", default_value_t = 1)]
    upload_response_max_inflight: usize,

    #[arg(
        long,
        env = "MEDIA_RESEARCH_STACK_LOG_TRANSCRIPTS",
        default_value_t = false
    )]
    log_transcripts: bool,
}

impl Config {
    fn upload_response_slot_bytes(&self) -> usize {
        self.upload_response_slot_size_kb
            .saturating_mul(1024)
            .max(1)
    }

    fn upload_response_slots_per_stream(&self) -> usize {
        self.upload_response_slots_per_stream
            .unwrap_or_else(|| {
                slots_for_ring_bytes(
                    self.upload_response_ring_bytes,
                    self.upload_response_slot_bytes(),
                )
            })
            .max(3)
    }

    fn upload_response_effective_ring_bytes(&self) -> usize {
        self.upload_response_slots_per_stream()
            .saturating_mul(self.upload_response_slot_bytes())
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.av_port != self.asr_ingress_port,
            "av-ingest and ASR must use different ports"
        );
        anyhow::ensure!(
            self.upload_response_num_streams > 0,
            "UPLOAD_RESPONSE_NUM_STREAMS must be greater than zero"
        );
        anyhow::ensure!(
            self.upload_response_max_inflight > 0,
            "UPLOAD_RESPONSE_MAX_INFLIGHT must be greater than zero"
        );
        Ok(())
    }
}

fn slots_for_ring_bytes(ring_bytes: usize, slot_bytes: usize) -> usize {
    let slot_bytes = slot_bytes.max(1);
    ring_bytes.max(slot_bytes).saturating_add(slot_bytes - 1) / slot_bytes
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();
    config.validate()?;
    init_tracing(&config.rust_log)?;
    configure_process_env(&config)?;

    let (tx, mut rx) = mpsc::channel::<ServiceExit>(8);

    spawn_service("av-ingest", tx.clone(), av_ingest_proxy::run_from_env());
    spawn_service("asr-local", tx, run_local_asr(config.clone()));

    info!(
        av_ingest = %format!("http://127.0.0.1:{}", config.av_port),
        asr_listen = %format!("https://127.0.0.1:{}/v1/listen", config.asr_ingress_port),
        "research stack started"
    );

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("shutdown requested");
            Ok(())
        }
        Some(exit) = rx.recv() => {
            match exit.result {
                Ok(()) => {
                    error!(service = %exit.name, "service exited");
                    anyhow::bail!("{} exited", exit.name)
                }
                Err(error) => {
                    error!(service = %exit.name, %error, "service failed");
                    anyhow::bail!("{} failed: {error}", exit.name)
                }
            }
        }
    }
}

fn init_tracing(rust_log: &str) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_new(rust_log).context("invalid RUST_LOG filter")?)
        .compact()
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize tracing: {error}"))
}

fn configure_process_env(config: &Config) -> Result<()> {
    let mlx_transcribe_bin = resolve_mlx_transcribe_bin(config)?;
    ensure_mlx_metallib(&mlx_transcribe_bin)?;
    env::set_var("ASR_COHERE_BACKEND", "mlx");
    env::set_var("ASR_MLX_TRANSCRIBE_BIN", &mlx_transcribe_bin);
    env::set_var("AV_INGEST_PROXY_LOCAL_HTTP", "1");
    env::set_var("AV_INGEST_PROXY_PORT", config.av_port.to_string());
    env::set_var("AV_INGEST_PROXY_RESOLVE_MODE", &config.av_resolve_mode);
    info!(
        mlx_transcribe_bin = %mlx_transcribe_bin.display(),
        "using Cohere MLX runtime"
    );
    Ok(())
}

fn ensure_mlx_metallib(runtime: &Path) -> Result<()> {
    let runtime_dir = runtime
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cohere MLX runtime has no parent directory"))?;
    let destination = runtime_dir.join("mlx.metallib");
    if destination.is_file() {
        return Ok(());
    }

    let build_dir = runtime_dir.parent().ok_or_else(|| {
        anyhow::anyhow!("Cohere MLX runtime is not inside a Swift build directory")
    })?;
    let candidates = [
        build_dir.join("mlx-metal/mlx.metallib"),
        build_dir.join("arm64-apple-macosx/debug/mlx.metallib"),
    ];
    let source = candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MLX Metal library was not found. Rebuild the runtime with `swift build -c release --package-path ../asr-api/apple`."
            )
        })?;
    fs::copy(&source, &destination).with_context(|| {
        format!(
            "failed to install MLX Metal library from {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    info!(
        source = %source.display(),
        destination = %destination.display(),
        "installed MLX Metal library"
    );
    Ok(())
}

fn resolve_mlx_transcribe_bin(config: &Config) -> Result<PathBuf> {
    if let Some(path) = &config.mlx_transcribe_bin {
        anyhow::ensure!(
            path.is_file(),
            "Cohere MLX runtime does not exist: {}",
            path.display()
        );
        return Ok(path.clone());
    }

    let current_dir = env::current_dir().context("failed to resolve current directory")?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        current_dir.join("../asr-api/apple/.build/release/asr-mlx-transcribe"),
        manifest_dir.join("../asr-api/apple/.build/release/asr-mlx-transcribe"),
    ];
    if let Some(path) = candidates.into_iter().find(|path| path.is_file()) {
        return Ok(path);
    }

    anyhow::bail!(
        "Cohere MLX runtime was not found. Build it with `swift build -c release --package-path ../asr-api/apple`, then set ASR_MLX_TRANSCRIBE_BIN to the resulting asr-mlx-transcribe binary."
    )
}

async fn run_local_asr(config: Config) -> Result<()> {
    let ingress_config = asr_config(
        &config,
        AppRole::Ingress,
        config.asr_ingress_port,
        "asr-api-ingress-local",
        Vec::new(),
        None,
    );
    let decoder_config = asr_config(
        &config,
        AppRole::Decoder,
        0,
        "asr-api-decoder-local",
        Vec::new(),
        None,
    );
    let worker_config = asr_config(
        &config,
        AppRole::Worker,
        0,
        "asr-api-worker-local-0",
        Vec::new(),
        Some(config.model_dir.clone()),
    );

    ingress_config.validate()?;

    let upload_service = Arc::new(UploadResponseService::new(
        ingress_config.upload_response_config(),
    ));
    let _watcher_handle = ResponseWatcher::new(upload_service.clone())
        .with_poll_interval_ms(ingress_config.upload_response_watch_poll_ms)
        .spawn();
    let _transcript_handle = config.log_transcripts.then(|| {
        spawn_transcript_watcher(
            upload_service.clone(),
            ingress_config.upload_response_watch_poll_ms,
        )
    });
    let _decoder_handle =
        Arc::new(DecoderState::new(decoder_config)).spawn_cache_worker(upload_service.clone());

    let backend = Arc::new(AsrBackend::new(
        worker_config.model_dir()?,
        worker_config.resolved_model_provider()?,
        &worker_config.device_ids,
        worker_config.onnx_sessions,
        worker_config.cohere_max_new_tokens,
    )?);
    let _worker_handle = Arc::new(WorkerState::new(worker_config, backend))
        .spawn_cache_worker(upload_service.clone());

    let upload_router = Arc::new(UploadResponseRouter::new(upload_service.clone()));
    let listen_ingress = Arc::new(ListenIngress::new(
        ingress_config.clone(),
        upload_service.clone(),
    ));
    let listen_ws = Arc::new(ListenIngressWebSocketHandler::new(listen_ingress.clone()));
    let router = Box::new(AppRouter::new(
        ingress_config.clone(),
        Some(upload_router),
        Some(listen_ingress),
        Some(listen_ws),
    ));

    let (cert_b64, key_b64) = ingress_config.tls_base64()?;
    let server = H2H3Server::builder()
        .with_tls(cert_b64, key_b64)
        .with_port(ingress_config.port)
        .enable_h2(true)
        .enable_h3(ingress_config.enable_h3)
        .enable_websocket(true)
        .with_router(router)
        .build()
        .context("failed to build local ASR ingress server")?;
    let handle = server
        .start()
        .await
        .context("failed to start ASR ingress")?;
    let _ = handle.ready_rx.await;

    info!(
        asr_listen = %format!("https://127.0.0.1:{}/v1/listen", ingress_config.port),
        upload_response_num_streams = ingress_config.upload_response_num_streams,
        upload_response_slot_size_kb = ingress_config.upload_response_slot_size_kb,
        upload_response_slots_per_stream = ingress_config.upload_response_slots_per_stream,
        upload_response_ring_bytes = config.upload_response_effective_ring_bytes(),
        "local ASR stack ready"
    );

    let _ = handle.finished_rx.await;
    anyhow::bail!("ASR ingress server exited")
}

fn spawn_transcript_watcher(service: Arc<UploadResponseService>, poll_interval_ms: u64) {
    tokio::spawn(async move {
        let mut poll_interval = interval(Duration::from_millis(poll_interval_ms.max(1)));
        let num_streams = service.config().num_streams;

        let mut stream_ids: Vec<u64> = vec![0; num_streams];
        let mut last_seen: Vec<usize> = vec![0; num_streams];
        let mut body_buffers: HashMap<u64, Vec<u8>> = HashMap::new();

        loop {
            poll_interval.tick().await;

            for stream_idx in 0..num_streams {
                let stream_id = service.slot_stream_id(stream_idx).unwrap_or(0);
                let previous_stream_id = stream_ids[stream_idx];

                if stream_id == 0 {
                    if previous_stream_id != 0 {
                        body_buffers.remove(&previous_stream_id);
                        last_seen[stream_idx] = 0;
                        stream_ids[stream_idx] = 0;
                    }
                    continue;
                }

                if previous_stream_id != stream_id {
                    if previous_stream_id != 0 {
                        body_buffers.remove(&previous_stream_id);
                    }
                    stream_ids[stream_idx] = stream_id;
                    last_seen[stream_idx] = 0;
                }

                let current_last = service.response_last(stream_id).unwrap_or(0);
                let seen = last_seen[stream_idx];
                if current_last <= seen {
                    continue;
                }

                let body = body_buffers.entry(stream_id).or_default();

                for slot_id in (seen + 1)..=current_last {
                    if let Some(bytes) = service.response_get(stream_id, slot_id).await {
                        if slot_id == 1 {
                            continue;
                        }

                        if UploadResponseService::is_end_marker(&bytes) {
                            let transcript = extract_asr_transcript(body);
                            if let Some(transcript) = transcript {
                                debug!(
                                    stream_id,
                                    stream_idx,
                                    asr_transcript = %transcript,
                                    "ASR response completed"
                                );
                            }
                            body_buffers.remove(&stream_id);
                            break;
                        }

                        body.extend_from_slice(&bytes);
                    }
                }

                last_seen[stream_idx] = current_last;
            }
        }
    });
}

fn extract_asr_transcript(body: &[u8]) -> Option<String> {
    let body = String::from_utf8_lossy(body);
    let mut final_segments: Vec<String> = Vec::new();
    let mut interim_segments: Vec<String> = Vec::new();

    for raw_line in body.lines() {
        let raw_line = raw_line.trim();
        if raw_line.is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(raw_line) {
            Ok(value) => value,
            Err(error) => {
                warn!(error = %error, "Failed to parse ASR response JSON line for transcript");
                continue;
            }
        };

        let transcript = value
            .pointer("/channel/alternatives/0/transcript")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .pointer("/results/channels/0/alternatives/0/transcript")
                    .and_then(Value::as_str)
            })
            .map(|transcript| transcript.trim())
            .unwrap_or("");

        if transcript.is_empty() {
            continue;
        }

        let is_final = value
            .get("is_final")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if is_final {
            final_segments.push(transcript.to_string());
        } else {
            interim_segments.push(transcript.to_string());
        }
    }

    if !final_segments.is_empty() {
        Some(final_segments.join(" "))
    } else if !interim_segments.is_empty() {
        Some(interim_segments.join(" "))
    } else {
        None
    }
}

fn spawn_service<F>(name: &'static str, tx: mpsc::Sender<ServiceExit>, future: F)
where
    F: Future<Output = Result<()>> + Send + 'static,
{
    tokio::spawn(async move {
        let result = future.await.map_err(|error| format!("{error:#}"));
        let _ = tx
            .send(ServiceExit {
                name: name.to_string(),
                result,
            })
            .await;
    });
}

struct ServiceExit {
    name: String,
    result: std::result::Result<(), String>,
}

fn asr_config(
    config: &Config,
    role: AppRole,
    port: u16,
    worker_id: &str,
    ingress_urls: Vec<String>,
    model_dir: Option<PathBuf>,
) -> AppConfig {
    AppConfig {
        role,
        rust_log: config.rust_log.clone(),
        log_format: LogFormat::Compact,
        port,
        enable_h3: false,
        tls_cert_path: None,
        tls_key_path: None,
        model_dir,
        model_provider: AsrModelProvider::Cohere,
        device_ids: Vec::new(),
        onnx_sessions: 1,
        cohere_max_new_tokens: config.cohere_max_new_tokens,
        chunk_seconds: config.chunk_seconds,
        overlap_seconds: config.overlap_seconds,
        final_min_seconds: config.final_min_seconds,
        utt_split_seconds: config.utt_split_seconds,
        upload_response_num_streams: config.upload_response_num_streams,
        upload_response_slot_size_kb: config.upload_response_slot_size_kb,
        upload_response_slots_per_stream: config.upload_response_slots_per_stream(),
        upload_response_timeout_ms: config.upload_response_timeout_ms,
        upload_response_watch_poll_ms: 1,
        upload_response_worker_poll_ms: 2,
        upload_response_max_inflight: config.upload_response_max_inflight,
        upload_response_worker_id: worker_id.to_string(),
        upload_response_ingress_urls: ingress_urls,
        upload_response_discovery_dns: None,
        upload_response_discovery_interval_ms: 2_000,
        upload_response_insecure_tls: true,
        upload_response_worker_heartbeat_interval_ms: 1_000,
        upload_response_worker_ttl_ms: 5_000,
    }
}
