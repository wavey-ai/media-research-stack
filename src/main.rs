use anyhow::{Context, Result};
use asr_api::config::{AppConfig, AppRole, AsrModelProvider, LogFormat};
use clap::Parser;
use std::env;
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "media-research-stack",
    about = "Run local av-ingest and asr-api services in one process"
)]
struct Config {
    #[arg(long, env = "ASR_MODEL_DIR")]
    model_dir: PathBuf,

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

    #[arg(long, default_value_t = 9443)]
    asr_decoder_port: u16,

    #[arg(long, default_value_t = 10443)]
    asr_worker_port: u16,

    #[arg(long, env = "RUST_LOG", default_value = "info")]
    rust_log: String,

    #[arg(long, env = "ASR_COHERE_MAX_NEW_TOKENS", default_value_t = 384)]
    cohere_max_new_tokens: usize,

    #[arg(long, env = "CHUNK_SECONDS", default_value_t = 30.0)]
    chunk_seconds: f32,

    #[arg(long, env = "OVERLAP_SECONDS", default_value_t = 2.0)]
    overlap_seconds: f32,

    #[arg(long, env = "FINAL_MIN_SECONDS", default_value_t = 0.5)]
    final_min_seconds: f32,

    #[arg(long, env = "UTT_SPLIT_SECONDS", default_value_t = 0.8)]
    utt_split_seconds: f64,

    #[arg(long, env = "UPLOAD_RESPONSE_NUM_STREAMS", default_value_t = 16)]
    upload_response_num_streams: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_SLOT_SIZE_KB", default_value_t = 32)]
    upload_response_slot_size_kb: usize,

    #[arg(
        long,
        env = "UPLOAD_RESPONSE_SLOTS_PER_STREAM",
        default_value_t = 1_024
    )]
    upload_response_slots_per_stream: usize,

    #[arg(long, env = "UPLOAD_RESPONSE_TIMEOUT_MS", default_value_t = 30_000)]
    upload_response_timeout_ms: u64,

    #[arg(long, env = "UPLOAD_RESPONSE_MAX_INFLIGHT", default_value_t = 2)]
    upload_response_max_inflight: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::parse();
    init_tracing(&config.rust_log)?;
    configure_process_env(&config);

    let ingress_url = format!("https://127.0.0.1:{}", config.asr_ingress_port);
    let (tx, mut rx) = mpsc::channel::<ServiceExit>(8);

    spawn_service("av-ingest", tx.clone(), av_ingest_proxy::run_from_env());
    spawn_service(
        "asr-ingress",
        tx.clone(),
        asr_api::run(asr_config(
            &config,
            AppRole::Ingress,
            config.asr_ingress_port,
            "asr-api-ingress-local",
            Vec::new(),
            None,
        )),
    );
    spawn_service(
        "asr-decoder",
        tx.clone(),
        asr_api::run(asr_config(
            &config,
            AppRole::Decoder,
            config.asr_decoder_port,
            "asr-api-decoder-local",
            vec![ingress_url.clone()],
            None,
        )),
    );
    spawn_service(
        "asr-worker",
        tx,
        asr_api::run(asr_config(
            &config,
            AppRole::Worker,
            config.asr_worker_port,
            "asr-api-worker-local-0",
            vec![ingress_url.clone()],
            Some(config.model_dir.clone()),
        )),
    );

    info!(
        av_ingest = %format!("http://127.0.0.1:{}", config.av_port),
        asr_listen = %format!("{ingress_url}/v1/listen"),
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

fn configure_process_env(config: &Config) {
    env::set_var("ASR_COHERE_BACKEND", "mlx");
    env::set_var("AV_INGEST_PROXY_LOCAL_HTTP", "1");
    env::set_var("AV_INGEST_PROXY_PORT", config.av_port.to_string());
    env::set_var("AV_INGEST_PROXY_RESOLVE_MODE", &config.av_resolve_mode);
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
        upload_response_slots_per_stream: config.upload_response_slots_per_stream,
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
