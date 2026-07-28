use anyhow::{Context, Result};
use av_ingest_proxy::TranscribeAudioResolver;
use clap::Parser;
use futures_util::{stream, StreamExt};
use media_research_stack::research_cache::{ensure_cached_source, source_urls_from_file};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(about = "Cache compressed research audio without starting ASR")]
struct Args {
    #[arg(long, env = "MEDIA_RESEARCH_STACK_URLS_FILE")]
    urls_file: PathBuf,

    #[arg(long, env = "MEDIA_RESEARCH_STACK_MEDIA_DIR")]
    media_dir: PathBuf,

    #[arg(
        long,
        env = "MEDIA_RESEARCH_STACK_CACHE_REPORT",
        default_value = "target/research/cache-report.jsonl"
    )]
    report: PathBuf,

    #[arg(
        long,
        env = "MEDIA_RESEARCH_STACK_CACHE_CONCURRENCY",
        default_value_t = 4
    )]
    concurrency: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.concurrency > 0, "cache concurrency must be positive");
    fs::create_dir_all(&args.media_dir)
        .with_context(|| format!("failed to create media cache {}", args.media_dir.display()))?;
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)?;
    }

    let urls = source_urls_from_file(&args.urls_file)?;
    anyhow::ensure!(!urls.is_empty(), "source manifest contains no URLs");
    let source_count = urls.len();
    let resolver = TranscribeAudioResolver::from_env()?;
    eprintln!(
        "cache process: {} source(s), concurrency {}, cache {}",
        source_count,
        args.concurrency,
        args.media_dir.display()
    );

    let started_at = Instant::now();
    let jobs = stream::iter(urls.into_iter().enumerate().map(|(index, source_url)| {
        let resolver = resolver.clone();
        let media_dir = args.media_dir.clone();
        async move {
            let result = ensure_cached_source(&resolver, &media_dir, index, &source_url).await;
            (index, source_url, result)
        }
    }))
    .buffer_unordered(args.concurrency);
    tokio::pin!(jobs);

    let mut ready = 0usize;
    let mut reused = 0usize;
    let mut failed = 0usize;
    while let Some((index, source_url, result)) = jobs.next().await {
        match result {
            Ok(outcome) => {
                ready += 1;
                reused += usize::from(outcome.reused);
                let metadata = &outcome.source.metadata;
                let bytes = metadata.content_length.unwrap_or(0);
                let media_rtfx = outcome.media_rtfx();
                append_json_line(
                    &args.report,
                    &json!({
                        "status": "ok",
                        "index": index,
                        "source_url": source_url,
                        "reused": outcome.reused,
                        "duration_seconds": metadata.duration_seconds,
                        "wall_seconds": outcome.wall_seconds,
                        "media_rtfx": media_rtfx,
                        "content_length": bytes,
                        "itag": metadata.itag,
                        "mime_type": metadata.source_mime_type,
                    }),
                )?;
                eprintln!(
                    "[{}/{}] cache ready: {} bytes, itag {:?}, {:.1}x real time, reused={}",
                    index + 1,
                    source_count,
                    bytes,
                    metadata.itag,
                    media_rtfx,
                    outcome.reused
                );
            }
            Err(error) => {
                failed += 1;
                let error_message = format!("{error:#}");
                append_json_line(
                    &args.report,
                    &json!({
                        "status": "error",
                        "index": index,
                        "source_url": source_url,
                        "error": &error_message,
                    }),
                )?;
                eprintln!("[{}] cache failed for {}: {error:#}", index + 1, source_url);
                if is_systemic_youtube_auth_error(&error_message) {
                    anyhow::bail!(
                        "YouTube rejected the cache process. Refresh its authentication before resuming. Original error: {error_message}"
                    );
                }
            }
        }
    }

    eprintln!(
        "cache process complete: {} ready, {} reused, {} failed in {:.1}s",
        ready,
        reused,
        failed,
        started_at.elapsed().as_secs_f64()
    );
    anyhow::ensure!(failed == 0, "{failed} cache job(s) failed");
    Ok(())
}

fn append_json_line(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn is_systemic_youtube_auth_error(error: &str) -> bool {
    error.contains("Sign in to confirm you’re not a bot")
        || error.contains("Sign in to confirm you're not a bot")
}
