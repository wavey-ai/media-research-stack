use anyhow::{anyhow, Context, Result};
use av_ingest_proxy::TranscribeAudioResolver;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceMetadata {
    pub source_url: String,
    pub duration_seconds: u64,
    pub content_length: Option<u64>,
    pub content_type: Option<String>,
    pub source_mime_type: Option<String>,
    pub resolver: String,
    pub itag: Option<u64>,
    pub cached: bool,
}

#[derive(Clone, Debug)]
pub struct CachedSource {
    pub metadata: SourceMetadata,
    pub media_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct CacheOutcome {
    pub source: CachedSource,
    pub reused: bool,
    pub wall_seconds: f64,
}

impl CacheOutcome {
    pub fn media_rtfx(&self) -> f64 {
        if self.reused {
            0.0
        } else {
            self.source.metadata.duration_seconds as f64 / self.wall_seconds.max(0.001)
        }
    }
}

pub async fn ensure_cached_source(
    resolver: &TranscribeAudioResolver,
    directory: &Path,
    source_index: usize,
    source_url: &str,
) -> Result<CacheOutcome> {
    if let Some(source) = open_cached_source(directory, source_index, source_url)? {
        return Ok(CacheOutcome {
            source,
            reused: true,
            wall_seconds: 0.0,
        });
    }

    fs::create_dir_all(directory)
        .with_context(|| format!("failed to create media cache {}", directory.display()))?;
    let stem = source_file_stem(source_index, source_url);
    let media_path = directory.join(format!("{stem}.audio"));
    let partial_media_path = directory.join(format!("{stem}.audio.download"));
    let metadata_path = directory.join(format!("{stem}.json"));
    let partial_metadata_path = directory.join(format!("{stem}.json.part"));
    let _ = fs::remove_file(&partial_media_path);
    let _ = fs::remove_file(&partial_metadata_path);

    let started_at = Instant::now();
    let downloaded = resolver
        .download_youtube_audio(source_url, &partial_media_path)
        .await
        .with_context(|| format!("failed to cache {source_url}"))?;
    let metadata = SourceMetadata {
        source_url: source_url.to_string(),
        duration_seconds: downloaded
            .duration_seconds
            .with_context(|| format!("av-ingest did not return duration for {source_url}"))?,
        content_length: Some(downloaded.content_length),
        content_type: downloaded.mime_type.clone(),
        source_mime_type: downloaded.mime_type,
        resolver: downloaded.resolver,
        itag: downloaded.itag,
        cached: true,
    };
    anyhow::ensure!(
        metadata_matches_required_mime_type(&metadata),
        "downloaded cache media for {source_url} has MIME type {:?}; expected {}",
        metadata.source_mime_type,
        required_cache_mime_type().unwrap_or_default()
    );
    fs::rename(&partial_media_path, &media_path)
        .with_context(|| format!("failed to publish cached audio {}", media_path.display()))?;
    fs::write(
        &partial_metadata_path,
        serde_json::to_vec_pretty(&metadata)?,
    )
    .with_context(|| {
        format!(
            "failed to write cache metadata {}",
            partial_metadata_path.display()
        )
    })?;
    fs::rename(&partial_metadata_path, &metadata_path).with_context(|| {
        format!(
            "failed to publish cache metadata {}",
            metadata_path.display()
        )
    })?;

    Ok(CacheOutcome {
        source: CachedSource {
            metadata,
            media_path,
        },
        reused: false,
        wall_seconds: started_at.elapsed().as_secs_f64(),
    })
}

pub fn open_cached_source(
    directory: &Path,
    source_index: usize,
    source_url: &str,
) -> Result<Option<CachedSource>> {
    let stem = source_file_stem(source_index, source_url);
    let media_path = directory.join(format!("{stem}.audio"));
    let metadata_path = directory.join(format!("{stem}.json"));
    let metadata_bytes = match fs::read(&metadata_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut metadata: SourceMetadata =
        serde_json::from_slice(&metadata_bytes).with_context(|| {
            format!(
                "failed to parse cached metadata {}",
                metadata_path.display()
            )
        })?;
    if metadata.source_url != source_url {
        return Ok(None);
    }
    if !metadata_matches_required_mime_type(&metadata) {
        return Ok(None);
    }
    let file_size = match fs::metadata(&media_path) {
        Ok(value) => value.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if file_size == 0
        || metadata
            .content_length
            .is_some_and(|length| length != file_size)
    {
        return Ok(None);
    }
    metadata.content_length = Some(file_size);
    metadata.cached = true;
    Ok(Some(CachedSource {
        metadata,
        media_path,
    }))
}

pub fn source_urls_from_file(path: &Path) -> Result<Vec<String>> {
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

pub fn source_file_stem(source_index: usize, source_url: &str) -> String {
    let source_id = source_url
        .split_once("v=")
        .map(|(_, value)| value)
        .unwrap_or(source_url)
        .split(['&', '?', '#', '/'])
        .find(|value| !value.is_empty())
        .unwrap_or("source");
    let source_id = source_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{:04}-{source_id}", source_index + 1)
}

fn split_urls(value: &str) -> Vec<String> {
    value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn required_cache_mime_type() -> Option<String> {
    env::var("MEDIA_RESEARCH_STACK_CACHE_MIME_TYPE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
}

fn metadata_matches_required_mime_type(metadata: &SourceMetadata) -> bool {
    let Some(required) = required_cache_mime_type() else {
        return true;
    };
    metadata_has_mime_type(metadata, &required)
}

fn metadata_has_mime_type(metadata: &SourceMetadata, required: &str) -> bool {
    metadata
        .source_mime_type
        .as_deref()
        .or(metadata.content_type.as_deref())
        .is_some_and(|value| {
            value
                .split_once(';')
                .map_or(value, |(mime_type, _)| mime_type)
                .trim()
                .eq_ignore_ascii_case(&required)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(mime_type: &str) -> SourceMetadata {
        SourceMetadata {
            source_url: "https://example.com/audio".to_string(),
            duration_seconds: 60,
            content_length: Some(10),
            content_type: Some(mime_type.to_string()),
            source_mime_type: Some(mime_type.to_string()),
            resolver: "test".to_string(),
            itag: Some(249),
            cached: true,
        }
    }

    #[test]
    fn matches_required_cache_mime_type() {
        assert!(metadata_has_mime_type(
            &metadata("audio/webm; codecs=opus"),
            "audio/webm"
        ));
        assert!(!metadata_has_mime_type(
            &metadata("audio/mp4"),
            "audio/webm"
        ));
    }
}
