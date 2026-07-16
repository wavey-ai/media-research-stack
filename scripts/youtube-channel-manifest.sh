#!/usr/bin/env bash
set -euo pipefail

channel_url="${1:-}"
output_path="${2:-target/research/channel-manifest.json}"

if [[ -z "${channel_url}" ]]; then
  echo "Usage: scripts/youtube-channel-manifest.sh CHANNEL_URL [OUTPUT_JSON]" >&2
  exit 2
fi

for command in yt-dlp jq; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required." >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${output_path}")"
raw_manifest="$(mktemp)"
normalized_manifest="$(mktemp)"
trap 'rm -f "${raw_manifest}" "${normalized_manifest}"' EXIT

yt-dlp \
  --flat-playlist \
  --dump-single-json \
  --skip-download \
  --no-warnings \
  "${channel_url}" >"${raw_manifest}"

jq '{
  schema_version: 1,
  generated_at: (now | todateiso8601),
  channel: {
    id: .channel_id,
    name: (.channel // .uploader // .title),
    url: .webpage_url
  },
  video_count: (.entries | length),
  total_known_duration_seconds: ([.entries[].duration // 0] | add),
  videos: [
    .entries[] | {
      id,
      title,
      url: ("https://www.youtube.com/watch?v=" + .id),
      duration_seconds: .duration,
      view_count,
      live_status
    }
  ]
}' "${raw_manifest}" >"${normalized_manifest}"

mv "${normalized_manifest}" "${output_path}"
jq '{
  channel,
  video_count,
  total_known_duration_seconds,
  total_known_duration_hours: (.total_known_duration_seconds / 3600)
}' "${output_path}"
echo "Saved ${output_path}" >&2
