# media-research-stack

Local macOS runner for media research tasks that need both `av-ingest` and
`asr-api`.

The binary starts these services in one Tokio process:

- `av-ingest-proxy` on `http://127.0.0.1:8444`
- `asr-api` ingress on `https://127.0.0.1:8443`
- an `asr-api` decoder worker attached to the same `upload-response` service
- one `asr-api` MLX worker attached to the same `upload-response` service

It defaults `av-ingest` to `AV_INGEST_PROXY_RESOLVE_MODE=transcribe`, so
`/resolve` returns audio-only formats when possible. If YouTube does not expose
an audio-only format, it keeps only the smallest muxed audio/video format.

## Requirements

```bash
brew install libvpx pkg-config yt-dlp
```

`ASR_MODEL_DIR` must point at a Cohere MLX bundle containing:

- `model.safetensors`
- `config.json`
- `vocab.json`

## Run

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
cargo run --release
```

Endpoints:

```text
AV ingest: http://127.0.0.1:8444
ASR:       https://127.0.0.1:8443/v1/listen
```

YouTube resolver options are still the `av-ingest-proxy` environment variables,
for example:

```bash
AV_INGEST_PROXY_YTDLP_EXTRACTOR_ARGS='youtube:player_client=mweb' \
AV_INGEST_PROXY_YTDLP_COOKIES_FROM_BROWSER=chrome \
MACOSX_DEPLOYMENT_TARGET=14.0 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
cargo run --release
```

## Smoke Checks

```bash
curl -fsS http://127.0.0.1:8444/healthz
```

```bash
curl --http2 -k -fsS \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  'https://127.0.0.1:8443/v1/listen?utterances=true&paragraphs=true&timestamps=true'
```

## Long Integration Benchmark

The mastering video benchmark is opt-in because it resolves public YouTube
sources and transcribes the full audio. It uses `av-ingest` as a Rust library
to open the selected audio stream, then writes that stream directly into
`asr-api`/`upload-response` in-process.

```bash
MEDIA_RESEARCH_STACK_MASTERING_BENCH=1 \
MACOSX_DEPLOYMENT_TARGET=14.0 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
cargo test --test mastering_videos -- --nocapture
```

By default it runs several public audio mastering videos end to end and writes
`target/mastering-videos/report.jsonl`. Each JSONL row includes source URL,
audio seconds, wall seconds, RTFx, selected format metadata, and transcript
size. It does not write transcript text into the report.

Useful overrides:

```bash
MEDIA_RESEARCH_STACK_MASTERING_URLS='https://www.youtube.com/watch?v=...,https://www.youtube.com/watch?v=...'
MEDIA_RESEARCH_STACK_MASTERING_REPORT=target/mastering-videos/my-run.jsonl
UPLOAD_RESPONSE_RING_BYTES=1073741824
```

`UPLOAD_RESPONSE_RING_BYTES` defaults to 1 GiB per stream. With the default
32 KiB slot size this derives `UPLOAD_RESPONSE_SLOTS_PER_STREAM=32768`; set
`UPLOAD_RESPONSE_SLOTS_PER_STREAM` directly to override the derived slot count.

## Why This Repo Exists

Research jobs usually need both source-media resolution and ASR. Running both
inside one local binary keeps the Mac setup small: one command, one log stream,
and fixed local endpoints for scripts or notebooks.
