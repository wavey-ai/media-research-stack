# media-research-stack

Local macOS runner for media research tasks that need both `av-ingest` and
`asr-api`.

The binary starts these services in one Tokio process:

- `av-ingest-proxy` on `http://127.0.0.1:8444`
- `asr-api` ingress on `https://127.0.0.1:8443`
- `asr-api` decoder on `https://127.0.0.1:9443`
- one `asr-api` MLX worker on `https://127.0.0.1:10443`

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

## Why This Repo Exists

Research jobs usually need both source-media resolution and ASR. Running both
inside one local binary keeps the Mac setup small: one command, one log stream,
and fixed local endpoints for scripts or notebooks.
