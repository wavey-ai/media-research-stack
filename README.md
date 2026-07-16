# media-research-stack

`media-research-stack` is the supported local macOS runner for research jobs
that need public-media resolution and streaming speech recognition in one
process.

It starts:

- `av-ingest` on `http://127.0.0.1:8444`;
- Deepgram-compatible ASR on `https://127.0.0.1:8443/v1/listen`;
- an in-process SoundKit decoder that converts compressed source audio to mono
  16 kHz PCM as bytes arrive; and
- one Cohere Transcribe MLX worker for Apple Silicon.

No FFmpeg process is used. `av-ingest` resolves and streams the selected source
format, SoundKit incrementally decodes formats such as WebM/Opus, and `asr-api`
chunks the normalized PCM into 30-second windows with two seconds of overlap.

## Platform and requirements

The current product target is Apple Silicon macOS 14 or newer.

Install the command-line dependencies:

```bash
brew install libvpx pkg-config yt-dlp jq
```

You also need Rust, Swift, and a Cohere MLX model bundle containing:

- `model.safetensors`
- `config.json`
- `preprocessor_config.json`
- `vocab.json`

The Rust dependencies are pinned in `Cargo.lock`. The MLX executable is built
from the current `asr-api` checkout because model weights are not stored in this
repository:

```bash
swift build -c release --package-path ../asr-api/apple
```

The runner automatically finds a sibling checkout at
`../asr-api/apple/.build/release/asr-mlx-transcribe`. For another layout, set
`ASR_MLX_TRANSCRIBE_BIN` explicitly. It also verifies that `mlx.metallib` is
beside the executable and installs the copy produced by SwiftPM when needed.

## Start the stack

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
ASR_MLX_TRANSCRIBE_BIN=../asr-api/apple/.build/release/asr-mlx-transcribe \
cargo run --locked --release
```

The process exits if either service fails. Stop both services with `Ctrl-C`.

Check readiness:

```bash
curl -fsS http://127.0.0.1:8444/healthz
curl -kfsS https://127.0.0.1:8443/healthz
curl -kfsS https://127.0.0.1:8443/ha/available
```

Submit a local recording:

```bash
curl --http2 -k -fsS \
  -H 'Content-Type: audio/webm' \
  --data-binary @recording.webm \
  'https://127.0.0.1:8443/v1/listen?utterances=true&paragraphs=true&timestamps=true'
```

The endpoint also accepts streaming HTTP request bodies and WebSockets. WAV,
MP3, FLAC, AAC, Ogg/Opus, and WebM/Opus are decoded by SoundKit; WebM/Opus is
preferred for YouTube research because it can produce PCM before end-of-file.

## Build a channel manifest

Channel discovery is metadata-only. It does not download media:

```bash
scripts/youtube-channel-manifest.sh \
  'https://www.youtube.com/@CHANNEL/videos' \
  target/research/channel-manifest.json
```

The normalized manifest contains channel metadata plus each video ID, title,
URL, and known duration. Files under `target/` are intentionally ignored by
Git.

## Run a research sweep

The opt-in integration runner sends each source directly from `av-ingest` into
SoundKit and ASR without an intermediate media file:

```bash
MEDIA_RESEARCH_STACK_BENCH=1 \
MEDIA_RESEARCH_STACK_URLS_FILE=target/research/channel-manifest.json \
MEDIA_RESEARCH_STACK_REPORT=target/research/report.jsonl \
MEDIA_RESEARCH_STACK_PROGRESS=target/research/progress.ndjson \
MEDIA_RESEARCH_STACK_RESUME=1 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
ASR_MLX_TRANSCRIBE_BIN=../asr-api/apple/.build/release/asr-mlx-transcribe \
MACOSX_DEPLOYMENT_TARGET=14.0 \
cargo test --locked --release --test mastering_videos -- --nocapture
```

`MEDIA_RESEARCH_STACK_URLS` can be used instead of a file for a comma- or
whitespace-separated URL list. The older `MEDIA_RESEARCH_STACK_MASTERING_*`
names remain supported as compatibility aliases.

### Research outputs

`report.jsonl` contains one row per completed source:

- source URL and selected resolver/format metadata;
- media and wall-clock duration;
- observed RTFx; and
- transcript character and word counts.

`progress.ndjson` records response status, timestamps, and transcript sizes. It
redacts transcript and word text by default. This is the right default for
third-party competitor research: keep manifests, measurements, tags, term
counts, and summaries as durable artifacts rather than creating a substitute
archive of the source material.

For recordings you own or are authorized to reproduce, set
`MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS=1` to retain full ASR events. Set
`MEDIA_RESEARCH_STACK_LOG_TRANSCRIPT_PREVIEWS=1` only when transcript text is
appropriate in local logs.

`MEDIA_RESEARCH_STACK_RESUME=1` reads successful source URLs from an existing
report and skips them, so a long sweep can be restarted safely.

## Runtime tuning

The defaults are designed for one MLX worker on a 16 GB Mac:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `CHUNK_SECONDS` | `30` | ASR window length |
| `OVERLAP_SECONDS` | `2` | Context shared by adjacent windows |
| `UPLOAD_RESPONSE_NUM_STREAMS` | `2` | Concurrent cache streams |
| `UPLOAD_RESPONSE_RING_BYTES` | `67108864` | Ring capacity per stream and lane |
| `UPLOAD_RESPONSE_MAX_INFLIGHT` | `1` | Requests processed by the MLX worker |
| `UPLOAD_RESPONSE_TIMEOUT_MS` | `300000` | Allows for a cold MLX compile/load |
| `ASR_COHERE_MAX_NEW_TOKENS` | `128` | Per-window generation cap, tuned for local MLX |

Increasing the ring or stream count multiplies memory across request, decoded,
and response lanes. Add capacity only after measuring a workload. A second MLX
worker is not recommended on a 16 GB Apple Silicon host because each worker
loads a separate model copy.

YouTube authentication, cookies, visitor data, and PO-token settings use the
standard `AV_INGEST_PROXY_YTDLP_*` and `AV_INGEST_PROXY_YOUTUBE_*` environment
variables documented by `av-ingest`.

## Development checks

```bash
cargo fmt --check
MACOSX_DEPLOYMENT_TARGET=14.0 cargo test --locked --all-targets
bash -n scripts/youtube-channel-manifest.sh
```

The long public-media integration run stays disabled unless
`MEDIA_RESEARCH_STACK_BENCH=1` is set.
