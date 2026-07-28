# media-research-stack

[`media-research-stack`](https://github.com/wavey-ai/media-research-stack) is
the supported local macOS runner for turning public media into research
transcripts, progress logs, and ASR timing data.

It can:

- resolve public media with [`av-ingest`](https://github.com/wavey-ai/av-ingest)
- decode compressed audio with [SoundKit](https://github.com/wavey-ai/soundkit)
  as bytes arrive
- run Cohere Transcribe through the local MLX runtime on Apple Silicon
- serve a Deepgram-compatible `/v1/listen` endpoint for clients that need the
  same ASR surface.

For channel research, [`av-ingest`](https://github.com/wavey-ai/av-ingest)
selects a source audio format. [SoundKit](https://github.com/wavey-ai/soundkit)
decodes formats such as WebM/Opus to mono 16 kHz PCM as bytes arrive.
[`asr-api`](https://github.com/wavey-ai/asr-api) transcribes 30-second windows
with a two-second overlap.

## Platform and requirements

The current product target is Apple Silicon macOS 14 or newer.

Install the command-line dependencies:

```bash
brew install libvpx pkg-config yt-dlp jq
```

Install the Python helpers used for model download/setup when needed:

```bash
python3 -m pip install -U huggingface_hub sentencepiece
```

You also need Rust, Swift, and the Cohere Transcribe MLX model bundle. Use a
Hugging Face login that has access to Cohere's gated model, then download and
prepare the bundle in the sibling
[`asr-api`](https://github.com/wavey-ai/asr-api) checkout:

```bash
../asr-api/scripts/setup-cohere-mlx-model.sh --login
```

The setup script downloads
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

After setup, the local model directory should be:

```text
../asr-api/models/cohere-transcribe-03-2026
```

and contain:

- `model.safetensors`
- `config.json`
- `preprocessor_config.json`
- `tokenizer.model`
- `vocab.json`

The Rust dependencies are pinned in `Cargo.lock`. Build the MLX executable from
the current [`asr-api`](https://github.com/wavey-ai/asr-api) checkout:

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
MP3, FLAC, AAC, Ogg/Opus, and WebM/Opus are decoded by SoundKit. WebM/Opus is
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

Use separate cache and ASR processes for channel sweeps.
The cache process can download multiple low-bitrate sources at the same time.
It publishes each cache entry atomically.

Run the cache process in one terminal:

```bash
MEDIA_RESEARCH_STACK_URLS_FILE=target/research/channel-manifest.json \
MEDIA_RESEARCH_STACK_MEDIA_DIR=target/research/media \
MEDIA_RESEARCH_STACK_CACHE_REPORT=target/research/cache-report.jsonl \
MEDIA_RESEARCH_STACK_CACHE_CONCURRENCY=4 \
AV_INGEST_PROXY_YTDLP_AUDIO_FORMAT='bestaudio[ext=webm][abr<=64]/bestaudio[abr<=64]/worstaudio[ext=webm]/worstaudio' \
cargo run --locked --release --bin cache-research-media
```

Run the ASR process in another terminal:

```bash
MEDIA_RESEARCH_STACK_BENCH=1 \
MEDIA_RESEARCH_STACK_URLS_FILE=target/research/channel-manifest.json \
MEDIA_RESEARCH_STACK_REPORT=target/research/report.jsonl \
MEDIA_RESEARCH_STACK_PROGRESS=target/research/progress.ndjson \
MEDIA_RESEARCH_STACK_MEDIA_DIR=target/research/media \
MEDIA_RESEARCH_STACK_TRANSCRIPTS_DIR=target/research/transcripts \
MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS=1 \
MEDIA_RESEARCH_STACK_RESUME=1 \
MEDIA_RESEARCH_STACK_REQUIRE_CACHE=1 \
MEDIA_RESEARCH_STACK_CACHE_WAIT_SECS=21600 \
ASR_MODEL_DIR=../asr-api/models/cohere-transcribe-03-2026 \
ASR_MLX_TRANSCRIBE_BIN=../asr-api/apple/.build/release/asr-mlx-transcribe \
MACOSX_DEPLOYMENT_TARGET=14.0 \
cargo test --locked --release --test mastering_videos -- --nocapture
```

The ASR process reads each cache file without real-time pacing.
It waits for atomic cache publication when the next source is not ready.
It never downloads media when `MEDIA_RESEARCH_STACK_REQUIRE_CACHE=1`.
Set `MEDIA_RESEARCH_STACK_CACHE_MIME_TYPE=audio/webm` in both processes.
The cache process then replaces incompatible entries before ASR reads them.

`MEDIA_RESEARCH_STACK_URLS` can be used instead of a file for a comma- or
whitespace-separated URL list. The older `MEDIA_RESEARCH_STACK_MASTERING_*`
names remain supported as compatibility aliases.

### Research outputs

`report.jsonl` contains one row per completed source:

- source URL and selected resolver/format metadata
- media and wall-clock duration
- total RTFx and ASR-only RTFx
- compressed-input throughput
- transcript character, word, and word-rate values
- rolling ASR and pipeline throughput.

`cache-report.jsonl` contains cache duration, selected format, bytes, and media
RTFx. A high media RTFx shows that download completed faster than real time.

`progress.ndjson` records response status, timestamps, and transcript sizes.
Transcript and word text are written only when transcript storage is enabled.
Set `MEDIA_RESEARCH_STACK_LOG_SEGMENT_METRICS=1` to print each result event.
For public-media research, keep manifests, measurements, tags, term counts, and
summaries as the durable artifacts. Retain full transcripts only when this is
appropriate for your sources.

Run the metrics watcher in a separate process:

```sh
python3 scripts/watch-asr-throughput.py \
  --progress target/research/progress.ndjson \
  --report target/research/report.jsonl \
  --output target/research/asr-throughput.jsonl
```

The watcher records interval RTFx during each source. It also copies completed
source and rolling run metrics. The watcher only reads ASR output files.

For recordings you own or are authorized to reproduce, set
`MEDIA_RESEARCH_STACK_STORE_TRANSCRIPTS=1` to retain full ASR events. Set
`MEDIA_RESEARCH_STACK_TRANSCRIPTS_DIR` to write one UTF-8 transcript for each
completed source. Transcript files are published atomically, and their paths are
recorded in `report.jsonl`. Set
`MEDIA_RESEARCH_STACK_LOG_TRANSCRIPT_PREVIEWS=1` only when transcript text is
appropriate in local logs.

Set `MEDIA_RESEARCH_STACK_MEDIA_DIR` for long or repeatable sweeps.
[`av-ingest`](https://github.com/wavey-ai/av-ingest) downloads the selected
compressed audio once, publishes it atomically into the cache, and subsequent
runs read it locally through [SoundKit](https://github.com/wavey-ai/soundkit).
The cache preserves the selected source encoding for repeatable ASR runs.

`MEDIA_RESEARCH_STACK_RESUME=1` reads successful source URLs from an existing
report and skips them, so a long sweep can be restarted safely.

Set `MEDIA_RESEARCH_STACK_CONTINUE_ON_ERROR=1` for channel-sized jobs. A failed
source is recorded in the report and the remaining sources continue. The run
still exits unsuccessfully after the sweep when a source fails. A supervisor can
restart it with resume enabled and retry only incomplete work.

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
| `MEDIA_RESEARCH_STACK_CACHE_CONCURRENCY` | `4` | Concurrent cache downloads |
| `MEDIA_RESEARCH_STACK_CACHE_WAIT_SECS` | `21600` | ASR wait for the next cache entry |

Increasing the ring or stream count multiplies memory across request, decoded,
and response lanes. Add capacity only after measuring a workload. A second MLX
worker is not recommended on a 16 GB Apple Silicon host because each worker
loads a separate model copy.

YouTube authentication, cookies, visitor data, and PO-token settings use the
standard `AV_INGEST_PROXY_YTDLP_*` and `AV_INGEST_PROXY_YOUTUBE_*` environment
variables documented by [`av-ingest`](https://github.com/wavey-ai/av-ingest).
For example, pass a Netscape-format cookie export without committing it:

```bash
AV_INGEST_PROXY_YTDLP_COOKIES=/absolute/path/to/youtube-cookies.txt \
  cargo test --locked --release --test mastering_videos -- --nocapture
```

Verify the same cookie source with one URL before a channel-sized run. The
research sweep treats YouTube's bot-confirmation response as a systemic
authentication failure and stops immediately instead of attempting every URL.

## Compare Apple and NVIDIA throughput

Use the fixed ten-source data set for architecture comparisons.
The data set contains 4,311 seconds of WebM/Opus audio.

Create the authorized GPU host:

```sh
scripts/linode-asr-benchmark-instance.sh create \
  --token-file ../.linode-token \
  --ssh-public-key target/linode/ssh_key.pub \
  --type g2-gpu-rtx4000a1-m \
  --region us-sea \
  --label asr-mpx-rtx4000a-medium-us-sea-20260728
```

Read the host address from `target/linode/instance.json`.
Then clone the source and copy the benchmark data:

```sh
address="$(jq -r '.ipv4[0]' target/linode/instance.json)"
scripts/sync-asr-benchmark-assets.sh \
  --host "root@${address}" \
  --identity target/linode/ssh_key
```

Install the pinned NVIDIA environment:

```sh
ssh -i target/linode/ssh_key "root@${address}" \
  'bash /opt/asr-bench/media-research-stack/scripts/bootstrap-ubuntu-nvidia-asr.sh --reboot'
```

Wait for the host to restart.
Then export the Cohere model on the host:

```sh
scripts/run-remote-cohere-export.sh \
  --host "root@${address}" \
  --identity target/linode/ssh_key \
  --token-file ../.hf_token
```

Run one CUDA configuration before the TensorRT matrix:

```sh
ssh -i target/linode/ssh_key "root@${address}" \
  'cd /opt/asr-bench/media-research-stack &&
   ~/.cargo/bin/cargo test --locked --release --test mastering_videos --no-run &&
   python3 scripts/run-asr-benchmark-matrix.py \
     --execution-provider cuda \
     --matrix 1:1:1 \
     --model-dir /opt/asr-bench/models/cohere-transcribe-03-2026 \
     --runtime-lib /opt/asr-bench/runtime/onnxruntime-linux-x64-gpu-1.23.2/lib/libonnxruntime.so \
     --results-dir target/audiomovers/benchmark-10/runs-remote'
```

Run the default TensorRT matrix:

```sh
ssh -i target/linode/ssh_key "root@${address}" \
  'cd /opt/asr-bench/media-research-stack &&
   python3 scripts/run-asr-benchmark-matrix.py \
     --execution-provider tensorrt \
     --model-dir /opt/asr-bench/models/cohere-transcribe-03-2026 \
     --runtime-lib /opt/asr-bench/runtime/onnxruntime-linux-x64-gpu-1.23.2/lib/libonnxruntime.so \
     --results-dir target/audiomovers/benchmark-10/runs-remote'
```

Run the matching Apple MLX baseline:

```sh
scripts/run-local-asr-baseline.sh
```

Copy the remote summary files before host deletion.
Use `scripts/compare-asr-benchmarks.py` to select the best stable configuration.
Select by effective RTFx, not ASR service RTFx.

Delete the host after you retrieve all summaries:

```sh
scripts/linode-asr-benchmark-instance.sh delete \
  --token-file ../.linode-token \
  --confirm-delete
```

## Development checks

```bash
cargo fmt --check
MACOSX_DEPLOYMENT_TARGET=14.0 cargo test --locked --all-targets
bash -n scripts/youtube-channel-manifest.sh
```

The long public-media integration run stays disabled unless
`MEDIA_RESEARCH_STACK_BENCH=1` is set.
