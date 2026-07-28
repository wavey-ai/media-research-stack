#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${ASR_MODEL_DIR:-../asr-api/models/cohere-transcribe-03-2026}"
MLX_RUNTIME="${ASR_MLX_TRANSCRIBE_BIN:-../asr-api/apple/.build/release/asr-mlx-transcribe}"
DATASET_DIR="${MEDIA_RESEARCH_STACK_BENCHMARK_DATASET:-target/audiomovers/benchmark-10}"
RESULTS_DIR="${MEDIA_RESEARCH_STACK_BENCHMARK_RESULTS:-target/audiomovers/benchmark-10/runs-local}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  run-local-asr-baseline.sh

Run the fixed one-session MLX baseline on the ten-source data set.
Use the ASR_MODEL_DIR and ASR_MLX_TRANSCRIBE_BIN variables to set local paths.
EOF
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "this script does not accept arguments" >&2
  exit 2
fi

exec python3 scripts/run-asr-benchmark-matrix.py \
  --execution-provider mlx \
  --matrix 1:1:1 \
  --model-dir "$MODEL_DIR" \
  --mlx-runtime "$MLX_RUNTIME" \
  --dataset-dir "$DATASET_DIR" \
  --results-dir "$RESULTS_DIR" \
  --cargo-config 'patch."https://github.com/wavey-ai/av-ingest.git".av-ingest-proxy.path="../av-ingest/crates/proxy"' \
  --cargo-config 'patch."https://github.com/wavey-ai/asr-api.git".asr-api.path="../asr-api"'
