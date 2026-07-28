#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODEL="${ASR_COHERE_SOURCE_MODEL:-CohereLabs/cohere-transcribe-03-2026}"
BENCH_ROOT="${ASR_BENCH_ROOT:-/opt/asr-bench}"
ASR_ONNX_ROOT="${ASR_ONNX_ROOT:-$BENCH_ROOT/asr-onnx}"
MODEL_DIR="${ASR_MODEL_DIR:-$BENCH_ROOT/models/cohere-transcribe-03-2026}"
HF_HOME="${HF_HOME:-$BENCH_ROOT/huggingface}"
VENV_DIR="${VENV_DIR:-$ASR_ONNX_ROOT/.venv-export}"

usage() {
  cat <<'EOF'
Usage:
  HF_TOKEN=<token> export-cohere-onnx-on-linux.sh

Download the gated Cohere checkpoint and export its four ONNX graphs.

Environment:
  ASR_COHERE_SOURCE_MODEL  Defaults to CohereLabs/cohere-transcribe-03-2026.
  ASR_BENCH_ROOT           Defaults to /opt/asr-bench.
  ASR_ONNX_ROOT            Defaults to /opt/asr-bench/asr-onnx.
  ASR_MODEL_DIR            Sets the completed model directory.
  HF_HOME                  Sets the Hugging Face download cache.
  VENV_DIR                 Sets the locked export environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "set HF_TOKEN before you run this script" >&2
  exit 1
fi
if [[ ! -f "$ASR_ONNX_ROOT/export/export_cohere_transcribe.py" ]]; then
  echo "the asr-onnx checkout is missing" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 1
fi

declare -a REQUIRED_FILES=(
  config.json
  decoder_cached_step.onnx
  decoder_cached_step.onnx.data
  decoder_last_token.onnx
  decoder_last_token.onnx.data
  decoder_prefill.onnx
  decoder_prefill.onnx.data
  encoder.onnx
  encoder.onnx.data
  export.json
  generation_config.json
  preprocessor_config.json
  processor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer.model
  tokenizer_config.json
)

model_is_complete() {
  local file_name
  [[ -f "$MODEL_DIR/SHA256SUMS" ]] || return 1
  for file_name in "${REQUIRED_FILES[@]}"; do
    [[ -s "$MODEL_DIR/$file_name" ]] || return 1
  done
  jq -e '
    (.dither | type == "number") and
    (.feature_size | type == "number") and
    (.n_fft | type == "number") and
    (.n_window_size | type == "number") and
    (.n_window_stride | type == "number") and
    (.normalize | type == "string") and
    (.padding_value | type == "number") and
    (.sampling_rate | type == "number") and
    (.window | type == "string")
  ' "$MODEL_DIR/preprocessor_config.json" >/dev/null || return 1
  (
    cd "$MODEL_DIR"
    sha256sum --check --quiet SHA256SUMS
  )
}

if model_is_complete; then
  echo "the verified ONNX export already exists at $MODEL_DIR"
  exit 0
fi

mkdir -p "$BENCH_ROOT/models" "$HF_HOME"
STAGED_DIR="$(mktemp -d "$BENCH_ROOT/models/.cohere-onnx-export.XXXXXX")"

cleanup() {
  local exit_code=$?
  trap - EXIT
  if [[ -d "$STAGED_DIR" ]]; then
    rm -rf "$STAGED_DIR"
  fi
  exit "$exit_code"
}
trap cleanup EXIT

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  VENV_DIR="$VENV_DIR" "$ASR_ONNX_ROOT/python/setup-export-env.sh"
fi

export HF_HOME
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$VENV_DIR/bin/python" \
  "$ASR_ONNX_ROOT/export/export_cohere_transcribe.py" \
  --source "$SOURCE_MODEL" \
  --output-dir "$STAGED_DIR" \
  --device cuda \
  --sample-audio-seconds 15 \
  --opset 18

for file_name in "${REQUIRED_FILES[@]}"; do
  if [[ ! -s "$STAGED_DIR/$file_name" ]]; then
    echo "the ONNX export did not create $file_name" >&2
    exit 1
  fi
done

(
  cd "$STAGED_DIR"
  sha256sum "${REQUIRED_FILES[@]}" >SHA256SUMS
  sha256sum --check SHA256SUMS
)

if [[ -e "$MODEL_DIR" ]]; then
  INCOMPLETE_DIR="${MODEL_DIR}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$MODEL_DIR" "$INCOMPLETE_DIR"
  echo "moved the incomplete export to $INCOMPLETE_DIR"
fi
mv "$STAGED_DIR" "$MODEL_DIR"

printf 'exported model bytes: '
du -sb "$MODEL_DIR" | awk '{print $1}'
echo "verified the ONNX export at $MODEL_DIR"
