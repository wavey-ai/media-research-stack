#!/usr/bin/env bash
set -euo pipefail

HOST=""
IDENTITY=""
REMOTE_ROOT="/opt/asr-bench"
DATASET_DIR="target/audiomovers/benchmark-10"

usage() {
  cat <<'EOF'
Usage:
  sync-asr-benchmark-assets.sh \
    --host root@example \
    --identity target/linode/ssh_key

The script clones each source repository from its main branch.
It transfers only the 40 MiB benchmark data set from the local host.
The Linode must download and export the model separately.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --identity)
      IDENTITY="${2:-}"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="${2:-}"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$HOST" || -z "$IDENTITY" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$IDENTITY" || ! -d "$DATASET_DIR" ]]; then
  echo "an input path does not exist" >&2
  exit 1
fi

SSH_COMMAND="ssh -i $IDENTITY -o BatchMode=yes"

"${SSH_COMMAND%% *}" -i "$IDENTITY" -o BatchMode=yes "$HOST" \
  "mkdir -p '$REMOTE_ROOT'"

remote_checkout() {
  local repository_url="$1"
  local directory_name="$2"
  local remote_command
  printf -v remote_command \
    'set -euo pipefail; root=%q; url=%q; name=%q; destination=\"$root/$name\"; if [[ -d \"$destination/.git\" ]]; then git -C \"$destination\" fetch origin main; git -C \"$destination\" checkout main; git -C \"$destination\" pull --ff-only origin main; else git clone --branch main --depth 1 \"$url\" \"$destination\"; fi' \
    "$REMOTE_ROOT" \
    "$repository_url" \
    "$directory_name"
  ssh -i "$IDENTITY" -o BatchMode=yes "$HOST" "$remote_command"
}

remote_checkout \
  https://github.com/wavey-ai/media-research-stack.git \
  media-research-stack
remote_checkout \
  https://github.com/wavey-ai/av-ingest.git \
  av-ingest
remote_checkout \
  https://github.com/wavey-ai/asr-api.git \
  asr-api
remote_checkout \
  https://github.com/wavey-ai/asr-onnx.git \
  asr-onnx

ssh -i "$IDENTITY" -o BatchMode=yes "$HOST" \
  "mkdir -p '$REMOTE_ROOT/media-research-stack/target/audiomovers/benchmark-10'"

rsync -a --delete --exclude '*.part' \
  -e "$SSH_COMMAND" \
  "$DATASET_DIR/" \
  "$HOST:$REMOTE_ROOT/media-research-stack/target/audiomovers/benchmark-10/"

echo "cloned source and synced the benchmark data set to $HOST:$REMOTE_ROOT"
