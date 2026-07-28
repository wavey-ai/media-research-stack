#!/usr/bin/env bash
set -euo pipefail

HOST=""
IDENTITY=""
TOKEN_FILE=""
REMOTE_SCRIPT="/opt/asr-bench/media-research-stack/scripts/export-cohere-onnx-on-linux.sh"

usage() {
  cat <<'EOF'
Usage:
  run-remote-cohere-export.sh \
    --host root@example \
    --identity target/linode/ssh_key \
    --token-file ../.hf_token

The script sends the token through SSH standard input.
It does not add the token to a command argument or remote file.
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
    --token-file)
      TOKEN_FILE="${2:-}"
      shift 2
      ;;
    --remote-script)
      REMOTE_SCRIPT="${2:-}"
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

if [[ -z "$HOST" || -z "$IDENTITY" || -z "$TOKEN_FILE" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$IDENTITY" || ! -s "$TOKEN_FILE" ]]; then
  echo "an identity or token file does not exist" >&2
  exit 1
fi

printf -v remote_command \
  'set -euo pipefail; IFS= read -r HF_TOKEN; export HF_TOKEN; exec bash %q' \
  "$REMOTE_SCRIPT"

{
  tr -d '\r\n' <"$TOKEN_FILE"
  printf '\n'
} | ssh -i "$IDENTITY" -o BatchMode=yes "$HOST" "$remote_command"
