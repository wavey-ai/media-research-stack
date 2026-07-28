#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  COMMAND=""
else
  COMMAND="${1:-}"
fi
if [[ -n "$COMMAND" ]]; then
  shift
fi

TOKEN_FILE="${LINODE_TOKEN_FILE:-}"
STATE_FILE="${LINODE_BENCH_STATE_FILE:-target/linode/instance.json}"
SSH_PUBLIC_KEY="${LINODE_BENCH_SSH_PUBLIC_KEY:-}"
INSTANCE_ID="${LINODE_BENCH_INSTANCE_ID:-}"
LABEL="${LINODE_BENCH_LABEL:-asr-mpx-rtx4000a-medium-us-sea}"
REGION="${LINODE_BENCH_REGION:-us-sea}"
TYPE="${LINODE_BENCH_TYPE:-g2-gpu-rtx4000a1-m}"
IMAGE="${LINODE_BENCH_IMAGE:-linode/ubuntu24.04}"
CONFIRM_DELETE="false"

usage() {
  cat <<'EOF'
Usage:
  linode-asr-benchmark-instance.sh create \
    --token-file <path> \
    --ssh-public-key <path> \
    [--type <plan>] \
    [--region <region>]
  linode-asr-benchmark-instance.sh status --token-file <path> [--instance-id <id>]
  linode-asr-benchmark-instance.sh delete --token-file <path> [--instance-id <id>] --confirm-delete

The script stores non-secret instance metadata in target/linode/instance.json.
It never stores or prints the Linode token or generated root password.

The default plan is g2-gpu-rtx4000a1-m.
This plan provides 32 GiB of host memory for the ONNX export.
EOF
}

if [[ -z "$COMMAND" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token-file)
      TOKEN_FILE="${2:-}"
      shift 2
      ;;
    --state-file)
      STATE_FILE="${2:-}"
      shift 2
      ;;
    --ssh-public-key)
      SSH_PUBLIC_KEY="${2:-}"
      shift 2
      ;;
    --instance-id)
      INSTANCE_ID="${2:-}"
      shift 2
      ;;
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --region)
      REGION="${2:-}"
      shift 2
      ;;
    --type)
      TYPE="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --confirm-delete)
      CONFIRM_DELETE="true"
      shift
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

if [[ ! "$COMMAND" =~ ^(create|status|delete)$ ]]; then
  usage >&2
  exit 2
fi
if [[ -z "$TOKEN_FILE" || ! -f "$TOKEN_FILE" ]]; then
  echo "--token-file must identify a readable token file" >&2
  exit 1
fi

LINODE_TOKEN="$(tr -d '\r\n' <"$TOKEN_FILE")"
if [[ -z "$LINODE_TOKEN" ]]; then
  echo "the token file is empty" >&2
  exit 1
fi

api() {
  local method="$1"
  local endpoint="$2"
  local data="${3:-}"
  if [[ -n "$data" ]]; then
    curl -fsS \
      -X "$method" \
      -H "Authorization: Bearer ${LINODE_TOKEN}" \
      -H "Content-Type: application/json" \
      --data "$data" \
      "https://api.linode.com/v4${endpoint}"
  else
    curl -fsS \
      -X "$method" \
      -H "Authorization: Bearer ${LINODE_TOKEN}" \
      "https://api.linode.com/v4${endpoint}"
  fi
}

load_instance_id() {
  if [[ -n "$INSTANCE_ID" ]]; then
    return
  fi
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "set --instance-id or provide the state file" >&2
    exit 1
  fi
  INSTANCE_ID="$(jq -er '.id' "$STATE_FILE")"
}

case "$COMMAND" in
  create)
    if [[ -z "$SSH_PUBLIC_KEY" || ! -f "$SSH_PUBLIC_KEY" ]]; then
      echo "create requires --ssh-public-key" >&2
      exit 1
    fi
    ROOT_PASSWORD="$(openssl rand -base64 48 | tr -d '\r\n')"
    PAYLOAD="$(jq -n \
      --arg label "$LABEL" \
      --arg region "$REGION" \
      --arg type "$TYPE" \
      --arg image "$IMAGE" \
      --arg root_pass "$ROOT_PASSWORD" \
      --arg authorized_key "$(tr -d '\r\n' <"$SSH_PUBLIC_KEY")" \
      '{
        label: $label,
        region: $region,
        type: $type,
        image: $image,
        root_pass: $root_pass,
        authorized_keys: [$authorized_key]
      }')"
    RESPONSE="$(api POST /linode/instances "$PAYLOAD")"
    mkdir -p "$(dirname "$STATE_FILE")"
    umask 077
    jq '{
      id,
      label,
      region,
      type,
      status,
      ipv4,
      created
    }' <<<"$RESPONSE" >"$STATE_FILE"
    jq '{id, label, region, type, status, ipv4}' "$STATE_FILE"
    ;;
  status)
    load_instance_id
    api GET "/linode/instances/${INSTANCE_ID}" |
      jq '{id, label, region, type, status, ipv4, specs}'
    ;;
  delete)
    load_instance_id
    if [[ "$CONFIRM_DELETE" != "true" ]]; then
      echo "delete requires --confirm-delete" >&2
      exit 1
    fi
    api DELETE "/linode/instances/${INSTANCE_ID}" >/dev/null
    for _ in {1..30}; do
      status_code="$(curl -sS -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer ${LINODE_TOKEN}" \
        "https://api.linode.com/v4/linode/instances/${INSTANCE_ID}")"
      if [[ "$status_code" == "404" ]]; then
        echo "deleted Linode ${INSTANCE_ID}; the API returned 404"
        exit 0
      fi
      if [[ "$status_code" != "200" ]]; then
        echo "unexpected Linode status response: HTTP ${status_code}" >&2
        exit 1
      fi
      sleep 2
    done
    echo "the Linode API still returns the deleted instance" >&2
    exit 1
    ;;
esac
