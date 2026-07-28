#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.23.2}"
NVIDIA_DRIVER_SERIES="${NVIDIA_DRIVER_SERIES:-570}"
CUDA_SERIES="${CUDA_SERIES:-12-8}"
TENSORRT_VERSION="${TENSORRT_VERSION:-10.9.0.34-1+cuda12.8}"
RUNTIME_ROOT="${ASR_BENCH_RUNTIME_ROOT:-/opt/asr-bench/runtime}"
PACKAGE_CACHE="${ASR_BENCH_PACKAGE_CACHE:-/var/cache/asr-benchmark}"
REBOOT_AFTER_INSTALL="false"

usage() {
  cat <<'EOF'
Usage:
  sudo scripts/bootstrap-ubuntu-nvidia-asr.sh [--reboot]

Install the pinned NVIDIA and ONNX Runtime packages for the ASR benchmark.

Environment:
  ORT_VERSION               Defaults to 1.23.2.
  NVIDIA_DRIVER_SERIES      Defaults to 570.
  CUDA_SERIES               Defaults to 12-8.
  TENSORRT_VERSION          Defaults to 10.9.0.34-1+cuda12.8.
  ASR_BENCH_RUNTIME_ROOT    Defaults to /opt/asr-bench/runtime.
  ASR_BENCH_PACKAGE_CACHE   Defaults to /var/cache/asr-benchmark.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reboot)
      REBOOT_AFTER_INSTALL="true"
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

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run this script as root" >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "this script requires Ubuntu 24.04" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
mkdir -p "$PACKAGE_CACHE" "$RUNTIME_ROOT"
apt-get update
apt-get install -y \
  build-essential \
  ca-certificates \
  clang \
  cmake \
  curl \
  git \
  jq \
  libclang-dev \
  libffi-dev \
  libopencore-amrnb-dev \
  libssl-dev \
  linux-headers-"$(uname -r)" \
  pkg-config \
  python3-dev \
  python3-venv \
  rsync \
  zstd

if dpkg-query -W -f='${binary:Package} ${db:Status-Status}\n' 'libopus*' \
    2>/dev/null | grep -q ' installed$'; then
  echo "the host contains a native libopus package" >&2
  exit 1
fi

KEYRING_PACKAGE="$PACKAGE_CACHE/cuda-keyring_1.1-1_all.deb"
if [[ ! -f "$KEYRING_PACKAGE" ]]; then
  curl -fL --retry 5 \
    -o "$KEYRING_PACKAGE" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
fi
dpkg -i "$KEYRING_PACKAGE"
apt-get update

apt-get install -y \
  "cuda-drivers-${NVIDIA_DRIVER_SERIES}" \
  "cuda-libraries-${CUDA_SERIES}" \
  "libcudnn9-cuda-12" \
  "tensorrt-libs=${TENSORRT_VERSION}" \
  "libnvinfer10=${TENSORRT_VERSION}" \
  "libnvinfer-lean10=${TENSORRT_VERSION}" \
  "libnvinfer-plugin10=${TENSORRT_VERSION}" \
  "libnvinfer-vc-plugin10=${TENSORRT_VERSION}" \
  "libnvinfer-dispatch10=${TENSORRT_VERSION}" \
  "libnvonnxparsers10=${TENSORRT_VERSION}"

if [[ ! -x /root/.cargo/bin/cargo ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal
fi

ORT_ARCHIVE="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
ORT_DIRECTORY="$RUNTIME_ROOT/onnxruntime-linux-x64-gpu-${ORT_VERSION}"
if [[ ! -f "$ORT_DIRECTORY/lib/libonnxruntime.so" ]]; then
  curl -fL --retry 5 \
    -o "$PACKAGE_CACHE/$ORT_ARCHIVE" \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
  tar -xzf "$PACKAGE_CACHE/$ORT_ARCHIVE" -C "$RUNTIME_ROOT"
fi

ldconfig
echo "ONNX Runtime: $ORT_DIRECTORY/lib/libonnxruntime.so"
echo "Restart the host before the first GPU test."

if [[ "$REBOOT_AFTER_INSTALL" == "true" ]]; then
  systemctl reboot
fi
