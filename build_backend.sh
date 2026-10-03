#!/bin/bash
# Build the omnivoice.cpp GGUF backend (omnivoice-tts binary).
# Usage: ./build_backend.sh [--cpu|--cuda|--vulkan]
# Default: auto (CUDA if nvcc exists, else CPU).
set -eu

MODE="${1:-auto}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/omnivoice.cpp"

if [ ! -d "$SRC_DIR" ]; then
    git clone --recurse-submodules \
        https://github.com/ServeurpersoCom/omnivoice.cpp.git "$SRC_DIR"
fi

cd "$SRC_DIR"
git pull --recurse-submodules 2>/dev/null || true

pick_backend() {
    case "$MODE" in
        --cuda) echo "cuda" ;;
        --cpu) echo "cpu" ;;
        --vulkan) echo "vulkan" ;;
        *) command -v nvcc >/dev/null 2>&1 && echo "cuda" || echo "cpu" ;;
    esac
}

BACKEND=$(pick_backend)
echo "Backend: $BACKEND (override: $0 --cpu|--cuda|--vulkan)"

# nvcc lags behind new GCC/glibc headers (e.g. CUDA 13.1 + GCC 15 fails
# on mathcalls.h rsqrt). Retry with an older host compiler when present.
build_cuda() {
    if ./buildcuda.sh; then
        return 0
    fi
    for cxx in g++-13 g++-12 g++-14 gcc-13 gcc-12; do
        if command -v "$cxx" >/dev/null 2>&1; then
            echo "CUDA build failed, retrying with NVCC_CCBIN=$cxx ..."
            if NVCC_CCBIN="$cxx" ./buildcuda.sh; then
                return 0
            fi
        fi
    done
    echo "ERROR: CUDA build failed. Options:" >&2
    echo "  sudo apt install -y g++-13   # then re-run $0" >&2
    echo "  $0 --cpu                    # CPU fallback, always works" >&2
    return 1
}

case "$BACKEND" in
    cuda) build_cuda ;;
    vulkan) ./buildvulkan.sh ;;
    *) ./buildcpu.sh ;;
esac

BIN=$(find build -name "omnivoice-tts" -type f | head -1)
echo "Built: $SRC_DIR/$BIN"
