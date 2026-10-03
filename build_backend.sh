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

case "$BACKEND" in
    cuda) ./buildcuda.sh ;;
    vulkan) ./buildvulkan.sh ;;
    *) ./buildcpu.sh ;;
esac

BIN=$(find build -name "omnivoice-tts" -type f | head -1)
echo "Built: $SRC_DIR/$BIN"
