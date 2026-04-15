#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$PROJECT_ROOT/lib}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.2.2}"
ARCHIVE_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip}"
ARCHIVE_PATH="$TARGET_DIR/libtorch-${LIBTORCH_VERSION}.zip"
TORCH_CONFIG_PATH="$TARGET_DIR/libtorch/share/cmake/Torch/TorchConfig.cmake"

download_archive() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 5 --retry-delay 2 --retry-connrefused \
            --output "$output" "$url"
        return 0
    fi

    if command -v wget >/dev/null 2>&1; then
        wget --tries=5 --waitretry=2 --retry-connrefused -O "$output" "$url"
        return 0
    fi

    echo "Neither curl nor wget is available to download LibTorch." >&2
    return 1
}

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

if [[ -f "$TORCH_CONFIG_PATH" ]]; then
    echo "LibTorch already installed at: $TARGET_DIR/libtorch"
    exit 0
fi

echo "Downloading LibTorch ${LIBTORCH_VERSION}..."
download_archive "$ARCHIVE_URL" "$ARCHIVE_PATH"

echo "Verifying archive..."
unzip -tq "$ARCHIVE_PATH" >/dev/null

echo "Extracting LibTorch..."
rm -rf "$TARGET_DIR/libtorch"
unzip -q "$ARCHIVE_PATH" -d "$TARGET_DIR"
rm -f "$ARCHIVE_PATH"

if [[ ! -f "$TORCH_CONFIG_PATH" ]]; then
    echo "LibTorch extraction completed, but TorchConfig.cmake was not found." >&2
    exit 1
fi

echo "LibTorch installed at: $TARGET_DIR/libtorch"
