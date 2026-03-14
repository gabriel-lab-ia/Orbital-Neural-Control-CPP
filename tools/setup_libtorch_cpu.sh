#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$PROJECT_ROOT/lib}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.2.2}"
ARCHIVE_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip}"
ARCHIVE_PATH="$TARGET_DIR/libtorch-${LIBTORCH_VERSION}.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading LibTorch ${LIBTORCH_VERSION}..."
wget -O "$ARCHIVE_PATH" "$ARCHIVE_URL"

echo "Extracting LibTorch..."
rm -rf "$TARGET_DIR/libtorch"
unzip -q "$ARCHIVE_PATH" -d "$TARGET_DIR"
rm -f "$ARCHIVE_PATH"

echo "LibTorch installed at: $TARGET_DIR/libtorch"
