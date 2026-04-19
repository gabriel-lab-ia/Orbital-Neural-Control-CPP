#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-cpu}"   # cpu | cu121 | cu124
TARGET_DIR="${LIBTORCH_DIR:-${REPO_ROOT}/third_party/libtorch}"
ARCHIVE_DIR="${REPO_ROOT}/third_party/.cache"

case "${LIBTORCH_VARIANT}" in
  cpu|cu121|cu124) ;;
  *)
    echo "[setup_libtorch] Unsupported LIBTORCH_VARIANT='${LIBTORCH_VARIANT}'. Use cpu|cu121|cu124."
    exit 1
    ;;
esac

if [[ "${LIBTORCH_VARIANT}" == "cpu" ]]; then
  BASE_URL="https://download.pytorch.org/libtorch/cpu"
  ARCHIVE_NAME="libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
else
  BASE_URL="https://download.pytorch.org/libtorch/${LIBTORCH_VARIANT}"
  ARCHIVE_NAME="libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${LIBTORCH_VARIANT}.zip"
fi

URL="${BASE_URL}/${ARCHIVE_NAME}"
ZIP_PATH="${ARCHIVE_DIR}/libtorch-${TORCH_VERSION}-${LIBTORCH_VARIANT}.zip"

mkdir -p "${ARCHIVE_DIR}" "${REPO_ROOT}/third_party"

if [[ -f "${TARGET_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "[setup_libtorch] Reusing existing libtorch at ${TARGET_DIR}"
  exit 0
fi

echo "[setup_libtorch] Downloading ${URL}"
curl -fL --retry 5 --retry-delay 2 "${URL}" -o "${ZIP_PATH}"

echo "[setup_libtorch] Extracting to ${REPO_ROOT}/third_party"
rm -rf "${TARGET_DIR}"
unzip -q -o "${ZIP_PATH}" -d "${REPO_ROOT}/third_party"

if [[ ! -f "${TARGET_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "[setup_libtorch] TorchConfig.cmake not found after extraction (${TARGET_DIR})"
  exit 1
fi

echo "[setup_libtorch] Ready at ${TARGET_DIR}"
