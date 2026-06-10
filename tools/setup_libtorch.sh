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
VARIANT_MARKER="${TARGET_DIR}/.nmc-libtorch-variant"
STAGING_DIR="${ARCHIVE_DIR}/extract-libtorch-${TORCH_VERSION}-${LIBTORCH_VARIANT}"

mkdir -p "${ARCHIVE_DIR}" "${REPO_ROOT}/third_party"

if [[ -f "${TARGET_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  INSTALLED_VARIANT=""
  if [[ -f "${VARIANT_MARKER}" ]]; then
    INSTALLED_VARIANT="$(<"${VARIANT_MARKER}")"
  elif [[ "${LIBTORCH_VARIANT}" == "cpu" && ! -f "${TARGET_DIR}/lib/libtorch_cuda.so" ]]; then
    INSTALLED_VARIANT="cpu"
  fi

  if [[ "${INSTALLED_VARIANT}" == "${LIBTORCH_VARIANT}" ]]; then
    echo "[setup_libtorch] Reusing ${LIBTORCH_VARIANT} LibTorch at ${TARGET_DIR}"
    exit 0
  fi

  echo "[setup_libtorch] Replacing existing LibTorch variant '${INSTALLED_VARIANT:-unknown}' with '${LIBTORCH_VARIANT}'"
fi

echo "[setup_libtorch] Downloading ${URL}"
curl -fL --retry 5 --retry-delay 2 "${URL}" -o "${ZIP_PATH}"

echo "[setup_libtorch] Extracting ${LIBTORCH_VARIANT} archive"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"
unzip -q -o "${ZIP_PATH}" -d "${STAGING_DIR}"

if [[ ! -f "${STAGING_DIR}/libtorch/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "[setup_libtorch] Extracted archive does not contain the expected LibTorch layout"
  exit 1
fi

if [[ -L "${TARGET_DIR}" ]]; then
  RESOLVED_TARGET="$(readlink -m "${TARGET_DIR}")"
  if [[ -z "${RESOLVED_TARGET}" || "${RESOLVED_TARGET}" == "/" ]]; then
    echo "[setup_libtorch] Refusing unsafe resolved target: '${RESOLVED_TARGET}'"
    exit 1
  fi
  rm -rf "${RESOLVED_TARGET}"
  mkdir -p "$(dirname "${RESOLVED_TARGET}")"
  mv "${STAGING_DIR}/libtorch" "${RESOLVED_TARGET}"
else
  rm -rf "${TARGET_DIR}"
  mv "${STAGING_DIR}/libtorch" "${TARGET_DIR}"
fi
rm -rf "${STAGING_DIR}"
printf '%s\n' "${LIBTORCH_VARIANT}" > "${VARIANT_MARKER}"

if [[ ! -f "${TARGET_DIR}/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "[setup_libtorch] TorchConfig.cmake not found after extraction (${TARGET_DIR})"
  exit 1
fi

echo "[setup_libtorch] Ready at ${TARGET_DIR}"
