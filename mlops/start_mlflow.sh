#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${ROOT_DIR}/artifacts/mlflow"
mkdir -p "${ROOT_DIR}/artifacts/mlflow-artifacts"

exec mlflow server \
  --backend-store-uri "sqlite:///${ROOT_DIR}/artifacts/mlflow/mlflow.db" \
  --default-artifact-root "${ROOT_DIR}/artifacts/mlflow-artifacts" \
  --host 0.0.0.0 \
  --port 5000
