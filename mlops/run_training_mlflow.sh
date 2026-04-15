#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${ROOT_DIR}/mlops/train_with_mlflow.py" \
  --tracking-uri "${MLFLOW_TRACKING_URI:-http://localhost:5000}" \
  --experiment "${MLFLOW_EXPERIMENT:-orbital_ppo}" \
  --run-id "${RUN_ID:-mlflow_train_$(date +%s)}" \
  --seed "${SEED:-7}" \
  --updates "${UPDATES:-30}" \
  --num-envs "${NUM_ENVS:-16}" \
  --env "${ENVIRONMENT:-point_mass}" \
  --binary "${NMC_BINARY:-${ROOT_DIR}/build/nmc}" \
  --config-json "${ROOT_DIR}/config/train.default.json" \
  --export-onnx
