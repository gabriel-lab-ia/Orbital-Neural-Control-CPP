#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts/mlflow artifacts/mlflow-artifacts

docker compose up --build -d mlflow backend frontend
sleep 4
docker compose run --rm training

echo "Demo stack online:"
echo "  Frontend: http://localhost:3000"
echo "  Backend : http://localhost:8080/health"
echo "  MLflow  : http://localhost:5000"
