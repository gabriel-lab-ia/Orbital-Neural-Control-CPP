#!/usr/bin/env python3
from __future__ import annotations

import argparse

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register ONNX model from MLflow run artifacts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default="orbital-ppo-policy")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--artifact-path", default="onnx")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"runs:/{args.run_id}/{args.artifact_path}"

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    print(f"Registered model version: name={result.name} version={result.version} source={model_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
