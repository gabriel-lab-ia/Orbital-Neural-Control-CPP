#!/usr/bin/env python3
"""MLflow-integrated orbital PPO training orchestrator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO training with MLflow tracking")
    parser.add_argument("--run-id", default=f"mlflow_train_{int(time.time())}")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env", default="point_mass")
    parser.add_argument("--binary", default="./build/nmc")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment", default="orbital_ppo")
    parser.add_argument("--config-json", default="config/train.default.json")
    parser.add_argument("--perturbation-level", default="nominal")
    parser.add_argument("--reward-shaping", default="dense")
    parser.add_argument("--mission-profile", default="rendezvous_3dof")
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--model-name", default="orbital-ppo-policy")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output", default="")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> int:
    cmd = [
        args.binary,
        "train",
        "--run-id",
        args.run_id,
        "--seed",
        str(args.seed),
        "--updates",
        str(args.updates),
        "--num-envs",
        str(args.num_envs),
        "--env",
        args.env,
    ]
    print("Launching training:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def read_training_metrics(csv_path: Path) -> dict[str, float]:
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    final = rows[-1]
    metrics: dict[str, float] = {}
    for key in (
        "avg_episode_return",
        "success_rate",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "samples_per_second",
        "inference_latency_ms",
    ):
        if key in final and final[key] != "":
            metrics[f"final_{key}"] = float(final[key])
    return metrics


def parse_checkpoint_meta(meta_path: Path) -> dict[str, Any]:
    content: dict[str, Any] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"observation_dim", "action_dim", "hidden_dim", "seed"}:
            content[key] = int(value)
        else:
            content[key] = value
    return content


def maybe_export_onnx(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if not args.export_onnx:
        return None

    meta_path = run_dir / "checkpoints" / "policy_last.meta"
    if not meta_path.exists():
        print(f"[warn] checkpoint meta not found at {meta_path}, skipping ONNX export")
        return None

    meta = parse_checkpoint_meta(meta_path)
    output_path = Path(args.onnx_output) if args.onnx_output else run_dir / "models" / "policy.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        "mlops/export_policy_onnx.py",
        "--observation-dim",
        str(meta["observation_dim"]),
        "--action-dim",
        str(meta["action_dim"]),
        "--hidden-dim",
        str(meta["hidden_dim"]),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print("[warn] ONNX export failed")
        return None

    return output_path


def log_reproducibility_metadata(args: argparse.Namespace) -> None:
    config_path = Path(args.config_json)
    if config_path.exists():
        config_content = config_path.read_text(encoding="utf-8")
        mlflow.log_text(config_content, artifact_file="config/train_config.json")
        mlflow.log_param("config_sha256", hashlib.sha256(config_content.encode("utf-8")).hexdigest())

    git_commit = os.environ.get("GITHUB_SHA") or subprocess.getoutput("git rev-parse --short HEAD")
    mlflow.set_tag("git_commit", git_commit.strip())
    mlflow.set_tag("cpu_first", "true")
    mlflow.set_tag("cuda_enabled", "false")


def main() -> int:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_id) as run:
        mlflow.set_tags(
            {
                "project": "orbital-neural-control-cpp",
                "orbital_dynamics": args.env,
                "perturbation_level": args.perturbation_level,
                "reward_shaping": args.reward_shaping,
                "mission_profile": args.mission_profile,
                "runtime_mode": "training",
            }
        )

        mlflow.log_params(
            {
                "seed": args.seed,
                "updates": args.updates,
                "num_envs": args.num_envs,
                "environment": args.env,
                "binary": args.binary,
            }
        )

        log_reproducibility_metadata(args)

        return_code = run_training(args)
        mlflow.log_metric("train_return_code", float(return_code))

        run_dir = Path("artifacts") / "runs" / args.run_id
        metrics_csv = run_dir / "training_metrics.csv"
        summary_json = run_dir / "training_summary.json"
        manifest_json = run_dir / "manifest.json"
        checkpoint_path = run_dir / "checkpoints" / "policy_last.pt"
        checkpoint_meta_path = run_dir / "checkpoints" / "policy_last.meta"

        if metrics_csv.exists():
            for key, value in read_training_metrics(metrics_csv).items():
                mlflow.log_metric(key, value)
            mlflow.log_artifact(str(metrics_csv), artifact_path="training")

        if summary_json.exists():
            mlflow.log_artifact(str(summary_json), artifact_path="reports")
        if manifest_json.exists():
            mlflow.log_artifact(str(manifest_json), artifact_path="reports")

        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")
            mlflow.set_tag("model_artifact", "models/policy_last.pt")
        if checkpoint_meta_path.exists():
            mlflow.log_artifact(str(checkpoint_meta_path), artifact_path="models")

        onnx_path = maybe_export_onnx(args, run_dir)
        if onnx_path and onnx_path.exists():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")

        status = "completed" if return_code == 0 else "failed"
        mlflow.log_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": status,
                    "run_dir": str(run_dir),
                },
                indent=2,
            ),
            artifact_file="run_manifest.json",
        )

        if return_code == 0 and args.register_model and onnx_path and onnx_path.exists():
            cmd = [
                "python3",
                "mlops/register_model.py",
                "--run-id",
                run.info.run_id,
                "--model-name",
                args.model_name,
                "--tracking-uri",
                args.tracking_uri,
            ]
            subprocess.run(cmd, check=False)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
