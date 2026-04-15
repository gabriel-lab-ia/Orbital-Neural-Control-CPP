#!/usr/bin/env python3
"""Training entrypoint for orbital PPO experiments.

This script orchestrates the C++ baseline executable while recording
structured run metadata for reproducible experiment management.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orbital PPO training pipeline")
    parser.add_argument("--run-id", default=f"train_py_{int(time.time())}")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env", default="point_mass")
    parser.add_argument("--binary", default="./build/nmc")
    parser.add_argument("--db", default="artifacts/experiments.sqlite")
    return parser.parse_args()


def ensure_tracking_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_invocations (
                run_id TEXT PRIMARY KEY,
                seed INTEGER NOT NULL,
                updates INTEGER NOT NULL,
                num_envs INTEGER NOT NULL,
                environment TEXT NOT NULL,
                launched_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.commit()


def register_invocation(db_path: Path, args: argparse.Namespace, status: str) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO training_invocations
            (run_id, seed, updates, num_envs, environment, launched_at, status)
            VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (args.run_id, args.seed, args.updates, args.num_envs, args.env, status),
        )
        connection.commit()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db)
    ensure_tracking_table(db_path)
    register_invocation(db_path, args, "running")

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

    print(json.dumps({"event": "launch_train", "cmd": cmd}, indent=2))
    completed = subprocess.run(cmd, check=False)
    register_invocation(db_path, args, "completed" if completed.returncode == 0 else "failed")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
