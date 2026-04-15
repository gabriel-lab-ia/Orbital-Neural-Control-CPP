#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate orbital policy checkpoints")
    parser.add_argument("--run-id", default=f"eval_py_{int(time.time())}")
    parser.add_argument("--checkpoint", default="artifacts/latest/checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--backend", default="libtorch")
    parser.add_argument("--binary", default="./build/nmc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        args.binary,
        "eval",
        "--run-id",
        args.run_id,
        "--checkpoint",
        args.checkpoint,
        "--episodes",
        str(args.episodes),
        "--backend",
        args.backend,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
