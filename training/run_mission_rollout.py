#!/usr/bin/env python3
"""Generate a deterministic orbital rollout via pybind11 core bindings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import py_orbital_core as orbital


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orbital rollout via pybind bindings")
    parser.add_argument("--mission-id", default="python_rollout")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output", default="artifacts/reports/python_rollout.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dynamics = orbital.DynamicsConfig()
    reward = orbital.RewardWeights()
    core = orbital.OrbitalControlCore(dynamics, reward)

    initial = orbital.OrbitalState3DOF()
    initial.position_m = [6_805_000.0, -500.0, 180.0]
    initial.velocity_mps = [0.0, 7670.0, 0.0]

    target = orbital.MissionTarget()
    target.target_position_m = [6_800_000.0, 0.0, 0.0]
    target.target_velocity_mps = [0.0, 7670.0, 0.0]

    result = core.run_open_loop_rollout(args.mission_id, initial, target, args.steps, args.dt)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "mission_time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "ux", "uy", "uz", "reward"])
        for step in result.timeline:
            writer.writerow(
                [
                    step.step_index,
                    step.mission_time_s,
                    step.position_m[0],
                    step.position_m[1],
                    step.position_m[2],
                    step.velocity_mps[0],
                    step.velocity_mps[1],
                    step.velocity_mps[2],
                    step.thrust_axis[0],
                    step.thrust_axis[1],
                    step.thrust_axis[2],
                    step.reward,
                ]
            )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
