#!/usr/bin/env python3
"""ONNX exporter for the orbital PPO policy network."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class PolicyValueModel(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_input = nn.Linear(observation_dim, hidden_dim)
        self.encoder_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.10, dtype=torch.float32))

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        hidden_1 = torch.nn.functional.silu(self.encoder_input(observations))
        hidden_2 = torch.nn.functional.silu(self.encoder_hidden(hidden_1) + hidden_1)
        return hidden_2

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(observations)
        mean = torch.tanh(self.actor_mean(latent))
        std = torch.exp(torch.clamp(self.log_std, -1.2, 0.35)).expand_as(mean)
        value = self.critic(latent)
        return mean, std, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export orbital policy to ONNX")
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--output", default="artifacts/models/policy.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model = PolicyValueModel(args.observation_dim, args.action_dim, args.hidden_dim)
    model.eval()

    sample = torch.zeros((1, args.observation_dim), dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_mean", "action_std", "state_value"],
        dynamic_axes={"observation": {0: "batch"}},
    )

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
