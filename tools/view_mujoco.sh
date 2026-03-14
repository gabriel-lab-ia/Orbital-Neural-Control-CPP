#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUJOCO_ROOT="${MUJOCO_ROOT:-$HOME/.local/mujoco-3.2.6}"
SIMULATE_BIN="$MUJOCO_ROOT/bin/simulate"
MODEL_PATH="${1:-$PROJECT_ROOT/assets/mujoco/cartpole.xml}"

if [[ ! -x "$SIMULATE_BIN" ]]; then
    echo "MuJoCo simulate binary not found at: $SIMULATE_BIN" >&2
    echo "Set MUJOCO_ROOT or install MuJoCo under ~/.local/mujoco-3.2.6" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model XML not found at: $MODEL_PATH" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$MUJOCO_ROOT/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_PLUGIN_PATH="$MUJOCO_ROOT/bin/mujoco_plugin"

echo "Opening MuJoCo viewer with:"
echo "  simulate: $SIMULATE_BIN"
echo "  model:    $MODEL_PATH"

exec "$SIMULATE_BIN" "$MODEL_PATH"
