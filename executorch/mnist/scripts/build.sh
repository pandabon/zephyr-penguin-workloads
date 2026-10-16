#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the ExecuTorch runner for one board and one model (a thin wrapper around `west build`;
# it prints the real command). Run inside an activated BSP environment (west, SDK on PATH).
#
#   EXECUTORCH_DIR=... XNNPACK_DIR=... scripts/build.sh [board] [model] [build-dir]
#   scripts/build.sh chipyard_penguin/penguin mnist_mlp_fp32 build/exec_fp32_chip
#   scripts/build.sh chipyard_penguin/penguin/vcu118_ta mnist_mlp_bf16 build/exec_bf16_ta
#
# Optional: PYTHON_EXECUTABLE / FLATC_EXECUTABLE (default: python3 and flatc on PATH), JOBS.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
board=${1:-chipyard_penguin/penguin}
model=${2:-mnist_mlp_fp32}
out=${3:-build/exec_${model}_$(echo "$board" | tr / _)}

: "${EXECUTORCH_DIR:?set EXECUTORCH_DIR to an ExecuTorch checkout (see the ExecuTorch section of the repo README.md)}"
extra=()
[ -n "${PYTHON_EXECUTABLE:-}" ] && extra+=("-DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE" "-DPython3_EXECUTABLE=$PYTHON_EXECUTABLE")
[ -n "${FLATC_EXECUTABLE:-}" ] && extra+=("-DFLATC_EXECUTABLE=$FLATC_EXECUTABLE")
[ -n "${XNNPACK_DIR:-}" ] && extra+=("-DXNNPACK_DIR=$XNNPACK_DIR")

# ExecuTorch's codegen runs python inside its source tree: do not litter it with __pycache__.
export PYTHONDONTWRITEBYTECODE=1
set -x
west build -p -b "$board" -d "$out" "$here" --build-opt=-j"${JOBS:-16}" -- \
    -DEXECUTORCH_DIR="$EXECUTORCH_DIR" -DEXEC_MODEL="$model" "${extra[@]}"
