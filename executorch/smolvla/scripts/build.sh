#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the ExecuTorch runner for one board and one model (a thin wrapper around `west build`;
# it prints the real command). Run inside an activated BSP environment (west, SDK on PATH).
#
#   EXECUTORCH_DIR=... XNNPACK_DIR=... scripts/build.sh [board] [model] [build-dir]
#   scripts/build.sh chipyard_penguin/penguin /path/to/model.pte build/smolvla/x
# Derived from executorch/mnist/scripts/build.sh in this repo; the model is now a .pte path.
# Optional environment, passed on as CMake options: EXEC_INPUTS, EXEC_EXPECTED, EXEC_TOL, EXEC_EMBED (hex|incbin),
# EXEC_STEP_DATA, EXEC_STEP_EXPECTED (step mode, see CMakeLists.txt),
# ET_METHOD_POOL_SIZE, ET_TEMP_POOL_SIZE, EXTRA_CMAKE (more words, e.g. "-DXNNPACK_ENABLE_RISCV_VECTOR=OFF").
#
# (Python3_EXECUTABLE is deliberately not set: Zephyr must keep using the west python.)
# Optional: PYTHON_EXECUTABLE / FLATC_EXECUTABLE (default: python3 and flatc on PATH), JOBS.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
board=${1:-chipyard_penguin/penguin}
model=${2:?give the path of a .pte model}
out=${3:-build/smolvla_$(basename "${model%.pte}")_$(echo "$board" | tr / _)}

: "${EXECUTORCH_DIR:?set EXECUTORCH_DIR to an ExecuTorch checkout (see the ExecuTorch section of the repo README.md)}"
extra=()
[ -n "${PYTHON_EXECUTABLE:-}" ] && extra+=("-DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE")
[ -n "${FLATC_EXECUTABLE:-}" ] && extra+=("-DFLATC_EXECUTABLE=$FLATC_EXECUTABLE")
[ -n "${XNNPACK_DIR:-}" ] && extra+=("-DXNNPACK_DIR=$XNNPACK_DIR")
for v in EXEC_INPUTS EXEC_EXPECTED EXEC_EMBED EXEC_STEP_DATA EXEC_STEP_EXPECTED EXEC_TOL ET_METHOD_POOL_SIZE ET_TEMP_POOL_SIZE; do
  [ -n "${!v:-}" ] && extra+=("-D$v=${!v}")
done
# shellcheck disable=SC2206
[ -n "${EXTRA_CMAKE:-}" ] && extra+=(${EXTRA_CMAKE})

# ExecuTorch's codegen runs python inside its source tree: do not litter it with __pycache__.
export PYTHONDONTWRITEBYTECODE=1
set -x
west build -p -b "$board" -d "$out" "$here" --build-opt=-j"${JOBS:-16}" -- \
    -DEXECUTORCH_DIR="$EXECUTORCH_DIR" -DEXEC_MODEL="$model" "${extra[@]}"
