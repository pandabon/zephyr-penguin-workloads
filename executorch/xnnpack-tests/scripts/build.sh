#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build one unmodified XNNPACK gtest file as a Zephyr app (a thin wrapper around `west build`;
# it prints the real command). Run inside an activated BSP environment (west, SDK on PATH).
#
#   XNNPACK_DIR=... EXECUTORCH_DIR=... scripts/build.sh [board] [test-name] [build-dir]
#   scripts/build.sh chipyard_penguin/penguin f32-bf16-vcvt build/xnnpack-tests/f32-bf16-vcvt
# Run on Spike:  west build -d <build-dir> -t run      (SPIKE_TIMEOUT=<seconds> for long files)
# Derived from executorch/smolvla/scripts/build.sh in this repo.
# Optional environment, passed on as CMake options: XNN_TEST_SUPPORT (";"-separated test/<name>.cc
# support sources, default unary-ops), XNN_TEST_FILTER (GoogleTest filter), XNN_TEST_SEED (random seed),
# EXTRA_CMAKE (more words), JOBS (default 16).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
board=${1:-chipyard_penguin/penguin}
test=${2:-f32-bf16-vcvt}
out=${3:-build/xnnpack-tests/${test}_$(echo "$board" | tr / _)}

: "${XNNPACK_DIR:?set XNNPACK_DIR to an XNNPACK checkout}"
: "${EXECUTORCH_DIR:?set EXECUTORCH_DIR to an ExecuTorch checkout with third-party/googletest initialised}"
extra=()
for v in XNN_TEST_SUPPORT XNN_TEST_FILTER XNN_TEST_SEED; do
  [ -n "${!v:-}" ] && extra+=("-D$v=${!v}")
done
# shellcheck disable=SC2206
[ -n "${EXTRA_CMAKE:-}" ] && extra+=(${EXTRA_CMAKE})

# XNNPACK's CMake runs python in its source tree: do not litter it with __pycache__.
export PYTHONDONTWRITEBYTECODE=1
set -x
west build -p -b "$board" -d "$out" "$here" --build-opt=-j"${JOBS:-16}" -- \
    -DXNNPACK_DIR="$XNNPACK_DIR" -DEXECUTORCH_DIR="$EXECUTORCH_DIR" -DXNN_TEST="$test" "${extra[@]}"
