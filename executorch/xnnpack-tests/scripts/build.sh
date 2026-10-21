#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build one unmodified XNNPACK gtest file, in one of two modes (decision WL-8):
#   --target baremetal  (default)  plain CMake + GCC/picolibc from the Zephyr SDK, no Zephyr, runs on Spike
#                                  with scripts/run.sh (see baremetal/CMakeLists.txt)
#   --target zephyr                the Zephyr app (a thin wrapper around `west build`; it prints the real
#                                  command). Needs an activated BSP environment (west, SDK on PATH).
# Both modes need `source <workspace>/env.sh` (compiler, ninja, spike on PATH).
#
#   XNNPACK_DIR=... EXECUTORCH_DIR=... scripts/build.sh [--target baremetal|zephyr] [--board B] [--out DIR] [test-name]
#   scripts/build.sh f32-bf16-vcvt                       # bare metal -> build/xnnpack-tests/baremetal_f32-bf16-vcvt
#   scripts/build.sh --target zephyr f32-bf16-vcvt       # Zephyr, board chipyard_penguin/penguin
#   scripts/run.sh build/xnnpack-tests/baremetal_f32-bf16-vcvt          # bare metal on Spike
#   west build -d <build-dir> -t run                                    # Zephyr on Spike (SPIKE_TIMEOUT=<s>)
# Optional environment, passed on as CMake options: XNN_TEST_SUPPORT (";"-separated test/<name>.cc
# support sources, default unary-ops), XNN_TEST_FILTER (GoogleTest filter), XNN_TEST_SEED (random seed),
# XNN_TEST_LINK (extra XNNPACK CMake targets), EXTRA_CMAKE (more words), JOBS (default 16).
# Derived from executorch/smolvla/scripts/build.sh in this repo.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
target=baremetal
board=chipyard_penguin/penguin
out=
test=f32-bf16-vcvt
while [ $# -gt 0 ]; do
  case "$1" in
    --target) target=$2; shift 2 ;;
    --board) board=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) test=$1; shift ;;
  esac
done
case "$target" in baremetal|zephyr) ;; *) echo "--target must be baremetal or zephyr, got '$target'" >&2; exit 2 ;; esac
[ -n "$out" ] || out=build/xnnpack-tests/${target}_${test}

: "${XNNPACK_DIR:?set XNNPACK_DIR to an XNNPACK checkout}"
: "${EXECUTORCH_DIR:?set EXECUTORCH_DIR to an ExecuTorch checkout with third-party/googletest initialised}"
extra=()
for v in XNN_TEST_SUPPORT XNN_TEST_FILTER XNN_TEST_SEED XNN_TEST_LINK; do
  [ -n "${!v:-}" ] && extra+=("-D$v=${!v}")
done
# shellcheck disable=SC2206
[ -n "${EXTRA_CMAKE:-}" ] && extra+=(${EXTRA_CMAKE})

# XNNPACK's CMake runs python in its source tree: do not litter it with __pycache__.
export PYTHONDONTWRITEBYTECODE=1
set -x
if [ "$target" = zephyr ]; then
  west build -p -b "$board" -d "$out" "$here" --build-opt=-j"${JOBS:-16}" -- \
      -DXNNPACK_DIR="$XNNPACK_DIR" -DEXECUTORCH_DIR="$EXECUTORCH_DIR" -DXNN_TEST="$test" "${extra[@]}"
else
  rm -rf "$out"
  cmake -S "$here/baremetal" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE= \
      -DCMAKE_TOOLCHAIN_FILE="$here/baremetal/toolchain.cmake" \
      -DXNNPACK_DIR="$XNNPACK_DIR" -DEXECUTORCH_DIR="$EXECUTORCH_DIR" -DXNN_TEST="$test" "${extra[@]}"
  cmake --build "$out" -j"${JOBS:-16}"
  set +x
  echo "ELF: $out/xnnpack_test.elf"
  echo "run: $here/scripts/run.sh $out"
fi
