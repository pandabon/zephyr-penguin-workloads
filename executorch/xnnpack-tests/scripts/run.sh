#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a bare-metal build (scripts/build.sh --target baremetal) on Spike. The exit status is the test's:
# 0 = every test passed, nonzero = a failure (or 124 = SPIKE_TIMEOUT hit, 100 = unexpected trap).
#
#   scripts/run.sh <build-dir or .elf>        e.g. scripts/run.sh build/xnnpack-tests/baremetal_f32-bf16-vcvt
# Environment:
#   SPIKE          the spike binary (default: `spike` on PATH, i.e. after `source <workspace>/env.sh`)
#   SPIKE_ISA      Spike's --isa string. If unset it is read, in this order, from
#                  1. ISA_FROM=<zephyr build dir>: SPIKE_ISA in its CMakeCache.txt (what the BSP's own
#                     `west build -t run` uses; spike.cmake writes it there), else
#                  2. baremetal/penguin-isa.cmake (PENGUIN_SPIKE_ISA; the file says where the string comes from)
#   SPIKE_TIMEOUT  seconds before a hung run is killed (default 120, like the BSP's `run` target)
#   SPIKE_EXTRA_ARGS  extra Spike arguments before the ELF
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
arg=${1:?usage: run.sh <build-dir or xnnpack_test.elf>}
elf=$arg; [ -d "$arg" ] && elf=$arg/xnnpack_test.elf
[ -f "$elf" ] || { echo "no such ELF: $elf (build it with scripts/build.sh --target baremetal)" >&2; exit 2; }

isa=${SPIKE_ISA:-}
src="SPIKE_ISA environment variable"
if [ -z "$isa" ] && [ -n "${ISA_FROM:-}" ]; then
  isa=$(sed -n 's/^SPIKE_ISA:INTERNAL=//p' "$ISA_FROM/CMakeCache.txt" 2>/dev/null || true)
  src="$ISA_FROM/CMakeCache.txt"
  [ -n "$isa" ] || { echo "no SPIKE_ISA in $ISA_FROM/CMakeCache.txt" >&2; exit 2; }
fi
if [ -z "$isa" ]; then
  isa=$(sed -n 's/^set(PENGUIN_SPIKE_ISA "\(.*\)")$/\1/p' "$here/baremetal/penguin-isa.cmake")
  src="baremetal/penguin-isa.cmake"
fi
spike=${SPIKE:-spike}
command -v "$spike" >/dev/null || { echo "spike not found ('$spike'): source <workspace>/env.sh or set SPIKE=" >&2; exit 2; }

echo "spike --isa=$isa   (ISA from $src)" >&2
# shellcheck disable=SC2086
exec timeout "${SPIKE_TIMEOUT:-120}" "$spike" --isa="$isa" ${SPIKE_EXTRA_ARGS:-} "$elf"
