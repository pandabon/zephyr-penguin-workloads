#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run already-built Atlas apps on Spike with the Atlas Spike model (a plugin with the Atlas
# registers, IMEM, VMEM and a scalar ISA, but NO tensor/DMA operations) and classify each run.
#
#   scripts/run_with_atlas_model.sh <libatlas_spike.so> <build-root-or-builddir>... 
#
# <build-root> holds one build dir per scenario (e.g. the --build-root of `make suite`:
# <ws>/build/test-runner); a single build dir (has zephyr/zephyr.elf) works too. Needs `spike`
# (SPIKE or PATH). Environment: TIMEOUT (seconds per run, default 60).
#
# Classes: PASS (the app printed its PASS line and exited 0); UNIMPLEMENTED (the model stopped at an
# operation it does not implement: expected for tensor/DMA kernels); ILLEGAL (Atlas halted with
# halt_reason=1, i.e. the model rejected an instruction: a model gap or mismatch to check, not a
# hardware verdict); HAZARD (failed after the model reported an access that would hang real
# hardware, e.g. IMEM read while running: timing-race tests); FAIL (anything else); TIMEOUT.
# The Atlas device addresses are the Penguin ones (IMEM 0x20000, CSR 0x40000, VMEM 0x20000000).
# Why a script and not `make suite`: the scenarios are build_only (plain Spike has no Atlas).
set -u
[ $# -ge 2 ] || { sed -n '4,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
LIB=$1; shift
SPIKE_BIN=${SPIKE:-spike}
command -v "$SPIKE_BIN" >/dev/null || { echo "error: spike not found (set SPIKE=)" >&2; exit 2; }
[ -f "$LIB" ] || { echo "error: $LIB not found (build it: make -C <atlas-spike-model>)" >&2; exit 2; }
TIMEOUT=${TIMEOUT:-60}
declare -A N; total=0
printf '%-72s %-14s %s\n' SCENARIO RESULT DETAIL
for root in "$@"; do
  for d in $( [ -f "$root/zephyr/zephyr.elf" ] && echo "$root" || ls -d "$root"/*/ 2>/dev/null ); do
    d=${d%/}; elf=$d/zephyr/zephyr.elf; [ -f "$elf" ] || continue
    isa=$(sed -n 's/^SPIKE_ISA:INTERNAL=//p' "$d/CMakeCache.txt")
    out=$(timeout "$TIMEOUT" "$SPIKE_BIN" --isa="$isa" --extlib="$LIB" \
          --device=atlas,imem,0x20000 --device=atlas,csr,0x40000 --device=atlas,vmem,0x20000000 \
          "$elf" 2>&1 </dev/null); rc=$?
    if   [ $rc -eq 124 ]; then res=TIMEOUT
    elif grep -q 'UNIMPLEMENTED' <<<"$out"; then res=UNIMPLEMENTED
    elif [ $rc -eq 0 ] && grep -Eq 'ATLASUP .* PASS|ATLAS KERNELS=PASS|ALL TESTS PASSED|^PASS' <<<"$out" && ! grep -Eq 'FAIL|TIMEOUT' <<<"$out"; then res=PASS
    elif grep -Eq 'halt_reason ?= ?1([^0-9]|$)' <<<"$out"; then res=ILLEGAL
    elif grep -q 'atlas-model: HAZARD' <<<"$out"; then res=HAZARD
    else res=FAIL; fi
    detail=$(grep -E 'UNIMPLEMENTED' <<<"$out" | head -n 1; grep -E '^FAIL|ATLASUP .* (FAIL|TIMEOUT)|ATLASK SUMMARY' <<<"$out" | head -n 1)
    detail=$(tr '\n' ' ' <<<"$detail" | cut -c1-120)
    printf '%-72s %-14s rc=%s %s\n' "$(basename "$d")" "$res" "$rc" "$detail"
    N[$res]=$(( ${N[$res]:-0} + 1 )); total=$((total+1))
  done
done
echo; printf 'total=%d' $total; for k in PASS UNIMPLEMENTED ILLEGAL HAZARD FAIL TIMEOUT; do printf '  %s=%d' $k ${N[$k]:-0}; done; echo
