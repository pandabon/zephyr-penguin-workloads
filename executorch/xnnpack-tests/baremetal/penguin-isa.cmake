# SPDX-License-Identifier: Apache-2.0
#
# The ISA of the Penguin SoC, as the BSP builds Zephyr apps for it. Read by toolchain.cmake (GCC flags)
# and by scripts/run.sh (Spike's --isa, via the line starting with "set(PENGUIN_SPIKE_ISA").
#
# WHERE THESE STRINGS COME FROM (they are not invented here):
#   * PENGUIN_MARCH/MABI/MCMODEL: the compile line `west build -b chipyard_penguin/penguin` gives GCC for the
#     base board (soc/chipyard/penguin/Kconfig of zephyr-chipyard-bsp turns the RISCV_ISA_EXT_* symbols into
#     -march; Zephyr's cmake/compiler/gcc/target_riscv.cmake writes them). Taken from build.ninja of the
#     Zephyr mode of this app (build/xnnpack-tests/gt2-f32-bf16-vcvt) on 2026-10-20.
#   * PENGUIN_SPIKE_ISA: the `--isa=` of that same build's `run` target; zephyr-chipyard-bsp's
#     cmake/spike.cmake builds it from the same Kconfig symbols plus SPIKE_ISA_EXTRA
#     (boards/chipyard/penguin/board.cmake: zicntr zfbfmin zvfbfa zvfofp8min, the chip's extensions that
#     GCC 14.3 cannot express in -march).
# If the BSP's ISA changes, these lines are the only place to update. To use the string of a Zephyr
# build directory instead of this file: `ISA_FROM=<zephyr build dir> scripts/run.sh` (reads SPIKE_ISA from
# its CMakeCache.txt, which spike.cmake writes), or SPIKE_ISA=<string> to give it directly.
set(PENGUIN_MARCH "rv64imafdcv_zicsr_zifencei_zba_zbb_zbs_zfh_zvfh_zvfbfmin")
set(PENGUIN_MABI "lp64d")
set(PENGUIN_MCMODEL "medany")
set(PENGUIN_SPIKE_ISA "rv64imafdcv_zicsr_zifencei_zba_zbb_zbs_zfh_zvfh_zvfbfmin_zaamo_zalrsc_zicntr_zfbfmin_zvfbfa_zvfofp8min_zvl256b")
