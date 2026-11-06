# SPDX-License-Identifier: Apache-2.0
#
# One Zephyr app that builds ONE of Saturn's vector benchmarks, picked by name:
#
#   west build -b chipyard_penguin/penguin saturn -- -DSATURN_BENCH=vec-sgemm
#   west build -b chipyard_penguin/penguin/vcu118_ta saturn -- -DSATURN_BENCH=vec-sgemm
#
# The benchmark sources under third_party/saturn-vectors/benchmarks/ are unmodified copies of Saturn's
# (see CREDITS.txt / NOTICE); everything of ours is in shim/, conf/ and this file.
# Same pattern as atlas/upstream (ATLAS_TEST).
#
# What differs from Saturn's own bare-metal flow (the reason shim/ exists):
#   * startup/exit: Saturn links its own crt.S + HTIF syscalls; Zephyr boots its
#     own kernel and calls main() from a thread. shim/exit_shim.c wraps main()
#     (linker --wrap=main) so a returning main() ends the run with its value as
#     exit code (0 = pass), like the board repo's former cmake/plain_app
#     (zephyr-chipyard-bsp 93ace03; removed by BSP-4).
#   * setStats(): Saturn's common/syscalls.c records mcycle/minstret; shim/stats_shim.c
#     does the same and prints "stats: mcycle=<n> minstret=<n>".
#   * l_trace_encoder.h (a trace-encoder driver outside Saturn's repo): no-op stub.

cmake_minimum_required(VERSION 3.20.0)

set(SATURN_BENCH "vec-sgemv" CACHE STRING "Benchmark name = a directory (or .c file) under third_party/saturn-vectors/benchmarks/")

# Sources: the copy in ../../third_party/saturn-vectors/benchmarks (see its NOTICE/CREDITS.txt),
# or SATURN_DIR=<a saturn-vectors checkout> to build against a newer upstream with the same layout.
if(DEFINED ENV{SATURN_DIR} AND NOT SATURN_DIR)
  set(SATURN_DIR $ENV{SATURN_DIR})
endif()
if(SATURN_DIR)
  set(_b ${SATURN_DIR}/benchmarks)
else()
  set(_b ${CMAKE_CURRENT_LIST_DIR}/../../third_party/saturn-vectors/benchmarks)
endif()
if(IS_DIRECTORY ${_b}/${SATURN_BENCH})
  set(_bdir ${_b}/${SATURN_BENCH})
  file(GLOB _srcs CONFIGURE_DEPENDS ${_bdir}/*.c ${_bdir}/*.cc ${_bdir}/*.cpp ${_bdir}/*.S)
  # gen_data.cpp / gen_data.c are host-side data generators, not part of the target
  list(FILTER _srcs EXCLUDE REGEX "/gen_data[^/]*$")
elseif(EXISTS ${_b}/${SATURN_BENCH}.c)
  set(_bdir ${_b})                      # single-file tests (fp16_convert_test, fp8*_test)
  set(_srcs ${_b}/${SATURN_BENCH}.c)
else()
  file(GLOB _all RELATIVE ${_b} ${_b}/vec-* ${_b}/*_test.c)
  list(JOIN _all "\n  " _l)
  message(FATAL_ERROR "SATURN_BENCH='${SATURN_BENCH}' not found under third_party/saturn-vectors/benchmarks/. Available:\n  ${_l}")
endif()

# Per-benchmark Kconfig additions (C++ switches, workarounds): conf/<bench>.conf
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/conf/${SATURN_BENCH}.conf)
  list(PREPEND EXTRA_CONF_FILE ${CMAKE_CURRENT_LIST_DIR}/conf/${SATURN_BENCH}.conf)
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(saturn_${SATURN_BENCH})

message(STATUS "saturn: benchmark ${SATURN_BENCH}")
foreach(_f ${_srcs})
  message(STATUS "saturn:   ${_f}")
endforeach()

# Ara's helper sources (similarity_check, fdotp, spmv, gemv): shared by many benchmarks,
# unused objects are dropped by the linker.
file(GLOB _ara CONFIGURE_DEPENDS ${_b}/common/ara/*.c)

target_sources(app PRIVATE ${_srcs} ${_ara}
  shim/stats_shim.c
  shim/exit_shim.c)
target_include_directories(app PRIVATE
  ${_bdir}                             # dataset1.h, local headers
  ${_b}/common                         # util.h bme.h rvv_mx.h ara/  (Saturn's)
  ${_b}/env                            # encoding.h
  ${_b}/utasks                         # utasks.h
  shim/include)                        # driver/.../l_trace_encoder.h stub (ours)

# main() -> __wrap_main() in shim/exit_shim.c
zephyr_ld_options(-Wl,--wrap=main)
