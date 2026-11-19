# SPDX-License-Identifier: Apache-2.0
#
# Shared by the two build modes of this app (the Zephyr one, ../CMakeLists.txt, and the bare-metal one,
# ../baremetal/CMakeLists.txt): where the opt-in dependencies are, which XNNPACK test is built, and the
# XNNPACK/pthreadpool/cpuinfo options. Moved here unchanged from the original Zephyr CMakeLists.txt
# (commit 7177e7d) so that both modes build the same thing from the same inputs.

# ---- opt-in dependencies, located by variable or environment -----------------
foreach(_v EXECUTORCH_DIR XNNPACK_DIR GOOGLETEST_DIR THIRD_PARTY_DIR)
  if(NOT DEFINED ${_v} AND DEFINED ENV{${_v}})
    set(${_v} "$ENV{${_v}}")
  endif()
endforeach()
set(_dep_help "\n  This app is an opt-in workload: XNNPACK and ExecuTorch are not part of this repo.\n  Example:\n    export XNNPACK_DIR=/path/to/XNNPACK\n    export EXECUTORCH_DIR=/path/to/executorch   # with third-party/googletest initialised\n")
if(NOT XNNPACK_DIR OR NOT EXISTS "${XNNPACK_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "XNNPACK_DIR is not set or has no CMakeLists.txt (got '${XNNPACK_DIR}').${_dep_help}")
endif()
get_filename_component(XNNPACK_DIR "${XNNPACK_DIR}" REALPATH)
if(NOT THIRD_PARTY_DIR AND EXECUTORCH_DIR)
  set(THIRD_PARTY_DIR "${EXECUTORCH_DIR}/backends/xnnpack/third-party")   # pthreadpool, FXdiv, cpuinfo
endif()
if(NOT GOOGLETEST_DIR AND EXECUTORCH_DIR)
  set(GOOGLETEST_DIR "${EXECUTORCH_DIR}/third-party/googletest")
endif()
if(NOT GOOGLETEST_DIR OR NOT EXISTS "${GOOGLETEST_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "GoogleTest not found (GOOGLETEST_DIR='${GOOGLETEST_DIR}'). In ExecuTorch run: git submodule update --init --depth 1 third-party/googletest${_dep_help}")
endif()
foreach(_d pthreadpool FXdiv cpuinfo)
  if(NOT THIRD_PARTY_DIR OR NOT EXISTS "${THIRD_PARTY_DIR}/${_d}/CMakeLists.txt")
    message(FATAL_ERROR "THIRD_PARTY_DIR/${_d} missing (THIRD_PARTY_DIR='${THIRD_PARTY_DIR}'; default is EXECUTORCH_DIR/backends/xnnpack/third-party).${_dep_help}")
  endif()
endforeach()
get_filename_component(GOOGLETEST_DIR "${GOOGLETEST_DIR}" REALPATH)
get_filename_component(THIRD_PARTY_DIR "${THIRD_PARTY_DIR}" REALPATH)

set(XNN_TEST "f32-bf16-vcvt" CACHE STRING "XNNPACK test/<name>.cc to run")
set(XNN_TEST_SUPPORT "unary-ops" CACHE STRING "test-support sources (test/<name>.cc) the test links against")
set(XNN_TEST_FILTER "" CACHE STRING "GoogleTest filter compiled into main() (empty = run all)")
set(XNN_TEST_SEED "" CACHE STRING "GoogleTest random seed compiled into main() (empty = GoogleTest's own, time-based)")
set(XNN_TEST_LINK "" CACHE STRING "extra XNNPACK CMake targets to link (e.g. XNNPACK for operator/subgraph-level tests)")
set(_test_src "${XNNPACK_DIR}/test/${XNN_TEST}.cc")
if(NOT EXISTS "${_test_src}")
  message(FATAL_ERROR "${_test_src} does not exist")
endif()


# ---- XNNPACK and its dependencies: options (the caller then does add_subdirectory) ----
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "")        # its own test build would FetchContent googletest
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(XNNPACK_BUILD_ALL_MICROKERNELS ON CACHE BOOL "")   # the tests also cover the non-"prod" kernels (scalar u1..u4)
set(XNNPACK_LIBRARY_TYPE static CACHE STRING "")
# cpuinfo has no Zephyr backend: with XNN_ENABLE_CPUINFO=1 XNNPACK would call into its empty stub
# and abort at xnn_initialize(). The option is added by our XNNPACK fork (CMakeLists.txt).
set(XNNPACK_ENABLE_CPUINFO OFF CACHE BOOL "")
set(XNNPACK_ENABLE_RISCV_VECTOR ON CACHE BOOL "")
set(XNNPACK_ENABLE_RISCV_GEMMINI OFF CACHE BOOL "")
set(XNNPACK_ENABLE_KLEIDIAI OFF CACHE BOOL "")
# Where XNNPACK finds its dependencies instead of downloading them (the variables its CMake documents).
set(PTHREADPOOL_SOURCE_DIR "${THIRD_PARTY_DIR}/pthreadpool" CACHE STRING "")
set(CPUINFO_SOURCE_DIR "${THIRD_PARTY_DIR}/cpuinfo" CACHE STRING "")
set(FXDIV_SOURCE_DIR "${THIRD_PARTY_DIR}/FXdiv" CACHE STRING "")
set(CPUINFO_LIBRARY_TYPE static CACHE STRING "")
set(PTHREADPOOL_LIBRARY_TYPE static CACHE STRING "")
set(PTHREADPOOL_ALLOW_DEPRECATED_API ON CACHE BOOL "")
set(CLOG_SOURCE_DIR "${CPUINFO_SOURCE_DIR}/deps/clog")
set(CPUINFO_LOG_LEVEL error CACHE STRING "")
