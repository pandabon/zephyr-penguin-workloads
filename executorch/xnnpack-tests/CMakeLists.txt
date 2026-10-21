# SPDX-License-Identifier: Apache-2.0
#
# One of XNNPACK's OWN, UNMODIFIED upstream GoogleTest files, run as a Zephyr application (on Spike or
# the chip). Pilot for WL-8 / decision D32: everything runs on Zephyr, no bare-metal runtime.
#
# What is compiled, all straight from the dependency trees (nothing is copied or patched):
#   * the test file   XNNPACK_DIR/test/${XNN_TEST}.cc     (default f32-bf16-vcvt; any test/<name>.cc whose
#                     support sources are in XNN_TEST_SUPPORT works the same way)
#   * its support     XNNPACK_DIR/test/{unary-ops}.cc (add others, e.g. next_prime, as a test needs them)   (XNN_TEST_SUPPORT)
#   * XNNPACK         add_subdirectory(XNNPACK_DIR) with tests/benchmarks off, all micro-kernels on
#   * GoogleTest      EXECUTORCH_DIR/third-party/googletest (ExecuTorch's pinned submodule; nothing is downloaded)
#   * pthreadpool, FXdiv, cpuinfo   EXECUTORCH_DIR/backends/xnnpack/third-party/* (the versions ExecuTorch pins)
# The only code of ours is src/main.cpp: a main() that runs GoogleTest and exits through the BSP.
#
# OPT-IN: needs two external trees that this repo does NOT contain:
#   XNNPACK_DIR     an XNNPACK checkout (the test file is read from its test/ directory)
#   EXECUTORCH_DIR  an ExecuTorch checkout with third-party/googletest initialised
#                   (git submodule update --init --depth 1 third-party/googletest)
# or GOOGLETEST_DIR / THIRD_PARTY_DIR to point at the individual pieces. No flatc, no python torchgen.
# Nothing is downloaded and nothing in the given trees is modified.
#
# Options (-D...): XNN_TEST=f32-bf16-vcvt   which test/<name>.cc
#                  XNN_TEST_SUPPORT="unary-ops"  test-support sources (test/<name>.cc)
#                  XNN_TEST_FILTER=""       GoogleTest filter compiled in (default: run everything)
#                  XNN_TEST_SEED=""         GoogleTest random seed compiled in (default: its own time-based one)
#
# Credits:
#  * The idea of running XNNPACK's gtests on Spike comes from Jonathan Lin's archived harness
#    (pandabon/XNNPACK, branch torch-1.0.0-bump-bf16, docs-ucb-bar/spike-runtime.md @ 6c25b9e9e8); no code was taken from it (it was bare-metal).
#  * The Zephyr flag propagation to third-party targets and the generated-header ordering
#    are derived from executorch/smolvla in this repo, itself derived from samples/exeggutorch of
#    pandabon/zephyr-chipyard-sw (branch exeggutorch @ e12bfa38034, Jonathan Lin) and ExecuTorch
#    examples/arm/executor_runner (Copyright 2023-2024 Arm Limited and/or its affiliates; BSD-style license).
#  * The test files are Copyright Google LLC (and Facebook, Inc.), BSD-style license of the XNNPACK repo.

cmake_minimum_required(VERSION 3.20)

# Dependency lookup and the XNN_TEST* options are shared with the bare-metal mode.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/xnnpack-tests-common.cmake)

add_compile_definitions(_POSIX_C_SOURCE=200809L __POSIX_VISIBLE=200809L)
# XNNPACK: no mmap/mprotect on Zephyr; cpuinfo's log.c wants STDERR_FILENO, which Zephyr's <unistd.h> lacks.
add_compile_definitions(XNN_HAS_MMAP=0 STDERR_FILENO=2)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(penguin_xnnpack_tests)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- XNNPACK and its dependencies, built as a normal CMake sub-project ----------
# XNNPACK/pthreadpool/cpuinfo options: see cmake/xnnpack-tests-common.cmake
add_subdirectory("${FXDIV_SOURCE_DIR}" fxdiv)

# Zephyr hands its include paths, definitions and ISA options (-march/-mabi/-mcmodel) to its own
# targets (zephyr_interface/app), not to CMAKE_<LANG>_FLAGS. The third-party libraries below are separate
# targets, so without this they are built for the compiler's default RISC-V target and against the SDK's
# default libc headers. So give every target created below the app's flag set.
include_directories($<TARGET_PROPERTY:app,INTERFACE_INCLUDE_DIRECTORIES>)
add_compile_options($<TARGET_PROPERTY:app,INTERFACE_COMPILE_OPTIONS>)
add_compile_definitions($<TARGET_PROPERTY:app,INTERFACE_COMPILE_DEFINITIONS>)

set(_xnn_build "${CMAKE_CURRENT_BINARY_DIR}/xnnpack")
add_subdirectory("${XNNPACK_DIR}" "${_xnn_build}" EXCLUDE_FROM_ALL)

# GoogleTest: no pthreads (Zephyr's POSIX layer is partial), no gmock, nothing installed.
set(gtest_disable_pthreads ON CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
set(gtest_build_tests OFF CACHE BOOL "")
add_subdirectory("${GOOGLETEST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)

# Zephyr's <unistd.h> lacks isatty/chdir/getcwd/dup/dup2 (see the header), and its POSIX limits clash
# with picolibc's under gtest's -Werror: shim the first, keep the second a warning (third-party code).
set(_gtest_shim "${CMAKE_CURRENT_SOURCE_DIR}/src/zephyr_compat/gtest_posix_shim.h")
target_compile_options(gtest PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-include${_gtest_shim}> -Wno-error -w)

# Every library created above must wait for Zephyr's generated headers (zephyr/syscall_list.h,
# autoconf.h), or a parallel build can compile them before those exist.
function(_depend_on_zephyr_headers DIR)
  get_property(_targets DIRECTORY "${DIR}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_t IN LISTS _targets)
    get_target_property(_type ${_t} TYPE)
    if(_type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
      add_dependencies(${_t} zephyr_generated_headers)
    endif()
  endforeach()
  get_property(_subdirs DIRECTORY "${DIR}" PROPERTY SUBDIRECTORIES)
  foreach(_d IN LISTS _subdirs)
    _depend_on_zephyr_headers("${_d}")
  endforeach()
endfunction()
_depend_on_zephyr_headers("${_xnn_build}")
_depend_on_zephyr_headers("${CMAKE_CURRENT_BINARY_DIR}/googletest")
_depend_on_zephyr_headers("${CMAKE_CURRENT_BINARY_DIR}/fxdiv")

# ---- the test application --------------------------------------------------------
# XNNPACK sets its configuration macros (XNN_ENABLE_RISCV_VECTOR, XNN_ENABLE_*...) with
# ADD_COMPILE_DEFINITIONS in its own directory; the test must see the same values (the .inc file
# lists the RVV kernels under XNN_ENABLE_RISCV_VECTOR), so take them from its directory property.
get_directory_property(_xnn_defs DIRECTORY "${XNNPACK_DIR}" COMPILE_DEFINITIONS)
set(_support)
foreach(_s IN LISTS XNN_TEST_SUPPORT)
  list(APPEND _support "${XNNPACK_DIR}/test/${_s}.cc")
endforeach()
target_sources(app PRIVATE src/main.cpp "${_test_src}" ${_support})
target_compile_definitions(app PRIVATE ${_xnn_defs} XNN_TEST_NAME="${XNN_TEST}" XNN_TEST_FILTER="${XNN_TEST_FILTER}" XNN_TEST_SEED="${XNN_TEST_SEED}")
# XNNPACK's headers are included as "src/xnnpack/..", "include/xnnpack.h", "test/..": the tree root is the include dir.
target_include_directories(app PRIVATE "${XNNPACK_DIR}" "${XNNPACK_DIR}/include" "${GOOGLETEST_DIR}/googletest/include")
target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-include${_gtest_shim}>)
target_link_libraries(app PRIVATE
  gtest
  xnnpack-datatype xnnpack-hardware-config xnnpack-logging xnnpack-microkernels-all xnnpack-microparams-init
  # reference weight packers (src/reference/packing.cc): GEMM/conv microkernel testers pack weights with them
  xnnpack-packing
  pthreadpool
  ${XNN_TEST_LINK}
)
zephyr_linker_sources(DATA_SECTIONS custom-sections.ld)
