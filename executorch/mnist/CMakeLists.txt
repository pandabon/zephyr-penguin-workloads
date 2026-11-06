# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch (+ XNNPACK RVV kernels) MNIST runner for the Penguin SoC.
#
# OPT-IN: this app needs two external source trees that this repo does NOT contain:
#   EXECUTORCH_DIR  an ExecuTorch checkout (a directory named "executorch")
#   XNNPACK_DIR     an XNNPACK checkout (optional if EXECUTORCH_DIR already has a working
#                   backends/xnnpack/third-party/XNNPACK)
# plus `flatc` and a python3 with torchgen/pyyaml for ExecuTorch's codegen (PYTHON_EXECUTABLE,
# FLATC_EXECUTABLE or on PATH). Without them the configure step stops with an explanation.
# Unset variables default to the checkouts of the west group `executorch` (west.yml, off by
# default; PW-2): <west topdir>/modules/lib/executorch and <west topdir>/modules/lib/XNNPACK.
# Turn it on once with `west config manifest.group-filter -- +executorch` and `west update`.
# CMake downloads nothing, and nothing in the given trees is modified.
#
# Model: -DEXEC_MODEL=mnist_mlp_fp32 (default) | mnist_mlp_bf16 | lenet_bf16_user | /path/to/x.pte
#
# Credits: derived from samples/exeggutorch/executor_runner/CMakeLists.txt of
# pandabon/zephyr-chipyard-sw (branch exeggutorch @ e12bfa38034, Jonathan Lin), itself from
# ExecuTorch examples/arm/executor_runner (Copyright 2023-2024 Arm Limited and/or its
# affiliates; BSD-style license of the ExecuTorch repo).

cmake_minimum_required(VERSION 3.20)

# ---- opt-in dependencies, located by variable or environment -----------------
foreach(_v EXECUTORCH_DIR XNNPACK_DIR)
  if(NOT DEFINED ${_v} AND DEFINED ENV{${_v}})
    set(${_v} "$ENV{${_v}}")
  endif()
endforeach()

# Defaults: the west group `executorch` (west.yml). Only used when the checkout exists.
set(_west_lib "")
find_program(_west_exe west)
if(_west_exe)
  execute_process(COMMAND ${_west_exe} topdir OUTPUT_VARIABLE _west_topdir
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _west_rc)
  if(_west_rc EQUAL 0 AND _west_topdir)
    set(_west_lib "${_west_topdir}/modules/lib")
  endif()
endif()
if(NOT EXECUTORCH_DIR AND _west_lib AND EXISTS "${_west_lib}/executorch/CMakeLists.txt")
  set(EXECUTORCH_DIR "${_west_lib}/executorch")
endif()
if(NOT XNNPACK_DIR AND _west_lib AND EXISTS "${_west_lib}/XNNPACK/CMakeLists.txt")
  set(XNNPACK_DIR "${_west_lib}/XNNPACK")
endif()

set(_dep_help "\n  This app is an opt-in workload: ExecuTorch and XNNPACK are not part of this repo.\n  Either turn on the west group that fetches them at the tested commits (once per workspace):\n    west config manifest.group-filter -- +executorch && west update\n  or point at your own checkouts:\n    export EXECUTORCH_DIR=/path/to/executorch      # directory is named 'executorch'\n    export XNNPACK_DIR=/path/to/XNNPACK\n  and in both cases:\n    export PATH=/path/to/env/bin:$PATH             # flatc + python3 with torchgen, pyyaml\n  See the ExecuTorch section of the repo README.md.\n")
if(NOT EXECUTORCH_DIR OR NOT EXISTS "${EXECUTORCH_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "ExecuTorch not found: EXECUTORCH_DIR is not set or has no CMakeLists.txt (got '${EXECUTORCH_DIR}'), and the west group checkout '${_west_lib}/executorch' is missing.${_dep_help}")
endif()
get_filename_component(EXECUTORCH_DIR "${EXECUTORCH_DIR}" REALPATH)
get_filename_component(_et_name "${EXECUTORCH_DIR}" NAME)
if(NOT _et_name STREQUAL "executorch")
  message(FATAL_ERROR "EXECUTORCH_DIR must be a directory named 'executorch' (its headers are included as <executorch/...> relative to its parent); got '${EXECUTORCH_DIR}'.")
endif()
if(XNNPACK_DIR)
  if(NOT EXISTS "${XNNPACK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "XNNPACK_DIR='${XNNPACK_DIR}' has no CMakeLists.txt.${_dep_help}")
  endif()
elseif(NOT EXISTS "${EXECUTORCH_DIR}/backends/xnnpack/third-party/XNNPACK/CMakeLists.txt")
  message(FATAL_ERROR "XNNPACK_DIR is not set and ${EXECUTORCH_DIR}/backends/xnnpack/third-party/XNNPACK is missing or a dangling link.${_dep_help}")
endif()

find_program(FLATC_EXECUTABLE flatc)
if(NOT FLATC_EXECUTABLE)
  message(FATAL_ERROR "flatc not found (pass -DFLATC_EXECUTABLE=... or put it on PATH).${_dep_help}")
endif()

# ---- model -------------------------------------------------------------------
set(EXEC_MODEL "mnist_mlp_fp32" CACHE STRING "Model: a name in models/ (without .pte) or a path to a .pte")
if(EXEC_MODEL MATCHES "\\.pte$")
  set(_pte "${EXEC_MODEL}")
else()
  set(_pte "${CMAKE_CURRENT_SOURCE_DIR}/models/${EXEC_MODEL}.pte")
endif()
if(NOT EXISTS "${_pte}")
  message(FATAL_ERROR "model '${_pte}' not found (EXEC_MODEL=${EXEC_MODEL}); regenerate with scripts/gen_model.py")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_pte}")

# .pte -> C array in section network_model_sec (placed by custom-sections.ld), like exeggutorch's
# pte_to_header.py. Done at configure time with CMake only, so the build needs no extra python.
file(READ "${_pte}" _hex HEX)
string(LENGTH "${_hex}" _hexlen)
math(EXPR _pte_size "${_hexlen} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/model_pte.c"
  "/* generated from ${_pte} */\n"
  "__attribute__((section(\"network_model_sec\"), aligned(16))) unsigned char model_pte[] = {${_bytes}};\n"
  "unsigned int model_pte_size = ${_pte_size};\n")

# ---- ExecuTorch view: a symlink mirror in the build dir --------------------------
# Two reasons not to build straight from EXECUTORCH_DIR:
#  * its CMake looks for XNNPACK at backends/xnnpack/third-party/XNNPACK; the mirror lets
#    XNNPACK_DIR replace that one entry without touching EXECUTORCH_DIR;
#  * flatcc's CMake writes its libraries and a flatcc binary into its own source dir
#    (third-party/flatcc/{lib,bin}); in the mirror that directory is a real one, so
#    EXECUTORCH_DIR stays read-only.
# Everything else in the mirror is a symlink back to EXECUTORCH_DIR.
function(_mirror_dir SRC DST)  # ARGN: entries NOT to link
  file(MAKE_DIRECTORY "${DST}")
  file(GLOB _entries RELATIVE "${SRC}" "${SRC}/*" "${SRC}/.[A-Za-z]*")
  foreach(_e IN LISTS _entries)
    if(NOT _e IN_LIST ARGN)
      file(CREATE_LINK "${SRC}/${_e}" "${DST}/${_e}" SYMBOLIC)
    endif()
  endforeach()
endfunction()

set(_view "${CMAKE_CURRENT_BINARY_DIR}/et_view/executorch")
file(REMOVE_RECURSE "${CMAKE_CURRENT_BINARY_DIR}/et_view")
_mirror_dir("${EXECUTORCH_DIR}" "${_view}" backends third-party)
_mirror_dir("${EXECUTORCH_DIR}/backends" "${_view}/backends" xnnpack)
_mirror_dir("${EXECUTORCH_DIR}/backends/xnnpack" "${_view}/backends/xnnpack" third-party)
_mirror_dir("${EXECUTORCH_DIR}/backends/xnnpack/third-party" "${_view}/backends/xnnpack/third-party" XNNPACK)
if(XNNPACK_DIR)
  file(CREATE_LINK "${XNNPACK_DIR}" "${_view}/backends/xnnpack/third-party/XNNPACK" SYMBOLIC)
else()
  file(CREATE_LINK "${EXECUTORCH_DIR}/backends/xnnpack/third-party/XNNPACK" "${_view}/backends/xnnpack/third-party/XNNPACK" SYMBOLIC)
endif()
_mirror_dir("${EXECUTORCH_DIR}/third-party" "${_view}/third-party" flatcc)
# bin/ and lib/ are flatcc's build outputs: never link to them, even if a stale copy exists
_mirror_dir("${EXECUTORCH_DIR}/third-party/flatcc" "${_view}/third-party/flatcc" bin lib)
set(ET_DIR_PATH "${_view}")

set(ET_BUILD_DIR_PATH "${CMAKE_CURRENT_BINARY_DIR}/executorch" CACHE PATH "ExecuTorch build dir")
set(ET_INCLUDE_PATH "${ET_DIR_PATH}/.." CACHE PATH "ExecuTorch headers (parent of the executorch dir)")

add_compile_definitions(_POSIX_C_SOURCE=200809L __POSIX_VISIBLE=200809L)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(penguin_executorch_runner)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON CACHE BOOL "")
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED OFF CACHE BOOL "")
set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "")
set(EXECUTORCH_BUILD_CPUINFO OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_PTHREADPOOL OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_GFLAGS OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_HOST_TARGETS OFF CACHE BOOL "")

set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "")
set(XNNPACK_ENABLE_RISCV_GEMMINI OFF CACHE BOOL "Build XNNPACK with RISC-V Gemmini micro-kernels")
set(XNNPACK_ENABLE_RISCV_VECTOR ON CACHE BOOL "Build XNNPACK with RISC-V Vector micro-kernels")
# Zvfbfmin/Zvfbfa kernels are used for bf16 models; Penguin has both (D17), Spike needs them in
# its ISA string (the Penguin board adds zfbfmin/zvfbfa). Set OFF if your core lacks them.
set(XNNPACK_ENABLE_RISCV_ZVFBFMIN ON CACHE BOOL "Build XNNPACK with RISC-V Zvfbfmin micro-kernels")
set(XNNPACK_ENABLE_RISCV_ZVFBFA ON CACHE BOOL "Build XNNPACK with RISC-V Zvfbfa micro-kernels")
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Build XNNPack unit tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Build XNNPack benchmarks")
set(FLATCC_RTONLY ON CACHE BOOL "enable build of runtime library of flatbuffers only")

add_subdirectory(${ET_DIR_PATH} ${ET_BUILD_DIR_PATH})

# ExecuTorch's third-party libs are built with the app's Zephyr include/flag set.
add_library(compile_commands INTERFACE)
target_include_directories(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_OPTIONS>)
target_compile_definitions(compile_commands INTERFACE $<TARGET_PROPERTY:app,INTERFACE_COMPILE_DEFINITIONS>)
add_dependencies(compile_commands zephyr_interface)
add_dependencies(compile_commands zephyr_property_target)
target_link_libraries(pthreadpool PUBLIC compile_commands)

target_link_libraries(app PRIVATE
  extension_runner_util executorch executorch_core portable_ops_lib portable_kernels
  xnnpack_backend XNNPACK xnnpack-microkernels-prod extension_threadpool
)

target_include_directories(app PRIVATE ${ET_INCLUDE_PATH} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_sources(app PRIVATE src/riscv_executor_runner.cpp ${CMAKE_CURRENT_BINARY_DIR}/model_pte.c)
zephyr_linker_sources(DATA_SECTIONS custom-sections.ld)
