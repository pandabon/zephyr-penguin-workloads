# SPDX-License-Identifier: Apache-2.0
#
# ExecuTorch (+ XNNPACK RVV kernels) runner for the Penguin SoC, built against the NEW forks
# (ExecuTorch v1.5.1 + XNNPACK 92a7ad501b95, branch `penguin`). Runs any .pte; can embed input
# tensors and an expected output and print a PASS/FAIL comparison (used for SmolVLA expert layers).
#
# OPT-IN: this app needs two external source trees that this repo does NOT contain:
#   EXECUTORCH_DIR  an ExecuTorch checkout (a directory named "executorch")
#   XNNPACK_DIR     an XNNPACK checkout (optional if EXECUTORCH_DIR already has a working
#                   backends/xnnpack/third-party/XNNPACK)
# plus `flatc` and a python3 with torchgen/pyyaml for ExecuTorch's codegen (PYTHON_EXECUTABLE,
# FLATC_EXECUTABLE or on PATH). Without them the configure step stops with an explanation.
# Nothing is downloaded and nothing in the given trees is modified.
#
# Model: -DEXEC_MODEL=/path/to/x.pte   (required; this app ships no models)
# Optional: -DEXEC_INPUTS=/path/in.bin    all input tensors concatenated (float32, in method order)
#           -DEXEC_EXPECTED=/path/exp.bin expected output 0 (same dtype as the model output)
#           -DEXEC_TOL=1e-4               pass if relative L2 error < EXEC_TOL
#           -DET_METHOD_POOL_SIZE=bytes -DET_TEMP_POOL_SIZE=bytes   (see src/riscv_executor_runner.cpp)
#
# Derived from executorch/mnist in this repo (same CMake opt-in/mirror scheme; adds the embedded
# input/expected arrays). That app is derived from the following:
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

set(_dep_help "\n  This app is an opt-in workload: ExecuTorch and XNNPACK are not part of this repo.\n  See the ExecuTorch section of the repo README.md. Example:\n    export EXECUTORCH_DIR=/path/to/executorch      # directory is named 'executorch'\n    export XNNPACK_DIR=/path/to/XNNPACK\n    export PATH=/path/to/env/bin:$PATH             # flatc + python3 with torchgen, pyyaml\n")
if(NOT EXECUTORCH_DIR OR NOT EXISTS "${EXECUTORCH_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "EXECUTORCH_DIR is not set or has no CMakeLists.txt (got '${EXECUTORCH_DIR}').${_dep_help}")
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
set(EXEC_MODEL "" CACHE STRING "Path to a .pte model")
set(EXEC_INPUTS "" CACHE STRING "Optional .bin with all input tensors concatenated")
set(EXEC_EXPECTED "" CACHE STRING "Optional .bin with the expected output tensor")
set(EXEC_TOL "1e-4" CACHE STRING "PASS if relative L2 error < this")
if(NOT EXEC_MODEL MATCHES "\\.pte$" OR NOT EXISTS "${EXEC_MODEL}")
  message(FATAL_ERROR "EXEC_MODEL must be the path of an existing .pte (got '${EXEC_MODEL}')")
endif()
set(_pte "${EXEC_MODEL}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_pte}")

# A file -> C array in section SEC (placed by custom-sections.ld). Done at configure time with
# CMake only (file(READ ... HEX)), so the build needs no extra python.
function(_embed_bin FILE SYM SEC OUT)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FILE}")
  file(READ "${FILE}" _hex HEX)
  string(LENGTH "${_hex}" _hexlen)
  math(EXPR _size "${_hexlen} / 2")
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
  file(WRITE "${OUT}"
    "/* generated from ${FILE} */\n"
    "__attribute__((section(\"${SEC}\"), aligned(16))) const unsigned char ${SYM}[] = {${_bytes}};\n"
    "const unsigned int ${SYM}_size = ${_size};\n")
endfunction()

# like exeggutorch's pte_to_header.py
_embed_bin("${_pte}" model_pte network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/model_pte.c")
set(_embedded ${CMAKE_CURRENT_BINARY_DIR}/model_pte.c)
if(EXEC_INPUTS)
  if(NOT EXISTS "${EXEC_INPUTS}")
    message(FATAL_ERROR "EXEC_INPUTS '${EXEC_INPUTS}' not found")
  endif()
  _embed_bin("${EXEC_INPUTS}" exec_inputs network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/exec_inputs.c")
  list(APPEND _embedded ${CMAKE_CURRENT_BINARY_DIR}/exec_inputs.c)
  add_compile_definitions(EXEC_HAVE_INPUTS=1)
endif()
if(EXEC_EXPECTED)
  if(NOT EXEC_INPUTS OR NOT EXISTS "${EXEC_EXPECTED}")
    message(FATAL_ERROR "EXEC_EXPECTED needs an existing file and EXEC_INPUTS")
  endif()
  _embed_bin("${EXEC_EXPECTED}" exec_expected network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/exec_expected.c")
  list(APPEND _embedded ${CMAKE_CURRENT_BINARY_DIR}/exec_expected.c)
  add_compile_definitions(EXEC_HAVE_EXPECTED=1)
endif()
add_compile_definitions(EXEC_TOL=${EXEC_TOL})
foreach(_p ET_METHOD_POOL_SIZE ET_TEMP_POOL_SIZE)
  if(DEFINED ${_p})
    add_compile_definitions(${_p}=${${_p}})
  endif()
endforeach()

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
# XNNPACK: no mmap/mprotect on Zephyr (common.h lets us override XNN_HAS_MMAP; the 1.0.1 fork
# hard-coded it to 0). cpuinfo's log.c wants STDERR_FILENO, which Zephyr's <unistd.h> lacks.
add_compile_definitions(XNN_HAS_MMAP=0 STDERR_FILENO=2)
# std::mutex shim (the SDK's libstdc++ has none), first on every target's include path.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src/zephyr_compat)

# Zephyr's python.cmake overwrites PYTHON_EXECUTABLE with the west python (which has no torchgen);
# remember the one the user passed for ExecuTorch's codegen and restore it after Zephyr is loaded.
set(_et_python "${PYTHON_EXECUTABLE}")
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(penguin_executorch_runner)
if(_et_python)
  set(PYTHON_EXECUTABLE "${_et_python}")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON CACHE BOOL "")
set(EXECUTORCH_BUILD_KERNELS_QUANTIZED OFF CACHE BOOL "")
set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "")
# ExecuTorch 1.5.1's preset REQUIRES cpuinfo + pthreadpool whenever XNNPACK is on (the 1.0.1 fork of
# the older app could turn them OFF). cpuinfo has no Zephyr support: its CMake builds an empty
# stub library for an unknown OS, and we never call it (see the Zephyr guard in
# extension/threadpool/threadpool.cpp of the fork).
set(EXECUTORCH_BUILD_CPUINFO ON CACHE BOOL "")
set(EXECUTORCH_BUILD_PTHREADPOOL ON CACHE BOOL "")
set(EXECUTORCH_BUILD_GFLAGS OFF CACHE BOOL "")
set(EXECUTORCH_BUILD_HOST_TARGETS OFF CACHE BOOL "")

set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "")
# cpuinfo has no Zephyr backend: with XNN_ENABLE_CPUINFO=1 XNNPACK calls into the stub library
# and aborts at xnn_initialize(). The option is added by our XNNPACK fork (CMakeLists.txt).
set(XNNPACK_ENABLE_CPUINFO OFF CACHE BOOL "Do not use cpuinfo in XNNPACK")
set(XNNPACK_ENABLE_RISCV_GEMMINI OFF CACHE BOOL "Build XNNPACK with RISC-V Gemmini micro-kernels")
set(XNNPACK_ENABLE_RISCV_VECTOR ON CACHE BOOL "Build XNNPACK with RISC-V Vector micro-kernels")
# Zvfbfmin/Zvfbfa kernels are used for bf16 models; Penguin has both (D17), Spike needs them in
# its ISA string (the BSP board adds zfbfmin/zvfbfa). Set OFF if your core lacks them.
set(XNNPACK_ENABLE_RISCV_ZVFBFMIN ON CACHE BOOL "Build XNNPACK with RISC-V Zvfbfmin micro-kernels")
set(XNNPACK_ENABLE_RISCV_ZVFBFA ON CACHE BOOL "Build XNNPACK with RISC-V Zvfbfa micro-kernels")
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Build XNNPack unit tests")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Build XNNPack benchmarks")
set(FLATCC_RTONLY ON CACHE BOOL "enable build of runtime library of flatbuffers only")

# Zephyr hands its include paths, definitions and ISA options (-march/-mabi/-mcmodel) to its own
# targets (zephyr_interface/app), not to CMAKE_<LANG>_FLAGS. ExecuTorch's libraries are a separate
# project(), so without this they are compiled for the compiler's default RISC-V target (32-bit
# objects: "ELFCLASS32 incompatible") and against the SDK's default libc headers, whose uint64_t
# differs from picolibc's (undefined reference to vlogf(..., unsigned long long, ...)). So give every
# target created below the app's flag set. (The 1.0.1 fork instead edited its top-level
# CMakeLists.txt to copy -march/-mabi from the parent, commit 0ba9c549b by Dima Nikiforov, and
# linked pthreadpool against a helper target; this does both for all targets.)
include_directories($<TARGET_PROPERTY:app,INTERFACE_INCLUDE_DIRECTORIES>)
add_compile_options($<TARGET_PROPERTY:app,INTERFACE_COMPILE_OPTIONS>)
add_compile_definitions($<TARGET_PROPERTY:app,INTERFACE_COMPILE_DEFINITIONS>)

add_subdirectory(${ET_DIR_PATH} ${ET_BUILD_DIR_PATH})

# The Zephyr headers we added to every target include generated ones (zephyr/syscall_list.h,
# autoconf.h): make every ExecuTorch library wait for them, or a parallel build can compile
# pthreadpool & co. before they exist. (Same job the old helper target's add_dependencies() did.)
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
_depend_on_zephyr_headers("${ET_BUILD_DIR_PATH}")

# ExecuTorch's third-party/CMakeLists.txt add_subdirectory()s gflags unconditionally, even with
# EXECUTORCH_BUILD_GFLAGS=OFF. We never link it (and it does not build for Zephyr: strcasecmp),
# so take its targets out of the default build.
foreach(_t gflags_static gflags_nothreads_static gflags_shared gflags_nothreads_shared)
  if(TARGET ${_t})
    set_target_properties(${_t} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endforeach()

# extension_data_loader is built by default by ExecuTorch but the runner uses BufferDataLoader (header
# only) and never links it; its FileDataLoader needs pread(), which Zephyr's <unistd.h> lacks.
if(TARGET extension_data_loader)
  set_target_properties(extension_data_loader PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

target_link_libraries(app PRIVATE
  extension_runner_util executorch executorch_core portable_ops_lib portable_kernels
  xnnpack_backend XNNPACK xnnpack-microkernels-prod extension_threadpool
)

# XNNPACK's internal headers (src/xnnpack/hardware-config.h), used to report the detected RISC-V config.
target_include_directories(app PRIVATE ${ET_DIR_PATH}/backends/xnnpack/third-party/XNNPACK)
target_include_directories(app PRIVATE ${ET_INCLUDE_PATH} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_sources(app PRIVATE src/riscv_executor_runner.cpp ${_embedded})
zephyr_linker_sources(DATA_SECTIONS custom-sections.ld)
