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
#           -DEXEC_EMBED=incbin           embed the files with an assembler .incbin instead of a generated C array
#                                         (needed for big models: a 200 MB .pte as a C array is ~1 GB of CMake string work
#                                         and C source; .incbin is a plain file copy at assemble time)
#           -DEXEC_STEP_DATA=/path/a.bin -DEXEC_STEP_EXPECTED=/path/b.bin
#                                         STEP MODE (SmolVLA action step: .pte with methods kv_proj + denoise_step, written by
#                                         smolvla-explore/export_expert_step.py): run kv_proj once and denoise_step 10 times,
#                                         compare the final actions; EXEC_INPUTS / EXEC_EXPECTED are not used
#           -DEXEC_CHAIN_DATA=/path/a.bin -DEXEC_CHAIN_EXPECTED=/path/b.bin
#                                         CHAIN MODE (SmolVLA mini chain: .pte with methods vision, vlm, kv_proj, denoise_step, written by
#                                         smolvla-explore/export_mini_chain.py): run them once in that order on the target (denoise_step
#                                         looped), print instructions per method and check every stage; not combinable with step mode
#           -DEXEC_PTD=/path/x.ptd -DEXEC_PTD_MODE=embed|prng
#                                         EXTERNAL WEIGHTS: the .pte was exported with its XNNPACK weights in a separate .ptd (ExecuTorch
#                                         flat_tensor data file, written by smolvla-explore/fpga-prep/ext_const.py + export_ext_all.py).
#                                         embed (default): the whole real .ptd goes into the image (correctness runs, same output as the
#                                         normal .pte). prng: only the .ptd header goes into the image; the weight bytes are a .noinit
#                                         buffer the runner fills with a PRNG at boot (tame values, timing only: the output check is
#                                         skipped), so the weights are not sent over the UART. EXEC_EXPECTED is ignored in prng mode.
#           -DEXEC_WEIGHT_CACHE=ON        build ExecuTorch's XNNPACK weights cache (packing at load_method() goes through it)
#           -DEXEC_PREPACK=ON|verify      with EXEC_PTD (embed) and EXEC_EMBED=incbin: pack the linears' weights on the BUILD machine
#                                         (prepack/: host packer from XNNPACK_DIR's reference packers, target RVV vlenb 32), link the
#                                         blob in and serve it through the weights cache, so load_method() does not pack them.
#                                         verify: link the blob but pack on the board anyway and compare byte for byte (one-time check)
#           -DEXEC_SELECT_OPS=OFF         link the whole portable kernel library. Default ON: link only the portable kernels the .pte
#                                         uses, in any of its methods (ExecuTorch's selective build, EXECUTORCH_SELECT_OPS_MODEL=EXEC_MODEL):
#                                         ~4.6 MB smaller image, i.e. ~7 min less UART load at 115200 baud. Turn it off only to run
#                                         a .pte that was not the one built in (the runner fails with a missing-kernel error otherwise)
#           -DEXEC_FP8_LINEAR=ON          build the ExecuTorch custom op penguin::linear_fp8.out (src/ops/linear_fp8_out.cpp) for a .pte exported with
#                                         fp8 weights (smolvla-explore/export_mini_chain_fp8.py). It calls the XNNPACK fork's
#                                         xnn_bf16_qf8w_f32_gemm_minmax_ukernel_5x4v__rvvzvfofp8, so the fork must have that kernel. Default OFF.
#                                         -DFP8_MAX_NPAD=n: largest padded out_features the op's static scratch holds (default 4096)
#           -DEXEC_REPEAT=N, -DEXEC_PC_SAMPLE=us, -DEXEC_XNN_PROFILE=ON   see "opt-in instrumentation" below
#           -DEXEC_ERR_STATS=ON           one ERRSTATS line per checked output: abs/rel error mean, min, max, RMSE, PSNR vs the reference
#           -DEXEC_ETDUMP=ON [-DEXEC_ETDUMP_BUF=bytes] [-DEXEC_CPU_HZ=hz]
#                                         ExecuTorch's OFFICIAL profiling: event tracer + ETDump, in plain, step and chain mode (one dump holds all methods
#                                         and runs; the order of the runs is printed as "ETDUMP RUNS ..."). After the run the
#                                         runner prints the ETDump base64 between "ETDUMP BEGIN" / "ETDUMP END"; the host script
#                                         tools/etdump_inspect.py turns it into a per-operator table with ExecuTorch's Inspector
#                                         (and an ETRecord from the export, if you have one). Times are mcycle ticks. Off by default.
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

set(EXEC_EMBED "hex" CACHE STRING "How files are embedded: hex (generated C array) or incbin (assembler .incbin)")
set(EXEC_STEP_DATA "" CACHE STRING "Step mode: runner inputs (.step_data.bin)")
set(EXEC_STEP_EXPECTED "" CACHE STRING "Step mode: golden outputs (.step_expected.bin)")
set(EXEC_CHAIN_DATA "" CACHE STRING "Chain mode: runner inputs (.chain_data.bin)")
set(EXEC_CHAIN_EXPECTED "" CACHE STRING "Chain mode: golden stages (.chain_expected.bin)")

# A file -> C array in section SEC (placed by custom-sections.ld). Done at configure time with
# CMake only (file(READ ... HEX)), so the build needs no extra python.
# Fine for small files (MNIST, one expert layer of 12 MB); for a 200 MB model the hex string alone is 400 MB and
# the C source ~1.2 GB (6 characters per byte), all built in CMake strings. Use EXEC_EMBED=incbin for those.
# Optional 5th/6th arguments (all _embed_* functions): LIMIT = embed only the first LIMIT bytes of FILE (default: all; 0 = all),
# ALIGN = alignment of the symbol in bytes (default 16).
function(_embed_hex FILE SYM SEC OUT)
  set(_limit 0)
  set(_align 16)
  if(ARGC GREATER 4)
    set(_limit ${ARGV4})
  endif()
  if(ARGC GREATER 5)
    set(_align ${ARGV5})
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FILE}")
  if(_limit GREATER 0)
    file(READ "${FILE}" _hex HEX LIMIT ${_limit})
  else()
    file(READ "${FILE}" _hex HEX)
  endif()
  string(LENGTH "${_hex}" _hexlen)
  math(EXPR _size "${_hexlen} / 2")
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
  file(WRITE "${OUT}"
    "/* generated from ${FILE} */\n"
    "__attribute__((section(\"${SEC}\"), aligned(${_align}))) const unsigned char ${SYM}[] = {${_bytes}};\n"
    "const unsigned int ${SYM}_size = ${_size};\n")
endfunction()

# The same two symbols from an assembler file: `.incbin` makes the assembler copy the file into the section, so
# nothing is read or converted at configure time. Same section flags GCC gives a `const` array ("a" = allocated,
# read-only), 16-byte aligned, and a 32-bit `<SYM>_size` word (the C array version has `const unsigned int`).
# The object depends on the file (OBJECT_DEPENDS), so changing the model rebuilds it.
function(_embed_incbin FILE SYM SEC OUT)
  set(_incbin "\"${FILE}\"")
  set(_align 16)
  if(ARGC GREATER 4 AND ARGV4 GREATER 0)
    set(_incbin "\"${FILE}\",0,${ARGV4}")   # GNU as: .incbin "file", skip, count
  endif()
  if(ARGC GREATER 5)
    set(_align ${ARGV5})
  endif()
  file(WRITE "${OUT}"
    "/* generated; embeds ${FILE} */\n"
    "\t.section ${SEC},\"a\",@progbits\n"
    "\t.balign ${_align}\n"
    "\t.global ${SYM}\n"
    "${SYM}:\n"
    "\t.incbin ${_incbin}\n"
    "${SYM}_end:\n"
    "\t.balign 4\n"
    "\t.global ${SYM}_size\n"
    "${SYM}_size:\n"
    "\t.word ${SYM}_end - ${SYM}\n")
  set_source_files_properties("${OUT}" PROPERTIES OBJECT_DEPENDS "${FILE}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FILE}")
endfunction()

# _embed_bin(FILE SYM SEC OUT_BASE [LIMIT [ALIGN]]): writes OUT_BASE.c or OUT_BASE.S depending on EXEC_EMBED; sets _embed_out to the file.
function(_embed_bin FILE SYM SEC OUT_BASE)
  set(_opt "")
  if(ARGC GREATER 4)
    set(_opt ${ARGV4})
  endif()
  if(ARGC GREATER 5)
    list(APPEND _opt ${ARGV5})
  endif()
  if(NOT _opt)
    set(_opt 0 16)
  endif()
  if(EXEC_EMBED STREQUAL "incbin")
    _embed_incbin("${FILE}" "${SYM}" "${SEC}" "${OUT_BASE}.S" ${_opt})
    set(_embed_out "${OUT_BASE}.S" PARENT_SCOPE)
  elseif(EXEC_EMBED STREQUAL "hex")
    _embed_hex("${FILE}" "${SYM}" "${SEC}" "${OUT_BASE}.c" ${_opt})
    set(_embed_out "${OUT_BASE}.c" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "EXEC_EMBED must be hex or incbin (got '${EXEC_EMBED}')")
  endif()
endfunction()

# like exeggutorch's pte_to_header.py
_embed_bin("${_pte}" model_pte network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/model_pte")
set(_embedded ${_embed_out})

# ---- external weights (opt-in): EXEC_PTD, EXEC_PTD_MODE ----------------------------
# A .ptd is ExecuTorch's FlatTensor file (extension/flat_tensor, forks/executorch penguin-1.6 85600f76e3): a 40-byte header at byte 8
# ("FH01", header length u32, flatbuffer_offset u64, flatbuffer_size u64, segment_base_offset u64, segment_data_size u64; see
# extension/flat_tensor/serialize/flat_tensor_header.h), the key table (flatbuffer), then the weight bytes from segment_base_offset.
# embed: the whole file is embedded (symbol ptd_embedded). prng: only bytes [0, segment_base_offset) are embedded (symbol ptd_header);
# the runner makes a .noinit buffer of EXEC_PTD_SIZE bytes and fills the rest with random values at boot.
set(EXEC_PTD "" CACHE STRING "Optional .ptd with the model's external weights (see the header of this file)")
set(EXEC_PTD_MODE "embed" CACHE STRING "embed (real weights in the image) or prng (generated on the target, timing only)")
option(EXEC_SELECT_OPS "Link only the portable kernels used by EXEC_MODEL (ExecuTorch selective build)" ON)
if(EXEC_PTD)
  if(NOT EXEC_PTD MATCHES "\\.ptd$" OR NOT EXISTS "${EXEC_PTD}")
    message(FATAL_ERROR "EXEC_PTD must be the path of an existing .ptd (got '${EXEC_PTD}')")
  endif()
  if(EXEC_STEP_DATA OR EXEC_CHAIN_DATA)
    message(WARNING "EXEC_PTD with step/chain mode is untested: the runner passes the data map to every load_method(), but no such .pte was exported")
  endif()
  file(SIZE "${EXEC_PTD}" _ptd_size)
  file(READ "${EXEC_PTD}" _ptd_hdr HEX OFFSET 8 LIMIT 40)
  string(SUBSTRING "${_ptd_hdr}" 0 8 _ptd_magic)
  if(NOT _ptd_magic STREQUAL "46483031")   # "FH01"
    message(FATAL_ERROR "${EXEC_PTD} is not a FlatTensor .ptd (no FH01 magic at byte 8)")
  endif()
  string(SUBSTRING "${_ptd_hdr}" 48 16 _ptd_base_le)   # segment_base_offset, u64 little endian, at header byte 24
  string(REGEX MATCHALL "[0-9a-f][0-9a-f]" _ptd_base_bytes "${_ptd_base_le}")
  list(REVERSE _ptd_base_bytes)
  list(JOIN _ptd_base_bytes "" _ptd_base_be)
  math(EXPR _ptd_base "0x${_ptd_base_be}")
  if(_ptd_base LESS 48 OR _ptd_base GREATER _ptd_size)
    message(FATAL_ERROR "${EXEC_PTD}: segment_base_offset ${_ptd_base} is not inside the file (${_ptd_size} bytes)")
  endif()
  if(EXEC_PTD_MODE STREQUAL "embed")
    # 128-byte aligned like the segments inside the .ptd (serialize.py aligns them to >= 128)
    _embed_bin("${EXEC_PTD}" ptd_embedded network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/ptd_embedded" 0 128)
    add_compile_definitions(EXEC_PTD_EMBED=1)
  elseif(EXEC_PTD_MODE STREQUAL "prng")
    _embed_bin("${EXEC_PTD}" ptd_header network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/ptd_header" ${_ptd_base})
    add_compile_definitions(EXEC_PTD_PRNG=1 EXEC_PTD_SIZE=${_ptd_size})
    if(EXEC_EXPECTED)
      message(STATUS "EXEC_PTD_MODE=prng: EXEC_EXPECTED is ignored (random weights, no output check)")
      set(EXEC_EXPECTED "")
    endif()
  else()
    message(FATAL_ERROR "EXEC_PTD_MODE must be embed or prng (got '${EXEC_PTD_MODE}')")
  endif()
  list(APPEND _embedded ${_embed_out})
  add_compile_definitions(EXEC_HAVE_PTD=1)
  message(STATUS "external weights: ${EXEC_PTD} (${_ptd_size} bytes, header+keys ${_ptd_base} bytes), mode ${EXEC_PTD_MODE}")
elseif(EXEC_PTD_MODE STREQUAL "prng")
  message(FATAL_ERROR "EXEC_PTD_MODE=prng needs EXEC_PTD")
endif()

# ---- weights cache and build-time prepacking (opt-in): EXEC_WEIGHT_CACHE, EXEC_PREPACK -----------------------------
# EXEC_WEIGHT_CACHE=ON builds ExecuTorch's XNNPACK weights cache (EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE): XNNPACK still packs at
# load_method(), into the heap, through the cache. EXEC_PREPACK=ON packs the linears on the BUILD machine instead: prepack/prepack.py
# reads EXEC_MODEL + EXEC_PTD, prepack/xnn_prepack (built here with the host c++ from XNNPACK_DIR's reference packers) packs every
# fully connected weight with the target's parameters (RISC-V RVV, vlenb 32), and the blob (XPWC v2, the cache's own file format) is
# linked in with .incbin; the runner registers it before load_method() and XNNPACK reads it in place. EXEC_PREPACK=verify links the
# same blob but makes the board pack anyway and compare byte for byte (one-time check of the host packer; "PREPACK VERIFY" lines).
# Needs EXEC_PTD (embed) and EXEC_EMBED=incbin. Weights the blob does not cover are packed at load as before.
option(EXEC_WEIGHT_CACHE "Build ExecuTorch's XNNPACK weights cache (packing at load goes through it)" OFF)
set(EXEC_PREPACK "OFF" CACHE STRING "OFF, ON (link weights packed at build time) or verify (link them, pack on the board too, compare)")
if(NOT EXEC_PREPACK STREQUAL "OFF")
  if(NOT EXEC_PREPACK MATCHES "^(ON|verify)$")
    message(FATAL_ERROR "EXEC_PREPACK must be OFF, ON or verify (got '${EXEC_PREPACK}')")
  endif()
  if(NOT EXEC_PTD OR NOT EXEC_PTD_MODE STREQUAL "embed" OR NOT EXEC_EMBED STREQUAL "incbin")
    message(FATAL_ERROR "EXEC_PREPACK needs EXEC_PTD with EXEC_PTD_MODE=embed and EXEC_EMBED=incbin")
  endif()
  set(EXEC_WEIGHT_CACHE ON CACHE BOOL "" FORCE)
  if(XNNPACK_DIR)
    set(_pp_xnn "${XNNPACK_DIR}")
  else()
    set(_pp_xnn "${EXECUTORCH_DIR}/backends/xnnpack/third-party/XNNPACK")
  endif()
  set(_pp_src "${CMAKE_CURRENT_SOURCE_DIR}/prepack")
  set(_pp_out "${CMAKE_CURRENT_BINARY_DIR}/prepack")
  set(_pp_tp "${EXECUTORCH_DIR}/backends/xnnpack/third-party")
  set(_pp_inc -I${_pp_xnn} -I${_pp_xnn}/include -I${_pp_tp}/pthreadpool/include -I${_pp_tp}/FP16/include -I${_pp_tp}/FXdiv/include)
  find_program(_pp_cxx NAMES c++ g++ REQUIRED)   # the HOST compiler: the helper runs on the build machine
  # xnn_prepack_support.cc is compiled as for RISC-V (x86 macros off, __riscv on) so XNNPACK's common.h gives the target's
  # XNN_ALLOCATION_ALIGNMENT; see the header of that file.
  add_custom_command(OUTPUT "${_pp_out}/xnn_prepack"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_pp_out}"
    COMMAND ${_pp_cxx} -std=c++17 -O2 -DXNN_LOG_LEVEL=0 -U__x86_64__ -U__amd64__ -U__x86_64 -U__amd64 -D__riscv=1 -c ${_pp_inc}
            "${_pp_src}/xnn_prepack_support.cc" -o "${_pp_out}/xnn_prepack_support.o"
    COMMAND ${_pp_cxx} -std=c++17 -O2 -DXNN_LOG_LEVEL=0 ${_pp_inc} "${_pp_src}/xnn_prepack.cc" "${_pp_xnn}/src/reference/packing.cc"
            "${_pp_out}/xnn_prepack_support.o" -o "${_pp_out}/xnn_prepack"
    DEPENDS "${_pp_src}/xnn_prepack.cc" "${_pp_src}/xnn_prepack_support.cc" "${_pp_xnn}/src/reference/packing.cc"
    COMMENT "prepack: building the host packer xnn_prepack" VERBATIM)
  set(_pp_flatc "")
  if(FLATC_EXECUTABLE)
    set(_pp_flatc --flatc "${FLATC_EXECUTABLE}")
  endif()
  add_custom_command(OUTPUT "${_pp_out}/prepacked.xpwc" "${_pp_out}/prepacked.json"
    COMMAND "${PYTHON_EXECUTABLE}" "${_pp_src}/prepack.py" --pte "${_pte}" --ptd "${EXEC_PTD}" --xnnpack-dir "${_pp_xnn}"
            --helper "${_pp_out}/xnn_prepack" ${_pp_flatc} --out "${_pp_out}/prepacked.xpwc" --json "${_pp_out}/prepacked.json"
    DEPENDS "${_pp_out}/xnn_prepack" "${_pp_src}/prepack.py" "${_pte}" "${EXEC_PTD}"
    COMMENT "prepack: packing the XNNPACK weights for the target on the build machine" VERBATIM)
  add_custom_target(prepacked_blob DEPENDS "${_pp_out}/prepacked.xpwc")
  # 64-byte aligned: the cache wants every packed entry 64-byte aligned in memory (the blob's offsets are multiples of 64).
  _embed_bin("${_pp_out}/prepacked.xpwc" prepacked_blob network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/prepacked_blob" 0 64)
  set_source_files_properties("${_embed_out}" PROPERTIES GENERATED_DEPENDS TRUE)
  list(APPEND _embedded ${_embed_out})
  if(EXEC_PREPACK STREQUAL "verify")
    add_compile_definitions(EXEC_PREPACK=1 EXEC_PREPACK_VERIFY=1)
  else()
    add_compile_definitions(EXEC_PREPACK=1 EXEC_PREPACK_VERIFY=0)
  endif()
  message(STATUS "prepacked weights: built from ${_pte} + ${EXEC_PTD}, mode ${EXEC_PREPACK}")
endif()
if(EXEC_WEIGHT_CACHE)
  add_compile_definitions(EXEC_WEIGHT_CACHE=1)
endif()
if(EXEC_STEP_DATA)
  if(NOT EXISTS "${EXEC_STEP_DATA}" OR NOT EXISTS "${EXEC_STEP_EXPECTED}")
    message(FATAL_ERROR "step mode needs existing EXEC_STEP_DATA and EXEC_STEP_EXPECTED")
  endif()
  _embed_bin("${EXEC_STEP_DATA}" step_data network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/step_data")
  list(APPEND _embedded ${_embed_out})
  _embed_bin("${EXEC_STEP_EXPECTED}" step_expected network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/step_expected")
  list(APPEND _embedded ${_embed_out})
  add_compile_definitions(EXEC_STEP_MODE=1)
endif()
if(EXEC_CHAIN_DATA)
  if(EXEC_STEP_DATA)
    message(FATAL_ERROR "EXEC_CHAIN_DATA and EXEC_STEP_DATA are separate modes; give one")
  endif()
  if(NOT EXISTS "${EXEC_CHAIN_DATA}" OR NOT EXISTS "${EXEC_CHAIN_EXPECTED}")
    message(FATAL_ERROR "chain mode needs existing EXEC_CHAIN_DATA and EXEC_CHAIN_EXPECTED")
  endif()
  _embed_bin("${EXEC_CHAIN_DATA}" chain_data network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/chain_data")
  list(APPEND _embedded ${_embed_out})
  _embed_bin("${EXEC_CHAIN_EXPECTED}" chain_expected network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/chain_expected")
  list(APPEND _embedded ${_embed_out})
  add_compile_definitions(EXEC_CHAIN_MODE=1)
endif()
if(EXEC_INPUTS)
  if(NOT EXISTS "${EXEC_INPUTS}")
    message(FATAL_ERROR "EXEC_INPUTS '${EXEC_INPUTS}' not found")
  endif()
  _embed_bin("${EXEC_INPUTS}" exec_inputs network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/exec_inputs")
  list(APPEND _embedded ${_embed_out})
  add_compile_definitions(EXEC_HAVE_INPUTS=1)
endif()
if(EXEC_EXPECTED)
  if(NOT EXEC_INPUTS OR NOT EXISTS "${EXEC_EXPECTED}")
    message(FATAL_ERROR "EXEC_EXPECTED needs an existing file and EXEC_INPUTS")
  endif()
  _embed_bin("${EXEC_EXPECTED}" exec_expected network_model_sec "${CMAKE_CURRENT_BINARY_DIR}/exec_expected")
  list(APPEND _embedded ${_embed_out})
  add_compile_definitions(EXEC_HAVE_EXPECTED=1)
endif()
add_compile_definitions(EXEC_TOL=${EXEC_TOL})

# ---- opt-in instrumentation for FPGA runs (all off by default; see smolvla-explore/fpga-prep/INSTRUMENTATION.md) ----
set(EXEC_REPEAT 1 CACHE STRING "Plain mode: run execute() this many times on the same inputs, print mcycle/minstret per run")
set(EXEC_PC_SAMPLE 0 CACHE STRING "Timer-interrupt PC sampler period in microseconds (0 = off); see src/pc_sampler.h")
set(EXEC_PC_TABLE 16384 CACHE STRING "PC sampler hash-table entries (power of two, 16 bytes each, .noinit)")
option(EXEC_XNN_PROFILE "Build ExecuTorch's XNNPACK backend with ENABLE_XNNPACK_PROFILING: one log line per XNNPACK operator and run" OFF)
option(EXEC_ERR_STATS "Print one ERRSTATS line per checked output (abs and rel error mean/min/max, RMSE, PSNR vs the reference), after the timed windows" OFF)
add_compile_definitions(EXEC_REPEAT=${EXEC_REPEAT} EXEC_PC_SAMPLE=${EXEC_PC_SAMPLE} EXEC_PC_TABLE=${EXEC_PC_TABLE})
# Debug aid: GCC expands memcpy() inline into e8,m8 vle8.v/vse8.v (not auto-vectorization: -fno-tree-vectorize keeps it). ETDump rows came
# back corrupted on FireSim but not on Spike, so this sends those copies to libc, to test whether they are the cause.
option(EXEC_NO_INLINE_VCOPY "Build ExecuTorch, XNNPACK and flatcc with -mstringop-strategy=libcall" OFF)
if(EXEC_NO_INLINE_VCOPY)
  add_compile_options(-mstringop-strategy=libcall)
endif()
# Debug aid: vfredusum in softmax and bf16 LayerNorm may add in any order, so hardware and Spike round differently. This fixes the order.
option(EXEC_ORDERED_FSUM "Use vfredosum in XNNPACK's RVV softmax and bf16 LayerNorm sums" OFF)
if(EXEC_ORDERED_FSUM)
  add_compile_definitions(XNN_RVV_ORDERED_FSUM=1)
endif()
if(EXEC_ERR_STATS)
  add_compile_definitions(EXEC_ERR_STATS=1)
endif()
option(EXEC_DUMP_ACTIONS "Chain mode: print the final actions as raw f32 bits, after the timed windows, to compare with lerobot on the host" OFF)
if(EXEC_DUMP_ACTIONS)
  add_compile_definitions(EXEC_DUMP_ACTIONS=1)
endif()
if(EXEC_STEP_DATA OR EXEC_CHAIN_DATA)
  if(EXEC_REPEAT GREATER 1 OR EXEC_PC_SAMPLE GREATER 0)
    message(WARNING "EXEC_REPEAT and EXEC_PC_SAMPLE apply to the plain mode only; step/chain mode ignores them")
  endif()
endif()
if(EXEC_XNN_PROFILE)
  # Directory-scope definition, so it reaches the xnnpack_backend target added below (XNNCompiler.cpp, XNNProfiler.cpp, XNNPACKBackend.cpp).
  # Needs an ExecuTorch without the stale print_avg_op_timings() call in XNNPACKBackend.cpp (upstream release/1.6 still has it; fixed
  # in our fork penguin-1.6).
  add_compile_definitions(ENABLE_XNNPACK_PROFILING)
endif()
# ExecuTorch's official profiling (ETRecord + ETDump + Inspector), opt-in. Derived from the Arm runner's ETDump support
# (ExecuTorch examples/arm/executor_runner/{CMakeLists.txt,arm_executor_runner.cpp} @ 681c3edb47, Copyright Arm Limited and/or its
# affiliates, BSD-style licence). What it switches on:
#   * EXECUTORCH_ENABLE_EVENT_TRACER: ET_EVENT_TRACER_ENABLED in the runtime and the delegates (every operator call is wrapped
#     in start_profiling()/end_profiling()). With it OFF none of that code is compiled.
#   * EXECUTORCH_BUILD_DEVTOOLS: builds the `etdump` library (ETDumpGen) and runs flatcc on its schema. flatcc's generator
#     (flatcc_cli) has to run on the BUILD machine: third-party/CMakeLists.txt of ExecuTorch builds it with an ExternalProject for the
#     host (CMAKE_TOOLCHAIN_FILE= empty), so cross-compiling needs nothing extra here. The runtime part (flatccrt) is built for RISC-V.
#   * the app (src/riscv_executor_runner.cpp) attaches ETDumpGen to load_method(), replaces the PAL clock with mcycle and prints
#     the dump after the timed execute() window.
# EXEC_ETDUMP_BUF: bytes of the static .noinit buffer ETDumpGen writes into (no malloc). up to 128 KiB of it (a quarter of a small buffer) is builder scratch;
#   the rest holds the events (about 100-200 bytes per operator and run). The runner prints how much it used.
# EXEC_CPU_HZ: core clock, only used to give the PAL a ticks->ns ratio (ns = ticks * 1e9 / EXEC_CPU_HZ). 50 MHz = the FPGA board.
#   The ETDump itself stores raw ticks, so the Inspector is run with --source_time_scale cycles. The ratio matters for the XNNPACK
#   per-operator rows (XNNProfiler::submit_trace() converts XNNPACK's microseconds into ticks with it, backends/xnnpack/runtime/
#   profiling/XNNProfiler.cpp) and for the ET_LOG timestamps.
option(EXEC_ETDUMP "Event tracer + ETDump profiling (ExecuTorch devtools); prints the dump base64 after the run (plain, step and chain mode)" OFF)
set(EXEC_ETDUMP_BUF 4194304 CACHE STRING "EXEC_ETDUMP: size in bytes of the static ETDump buffer")
set(EXEC_CPU_HZ 50000000 CACHE STRING "EXEC_ETDUMP: core clock in Hz for the PAL ticks->ns ratio")
if(EXEC_ETDUMP)
  add_compile_definitions(EXEC_ETDUMP=1 EXEC_ETDUMP_BUF=${EXEC_ETDUMP_BUF} EXEC_CPU_HZ=${EXEC_CPU_HZ})
  # XNNPACK's per-operator rows then read mcycle too, instead of the 100 us Zephyr clock (forks/XNNPACK src/runtime.c xnn_read_timer)
  add_compile_definitions(XNN_TIMER_MCYCLE_HZ=${EXEC_CPU_HZ})
  set(EXECUTORCH_ENABLE_EVENT_TRACER ON CACHE BOOL "" FORCE)
  set(EXECUTORCH_BUILD_DEVTOOLS ON CACHE BOOL "" FORCE)
  message(STATUS "ETDump: event tracer on, buffer ${EXEC_ETDUMP_BUF} bytes, PAL clock mcycle at ${EXEC_CPU_HZ} Hz")
endif()
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

# Selective build of the portable kernels: ExecuTorch's own option, fed with this app's model. executorch_selected_kernels (generated by
# ExecuTorch's CMake from the ops listed in the .pte, forks/executorch CMakeLists.txt "executorch_selected_kernels") replaces
# portable_ops_lib + portable_kernels below. No source is changed.
if(EXEC_SELECT_OPS)
  set(EXECUTORCH_SELECT_OPS_MODEL "${_pte}" CACHE STRING "" FORCE)
endif()

set(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ${EXEC_WEIGHT_CACHE} CACHE BOOL "" FORCE)   # EXEC_WEIGHT_CACHE / EXEC_PREPACK above
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

if(EXEC_SELECT_OPS)
  set(_kernel_libs executorch_selected_kernels)
else()
  set(_kernel_libs portable_ops_lib portable_kernels)
endif()
target_link_libraries(app PRIVATE
  extension_runner_util executorch executorch_core ${_kernel_libs}
  xnnpack_backend XNNPACK xnnpack-microkernels-prod extension_threadpool
)

if(EXEC_ETDUMP)
  target_link_libraries(app PRIVATE etdump flatccrt)
endif()

if(EXEC_PTD)
  target_link_libraries(app PRIVATE extension_flat_tensor)   # FlatTensorDataMap (built by ExecuTorch's XNNPACK preset)
endif()

# XNNPACK's internal headers (src/xnnpack/hardware-config.h), used to report the detected RISC-V config.
target_include_directories(app PRIVATE ${ET_DIR_PATH}/backends/xnnpack/third-party/XNNPACK)
target_include_directories(app PRIVATE ${ET_INCLUDE_PATH} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_sources(app PRIVATE src/riscv_executor_runner.cpp ${_embedded})
if(NOT EXEC_PREPACK STREQUAL "OFF")
  add_dependencies(app prepacked_blob)
endif()
# ExecuTorch custom op for fp8 weight-only linears, compiled into the app itself: its static registration object (EXECUTORCH_LIBRARY) is
# then always linked, which a static library would drop. The registry size (MAX_KERNEL_NUM) is generated from the ops of EXEC_MODEL, which
# lists penguin::linear_fp8.out, so the op has a slot.
option(EXEC_FP8_LINEAR "Build the ExecuTorch custom op penguin::linear_fp8.out (needs the XNNPACK fork's rvvzvfofp8 gemm kernel)" OFF)
set(FP8_MAX_NPAD 4096 CACHE STRING "EXEC_FP8_LINEAR: largest padded out_features (multiple of 32) the op's static f32 scratch (5 rows) can hold")
if(EXEC_FP8_LINEAR)
  target_sources(app PRIVATE src/ops/linear_fp8_out.cpp)
  target_compile_definitions(app PRIVATE FP8_MAX_NPAD=${FP8_MAX_NPAD})
  # The op's micro-kernel. No XNNPACK config references it, so XNNPACK lists it as non-production and leaves it out of
  # xnnpack-microkernels-prod; compile the one generated file here, with that library's include paths.
  set(_fp8_ukernel "${XNNPACK_DIR}/src/bf16-qf8w-gemm/gen/bf16-qf8w-f32-gemm-5x4v-minmax-rvvzvfofp8.c")
  if(NOT EXISTS "${_fp8_ukernel}")
    message(FATAL_ERROR "EXEC_FP8_LINEAR: ${_fp8_ukernel} not found; XNNPACK_DIR must be the fork with the rvvzvfofp8 kernels")
  endif()
  target_sources(app PRIVATE ${_fp8_ukernel})
  set_source_files_properties(${_fp8_ukernel} PROPERTIES
    INCLUDE_DIRECTORIES "$<TARGET_PROPERTY:xnnpack-microkernels-prod,INCLUDE_DIRECTORIES>")
endif()
zephyr_linker_sources(DATA_SECTIONS custom-sections.ld)
