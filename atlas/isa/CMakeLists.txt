# SPDX-License-Identifier: Apache-2.0
#
# CREDIT: the tests this app builds (assembly/*.S, generators/gen_*.py, assembler.py and the C harness
# that assembler.py emits) are the Atlas team's work, copied unmodified from the course tapeout repo
# (tapeout-chipyard 48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8, submodule sp26-atlas-acc 2ae0bef209df,
# baremetal/) into ../../third_party/tapeout-chipyard. Authors and licence:
# ../../third_party/tapeout-chipyard/NOTICE. The generation recipe follows their baremetal/Makefile and
# Baremetal-IDE PR #65 (Crackle07, scripts/atlas/gen_atlas_tests.py @ 0a22f5a820ca), see
# ../../scripts/gen_atlas_isa_tests.py. We own only the files in this directory and that script.
#
# Build ONE of the Atlas ISA-level baremetal tests (121 of them) as a Zephyr app:
#
#   west build -b chipyard_penguin/penguin atlas/isa -- -DATLAS_ISA_TEST=alu_basic
#
# What happens (all into the build dir; the source tree is only read):
#   1. scripts/gen_atlas_isa_tests.py imports the team's assembler.py, assembles assembly/<test>.S
#      into Atlas instruction words, runs generators/gen_<test>.py (if there is one) for the golden
#      DRAM data, and writes atlas_<test>.c with assembler.py's own emit_c_file(): IMEM program array,
#      DRAM preload/check arrays, then main() that stops Atlas, writes IMEM, reads it back, starts
#      Atlas, polls CSR_DBG0 (up to 200M reads), stops Atlas, checks DRAM, prints
#      "*** PASSED ***" / "*** FAILED ***" and returns 0/1.
#   2. That C file is compiled unmodified, with shim/atlas_isa_shim.c which wraps main(): it adds the
#      greppable "ATLASISA <test> PASS|FAIL|TIMEOUT" line, ends the run with main()'s value as exit
#      code and bounds the wall time (Kconfig ATLAS_ISA_TIMEOUT_S).
#
# Variables:
#   ATLAS_ISA_TEST   test name = file stem in baremetal/assembly/ (default alu_basic)
#   ATLAS_PYTHON     python for the generators; default: <repo>/.venv/bin/python if present, else the
#                    build's python. Needs numpy, numba, torch (requirements-atlas.txt, opt-in).
#   TAPEOUT_CHIPYARD optional: a tapeout-chipyard checkout instead of third_party/ (same layout).
cmake_minimum_required(VERSION 3.20.0)

if(NOT TAPEOUT_CHIPYARD)
  set(TAPEOUT_CHIPYARD "$ENV{TAPEOUT_CHIPYARD}")
endif()
if(NOT TAPEOUT_CHIPYARD)
  set(TAPEOUT_CHIPYARD "${CMAKE_CURRENT_LIST_DIR}/../../third_party/tapeout-chipyard")
endif()
get_filename_component(TAPEOUT_CHIPYARD "${TAPEOUT_CHIPYARD}" ABSOLUTE)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(BM "${TAPEOUT_CHIPYARD}/generators/sp26-atlas-acc/baremetal")

if(NOT ATLAS_ISA_TEST)
  set(ATLAS_ISA_TEST alu_basic)
endif()
if(NOT EXISTS "${BM}/assembly/${ATLAS_ISA_TEST}.S")
  message(FATAL_ERROR "ATLAS_ISA_TEST='${ATLAS_ISA_TEST}': no ${BM}/assembly/${ATLAS_ISA_TEST}.S "
                      "(a test name is a file stem of baremetal/assembly/*.S).")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(atlas_isa)

# python for the generators: explicit > repo .venv > the python Zephyr found
if(NOT ATLAS_PYTHON)
  set(ATLAS_PYTHON "$ENV{ATLAS_PYTHON}")
endif()
if(NOT ATLAS_PYTHON AND EXISTS "${REPO_ROOT}/.venv/bin/python")
  set(ATLAS_PYTHON "${REPO_ROOT}/.venv/bin/python")
endif()
if(NOT ATLAS_PYTHON)
  set(ATLAS_PYTHON "${Python3_EXECUTABLE}")
  if(NOT ATLAS_PYTHON)
    set(ATLAS_PYTHON "${PYTHON_EXECUTABLE}")
  endif()
endif()
message(STATUS "atlas_isa: test ${ATLAS_ISA_TEST}, generators run with ${ATLAS_PYTHON}")

set(GEN_DIR "${CMAKE_BINARY_DIR}/atlas_isa_gen")
set(GEN_C "${GEN_DIR}/atlas_${ATLAS_ISA_TEST}.c")
file(GLOB_RECURSE _gen_deps CONFIGURE_DEPENDS
  ${BM}/generators/*.py ${TAPEOUT_CHIPYARD}/generators/sp26-atlas-acc/dependencies/fpex/model/*.py)
add_custom_command(
  OUTPUT ${GEN_C}
  COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
          ${ATLAS_PYTHON} ${REPO_ROOT}/scripts/gen_atlas_isa_tests.py
          --atlas-dir ${BM} --out-dir ${GEN_DIR} --python ${ATLAS_PYTHON} ${ATLAS_ISA_TEST}
  DEPENDS ${BM}/assembly/${ATLAS_ISA_TEST}.S ${BM}/assembler.py
          ${REPO_ROOT}/scripts/gen_atlas_isa_tests.py ${_gen_deps}
  COMMENT "Atlas ${ATLAS_ISA_TEST}: assemble + golden data -> atlas_${ATLAS_ISA_TEST}.c"
  VERBATIM)
add_custom_target(atlas_isa_gen DEPENDS ${GEN_C})
add_dependencies(app atlas_isa_gen)

target_sources(app PRIVATE ${GEN_C} shim/atlas_isa_shim.c)
target_compile_definitions(app PRIVATE ATLAS_ISA_NAME="${ATLAS_ISA_TEST}")
# main() -> __wrap_main() in the shim: result line, exit code, watchdog.
zephyr_ld_options(-Wl,--wrap=main)
