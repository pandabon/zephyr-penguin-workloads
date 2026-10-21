# SPDX-License-Identifier: Apache-2.0
#
# BARE-METAL mode of the XNNPACK-gtest app (decision WL-8): the same UNMODIFIED XNNPACK test files as the Zephyr
# mode (../CMakeLists.txt), but with no Zephyr at all: GCC + picolibc from the Zephyr SDK, our small start-up
# and HTIF console/exit (src/crt_htif.S, src/htif.c), run on Spike (../scripts/run.sh) or the chip.
#
# Configure through ../scripts/build.sh (it passes the toolchain file and the opt-in dependency paths):
#   scripts/build.sh --target baremetal f32-bf16-vcvt
# Inputs and options are the same as in the Zephyr mode, see ../cmake/xnnpack-tests-common.cmake:
#   XNN_TEST, XNN_TEST_SUPPORT, XNN_TEST_FILTER, XNN_TEST_SEED (and XNNPACK_DIR, EXECUTORCH_DIR ...).
#
# Differences from the Zephyr mode (everything else is shared):
#   * no Zephyr include paths/flags; the ISA flags come from penguin-isa.cmake via toolchain.cmake;
#   * XNNPACK's RISC-V hardware detection: see "RVV detection" below;
#   * a one-thread <pthread.h> shim (src/shim) instead of Zephyr's POSIX layer;
#   * the SDK's own linker script picolibc.ld with the memory map given on the command line (below).
#
# Credits: see the top of ../CMakeLists.txt (idea, flag propagation) and the headers of src/crt_htif.S and
# src/htif.c (riscv-software-src/riscv-tests, ucb-bar/libgloss-htif, picolibc).

cmake_minimum_required(VERSION 3.20)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/xnnpack-tests-common.cmake)

project(penguin_xnnpack_tests_baremetal C CXX ASM)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The flags the Zephyr mode uses for the same sources (-O2, no exceptions, no RTTI, no auto-vectorization).
add_compile_options(-O2 -g -fno-strict-aliasing -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize
                    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions> $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
                    $<$<COMPILE_LANGUAGE:CXX>:-fcheck-new> $<$<COMPILE_LANGUAGE:CXX>:-Wno-register>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wno-volatile>)
add_compile_definitions(_POSIX_C_SOURCE=200809L __POSIX_VISIBLE=200809L)
# XNNPACK: no mmap/mprotect; cpuinfo's log.c wants STDERR_FILENO.
add_compile_definitions(XNN_HAS_MMAP=0 STDERR_FILENO=2)
# XNNPACK's runtime (src/runtime.c, operator timing for profiling) calls clock_gettime(CLOCK_MONOTONIC). picolibc
# declares CLOCK_MONOTONIC only when the platform announces a monotonic clock; announce it, and
# src/syscall_stubs.c provides a clock_gettime() that returns 0 like gettimeofday() (no clock here).
add_compile_definitions(_POSIX_MONOTONIC_CLOCK=200112L)
# <pthread.h> for XNNPACK/pthreadpool: the one-thread shim (picolibc has none).
include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/src/shim)

# ---- memory map (Spike's default RAM starts at 0x80000000, 2 GiB) --------------------------------------
# picolibc.ld has two regions: "flash" (code + read-only data, and the initial values of .data) and "ram"
# (.data, .bss, heap, stack). On Spike both are plain RAM, so load image at 0x80000000 and put "ram" after it.
# The heap is whatever is left of ram, below the stack; XNNPACK's tests allocate big buffers.
set(BM_FLASH_BASE "0x80000000" CACHE STRING "start of the image (Spike: RAM base)")
set(BM_FLASH_SIZE "0x10000000" CACHE STRING "room for code + rodata + .data initial values (256 MiB)")
set(BM_RAM_BASE   "0x90000000" CACHE STRING "start of .data/.bss/heap/stack")
set(BM_RAM_SIZE   "0x10000000" CACHE STRING "size of .data + .bss + heap + stack (256 MiB)")
set(BM_STACK_SIZE "0x00100000" CACHE STRING "stack size, taken from the top of ram (1 MiB)")

# pthreadpool does find_package(Threads). On a "Generic" system with a static-library try-compile CMake would
# decide that -lpthreads exists; there is no such library (picolibc has no threads: our <pthread.h> shim is
# header-only), so say up front that pthread functions are "in libc" and no extra library is needed.
set(CMAKE_HAVE_LIBC_PTHREAD ON CACHE BOOL "pthread_create is in libc (our shim)")

# ---- XNNPACK and its dependencies, built as a normal CMake sub-project ----------
add_subdirectory("${FXDIV_SOURCE_DIR}" fxdiv)
set(_xnn_build "${CMAKE_CURRENT_BINARY_DIR}/xnnpack")
add_subdirectory("${XNNPACK_DIR}" "${_xnn_build}" EXCLUDE_FROM_ALL)

# ---- RVV detection (the point that needed care) ---------------------------------------------------------
# XNNPACK decides at run time which kernels it may use (src/configs/hardware-config.c). On Linux it asks the OS
# (getauxval(AT_HWCAP), riscv_hwprobe); bare metal has neither. Our XNNPACK fork has a `#if defined(__ZEPHYR__)`
# branch there for exactly this case: take "vector" and "fp16 vector" from the compile-time ISA
# (__riscv_vector / __riscv_zvfh, i.e. from -march) and read only the vector length from the vlenb CSR.
# That branch is "no OS" in substance, so we switch it on for the two XNNPACK libraries that have such branches
# (xnnpack-hardware-config: the detection; xnnpack-logging: its __ZEPHYR__ branch uses printf() instead of
# write(), and picolibc has no write()). A compile definition, so forks/XNNPACK stays unmodified.
# It is NOT defined for GoogleTest or the test app (gtest would then include Zephyr headers).
foreach(_t xnnpack-hardware-config xnnpack-logging)
  target_compile_definitions(${_t} PRIVATE __ZEPHYR__=1)
endforeach()

# GoogleTest: no pthreads, no gmock, nothing installed.
set(gtest_disable_pthreads ON CACHE BOOL "")
set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
set(gtest_build_tests OFF CACHE BOOL "")
add_subdirectory("${GOOGLETEST_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
# picolibc declares isatty/chdir/getcwd/dup/dup2 in <unistd.h> but does not define them: src/syscall_stubs.c does
# (the Zephyr mode needs a force-included header instead, because Zephyr's <unistd.h> does not declare them).
target_compile_options(gtest PRIVATE -Wno-error -w)

# ---- the test application --------------------------------------------------------
# The test must see the same XNN_ENABLE_* values XNNPACK was configured with (see ../CMakeLists.txt).
get_directory_property(_xnn_defs DIRECTORY "${XNNPACK_DIR}" COMPILE_DEFINITIONS)
set(_support)
foreach(_s IN LISTS XNN_TEST_SUPPORT)
  list(APPEND _support "${XNNPACK_DIR}/test/${_s}.cc")
endforeach()
add_executable(xnnpack_test.elf
  ${CMAKE_CURRENT_LIST_DIR}/../src/main.cpp "${_test_src}" ${_support}
  ${CMAKE_CURRENT_LIST_DIR}/src/crt_htif.S ${CMAKE_CURRENT_LIST_DIR}/src/htif.c ${CMAKE_CURRENT_LIST_DIR}/src/syscall_stubs.c)
target_compile_definitions(xnnpack_test.elf PRIVATE ${_xnn_defs}
  XNN_TEST_NAME="${XNN_TEST}" XNN_TEST_FILTER="${XNN_TEST_FILTER}" XNN_TEST_SEED="${XNN_TEST_SEED}")
target_include_directories(xnnpack_test.elf PRIVATE "${XNNPACK_DIR}" "${XNNPACK_DIR}/include" "${GOOGLETEST_DIR}/googletest/include")
target_link_libraries(xnnpack_test.elf PRIVATE
  gtest
  xnnpack-datatype xnnpack-hardware-config xnnpack-logging xnnpack-microkernels-all xnnpack-microparams-init
  xnnpack-packing
  pthreadpool
  ${XNN_TEST_LINK}
)
# -Tpicolibc.ld: the SDK's linker script. picolibc.specs adds it itself, but twice when the C++ driver is used
# ("linker script file 'picolibc.ld' appears multiple times"); giving -T makes the spec skip its own.
# -e _htif_start: start in our entry (enables FPU/vector, then jumps to picolibc's crt0 _start).
# --defsym: the memory map picolibc.ld asks the application for (its MEMORY block uses them if defined).
target_link_options(xnnpack_test.elf PRIVATE
  -Tpicolibc.ld -Wl,-e,_htif_start -Wl,--gc-sections -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/xnnpack_test.map
  -Wl,--defsym=__flash=${BM_FLASH_BASE} -Wl,--defsym=__flash_size=${BM_FLASH_SIZE}
  -Wl,--defsym=__ram=${BM_RAM_BASE} -Wl,--defsym=__ram_size=${BM_RAM_SIZE}
  -Wl,--defsym=__stack_size=${BM_STACK_SIZE})
