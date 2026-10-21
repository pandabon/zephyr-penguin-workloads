# SPDX-License-Identifier: Apache-2.0
#
# CMake toolchain file: bare-metal RISC-V with the Zephyr SDK's GCC and its picolibc (no Zephyr kernel,
# no OS). Used as  cmake -DCMAKE_TOOLCHAIN_FILE=baremetal/toolchain.cmake ...  by scripts/build.sh.
#
# The compiler is the one `source <workspace>/env.sh` puts on PATH (riscv64-zephyr-elf-gcc, Zephyr SDK
# 1.0.0-beta1, GCC 14.3); or set ZEPHYR_SDK_INSTALL_DIR. Nothing is installed or downloaded.
#
# Credit: the shape of a "Generic" cross toolchain file is the standard CMake pattern (CMake docs,
# cmake-toolchains(7)); the picolibc spec file is the SDK's own (picolibc, BSD-3-Clause, Keith Packard).
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(_triple riscv64-zephyr-elf)
if(DEFINED ENV{ZEPHYR_SDK_INSTALL_DIR})
  set(_hint "$ENV{ZEPHYR_SDK_INSTALL_DIR}/gnu/${_triple}/bin")
endif()
find_program(CMAKE_C_COMPILER   ${_triple}-gcc HINTS ${_hint})
find_program(CMAKE_CXX_COMPILER ${_triple}-g++ HINTS ${_hint})
find_program(CMAKE_ASM_COMPILER ${_triple}-gcc HINTS ${_hint})
find_program(CMAKE_AR           ${_triple}-gcc-ar HINTS ${_hint})
find_program(CMAKE_RANLIB       ${_triple}-gcc-ranlib HINTS ${_hint})
find_program(CMAKE_OBJCOPY      ${_triple}-objcopy HINTS ${_hint})
find_program(CMAKE_SIZE_UTIL    ${_triple}-size HINTS ${_hint})
if(NOT CMAKE_C_COMPILER)
  message(FATAL_ERROR "${_triple}-gcc not found: run `source <workspace>/env.sh` first")
endif()

# The ISA (-march/-mabi/-mcmodel) is in penguin-isa.cmake, in one place, with where it comes from.
include(${CMAKE_CURRENT_LIST_DIR}/penguin-isa.cmake)
set(_arch "-march=${PENGUIN_MARCH} -mabi=${PENGUIN_MABI} -mcmodel=${PENGUIN_MCMODEL}")
# -specs=picolibc.specs: the SDK's picolibc (headers, libc.a, libstdc++, default linker script picolibc.ld).
# (CMake passes CMAKE_<LANG>_FLAGS to the link command too, so -specs is given once, here.)
set(_common "${_arch} -specs=picolibc.specs -ffunction-sections -fdata-sections -fno-common -fno-pic -fno-pie -fno-asynchronous-unwind-tables")
set(CMAKE_C_FLAGS_INIT   "${_common}")
set(CMAKE_CXX_FLAGS_INIT "${_common}")
set(CMAKE_ASM_FLAGS_INIT "${_arch}")

# Cross-compiling: never try to run what we build; link checks use a static library.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
