# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(rvv_ctx_switch)

target_sources(app PRIVATE src/main.c)
# The compiler must never auto-vectorize our checking code: any compiler-emitted
# vector op between the asm blocks would clobber the registers under test.
target_compile_options(app PRIVATE -fno-tree-vectorize)
