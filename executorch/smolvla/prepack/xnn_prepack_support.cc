// Support file for xnn_prepack.cc (see its header for the big picture).
//
// 1. XNN_ALLOCATION_ALIGNMENT for RISC-V. The macro in XNNPACK's
//    src/xnnpack/common.h depends on the compile target (64 on x86, 16 on
//    RISC-V). The helper runs on x86, so this file is compiled with the
//    compiler's x86 macros removed and __riscv set (flags in prepack.py's
//    build command). common.h then takes its RISC-V branch and the value is
//    read from the fork's header instead of being typed in here. The #error
//    below stops the build if the trick ever stops working.
//    Source: pandabon/XNNPACK src/xnnpack/common.h (commit 7264d39dab).
//
// 2. Link stubs. src/reference/packing.cc (linked unchanged) also contains
//    int8 packers that call two parameter-init functions we do not link.
//    They are never called for bf16 / f32 weights; __builtin_trap() makes a
//    wrong call obvious instead of silent. (No libc headers here: with the x86
//    macros removed, glibc's headers do not compile.)

#include "src/xnnpack/common.h"

#if !XNN_ARCH_RISCV
#error "xnn_prepack_support.cc must be compiled with -D__riscv=1 and the x86 macros undefined"
#endif

extern "C" const unsigned long xnn_riscv_allocation_alignment =
    XNN_ALLOCATION_ALIGNMENT;

extern "C" void xnn_init_blockwise_scale_bf16_params() { __builtin_trap(); }
extern "C" void xnn_init_qs8_qc8w_scale_fp32_params() { __builtin_trap(); }
