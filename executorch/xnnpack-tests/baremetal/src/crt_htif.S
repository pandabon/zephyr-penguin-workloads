/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Entry point of the bare-metal XNNPACK test app (Spike / Penguin, one hart, machine mode).
 *
 * It does only what picolibc's own start-up code (crt0, which we jump to at the end) does not:
 * turn on the FPU and the VECTOR unit, and install a trap handler that reports and exits.
 * picolibc's crt0 then sets gp/sp, copies .data, zeroes .bss, sets up TLS, runs the C++ constructors,
 * calls main() and calls exit().
 *
 * Credits (derived from, not copied wholesale):
 *   * riscv-software-src/riscv-tests, benchmarks/common/crt.S @ b82006e3ca47 (BSD-3-Clause,
 *     Copyright (c) 2012-2015 The Regents of the University of California): the "enable FPU, vector,
 *     and accelerator if present" mstatus write (MSTATUS_FS | MSTATUS_XS | MSTATUS_VS) and the
 *     mtvec set-up. Bit values from env/encoding.h of the same repo.
 *   * picolibc (https://github.com/picolibc/picolibc, BSD-3-Clause, Copyright (c) Keith Packard):
 *     the _start we jump to (picocrt/machine/riscv/crt0.c).
 */
#define MSTATUS_FS  0x00006000   /* FPU state: 3 = dirty (so enabled) */
#define MSTATUS_XS  0x00018000   /* custom-extension state; harmless if there is none */
#define MSTATUS_VS  0x00000600   /* VECTOR state: 3 = dirty (so enabled). Without it every RVV instruction traps. */

	.section .text.init.enter, "ax"   /* picolibc.ld puts this first, so it is the image start */
	.globl _htif_start
_htif_start:
	la	t0, _htif_trap_entry     /* first, so that even a missing unit below is reported, not a silent hang */
	csrw	mtvec, t0

	li	t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS
	csrs	mstatus, t0
	fscsr	x0                       /* FP flags and rounding mode: clean */
	csrw	vstart, x0               /* vector state: clean (vl/vtype are set by vsetvli); traps if there is no V */

	j	_start                   /* picolibc crt0 */

	.align	2
_htif_trap_entry:
	/* Nothing is expected to trap. If it does, the stack may be what is broken, so use a fresh one:
	   the top of the stack area (picolibc __stack) minus a margin, then report mcause/mepc/mtval and exit. */
	.option push
	.option norelax
	la	gp, __global_pointer$    /* the C code is compiled gp-relative; a trap before crt0 ran has gp = 0 */
	.option pop
	la	sp, __stack
	addi	sp, sp, -16
	csrr	a0, mcause
	csrr	a1, mepc
	csrr	a2, mtval
	call	htif_fatal_trap
