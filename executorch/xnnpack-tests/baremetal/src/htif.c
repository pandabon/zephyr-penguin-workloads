// SPDX-License-Identifier: BSD-3-Clause
//
// Console and exit for a bare-metal program on Spike (or any HTIF host, e.g. Chipyard's simulators):
// the HTIF "tohost"/"fromhost" mailbox, plus the one hook picolibc needs for stdio (stdout/stdin/stderr
// objects) and the exit code.
//
// What HTIF is: Spike (like Chipyard's simulators) polls the 64-bit word `tohost` in the program's memory
// (it finds it by the ELF symbol "tohost"). Writing (code << 1) | 1 there ends the simulation with that
// exit code. Writing the address of a small 8-word block there asks the host to run a proxy-kernel system
// call (here: write(1, buf, len)), and the host answers by writing a nonzero value to `fromhost`.
//
// Credits:
//   * htif_syscall(), _exit(): ucb-bar/libgloss-htif, misc/htif.c and sys/exit.c, sys/write.c @ 177ce07db657
//     (BSD-3-Clause, Copyright (c) 2019 The Regents of the University of California), re-typed here without
//     the spinlock (one hart) and with the HTIF_TOHOST() macro written out.
//   * the same protocol, one command per character: riscv-software-src/riscv-tests, benchmarks/common/
//     syscalls.c @ 3eeb9a1fecb5 (BSD-3-Clause, Copyright (c) 2012-2015 The Regents of the University of
//     California): tohost/fromhost in ".tohost", syscall(SYS_write, 1, buf, 1), tohost_exit().
//   * stdio hook: picolibc's documented way to give tinystdio a console, FDEV_SETUP_STREAM with a
//     put function and `FILE *const stdout = &__stdio` (picolibc doc/os.md; picolibc BSD-3-Clause,
//     Copyright (c) Keith Packard). 
//   * protocol description: riscv-isa-sim (Spike), fesvr/htif.h and fesvr/syscall.cc (BSD-3-Clause).
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define SYS_write 64

// 64-byte aligned. Plain (.bss) variables are enough: Spike finds them by the ELF symbol names and the
// memory they live in starts zeroed. (riscv-tests / libgloss-htif put them in a section of their own, which
// does not fit picolibc.ld's regions; they would need a linker-script entry.)
volatile uint64_t tohost __attribute__((aligned(64)));
volatile uint64_t fromhost __attribute__((aligned(64)));

static long htif_syscall(uint64_t a0, uint64_t a1, uint64_t a2, unsigned long n) {
  volatile uint64_t buf[8] __attribute__((aligned(64)));
  buf[0] = n;
  buf[1] = a0;
  buf[2] = a1;
  buf[3] = a2;
  __sync_synchronize();
  tohost = (uint64_t)(uintptr_t)buf;          // HTIF device 0 (syscall proxy), command 0, payload = address
  while (fromhost == 0) {
  }
  fromhost = 0;
  __sync_synchronize();
  return (long)buf[0];
}

// Standard C exit(): picolibc's exit() runs the destructors and calls _exit().
__attribute__((noreturn)) void _exit(int code) {
  uint64_t cmd = ((uint64_t)(uint32_t)code << 1) | 1;
  for (;;) {
    fromhost = 0;
    tohost = cmd;
  }
}

// POSIX write(): fd 1 and 2 go to the host's stdout/stderr (proxy-kernel write). Used by GoogleTest and fdopen().
ssize_t write(int fd, const void* buf, size_t len) {
  if (fd != 1 && fd != 2) { errno = EBADF; return -1; }
  return (ssize_t)htif_syscall((uint64_t)fd, (uint64_t)(uintptr_t)buf, len, SYS_write);
}

static int htif_putc(char c, FILE*) {
  htif_syscall(1, (uint64_t)(uintptr_t)&c, 1, SYS_write);
  return (unsigned char)c;
}
static int htif_getc(FILE*) { return _FDEV_EOF; }   // no input on the chip's test runs

static FILE __stdio = FDEV_SETUP_STREAM(htif_putc, htif_getc, NULL, _FDEV_SETUP_RW);
FILE* const stdin = &__stdio;
FILE* const stdout = &__stdio;
FILE* const stderr = &__stdio;

// Called from the trap vector in crt_htif.S: print what happened, then exit with a code that is not 0.
static void put_hex(uint64_t v) {
  for (int i = 60; i >= 0; i -= 4) htif_putc("0123456789abcdef"[(v >> i) & 15], NULL);
}
static void put_str(const char* s) {
  while (*s) htif_putc(*s++, NULL);
}
__attribute__((noreturn)) void htif_fatal_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval) {
  put_str("\nFATAL TRAP mcause=0x"); put_hex(mcause);
  put_str(" mepc=0x"); put_hex(mepc);
  put_str(" mtval=0x"); put_hex(mtval);
  put_str("\n");
  _exit(100);
}
