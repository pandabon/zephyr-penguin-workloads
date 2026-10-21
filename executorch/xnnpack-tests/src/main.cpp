// SPDX-License-Identifier: Apache-2.0
//
// Entry point for running one unmodified XNNPACK GoogleTest file on Zephyr or bare metal (both build
// modes of this app share this file). The test file itself
// (XNNPACK test/<name>.cc) supplies the TEST()s; XNNPACK's CMake would link gtest_main, but that needs
// a hosted environment, so this is the only code of ours: initialise GoogleTest, run, and leave through
// the BSP (Zephyr) or HTIF (bare metal, baremetal/src/htif.c) so the host (Spike/uart_tsi) sees the
// result as the exit code.
//
// Credits: the main() shape is GoogleTest's own gtest_main.cc (google/googletest, BSD-3-Clause);
// chipyard_sim_exit usage as in executorch/smolvla/src/riscv_executor_runner.cpp of this repo.
#include <stdio.h>

#include <gtest/gtest.h>

#ifdef __ZEPHYR__
#include <zephyr/toolchain.h>

extern "C" {
#include <chipyard/sim_exit.h>
}
#else
// Bare metal: exit() runs picolibc's exit handlers and ends in _exit() = HTIF "tohost" write (htif.c).
#include <stdlib.h>
static inline void chipyard_sim_exit(int code) { exit(code); }
#endif

int main(void) {
  // No command line on the chip: pass the filter chosen at build time (XNN_TEST_FILTER, empty = all).
  int argc = 1;
  char name[] = "xnnpack-test";
  char filter[] = "--gtest_filter=" XNN_TEST_FILTER;
  // XNN_TEST_SEED: the tests seed their random numbers from GoogleTest's random seed
  // (UnitTest::random_seed()), which is 0 -> time-based unless --gtest_shuffle is given. Spike's clock is
  // deterministic, so without a seed every run of one binary gets the same "random" numbers. A fixed seed
  // can be chosen at build time; it needs --gtest_shuffle too (test ORDER is then shuffled, results do not
  // depend on it). Default (empty): GoogleTest's own behaviour, no shuffle.
  char seed[] = "--gtest_random_seed=" XNN_TEST_SEED;
  char shuffle[] = "--gtest_shuffle";
  char* argv[5] = {name, nullptr, nullptr, nullptr, nullptr};
  if (sizeof(XNN_TEST_FILTER) > 1) {
    argv[argc++] = filter;
  }
  if (sizeof(XNN_TEST_SEED) > 1) {
    argv[argc++] = shuffle;
    argv[argc++] = seed;
  }
  printf("XNNPACK test %s, filter '%s', seed '%s'\n", XNN_TEST_NAME, XNN_TEST_FILTER, XNN_TEST_SEED);
  testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  printf("%s (rc=%d)\n", rc == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", rc);
  chipyard_sim_exit(rc);
  return rc;
}
