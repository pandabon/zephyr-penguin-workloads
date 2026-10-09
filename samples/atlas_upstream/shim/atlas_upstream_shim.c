/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CREDIT: the tests this sample builds, and the generators/assembler it calls, are
 * the Atlas/Saturn team's work, used unmodified from ~/toolsB/tapeout-chipyard
 * (tests/, generators/sp26-atlas-acc/baremetal/), commit
 * 48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8, submodule sp26-atlas-acc 2ae0bef209df. git authors of tests/: kellytou
 * ("T.K."), Eric Wen, Nicolas Rakela (Herrera), Nicolas Reed; sp26-atlas-acc
 * baremetal: kellytou, Nicolas Reed, jason.tang, Eric Wen, nicorakela. The build
 * logic mirrors tests/CMakeLists.txt there. We own only the files in this directory.
 *
 * Glue between the Atlas team's baremetal tests (compiled unmodified) and Zephyr.
 * Everything that differs from their world lives here, in one file:
 *
 *  - Startup/exit: upstream is linked with htif.ld + -specs=htif_nano.specs, whose
 *    crt0 calls main() and writes its return value to HTIF `tohost`. Zephyr boots
 *    its own kernel and calls main() from a thread, so we link with --wrap=main
 *    (see CMakeLists.txt): the kernel's main() call lands in __wrap_main() below,
 *    which runs the upstream main() and ends the simulation/bench run with its
 *    return value, as the baremetal crt0 would (0 = pass).
 *  - printf(): htif_nano.specs routes stdout to HTIF; Zephyr+picolibc routes it
 *    to the console (HTIF or UART, whichever the board selects). Nothing to do.
 *  - rdcycle (dmem_sbus_dma): runs in M-mode on this board, works unchanged.
 *  - Memory: upstream's linker script puts the program at 0x80000000; Zephyr's
 *    does the same (ram0). The tests' DRAM window ATLAS_DRAM_BASE 0x90000000 is
 *    plain pointer access, no linker symbol involved.
 *  - Timeouts: some upstream loops can wait very long (poll counts of 200 million)
 *    or forever. A watchdog thread bounds the wall time (Kconfig
 *    ATLAS_UPSTREAM_TIMEOUT_S), stops Atlas, and reports TIMEOUT.
 *
 * Output convention (greppable, in addition to upstream's own lines):
 *   ATLASUP <test> PASS | FAIL rc=<n> | TIMEOUT after <s> s
 * Upstream's own final "PASS: ..." line is the reference result; main() returns
 * nonzero on any failure, so the exit code agrees.
 */
#include <stdio.h>
#include <zephyr/kernel.h>
#include <chipyard/sim_exit.h>

#include "atlas_mmio.h"   /* upstream header: ATLAS_CSR_BASE etc. */

#ifndef ATLAS_UPSTREAM_NAME
#define ATLAS_UPSTREAM_NAME "unknown"
#endif

extern int __real_main(void);

/* Cooperative priority (-1) so that it preempts the busy-polling main thread
 * (priority 0, preemptible) as soon as its sleep expires. */
static void watchdog(void *a, void *b, void *c)
{
	(void)a; (void)b; (void)c;
	k_sleep(K_SECONDS(CONFIG_ATLAS_UPSTREAM_TIMEOUT_S));
	atlas_stop();   /* execControl = 0: do not leave Atlas running */
	printf("ATLASUP %s TIMEOUT after %d s\n", ATLAS_UPSTREAM_NAME,
	       CONFIG_ATLAS_UPSTREAM_TIMEOUT_S);
	chipyard_sim_exit(124);
}
K_THREAD_DEFINE(atlas_wd, 1024, watchdog, NULL, NULL, NULL, -1, 0, 0);

int __wrap_main(void)
{
	printf("ATLASUP %s start\n", ATLAS_UPSTREAM_NAME);
	int rc = __real_main();

	if (rc == 0) {
		printf("ATLASUP %s PASS\n", ATLAS_UPSTREAM_NAME);
	} else {
		printf("ATLASUP %s FAIL rc=%d\n", ATLAS_UPSTREAM_NAME, rc);
	}
	chipyard_sim_exit(rc);
}
