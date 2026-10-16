/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Replaces setStats() from Saturn's benchmarks/common/syscalls.c (which needs the
 * HTIF/pk environment, not available under Zephyr). Same behaviour: setStats(1)
 * records mcycle/minstret, setStats(0) prints the deltas.
 * Output line (greppable):  stats: mcycle=<n> minstret=<n>
 */
#include <stdio.h>
#include "encoding.h"

static unsigned long c0, i0;

void setStats(int enable)
{
	if (enable) {
		c0 = read_csr(mcycle);
		i0 = read_csr(minstret);
		return;
	}
	printf("stats: mcycle=%lu minstret=%lu\n", read_csr(mcycle) - c0, read_csr(minstret) - i0);
}
