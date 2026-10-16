/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Credits: Zephyr's samples/hello_world (Wind River Systems, Apache-2.0), with the HTIF exit added
 * by Dima Nikiforov in ucb-bar/zephyr-chipyard-sw (branch dev @ 838a4fed2be, samples/hello_world).
 * Copied into penguin-workloads unchanged except for this comment.
 */

#include <stdio.h>
#include <zephyr/sys/reboot.h>

int main(void)
{
	// Print to the host via TSI
	printf("Hello World! %s\n", CONFIG_BOARD_TARGET);

	// Send an exit command to the host to terminate the simulation
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
