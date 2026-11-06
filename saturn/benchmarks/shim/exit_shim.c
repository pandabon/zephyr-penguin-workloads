/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * CREDIT: same idea and code as cmake/plain_app/exit_shim.c in our board repo
 * (zephyr-chipyard-bsp @93ace03, now zephyr-penguin; BSP-4 removes plain_app
 * there).
 *
 * The app is linked with -Wl,--wrap=main: the kernel's call to main() lands
 * here. We run the benchmark's main() and end the run with its return value
 * (0 = pass); without this a returning main() leaves Zephyr idling.
 */
#include <chipyard/sim_exit.h>

extern int __real_main(void);

int __wrap_main(void)
{
	chipyard_sim_exit(__real_main());
}
