/*
 * pc_sampler.h - opt-in timer-interrupt PC sampler (EXEC_PC_SAMPLE=<period in us>, 0 = compiled out).
 *
 * Every timer tick records the PC of the interrupted code in a fixed-size open-addressing hash table (pc -> count) and
 * pc_sampler_report() prints it as "<hex pc> <count>" lines between "PCHIST BEGIN" / "PCHIST END" markers, the format
 * penguin-bringup-tools/profile/spike_profile.py --hist reads (see fpga-prep/pchist_from_log.py).
 *
 * How the interrupted PC is found (CONFIG_SMP=y, one CPU, so Zephyr's own sampler cannot be enabled as is):
 * derived from arch_perf_current_stack_trace() in zephyr/subsys/profiling/perf/backends/perf_riscv.c
 * (Copyright (c) 2023 KNS Group LLC (YADRO), SPDX-License-Identifier: Apache-2.0), and the timer-callback idea of
 * perf_tracer() in zephyr/subsys/profiling/perf/perf.c (same origin). arch/riscv/core/isr.S saves the interrupted context
 * (struct arch_esf: ra, mepc, s0, ...) on the thread stack, switches to the CPU's irq stack and stores the thread's sp
 * (= the ESF address) at irq_stack - 16. A k_timer callback runs inside the system-timer ISR, so
 * *(struct arch_esf **)(_current_cpu->irq_stack - 16) is the ESF and ->mepc the interrupted PC. This works the same with
 * SMP on or off (it does not use anything of the perf subsystem, which needs !SMP and SHELL).
 * Everything else (hash table, report, markers) is written for this repo.
 */
#ifndef PC_SAMPLER_H
#define PC_SAMPLER_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef EXEC_PC_SAMPLE
#define EXEC_PC_SAMPLE 0
#endif

#if EXEC_PC_SAMPLE > 0

#include <zephyr/kernel.h>
#include <zephyr/kernel_structs.h>
#include <zephyr/arch/riscv/exception.h>

#ifndef EXEC_PC_TABLE
#define EXEC_PC_TABLE 16384 /* entries; power of two; 16 bytes each on rv64, in .noinit (not part of the loaded image) */
#endif
static_assert((EXEC_PC_TABLE & (EXEC_PC_TABLE - 1)) == 0, "EXEC_PC_TABLE must be a power of two");

struct pc_entry {
	uintptr_t pc; /* 0 = empty */
	uint32_t count;
};

static struct pc_entry __noinit pc_table[EXEC_PC_TABLE];
static struct k_timer pc_timer;
static volatile uint32_t pc_samples, pc_dropped, pc_used;
static uint64_t pc_cycle0, pc_cycles_run; /* mtime cycles with the timer running (pause/resume excluded) */
static uint32_t pc_period_ticks;

#define PC_MAX_PROBE 64 /* bounds the time spent in the ISR; a longer probe chain counts as dropped */

/* Runs in the system-timer ISR. Integer code only (no FPU / vector registers touched). */
static void pc_sample_cb(struct k_timer *t)
{
	(void)t;
	if (_current_cpu->nested != 1) { /* not the first-level interrupt: no ESF at irq_stack - 16 */
		pc_dropped++;
		return;
	}
	const struct arch_esf *esf = *((struct arch_esf **)(((uintptr_t)_current_cpu->irq_stack) - 16));
	uintptr_t pc = esf->mepc;
	uint32_t h = (uint32_t)((pc >> 1) * 2654435761u) & (EXEC_PC_TABLE - 1);

	for (unsigned i = 0; i < PC_MAX_PROBE; i++) {
		struct pc_entry *e = &pc_table[(h + i) & (EXEC_PC_TABLE - 1)];
		if (e->pc == pc) {
			e->count++;
			pc_samples++;
			return;
		}
		if (e->pc == 0) {
			e->pc = pc;
			e->count = 1;
			pc_used++;
			pc_samples++;
			return;
		}
	}
	pc_dropped++;
}

static inline void pc_sampler_start(void)
{
	memset(pc_table, 0, sizeof(pc_table));
	pc_samples = pc_dropped = pc_used = 0;
	pc_period_ticks = (uint32_t)k_us_to_ticks_ceil64((uint64_t)EXEC_PC_SAMPLE);
	if (pc_period_ticks < 1)
		pc_period_ticks = 1;
	k_timer_init(&pc_timer, pc_sample_cb, NULL);
	pc_cycles_run = 0;
	pc_cycle0 = k_cycle_get_64();
	k_timer_start(&pc_timer, K_TICKS(pc_period_ticks), K_TICKS(pc_period_ticks));
}

/* Stop the timer, keep the table (used around work that must not be sampled, e.g. re-setting the inputs between runs). */
static inline void pc_sampler_pause(void)
{
	k_timer_stop(&pc_timer);
	pc_cycles_run += k_cycle_get_64() - pc_cycle0;
}

static inline void pc_sampler_resume(void)
{
	pc_cycle0 = k_cycle_get_64();
	k_timer_start(&pc_timer, K_TICKS(pc_period_ticks), K_TICKS(pc_period_ticks));
}

static inline void pc_sampler_stop(void)
{
	pc_sampler_pause();
}

static inline void pc_sampler_report(void)
{
	const uint64_t hz = sys_clock_hw_cycles_per_sec();
	const uint64_t elapsed_us = pc_cycles_run * 1000000ull / hz;
	printf("PCHIST BEGIN\n");
	printf("PCHIST total_samples=%u dropped=%u distinct_pcs=%u table=%u\n", (unsigned)pc_samples, (unsigned)pc_dropped,
	       (unsigned)pc_used, (unsigned)EXEC_PC_TABLE);
	printf("PCHIST requested_period_us=%u period_ticks=%u tick_us=%u mtime_hz=%u elapsed_us=%u effective_period_us=%u\n",
	       (unsigned)EXEC_PC_SAMPLE, (unsigned)pc_period_ticks, (unsigned)(1000000u / CONFIG_SYS_CLOCK_TICKS_PER_SEC),
	       (unsigned)hz, (unsigned)elapsed_us, (unsigned)(pc_samples ? elapsed_us / pc_samples : 0));
	for (unsigned i = 0; i < EXEC_PC_TABLE; i++)
		if (pc_table[i].pc)
			printf("%lx %u\n", (unsigned long)pc_table[i].pc, (unsigned)pc_table[i].count);
	printf("PCHIST END\n");
}

#else /* EXEC_PC_SAMPLE == 0: nothing is compiled in */

static inline void pc_sampler_start(void) {}
static inline void pc_sampler_pause(void) {}
static inline void pc_sampler_resume(void) {}
static inline void pc_sampler_stop(void) {}
static inline void pc_sampler_report(void) {}

#endif
#endif
