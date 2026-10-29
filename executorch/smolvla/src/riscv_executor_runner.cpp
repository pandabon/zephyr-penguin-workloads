/*
 * ExecuTorch runner for the Penguin SoC (Zephyr, HTIF/UART console).
 *
 * Credits:
 *  - Derived from ExecuTorch's examples/arm/executor_runner/arm_executor_runner.cpp
 *    (Copyright (c) Meta Platforms, Inc. and affiliates; Copyright 2023-2024 Arm Limited
 *    and/or its affiliates; BSD-style license, see the LICENSE file in the ExecuTorch root).
 *  - Ported to RISC-V/Zephyr by Jonathan Lin (UC Berkeley), repo pandabon/zephyr-chipyard-sw,
 *    branch exeggutorch @ e12bfa38034, samples/exeggutorch/executor_runner/riscv_executor_runner.cpp
 *    (bf16 output printing, input fill, memory statistics).
 *  - Heap fix (empty BufferCleanup) and clean exit code: found while building the 2026-10-09
 *    bench ELFs; the same BufferCleanup change is in the author's working tree of the repo above.
 *
 * This copy (executorch/smolvla) derives from executorch/mnist's runner in this repo and adds
 * (written for this repo, no external origin): embedded input tensors (EXEC_INPUTS), an embedded
 * expected output with error metrics and PASS/FAIL (EXEC_EXPECTED, EXEC_TOL), minstret/mcycle around
 * execute(), a report of XNNPACK's detected RISC-V vector config, and the two pools in .noinit.
 *
 * STEP MODE (EXEC_STEP_MODE, written for this repo): the SmolVLA action-expert step. The .pte has two methods
 * (smolvla-explore/export_expert_step.py): kv_proj runs once, denoise_step runs 10 times (lerobot's flow-matching Euler
 * loop, x_t fed back); see run_step_mode() below.
 *
 * CHAIN MODE (EXEC_CHAIN_MODE, written for this repo): the SmolVLA "mini chain" (smolvla-explore/mini_chain.py,
 * export_mini_chain.py): one .pte with four methods - vision, vlm, kv_proj, denoise_step - run once in order, each method's
 * outputs fed to the next one on the target; execute() instructions per method call and a check of every stage against the
 * PyTorch reference; see run_chain_mode() below.
 *
 * EXTERNAL WEIGHTS (EXEC_PTD, written for this repo; recipe and measurements: smolvla-explore/fpga-prep/FPGA-PLAN.md section 3a): the .pte
 * holds the graph, its XNNPACK weights are in a .ptd (ExecuTorch FlatTensorDataMap, passed to load_method()). Mode embed: the real .ptd
 * is part of the image. Mode prng: only the .ptd header is in the image, the weight bytes are generated at boot into a .noinit buffer
 * (timing runs: nothing to send over the UART, no output check).
 *
 * Changes versus the exeggutorch version:
 *  - method_allocation_pool_size is a compile definition (ET_METHOD_POOL_SIZE, default 64 KiB;
 *    the MNIST MLP needs ~7 KB. The pool is a static array, so it is part of the loaded image).
 *  - main() ends with chipyard_sim_exit(rc): rc 0 only after "Model executed successfully",
 *    so the host (uart_tsi / Spike) sees PASS/FAIL instead of a fault after main() returns.
 */

#include <cmath>
#include <cstring>
#include <errno.h>
#include <inttypes.h>
#include <memory>
#include <utility>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#ifdef EXEC_HAVE_PTD
#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/runtime/core/named_data_map.h>
#endif
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

struct _reent * _impure_ptr = nullptr;

void *__dso_handle = nullptr;

#include <zephyr/toolchain.h>

extern "C" {
#include <chipyard/sim_exit.h>
}
#include "model_pte.h"
#include "pc_sampler.h" /* EXEC_PC_SAMPLE: opt-in PC sampler, compiled out when 0 */

/* EXEC_REPEAT=N (plain mode only): run execute() N times on the same inputs; default 1 = the old single run. */
#ifndef EXEC_REPEAT
#define EXEC_REPEAT 1
#endif

// XNNPACK internals, to report which RISC-V vector config it detected (see report_xnnpack_config()).
extern "C" {
#include "src/xnnpack/hardware-config.h"
}

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferCleanup;
using executorch::extension::BufferDataLoader;
#ifdef EXEC_HAVE_PTD
using executorch::extension::FlatTensorDataMap;
#endif
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

/**
 * The method_allocation_pool should be large enough to fit the setup, input
 * used and other data used like the planned memory pool (e.g. memory-planned
 * buffers to use for mutable tensor data).
 *
 * Both pools are __noinit (Zephyr's ".noinit" section, NOLOAD): they occupy RAM but not the
 * loaded image, so a big pool does not lengthen the UART/TSI load. They are NOT zeroed at boot;
 * ExecuTorch does not need that (MemoryAllocator hands out raw memory and every user initialises
 * what it allocates). Before, they sat in input_data_sec, a PROGBITS section, i.e. in the image.
 */
#ifndef ET_METHOD_POOL_SIZE
#define ET_METHOD_POOL_SIZE (64 * 1024)
#endif
const size_t method_allocation_pool_size = ET_METHOD_POOL_SIZE;
static unsigned char __noinit __attribute__((aligned(16))) method_allocation_pool[method_allocation_pool_size];

/**
 * The temp_allocation_pool is used for allocating temporary data during kernel
 * or delegate execution. This will be reset after each kernel or delegate call.
 * Currently a MemoryAllocator is used but a PlatformMemoryAllocator is probably
 * a better fit
 */
#ifndef ET_TEMP_POOL_SIZE
#define ET_TEMP_POOL_SIZE 1024
#endif
const size_t temp_allocation_pool_size = ET_TEMP_POOL_SIZE;
static unsigned char __noinit __attribute__((aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

#ifdef EXEC_HAVE_PTD
/* External weights. g_named_data_map is what every load_method() gets; it stays null without EXEC_PTD (then load_method() is called as before). */
static const executorch::runtime::NamedDataMap *g_named_data_map = nullptr;
#endif

#ifdef EXEC_PTD_PRNG
/* Weights generated on the target (EXEC_PTD_MODE=prng). This .noinit buffer (NOLOAD: in RAM, not in the image, so not sent over the UART)
 * holds [copy of the .ptd header and key table | generated weight bytes] and is what the FlatTensorDataMap reads. 4096-aligned, which
 * covers the segment alignment inside a .ptd (>= 128, see extension/flat_tensor/serialize/serialize.py in forks/executorch
 * 85600f76e3). The values only have to be harmless, since timing does not depend on them: every 16-bit word is a small bf16 (sign random,
 * exponent 2^-13..2^-6, random mantissa). Two such words read as an f32 (lo, hi) are also finite and small, so bf16 and f32 weights both
 * stay free of NaN, inf and denormals. Generator: xorshift64* (Vigna, "An experimental exploration of Marsaglia's xorshift generators,
 * scrambled", 2014; public-domain constants 0x2545F4914F6CDD1D), four bf16 lanes per 64-bit word. */
static unsigned char __noinit __attribute__((aligned(4096))) ptd_buf[EXEC_PTD_SIZE];

static void fill_ptd_buffer()
{
	memcpy(ptd_buf, ptd_header, ptd_header_size);
	uint64_t x = 0x9E3779B97F4A7C15ull;
	uint64_t *p = (uint64_t *)(ptd_buf + ptd_header_size);
	size_t n = (EXEC_PTD_SIZE - ptd_header_size) / 8;
	for (size_t i = 0; i < n; i++)
	{
		x ^= x >> 12;
		x ^= x << 25;
		x ^= x >> 27;
		uint64_t r = x * 0x2545F4914F6CDD1Dull;
		/* per lane: sign = r bit 15, exponent = 0x72 + (r bits 8..10) -> 2^-13..2^-6, mantissa = r bits 0..6 */
		p[i] = (r & 0x8000800080008000ull) | ((((r >> 8) & 0x0007000700070007ull) + 0x0072007200720072ull) << 7) |
			   (r & 0x007F007F007F007Full);
	}
	memset(ptd_buf + ptd_header_size + n * 8, 0, (EXEC_PTD_SIZE - ptd_header_size) % 8); /* none for 128-aligned sizes */
}
#endif

#ifndef EXEC_TOL
#define EXEC_TOL 1e-4
#endif

static inline uint64_t read_minstret()
{
	uint64_t v;
	__asm__ volatile("csrr %0, minstret" : "=r"(v));
	return v;
}
static inline uint64_t read_mcycle()
{
	uint64_t v;
	__asm__ volatile("csrr %0, mcycle" : "=r"(v));
	return v;
}

/* void et_pal_init(void)
{
} */

ET_NORETURN void et_pal_abort(void)
{
#ifndef SEMIHOSTING
	__builtin_trap();
#else
	_exit(-1);
#endif
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 */
void et_pal_emit_log_message(ET_UNUSED et_timestamp_t timestamp, et_pal_log_level_t level, const char *filename,
							 ET_UNUSED const char *function, size_t line, const char *message, ET_UNUSED size_t length)
{
	fprintf(stderr, "%c [executorch:%s:%zu] %s\n", level, filename, line, message);
}

namespace
{

// Setup our own allocator that can show some extra stuff like used and free memory info
class RiscvMemoryAllocator : public executorch::runtime::MemoryAllocator
{
public:
	RiscvMemoryAllocator(uint32_t size, uint8_t *base_address) : MemoryAllocator(size, base_address), used_(0)
	{
	}

	void *allocate(size_t size, size_t alignment = kDefaultAlignment) override
	{
		void *ret = executorch::runtime::MemoryAllocator::allocate(size, alignment);
		if (ret != nullptr)
		{
			// Align with the same code as in MemoryAllocator::allocate() to keep
			// used_ "in sync" As alignment is expected to be power of 2 (checked by
			// MemoryAllocator::allocate()) we can check it the lower bits
			// (same as alignment - 1) is zero or not.
			if ((size & (alignment - 1)) == 0)
			{
				// Already aligned.
				used_ += size;
			}
			else
			{
				used_ = (used_ | (alignment - 1)) + 1 + size;
			}
		}
		return ret;
	}

	// Returns the used size of the allocator's memory buffer.
	size_t used_size() const
	{
		return used_;
	}

	// Returns the free size of the allocator's memory buffer.
	size_t free_size() const
	{
		return executorch::runtime::MemoryAllocator::size() - used_;
	}

private:
	size_t used_;
};

float tensor_value(const Tensor &t, size_t j)
{
	if (t.scalar_type() == ScalarType::BFloat16)
	{
		uint16_t raw = reinterpret_cast<const uint16_t *>(t.const_data_ptr())[j];
		uint32_t bits = (uint32_t)raw << 16;
		float val;
		std::memcpy(&val, &bits, sizeof(float));
		return val;
	}
	if (t.scalar_type() == ScalarType::Int)
		return (float)t.const_data_ptr<int>()[j];
	return t.const_data_ptr<float>()[j];
}

// Which RISC-V vector support did XNNPACK detect? (On Linux it asks getauxval(); in our Zephyr
// patch of hardware-config.c it uses the compile-time ISA plus the vlenb CSR.)
void report_xnnpack_config()
{
	const struct xnn_hardware_config *hw = xnn_init_hardware_config();
	if (hw == nullptr)
	{
		printf("XNNPACK hardware config: NULL\n");
		return;
	}
	printf("XNNPACK hw: arch_flags=0x%" PRIx64 " riscv_vector=%d fp16_arith=%d vlenb=%u\n", (uint64_t)hw->arch_flags,
		   (int)((hw->arch_flags & xnn_arch_riscv_vector) != 0),
		   (int)((hw->arch_flags & xnn_arch_riscv_vector_fp16_arith) != 0), (unsigned)hw->vlenb);
}

Result<BufferCleanup> prepare_input_tensors(Method &method, MemoryAllocator &allocator,
											std::vector<std::pair<char *, size_t>> &input_buffers)
{
	MethodMeta method_meta = method.method_meta();
	size_t num_inputs = method_meta.num_inputs();
	size_t num_allocated = 0;

	void **inputs = static_cast<void **>(allocator.allocate(num_inputs * sizeof(void *)));

	ET_CHECK_OR_RETURN_ERROR(inputs != nullptr, MemoryAllocationFailed,
							 "Could not allocate memory for pointers to input buffers.");

	for (size_t i = 0; i < num_inputs; i++)
	{
		auto tag = method_meta.input_tag(i);
		ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

		if (tag.get() != Tag::Tensor)
		{
			ET_LOG(Debug, "Skipping non-tensor input %zu", i);
			continue;
		}
		Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
		ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

		// Input is a tensor. Allocate a buffer for it.
		void *data_ptr = allocator.allocate(tensor_meta->nbytes());
		ET_CHECK_OR_RETURN_ERROR(data_ptr != nullptr, MemoryAllocationFailed,
								 "Could not allocate memory for input buffers.");
		inputs[num_allocated++] = data_ptr;

		Error err = Error::Ok;
		if (input_buffers.size() > 0)
		{
			auto [buffer, buffer_size] = input_buffers.at(i);
			if (buffer_size != tensor_meta->nbytes())
			{
				ET_LOG(Error, "input size (%zu) and tensor size (%zu) missmatch!", buffer_size, tensor_meta->nbytes());
				return Error::InvalidArgument;
			}
			else
			{
				ET_LOG(Info, "Copying embedded input %zu to tensor.", i);
				std::memcpy(data_ptr, buffer, buffer_size);
			}
		}

		TensorImpl impl = TensorImpl(tensor_meta.get().scalar_type(), tensor_meta.get().sizes().size(),
									 const_cast<TensorImpl::SizesType *>(tensor_meta.get().sizes().data()), data_ptr,
									 const_cast<TensorImpl::DimOrderType *>(tensor_meta.get().dim_order().data()));
		Tensor t(&impl);

		// If input_buffers.size <= 0, we don't have any input, fill t with 1's.
		if (input_buffers.size() <= 0)
		{
			for (size_t j = 0; j < t.numel(); j++)
			{
				switch (t.scalar_type())
				{
				case ScalarType::Int:
					t.mutable_data_ptr<int>()[j] = 1;
					break;
				case ScalarType::Float:
					t.mutable_data_ptr<float>()[j] = 1.;
					break;
				case ScalarType::BFloat16:
				{
					// bf16 1.0 = 0x3F80
					reinterpret_cast<uint16_t*>(data_ptr)[j] = 0x3F80u;
					break;
				}
				default:
					break;
				}
			}
		}

		err = method.set_input(t, i);

		if (err != Error::Ok)
		{
			ET_LOG(Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
			// The inputs live in the static method pool (allocator above), not
			// on the heap, so there is nothing to free(): an empty span keeps
			// ~BufferCleanup from calling free() on pool addresses.
			BufferCleanup cleanup{executorch::runtime::Span<void *>()};
			return err;
		}
	}
	// Same reason: returning BufferCleanup({inputs, num_allocated}) made its
	// destructor free() pool memory when main() returned, corrupting the
	// Zephyr sys_heap (fault in set_chunk_used after "Program complete").
	return BufferCleanup(executorch::runtime::Span<void *>());
}

} // namespace

#if defined(EXEC_STEP_MODE) || defined(EXEC_CHAIN_MODE)
/*
 * STEP MODE: the whole SmolVLA action-expert step on one .pte with two methods.
 * (Chain mode, further below, reuses the helpers of this block: Loaded, load_named, set_in, rel_l2.)
 *
 * Data layout (export_expert_step.py, "step_data.bin" / "step_expected.bin"):
 *   step_data     : noise x_1 f32 [1,50,32] | time table f32 [10,720] | 16 bf16 [1,241,320] = kv_proj inputs
 *                   (K1,V1,K3,V3,..K15,V15) | 16 bf16 [1,241,320] = prefix K/V of the self-attn layers (K0,V0,K2,..V14)
 *                   | WL-11: prefix_bias f32 [1,241] (0 valid / -FLT_MAX padded; slot padded to 16 bytes) | RoPE sin f32 [50,1,32] | cos f32 [50,1,32]
 *                   (positions n_valid..n_valid+49 of the self-attn layers)
 *   step_expected : final actions of the PyTorch lerobot-bf16 loop f32 | final actions of the fp32 loop f32 |
 *                   x after each of the 10 steps of the bf16 loop (10 x f32 [1,50,32])
 *                   | WL-11: lerobot's own final actions f32 [1,50,6] (the PASS/FAIL reference)
 * kv_proj  : 16 inputs -> 16 f32 outputs [1,241,320] (k_proj(K), v_proj(V) of the 8 cross-attn layers).
 * denoise_step: inputs x_t, time row, prefix_bias, sin, cos (WL-11), then K,V of layer 0..15 (cross-attn layers get
 *               kv_proj's outputs); output x_t + dt * v_t.
 * The measured window is exactly one Method::execute() call (minstret before/after). The set_input() copies (the
 * method's inputs are memory-planned buffers that ExecuTorch may reuse once an input is dead, so every input is set
 * again before every step) are outside it.
 */
namespace
{
constexpr size_t kLayers = 16, kSteps = 10, kChunk = 50, kAct = 32, kHidden = 720, kPrefix = 241, kKv = 320;
constexpr size_t kRealAct = 6, kRopeHalf = 32, kFirstKv = 5; // WL-11: lerobot's action dims; head_dim/2; denoise_step inputs before K0
constexpr size_t kXBytes = kChunk * kAct * sizeof(float);                // 6400
constexpr size_t kTimeBytes = kSteps * kHidden * sizeof(float);          // 28800
constexpr size_t kKvBytes = kPrefix * kKv * sizeof(uint16_t);            // one bf16 prefix K or V
constexpr size_t kOffTime = kXBytes, kOffKvIn = kOffTime + kTimeBytes, kOffSelf = kOffKvIn + 16 * kKvBytes;
constexpr size_t kOffPad = kOffSelf + 16 * kKvBytes;                     // WL-11 inputs: prefix_bias, sin, cos
constexpr size_t kPadBytes = (kPrefix * sizeof(float) + 15) / 16 * 16; // prefix_bias slot, padded to 16 bytes
constexpr size_t kRopeBytes = kChunk * kRopeHalf * sizeof(float);
constexpr size_t kDataSize = kOffPad + kPadBytes + 2 * kRopeBytes;
constexpr size_t kExpSize = (2 + kSteps) * kXBytes + kChunk * kRealAct * sizeof(float);

struct Loaded
{
	std::vector<Span<uint8_t>> spans;
	std::unique_ptr<HierarchicalAllocator> planned;
	std::unique_ptr<MemoryManager> manager;
	std::unique_ptr<Method> method;
	size_t planned_bytes = 0;
};

bool load_named(Program &program, const char *name, MemoryAllocator &method_alloc, MemoryAllocator &temp_alloc,
				Loaded &out)
{
	Result<MethodMeta> meta = program.method_meta(name);
	if (!meta.ok())
	{
		printf("FAIL: no method %s (0x%x)\n", name, (unsigned)meta.error());
		return false;
	}
	for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id)
	{
		size_t n = static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
		uint8_t *b = reinterpret_cast<uint8_t *>(method_alloc.allocate(n));
		if (b == nullptr)
		{
			printf("FAIL: method pool too small for %s planned buffer %zu (%zu bytes); raise ET_METHOD_POOL_SIZE\n", name,
				   id, n);
			return false;
		}
		out.spans.push_back({b, n});
		out.planned_bytes += n;
	}
	out.planned = std::make_unique<HierarchicalAllocator>(Span<Span<uint8_t>>(out.spans.data(), out.spans.size()));
	out.manager = std::make_unique<MemoryManager>(&method_alloc, out.planned.get(), &temp_alloc);
#ifdef EXEC_HAVE_PTD
	Result<Method> m = program.load_method(name, out.manager.get(), nullptr, g_named_data_map);
#else
	Result<Method> m = program.load_method(name, out.manager.get());
#endif
	if (!m.ok())
	{
		printf("FAIL: load_method(%s) = 0x%x\n", name, (unsigned)m.error());
		return false;
	}
	out.method = std::make_unique<Method>(std::move(m.get()));
	printf("loaded method %s: %zu inputs, %zu outputs, planned memory %zu bytes\n", name, out.method->inputs_size(),
		   out.method->outputs_size(), out.planned_bytes);
	return true;
}

// Set input i of `m` from raw data (shape and dtype from the method's own metadata); ExecuTorch copies it.
Error set_in(Method &m, size_t i, const void *data)
{
	MethodMeta mm = m.method_meta();
	auto info = mm.input_tensor_meta(i);
	ET_CHECK_OK_OR_RETURN_ERROR(info.error());
	TensorImpl impl(info->scalar_type(), info->sizes().size(), const_cast<TensorImpl::SizesType *>(info->sizes().data()),
					const_cast<void *>(data), const_cast<TensorImpl::DimOrderType *>(info->dim_order().data()));
	Tensor t(&impl);
	return m.set_input(t, i);
}

double rel_l2(const float *a, const float *ref, size_t n)
{
	double e = 0, r = 0;
	for (size_t j = 0; j < n; ++j)
	{
		double d = (double)a[j] - (double)ref[j];
		e += d * d;
		r += (double)ref[j] * (double)ref[j];
	}
	return std::sqrt(e) / std::sqrt(r);
}
} // namespace
#endif // EXEC_STEP_MODE || EXEC_CHAIN_MODE

#ifdef EXEC_STEP_MODE
static int run_step_mode(Program &program)
{
	if (step_data_size != kDataSize || step_expected_size != kExpSize)
	{
		printf("FAIL: step data sizes %u / %u, expected %zu / %zu\n", step_data_size, step_expected_size, kDataSize,
			   kExpSize);
		return 1;
	}
	RiscvMemoryAllocator method_alloc(method_allocation_pool_size, method_allocation_pool);
	RiscvMemoryAllocator temp_alloc(temp_allocation_pool_size, temp_allocation_pool);
	Loaded kv, dn;
	if (!load_named(program, "kv_proj", method_alloc, temp_alloc, kv) ||
		!load_named(program, "denoise_step", method_alloc, temp_alloc, dn))
		return 1;
	printf("method pool used after loading both methods: %zu of %zu bytes\n", method_alloc.used_size(),
		   method_allocation_pool_size);
	if (kv.method->inputs_size() != 16 || kv.method->outputs_size() != 16 || dn.method->inputs_size() != kFirstKv + 2 * kLayers ||
		dn.method->outputs_size() != 1)
	{
		printf("FAIL: unexpected method signatures\n");
		return 1;
	}

	const uint8_t *data = step_data;
	const float *exp_final = reinterpret_cast<const float *>(step_expected);
	const float *exp_fp32 = exp_final + kChunk * kAct;
	const float *exp_steps = exp_fp32 + kChunk * kAct;
	const float *exp_lerobot = exp_steps + kSteps * kChunk * kAct; // [50,6]

	// ---- kv_proj, once ----
	for (size_t i = 0; i < 16; ++i)
		if (set_in(*kv.method, i, data + kOffKvIn + i * kKvBytes) != Error::Ok)
		{
			printf("FAIL: kv_proj set_input %zu\n", i);
			return 1;
		}
	uint64_t i0 = read_minstret();
	Error st = kv.method->execute();
	uint64_t kv_instret = read_minstret() - i0;
	printf("STEP kv_proj: minstret delta = %" PRIu64 " status=0x%x\n", kv_instret, (unsigned)st);
	if (st != Error::Ok)
		return 1;
	std::vector<EValue> kvo(16);
	if (kv.method->get_outputs(kvo.data(), kvo.size()) != Error::Ok)
		return 1;

	// ---- denoise_step x 10 ----
	static float xbuf[kChunk * kAct];
	std::memcpy(xbuf, data, kXBytes);
	uint64_t dn_total = 0;
	double worst_step = 0;
	for (size_t step = 0; step < kSteps; ++step)
	{
		bool ok = set_in(*dn.method, 0, xbuf) == Error::Ok &&
				  set_in(*dn.method, 1, data + kOffTime + step * kHidden * sizeof(float)) == Error::Ok &&
				  set_in(*dn.method, 2, data + kOffPad) == Error::Ok &&
				  set_in(*dn.method, 3, data + kOffPad + kPadBytes) == Error::Ok &&
				  set_in(*dn.method, 4, data + kOffPad + kPadBytes + kRopeBytes) == Error::Ok;
		for (size_t l = 0; ok && l < kLayers; ++l)
			for (size_t j = 0; ok && j < 2; ++j)
			{
				size_t in_idx = kFirstKv + 2 * l + j, pair = 2 * (l / 2) + j;
				if (l % 2 == 1) // cross-attn layer: kv_proj's output for this layer
					ok = dn.method->set_input(kvo[pair], in_idx) == Error::Ok;
				else // self-attn layer: the prefix K/V as they are
					ok = set_in(*dn.method, in_idx, data + kOffSelf + pair * kKvBytes) == Error::Ok;
			}
		if (!ok)
		{
			printf("FAIL: denoise_step set_input (step %zu)\n", step);
			return 1;
		}
		uint64_t s0 = read_minstret();
		st = dn.method->execute();
		uint64_t d = read_minstret() - s0;
		dn_total += d;
		if (st != Error::Ok)
		{
			printf("FAIL: denoise_step execute step %zu status 0x%x\n", step, (unsigned)st);
			return 1;
		}
		std::vector<EValue> o(1);
		if (dn.method->get_outputs(o.data(), 1) != Error::Ok)
			return 1;
		const Tensor ot = o[0].toTensor();
		std::memcpy(xbuf, ot.const_data_ptr<float>(), kXBytes);
		double r = rel_l2(xbuf, exp_steps + step * kChunk * kAct, kChunk * kAct);
		worst_step = r > worst_step ? r : worst_step;
		printf("STEP denoise %zu: minstret delta = %" PRIu64 "  relL2 of x vs PyTorch lerobot-bf16 = %e\n", step + 1, d, r);
	}

	double r_bf16 = rel_l2(xbuf, exp_final, kChunk * kAct), r_fp32 = rel_l2(xbuf, exp_fp32, kChunk * kAct);
	// lerobot returns only the first 6 of the 32 action dims
	double le = 0, lr = 0;
	for (size_t t = 0; t < kChunk; ++t)
		for (size_t c = 0; c < kRealAct; ++c)
		{
			double d = (double)xbuf[t * kAct + c] - (double)exp_lerobot[t * kRealAct + c];
			le += d * d;
			lr += (double)exp_lerobot[t * kRealAct + c] * (double)exp_lerobot[t * kRealAct + c];
		}
	double r_lerobot = std::sqrt(le) / std::sqrt(lr);
	bool finite = true;
	for (size_t j = 0; j < kChunk * kAct; ++j)
		finite = finite && std::isfinite(xbuf[j]);
	printf("kv_proj minstret = %" PRIu64 "\n", kv_instret);
	printf("denoise_step minstret total (10 steps) = %" PRIu64 ", per step mean = %" PRIu64 "\n", dn_total,
		   dn_total / kSteps);
	printf("TOTAL action step (kv_proj + 10 denoise_step) minstret = %" PRIu64 "\n", kv_instret + dn_total);
	printf("final actions[0..3] = %f %f %f %f\n", xbuf[0], xbuf[1], xbuf[2], xbuf[3]);
	printf("final actions relL2 vs PyTorch lerobot-bf16 = %e (tol %e), vs PyTorch fp32 = %e\n", r_bf16, (double)EXEC_TOL,
		   r_fp32);
	printf("final actions[:, :6] relL2 vs LEROBOT's actions = %e (tol %e)\n", r_lerobot, (double)EXEC_TOL);
	bool pass = finite && r_lerobot < (double)EXEC_TOL;
	printf("%s\n", pass ? "PASS" : "FAIL");
	return pass ? 0 : 1;
}
#endif // EXEC_STEP_MODE

#ifdef EXEC_CHAIN_MODE
/*
 * CHAIN MODE: the SmolVLA mini chain (smolvla-explore/mini_chain.py), four methods in one .pte, run once in this order:
 *   vision(pixels) -> image tokens;  vlm(image tokens, host embeddings, RoPE tables, bias) -> K0,V0,K1,V1,..;
 *   kv_proj(K,V of the cross-attention layers) -> projected K/V;  denoise_step x `steps` (flow matching Euler loop, x fed back).
 * Every method's output is passed to the next one's input with set_input(EValue), nothing leaves the target in between.
 *
 * Data (mini_chain.write_chain_data / write_chain_expected): a 32-byte header (u32 magic "MCH1", steps, layers, P, 4 x 0), then
 * the tensors, each padded to 16 bytes. Their sizes come from the methods' own metadata, so the same runner serves any P,
 * layer count or step count. chain_data: pixels | language emb | state emb | VLM sin | cos | bias | noise | time table
 * f32 [steps,720] | prefix_bias | expert sin | cos. chain_expected: image tokens | K0,V0,K1,V1,.. | x after each step |
 * the fp32 chain's final actions (reported only).
 * PASS = every K/V and the final actions within EXEC_TOL of the bf16 PyTorch reference (image tokens are reported, not gated:
 * their error shows in K/V). The measured window is one Method::execute() call; set_input() copies are outside it.
 */
namespace
{
constexpr uint32_t kChainMagic = 0x3148434D;
constexpr size_t kTimeRow = kHidden * sizeof(float);

struct Cursor // walks a blob of 16-byte padded parts
{
	const uint8_t *base;
	size_t off;
	const uint8_t *take(size_t n)
	{
		const uint8_t *p = base + off;
		off += (n + 15) / 16 * 16;
		return p;
	}
};

size_t in_bytes(Method &m, size_t i) { return m.method_meta().input_tensor_meta(i)->nbytes(); }
size_t out_bytes(Method &m, size_t i) { return m.method_meta().output_tensor_meta(i)->nbytes(); }

// rel L2 of a method output against expected bytes in the output's own dtype (f32 or bf16 bit patterns)
double rel_l2_tensor(const Tensor &t, const uint8_t *exp)
{
	double e = 0, r = 0;
	for (size_t j = 0; j < (size_t)t.numel(); ++j)
	{
		double x;
		if (t.scalar_type() == ScalarType::BFloat16)
		{
			uint32_t bits = (uint32_t)reinterpret_cast<const uint16_t *>(exp)[j] << 16;
			float f;
			std::memcpy(&f, &bits, sizeof f);
			x = f;
		}
		else
			x = reinterpret_cast<const float *>(exp)[j];
		double d = (double)tensor_value(t, j) - x;
		e += d * d;
		r += x * x;
	}
	return std::sqrt(e) / std::sqrt(r);
}
} // namespace

static int run_chain_mode(Program &program)
{
	if (chain_data_size < 32 || chain_expected_size < 32)
	{
		printf("FAIL: chain data missing\n");
		return 1;
	}
	const uint32_t *hd = reinterpret_cast<const uint32_t *>(chain_data), *he = reinterpret_cast<const uint32_t *>(chain_expected);
	if (hd[0] != kChainMagic || he[0] != kChainMagic || hd[1] != he[1] || hd[2] != he[2] || hd[3] != he[3])
	{
		printf("FAIL: chain data / expected headers differ or have a wrong magic\n");
		return 1;
	}
	const size_t steps = hd[1], layers = hd[2], prefix = hd[3], n_cross = layers / 2;
	printf("chain: %zu layers, %zu denoise steps, prefix P = %zu\n", layers, steps, prefix);

	RiscvMemoryAllocator method_alloc(method_allocation_pool_size, method_allocation_pool);
	RiscvMemoryAllocator temp_alloc(temp_allocation_pool_size, temp_allocation_pool);
	Loaded vis, vlm, kvp, dn;
	if (!load_named(program, "vision", method_alloc, temp_alloc, vis) || !load_named(program, "vlm", method_alloc, temp_alloc, vlm) ||
		!load_named(program, "kv_proj", method_alloc, temp_alloc, kvp) ||
		!load_named(program, "denoise_step", method_alloc, temp_alloc, dn))
		return 1;
	printf("method pool used after loading the four methods: %zu of %zu bytes\n", method_alloc.used_size(),
		   method_allocation_pool_size);
	Method &V = *vis.method, &L = *vlm.method, &K = *kvp.method, &D = *dn.method;
	if (V.inputs_size() != 1 || V.outputs_size() != 1 || L.inputs_size() != 6 || L.outputs_size() != 2 * layers ||
		K.inputs_size() != 2 * n_cross || K.outputs_size() != 2 * n_cross || D.inputs_size() != kFirstKv + 2 * layers ||
		D.outputs_size() != 1 || in_bytes(D, 1) != kTimeRow || in_bytes(D, 2) != prefix * sizeof(float))
	{
		printf("FAIL: unexpected method signatures\n");
		return 1;
	}

	// ---- slice the data (sizes from the methods' metadata)
	Cursor dc{chain_data, 32}, ec{chain_expected, 32};
	const uint8_t *pix = dc.take(in_bytes(V, 0)), *lang = dc.take(in_bytes(L, 1)), *state = dc.take(in_bytes(L, 2)),
				  *vsin = dc.take(in_bytes(L, 3)), *vcos = dc.take(in_bytes(L, 4)), *vbias = dc.take(in_bytes(L, 5)),
				  *noise = dc.take(in_bytes(D, 0)), *times = dc.take(steps * kTimeRow), *pbias = dc.take(in_bytes(D, 2)),
				  *esin = dc.take(in_bytes(D, 3)), *ecos = dc.take(in_bytes(D, 4));
	const uint8_t *x_img = ec.take(out_bytes(V, 0));
	std::vector<const uint8_t *> x_kv, x_steps;
	for (size_t i = 0; i < 2 * layers; ++i)
		x_kv.push_back(ec.take(out_bytes(L, i)));
	for (size_t s = 0; s < steps; ++s)
		x_steps.push_back(ec.take(out_bytes(D, 0)));
	const uint8_t *x_fp32 = ec.take(out_bytes(D, 0));
	if (dc.off != chain_data_size || ec.off != chain_expected_size)
	{
		printf("FAIL: chain data sizes %zu / %zu bytes, the methods need %zu / %zu\n", (size_t)chain_data_size,
			   (size_t)chain_expected_size, dc.off, ec.off);
		return 1;
	}

	struct Row
	{
		std::string name; // a copy: callers pass a reused stack buffer
		uint64_t instr;
	};
	std::vector<Row> rows;
	auto run = [&](Method &m, const char *name) {
		uint64_t i0 = read_minstret();
		Error st = m.execute();
		uint64_t d = read_minstret() - i0;
		rows.push_back({name, d});
		printf("CHAIN %s: minstret delta = %" PRIu64 " status=0x%x\n", name, d, (unsigned)st);
		return st == Error::Ok;
	};
	bool pass = true;
	auto stage = [&](const char *name, const Tensor &t, const uint8_t *exp, bool gated) {
		double r = rel_l2_tensor(t, exp);
		bool ok = std::isfinite(r) && r <= (double)EXEC_TOL;
		printf("STAGE %-14s relL2 vs PyTorch bf16 chain = %e  %s\n", name, r, gated ? (ok ? "ok" : "OVER TOL") : "(reported)");
		if (gated && !ok)
			pass = false;
	};

	// ---- vision
	if (set_in(V, 0, pix) != Error::Ok || !run(V, "vision"))
		return 1;
	std::vector<EValue> img(1);
	if (V.get_outputs(img.data(), 1) != Error::Ok)
		return 1;
	stage("image tokens", img[0].toTensor(), x_img, false);

	// ---- vlm: image tokens (chained) + host embeddings, RoPE tables, bias
	bool ok = L.set_input(img[0], 0) == Error::Ok && set_in(L, 1, lang) == Error::Ok && set_in(L, 2, state) == Error::Ok &&
			  set_in(L, 3, vsin) == Error::Ok && set_in(L, 4, vcos) == Error::Ok && set_in(L, 5, vbias) == Error::Ok;
	if (!ok || !run(L, "vlm"))
	{
		printf("FAIL: vlm\n");
		return 1;
	}
	std::vector<EValue> kv(2 * layers);
	if (L.get_outputs(kv.data(), kv.size()) != Error::Ok)
		return 1;
	for (size_t i = 0; i < 2 * layers; ++i)
	{
		char nm[16];
		snprintf(nm, sizeof nm, "%c%zu", i % 2 ? 'V' : 'K', i / 2);
		stage(nm, kv[i].toTensor(), x_kv[i], true);
	}

	// ---- kv_proj: K/V of the cross-attention layers (odd layer indices)
	for (size_t c = 0; ok && c < n_cross; ++c)
		for (size_t j = 0; ok && j < 2; ++j)
			ok = K.set_input(kv[2 * (2 * c + 1) + j], 2 * c + j) == Error::Ok;
	if (!ok || !run(K, "kv_proj"))
	{
		printf("FAIL: kv_proj\n");
		return 1;
	}
	std::vector<EValue> proj(2 * n_cross);
	if (K.get_outputs(proj.data(), proj.size()) != Error::Ok)
		return 1;

	// ---- denoise_step x steps
	static float xbuf[kChunk * kAct];
	std::memcpy(xbuf, noise, kXBytes);
	for (size_t s = 0; s < steps; ++s)
	{
		ok = set_in(D, 0, xbuf) == Error::Ok && set_in(D, 1, times + s * kTimeRow) == Error::Ok && set_in(D, 2, pbias) == Error::Ok &&
			 set_in(D, 3, esin) == Error::Ok && set_in(D, 4, ecos) == Error::Ok;
		for (size_t l = 0; ok && l < layers; ++l)
			for (size_t j = 0; ok && j < 2; ++j)
				ok = D.set_input(l % 2 == 1 ? proj[2 * (l / 2) + j] : kv[2 * l + j], kFirstKv + 2 * l + j) == Error::Ok;
		char nm[16];
		snprintf(nm, sizeof nm, "denoise_%zu", s + 1);
		if (!ok || !run(D, nm))
		{
			printf("FAIL: denoise_step %zu\n", s + 1);
			return 1;
		}
		std::vector<EValue> o(1);
		if (D.get_outputs(o.data(), 1) != Error::Ok)
			return 1;
		const Tensor ot = o[0].toTensor();
		std::memcpy(xbuf, ot.const_data_ptr<float>(), kXBytes);
		snprintf(nm, sizeof nm, "x after step %zu", s + 1);
		stage(nm, ot, x_steps[s], s + 1 == steps); // the last one is the final actions
	}
	double r32 = rel_l2(xbuf, reinterpret_cast<const float *>(x_fp32), kChunk * kAct);
	bool finite = true;
	for (size_t j = 0; j < kChunk * kAct; ++j)
		finite = finite && std::isfinite(xbuf[j]);

	// ---- instruction table
	uint64_t total = 0;
	printf("CHAIN method        instructions (one execute() call each)\n");
	for (const Row &r : rows)
	{
		printf("CHAIN %-14s %12" PRIu64 "\n", r.name.c_str(), r.instr);
		total += r.instr;
	}
	printf("CHAIN TOTAL          %12" PRIu64 "  (sum of execute() windows)\n", total);
	printf("final actions[0..3] = %f %f %f %f\n", xbuf[0], xbuf[1], xbuf[2], xbuf[3]);
	printf("final actions relL2 vs PyTorch fp32 chain = %e (reported)\n", r32);
	pass = pass && finite;
	printf("%s (tol %e on every K/V and the final actions)\n", pass ? "PASS" : "FAIL", (double)EXEC_TOL);
	return pass ? 0 : 1;
}
#endif // EXEC_CHAIN_MODE

static int g_rc = 1; // 0 only after "Model executed successfully"
int main()
{
	executorch::runtime::runtime_init();
	report_xnnpack_config();
	std::vector<std::pair<char *, size_t>> input_buffers;
	size_t pte_size = model_pte_size;

	ET_LOG(Info, "Model in %p %c", model_pte, model_pte[0]);
	auto loader = BufferDataLoader(model_pte, pte_size);
	ET_LOG(Info, "Model PTE file loaded. Size: %lu bytes.", pte_size);
	Result<Program> program = Program::load(&loader);
	if (!program.ok())
	{
		ET_LOG(Info, "Program loading failed @ 0x%p: 0x%" PRIx32, model_pte, program.error());
	}
#ifdef EXEC_HAVE_PTD
	/* External weights: generate (prng) or just point at (embed) the .ptd bytes, and open them as a NamedDataMap that stays alive until
	 * main() ends (the Methods keep pointers into it). */
#ifdef EXEC_PTD_PRNG
	uint64_t fill_i0 = read_minstret(), fill_c0 = read_mcycle();
	fill_ptd_buffer();
	uint64_t fill_i1 = read_minstret(), fill_c1 = read_mcycle();
	printf("PRNG weights: %u bytes (.ptd header %u + generated %u), fill: mcycle delta = %" PRIu64 ", minstret delta = %" PRIu64 "\n",
		   (unsigned)EXEC_PTD_SIZE, ptd_header_size, (unsigned)(EXEC_PTD_SIZE - ptd_header_size), fill_c1 - fill_c0, fill_i1 - fill_i0);
	BufferDataLoader ptd_loader(ptd_buf, EXEC_PTD_SIZE);
#else
	printf("external weights: real .ptd embedded in the image, %u bytes\n", ptd_embedded_size);
	BufferDataLoader ptd_loader(ptd_embedded, ptd_embedded_size);
#endif
	Result<FlatTensorDataMap> ptd_map = FlatTensorDataMap::load(&ptd_loader);
	if (!ptd_map.ok())
	{
		printf("FAIL: FlatTensorDataMap::load = 0x%x\n", (unsigned)ptd_map.error());
		chipyard_sim_exit(1);
	}
	g_named_data_map = &*ptd_map;
#endif
#ifdef EXEC_STEP_MODE
	if (!program.ok())
		chipyard_sim_exit(1);
	g_rc = run_step_mode(*program);
	ET_LOG(Info, "Program complete, exiting.");
	ET_LOG(Info, "\04");
	chipyard_sim_exit(g_rc);
#endif
#ifdef EXEC_CHAIN_MODE
	if (!program.ok())
		chipyard_sim_exit(1);
	g_rc = run_chain_mode(*program);
	ET_LOG(Info, "Program complete, exiting.");
	ET_LOG(Info, "\04");
	chipyard_sim_exit(g_rc);
#endif

	ET_LOG(Info, "Model buffer loaded, has %lu methods", program->num_methods());

	const char *method_name = nullptr;
	{
		const auto method_name_result = program->get_method_name(0);
		ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
		method_name = *method_name_result;
	}
	ET_LOG(Info, "Running method %s", method_name);

	Result<MethodMeta> method_meta = program->method_meta(method_name);
	if (!method_meta.ok())
	{
		ET_LOG(Info, "Failed to get method_meta for %s: 0x%x", method_name, (unsigned int)method_meta.error());
	}

	ET_LOG(Info, "Setup Method allocator pool. Size: %lu bytes.", method_allocation_pool_size);

	RiscvMemoryAllocator method_allocator(method_allocation_pool_size, method_allocation_pool);

	std::vector<uint8_t *> planned_buffers;	  // Owns the memory
	std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
	size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

	size_t planned_buffer_membase = method_allocator.used_size();

	for (size_t id = 0; id < num_memory_planned_buffers; ++id)
	{
		size_t buffer_size = static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
		ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);

		/* Move to it's own allocator when MemoryPlanner is in place. */
		uint8_t *buffer = reinterpret_cast<uint8_t *>(method_allocator.allocate(buffer_size));
		planned_buffers.push_back(buffer);
		planned_spans.push_back({planned_buffers.back(), buffer_size});
	}

	size_t planned_buffer_memsize = method_allocator.used_size() - planned_buffer_membase;

	HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});

	RiscvMemoryAllocator temp_allocator(temp_allocation_pool_size, temp_allocation_pool);

	MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

	size_t method_loaded_membase = method_allocator.used_size();

#ifdef EXEC_HAVE_PTD
	Result<Method> method = program->load_method(method_name, &memory_manager, nullptr, g_named_data_map);
#else
	Result<Method> method = program->load_method(method_name, &memory_manager);
#endif
	if (!method.ok())
	{
		ET_LOG(Info, "Loading of method %s failed with status 0x%" PRIx32, method_name, method.error());
	}
	size_t method_loaded_memsize = method_allocator.used_size() - method_loaded_membase;
	ET_LOG(Info, "Method loaded.");

#ifdef EXEC_HAVE_INPUTS
	// Split the embedded blob into one buffer per tensor input, sized by the method's own metadata.
	{
		size_t off = 0;
		for (size_t i = 0; i < method_meta->num_inputs(); i++)
		{
			if (method_meta->input_tag(i).get() != Tag::Tensor)
				continue;
			size_t n = method_meta->input_tensor_meta(i)->nbytes();
			input_buffers.push_back({(char *)(exec_inputs + off), n});
			off += n;
		}
		ET_LOG(Info, "Embedded inputs: %zu tensors, %zu of %u bytes used.", input_buffers.size(), off,
			   exec_inputs_size);
		if (off != exec_inputs_size)
			ET_LOG(Error, "EXEC_INPUTS size does not match the model's inputs!");
	}
#endif
	ET_LOG(Info, "Preparing inputs...");
	size_t input_membase = method_allocator.used_size();

	auto inputs = ::prepare_input_tensors(*method, method_allocator, input_buffers);

	if (!inputs.ok())
	{
		ET_LOG(Info, "Preparing inputs tensors for method %s failed with status 0x%" PRIx32, method_name,
			   inputs.error());
	}
	size_t input_memsize = method_allocator.used_size() - input_membase;
	ET_LOG(Info, "Input prepared.");

	ET_LOG(Info, "Starting the model execution...");
	size_t executor_membase = method_allocator.used_size();
	Error status = Error::Ok;
	size_t reprepared_bytes = 0; /* pool bytes used by re-setting the inputs between runs (not part of execute()) */
	pc_sampler_start();			 /* no-op unless EXEC_PC_SAMPLE > 0 */
	for (int rep = 0; rep < EXEC_REPEAT; ++rep)
	{
		if (rep > 0)
		{
			// execute() may overwrite its own input buffers (the memory planner reuses an input's memory once its last use is
			// done; without this the 2nd run read garbage and the output was -inf). So give the method its inputs again, outside the
			// timed and sampled window. Costs one more copy of the inputs in the method pool per extra run.
			pc_sampler_pause();
			size_t before = method_allocator.used_size();
			auto again = ::prepare_input_tensors(*method, method_allocator, input_buffers);
			ET_CHECK_MSG(again.ok(), "re-preparing inputs for execute() #%d failed", rep + 1);
			reprepared_bytes += method_allocator.used_size() - before;
			pc_sampler_resume();
		}
		uint64_t instret0 = read_minstret(), cycle0 = read_mcycle();
		status = method->execute();
		uint64_t instret1 = read_minstret(), cycle1 = read_mcycle();
#if EXEC_REPEAT > 1
		printf("execute() #%d: mcycle delta = %" PRIu64 ", minstret delta = %" PRIu64 "\n", rep + 1, cycle1 - cycle0,
			   instret1 - instret0);
#else
		printf("execute(): minstret delta = %" PRIu64 ", mcycle delta = %" PRIu64 "\n", instret1 - instret0,
			   cycle1 - cycle0);
#endif
		if (status != Error::Ok)
			break;
	}
	pc_sampler_stop();
	pc_sampler_report(); /* prints PCHIST BEGIN ... PCHIST END; no-op unless EXEC_PC_SAMPLE > 0 */
	size_t executor_memsize = method_allocator.used_size() - executor_membase - reprepared_bytes;

	ET_LOG(Info, "model_pte_loaded_size:     %lu bytes.", pte_size);

	if (method_allocator.size() != 0)
	{
		size_t method_allocator_used = method_allocator.used_size();
		ET_LOG(Info, "method_allocator_used:     %zu / %zu  free: %zu ( used: %zu %% ) ", method_allocator_used,
			   method_allocator.size(), method_allocator.free_size(),
			   100 * method_allocator_used / method_allocator.size());
		ET_LOG(Info, "method_allocator_planned:  %zu bytes", planned_buffer_memsize);
		ET_LOG(Info, "method_allocator_loaded:   %zu bytes", method_loaded_memsize);
		ET_LOG(Info, "method_allocator_input:    %zu bytes", input_memsize);
		ET_LOG(Info, "method_allocator_executor: %zu bytes", executor_memsize);
	}
	if (temp_allocator.size() > 0)
	{
		ET_LOG(Info, "temp_allocator_used:       %zu / %zu free: %zu ( used: %zu %% ) ", temp_allocator.used_size(),
			   temp_allocator.size(), temp_allocator.free_size(),
			   100 * temp_allocator.used_size() / temp_allocator.size());
	}

	if (status != Error::Ok)
	{
		ET_LOG(Info, "Execution of method %s failed with status 0x%" PRIx32, method_name, status);
	}
	else
	{
		ET_LOG(Info, "Model executed successfully.");
		g_rc = 0;
	}

	std::vector<EValue> outputs(method->outputs_size());
	ET_LOG(Info, "%zu outputs: ", outputs.size());
	status = method->get_outputs(outputs.data(), outputs.size());
	ET_CHECK(status == Error::Ok);
#ifdef EXEC_HAVE_INPUTS
	const size_t print_limit = 8; // big outputs: first values only
#else
	const size_t print_limit = 64; // all-ones runs (MNIST: 10 logits): print everything
#endif
#ifdef EXEC_HAVE_EXPECTED
	// Multi-output check (SmolVLA steps 6e/6f): when the expected blob is R >= 1 whole copies of ALL outputs back to back (same order,
	// each in its own dtype, no padding), every output is compared with every copy: "MULTI set s output i relL2". Copy 0 is the gated
	// one (tolerance EXEC_TOL on every output); further copies (e.g. lerobot's own K/V) are reported only. Single-output models and
	// expected blobs of exactly one output keep the old behaviour.
	bool multi_expected = false;
	size_t multi_total = 0;
	if (outputs.size() > 1)
	{
		for (size_t i = 0; i < outputs.size(); ++i)
			multi_total += outputs[i].toTensor().nbytes();
		multi_expected = multi_total > 0 && exec_expected_size % multi_total == 0;
	}
	if (multi_expected)
	{
		const size_t sets = exec_expected_size / multi_total;
		bool all_pass = true;
		for (size_t s = 0; s < sets; ++s)
		{
			size_t off = s * multi_total;
			for (size_t i = 0; i < outputs.size(); ++i)
			{
				Tensor t = outputs[i].toTensor();
				size_t n = t.numel(), esz = t.element_size();
				double err2 = 0, exp2 = 0;
				for (size_t j = 0; j < n; ++j)
				{
					double e;
					if (t.scalar_type() == ScalarType::BFloat16)
					{
						uint16_t raw;
						std::memcpy(&raw, exec_expected + off + j * 2, 2);
						uint32_t bits = (uint32_t)raw << 16;
						float v;
						std::memcpy(&v, &bits, sizeof(float));
						e = v;
					}
					else
					{
						float v;
						std::memcpy(&v, exec_expected + off + j * sizeof(float), sizeof(float));
						e = v;
					}
					double d = tensor_value(t, j) - e;
					err2 += d * d;
					exp2 += e * e;
				}
				double r = std::sqrt(err2) / std::sqrt(exp2);
				printf("MULTI set %zu output %zu relL2 = %e\n", s, i, r);
				if (s == 0 && !(r < (double)EXEC_TOL))
					all_pass = false;
				off += n * esz;
			}
		}
		printf("MULTI %s (tol %e on every output of set 0)\n", all_pass ? "PASS" : "FAIL", (double)EXEC_TOL);
		if (!all_pass)
			g_rc = 1;
	}
#endif
	for (int i = 0; i < outputs.size(); ++i)
	{
		Tensor t = outputs[i].toTensor();
		size_t n = t.numel();
		// The output might be collected and parsed so printf() is used instead of ET_LOG() here
		for (size_t j = 0; j < n && j < print_limit; ++j)
		{
			if (t.scalar_type() == ScalarType::Int)
				printf("Output[%d][%zu]: %d\n", i, j, t.const_data_ptr<int>()[j]);
			else
				printf("Output[%d][%zu]: %f\n", i, j, tensor_value(t, j));
		}
		if (n > print_limit)
			printf("Output[%d]: ... %zu values total\n", i, n);

		if (t.scalar_type() == ScalarType::Int)
			continue;
		double sum = 0;
		for (size_t j = 0; j < n; ++j)
			sum += tensor_value(t, j);
		printf("Output[%d] numel=%zu sum=%f\n", i, n, sum);
#ifdef EXEC_HAVE_EXPECTED
		if (i == 0 && !multi_expected)
		{
			// expected output has the output's own dtype (float32, or bf16 bit patterns), as written by
			// the export script
			const size_t esz = t.element_size();
			if (exec_expected_size != n * esz)
			{
				printf("FAIL: expected size %u != output %zu bytes\n", exec_expected_size, n * esz);
				g_rc = 1;
				continue;
			}
			auto expected_value = [&](size_t j) -> double {
				if (t.scalar_type() == ScalarType::BFloat16)
				{
					uint16_t raw;
					std::memcpy(&raw, exec_expected + j * 2, 2);
					uint32_t bits = (uint32_t)raw << 16;
					float v;
					std::memcpy(&v, &bits, sizeof(float));
					return v;
				}
				float v;
				std::memcpy(&v, exec_expected + j * sizeof(float), sizeof(float));
				return v;
			};
			double max_err = 0, max_exp = 0, err2 = 0, exp2 = 0;
			for (size_t j = 0; j < n; ++j)
			{
				double o = tensor_value(t, j), e = expected_value(j);
				double d = std::fabs(o - e);
				max_err = d > max_err ? d : max_err;
				max_exp = std::fabs(e) > max_exp ? std::fabs(e) : max_exp;
				err2 += d * d;
				exp2 += e * e;
			}
			double rel_l2 = std::sqrt(err2) / std::sqrt(exp2);
			printf("max|out-exp|=%e max|exp|=%e relL2=%e tol=%e\n", max_err, max_exp, rel_l2, (double)EXEC_TOL);
			bool pass = rel_l2 < (double)EXEC_TOL; // false for NaN as well
			printf("%s\n", pass ? "PASS" : "FAIL");
			if (!pass)
				g_rc = 1;
		}
#endif
	}
out:
#ifdef EXEC_PTD_PRNG
	printf("PRNG weights: no output check\n"); /* the outputs above come from random weights; only timing and finiteness mean anything */
#endif
	ET_LOG(Info, "Program complete, exiting.");
	ET_LOG(Info, "\04");
	chipyard_sim_exit(g_rc);
}
