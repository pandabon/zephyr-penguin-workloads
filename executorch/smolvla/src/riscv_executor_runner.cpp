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
#include <stdio.h>
#include <unistd.h>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
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

// XNNPACK internals, to report which RISC-V vector config it detected (see report_xnnpack_config()).
extern "C" {
#include "src/xnnpack/hardware-config.h"
}

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferCleanup;
using executorch::extension::BufferDataLoader;
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

	Result<Method> method = program->load_method(method_name, &memory_manager);
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
	uint64_t instret0 = read_minstret(), cycle0 = read_mcycle();
	Error status = method->execute();
	uint64_t instret1 = read_minstret(), cycle1 = read_mcycle();
	printf("execute(): minstret delta = %" PRIu64 ", mcycle delta = %" PRIu64 "\n", instret1 - instret0,
		   cycle1 - cycle0);
	size_t executor_memsize = method_allocator.used_size() - executor_membase;

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
		if (i == 0)
		{
			// expected output is float32 (the .bin written by the export script)
			if (exec_expected_size != n * sizeof(float))
			{
				printf("FAIL: expected size %u != output %zu bytes\n", exec_expected_size, n * sizeof(float));
				g_rc = 1;
				continue;
			}
			const float *exp = reinterpret_cast<const float *>(exec_expected);
			double max_err = 0, max_exp = 0, err2 = 0, exp2 = 0;
			for (size_t j = 0; j < n; ++j)
			{
				double o = tensor_value(t, j), e = exp[j];
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
	ET_LOG(Info, "Program complete, exiting.");
	ET_LOG(Info, "\04");
	chipyard_sim_exit(g_rc);
}
