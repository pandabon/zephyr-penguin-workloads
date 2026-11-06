// SPDX-License-Identifier: Apache-2.0
//
// penguin::linear_fp8.out: ExecuTorch custom-op kernel for fp8 (OFP8 E4M3) weight-only linears on Penguin (Saturn, RVV).
//
//   linear_fp8.out(Tensor x, Tensor w_packed, int out_features, *, Tensor(a!) out) -> Tensor(a!)
//   x: bf16 [..., K]   w_packed: uint8, 1-D, tiles of 32 output columns (layout below)   out: bf16 [..., N]
//   out[i][n] = bf16_RNE( s[n] * sum_k x[i][k] * float(q[n][k]) + bias[n] ), f32 accumulate, bias added after the scale.
//
// Numerics, packed layout and micro-kernel symbols are the interface contract of the fp8 effort (2026-11-06); the host reference that
// defines them is smolvla-explore/fp8_linear.py (pack_fp8 / unpack_fp8 / _ref). The same file registers the op in ExecuTorch.
//
// Packed layout, per tile t of NR = 32 columns (n0 = 32 t), tiles back to back, N padded up to a multiple of 32 (pad columns: q, s, bias = 0):
//     float bias[32] | uint8 q[K][32] | float scale[32]        = 128 + 32 K + 128 bytes
// The micro-kernel consumes this as is. It writes f32 rows of 32 columns per tile; this file converts f32 -> bf16 (round to nearest even).
//
// Scratch: a static f32 buffer of 5 rows x FP8_MAX_NPAD (default 4096, 80 KiB, .bss). Rows are processed in chunks of 5 (the 5x4v
// kernel's MR) and each chunk is converted to bf16 right after the GEMM, so the scratch does not grow with M. (The contract sketched a
// full [M][Npad] scratch; the result is identical.) The runtime temp allocator is not used: the runner's default pool is only 1 KiB.
//
// Credits: registration pattern from ExecuTorch examples/portable/custom_ops/custom_ops_2_out.cpp and
// extension/kernel_util/make_boxed_from_unboxed_functor.h (forks/executorch branch penguin-1.6 @ 85600f76e3, BSD-style licence of the
// ExecuTorch repo). The micro-kernel is xnn_bf16_qf8w_f32_gemm_minmax_ukernel_5x4v__rvvzvfofp8 of our XNNPACK fork.

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef FP8_MAX_NPAD
#define FP8_MAX_NPAD 4096
#endif

// Declared here, not through src/xnnpack/gemm.h: this file needs only the symbol, and the types below match the contract
// (a: bf16 as raw uint16_t, params = struct xnn_f32_minmax_params { struct { float min, max; } scalar; }, src/xnnpack/microparams.h).
struct fp8_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};
extern "C" void xnn_bf16_qf8w_f32_gemm_minmax_ukernel_5x4v__rvvzvfofp8(
    size_t mr, size_t nc, size_t kc, const uint16_t* a, size_t a_stride, const void* w, float* c, size_t cm_stride,
    size_t cn_stride, const fp8_minmax_params* params);

namespace penguin {
namespace native {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using executorch::runtime::Error;

namespace {

constexpr size_t kNR = 32;  // columns per packed tile (VLEN 256, one e32 m4 register group)
constexpr size_t kMR = 5;   // rows per micro-kernel call

alignas(16) float g_scratch[kMR * FP8_MAX_NPAD];

// f32 -> bf16, round to nearest even; NaN stays NaN (quiet bit set). Same result as XNNPACK's f32-bf16 vcvt kernels.
inline uint16_t f32_to_bf16_rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((u >> 16) | 0x40u);
  }
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

}  // namespace

Tensor& linear_fp8_out(KernelRuntimeContext& ctx, const Tensor& x, const Tensor& w_packed, int64_t out_features, Tensor& out) {
  ET_KERNEL_CHECK_MSG(ctx, x.scalar_type() == ScalarType::BFloat16, InvalidArgument, out, "linear_fp8: x must be bf16");
  ET_KERNEL_CHECK_MSG(ctx, out.scalar_type() == ScalarType::BFloat16, InvalidArgument, out, "linear_fp8: out must be bf16");
  ET_KERNEL_CHECK_MSG(ctx, w_packed.scalar_type() == ScalarType::Byte, InvalidArgument, out, "linear_fp8: w_packed must be uint8");
  ET_KERNEL_CHECK_MSG(ctx, x.dim() >= 1 && out.dim() == x.dim(), InvalidArgument, out, "linear_fp8: out rank must equal x rank");

  const size_t K = static_cast<size_t>(x.size(x.dim() - 1));
  const size_t N = static_cast<size_t>(out_features);
  const size_t Npad = (N + kNR - 1) / kNR * kNR;
  const size_t M = K ? static_cast<size_t>(x.numel()) / K : 0;
  ET_KERNEL_CHECK_MSG(ctx, N > 0 && K > 0 && static_cast<size_t>(out.size(out.dim() - 1)) == N, InvalidArgument, out,
                      "linear_fp8: out last dim must equal out_features");
  ET_KERNEL_CHECK_MSG(ctx, static_cast<size_t>(out.numel()) == M * N, InvalidArgument, out, "linear_fp8: out numel != M * N");
  ET_KERNEL_CHECK_MSG(ctx, static_cast<size_t>(w_packed.numel()) == Npad / kNR * (256 + kNR * K), InvalidArgument, out,
                      "linear_fp8: w_packed size does not match K and out_features");
  ET_KERNEL_CHECK_MSG(ctx, Npad <= FP8_MAX_NPAD, InvalidArgument, out,
                      "linear_fp8: out_features exceeds FP8_MAX_NPAD (rebuild with -DFP8_MAX_NPAD=...)");

  const uint16_t* a = reinterpret_cast<const uint16_t*>(x.const_data_ptr<executorch::aten::BFloat16>());
  uint16_t* o = reinterpret_cast<uint16_t*>(out.mutable_data_ptr<executorch::aten::BFloat16>());
  const void* w = w_packed.const_data_ptr<uint8_t>();
  const fp8_minmax_params params = {{-__builtin_inff(), __builtin_inff()}};   // no clamping

  for (size_t i = 0; i < M; i += kMR) {
    const size_t mr = (M - i < kMR) ? (M - i) : kMR;
    xnn_bf16_qf8w_f32_gemm_minmax_ukernel_5x4v__rvvzvfofp8(
        mr, N, K * sizeof(uint16_t), a + i * K, K * sizeof(uint16_t), w, g_scratch, Npad * sizeof(float), kNR * sizeof(float), &params);
    for (size_t r = 0; r < mr; ++r) {
      const float* src = g_scratch + r * Npad;
      uint16_t* dst = o + (i + r) * N;
      for (size_t n = 0; n < N; ++n) {
        dst[n] = f32_to_bf16_rne(src[n]);
      }
    }
  }
  return out;
}

}  // namespace native
}  // namespace penguin

// Schema (registered by smolvla-explore/fp8_linear.py on the export side):
//   penguin::linear_fp8.out(Tensor x, Tensor w_packed, int out_features, *, Tensor(a!) out) -> Tensor(a!)
EXECUTORCH_LIBRARY(penguin, "linear_fp8.out", penguin::native::linear_fp8_out);
