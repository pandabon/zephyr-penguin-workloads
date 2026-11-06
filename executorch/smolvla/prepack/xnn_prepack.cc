// xnn_prepack: pack XNNPACK fully-connected weights on a desktop (x86) host the
// way XNNPACK packs them on the Penguin board (RISC-V RVV, VLEN 256).
//
// What it does: reads a job list (one line per fully-connected layer: weight
// bytes, optional bias bytes, shapes), packs each layer into the exact byte
// layout that create_fully_connected_nc() would build on the board, and writes
// the packed buffers back to back (each at a 64-byte aligned offset) plus a
// result table (sizes, nr/kr/sr, cache seed, fingerprint id, ...). prepack.py
// wraps the result into an XPWC v2 file that ExecuTorch's weights cache can
// serve in place.
//
// Why it is built from the fork's own sources: enum values (operator type,
// fingerprint id, flag bits), the reference packers and xnn_use_nr2() all come
// from the XNNPACK fork, so a change there is picked up by a rebuild and
// nothing is copied by hand. Only the RISC-V gemm-config values (nr, kr, sr,
// element sizes) are written out here, each with the config line it comes from.
//
// Derived from (all in pandabon/XNNPACK, branch penguin, commit 7264d39dab):
//   src/operators/fully-connected-nc.c  create_fully_connected_nc(): strides,
//       packed size, allocation alignment, cache seed, which packer runs,
//       setup_variant_and_gemm_config(): nr2 choice, fingerprint ids,
//       f32_variant / bf16 create function: extra_weights_bytes = 0
//   src/configs/gemm-config.c  init_bf16_f32_gemm_config() (RISC-V branch),
//       init_f32_gemm_config_impl() and init_f32_gemm_nr2_config_impl()
//       (RISC-V branches): nr, packers
//   src/reference/packing.cc    xnn_pack_{f32,bf16_f32}_gemm_{goi,gio}_w
//       (linked unchanged; the RVV packw kernels write the same bytes)
//   src/microkernel-utils.c     xnn_use_nr2() (included unchanged)
//   src/xnnpack/common.h        XNN_ALLOCATION_ALIGNMENT (RISC-V value, see
//       xnn_prepack_support.cc)
//
// Build: see the header of prepack.py (one command).

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

// These headers come from the fork (-I <XNNPACK> -I <XNNPACK>/include).
#include "include/xnnpack.h"  // XNN_FLAG_TRANSPOSE_WEIGHTS
#include "src/operators/fingerprint_id.h"  // enum xnn_fingerprint_id
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"  // round_up, round_up_po2
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/operator-type.h"  // enum xnn_operator_type
#include "src/xnnpack/pack.h"

// microkernel-utils.c includes xnn_init_hardware_config(); xnn_use_nr2() does
// not call it, so a stub keeps the link small.
extern "C" const struct xnn_hardware_config* xnn_init_hardware_config() {
  return nullptr;
}
extern "C" {
#include "src/microkernel-utils.c"
}

// From xnn_prepack_support.cc, which is compiled as if for RISC-V so that the
// RISC-V branch of common.h is evaluated.
extern "C" const unsigned long xnn_riscv_allocation_alignment;

namespace {

// What one packing recipe needs. Values are the board's, see the comments.
struct Recipe {
  const char* name;                 // job "op" field
  uint32_t operator_type;           // enum from operator-type.h
  uint32_t fingerprint_id;          // enum from fingerprint_id.h
  const char* fingerprint_name;
  uint32_t nr;
  uint32_t log2_kr;
  uint32_t log2_sr;
  uint32_t log2_filter_element_size;
  uint32_t bias_element_size;
  size_t extra_weights_bytes;
  bool uses_bf16_packer;
  bool is_nr2;
};

struct Job {
  long idx = 0;
  std::string op;
  uint32_t flags = 0;
  size_t d0 = 0, dlast = 0;  // filter shape: dim[0] and dim[1] (2-D filter)
  std::string wfile;
  uint64_t woff = 0, wlen = 0;
  std::string bfile;  // "-" when the layer has no bias
  uint64_t boff = 0, blen = 0;
};

bool read_bytes(const std::string& path, uint64_t off, uint64_t len,
                std::vector<uint8_t>* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s: %s\n", path.c_str(), strerror(errno));
    return false;
  }
  out->assign(len + 64, 0);  // slack so the packers never read past the end
  if (fseeko(f, (off_t)off, SEEK_SET) != 0 ||
      fread(out->data(), 1, len, f) != len) {
    fprintf(stderr, "short read of %llu bytes at %llu from %s\n",
            (unsigned long long)len, (unsigned long long)off, path.c_str());
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t vlenb = 32;  // Penguin: VLEN 256 bits
  int zvfbfa = 1;
  std::string jobs_path, out_data, out_tsv;
  bool print_constants = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value after %s\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--vlenb") vlenb = (uint32_t)strtoul(next(), nullptr, 0);
    else if (a == "--zvfbfa") zvfbfa = atoi(next());
    else if (a == "--jobs") jobs_path = next();
    else if (a == "--out-data") out_data = next();
    else if (a == "--out-tsv") out_tsv = next();
    else if (a == "--print-constants") print_constants = true;
    else {
      fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  // ---- Recipes (RISC-V, XNN_ENABLE_RISCV_VECTOR) -------------------------
  // bf16 weights x bf16 input -> f32: init_bf16_f32_gemm_config(), RISC-V branch:
  //   nr = 4 * vlenb / sizeof(float); log2_kr = 0; log2_sr is never set (0);
  //   log2_filter_element_size = XNN_LOG2_SIZEOF_BFLOAT16 (1); bias is float.
  //   Packers: goi = xnn_x16_x32_packw_gemm_goi_ukernel_x4v__rvv_u4 (same bytes
  //   as xnn_pack_bf16_f32_gemm_goi_w), gio = xnn_pack_bf16_f32_gemm_gio_w.
  //   zvfbfa only swaps the gemm microkernel (5x4v__rvvzvfbfa vs 5x4v__rvv);
  //   nr, kr and the packers are the same, so --zvfbfa does not change a byte.
  // f32 weights x f32 input -> f32: init_f32_gemm_config_impl(), RISC-V branch:
  //   nr = 4 * vlenb / sizeof(float); kr = sr = 1; goi packer
  //   xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8; gio xnn_pack_f32_gemm_gio_w.
  // f32 "nr2" alternative: init_f32_gemm_nr2_config_impl(), RISC-V branch:
  //   nr = 2 * vlenb / sizeof(float); packer xnn_x32_packw_gemm_goi_ukernel_
  //   x2v__rvv_u8 (same bytes as xnn_pack_f32_gemm_goi_w). Chosen per layer by
  //   setup_variant_and_gemm_config() through xnn_use_nr2(nr, nr2, oc).
  // extra_weights_bytes = 0 for f32_variant (both ops use f32_variant).
  const Recipe bf16_f32 = {
      "bf16_f32",
      (uint32_t)xnn_operator_type_fully_connected_nc_bf16_f32,
      (uint32_t)xnn_fingerprint_id_fully_connected_nc_bf16_bf16_f32,
      "xnn_fingerprint_id_fully_connected_nc_bf16_bf16_f32",
      4 * vlenb / (uint32_t)sizeof(float), 0, 0, 1, (uint32_t)sizeof(float), 0,
      true, false};
  const Recipe f32_default = {
      "f32",
      (uint32_t)xnn_operator_type_fully_connected_nc_f32,
      (uint32_t)xnn_fingerprint_id_fully_connected_nc_f32_f32_f32,
      "xnn_fingerprint_id_fully_connected_nc_f32_f32_f32",
      4 * vlenb / (uint32_t)sizeof(float), 0, 0, 2, (uint32_t)sizeof(float), 0,
      false, false};
  const Recipe f32_nr2 = {
      "f32",
      (uint32_t)xnn_operator_type_fully_connected_nc_f32,
      (uint32_t)xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2,
      "xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2",
      2 * vlenb / (uint32_t)sizeof(float), 0, 0, 2, (uint32_t)sizeof(float), 0,
      false, true};

  const size_t alloc_align = (size_t)xnn_riscv_allocation_alignment;

  if (print_constants) {
    printf("XNN_ALLOCATION_ALIGNMENT_RISCV=%zu\n", alloc_align);
    printf("XNN_FLAG_TRANSPOSE_WEIGHTS=%u\n", (unsigned)XNN_FLAG_TRANSPOSE_WEIGHTS);
    printf("operator_type_fully_connected_nc_bf16_f32=%u\n", bf16_f32.operator_type);
    printf("operator_type_fully_connected_nc_f32=%u\n", f32_default.operator_type);
    printf("%s=%u\n", bf16_f32.fingerprint_name, bf16_f32.fingerprint_id);
    printf("%s=%u\n", f32_default.fingerprint_name, f32_default.fingerprint_id);
    printf("%s=%u\n", f32_nr2.fingerprint_name, f32_nr2.fingerprint_id);
    printf("vlenb=%u\nzvfbfa=%d\n", vlenb, zvfbfa);
    return 0;
  }
  if (jobs_path.empty() || out_data.empty() || out_tsv.empty()) {
    fprintf(stderr,
            "usage: xnn_prepack [--vlenb 32] [--zvfbfa 1] --jobs jobs.tsv "
            "--out-data packed.bin --out-tsv result.tsv | --print-constants\n");
    return 2;
  }

  // ---- Read jobs ---------------------------------------------------------
  std::vector<Job> jobs;
  {
    FILE* f = fopen(jobs_path.c_str(), "r");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", jobs_path.c_str());
      return 1;
    }
    char line[8192];
    while (fgets(line, sizeof line, f)) {
      if (line[0] == '#' || line[0] == '\n') continue;
      // idx op flags d0 dlast wfile woff wlen bfile boff blen   (tab separated)
      std::vector<std::string> t;
      std::string cur;
      for (char* p = line; *p && *p != '\n'; ++p) {
        if (*p == '\t') {
          t.push_back(cur);
          cur.clear();
        } else {
          cur.push_back(*p);
        }
      }
      t.push_back(cur);
      if (t.size() != 11) {
        fprintf(stderr, "bad job line (%zu fields): %s", t.size(), line);
        return 1;
      }
      Job j;
      j.idx = atol(t[0].c_str());
      j.op = t[1];
      j.flags = (uint32_t)strtoul(t[2].c_str(), nullptr, 0);
      j.d0 = strtoull(t[3].c_str(), nullptr, 10);
      j.dlast = strtoull(t[4].c_str(), nullptr, 10);
      j.wfile = t[5];
      j.woff = strtoull(t[6].c_str(), nullptr, 10);
      j.wlen = strtoull(t[7].c_str(), nullptr, 10);
      j.bfile = t[8];
      j.boff = strtoull(t[9].c_str(), nullptr, 10);
      j.blen = strtoull(t[10].c_str(), nullptr, 10);
      jobs.push_back(j);
    }
    fclose(f);
  }

  FILE* fd = fopen(out_data.c_str(), "wb");
  FILE* ft = fopen(out_tsv.c_str(), "w");
  if (!fd || !ft) {
    fprintf(stderr, "cannot open outputs\n");
    return 1;
  }
  fprintf(ft,
          "idx\toperator_type\tnr\tkr\tsr\textra_weights_bytes\tn_stride\tk_stride\t"
          "weights_stride\tpacked_weights_size\taligned_total_weights_size\tseed\t"
          "fingerprint_id\tfingerprint_name\tuse_nr2\tpartial_nr_block\tdata_offset\t"
          "ic\toc\ttranspose\n");

  uint64_t data_pos = 0;
  for (const Job& j : jobs) {
    // Which input/output channels: create_fully_connected_operator() in
    // src/subgraph/fully-connected.c.
    const bool transpose = (j.flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0;
    size_t input_channels, output_channels;
    if (transpose) {
      input_channels = j.d0;
      output_channels = j.dlast;
    } else {
      output_channels = j.d0;
      input_channels = j.dlast;
    }

    Recipe r;
    if (j.op == "bf16_f32") {
      r = bf16_f32;
    } else if (j.op == "f32") {
      // setup_variant_and_gemm_config(), case ..._fully_connected_nc_f32.
      // The extra conditions there (mr != 0 and a default-uarch minmax gemm at
      // index mr-1) hold for the RISC-V nr2 config: mr = 7 and
      // minmax.gemm[XNN_MR_TO_INDEX(7)] is set. flags have no slow-consistent bit.
      r = f32_default;
      if (xnn_use_nr2(f32_default.nr, f32_nr2.nr, output_channels)) r = f32_nr2;
    } else {
      fprintf(stderr, "job %ld: unknown op %s\n", j.idx, j.op.c_str());
      return 1;
    }

    const size_t nr = r.nr;
    const size_t kr = (size_t)1 << r.log2_kr;
    const size_t sr = (size_t)1 << r.log2_sr;
    // create_fully_connected_nc():
    const size_t n_stride = round_up(output_channels, nr);
    const size_t k_stride = round_up_po2(input_channels, kr * sr);
    const size_t weights_stride = (k_stride << r.log2_filter_element_size) +
                                  r.bias_element_size + r.extra_weights_bytes;
    const size_t packed_weights_size = n_stride * weights_stride;
    const size_t aligned_total_weights_size =
        round_up_po2(packed_weights_size, alloc_align);
    uint32_t seed = (uint32_t)(output_channels ^ input_channels ^ nr ^ kr ^ sr ^
                               r.extra_weights_bytes ^ (size_t)r.operator_type);
    if (transpose) seed = ~seed;

    const size_t elt = (size_t)1 << r.log2_filter_element_size;
    if (j.wlen != input_channels * output_channels * elt) {
      fprintf(stderr, "job %ld: weight bytes %llu != %zu*%zu*%zu\n", j.idx,
              (unsigned long long)j.wlen, output_channels, input_channels, elt);
      return 1;
    }
    const bool has_bias = j.bfile != "-";
    if (has_bias && j.blen != output_channels * sizeof(float)) {
      fprintf(stderr, "job %ld: bias bytes %llu != %zu*4\n", j.idx,
              (unsigned long long)j.blen, output_channels);
      return 1;
    }

    std::vector<uint8_t> w, b;
    if (!read_bytes(j.wfile, j.woff, j.wlen, &w)) return 1;
    if (has_bias && !read_bytes(j.bfile, j.boff, j.blen, &b)) return 1;

    // Zero-filled, so bytes the packer skips in a partial last nr block are 0.
    // (On the board they are whatever the allocator returned: extra_weights_bytes
    // is 0 here, so create_fully_connected_nc() does not memset.)
    std::vector<uint8_t> packed(aligned_total_weights_size + 64, 0);
    const void* bias_ptr = has_bias ? (const void*)b.data() : nullptr;
    if (!transpose) {
      if (r.uses_bf16_packer) {
        xnn_pack_bf16_f32_gemm_goi_w(1, output_channels, input_channels, nr, kr,
                                     sr, (const xnn_bfloat16*)w.data(),
                                     (const float*)bias_ptr, nullptr,
                                     packed.data(), nr * r.extra_weights_bytes,
                                     nullptr);
      } else {
        xnn_pack_f32_gemm_goi_w(1, output_channels, input_channels, nr, kr, sr,
                                (const float*)w.data(), (const float*)bias_ptr,
                                nullptr, (float*)packed.data(),
                                nr * r.extra_weights_bytes, nullptr);
      }
    } else {
      // gio: k_stride argument is output_channels (create_fully_connected_nc()).
      if (r.uses_bf16_packer) {
        xnn_pack_bf16_f32_gemm_gio_w(1, output_channels, input_channels, nr, kr,
                                     sr, output_channels,
                                     (const xnn_bfloat16*)w.data(),
                                     (const float*)bias_ptr, nullptr,
                                     packed.data(), nr * r.extra_weights_bytes,
                                     nullptr);
      } else {
        xnn_pack_f32_gemm_gio_w(1, output_channels, input_channels, nr, kr, sr,
                                output_channels, (const float*)w.data(),
                                (const float*)bias_ptr, nullptr,
                                (float*)packed.data(),
                                nr * r.extra_weights_bytes, nullptr);
      }
    }
    // The packers must stay inside packed_weights_size.
    for (size_t i = packed_weights_size; i < packed.size(); ++i) {
      if (packed[i] != 0) {
        fprintf(stderr, "job %ld: packer wrote past packed_weights_size\n", j.idx);
        return 1;
      }
    }

    // 64-byte aligned placement in the output stream.
    const uint64_t aligned_pos = (data_pos + 63) & ~(uint64_t)63;
    static const uint8_t zeros[64] = {0};
    if (aligned_pos > data_pos) fwrite(zeros, 1, aligned_pos - data_pos, fd);
    fwrite(packed.data(), 1, aligned_total_weights_size, fd);
    data_pos = aligned_pos + aligned_total_weights_size;

    fprintf(ft,
            "%ld\t%u\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%u\t%u\t%s\t%d\t%d\t%llu\t%zu\t%zu\t%d\n",
            j.idx, r.operator_type, nr, kr, sr, r.extra_weights_bytes, n_stride,
            k_stride, weights_stride, packed_weights_size,
            aligned_total_weights_size, seed, r.fingerprint_id,
            r.fingerprint_name, r.is_nr2 ? 1 : 0,
            (output_channels % nr) != 0 ? 1 : 0,
            (unsigned long long)aligned_pos, input_channels, output_channels,
            transpose ? 1 : 0);
  }
  fclose(fd);
  fclose(ft);
  return 0;
}
