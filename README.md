# penguin-workloads

Programs that run on the **Penguin** SoC (Shuttle RISC-V core + Saturn vector unit + Atlas accelerator): tests,
benchmarks and workloads, as Zephyr applications. The board port itself (SoC, board, drivers, runner) lives in the
[Chipyard Zephyr BSP](https://github.com/pandabon/zephyr-chipyard-bsp); this repo only holds what runs on it.

**Private** for now. Part of it is the Atlas team's code, which must be cleared with them before this repo is
published (see Credits).

## What is in it

| Folder | What | Needs |
|---|---|---|
| `atlas/kernels/` | The Atlas team's first bring-up tests, ported to one Zephyr app (IMEM programs as arrays, Zephyr I/O, timeouts). | nothing |
| `atlas/upstream/` | The Atlas/Saturn team's 22 host tests (`tests/saturn_atlas_*.c`, `atlas_alu_basic.c`) built from source, one test per build (`-DATLAS_TEST=<name>`). 13 build as they are; 9 need `torch` to generate golden data. | nothing (torch for 9) |
| `atlas/isa/` | The Atlas team's 121 ISA-level baremetal tests (`baremetal/assembly/*.S`), assembled and wrapped in the team's own C harness at build time, one test per build (`-DATLAS_ISA_TEST=<name>`). | python venv (numpy, numba, torch) for the golden data |
| `basics/` | `hello_world`, `timer`, `mt_rvv` (4 threads, each keeps its vector register across a yield): small bring-up checks, single hart. | nothing |
| `saturn/benchmarks/` | 55 of Saturn's vector benchmarks as one Zephyr app (`-DSATURN_BENCH=<name>`). | nothing |
| `executorch/mnist/` | A small ExecuTorch model (XNNPACK RVV kernels) on the SoC. **Opt-in.** | ExecuTorch + XNNPACK checkouts, `flatc`, python |
| `third_party/tapeout-chipyard/` | The copied Atlas team sources those apps build. Same paths as in the tapeout-chipyard repo, so a checkout can replace it. Read `NOTICE`. | - |
| `tests/rvv/` | Tests of the vector unit and the kernel's V state: `vlen` (VLEN/VLENB), `bf16` (Zvfbfmin/Zvfbfa), `ctx_switch` (V state across context switches; the stock fork fails 3 of 6 cases, expected). Moved from the BSP with their history (D35). | nothing |
| `tests/linker/memregion/` | A linker-region check (an app section placed in a board memory region). Moved from the BSP with its history. | nothing |
| `known_failures.yaml` | Scenarios expected to fail: `arch.riscv.rvv.ctx_switch` and `.lazy` (kernel bugs, see the file). | - |

Each group is a folder with its own apps (`<group>/<app>/`) and its own opt-in dependencies; a new group never adds
dependencies to the others.

Every app has a `sample.yaml` with scenarios for the two boards: `chipyard_penguin/penguin` (Spike-style board) and
`chipyard_penguin/penguin/vcu118_ta` (FPGA, HTIF console).

## Get started

Inside the maintainers' workspace (`~/toolsC/bringup`, which already has the BSP and Zephyr) nothing is needed
beyond `source ../env.sh`. Anywhere else:

```sh
mkdir ws && cd ws
git clone <this repo> penguin-workloads
west init -l penguin-workloads && west update      # fetches the BSP and the Zephyr fork it pins
cd penguin-workloads && make -s help
```

You also need the Zephyr SDK 1.0.0-beta1, Python >= 3.10 with west, cmake < 4 and ninja (`make doctor` from the BSP
tells you what is missing). `west.yml` pins a BSP commit that exists only locally until the BSP restructure is
pushed.

## Build and get ELFs

```sh
make build APP=atlas/kernels                       # ELF: <workspace>/build/kernels/zephyr/zephyr.elf
make build APP=atlas/upstream CMAKE_ARGS=-DATLAS_TEST=saturn_atlas_pause_resume
make build APP=atlas/kernels BOARD=chipyard_penguin/penguin/vcu118_ta     # the FPGA variant
```

(Plain `west build -b <board> atlas/kernels -d build/x` works too; the make targets print the command they run.)
The ELF is a normal Zephyr ELF (`zephyr.elf`, also `zephyr.bin`); hand it to whatever loads the board (for the
FPGA: the TSI loader, e.g. `uart_tsi` from `penguin-bringup-tools`).

**For the bench**: build every app for `vcu118_ta` and collect the ELFs (the build dir of each scenario holds one):

```sh
make suite BOARD=chipyard_penguin/penguin/vcu118_ta ARGS=--build-only   # builds the 14 buildable scenarios of that board
# ELFs: <workspace>/build/test-runner/<scenario>/zephyr/zephyr.elf
```

## Atlas ISA-level tests (`atlas/isa/`)

All 121 tests of `baremetal/assembly/*.S` (sp26-atlas-acc 2ae0bef209df, the tapeout pin) are built as Zephyr
apps. Nothing generated is committed: `scripts/gen_atlas_isa_tests.py` imports the team's `assembler.py`,
assembles the `.S` into Atlas instruction words, runs `generators/gen_<test>.py` for the golden DRAM data (as
their `baremetal/Makefile` does) and writes the team's own C harness (`emit_c_file()`: write IMEM, read it back,
start Atlas, poll `CSR_DBG0`, check DRAM, `*** PASSED ***`/`*** FAILED ***`) into the build directory. A small
shim adds `ATLASISA <test> PASS|FAIL|TIMEOUT`, the exit code and a watchdog.

```sh
uv venv .venv --python 3.12                       # opt-in, gitignored; the generators import torch/numpy/numba
uv pip install --python .venv/bin/python -r requirements-atlas.txt     # CPU torch, about 1 GB in total
west build -b chipyard_penguin/penguin/vcu118_ta atlas/isa -- -DATLAS_ISA_TEST=mxu0_push_pop
make suite ARGS='-f atlas_isa' BOARD=chipyard_penguin/penguin/vcu118_ta     # all 121 (about 7 min with -j 12)
```

`.venv/bin/python` is picked automatically (override: `-DATLAS_PYTHON=`). The model/bench notes under "Run on
Spike" apply too. Copy and licence details: `third_party/tapeout-chipyard/NOTICE`.

## Run on Spike

Plain Spike has no Atlas, so these apps are **build-only** by default (they would trap on the first Atlas register
access). The Atlas Spike model (`atlas-spike-model`, a Spike plugin with the Atlas registers, IMEM, VMEM and a
scalar ISA, but no tensor/DMA operations) can run them. Its Atlas devices are given to Spike as extra arguments
(`SPIKE_EXTRA_ARGS`, honoured by the BSP's run target):

```sh
make suite ARGS=--build-only                  # build everything first (ELFs under <ws>/build/test-runner)
scripts/run_with_atlas_model.sh <model>/libatlas_spike.so <ws>/build/test-runner
# one app by hand: SPIKE_EXTRA_ARGS="--extlib=<model>/libatlas_spike.so --device=atlas,imem,0x20000 \
#   --device=atlas,csr,0x40000 --device=atlas,vmem,0x20000000" west build -d <builddir> -t run
```

The script prints PASS / UNIMPLEMENTED (the model stops at a tensor or DMA operation: expected) / ILLEGAL (the
model rejected an instruction, a model gap to check) / HAZARD (a timing-race test that reads IMEM while Atlas runs,
which would hang real hardware) / FAIL / TIMEOUT. Status 2026-10-16 with the model at `atlas-spike-model`: of the
14 buildable apps, 4 PASS (`atlas_alu_basic`, `vmem_smoke`, `buffered_optA_end_jal`, `buffered_optB_restart`),
6 UNIMPLEMENTED (all DMA kernels), 3 ILLEGAL (`pause_resume`, `pi0_slice`, and `atlas/kernels`, whose summary is
5 of 7 pass), 1 HAZARD (`buffered_lower_first`). None of that is a hardware verdict; the bench is.

## Run the whole suite

```sh
make suite                                                  # build all scenarios of the default board
make suite BOARD=chipyard_penguin/penguin/vcu118_ta         # the other board
make suite ARGS='-f saturn_atlas_pause'                     # filter by name or tag
```

`make suite` is the BSP's `scripts/run_tests.py --root .`; it reads `known_failures.yaml` from here. Scenarios of
the other board SKIP; scenarios with `skip: true` (the torch ones) SKIP with a reason.

## The 9 tests that need torch

`saturn_atlas_{smolvla_compose,smolvla_compose_large,tri_phase,tri_phase_large,fp8_quantize_bridge,
fp8_dequant_bridge,quad_phase_large,smolvla_compose_native_bf16,tri_phase_native_bf16}` compare against golden data
produced by the team's bit-exact Python models (`import torch`). No pre-generated headers exist in the source repo,
and our environment has no torch, so they are marked `skip: true`. In an environment with `torch` and `numpy`:
`west build ... atlas/upstream -- -DATLAS_TEST=saturn_atlas_tri_phase` generates the header at build time.

## Using a newer tapeout-chipyard

`-DTAPEOUT_CHIPYARD=<checkout>` (or the environment variable) makes `atlas/upstream` build from that checkout
instead of `third_party/`. Nothing in the checkout is written to.

## Saturn vector benchmarks (`saturn/benchmarks/`)

55 of Saturn's vector benchmarks (copied from `ucb-ee194-tapeout/saturn-vectors@de04946`, credits and licences
in `third_party/saturn-vectors/CREDITS.txt` and `NOTICE`), as one Zephyr app: the benchmark is picked with `-DSATURN_BENCH`.

**One benchmark on Spike**
```
west build -p always -b chipyard_penguin/penguin saturn/benchmarks -- -DSATURN_BENCH=vec-sgemm
west build -t run            # exit code 0 = pass; a "stats: mcycle=... minstret=..." line when the benchmark prints it
```
**The whole set on Spike** (build + run every benchmark; ~45 s of build each, runs take about a second):
```
for b in $(ls third_party/saturn-vectors/benchmarks | grep -E '^vec-|_test'); do
  west build -p always -b chipyard_penguin/penguin -d build/saturn/$b saturn/benchmarks -- -DSATURN_BENCH=$b >/dev/null &&
  west build -d build/saturn/$b -t run >build/saturn/$b.log 2>&1; echo "$b rc=$?"
done
```
Expected: 41 pass (`vec-mx-unary` too, since the BSP's Spike ISA includes `zvfofp8min`, BSP cfdcd0f); `vec-spmv` and `vec-conjugate-gradient` pass since the Ara helper sets vtype inside its inline asm (they failed under GCC 14.3 before); the 14 FP8 /
bf16-reduction ones trap on Spike by design (`build_only` in `saturn/benchmarks/testcase.yaml`, run them on the FPGA). The
BSP's `scripts/run_tests.py` reads `testcase.yaml` files only inside the BSP; point it at `saturn/benchmarks/testcase.yaml`
once it has a `--root` option (proposed).

**For vcu118_ta (FPGA)**: same commands with `-b chipyard_penguin/penguin/vcu118_ta` (Spike also runs these builds);
load the printed ELF with uart_tsi.

**Good on the bench** (self-checking, Spike-verified, small): `vec-strlen vec-mx-l2norm vec-mx-dotprod vec-mx-rmsnorm
vec-sgemm-v2 vec-mx-conv1d vec-mx-fma vec-mx-masked-binary vec-conditional vec-mx-unary vec-transpose-load
vec-transpose-store vec-mixed_width_mask vec-dropout vec-softmax vec-dotprod vec-fft vec-log vec-exp vec-cos
vec-slide-conv vec-sgemm vec-sgemv vec-conv-3 vec-sgemm-v3 vec-sep-conv-3 vec-igemm vec-spmv` (tag `bench`, in this order = ELF size).
Not for the bench: `vec-roi-align` (1.8 MB), `vec-jacobi2d`, `vec-fconv3d` (0.5-0.6 MB each). Put the FP8 and bf16-reduction
programs on the bench only to test those instructions: they are the point of running on hardware, since Spike cannot.

## ExecuTorch (`executorch/mnist/`, opt-in)

A small ExecuTorch model (XNNPACK delegate, RISC-V Vector micro-kernels) runs on the SoC under Zephyr. Console is HTIF on
`chipyard_penguin/penguin` and the UART on `chipyard_penguin/penguin/vcu118_ta`. ExecuTorch and XNNPACK are **not** part
of this repo (no copy, no submodule): you provide them, nothing is downloaded and nothing in them is modified. Without
them CMake stops at configure with a message listing what is missing, and the rest of the repo is unaffected.

| What | Variable | Tested with |
|---|---|---|
| ExecuTorch checkout, directory **named `executorch`** | `EXECUTORCH_DIR` | pandabon/executorch branch `zephyr-1.0.1-bf16` @ `c19e05968e8d` (ExecuTorch 1.0.1 + bf16/RVV changes) |
| XNNPACK checkout | `XNNPACK_DIR` (optional if `EXECUTORCH_DIR/backends/xnnpack/third-party/XNNPACK` works) | pandabon/XNNPACK branch `torch-1.0.0-bump-bf16` @ `c0d34218458` (RVV, Zvfbfmin, Zvfbfa kernels) |
| `flatc` and `python3` with `torchgen`, `pyyaml` | on `PATH`, or `FLATC_EXECUTABLE`, `PYTHON_EXECUTABLE` | an environment with flatc, torch 2.9.1, executorch 1.0.1 |
| the BSP and its SDK environment | your BSP workspace (`west`) | zephyr-chipyard-bsp `dev/main` d1cff22, SDK 1.0.0-beta1 |

Only `executorch/mnist/scripts/gen_model.py` needs the `executorch`/`torch` python packages (to re-export a model);
building does not. The CMake builds ExecuTorch through a symlink mirror in the build directory, so flatcc's build outputs
land in the build directory and `EXECUTORCH_DIR` stays untouched.

```sh
export EXECUTORCH_DIR=/path/to/executorch XNNPACK_DIR=/path/to/XNNPACK
export PATH=/path/to/env/bin:$PATH             # flatc, python3 (or PYTHON_EXECUTABLE / FLATC_EXECUTABLE)
executorch/mnist/scripts/build.sh chipyard_penguin/penguin           mnist_mlp_fp32 build/exec_fp32_chip   # about 4 min at -j16
executorch/mnist/scripts/build.sh chipyard_penguin/penguin           mnist_mlp_bf16 build/exec_bf16_chip
executorch/mnist/scripts/build.sh chipyard_penguin/penguin/vcu118_ta mnist_mlp_fp32 build/exec_fp32_ta
west build -d build/exec_fp32_chip -t run          # Spike, ISA derived from the board (includes zvfbfa)
```
`build.sh` prints the `west build` line it runs (`-DEXEC_MODEL=<name or path to .pte>`). Models in
`executorch/mnist/models/`: `mnist_mlp_fp32`, `mnist_mlp_bf16`, `lenet_bf16_user` (provenance in `models/SOURCES.txt`).
Re-export: `python executorch/mnist/scripts/gen_model.py --precision bf16 --pte executorch/mnist/models/mnist_mlp_bf16.pte`.
Twister/`make suite` scenarios (`workload.executorch.*`, tag `executorch`) are build-only and need the variables above.

Expected output (fixed all-ones input; the values, not their meaning, are the check):
```
I [executorch:...] Model PTE file loaded. Size: 106792 bytes.
I [executorch:...] Model executed successfully.
Output[0][0]: 0.000000
Output[0][1]: 493.999390
Output[0][2]: -3.000000
Output[0][3]: -3.000000
Output[0][4]: 3592.199463
Output[0][5]: -3.000000
Output[0][6]: -3.000000
Output[0][7]: 1047.199829
Output[0][8]: -3.000000
Output[0][9]: -3.000000
I [executorch:...] Program complete, exiting.
```
then a clean HTIF exit with code 0 (FAIL = exit code 1). `mnist_mlp_bf16` prints `496.000000`, `3600.000000`,
`1048.000000` at indices 1, 4, 7 (bf16 rounding). `lenet_bf16_user` (random weights) prints
`0.014099 0.032959 0.090332 0.062500 0.027710 0.014648 -0.107422 0.115723 -0.099121 0.008972`.
Chip and TA builds give identical numbers on Spike. The bf16 models use Zvfbfmin/Zvfbfa instructions: a core without
them traps (illegal instruction) in `xnn_bf16_*_rvv_zvfbfmin_zvfbfa`; set `XNNPACK_ENABLE_RISCV_ZVFBFMIN/ZVFBFA` OFF then.

**On the bench.** Load the ELF with uart_tsi (chip: 921600 baud, TA VCU118: 115200 baud). **Load time is set by `memsz`,
not the file size**: the loader also sends the zero-filled `.bss`/noinit over the link. Strip debug info only
(`riscv64-zephyr-elf-strip --strip-debug`; never a plain `strip`, the host needs `tohost` from the symbol table).

| ELF | memsz | filesz | 921600 baud (chip) full / `.fast` | 115200 baud (TA) full / `.fast` |
|---|---|---|---|---|
| mnist fp32 | 14.39 MB | 12.17 MB | 159 s / 135 s | 21.2 min / 17.9 min |
| mnist bf16 | 14.34 MB | 12.12 MB | 159 s / 134 s | 21.1 min / 17.9 min |
| lenet bf16 | 14.38 MB | 12.16 MB | 159 s / 135 s | 21.2 min / 17.9 min |

`seconds = memsz * 1.0195 / (baud / 10)` (TSI 1024-byte chunks, 20-byte header each, 8N1); estimates, not measured on
hardware. A `.fast.elf` is the same ELF with `p_memsz = p_filesz` (`fastload_elf.py` from penguin-bringup-tools), so the
zeros are not sent (Zephyr clears its own `.bss`); opt-in until validated on the bench. About 12 MB of the image is
XNNPACK micro-kernels (the "prod" set links all of them); trimming that is the way to a shorter load.

Notes: the runner's inputs come from a static pool, so the stock `BufferCleanup` would `free()` them when `main()`
returns (fault in `set_chunk_used`); the runner passes an empty span instead. `CONFIG_SMP=y` with
`MP_MAX_NUM_CPUS=1` and POSIX threads are ExecuTorch's threadpool requirements, not a second hart. XNNPACK
`c0d34218458` was only in the author's local clone when this was written (not yet on the `pandabon/XNNPACK` remote).

## Credits and licences

* `third_party/tapeout-chipyard/` is the **Atlas/Saturn team's** work (authors: kellytou, Nicolas Reed, Eric Wen,
  Jason Tang, rylee4761, aabel14, Brandon Kim, Shashank Anand; per file in `third_party/tapeout-chipyard/NOTICE`),
  copied unmodified from the course tapeout repo (tapeout-chipyard) at 48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8.
  `tests/` is BSD-3-Clause (Regents of the University of California; `LICENSE` there). The
  `generators/sp26-atlas-acc` part has **no licence file** upstream: fine for a private repo, to be cleared with the
  Atlas team before publication.
* `atlas/kernels/` adapts the team's tests (credits in `src/main.c`, licence text `LICENSE.tapeout-chipyard`);
  `atlas/upstream/` is our build glue around their sources. Our own files are Apache-2.0 (SPDX headers).
* `executorch/mnist/` is derived from ExecuTorch's `examples/arm/executor_runner` (Meta Platforms, Inc. and Arm Limited,
  BSD-style licence of the ExecuTorch repo), as ported to RISC-V by Jonathan Lin in pandabon/zephyr-chipyard-sw, branch
  `exeggutorch` @ e12bfa38034 (`samples/exeggutorch/executor_runner`, `samples/exeggutorch/model/gen_pte.py`). Per-file
  credits are in the file headers; the models' provenance is in `executorch/mnist/models/SOURCES.txt`.
* `basics/{hello_world,timer,mt_rvv}` are Dima Nikiforov's samples (hello_world, timer, test_mt_rvv) from
  ucb-bar/zephyr-chipyard-sw, branch `dev` @ 838a4fed2be; `hello_world` and `mt_rvv` derive from Zephyr's
  `samples/hello_world` (Wind River Systems, Apache-2.0). That repo has no LICENSE file (only the SPDX lines in the
  files): fine for a private repo, to be cleared before publication. Our changes (single hart, exit code, scenarios)
  are in the second commit of `basics/`.
* History: `atlas/kernels` and `atlas/upstream` come with their full git history from
  zephyr-chipyard-bsp `dev/main` d1cff22 (`samples/atlas_kernels`, `external/atlas_upstream`).
