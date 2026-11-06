# zephyr-penguin-workloads

Programs that run on the **Penguin** SoC (Shuttle RISC-V core + Saturn vector unit + Atlas accelerator): tests,
benchmarks and workloads, as Zephyr applications. The board port itself (SoC, board, runner, test runner) and the samples that need nothing but the board live
in the board repo, [zephyr-penguin](https://github.com/pandabon/zephyr-chipyard-bsp) (still at its old URL and
folder name, `zephyr-chipyard-bsp`); this repo holds the workloads with heavier dependencies. The local folder is
still called `penguin-workloads` until it is renamed.

**Private** for now. Part of it is the Atlas team's code, which must be cleared with them before this repo is
published (see Credits).

## What is in it

| Folder | What | Needs |
|---|---|---|
| `atlas/upstream/` | The Atlas/Saturn team's 22 host tests (`tests/saturn_atlas_*.c`, `atlas_alu_basic.c`) built from source, one test per build (`-DATLAS_TEST=<name>`). 13 build as they are; 9 need `torch` to generate golden data. | nothing (torch for 9) |
| `atlas/isa/` | The Atlas team's 121 ISA-level baremetal tests (`baremetal/assembly/*.S`), assembled and wrapped in the team's own C harness at build time, one test per build (`-DATLAS_ISA_TEST=<name>`). | python venv (numpy, numba, torch) for the golden data |
| `basics/` | `hello_world`: the smallest bring-up check, single hart. (`timer` and `mt_rvv` moved to the board repo's `samples/`.) | nothing |
| `saturn/benchmarks/` | 55 of Saturn's vector benchmarks as one Zephyr app (`-DSATURN_BENCH=<name>`). | nothing |
| `executorch/mnist/` | A small ExecuTorch model (XNNPACK RVV kernels) on the SoC. **Opt-in.** | ExecuTorch + XNNPACK checkouts, `flatc`, python |
| `third_party/tapeout-chipyard/` | The copied Atlas team sources those apps build. Same paths as in the tapeout-chipyard repo, so a checkout can replace it. Read `NOTICE`. | - |
| `known_failures.yaml` | Scenarios expected to fail (none today). The RVV context-switch reproducer `ctx_switch_min` and its entry moved to the board repo (`tests/arch/rvv/ctx_switch_min`). | - |

Each group is a folder with its own apps (`<group>/<app>/`) and its own opt-in dependencies; a new group never adds
dependencies to the others.

Every app has a `sample.yaml` with scenarios for the two boards: `chipyard_penguin/penguin` (Spike-style board) and
`chipyard_penguin/penguin/vcu118_ta` (FPGA, HTIF console).

## Get started

Inside the maintainers' workspace (`~/toolsC/bringup`, which already has the board repo and Zephyr) nothing is
needed beyond `source ../env.sh`. Anywhere else:

```sh
mkdir ws && cd ws
git clone <this repo> zephyr-penguin-workloads
west init -l zephyr-penguin-workloads && west update   # fetches the board repo and the Zephyr fork it pins
cd zephyr-penguin-workloads
```

Builds use plain `west build`; the suite runner and the machine check come from the board repo. Find it once per
shell (west project `penguin` in a workspace made from this repo, the manifest repo in the maintainers' workspace):

```sh
ZEPHYR_PENGUIN=$(west list -f '{abspath}' penguin 2>/dev/null || west list -f '{abspath}' manifest)
$ZEPHYR_PENGUIN/scripts/doctor.sh       # says what is missing
```

You also need the Zephyr SDK 1.0.0-beta1, Python >= 3.10 with west, cmake < 4 and ninja. `west.yml` pins a board
repo commit that exists only locally until that repo's reorg is published.

## Build and get ELFs

```sh
west build -p auto -b chipyard_penguin/penguin -d ../build/hello_world basics/hello_world   # ELF: ../build/hello_world/zephyr/zephyr.elf
west build -d ../build/hello_world -t run          # run it on Spike; the exit code is the program's
west build -p auto -b chipyard_penguin/penguin -d ../build/pause_resume atlas/upstream -- -DATLAS_TEST=saturn_atlas_pause_resume
west build -p auto -b chipyard_penguin/penguin/vcu118_ta -d ../build/hello_ta basics/hello_world   # the FPGA variant
```

Keep build directories outside the repo (`-d`). The ELF is a normal Zephyr ELF (`zephyr.elf`, also `zephyr.bin`); hand it to whatever loads the board (for the
FPGA: the TSI loader, e.g. `uart_tsi` from `penguin-bringup-tools`).

**For the bench**: build every app for `vcu118_ta` and collect the ELFs (the build dir of each scenario holds one):

```sh
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . --board chipyard_penguin/penguin/vcu118_ta --build-only   # every buildable scenario of that board
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
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . -f atlas_isa --board chipyard_penguin/penguin/vcu118_ta -j 12   # all 121 (about 7 min)
```

`.venv/bin/python` is picked automatically (override: `-DATLAS_PYTHON=`). The model/bench notes under "Run on
Spike" apply too. Copy and licence details: `third_party/tapeout-chipyard/NOTICE`.

## Run on Spike

Plain Spike has no Atlas, so these apps are **build-only** by default (they would trap on the first Atlas register
access). The Atlas Spike model (`atlas-spike-model`, a Spike plugin with the Atlas registers, IMEM, VMEM and a
scalar ISA, but no tensor/DMA operations) can run them. Its Atlas devices are given to Spike as extra arguments
(`SPIKE_EXTRA_ARGS`, honoured by the board repo's run target):

```sh
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . --build-only   # build everything first (ELFs under <ws>/build/test-runner)
scripts/run_with_atlas_model.sh <model>/libatlas_spike.so <ws>/build/test-runner
# one app by hand: SPIKE_EXTRA_ARGS="--extlib=<model>/libatlas_spike.so --device=atlas,imem,0x20000 \
#   --device=atlas,csr,0x40000 --device=atlas,vmem,0x20000000" west build -d <builddir> -t run
```

The script prints PASS / UNIMPLEMENTED (the model stops at a tensor or DMA operation: expected) / ILLEGAL (the
model rejected an instruction, a model gap to check) / HAZARD (a timing-race test that reads IMEM while Atlas runs,
which would hang real hardware) / FAIL / TIMEOUT. Status 2026-10-16 with the model at `atlas-spike-model`: of the
13 buildable apps, 4 PASS (`atlas_alu_basic`, `vmem_smoke`, `buffered_optA_end_jal`, `buffered_optB_restart`),
6 UNIMPLEMENTED (all DMA kernels), 2 ILLEGAL (`pause_resume`, `pi0_slice`),
1 HAZARD (`buffered_lower_first`). None of that is a hardware verdict; the bench is.

## Run the whole suite

```sh
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . -j 16                                      # every scenario of the default board, on Spike
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . --board chipyard_penguin/penguin/vcu118_ta  # the other board
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . -f saturn_atlas_pause                       # filter by name or tag
$ZEPHYR_PENGUIN/scripts/run_tests.py --root . --list                                      # list the scenarios
```

`--root .` makes the board repo's runner read this repo's `sample.yaml`/`testcase.yaml` files and its
`known_failures.yaml`. Builds and logs go to `<west topdir>/build/test-runner` (`--build-root` to change). Scenarios of
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
Expected: 41 pass (`vec-mx-unary` too, since the board repo's Spike ISA includes `zvfofp8min`, zephyr-chipyard-bsp 80ae1e9); `vec-spmv` and `vec-conjugate-gradient` pass since the Ara helper sets vtype inside its inline asm (they failed under GCC 14.3 before); the 14 FP8 /
bf16-reduction ones trap on Spike by design (`build_only` in `saturn/benchmarks/testcase.yaml`, run them on the FPGA). The
suite runner covers them too: `$ZEPHYR_PENGUIN/scripts/run_tests.py --root . -f saturn`.

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
| the board repo (zephyr-penguin) and its SDK environment | your west workspace | zephyr-chipyard-bsp `dev/main` 10fc321, SDK 1.0.0-beta1 |

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
Twister/`run_tests.py` scenarios (`workload.executorch.*`, tag `executorch`) are build-only and need the variables above.

**In the maintainers' workspace** (`~/toolsC/bringup`) the verified build used:
```sh
export EXECUTORCH_DIR=~/toolsC/archive/zephyr-chipyard-sw/third-party/executorch   # read-only; has local changes, don't modify
export XNNPACK_DIR=~/toolsC/archive/XNNPACK
export FLATC_EXECUTABLE=~/miniforge3/envs/zephyr/bin/flatc PYTHON_EXECUTABLE=~/miniforge3/envs/zephyr/bin/python   # old env, read-only
```

**Running another model.**
- **Same runner:** `-DEXEC_MODEL=/abs/path/model.pte` builds the existing runner with any `.pte`. This works as long as
  every op in the model is either XNNPACK-delegated or in the runner's portable-kernel set.
- **A model with its own export script, ops or memory needs** gets its own app next to this one,
  `executorch/<name>/`. Copy `executorch/mnist/`'s structure (CMakeLists, prj.conf, src/, scripts/, models/ with a
  `SOURCES.txt`) and keep it a separate app, with its dependencies opt-in.
- **Memory:** the heap, the method pool and the planned-memory arena are set in `prj.conf` and `src/`. Everything must
  fit in the board's DRAM node (256 MiB in the board repo's devicetree today; a larger model needs a change there).
- **Load time:** loading takes about `memsz*1.02/11520` s at 115200 baud; `.fast.elf` copies skip `.bss`.

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
* `atlas/upstream/` is our build glue around their sources. Our own files are Apache-2.0 (SPDX headers).
* `executorch/mnist/` is derived from ExecuTorch's `examples/arm/executor_runner` (Meta Platforms, Inc. and Arm Limited,
  BSD-style licence of the ExecuTorch repo), as ported to RISC-V by Jonathan Lin in pandabon/zephyr-chipyard-sw, branch
  `exeggutorch` @ e12bfa38034 (`samples/exeggutorch/executor_runner`, `samples/exeggutorch/model/gen_pte.py`). Per-file
  credits are in the file headers; the models' provenance is in `executorch/mnist/models/SOURCES.txt`.
* `basics/hello_world` is Dima Nikiforov's hello_world sample from ucb-bar/zephyr-chipyard-sw, branch `dev` @
  838a4fed2be, derived from Zephyr's `samples/hello_world` (Wind River Systems, Apache-2.0). (`timer` and `mt_rvv`, from
  the same repo, moved to the board repo with their history and credits.) That repo has no LICENSE file (only the SPDX lines in the
  files): fine for a private repo, to be cleared before publication. Our changes (single hart, exit code, scenarios)
  are in the second commit of `basics/`.
* History: `atlas/upstream` comes with its full git history from zephyr-chipyard-bsp `dev/main` 10fc321
  (`external/atlas_upstream`). `atlas/kernels` (our 7-in-1 compaction of the Atlas tests, ex `samples/atlas_kernels`) was
  removed in favour of the faithful `atlas/upstream`; it stays in git history.
