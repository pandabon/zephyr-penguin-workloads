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
| `third_party/tapeout-chipyard/` | The copied Atlas team sources those apps build. Same paths as in the tapeout-chipyard repo, so a checkout can replace it. Read `NOTICE`. | - |
| `known_failures.yaml` | Scenarios expected to fail (empty). | - |

Planned, each as its own group folder with its own apps and its own opt-in dependencies (D29): `saturn/benchmarks/`
(Saturn's tests, built in place from their repo), `executorch/` (MNIST etc.). A new group never adds dependencies
to the others.

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
Expected: 38 pass; `vec-mx-unary` passes only with `zvfofp8min` added to Spike's ISA (`SPIKE_EXTRA_ARGS`); `vec-spmv`, `vec-conjugate-gradient` fail (GCC vtype issue, `known_failures.yaml`); the 14 FP8 /
bf16-reduction ones trap on Spike by design (`build_only` in `saturn/benchmarks/testcase.yaml`, run them on the FPGA). The
BSP's `scripts/run_tests.py` reads `testcase.yaml` files only inside the BSP; point it at `saturn/benchmarks/testcase.yaml`
once it has a `--root` option (proposed).

**For vcu118_ta (FPGA)**: same commands with `-b chipyard_penguin/penguin/vcu118_ta` (Spike also runs these builds);
load the printed ELF with uart_tsi.

**Good on the bench** (self-checking, Spike-verified, small): `vec-strlen vec-mx-l2norm vec-mx-dotprod vec-mx-rmsnorm
vec-sgemm-v2 vec-mx-conv1d vec-mx-fma vec-mx-masked-binary vec-conditional vec-mx-unary vec-transpose-load
vec-transpose-store vec-mixed_width_mask vec-dropout vec-softmax vec-dotprod vec-fft vec-log vec-exp vec-cos
vec-slide-conv vec-sgemm vec-sgemv vec-conv-3 vec-sgemm-v3 vec-sep-conv-3 vec-igemm` (tag `bench`, in this order = ELF size).
Not for the bench: `vec-roi-align` (1.8 MB), `vec-jacobi2d`, `vec-fconv3d` (0.5-0.6 MB each). Put the FP8 and bf16-reduction
programs on the bench only to test those instructions: they are the point of running on hardware, since Spike cannot.

## Credits and licences

* `third_party/tapeout-chipyard/` is the **Atlas/Saturn team's** work (authors: kellytou, Nicolas Reed, Eric Wen,
  Jason Tang, rylee4761, aabel14, Brandon Kim, Shashank Anand; per file in `third_party/tapeout-chipyard/NOTICE`),
  copied unmodified from the course tapeout repo (tapeout-chipyard) at 48c3a7d11f6de3f0cecb33ae6a31c24ec26ff9d8.
  `tests/` is BSD-3-Clause (Regents of the University of California; `LICENSE` there). The
  `generators/sp26-atlas-acc` part has **no licence file** upstream: fine for a private repo, to be cleared with the
  Atlas team before publication.
* `atlas/kernels/` adapts the team's tests (credits in `src/main.c`, licence text `LICENSE.tapeout-chipyard`);
  `atlas/upstream/` is our build glue around their sources. Our own files are Apache-2.0 (SPDX headers).
* History: `atlas/kernels` and `atlas/upstream` come with their full git history from
  zephyr-chipyard-bsp `dev/main` d1cff22 (`samples/atlas_kernels`, `external/atlas_upstream`).
