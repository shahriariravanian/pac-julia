# CLAUDE.md: the Metal runner (c/metal/), for the Claude instance on the Mac

This file guides work on PacRunner's Metal runner on macOS (Apple silicon). It is in the repository
and shared with the Linux checkout: notes about one machine go under "Machine notes", marked as such.

## What this is

pac-julia's UIR interpreter in the Metal Shading Language, mixed f32/df64 (Metal has no double),
as UMAMI libraries MadSpace loads as CPU providers. Read `c/metal/PLAN.md` first: the design, the
precision split, the files, and what is done. Everything was written on Linux, where the `.metal`
sources compile as C++ through `msl_shim.h` and agree with f64 on 629 programs (PLAN.md section 4).
**Nothing has been compiled as MSL or run on a GPU yet**: that is this machine's job.

The pieces: `df64.h`, `pacops.metal`, `uirvm.metal` (device code, concatenated per program with its
kernels into `runner.metal` by `src/compile_metal.jl`); `uirmetal.m` (host: compile with
`MTLMathModeSafe`, launch); `metalpool.m` (process-wide slot pool, `libpacmetal.dylib`);
`umami_metal.c` (the UMAMI layer); `src/kernel_msl.jl` (symjit kernels as MSL).

## Rules

- Work on the branch `metal` (or a branch from it); commit there and push it. Never push `main`.
- `c/pacops.c` ↔ `c/metal/pacops.metal` and `c/uirvm.c` ↔ `c/metal/uirvm.metal` are kept in step:
  a fix in one belongs in the other. `src/kernel_c.jl`'s C output must not change (CUDA and the CPU
  are bit for bit RustiCol): after touching it, compare `kernel_c`/`kernel_c_mixed` output before and
  after on the test data's kernels.
- Accuracy is checked with tolerances (median ~1e-7, max ~1e-5 on a matrix element), never by
  loosening a tolerance to make a failure pass: find the cause.
- Record results (numbers, dates, the machine) in `c/metal/PLAN.md` (phases table, a "Results on the
  Mac" section); keep this file for instructions.

## Steps

0. **Setup.** `git fetch && git checkout metal`; `julia --project -e 'using Pkg; Pkg.instantiate()'`.
   Needs the Command Line Tools (`cc` with Objective-C, the Metal and Foundation frameworks); no Xcode
   (the sources compile at run time). GPU runs may need the Bash sandbox disabled.

1. **The sources on the CPU** (as on Linux, with clang): `julia --project test/test_metal_shim.jl`.
   Must pass as on Linux. If clang rejects something g++ accepted, fix the source, not the test.

2. **The sources as MSL** (phase 0). Compile one program's folder and run it:
   ```sh
   julia --project -e 'using PacRunner; l = load_recurrence_export("test/data/export/zggg_union.json")[1];
                       println(compile_metal("/tmp/zggg_metal", l))'
   make -C /tmp/zggg_metal         # librunner_metal.dylib
   ```
   The first call of `run_metal` compiles `runner.metal`; a compile error is printed with
   `uirmetal (...): the program's source does not compile: <Metal's message>`. To see all of
   Metal's diagnostics at once, write a tiny Objective-C or Swift driver around `newLibraryWithSource`
   on `/tmp/zggg_metal/runner.metal` (or install Xcode's `xcrun metal -c`). Fix the sources in
   `c/metal/` (and `src/kernel_msl.jl` for kernel code), regenerate, retry. Suspects are listed in
   PLAN.md section 5 (pointer casts in `uir_view`, `long`, `constant uir_launch &`, the atomic store,
   float literals, the `static` forward declaration of `uir_stub`, operator overloads on `df64`).
   Then check that the shim still passes (step 1).

3. **On the GPU** (phase 3): `julia --project test/test_metal.jl` (test data), then the mg7 processes'
   exports if they exist on this machine (`julia --project test/test_metal.jl EXPORT.json ...`).
   Expect: ME error median ~2e-7, max ≤ ~3e-5 (PLAN.md's Linux numbers), and **the GPU equal to the
   CPU shim bit for bit** (same correctly rounded f32 operations, flush-to-zero). If the GPU differs
   from the shim, look at the compile options first (`MTLMathModeSafe`, precise functions in
   `uirmetal.m`), then at fma contraction, then at denormals; report what differs and where.

4. **UMAMI** (phase 4): the library check `test_umami.jl` with `metal = true` runs inside
   `test/test_metal.jl`. Then mg7, with this machine's MG7/MadSpace (ask the user where they are):
   ```sh
   python3 test/mg7_run.py prepare OUTPUT --device metal     # lib/pacrunner-metal/...
   python3 test/mg7_run.py run OUTPUT --device metal -f      # run.device = ["cpu"], color_accuracy = "lc"
   ```
   Compare σ and the events with the stock run and with `--device cpu` (f64): σ within ~1e-6
   relative of our CPU build (mixed precision), events not bit for bit.

5. **Speed** (phase 5): `julia --project test/test_metal.jl --time 65536` (µs per point, warm);
   then the mg7 walls against stock and `--device cpu`, the same settings as the Linux benchmark
   (~/mg7-pr452-runs/bench_main.sh there: 64k batches, 100k events; MadSpace threads = the CPUs).
   Knobs: `PAC_METAL_CHUNK` (points per launch, default 4096: try 2048, 8192, 16384),
   `PAC_METAL_SLOTS` (default the CPUs), `PAC_METAL_SLOT_BYTES` (default 256 MiB). symjit-utils'
   Metal probe found 4096 best and the CPU-side staging a visible share for short kernels.

## Things that bite (from symjit-utils' Metal work on an M4, 2026-11-01)

- Metal compiles sources at run time and caches them: the first run of a big program is slow (a
  41 MB source took 22 s + 100 s of pipeline builds), later runs fast. Time warm runs only.
- f32 denormals are flushed to zero on the GPU (the shim does the same).
- f32 +, −, ×, ÷, fma, sqrt were the CPU's bits under `MTLMathModeSafe`; df64 depends on that.
- `MTLMathModeSafe` does **not** stop fma contraction (found on the M4, 2026-11-15): runner.metal's
  prelude has `#pragma METAL fp contract(off)`, without which the GPU differs from the shim by an
  ulp here and there. To find such a difference: run uir_points on the GPU and through the shim with
  the same buffers, compare forms, amplitudes and planes, and bisect on the number of rows.
- GPU runs from the Bash sandbox may need it disabled.
- Timing mg7 runs: `test/mg7_bench.py OUTPUT` (stock, cpu, metal; a warm-up run each). The timer
  shims (PAC_UMAMI_TIMER) are built per library at the run's start: with 156 libraries that adds
  ~40 s to the process's wall, so take wall times from `--no-timer` runs and the matrix element's
  from timed ones. A Metal run may sample other points than the CPU one (MadSpace adapts on values
  ~1e-7 apart): compare σ within errors there, not to 1e-6.

## Machine notes

**Apple M4 (10 cores, 16 GB), macOS 26.6.2, Command Line Tools (Apple clang 17), Julia 1.13.4** (2026-11-15):
- MadGraph7 is beside this repository (`../MadGraph7`), on a local branch `pr452` (MadGraphTeam/MadGraph7
  PR #452, the recurrence export). MadSpace and pyAmpliCol are built from source in it
  (`python3 madspace/install.py --source -y --no-cuda --no-hip --no-simd`; `python3 pyamplicol/install.py
  --source -y`). pyAmpliCol reports 0.2.1: MadGraph7's version label on upstream pyAmpliCol 1.0.3's code.
- The python is `~/rust/.venv` (3.13); cmake and ninja come from pip there (no Homebrew cmake), and
  juliacall (for `test/mg7_example.py`) too.
- clang's `-Wall` flags the generated C's self-moves and unused kernel helpers, which gcc does not:
  `test/compile_common.jl` silences those two warnings.
- Homebrew's `g++-16` needs `SDKROOT=$(xcrun --show-sdk-path)`; with it the shim gives the same bits
  as Apple clang.
- `test/runtests.jl` runs the Metal GPU testset on macOS (it fails until phase 0 is done).
