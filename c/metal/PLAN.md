# pac-julia on Apple GPUs (Metal)

Plan 2026-11-14, decisions 2026-11-15, implementation on branch `metal` (2026-11-15). Status:
**checked on Linux (phases 1–2); on the Mac (M4) the sources compile as MSL and the GPU's f32 and
df64 operations are the CPU shim's bits (phase 0, 2026-11-15); the programs on the GPU are the
shim's bits (phase 3), and mg7 runs on them agree with stock and are faster (phase 4,
2026-11-15); phase 5 (tuning) to do**. The Claude instance on the Mac starts from
`c/metal/CLAUDE.md`.

**Goal.** PacRunner's UIR programs on Apple-silicon GPUs through Metal, as UMAMI libraries MadSpace
loads on macOS, in one precision mode: CUDA's `mixed` split with every f64 part in df64 (a pair of
f32), since Metal has no `double`.

**Decisions (user, 2026-11-15).** Common files stay in `c/`, CUDA's are in `c/cuda/`, Metal's in
`c/metal/`. Kernels are generated in Julia (`src/kernel_msl.jl`): pac-julia stays independent of
symjit-utils (sjb2obj remains an optional test dependency). The event's end (helicity sum or draw,
alpha_s scale, factor) runs on the host first. macOS is the primary development platform: MG7 and
MadSpace build there.

## 1. Design

- **The UIR interpreter in MSL**, one GPU thread per point (`c/metal/uirvm.metal`, a port of
  `c/uirvm.c`); the program is data in one buffer (a header of byte offsets, then encode(l)'s rows,
  the edge table and the tables they index); the symjit kernels and the stub switch are generated
  MSL per program (`compile_metal`).
- **The library is a UMAMI CPU provider**: UMAMI has no Metal device and MadSpace no Metal runtime.
  MadSpace calls it with host memory from its CPU threads; with unified memory the GPU reads the
  staged momenta and writes the values in place (no copies). The host checks the inputs, runs the
  call's events on the GPU in one command buffer, and finishes each event in f64 with the existing
  `c/umami_finish.h` (`c/metal/umami_metal.c`).
- **One slot pool per process** (`libpacmetal.dylib`, `c/metal/metalpool.m`; the analogue of
  `c/cuda/pool.cu`): the device, one command queue, K slots of shared memory grown on demand. A call
  borrows a slot and carves its momenta, outputs and arena out of it, so the memory is bounded by
  the slots, not by the number of program libraries (pp → Zjj loads 53). No Objective-C classes:
  many program libraries share one process.
- **Runtime compilation**: the source is embedded in the library and compiled on first use with
  `MTLMathModeSafe` (no fast math, no contraction: df64 needs exact f32 operations); Metal caches
  the compiled library across processes.

## 2. Precision: one mode, mixed f32/df64

CUDA's `PAC_MIXED` split (`c/pacops.h`, `c/uirvm.c`, `kernel_c_mixed`) with `double` → df64:

| Quantity | CUDA mixed | Metal |
|---|---|---|
| currents (planes), vertex / propagator-numerator / scale / closure-product / copy / scatter arithmetic, vertex-run sums, the rows' factors | f32 | f32 |
| momentum forms | f64 | df64 |
| propagator denominators | f64, rounded once to f32 | df64, rounded once to f32 |
| wavefunctions (SOURCE rows), then rounded into the f32 planes | f64 | df64 |
| amplitudes, REDUCE rows, group factor | f64 | df64 |
| symjit kernels | f32, f64 on the divisors' backward slices | f32, df64 there |
| the event's end (umami_finish.h) | f64 on the device | f64 on the host |

df64 (`c/metal/df64.h`): pacuda's algorithms (Joldes, Muller, Popescu 2017: AccurateDWPlusDW,
DWTimesDW3, DWDivDW2), a Karp–Markstein square root, componentwise bit operations for the kernels'
masks. f32's exponent range and, on Apple GPUs, flush-to-zero of f32 denormals.

Target: per matrix element about 1e-7 relative to f64 (median), 1e-5 at worst; σ within 1e-6.

## 1b. Review of MG7's `aloha/template_files/madmatrix` (helas.h, helas_fd.h, helas.cu)

Nothing to import. It confirms the split: madmatrix's mixed mode keeps `fptype_momenta` =
`fptype_denom` (momenta, in-vertex denominators) wide and `fptype_amp` narrow. It has a cancellation
guard we do not (`pp + pz` for `pz < 0` as `(px² + py²) / (pp − pz)`); pacops keeps RustiCol's
`sqrt(max(E + pz, 0))` so that the f64 reference stays the comparison (a candidate improvement for
all back ends). Its branchless helicities and on-axis specializations are speed devices our
interpreter does not need; `helas_fd.h` (flavour-indexed) and `fptype2` (colour) do not apply.

## 3. Files

```
c/        pacops.{c,h} uirvm.{c,h} umami_pac.c umami_finish.h umami.h UMAMI_LICENSE   CPU + shared
c/cuda/   uirgpu.{c,h} uirgpu_pool.c uirgpu_umami.c uirgpu_provider.c pool.{cu,h}
c/metal/  df64.h          the df64 type and operations (MSL and C++)
          msl_shim.h      enough MSL for the .metal sources to compile as C++ (Linux tests)
          pacops.metal    c/pacops.c in mixed f32/df64 (factors by value; no recursion)
          uirvm.metal     c/uirvm.c, the program as one buffer, the compute function uir_points
          uirmetal.{h,m}  the host (Objective-C): compile, pipeline, a call on a pool slot
          metalpool.{h,m} the process-wide slot pool (libpacmetal.dylib)
          umami_metal.c   the UMAMI layer: c/umami_pac.c's CPU library, one GPU run per call
          PLAN.md, CLAUDE.md
src/      kernel_c.jl     the mixed kernels' renderers now take a language (:c, :msl); C output unchanged
          kernel_msl.jl   inline_funclets, kernel_msl (a kernel as an MSL function, df64)
          compile_metal.jl  compile_metal(dir, l): runner.metal, program.bin, runner_metal.{c,h}, Makefile
          umami.jl        device = :metal (build_umami_metal, build_metal_runtime)
test/     metal_shim_ops.cpp, metal_shim_run.cpp, test_metal_shim.jl (Linux and macOS: CPU)
          test_metal.jl (macOS: GPU), test_umami.jl check(...; metal = true), mg7_run.py --device metal
          metal_compile.m (macOS: Metal's diagnostics of a .metal file),
          metal_probe.mm + metal_probe_body.metal (macOS: f32 and df64 operations, GPU vs shim)
          mg7_bench.py (an mg7 output on stock, cpu and metal: σ, wall, phases, matrix-element time)
```

The `.metal` files are copies of `c/pacops.c` and `c/uirvm.c`, modified: a fix in one belongs in the
other; `test_metal_shim.jl` compares them on the same inputs.

**Kernels.** `kernel_c_mixed`'s steps and `need` rule, rendered in MSL. MSL has no goto, so funclets
are inlined (`inline_funclets`) and any other control flow is refused. Every kernel of the test data
and of the four mg7 processes is straight-line after inlining (47 kernels: 21 straight-line, 26 with
one inlined complex square root; no math-library calls).

## 4. Phases

| Phase | Where | Status |
|---|---|---|
| 0. MSL probes: the sources compile with newLibraryWithSource; df64 exact on the GPU | Mac | done 2026-11-15 (M4), see "Results on the Mac" |
| 1. df64 and pacops on the CPU (shim) | Linux | done: df64 ops ≤ 1.8e-14 relative, wavefunctions 7e-14, f32 vertices and propagators 4.4e-7 |
| 2. Interpreter and kernels on the CPU (shim) | Linux | done: 629 programs (test data + the 4 mg7 processes), see below |
| 3. On the GPU: test_metal.jl (GPU vs f64 vs shim) | Mac | done 2026-11-15: 653 programs bit for bit with the shim, see "Results on the Mac" |
| 4. UMAMI: test_umami (metal), mg7 runs vs stock and our CPU | Mac | done 2026-11-15: five processes, see "Results on the Mac" |
| 5. Speed: chunk size, slots, threads, profile | Mac | baseline measured (phase 4's table, default knobs); tuning to do |

Phase 2 on Linux (2026-11-15, `julia --project test/test_metal_shim.jl EXPORTS`): matrix element
relative to f64 at the reference points (or 16 RAMBO points):

| Export | Programs | ME error median | max |
|---|---|---|---|
| test/data pp_3j | 12 | 1.6e-7 | 6.8e-7 |
| test/data zggg_union | 1 | 2.2e-7 | 6.7e-7 |
| gg → tt̄gg | 24 | 3.0e-7 | 1.4e-6 |
| pp → Zjj | 440 | 2.2e-7 | 2.7e-5 |
| pp → jjj | 128 | 1.5e-7 | 5.9e-6 |
| uū → Z+4g | 24 | 2.6e-7 | 8.4e-7 |

The same as CUDA mixed (median ~1e-7, worst ~1e-5).

## 4b. Results on the Mac

**Apple M4 (10 GPU cores), macOS 26.6.2, Apple clang 17, 2026-11-15.**

Phase 0. The test data's 13 programs (zggg_union, pp_3j's 12) compile with newLibraryWithSource
(MTLMathModeSafe, precise functions) in about 0.2 s each (0.00 s from Metal's cache), without
warnings, once two errors in `df64.h` were fixed: `operator+=` returned a `df64 &` (MSL needs the
address space: `thread df64 &`), and `dd_rint` named a variable `half` (an MSL type). The suspects of
section 5 were all fine. The host (`make`: librunner_metal.dylib) builds without warnings.
`test/metal_probe.mm` runs f32 + − × ÷ fma sqrt and df64 + − × ÷ sqrt floor rint fma, two_prod and
two_sum on 2^20 inputs (2^±60, cancellations, values near 2^-120) on the GPU and through the shim:
**all bit for bit**. Its controls: the CPU without flush-to-zero differs on every operation (on
0.6–5 % of the inputs), the GPU with MTLMathModeFast on ÷ and every df64 operation (up to 98 %).
So `msl_shim_flush_denormals` now sets FPCR.FZ on arm64 too (it only set FTZ + DAZ on x86-64), or the
shim on a Mac would not be the GPU's bits.

The shim (test_metal_shim.jl) on the Mac, Apple clang and Homebrew g++ 16 alike (the same bits):
df64 and pacops as on Linux; pp_3j ME error median 1.6e-7, max 5.4e-6 (one point of
p_p_to_j_j_j_16 flow:3,1,4,2,5; the Linux row above says 6.8e-7: to be checked there), zggg_union
2.2e-7 / 6.7e-7 as on Linux.

Phase 3 (2026-11-15). The first run had the programs within the target but **not bit for bit with
the shim** (95 of pp_3j's 96 points, up to 1.4e-6), although every operation of the probe was. A
driver running uir_points on the GPU and through the shim with the same buffers found forms and
amplitudes equal, planes differing from the 23rd row of zggg_union (a vertex: one f32 ulp);
**MTLMathModeSafe still contracts a * b + c into fma**. With `#pragma METAL fp contract(off)` in
runner.metal's prelude (compile_metal) every buffer is equal. test_metal.jl, the GPU vs f64 and vs
the shim:

| Export | Programs | Points | ME error median | max | GPU vs shim |
|---|---|---|---|---|---|
| test/data pp_3j | 12 | 96 | 1.6e-7 | 5.4e-6 | bit for bit |
| test/data zggg_union | 1 | 4 | 2.2e-7 | 6.7e-7 | bit for bit |
| g g > g g g (mg7, this Mac) | 24 | 384 | 1.7e-7 | 1.7e-6 | bit for bit |
| g g > t t~ g g (mg7) | 24 | 384 | 4.0e-7 | 1.6e-6 | bit for bit |
| p p > j j j (mg7) | 440 | 7040 | 2.6e-7 | 7.5e-5 | bit for bit |
| u u~ > z g g g g (mg7) | 24 | 384 | 1.9e-7 | 1.8e-6 | bit for bit |
| p p > z j j (mg7) | 128 | 2048 | 2.1e-7 | 8.5e-6 | bit for bit |

and the UMAMI libraries (test_umami's check, metal = true) of the test data as expected.

The mg7 processes were generated and exported here (MadGraph7 PR #452, ~/mg7-pr452-runs/NAME, 16
RAMBO points per program). Two things to check on Linux: (1) here p p > j j j has 440 programs and
p p > z j j 128, the reverse of the Linux table's labels in section 4 (the labels there are probably
swapped); (2) p p > j j j's worst point, 7.5e-5 (program 244, mp0_mp0_to_mp0_mp0_mp0_76
flow:2,3,5,4,1, point 16), with 5.3e-5 (program 289, point 6) the only two of 7040 above 3e-5 and
four above 1e-5. The GPU is the shim's bits there, so it is the mixed split's arithmetic; CUDA mixed
on the same export and points (not runnable here: c/uirvm.c's PAC_MIXED is nvcc's) should show
whether it is the f32 currents' cancellation (expected) or something of df64.

Phase 4 (2026-11-15). `test/mg7_bench.py`: the five processes generated here (g g > g g g and
the four of section 4, MadGraph7 PR #452), 100k events, batches of 64000, seed 7, fixed scales, LC,
MadSpace on 10 threads; per mode a warm-up run, then the timed one. Wall: the whole generate_events
process without timers (survey + generate + combine within 0.2–0.9 s of it); ME: umami_timer.c in
front of every library (the stock one too), seconds summed over the threads, in a second pass
(the same runs: same σ, same points). PacRunner's CPU f64 is the stock σ to ≤ 1e-13, the same points.

| Process | Libraries | Mode | σ vs CPU f64 | Wall s | µs / point (wall) | ME µs / event (thread time) | Wall vs stock | ME vs stock |
|---|---|---|---|---|---|---|---|---|
| g g > g g g | 1 / 4 | stock | | 12.3 | 1.55 | 13.5 | 1× | 1× |
| | | cpu | | 3.5 | 0.44 | 2.39 | 3.5× | 5.6× |
| | | metal | −1.6e-7 | 2.5 | 0.31 | 1.00 | 5.0× | 13.5× |
| g g > t t~ g g | 1 / 12 | stock | | 64.5 | 5.16 | 49.0 | 1× | 1× |
| | | cpu | | 13.2 | 1.06 | 8.18 | 4.9× | 6.0× |
| | | metal | +1.3e-4 (0.12σ) | 9.2 | 0.74 | 5.06 | 7.0× | 9.7× |
| u u~ > z g g g g | 1 / 1 | stock | | 150.2 | 2.73 | 49.2 | 1× | 1× |
| | | cpu | | 58.0 | 1.05 | 15.8 | 2.6× | 3.1× |
| | | metal | −1.1e-3 (1.25σ) | 23.3 | 0.43 | 4.13 | 6.4× | 11.9× |
| p p > z j j | 1 / 156 | stock | | 46.4 | 3.09 | 25.7 | 1× | 1× |
| | | cpu | | 7.4 | 0.50 | 1.51 | 6.2× | 17.0× |
| | | metal | −5.4e-8 | 6.2 | 0.41 | 0.87 | 7.5× | 29.6× |
| p p > j j j | 1 / 53 | stock | | 206.6 | 1.18 | 15.8 | 1× | 1× |
| | | cpu | | 47.0 | 0.27 | 1.19 | 4.4× | 13.2× |
| | | metal | −4.5e-6 | 46.1 | 0.26 | 0.81 | 4.5× | 19.5× |

σ: where the Metal run samples the CPU run's points (g g > g g g, p p > z j j; p p > j j j but two
points), it is within 1e-6 (p p > j j j 4.5e-6). Elsewhere MadSpace's adaptation, fed values that
differ by ~1e-7, takes other points (u u~ > z g g g g: 54.0M against 55.1M), and σ agrees within
the statistical error (0.12σ, 1.25σ): the Metal libraries cannot be held to 1e-6 on σ there, only
to their matrix elements (phase 3). Speed: Metal is fastest everywhere, by most where the matrix
element dominates (u u~ > z g g g g: 2.5× our CPU, 6.4× stock); on p p > j j j the matrix element
is a fifth of the thread time and the rest MadSpace's, so the wall barely moves. The stock
library spends some seconds in each run's survey (p p > z j j 4.9 s, ours 0.0–0.1 s).

## 5. Risks

- **MSL details not checked here**: the sources compile as C++ on Linux, not yet as MSL. Likely
  spots: pointer casts in `uir_view`, `long` in a thread struct, the `kernel` function's
  `constant uir_launch &`, `atomic_store_explicit` on `device atomic_int *`, float literals
  (`1.5e-30f`, `INFINITY`, `as_type<float>(...)`), `static` forward declaration of `uir_stub`.
- **Exact f32 on the GPU**: df64 needs correctly rounded f32 +, −, ×, ÷, sqrt and a real fma
  (`MTLMathModeSafe`; symjit-utils measured them on the M4). test_metal.jl compares the GPU with the
  CPU shim: they should agree bit for bit; a difference points at the compile options.
- **Compile time** of large sources (minutes the first time for 40 MB; Metal caches). Ours are
  smaller (zggg_union 75 kB), but 53 libraries compile one by one at first use.
- **Exponent range** of f32 and df64 (~1e-38): not hit on our processes (shim results above).

## 6. Questions (answered 2026-11-15)

- Q1 madmatrix: reviewed (1b). Q2 folders: `c/`, `c/cuda/`, `c/metal/`. Q3 MG7 and MadSpace build on
  macOS. Q4 kernels in Julia. Q5 the event's end on the host first.
