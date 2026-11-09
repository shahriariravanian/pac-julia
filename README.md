# PacRunner

pyAmpliCol's recurrence programs (pac2json folders: `main.json`, `examples.json`, the symjit
kernels) run in Julia, and compiled to C. The C runner is the base for the GPU runner and the
reference it is tested against: its values are bit for bit the Julia runner's, which are RustiCol's.

```julia
using PacRunner
p = load_artifact("lc", 1)                     # program 1 of a pyAmpliCol LC artifact (or a pac2json folder, or its main.json)
ps = load_programs("lc")                       # all of them
a = Arena(p)
outputs = evaluate(p, a, momenta)              # momenta[leg] = (E, px, py, pz), the caller's legs
compile("out", p; examples = load_examples("lc", 1))   # the folder out/, below (from lower(p))
app = read_sjb("kernel.plane.symjit"; lowered = true)   # a kernel's MIR on real registers (ComplexBi kept)
kernel_c("k0", "kernel.plane.symjit")          # the kernel as C: static int k0(double *mem)
l = lower(p)                                   # UIR v0.8: rows on edges, the layout, the tables
evaluate(l, LArena(l), momenta)                # bit for bit evaluate(p, a, momenta)
o = optimize(l)                                # sort_runs, fuse, reuse_planes: the same values
e = encode(o)                                  # one stream of 8-byte words (sources, then rows), the edge table
compile("out-vm", p; interpreter = true)       # the same folder, runner.c on the UIR interpreter
```

## Artifacts and installation

`load_artifact`, `load_programs` and `load_examples` take a pyAmpliCol LC artifact directory (its
`artifact.json`) as they take a pac2json folder. The first time, the artifact is recorded by
pac2json (`capture(artifact)`), into this package's scratch space (Scratch.jl), in a folder named by
the artifact's `artifact_id`; afterwards that folder is used. `capture(artifact; parameters =
"params.json")` records with other model parameters (a folder of its own).

A program's `alpha_s` is `(value, power)`: its scales are at alpha_s = value, and |M|² ∝
alpha_s^power. A capture that did not measure the power (pac-miner's pac2json evaluates nothing)
has `power` null, read as `nothing`; `alpha_s_power(artifact)` measures it as pac2json did with
RustiCol: the artifact captured again at twice its alpha_s, both programs run by the Julia runner at
16 phase-space points, every ratio of totals 2^power to 1e-12 (else an error).
`alpha_s_power(base, doubled)` does it for two folders, and `load_programs(artifact;
measure_alpha_s = true)` fills the powers in.

pac2json is built on first use, or by `PacRunner.install()`: pacuda (github.com/siravan/pacuda at
a pinned commit) and its patched RustiCol (MadGraph7's `pyamplicol/` at the commit pacuda pins,
with pacuda's patches; only RustiCol's Rust crate is compiled: no Python, no pyAmpliCol build, no
Symbolica), into the scratch space; only the pac2json binary is kept. It needs git, tar, patch, a C
compiler and Rust installed with rustup (pacuda names its toolchain; rustup fetches it). On Linux and
macOS. A first install takes a few minutes (about 4 on 2 cores). Rustup's default install adds a
"stable" toolchain beside the one pacuda pins (about 1 GB each); to have only the pinned one:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
```

The Julia runner calls each symjit kernel through a shared library, built on first use into the
scratch space (`load_programs(path; kernels = ...)`): `:auto` (the default) with symjit-utils'
`sjb2obj` when it is found (`SJB2OBJ`, or the PATH), else from `kernel_c`'s C with `cc`; `:sjb2obj`,
`:c`, or `:none` (no library: only programs without kernel rows run). Both give the same values bit
for bit. The compiled runners need neither: their kernels are C in `runner.c`.

## The compiled folder

`compile(dir, p)` writes a folder that builds on its own:

| File | What it is |
| --- | --- |
| `runner.c` | the program: `int run(double *out, double *p, const double *momenta)` for one point |
| `runner.h` | its prototype and sizes: `RUNNER_LEGS`, `RUNNER_OUTPUTS`, `RUNNER_ARENA_SIZE` (the scratch `p`) |
| `pacops.c`, `pacops.h` | the sources and instructions |
| `testing.c` | with `examples`: the points of examples.json, each output against RustiCol's within 1e-14 of the total |
| `Makefile` | `make` (`runner.o`, `pacops.o`, `testing`), `make test`, `make clean`; `CC`, `CFLAGS` may be set |

```sh
cd out && make test     # u_ubar_to_z_g_g flow:2,4,5,1: 8 points, 384 outputs; 0 beyond 1e-14 of the total ...
```

With `interpreter = true`, `runner.c` holds the program as data (the encoded words, sources first, the
edge table and the tables they index, as one `uir_program`) and includes `uirvm.c`, the UIR
interpreter, copied next to it with `uirvm.h`. `run()`, `runner.h` and the values are the same, bit
for bit. The straight-line C is the CPU runner (about twice as fast); the interpreter is the GPU's,
and its CPU build is its test. The interpreter reads each row's fields straight from its word (the
`UIR_` macros of `uirvm.h`) and takes one-word rows only: a program with a wide row (an index beyond
the one-word fields) is refused.

The interpreter's `runner.c` also builds with nvcc: `make gpu` gives `runner_gpu.o` with
`int run_gpu(double *out, const double *momenta, size_t points)` (`uirvm.h`), every point on the GPU,
one thread each, momenta and outputs point by point on the host. Every function of `pacops`, `uirvm`
and the kernels is then a device function too (`PAC_FN`), and the tables are `__device__` data
(`UIR_DATA`). The outputs are the CPU build's bit for bit (`--fmad=false`, as `-ffp-contract=off`).

## Layout

| Path | What it is |
| --- | --- |
| `src/PacRunner.jl` | the module |
| `src/types.jl` | the program (`Program`, its instructions) |
| `src/program.jl` | the loader of `main.json` |
| `src/evaluators/` | the Julia runners: `arena.jl` (`Arena`, the values of one point of a `Program`), `wavefunctions.jl`, `operations.jl` (the sources and the instructions, RustiCol's arithmetic in the same order), `progeval.jl` (`evaluate(::Program)`), `uireval.jl` (`evaluate(::Lowered)`, the third oracle) |
| `src/compile.jl` | the C generator, from a `Lowered` (`compile(dir, p::Program)` lowers `p` first); `ArenaIndex`, runner.c's arena |
| `src/sjb.jl`, `src/complexify.jl` | the reader of saved symjit applications and symjit's Complexifier |
| `src/install.jl`, `src/kernel_lib.jl` | `install`, `capture` (pac2json for pyAmpliCol artifacts) and the Julia runner's kernel libraries |
| `src/alpha_s.jl` | `alpha_s_power`: the alpha_s power of the programs, from captures at two alpha_s |
| `src/kernel_c.jl` | `kernel_c(name, path)`: a kernel's MIR as one C function, the native code's operations (bit for bit sjb2obj's library); the compiled folders' kernels |
| `src/uir.jl`, `src/lower.jl` | UIR v0.8, the lowered form the back ends read (`Lowered`); `lower(p)` with its checks (`validate`) |
| `src/passes.jl` | the optional passes over a `Lowered`: `sort_runs`, `fuse`, `reuse_planes` (the liveness layout), `optimize` (all three) |
| `src/encode.jl` | `encode(l)`: the device form (one stream of 64-bit words, the SOURCE rows first, END last; the 32-bit edge table), and `decode` |
| `c/pacops.c`, `c/pacops.h` | the sources and instructions in C, copied next to each `runner.c` |
| `src/compile_vm.jl`, `c/uirvm.c`, `c/uirvm.h` | `compile(dir, p; interpreter = true)`: the program as tables for the UIR interpreter, in portable C (one source for the CPU and the GPU) |
| `CONVENTIONS.md` | the meaning of main.json's fields and instructions (pac2json's legend) |
| `test/runtests.jl` | the short tests |
| `test/test_*.jl` | the longer comparisons, one script each (usage at the top of each); `test_lower.jl` reads every lowered row back to its operation and runs `evaluate(::Lowered)` against `evaluate(::Program)`, plain and after each pass |
| `test/bench_compile.jl` | timing of the compiled C (straight-line and interpreted) against the Julia runner |
| `test/data/` | the test cases (`test/data/README.md`) |

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Without external tools this tests the loader, the `.sjb` reader, the Complexifier, the Julia runner
against RustiCol's outputs, and, with a C compiler (`CC`, default `cc`), the compiled C: its kernels
are C (`kernel_c`), so every program's folder builds and passes `make test`, and every program runs
in Julia with kernel libraries built from that C. With symjit-utils' `sjb2obj`, the tests also build
the kernels' native libraries: Julia and C against them bit for bit, and each kernel's C against its
library. With nvcc and a GPU, the interpreter's GPU build against its CPU build. With
`PACRUNNER_ARTIFACT` an LC artifact and pac2json installed, its capture:

```sh
SJB2OBJ=/path/to/sjb2obj PACRUNNER_ARTIFACT=/path/to/lc julia --project -e 'using Pkg; Pkg.test()'
```

The scripts compare with symjit and RustiCol directly, for example
`julia --project test/test_compile.jl SJB2OBJ` (every program of every case, C against Julia and
RustiCol), `julia --project test/test_kernel_c.jl SJB2OBJ` (each kernel's C against its sjb2obj
library), `julia --project test/test_gpu.jl` (the GPU build against the CPU build, every program;
with nvcc and a GPU, `Pkg.test()` runs it too), `julia --project test/test_sjb.jl SJBDUMP DIR` (the reader against symjit's own, with
`test/sjbdump`) and `julia --project test/test_wavefunctions.jl` (against RustiCol's
wavefunctions, built with rustc).

The C must be built with `-ffp-contract=off` (nvcc: `--fmad=false`): the values are bit for bit only
without contracted multiply-adds.
