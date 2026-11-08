# PacRunner

pyAmpliCol's recurrence programs (pac2json folders: `main.json`, `examples.json`, the symjit
kernels) run in Julia, and compiled to C. The C runner is the base for the GPU runner and the
reference it is tested against: its values are bit for bit the Julia runner's, which are RustiCol's.

```julia
using PacRunner
p = load_artifact("folder/main.json", 1)       # or load_programs("folder/main.json")
a = Arena(p)
outputs = evaluate(p, a, momenta)              # momenta[leg] = (E, px, py, pz), the caller's legs
compile("out", p; examples = load_examples("folder/examples.json", 1))   # the folder out/, below (from lower(p))
app = read_sjb("kernel.plane.symjit"; lowered = true)   # a kernel's MIR on real registers (ComplexBi kept)
kernel_c("k0", "kernel.plane.symjit")          # the kernel as C: static int k0(double *mem)
l = lower(p)                                   # UIR v0.7: rows on edges, the layout, the tables
evaluate(l, LArena(l), momenta)                # bit for bit evaluate(p, a, momenta)
o = optimize(l)                                # sort_runs, fuse, reuse_planes: the same values
e = encode(o)                                  # one stream of 8-byte words (sources, then rows), the edge table
compile("out-vm", p; interpreter = true)       # the same folder, runner.c on the UIR interpreter
```

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
and its CPU build is its test.

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
| `src/kernel_c.jl` | `kernel_c(name, path)`: a kernel's MIR as one C function, the native code's operations (bit for bit sjb2obj's library); the compiled folders' kernels |
| `src/uir.jl`, `src/lower.jl` | UIR v0.7, the lowered form the back ends read (`Lowered`); `lower(p)` with its checks (`validate`) |
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
are C (`kernel_c`), so every program's folder builds and passes `make test`. The Julia runner calls
the kernels' libraries, built by symjit-utils' `sjb2obj`; with them, every program also runs in
Julia, C against Julia bit for bit, and each kernel's C against its library:

```sh
SJB2OBJ=/path/to/sjb2obj julia --project -e 'using Pkg; Pkg.test()'
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
