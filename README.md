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
app = read_sjb("kernel.plane.symjit"; lowered = true)   # a kernel's MIR, lowered to real operations
l = lower(p)                                   # UIR v0.6: rows on edges, the layout, the tables
evaluate(l, LArena(l), momenta)                # bit for bit evaluate(p, a, momenta)
o = optimize(l)                                # sort_runs, fuse, reuse_planes: the same values
e = encode(o)                                  # the 8-byte rows, the edge table, the sources
```

## The compiled folder

`compile(dir, p)` writes a folder that builds on its own:

| File | What it is |
| --- | --- |
| `runner.c` | the program: `int run(double *out, double *p, const double *momenta)` for one point |
| `runner.h` | its prototype and sizes: `RUNNER_LEGS`, `RUNNER_OUTPUTS`, `RUNNER_ARENA_SIZE` (the scratch `p`) |
| `pacops.c`, `pacops.h` | the sources and instructions |
| `<sha256>.so` | the symjit kernels' libraries, when the program calls kernels |
| `testing.c` | with `examples`: the points of examples.json, each output against RustiCol's within 1e-14 of the total |
| `Makefile` | `make` (`runner.o`, `pacops.o`, `testing`), `make test`, `make clean`; `CC`, `CFLAGS` may be set |

```sh
cd out && make test     # u_ubar_to_z_g_g flow:2,4,5,1: 8 points, 384 outputs; 0 beyond 1e-14 of the total ...
```

## Layout

| Path | What it is |
| --- | --- |
| `src/PacRunner.jl` | the module |
| `src/types.jl`, `src/arena.jl` | the program (`Program`, its instructions) and the values of one point (`Arena`) |
| `src/pacrun.jl` | the loader of `main.json` and `evaluate` |
| `src/wavefunctions.jl`, `src/operations.jl` | the sources and the instructions, RustiCol's arithmetic in the same order |
| `src/compile.jl` | the C generator, from a `Lowered` (`compile(dir, p::Program)` lowers `p` first); `ArenaIndex`, runner.c's arena |
| `src/sjb.jl`, `src/complexify.jl` | the reader of saved symjit applications and symjit's Complexifier |
| `src/uir.jl`, `src/lower.jl`, `src/uireval.jl` | UIR v0.6, the lowered form the back ends will read (`Lowered`); `lower(p)` with its checks (`validate`); `evaluate(::Lowered)`, the third oracle |
| `src/passes.jl` | the optional passes over a `Lowered`: `sort_runs`, `fuse`, `reuse_planes` (the liveness layout), `optimize` (all three) |
| `src/encode.jl` | `encode(l)`: the device form (64-bit rows, 32-bit edge table, source records), and `decode` |
| `c/pacops.c`, `c/pacops.h` | the sources and instructions in C, copied next to each `runner.c` |
| `CONVENTIONS.md` | the meaning of main.json's fields and instructions (pac2json's legend) |
| `test/runtests.jl` | the short tests |
| `test/test_*.jl` | the longer comparisons, one script each (usage at the top of each); `test_lower.jl` reads every lowered row back to its operation and runs `evaluate(::Lowered)` against `evaluate(::Program)`, plain and after each pass |
| `test/bench_compile.jl` | timing of the compiled C against the Julia runner |
| `test/data/` | the test cases (`test/data/README.md`) |

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Without external tools this tests the loader, the `.sjb` reader, the Complexifier, the Julia runner
against RustiCol's outputs, and, with a C compiler (`CC`, default `cc`), the compiled C. The
programs that call symjit kernels need the kernels' libraries, built by symjit-utils' `sjb2obj`:

```sh
SJB2OBJ=/path/to/sjb2obj julia --project -e 'using Pkg; Pkg.test()'
```

The scripts compare with symjit and RustiCol directly, for example
`julia --project test/test_compile.jl SJB2OBJ` (every program of every case, C against Julia and
RustiCol), `julia --project test/test_sjb.jl SJBDUMP DIR` (the reader against symjit's own, with
`test/sjbdump`) and `julia --project test/test_wavefunctions.jl` (against RustiCol's
wavefunctions, built with rustc).

The C must be built with `-ffp-contract=off` (nvcc: `--fmad=false`): the values are bit for bit only
without contracted multiply-adds.
