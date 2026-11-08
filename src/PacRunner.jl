# PacRunner: pyAmpliCol's recurrence programs (pac2json folders) in Julia and C.
#
# The runner: `load_artifact` reads one program of main.json into a Program (the template: every
# name resolved to an index), `evaluate` runs it on one phase-space point with an Arena (the
# values). The instructions are evaluators/operations.jl's, the sources evaluators/wavefunctions.jl's.
#
#   p = load_artifact("folder", 1)                  # or load_programs("folder"): a pac2json folder (or
#                                                   # its main.json), or a pyAmpliCol LC artifact, which
#                                                   # pac2json captures once (install.jl)
#   a = Arena(p)
#   outputs = evaluate(p, a, momenta)               # momenta[leg] = (E, px, py, pz), the caller's
#
# UIR: `lower(p)` turns a Program into a Lowered (uir.jl, lower.jl), the form the back ends read;
# `evaluate(l, LArena(l), momenta)` runs it, bit for bit as evaluate(p, ...).
#
# The C: `compile(dir, p)` writes dir/runner.c (`int run(double *out, double *p, const double
# *momenta)`) from lower(p), with pacops.c/h and the kernels' libraries; bit for bit the Julia
# runner's values.
#
# The kernels: `read_sjb(path)` reads a saved symjit application (.sjb, .plane.symjit) and its MIR;
# `read_sjb(path; lowered = true)` lowers a complex MIR to real operations (complexify.jl).
module PacRunner

using JSON
using Libdl
using Scratch
using SHA

include("types.jl")
include("program.jl")
include("uir.jl")
include("lower.jl")
include("passes.jl")
include("encode.jl")
# the evaluators: the arithmetic, then the runners of a Program and of a Lowered
include("evaluators/arena.jl")
include("evaluators/wavefunctions.jl")
include("evaluators/operations.jl")
include("evaluators/progeval.jl")
include("evaluators/uireval.jl")
include("compile.jl")
include("compile_vm.jl")
include("sjb.jl")
include("complexify.jl")
include("kernel_c.jl")
include("kernel_lib.jl")
include("install.jl")

# the runner
export Program, Arena, ArenaIndex, read_main, load_artifact, load_programs, load_examples, evaluate
# UIR
export Lowered, LArena, lower, validate, sort_runs, fuse, reuse_planes, optimize, Encoded, encode, decode
# the C
export compile
# the kernels
export read_sjb, SavedApplication, MIR, complexify, listing, kernel_c
# pac2json, for pyAmpliCol artifacts
export install, capture

end
