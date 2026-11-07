# PacRunner: pyAmpliCol's recurrence programs (pac2json folders) in Julia and C.
#
# The runner: `load_artifact` reads one program of main.json into a Program (the template: every
# name resolved to an index), `evaluate` runs it on one phase-space point with an Arena (the
# values). The instructions are operations.jl's, the sources wavefunctions.jl's.
#
#   p = load_artifact("folder/main.json", 1)        # or load_programs("folder/main.json")
#   a = Arena(p)
#   outputs = evaluate(p, a, momenta)               # momenta[leg] = (E, px, py, pz), the caller's
#
# The C: `compile(dir, p)` writes dir/runner.c (`int run(double *out, double *p, const double
# *momenta)`) with pacops.c/h and the kernels' libraries; bit for bit the Julia runner's values.
#
# The kernels: `read_sjb(path)` reads a saved symjit application (.sjb, .plane.symjit) and its MIR;
# `read_sjb(path; lowered = true)` lowers a complex MIR to real operations (complexify.jl).
module PacRunner

using JSON
using Libdl

include("types.jl")
include("wavefunctions.jl")
include("operations.jl")
include("arena.jl")
include("pacrun.jl")
include("compile.jl")
include("sjb.jl")
include("complexify.jl")

# the runner
export Program, Arena, ArenaIndex, read_main, load_artifact, load_programs, evaluate
# the C
export compile
# the kernels
export read_sjb, SavedApplication, MIR, complexify, listing

end
