# A runner for pac2json folders: `load_artifact` reads one program of main.json into a Program
# (the template: every name resolved to an index), `evaluate` runs it on one phase-space point
# with an Arena (the values). The instructions are operations.jl's, the sources wavefunctions.jl's.
#
#   p = load_artifact("folder/main.json", 1)        # or load_programs("folder/main.json")
#   a = Arena(p)
#   outputs = evaluate(p, a, momenta)               # momenta[leg] = (E, px, py, pz), the caller's
using JSON
using Libdl

include("types.jl")
include("wavefunctions.jl")
include("operations.jl")
include("arena.jl")
include("pacrun.jl")
include("compile.jl")
