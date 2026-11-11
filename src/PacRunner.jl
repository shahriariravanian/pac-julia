# PacRunner: pyAmpliCol's LC recurrence programs in Julia, C and on the GPU, bit for bit RustiCol.
#
# The programs come from RustiCol's recurrence export (pyAmpliCol 1.0.x, schema 2):
#
#   ls = load_recurrence_export("capture.json")    # one Lowered (UIR v0.8) per program entry
#   outputs = evaluate(ls[1], LArena(ls[1]), momenta)   # momenta[leg] = (E, px, py, pz), the caller's
#   compile("out", ls[1])                          # dir/runner.c, pacops.c/h, a Makefile; with
#                                                  # `interpreter = true` on the UIR interpreter
#
# UIR (uir.jl, validate.jl) is the form every back end reads; passes.jl rewrites it (sort_runs,
# fuse, reuse_planes), encode.jl packs it into 8-byte words for the interpreters. The arithmetic is
# evaluators/operations.jl's, the sources evaluators/wavefunctions.jl's.
#
# The kernels: `read_sjb(path)` reads a saved symjit application (.sjb, .plane.symjit) and its MIR;
# `read_sjb(path; lowered = true)` lowers a complex MIR to real operations (complexify.jl), and
# kernel_c.jl writes it as C.
#
# v1/: the earlier path, pac2json's main.json (pyAmpliCol 0.2.x): `load_artifact`/`load_programs`
# read a main.json into a Program, `evaluate(p, Arena(p), momenta)` runs it, `lower(p)` makes its
# Lowered; install.jl builds pac2json (pacuda's patched RustiCol, with cargo) to capture an artifact.
module PacRunner

using JSON
using Base64
using Libdl
using Scratch
using SHA

include("types.jl")
include("uir.jl")
include("validate.jl")
include("passes.jl")
include("encode.jl")
# the evaluators: the arithmetic, the sources, the runner of a Lowered
include("evaluators/wavefunctions.jl")
include("evaluators/operations.jl")
include("evaluators/uireval.jl")
include("compile.jl")
include("compile_vm.jl")
# the kernels
include("sjb.jl")
include("complexify.jl")
include("kernel_c.jl")
include("kernel_lib.jl")
# RustiCol's recurrence export (pyAmpliCol 1.0.x)
include("recurrence_export.jl")
# UMAMI libraries for MadSpace, one per program
include("umami.jl")
# v1: pac2json's main.json (pyAmpliCol 0.2.x)
include("v1/types.jl")
include("v1/program.jl")
include("v1/lower.jl")
include("v1/arena.jl")
include("v1/operations.jl")
include("v1/progeval.jl")
include("v1/compile.jl")
include("v1/install.jl")
include("v1/alpha_s.jl")

# RustiCol's recurrence export
export load_recurrence_export, recurrence_references, umami_libraries
# UIR
export Lowered, LArena, evaluate, validate, sort_runs, fuse, reuse_planes, optimize, Encoded, encode, decode
# the C
export compile, ArenaIndex
# the kernels
export read_sjb, SavedApplication, MIR, complexify, listing, kernel_c
# v1: main.json, pac2json
export Program, Arena, read_main, load_artifact, load_programs, load_examples, lower, install, capture, alpha_s_power

end
