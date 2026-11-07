# operations.jl and wavefunctions.jl on a pac2json folder: every program of main.json run on its
# examples.json points by a minimal interpreter of the body, each output compared with the CPU's
# (RustiCol's) bit for bit. The folder must have the kernels' shared libraries (pac2json --sjb2obj).
# usage: julia --project test/test_operations.jl PAC2JSON_DIR
include(joinpath(@__DIR__, "minijson.jl"))
using PacRunner
using PacRunner: Source, apply_source, wavefunction, vertex!, propagator, scale!, clear!, closure, kernel!, reduction,
    KernelLibrary, load_kernel
import PacRunner: get_vector, get_scale, get_factor

include(joinpath(@__DIR__, "interpreter.jl"))

# Every output against examples.json: bit for bit, or within TOLERANCE of the point's total. The
# CPU runs symjit 2.27.3's AVX kernels for whole 4-point blocks, and their complex products round
# differently from the scalar kernels in the libraries (PACUDA.md §9): last-bit differences.
const TOLERANCE = 1e-14

function main(dir)
    m = read_json(joinpath(dir, "main.json"))
    ex = read_json(joinpath(dir, "examples.json"))
    kernels = Dict{String,KernelLibrary}()
    for k in m["kernels"]
        lib = k["library"]
        lib === nothing && error("kernel $(k["name"]) has no library: run pac2json with --sjb2obj")
        kernels[k["name"]] = load_kernel(joinpath(dir, lib["path"]), lib["symbol"], lib["states"], lib["outputs"])
    end
    (points, outputs, differ, worst) = (0, 0, 0, 0.0)
    for (prog, e) in zip(m["programs"], ex["programs"])
        (prog["process"], prog["colour"]) == (e["process"], e["colour"]) || error("examples.json is not in main.json's order")
        for pt in e["points"]
            got = run_program(prog, kernels, [Float64.(v) for v in pt["momenta"]])
            want = Float64.(pt["outputs"])
            total = abs(Float64(pt["total"]))
            points += 1
            outputs += length(want)
            for (g, w) in zip(got, want)
                if reinterpret(UInt64, g) != reinterpret(UInt64, w)
                    differ += 1
                    worst = max(worst, abs(g - w) / max(total, floatmin()))
                end
            end
        end
    end
    println("$(length(m["programs"])) programs, $points points, $outputs outputs: ", outputs - differ, " bit for bit, $differ differ",
            differ > 0 ? " (max $(round(worst, sigdigits = 2)) of the point's total)" : "")
    return worst <= TOLERANCE ? 0 : 1
end

# as a script (runtests.jl includes this file for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || (println("usage: julia --project test/test_operations.jl PAC2JSON_DIR"); exit(2))
    exit(main(ARGS[1]))
end
