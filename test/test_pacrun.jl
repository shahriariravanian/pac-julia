# pacrun.jl on a pac2json folder: every program loaded with load_programs and evaluated on its
# examples.json points (one Arena per program, reused from point to point), each output compared
# bit for bit with the reference interpreter (interpreter.jl: string-keyed, the same operations),
# and with the CPU's outputs in examples.json within TOLERANCE of the point's total (the CPU's
# symjit 2.27.3 AVX kernels round differently from the scalar kernel libraries, see
# test_operations.jl). The folder must have the kernels' shared libraries (pac2json --sjb2obj).
# usage: julia test_pacrun.jl PAC2JSON_DIR
include(joinpath(@__DIR__, "..", "pacrun.jl"))
include(joinpath(@__DIR__, "minijson.jl"))
include(joinpath(@__DIR__, "interpreter.jl"))

const TOLERANCE = 1e-14

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

function main(dir)
    path = joinpath(dir, "main.json")
    programs = load_programs(path)
    m = read_json(path)
    ex = read_json(joinpath(dir, "examples.json"))
    libraries = Dict(k["name"] => load_kernel(joinpath(dir, k["library"]["path"]), k["library"]["symbol"],
                                              k["library"]["states"], k["library"]["outputs"]) for k in m["kernels"])
    (points, outputs, vs_reference, vs_cpu, worst) = (0, 0, 0, 0, 0.0)
    (t_indexed, t_reference) = (0.0, 0.0)
    for (p, prog, e) in zip(programs, m["programs"], ex["programs"])
        (p.process, p.colour) == (e["process"], e["colour"]) || error("examples.json is not in main.json's order")
        a = Arena(p)
        for pt in e["points"]
            momenta = [Tuple(Float64.(v)) for v in pt["momenta"]]
            t_indexed += @elapsed got = evaluate(p, a, momenta)
            t_reference += @elapsed ref = run_program(prog, libraries, [Float64.(v) for v in pt["momenta"]])
            want = Float64.(pt["outputs"])
            total = abs(Float64(pt["total"]))
            length(got) == length(want) == length(ref) || error("$(p.process): $(length(got)) outputs")
            points += 1
            outputs += length(got)
            vs_reference += count(!same(g, r) for (g, r) in zip(got, ref))
            for (g, w) in zip(got, want)
                if !same(g, w)
                    vs_cpu += 1
                    worst = max(worst, abs(g - w) / max(total, floatmin()))
                end
            end
        end
    end
    println("$(length(programs)) programs, $points points, $outputs outputs")
    println("  against the reference interpreter: $vs_reference differ")
    println("  against the CPU (examples.json):   $vs_cpu differ", vs_cpu > 0 ? " (max $(round(worst, sigdigits = 2)) of the point's total)" : "")
    println("  time per point: indexed $(round(1e3 * t_indexed / points, sigdigits = 3)) ms, reference $(round(1e3 * t_reference / points, sigdigits = 3)) ms (first calls compile)")
    return vs_reference == 0 && worst <= TOLERANCE ? 0 : 1
end

length(ARGS) == 1 || (println("usage: julia test_pacrun.jl PAC2JSON_DIR"); exit(2))
exit(main(ARGS[1]))
