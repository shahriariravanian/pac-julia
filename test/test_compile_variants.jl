# compile.jl on the paths the test cases do not reach: programs of test/data changed the ways the
# data never are (a group factor, several terms per output, crossing with signs −1, scale and
# clear instructions, direct bindings, a kernel input read from the destination or from the row's
# factor), each compiled,
# built with test/compile_driver.c and compared bit for bit with the Julia runner on the
# examples.json points. A variant must change the Julia runner's outputs (else it tests nothing).
# What lower refuses, compile refuses (a null vertex operand).
# usage: julia --project test/test_compile_variants.jl SJB2OBJ     (CC: the C compiler, default cc)
include(joinpath(@__DIR__, "compile_common.jl"))
using PacRunner: Operation, Vertex, Scale, Clear, Binding, KernelInput, OUT, FACTOR

const DRIVER = joinpath(@__DIR__, "compile_driver.c")

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

# a copy of `p` with some fields replaced
with(p::Program; kw...) = Program((haskey(kw, f) ? kw[f] : getfield(p, f) for f in fieldnames(Program))...)

# the outputs of `p` at `points` by the compiled C and by the Julia runner
function both(p::Program, points, dir::String)
    compile(dir, p)
    exe = build(dir, DRIVER, joinpath(dir, "driver"), `-O2 -ffp-contract=off -Wall -Werror`)
    b = ArenaIndex(lower(p))
    text = read(pipeline(`$exe $(p.legs) $(length(p.reduction)) $(b.size)`; stdin = IOBuffer(points_text(points))), String)
    c = [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in split(strip(text), "\n")]
    a = Arena(p)
    julia = [evaluate(p, a, [Tuple(Float64.(v)) for v in pt.momenta]) for pt in points]
    return (c, julia)
end

# a factor index whose value is not 1 (for the scale instructions), else 1
other_factor(p::Program) = something(findfirst(f -> f != complex(1.0, 0.0), p.factors), 1)

function variants(p::Program)
    legs, H = p.legs, length(p.reduction)
    out = Pair{String,Program}[]
    push!(out, "group factor 3/7" => with(p; group_factor = (3, 7)))
    merged = [vcat(p.reduction[h], [(a, 0.25 * w) for (a, w) in p.reduction[mod1(h + 1, H)]]) for h in 1:H]
    push!(out, "several terms per output" => with(p; reduction = merged))
    # the program's leg l reads the caller's leg legs + 1 - l, the first two with sign −1
    push!(out, "crossing with signs -1" => with(p; crossing = [(legs + 1 - l, l <= 2 ? -1.0 : 1.0) for l in 1:legs]))
    # after every 5th vertex, scale its destination's first n - 1 components; after every 7th,
    # clear its first component
    body = Operation[]
    x = other_factor(p)
    for (k, op) in enumerate(p.body)
        push!(body, op)
        if op isa Vertex
            n = p.currents[op.out].components
            k % 5 == 0 && push!(body, Scale(op.out, max(n - 1, 1), x))
            k % 7 == 0 && push!(body, Clear(op.out, 1))
        end
    end
    push!(out, "scale and clear instructions" => with(p; body = body))
    if !isempty(p.bindings)
        push!(out, "direct bindings" => with(p; bindings = [Binding(b.kernel, true, b.inputs, b.outputs) for b in p.bindings]))
        # the first input of every binding read from the destination's component 0 (re)
        push!(out, "kernel input from the destination" =>
              with(p; bindings = [Binding(b.kernel, b.direct, vcat([KernelInput(OUT, 1, false)], b.inputs[2:end]), b.outputs)
                                  for b in p.bindings]))
        # the first input of every binding read from the row's factor (re)
        push!(out, "a FACTOR kernel input" =>
              with(p; bindings = [Binding(b.kernel, b.direct, vcat([KernelInput(FACTOR, 0, false)], b.inputs[2:end]), b.outputs)
                                  for b in p.bindings]))
    end
    return out
end

# compile must refuse these
function refusals(p::Program)
    out = Pair{String,Program}[]
    k = findfirst(op -> op isa Vertex, p.body)
    if k !== nothing
        v = p.body[k]
        body = copy(p.body)
        body[k] = Vertex(v.type, v.out, v.mode, v.l, 0, v.u, v.v, v.coupling, v.factor)
        push!(out, "a null vertex operand" => with(p; body = body))
    end
    return out
end

function main(sjb2obj)
    failed = 0
    # the running test, a gluon program with no kernels, a program with 6 kernels, a crossed one
    picks = [("uux_zgg", 1), ("ppjjj_crossed", 1), ("ppjjj_crossed", 4), ("ppjjj_crossed", 10)]
    for (case, n) in picks
        dir = prepare(sjb2obj, joinpath(@__DIR__, "data", case); quiet = true)
        p = load_programs(joinpath(dir, "main.json"))[n]
        points = read_main(joinpath(dir, "examples.json")).programs[n].points
        (_, base) = both(p, points, joinpath(dir, "base"))
        println("$case program $n ($(p.process), $(p.colour)):")
        for (k, (name, v)) in enumerate(variants(p))
            (c, julia) = both(v, points, joinpath(dir, "variant-$k"))
            differ = count(!same(x, y) for (cr, jr) in zip(c, julia) for (x, y) in zip(cr, jr))
            changed = any(!same(x, y) for (jr, br) in zip(julia, base) for (x, y) in zip(jr, br))
            ok = differ == 0 && changed
            ok || (failed += 1)
            println("  $(rpad(name, 34)) C vs Julia: $differ differ; changes the outputs: $(changed ? "yes" : "NO")",
                    ok ? "" : "  FAILED")
        end
        for (k, (name, v)) in enumerate(refusals(p))
            refused = try
                compile(joinpath(dir, "refused-$k"), v)
                false
            catch e
                e isa ErrorException || rethrow()
                true
            end
            refused || (failed += 1)
            println("  $(rpad(name, 34)) refused by compile: $(refused ? "yes" : "NO  FAILED")")
        end
    end
    println(failed == 0 ? "all variants pass" : "$failed variants FAILED")
    return failed == 0 ? 0 : 1
end

# as a script (runtests.jl includes this file for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || (println("usage: julia --project test/test_compile_variants.jl SJB2OBJ"); exit(2))
    exit(main(ARGS[1]))
end
