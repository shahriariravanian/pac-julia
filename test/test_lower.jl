# lower.jl and evaluators/uireval.jl on pac2json folders (default: every case in test/data): every program is
# lowered and validated; every row is read back through the edge table to the Program operation it
# came from (operands, momenta, tables, flags); and on examples.json's points evaluate(::Lowered)
# must equal evaluate(::Program) bit for bit, plain, after each pass (passes.jl: sort_runs, fuse,
# reuse_planes, optimize) and decoded from its encoding (encode.jl). Programs with kernel rows run only with SJB2OBJ (their libraries are built
# in a copy of the folder, as test_compile.jl does).
# usage: julia --project test/test_lower.jl [--sjb2obj SJB2OBJ] [PAC2JSON_DIR...]
include(joinpath(@__DIR__, "compile_common.jl"))
using PacRunner
using PacRunner: LOp, LCurrent, Vertex, Propagator, Kernel, Closure, Scale, Clear, KernelInput, SCALE,
                 MOMENTUM_U, MOMENTUM_V, KERNEL_DIRECT, VERTEX_OPS, PROPAGATOR_OPS, FAMILY_TYPES, problems,
                 edge_current, edge_form, is_vertex


# the binding inputs a stub and scale set stand for
function binding_inputs(l::Lowered, t::Int)::Vector{KernelInput}
    (s, set) = l.instances[t]
    return [i.kind == SCALE ? KernelInput(SCALE, l.scalesets[set][i.n], false) : i for i in l.stubs[s].inputs]
end

# The differences between program `p` and its lowering `l`, read back row by row
function roundtrip(p::Program, l::Lowered)::Vector{String}
    out = String[]
    check(ok, k, what) = ok || push!(out, "row $k: $what")
    # the rows: the sources (SOURCE rows), the body, the reduction (REDUCE rows)
    srcs = [(c, cur.source) for (c, cur) in enumerate(p.currents) if cur.source !== nothing]
    terms = [(h, a, w) for (h, ts) in enumerate(p.reduction) for (a, w) in ts]
    (ns, nb) = (length(srcs), length(p.body))
    length(l.ops) == ns + nb + length(terms) ||
        return ["$(length(l.ops)) rows for $ns sources, $nb operations and $(length(terms)) reduction terms"]
    [c.components for c in l.currents] == [c.components for c in p.currents] || push!(out, "component counts")
    for ((c, s), o) in zip(srcs, l.ops[1:ns])
        (o.op, edge_current(l, o.d), edge_form(l, o.d), PacRunner.source_fields(o)..., o.t, o.x, o.l) ==
            (PacRunner.SOURCE, c, s.form, FAMILY_TYPES[s.family], s.orientation == "antiparticle", s.helicity, s.chirality,
             s.mass, s.factor, 0) || push!(out, "source of current $c")
    end
    for ((h, a, w), o) in zip(terms, l.ops[ns+nb+1:end])
        (o.op, o.d, o.t, o.x) == (PacRunner.REDUCE, h, a, 0) && isequal(l.weights[o.n], w) || push!(out, "reduction term ($h, $a)")
    end
    cur(e) = edge_current(l, e)
    form(e) = edge_form(l, e)
    for (k, (op, o)) in enumerate(zip(p.body, l.ops[ns+1:ns+nb]))
        if op isa Vertex
            check(o.op == VERTEX_OPS[op.type][1], k, "opcode")
            check((cur(o.d), cur(o.l), cur(o.r)) == (op.out, op.l, op.r), k, "operands")
            o.op == PacRunner.V_THREE_VECTOR && check((form(o.l), form(o.r)) == (op.u, op.v), k, "momenta")
            check((o.t, o.x, o.set) == (op.coupling, op.factor, op.mode == "="), k, "coupling, factor, mode")
            check(o.first && o.last, k, "a vertex row is its own run")
        elseif op isa Propagator
            check(o.op == PROPAGATOR_OPS[op.type][1], k, "opcode")
            check((cur(o.d), form(o.d), o.x) == (op.current, op.momentum, op.factor), k, "operands")
        elseif op isa Kernel
            b = p.bindings[op.binding]
            check(binding_inputs(l, o.t) == b.inputs, k, "binding inputs")
            check(l.stubs[l.instances[o.t][1]].outputs == b.outputs, k, "outputs")
            check(l.stubs[l.instances[o.t][1]].kernel.name == b.kernel.name, k, "kernel")
            direct = b.direct && PacRunner.is_one(p.factors[op.factor])
            check((o.op == KERNEL_DIRECT) == direct, k, "direct")
            check((cur(o.d), cur(o.l), cur(o.r)) == (op.out, op.l, op.r), k, "operands")
            kinds = [i.kind for i in b.inputs]
            MOMENTUM_U in kinds && check(form(o.l) == op.u, k, "momentum u")
            MOMENTUM_V in kinds && check(form(o.r) == op.v, k, "momentum v")
            check((o.x, o.set) == (op.factor, direct || op.mode == "="), k, "factor, mode")
        elseif op isa Closure
            check((o.op, o.d, cur(o.l), cur(o.r), o.x) == (PacRunner.CLOSURE, op.amplitude, op.l, op.r, op.factor), k, "operands")
            check(l.lists[o.t] == op.component_factors, k, "component factors")
        elseif op isa Scale
            check((o.op, cur(o.d), o.n, o.x) == (PacRunner.SCALE_ROW, op.current, op.components, op.factor), k, "operands")
        elseif op isa Clear
            check((o.op, cur(o.d), o.n) == (PacRunner.CLEAR, op.current, op.components), k, "operands")
        end
    end
    return out
end

# `l` with row k replaced by `f(row k)` (a copy)
function with_row(f, l::Lowered, k::Int)::Lowered
    m = deepcopy(l)
    m.ops[k] = f(m.ops[k])
    return m
end
setrow(o::LOp; kw...) = LOp(o.op; set = o.set, first = o.first, last = o.last, d = o.d, t = o.t, l = o.l, r = o.r,
                            n = o.n, x = o.x, kw...)

# two consecutive vertex rows on one destination, the second "+=": a run `lower` could fuse
function fusable(l::Lowered)
    k = findfirst(k -> is_vertex(l.ops[k].op) && is_vertex(l.ops[k+1].op) && l.ops[k].d == l.ops[k+1].d &&
                       !l.ops[k+1].set, 1:length(l.ops)-1)
    return k
end

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y) || (isnan(x) && isnan(y))

# `lower(p)` and its forms after each pass, by name
# and what a device reads: the encoded rows, edges and sources decoded (encode.jl), plain and optimized
function passes(l::Lowered)
    o = optimize(l)
    return ["plain" => l, "sort_runs" => sort_runs(l), "fuse" => fuse(l), "reuse_planes" => reuse_planes(l),
            "encoded" => decode(encode(l), l), "encoded optimized" => decode(encode(o), o), "optimize" => o]
end

# The outputs of `lower(p)` and of each pass on `points` that differ (bit for bit) from the Program's
# (`differ`, by pass in `by`), with what the passes did: the rows fused by `optimize`, the planes of
# the plain and the optimized layout, and the peak of live planes
function compare(p::Program, points)
    l = lower(p)
    variants = passes(l)
    a = Arena(p)
    arenas = [LArena(m) for (_, m) in variants]
    by = Dict(name => 0 for (name, _) in variants)
    for pt in points
        momenta = [Tuple(Float64.(v)) for v in pt.momenta]
        want = evaluate(p, a, momenta)
        for ((name, m), la) in zip(variants, arenas)
            by[name] += count(!, same.(evaluate(m, la, momenta), want))
        end
    end
    o = last(variants)[2]
    return (differ = sum(values(by)), by = by, fused = PacRunner.fused_rows(o), planes = l.planes,
            reused = o.planes, peak = PacRunner.liveness_layout(o)[3])
end

has_kernel_rows(p::Program) = any(op -> op isa Kernel, p.body)

# `p` with every "=" vertex row made "+=": the currents then start from zero and accumulate, with the
# same values (0 + c is c) but no fresh first write, so reuse_planes must keep them live from the start
function first_writes_add(p::Program)::Program
    body = [op isa Vertex && op.mode == "=" ? Vertex(op.type, op.out, "+=", op.l, op.r, op.u, op.v, op.coupling, op.factor) : op
            for op in p.body]
    return Program((f == :body ? body : getfield(p, f) for f in fieldnames(Program))...)
end

function test_folder(dir::String; sjb2obj = nothing)::Int
    sjb2obj === nothing || (dir = prepare(sjb2obj, dir; quiet = true))
    ex = isfile(joinpath(dir, "examples.json")) ? read_main(joinpath(dir, "examples.json")).programs : nothing
    (failed, skipped, outputs, joined, plain, reused, peak) = (0, 0, 0, 0, 0, 0, 0)
    for (n, p) in enumerate(load_programs(joinpath(dir, "main.json")))
        diffs = try
            roundtrip(p, lower(p))
        catch e
            [sprint(showerror, e)]
        end
        if isempty(diffs) && ex !== nothing
            if has_kernel_rows(p) && sjb2obj === nothing
                skipped += 1
            else
                r = compare(p, ex[n].points)
                outputs += length(ex[n].points) * length(p.reduction)
                (joined, plain, reused, peak) = (joined + r.fused, plain + r.planes, reused + r.reused, peak + r.peak)
                for (name, d) in r.by
                    d == 0 || push!(diffs, "evaluate(::Lowered), $name: $d outputs differ")
                end
            end
        end
        isempty(diffs) || (failed += 1; println("  program $n ($(p.process), $(p.colour)): ", join(first(diffs, 5), "; ")))
    end
    println("$(basename(dir)): ", failed == 0 ? "every program lowers and reads back" : "$failed programs FAILED",
            ex === nothing ? " (no examples.json: not evaluated)" :
            "; $outputs outputs bit for bit, plain, after each pass and encoded; optimize: $joined rows fused, " *
            "$plain → $reused planes (peak live $peak)" *
            (skipped > 0 ? "; $skipped programs with kernel rows not run (no --sjb2obj)" : ""))
    return failed
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    k = findfirst(==("--sjb2obj"), args)
    sjb2obj = k === nothing ? nothing : args[k+1]
    k === nothing || deleteat!(args, k:k+1)
    dirs = isempty(args) ? cases() : args
    exit(sum(d -> test_folder(d; sjb2obj), dirs) == 0 ? 0 : 1)
end
