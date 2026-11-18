# A compiled plan (compiled_plan.jl) against the recurrence export of the same process (pyAmpliCol's
# `export-recurrence` of a recurrence-mode artifact), per colour sector: the same currents (keyed by
# what they are: tensor or vector, the external labels, the helicities of the sources they descend
# from), the same interactions and vertex kinds, coefficients equal up to one sign per current (the
# two orientation conventions: the C++ builds currents in ascending label order, the export in the
# colour flow's), and the same closures. Used by runtests.jl ("compiled plan") and by hand on larger
# processes.

module CompareExport

using JSON
using PacRunner: CompiledPlan, CompiledExecution, ModelKernel

const KINDS = Dict("color-ordered-three-vector" => :three_vector, "vector-wedge-vector" => :vector_wedge,
                   "antisymmetric-tensor-vector" => :tensor_vector, "weyl-vector-to-weyl-a" => :weyl_vector_a,
                   "weyl-vector-to-weyl-b" => :weyl_vector_b)
const PROPAGATORS = Dict("vector-propagator-feynman" => :vector, "weyl-propagator-a" => :weyl_a, "weyl-propagator-b" => :weyl_b)

# a program's values as (slot, version): slots are reused (a source, a clear or a `set` not right
# after a clear starts a value)
function program_values(p)
    run_sources = [s for r in p["runs"] for s in r["sources"]]
    version, last = Dict{Int,Int}(), Dict{Int,Symbol}()
    cur(s) = (s, get(version, s, 0))
    start!(s) = (version[s] = get(version, s, 0) + 1)
    sources = Dict{Tuple{Int,Int},Tuple{Int,Int}}()   # value -> (label 1-based, helicity)
    rows = Dict{Tuple{Int,Int},Vector{Any}}()         # value -> [(kind, left, right, coupling × factor)]
    props, closures = Dict{Tuple{Int,Int},Symbol}(), Any[]
    kernels = Dict{Tuple{Int,Int},Vector{Any}}()      # value -> [(binding, left, right, factor, mode)]
    for r in p["body"]
        k = r["kind"]
        if k == "source"
            s = run_sources[r["source"]+1]
            start!(s["current"]); last[s["current"]] = :source
            sources[cur(s["current"])] = (s["leg"] + 1, s["helicity"])
        elseif k == "clear"
            start!(r["current"]); last[r["current"]] = :clear
        elseif k == "vertex"
            (l, rr) = (cur(r["left"]), cur(r["right"]))
            r["mode"] == "set" && get(last, r["current"], :none) != :clear && start!(r["current"])
            last[r["current"]] = :write
            c = complex(r["coupling"]["re"], r["coupling"]["im"]) * complex(r["factor"]["re"], r["factor"]["im"])
            push!(get!(rows, cur(r["current"]), Any[]), (KINDS[split(r["vertex"], ".")[end-1]], l, rr, c))
        elseif k == "propagator"
            props[cur(r["current"])] = PROPAGATORS[split(r["propagator"], ".")[end-1]]
        elseif k == "kernel"
            r["amplitude"] && error("a kernel with an amplitude destination")
            (l, rr) = (cur(r["left"]), cur(r["right"]))
            r["mode"] == "set" && get(last, r["destination"], :none) != :clear && start!(r["destination"])
            last[r["destination"]] = :write
            push!(get!(kernels, cur(r["destination"]), Any[]),
                  (r["binding"], l, rr, complex(r["factor"]["re"], r["factor"]["im"]), r["mode"]))
        elseif k == "closure"
            cf = [complex(x["re"], x["im"]) for x in r["component_factors"]] .* complex(r["factor"]["re"], r["factor"]["im"])
            push!(closures, (cur(r["left"]), cur(r["right"]), cf))
        end
    end
    return (; sources, rows, props, closures, kernels)
end

# keys from the sources up: (tensor, labels, (label, helicity) pairs)
function keys_from(sources::Dict, inputs::Dict, is_tensor)
    key = Dict{Any,Tuple{Bool,Vector{Int},Vector{Tuple{Int,Int}}}}()
    for (v, (label, h)) in sources
        key[v] = (false, [label], [(label, h)])
    end
    changed = true
    while changed
        changed = false
        for (v, sets) in inputs
            haskey(key, v) && continue
            all(s -> all(i -> haskey(key, i), s), sets) || continue
            ks = Set((sort!(unique!(reduce(vcat, [key[i][2] for i in s]))), sort!(unique!(reduce(vcat, [key[i][3] for i in s])))) for s in sets)
            length(ks) == 1 || error("value $v: its inputs disagree on its labels/helicities")
            (labels, hels) = only(ks)
            key[v] = (is_tensor(v), labels, hels)
            changed = true
        end
    end
    return key
end

# the labels (1-based) a flow's currents are built of: consecutive runs of all but its last label
function flow_shape(flow::AbstractString)
    o = parse.(Int, split(split(flow, ":")[2], ","))[1:end-1]
    return Set(sort(o[i:j]) for i in eachindex(o) for j in i+1:length(o))
end

# GF(2): is there one sign per variable with the product of each equation's signs as given?
function signs_consistent(eqs::Vector{Tuple{Vector{Any},Bool}})
    vars = Dict{Any,Int}()
    rows = Tuple{BitVector,Bool}[]
    for (vs, b) in eqs
        foreach(v -> get!(vars, v, length(vars) + 1), vs)
    end
    pivots = Dict{Int,Tuple{BitVector,Bool}}()
    for (vs, b) in eqs
        row = falses(length(vars))
        foreach(v -> row[vars[v]] ⊻= true, vs)
        for (p, (pr, pb)) in pivots
            row[p] && (row .⊻= pr; b ⊻= pb)
        end
        if !any(row)
            b && return false
            continue
        end
        p = findfirst(row)
        for q in collect(keys(pivots))
            (qr, qb) = pivots[q]
            qr[p] && (pivots[q] = (qr .⊻ row, qb ⊻ b))
        end
        pivots[p] = (row, b)
    end
    return true
end

# a program's values with its kernel rows: two-operand ones as interactions, one-operand ones (a
# propagator kernel) folded into the value they write; keyed from the sources up
function export_side(p)
    v = program_values(p)
    bsha = Dict(b["id"] => b["sha256"] for b in p["kernel_bindings"])
    r_rows = Dict{Any,Vector{Any}}(x => Any[(kind, l, r, c, "") for (kind, l, r, c) in rows] for (x, rows) in v.rows)
    single = Dict{Any,Any}()
    for (x, ks) in v.kernels, (b, l, r, f, mode) in ks
        if r[1] === nothing
            single[x] = (l, bsha[b], f)
        else
            push!(get!(r_rows, x, Any[]), (:kernel, l, r, f, bsha[b]))
        end
    end
    r_props = Dict{Any,Any}(x => k for (x, k) in v.props)
    for (x, (l, sha, f)) in single
        r_rows[x] = Any[(kind, a, b, c * f, k) for (kind, a, b, c, k) in r_rows[l]]
        r_props[x] = sha
        delete!(r_rows, l)
    end
    intermediates = Set(l for (l, _, _) in values(single))
    r_inputs = Dict{Any,Set{Vector{Any}}}(x => Set([[l, r] for (_, l, r, _, _) in rows]) for (x, rows) in r_rows)
    tensor_out = Set(x for (x, rows) in v.rows if any(r -> r[1] === :vector_wedge, rows))
    r_key = keys_from(v.sources, r_inputs, x -> x in tensor_out)
    by_r = Dict(k => x for (x, k) in r_key if !(x in intermediates))
    return (; v, r_rows, r_props, r_key, by_r)
end

"""
    compare(plan, export_path; kernels = ModelKernel[]) -> (reports, kernel_differences)

Every materialized colour sector of `plan` against the export's program of the same shape. Kernel
rows: a two-operand row is an interaction (the kernel by SHA-256, its factor); a one-operand row is
a propagator kernel on an intermediate value, folded as compiled mode folds it (the destination has
the intermediate's interactions and the kernel as its propagator). Interactions whose kernel differs
from the export's (a numerically equivalent one) are listed, not errors; everything else must agree.
"""
function compare(plan::CompiledPlan, export_path::AbstractString; kernels = ModelKernel[])
    ex = plan.execution
    sha_of = Dict(k.id => k.sha for k in kernels)
    programs = JSON.parsefile(export_path)["programs"]
    cur = ex.currents
    chirality = Dict(c["current_id"] => c["chirality"] for c in ex.raw["runtime_schema"]["current_storage"]["current_slots"])
    c_sources = Dict(c.id => (c.labels[1], c.spin) for c in values(cur) if c.source)
    c_inputs = Dict{Any,Set{Vector{Any}}}()
    for i in plan.interactions
        push!(get!(c_inputs, i.current, Set{Vector{Any}}()), i.kind === :identity ? Any[i.left] : Any[i.left, i.right])
    end
    c_key = keys_from(c_sources, c_inputs, id -> cur[id].tensor)
    sectors = sort(unique(reduce(vcat, [c.sectors for c in values(cur)])))
    reports, differences = [], []
    eqs = Tuple{Vector{Any},Bool}[]
    sides = map(export_side, programs)
    for s in sectors
        members = Set(c.id for c in values(cur) if isempty(c.sectors) || s in c.sectors)
        want = Set(c_key[id] for id in members)
        matching = [i for (i, side) in enumerate(sides) if Set(keys(side.by_r)) == want]
        isempty(matching) && error("sector $s: no export program with its currents")
        p = programs[first(matching)]
        (; v, r_rows, r_props, r_key, by_r) = sides[first(matching)]
        by_c = Dict(c_key[id] => id for id in members)
        Set(keys(by_c)) == Set(keys(by_r)) || error("sector $s ($(p["colour_id"])): different currents")
        n = 0
        for (k, x) in by_r
            haskey(r_rows, x) || continue
            id = by_c[k]
            c_rows = [i for i in plan.interactions if i.current == id]
            ckey(i) = Set(i.kind === :identity ? [c_key[i.left]] : [c_key[i.left], c_key[i.right]])
            Set(ckey(i) for i in c_rows) == Set(Set([r_key[l], r_key[r]]) for (_, l, r, _, _) in r_rows[x]) ||
                error("sector $s: current $(k): different interactions")
            for (kind, l, r, coef, sha) in r_rows[x]
                i = only(i for i in c_rows if ckey(i) == Set([r_key[l], r_key[r]]))
                swapped = c_key[i.left] != r_key[l]
                if kind === :kernel
                    i.kind === :kernel || error("sector $s: current $k: $(i.kind) vs a kernel row")
                    if get(sha_of, i.kernel, "") != sha || swapped
                        push!(differences, (sector = s, current = k, compiled = i.kernel, compiled_sha = get(sha_of, i.kernel, ""),
                                            export_sha = sha, swapped = swapped,
                                            chiralities = (chirality[i.left], chirality[i.right], chirality[id])))
                        n += 1
                        continue   # an equivalent kernel: its coefficient is relative to it
                    end
                else
                    i.kind === kind || error("sector $s: current $k: $(i.kind) vs $kind")
                    swapped && !(kind in (:three_vector, :vector_wedge)) && error("sector $s: current $k: $kind swapped")
                end
                σ = i.coefficient / (coef * (swapped ? -1 : 1))
                abs(abs(σ) - 1) < 1e-12 && abs(imag(σ)) < 1e-12 || error("sector $s: current $k: coefficient $(i.coefficient) vs $coef")
                push!(eqs, (Any[id, i.left, i.right], real(σ) < 0))
                n += 1
            end
            (id in ex.propagated) == haskey(r_props, x) || error("sector $s: current $k: propagated in one only")
            if haskey(r_props, x)
                cp, rp = plan.propagators[id], r_props[x]
                (cp isa Symbol ? cp === rp : get(sha_of, cp, "") == rp) ||
                    error("sector $s: current $k: propagator $cp vs $rp")
            end
        end
        # closures: the same pairs, the same contraction up to the pair's signs
        c_cl = [c for c in plan.closures if all(i -> i in members, (c.left, c.right))]
        r_cl = Dict(Set([r_key[l], r_key[r]]) => cf for (l, r, cf) in v.closures)
        length(c_cl) == length(r_cl) || error("sector $s: $(length(c_cl)) closures vs $(length(r_cl))")
        for c in c_cl
            cf = r_cl[Set([c_key[c.left], c_key[c.right]])]
            σ = c.coefficient .* c.contraction ./ cf
            all(x -> abs(x - σ[1]) < 1e-12, σ) && abs(abs(σ[1]) - 1) < 1e-12 || error("sector $s: closure $(c.root): $(c.contraction) vs $cf")
            push!(eqs, (Any[c.left, c.right], real(σ[1]) < 0))
        end
        push!(reports, (sector = s, flow = p["colour_id"], currents = length(members), interactions = n,
                        propagators = length(r_props), closures = length(c_cl)))
    end
    signs_consistent(eqs) || error("the coefficients' signs are not one per current")
    return reports, differences
end

end
