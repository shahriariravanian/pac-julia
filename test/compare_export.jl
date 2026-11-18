# A compiled plan (compiled_plan.jl) against the recurrence export of the same process (pyAmpliCol's
# `export-recurrence` of a recurrence-mode artifact), per colour sector: the same currents (keyed by
# what they are: tensor or vector, the external labels, the helicities of the sources they descend
# from), the same interactions and vertex kinds, coefficients equal up to one sign per current (the
# two orientation conventions: the C++ builds currents in ascending label order, the export in the
# colour flow's), and the same closures. Used by runtests.jl ("compiled plan") and by hand on larger
# processes.

module CompareExport

using JSON
using PacRunner: CompiledPlan, CompiledExecution

const KINDS = Dict("color-ordered-three-vector" => :three_vector, "vector-wedge-vector" => :vector_wedge,
                   "antisymmetric-tensor-vector" => :tensor_vector)

# a program's values as (slot, version): slots are reused (a source, a clear or a `set` not right
# after a clear starts a value)
function program_values(p)
    run_sources = [s for r in p["runs"] for s in r["sources"]]
    version, last = Dict{Int,Int}(), Dict{Int,Symbol}()
    cur(s) = (s, get(version, s, 0))
    start!(s) = (version[s] = get(version, s, 0) + 1)
    sources = Dict{Tuple{Int,Int},Tuple{Int,Int}}()   # value -> (label 1-based, helicity)
    rows = Dict{Tuple{Int,Int},Vector{Any}}()         # value -> [(kind, left, right, coupling × factor)]
    props, closures = Set{Tuple{Int,Int}}(), Any[]
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
            push!(props, cur(r["current"]))
        elseif k == "closure"
            cf = [complex(x["re"], x["im"]) for x in r["component_factors"]] .* complex(r["factor"]["re"], r["factor"]["im"])
            push!(closures, (cur(r["left"]), cur(r["right"]), cf))
        end
    end
    return (; sources, rows, props, closures)
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

"""
    compare(plan, export_path) -> a report (NamedTuple per sector) or an error

Every materialized colour sector of `plan` against the export's program of the same shape.
"""
function compare(plan::CompiledPlan, export_path::AbstractString)
    ex = plan.execution
    programs = JSON.parsefile(export_path)["programs"]
    cur = ex.currents
    c_sources = Dict(c.id => (c.labels[1], c.spin) for c in values(cur) if c.source)
    c_inputs = Dict{Any,Set{Vector{Any}}}()
    for i in plan.interactions
        push!(get!(c_inputs, i.current, Set{Vector{Any}}()), [i.left, i.right])
    end
    c_key = keys_from(c_sources, c_inputs, id -> cur[id].tensor)
    sectors = sort(unique(reduce(vcat, [c.sectors for c in values(cur)])))
    reports = []
    eqs = Tuple{Vector{Any},Bool}[]
    for s in sectors
        members = Set(c.id for c in values(cur) if isempty(c.sectors) || s in c.sectors)
        shape = Set(c_key[id][2] for id in members if length(c_key[id][2]) > 1)
        matching = [p for p in programs if flow_shape(p["colour_id"]) == shape]
        isempty(matching) && error("sector $s: no export program of its shape")
        p = first(matching)
        v = program_values(p)
        r_inputs = Dict{Any,Set{Vector{Any}}}(x => Set([[l, r] for (_, l, r, _) in rows]) for (x, rows) in v.rows)
        tensor_out = Set(x for (x, rows) in v.rows if any(r -> r[1] === :vector_wedge, rows))
        r_key = keys_from(v.sources, r_inputs, x -> x in tensor_out)
        by_c = Dict(c_key[id] => id for id in members)
        by_r = Dict(k => x for (x, k) in r_key)
        Set(keys(by_c)) == Set(keys(by_r)) || error("sector $s ($(p["colour_id"])): different currents")
        n = 0
        for (k, x) in by_r
            haskey(v.rows, x) || continue
            id = by_c[k]
            c_rows = [i for i in plan.interactions if i.current == id]
            r_rows = v.rows[x]
            Set(Set([c_key[i.left], c_key[i.right]]) for i in c_rows) == Set(Set([r_key[l], r_key[r]]) for (_, l, r, _) in r_rows) ||
                error("sector $s: current $(k): different interactions")
            for (kind, l, r, coef) in r_rows
                i = only(i for i in c_rows if Set([c_key[i.left], c_key[i.right]]) == Set([r_key[l], r_key[r]]))
                i.kind === kind || error("sector $s: current $k: $(i.kind) vs $kind")
                swapped = c_key[i.left] != r_key[l]
                swapped && kind === :tensor_vector && error("sector $s: a tensor on the right")
                σ = i.coefficient / (coef * (swapped ? -1 : 1))
                abs(abs(σ) - 1) < 1e-12 && abs(imag(σ)) < 1e-12 || error("sector $s: current $k: coefficient $(i.coefficient) vs $coef")
                push!(eqs, (Any[id, i.left, i.right], real(σ) < 0))
                n += 1
            end
            (id in ex.propagated) == (x in v.props) || error("sector $s: current $k: propagated in one only")
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
                        propagators = length(v.props), closures = length(c_cl)))
    end
    signs_consistent(eqs) || error("the coefficients' signs are not one per current")
    return reports
end

end
