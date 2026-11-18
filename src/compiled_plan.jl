# planc phase 2: the recurrence plan of a compiled-mode artifact of pyAmpliCol's cpp backend, from
# its leaves (direct_cpp.jl) and processes/<p>/execution.json.
#
# execution.json (its `helicity_sum_execution`: every helicity of the materialized colour sectors)
# describes the currents (external labels, particle, spin state of a source, `auxiliary_kind` of the
# four-gluon contact's tensor, colour support `lc-sector-support:i[,j]`, components in the value
# arena), the momentum slots (sums of external momenta), the stages (stage k: the currents of k + 1
# legs), each leaf's inputs (a value component or a momentum component; `real_valued`) and outputs,
# and the amplitude roots (the closures: two currents and a contraction). It does not say which
# currents combine into which ("interactions_compacted"): that is in the leaves' arithmetic.
#
# `compiled_plan(read_compiled(artifact))` recovers it:
#   - interactions: an output current's terms (the monomials of its components, as products of input
#     currents) name the sets of currents it combines: two per term here, a vertex;
#   - kinds: from the types, vector × vector → vector (the colour-ordered three-gluon vertex,
#     operations.jl `vertex_three_vector`), vector × vector → tensor (`vertex_vector_wedge`), tensor ×
#     vector → vector (`vertex_tensor_vector`);
#   - coefficients: the leaf evaluated with only the two inputs nonzero (random) and random momenta,
#     divided by the bare vertex (× −i/k² for a propagated output, the Feynman-gauge vector
#     propagator): one constant over components and samples, or an error;
#   - closures: the amplitude leaves against Σⱼ lⱼ rⱼ cⱼ (the root's contraction coefficients).
# Orientation: a tensor is the left input; of two vectors, the one with the lower first external
# label (the C++ builds currents in ascending label order). The coefficient is measured in that
# orientation, so it includes whatever sign the orientation brings.

struct CompiledPlanError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CompiledPlanError) = print(io, "compiled plan: ", e.msg)

struct CompiledCurrent
    id::Int
    labels::Vector{Int}      # external labels, 1-based, sorted
    particle::Int            # PDG id
    spin::Int                # a source's spin state (helicity); 0 otherwise
    source::Bool
    tensor::Bool             # the antisymmetric tensor of the four-gluon contact (auxiliary_kind)
    dimension::Int
    components::UnitRange{Int}  # in the value arena, 0-based
    sectors::Vector{Int}     # the materialized colour sectors it serves (empty: a source, all)
end

struct CompiledInput          # one leaf input
    kind::Symbol              # :current, :momentum, :other
    id::Int                   # current id or momentum slot id
    component::Int            # 0-based
    real::Bool
end

struct CompiledLeaf
    kernel::DirectKernel
    inputs::Vector{CompiledInput}
    planes::Vector{UnitRange{Int}}   # input i's planes (1-based; 2 for a complex input, 1 for a real one)
    outputs::Vector{Tuple{Int,Int}}  # complex output j: (current id, component) or (root index, 0) in the amplitude stage
end

struct CompiledExecution
    path::String              # execution.json
    process::String
    currents::Dict{Int,CompiledCurrent}
    momentum_slots::Dict{Vector{Int},Int}  # sorted external labels -> slot id
    propagated::Set{Int}      # the currents a stage stores propagated
    stages::Vector{Vector{CompiledLeaf}}
    amplitude::Vector{CompiledLeaf}
    roots::Vector{Any}        # the amplitude roots, as execution.json has them
    raw::Dict{String,Any}     # the execution's JSON
end

struct CompiledInteraction
    current::Int
    kind::Symbol              # :three_vector, :vector_wedge, :tensor_vector
    left::Int
    right::Int
    coefficient::ComplexF64   # coupling × factor of the recurrence row, in this orientation
end

struct CompiledClosure
    root::Int                 # 1-based, in execution.json's order
    left::Int
    right::Int
    contraction::Vector{ComplexF64}
    coefficient::ComplexF64
end

struct CompiledPlan
    execution::CompiledExecution
    interactions::Vector{CompiledInteraction}
    closures::Vector{CompiledClosure}
end

# the probes' numbers: SplitMix64, uniform in [-1, 1) (reproducible, no dependency)
mutable struct ProbeRng
    state::UInt64
end
function next!(g::ProbeRng)::Float64
    g.state += 0x9e3779b97f4a7c15
    z = g.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻= z >> 31
    return 2.0 * ((z >> 11) * 2.0^-53) - 1.0
end

# ---- reading -------------------------------------------------------------------------------------

function sector_support(c)
    keys = c["color_state"]["basis_key"]
    isempty(keys) && return Int[]
    return sort!([parse(Int, x) for k in keys for x in split(split(k, ":")[2], ",")])
end

function read_leaf(dir::AbstractString, leaf, bindings, value_count::Int, component_current, mom_slot, out_at)
    p = joinpath(dir, leaf["application_path"])
    cpp = joinpath(dirname(p), replace(basename(p), r"^lib" => "") * ".cpp")
    isfile(cpp) || throw(CompiledPlanError("$cpp: no C++ (an artifact of the cpp backend: --backend cpp)"))
    k = read_direct_cpp(cpp)
    inputs, planes, plane = CompiledInput[], UnitRange{Int}[], 1
    for i in leaf["input_indices"]
        b = bindings[i]
        real = b["real_valued"]
        g = b["global_component"]
        input = if b["kind"] == "value"
            (cur, comp) = component_current[g]
            CompiledInput(:current, cur, comp, real)
        elseif b["kind"] == "momentum"
            (slot, comp) = mom_slot[g]
            CompiledInput(:momentum, slot, comp, real)
        else
            CompiledInput(:other, -1, 0, real)
        end
        push!(inputs, input)
        push!(planes, plane:plane+(real ? 0 : 1))
        plane += real ? 1 : 2
    end
    plane - 1 == k.input_planes || throw(CompiledPlanError("$cpp: $(plane - 1) input planes from the bindings, $(k.input_planes) in the C++"))
    outputs = [out_at[leaf["output_start"]+j] for j in 0:length(k.outputs)-1]
    return CompiledLeaf(k, inputs, planes, outputs)
end

"""
    read_compiled(artifact; process = nothing) -> CompiledExecution

The helicity-sum execution of a compiled-mode artifact of pyAmpliCol's cpp backend (`process`: its
key, default the only one): its currents, momentum slots and leaves (src/compiled_plan.jl).
"""
function read_compiled(artifact::AbstractString; process::Union{Nothing,AbstractString} = nothing)::CompiledExecution
    procs = joinpath(artifact, "processes")
    isdir(procs) || throw(CompiledPlanError("$artifact: not a pyAmpliCol artifact (no processes/)"))
    keys_ = sort([d for d in readdir(procs) if isfile(joinpath(procs, d, "execution.json"))])
    if process === nothing
        length(keys_) == 1 || throw(CompiledPlanError("$artifact: processes $keys_: pick one (process = ...)"))
        process = keys_[1]
    end
    process in keys_ || throw(CompiledPlanError("$artifact: no process $(repr(process)) (there are $keys_)"))
    path = joinpath(procs, process, "execution.json")
    top = JSON.parsefile(path)
    ex = top["helicity_sum_execution"]
    rs = ex["runtime_schema"]
    currents = Dict{Int,CompiledCurrent}()
    component_current = Dict{Int,Tuple{Int,Int}}()
    for c in rs["current_storage"]["current_slots"]
        id = c["current_id"]
        comps = c["component_start"]:c["component_stop"]-1
        currents[id] = CompiledCurrent(id, sort(Int.(c["external_labels"])), c["particle_id"], c["spin_state"], c["is_source"],
                                       c["auxiliary_kind"] !== nothing, c["dimension"], comps, sector_support(c))
        foreach(g -> component_current[g] = (id, g - first(comps)), comps)
    end
    value_count = rs["parameter_layout"]["value_component_count"]
    mom_slot = Dict{Int,Tuple{Int,Int}}()
    momentum_slots = Dict{Vector{Int},Int}()
    for m in rs["momentum_slots"]
        momentum_slots[sort(Int.(m["external_labels"]))] = m["momentum_slot_id"]
        foreach(k -> mom_slot[value_count+k] = (m["momentum_slot_id"], k - m["component_start"]), m["component_start"]:m["component_stop"]-1)
    end
    dir = joinpath(procs, process)
    stage_ev = ex["compiled"]["stage_evaluators"]
    propagated = Set{Int}()
    stages = map(stage_ev["stages"]) do stage
        arena = stage["compiled_plane_arena"]
        bindings = Dict(b["parameter_index"] => b for b in arena["input_bindings"])
        out_at = Dict{Int,Tuple{Int,Int}}()
        for s in stage["output_slots"]
            s["variant"] == "propagated" && push!(propagated, s["current_id"])
            foreach(j -> out_at[j] = (s["current_id"], j - s["output_start"]), s["output_start"]:s["output_stop"]-1)
        end
        [read_leaf(dir, leaf, bindings, value_count, component_current, mom_slot, out_at) for leaf in arena["leaves"]]
    end
    roots = rs["amplitude_stage"]["roots"]
    amp = stage_ev["amplitude_stage"]["compiled_plane_arena"]
    bindings = Dict(b["parameter_index"] => b for b in amp["input_bindings"])
    out_at = Dict(r["output_index"] => (i, 0) for (i, r) in enumerate(roots))
    amplitude = [read_leaf(dir, leaf, bindings, value_count, component_current, mom_slot, out_at) for leaf in amp["leaves"]]
    return CompiledExecution(path, process, currents, momentum_slots, propagated, stages, amplitude, roots, ex)
end

# ---- interactions: the sets of input currents an output current's terms multiply ------------------

# a node's monomials, each as the sorted current ids it multiplies (momenta and constants: none)
function monomials(leaf::CompiledLeaf)
    plane_input = Dict(p => i for (i, r) in enumerate(leaf.planes) for p in r)
    m = Vector{Set{Vector{Int}}}(undef, length(leaf.kernel.nodes))
    one = Set([Int[]])
    for (i, n) in enumerate(leaf.kernel.nodes)
        m[i] = if n.op === :complex_plane || n.op === :real_plane
            input = leaf.inputs[plane_input[n.args[1]+1]]
            input.kind === :current ? Set([[input.id]]) : one
        elseif n.op === :constant
            one
        elseif n.op === :add
            union(m[n.args[1]], m[n.args[2]])
        elseif n.op === :multiply
            Set(sort!(unique!(vcat(a, b))) for a in m[n.args[1]] for b in m[n.args[2]])
        else  # a reciprocal: of momenta (a propagator's k²); its currents, if any, as one factor
            Set([sort!(unique!(reduce(vcat, collect(m[n.args[1]]); init = Int[])))])
        end
    end
    return m
end

# the measurement's inputs: the given currents' components random, momenta random, the rest zero
function probe_planes(rng, leaf::CompiledLeaf, J::Dict{Int,Vector{ComplexF64}}, P::Dict{Int,Vector{Float64}})
    planes = [zeros(1) for _ in 1:leaf.kernel.input_planes]
    for (input, r) in zip(leaf.inputs, leaf.planes)
        z = input.kind === :current && haskey(J, input.id) ? J[input.id][input.component+1] :
            input.kind === :momentum ? complex(P[input.id][input.component+1]) : 0.0im
        planes[first(r)][1] = real(z)
        length(r) == 2 && (planes[last(r)][1] = imag(z))
    end
    return planes
end

function bare_vertex(kind::Symbol, l, r, u, v)
    kind === :three_vector && return vertex_three_vector(l, r, u, v)
    kind === :vector_wedge && return vertex_vector_wedge(l, r)
    return vertex_tensor_vector(l, r)
end

# the one constant c with kernel = c × reference over the components and `samples` samples
function measure(f_kernel, f_reference, samples::Int, what)
    ratios = ComplexF64[]
    for s in 1:samples
        (got, ref) = (f_kernel(s), f_reference(s))
        scale = maximum(abs, ref)
        append!(ratios, [g / r for (g, r) in zip(got, ref) if abs(r) > 1e-9 * scale])
    end
    isempty(ratios) && throw(CompiledPlanError("$what: no component to compare"))
    c = ratios[1]
    all(r -> abs(r - c) <= 1e-9 * max(1.0, abs(c)), ratios) ||
        throw(CompiledPlanError("$what: not one constant times the bare form (ratios $(unique(round.(ratios; sigdigits = 6))[1:min(end, 4)]))"))
    return c
end

"""
    compiled_plan(exec::CompiledExecution; samples = 3, rng = ProbeRng(1)) -> CompiledPlan

The interactions (vertex kind, inputs, coefficient) of every computed current and the closures of
the amplitude roots (src/compiled_plan.jl).
"""
function compiled_plan(exec::CompiledExecution; samples::Int = 3, rng::ProbeRng = ProbeRng(1))::CompiledPlan
    cur = exec.currents
    randJ(c) = [complex(next!(rng), next!(rng)) for _ in 1:cur[c].dimension]
    randP() = Dict(s => [next!(rng) for _ in 1:4] for s in values(exec.momentum_slots))
    interactions = CompiledInteraction[]
    for leaves in exec.stages, leaf in leaves
        m = monomials(leaf)
        outs = Dict{Int,Vector{Tuple{Int,Int}}}()   # current -> (complex output j, component)
        for (j, (c, comp)) in enumerate(leaf.outputs)
            push!(get!(outs, c, Tuple{Int,Int}[]), (j, comp))
        end
        for (x, comps) in sort!(collect(outs))
            length(comps) == cur[x].dimension ||
                throw(CompiledPlanError("current $x: $(length(comps)) of its $(cur[x].dimension) components in one leaf"))
            sets = sort!(collect(Set(s for (j, _) in comps for s in m[leaf.kernel.outputs[j]] if !isempty(s))))
            for s in sets
                length(s) == 2 || throw(CompiledPlanError("current $x: a term of $(length(s)) currents $s (vertices of two inputs only)"))
                (a, b) = (cur[s[1]], cur[s[2]])
                kind, l, r = if a.tensor != b.tensor
                    (:tensor_vector, a.tensor ? a : b, a.tensor ? b : a)
                elseif !a.tensor && !b.tensor
                    (cur[x].tensor ? :vector_wedge : :three_vector, first(a.labels) < first(b.labels) ? a : b, first(a.labels) < first(b.labels) ? b : a)
                else
                    throw(CompiledPlanError("current $x: two tensors $(a.id), $(b.id)"))
                end
                (kind === :tensor_vector && cur[x].tensor) && throw(CompiledPlanError("current $x: tensor × vector → tensor"))
                k = exec.momentum_slots[cur[x].labels]
                propagated = x in exec.propagated
                probes = [(Dict(l.id => randJ(l.id), r.id => randJ(r.id)), randP()) for _ in 1:samples]
                kernel(s) = begin
                    (J, P) = probes[s]
                    o = evaluate_direct(leaf.kernel, probe_planes(rng, leaf, J, P))
                    [complex(o[2j-1][1], o[2j][1]) for (j, _) in sort(comps; by = last)]
                end
                reference(s) = begin
                    (J, P) = probes[s]
                    u, v = P[exec.momentum_slots[l.labels]], P[exec.momentum_slots[r.labels]]
                    o = bare_vertex(kind, J[l.id], J[r.id], u, v)
                    propagated || return o
                    q = P[k]
                    o .* (-1im / (q[1]^2 - q[2]^2 - q[3]^2 - q[4]^2))
                end
                c = measure(kernel, reference, samples, "current $x from $(l.id), $(r.id) ($kind)")
                push!(interactions, CompiledInteraction(x, kind, l.id, r.id, c))
            end
        end
    end
    closures = CompiledClosure[]
    for leaf in exec.amplitude
        for (j, (root, _)) in enumerate(leaf.outputs)
            rt = exec.roots[root]
            (l, r) = (rt["left_current_id"], rt["right_current_id"])
            cf = [complex(x[1], x[2]) for x in rt["contraction_ir"]["coefficients"]]
            probes = [Dict(l => randJ(l), r => randJ(r)) for _ in 1:samples]
            kernel(s) = (o = evaluate_direct(leaf.kernel, probe_planes(rng, leaf, probes[s], randP())); [complex(o[2j-1][1], o[2j][1])])
            reference(s) = [sum(probes[s][l][c] * probes[s][r][c] * cf[c] for c in eachindex(cf))]
            push!(closures, CompiledClosure(root, l, r, cf, measure(kernel, reference, samples, "root $root")))
        end
    end
    return CompiledPlan(exec, interactions, closures)
end
