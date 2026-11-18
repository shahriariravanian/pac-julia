# planc phase 3, step 6: a compiled plan (compiled_plan.jl) written as a recurrence export (schema 2,
# the format of `pyamplicol export-recurrence`, recurrence_export.jl), so `load_recurrence_export`
# builds its Lowered programs: the importer, its checks and every back end serve both.
#
#   programs     one per materialized colour sector (its currents: execution.json's colour support),
#                and one alias (program_reference + crossing) per sector the colour replay derives
#                from it (lc_topology_replay: a permutation of the external labels); helicity
#                outputs: every helicity configuration (the export's order), an alias's relabelled
#   storage      each current its own components (no reuse), sources first
#   rows         per current (stages in order): its interactions, the first `set`, the others `add`
#                (a UIR vertex with the measured coefficient as its coupling; a model kernel as a
#                kernel row with its binding and the coefficient as its factor), then its propagator
#                (a UIR propagator row, or a propagator kernel's row in place); the closures (the
#                roots, with their contraction × coefficient as the component factors)
#   reduction    each root's |amplitude|² into its physics_reduction group's physical helicities,
#                weight global_coupling_factor × color_factor / (average_factor × identical_factor) ×
#                the group's physical colour flows (a flow and its reverse: the same |amplitude|²)
#   momenta      a momentum slot is a form of its labels, incoming legs negated (source_fill's
#                "negate-incoming-momentum")

const COMPILED_VERTEX_NAMES = Dict(:weyl_vector_a => "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1",
                                   :weyl_vector_b => "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1",
                                   :tensor_vector => "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1",
                                   :vector_wedge => "rusticol.recurrence-intrinsic.vector-wedge-vector.v1",
                                   :three_vector => "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1")
const COMPILED_PROPAGATOR_NAMES = Dict(:weyl_a => "rusticol.recurrence-intrinsic.weyl-propagator-a.v1",
                                       :weyl_b => "rusticol.recurrence-intrinsic.weyl-propagator-b.v1",
                                       :vector => "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1")

cjson(z) = Dict("re" => real(complex(z)), "im" => imag(complex(z)))

helicity_id(h) = "h:" * join([x > 0 ? "+1" : "-1" for x in h], ",")

# every helicity configuration of n legs, the export's order (the last leg fastest, -1 first)
all_helicities(n::Int) = [[((i >> (n - k)) & 1) == 1 ? 1 : -1 for k in 1:n] for i in 0:2^n-1]

# a kernel binding's inputs by the kernel's contract (model_kernels.jl): operands' components,
# momenta (operand 0 or 1 of the row), parameters and couplings as literals
function binding_inputs(k::ModelKernel, parameters::Dict{String,Float64})
    seen = Dict{Symbol,Int}()
    out = Any[]
    lit(x) = Dict("kind" => "literal", "value" => Float64(x))
    for (role, comp, pname) in k.roles
        if role in (:left_current, :right_current, :current)
            i = seen[role] = get(seen, role, 0) + 1
            parent = role === :right_current ? 1 : 0
            for imaginary in (false, true)
                push!(out, Dict("kind" => "parent", "parent" => parent, "component" => i - 1, "imaginary" => imaginary))
            end
        elseif role in (:momentum, :left_momentum, :right_momentum)
            push!(out, Dict("kind" => "momentum", "operand" => role === :right_momentum ? 1 : 0, "component" => comp), lit(0.0))
        elseif role === :model_parameter
            haskey(parameters, pname) || throw(CompiledPlanError("kernel $(k.id): no value of $pname"))
            push!(out, lit(parameters[pname]), lit(0.0))
        elseif role === :coupling_real
            push!(out, lit(1.0), lit(0.0))
        elseif role === :coupling_imag
            push!(out, lit(0.0), lit(0.0))
        else
            throw(CompiledPlanError("kernel $(k.id): input role $role"))
        end
    end
    return out
end

binding_outputs_json(k::ModelKernel) =
    [Dict("kind" => "destination", "component" => j, "imaginary" => im, "amplitude" => false) for j in 0:k.output_arity-1 for im in (false, true)]

# the flow id of a sector derived by a label permutation (representative label -> sector label)
function permuted_flow(flow::AbstractString, perm::Dict{Int,Int})
    startswith(flow, "flow:") || return flow * "@" * join([perm[k] for k in sort(collect(keys(perm)))], ",")
    return "flow:" * join([perm[parse(Int, x)] for x in split(flow[6:end], ",")], ",")
end

"""
    compiled_export(plan; kernels) -> Dict

The plan as a recurrence export (schema 2): `JSON.json` it to a file and `load_recurrence_export` it
(src/compiled_export.jl). `kernels`: the model kernels the plan was matched with.
"""
function compiled_export(plan::CompiledPlan; kernels::Vector{ModelKernel} = ModelKernel[])::Dict{String,Any}
    ex = plan.execution
    rs = ex.raw["runtime_schema"]
    cur = ex.currents
    kernel_of = Dict(k.id => k for k in kernels)
    any(i -> i.kind === :identity, plan.interactions) && throw(CompiledPlanError("a current from one operand alone: not written yet"))
    isempty(plan.unmatched) || throw(CompiledPlanError("$(length(plan.unmatched)) terms match no vertex or kernel"))

    particles = sort(rs["external_particles"]; by = p -> p["label"])
    n = length(particles)
    initial = Set(p["label"] for p in particles if p["role"] == "initial")
    legs = [Dict("pdg" => p["pdg"], "name" => string(p["pdg"]), "role" => p["role"]) for p in particles]
    # the forms: one per momentum slot, by its labels
    slots = sort(collect(ex.momentum_slots); by = last)
    form_of = Dict{Vector{Int},Int}()
    forms = Any[]
    for (labels, _) in slots
        form_of[labels] = length(forms)
        push!(forms, [Dict("leg" => l - 1, "coefficient" => l in initial ? -1.0 : 1.0) for l in labels])
    end
    stage_of = Dict(c => k for (k, leaves) in enumerate(ex.stages) for leaf in leaves for (c, _) in leaf.outputs)
    sources = Dict(s["current_id"] => s for s in rs["source_fill"]["sources"])
    groups = ex.raw["physics_reduction"]["groups"]
    norm = rs["normalization"]
    w0 = norm["global_coupling_factor"] * norm["color_factor"] / (norm["average_factor"] * norm["identical_factor"])
    hel = all_helicities(n)
    hel_index = Dict(helicity_id(h) => i - 1 for (i, h) in enumerate(hel))
    replay = get(ex.raw["compiled"], "lc_topology_replay", nothing)
    derived = Dict{Int,Vector{Any}}()   # materialized sector -> the permutations deriving the others
    for g in (replay === nothing || !replay["enabled"] ? Any[] : replay["groups"]), perm in g["sector_permutations"]
        perm["sector_id"] == g["materialized_sector_id"] && continue
        push!(get!(derived, g["materialized_sector_id"], Any[]), perm)
    end
    used = Set{String}()
    programs = Any[]
    process_key = String(rs["process_key"])
    sectors = sort(unique(reduce(vcat, [c.sectors for c in values(cur)])))
    for s in sectors
        members = sort([c.id for c in values(cur) if isempty(c.sectors) || s in c.sectors])
        roots = [c for c in plan.closures if c.left in members && c.right in members]
        isempty(roots) && continue
        offset = Dict{Int,Int}()
        top = 0
        for id in sort(members; by = id -> (cur[id].source ? 0 : 1, get(stage_of, id, 0), id))
            offset[id] = top
            top += cur[id].dimension
        end
        form(id) = form_of[cur[id].labels]
        bindings = Any[]
        binding_id = Dict{Int,Int}()
        bind!(kid) = get!(binding_id, kid) do
            k = kernel_of[kid]
            push!(used, k.sha)
            push!(bindings, Dict("id" => length(bindings), "sha256" => k.sha, "inputs" => binding_inputs(k, ex.parameters),
                                 "outputs" => binding_outputs_json(k), "direct" => false))
            length(bindings) - 1
        end
        body = Any[Dict("kind" => "clear_amplitude", "amplitude" => a - 1) for a in eachindex(roots)]
        run_sources = Any[]
        for id in members
            cur[id].source || continue
            src = sources[id]
            ir = src["source_ir"]
            family = src["source_basis"] == "lorentz-vector" ? "vector" : src["source_basis"] == "weyl-chiral" ? "weyl" :
                     throw(CompiledPlanError("source $id: basis $(src["source_basis"])"))
            mass = ir["mass_parameter"] === nothing ? 0.0 : ex.parameters[String(ir["mass_parameter"])]
            phase = ir["crossing"]["phase"]
            push!(run_sources, Dict("current" => offset[id], "component_count" => cur[id].dimension, "family" => family,
                                    "orientation" => ir["identity"]["orientation"], "leg" => src["leg_label"] - 1,
                                    "momentum_form" => form(id), "helicity" => src["source_helicity"], "chirality" => src["chirality"],
                                    "mass" => cjson(mass), "factor" => cjson(complex(phase[1], phase[2])), "embedding" => Any[]))
            push!(body, Dict("kind" => "source", "source" => length(run_sources) - 1))
        end
        computed = sort([id for id in members if !cur[id].source]; by = id -> (get(stage_of, id, 0), id))
        for x in computed
            ints = [i for i in plan.interactions if i.current == x]
            isempty(ints) && throw(CompiledPlanError("current $x: no interaction"))
            for (k, i) in enumerate(ints)
                mode = k == 1 ? "set" : "add"
                if i.kind === :kernel
                    push!(body, Dict("kind" => "kernel", "binding" => bind!(i.kernel), "destination" => offset[x], "amplitude" => false,
                                     "left" => offset[i.left], "right" => offset[i.right], "left_momentum" => form(i.left),
                                     "right_momentum" => form(i.right), "mode" => mode, "factor" => cjson(i.coefficient)))
                else
                    push!(body, Dict("kind" => "vertex", "vertex" => COMPILED_VERTEX_NAMES[i.kind], "current" => offset[x], "mode" => mode,
                                     "left" => offset[i.left], "right" => offset[i.right], "left_momentum" => form(i.left),
                                     "right_momentum" => form(i.right), "coupling" => cjson(i.coefficient), "factor" => cjson(1.0)))
                end
            end
            if x in ex.propagated
                p = plan.propagators[x]
                if p isa Symbol
                    push!(body, Dict("kind" => "propagator", "propagator" => COMPILED_PROPAGATOR_NAMES[p], "current" => offset[x],
                                     "component_count" => cur[x].dimension, "momentum_form" => form(x), "factor" => cjson(1.0)))
                else
                    push!(body, Dict("kind" => "kernel", "binding" => bind!(p), "destination" => offset[x], "amplitude" => false,
                                     "left" => offset[x], "right" => nothing, "left_momentum" => form(x), "right_momentum" => nothing,
                                     "mode" => "set", "factor" => cjson(1.0)))
                end
            end
        end
        reduction = Any[]
        for (a, c) in enumerate(roots)
            push!(body, Dict("kind" => "closure", "amplitude" => a - 1, "left" => offset[c.left], "right" => offset[c.right],
                             "component_factors" => [cjson(f * c.coefficient) for f in c.contraction], "factor" => cjson(1.0)))
            root = ex.roots[c.root]
            g = groups[root["coherent_group_id"]+1]
            w = w0 * length(g["physical_color_ids"])
            for h in g["physical_helicity_ids"]
                push!(reduction, Dict("helicity_output" => hel_index[h], "amplitude" => a - 1, "amplitude_scale" => cjson(1.0),
                                      "apply_amplitude_scale" => true, "weight" => w))
            end
        end
        flow = String(groups[ex.roots[roots[1].root]["coherent_group_id"]+1]["representative_color_id"])
        rep = Dict{String,Any}(
            "process_key" => process_key, "process" => rs["process"], "representative_process_key" => process_key,
            "representative_colour_id" => flow, "colour_id" => flow, "external_count" => n, "legs" => legs, "caller_legs" => legs,
            "crossing" => nothing, "momentum_forms" => forms, "currents" => Any[], "current_component_count" => top,
            "amplitude_count" => length(roots), "scales" => Any[], "factors" => Any[],
            "helicity_outputs" => [Dict("id" => helicity_id(h), "helicities" => h) for h in hel],
            "body" => body, "kernel_bindings" => bindings, "runs" => [Dict("sources" => run_sources, "reduction" => reduction)],
            "group_factor" => Dict("numerator" => 1, "denominator" => 1),
            "alpha_s" => Dict("parameter" => "normalization.alpha_s_me_check", "value" => cjson(ex.parameters["alpha_s"]),
                              "squared_amplitude_power" => norm["qcd_coupling_power"]))
        push!(programs, rep)
        for perm in get(derived, s, Any[])
            σ = Dict(l["representative_label"] => l["sector_label"] for l in perm["label_permutation"])
            alias = copy(rep)
            delete!(alias, "body"); delete!(alias, "runs"); delete!(alias, "kernel_bindings")
            alias["colour_id"] = permuted_flow(flow, σ)
            alias["program_reference"] = Dict("process_key" => process_key, "colour_id" => flow)
            alias["crossing"] = [Dict("program_leg" => r - 1, "caller_leg" => σ[r] - 1, "sign" => 1.0) for r in 1:n]
            alias["helicity_outputs"] = map(hel) do h
                hc = zeros(Int, n)
                foreach(r -> hc[σ[r]] = h[r], 1:n)
                Dict("id" => helicity_id(hc), "helicities" => hc)
            end
            push!(programs, alias)
        end
    end
    kernel_files = [Dict("sha256" => k.sha, "application_base64" => base64encode(read(k.library.source)))
                    for k in kernels if k.sha in used]
    unique!(k -> k["sha256"], kernel_files)
    return Dict{String,Any}("schema_version" => EXPORT_SCHEMA, "rusticol_version" => "planc", "artifact_id" => basename(dirname(dirname(dirname(ex.path)))),
                            "execution_semantics" => EXPORT_SEMANTICS,
                            "parameters" => [Dict("name" => "normalization.alpha_s_me_check", "value" => cjson(ex.parameters["alpha_s"]))],
                            "programs" => programs, "kernels" => kernel_files)
end
