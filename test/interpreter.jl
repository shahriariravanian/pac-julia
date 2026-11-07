# A minimal interpreter of a main.json program on string-keyed values (the reference of the tests):
# run_program(prog, kernels, caller) with prog a parsed program (minijson.jl), kernels the
# KernelLibrary of each kernel name, caller[leg] the caller's momenta. Needs wavefunctions.jl and
# operations.jl.
using PacRunner
using PacRunner: Source, apply_source, wavefunction, vertex!, propagator, scale!, clear!, closure, kernel!, reduction,
    KernelLibrary, load_kernel
import PacRunner: get_vector, get_scale, get_factor

# one program's values at one point: the momentum forms, the scales, the factors
struct ProgramArena
    forms::Dict{String,NTuple{4,Float64}}
    scales::Dict{String,Complex{Float64}}
    factors::Dict{String,Complex{Float64}}
end
get_vector(a::ProgramArena, name) = a.forms[name]
get_scale(a::ProgramArena, name) = a.scales[name]
get_factor(a::ProgramArena, name) = a.factors[name]

cplx(v) = complex(Float64(v[1]), Float64(v[2]))

# p<l> from the caller's momenta (inputs.crossing), then q<i> = Σ coefficient × p<leg>, the first
# term added to 0.0, in term order
function forms_of(inputs, caller)
    legs = inputs["legs"]
    crossing = inputs["crossing"]
    p = [crossing === nothing ? caller[l+1] : Float64(crossing[l+1]["sign"]) .* caller[crossing[l+1]["leg"]+1] for l in 0:legs-1]
    forms = Dict{String,NTuple{4,Float64}}()
    for f in inputs["forms"]
        q = zeros(4)
        for (t, (leg, c)) in enumerate(f["terms"])
            for k in 1:4
                term = Float64(c) * p[leg+1][k]
                q[k] = t == 1 ? 0.0 + term : q[k] + term
            end
        end
        forms[f["name"]] = Tuple(q)
    end
    return forms
end

function run_program(prog, kernels, caller)
    inputs = prog["inputs"]
    arena = ProgramArena(forms_of(inputs, caller),
                         Dict(s["name"] => cplx(s["value"]) for s in inputs["scales"]),
                         Dict(x["name"] => cplx(x["value"]) for x in inputs["factors"]))
    currents = Dict{String,Vector{Complex{Float64}}}()
    for c in prog["currents"]
        init = c["initial"]
        if init == "source"
            s = c["source"]
            src = Source(s["wavefunction"], s["family"], s["orientation"], s["leg"], s["momentum"],
                         s["helicity"], s["chirality"], s["mass"], s["factor"])
            currents[c["name"]] = apply_source(src, arena)
        elseif init == "zero" || init == "written"
            currents[c["name"]] = zeros(Complex{Float64}, c["components"])
        else
            error("current $(c["name"]): initial $init")
        end
    end
    amplitudes = Dict(a["name"] => complex(0.0, 0.0) for a in prog["amplitudes"])
    bindings = Dict(b["name"] => b for b in prog["bindings"])
    q(name) = get_vector(arena, name)
    operand(name) = name === nothing ? Complex{Float64}[] : currents[name]  # an unused operand is null
    for op in prog["body"]
        kind = op["op"]
        if kind == "vertex"
            (l, r) = (operand(op["in"][1]), operand(op["in"][2]))
            vertex!(currents[op["out"]], op["type"], op["mode"], l, r, q(op["momenta"][1]), q(op["momenta"][2]),
                    get_scale(arena, op["coupling"]), get_factor(arena, op["factor"]))
        elseif kind == "propagator"
            c = currents[op["current"]]
            v = propagator(op["type"], c, q(op["momentum"]), get_factor(arena, op["factor"]))
            c[1:length(v)] = v
        elseif kind == "scale"
            scale!(currents[op["current"]], op["components"], get_factor(arena, op["factor"]))
        elseif kind == "clear"
            clear!(currents[op["current"]], op["components"])
        elseif kind == "closure"
            (l, r) = (operand(op["in"][1]), operand(op["in"][2]))
            cf = [get_factor(arena, x) for x in op["component_factors"]]
            amplitudes[op["amplitude"]] = closure(amplitudes[op["amplitude"]], l, r, cf, get_factor(arena, op["factor"]))
        elseif kind == "kernel"
            b = bindings[op["binding"]]
            (l, r) = (operand(op["in"][1]), operand(op["in"][2]))
            kernel!(currents[op["out"]], kernels[b["kernel"]], b["inputs"], b["outputs"], b["direct"], op["mode"],
                    l, r, q(op["momenta"][1]), q(op["momenta"][2]), get_factor(arena, op["factor"]), arena)
        else
            error("unknown instruction $kind")
        end
    end
    gf = prog["output"]["group_factor"]
    return [reduction([amplitudes[a] for (a, _) in red["terms"]], [Float64(w) for (_, w) in red["terms"]], (gf[1], gf[2]))
            for red in prog["reduction"]]
end

