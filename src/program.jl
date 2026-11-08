# The loader of main.json (pac2json's programs): every name resolved to an index (1-based), into a
# Program (types.jl). evaluate(::Program) is evaluators/progeval.jl.

# "c12" → 13 (1-based), nothing → 0 (a null operand)
index(name::Nothing, prefix::String) = 0
function index(name::AbstractString, prefix::String)::Int
    startswith(name, prefix) || error("$name is not a $prefix<i>")
    return parse(Int, name[length(prefix)+1:end]) + 1
end

ci(name)::CurrentIndex = index(name, "c")
qi(name)::FormIndex = index(name, "q")
ai(name)::AmplitudeIndex = index(name, "a")
si(name)::ScaleIndex = index(name, "s")
xi(name)::FactorIndex = index(name, "x")
bi(name)::BindingIndex = index(name, "b")

cplx(v) = complex(Float64(v[1]), Float64(v[2]))

#################################### Form ########################################

load_forms(prog) = [Form([(LegIndex(t[1]) + 1, Float64(t[2])) for t in f.terms]) for f in prog.inputs.forms]

################################### Current #########################################

function load_currents(prog)
    currents = Current[]
    for (i, c) in enumerate(prog.currents)
        ci(c.name) == i || error("currents[$i] is $(c.name)")
        init = c.initial
        if init == "source"
            s = c.source
            src = CurrentSource(s.family, s.orientation, qi(s.momentum), s.helicity, s.chirality, si(s.mass), xi(s.factor))
            push!(currents, Current(c.components, src))
        elseif init == "zero" || init == "written"  # written: the body's first write is "="
            push!(currents, Current(c.components, nothing))
        else
            error("current $(c.name): initial \"$init\" is not supported")
        end
    end
    return currents
end

###################################### Kernel #########################################

function load_input(code::AbstractString)::KernelInput
    if (m = match(r"^(re|im) in\[([01])\]\[(\d+)\]$", code)) !== nothing
        return KernelInput(m[2] == "0" ? IN_L : IN_R, parse(ComponentIndex, m[3]) + 1, m[1] == "im")
    elseif (m = match(r"^(re|im) out\[(\d+)\]$", code)) !== nothing
        return KernelInput(OUT, parse(ComponentIndex, m[2]) + 1, m[1] == "im")
    elseif (m = match(r"^momenta\[([01])\]\[(\d+)\]$", code)) !== nothing
        return KernelInput(m[1] == "0" ? MOMENTUM_U : MOMENTUM_V, parse(ComponentIndex, m[2]) + 1, false)
    elseif (m = match(r"^s(\d+)$", code)) !== nothing
        return KernelInput(SCALE, parse(ScaleIndex, m[1]) + 1, false)
    elseif (m = match(r"^(re|im) factor$", code)) !== nothing
        return KernelInput(FACTOR, 0, m[1] == "im")
    else
        error("unknown binding input $code")
    end
end

# the kernels of main.json, by name: each with its .plane.symjit (the compiled runners' kernels) and
# its shared library (pac2json --sjb2obj; the Julia runner's)
function load_kernels(main, path)
    kernels = Dict{String,KernelLibrary}()
    for k in main.kernels
        lib = k.library
        lib === nothing && error("kernel $(k.name) has no library: run pac2json with --sjb2obj")
        kernels[k.name] = load_kernel(joinpath(dirname(path), lib.path), lib.symbol, lib.states, lib.outputs;
                                      source = joinpath(dirname(path), k.path))
    end
    return kernels
end

function load_bindings(prog, kernels)
    bindings = Binding[]
    for (i, b) in enumerate(prog.bindings)
        bi(b.name) == i || error("bindings[$i] is $(b.name)")
        push!(bindings, Binding(kernels[b.kernel], b.direct, [load_input(c) for c in b.inputs],
                              [ComponentIndex(j) + 1 for j in b.outputs]))
    end
    return bindings
end

##################################### Operations ######################################



function load_operation(o)::Operation
    op = o.op
    if op == "vertex"
        return Vertex(o.type, ci(o.out), o.mode, ci(o.in[1]), ci(o.in[2]), qi(o.momenta[1]), qi(o.momenta[2]),
                      si(o.coupling), xi(o.factor))
    elseif op == "propagator"
        return Propagator(o.type, ci(o.current), qi(o.momentum), xi(o.factor))
    elseif op == "scale"
        return Scale(ci(o.current), o.components, xi(o.factor))
    elseif op == "clear"
        return Clear(ci(o.current), o.components)
    elseif op == "closure"
        return Closure(ai(o.amplitude), ci(o.in[1]), ci(o.in[2]), [xi(f) for f in o.component_factors], xi(o.factor))
    elseif op == "kernel"
        return Kernel(bi(o.binding), ci(o.out), o.mode, ci(o.in[1]), ci(o.in[2]), qi(o.momenta[1]), qi(o.momenta[2]),
                      xi(o.factor))
    else
        error("unknown instruction $op")
    end
end

load_body(prog)::Vector{Operation} = [load_operation(o) for o in prog.body]

###################################### Program ########################################

count_legs(p::Program) = p.legs

function load_program(prog, kernels)::Program
    inputs = prog.inputs
    crossing = inputs.crossing === nothing ? nothing : [(LegIndex(c.leg) + 1, Float64(c.sign)) for c in inputs.crossing]
    for (k, s) in enumerate(inputs.scales)
        si(s.name) == k || error("scales[$k] is $(s.name)")
    end
    for (k, f) in enumerate(inputs.factors)
        xi(f.name) == k || error("factors[$k] is $(f.name)")
    end
    for (k, a) in enumerate(prog.amplitudes)
        ai(a.name) == k && a.initial == 0 || error("amplitudes[$k] is $(a.name), initial $(a.initial)")
    end
    reduction = Vector{Tuple{AmplitudeIndex,Float64}}[]
    for (h, r) in enumerate(prog.reduction)
        r.output == h - 1 || error("reduction[$h] is output $(r.output)")
        push!(reduction, [(ai(t[1]), Float64(t[2])) for t in r.terms])
    end
    out = prog.output
    alpha_s = inputs.alpha_s === nothing ? nothing : (Float64(inputs.alpha_s.value), Int(inputs.alpha_s.power))
    return Program(
        prog.process,
        prog.colour,
        prog.expression,
        inputs.legs,
        crossing,
        load_forms(prog),
        [cplx(s.value) for s in inputs.scales],
        [cplx(f.value) for f in inputs.factors],
        alpha_s,
        [h.legs === nothing ? Helicity[] : Helicity.(h.legs) for h in out.helicities],
        load_currents(prog),
        length(prog.amplitudes),
        load_bindings(prog, kernels),
        load_body(prog),
        reduction,
        (Int(out.group_factor[1]), Int(out.group_factor[2])),
    )
end

read_main(path::String) = JSON.parse(read(path, String))

# program `program` (1-based) of main.json at `path`
function load_artifact(path::String, program::Int = 1)::Program
    main = read_main(path)
    return load_program(main.programs[program], load_kernels(main, path))
end

# program `program`'s entry of examples.json at `path` (process, colour, points: momenta, outputs),
# for compile's testing.c
load_examples(path::String, program::Int = 1) = read_main(path).programs[program]

# every program of main.json at `path`
function load_programs(path::String)::Vector{Program}
    main = read_main(path)
    kernels = load_kernels(main, path)
    return [load_program(prog, kernels) for prog in main.programs]
end
