# The checks of a Program at load time (`check_program`, called by load_program), before anything
# runs:
#   - every index in range: currents, forms, scales, factors, amplitudes, bindings, components;
#   - the kind of every current (CurrentKind): from its source, or from the instructions that
#     write and read it, which must agree, and with its number of components;
#   - every instruction's operands of the kinds the instruction takes (vertex and propagator
#     types); closures, scales, clears and kernels within their operands' components;
#   - no current read before it has a value: a current with initial "written" gets its value from
#     its first write, which must have mode "=".
# A failure is an error naming the instruction (body[n], 1-based) and the currents (main.json's
# names).

# the operand and output kinds of each vertex type: (in[0], in[1]) → out
const VERTEX_KINDS = Dict(
    "weyl_vector_a" => (WEYL, VECTOR, WEYL),
    "weyl_vector_b" => (WEYL, VECTOR, WEYL),
    "tensor_vector" => (TENSOR, VECTOR, VECTOR),
    "vector_wedge" => (VECTOR, VECTOR, TENSOR),
    "three_vector" => (VECTOR, VECTOR, VECTOR),
)

# the current kind of each propagator type
const PROPAGATOR_KINDS = Dict("weyl_a" => WEYL, "weyl_b" => WEYL, "vector" => VECTOR)

const SOURCE_KINDS = Dict("scalar" => SCALAR, "weyl" => WEYL, "dirac" => DIRAC, "vector" => VECTOR)

# the kind of a current no typed instruction constrains (a kernel's operand or output, a closure's
# operand), from its components; 4 is a vector (a Dirac current is a source)
const KIND_OF_COMPONENTS = Dict(1 => SCALAR, 2 => WEYL, 4 => VECTOR, 6 => TENSOR)

struct ProgramError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ProgramError) = print(io, "invalid program: ", e.msg)

current_name(i::CurrentIndex) = "c$(i - 1)"

# The kinds of the currents, after checking the program; a ProgramError otherwise.
function check_program(currents::Vector{CurrentSpec}, body::Vector{Operation}, bindings::Vector{Binding};
                       forms::Int, scales::Int, factors::Int, amplitudes::Int)::Vector{CurrentKind}
    n = length(currents)
    kinds = Vector{Union{Nothing,CurrentKind}}(nothing, n)
    why = fill("", n)  # what set each kind, for the messages
    fail(msg) = throw(ProgramError(msg))

    function require_kind(i::CurrentIndex, k::CurrentKind, at::String)
        if kinds[i] === nothing
            components(k) == currents[i].components ||
                fail("$at: $(current_name(i)) has $(currents[i].components) components, not those of a $(kind_name(k)) ($(components(k)))")
            kinds[i] = k
            why[i] = at
        elseif kinds[i] != k
            fail("$at: $(current_name(i)) is a $(kind_name(kinds[i])) ($(why[i])), not a $(kind_name(k))")
        end
    end

    for (i, c) in enumerate(currents)
        if c.source !== nothing
            haskey(SOURCE_KINDS, c.source.family) || fail("$(current_name(i)): unknown source family $(c.source.family)")
            require_kind(i, SOURCE_KINDS[c.source.family], "its source")
            in_range(c.source.form, forms) || fail("$(current_name(i)): source momentum form $(c.source.form) of $forms")
            in_range(c.source.mass, scales) || fail("$(current_name(i)): source mass scale $(c.source.mass) of $scales")
            in_range(c.source.factor, factors) || fail("$(current_name(i)): source factor $(c.source.factor) of $factors")
        end
    end

    defined = [c.initially for c in currents]

    for (step, op) in enumerate(body)
        at = "body[$step] ($(describe(op)))"
        check_indices(op, at, n; forms, scales, factors, amplitudes, bindings = length(bindings))
        needs_value(i::CurrentIndex) = defined[i] || fail("$at: $(current_name(i)) is read before it has a value")
        function assigns(i::CurrentIndex, mode::String)
            mode == "=" || mode == "+=" || fail("$at: mode \"$mode\"")
            mode == "+=" && needs_value(i)
            defined[i] = true
        end
        within(i::CurrentIndex, j::Int, what::String) =
            j <= currents[i].components || fail("$at: $what reads component $j of $(current_name(i)), which has $(currents[i].components)")

        if op isa Vertex
            haskey(VERTEX_KINDS, op.type) || fail("$at: unknown vertex type $(op.type)")
            (kl, kr, ko) = VERTEX_KINDS[op.type]
            (op.l > 0 && op.r > 0) || fail("$at: a vertex needs both operands")
            require_kind(op.l, kl, at)
            require_kind(op.r, kr, at)
            require_kind(op.out, ko, at)
            needs_value(op.l)
            needs_value(op.r)
            assigns(op.out, op.mode)
        elseif op isa Propagator
            haskey(PROPAGATOR_KINDS, op.type) || fail("$at: unknown propagator type $(op.type)")
            require_kind(op.current, PROPAGATOR_KINDS[op.type], at)
            needs_value(op.current)
        elseif op isa Scale
            within(op.current, op.components, "scale")
            needs_value(op.current)
        elseif op isa Clear
            within(op.current, op.components, "clear")
            defined[op.current] = true
        elseif op isa Closure
            (op.l > 0 && op.r > 0) || fail("$at: a closure needs both operands")
            m = length(op.component_factors)
            within(op.l, m, "the closure")
            within(op.r, m, "the closure")
            needs_value(op.l)
            needs_value(op.r)
        elseif op isa Kernel
            b = bindings[op.binding]
            length(b.inputs) == b.kernel.inputs ||
                fail("$at: binding b$(op.binding - 1) has $(length(b.inputs)) inputs, its kernel takes $(b.kernel.inputs)")
            2 * length(b.outputs) == b.kernel.outputs ||
                fail("$at: binding b$(op.binding - 1) has $(length(b.outputs)) output pairs, its kernel gives $(b.kernel.outputs ÷ 2)")
            for x in b.inputs
                if x.kind == IN_L || x.kind == IN_R || x.kind == OUT
                    i = x.kind == IN_L ? op.l : x.kind == IN_R ? op.r : op.out
                    i > 0 || fail("$at: the kernel reads in[$(x.kind == IN_L ? 0 : 1)], which is null")
                    within(i, x.n, "the kernel")
                    needs_value(i)
                elseif x.kind == MOMENTUM_U || x.kind == MOMENTUM_V
                    1 <= x.n <= 4 || fail("$at: momentum component $(x.n)")
                elseif x.kind == SCALE
                    in_range(x.n, scales) || fail("$at: kernel input scale $(x.n) of $scales")
                end
            end
            for j in b.outputs
                within(op.out, j, "the kernel output")
            end
            allunique(b.outputs) || fail("$at: binding b$(op.binding - 1) writes a component twice")
            # all components written ("="): the current has a value
            if op.mode == "=" && sort(b.outputs) != collect(1:currents[op.out].components)
                needs_value(op.out)
            end
            assigns(op.out, op.mode)
        end
    end

    # the currents no typed instruction constrains: by their components
    for (i, c) in enumerate(currents)
        if kinds[i] === nothing
            haskey(KIND_OF_COMPONENTS, c.components) || fail("$(current_name(i)) has $(c.components) components")
            kinds[i] = KIND_OF_COMPONENTS[c.components]
        end
    end
    return CurrentKind[k for k in kinds]
end

in_range(i::Int, n::Int) = 1 <= i <= n

# every index of an instruction within the program's lists (0 only for a null operand)
function check_indices(op::Operation, at::String, currents::Int; forms, scales, factors, amplitudes, bindings)
    fail(what, i, n) = throw(ProgramError("$at: $what $i of $n"))
    cur(i, null_ok = false) = (null_ok && i == 0) || in_range(i, currents) || fail("current", i, currents)
    frm(i) = in_range(i, forms) || fail("momentum form", i, forms)
    fac(i) = in_range(i, factors) || fail("factor", i, factors)
    if op isa Vertex
        cur(op.out); cur(op.l, true); cur(op.r, true); frm(op.u); frm(op.v); fac(op.factor)
        in_range(op.coupling, scales) || fail("scale", op.coupling, scales)
    elseif op isa Propagator
        cur(op.current); frm(op.momentum); fac(op.factor)
    elseif op isa Scale
        cur(op.current); fac(op.factor)
    elseif op isa Clear
        cur(op.current)
    elseif op isa Closure
        cur(op.l, true); cur(op.r, true); fac(op.factor); foreach(fac, op.component_factors)
        in_range(op.amplitude, amplitudes) || fail("amplitude", op.amplitude, amplitudes)
    elseif op isa Kernel
        cur(op.out); cur(op.l, true); cur(op.r, true); frm(op.u); frm(op.v); fac(op.factor)
        in_range(op.binding, bindings) || fail("binding", op.binding, bindings)
    end
end

# an instruction for the messages, in main.json's names
describe(op::Vertex) = "vertex $(op.type) $(current_name(op.out)) $(op.mode) $(cn(op.l)), $(cn(op.r))"
describe(op::Propagator) = "propagator $(op.type) $(current_name(op.current))"
describe(op::Scale) = "scale $(current_name(op.current))"
describe(op::Clear) = "clear $(current_name(op.current))"
describe(op::Closure) = "closure a$(op.amplitude - 1) += $(cn(op.l)), $(cn(op.r))"
describe(op::Kernel) = "kernel b$(op.binding - 1) $(current_name(op.out)) $(op.mode) $(cn(op.l)), $(cn(op.r))"
cn(i::CurrentIndex) = i == 0 ? "null" : current_name(i)
