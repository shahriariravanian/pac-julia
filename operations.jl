# The instructions of a pac2json program body (main.json `programs[i].body`, legend in
# `configs.instructions`), one function per instruction, per phase-space point. The arithmetic is
# pacuda's runner's (runner/recurrence_runner.cu: `contribution_values`, `propagator_values`,
# `identity_value`, `closure_increment`, `symjit_row`, `reduce_kernel`), which is RustiCol's
# operation for operation; the same operations in the same order, so the values are bit for bit
# RustiCol's. Julia's Complex{Float64} +, -, * are the runner's add, sub, mul (no fma).
#
# Values: a current is a tuple of its kind (Weyl, Vec4, Tensor, ...; types.jl), a momentum form a
# Momentum (E, px, py, pz). The functions return new values; nothing is changed in place.
# Indices count from 1; the formulas in the comments use main.json's 0-based components (l[0] is
# l[1] here).
#
# The source instruction is `apply_source` in wavefunctions.jl. The kernel instruction calls the
# kernel's shared library (main.json `kernels[i].library`, built by pac2json with `--sjb2obj`).
# `kernel_inputs` reads scales through the arena: get_scale(arena, "s<i>") -> Complex{Float64}.

using Libdl

include(joinpath(@__DIR__, "types.jl"))

mul_i(z::Complex{Float64}) = complex(-imag(z), real(z))  # i × z

# the new value of a component: value ("=") or old + value ("+=")
function combine(old::Complex{Float64}, value::Complex{Float64}, mode::String)::Complex{Float64}
    if mode == "="
        return value
    elseif mode == "+="
        return old + value
    else
        error("unknown mode $mode")
    end
end

# ---- vertex: o, before the coupling and the factor -------------------------------------------------

# l Weyl, r vector → Weyl:
#   o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0]
#   o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3]
function vertex_weyl_vector_a(l::Weyl, r::Vec4)::Weyl
    (l0, l1) = l
    (r0, r1, r2, r3) = r
    o0 = ((-mul_i(l0 * r3) - mul_i(l1 * r1)) + l1 * r2) + mul_i(l0 * r0)
    o1 = ((-(l0 * r2) - mul_i(l0 * r1)) + mul_i(l1 * r0)) + mul_i(l1 * r3)
    return (o0, o1)
end

# l Weyl, r vector → Weyl:
#   o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1]
#   o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0]
function vertex_weyl_vector_b(l::Weyl, r::Vec4)::Weyl
    (l0, l1) = l
    (r0, r1, r2, r3) = r
    o0 = ((-(l1 * r2) + mul_i(l0 * r0)) + mul_i(l0 * r3)) + mul_i(l1 * r1)
    o1 = ((-mul_i(l1 * r3) + l0 * r2) + mul_i(l0 * r1)) + mul_i(l1 * r0)
    return (o0, o1)
end

# l antisymmetric tensor, r vector → vector:
#   o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3]
#   o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3]
#   o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3]
#   o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2]
function vertex_tensor_vector(l::Tensor, r::Vec4)::Vec4
    (l0, l1, l2, l3, l4, l5) = l
    (r0, r1, r2, r3) = r
    return (
        (l0 * r1 + l1 * r2) + l2 * r3,
        (l0 * r0 + l3 * r2) + l4 * r3,
        (l1 * r0 - l3 * r1) + l5 * r3,
        (l2 * r0 - l4 * r1) - l5 * r2,
    )
end

# l vector, r vector → antisymmetric tensor:
#   o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0],
#        l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2])
function vertex_vector_wedge(l::Vec4, r::Vec4)::Tensor
    (l0, l1, l2, l3) = l
    (r0, r1, r2, r3) = r
    return (l0 * r1 - l1 * r0, l0 * r2 - l2 * r0, l0 * r3 - l3 * r0, l1 * r2 - l2 * r1, l1 * r3 - l3 * r1, l2 * r3 - l3 * r2)
end

# a·b of two complex vectors, or of a complex vector and a momentum: a0 b0 - a1 b1 - a2 b2 - a3 b3
minkowski_dot(a::Vec4, b::Union{Vec4,Momentum}) = ((a[1] * b[1] - a[2] * b[2]) - a[3] * b[3]) - a[4] * b[4]

# l vector, r vector → vector, the colour-ordered three-gluon vertex; u = momenta[0],
# v = momenta[1]: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j])
function vertex_three_vector(l::Vec4, r::Vec4, u::Momentum, v::Momentum)::Vec4
    lr = minkowski_dot(l, r)
    lv = minkowski_dot(l, v)
    ru = minkowski_dot(r, u)
    return ntuple(j -> lr * (u[j] - v[j]) + (lv * r[j] - ru * l[j]) * 2.0, 4)
end

# the vertex's o by its type (main.json `type`)
function vertex_values(type::String, l::CurrentValue, r::CurrentValue, u::Momentum, v::Momentum)::CurrentValue
    if type == "weyl_vector_a"
        return vertex_weyl_vector_a(l, r)
    elseif type == "weyl_vector_b"
        return vertex_weyl_vector_b(l, r)
    elseif type == "tensor_vector"
        return vertex_tensor_vector(l, r)
    elseif type == "vector_wedge"
        return vertex_vector_wedge(l, r)
    elseif type == "three_vector"
        return vertex_three_vector(l, r, u, v)
    else
        error("unknown vertex type $type")
    end
end

# The vertex instruction: the new value of `out`, out[j] (mode)= o[j] × coupling × factor
# (coupling × factor first, as the runner)
function vertex(type::String, mode::String, out::T, l::CurrentValue, r::CurrentValue, u::Momentum, v::Momentum,
                coupling::Complex{Float64}, factor::Complex{Float64})::T where {T<:CurrentValue}
    return assign(mode, out, vertex_values(type, l, r, u, v)::T, coupling * factor)
end

# the new value of `out` from a vertex's o: out[j] (mode)= o[j] × scale
assign(mode::String, out::T, o::T, scale::Complex{Float64}) where {T<:CurrentValue} =
    ntuple(j -> combine(out[j], o[j] * scale, mode), length(out))::T

# ---- propagator: current[j] = o[j] × factor / (k·k), the new value of the current ----------------

reciprocal_norm(k::Momentum) = 1.0 / (((k[1] * k[1] - k[2] * k[2]) - k[3] * k[3]) - k[4] * k[4])

# c Weyl:
#   o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0]
#   o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3]
function propagator_weyl_a(c::Weyl, k::Momentum, factor::Complex{Float64})::Weyl
    (c0, c1) = c
    (k0, k1, k2, k3) = k
    scale = factor * reciprocal_norm(k)
    o0 = ((-mul_i(c0 * k3) - mul_i(c1 * k1)) + c1 * k2) + mul_i(c0 * k0)
    o1 = ((-(c0 * k2) - mul_i(c0 * k1)) + mul_i(c1 * k0)) + mul_i(c1 * k3)
    return (o0 * scale, o1 * scale)
end

# c Weyl:
#   o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1]
#   o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0]
function propagator_weyl_b(c::Weyl, k::Momentum, factor::Complex{Float64})::Weyl
    (c0, c1) = c
    (k0, k1, k2, k3) = k
    scale = factor * reciprocal_norm(k)
    o0 = ((-(c1 * k2) + mul_i(c0 * k0)) + mul_i(c0 * k3)) + mul_i(c1 * k1)
    o1 = ((-mul_i(c1 * k3) + c0 * k2) + mul_i(c0 * k1)) + mul_i(c1 * k0)
    return (o0 * scale, o1 * scale)
end

# c vector, Feynman gauge: o[j] = -i c[j]
function propagator_vector(c::Vec4, k::Momentum, factor::Complex{Float64})::Vec4
    scale = (factor * complex(0.0, -1.0)) * reciprocal_norm(k)
    return ntuple(j -> c[j] * scale, 4)
end

function propagator(type::String, c::T, k::Momentum, factor::Complex{Float64})::T where {T<:CurrentValue}
    if type == "weyl_a"
        return propagator_weyl_a(c, k, factor)
    elseif type == "weyl_b"
        return propagator_weyl_b(c, k, factor)
    elseif type == "vector"
        return propagator_vector(c, k, factor)
    else
        error("unknown propagator type $type")
    end
end

# ---- scale, clear, closure ----------------------------------------------------------------------

# current[j] = factor × current[j], j < components (the other components unchanged): the new value
scale(c::T, components::Int, factor::Complex{Float64}) where {T<:CurrentValue} =
    ntuple(j -> j <= components ? factor * c[j] : c[j], length(c))::T

# current[j] = 0, j < components: the new value
clear(c::T, components::Int) where {T<:CurrentValue} = ntuple(j -> j <= components ? ZERO : c[j], length(c))::T

# amplitude += Σ_j l[j] × r[j] × component_factors[j] × factor, j < components: the new amplitude
# (each component's increment added in turn)
function closure(amplitude::Complex{Float64}, l::CurrentValue, r::CurrentValue, component_factors::AbstractVector{Complex{Float64}},
                 factor::Complex{Float64})::Complex{Float64}
    for j in eachindex(component_factors)
        amplitude = amplitude + (l[j] * r[j]) * (component_factors[j] * factor)
    end
    return amplitude
end

# ---- kernel ---------------------------------------------------------------------------------------

struct SymjitSlice  # symjit_slice = {double *ptr; size_t len}
    ptr::Ptr{Float64}
    len::Csize_t
end

# A kernel's shared library (main.json kernels[i].library): its function, its counts, and the
# buffers of one point that kernel_call! passes it (made once; one KernelLibrary per thread)
struct KernelLibrary
    handle::Ptr{Cvoid}
    fn::Ptr{Cvoid}
    inputs::Int                  # library.states
    outputs::Int                 # library.outputs: 2 × the output pairs
    input::Vector{Float64}       # the inputs of the next call
    output::Vector{Float64}      # the outputs of the last call: re, im of each pair
    states::Vector{SymjitSlice}  # one plane of one point per input, then per output, in the buffers
end

function load_kernel(path::String, symbol::String, inputs::Int, outputs::Int)::KernelLibrary
    h = Libdl.dlopen(path)
    (input, output) = (zeros(inputs), zeros(outputs))
    # the buffers are never resized, and Julia does not move arrays: the pointers stay valid
    states = vcat([SymjitSlice(pointer(input, k), 1) for k in 1:inputs], [SymjitSlice(pointer(output, k), 1) for k in 1:outputs])
    return KernelLibrary(h, Libdl.dlsym(h, symbol), inputs, outputs, input, output, states)
end

# The kernel on lib.input (one point): its outputs, in lib.output (re, im of each pair, in order)
function kernel_call!(lib::KernelLibrary)::Vector{Float64}
    status = GC.@preserve lib ccall(lib.fn, Cint, (Ptr{Float64}, Ptr{SymjitSlice}, Csize_t, Ptr{Float64}),
                                     C_NULL, lib.states, 0, C_NULL)
    status == 0 || error("the kernel failed ($status)")
    return lib.output
end

# The kernel on `inputs`: a copy of its outputs
function kernel_call(lib::KernelLibrary, inputs::Vector{Float64})::Vector{Float64}
    length(inputs) == lib.inputs || error("the kernel takes $(lib.inputs) inputs, not $(length(inputs))")
    copyto!(lib.input, inputs)
    return copy(kernel_call!(lib))
end

const Operand = Union{CurrentValue,NoOperand}

# One binding input (main.json `bindings[i].inputs`), from the kernel row's operands:
#   "re|im in[0][j]", "re|im in[1][j]": a part of a component of l = in[0], r = in[1]
#   "re|im out[j]": of the output current before the row writes it
#   "momenta[0][j]", "momenta[1][j]": a component of u = momenta[0], v = momenta[1]
#   "s<i>": the real part of scale s<i>; "re|im factor": of the row's factor
function kernel_input(code::AbstractString, l::Operand, r::Operand, out::CurrentValue, u::Momentum, v::Momentum,
                      factor::Complex{Float64}, arena)::Float64
    part(z, p) = p == "re" ? real(z) : p == "im" ? imag(z) : error("binding input $code")
    index(s) = parse(ComponentIndex, s) + 1
    if (m = match(r"^(re|im) in\[([01])\]\[(\d+)\]$", code)) !== nothing
        return part((m[2] == "0" ? l : r)[index(m[3])], m[1])
    elseif (m = match(r"^(re|im) out\[(\d+)\]$", code)) !== nothing
        return part(out[index(m[2])], m[1])
    elseif (m = match(r"^momenta\[([01])\]\[(\d+)\]$", code)) !== nothing
        return (m[1] == "0" ? u : v)[index(m[2])]
    elseif (m = match(r"^s\d+$", code)) !== nothing
        return real(get_scale(arena, code))
    elseif (m = match(r"^(re|im) factor$", code)) !== nothing
        return part(factor, m[1])
    else
        error("unknown binding input $code")
    end
end

kernel_inputs(codes, l::Operand, r::Operand, out::CurrentValue, u::Momentum, v::Momentum, factor::Complex{Float64},
              arena)::Vector{Float64} = [kernel_input(c, l, r, out, u, v, factor, arena) for c in codes]

# The kernel instruction: the new value of `out`, output pair n (re, im) × factor (mode)=
# out[outputs[n]] (outputs: the binding's 0-based components); with binding.direct and factor
# exactly 1 + 0i the pair is stored as it is
function kernel(out::T, lib::KernelLibrary, binding_inputs, binding_outputs, direct::Bool, mode::String,
                l::Operand, r::Operand, u::Momentum, v::Momentum, factor::Complex{Float64}, arena)::T where {T<:CurrentValue}
    values = kernel_call(lib, kernel_inputs(binding_inputs, l, r, out, u, v, factor, arena))
    return kernel_store(out, values, [ComponentIndex(j) + 1 for j in binding_outputs], direct, mode, factor)
end

# The kernel's outputs `values` (re, im of each pair) into `out`: the new value, pair n × factor
# (mode)= out[outputs[n]], or the pair as it is with `direct` and factor exactly 1 + 0i; the other
# components unchanged (the outputs are distinct components, checked at load)
function kernel_store(out::T, values::AbstractVector{Float64}, outputs::Vector{ComponentIndex}, direct::Bool, mode::String,
                      factor::Complex{Float64})::T where {T<:CurrentValue}
    as_is = direct && reinterpret(UInt64, real(factor)) == reinterpret(UInt64, 1.0) &&
            reinterpret(UInt64, imag(factor)) == reinterpret(UInt64, 0.0)
    return ntuple(length(out)) do j
        n = findfirst(==(j), outputs)
        n === nothing && return out[j]
        pair = complex(values[2n-1], values[2n])
        return as_is ? pair : combine(out[j], pair * factor, mode)
    end
end

# ---- reduction (after the body) --------------------------------------------------------------------

# outputs[h] = group_factor × Σ weight × |a|² over reduction[h]'s terms, in order
# (|a|² = fma(re, re, im × im)); group_factor = (numerator, denominator)
function reduction(amplitudes, weights, group_factor)::Float64
    acc = 0.0
    for (a, w) in zip(amplitudes, weights)
        acc = acc + w * fma(real(a), real(a), imag(a) * imag(a))
    end
    (n, d) = group_factor
    return n == d ? acc : acc * Float64(n) / Float64(d)
end
