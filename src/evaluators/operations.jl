# The instructions of a pac2json program body (main.json `programs[i].body`, legend in
# CONVENTIONS.md), one function per instruction, per phase-space point. The arithmetic is
# pacuda's runner's (runner/recurrence_runner.cu: `contribution_values`, `propagator_values`,
# `identity_value`, `closure_increment`, `symjit_row`, `reduce_kernel`), which is RustiCol's
# operation for operation; the same operations in the same order, so the values are bit for bit
# RustiCol's. Julia's Complex{Float64} +, -, * are the runner's add, sub, mul (no fma).
#
# Indices: Julia vectors count from 1; the formulas in the comments use main.json's 0-based
# components (l[0] is l[1] here). A current is a Vector{Complex{Float64}}; a momentum form is
# (E, px, py, pz).
#
# The source instruction is `apply_source` in wavefunctions.jl. The kernel instruction calls the
# kernel's shared library (main.json `kernels[i].library`, built by pac2json with `--sjb2obj`).
# `kernel_inputs` reads scales through the arena: get_scale(arena, "s<i>") -> Complex{Float64}.

mul_i(z::Complex{Float64}) = complex(-imag(z), real(z))  # i × z

# the operand's components as locals
c2(x) = (x[1], x[2])
c4(x) = (x[1], x[2], x[3], x[4])

# ---- vertex: o, before the coupling and the factor -------------------------------------------------

# l Weyl (2), r vector (4) → Weyl (2):
#   o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0]
#   o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3]
function vertex_weyl_vector_a(l, r)::Vector{Complex{Float64}}
    (l0, l1) = c2(l)
    (r0, r1, r2, r3) = c4(r)
    o0 = ((-mul_i(l0 * r3) - mul_i(l1 * r1)) + l1 * r2) + mul_i(l0 * r0)
    o1 = ((-(l0 * r2) - mul_i(l0 * r1)) + mul_i(l1 * r0)) + mul_i(l1 * r3)
    return [o0, o1]
end

# l Weyl (2), r vector (4) → Weyl (2):
#   o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1]
#   o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0]
function vertex_weyl_vector_b(l, r)::Vector{Complex{Float64}}
    (l0, l1) = c2(l)
    (r0, r1, r2, r3) = c4(r)
    o0 = ((-(l1 * r2) + mul_i(l0 * r0)) + mul_i(l0 * r3)) + mul_i(l1 * r1)
    o1 = ((-mul_i(l1 * r3) + l0 * r2) + mul_i(l0 * r1)) + mul_i(l1 * r0)
    return [o0, o1]
end

# l antisymmetric tensor (6), r vector (4) → vector (4):
#   o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3]
#   o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3]
#   o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3]
#   o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2]
function vertex_tensor_vector(l, r)::Vector{Complex{Float64}}
    (l0, l1, l2, l3, l4, l5) = (l[1], l[2], l[3], l[4], l[5], l[6])
    (r0, r1, r2, r3) = c4(r)
    return [
        (l0 * r1 + l1 * r2) + l2 * r3,
        (l0 * r0 + l3 * r2) + l4 * r3,
        (l1 * r0 - l3 * r1) + l5 * r3,
        (l2 * r0 - l4 * r1) - l5 * r2,
    ]
end

# l vector (4), r vector (4) → antisymmetric tensor (6):
#   o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0],
#        l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2])
function vertex_vector_wedge(l, r)::Vector{Complex{Float64}}
    (l0, l1, l2, l3) = c4(l)
    (r0, r1, r2, r3) = c4(r)
    return [l0 * r1 - l1 * r0, l0 * r2 - l2 * r0, l0 * r3 - l3 * r0, l1 * r2 - l2 * r1, l1 * r3 - l3 * r1, l2 * r3 - l3 * r2]
end

# a·b of two complex vectors and of a complex and a real vector: a0 b0 - a1 b1 - a2 b2 - a3 b3
minkowski_dot(a, b) = ((a[1] * b[1] - a[2] * b[2]) - a[3] * b[3]) - a[4] * b[4]

# l vector (4), r vector (4) → vector (4), the colour-ordered three-gluon vertex; u = momenta[0],
# v = momenta[1]: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j])
function vertex_three_vector(l, r, u, v)::Vector{Complex{Float64}}
    lr = minkowski_dot(l, r)
    lv = minkowski_dot(l, v)
    ru = minkowski_dot(r, u)
    return [lr * (u[j] - v[j]) + (lv * r[j] - ru * l[j]) * 2.0 for j in 1:4]
end

# the vertex's o by its type (main.json `type`)
function vertex_values(type::String, l, r, u, v)::Vector{Complex{Float64}}
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

# The vertex instruction: out[j] (mode)= o[j] × coupling × factor (coupling × factor first, as the
# runner). Changes `out` in place.
function vertex!(out, type::String, mode::String, l, r, u, v, coupling::Complex{Float64}, factor::Complex{Float64})
    scale = coupling * factor
    o = vertex_values(type, l, r, u, v)
    for j in eachindex(o)
        store!(out, j, o[j] * scale, mode)
    end
    return out
end

# out[j] = value ("=") or out[j] = out[j] + value ("+=")
function store!(out, j::ComponentIndex, value::Complex{Float64}, mode::String)
    if mode == "="
        out[j] = value
    elseif mode == "+="
        out[j] = out[j] + value
    else
        error("unknown mode $mode")
    end
end

# ---- propagator: current[j] = o[j] × factor / (k·k), the new values of the current ---------------

reciprocal_norm(k) = 1.0 / (((k[1] * k[1] - k[2] * k[2]) - k[3] * k[3]) - k[4] * k[4])

# c Weyl (2):
#   o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0]
#   o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3]
function propagator_weyl_a(c, k, factor::Complex{Float64})::Vector{Complex{Float64}}
    (c0, c1) = c2(c)
    (k0, k1, k2, k3) = c4(k)
    scale = factor * reciprocal_norm(k)
    o0 = ((-mul_i(c0 * k3) - mul_i(c1 * k1)) + c1 * k2) + mul_i(c0 * k0)
    o1 = ((-(c0 * k2) - mul_i(c0 * k1)) + mul_i(c1 * k0)) + mul_i(c1 * k3)
    return [o0 * scale, o1 * scale]
end

# c Weyl (2):
#   o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1]
#   o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0]
function propagator_weyl_b(c, k, factor::Complex{Float64})::Vector{Complex{Float64}}
    (c0, c1) = c2(c)
    (k0, k1, k2, k3) = c4(k)
    scale = factor * reciprocal_norm(k)
    o0 = ((-(c1 * k2) + mul_i(c0 * k0)) + mul_i(c0 * k3)) + mul_i(c1 * k1)
    o1 = ((-mul_i(c1 * k3) + c0 * k2) + mul_i(c0 * k1)) + mul_i(c1 * k0)
    return [o0 * scale, o1 * scale]
end

# c vector (4), Feynman gauge: o[j] = -i c[j]
function propagator_vector(c, k, factor::Complex{Float64})::Vector{Complex{Float64}}
    scale = (factor * complex(0.0, -1.0)) * reciprocal_norm(k)
    return [c[j] * scale for j in 1:4]
end

function propagator(type::String, c, k, factor::Complex{Float64})::Vector{Complex{Float64}}
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

# current[j] = factor × current[j], j < components (the other components are unchanged)
function scale!(current, components::Int, factor::Complex{Float64})
    for j in 1:components
        current[j] = factor * current[j]
    end
    return current
end

# current[j] = 0, j < components
function clear!(current, components::Int)
    for j in 1:components
        current[j] = complex(0.0, 0.0)
    end
    return current
end

# amplitude += Σ_j l[j] × r[j] × component_factors[j] × factor, j < components: the new amplitude
# (each component's increment added in turn)
function closure(amplitude::Complex{Float64}, l, r, component_factors, factor::Complex{Float64})::Complex{Float64}
    for j in eachindex(component_factors)
        amplitude = amplitude + (l[j] * r[j]) * (component_factors[j] * factor)
    end
    return amplitude
end

# ---- kernel ---------------------------------------------------------------------------------------

# A kernel's library, opened. A library not built yet (pac2json without --sjb2obj, or test/data,
# which holds no binaries) is not an error here: the program loads, and only a call of the kernel
# (kernel_call, compile) fails, so programs without kernel rows run without any library.
# (`name`: the kernel's name in its program, default the library's `symbol`)
function load_kernel(path::String, symbol::String, inputs::Int, outputs::Int; source::String = "", name::String = symbol)
    isfile(path) || return KernelLibrary(name, path, source, C_NULL, C_NULL, inputs, outputs)
    h = Libdl.dlopen(path)
    return KernelLibrary(name, path, source, h, Libdl.dlsym(h, symbol), inputs, outputs)
end

built(lib::KernelLibrary) = lib.fn != C_NULL
not_built(lib::KernelLibrary) = error("kernel $(lib.name): no library (no C compiler, or loaded with kernels = :none; " *
                                      "kernel_lib.jl)")

# The kernel on one point's inputs: its outputs (re, im of each output pair, in order)
function kernel_call(lib::KernelLibrary, inputs::Vector{Float64})::Vector{Float64}
    built(lib) || not_built(lib)
    length(inputs) == lib.inputs || error("the kernel takes $(lib.inputs) inputs, not $(length(inputs))")
    planes = [[x] for x in inputs]
    outs = [[0.0] for _ in 1:lib.outputs]
    GC.@preserve planes outs begin
        states = [SymjitSlice(pointer(p), 1) for p in vcat(planes, outs)]
        status = ccall(lib.fn, Cint, (Ptr{Float64}, Ptr{SymjitSlice}, Csize_t, Ptr{Float64}), C_NULL, states, 0, C_NULL)
        status == 0 || error("the kernel failed ($status)")
    end
    return [o[1] for o in outs]
end

# One binding input (main.json `bindings[i].inputs`), from the kernel row's operands:
#   "re|im in[0][j]", "re|im in[1][j]": a part of a component of l = in[0], r = in[1]
#   "re|im out[j]": of the output current before the row writes it
#   "momenta[0][j]", "momenta[1][j]": a component of u = momenta[0], v = momenta[1]
#   "s<i>": the real part of scale s<i>; "re|im factor": of the row's factor
function kernel_input(code::AbstractString, l, r, out, u, v, factor::Complex{Float64}, arena)::Float64
    part(z, p) = p == "re" ? real(z) : p == "im" ? imag(z) : error("binding input $code")
    index(s) = parse(Int, s) + 1
    if (m = match(r"^(re|im) in\[([01])\]\[(\d+)\]$", code)) !== nothing
        return part((m[2] == "0" ? l : r)[index(m[3])], m[1])
    elseif (m = match(r"^(re|im) out\[(\d+)\]$", code)) !== nothing
        return part(out[index(m[2])], m[1])
    elseif (m = match(r"^momenta\[([01])\]\[(\d+)\]$", code)) !== nothing
        return Float64((m[1] == "0" ? u : v)[index(m[2])])
    elseif (m = match(r"^s\d+$", code)) !== nothing
        return real(get_scale(arena, code))
    elseif (m = match(r"^(re|im) factor$", code)) !== nothing
        return part(factor, m[1])
    else
        error("unknown binding input $code")
    end
end

kernel_inputs(codes, l, r, out, u, v, factor::Complex{Float64}, arena)::Vector{Float64} =
    [kernel_input(c, l, r, out, u, v, factor, arena) for c in codes]

# The kernel instruction: output pair n (re, im) × factor (mode)= out[outputs[n]] (outputs: the
# binding's 0-based components); with binding.direct and factor exactly 1 + 0i the pair is stored
# as it is. Changes `out` in place.
function kernel!(out, lib::KernelLibrary, binding_inputs, binding_outputs, direct::Bool, mode::String,
                 l, r, u, v, factor::Complex{Float64}, arena)
    values = kernel_call(lib, kernel_inputs(binding_inputs, l, r, out, u, v, factor, arena))
    return kernel_store!(out, values, [ComponentIndex(j) + 1 for j in binding_outputs], direct, mode, factor)
end

# The kernel's outputs `values` (re, im of each pair) into `out`: pair n × factor (mode)=
# out[outputs[n]], or the pair as it is with `direct` and factor exactly 1 + 0i
function kernel_store!(out, values::Vector{Float64}, outputs::Vector{ComponentIndex}, direct::Bool, mode::String,
                       factor::Complex{Float64})
    as_is = direct && reinterpret(UInt64, real(factor)) == reinterpret(UInt64, 1.0) &&
            reinterpret(UInt64, imag(factor)) == reinterpret(UInt64, 0.0)
    for (n, j) in enumerate(outputs)
        pair = complex(values[2n-1], values[2n])
        if as_is
            out[j] = pair
        else
            store!(out, j, pair * factor, mode)
        end
    end
    return out
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
