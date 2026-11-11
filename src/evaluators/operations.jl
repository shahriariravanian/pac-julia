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

const NO_CURRENT = Complex{Float64}[]  # a null operand (an operand the instruction does not read)

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

# ---- reduction (after the body) --------------------------------------------------------------------

# x × the group factor (numerator, denominator): x as it is when they are equal, else × numerator,
# then ÷ denominator (no precomputed ratio: its rounding would differ)
group(x::Float64, (n, d)::Tuple{Int,Int})::Float64 = n == d ? x : x * Float64(n) / Float64(d)

# ---- scatter (RustiCol's fanout) ------------------------------------------------------------------

# v × f as RustiCol's fanout scatters it: a real factor: v (1), -v (-1), else (re k, im k); a purely
# imaginary one: (-im k, re k); else the complex product (value first)
function scatter_value(v::Complex{Float64}, f::Complex{Float64})::Complex{Float64}
    if imag(f) == 0.0
        k = real(f)
        k == 1.0 && return v
        k == -1.0 && return -v
        return complex(real(v) * k, imag(v) * k)
    elseif real(f) == 0.0
        k = imag(f)
        return complex(-imag(v) * k, real(v) * k)
    end
    return v * f
end
