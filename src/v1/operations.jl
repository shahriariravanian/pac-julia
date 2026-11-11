# The instructions of a main.json Program on an Arena (progeval.jl): the vertex, kernel and
# reduction helpers that read main.json's names (binding inputs as text, modes as "="/"+=");
# the arithmetic itself is ../evaluators/operations.jl's.

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
    return group(acc, group_factor)
end

