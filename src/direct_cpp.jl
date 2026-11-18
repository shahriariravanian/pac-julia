# pyAmpliCol's cpp backend (planc): a compiled-mode leaf as C++ (`pyamplicol generate ...
# --execution-mode compiled --backend cpp`, `processes/<p>/[...]/compiled/repackaged/
# pyamplicol_<leaf>.direct.cpp`, "pyAmpliCol genuine native DirectApplication producer v1"), read
# into a graph of complex operations.
#
# The C++ (`evaluate_direct_bundle`) is Symbolica's instruction list, one statement per instruction,
# over complex registers `output[j]` (the outputs, also scratch) and `temporary[t]`, which it reuses
# as Symbolica reuses them. Its building blocks:
#   direct_complex_plane<Lane>(inputs[a].values, inputs[a + 1].values, point)   a complex input
#   direct_real_plane<Lane>(inputs[p].values, point)                            a real input (im 0)
#   direct_constant<Lane>(re, im)                                               hex floats
#   direct_add(x, y), direct_multiply(x, y)                                     +, × (textbook formula)
#   direct_real_reciprocal(x), direct_complex_reciprocal(x)                     (1/re x, 0), 1/x
# then `direct_store_lane(outputs[2j].values, point, output[j].real)` (and `.imaginary` at 2j + 1).
#
# `read_direct_cpp(path)` gives a `DirectKernel`: the header's counts and the graph, every value a
# node defined once (the registers' reuse resolved: a read of `output[j]` is the node it holds then),
# inputs as planes (a complex input reads two, a real one one: which leaf input a plane is comes
# from execution.json's bindings). `evaluate_direct(k, planes)` runs it per point with the C++'s
# formulas (no fused operations), so a library built from the C++ with `-ffp-contract=off` gives
# the same bits.

struct DirectCppError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DirectCppError) = print(io, "direct cpp: ", e.msg)

# a node of the graph: an input plane (or pair), a constant, or an operation of earlier nodes
struct DirectNode
    op::Symbol               # :complex_plane, :real_plane, :constant, :add, :multiply, :real_reciprocal, :complex_reciprocal
    args::Vector{Int}        # planes (0-based, the C++'s) of an input; nodes (1-based) of an operation
    value::ComplexF64        # a constant's
end

struct DirectKernel
    name::String             # the leaf's symbol prefix: `<name>_direct_application_v1` is its entry point
    path::String
    input_planes::Int        # kInputPlaneCount
    scalar_inputs::Int       # kScalarInputCount
    output_planes::Int       # kOutputPlaneCount (2 per complex output)
    lane_width::Int          # kSimdLaneWidth
    instruction_count::Int   # kInstructionCount: Symbolica's instructions (the statements)
    temporary_count::Int     # kTemporaryCount
    nodes::Vector{DirectNode}
    outputs::Vector{Int}     # the node of complex output j (planes 2j, 2j + 1), 1-based j
end

const DIRECT_OPS = Dict("direct_add" => :add, "direct_multiply" => :multiply,
                        "direct_real_reciprocal" => :real_reciprocal, "direct_complex_reciprocal" => :complex_reciprocal)
const DIRECT_ARITY = Dict(:add => 2, :multiply => 2, :real_reciprocal => 1, :complex_reciprocal => 1)

function header_count(text::String, name::String)
    m = match(Regex("constexpr std::uint32_t $name = (\\d+)u;"), text)
    m === nothing && throw(DirectCppError("no $name"))
    return parse(Int, m.captures[1])
end

"""
    read_direct_cpp(path) -> DirectKernel

A compiled-mode leaf of pyAmpliCol's cpp backend (`*.direct.cpp`) as a graph of complex operations
(src/direct_cpp.jl).
"""
function read_direct_cpp(path::AbstractString)::DirectKernel
    text = read(path, String)
    startswith(text, "// pyAmpliCol genuine native DirectApplication producer v1") ||
        throw(DirectCppError("$path: not a pyAmpliCol DirectApplication (producer v1) source"))
    m = match(r"extern \"C\" int (\w+)_direct_application_v1\(", text)
    m === nothing && throw(DirectCppError("$path: no entry point"))
    name = String(m.captures[1])
    counts = [header_count(text, n) for n in ("kInputPlaneCount", "kScalarInputCount", "kOutputPlaneCount",
                                             "kSimdLaneWidth", "kInstructionCount", "kTemporaryCount")]
    i = findfirst("inline void evaluate_direct_bundle", text)
    i === nothing && throw(DirectCppError("$path: no evaluate_direct_bundle"))
    body = text[last(i):end]
    body = body[findfirst('{', body)+1:first(findfirst("\n}\n", body))]
    output_count = parse(Int, match(r"std::array<DirectComplex<Lane>, (\d+)> output;", body).captures[1])

    nodes = DirectNode[]
    regs = Dict{Tuple{Symbol,Int},Int}()   # (:output | :temporary, index) -> the node it holds
    node!(op, args, value = 0.0im) = (push!(nodes, DirectNode(op, args, value)); length(nodes))

    # the expression at s[i:]: (its node, the index after it)
    function expr(s::AbstractString, i::Int)
        local m   # not the statement's match (a closure would share it)
        i = something(findnext(!isspace, s, i), i)
        if (m = match(r"^(output|temporary)\[(\d+)\]", SubString(s, i))) !== nothing
            key = (Symbol(m.captures[1]), parse(Int, m.captures[2]))
            haskey(regs, key) || throw(DirectCppError("$path: $(key[1])[$(key[2])] read before it is written"))
            return regs[key], i + length(m.match)
        elseif (m = match(r"^direct_complex_plane<Lane>\(inputs\[(\d+)\]\.values, inputs\[(\d+)\]\.values, point\)", SubString(s, i))) !== nothing
            (a, b) = parse.(Int, m.captures)
            b == a + 1 || throw(DirectCppError("$path: complex input planes $a, $b are not a pair"))
            return node!(:complex_plane, [a, b]), i + length(m.match)
        elseif (m = match(r"^direct_real_plane<Lane>\(inputs\[(\d+)\]\.values, point\)", SubString(s, i))) !== nothing
            return node!(:real_plane, [parse(Int, m.captures[1])]), i + length(m.match)
        elseif (m = match(r"^direct_constant<Lane>\(([^,()]+), ([^,()]+)\)", SubString(s, i))) !== nothing
            value = complex(hexfloat(m.captures[1]), hexfloat(m.captures[2]))
            return node!(:constant, Int[], value), i + length(m.match)
        elseif (m = match(r"^(direct_[a-z_]+)\(", SubString(s, i))) !== nothing && haskey(DIRECT_OPS, m.captures[1])
            op = DIRECT_OPS[m.captures[1]]
            args = Int[]
            j = i + length(m.match)
            while true
                (a, j) = expr(s, j)
                push!(args, a)
                j = something(findnext(!isspace, s, j), j)
                s[j] == ',' && (j += 1; continue)
                s[j] == ')' || throw(DirectCppError("$path: expected ) at $(repr(s[j:min(end, j + 40)]))"))
                j += 1
                break
            end
            length(args) == DIRECT_ARITY[op] || throw(DirectCppError("$path: $(m.captures[1]) of $(length(args)) arguments"))
            return node!(op, args), j
        end
        throw(DirectCppError("$path: cannot read $(repr(s[i:min(end, i + 60)]))"))
    end

    stores = Tuple{Int,Int,Symbol}[]
    for stmt in split(body, ';')
        stmt = strip(stmt)
        (isempty(stmt) || startswith(stmt, "std::array")) && continue
        if (m = match(r"^(output|temporary)\[(\d+)\] = (.*)$"s, stmt)) !== nothing
            (kind, index, rhs) = (Symbol(m.captures[1]), parse(Int, m.captures[2]), m.captures[3])
            (n, j) = expr(rhs, 1)
            j > lastindex(rhs) || throw(DirectCppError("$path: trailing text in $(repr(stmt))"))
            regs[(kind, index)] = n
        elseif (m = match(r"^direct_store_lane\(outputs\[(\d+)\]\.values, point, output\[(\d+)\]\.(real|imaginary)\)$", stmt)) !== nothing
            push!(stores, (parse(Int, m.captures[1]), parse(Int, m.captures[2]), Symbol(m.captures[3])))
        else
            throw(DirectCppError("$path: unknown statement $(repr(stmt[1:min(end, 80)]))"))
        end
    end
    # every complex output j stored as planes 2j (real), 2j + 1 (imaginary)
    expected = [(2j + (part == :imaginary), j, part) for j in 0:output_count-1 for part in (:real, :imaginary)]
    sort(stores) == sort(expected) || throw(DirectCppError("$path: outputs not stored as planes 2j, 2j + 1"))
    2 * output_count == counts[3] || throw(DirectCppError("$path: $output_count outputs for $(counts[3]) output planes"))
    outputs = [get(regs, (:output, j)) do
                   throw(DirectCppError("$path: output[$j] never written"))
               end for j in 0:output_count-1]
    return DirectKernel(name, String(path), counts..., nodes, outputs)
end

hexfloat(s::AbstractString) = parse(Float64, strip(s))   # Julia reads C's hex floats (0x1.8p+1)

"""
    evaluate_direct(k, planes::Vector{Vector{Float64}}) -> Vector{Vector{Float64}}

The leaf on input planes (`k.input_planes` of them, one value per point each): its output planes
(`k.output_planes`), with the C++'s formulas operation for operation.
"""
function evaluate_direct(k::DirectKernel, planes::AbstractVector{<:AbstractVector{Float64}})
    length(planes) == k.input_planes || throw(DirectCppError("$(k.input_planes) input planes, not $(length(planes))"))
    n = length(first(planes))
    out = [Vector{Float64}(undef, n) for _ in 1:k.output_planes]
    v = Vector{ComplexF64}(undef, length(k.nodes))
    for p in 1:n
        for (i, node) in enumerate(k.nodes)
            op = node.op
            v[i] = if op === :complex_plane
                complex(planes[node.args[1]+1][p], planes[node.args[2]+1][p])
            elseif op === :real_plane
                complex(planes[node.args[1]+1][p], 0.0)
            elseif op === :constant
                node.value
            elseif op === :add
                (a, b) = (v[node.args[1]], v[node.args[2]])
                complex(real(a) + real(b), imag(a) + imag(b))
            elseif op === :multiply
                (a, b) = (v[node.args[1]], v[node.args[2]])
                complex(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b))
            elseif op === :real_reciprocal
                complex(1.0 / real(v[node.args[1]]), 0.0)
            else  # :complex_reciprocal
                a = v[node.args[1]]
                d = real(a) * real(a) + imag(a) * imag(a)
                complex(real(a) / d, -imag(a) / d)
            end
        end
        for (j, o) in enumerate(k.outputs)
            out[2j-1][p] = real(v[o])
            out[2j][p] = imag(v[o])
        end
    end
    return out
end

"""
    direct_library_call(library, k, planes) -> Vector{Vector{Float64}}

The leaf's C++ entry point in a shared library (built from `k.path`) on `planes`: its output planes.
"""
function direct_library_call(library::AbstractString, k::DirectKernel, planes::AbstractVector{<:AbstractVector{Float64}})
    n = length(first(planes))
    out = [zeros(n) for _ in 1:k.output_planes]
    h = Libdl.dlopen(library)
    try
        f = Libdl.dlsym(h, Symbol(k.name, "_direct_application_v1"))
        ins = [pointer(p) for p in planes]
        outs = [pointer(o) for o in out]
        status = GC.@preserve planes out ins outs ccall(f, Cint,
            (Ptr{Ptr{Float64}}, UInt32, Ptr{Ptr{Float64}}, UInt32, Ptr{Ptr{Float64}}, UInt32, UInt32, UInt32),
            ins, length(ins), C_NULL, 0, outs, length(outs), 0, n)
        status == 0 || throw(DirectCppError("$(k.name): the library returned $status"))
    finally
        Libdl.dlclose(h)
    end
    return out
end
