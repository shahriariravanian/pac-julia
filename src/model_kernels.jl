# planc phase 3: the model's prepared kernels, from pyAmpliCol's prepared-model pack
# (`pyamplicol/assets/prepared_models/<model>-jit-o2-<arch>.pyamplicol-model`, an uncompressed zip:
# `manifest.json`, whose `kernel_pack.kernels` describes each kernel, and per kernel
# `kernels/<id>/application-1.plane.symjit`, the symjit plane kernel recurrence mode runs).
#
# In recurrence mode a vertex or propagator without a RustiCol intrinsic runs as one of these
# kernels (a `kernel` row of the export, its binding feeding the kernel's inputs); in compiled mode
# its formula is inlined in the leaves. They are the model's, not a process's: the same kernel (by
# SHA-256) serves every process of the model.
#
# A kernel's inputs (complex, as re, im planes) in its contract order, by role:
#   left-current, right-current   the left/right operand's components, in order (the operand's own:
#                                 a Weyl spinor's 0, 1 where the contract names Dirac components)
#   current                       a propagator's operand
#   momentum                      the propagated current's momentum (left-, right-momentum: the
#                                 operands'), a real input (imaginary part 0)
#   model-parameter               a model parameter (a mass, a width), real
#   coupling-real, coupling-imag  the coupling's parts (fed 1, 0 when measuring: the ratio holds it)
# and its outputs the destination's components (output_arity of them).

struct ModelKernelError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ModelKernelError) = print(io, "model kernel: ", e.msg)

struct ModelKernel
    id::Int
    contract::Symbol                       # :vertex, :propagator
    roles::Vector{Tuple{Symbol,Int,String}} # (role, component, model parameter name or "")
    output_arity::Int
    output_layout::Vector{String}
    sha::String                            # of application-1.plane.symjit
    library::KernelLibrary
end

role_count(k::ModelKernel, role::Symbol) = count(r -> r[1] === role, k.roles)

# the entries of an uncompressed (stored) zip: name -> bytes
function read_stored_zip(path::AbstractString)::Dict{String,Vector{UInt8}}
    data = read(path)
    u16(i) = Int(data[i+1]) | Int(data[i+2]) << 8
    u32(i) = u16(i) | u16(i + 2) << 16
    # the end of central directory record: its signature 0x06054b50, searched from the end
    e = findlast(i -> data[i] == 0x50 && data[i+1] == 0x4b && data[i+2] == 0x05 && data[i+3] == 0x06, 1:length(data)-21)
    e === nothing && throw(ModelKernelError("$path: not a zip archive"))
    (count, offset) = (u16(e - 1 + 10), u32(e - 1 + 16))
    out = Dict{String,Vector{UInt8}}()
    p = offset
    for _ in 1:count
        u32(p) == 0x02014b50 || throw(ModelKernelError("$path: a bad central directory entry"))
        method, size = u16(p + 10), u32(p + 20)
        (name_len, extra_len, comment_len, local_at) = (u16(p + 28), u16(p + 30), u16(p + 32), u32(p + 42))
        name = String(data[p+47:p+46+name_len])
        method == 0 || throw(ModelKernelError("$path: $name is compressed (method $method): only stored entries are read"))
        u32(local_at) == 0x04034b50 || throw(ModelKernelError("$path: a bad local header for $name"))
        start = local_at + 30 + u16(local_at + 26) + u16(local_at + 28)
        out[name] = data[start+1:start+size]
        p += 46 + name_len + extra_len + comment_len
    end
    return out
end

"""
    model_kernels(pack; kernels = :c) -> Vector{ModelKernel}

The prepared kernels of a pyAmpliCol prepared-model pack (`.pyamplicol-model`), each with its
library built (`kernels`: kernel_lib.jl's ways; the plane kernels go to the scratch space by
SHA-256, as the export's do).
"""
function model_kernels(pack::AbstractString; kernels::Symbol = :c)::Vector{ModelKernel}
    entries = read_stored_zip(pack)
    haskey(entries, "manifest.json") || throw(ModelKernelError("$pack: no manifest.json"))
    manifest = JSON.parse(String(entries["manifest.json"]))
    way = resolve_way(kernels)
    out = ModelKernel[]
    for k in manifest["kernel_pack"]["kernels"]
        id = Int(k["kernel_id"])
        member = "kernels/$(lpad(id, 6, '0'))/application-1.plane.symjit"
        haskey(entries, member) || throw(ModelKernelError("$pack: no $member"))
        bytes = entries[member]
        sha = bytes2hex(sha256(bytes))
        source = joinpath(export_kernels_dir(), "$sha.plane.symjit")
        if !isfile(source)
            tmp = tempname(export_kernels_dir())
            write(tmp, bytes)
            mv(tmp, source; force = true)
        end
        app = read_sjb(source)
        (inputs, outputs) = (app.count_states, app.count_obs)
        name = "k_" * sha[1:16]
        lib = way == :none ? load_kernel("", name, inputs, outputs; source) :
              load_kernel(something(kernel_library(source, sha, inputs, outputs, way), ""), KERNEL_SYMBOL, inputs, outputs; source, name)
        roles = [(Symbol(replace(c["role"], "-" => "_")), Int(c["component"]), something(c["model_parameter_name"], ""))
                 for c in k["input_contracts"]]
        2 * length(roles) == inputs || throw(ModelKernelError("kernel $id: $(length(roles)) inputs, $inputs planes"))
        2 * k["output_arity"] == outputs || throw(ModelKernelError("kernel $id: $(k["output_arity"]) outputs, $outputs planes"))
        push!(out, ModelKernel(id, Symbol(k["contract_kind"]), roles, k["output_arity"], String.(k["output_layout"]), sha, lib))
    end
    return out
end

"""
    run_model_kernel(k, operands::Dict{Symbol,Vector{ComplexF64}}, momenta::Dict{Symbol,Vector{Float64}},
                     parameters::Dict{String,Float64}) -> Vector{ComplexF64}

The kernel on one point: `operands` by role (:left_current, :right_current, :current), `momenta`
(:momentum, :left_momentum, :right_momentum), the model parameters by name; couplings 1.
"""
function run_model_kernel(k::ModelKernel, operands, momenta, parameters)::Vector{ComplexF64}
    seen = Dict{Symbol,Int}()
    planes = Float64[]
    for (role, comp, pname) in k.roles
        z = if role in (:left_current, :right_current, :current)
            i = seen[role] = get(seen, role, 0) + 1
            operands[role][i]
        elseif role in (:momentum, :left_momentum, :right_momentum)
            complex(momenta[role][comp+1])
        elseif role === :model_parameter
            haskey(parameters, pname) || throw(ModelKernelError("kernel $(k.id): no value of $pname"))
            complex(parameters[pname])
        elseif role === :coupling_real
            1.0 + 0im
        elseif role === :coupling_imag
            0.0 + 0im
        else
            throw(ModelKernelError("kernel $(k.id): input role $role"))
        end
        push!(planes, real(z), imag(z))
    end
    o = kernel_call(k.library, planes)
    return [complex(o[2j-1], o[2j]) for j in 1:k.output_arity]
end
