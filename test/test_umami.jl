# The UMAMI library of one program (build_umami_library, without an artifact's metadata: the program's
# own leg order and helicities, factor 1), straight-line C and on the interpreter, called through its
# C API: the matrix element is the sum of RustiCol's reference values, in order, bit for bit; the
# alpha_s input is always supported: at the exported alpha_s the values are unchanged, elsewhere
# scaled by (alpha_s / alpha_s0)^power, and a program without a proven power accepts only the
# exported alpha_s (relative 1e-12) and refuses another one.
# usage: julia --project test/test_umami.jl EXPORT.json
using PacRunner
using PacRunner: UmamiLibrary, build_umami_library, output_count, with
using Libdl

# umami.h
const UMAMI_SUCCESS = Cint(0)
const UMAMI_ERROR = Cint(1)
const UMAMI_IN_MOMENTA = Cint(0)
const UMAMI_IN_ALPHA_S = Cint(1)
const UMAMI_OUT_MATRIX_ELEMENT = Cint(0)

struct Library
    handle::Ptr{Cvoid}     # dlopen's
    instance::Ptr{Cvoid}   # umami_initialize's
end

function open_library(l, dir::String; interpreter::Bool)::Library
    n = output_count(l)
    lib = UmamiLibrary(l.process, l.colour, collect(0:l.legs-1), 0, 1.0, collect(0:n-1))
    provider = Dict("particle_count" => l.legs, "helicity_count" => n, "color_count" => 1, "masses" => zeros(l.legs))
    so = build_umami_library(dir, lib, l, provider, n, Tuple{String,ComplexF64}[]; interpreter)
    h = dlopen(so)
    instance = Ref{Ptr{Cvoid}}(C_NULL)
    ccall(dlsym(h, :umami_initialize), Cint, (Ptr{Ptr{Cvoid}}, Cstring), instance, C_NULL) == UMAMI_SUCCESS ||
        error("umami_initialize failed")
    return Library(h, instance[])
end

function close_library(lib::Library)
    ccall(dlsym(lib.handle, :umami_free), Cint, (Ptr{Cvoid},), lib.instance)
    dlclose(lib.handle)
end

function alpha_s_supported(lib::Library)::Bool
    supported = Ref{Ptr{Bool}}(C_NULL)
    count = Ref{Cint}(0)
    ccall(dlsym(lib.handle, :umami_supported_inputs), Cint, (Ptr{Ptr{Bool}}, Ptr{Cint}), supported, count) == UMAMI_SUCCESS ||
        error("umami_supported_inputs failed")
    return unsafe_load(supported[], UMAMI_IN_ALPHA_S + 1)
end

# (status, matrix element) at one point (legs × 4), with an alpha_s input unless `alpha_s` is nothing
function matrix_element(lib::Library, point, alpha_s)
    legs = length(point)
    momenta = [point[j][k] for k in 1:4 for j in 1:legs]   # input[particle + particle_count*component]
    me = Ref(NaN)
    keys = alpha_s === nothing ? [UMAMI_IN_MOMENTA] : [UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S]
    a = Ref(Float64(something(alpha_s, 0.0)))
    GC.@preserve momenta a begin
        inputs = [pointer(momenta), Base.unsafe_convert(Ptr{Float64}, a)][1:length(keys)]
        outputs = [Base.unsafe_convert(Ptr{Float64}, me)]
        status = ccall(dlsym(lib.handle, :umami_matrix_element), Cint,
                       (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Ptr{Cint}, Ptr{Ptr{Float64}}, Csize_t, Ptr{Cint}, Ptr{Ptr{Float64}}),
                       lib.instance, 1, 1, 0, length(keys), keys, inputs, 1, [UMAMI_OUT_MATRIX_ELEMENT], outputs)
    end
    return (status, me[])
end

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

# the failures for the first program of `path` that has an alpha_s power
function check(path::String; quiet::Bool = false)
    ls = load_recurrence_export(path; kernels = :none)
    refs = recurrence_references(path)
    k = findfirst(k -> refs[k] !== nothing && ls[k].alpha_s !== nothing && ls[k].alpha_s[2] !== nothing && ls[k].alpha_s[1] > 0, eachindex(ls))
    k === nothing && error("$path: no program with an alpha_s power and reference values")
    (l, (points, want)) = (ls[k], refs[k])
    (a0, power) = l.alpha_s
    bad = String[]
    for interpreter in (false, true), known in (true, false)
        what = "$(interpreter ? "interpreter" : "straight-line C"), alpha_s power $(known ? power : "unknown")"
        lib = open_library(known ? l : with(l; alpha_s = (a0, nothing)), mktempdir(); interpreter)
        try
            alpha_s_supported(lib) || push!(bad, "$what: alpha_s not a supported input")
            for (point, values) in zip(points, want)
                total = foldl(+, values; init = 0.0)
                # a power scales even a rounding difference; without one, it is the exported alpha_s
                for alpha_s in (known ? (nothing, a0) : (nothing, a0, a0 * (1 + 1e-15)))
                    (status, me) = matrix_element(lib, point, alpha_s)
                    (status == UMAMI_SUCCESS && same(me, total)) || push!(bad, "$what: alpha_s $alpha_s: status $status, $me ≠ $total")
                end
                (status, me) = matrix_element(lib, point, 1.1 * a0)
                if known
                    (status == UMAMI_SUCCESS && isapprox(me, 1.1^power * total; rtol = 1e-14)) ||
                        push!(bad, "$what: alpha_s × 1.1: status $status, $me ≉ $(1.1^power * total)")
                else
                    status == UMAMI_ERROR || push!(bad, "$what: alpha_s × 1.1 accepted (status $status)")
                end
            end
        finally
            close_library(lib)
        end
    end
    quiet || println("$(basename(path)): UMAMI library of $(l.process) $(l.colour) (alpha_s power $power): ",
                     isempty(bad) ? "all as expected" : "failures: " * join(first(bad, 8), "; "))
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(all(f -> isempty(check(f)), ARGS) ? 0 : 1)
end
