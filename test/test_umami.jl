# The UMAMI library of one program (build_umami_library, without an artifact's metadata: the program's
# own leg order and helicities, factor 1), straight-line C and on the interpreter, or (cuda = true)
# on the GPU, called through its C API: the matrix element is the sum of RustiCol's reference values, in order, bit for bit; the
# alpha_s input is always supported: at the exported alpha_s the values are unchanged, elsewhere
# scaled by (alpha_s / alpha_s0)^power, and a program without a proven power accepts only the
# exported alpha_s (relative 1e-12) and refuses another one.
# usage: julia --project test/test_umami.jl EXPORT.json
using PacRunner
using PacRunner: UmamiLibrary, build_umami_library, output_count, with
using Libdl
using Random

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

function open_library(l, dir::String; interpreter::Bool, device::Symbol = :cpu, precision::Symbol = :f64,
                      gpu_memory::Real = PacRunner.GPU_MEMORY_DEFAULT)::Library
    n = output_count(l)
    lib = UmamiLibrary(l.process, l.colour, collect(0:l.legs-1), 0, 1.0, collect(0:n-1))
    provider = Dict("particle_count" => l.legs, "helicity_count" => n, "color_count" => 1, "masses" => zeros(l.legs))
    so = build_umami_library(dir, lib, l, provider, n, Tuple{String,ComplexF64}[]; interpreter, device, precision, gpu_memory)
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

# the failures for the first program of `path` that has an alpha_s power; precision = :mixed (CUDA):
# within MIXED_RTOL instead of bit for bit
const MIXED_RTOL = 1e-4
function check(path::String; quiet::Bool = false, cuda::Bool = false, precision::Symbol = :f64)
    agree(x, y) = precision == :f64 ? same(x, y) : isapprox(x, y; rtol = MIXED_RTOL)
    ls = load_recurrence_export(path; kernels = :none)
    refs = recurrence_references(path)
    k = findfirst(k -> refs[k] !== nothing && ls[k].alpha_s !== nothing && ls[k].alpha_s[2] !== nothing && ls[k].alpha_s[1] > 0, eachindex(ls))
    k === nothing && error("$path: no program with an alpha_s power and reference values")
    (l, (points, want)) = (ls[k], refs[k])
    (a0, power) = l.alpha_s
    bad = String[]
    # (interpreter, device); CUDA f64 after the straight-line C, to compare them at another alpha_s
    builds = cuda ? (precision == :f64 ? [(false, :cpu), (true, :cuda)] : [(true, :cuda)]) : [(false, :cpu), (true, :cpu)]
    scaled = Dict{Int,Float64}()   # the first build's values at 1.1 alpha_s0: every f64 build's, bit for bit
    for (interpreter, device) in builds, known in (true, false)
        what = "$(device == :cuda ? "CUDA $precision" : interpreter ? "interpreter" : "straight-line C"), alpha_s power $(known ? power : "unknown")"
        lib = open_library(known ? l : with(l; alpha_s = (a0, nothing)), mktempdir(); interpreter, device, precision)
        try
            alpha_s_supported(lib) || push!(bad, "$what: alpha_s not a supported input")
            for (p, (point, values)) in enumerate(zip(points, want))
                total = foldl(+, values; init = 0.0)
                # a power scales even a rounding difference; without one, it is the exported alpha_s
                for alpha_s in (known ? (nothing, a0) : (nothing, a0, a0 * (1 + 1e-15)))
                    (status, me) = matrix_element(lib, point, alpha_s)
                    (status == UMAMI_SUCCESS && agree(me, total)) || push!(bad, "$what: alpha_s $alpha_s: status $status, $me ≠ $total")
                end
                (status, me) = matrix_element(lib, point, 1.1 * a0)
                if known
                    (status == UMAMI_SUCCESS && isapprox(me, 1.1^power * total; rtol = precision == :f64 ? 1e-14 : MIXED_RTOL)) ||
                        push!(bad, "$what: alpha_s × 1.1: status $status, $me ≉ $(1.1^power * total)")
                    if precision == :f64
                        first_me = get!(scaled, p, me)
                        same(me, first_me) || push!(bad, "$what: alpha_s × 1.1: $me, not the first build's $first_me")
                    end
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

# ---- the CUDA provider (device = :cuda_provider), driven as MadSpace's GPU runtime drives it -----

const UMAMI_IN_FLAVOR_INDEX = Cint(2)
const UMAMI_IN_RANDOM_HELICITY = Cint(4)
const UMAMI_IN_HELICITY_INDEX = Cint(6)
const UMAMI_OUT_COLOR_INDEX = Cint(2)
const UMAMI_OUT_HELICITY_INDEX = Cint(3)
const UMAMI_OUT_GPU_STREAM = Cint(5)
const UMAMI_META_DEVICE = Cint(0)
const UMAMI_DEVICE_CUDA = Cint(1)

# one umami_matrix_element call: inputs and outputs as (key, pointer) pairs
function call_batch(lib::Library, count, stride, offset, inputs, outputs)::Cint
    ik = Cint[k for (k, _) in inputs]; ip = Ptr{Cvoid}[p for (_, p) in inputs]
    ok = Cint[k for (k, _) in outputs]; op = Ptr{Cvoid}[p for (_, p) in outputs]
    return ccall(dlsym(lib.handle, :umami_matrix_element), Cint,
                 (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Ptr{Cint}, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Cint}, Ptr{Ptr{Cvoid}}),
                 lib.instance, count, stride, offset, length(ik), ik, ip, length(ok), ok, op)
end

# the CUDA runtime the provider links (its copy in the build folder)
struct Cuda
    rt::Ptr{Cvoid}
end
cuda_check(c::Cuda, s::Cint, what) = s == 0 || error("$what: CUDA error $s")
function upload(c::Cuda, x::Vector{T}) where {T}
    r = Ref{Ptr{Cvoid}}(C_NULL)
    cuda_check(c, ccall(dlsym(c.rt, :cudaMalloc), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), r, sizeof(x)), "cudaMalloc")
    cuda_check(c, ccall(dlsym(c.rt, :cudaMemcpy), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint), r[], x, sizeof(x), 1), "cudaMemcpy")
    return r[]
end
function download!(c::Cuda, x::Vector{T}, p::Ptr{Cvoid}) where {T}
    cuda_check(c, ccall(dlsym(c.rt, :cudaMemcpy), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint), x, p, sizeof(x), 2), "cudaMemcpy")
    return x
end
release(c::Cuda, p::Ptr{Cvoid}) = ccall(dlsym(c.rt, :cudaFree), Cint, (Ptr{Cvoid},), p)

# The provider against the straight-line CPU build on one batch of `events` (the export's reference
# points, cycled) at offset 5 of a longer stride, with a workspace of about 97 events (several
# launches): the sampled pass (random helicity in; ME, colour and helicity index out) and the
# fixed-helicity pass, half the events at alpha_s0 and half at 1.1 alpha_s0; then a flavour other
# than 0: that event's ME is NaN, the next call reports it once and the one after runs. f64: bit for bit at alpha_s0, the device's pow within 4 ulp
# elsewhere; mixed: within MIXED_RTOL. Returns the failures.
function check_provider(path::String; precision::Symbol = :f64, events::Int = 300, quiet::Bool = false)
    ls = load_recurrence_export(path; kernels = :none)
    refs = recurrence_references(path)
    k = findfirst(k -> refs[k] !== nothing && ls[k].alpha_s !== nothing && ls[k].alpha_s[2] !== nothing && ls[k].alpha_s[1] > 0, eachindex(ls))
    k === nothing && error("$path: no program with an alpha_s power and reference values")
    (l, (points, _)) = (ls[k], refs[k])
    a0 = l.alpha_s[1]
    n, exported, N = l.legs, output_count(l), events
    stride, offset = N + 7, 5
    bad = String[]
    cpu = open_library(l, mktempdir(); interpreter = false)
    dir = mktempdir()
    gpu = open_library(l, dir; interpreter = true, device = :cuda_provider, precision,
                       gpu_memory = 97 * PacRunner.gpu_bytes_per_point(l; precision) + 4096)
    rt = dlopen(first(filter(f -> startswith(basename(f), "libcudart.so"), readdir(dir; join = true))))
    c = Cuda(rt)
    try
        device = Ref{Cint}(-1)
        ccall(dlsym(gpu.handle, :umami_get_meta), Cint, (Cint, Ptr{Cvoid}), UMAMI_META_DEVICE, device)
        device[] == UMAMI_DEVICE_CUDA || push!(bad, "the provider reports device $(device[]), not CUDA")
        rng = Random.Xoshiro(20261113)
        momenta = fill(NaN, 4n * stride)
        for e in 1:N, p in 1:n, j in 1:4
            momenta[(n * (j - 1) + p - 1) * stride + offset + e] = points[mod1(e, length(points))][p][j]
        end
        random = rand(rng, stride)
        alpha_s = [isodd(e) ? a0 : 1.1 * a0 for e in 1:stride]
        fixed = Cint.(rand(rng, 0:exported-1, stride))
        stream = Ref{Ptr{Cvoid}}(C_NULL)
        cuda_check(c, ccall(dlsym(rt, :cudaStreamCreate), Cint, (Ptr{Ptr{Cvoid}},), stream), "cudaStreamCreate")
        sync() = cuda_check(c, ccall(dlsym(rt, :cudaStreamSynchronize), Cint, (Ptr{Cvoid},), stream[]), "cudaStreamSynchronize")
        d = Dict(:momenta => upload(c, momenta), :random => upload(c, random), :alpha_s => upload(c, alpha_s), :fixed => upload(c, fixed))
        events_in(e) = offset < e <= offset + N
        for pass in (:sampled, :fixed)
            # the CPU build on host arrays
            me = fill(-1.0, stride); hel = fill(Cint(-7), stride); col = fill(Cint(-7), stride)
            ins = pass == :sampled ? [(UMAMI_IN_MOMENTA, momenta), (UMAMI_IN_ALPHA_S, alpha_s), (UMAMI_IN_RANDOM_HELICITY, random)] :
                                     [(UMAMI_IN_MOMENTA, momenta), (UMAMI_IN_ALPHA_S, alpha_s), (UMAMI_IN_HELICITY_INDEX, fixed)]
            GC.@preserve momenta alpha_s random fixed me hel col begin
                outs = [(UMAMI_OUT_MATRIX_ELEMENT, pointer(me)), (UMAMI_OUT_COLOR_INDEX, pointer(col))]
                pass == :sampled && push!(outs, (UMAMI_OUT_HELICITY_INDEX, pointer(hel)))
                status = call_batch(cpu, N, stride, offset, [(key, pointer(x)) for (key, x) in ins], outs)
                status == UMAMI_SUCCESS || push!(bad, "$pass: the CPU build: status $status")
            end
            # the provider on device arrays and a stream
            dme, dhel, dcol = upload(c, fill(-1.0, stride)), upload(c, fill(Cint(-7), stride)), upload(c, fill(Cint(-7), stride))
            dins = [(UMAMI_IN_MOMENTA, d[:momenta]), (UMAMI_IN_ALPHA_S, d[:alpha_s]),
                    pass == :sampled ? (UMAMI_IN_RANDOM_HELICITY, d[:random]) : (UMAMI_IN_HELICITY_INDEX, d[:fixed])]
            douts = [(UMAMI_OUT_MATRIX_ELEMENT, dme), (UMAMI_OUT_COLOR_INDEX, dcol), (UMAMI_OUT_GPU_STREAM, stream[])]
            pass == :sampled && push!(douts, (UMAMI_OUT_HELICITY_INDEX, dhel))
            status = call_batch(gpu, N, stride, offset, dins, douts)
            status == UMAMI_SUCCESS || push!(bad, "$pass: the provider: status $status")
            sync()
            gme, ghel, gcol = download!(c, fill(0.0, stride), dme), download!(c, fill(Cint(0), stride), dhel), download!(c, fill(Cint(0), stride), dcol)
            foreach(p -> release(c, p), (dme, dhel, dcol))
            for e in 1:stride
                if !events_in(e)
                    (gme[e] == -1.0 && ghel[e] == -7 && gcol[e] == -7) || push!(bad, "$pass: slot $e outside the call written")
                    continue
                end
                exact = alpha_s[e] == a0
                ok = precision == :f64 ? (exact ? same(gme[e], me[e]) : abs(gme[e] - me[e]) <= 4 * eps(abs(me[e]))) :
                     isapprox(gme[e], me[e]; rtol = MIXED_RTOL)
                ok || push!(bad, "$pass: event $e (alpha_s $(alpha_s[e])): $(gme[e]) vs the CPU's $(me[e])")
                gcol[e] == col[e] || push!(bad, "$pass: event $e: colour $(gcol[e]) vs $(col[e])")
                if pass == :sampled && precision == :f64 && exact
                    ghel[e] == hel[e] || push!(bad, "$pass: event $e: helicity $(ghel[e]) vs $(hel[e])")
                end
            end
        end
        # a flavour other than 0: queued without complaint, reported by the next call
        flavour = zeros(Cint, stride); flavour[offset+3] = 1
        dflav, dme = upload(c, flavour), upload(c, zeros(stride))
        ins = [(UMAMI_IN_MOMENTA, d[:momenta]), (UMAMI_IN_FLAVOR_INDEX, dflav)]
        outs = [(UMAMI_OUT_MATRIX_ELEMENT, dme), (UMAMI_OUT_GPU_STREAM, stream[])]
        call_batch(gpu, N, stride, offset, ins, outs) == UMAMI_SUCCESS || push!(bad, "flavour 1: the queuing call failed")
        sync()
        gme = download!(c, fill(0.0, stride), dme)
        (isnan(gme[offset+3]) && all(isfinite, gme[offset+1:offset+2]) && all(isfinite, gme[offset+4:offset+N])) ||
            push!(bad, "flavour 1: not exactly that event's matrix element NaN")
        call_batch(gpu, N, stride, offset, ins[1:1], outs) == UMAMI_ERROR || push!(bad, "flavour 1: the next call did not report it")
        # reported once: the flag is cleared and the library goes on
        call_batch(gpu, N, stride, offset, ins[1:1], outs) == UMAMI_SUCCESS || push!(bad, "flavour 1: the call after the report failed")
        sync()
        all(isfinite, download!(c, fill(0.0, stride), dme)[offset+1:offset+N]) || push!(bad, "after the report: a non-finite matrix element")
        foreach(p -> release(c, p), (dflav, dme, values(d)...))
        ccall(dlsym(rt, :cudaStreamDestroy), Cint, (Ptr{Cvoid},), stream[])
    finally
        close_library(cpu)
        close_library(gpu)
    end
    quiet || println("$(basename(path)): CUDA provider ($precision) of $(l.process) $(l.colour), $N events: ",
                     isempty(bad) ? "as the CPU build" : "failures: " * join(first(bad, 8), "; "))
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(all(f -> isempty(check(f)), ARGS) ? 0 : 1)
end
