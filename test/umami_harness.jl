# PacRunner's CPU UMAMI libraries (umami_libraries) against the stock pyAmpliCol library, called
# directly through the UMAMI C API the way MadSpace's CPU runtime calls a matrix element
# (madspace/src/cpu/runtime.cpp op_matrix_element, src/cpu/device.hpp AsyncCpuDevice::foreach):
#
#   - a batch of B events in column layout (input[event + B*(...)]), split into jobs aligned to the
#     SIMD width (job_count_and_size), each job one umami_matrix_element(count, stride = B, offset)
#     call on its worker's instance, the jobs run concurrently;
#   - one UMAMI instance per worker thread and loaded matrix element (MadSpace's ThreadResource),
#     kept across batches;
#   - the LC pass of mg7's [amplicol] subprocess (subprocess_amplicol.py): inputs momenta, flavour,
#     channel, random helicity (+ alpha_s when the stock library supports it), outputs matrix
#     element, colour index, helicity index;
#   - the stock library: one matrix element per channel (mg7 today), the channel's batch with mixed
#     flavours (flavour input per event);
#   - PacRunner's libraries: MadSpace's own flavour split (phasespace/integrand.cpp, the
#     multi-diff_xs path), with one differential cross section per flavour
#     (flavor_diff_xs_indices[f] = f, flavor_remap[f] = 0): one matrix element loaded per flavour
#     (its own instances, even when flavours share a library), batch_split_by_index +
#     batch_gather give each flavour's events as a batch of their own (flavour input 0), and the
#     flavours' matrix-element instructions run concurrently (CpuRuntime::run_concurrent: their
#     jobs share the thread pool), the results merged back by index (batch_merge_by_index);
#     --sequential runs the flavours one after another instead (a non-concurrent runtime);
#   - optionally the fixed-helicity pass (helicity input, matrix element output).
#
# Compared bit for bit: matrix element, colour index, helicity index (with a running alpha_s that
# the export rescales, the matrix element within --rtol and the indices may differ where the
# rescaling moves a sampled helicity's boundary: counted, not failed).
#
# usage: julia -t THREADS --project test/umami_harness.jl UMAMI_PAC_JSON STOCK_LIBUMAMI_SO ARTIFACT_DIR
#            [--batches 1,3,17,64,257,1000] [--rounds 2] [--warmup 0] [--vec 8] [--fixed] [--sequential] [--seed N]
#            [--rtol 1e-13]
# (umami_pac.json from umami_libraries; the stock library from `make` in ARTIFACT_DIR/API/umami, as
# artifacts/.pyamplicol-api-build/NAME/umami/libumami.so; ARTIFACT_DIR is its param card argument)
using JSON
using Libdl
using Random

# umami.h
const UMAMI_SUCCESS = Cint(0)
const IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR, IN_RANDOM_HELICITY, IN_HELICITY, IN_CHANNEL = Cint.((0, 1, 2, 4, 6, 8))
const OUT_ME, OUT_COLOR, OUT_HELICITY = Cint.((0, 2, 3))
const META_PARTICLE_COUNT, META_MASSES = Cint.((1, 5))

# one library's entry points
struct Api
    handle::Ptr{Cvoid}
    initialize::Ptr{Cvoid}
    matrix_element::Ptr{Cvoid}
    free::Ptr{Cvoid}
    supported_inputs::Ptr{Cvoid}
    get_meta::Ptr{Cvoid}
    param_card::String
end

function Api(path::String, param_card::String)
    h = dlopen(path, RTLD_LOCAL | RTLD_NOW)
    Api(h, dlsym(h, :umami_initialize), dlsym(h, :umami_matrix_element), dlsym(h, :umami_free),
        dlsym(h, :umami_supported_inputs), dlsym(h, :umami_get_meta), param_card)
end

function supports(api::Api, key::Cint)::Bool
    p = Ref{Ptr{Bool}}(C_NULL); n = Ref{Cint}(0)
    ccall(api.supported_inputs, Cint, (Ptr{Ptr{Bool}}, Ptr{Cint}), p, n) == UMAMI_SUCCESS || error("umami_supported_inputs")
    return key < n[] && unsafe_load(p[], key + 1)
end

function meta_int(api::Api, key::Cint)::Int
    r = Ref{Cint}(0)
    ccall(api.get_meta, Cint, (Cint, Ptr{Cvoid}), key, r) == UMAMI_SUCCESS || error("umami_get_meta $key")
    return r[]
end

function masses(api::Api, n::Int)::Vector{Float64}
    m = zeros(n)
    ccall(api.get_meta, Cint, (Cint, Ptr{Cvoid}), META_MASSES, m) == UMAMI_SUCCESS || error("umami_get_meta masses")
    return m
end

function new_instance(api::Api)::Ptr{Cvoid}
    r = Ref{Ptr{Cvoid}}(C_NULL)
    s = ccall(api.initialize, Cint, (Ptr{Ptr{Cvoid}}, Cstring), r, api.param_card)
    s == UMAMI_SUCCESS || error("umami_initialize: status $s")
    return r[]
end

# a loaded matrix element (MadSpace's MatrixElementApi): a library and its own instances, `id`
# distinguishing two loads of one library (two flavours sharing a program)
struct Me
    api::Api
    id::Int
end

# the worker threads' instances: instances[worker][me.id] (MadSpace's ThreadResource)
const Instances = Vector{Dict{Int,Tuple{Api,Ptr{Cvoid}}}}
instance!(instances::Instances, w::Int, me::Me) = get!(() -> (me.api, new_instance(me.api)), instances[w], me.id)[2]

function free_all(instances::Instances)
    for d in instances, (api, p) in values(d)
        ccall(api.free, Cint, (Ptr{Cvoid},), p)
    end
end

# MadSpace's AsyncCpuDevice::job_count_and_size: (job count, job size)
function jobs(batch::Int, threads::Int, vec::Int)
    batch == 0 && return (0, 0)
    bv = cld(batch, vec)
    min_batch = max(64 ÷ vec, 1)
    count = bv < threads * min_batch ? cld(bv, min_batch) : threads
    size = cld(bv, count) * vec
    return (cld(batch, size), size)
end

# a batch's columns: keys and arrays (each Vector{Float64} or Vector{Cint}, element `event` of each
# column block at event + B*block); outputs likewise
struct Batch
    size::Int
    in_keys::Vector{Cint}
    inputs::Vector{Vector}
    out_keys::Vector{Cint}
    outputs::Vector{Vector}
end

function Batch(size::Int, in_keys, inputs, out_keys)
    outputs = Vector[k == OUT_ME ? fill(NaN, size) : fill(Cint(-7), size) for k in out_keys]
    Batch(size, Cint.(in_keys), inputs, Cint.(out_keys), outputs)
end

# matrix-element instructions, each a batch on a loaded matrix element, run as MadSpace's
# concurrent CPU runtime runs independent instructions: every instruction's jobs
# (job_count_and_size of its own batch) in one pool, taken by the worker threads in order, each job
# on its worker's instance of that matrix element. The statuses that were not UMAMI_SUCCESS, per
# instruction.
function run!(calls::Vector{Tuple{Batch,Me}}, instances::Instances, vec::Int)::Vector{Vector{Cint}}
    queue = Tuple{Int,Int,Int}[]   # (call, count, offset)
    for (k, (b, _)) in enumerate(calls)
        (njobs, size) = jobs(b.size, length(instances), vec)
        for j in 0:njobs-1
            push!(queue, (k, min(size, b.size - j * size), j * size))
        end
    end
    pool = Channel{Tuple{Int,Int,Int}}(length(queue))
    foreach(j -> put!(pool, j), queue)
    close(pool)
    bad = [Cint[] for _ in calls]
    lock = ReentrantLock()
    GC.@preserve calls begin
        in_ptrs = [Ptr{Cvoid}[pointer(x) for x in b.inputs] for (b, _) in calls]
        out_ptrs = [Ptr{Cvoid}[pointer(x) for x in b.outputs] for (b, _) in calls]
        @sync for w in eachindex(instances)
            Threads.@spawn for (k, count, offset) in pool
                (b, me) = calls[k]
                inst = instance!(instances, w, me)
                s = ccall(me.api.matrix_element, Cint,
                          (Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, Csize_t, Ptr{Cint}, Ptr{Ptr{Cvoid}}, Csize_t, Ptr{Cint}, Ptr{Ptr{Cvoid}}),
                          inst, count, b.size, offset, length(b.in_keys), b.in_keys, in_ptrs[k], length(b.out_keys), b.out_keys, out_ptrs[k])
                s == UMAMI_SUCCESS || @lock lock push!(bad[k], s)
            end
        end
    end
    return bad
end

# random momenta (on shell, not conserved: the comparison does not need physical points), column
# layout momenta[event + B*(particle + n*component)]
function momenta(rng, ms::Vector{Float64}, B::Int)::Vector{Float64}
    n = length(ms)
    m = zeros(B * n * 4)
    for e in 1:B, j in 1:n
        p = rand(rng, 3) .* 600 .- 300
        m[e + B * (j - 1)] = sqrt(sum(abs2, p) + ms[j]^2)
        for k in 1:3
            m[e + B * ((j - 1) + n * k)] = p[k]
        end
    end
    return m
end

# the rows `events` (1-based) of a column-layout array of B rows
function gather(x::Vector{T}, B::Int, events::Vector{Int})::Vector{T} where {T}
    blocks = length(x) ÷ B
    b = length(events)
    y = Vector{T}(undef, b * blocks)
    for k in 0:blocks-1, (i, e) in enumerate(events)
        y[i + b * k] = x[e + B * k]
    end
    return y
end

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

mutable struct Tally
    batches::Int
    events::Int
    calls::Int
    me_differ::Int
    index_differ::Int
    worst::Float64
    failures::Vector{String}
    jobs::Int          # PacRunner's umami_matrix_element calls (flavours × thread jobs)
    job_events::Int    # their events
    stock_time::Float64   # seconds in the stock library's instructions
    pac_time::Float64     # seconds in PacRunner's (the gather and merge not included)
end
Tally() = Tally(0, 0, 0, 0, 0, 0.0, String[], 0, 0, 0.0, 0.0)

# one channel's batch of B events: `stock` the channel's stock matrix element, `flavours[f + 1]`
# the matrix element loaded for flavour f
function check_channel!(t::Tally, c::Int, stock::Me, flavours::Vector{Me}, instances::Instances, rng, B::Int;
                        vec::Int, fixed::Bool, sequential::Bool, alpha_s0, rtol::Float64, helicities::Int)
    nf = length(flavours)
    mom = momenta(rng, stock_masses[], B)
    flav = Cint.(rand(rng, 0:nf-1, B))
    chan = fill(Cint(c), B)
    rnd = rand(rng, B)
    running = alpha_s0 === nothing ? nothing : alpha_s0 .* (0.8 .+ 0.4 .* rand(rng, B))
    label = "channel $c, batch $B"

    passes = [("sampled", [IN_MOMENTA, IN_FLAVOR, IN_CHANNEL, IN_RANDOM_HELICITY], Vector[mom, flav, chan, rnd],
               [OUT_ME, OUT_COLOR, OUT_HELICITY], false)]
    if running !== nothing
        push!(passes, ("sampled, alpha_s = alpha_s0", [IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR, IN_CHANNEL, IN_RANDOM_HELICITY],
                       Vector[mom, fill(alpha_s0, B), flav, chan, rnd], [OUT_ME, OUT_COLOR, OUT_HELICITY], false))
        push!(passes, ("sampled, running alpha_s", [IN_MOMENTA, IN_ALPHA_S, IN_FLAVOR, IN_CHANNEL, IN_RANDOM_HELICITY],
                       Vector[mom, running, flav, chan, rnd], [OUT_ME, OUT_COLOR, OUT_HELICITY], true))
    end
    if fixed
        push!(passes, ("fixed helicity", [IN_MOMENTA, IN_FLAVOR, IN_CHANNEL, IN_HELICITY],
                       Vector[mom, flav, chan, Cint.(rand(rng, 0:helicities-1, B))], [OUT_ME], false))
    end

    for (name, keys, inputs, outs, tolerant) in passes
        # stock: the channel's batch, mixed flavours
        s = Batch(B, keys, inputs, outs)
        t.stock_time += @elapsed (bad = run!([(s, stock)], instances, vec)[1])
        isempty(bad) || (push!(t.failures, "$label, $name: stock status $(unique(bad))"); continue)
        t.calls += 1
        # PacRunner: MadSpace's flavour split (batch_split_by_index, batch_gather per flavour), the
        # flavours' instructions concurrently, then batch_merge_by_index
        p = Batch(B, keys, inputs, outs)
        split = [(f, findall(==(f), flav)) for f in 0:nf-1]
        filter!(x -> !isempty(x[2]), split)
        calls = Tuple{Batch,Me}[]
        for (f, events) in split
            sub_inputs = Vector[k == IN_FLAVOR ? zeros(Cint, length(events)) : gather(x, B, events) for (k, x) in zip(keys, inputs)]
            push!(calls, (Batch(length(events), keys, sub_inputs, outs), flavours[f+1]))
        end
        t.pac_time += @elapsed (bad = sequential ? [run!([call], instances, vec)[1] for call in calls] : run!(calls, instances, vec))
        t.calls += length(calls)
        t.jobs += sum(c -> jobs(c[1].size, length(instances), vec)[1], calls; init = 0)
        t.job_events += sum(c -> c[1].size, calls; init = 0)
        for ((f, events), (sub, _), st) in zip(split, calls, bad)
            isempty(st) || (push!(t.failures, "$label, $name: flavour $f status $(unique(st))"); continue)
            for (o, x) in zip(p.outputs, sub.outputs), (i, e) in enumerate(events)
                o[e] = x[i]
            end
        end
        # compare
        for (k, x, y) in zip(outs, s.outputs, p.outputs)
            if k == OUT_ME
                for e in 1:B
                    same(x[e], y[e]) && continue
                    rel = abs(x[e] - y[e]) / abs(x[e])
                    t.worst = max(t.worst, rel)
                    if !tolerant || !(rel <= rtol)
                        t.me_differ += 1
                        length(t.failures) < 20 && push!(t.failures, "$label, $name: event $e flavour $(flav[e]): ME $(x[e]) vs $(y[e]) (rel $rel)")
                    end
                end
            else
                d = count(e -> x[e] != y[e], 1:B)
                d == 0 && continue
                if tolerant
                    t.index_differ += d
                else
                    t.me_differ += d
                    push!(t.failures, "$label, $name: $(k == OUT_COLOR ? "colour" : "helicity") index differs at $d events")
                end
            end
        end
    end
    t.batches += 1
    t.events += B
end

const stock_masses = Ref{Vector{Float64}}()

function main(args)
    length(args) >= 3 || (println(stderr, "usage: julia -t N --project test/umami_harness.jl UMAMI_PAC_JSON STOCK_SO ARTIFACT_DIR [options]"); return 2)
    (table_path, stock_path, artifact) = args[1:3]
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 4
    while i <= length(args)
        a = args[i]
        if a in ("--fixed", "--sequential")
            push!(flags, a); i += 1
        else
            opts[a] = args[i+1]; i += 2
        end
    end
    batches = parse.(Int, split(get(opts, "--batches", "1,3,17,64,257,1000"), ","))
    rounds = parse(Int, get(opts, "--rounds", "2"))
    warmup = parse(Int, get(opts, "--warmup", "0"))   # untimed rounds first (checked as well)
    vec = parse(Int, get(opts, "--vec", "8"))
    seed = parse(Int, get(opts, "--seed", "20261111"))
    rtol = parse(Float64, get(opts, "--rtol", "1e-13"))
    fixed = "--fixed" in flags
    sequential = "--sequential" in flags

    table = JSON.parse(read(table_path, String))
    base = dirname(abspath(table_path))
    # the metadata and export paths: absolute, or relative to umami_pac.json's folder (a moved bundle)
    path(x) = isabspath(x) ? x : joinpath(base, x)
    meta = JSON.parse(read(path(table["metadata"]), String))
    helicities = length(meta["helicities"])
    stock = Api(abspath(stock_path), abspath(artifact))
    libs = [Api(joinpath(base, l["path"]), "") for l in table["libraries"]]
    n = meta_int(stock, META_PARTICLE_COUNT)
    n == table["particle_count"] || error("stock library: $n particles, umami_pac.json $(table["particle_count"])")
    stock_masses[] = masses(stock, n)
    for (k, l) in enumerate(libs)
        meta_int(l, META_PARTICLE_COUNT) == n || error("library $(k-1): particle count")
        masses(l, n) == stock_masses[] || error("library $(k-1): masses differ from the stock library's")
    end

    alpha_s0 = nothing
    if supports(stock, IN_ALPHA_S)
        all(l -> supports(l, IN_ALPHA_S), libs) || error("the stock library takes alpha_s, a PacRunner library does not")
        export_doc = JSON.parse(read(path(table["export"]), String))
        a = findfirst(p -> p["alpha_s"] !== nothing, export_doc["programs"])
        a === nothing || (alpha_s0 = Float64(export_doc["programs"][a]["alpha_s"]["value"]["re"]))
    end

    # the loaded matrix elements: the stock library once per channel (one Integrand per channel),
    # PacRunner's library once per (channel, flavour) (one differential cross section per flavour)
    ids = Ref(0)
    me(api) = Me(api, ids[] += 1)
    channels = table["channels"]
    stock_mes = [me(stock) for _ in channels]
    flavour_mes = [[me(libs[k+1]) for k in Int.(ch["flavor_libraries"])] for ch in channels]

    threads = Threads.nthreads()
    instances = Instances([Dict{Int,Tuple{Api,Ptr{Cvoid}}}() for _ in 1:threads])
    rng = Xoshiro(seed)
    t = Tally()
    untimed_events = 0
    elapsed = @elapsed try
        for r in 1:warmup+rounds
            if r == warmup + 1   # the timed rounds start: every instance set up, its buffers grown
                t.stock_time = t.pac_time = 0.0
                untimed_events = t.events
            end
            for B in batches, c in eachindex(channels)
                check_channel!(t, c - 1, stock_mes[c], flavour_mes[c], instances, rng, B;
                               vec, fixed, sequential, alpha_s0, rtol, helicities)
            end
        end
    finally
        free_all(instances)
    end
    nflav = sum(ch -> length(ch["flavor_libraries"]), table["channels"])
    println("$(basename(base)): $(length(table["channels"])) channels, $nflav flavours, $(length(libs)) libraries; ",
            "$threads threads, SIMD width $vec, batches $(join(batches, ",")) × $rounds rounds$(warmup > 0 ? " after $warmup warm-up" : ""); ",
            "alpha_s ", alpha_s0 === nothing ? "not supported by the stock library" : "at $alpha_s0 and running",
            fixed ? "; fixed-helicity pass" : "", sequential ? "; flavours sequential" : "; flavours concurrent")
    println("  $(t.batches) batches, $(t.events) events, $(t.calls) batch calls in $(round(elapsed; digits = 1)) s: ",
            t.me_differ == 0 && isempty(t.failures) ? "all bit for bit" : "$(t.me_differ) differences")
    t.jobs > 0 && println("  PacRunner: $(t.jobs) umami_matrix_element calls, $(round(t.job_events / t.jobs; digits = 1)) events per call on average")
    println("  time in the matrix elements: stock $(round(t.stock_time; digits = 3)) s, PacRunner $(round(t.pac_time; digits = 3)) s over $(t.events - untimed_events) events", warmup > 0 ? " (timed rounds only)" : " (the first calls include their setup)")
    alpha_s0 === nothing || println("  running alpha_s: largest relative ME difference $(t.worst), $(t.index_differ) sampled indices differ")
    for f in first(t.failures, 20)
        println("  ", f)
    end
    return isempty(t.failures) && t.me_differ == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
