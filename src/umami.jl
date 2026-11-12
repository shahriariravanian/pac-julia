# UMAMI libraries for MadSpace, one per program: umami_libraries(export, metadata, dir) writes, for
# each (process, colour flow, leg permutation) a pyAmpliCol artifact's UMAMI provider uses
# (API/umami/metadata.json: its channels, each channel's flavours, each flavour one contribution),
# a shared library made of the program's compiled C (compile: runner.c, pacops.c) and
# c/umami_pac.c, a thin UMAMI layer: the permutation, the helicity tables, the factor, the colour
# index, the masses and the exported model parameters are constants of the library.
#
# The flavour splitting is MadSpace's: dir/umami_pac.json lists the libraries and, per channel, the
# library of each flavour (`flavor_libraries`) and the channel's libraries (`libraries`). An
# integrand per channel loads flavour f's library as a matrix element of its own, with one
# differential cross section per flavour:
#   flavor_diff_xs_indices[f] = f,
#   flavor_remap[f] = 0 (each library sees flavour 0),
# flavor_per_subproc_remap and flavor_subproc_indices as for the stock provider (the event output's
# flavour and subprocess ids). MadSpace splits each batch by flavour itself; a library never sees
# another flavour's events.
#
# device = :cuda: the program runs on a CUDA GPU (the UIR interpreter, c/uirgpu.c, built by nvcc);
# the library still identifies as a CPU provider (host pointers, MadSpace's CPU threads).
#
# The programs come from RustiCol's recurrence export (load_recurrence_export) of the same artifact,
# with the same model parameters: a library refuses other values (umami_set_parameter).

const UMAMI_C = joinpath(dirname(@__DIR__), "c", "umami_pac.c")
const UMAMI_H = joinpath(dirname(@__DIR__), "c", "umami.h")

struct UmamiLibrary
    process::String              # the provider's runtime process (the export's process_key)
    colour::String               # its colour flow (colour_id)
    permutation::Vector{Int}     # runtime leg j = input leg permutation[j] (0-based)
    colour_index::Int
    factor::Float64
    helicity_export::Vector{Int} # runtime helicity → exported helicity (0-based)
end

# libraries are equal when every field is (the default == of a struct compares vectors by identity)
key(lib::UmamiLibrary) = Tuple(getfield(lib, f) for f in fieldnames(UmamiLibrary))
Base.:(==)(a::UmamiLibrary, b::UmamiLibrary) = key(a) == key(b)
Base.hash(lib::UmamiLibrary, h::UInt) = hash(key(lib), h)

clit(x::Float64) = isfinite(x) ? repr(x) : error("not finite: $x")

# umami_program.h of `lib`, whose program is `l`
function umami_program_h(lib::UmamiLibrary, l::Lowered, provider, exported_count::Int, parameters)::String
    what = "$(lib.process) $(lib.colour)"
    exported_count > 0 || error("$what: the provider exports no helicities")
    isempty(lib.helicity_export) && error("$what: no runtime helicities")
    runtime = fill(-1, exported_count)
    for (r, e) in enumerate(lib.helicity_export)
        0 <= e < exported_count || error("$what: runtime helicity $(r - 1) is exported helicity $e, not in 0:$(exported_count - 1)")
        runtime[e+1] == -1 || error("$what: exported helicity $e is runtime helicity $(runtime[e+1]) and $(r - 1)")
        runtime[e+1] = r - 1
    end
    (alpha_s0, power) = l.alpha_s === nothing ? (0.0, nothing) : l.alpha_s
    scales = power !== nothing && alpha_s0 > 0   # values at another alpha_s: × (alpha_s / alpha_s0)^power
    masses = Float64.(provider["masses"])
    out = IOBuffer()
    println(out, "/* $(lib.process) $(lib.colour), permutation $(lib.permutation): written by PacRunner's umami_libraries */")
    println(out, "#define PAC_PARTICLES $(Int(provider["particle_count"]))")
    println(out, "#define PAC_HELICITY_COUNT $(Int(provider["helicity_count"]))")
    println(out, "#define PAC_EXPORTED_HELICITY_COUNT $exported_count")
    println(out, "#define PAC_COLOR_COUNT $(Int(provider["color_count"]))")
    println(out, "#define PAC_COLOR_INDEX $(lib.colour_index)")
    println(out, "#define PAC_FACTOR $(clit(lib.factor))")
    println(out, "#define PAC_ALPHA_S_SCALES $(Int(scales))")
    println(out, "#define PAC_ALPHA_S0 $(clit(alpha_s0))")
    println(out, "#define PAC_ALPHA_S_POWER $(scales ? power : 0)")
    println(out, "static const double pac_masses[] = {", join(clit.(masses), ", "), "};")
    println(out, "static const size_t pac_permutation[] = {", join(lib.permutation, ", "), "};")
    println(out, "static const int pac_helicity_runtime[] = {", join(runtime, ", "), "};")
    println(out, "static const int pac_helicity_export[] = {", join(lib.helicity_export, ", "), "};")
    println(out, "static const PacParameter pac_parameters[] = {")
    for (name, z) in parameters
        println(out, "    {", cstring(name), ", ", clit(real(z)), ", ", clit(imag(z)), "},")
    end
    println(out, "    {NULL, 0.0, 0.0},")
    println(out, "};")
    return String(take!(out))
end

const UMAMI_DEVICES = (:cpu, :cuda)

find_nvcc() = (c = get(ENV, "NVCC", something(Sys.which("nvcc"), "/usr/local/cuda/bin/nvcc"));
               Sys.which(c) !== nothing || isfile(c) ? c : nothing)

const PAC_CUDA_CU = joinpath(dirname(@__DIR__), "c", "pac_cuda.cu")
const PAC_CUDA_H = joinpath(dirname(@__DIR__), "c", "pac_cuda.h")

# the CUDA runtime library's file next to nvcc (libcudart.so.N, its soname)
function find_cudart(nvcc::String)::String
    bin = dirname(something(Sys.which(nvcc), nvcc))
    for lib in (joinpath(bin, "..", "lib64"), joinpath(bin, "..", "targets", "x86_64-linux", "lib"))
        isdir(lib) || continue
        names = filter(f -> occursin(r"^libcudart\.so\.\d+$", f), readdir(lib))
        isempty(names) || return realpath(joinpath(lib, first(names)))   # the file it links to
    end
    error("no libcudart.so.N next to $nvcc")
end

# the process-wide GPU pool (pac_cuda.cu) in `dir`: libpac_cuda.so, with a copy of the CUDA runtime
# it and the program libraries share (libcudart.so.N, found through rpath \$ORIGIN)
function build_cuda_runtime(dir::String; nvcc::String = something(find_nvcc(), "nvcc"), cuda_arch::String = "native")::String
    so = joinpath(dir, "libpac_cuda.$(Libdl.dlext)")
    cudart = find_cudart(nvcc)
    m = match(r"^(libcudart\.so\.\d+)", basename(cudart))
    cp(cudart, joinpath(dir, m === nothing ? basename(cudart) : m.captures[1]); force = true)
    cp(PAC_CUDA_H, joinpath(dir, "pac_cuda.h"); force = true)
    run(`$nvcc -O3 -arch=$cuda_arch -shared -Xcompiler -fPIC -cudart shared -Xlinker -soname=libpac_cuda.$(Libdl.dlext)
         -Xlinker -rpath -Xlinker \$ORIGIN -I$dir -o $so $PAC_CUDA_CU`)
    return so
end

# the shared library of `lib` in `dir` (compile's folder plus the UMAMI layer). device = :cuda: the
# interpreter's runner.c built by nvcc for `cuda_arch` (--fmad=false: the CPU's bits), on the
# process-wide GPU slot pool of libpac_cuda.so in `runtime_dir` (built there if missing; one CUDA
# runtime, libcudart.so.N, shared through rpath); every symbol but the UMAMI functions hidden. The
# pool the library asks for (the first library of a process decides): `gpu_slots` slots (0: one
# per CPU) of `gpu_memory` device bytes each (a call needing more runs as several launches).
function build_umami_library(dir::String, lib::UmamiLibrary, l::Lowered, provider, exported_count, parameters;
                             cc::String = something(find_cc(), "cc"), cflags::Cmd = `-O2`, interpreter::Bool = false,
                             device::Symbol = :cpu, nvcc::Union{String,Nothing} = find_nvcc(),
                             cuda_arch::String = "native", nvflags::Cmd = `-O3`,
                             gpu_memory::Real = GPU_MEMORY_DEFAULT, gpu_slots::Integer = 0,
                             runtime_dir::String = dir)::String
    device in UMAMI_DEVICES || error("device $device: one of $UMAMI_DEVICES")
    cuda = device == :cuda
    cuda && nvcc === nothing && error("device = :cuda needs nvcc (ENV[\"NVCC\"], the PATH or /usr/local/cuda/bin)")
    compile(dir, l; interpreter = interpreter || cuda)
    write(joinpath(dir, "umami_program.h"), umami_program_h(lib, l, provider, exported_count, parameters))
    cp(UMAMI_C, joinpath(dir, "umami_pac.c"); force = true)
    cp(UMAMI_H, joinpath(dir, "umami.h"); force = true)
    so = joinpath(dir, "libumami_pac.$(Libdl.dlext)")
    if cuda
        isfile(joinpath(runtime_dir, "libpac_cuda.$(Libdl.dlext)")) || build_cuda_runtime(runtime_dir; nvcc, cuda_arch)
        file(f) = joinpath(dir, f)
        rpath = "\$ORIGIN/" * relpath(abspath(runtime_dir), abspath(dir))
        run(`$nvcc $nvflags -arch=$cuda_arch --fmad=false -Xcompiler -fPIC,-fvisibility=hidden -DPAC_CUDA_EXTERNAL_POOL
             -DUIRGPU_SLOTS=$gpu_slots -DUIRGPU_SLOT_BYTES=$(Int(gpu_memory)) -x cu -c -o $(file("runner_gpu.o")) $(file("runner.c"))`)
        run(`$cc $cflags -std=c99 -ffp-contract=off -fPIC -fvisibility=hidden -DPAC_CUDA -c
             -o $(file("umami_pac.o")) $(file("umami_pac.c"))`)
        run(`$nvcc -shared -arch=$cuda_arch -cudart shared -o $so $(file("umami_pac.o")) $(file("runner_gpu.o"))
             -L$runtime_dir -lpac_cuda -Xlinker -rpath -Xlinker $rpath -lm`)
    else
        sources = [joinpath(dir, f) for f in ("umami_pac.c", "runner.c", "pacops.c")]
        run(`$cc $cflags -std=c99 -ffp-contract=off -fPIC -fvisibility=hidden -shared -o $so $sources -lm`)
    end
    return so
end

"""
    umami_libraries(export, metadata, dir; optimize_programs = true, interpreter = false, cc, cflags = `-O2`,
                    device = :cpu, cuda_arch = "native", nvcc, gpu_slots = 0, gpu_memory = 256 MiB,
                    quiet = false) -> String

One UMAMI library per (process, colour flow, permutation) of the artifact's UMAMI provider
(`metadata`: the artifact's API/umami/metadata.json), its program from the recurrence export
`export` of the same artifact; written into `dir`, with `dir/umami_pac.json` (the libraries and
each channel's flavours, for MadSpace). Returns the path of umami_pac.json. `optimize_programs`:
the programs after `optimize` (the default; the same values) or as imported. `interpreter`: each
program as the UIR interpreter's data (`compile(...; interpreter = true)`) instead of straight-line
C; `cc` and `cflags` build the libraries. `device = :cuda`: each program runs on a CUDA GPU (the
interpreter, built by `nvcc` for `cuda_arch`, default "native": this machine's GPU), the libraries
still CPU providers for MadSpace. The GPU memory and streams are a process-wide pool
(`dir/libpac_cuda.so`, with a copy of the CUDA runtime): `gpu_slots` slots (default 0: one per CPU,
so every MadSpace thread can run at once; 1: one call at a time) of `gpu_memory` device bytes
each, allocated when first used (a call needing more runs as several launches). A call borrows any
free slot and waits while all are busy. The libraries are built in parallel.
"""
function umami_libraries(export_path::AbstractString, metadata_path::AbstractString, dir::AbstractString;
                         optimize_programs::Bool = true, interpreter::Bool = false,
                         cc::String = something(find_cc(), "cc"), cflags::Cmd = `-O2`,
                         device::Symbol = :cpu, cuda_arch::String = "native", nvcc = find_nvcc(),
                         gpu_slots::Integer = 0, gpu_memory::Real = GPU_MEMORY_DEFAULT,
                         quiet::Bool = false)::String
    d = read_export(export_path)
    meta = JSON.parse(read(metadata_path, String))
    get(meta, "kind", nothing) == "pyamplicol-umami-metadata" || error("$metadata_path: not a pyAmpliCol UMAMI metadata file")
    provider = meta["provider"]
    provider["artifact_id"] == d["artifact_id"] ||
        error("the export is of artifact $(d["artifact_id"]), the metadata of $(provider["artifact_id"])")
    provider["color_accuracy"] == "lc" || error("the provider is $(provider["color_accuracy"]), not lc")
    exported_count = length(meta["helicities"])
    parameters = [(String(p["name"]), cvalue(p["value"])) for p in d["parameters"]]

    # the programs by (process_key, colour_id), with their helicity ids
    ls = load_recurrence_export(export_path; kernels = :none)
    programs = Dict{Tuple{String,String},Tuple{Lowered,Vector{String}}}()
    for (l, p) in zip(ls, d["programs"])
        programs[(l.process, l.colour)] = (l, [String(h["id"]) for h in p["helicity_outputs"]])
    end

    procs = meta["runtime_processes"]
    libraries = UmamiLibrary[]
    channels = []
    for (c, channel) in enumerate(meta["channels"])
        flavours = Int[]
        for (f, flavour) in enumerate(channel["processes"])
            r = flavour["runtime"]
            process = String(procs[Int(r["process_index"])+1]["id"])
            lib = UmamiLibrary(process, String(r["color_id"]), Int.(r["permutation"]), Int(r["color_index"]),
                               Float64(r["factor"]), Int.(r["helicity_export"]))
            haskey(programs, (lib.process, lib.colour)) ||
                error("channel $(c - 1) flavour $(f - 1): no program $(lib.process) $(lib.colour) in $export_path")
            (l, ids) = programs[(lib.process, lib.colour)]
            want = [String(h) for h in procs[Int(r["process_index"])+1]["helicity_ids"]]
            ids == want || error("$(lib.process) $(lib.colour): the export's helicities are not the provider's, in order")
            length(lib.helicity_export) == output_count(l) ||
                error("$(lib.process) $(lib.colour): $(length(lib.helicity_export)) exported helicities for $(output_count(l)) outputs")
            push!(flavours, intern!(libraries, lib))
        end
        push!(channels, flavours)
    end

    mkpath(dir)
    device == :cuda && build_cuda_runtime(dir; nvcc, cuda_arch)
    # the libraries' folders and programs first (Julia work), then their builds (compilers) in parallel
    jobs = map(enumerate(libraries)) do (k, lib)
        sub = joinpath(dir, "programs", lpad(k - 1, 4, '0'))
        mkpath(sub)
        l = programs[(lib.process, lib.colour)][1]
        (k, lib, sub, optimize_programs ? optimize(l) : l)
    end
    sos = asyncmap(jobs; ntasks = Sys.CPU_THREADS) do (k, lib, sub, l)
        so = build_umami_library(sub, lib, l, provider, exported_count, parameters;
                                 cc, cflags, interpreter, device, cuda_arch, nvcc, gpu_memory, gpu_slots, runtime_dir = abspath(dir))
        quiet || println("library $(k - 1) of $(length(libraries)): $(lib.process) $(lib.colour) $(lib.permutation)")
        so
    end
    entries = [Dict("index" => k - 1, "path" => relpath(so, dir), "process" => lib.process, "colour" => lib.colour,
                    "permutation" => lib.permutation, "colour_index" => lib.colour_index, "factor" => lib.factor)
               for ((k, lib, _, _), so) in zip(jobs, sos)]
    table = Dict(
        "kind" => "pacrunner-umami-libraries", "schema_version" => 1, "device" => String(device), "gpu_slots" => gpu_slots, "gpu_memory" => gpu_memory,
        "artifact_id" => d["artifact_id"], "export" => abspath(export_path), "metadata" => abspath(metadata_path),
        "particle_count" => provider["particle_count"], "libraries" => entries,
        "channels" => [Dict("flavor_libraries" => [k - 1 for k in fl], "libraries" => sort(unique(fl)) .- 1) for fl in channels],
    )
    path = joinpath(dir, "umami_pac.json")
    write(path, JSON.json(table; pretty = true))
    return path
end
