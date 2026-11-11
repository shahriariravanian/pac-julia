# UMAMI libraries for MadSpace, one per program: umami_libraries(export, metadata, dir) writes, for
# each (process, colour flow, leg permutation) a pyAmpliCol artifact's UMAMI provider uses
# (API/umami/metadata.json: its channels, each channel's flavours, each flavour one contribution),
# a shared library made of the program's compiled C (compile: runner.c, pacops.c) and
# c/umami_pac.c, a thin UMAMI layer: the permutation, the helicity tables, the factor, the colour
# index, the masses and the exported model parameters are constants of the library.
#
# The flavour splitting is MadSpace's: dir/umami_pac.json lists the libraries and, per channel, the
# library of each flavour (`flavor_libraries`) and the channel's libraries (`libraries`), so that an
# integrand of the channel has one differential cross section per library of the channel and
#   flavor_diff_xs_indices[f] = position of flavor_libraries[f] in libraries,
#   flavor_remap[f] = 0 (each library sees flavour 0).
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
    runtime = fill(-1, exported_count)
    for (r, e) in enumerate(lib.helicity_export)
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

# the shared library of `lib` in `dir` (compile's folder plus the UMAMI layer)
function build_umami_library(dir::String, lib::UmamiLibrary, l::Lowered, provider, exported_count, parameters;
                             cc::String = something(find_cc(), "cc"), cflags::Cmd = `-O2`, interpreter::Bool = false)::String
    compile(dir, l; interpreter)
    write(joinpath(dir, "umami_program.h"), umami_program_h(lib, l, provider, exported_count, parameters))
    cp(UMAMI_C, joinpath(dir, "umami_pac.c"); force = true)
    cp(UMAMI_H, joinpath(dir, "umami.h"); force = true)
    so = joinpath(dir, "libumami_pac.$(Libdl.dlext)")
    sources = [joinpath(dir, f) for f in ("umami_pac.c", "runner.c", "pacops.c")]
    run(`$cc $cflags -std=c99 -ffp-contract=off -fPIC -fvisibility=hidden -shared -o $so $sources -lm`)
    return so
end

"""
    umami_libraries(export, metadata, dir; optimize_programs = true, interpreter = false, cc, cflags = `-O2`,
                    quiet = false) -> String

One UMAMI library per (process, colour flow, permutation) of the artifact's UMAMI provider
(`metadata`: the artifact's API/umami/metadata.json), its program from the recurrence export
`export` of the same artifact; written into `dir`, with `dir/umami_pac.json` (the libraries and
each channel's flavours, for MadSpace). Returns the path of umami_pac.json. `optimize_programs`:
the programs after `optimize` (the default; the same values) or as imported. `interpreter`: each
program as the UIR interpreter's data (`compile(...; interpreter = true)`) instead of straight-line
C; `cc` and `cflags` build the libraries.
"""
function umami_libraries(export_path::AbstractString, metadata_path::AbstractString, dir::AbstractString;
                         optimize_programs::Bool = true, interpreter::Bool = false,
                         cc::String = something(find_cc(), "cc"), cflags::Cmd = `-O2`, quiet::Bool = false)::String
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
    entries = []
    for (k, lib) in enumerate(libraries)
        sub = joinpath(dir, "programs", lpad(k - 1, 4, '0'))
        mkpath(sub)
        l = programs[(lib.process, lib.colour)][1]
        so = build_umami_library(sub, lib, optimize_programs ? optimize(l) : l, provider, exported_count, parameters;
                                 cc, cflags, interpreter)
        quiet || println("library $(k - 1) of $(length(libraries)): $(lib.process) $(lib.colour) $(lib.permutation)")
        push!(entries, Dict("index" => k - 1, "path" => relpath(so, dir), "process" => lib.process, "colour" => lib.colour,
                            "permutation" => lib.permutation, "colour_index" => lib.colour_index, "factor" => lib.factor))
    end
    table = Dict(
        "kind" => "pacrunner-umami-libraries", "schema_version" => 1,
        "artifact_id" => d["artifact_id"], "export" => abspath(export_path), "metadata" => abspath(metadata_path),
        "particle_count" => provider["particle_count"], "libraries" => entries,
        "channels" => [Dict("flavor_libraries" => [k - 1 for k in fl], "libraries" => sort(unique(fl)) .- 1) for fl in channels],
    )
    path = joinpath(dir, "umami_pac.json")
    write(path, JSON.json(table; pretty = true))
    return path
end
