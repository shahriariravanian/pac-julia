# pac2json, built and run for the user: install() builds it, capture(artifact) runs it on a pyAmpliCol
# LC artifact once and keeps the result, and the loaders (load_programs, load_artifact,
# load_examples) take an artifact directory as they take a pac2json folder.
#
# install(): pacuda (PACUDA_REPO at PACUDA_COMMIT, which pins the RustiCol pac2json records with)
# into this package's scratch space ("tools"); its patched RustiCol prepared as pacuda's
# rusticol/prepare.sh does, here in Julia so that it runs on Linux and macOS alike (git, tar and
# patch; no sha256sum): the pyamplicol/ directory of MadGraph7 at rusticol/UPSTREAM, with
# rusticol/patches; then `cargo build --release --bin pac2json` (pacuda's rust-toolchain.toml
# names the toolchain; rustup installs it), the binary kept in tools/bin and cargo's build folder
# removed. Only RustiCol's Rust crate is built: no Python, no
# pyAmpliCol build, no Symbolica.
#
# capture(artifact): the folder pac2json writes for the artifact, in the scratch space
# ("captures"), named by the artifact's own artifact_id (artifact.json), with a suffix for
# parameters other than the artifact's. Made in a temporary folder and moved into place, so a
# folder there is always whole.

const PACUDA_REPO = "https://github.com/siravan/pacuda.git"
const PACUDA_COMMIT = "957dc16fde238d0961b0c21c306c9a52b20d7835"
const MADGRAPH7_REPO = "https://github.com/MadGraphTeam/MadGraph7.git"

tools_dir() = @get_scratch!("tools")
captures_dir() = @get_scratch!("captures")
pacuda_dir() = joinpath(tools_dir(), "pacuda")
exe_name(name) = Sys.iswindows() ? name * ".exe" : name
pac2json_path() = joinpath(tools_dir(), "bin", exe_name("pac2json"))

function need(tool::String, why::String)
    Sys.which(tool) === nothing && error("install: `$tool` is not on the PATH ($why)")
end

# git's output, its errors shown
git(dir, args...) = run(`git -C $dir $args`)

"""
    install(; force = false, madgraph7 = nothing)

Builds pac2json (pacuda at `PACUDA_COMMIT`, with its patched RustiCol) in this package's scratch
space; `capture` and the loaders call it when pac2json is missing. `madgraph7`: a local clone of
MadGraph7 that has the RustiCol commit pacuda pins (else it is fetched from GitHub). Needs git,
tar, patch, a C compiler and Rust's cargo (rustup installs the toolchain pacuda names).
"""
function install(; force::Bool = false, madgraph7::Union{String,Nothing} = nothing)
    exe = pac2json_path()
    stamp = joinpath(pacuda_dir(), "PACRUNNER_INSTALLED")
    !force && isfile(exe) && isfile(stamp) && read(stamp, String) == PACUDA_COMMIT && return exe
    for (tool, why) in (("git", "to fetch pacuda and pyAmpliCol"), ("tar", "to unpack pyAmpliCol"),
                        ("patch", "to patch RustiCol"), ("cargo", "to build pac2json: install Rust from rustup.rs"))
        need(tool, why)
    end
    dir = pacuda_dir()
    # pacuda at the pinned commit
    if !isdir(joinpath(dir, ".git"))
        rm(dir; recursive = true, force = true)
        mkpath(dir)
        git(dir, "init", "-q")
        git(dir, "remote", "add", "origin", PACUDA_REPO)
    end
    git(dir, "fetch", "-q", "--depth", "1", "origin", PACUDA_COMMIT)
    git(dir, "checkout", "-q", "--force", "FETCH_HEAD")
    prepare_rusticol(dir, madgraph7)
    @info "install: building pac2json (cargo, a few minutes the first time)"
    convert = joinpath(dir, "convert")
    run(Cmd(`cargo build --release --bin pac2json`; dir = convert))
    built = joinpath(convert, "target", "release", exe_name("pac2json"))
    isfile(built) || error("install: cargo did not build $built")
    # only the binary is kept: cargo's build folder is about half a gigabyte
    mkpath(dirname(exe))
    cp(built, exe; force = true)
    rm(joinpath(convert, "target"); recursive = true, force = true)
    write(stamp, PACUDA_COMMIT)
    return exe
end

# pacuda's rusticol/prepare.sh, in Julia: build/rusticol = MadGraph7's pyamplicol/ at
# rusticol/UPSTREAM with rusticol/patches (kept while UPSTREAM and the patches are the same)
function prepare_rusticol(dir::String, madgraph7::Union{String,Nothing})
    here = joinpath(dir, "rusticol")
    base = strip(read(joinpath(here, "UPSTREAM"), String))
    patches = sort(filter(f -> endswith(f, ".patch"), readdir(joinpath(here, "patches"); join = true)))
    want = base * " " * bytes2hex(sha256(join(read.(patches, String))))
    out = joinpath(dir, "build", "rusticol")
    stamp = joinpath(out, "PREPARED")
    isfile(stamp) && read(stamp, String) == want && return out
    rm(out; recursive = true, force = true)
    mkpath(out)
    source = madgraph7
    tmp = nothing
    if source === nothing
        tmp = mktempdir()
        source = tmp
        git(tmp, "init", "-q")
        @info "install: fetching pyAmpliCol (MadGraph7 at $(base[1:12]))"
        git(tmp, "fetch", "-q", "--depth", "1", "--filter=blob:none", MADGRAPH7_REPO, base)
    end
    try
        run(pipeline(`git -C $source archive $base pyamplicol`, `tar -x -C $out`))
    finally
        tmp === nothing || rm(tmp; recursive = true, force = true)
    end
    for p in patches
        run(`patch -s -p1 -N -d $out -i $p`)   # (no GNU-only options: macOS's patch too)
    end
    write(stamp, want)
    return out
end

# pac2json, installed when missing
pac2json() = isfile(pac2json_path()) ? pac2json_path() : install()

# ---- artifacts ------------------------------------------------------------------------------------

# a pyAmpliCol artifact directory (artifact.json) or not
is_artifact(path::AbstractString) = isdir(path) && isfile(joinpath(path, "artifact.json"))

# The artifact's id, after the checks that it is one pac2json records: a pyAmpliCol process (or
# process-set) artifact of the schema the pinned RustiCol reads (3), leading colour only, run by RustiCol, with
# its UMAMI provider header (the processes and colours pac2json records)
function check_artifact(dir::AbstractString)::String
    is_artifact(dir) || error("$dir: not a pyAmpliCol artifact (no artifact.json)")
    a = JSON.parse(read(joinpath(dir, "artifact.json"), String))
    what = "$dir"
    get(a, "kind", nothing) in ("pyamplicol-process", "pyamplicol-process-set") ||
        error("$what: not a pyAmpliCol process or process-set artifact (kind $(get(a, "kind", nothing)))")
    get(a, "schema_version", nothing) == 3 ||
        error("$what: artifact schema $(get(a, "schema_version", nothing)); pac2json's RustiCol reads schema 3")
    producer = get(a, "producer", nothing)
    if producer !== nothing && get(producer, "versions", nothing) !== nothing
        v = get(producer.versions, "process_artifact", nothing)
        v == 3 || error("$what: process artifact version $v (pyAmpliCol $(get(producer, "version", "?"))); pac2json's RustiCol reads 3")
    end
    runtime = get(a, "runtime", nothing)
    runtime !== nothing && get(runtime, "engine", nothing) != "rusticol" &&
        error("$what: run by $(get(runtime, "engine", nothing)), not RustiCol")
    for p in get(a, "processes", [])
        get(p, "color_accuracy", "lc") == "lc" ||
            error("$what: process $(get(p, "id", "?")) is $(p.color_accuracy) colour; pac2json records leading colour (LC) artifacts")
    end
    isfile(joinpath(dir, "API", "umami", "provider_p0.h")) ||
        error("$what: no API/umami/provider_p0.h (the artifact's UMAMI provider: generate it with the API bundle)")
    id = get(a, "artifact_id", nothing)
    id isa AbstractString && occursin(r"^[0-9a-f]{64}$", id) || error("$what: no artifact_id in artifact.json")
    return String(id)
end

"""
    capture(artifact; parameters = nothing, force = false) -> folder

The pac2json folder (main.json, examples.json, kernels/) of the pyAmpliCol LC artifact directory
`artifact`: recorded once by pac2json (installed when missing) into this package's scratch space
and found there afterwards, by the artifact's artifact_id. `parameters`: a JSON file of model
parameters ({"name": value}) to record with instead of the artifact's (pac2json --parameters).
"""
function capture(artifact::AbstractString; parameters::Union{AbstractString,Nothing} = nothing, force::Bool = false)::String
    artifact = realpath(abspath(artifact))   # RustiCol refuses an artifact path through a symlink
    id = check_artifact(artifact)
    key = parameters === nothing ? id : id * "-" * bytes2hex(sha256(read(parameters)))[1:16]
    dest = joinpath(captures_dir(), key)
    !force && isfile(joinpath(dest, "main.json")) && return dest
    exe = pac2json()
    work = mktempdir(captures_dir())
    try
        out = joinpath(work, key)
        args = parameters === nothing ? String[] : ["--parameters", abspath(parameters)]
        @info "capture: recording $artifact with pac2json"
        # pac2json's report (and its note that it builds no kernel libraries: kernel_lib.jl does)
        # goes to a log, shown when it fails
        log = joinpath(work, "pac2json.log")
        ok = success(pipeline(`$exe $artifact -o $out $args`; stdout = log, stderr = log))
        ok && isfile(joinpath(out, "main.json")) || error("capture: pac2json failed on $artifact:\n" * read(log, String))
        rm(dest; recursive = true, force = true)
        mv(out, dest)
    finally
        rm(work; recursive = true, force = true)
    end
    return dest
end

# main.json of `path`: main.json itself, a pac2json folder, or a pyAmpliCol artifact (captured)
function main_json(path::AbstractString; parameters = nothing)::String
    is_artifact(path) && return joinpath(capture(path; parameters), "main.json")
    isdir(path) && return joinpath(path, "main.json")
    return String(path)
end

# examples.json of `path`, as main_json finds main.json (or examples.json itself)
function examples_json(path::AbstractString; parameters = nothing)::String
    basename(path) == "examples.json" && return String(path)
    return joinpath(dirname(main_json(path; parameters)), "examples.json")
end
