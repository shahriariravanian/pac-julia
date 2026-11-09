# The kernels' real MIR, lowered by symjit itself: `complexified(path)` is a saved application whose
# MIR symjit's Complexifier has lowered (symjit's Application::complexify, with siravan/symjit
# PR #32), written by the tool tools/sjb-complexify and kept in this package's scratch space
# ("complexified"), one file per input content (sha256). A real application is its own file.
# kernel_c and `read_sjb(path; lowered = true)` read it.
#
# install_complexify(): builds tools/sjb-complexify (cargo, `--locked`) into the scratch space's
# tools/bin, cargo's build folder removed; complexified calls it when the tool is missing.
# ENV["SJB_COMPLEXIFY"]: a built sjb-complexify to use instead.

const SJB_COMPLEXIFY_SOURCE = joinpath(dirname(@__DIR__), "tools", "sjb-complexify")

complexified_dir() = @get_scratch!("complexified")
sjb_complexify_path() = joinpath(tools_dir(), "bin", exe_name("sjb-complexify"))

# the tool's sources, hashed: an installed tool built from other sources is rebuilt
function sjb_complexify_stamp()::String
    files = ("Cargo.toml", "Cargo.lock", joinpath("src", "main.rs"))
    return bytes2hex(sha256(join(read(joinpath(SJB_COMPLEXIFY_SOURCE, f), String) for f in files)))
end

# a usable tool, or nothing
function find_sjb_complexify()::Union{String,Nothing}
    s = get(ENV, "SJB_COMPLEXIFY", nothing)
    s !== nothing && return isfile(s) ? s : nothing
    exe = sjb_complexify_path()
    stamp = exe * ".stamp"
    return isfile(exe) && isfile(stamp) && read(stamp, String) == sjb_complexify_stamp() ? exe : nothing
end

"""
    install_complexify(; force = false)

Builds sjb-complexify (tools/sjb-complexify: symjit's `Application::complexify` on saved
applications) in this package's scratch space; `complexified` calls it when the tool is missing.
Needs Rust's cargo (and the network, for the symjit crate).
"""
function install_complexify(; force::Bool = false)::String
    if !force
        exe = find_sjb_complexify()
        exe !== nothing && return exe
    end
    need("cargo", "to build sjb-complexify: install Rust from rustup.rs")
    exe = sjb_complexify_path()
    target = mktempdir()
    try
        @info "install_complexify: building sjb-complexify (cargo, a minute the first time)"
        run(addenv(Cmd(`cargo build --release --locked`; dir = SJB_COMPLEXIFY_SOURCE), "CARGO_TARGET_DIR" => target))
        built = joinpath(target, "release", exe_name("sjb-complexify"))
        isfile(built) || error("install_complexify: cargo did not build $built")
        mkpath(dirname(exe))
        cp(built, exe; force = true)
        write(exe * ".stamp", sjb_complexify_stamp())
    finally
        rm(target; recursive = true, force = true)
    end
    return exe
end

"""
    complexified(path) -> String

The saved application `path` (.sjb, .plane.symjit) with its MIR lowered to real MIR by symjit's
Complexifier: a file in this package's scratch space (made once per content), or `path` itself for
a real application. The lowered MIR has real registers only, but for LoadComplex/SaveComplex (a
pair of values at i, i + 1) and ComplexBi (+, -, ×, ÷ of register pairs). symjit does not load the
file again; `read_sjb` reads it.
"""
function complexified(path::AbstractString)::String
    bytes = read(path)
    out = joinpath(complexified_dir(), bytes2hex(sha256(bytes)) * ".plane.symjit")
    isfile(out) && return out
    is_complex(read_sjb(IOBuffer(bytes)).config) || return String(path)
    exe = install_complexify()
    # written beside and moved into place, so a file there is always whole
    tmp = tempname(complexified_dir())
    try
        run(`$exe $path $tmp`)
        is_complexified(read_sjb(tmp).mir) || error("complexified: sjb-complexify left $path's MIR complex")
        mv(tmp, out; force = true)
    finally
        rm(tmp; force = true)
    end
    return out
end
