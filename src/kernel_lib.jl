# The Julia runner's kernel libraries. A kernel of main.json with a library pac2json built
# (`--sjb2obj`) is called as it is; otherwise its library is built here from its .plane.symjit and
# kept in this package's scratch space (Scratch.jl, "kernels"), one per build (build_key: the
# generated C, or the kernel's content with the sjb2obj binary, so a changed generator or tool builds
# anew) and way of building:
#   :sjb2obj  symjit-utils' sjb2obj (ENV["SJB2OBJ"], or sjb2obj on the PATH), then cc -shared: the
#             native code symjit would run (the reference the tests compare with);
#   :c        (the default) kernel_c.jl's C, compiled by cc (ENV["CC"], default cc), bit for bit the
#             same values; needs nothing but a C compiler;
#   :auto     :sjb2obj when sjb2obj is found, else :c;
#   :none     no library (a call of the kernel is an error; programs without kernel rows run).
# Every library exports KERNEL_SYMBOL with symjit's calling convention (kernel_call), whatever the
# kernel's name in its program.

const KERNEL_SYMBOL = "pacrunner_kernel"
const KERNEL_WAYS = (:auto, :sjb2obj, :c, :none)

kernels_dir() = @get_scratch!("kernels")
find_sjb2obj() = (s = get(ENV, "SJB2OBJ", Sys.which("sjb2obj"));
                  s !== nothing && isfile(s) ? s : nothing)
find_cc() = (c = get(ENV, "CC", "cc"); Sys.which(c) !== nothing ? c : nothing)

# kernel_c's kernel behind symjit's calling convention: int f(double *mem, const symjit_slice
# *states, size_t idx, const double *params), states = the input planes, then the output planes
function kernel_c_library_source(source::String, inputs::Int, outputs::Int)::String
    return KERNEL_PRELUDE * kernel_c("sj_kernel", source) * """

    typedef struct { double *ptr; size_t len; } sj_slice;
    int $KERNEL_SYMBOL(double *unused, const sj_slice *states, size_t idx, const double *params)
    {
        double mem[$(inputs + outputs + 1)];
        (void)unused; (void)params;
        for (int j = 0; j < $inputs; j++) mem[j] = states[j].ptr[idx];
        int status = sj_kernel(mem);
        for (int j = 0; j < $outputs; j++) states[$inputs + j].ptr[idx] = mem[$inputs + j];
        return status;
    }
    """
end

const KERNEL_CFLAGS = `-std=c99 -O2 -ffp-contract=off -fPIC`

# what a cached library was built from: for :c the generated C and the compiler command (so a
# change of kernel_c.jl, complexify.jl or the prelude builds anew), for :sjb2obj the kernel's
# content and the sjb2obj binary (its path, size and modification time)
function build_key(sha::String, way::Symbol; c_source = nothing, cc = nothing, sjb2obj = nothing)::String
    parts = if way == :sjb2obj
        st = stat(sjb2obj)
        ["sjb2obj", sha, abspath(sjb2obj), string(st.size), string(st.mtime), KERNEL_SYMBOL]
    else
        ["c", c_source, cc, string(KERNEL_CFLAGS)]
    end
    return bytes2hex(sha256(join(parts, "\0")))
end

# the library of the kernel at `source` (its sha256 `sha`) built the way `way` (:sjb2obj or :c),
# from the scratch space when it is there; nothing when the tool is missing
function kernel_library(source::String, sha::String, inputs::Int, outputs::Int, way::Symbol)::Union{String,Nothing}
    cc = find_cc()
    cc === nothing && return nothing
    sjb2obj = c_source = nothing
    if way == :sjb2obj
        sjb2obj = find_sjb2obj()
        sjb2obj === nothing && return nothing
    else
        c_source = kernel_c_library_source(source, inputs, outputs)
    end
    lib = joinpath(kernels_dir(), "$(build_key(sha, way; c_source, cc, sjb2obj))-$way.$(Libdl.dlext)")
    isfile(lib) && return lib
    work = mktempdir(kernels_dir())
    try
        object = joinpath(work, "kernel.o")
        if way == :sjb2obj
            run(pipeline(`$sjb2obj $source -o $object --name $KERNEL_SYMBOL`; stdout = devnull))
        else
            c = joinpath(work, "kernel.c")
            write(c, c_source)
            run(`$cc $KERNEL_CFLAGS -c -o $object $c`)
        end
        run(`$cc -shared -o $(joinpath(work, "kernel.$(Libdl.dlext)")) $object -lm`)
        mv(joinpath(work, "kernel.$(Libdl.dlext)"), lib; force = true)   # whole, or not at all
    finally
        rm(work; recursive = true, force = true)
    end
    return lib
end

# the way :auto means here
resolve_way(way::Symbol) = way == :auto ? (find_sjb2obj() !== nothing ? :sjb2obj : :c) : way
