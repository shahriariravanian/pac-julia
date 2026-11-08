# The Julia runner's kernel libraries. A kernel of main.json with a library pac2json built
# (`--sjb2obj`) is called as it is; otherwise its library is built here from its .plane.symjit and
# kept in this package's scratch space (Scratch.jl, "kernels"), one per kernel content (sha256) and
# way of building:
#   :sjb2obj  symjit-utils' sjb2obj (ENV["SJB2OBJ"], or sjb2obj on the PATH), then cc -shared: the
#             native code symjit would run (the reference the tests compare with);
#   :c        kernel_c.jl's C, compiled by cc (ENV["CC"], default cc), bit for bit the same values;
#   :auto     (the default) :sjb2obj when sjb2obj is found, else :c;
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

# the library of the kernel at `source` (its sha256 `sha`) built the way `way` (:sjb2obj or :c),
# from the scratch space when it is there; nothing when the tool is missing
function kernel_library(source::String, sha::String, inputs::Int, outputs::Int, way::Symbol)::Union{String,Nothing}
    lib = joinpath(kernels_dir(), "$sha-$way.$(Libdl.dlext)")
    isfile(lib) && return lib
    cc = find_cc()
    cc === nothing && return nothing
    work = mktempdir(kernels_dir())
    try
        object = joinpath(work, "kernel.o")
        if way == :sjb2obj
            sjb2obj = find_sjb2obj()
            sjb2obj === nothing && return nothing
            run(pipeline(`$sjb2obj $source -o $object --name $KERNEL_SYMBOL`; stdout = devnull))
        else
            c = joinpath(work, "kernel.c")
            write(c, kernel_c_library_source(source, inputs, outputs))
            run(`$cc -std=c99 -O2 -ffp-contract=off -fPIC -c -o $object $c`)
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
