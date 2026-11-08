# kernel_c.jl against symjit-utils' sjb2obj: every kernel of the pac2json folders (default: every
# case in test/data) as C (kernel_c), built as a shared library with cc, against its sjb2obj library
# on random inputs (magnitudes 1e-3 to 1e3, signs both ways) and on special ones (zeros, signed
# zeros, a negative or zero real part only): the outputs must be the same bits, but that any two NaNs
# are the same (the sign of a NaN from 0/0 follows the operand order x86 sees, which cc may change).
# usage: julia --project test/test_kernel_c.jl SJB2OBJ [PAC2JSON_DIR...]
include(joinpath(@__DIR__, "compile_common.jl"))
using PacRunner
using PacRunner: KERNEL_PRELUDE, kernel_call, load_kernel
using Libdl
using Random

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y) || (isnan(x) && isnan(y))

# special inputs of n values: zeros, signed zeros, every other value negative or zero
function special_inputs(n::Int)::Vector{Vector{Float64}}
    return [zeros(n), fill(-0.0, n), [isodd(j) ? -1.5 : 0.0 for j in 1:n], [isodd(j) ? 0.0 : -2.0 for j in 1:n],
            [isodd(j) ? 1e-300 : -1e300 for j in 1:n], ones(n)]
end

# the kernels of the folder `dir` (with their sjb2obj libraries built): the points whose outputs differ
function compare_kernels(dir::String; points::Int = 2000, quiet = false)::Int
    main = read_main(joinpath(dir, "main.json"))
    rng = Xoshiro(11)
    differ = 0
    work = mktempdir()
    for k in main.kernels
        lib = load_kernel(joinpath(dir, k.library.path), k.library.symbol, k.library.states, k.library.outputs)
        src = joinpath(work, "$(k.name).c")
        write(src, KERNEL_PRELUDE * kernel_c(k.name, joinpath(dir, k.path)) * "\nint c_$(k.name)(double *mem) { return $(k.name)(mem); }\n")
        so = joinpath(work, "$(k.name).so")
        run(`$CC -std=c99 -O2 -ffp-contract=off -Wall -Werror -shared -fPIC -o $so $src -lm`)
        h = Libdl.dlopen(so)
        fn = Libdl.dlsym(h, "c_$(k.name)")
        n = lib.inputs
        inputs = vcat(special_inputs(n), [[randn(rng) * 10.0^rand(rng, -3:3) for _ in 1:n] for _ in 1:points])
        bad = 0
        for x in inputs
            want = kernel_call(lib, x)
            mem = vcat(x, zeros(lib.outputs + 1))
            status = ccall(fn, Cint, (Ptr{Float64},), mem)
            got = mem[n+1:n+lib.outputs]
            bad += status != 0 || !all(same.(got, want))
        end
        Libdl.dlclose(h)
        quiet || println("  kernel $(k.name): $(length(inputs)) points, $(bad == 0 ? "bit for bit" : "$bad differ")")
        differ += bad
    end
    return differ
end

# the folder `source` with its kernels' libraries built (prepare), compared
function test_folder(sjb2obj::String, source::String; quiet = false)::Int
    quiet || println(basename(normpath(source)), ":")
    return compare_kernels(prepare(sjb2obj, source; quiet = true); quiet)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 1 || (println("usage: julia --project test/test_kernel_c.jl SJB2OBJ [PAC2JSON_DIR...]"); exit(2))
    dirs = length(ARGS) > 1 ? ARGS[2:end] : cases()
    exit(sum(d -> test_folder(ARGS[1], d), dirs) == 0 ? 0 : 1)
end
