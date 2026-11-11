# The compiled back ends on imported recurrence exports: for programs of each export (all, or every
# k-th to keep at most MAX), the straight-line C, the UIR interpreter and, with nvcc and a GPU, its
# GPU build, each on the file's reference points against RustiCol's reference values bit for bit.
# usage: julia --project test/test_recurrence_export_c.jl [--max N] EXPORT.json...   (default N 12)
include(joinpath(@__DIR__, "compile_common.jl"))
using PacRunner
using PacRunner: output_count, LOp

const DRIVER = joinpath(@__DIR__, "compile_driver.c")
const GPU_DRIVER = joinpath(@__DIR__, "gpu_driver.cu")

text(momenta) = join([join([repr(x) for leg in m for x in leg], " ") for m in momenta], "\n") * "\n"
parse_rows(s) = [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in split(strip(s), "\n")]
same(x::Vector{Float64}, y::Vector{Float64}) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

function run_c(dir, l, momenta)
    exe = build(dir, DRIVER, joinpath(dir, "driver"), `-O2 -ffp-contract=off -Wall -Werror`)
    parse_rows(read(pipeline(`$exe $(l.legs) $(output_count(l)) $(arena_size(dir))`; stdin = IOBuffer(text(momenta))), String))
end

function run_gpu(dir, l, momenta, nvcc)
    run(pipeline(`make -s -C $dir gpu NVCC=$nvcc`; stderr = devnull))
    exe = joinpath(dir, "gpu_driver")
    run(`$nvcc -O3 -arch=native -o $exe $GPU_DRIVER $(joinpath(dir, "runner_gpu.o"))`)
    parse_rows(read(pipeline(`$exe $(l.legs) $(output_count(l))`; stdin = IOBuffer(text(momenta))), String))
end

# (programs checked, those every back end gives bit for bit, the failures)
function check(path::String; max::Int = 12, gpu = nothing)
    ls = load_recurrence_export(path; kernels = :none)
    refs = recurrence_references(path)
    picked = [k for k in eachindex(ls) if refs[k] !== nothing]
    step = cld(length(picked), max)
    picked = picked[1:step:end]
    bad = String[]
    for k in picked
        (l, (momenta, want)) = (ls[k], refs[k])
        for interpreter in (false, true)
            dir = compile(mktempdir(), l; interpreter)
            all(same.(run_c(dir, l, momenta), want)) || push!(bad, "$(l.process) $(l.colour): $(interpreter ? "interpreter" : "straight-line C")")
            if interpreter && gpu !== nothing
                got = run_gpu(dir, l, repeat(momenta, 32), gpu)
                all(same(got[j], want[mod1(j, length(want))]) for j in eachindex(got)) || push!(bad, "$(l.process) $(l.colour): GPU")
            end
        end
    end
    backs = gpu === nothing ? "straight-line C, interpreter" : "straight-line C, interpreter, GPU"
    println("$(basename(path)): $(length(picked)) programs ($backs): ", isempty(bad) ? "all bit for bit" : "differ: " * join(first(bad, 8), "; "))
    return (length(picked), bad)
end

# Direct kernel bindings in both modes. The exports have none yet: in the first program of `path`, the
# first add-mode KERNEL row whose output current already holds a value (so that set and add differ)
# becomes a KERNEL_DIRECT row, once in add mode and once in set mode; the straight-line C and the
# interpreter must give the Julia runner's bits (the kernels as C) on the file's reference points.
# Returns the failures.
function check_direct(path::String)
    ls = load_recurrence_export(path; kernels = :c)
    (l, (momenta, _)) = (ls[1], recurrence_references(path)[1])
    variant(i, set) = PacRunner.with(l; ops = [k == i ? LOp(PacRunner.KERNEL_DIRECT, set, o.first, o.last, o.d, o.t, o.l, o.r, o.n, o.x) : o
                                               for (k, o) in enumerate(l.ops)])
    i = findfirst(eachindex(l.ops)) do i
        o = l.ops[i]
        o.op == PacRunner.KERNEL && !o.set &&
            evaluate(variant(i, false), LArena(l), momenta[1]) != evaluate(variant(i, true), LArena(l), momenta[1])
    end
    i === nothing && error("$path: no add-mode kernel row adds to a value")
    bad = String[]
    for set in (false, true), interpreter in (false, true)
        v = variant(i, set)
        want = [evaluate(v, LArena(v), m) for m in momenta]
        got = run_c(compile(mktempdir(), v; interpreter), v, momenta)
        all(same.(got, want)) || push!(bad, "row $i direct $(set ? "set" : "add"): $(interpreter ? "interpreter" : "straight-line C")")
    end
    println("$(basename(path)): direct kernel rows (set, add; straight-line C, interpreter): ",
            isempty(bad) ? "all bit for bit" : "differ: " * join(bad, "; "))
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    max = 12
    if !isempty(args) && args[1] == "--max"
        max = parse(Int, args[2])
        args = args[3:end]
    end
    nvcc = something(Sys.which("nvcc"), "/usr/local/cuda/bin/nvcc")
    gpu = isfile(nvcc) && (try success(pipeline(`nvidia-smi -L`; stdout = devnull, stderr = devnull)) catch; false end) ? nvcc : nothing
    results = [check(f; max, gpu) for f in args]
    exit(all(isempty(r[2]) for r in results) ? 0 : 1)
end
