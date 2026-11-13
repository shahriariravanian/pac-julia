# The GPU build of the UIR interpreter against its CPU build: every program of the pac2json folders
# (default: every case in test/data) compiled with interpreter = true, its runner.c built by nvcc
# (`make gpu`: runner_gpu.o, run_gpu) and by cc, both run on examples.json's points (repeated, so
# that a launch has many threads): the outputs must be the same bits. The kernels are C
# (kernel_c.jl), so no sjb2obj is needed.
# usage: julia --project test/test_gpu.jl [PAC2JSON_DIR...]     (NVCC: nvcc, default nvcc on the PATH)
include(joinpath(@__DIR__, "compile_common.jl"))
module GpuCompile include(joinpath(@__DIR__, "test_compile.jl")) end
using PacRunner

const NVCC = get(ENV, "NVCC", something(Sys.which("nvcc"), "nvcc"))
const GPU_DRIVER = joinpath(@__DIR__, "gpu_driver.cu")

# a usable GPU: nvcc and nvidia-smi listing a device
have_gpu() = Sys.which(NVCC) !== nothing || isfile(NVCC) ?
             (try success(pipeline(`nvidia-smi -L`; stdout = devnull, stderr = devnull)) catch; false end) : false

# the outputs of the GPU build of the compiled folder `dir` on `points`
function run_gpu_folder(dir::String, p::Program, points; nvflags::String = "")::Vector{Vector{Float64}}
    isempty(nvflags) || rm(joinpath(dir, "runner_gpu.o"); force = true)
    run(pipeline(`make -s -C $dir gpu NVCC=$NVCC $(isempty(nvflags) ? String[] : ["NVFLAGS=-O3 -arch=native $nvflags"])`; stderr = devnull))
    exe = joinpath(dir, "gpu_driver")
    run(`$NVCC -O3 -arch=native -o $exe $GPU_DRIVER $(joinpath(dir, "runner_gpu.o"))`)
    text = read(pipeline(`$exe $(p.legs) $(length(p.reduction))`; stdin = IOBuffer(points_text(points))), String)
    rows = split(strip(text), "\n")
    length(rows) == length(points) || error("the GPU driver printed $(length(rows)) lines for $(length(points)) points")
    return [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in rows]
end

# the folder `source`: the points whose GPU outputs differ from the CPU's
function test_folder(source::String; repeats::Int = 64, quiet = false)::Int
    ps = load_programs(joinpath(source, "main.json"))
    ex = read_main(joinpath(source, "examples.json")).programs
    differ = 0
    for (n, p) in enumerate(ps)
        dir = compile(mktempdir(), p; interpreter = true)
        pts = ex[n].points
        gpu = run_gpu_folder(dir, p, repeat(pts, repeats))
        cpu = GpuCompile.run_compiled(dir, p, pts)
        differ += count(k -> reinterpret(UInt64, gpu[k]) != reinterpret(UInt64, cpu[mod1(k, length(pts))]), eachindex(gpu))
    end
    quiet || println(basename(normpath(source)), ": $(length(ps)) programs, $(repeats) × examples.json's points each; ",
                     differ == 0 ? "the GPU's outputs are the CPU's bit for bit" : "$differ points differ")
    return differ
end

# the folder `source` on a GPU build with `nvflags` (e.g. "-DPAC_MIXED", pacops.h): per point, the
# relative error of the sum of its outputs (the helicity sum) against the CPU's (f64), and the
# largest error of one output relative to that sum (helicities that vanish exactly are ~1e-36 in f64
# but the f32 rounding residue in mixed)
function errors_folder(source::String; nvflags::String = "-DPAC_MIXED", repeats::Int = 1)
    ps = load_programs(joinpath(source, "main.json"))
    ex = read_main(joinpath(source, "examples.json")).programs
    errs, each = Float64[], Float64[]
    for (n, p) in enumerate(ps)
        dir = compile(mktempdir(), p; interpreter = true)
        pts = ex[n].points
        gpu = run_gpu_folder(dir, p, repeat(pts, repeats); nvflags)
        cpu = GpuCompile.run_compiled(dir, p, pts)
        for k in eachindex(gpu)
            x, y = cpu[mod1(k, length(pts))], gpu[k]
            t = sum(abs, x)
            push!(errs, t == 0 ? Float64(sum(abs, y) != 0) : abs(sum(x) - sum(y)) / t)
            push!(each, t == 0 ? Float64(sum(abs, y) != 0) : maximum(abs.(x .- y)) / t)
        end
    end
    return (errs, each)
end

if abspath(PROGRAM_FILE) == (@__FILE__) && get(ENV, "NVFLAGS_ERRORS", "") != ""
    for d in (isempty(ARGS) ? cases() : ARGS)
        (e, o) = sort.(errors_folder(d; nvflags = ENV["NVFLAGS_ERRORS"]))
        println(basename(normpath(d)), " ($(ENV["NVFLAGS_ERRORS"])): $(length(e)) points; sum: median ", e[cld(end, 2)],
                ", max ", e[end], "; one output: max ", o[end], "; NaN ", count(isnan, e))
    end
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    have_gpu() || (println("no nvcc ($NVCC) or no GPU (nvidia-smi -L)"); exit(2))
    dirs = isempty(ARGS) ? cases() : ARGS
    exit(sum(test_folder, dirs) == 0 ? 0 : 1)
end
