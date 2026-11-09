# UIR v0.8's new rows on every back end. Until RustiCol's export is imported, no lowered program has
# COPY, SCATTER, CLEAR_AMPLITUDE, embedded sources or REDUCE rows with an amplitude scale; this test
# builds one from a real program (a kernel-free program of a pac2json folder), in two runs:
#   run 1: the program's rows, every REDUCE with an amplitude scale;
#   CLEAR_AMPLITUDE of every amplitude;
#   run 2: its sources as embedded sources (the first one's embedding reversed, with a zero), its
#          body, then COPY and SCATTER rows on four vector currents (every scatter shortcut: 1, -1, a
#          real, an imaginary and a general factor; set and add) read by one more closure, and the
#          REDUCE rows again (plus one for the new closure);
# and requires the same bits from the Julia runner of a Lowered (plain, with reused planes, decoded
# from its encoding), the straight-line C, the UIR interpreter and, with a GPU, its GPU build.
# The values mean nothing physically; the back ends must agree on them.
# usage: julia --project test/test_uir08.jl [PAC2JSON_DIR]     (default: test/data/ppjjj_crossed)
include(joinpath(@__DIR__, "compile_common.jl"))
using PacRunner
using PacRunner: LOp, Lowered, with, SOURCE, REDUCE, CLOSURE, CLEAR_AMPLITUDE, COPY, SCATTER, VECTOR, source_fields,
                 TYPE_COMPONENTS, edge_current, intern!, intern_value!, decode, encode, validate, output_count

const DRIVER = joinpath(@__DIR__, "compile_driver.c")
const GPU_DRIVER = joinpath(@__DIR__, "gpu_driver.cu")

same(x::Vector{Float64}, y::Vector{Float64}) = reinterpret(UInt64, x) == reinterpret(UInt64, y)

# the two-run program built from `l` (lowered, kernel-free)
function synthetic(l::Lowered)::Lowered
    factors = copy(l.factors)
    f(z) = intern_value!(factors, complex(Float64(real(z)), Float64(imag(z))))
    (f_scale, f_one, f_neg, f_re, f_im, f_gen, f_zero) =
        f.((0.5 - 0.25im, 1.0 + 0.0im, -1.0 + 0.0im, -2.0 + 0.0im, 0.0 + 1.5im, 0.3 + 0.7im, 0.0 + 0.0im))
    embeddings = Vector{PacRunner.EmbeddingEntry}[]
    with_scale(o::LOp) = LOp(REDUCE; d = o.d, t = o.t, n = o.n, x = f_scale)
    run1 = [o.op == REDUCE ? with_scale(o) : o for o in l.ops]
    clears = [LOp(CLEAR_AMPLITUDE; d = a) for a in 1:l.amplitudes]
    # run 2's sources: embedded (the first: reversed, its last entry zero; the others: identity)
    first_source = true
    run2 = LOp[]
    for o in l.ops
        o.op == REDUCE && continue
        if o.op == SOURCE
            n = TYPE_COMPONENTS[source_fields(o)[1]]
            list = first_source ? [(j, j == n ? 0 : n + 1 - j, o.x) for j in 1:n] : [(j, j, o.x) for j in 1:n]
            first_source = false
            push!(embeddings, list)
            push!(run2, LOp(SOURCE; d = o.d, t = o.t, n = o.n, l = length(embeddings)))
        else
            push!(run2, o)
        end
    end
    # four vector currents (their own edges): x = a × gen, x += b i 1.5, x += a × -2, x += -b; y = a,
    # y += b × gen (general), y = y set from x (1); then a closure of x and y into amplitude 1
    vectors = [c for (c, cur) in enumerate(l.currents) if cur.type == VECTOR]
    length(vectors) >= 4 || error("$(l.process) $(l.colour): fewer than 4 vector currents")
    (a, b, x, y) = vectors[1:4]
    extra = [LOp(COPY; d = x, l = a, x = f_gen), LOp(SCATTER; d = x, l = b, x = f_im),
             LOp(SCATTER; d = x, l = a, x = f_re), LOp(SCATTER; d = x, l = b, x = f_neg),
             LOp(SCATTER; set = true, d = y, l = a, x = f_one), LOp(SCATTER; d = y, l = b, x = f_gen),
             LOp(COPY; d = b, l = y, x = f_zero),   # a zero factor: b = +0
             LOp(SCATTER; d = y, l = b, x = f_neg),
             LOp(CLEAR_AMPLITUDE; d = 1)]
    k = findfirst(o -> o.op == CLOSURE && length(l.lists[o.t]) == 4, l.ops)
    k === nothing && error("$(l.process) $(l.colour): no closure of 4 components")
    push!(extra, LOp(CLOSURE; d = 1, t = l.ops[k].t, l = x, r = y, x = l.ops[k].x))
    weights = copy(l.weights)
    reduces = [with_scale(o) for o in l.ops if o.op == REDUCE]
    push!(reduces, LOp(REDUCE; d = 1, t = 1, n = intern_value!(weights, 0.125), x = 0))
    ops = vcat(run1, clears, run2, extra, reduces)
    return validate(with(l; factors, embeddings, weights, ops))
end

# the outputs of the compiled folder `dir` (runner.c with the driver) on `points`
function run_lowered(dir::String, l::Lowered, points)::Vector{Vector{Float64}}
    exe = build(dir, DRIVER, joinpath(dir, "driver"), `-O2 -ffp-contract=off -Wall -Werror`)
    text = read(pipeline(`$exe $(l.legs) $(output_count(l)) $(arena_size(dir))`; stdin = IOBuffer(points_text(points))), String)
    return [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in split(strip(text), "\n")]
end

# the same on the GPU build of an interpreter folder (nvcc)
function run_lowered_gpu(dir::String, l::Lowered, points, nvcc::String)::Vector{Vector{Float64}}
    run(pipeline(`make -s -C $dir gpu NVCC=$nvcc`; stderr = devnull))
    exe = joinpath(dir, "gpu_driver")
    run(`$nvcc -O3 -arch=native -o $exe $GPU_DRIVER $(joinpath(dir, "runner_gpu.o"))`)
    text = read(pipeline(`$exe $(l.legs) $(output_count(l))`; stdin = IOBuffer(points_text(points))), String)
    return [[reinterpret(Float64, parse(UInt64, h; base = 16)) for h in split(r)] for r in split(strip(text), "\n")]
end

# The back ends that differ from the Julia runner of the synthetic program of `case`'s first
# kernel-free program (empty: all agree); `gpu`: nvcc's path, or nothing
function check(case::String; gpu = nothing, quiet = false)::Vector{String}
    ps = load_programs(joinpath(case, "main.json"))
    ex = read_main(joinpath(case, "examples.json")).programs
    n = findfirst(p -> !any(op -> op isa PacRunner.Kernel, p.body), ps)
    n === nothing && error("$case: no kernel-free program")
    s = synthetic(lower(ps[n]))
    points = ex[n].points
    momenta = [[Tuple(Float64.(v)) for v in pt.momenta] for pt in points]
    want = [evaluate(s, LArena(s), m) for m in momenta]
    all(v -> all(isfinite, v), want) || error("the synthetic program's values are not finite")
    bad = String[]
    for (name, v) in (("reused planes", reuse_planes(s)), ("decoded", decode(encode(s), s)))
        all(same(evaluate(v, LArena(v), m), w) for (m, w) in zip(momenta, want)) || push!(bad, "Julia, $name")
    end
    if Sys.which(CC) !== nothing
        for interpreter in (false, true)
            dir = compile(mktempdir(), s; interpreter)
            all(same.(run_lowered(dir, s, points), want)) || push!(bad, interpreter ? "the interpreter" : "the straight-line C")
            if interpreter && gpu !== nothing
                got = run_lowered_gpu(dir, s, repeat(points, 64), gpu)
                all(same(got[k], want[mod1(k, length(want))]) for k in eachindex(got)) || push!(bad, "the GPU")
            end
        end
    end
    quiet || println("$(s.process) $(s.colour): $(length(s.ops)) rows ($(count(o -> o.op in (COPY, SCATTER), s.ops)) copy/scatter, ",
                     "$(length(s.embeddings)) embeddings), $(length(points)) points: ",
                     isempty(bad) ? "every back end agrees bit for bit" : "differ: " * join(bad, ", "))
    return bad
end

if abspath(PROGRAM_FILE) == @__FILE__
    case = isempty(ARGS) ? joinpath(@__DIR__, "data", "ppjjj_crossed") : ARGS[1]
    nvcc = something(Sys.which("nvcc"), "/usr/local/cuda/bin/nvcc")
    gpu = isfile(nvcc) && (try success(pipeline(`nvidia-smi -L`; stdout = devnull, stderr = devnull)) catch; false end) ? nvcc : nothing
    exit(isempty(check(case; gpu)) ? 0 : 1)
end
