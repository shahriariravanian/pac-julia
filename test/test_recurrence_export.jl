# load_recurrence_export against RustiCol: every program entry of each recurrence export (written
# with `pyamplicol export-recurrence ... --reference-points FILE`) run by the Julia runner of a
# Lowered on the file's reference points, every output against RustiCol's reference value bit for
# bit. Prints one line per file and the entries that differ (with the largest relative difference).
# usage: julia --project test/test_recurrence_export.jl [--kernels sjb2obj|c] [--optimize] EXPORT.json...
#   --optimize: each Lowered after `optimize` (sort_runs, fuse; reuse_planes keeps a fixed layout)
using PacRunner

same(x::Float64, y::Float64) = reinterpret(UInt64, x) == reinterpret(UInt64, y) || (isnan(x) && isnan(y))

# (entries, entries bit for bit, values, values bit for bit, the differing entries' worst relative error)
function check(path::String; kernels::Symbol = :auto, passes = false, quiet = false)
    ls = load_recurrence_export(path; kernels)
    passes && (ls = optimize.(ls))
    refs = recurrence_references(path)
    (n, ok, nv, okv) = (0, 0, 0, 0)
    bad = Tuple{String,Float64}[]
    for (l, r) in zip(ls, refs)
        r === nothing && continue
        n += 1
        a = LArena(l)
        good = true
        worst = 0.0
        for (m, want) in zip(r...)
            got = evaluate(l, a, m)
            for (x, y) in zip(got, want)
                nv += 1
                if same(x, y)
                    okv += 1
                else
                    good = false
                    worst = max(worst, abs(x - y) / max(abs(y), floatmin()))
                end
            end
        end
        good ? (ok += 1) : push!(bad, ("$(l.process) $(l.colour)", worst))
    end
    if !quiet
        println("$(basename(path)): $ok of $n programs bit for bit ($okv of $nv values)")
        for (what, w) in first(bad, 10)
            println("  differs: $what (worst relative $w)")
        end
    end
    return (n, ok, nv, okv, bad)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    kernels = :auto
    if !isempty(args) && args[1] == "--kernels"
        kernels = Symbol(args[2])
        args = args[3:end]
    end
    passes = !isempty(args) && args[1] == "--optimize"
    passes && (args = args[2:end])
    results = [check(f; kernels, passes) for f in args]
    total = sum(r[1] for r in results; init = 0)
    good = sum(r[2] for r in results; init = 0)
    println("all: $good of $total programs bit for bit")
    exit(good == total ? 0 : 1)
end
