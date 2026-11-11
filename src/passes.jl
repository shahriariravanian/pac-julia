# The optional passes over a Lowered (UIR v0.8). Each returns a new, validated Lowered whose values
# are the input's bit for bit; `optimize` runs all three.
#
#   sort_runs     each segment of consecutive vertex rows stably sorted by destination, when safe
#                 (pacuda's sort_contribution_runs), so the rows on one destination become adjacent
#   fuse          adjacent vertex rows on one destination joined into fused runs (FIRST ... LAST)
#   reuse_planes  the liveness layout: a current's planes are reused once it is dead; every current
#                 still starts at an even plane (not for a fixed layout: an imported program keeps its
#                 producer's, whose ranges may share planes)
#
# Order: sort_runs before fuse (it sorts single rows, not runs); reuse_planes may come anywhere, since
# a run's destination is live across the run, so no other current gets its planes meanwhile.

# a copy of `l` with some fields replaced
with(l::Lowered; kw...) = Lowered((haskey(kw, f) ? kw[f] : getfield(l, f) for f in fieldnames(Lowered))...)

# an edge's planes as a range of plane offsets
function plane_range(l::Lowered, e::EdgeIndex)::UnitRange{Int}
    c = edge_layout(l, e)
    return c.start:c.start+c.components-1
end
overlaps(a::UnitRange, b::UnitRange) = !isempty(intersect(a, b))

# the vertex rows that continue a fused run
fused_rows(l::Lowered) = count(o -> is_vertex(o.op) && !o.first, l.ops)

optimize(l::Lowered)::Lowered = reuse_planes(fuse(sort_runs(l)))

#################################### sort_runs #######################################

# A segment of consecutive vertex rows is sorted (stably, by its destinations' first planes) when no
# two different destinations share a plane and no row reads a plane any row of the segment writes.
# Then every destination keeps its own rows in their order, so its sums are unchanged, and no row
# sees another destination's partial value. The rows must be unfused (first and last).
function sort_runs(l::Lowered)::Lowered
    ops = copy(l.ops)
    k = 1
    while k <= length(ops)
        if !is_vertex(ops[k].op)
            k += 1
            continue
        end
        e = k
        while e <= length(ops) && is_vertex(ops[e].op)
            e += 1
        end
        seg = k:e-1
        k = e
        all(i -> ops[i].first && ops[i].last, seg) || error("sort_runs comes before fuse: a segment holds a fused run")
        dests = unique(ops[i].d for i in seg)
        ranges = [plane_range(l, d) for d in dests]
        safe = all(!overlaps(ranges[i], ranges[j]) for i in eachindex(ranges) for j in i+1:length(ranges))
        safe = safe && all(!overlaps(plane_range(l, x), r) for i in seg for x in (ops[i].l, ops[i].r) for r in ranges)
        safe || continue
        ops[seg] = sort(ops[seg]; by = o -> l.currents[edge_current(l, o.d)].start, alg = MergeSort)
    end
    return validate(with(l; ops = ops))
end

###################################### fuse ##########################################

# A vertex row joins the run of the vertex row before it when both write the same destination, it
# is "+=" (a "=" row starts its own run), and it reads none of the destination's planes.
function fuse(l::Lowered)::Lowered
    ops = copy(l.ops)
    for k in 2:length(ops)
        (a, b) = (ops[k-1], ops[k])
        is_vertex(a.op) && is_vertex(b.op) && a.d == b.d && !b.set || continue
        dest = plane_range(l, b.d)
        any(e -> overlaps(plane_range(l, e), dest), (b.l, b.r)) && continue
        ops[k-1] = LOp(a.op, a.set, a.first, false, a.d, a.t, a.l, a.r, a.n, a.x)
        ops[k] = LOp(b.op, b.set, false, b.last, b.d, b.t, b.l, b.r, b.n, b.x)
    end
    return validate(with(l; ops = ops))
end

################################### reuse_planes #####################################

# the components a row writes into its destination current, all of them without reading any: then the
# current needs no earlier value, and its planes may have held another current's
function writes_fresh(l::Lowered, op::LOp)::Bool
    n = edge_layout(l, op.d).components
    if is_vertex(op.op)
        return op.set && TYPE_COMPONENTS[OPCODE_SHAPES[op.op][3]] == n
    elseif is_kernel(op.op)
        (s, _) = l.instances[op.t]
        stub = l.stubs[s]
        return op.set && sort(stub.outputs) == 1:n && !any(i -> i.kind == OUT, stub.inputs)
    elseif op.op == CLEAR
        return op.n == n
    elseif op.op == SOURCE
        return op.l == 0 || sort([j for (j, _, _) in l.embeddings[op.l]]) == 1:n
    elseif op.op == COPY
        return true
    elseif op.op == SCATTER
        return op.set
    end
    return false
end

# the rows whose d is not an edge (an amplitude, an output)
no_destination(op::LOp) = op.op in (CLOSURE, CLEAR_AMPLITUDE, REDUCE)

# the edges a row reads as operands (a source's l is its embedding, not an edge)
operand_edges(op::LOp) = op.op == SOURCE || op.op == CLEAR_AMPLITUDE || op.op == REDUCE ? () : (op.l, op.r)

# the currents a row reads (a destination it updates counts as read)
function reads(l::Lowered, op::LOp)::Vector{CurrentIndex}
    cs = CurrentIndex[]
    for e in operand_edges(op)
        e == 0 || push!(cs, edge_current(l, e))
    end
    no_destination(op) || writes_fresh(l, op) || push!(cs, edge_current(l, op.d))
    return cs
end

# Each current is live from its first access to its last. A current whose first access is a fresh
# write (writes_fresh: a SOURCE, COPY or SET row that writes every component) starts there; every other
# one (a current that starts at zero or is read before written) is live from the start, so it gets planes nobody used before: zero, as the
# runners leave the arena. Before row k: the currents starting at k get planes (first fit, even
# blocks); after it, those last used at k free theirs. So a row never writes planes it reads.
# Returns the new layout and the peak of live planes (the bound no layout of this order beats).
function liveness_layout(l::Lowered)::Tuple{Vector{LCurrent},Int,Int}
    nc, nr = length(l.currents), length(l.ops)
    start = fill(-1, nc)     # the row a current becomes live (0: before the rows)
    stop = zeros(Int, nc)    # its last row
    for (k, op) in enumerate(l.ops)
        for c in reads(l, op)
            start[c] == -1 && (start[c] = 0)   # read before written: its zero start counts
            stop[c] = k
        end
        if !no_destination(op)
            c = edge_current(l, op.d)
            start[c] == -1 && (start[c] = writes_fresh(l, op) ? k : 0)
            stop[c] = k
        end
    end
    for c in 1:nc
        start[c] == -1 && (start[c] = 0)       # never used
        stop[c] = max(stop[c], start[c])
    end
    size(c) = l.currents[c].components + isodd(l.currents[c].components)

    free = UnitRange{Int}[]   # sorted, disjoint, even-aligned
    top = 0
    function alloc(n)
        for (i, r) in enumerate(free)
            length(r) >= n || continue
            free[i] = first(r)+n:last(r)
            isempty(free[i]) && deleteat!(free, i)
            return first(r)
        end
        top += n
        return top - n
    end
    function release(r)
        i = searchsortedfirst(free, r; by = first)
        insert!(free, i, r)
        if i < length(free) && last(free[i]) + 1 == first(free[i+1])     # join the next block
            free[i] = first(free[i]):last(free[i+1])
            deleteat!(free, i + 1)
        end
        if i > 1 && last(free[i-1]) + 1 == first(free[i])                # and the previous
            free[i-1] = first(free[i-1]):last(free[i])
            deleteat!(free, i)
        end
    end
    starting = [CurrentIndex[] for _ in 0:nr]
    ending = [CurrentIndex[] for _ in 0:nr]
    for c in 1:nc
        push!(starting[start[c]+1], c)
        push!(ending[stop[c]+1], c)
    end
    at = zeros(PlaneOffset, nc)
    (live, peak) = (0, 0)
    for t in 0:nr
        for c in starting[t+1]
            at[c] = alloc(size(c))
            live += size(c)
        end
        peak = max(peak, live)
        for c in ending[t+1]
            release(at[c]:at[c]+size(c)-1)
            live -= size(c)
        end
    end
    return ([LCurrent(at[c], cur.components, cur.type) for (c, cur) in enumerate(l.currents)], top, peak)
end

function reuse_planes(l::Lowered)::Lowered
    l.fixed_layout && return l
    (currents, planes, _) = liveness_layout(l)
    return validate(with(l; currents = currents, planes = planes))
end
