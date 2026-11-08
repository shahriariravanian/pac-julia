# encode(l::Lowered) -> Encoded: the device form of UIR v0.6 (design doc, "The 8-byte encoding (UIR
# v0.6)"): one stream of 64-bit words, the sources first (SOURCE rows), then the rows; and the edge
# table as 32-bit entries. The other tables (scales, factors, component-factor lists, stubs, scale
# sets, binding instances, forms, reduction) are the Lowered's.
#
# Indices are 0-based here (the Lowered's minus 1), the only place besides the C generator that
# converts. In an index field, all ones means none (the null operand of a kernel row).
#
# Word 0 of a row (no field crosses bit 32, so each half decodes with 32-bit bit-field operations):
#   bits 0-4 op, 5 EXT (word 1 follows), 6 SET, 7 FIRST, 8 LAST,
#   9-21 D (13), 22-31 T (10), 32-44 L (13), 45-57 R (13), 32-47 N (16, on L and R's bits), 58-63 X (6)
# Word 1 of a wide row (EXT), each field's high bits:
#   bit 0 EXT (another word: none in v0.6), 1-7 D (+7 → 20), 8-14 L (+7), 15-21 R (+7),
#   22-27 T (+6 → 16), 8-23 N (+16 → 32, on L and R's bits), 32-41 X (+10 → 16)
# A row is wide when one of its values does not fit word 0 (an index ≥ 2^width − 1, all ones being none).
#
# An edge-table entry (32 bits): bits 0-14 start plane / 2, 15-22 form (8; 255 none), 23-26
# components, 27-30 physical type (PhysicalType's number), 31 spare.
# A source is a SOURCE row: D its edge (the current and the form), T its mass scale, X its factor,
# N its template: bits 0-2 family (PhysicalType's number), 3 antiparticle, 4-6 helicity + 2,
# 7-8 chirality + 1. A SOURCE's N has no high bits: in a wide SOURCE, word 1 holds D, T and X only.

struct Encoded
    version::VersionNumber
    rows::Vector{UInt64}     # the sources' words, then the rows'
    source_words::Int        # the sources' words at the head of `rows`
    edges::Vector{UInt32}
end

# a field: (shift, width) in word 0, and (shift, width) of its high bits in word 1
struct Field
    shift::Int
    width::Int
    hishift::Int
    hiwidth::Int
end
const FD = Field(9, 13, 1, 7)
const FT = Field(22, 10, 22, 6)
const FL = Field(32, 13, 8, 7)
const FR = Field(45, 13, 15, 7)
const FN = Field(32, 16, 8, 16)
const FX = Field(58, 6, 32, 10)
const FTEMPLATE = Field(32, 16, 8, 0)   # a SOURCE's N: word 0 only (its high bits would cross T's in word 1)
const EXT = UInt64(1) << 5

none(f::Field; wide::Bool) = (UInt64(1) << (f.width + (wide ? f.hiwidth : 0))) - 1
mask(n::Int) = (UInt64(1) << n) - 1

# the fields each opcode uses, and whether they hold indices (all ones: none) or counts
function fields(op::Opcode)::Vector{Tuple{Symbol,Field,Bool}}
    if is_vertex(op) || is_kernel(op) || op == CLOSURE
        return [(:d, FD, true), (:t, FT, true), (:l, FL, true), (:r, FR, true), (:x, FX, true)]
    elseif is_propagator(op)
        return [(:d, FD, true), (:x, FX, true)]
    elseif op == SCALE_ROW
        return [(:d, FD, true), (:n, FN, false), (:x, FX, true)]
    elseif op == SOURCE
        return [(:d, FD, true), (:t, FT, true), (:n, FTEMPLATE, false), (:x, FX, true)]
    elseif op == CLEAR
        return [(:d, FD, true), (:n, FN, false)]
    end
    error("unknown opcode $op")
end

# a field's value in the encoding: an index − 1 (0, a Lowered's none, becomes all ones), a count as it is
encoded_value(v::Int, index::Bool, f::Field, wide::Bool)::UInt64 =
    index ? (v == 0 ? none(f; wide) : UInt64(v - 1)) : UInt64(v)
fits(v::Int, index::Bool, f::Field, wide::Bool) =
    index ? v == 0 || v - 1 < none(f; wide) : v < (UInt64(1) << (f.width + (wide ? f.hiwidth : 0)))

# the words of a row, appended to `words`
function encode_row!(words::Vector{UInt64}, op::LOp)
    fs = fields(op.op)
    wide = !all(fits(getfield(op, s), index, f, false) for (s, f, index) in fs)
    for (s, f, index) in fs
        fits(getfield(op, s), index, f, true) || error("$(op.op): $s = $(getfield(op, s)) does not fit UIR v0.6")
    end
    w0 = UInt64(Int(op.op)) | (wide ? EXT : 0) | UInt64(op.set) << 6 | UInt64(op.first) << 7 | UInt64(op.last) << 8
    w1 = UInt64(0)
    for (s, f, index) in fs
        v = encoded_value(getfield(op, s), index, f, wide)
        w0 |= (v & mask(f.width)) << f.shift
        wide && (w1 |= (v >> f.width) << f.hishift)
    end
    push!(words, w0)
    wide && push!(words, w1)
    return words
end

# the row at words[k] and the index of the next row
function decode_row(words::Vector{UInt64}, k::Int)::Tuple{LOp,Int}
    w0 = words[k]
    op = Opcode(w0 & 31)
    wide = w0 & EXT != 0
    w1 = wide ? words[k+1] : UInt64(0)
    wide && w1 & 1 != 0 && error("row at word $k: a third word (not in UIR v0.6)")
    vals = Dict{Symbol,Int}()
    for (s, f, index) in fields(op)
        v = (w0 >> f.shift) & mask(f.width)
        wide && (v |= ((w1 >> f.hishift) & mask(f.hiwidth)) << f.width)
        vals[s] = index ? (v == none(f; wide) ? 0 : Int(v) + 1) : Int(v)
    end
    get0(s) = get(vals, s, 0)
    o = LOp(op, w0 >> 6 & 1 == 1, w0 >> 7 & 1 == 1, w0 >> 8 & 1 == 1, get0(:d), get0(:t), get0(:l), get0(:r), get0(:n), get0(:x))
    return (o, k + (wide ? 2 : 1))
end

const EDGE_NOFORM = 255

function encode_edge(l::Lowered, e::EdgeIndex)::UInt32
    c = edge_layout(l, e)
    f = l.edges[e].form
    (iseven(c.start) && c.start ÷ 2 < 2^15) || error("edge $e: start plane $(c.start) does not fit UIR v0.6")
    f - 1 < EDGE_NOFORM || error("edge $e: form $f does not fit UIR v0.6")
    c.components < 16 || error("edge $e: $(c.components) components")
    form = f == 0 ? EDGE_NOFORM : f - 1
    return UInt32(c.start ÷ 2) | UInt32(form) << 15 | UInt32(c.components) << 23 | UInt32(Int(c.type)) << 27
end

# an edge-table entry: (start plane, form, components, physical type), 1-based form (0: none)
function decode_edge(x::UInt32)::Tuple{PlaneOffset,FormIndex,Int,PhysicalType}
    form = Int((x >> 15) & 0xff)
    return (Int(x & 0x7fff) * 2, form == EDGE_NOFORM ? 0 : form + 1, Int((x >> 23) & 0xf), PhysicalType((x >> 27) & 0xf))
end

# a source as its SOURCE row, and back
function source_row(s::LSource)::LOp
    -2 <= s.helicity <= 5 && -1 <= s.chirality <= 2 || error("source on edge $(s.edge): helicity $(s.helicity), chirality $(s.chirality)")
    template = Int(s.family) | Int(s.antiparticle) << 3 | (s.helicity + 2) << 4 | (s.chirality + 1) << 7
    return LOp(SOURCE; d = s.edge, t = s.mass, n = template, x = s.factor)
end

function row_source(o::LOp)::LSource
    o.op == SOURCE || error("a $(o.op) row among the sources")
    return LSource(o.d, PhysicalType(o.n & 7), (o.n >> 3) & 1 == 1, ((o.n >> 4) & 7) - 2, ((o.n >> 7) & 3) - 1, o.t, o.x)
end

function encode(l::Lowered)::Encoded
    validate(l)
    words = UInt64[]
    foreach(s -> encode_row!(words, source_row(s)), l.sources)
    n = length(words)
    foreach(op -> encode_row!(words, op), l.ops)
    return Encoded(l.version, words, n, [encode_edge(l, e) for e in eachindex(l.edges)])
end

# the rows of an encoded stream
function decode_rows(words::Vector{UInt64})::Vector{LOp}
    ops = LOp[]
    k = 1
    while k <= length(words)
        (o, k) = decode_row(words, k)
        push!(ops, o)
    end
    return ops
end

# What a device sees, as a Lowered: the sources, rows and edges decoded from `e`, the other tables
# from `l`. Each edge becomes its own current (its planes, components and type from the entry), so
# the decoded Lowered holds only what the encoding carries; its values must still equal `l`'s.
function decode(e::Encoded, l::Lowered)::Lowered
    entries = decode_edge.(e.edges)
    currents = [LCurrent(start, n, ty) for (start, _, n, ty) in entries]
    edges = [LEdge(k, form) for (k, (_, form, _, _)) in enumerate(entries)]
    sources = row_source.(decode_rows(e.rows[1:e.source_words]))
    return validate(with(l; version = e.version, currents = currents, edges = edges, sources = sources,
                         ops = decode_rows(e.rows[e.source_words+1:end])))
end
