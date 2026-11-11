# UIR, the Unified Intermediate Representation (v0.8): a program as a flat list of fixed-size rows
# with every operand resolved, which the back ends read (the C generator, the GPU interpreters' 8-byte
# encoding). `load_recurrence_export` (recurrence_export.jl) reads RustiCol's recurrence export into
# it row for row; `lower(p::Program)` (v1/lower.jl) builds it from a pac2json program. The design is
# section 7 of the pac-julia design doc ("The lowered IR").
#
# v0.8 adds to v0.7: sources are rows (SOURCE, anywhere in the stream: a program of several runs has
# each run's sources in it) and may scatter their components (an embedding); the reduction is rows
# (REDUCE, after each run: out[h] += weight × |scale × amplitude|²); CLEAR_AMPLITUDE, COPY and SCATTER
# (RustiCol's certified reuse and fanout); KERNEL_DIRECT stores the kernel's pairs unscaled, with
# its mode.
#
# Operands are edges: an edge is a (current, momentum form) pair, and the edge table gives each
# edge's current, whose layout entry gives its planes. So a layout change (plane reuse) touches only
# the current table, never the operations.
#
# Indices: 1-based in every Julia structure (Program, Lowered; 0: none, as a null operand), 0-based
# in the encoding and in C. Only the boundary converts: `encode` and the C generator subtract 1, and
# nothing else does. Plane offsets (LCurrent.start, Lowered.planes) are offsets, not indices: 0-based
# in Julia too, and written as they are.

const EdgeIndex = Int       # an edge (current, form): Lowered.edges; 0: a null operand
const StubIndex = Int       # a kernel stub (kernel, input layout): Lowered.stubs
const ScaleSetIndex = Int   # a kernel scale set: Lowered.scalesets
const InstanceIndex = Int   # a binding instance (stub, scale set): Lowered.instances
const ListIndex = Int       # a closure's component-factor list: Lowered.lists
const EmbeddingIndex = Int  # a source's embedding: Lowered.embeddings
const WeightIndex = Int     # a reduction weight: Lowered.weights
const PlaneOffset = Int     # a plane (one complex component of the arena), 0-based
const UIR_VERSION = v"0.8"

# what a current holds; the component count follows from it (SCALAR 1, WEYL 2, DIRAC 4, VECTOR 4,
# TENSOR 6). UNTYPED: no row or source says (a current only kernels touch)
@enum PhysicalType UNTYPED SCALAR WEYL DIRAC VECTOR TENSOR

const TYPE_COMPONENTS = Dict(SCALAR => 1, WEYL => 2, DIRAC => 4, VECTOR => 4, TENSOR => 6)

# The opcodes, numbered as in the 8-byte word (bits 0-4): the vertex types, the propagator types,
# then the other rows. A vertex's mode is the SET flag, a kernel's direct store its own opcode.
@enum Opcode::UInt8 begin
    V_WEYL_VECTOR_A = 0   # l Weyl, r vector → Weyl
    V_WEYL_VECTOR_B = 1   # l Weyl, r vector → Weyl
    V_TENSOR_VECTOR = 2   # l tensor, r vector → vector
    V_VECTOR_WEDGE = 3    # l vector, r vector → tensor
    V_THREE_VECTOR = 4    # l vector, r vector → vector; reads both momenta
    P_WEYL_A = 5          # the propagators: current Weyl, reads its momentum
    P_WEYL_B = 6
    P_VECTOR = 7          # current vector (Feynman gauge)
    CLOSURE = 8
    SCALE_ROW = 9         # scale (SCALE is a KernelInput kind)
    CLEAR = 10
    KERNEL = 11
    KERNEL_DIRECT = 12    # a direct binding: the pairs stored as they are (the factor not applied), with the mode
    SOURCE = 13           # a source current
    CLEAR_AMPLITUDE = 14  # an amplitude set to 0
    COPY = 15             # a current's components × a factor into another (RustiCol's certified reuse)
    SCATTER = 16          # a fanout scatter: as COPY, with a mode and RustiCol's factor shortcuts
    REDUCE = 17           # out[h] += weight × |amplitude|² (the amplitude × its replay scale first)
    END = 31              # the end of the stream: in the encoding only, its last word
end

is_vertex(op::Opcode) = op <= V_THREE_VECTOR
is_propagator(op::Opcode) = P_WEYL_A <= op <= P_VECTOR
is_kernel(op::Opcode) = op == KERNEL || op == KERNEL_DIRECT

# a current's place in the arena: planes start .. start + components - 1, start even
struct LCurrent
    start::PlaneOffset
    components::Int
    type::PhysicalType
end

# an edge: a current with the momentum it is read with (FormIndex 0: none)
struct LEdge
    current::CurrentIndex
    form::FormIndex
end

# A row. The fields are the UIR word's (D, T, L, R, N, X), with the same meaning per opcode, but
# full integers: the widths are encode's concern.
#   vertex           d out edge, t coupling (ScaleIndex), l, r edges, x factor
#   kernel           d out edge, t binding instance, l, r edges (0: a null operand), x factor
#   propagator       d the current's edge, x factor
#   closure          d amplitude, t component-factor list, l, r edges, x factor
#   scale            d the current's edge, n components, x factor
#   clear            d the current's edge, n components
#   source           d the current's edge (its form: the wavefunction's momentum), t mass (ScaleIndex: its
#                    real part), n template (source_template), l embedding (0: none), x factor (none
#                    with an embedding, whose entries carry their own)
#   clear_amplitude  d amplitude
#   copy             d, l edges (destination, source; d's components are copied), x factor
#   scatter          d, l edges (as copy), x factor; set = mode "="
#   reduce           d output (helicity), t amplitude, x the amplitude's scale (0: none), n weight
# Flags: set = mode "=" (vertex, kernel, scatter); first, last = a fused run (vertex rows only).
struct LOp
    op::Opcode
    set::Bool
    first::Bool
    last::Bool
    d::Int
    t::Int
    l::EdgeIndex
    r::EdgeIndex
    n::Int
    x::FactorIndex
end

LOp(op::Opcode; set = false, first = false, last = false, d = 0, t = 0, l = 0, r = 0, n = 0, x = 0) =
    LOp(op, set, first, last, d, t, l, r, n, x)

# A source row's template (its N): family (PhysicalType's number) bits 0-2, antiparticle 3,
# helicity + 2 bits 4-6, chirality + 1 bits 7-8
function source_template(family::PhysicalType, antiparticle::Bool, helicity::Helicity, chirality::Chirality)::Int
    -2 <= helicity <= 5 && -1 <= chirality <= 2 || error("source helicity $helicity, chirality $chirality")
    return Int(family) | Int(antiparticle) << 3 | (helicity + 2) << 4 | (chirality + 1) << 7
end

# a SOURCE row's template as (family, antiparticle, helicity, chirality)
source_fields(op::LOp) = (PhysicalType(op.n & 7), (op.n >> 3) & 1 == 1, ((op.n >> 4) & 7) - 2, ((op.n >> 7) & 3) - 1)

# A source row: factor × the wavefunction of the template at its edge's form into its edge's current;
# with an embedding, component j of the current = factor_j × wavefunction component k_j (0: zero)
source_op(edge::EdgeIndex, family::PhysicalType, antiparticle::Bool, helicity::Helicity, chirality::Chirality,
          mass::ScaleIndex, factor::FactorIndex; embedding::EmbeddingIndex = 0) =
    LOp(SOURCE; d = edge, t = mass, n = source_template(family, antiparticle, helicity, chirality), l = embedding,
        x = embedding == 0 ? factor : 0)

# an embedding entry: (the current's component, the wavefunction's component or 0 for zero, factor)
const EmbeddingEntry = Tuple{ComponentIndex,ComponentIndex,FactorIndex}

# A kernel stub: a kernel with its input layout compiled in. The layout is the binding's inputs
# with each scale replaced by its slot in the scale set (KernelInput(SCALE, slot, false)), so
# bindings that differ only in their scales share a stub.
struct LStub
    kernel::KernelLibrary
    inputs::Vector{KernelInput}
    outputs::Vector{ComponentIndex}  # the components of out, one per output pair
end

struct Lowered
    version::VersionNumber
    process::String
    colour::String
    legs::Int
    crossing::Union{Nothing,Vector{Tuple{LegIndex,Float64}}}
    forms::Vector{Form}
    scales::Vector{Complex{Float64}}
    factors::Vector{Complex{Float64}}
    alpha_s::Union{Nothing,Tuple{Float64,Union{Int,Nothing}}}
    helicities::Vector{Vector{Helicity}}
    currents::Vector{LCurrent}           # indexed by CurrentIndex, as Program.currents
    planes::Int                          # the arena's planes
    edges::Vector{LEdge}                 # edge c is current c with its own momentum; extra pairs follow
    lists::Vector{Vector{FactorIndex}}   # closure component factors
    embeddings::Vector{Vector{EmbeddingEntry}}  # source embeddings
    stubs::Vector{LStub}
    scalesets::Vector{Vector{ScaleIndex}}
    instances::Vector{Tuple{StubIndex,ScaleSetIndex}}
    ops::Vector{LOp}
    amplitudes::Int
    weights::Vector{Float64}             # the REDUCE rows' weights
    group_factor::Tuple{Int,Int}         # applied to every output once, after the rows
    fixed_layout::Bool                   # the layout is the producer's (RustiCol's): no reuse_planes
end

output_count(l::Lowered) = length(l.helicities)   # the helicity outputs

# an edge's current, form, layout entry and first plane
edge_current(l::Lowered, e::EdgeIndex)::CurrentIndex = e == 0 ? 0 : l.edges[e].current
edge_form(l::Lowered, e::EdgeIndex)::FormIndex = e == 0 ? 0 : l.edges[e].form
edge_layout(l::Lowered, e::EdgeIndex)::LCurrent = l.currents[l.edges[e].current]
edge_plane(l::Lowered, e::EdgeIndex)::PlaneOffset = edge_layout(l, e).start

# A row as text, for the C comments and dumps: main.json's names (c<i>, q<i>, s<i>, x<i>, a<i>, which
# count from 0), an edge as its current with the momentum it is read with (c12@q3)
function describe(l::Lowered, op::LOp)::String
    edge(e) = e == 0 ? "-" : "c$(edge_current(l, e) - 1)" * (edge_form(l, e) == 0 ? "" : "@q$(edge_form(l, e) - 1)")
    mode = op.set ? "=" : "+="
    run = !is_vertex(op.op) || op.first && op.last ? "" : op.first ? " [run starts]" : op.last ? " [run ends]" : " [in run]"
    if is_vertex(op.op) || is_kernel(op.op)
        what = is_vertex(op.op) ? "s$(op.t - 1)" : "binding instance $(op.t)"
        return "$(op.op) $(edge(op.d)) $mode $(edge(op.l)), $(edge(op.r)); $what x$(op.x - 1)$run"
    elseif is_propagator(op.op)
        return "$(op.op) $(edge(op.d)); x$(op.x - 1)"
    elseif op.op == CLOSURE
        return "CLOSURE a$(op.d - 1) += $(edge(op.l)) . $(edge(op.r)); list $(op.t) x$(op.x - 1)"
    elseif op.op == SCALE_ROW
        return "SCALE $(edge(op.d)), $(op.n) components; x$(op.x - 1)"
    elseif op.op == SOURCE
        (family, anti, h, ch) = source_fields(op)
        how = op.l == 0 ? "x$(op.x - 1)" : "embedding $(op.l)"
        return "SOURCE $(edge(op.d)) = $family$(anti ? " antiparticle" : ""), helicity $h, chirality $ch, mass s$(op.t - 1); $how"
    elseif op.op == CLEAR_AMPLITUDE
        return "CLEAR_AMPLITUDE a$(op.d - 1)"
    elseif op.op == COPY
        return "COPY $(edge(op.d)) = $(edge(op.l)); x$(op.x - 1)"
    elseif op.op == SCATTER
        return "SCATTER $(edge(op.d)) $mode $(edge(op.l)); x$(op.x - 1)"
    elseif op.op == REDUCE
        scale = op.x == 0 ? "" : " (× x$(op.x - 1))"
        return "REDUCE out[$(op.d - 1)] += w$(op.n) × |a$(op.t - 1)$scale|²"
    else
        return "$(op.op) $(edge(op.d)), $(op.n) components"
    end
end

function Base.show(io::IO, ::MIME"text/plain", l::Lowered)
    count(f) = Base.count(o -> f(o.op), l.ops)
    println(io, "Lowered (UIR v$(l.version)): $(l.process), $(l.colour)")
    println(io, "  $(length(l.ops)) rows: $(count(is_vertex)) vertex, $(count(is_propagator)) propagator, ",
            "$(count(is_kernel)) kernel, $(count(==(CLOSURE))) closure, $(count(==(SCALE_ROW))) scale, $(count(==(CLEAR))) clear, ",
            "$(count(==(SOURCE))) source, $(count(==(CLEAR_AMPLITUDE))) clear_amplitude, $(count(==(COPY))) copy, ",
            "$(count(==(SCATTER))) scatter, $(count(==(REDUCE))) reduce")
    println(io, "  $(length(l.currents)) currents on $(l.planes) planes$(l.fixed_layout ? " (fixed layout)" : ""), ",
            "$(length(l.edges)) edges, $(length(l.forms)) forms, $(l.amplitudes) amplitudes, $(output_count(l)) outputs")
    print(io, "  $(length(l.stubs)) stubs, $(length(l.scalesets)) scale sets, $(length(l.instances)) binding instances, ",
          "$(length(l.lists)) component-factor lists, $(length(l.embeddings)) embeddings")
end
