# UIR, the Unified Intermediate Representation (v0.6): a Program lowered to a flat list of
# fixed-size operations with every operand resolved, which the back ends read (the C generator, the
# GPU interpreters' 8-byte encoding). `lower(p::Program)` (lower.jl) builds it. The design is section
# 7 of the pac-julia design doc ("The lowered IR: UIR v0.6").
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
const PlaneOffset = Int     # a plane (one complex component of the arena), 0-based
const UIR_VERSION = v"0.6"

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
    KERNEL_DIRECT = 12    # a direct binding whose factor is exactly 1: the pairs stored as they are
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
#   vertex      d out edge, t coupling (ScaleIndex), l, r edges, x factor
#   kernel      d out edge, t binding instance, l, r edges (0: a null operand), x factor
#   propagator  d the current's edge, x factor
#   closure     d amplitude, t component-factor list, l, r edges, x factor
#   scale       d the current's edge, n components, x factor
#   clear       d the current's edge, n components
# Flags: set = mode "=" (vertex, kernel); first, last = a fused run (vertex rows only).
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

# A source: its own table, run before the rows (factor × the wavefunction at `form`)
struct LSource
    current::CurrentIndex
    form::FormIndex
    family::PhysicalType   # SCALAR, WEYL, DIRAC, VECTOR
    antiparticle::Bool
    helicity::Helicity
    chirality::Chirality
    mass::ScaleIndex       # its real part is the mass
    factor::FactorIndex
end

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
    alpha_s::Union{Nothing,Tuple{Float64,Int}}
    helicities::Vector{Vector{Helicity}}
    currents::Vector{LCurrent}           # indexed by CurrentIndex, as Program.currents
    planes::Int                          # the arena's planes
    edges::Vector{LEdge}                 # edge c is current c with its own momentum; extra pairs follow
    sources::Vector{LSource}
    lists::Vector{Vector{FactorIndex}}   # closure component factors
    stubs::Vector{LStub}
    scalesets::Vector{Vector{ScaleIndex}}
    instances::Vector{Tuple{StubIndex,ScaleSetIndex}}
    ops::Vector{LOp}
    amplitudes::Int
    reduction::Vector{Vector{Tuple{AmplitudeIndex,Float64}}}
    group_factor::Tuple{Int,Int}
end

# an edge's current, form, layout entry and first plane
edge_current(l::Lowered, e::EdgeIndex)::CurrentIndex = e == 0 ? 0 : l.edges[e].current
edge_form(l::Lowered, e::EdgeIndex)::FormIndex = e == 0 ? 0 : l.edges[e].form
edge_layout(l::Lowered, e::EdgeIndex)::LCurrent = l.currents[l.edges[e].current]
edge_plane(l::Lowered, e::EdgeIndex)::PlaneOffset = edge_layout(l, e).start

function Base.show(io::IO, ::MIME"text/plain", l::Lowered)
    count(f) = Base.count(o -> f(o.op), l.ops)
    println(io, "Lowered (UIR v$(l.version)): $(l.process), $(l.colour)")
    println(io, "  $(length(l.ops)) rows: $(count(is_vertex)) vertex, $(count(is_propagator)) propagator, ",
            "$(count(is_kernel)) kernel, $(count(==(CLOSURE))) closure, $(count(==(SCALE_ROW))) scale, $(count(==(CLEAR))) clear")
    println(io, "  $(length(l.currents)) currents on $(l.planes) planes, $(length(l.edges)) edges, ",
            "$(length(l.forms)) forms, $(length(l.sources)) sources, $(l.amplitudes) amplitudes")
    print(io, "  $(length(l.stubs)) stubs, $(length(l.scalesets)) scale sets, $(length(l.instances)) binding instances, ",
          "$(length(l.lists)) component-factor lists")
end
