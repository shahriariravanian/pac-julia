# The SSA form of a MIR: a port of symjit-utils' `to_ssa` (src/ssa.rs), which is symjit's
# `GreedyAllocator::create` (the first pass of its register allocator, allocator.rs): each definition
# of a general register `Gen(r)` gets a new `Static(k)` register, and each use the `Static` its
# register holds. `Ret`, `Temp`, `Left` and `Right` stay as they are (library calls and funclets pass
# values in them), and so do the stack and memory slots. A label or a call forgets the registers.
#
# The MIR of a lowered complex model passes values across labels and calls (the funclets of the
# lowering, e.g. `@complex_root`, read `Gen` registers set before the call), so a register read
# before any definition since the last label or call is *pinned*. The definition of a pinned
# register that is current where control may leave the straight-line code (a label, call, jump or
# the end) is *kept*: it writes `Gen(r)`, and reads see `Gen(r)` until the next definition, so the
# register holds the original MIR's value at every jump and label. Its other definitions
# (overwritten before control leaves) get a `Static`.
#
# Compression (`LoadArgs`, `SaveArgs`) has no SSA form. `to_ssa(read_sjb(path; lowered = true).mir)`
# is what `sjb2ssa` writes to an .ssa (register for register: the "ssa" tests).

# the registers an instruction reads and writes, in the order `create` visits them (reads first)
const READS_WRITES = Dict{DataType,Tuple{Tuple,Tuple}}(
    Uni => ((:s1,), (:dst,)),
    Mov => ((:s1,), (:dst,)),
    Bi => ((:s1, :s2), (:dst,)),
    LoadConst => ((), (:dst,)),
    Load => ((), (:dst,)),
    Save => ((:src,), ()),
    LoadComplex => ((), (:xd, :yd)),
    SaveComplex => ((:xs, :ys), ()),
    Fused => ((:a, :b, :c), (:dst,)),
    IfElse => ((:true_val, :false_val), (:dst,)),
    BranchIf => ((:cond,), ()),
    LoadMath => ((:s1,), (:dst,)),
    LoadConstMath => ((:s1,), (:dst,)),
    ComplexBi => ((:x1, :y1, :x2, :y2), (:xd, :yd)),
)

struct SsaError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SsaError) = print(io, "ssa: ", e.msg)

function reads_writes(ins::Instruction)
    (ins isa LoadArgs || ins isa SaveArgs) && throw(SsaError("compression mode (LoadArgs, SaveArgs) has no SSA form"))
    return get(READS_WRITES, typeof(ins), ((), ()))
end

operands(ins::Instruction) = let (r, w) = reads_writes(ins)
    (Reg[getfield(ins, f) for f in r], Reg[getfield(ins, f) for f in w])
end

# whether control may leave the straight-line code at the instruction (or after it)
leaves(ins::Instruction) = ins isa Call || ins isa Label || ins isa Branch || ins isa BranchIf || ins isa End
# whether the instruction forgets the registers (`create`'s `reset_regs`)
resets(ins::Instruction) = ins isa Call || ins isa Label

# `ins` with its registers mapped by `f(reg, is_write)`, reads before writes
function rewrite(f, ins::Instruction)
    (r, w) = reads_writes(ins)
    isempty(r) && isempty(w) && return ins
    mapped = Dict{Symbol,Reg}()
    foreach(n -> mapped[n] = f(getfield(ins, n), false), r)
    foreach(n -> mapped[n] = f(getfield(ins, n), true), w)
    T = typeof(ins)
    return T((get(mapped, n, getfield(ins, n)) for n in fieldnames(T))...)
end

"""
    to_ssa(mir::MIR) -> (MIR, stats)

`mir` in SSA form (symjit-utils' `to_ssa`, the first pass of symjit's register allocator; see
src/ssa.jl): every definition of a `Gen` register a new `Static` one, but for the kept definitions
of pinned registers. `stats`: `(instructions, statics, pinned, kept)`.
"""
function to_ssa(mir::MIR)
    is_gen(r::Reg) = r.kind == REG_GEN
    for ins in mir.code, r in Iterators.flatten(operands(ins))
        r.kind == REG_STATIC && throw(SsaError("the MIR is in SSA form already"))
    end

    # 1. the pinned registers: read before any definition since the last label or call
    defined, pinned = falses(64), falses(64)
    for ins in mir.code
        (reads, writes) = operands(ins)
        for r in reads
            is_gen(r) && !defined[r.index+1] && (pinned[r.index+1] = true)
        end
        for r in writes
            is_gen(r) && (defined[r.index+1] = true)
        end
        resets(ins) && fill!(defined, false)
    end

    # 2. backward: the definitions of pinned registers that must write them, the last before each
    # label, call, jump or end; the others get a `Static` as well
    keep = zeros(UInt64, length(mir.code))
    later = falses(64)
    for i in reverse(eachindex(mir.code))
        ins = mir.code[i]
        if leaves(ins)
            fill!(later, false)
            continue
        end
        for r in operands(ins)[2]
            if is_gen(r) && pinned[r.index+1] && !later[r.index+1]
                keep[i] |= UInt64(1) << r.index
                later[r.index+1] = true
            end
        end
    end

    # 3. the rewrite: a new `Static` per definition, but the kept ones
    current = Vector{Union{Nothing,Int}}(nothing, 64)
    statics, kept = 0, 0
    code = Instruction[]
    for (i, ins) in enumerate(mir.code)
        missing_reg = nothing
        out = rewrite(ins) do r, write
            is_gen(r) || return r
            g = r.index
            if write && keep[i] & (UInt64(1) << g) != 0
                current[g+1] = nothing
                kept += 1
                return r
            elseif write
                current[g+1] = statics
                statics += 1
                return Reg(REG_STATIC, statics - 1)
            end
            s = current[g+1]
            s !== nothing && return Reg(REG_STATIC, s)
            pinned[g+1] || (missing_reg = g)   # a pinned register holds its value
            return r
        end
        # pass 1 pins every register read before a definition
        missing_reg === nothing || throw(SsaError("internal error: Gen($missing_reg) read before it is defined"))
        push!(code, out)
        if resets(ins)
            fill!(current, nothing)
        elseif leaves(ins)
            foreach(g -> pinned[g] && (current[g] = nothing), 1:64)
        end
    end
    stats = (instructions = length(mir.code), statics = statics, pinned = [g - 1 for g in 1:64 if pinned[g]], kept = kept)
    return MIR(code, copy(mir.consts)), stats
end

"""
    to_ssa(app::SavedApplication) -> (SavedApplication, stats)

The application with its MIR in SSA form (`app.ssa`). A complex model is lowered first (as
`sjb2ssa` does: the lowering cannot read `Static` registers), from `read_sjb(path; lowered = true)`.
"""
function to_ssa(app::SavedApplication)
    app.ssa && throw(SsaError("the application is in SSA form already"))
    is_complex(app.config) && !app.lowered &&
        throw(SsaError("a complex model is lowered first: read_sjb(path; lowered = true)"))
    (mir, stats) = to_ssa(app.mir)
    fields = (f == :mir ? mir : f == :ssa ? true : getfield(app, f) for f in fieldnames(SavedApplication))
    return SavedApplication(fields...), stats
end
