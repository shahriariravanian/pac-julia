# A reader of saved symjit applications (.sjb; pac2json's kernels are .plane.symjit files of the
# same format) in pure Julia: the file's header and its MIR, as symjit wrote them. Nothing is
# lowered or compiled: for a complex model (`is_complex(app.config)`) the MIR is the complex MIR,
# before symjit's Complexifier lowers it to real operations.
#
# The format (symjit rust/runnable.rs, model.rs, config.rs, builder.rs, serializer.rs, strings.rs;
# all integers little-endian u64 unless noted):
#   Application  magic, version (3 or 4), Program, mask (the kernels it was saved with), Mir, reals
#   Program      magic, Config, count_states, count_params, count_obs, count_diffs, count_loops, Builder
#   Config       magic, opt | ty << 32, [version 4: stack_limit, num_args]
#   Builder      magic, count_loops, stack slots, n, n f64 constants, n, n function names (strings)
#   Mir          magic, n, n bytes of code, n, n f64 constants
#   reals        n, n codes (kind << 32 | index; kind 1 Mem, 2 Stack, 3 Param)
#   string       a length byte (255, 255: a u64 length follows) and the UTF-8 bytes
#
# Indices are kept as the file has them (0-based): Mem[i], Stack[i], Param[i], consts[i]
# (app.mir.consts[i + 1] in Julia), Gen(r) registers. `show` prints an instruction as symjit's
# Debug does (e.g. `%0 := Mem[3]`), so a listing can be compared with symjit's.

# ---- registers and locations ----------------------------------------------------------------------

@enum RegKind REG_RET REG_TEMP REG_LEFT REG_RIGHT REG_GEN REG_STATIC
@enum LocKind LOC_MEM LOC_PARAM LOC_STACK

struct Reg
    kind::RegKind
    index::Int  # of REG_GEN and REG_STATIC; 0 otherwise
end

struct Loc
    kind::LocKind
    index::Int
end

# ---- instructions (symjit's mir::Instruction, field for field) --------------------------------------

const UNI_OPS = (:Neg, :Not, :Abs, :Root, :RealRoot, :Recip, :Round, :Floor, :Ceiling, :Trunc, :Real,
                 :Imaginary, :Conjugate, :Half, :IsZero, :IsNotZero, :Sign, :Abs2, :TimesI, :TimesNegI)
const BIN_OPS = (:Plus, :Minus, :Times, :Divide, :GreaterThan, :GreaterThanEqual, :LittleThan,
                 :LittleThanEqual, :Equal, :NotEqual, :And, :AndNot, :Or, :Xor, :Complex)
const ARITH_OPS = (:Plus, :Minus, :Times, :Divide)
const FUSED_OPS = (:MulAdd, :MulSub, :NegMulAdd, :NegMulSub)  # a*b + c, a*b - c, -a*b + c, -a*b - c

abstract type Instruction end

struct Nop <: Instruction end
struct End <: Instruction end
struct Uni <: Instruction  # dst := op(s1)
    op::Symbol             # one of UNI_OPS
    dst::Reg
    s1::Reg
end
struct Bi <: Instruction   # dst := s1 op s2
    op::Symbol             # one of BIN_OPS
    dst::Reg
    s1::Reg
    s2::Reg
end
struct Mov <: Instruction  # dst := s1
    dst::Reg
    s1::Reg
end
struct Load <: Instruction  # dst := loc
    dst::Reg
    loc::Loc
end
struct Save <: Instruction  # loc := src
    src::Reg
    loc::Loc
end
struct LoadConst <: Instruction  # dst := consts[idx]
    dst::Reg
    idx::Int
end
struct LoadComplex <: Instruction  # xd + yd*im := loc
    xd::Reg
    yd::Reg
    loc::Loc
end
struct SaveComplex <: Instruction  # loc := xs + ys*im
    xs::Reg
    ys::Reg
    loc::Loc
end
struct LoadArgs <: Instruction  # the arguments of the next call
    locs::Vector{Loc}
    complex::Bool
    ultra::Bool
end
struct SaveArgs <: Instruction
    num_args::Int
    complex::Bool
    ultra::Bool
end
struct Fused <: Instruction  # dst := ±a*b ± c
    op::Symbol               # one of FUSED_OPS
    dst::Reg
    a::Reg
    b::Reg
    c::Reg
end
struct IfElse <: Instruction  # dst := cond ? true_val : false_val
    dst::Reg
    true_val::Reg
    false_val::Reg
    cond::Loc
end
struct Label <: Instruction
    label::String
end
struct Branch <: Instruction  # goto label
    label::String
end
struct BranchIf <: Instruction  # if cond goto label (is_else: if not cond)
    cond::Reg
    label::String
    is_else::Bool
end
struct Call <: Instruction  # a math function (or a label of the model)
    label::String
    num_args::Int
end
struct LoadMath <: Instruction  # dst := s1 op loc
    op::Symbol                  # one of ARITH_OPS
    dst::Reg
    s1::Reg
    loc::Loc
end
struct LoadConstMath <: Instruction  # dst := s1 op consts[idx]
    op::Symbol
    dst::Reg
    s1::Reg
    idx::Int
end
struct ComplexBi <: Instruction  # xd + yd*im := (x1 + y1*im) op (x2 + y2*im)
    op::Symbol
    xd::Reg
    yd::Reg
    x1::Reg
    y1::Reg
    x2::Reg
    y2::Reg
end

# ---- the application ------------------------------------------------------------------------------

struct Config
    opt::UInt32       # the option bits (CONFIG_OPTIONS) and the opt level
    ty::Int           # the compiler type: 0 native, 1 amd, 2 amd-avx, 3 amd-sse, 4 arm, 5 riscv, 6 bytecode, 7 debug
    stack_limit::Int
    num_args::Int
end

struct MIR
    code::Vector{Instruction}
    consts::Vector{Float64}  # the constant pool: consts[idx] is consts[idx + 1] here
end

struct SavedApplication
    version::Int
    config::Config
    count_states::Int
    count_params::Int
    count_obs::Int
    count_diffs::Int
    count_loops::Int
    stack_size::Int            # the stack slots of the model
    consts::Vector{Float64}    # the builder's constants (the MIR's, but for `stable_math` ones)
    functions::Vector{String}  # the function table (the math functions it calls), sorted
    mask::Int                  # 1 scalar, 2 fast, 4 SIMD: the kernels it was saved with
    mir::MIR
    reals::Vector{Loc}         # the locations of a complex model that hold real numbers, sorted
end

# the option bits of Config.opt (symjit config.rs)
const CONFIG_OPTIONS = (
    use_simd = 0x0000_0001, use_threads = 0x0000_0002, cse = 0x0000_0004, fastmath = 0x0000_0008,
    enable_simd512 = 0x0000_0010, complex = 0x0000_0020, symbolica = 0x0000_0040, simd_branch = 0x0000_0080,
    compact = 0x0000_1000, compress = 0x0000_2000, direct = 0x0000_4000, fast_complex = 0x0000_8000,
    huge = 0x0010_0000, parallel_mul = 0x0020_0000, direct_arena = 0x0040_0000,
    direct_arena_identity_output = 0x0080_0000, stable_math = 0x0800_0000, coroutines = 0x4000_0000,
    ultra = 0x8000_0000,
)

is_complex(c::Config) = c.opt & CONFIG_OPTIONS.complex != 0
opt_level(c::Config) = Int((c.opt & 0x0000_0f00) >> 8)
options(c::Config) = [String(k) for (k, bit) in pairs(CONFIG_OPTIONS) if c.opt & bit != 0]

# the memory of one point: states, observables, derivatives, one spare value (f64 values; a
# complex model counts two per number)
mem_size(a::SavedApplication) = a.count_states + a.count_obs + a.count_diffs + 1

# ---- reading --------------------------------------------------------------------------------------

const APPLICATION_MAGIC = 0x40568795410d08e9
const PROGRAM_MAGIC = 0xc2b244aefb8e4d5d
const CONFIG_MAGIC = 0x802c3c77c7422e70
const BUILDER_MAGIC = 0x12f21e25abe627bc
const MIR_MAGIC = 0x876a9b6323b00c9e

struct SjbError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SjbError) = print(io, "sjb: ", e.msg)

u64(io::IO) = ltoh(read(io, UInt64))
int(io::IO) = Int(u64(io))
f64s(io::IO, n::Int) = [reinterpret(Float64, u64(io)) for _ in 1:n]

function magic(io::IO, expected::UInt64, what::String)
    m = u64(io)
    m == expected || throw(SjbError("invalid magic number ($what): 0x$(string(m, base = 16))"))
end

# a string of the Builder (a stream) or of the code (a byte cursor): strings.rs
function read_string(next::Function)::String
    n = Int(next())
    if n == 255
        b = next()
        if b == 255
            n = Int(sum(UInt64(next()) << (8k) for k in 0:7))
            n > 255 || throw(SjbError("noncanonical extended label length"))
            return String([next() for _ in 1:n])
        end
        return String(vcat(b, [next() for _ in 2:255]))  # a 255-byte name: its first byte is read
    end
    return String([next() for _ in 1:n])
end

function read_config(io::IO, version::Int)::Config
    magic(io, CONFIG_MAGIC, "Config")
    v = u64(io)
    (stack, args) = version == 3 ? (1 << 20, 64) : (int(io), int(io))  # version 3: symjit's defaults
    ty = Int(v >> 32)
    0 <= ty <= 7 || throw(SjbError("invalid compiler type $ty"))
    return Config(UInt32(v & 0xffffffff), ty, stack, args)
end

# The code of the Mir: a byte cursor over it
mutable struct Cursor
    buf::Vector{UInt8}
    pos::Int
end
at_end(c::Cursor) = c.pos > length(c.buf)
function next_byte(c::Cursor)::UInt8
    at_end(c) && throw(SjbError("unexpected end of the MIR code"))
    c.pos += 1
    return c.buf[c.pos-1]
end

# a little-endian number of (b & 15) bytes
number(c::Cursor, b::UInt8) = sum((Int(next_byte(c)) << (8k) for k in 0:(b&15)-1); init = 0)

function reg(c::Cursor)::Reg
    b = next_byte(c)
    top = b & 0xc0
    top == 0x40 && return Reg(REG_GEN, b & 0x3f)
    top == 0x80 && return Reg(REG_STATIC, number(c, b))
    b == 0xc0 && return Reg(REG_RET, 0)
    b == 0xc1 && return Reg(REG_TEMP, 0)
    b == 0xc2 && return Reg(REG_LEFT, 0)
    b == 0xc3 && return Reg(REG_RIGHT, 0)
    throw(SjbError("undefined register 0x$(string(b, base = 16))"))
end

function loc(c::Cursor)::Loc
    b = next_byte(c)
    top = b & 0xc0
    top == 0x40 && return Loc(LOC_MEM, number(c, b))
    top == 0x80 && return Loc(LOC_PARAM, number(c, b))
    top == 0xc0 && return Loc(LOC_STACK, number(c, b))
    throw(SjbError("undefined location 0x$(string(b, base = 16))"))
end

constant(c::Cursor) = number(c, next_byte(c))  # an index into the constant pool
label(c::Cursor) = read_string(() -> next_byte(c))
args_flags(b::UInt8) = (Int(b & 0x3f), b & 0x80 != 0, b & 0x40 != 0)  # (count, complex, ultra)

# One instruction (serializer.rs `MirIterator`): the first byte's top two bits are 01 a unary
# operation, 10 a binary one, otherwise the six low bits are the instruction
function instruction(c::Cursor)::Instruction
    h = next_byte(c)
    op = Int(h & 0x3f)
    if h & 0xc0 == 0x40
        op < length(UNI_OPS) || throw(SjbError("undefined unary operation $op"))
        return Uni(UNI_OPS[op+1], reg(c), reg(c))
    elseif h & 0xc0 == 0x80
        op < length(BIN_OPS) || throw(SjbError("undefined binary operation $op"))
        return Bi(BIN_OPS[op+1], reg(c), reg(c), reg(c))
    end
    op == 0 && return Nop()
    op == 1 && return End()
    op == 2 && return Mov(reg(c), reg(c))
    op == 3 && return Load(reg(c), loc(c))
    op == 4 && return Save(reg(c), loc(c))
    op == 5 && return LoadConst(reg(c), constant(c))
    op == 6 && return LoadComplex(reg(c), reg(c), loc(c))
    op == 7 && return SaveComplex(reg(c), reg(c), loc(c))
    op == 8 && return Branch(label(c))
    op == 9 && return BranchIf(reg(c), label(c), false)
    op == 10 && return BranchIf(reg(c), label(c), true)
    if op == 11
        n = Int(next_byte(c))
        n & 0x80 != 0 && (n = (n & 0x7f) | Int(next_byte(c)) << 7)
        return Call(label(c), n)
    end
    op == 12 && return Label(label(c))
    op == 13 && return IfElse(reg(c), reg(c), reg(c), loc(c))
    if op == 14
        (n, complex, ultra) = args_flags(next_byte(c))
        return LoadArgs([loc(c) for _ in 1:n], complex, ultra)
    end
    op == 15 && return SaveArgs(args_flags(next_byte(c))...)
    32 <= op <= 35 && return Fused(FUSED_OPS[op-31], reg(c), reg(c), reg(c), reg(c))
    36 <= op <= 39 && return LoadMath(ARITH_OPS[op-35], reg(c), reg(c), loc(c))
    40 <= op <= 43 && return LoadConstMath(ARITH_OPS[op-39], reg(c), reg(c), constant(c))
    44 <= op <= 47 && return ComplexBi(ARITH_OPS[op-43], reg(c), reg(c), reg(c), reg(c), reg(c), reg(c))
    throw(SjbError("undefined instruction $op"))
end

function decode(code::Vector{UInt8})::Vector{Instruction}
    c = Cursor(code, 1)
    out = Instruction[]
    while !at_end(c)
        push!(out, instruction(c))
    end
    return out
end

# The application of a .sjb. With `lowered`, a complex model's MIR is lowered to real MIR (a real
# model's is as saved); `stable_math` (true or false) replaces the saved configuration's option
# before it is lowered, as symjit-utils' `--set stable_math=...`.
function read_sjb(io::IO; lowered::Bool = false, stable_math::Union{Bool,Nothing} = nothing)::SavedApplication
    magic(io, APPLICATION_MAGIC, "Application")
    version = int(io)
    version in (3, 4) || throw(SjbError("unsupported sjb version $version"))

    magic(io, PROGRAM_MAGIC, "Program")
    config = read_config(io, version)
    (states, params, obs, diffs, loops) = (int(io), int(io), int(io), int(io), int(io))

    magic(io, BUILDER_MAGIC, "Builder")
    int(io)  # the builder's count of loops, the program's again
    stack = int(io)
    consts = f64s(io, int(io))
    functions = [read_string(() -> read(io, UInt8)) for _ in 1:int(io)]

    mask = int(io)

    magic(io, MIR_MAGIC, "Mir")
    code = read(io, int(io))
    mir = MIR(decode(code), f64s(io, int(io)))

    reals = map(1:int(io)) do _
        b = u64(io)
        kind = b >> 32
        kind in (1, 2, 3) || throw(SjbError("invalid real location 0x$(string(b, base = 16))"))
        Loc((LOC_MEM, LOC_STACK, LOC_PARAM)[kind], Int(b & 0xffffffff))
    end
    Base.eof(io) || throw(SjbError("bytes after the application"))

    if stable_math !== nothing
        bit = CONFIG_OPTIONS.stable_math
        config = Config(stable_math ? config.opt | bit : config.opt & ~bit, config.ty, config.stack_limit, config.num_args)
    end
    if lowered && is_complex(config)
        mir = complexify(mir, reals; stable_math = config.opt & CONFIG_OPTIONS.stable_math != 0)
        # stable_math appends constants, which symjit copies to the program's
        length(mir.consts) > length(consts) && (consts = copy(mir.consts))
    end

    return SavedApplication(version, config, states, params, obs, diffs, loops, stack, consts, functions, mask, mir, reals)
end

read_sjb(path::AbstractString; kw...) = open(io -> read_sjb(io; kw...), path)

# ---- printing, as symjit's Debug ------------------------------------------------------------------

function Base.show(io::IO, r::Reg)
    r.kind == REG_GEN ? print(io, "%", r.index) :
    r.kind == REG_STATIC ? print(io, "σ", r.index) :
    print(io, ("%\$", "%t", "%l", "%r")[Int(r.kind)+1])
end

Base.show(io::IO, l::Loc) = print(io, ("Mem", "Param", "Stack")[Int(l.kind)+1], "[", l.index, "]")

# Rust's {:?} of a string: \\, \", \n, \r, \t, \0 and \u{..} for the other control characters
const ESCAPES = Dict('\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t", '\0' => "\\0")
function quoted(s::String)
    io = IOBuffer()
    print(io, '"')
    for ch in s
        e = get(ESCAPES, ch, nothing)
        e !== nothing ? print(io, e) : iscntrl(ch) ? print(io, "\\u{", string(UInt32(ch), base = 16), "}") : print(io, ch)
    end
    print(io, '"')
    return String(take!(io))
end
cx(io, x, y) = print(io, "(", x, " + ", y, "*im)")

Base.show(io::IO, ::Nop) = print(io, "nop")
Base.show(io::IO, ::End) = print(io, "end")
Base.show(io::IO, i::Uni) = print(io, i.dst, " := ", i.op, "(", i.s1, ")")
Base.show(io::IO, i::Bi) = print(io, i.dst, " := ", i.s1, " ", i.op, " ", i.s2)
Base.show(io::IO, i::Mov) = print(io, i.dst, " := ", i.s1)
Base.show(io::IO, i::Load) = print(io, i.dst, " := ", i.loc)
Base.show(io::IO, i::Save) = print(io, i.loc, " := ", i.src)
Base.show(io::IO, i::LoadConst) = print(io, i.dst, " := consts[", i.idx, "]")
Base.show(io::IO, i::LoadComplex) = (cx(io, i.xd, i.yd); print(io, " := ", i.loc))
Base.show(io::IO, i::SaveComplex) = (print(io, i.loc, " := "); cx(io, i.xs, i.ys))
Base.show(io::IO, i::LoadArgs) =
    print(io, "Load Args [", join(sprint.(show, i.locs), ", "), "]; ", i.complex ? "complex" : "real", ", ", i.ultra)
Base.show(io::IO, i::SaveArgs) = print(io, "Save Args; n = ", i.num_args, "; ", i.complex ? "complex" : "real", ", ", i.ultra)
function Base.show(io::IO, i::Fused)
    (neg, sign) = (i.op in (:NegMulAdd, :NegMulSub) ? "- " : "", i.op in (:MulAdd, :NegMulAdd) ? "+" : "-")
    print(io, i.dst, " := ", neg, i.a, " * ", i.b, " ", sign, " ", i.c)
end
Base.show(io::IO, i::IfElse) = print(io, i.dst, " := ", i.cond, " ? ", i.true_val, " : ", i.false_val)
Base.show(io::IO, i::Label) = print(io, quoted(i.label), ":")
Base.show(io::IO, i::Branch) = print(io, "goto ", quoted(i.label))
Base.show(io::IO, i::BranchIf) = print(io, i.is_else ? "if not " : "if ", i.cond, " goto ", quoted(i.label))
Base.show(io::IO, i::Call) = print(io, "call ", i.label)
Base.show(io::IO, i::LoadMath) = print(io, i.dst, " := ", i.s1, " ", i.op, " ", i.loc, " # load/math")
Base.show(io::IO, i::LoadConstMath) = print(io, i.dst, " := ", i.s1, " ", i.op, " consts[", i.idx, "] # load const/math")
Base.show(io::IO, i::ComplexBi) = (cx(io, i.xd, i.yd); print(io, " := "); cx(io, i.x1, i.y1); print(io, " ", i.op, " "); cx(io, i.x2, i.y2))

# a listing of the MIR: one instruction a line, numbered from 0
function listing(io::IO, mir::MIR)
    for (k, ins) in enumerate(mir.code)
        println(io, lpad(k - 1, 5, '0'), "\t", ins)
    end
end
listing(mir::MIR) = listing(stdout, mir)

function Base.show(io::IO, ::MIME"text/plain", a::SavedApplication)
    println(io, "SavedApplication (sjb version $(a.version)), ", is_complex(a.config) ? "complex" : "real",
            ", opt level $(opt_level(a.config)): $(join(options(a.config), ", "))")
    println(io, "  states $(a.count_states), params $(a.count_params), observables $(a.count_obs), ",
            "derivatives $(a.count_diffs), loops $(a.count_loops), stack $(a.stack_size), memory $(mem_size(a))")
    println(io, "  $(length(a.mir.code)) instructions, $(length(a.mir.consts)) constants, functions: ",
            isempty(a.functions) ? "none" : join(a.functions, ", "))
end

include(joinpath(@__DIR__, "complexify.jl"))
