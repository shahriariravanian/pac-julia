mutable struct CodeWriterCpu
    fd::IOStream
    indent::Int
    path::String

    function CodeWriterCpu(path)
        mkpath(path)

        fd = open(joinpath(path, "runner.c"), "w")
        writer = new(fd, 0, path)

        copy_home(writer, "c/pacops.c")
        copy_home(writer, "c/pacops.h")

        return writer
    end
end

Base.close(writer::CodeWriterCpu) = close(writer.fd)

function emit(writer::CodeWriterCpu, s)
    for i in 1:writer.indent
        print(writer.fd, '\t')
    end

    writer.indent = max(writer.indent + count(isequal('{'), s) - count(isequal('}'), s), 0)

    println(writer.fd, s)
end

function newline(writer::CodeWriterCpu)
    println(writer.fd)
end

# `file` relative to the repository (c/pacops.c), or an absolute path, into the output folder
function copy_home(writer::CodeWriterCpu, file)
    name = basename(file)
    cp(joinpath(@__DIR__, "..", file), joinpath(writer.path, name); force=true)
end

#####################################################################################
#
# The C is generated from a Lowered (UIR v0.8, uir.jl): compile(path, p::Program) lowers `p`
# first. Indices are 1-based in the Lowered and 0-based in the C: the conversions (`- 1`) are made
# here, where the C is written, and nowhere else.
#
# A fused run (FIRST ... LAST) is written as its rows one by one, each storing with its own mode:
# by the run invariants (validate) no row of a run reads its destination and the sums keep their
# order, so the values are the run's, and the C compiler may keep the destination in registers.

# The arena of runner.c, in doubles (0-based offsets): the forms (4 each), the amplitudes (2 each),
# then the planes (2 each), a current at planes + 2 × its plane offset
struct ArenaIndex
    forms::Int        # the first form's offset
    amplitudes::Int   # the first amplitude's
    planes::Int       # plane offset 0's
    size::Int         # the arena's doubles (RUNNER_ARENA_SIZE)
end

function ArenaIndex(l::Lowered)
    forms = 0
    amplitudes = forms + 4 * length(l.forms)
    planes = amplitudes + 2 * l.amplitudes
    return ArenaIndex(forms, amplitudes, planes, planes + 2 * l.planes)
end

# the arena offsets of a current, an edge's current, a form and an amplitude (`- 1`: 1-based → 0-based)
current_at(b::ArenaIndex, l::Lowered, c::CurrentIndex) = b.planes + 2 * l.currents[c].start
function edge_at(b::ArenaIndex, l::Lowered, e::EdgeIndex)
    e == 0 && error("an instruction reads a null operand")
    return current_at(b, l, edge_current(l, e))
end
form_at(b::ArenaIndex, f::FormIndex) = (f == 0 && error("an instruction reads no momentum form"); b.forms + 4 * (f - 1))
amplitude_at(b::ArenaIndex, a::AmplitudeIndex) = b.amplitudes + 2 * (a - 1)
scale(i::ScaleIndex) = "scales[$(2*(i-1))]"
factor(i::FactorIndex) = "factors[$(2*(i-1))]"

# a double as a C literal: Julia prints the shortest string that reads back as the same double,
# which C reads back exactly too; NaN and Inf have no C literal
function lit(x::Real)::String
    y = Float64(x)
    isfinite(y) || error("$y has no C literal")
    return repr(y)
end

# pacops.h's names of the opcodes and source families
const C_VERTEX = Dict(V_WEYL_VECTOR_A => "VERTEX_WEYL_VECTOR_A", V_WEYL_VECTOR_B => "VERTEX_WEYL_VECTOR_B",
                      V_TENSOR_VECTOR => "VERTEX_TENSOR_VECTOR", V_VECTOR_WEDGE => "VERTEX_VECTOR_WEDGE",
                      V_THREE_VECTOR => "VERTEX_THREE_VECTOR")
const C_PROPAGATOR = Dict(P_WEYL_A => "PROPAGATOR_WEYL_A", P_WEYL_B => "PROPAGATOR_WEYL_B", P_VECTOR => "PROPAGATOR_VECTOR")
const C_FAMILY = Dict(SCALAR => "FAMILY_SCALAR", WEYL => "FAMILY_WEYL", DIRAC => "FAMILY_DIRAC", VECTOR => "FAMILY_VECTOR")
c_mode(op::LOp) = op.set ? "MODE_SET" : "MODE_ADD"

# The folder `path` for program `prog`: runner.c (`int run(double *out, double *p, const double
# *momenta)`), runner.h (its sizes and prototype), pacops.c and pacops.h, and a Makefile (`make`: runner.o and pacops.o). With `examples`, the program's entry of examples.json
# (load_examples), also testing.c, which runs those points and compares with RustiCol's outputs
# (`make test`). A Program is lowered first, and with `optimize` (the default)
# sorted, fused and laid out by liveness (passes.jl); a Lowered is compiled as it is. With
# `interpreter`, runner.c runs the program on the UIR interpreter (compile_vm.jl, c/uirvm.c) instead
# of straight-line code, with the same runner.h and values. The kernels are C in runner.c, made
# from their .plane.symjit (kernel_c.jl), bit for bit sjb2obj's libraries: the folder needs neither.
function compile(path, prog::Lowered; examples = nothing, interpreter::Bool = false)
    interpreter && return compile_interpreter(path, prog; examples)
    validate(prog)
    b = ArenaIndex(prog)
    examples === nothing || check_examples(prog, examples)
    writer = CodeWriterCpu(path)
    try
        emit(writer, "#include <stdlib.h>")
        emit(writer, "#include \"pacops.h\"")
        emit(writer, "#include \"runner.h\"")
        newline(writer)

        compile_scales(writer, prog)
        compile_factors(writer, prog)

        compile_kernels(writer, prog)

        compile_forms(writer, prog, b)
        compile_tables(writer, prog)
        compile_body(writer, prog, b)

        # the arena and the outputs at zero, the forms, the rows (REDUCE rows add to out), the group factor
        (num, den) = prog.group_factor
        emit(writer, "int run(double *out, double *p, const double *momenta) {")
        emit(writer, "for(int i=0; i<$(b.size); i++) p[i] = 0.0;")
        emit(writer, "for(int h=0; h<$(output_count(prog)); h++) out[h] = 0.0;")
        emit(writer, "forms(p, momenta);")
        emit(writer, "body(out, p);")
        num == den || emit(writer, "for(int h=0; h<$(output_count(prog)); h++) out[h] = group(out[h], $(num)L, $(den)L);")
        emit(writer, "return 0;")
        emit(writer, "}")
    finally
        close(writer)
    end
    write(joinpath(path, "runner.h"), runner_h(prog, b))
    examples === nothing || write(joinpath(path, "testing.c"), testing_c(prog, examples))
    write(joinpath(path, "Makefile"), makefile(prog, examples !== nothing))
    return path
end

# a string as a C string literal
cstring(s::AbstractString) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""

# the kernels of `prog` in order of first use by a stub, each checked against its stubs: a stub
# lists as many inputs and output pairs as its kernel has (the outputs array would otherwise be
# read past its end)
function used_kernels(prog::Lowered)::Vector{KernelLibrary}
    kernels = KernelLibrary[]
    for stub in prog.stubs
        ker = stub.kernel
        length(stub.inputs) == ker.inputs && length(stub.outputs) == ker.outputs ÷ 2 ||
            error("a binding of kernel $(ker.name) lists $(length(stub.inputs)) inputs, $(length(stub.outputs)) outputs; " *
                  "the kernel has $(ker.inputs), $(ker.outputs ÷ 2)")
        any(k -> k.name == ker.name, kernels) || push!(kernels, ker)
    end
    return kernels
end

# the C of the kernels (kernel_c.jl), the helpers first: `static int <name>(double *mem)` each
# mixed: each kernel also in mixed precision (kernel_c_mixed), the one compiled under PAC_MIXED
function emit_kernels(writer, kernels::Vector{KernelLibrary}; mixed::Bool = false)
    isempty(kernels) && return
    lines(text) = foreach(l -> emit(writer, l), split(chomp(text), '\n'))
    lines(KERNEL_PRELUDE)
    for ker in kernels
        isfile(ker.source) || error("kernel $(ker.name): no .plane.symjit at $(ker.source)")
        if mixed
            app = read_sjb(ker.source; lowered = true)
            emit(writer, "#ifdef PAC_MIXED")
            lines(kernel_c_mixed(ker.name, app))
            emit(writer, "#else")
            lines(kernel_c(ker.name, app))
            emit(writer, "#endif")
        else
            lines(kernel_c(ker.name, ker.source))
        end
        newline(writer)
    end
end

# `examples` must be this program's entry, its points of the program's shape
function check_examples(prog::Lowered, examples)
    (examples.process, examples.colour) == (prog.process, prog.colour) ||
        error("the examples are of $(examples.process) $(examples.colour), the program $(prog.process) $(prog.colour)")
    isempty(examples.points) && error("the examples have no points")
    for pt in examples.points
        length(pt.momenta) == prog.legs && all(v -> length(v) == 4, pt.momenta) && length(pt.outputs) == output_count(prog) ||
            error("an example point has $(length(pt.momenta)) momenta and $(length(pt.outputs)) outputs; " *
                  "the program has $(prog.legs) legs and $(output_count(prog)) outputs")
    end
end

function runner_h(prog::Lowered, b::ArenaIndex)::String
    return """
    /* $(prog.process) $(prog.colour): the program compiled by PacRunner's compile (runner.c).
     * run(out, p, momenta) evaluates one phase-space point: momenta[4 * RUNNER_LEGS] are the caller's
     * legs (E, px, py, pz each, before the crossing), out[RUNNER_OUTPUTS] receives the helicity
     * outputs (group factor applied; the matrix element is their sum), p[RUNNER_ARENA_SIZE] is
     * scratch. Returns 0. */
    #ifndef RUNNER_H
    #define RUNNER_H

    #define RUNNER_PROCESS $(cstring(prog.process))
    #define RUNNER_COLOUR $(cstring(prog.colour))
    #define RUNNER_LEGS $(prog.legs)
    #define RUNNER_OUTPUTS $(output_count(prog))
    #define RUNNER_ARENA_SIZE $(b.size)

    #ifdef __cplusplus
    extern "C" {
    #endif

    int run(double *out, double *p, const double *momenta);

    #ifdef __cplusplus
    }
    #endif

    #endif
    """
end

function testing_c(prog::Lowered, examples)::String
    points = examples.points
    rows(f) = join(["    {" * join([lit(Float64(x)) for x in f(pt)], ", ") * "}" for pt in points], ",\n")
    momenta = rows(pt -> (x for v in pt.momenta for x in v))
    expected = rows(pt -> pt.outputs)
    return """
    /* The tests of $(prog.process) $(prog.colour), generated by PacRunner's compile: run() (runner.c)
     * on the $(length(points)) points of examples.json, each output compared with RustiCol's (the CPU's
     * outputs there) within TOLERANCE × |the point's total|.
     * usage: ./testing [TOLERANCE]     (default 1e-14; `make test`)
     * Exits 0 when every output is within the tolerance. The C is bit for bit the Julia runner's;
     * RustiCol's SIMD kernels may round differently in the last bits, hence the tolerance. */
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <float.h>
    #include "runner.h"

    #define POINTS $(length(points))

    static const double momenta[POINTS][4 * RUNNER_LEGS] = {
    $momenta
    };

    static const double expected[POINTS][RUNNER_OUTPUTS] = {
    $expected
    };

    static double p[RUNNER_ARENA_SIZE];

    int main(int argc, char **argv)
    {
        double tolerance = argc > 1 ? atof(argv[1]) : 1e-14;
        double out[RUNNER_OUTPUTS], worst = 0.0;
        int beyond = 0, identical = 0;
        for (int i = 0; i < POINTS; i++) {
            if (run(out, p, momenta[i]) != 0) {
                printf("point %d: run failed\\n", i);
                return 1;
            }
            double total = 0.0;
            for (int h = 0; h < RUNNER_OUTPUTS; h++) total += expected[i][h];
            double scale = fabs(total) > 0.0 ? fabs(total) : DBL_MIN;
            for (int h = 0; h < RUNNER_OUTPUTS; h++) {
                double d = fabs(out[h] - expected[i][h]) / scale;
                if (memcmp(&out[h], &expected[i][h], sizeof(double)) == 0) identical++;
                if (!(d <= tolerance)) {
                    if (beyond < 10)
                        printf("point %d, output %d: %.17g, RustiCol %.17g (%.3g of the total)\\n", i, h, out[h], expected[i][h], d);
                    beyond++;
                }
                if (d > worst || d != d) worst = d;
            }
        }
        printf("%s %s: %d points, %d outputs; %d beyond %g of the total (worst %.3g), %d bit-identical to RustiCol%s\\n",
               RUNNER_PROCESS, RUNNER_COLOUR, POINTS, POINTS * RUNNER_OUTPUTS, beyond, tolerance, worst, identical,
               beyond ? "  FAILED" : "");
        return beyond ? 1 : 0;
    }
    """
end

function makefile(prog::Lowered, testing::Bool; interpreter::Bool = false)::String
    vm = interpreter ? " uirvm.c uirvm.h uirgpu.c uirgpu.h pool.cu pool.h" : ""
    return """
    # $(prog.process) $(prog.colour), compiled by PacRunner's compile.
    #   make         runner.o and pacops.o$(testing ? ", and testing" : "")
    $(testing ? "#   make test    runs testing: the points of examples.json against RustiCol's outputs\n" : "")$(interpreter ? "#   make gpu     runner_gpu.o (nvcc): run_gpu, every point on the GPU (uirvm.h)\n" : "")#   make clean
    # CFLAGS may be changed; PACFLAGS may not: the values are bit for bit the Julia runner's only
    # without contracted multiply-adds (-ffp-contract=off).
    CC ?= cc
    CFLAGS ?= -O2
    PACFLAGS = -std=c99 -ffp-contract=off
    all: runner.o pacops.o$(testing ? " testing" : "")

    runner.o: runner.c runner.h pacops.h$vm
    \t\$(CC) \$(CFLAGS) \$(PACFLAGS) -c -o \$@ runner.c

    pacops.o: pacops.c pacops.h
    \t\$(CC) \$(CFLAGS) \$(PACFLAGS) -c -o \$@ pacops.c
    $(interpreter ? """

    # the same runner.c on the GPU; NVFLAGS may be changed, NVPACFLAGS may not (bit for bit the CPU's
    # only without contracted multiply-adds)
    NVCC ?= nvcc
    NVFLAGS ?= -O3 -arch=native
    NVPACFLAGS = --fmad=false
    gpu: runner_gpu.o

    runner_gpu.o: runner.c runner.h pacops.c pacops.h$vm
    \t\$(NVCC) \$(NVFLAGS) \$(NVPACFLAGS) -x cu -c -o \$@ runner.c
    """ : "")$(testing ? """

    testing: testing.c runner.h runner.o pacops.o
    \t\$(CC) \$(CFLAGS) \$(PACFLAGS) -o \$@ testing.c runner.o pacops.o -lm

    test: testing
    \t./testing
    """ : "")
    clean:
    \trm -f runner.o pacops.o$(testing ? " testing" : "")$(interpreter ? " runner_gpu.o" : "")

    .PHONY: all$(testing ? " test" : "")$(interpreter ? " gpu" : "") clean
    """
end
function compile_scales(writer, prog::Lowered)
     emit(writer, "const double scales[] = {")

     for c in prog.scales
         emit(writer, "$(lit(real(c))), $(lit(imag(c))),")
     end

     emit(writer, "};")
     newline(writer)
end

function compile_factors(writer, prog::Lowered)
     emit(writer, "const double factors[] = {")

     for c in prog.factors
         emit(writer, "$(lit(real(c))), $(lit(imag(c))),")
     end

     emit(writer, "};")
     newline(writer)
end

# the constant arrays of the rows, at file scope (C++ and CUDA have no compound literals): the
# component factors of closure list t, `cf<t>`; the output components of stub s (0-based), `outputs<s>`
function compile_tables(writer, prog::Lowered)
    for (t, list) in enumerate(prog.lists)
        isempty(list) && continue
        values = join(["$(lit(real(prog.factors[x]))), $(lit(imag(prog.factors[x])))" for x in list], ", ")
        emit(writer, "static const double cf$t[] = {$values};")
    end
    for (s, stub) in enumerate(prog.stubs)
        emit(writer, "static const int outputs$s[] = {$(join(stub.outputs .- 1, ", "))};")
    end
    # embedding k (0-based): the current's components, the wavefunction's (-1: zero), the factors
    for (k, list) in enumerate(prog.embeddings)
        emit(writer, "static const int emb_components$k[] = {$(join([j - 1 for (j, _, _) in list], ", "))};")
        emit(writer, "static const int emb_waves$k[] = {$(join([w - 1 for (_, w, _) in list], ", "))};")
        values = join(["$(lit(real(prog.factors[x]))), $(lit(imag(prog.factors[x])))" for (_, _, x) in list], ", ")
        emit(writer, "static const double emb_factors$k[] = {$values};")
    end
    newline(writer)
end

# one adapter per kernel: its inputs as arguments, then its output pairs into `out` (as they are
# for KERNEL_DIRECT, else × factor with the mode)
function compile_kernels(writer, prog::Lowered)
    kernels = used_kernels(prog)
    emit_kernels(writer, kernels)
    for ker in kernels
        name = ker.name
        inputs = ker.inputs
        outputs = ker.outputs

        s = "void adapter_$name(double *out, "

        for i in 0:inputs-1
            s *= "double x$i, "
        end

        emit(writer, "$s const int *outputs, int direct, int mode, const double *factor) {")

        emit(writer, "double mem[$(inputs+outputs+1)];")
        for i in 0:inputs-1
            emit(writer, "mem[$i] = x$i;")
        end

        emit(writer, "if ($name(mem) != 0) abort();  /* the kernel failed */")

        emit(writer, "cplx f = get(factor, 0);")
        emit(writer, "for (int n = 0; n < $(outputs ÷ 2); n++) {")
        emit(writer, "cplx pair = get(mem + $inputs, n);")
        emit(writer, "put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));")
        emit(writer, "}")

        emit(writer, "}")
        newline(writer)
    end
end

function compile_forms(writer, prog::Lowered, b::ArenaIndex)
    emit(writer, "void forms(double *p, const double *momenta) {")

    # the program's leg l: the caller's leg `leg` (1-based) times `sign`, or the caller's leg l
    # (as evaluators/uireval.jl's fill_forms!: q = 0.0 + Σ coef × (sign × p[leg]), in term order)
    external(l, j) = prog.crossing === nothing ? "momenta[$(4*(l-1)+j)]" :
        "($(lit(prog.crossing[l][2])) * momenta[$(4*(prog.crossing[l][1]-1)+j)])"

    for (i, f) in enumerate(prog.forms)
        q = form_at(b, i)
        for j in 0:3
            s = "p[$(q+j)] = 0.0"
            for (leg, coef) in f.terms
                s *= " + $(lit(coef)) * $(external(leg, j))"
            end
            s *= ";"
            emit(writer, s)
        end
    end

    emit(writer, "}")
    newline(writer)
end

function compile_body(writer, prog::Lowered, b::ArenaIndex)
    emit(writer, "void body(double *out, double *p) {")

    for (k, op) in enumerate(prog.ops)
        emit(writer, "// $k: $(describe(prog, op))")
        compile_op(writer, op, prog, b)
    end

    emit(writer, "}")
    newline(writer)
end

function compile_op(writer, op::LOp, prog::Lowered, b::ArenaIndex)
    p(e) = "&p[$(edge_at(b, prog, e))]"
    momentum(e) = "&p[$(form_at(b, edge_form(prog, e)))]"
    if is_vertex(op.op)
        (u, v) = op.op == V_THREE_VECTOR ? (momentum(op.l), momentum(op.r)) : ("NULL", "NULL")
        emit(writer, "vertex($(p(op.d)), $(C_VERTEX[op.op]), $(c_mode(op)), $(p(op.l)), $(p(op.r)), $u, $v, " *
                     "&$(scale(op.t)), &$(factor(op.x)));")
    elseif is_propagator(op.op)
        emit(writer, "propagator($(p(op.d)), $(C_PROPAGATOR[op.op]), $(p(op.d)), $(momentum(op.d)), &$(factor(op.x)));")
    elseif op.op == SCALE_ROW
        emit(writer, "scale($(p(op.d)), $(op.n), &$(factor(op.x)));")
    elseif op.op == CLEAR
        emit(writer, "clear($(p(op.d)), $(op.n));")
    elseif op.op == SOURCE
        (family, anti, h, ch) = source_fields(op)
        ori = anti ? "ANTIPARTICLE" : "PARTICLE"
        mass = lit(real(prog.scales[op.t]))
        if op.l == 0
            emit(writer, "source($(p(op.d)), $(C_FAMILY[family]), $ori, $(momentum(op.d)), $h, $ch, $mass, &$(factor(op.x)));")
        else
            k = op.l
            emit(writer, "source_embedded($(p(op.d)), $(C_FAMILY[family]), $ori, $(momentum(op.d)), $h, $ch, $mass, " *
                         "emb_components$k, emb_waves$k, emb_factors$k, $(length(prog.embeddings[k])));")
        end
    elseif op.op == CLEAR_AMPLITUDE
        emit(writer, "p[$(amplitude_at(b, op.d))] = 0.0; p[$(amplitude_at(b, op.d) + 1)] = 0.0;")
    elseif op.op == COPY
        emit(writer, "copy_row($(p(op.d)), $(p(op.l)), $(edge_layout(prog, op.d).components), &$(factor(op.x)));")
    elseif op.op == SCATTER
        emit(writer, "scatter($(p(op.d)), $(p(op.l)), $(edge_layout(prog, op.d).components), $(c_mode(op)), &$(factor(op.x)));")
    elseif op.op == REDUCE
        sc = op.x == 0 ? "NULL" : "&$(factor(op.x))"
        emit(writer, "out[$(op.d - 1)] = reduce_term(out[$(op.d - 1)], &p[$(amplitude_at(b, op.t))], $sc, $(lit(prog.weights[op.n])));")
    elseif op.op == CLOSURE
        list = prog.lists[op.t]
        cf = isempty(list) ? "NULL" : "cf$(op.t)"
        emit(writer, "closure(&p[$(amplitude_at(b, op.d))], $(p(op.l)), $(p(op.r)), $cf, $(length(list)), &$(factor(op.x)));")
    elseif is_kernel(op.op)
        (s, set) = prog.instances[op.t]
        stub = prog.stubs[s]
        args = join([kernel_arg(i, op, prog, b, prog.scalesets[set]) for i in stub.inputs], ", ")
        direct = op.op == KERNEL_DIRECT ? 1 : 0
        emit(writer, "adapter_$(stub.kernel.name)($(p(op.d)), $args, outputs$s, $direct, $(c_mode(op)), &$(factor(op.x)));")
    else
        error("unknown opcode $(op.op)")
    end
end

# a kernel input as a C expression: a part of an operand's component, a momentum component, the
# real part of a scale (by its slot in the scale set), a part of the row's factor
function kernel_arg(i::KernelInput, op::LOp, prog::Lowered, b::ArenaIndex, scaleset::Vector{ScaleIndex})::String
    j = i.n - 1
    im_ = i.im ? 1 : 0
    if i.kind == IN_L
        return "p[$(edge_at(b, prog, op.l) + 2*j + im_)]"
    elseif i.kind == IN_R
        return "p[$(edge_at(b, prog, op.r) + 2*j + im_)]"
    elseif i.kind == OUT
        return "p[$(edge_at(b, prog, op.d) + 2*j + im_)]"
    elseif i.kind == MOMENTUM_U
        return "p[$(form_at(b, edge_form(prog, op.l)) + j)]"
    elseif i.kind == MOMENTUM_V
        return "p[$(form_at(b, edge_form(prog, op.r)) + j)]"
    elseif i.kind == SCALE
        return scale(scaleset[i.n])
    else  # FACTOR
        return "factors[$(2*(op.x-1) + im_)]"
    end
end
