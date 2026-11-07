include("types.jl")

mutable struct CodeWriterCpu
    fd::IOStream
    indent::Int
    path::String

    function CodeWriterCpu(path)
        if !ispath(path)
            mkdir(path)
        end

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

# `file` relative to this directory (or an absolute path) into the output folder
function copy_home(writer::CodeWriterCpu, file)
    name = basename(file)
    cp(joinpath(@__DIR__, file), joinpath(writer.path, name); force=true)
end

#####################################################################################

# an operand the instruction reads: a null operand (index 0) is an error here, not `&NULL` in the C
function current(b::ArenaIndex, i::CurrentIndex, p = "p")
    i == 0 && error("an instruction reads a null current operand")
    return "$p[$(b.currents[i])]"
end
form(b::ArenaIndex, i::FormIndex, p = "p") = "$p[$(b.forms[i])]"
amplitude(b::ArenaIndex, i::AmplitudeIndex, p = "p") = "$p[$(b.amplitudes[i])]"
scale(i::ScaleIndex) = "scales[$(2*(i-1))]"
factor(i::FactorIndex) = "factors[$(2*(i-1))]"

# a double as a C literal: Julia prints the shortest string that reads back as the same double,
# which C reads back exactly too; NaN and Inf have no C literal
function lit(x::Real)::String
    y = Float64(x)
    isfinite(y) || error("$y has no C literal")
    return repr(y)
end

const FAMILY_SCALAR = 0
const FAMILY_WEYL = 1
const FAMILY_DIRAC = 2
const FAMILY_VECTOR = 3

const PARTICLE = 0
const ANTIPARTICLE = 1

const MODE_SET = 0
const MODE_ADD = 1

const VERTEX_WEYL_VECTOR_A = 0
const VERTEX_WEYL_VECTOR_B = 1
const VERTEX_TENSOR_VECTOR = 2
const VERTEX_VECTOR_WEDGE = 3
const VERTEX_THREE_VECTOR = 4

const PROPAGATOR_WEYL_A = 0
const PROPAGATOR_WEYL_B = 1
const PROPAGATOR_VECTOR = 2


function compile(path, prog)
    writer = CodeWriterCpu(path)
    try
        a = Arena(prog)
        b = ArenaIndex(prog, a)

        emit(writer, "#include <stdlib.h>")
        emit(writer, "#include \"pacops.h\"")
        newline(writer)

        compile_scales(writer, prog)
        compile_factors(writer, prog)

        compile_kernels(writer, prog)

        compile_forms(writer, prog, b)
        compile_currents(writer, prog, b)
        compile_tables(writer, prog)
        compile_body(writer, prog, b)
        compile_reduction(writer, prog, b)

        emit(writer, "int run(double *out, double *p, const double *momenta) {")
        emit(writer, "for(int i=0; i<$(b.size); i++) p[i] = 0.0;")
        emit(writer, "forms(p, momenta);")
        emit(writer, "currents(p);")
        emit(writer, "body(p);")
        emit(writer, "reducer(out, p);")
        emit(writer, "return 0;")
        emit(writer, "}")
    finally
        close(writer)
    end
end

function compile_scales(writer, prog::Program)
     emit(writer, "const double scales[] = {")

     for c in prog.scales
         emit(writer, "$(lit(real(c))), $(lit(imag(c))),")
     end

     emit(writer, "};")
     newline(writer)
end

function compile_factors(writer, prog::Program)
     emit(writer, "const double factors[] = {")

     for c in prog.factors
         emit(writer, "$(lit(real(c))), $(lit(imag(c))),")
     end

     emit(writer, "};")
     newline(writer)
end

# the constant arrays of the instructions, at file scope (C++ and CUDA have no compound literals):
# closure k's component factors `cf<k>`, kernel row k's output components `outputs<k>`
function compile_tables(writer, prog::Program)
    for (k, op) in enumerate(prog.body)
        if op isa Closure && !isempty(op.component_factors)
            values = join(["$(lit(real(prog.factors[x]))), $(lit(imag(prog.factors[x])))" for x in op.component_factors], ", ")
            emit(writer, "static const double cf$k[] = {$values};")
        elseif op isa Kernel
            emit(writer, "static const int outputs$k[] = {$(join(prog.bindings[op.binding].outputs .- 1, ", "))};")
        end
    end
    newline(writer)
end

function compile_kernels(writer, prog::Program)
    done = Set{String}()
    for binding in prog.bindings
        ker = binding.kernel
        # the adapter takes the kernel's inputs and reads its output pairs: the binding must list
        # as many (the outputs array would otherwise be read past its end)
        length(binding.inputs) == ker.inputs && length(binding.outputs) == ker.outputs ÷ 2 ||
            error("a binding of kernel $(ker.name) lists $(length(binding.inputs)) inputs, $(length(binding.outputs)) outputs; " *
                  "the kernel has $(ker.inputs), $(ker.outputs ÷ 2)")
        ker.name in done && continue
        push!(done, ker.name)
        name = ker.name
        inputs = ker.inputs
        outputs = ker.outputs

        emit(writer, "int $name(double *mem, const void *states, size_t idx, const double *params);")
        newline(writer)

        s = "void adapter_$name(double *out, "

        for i in 0:inputs-1
            s *= "double x$i, "
        end

        emit(writer, "$s const int *outputs, int direct, int mode, const double *factor) {")

        emit(writer, "double mem[$(inputs+outputs+1)];")
        for i in 0:inputs-1
            emit(writer, "mem[$i] = x$i;")
        end

        emit(writer, "if ($name(mem, NULL, 0, NULL) != 0) abort();  /* the kernel failed */")

        emit(writer, "cplx f = get(factor, 0);")
        emit(writer, "int as_is = direct && f.re == 1.0 && f.im == 0.0 && !signbit(f.im);  /* exactly 1 + 0i */");
        emit(writer, "for (int n = 0; n < $(outputs ÷ 2); n++) {")
        emit(writer, "cplx pair = get(mem + $inputs, n);")
        emit(writer, "put(out, outputs[n], as_is ? pair : combine(get(out, outputs[n]), mul(pair, f), mode));")
        emit(writer, "}")

        emit(writer, "}")
        newline(writer)

        copy_home(writer, ker.path)
    end
end

function compile_forms(writer, prog::Program, b::ArenaIndex)
    emit(writer, "void forms(double *p, const double *momenta) {")

    # the program's leg l: the caller's leg `leg` (1-based) times `sign`, or the caller's leg l
    # (as pacrun.jl's fill_forms!: q = 0.0 + Σ coef × (sign × p[leg]), in term order)
    external(l, j) = prog.crossing === nothing ? "momenta[$(4*(l-1)+j)]" :
        "($(lit(prog.crossing[l][2])) * momenta[$(4*(prog.crossing[l][1]-1)+j)])"

    for (i, f) in enumerate(prog.forms)
        q = b.forms[i]
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

function compile_currents(writer, prog::Program, b::ArenaIndex)
    emit(writer, "void currents(double *p) {")

    for (i, c) in enumerate(prog.currents)
        idx = b.currents[i]
        n = b.currents[i+1] - b.currents[i]
        s = c.source
        if s === nothing
            # zero or written: run() zeroes the arena
        else
            ori = s.orientation == "antiparticle" ? ANTIPARTICLE : PARTICLE

            if s.family == "scalar"
                f = FAMILY_SCALAR
            elseif s.family == "weyl"
                f = FAMILY_WEYL
            elseif s.family == "dirac"
                f = FAMILY_DIRAC
            elseif s.family == "vector"
                f = FAMILY_VECTOR
            else
                error("unknown source family $(s.family)")
            end

            emit(writer, "source(p+$idx, $f, $ori, p+$(b.forms[s.form]), $(s.helicity), $(s.chirality), $(lit(real(prog.scales[s.mass]))), &$(factor(s.factor)));")
        end
    end

    emit(writer, "}")
    newline(writer)
end


function compile_body(writer, prog::Program, b::ArenaIndex)
    emit(writer, "void body(double *p) {")

    for (k, op) in enumerate(prog.body)
        compile_op(writer, op, prog, b, k)
    end

    emit(writer, "}")
    newline(writer)
end

function type(op::Vertex)
    type = op.type

    if type == "weyl_vector_a"
        return VERTEX_WEYL_VECTOR_A
    elseif type == "weyl_vector_b"
        return VERTEX_WEYL_VECTOR_B
    elseif type == "tensor_vector"
        return VERTEX_TENSOR_VECTOR
    elseif type == "vector_wedge"
        return VERTEX_VECTOR_WEDGE
    elseif type == "three_vector"
        return VERTEX_THREE_VECTOR
    else
        error("unknown vertex type $type")
    end
end

function type(op::Propagator)
    type = op.type

    if type == "weyl_a"
        return PROPAGATOR_WEYL_A
    elseif type == "weyl_b"
        return PROPAGATOR_WEYL_B
    elseif type == "vector"
        return PROPAGATOR_VECTOR
    else
        error("unknown propagator type $type")
    end
end

function mode(op)
    if op.mode == "="
        return MODE_SET
    elseif op.mode == "+="
        return MODE_ADD
    else
        error("unknown mode $(op.mode)")
    end
end

function compile_op(writer, op::Vertex, prog::Program, b::ArenaIndex, k::Int)
    emit(writer, "// $op")

    # vertex!(a.currents[op.out], op.type, op.mode, current(a, op.l), current(a, op.r), form(a, op.u), form(a, op.v),
    #        p.scales[op.coupling], p.factors[op.factor])
    s = "vertex(&$(current(b,op.out)), $(type(op)), $(mode(op)), &$(current(b,op.l)), &$(current(b,op.r)), &$(form(b,op.u)), &$(form(b,op.v)), &$(scale(op.coupling)), &$(factor(op.factor)));"
    emit(writer, s)
end

function compile_op(writer, op::Propagator, prog::Program, b::ArenaIndex, k::Int)
    emit(writer, "// $op")
    # c = a.currents[op.current]
    # v = propagator(op.type, c, form(a, op.momentum), p.factors[op.factor])
    # c[1:length(v)] = v
    s = "propagator(&$(current(b,op.current)), $(type(op)), &$(current(b,op.current)), &$(form(b,op.momentum)), &$(factor(op.factor)));"
    emit(writer, s)
end

function compile_op(writer, op::Scale, prog::Program, b::ArenaIndex, k::Int)
    # scale!(a.currents[op.current], op.components, p.factors[op.factor])
    components = op.components
    s = "scale(&$(current(b,op.current)), $components, &$(factor(op.factor)));"
    emit(writer, s)
end

function compile_op(writer, op::Clear, prog::Program, b::ArenaIndex, k::Int)
    # clear!(a.currents[op.current], op.components)
    components = op.components
    s = "clear(&$(current(b,op.current)), $components);"
    emit(writer, s)
end

function compile_op(writer, op::Closure, prog::Program, b::ArenaIndex, k::Int)
    emit(writer, "// $op")

    cf = isempty(op.component_factors) ? "NULL" : "cf$k"

    # cf = [p.factors[k] for k in op.component_factors]
    # a.amplitudes[op.amplitude] = closure(a.amplitudes[op.amplitude], current(a, op.l), current(a, op.r), cf, p.factors[op.factor])
    s = "closure(&$(amplitude(b,op.amplitude)), &$(current(b,op.l)), &$(current(b,op.r)), $cf, $(length(op.component_factors)), &$(factor(op.factor)));"
    emit(writer, s)
end

function kernel_arg(i::KernelInput, op::Kernel, b::ArenaIndex)::String
    j = i.n - 1

    if i.kind == IN_L
        k = b.currents[op.l] + 2*j + (i.im ? 1 : 0)
        return  "p[$k]"
    elseif i.kind == IN_R
        k = b.currents[op.r] + 2*j + (i.im ? 1 : 0)
        return  "p[$k]"
    elseif i.kind == OUT
        k = b.currents[op.out] + 2*j + (i.im ? 1 : 0)
        return  "p[$k]"
    elseif i.kind == MOMENTUM_U
        k = b.forms[op.u] + j
        return  "p[$k]"
    elseif i.kind == MOMENTUM_V
        k = b.forms[op.v] + j
        return  "p[$k]"
    elseif i.kind == SCALE
        return "scales[$(2*j)]"
    else
        error("unsupported kernel input")
    end
end

function compile_op(writer, op::Kernel, prog::Program, b::ArenaIndex, k::Int)
    emit(writer,"// $op")

    binding = prog.bindings[op.binding]
    name = binding.kernel.name

    s = "adapter_$name(&$(current(b,op.out)), "
    for i in binding.inputs
        s *= "$(kernel_arg(i, op, b)), "
    end

    direct = binding.direct ? 1 : 0

    s *= "outputs$k, $direct, $(mode(op)), &$(factor(op.factor)));"
    emit(writer, s)
end

# out[h - 1] = group factor × Σ weight × |amplitude|² over output h's terms (none: 0); the terms
# (0-based amplitudes) and weights as constant arrays `terms<h>`, `weights<h>`
function compile_reduction(writer, prog::Program, b::ArenaIndex)
    for (h, terms) in enumerate(prog.reduction)
        isempty(terms) && continue
        emit(writer, "static const int terms$h[] = {$(join([x[1]-1 for x in terms], ", "))};")
        emit(writer, "static const double weights$h[] = {$(join([lit(x[2]) for x in terms], ", "))};")
    end
    newline(writer)
    emit(writer, "void reducer(double *out, double *p) {")

    (num, den) = prog.group_factor
    for (h, terms) in enumerate(prog.reduction)
        if isempty(terms)
            emit(writer, "out[$(h-1)] = 0.0;")
        else
            emit(writer, "out[$(h-1)] = reduction(&$(amplitude(b, 1)), terms$h, weights$h, $(length(terms)), $num, $den);")
        end
    end

    emit(writer, "}")
    newline(writer)
end
