# compile(path, l; interpreter = true): the folder of compile.jl, but runner.c runs the program on
# the UIR interpreter (c/uirvm.c, which runner.c includes) instead of straight-line code. The
# program is data there: encode(l)'s rows, edge table and sources, and the tables they index, as C
# arrays of one uir_program (c/uirvm.h); the kernels are reached through uir_kernel, a switch over
# the kernels' numbers. runner.h, testing.c and the Makefile are compile.jl's, so run() has the
# same prototype and the values must be the straight-line C's bit for bit.
#
# The CPU build is the interpreter's test; the same uirvm.c is the GPU's interpreter.

# a C array: `static const <ty> <name>[] = {...};` (an empty one holds a single unused 0: C has no
# empty arrays)
function carray(writer, ty::String, name::String, values; per_line = 8)
    if isempty(values)
        emit(writer, "static const $ty $name[1] = {0};")
        return
    end
    emit(writer, "static const $ty $name[] = {")
    v = collect(values)
    for k in 1:per_line:length(v)
        emit(writer, join(v[k:min(k + per_line - 1, end)], ", ") * ",")
    end
    emit(writer, "};")
end

hex64(w::UInt64) = "0x" * string(w; base = 16, pad = 16) * "ull"
cdoubles(zs::Vector{Complex{Float64}}) = [lit(f(z)) for z in zs for f in (real, imag)]
# a "first" table: the offsets of each entry's items, then their total
firsts(lengths) = cumsum([0; collect(lengths)])

# a kernel input as uirvm's code: kind, imaginary part, n (0-based component or scale-set slot)
input_code(i::KernelInput) = UInt32(Int(i.kind)) | UInt32(i.im) << 3 | UInt32(i.kind == FACTOR ? 0 : i.n - 1) << 4

# the kernels in order of first use by a stub, each checked against its stubs and built
function vm_kernels(prog::Lowered)::Vector{KernelLibrary}
    kernels = KernelLibrary[]
    for stub in prog.stubs
        ker = stub.kernel
        length(stub.inputs) == ker.inputs && length(stub.outputs) == ker.outputs ÷ 2 ||
            error("a binding of kernel $(ker.name) lists $(length(stub.inputs)) inputs, $(length(stub.outputs)) outputs; " *
                  "the kernel has $(ker.inputs), $(ker.outputs ÷ 2)")
        any(k -> k.name == ker.name, kernels) && continue
        built(ker) || not_built(ker)
        push!(kernels, ker)
    end
    return kernels
end

function compile_interpreter(path, prog::Lowered; examples = nothing)
    validate(prog)
    b = ArenaIndex(prog)
    examples === nothing || check_examples(prog, examples)
    e = encode(prog)
    kernels = vm_kernels(prog)
    number = Dict(k.name => n - 1 for (n, k) in enumerate(kernels))
    writer = CodeWriterCpu(path)
    try
        copy_home(writer, "c/uirvm.c")
        copy_home(writer, "c/uirvm.h")
        emit(writer, "/* $(prog.process) $(prog.colour) on the UIR interpreter (uirvm.c), compiled by PacRunner's compile */")
        emit(writer, "#include <stdlib.h>")
        emit(writer, "#define UIR_KERNEL_IN $(max(1, maximum((k.inputs for k in kernels); init = 0)))")
        emit(writer, "#define UIR_KERNEL_OUT $(max(1, maximum((k.outputs for k in kernels); init = 0)))")
        emit(writer, "#include \"uirvm.c\"")
        emit(writer, "#include \"runner.h\"")
        newline(writer)

        carray(writer, "uint64_t", "rows", hex64.(e.rows); per_line = 4)
        carray(writer, "uint32_t", "edges", e.edges)
        carray(writer, "int32_t", "sources", [x for s in e.sources for x in s])
        carray(writer, "double", "scales", cdoubles(prog.scales); per_line = 4)
        carray(writer, "double", "factors", cdoubles(prog.factors); per_line = 4)
        carray(writer, "uint32_t", "form_first", firsts(length(f.terms) for f in prog.forms))
        carray(writer, "uint32_t", "term_leg", [leg - 1 for f in prog.forms for (leg, _) in f.terms])
        carray(writer, "double", "term_coefficient", [lit(c) for f in prog.forms for (_, c) in f.terms]; per_line = 4)
        if prog.crossing !== nothing
            carray(writer, "uint32_t", "crossing_leg", [leg - 1 for (leg, _) in prog.crossing])
            carray(writer, "double", "crossing_sign", [lit(s) for (_, s) in prog.crossing]; per_line = 4)
        end
        carray(writer, "uint32_t", "list_first", firsts(length.(prog.lists)))
        carray(writer, "uint32_t", "list_factor", [x - 1 for list in prog.lists for x in list])
        carray(writer, "uint32_t", "stub_kernel", [number[s.kernel.name] for s in prog.stubs])
        carray(writer, "uint32_t", "stub_inputs", firsts(length(s.inputs) for s in prog.stubs))
        carray(writer, "uint32_t", "input_code", [input_code(i) for s in prog.stubs for i in s.inputs])
        carray(writer, "uint32_t", "stub_outputs", firsts(length(s.outputs) for s in prog.stubs))
        carray(writer, "uint32_t", "output_component", [j - 1 for s in prog.stubs for j in s.outputs])
        carray(writer, "uint32_t", "scaleset_first", firsts(length.(prog.scalesets)))
        carray(writer, "uint32_t", "scaleset_scale", [x - 1 for set in prog.scalesets for x in set])
        carray(writer, "uint32_t", "instance", [x - 1 for (s, set) in prog.instances for x in (s, set)])
        carray(writer, "uint32_t", "output_first", firsts(length.(prog.reduction)))
        carray(writer, "uint32_t", "term_amplitude", [k - 1 for terms in prog.reduction for (k, _) in terms])
        carray(writer, "double", "term_weight", [lit(w) for terms in prog.reduction for (_, w) in terms]; per_line = 4)
        newline(writer)

        crossed = prog.crossing !== nothing
        (num, den) = prog.group_factor
        emit(writer, "static const uir_program program = {")
        emit(writer, "UIR_VERSION, $(prog.legs), $(length(prog.forms)), $(prog.planes), $(prog.amplitudes), $(length(prog.reduction)),")
        emit(writer, "rows, $(length(e.rows)), edges, $(length(e.sources)), sources, scales, factors,")
        emit(writer, "form_first, term_leg, term_coefficient, $(crossed ? "crossing_leg" : "NULL"), $(crossed ? "crossing_sign" : "NULL"),")
        emit(writer, "list_first, list_factor,")
        emit(writer, "stub_kernel, stub_inputs, input_code, stub_outputs, output_component,")
        emit(writer, "scaleset_first, scaleset_scale, instance,")
        emit(writer, "output_first, term_amplitude, term_weight, $(num)L, $(den)L")
        emit(writer, "};")
        newline(writer)

        # the kernels: each called as compile.jl's adapters call it, its inputs then its outputs in mem
        for k in kernels
            emit(writer, "int $(k.name)(double *mem, const void *states, size_t idx, const double *params);")
            copy_home(writer, k.path)
        end
        emit(writer, "int uir_kernel(uint32_t kernel, const double *in, double *out) {")
        emit(writer, "switch (kernel) {")
        for k in kernels
            emit(writer, "case $(number[k.name]): {")
            emit(writer, "double mem[$(k.inputs + k.outputs + 1)];")
            emit(writer, "for (int j = 0; j < $(k.inputs); j++) mem[j] = in[j];")
            emit(writer, "int status = $(k.name)(mem, NULL, 0, NULL);")
            emit(writer, "for (int j = 0; j < $(k.outputs); j++) out[j] = mem[$(k.inputs) + j];")
            emit(writer, "return status;")
            emit(writer, "}")
        end
        emit(writer, "default: return -1;")
        emit(writer, "}")
        emit(writer, "}")
        newline(writer)

        emit(writer, "int run(double *out, double *p, const double *momenta) {")
        emit(writer, "for (int i = 0; i < $(b.size); i++) p[i] = 0.0;")
        emit(writer, "uir_arena a = {p + $(b.forms), p + $(b.amplitudes), p + $(b.planes), 1, 0};")
        emit(writer, "if (uir_run(&program, &a, momenta, out) != 0) abort();  /* a kernel failed */")
        emit(writer, "return 0;")
        emit(writer, "}")
    finally
        close(writer)
    end
    write(joinpath(path, "runner.h"), runner_h(prog, b))
    examples === nothing || write(joinpath(path, "testing.c"), testing_c(prog, examples))
    write(joinpath(path, "Makefile"), makefile(prog, examples !== nothing; interpreter = true))
    return path
end
