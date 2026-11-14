# compile(path, l; interpreter = true): the folder of compile.jl, but runner.c runs the program on
# the UIR interpreter (c/uirvm.c, which runner.c includes) instead of straight-line code. The
# program is data there: encode(l)'s words (the rows, SOURCE rows among them) and edge table, and the tables they index, as C
# arrays of one uir_program (c/uirvm.h); the kernels are reached through uir_stub, a switch over
# the stubs (each a kernel and its input layout, the kernels' C as compile.jl's). runner.h, testing.c and the Makefile are compile.jl's, so run() has the
# same prototype and the values must be the straight-line C's bit for bit.
#
# The CPU build is the interpreter's test; the same uirvm.c is the GPU's interpreter.

# a C array: `UIR_DATA <ty> <name>[] = {...};` (static const, or __device__ const under nvcc; an
# empty one holds a single unused 0: C has no empty arrays)
function carray(writer, ty::String, name::String, values; per_line = 8)
    if isempty(values)
        emit(writer, "UIR_DATA $ty $name[1] = {0};")
        return
    end
    emit(writer, "UIR_DATA $ty $name[] = {")
    v = collect(values)
    for k in 1:per_line:length(v)
        emit(writer, join(v[k:min(k + per_line - 1, end)], ", ") * ",")
    end
    emit(writer, "};")
end

hex32(w::UInt32) = "0x" * string(w; base = 16, pad = 8) * "u"
# the words as uirvm reads them: each as two 32-bit halves, low first
halves(words::Vector{UInt64}) = [UInt32((w >> s) & 0xffffffff) for w in words for s in (0, 32)]
cdoubles(zs::Vector{Complex{Float64}}) = [lit(f(z)) for z in zs for f in (real, imag)]
# a "first" table: the offsets of each entry's items, then their total
firsts(lengths) = cumsum([0; collect(lengths)])

# a kernel input as uirvm's code: kind, imaginary part, n (0-based component or scale-set slot)
input_code(i::KernelInput) = UInt32(Int(i.kind)) | UInt32(i.im) << 3 | UInt32(i.kind == FACTOR ? 0 : i.n - 1) << 4

const GPU_MEMORY_DEFAULT = 256 * 2^20   # device bytes per slot of the GPU pool (pool.h)

# the device bytes per point of a program's arena on the GPU (uirgpu.c: momenta, outputs, forms,
# amplitudes, planes; precision = :mixed: planes in f32)
gpu_bytes_per_point(prog::Lowered; precision::Symbol = :f64) =
    8 * sum(max(n, 1) for n in (4 * prog.legs, output_count(prog), 4 * length(prog.forms), 2 * prog.amplitudes)) +
    (precision == :mixed ? 4 : 8) * max(2 * prog.planes, 1)

# the GPU part of runner.c (under nvcc): the sizes uirgpu.c needs, then uirgpu.c (the program on a
# slot of the pool, pool.h: plane-major arena, ld = points). The pool's size (UIRGPU_SLOTS,
# UIRGPU_SLOT_BYTES) defaults to one slot per CPU of GPU_MEMORY_DEFAULT bytes; nvcc -D overrides.
function run_gpu_c(prog::Lowered)::String
    return """
    #define UIRGPU_LEGS $(prog.legs)
    #define UIRGPU_OUTPUTS $(output_count(prog))
    #define UIRGPU_FORMS $(length(prog.forms))
    #define UIRGPU_AMPLITUDES $(prog.amplitudes)
    #define UIRGPU_PLANES $(prog.planes)
    #ifndef UIRGPU_SLOT_BYTES
    #define UIRGPU_SLOT_BYTES ((size_t)$(GPU_MEMORY_DEFAULT))  /* $(gpu_bytes_per_point(prog)) bytes per point ($(gpu_bytes_per_point(prog; precision = :mixed)) mixed) */
    #endif
    #include "uirgpu.c"
    """
end

# the most components uirvm's closure and scale rows take (UIR_MAX_COMPONENTS: each count 0-6 has its
# constant-count code); a kernel-only (UNTYPED) current may have more
const VM_MAX_COMPONENTS = 6

# what uirvm cannot run in `prog`: closure, scale, copy and scatter rows beyond VM_MAX_COMPONENTS
function vm_unsupported(prog::Lowered)::Vector{String}
    out = String[]
    for (k, op) in enumerate(prog.ops)
        n = op.op == CLOSURE ? length(prog.lists[op.t]) : op.op == SCALE_ROW ? op.n :
            op.op in (COPY, SCATTER) ? edge_layout(prog, op.d).components : 0
        n > VM_MAX_COMPONENTS && push!(out, "row $k ($(op.op)): $n components, beyond the interpreter's $VM_MAX_COMPONENTS")
    end
    return out
end

# what the interpreter cannot run: a wide row (uirvm reads one-word rows only) or too many components
function check_interpretable(prog::Lowered, e::Encoded)
    length(e.rows) == length(prog.ops) + 1 ||
        error("$(prog.process) $(prog.colour): $(length(e.rows) - length(prog.ops) - 1) wide rows " *
              "(an index beyond the one-word fields: 13-bit edges, 10-bit T, 6-bit factors); the UIR interpreter " *
              "reads one-word rows only")
    bad = vm_unsupported(prog)
    isempty(bad) || error("$(prog.process) $(prog.colour): " * join(bad, "; "))
end

# the interpreter's sources beside runner.c, and runner.c's head
function compile_vm_head(writer, prog::Lowered)
    foreach(f -> copy_home(writer, f), ("c/uirvm.c", "c/uirvm.h", "c/uirgpu.c", "c/uirgpu.h", "c/pool.cu", "c/pool.h"))
    emit(writer, "/* $(prog.process) $(prog.colour) on the UIR interpreter (uirvm.c), compiled by PacRunner's compile */")
    emit(writer, "#include <stdlib.h>")
    emit(writer, "#include \"uirvm.c\"")
    emit(writer, "#ifdef __CUDACC__")
    emit(writer, "#include \"pacops.c\"  /* one translation unit for the device */")
    emit(writer, "#endif")
    emit(writer, "#include \"runner.h\"")
    newline(writer)
end

# the program's tables as C arrays: the encoded rows and edges, then the tables they index (each
# list table as its entries' firsts and its flattened entries), 0-based
function compile_vm_tables(writer, prog::Lowered, e::Encoded)
    carray(writer, "uint32_t", "rows", hex32.(halves(e.rows)); per_line = 8)
    carray(writer, "uint32_t", "edges", e.edges)
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
    carray(writer, "uint32_t", "embedding_first", firsts(length.(prog.embeddings)))
    carray(writer, "int32_t", "embedding_component", [j - 1 for list in prog.embeddings for (j, _, _) in list])
    carray(writer, "int32_t", "embedding_wave", [w - 1 for list in prog.embeddings for (_, w, _) in list])
    carray(writer, "uint32_t", "embedding_factor", [x - 1 for list in prog.embeddings for (_, _, x) in list])
    carray(writer, "uint32_t", "scaleset_first", firsts(length.(prog.scalesets)))
    carray(writer, "uint32_t", "scaleset_scale", [x - 1 for set in prog.scalesets for x in set])
    carray(writer, "uint32_t", "instance", [x - 1 for (s, set) in prog.instances for x in (s, set)])
    carray(writer, "double", "weights", [lit(w) for w in prog.weights]; per_line = 4)
    # the scales and factors as pac_real: rounded copies under PAC_MIXED (pacops.h), else the same tables
    emit(writer, "#ifdef PAC_MIXED")
    carray(writer, "pac_real", "scales_r", cdoubles(prog.scales); per_line = 4)
    carray(writer, "pac_real", "factors_r", cdoubles(prog.factors); per_line = 4)
    emit(writer, "#else")
    emit(writer, "#define scales_r scales")
    emit(writer, "#define factors_r factors")
    emit(writer, "#endif")
    newline(writer)
end

# the uir_program (uirvm.h) of the tables
function compile_vm_program(writer, prog::Lowered, e::Encoded)
    crossed = prog.crossing !== nothing
    (num, den) = prog.group_factor
    emit(writer, "UIR_DATA uir_program program = {")
    emit(writer, "UIR_VERSION, $(prog.legs), $(length(prog.forms)), $(prog.planes), $(prog.amplitudes), $(output_count(prog)),")
    emit(writer, "rows, $(length(e.rows)), edges, scales, factors,")
    emit(writer, "form_first, term_leg, term_coefficient, $(crossed ? "crossing_leg" : "NULL"), $(crossed ? "crossing_sign" : "NULL"),")
    emit(writer, "list_first, list_factor,")
    emit(writer, "embedding_first, embedding_component, embedding_wave, embedding_factor,")
    emit(writer, "scaleset_first, scaleset_scale, instance,")
    emit(writer, "weights, $(num)L, $(den)L,")
    emit(writer, "scales_r, factors_r")
    emit(writer, "};")
    newline(writer)
end

# the kernels (kernel_c.jl) and uir_stub, one case per stub: its inputs gathered into the kernel's
# mem (constant codes), the kernel, its output pairs stored (constant components)
function compile_vm_stubs(writer, prog::Lowered)
    emit_kernels(writer, used_kernels(prog); mixed = true)
    emit(writer, "PAC_FN int uir_stub(uint32_t stub, const uir_program *g, const uir_arena *a, const uint32_t *w, const uint32_t *scaleset) {")
    emit(writer, "switch (stub) {")
    for (s, stub) in enumerate(prog.stubs)
        k = stub.kernel
        emit(writer, "case $(s - 1): {  /* $(k.name) */")
        emit(writer, "double mem[$(k.inputs + k.outputs + 1)];")
        for (j, i) in enumerate(stub.inputs)
            emit(writer, "mem[$(j - 1)] = uir_input(g, a, w, $(input_code(i))u, scaleset);")
        end
        emit(writer, "int status = $(k.name)(mem);")
        emit(writer, "if (status != 0) return status;")
        for (n, c) in enumerate(stub.outputs)
            emit(writer, "uir_output(g, a, w, $(c - 1), cx(mem[$(k.inputs + 2(n - 1))], mem[$(k.inputs + 2(n - 1) + 1)]));")
        end
        emit(writer, "return 0;")
        emit(writer, "}")
    end
    emit(writer, "default: return -1;")
    emit(writer, "}")
    emit(writer, "}")
    newline(writer)
end

# run(): one point on the CPU (the arena of compile.jl's layout, ld = 1); under nvcc the GPU entries
function compile_vm_run(writer, prog::Lowered, b::ArenaIndex)
    emit(writer, "#ifndef __CUDACC__")
    emit(writer, "int run(double *out, double *p, const double *momenta) {")
    emit(writer, "for (int i = 0; i < $(b.size); i++) p[i] = 0.0;")
    emit(writer, "uir_arena a = {p + $(b.forms), p + $(b.amplitudes), p + $(b.planes), 1, 0};")
    emit(writer, "return uir_run(&program, &a, momenta, out);  /* 0, or a failing kernel's status */")
    emit(writer, "}")
    emit(writer, "#else")
    foreach(l -> emit(writer, l), split(chomp(run_gpu_c(prog)), '\n'))
    emit(writer, "#endif")
end

function compile_interpreter(path, prog::Lowered; examples = nothing)
    validate(prog)
    b = ArenaIndex(prog)
    examples === nothing || check_examples(prog, examples)
    e = encode(prog)
    check_interpretable(prog, e)
    writer = CodeWriterCpu(path)
    try
        compile_vm_head(writer, prog)
        compile_vm_tables(writer, prog, e)
        compile_vm_program(writer, prog, e)
        compile_vm_stubs(writer, prog)
        compile_vm_run(writer, prog, b)
    finally
        close(writer)
    end
    write(joinpath(path, "runner.h"), runner_h(prog, b))
    examples === nothing || write(joinpath(path, "testing.c"), testing_c(prog, examples))
    write(joinpath(path, "Makefile"), makefile(prog, examples !== nothing; interpreter = true))
    return path
end
