# compile_metal(path, l): one program for Apple GPUs (c/metal/): the UIR interpreter in the Metal
# Shading Language, mixed f32/df64 (c/metal/PLAN.md). The folder holds
#
#   runner.metal     the program's whole MSL source: c/metal/df64.h, pacops.metal, uirvm.metal (their
#                    text: runtime compilation resolves no #include), then the kernels (kernel_msl.jl)
#                    and uir_stub (a switch over the stubs, as compile_vm.jl's)
#   program.bin      the program as data: a uir_header (uirvm.metal) of counts and byte offsets, then
#                    encode(l)'s rows and edges and the tables they index (df64 values as (hi, lo)
#                    float pairs, rscales/rfactors f32)
#   runner_metal.h   the program's sizes (RUNNER_*) and the entries of c/metal/uirmetal.h
#   runner_metal.c   runner.metal and program.bin embedded as C arrays, for uirmetal.m
#   uirmetal.h, uirmetal.m (the host: compile, launch), metalpool.h, metalpool.m (the process-wide slot
#                    pool), Makefile (macOS: cc -fobjc-arc, the Metal and Foundation frameworks)
#
# runner.metal is also C++: with c/metal/msl_shim.h before it, it compiles on any CPU, which is how
# test/test_metal_shim.jl checks it on Linux.

const METAL_DIR = joinpath(dirname(@__DIR__), "c", "metal")
const METAL_SOURCES = ("df64.h", "pacops.metal", "uirvm.metal")
const METAL_HOST = ("uirmetal.h", "uirmetal.m", "metalpool.h", "metalpool.m")

# a double as df64: (hi, lo), hi = the nearest f32, lo = the rest rounded (0 when hi is not finite)
function df64_pair(x::Float64)::Tuple{Float32,Float32}
    hi = Float32(x)
    return (hi, isfinite(hi) ? Float32(x - Float64(hi)) : 0.0f0)
end
df64_pairs(xs) = Float32[v for x in xs for v in df64_pair(Float64(x))]
cdf64_pairs(zs) = Float32[v for z in zs for part in (real(z), imag(z)) for v in df64_pair(part)]

# the header's fields, in uirvm.metal's (and uirmetal.h's) order
const METAL_TABLES = (:rows, :edges, :scales, :factors, :rscales, :rfactors, :form_first, :term_leg, :term_coefficient,
                      :crossing_leg, :crossing_sign, :list_first, :list_factor, :embedding_first, :embedding_component,
                      :embedding_wave, :embedding_factor, :scaleset_first, :scaleset_scale, :instance, :weights)
const METAL_HEADER_BYTES = 128   # 10 counts + 21 offsets, 4 bytes each (124), padded to 8

# the tables of `prog` (0-based indices), in METAL_TABLES order
function metal_tables(prog::Lowered, e::Encoded)
    crossed = prog.crossing !== nothing
    return (
        rows = halves(e.rows),
        edges = e.edges,
        scales = cdf64_pairs(prog.scales),
        factors = cdf64_pairs(prog.factors),
        rscales = Float32[Float32(part) for z in prog.scales for part in (real(z), imag(z))],
        rfactors = Float32[Float32(part) for z in prog.factors for part in (real(z), imag(z))],
        form_first = UInt32.(firsts(length(f.terms) for f in prog.forms)),
        term_leg = UInt32[leg - 1 for f in prog.forms for (leg, _) in f.terms],
        term_coefficient = df64_pairs(c for f in prog.forms for (_, c) in f.terms),
        crossing_leg = crossed ? UInt32[leg - 1 for (leg, _) in prog.crossing] : UInt32[],
        crossing_sign = crossed ? df64_pairs(s for (_, s) in prog.crossing) : Float32[],
        list_first = UInt32.(firsts(length.(prog.lists))),
        list_factor = UInt32[x - 1 for list in prog.lists for x in list],
        embedding_first = UInt32.(firsts(length.(prog.embeddings))),
        embedding_component = Int32[j - 1 for list in prog.embeddings for (j, _, _) in list],
        embedding_wave = Int32[w - 1 for list in prog.embeddings for (_, w, _) in list],
        embedding_factor = UInt32[x - 1 for list in prog.embeddings for (_, _, x) in list],
        scaleset_first = UInt32.(firsts(length.(prog.scalesets))),
        scaleset_scale = UInt32[x - 1 for set in prog.scalesets for x in set],
        instance = UInt32[x - 1 for (s, set) in prog.instances for x in (s, set)],
        weights = df64_pairs(prog.weights),
    )
end

# program.bin: the header, then each table at a multiple of 8 bytes
function metal_program(prog::Lowered)::Vector{UInt8}
    e = encode(prog)
    check_interpretable(prog, e)
    tables = metal_tables(prog, e)
    (num, den) = prog.group_factor
    typemin(Int32) <= num <= typemax(Int32) && typemin(Int32) <= den <= typemax(Int32) ||
        error("$(prog.process) $(prog.colour): group factor $num/$den beyond 32 bits")
    io = IOBuffer()
    write(io, zeros(UInt8, METAL_HEADER_BYTES))
    offsets = UInt32[]
    for name in METAL_TABLES
        while position(io) % 8 != 0
            write(io, UInt8(0))
        end
        push!(offsets, position(io))
        foreach(x -> write(io, htol(x)), tables[name])
    end
    blob = take!(io)
    header = IOBuffer()
    counts = UInt32[(0 << 16) | 8, prog.legs, length(prog.forms), prog.planes, prog.amplitudes, output_count(prog),
                    length(e.rows), prog.crossing !== nothing ? 1 : 0]
    foreach(x -> write(header, htol(x)), counts)
    write(header, htol(Int32(num)), htol(Int32(den)))
    foreach(x -> write(header, htol(x)), offsets)
    h = take!(header)
    blob[1:length(h)] = h
    return blob
end

# uir_stub: one case per stub, its inputs gathered into the kernel's mem (df64), the kernel, its
# output pairs stored (as compile_vm_stubs)
function metal_stubs(prog::Lowered)::String
    out = IOBuffer()
    println(out, "static int uir_stub(uint stub, thread const uir_program &g, thread const uir_arena &a, device const uint *w,")
    println(out, "                    device const uint *scaleset) {")
    println(out, "switch (stub) {")
    for (s, stub) in enumerate(prog.stubs)
        k = stub.kernel
        println(out, "case $(s - 1): {  /* $(k.name) */")
        println(out, "df64 mem[$(k.inputs + k.outputs + 1)];")
        for (j, i) in enumerate(stub.inputs)
            println(out, "mem[$(j - 1)] = uir_input(g, a, w, $(input_code(i))u, scaleset);")
        end
        println(out, "int status = $(k.name)(mem);")
        println(out, "if (status != 0) return status;")
        for (n, c) in enumerate(stub.outputs)
            println(out, "uir_output(g, a, w, $(c - 1)u, cdd(mem[$(k.inputs + 2(n - 1))], mem[$(k.inputs + 2(n - 1) + 1)]));")
        end
        println(out, "return 0;")
        println(out, "}")
    end
    println(out, "default: return -1;")
    println(out, "}")
    println(out, "}")
    return String(take!(out))
end

# runner.metal: the whole source of the program
function metal_source(prog::Lowered)::String
    out = IOBuffer()
    println(out, "// $(prog.process) $(prog.colour): the program's Metal source, by PacRunner's compile_metal")
    println(out, "#ifndef MSL_SHIM_H")
    println(out, "#include <metal_stdlib>")
    println(out, "using namespace metal;")
    # no fma contraction (MTLMathModeSafe alone still contracts a * b + c): the C sources and the
    # shim are compiled with -ffp-contract=off, and the GPU gives their bits only with this
    println(out, "#pragma METAL fp contract(off)")
    println(out, "#endif")
    for f in METAL_SOURCES
        println(out, "\n// ---- c/metal/$f ----")
        print(out, read(joinpath(METAL_DIR, f), String))
    end
    println(out, "\n// ---- the program's kernels (kernel_msl.jl) and stubs ----")
    for ker in used_kernels(prog)
        isfile(ker.source) || error("kernel $(ker.name): no .plane.symjit at $(ker.source)")
        println(out, kernel_msl(ker.name, read_sjb(ker.source; lowered = true)))
    end
    print(out, metal_stubs(prog))
    return String(take!(out))
end

# bytes as a C array body, 16 a line
function c_bytes(io::IO, bytes::AbstractVector{UInt8})
    for (k, b) in enumerate(bytes)
        print(io, "0x", string(b; base = 16, pad = 2), k == length(bytes) ? "" : ",", k % 16 == 0 ? "\n" : "")
    end
    isempty(bytes) && print(io, "0")
    println(io)
end

# runner_metal.c: the source and the program for uirmetal.m
function metal_runner_c(prog::Lowered, source::String, program::Vector{UInt8})::String
    out = IOBuffer()
    println(out, "/* $(prog.process) $(prog.colour): its Metal source and program, by PacRunner's compile_metal */")
    println(out, "#include <stddef.h>")
    println(out, "#include \"runner_metal.h\"")
    println(out, "const char uir_metal_source[] = {")
    c_bytes(out, Vector{UInt8}(codeunits(source)))
    println(out, ", 0};")
    println(out, "const size_t uir_metal_source_size = $(ncodeunits(source));")
    println(out, "const unsigned char uir_metal_program[] __attribute__((aligned(16))) = {")
    c_bytes(out, program)
    println(out, "};")
    println(out, "const size_t uir_metal_program_size = $(length(program));")
    return String(take!(out))
end

function metal_runner_h(prog::Lowered)::String
    return """
    /* $(prog.process) $(prog.colour): the program for Metal, by PacRunner's compile_metal
     * (runner_metal.c: its source and program; uirmetal.h: the entries). */
    #ifndef RUNNER_METAL_H
    #define RUNNER_METAL_H

    #define RUNNER_PROCESS $(cstring(prog.process))
    #define RUNNER_COLOUR $(cstring(prog.colour))
    #define RUNNER_LEGS $(prog.legs)
    #define RUNNER_OUTPUTS $(output_count(prog))
    #define RUNNER_FORMS $(length(prog.forms))
    #define RUNNER_AMPLITUDES $(prog.amplitudes)
    #define RUNNER_PLANES $(prog.planes)

    #include <stddef.h>
    extern const char uir_metal_source[];
    extern const size_t uir_metal_source_size;
    extern const unsigned char uir_metal_program[];
    extern const size_t uir_metal_program_size;

    #include "uirmetal.h"

    #endif
    """
end

function metal_makefile(prog::Lowered)::String
    return """
    # $(prog.process) $(prog.colour) for Metal, compiled by PacRunner's compile_metal (macOS).
    #   make         librunner_metal.dylib: run_metal, pac_metal_run, pac_metal_values (uirmetal.h)
    #   make clean
    CC ?= cc
    CFLAGS ?= -O2
    PACFLAGS = -ffp-contract=off -fPIC
    FRAMEWORKS = -framework Metal -framework Foundation
    all: librunner_metal.dylib

    librunner_metal.dylib: uirmetal.m uirmetal.h metalpool.m metalpool.h runner_metal.c runner_metal.h
    \t\$(CC) \$(CFLAGS) \$(PACFLAGS) -fobjc-arc -dynamiclib -o \$@ uirmetal.m metalpool.m runner_metal.c \$(FRAMEWORKS)

    clean:
    \trm -f librunner_metal.dylib

    .PHONY: all clean
    """
end

# the folder of `prog` for Metal; returns path
function compile_metal(path, prog::Lowered)
    validate(prog)
    mkpath(path)
    source = metal_source(prog)
    program = metal_program(prog)
    write(joinpath(path, "runner.metal"), source)
    write(joinpath(path, "program.bin"), program)
    write(joinpath(path, "runner_metal.h"), metal_runner_h(prog))
    write(joinpath(path, "runner_metal.c"), metal_runner_c(prog, source, program))
    foreach(f -> cp(joinpath(METAL_DIR, f), joinpath(path, f); force = true), METAL_HOST)
    write(joinpath(path, "Makefile"), metal_makefile(prog))
    return path
end
