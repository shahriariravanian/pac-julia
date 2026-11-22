# A symjit kernel in the Metal Shading Language, for the Metal runner (c/metal/): kernel_msl(name, app)
# writes kernel_c_mixed's kernel (kernel_c.jl: the same steps, the same `need` rule) with df64
# (c/metal/df64.h) in place of double, as `static inline int name(thread df64 *mem)`: the inputs at
# mem[0 .. states), the outputs (re, im of each output pair) at mem[states ..), as kernel_c's.
#
# MSL has no goto, so a kernel must be straight-line once its funclets are inlined: the
# Complexifier's lowered complex square root and division are funclets whose bodies are straight-line
# (selects, no branches); each call is replaced by the funclet's body (inline_funclets), and any other
# control flow, or a math-library call, is refused. Every kernel of our test data and of the mg7
# processes is of this kind (2026-11-15: 21 straight-line, 26 with one inlined complex root).
#
# The bit operations act on both halves of a df64 (df64.h): masks and signs have their bits in both,
# so the kernels' select, NaN-clearing and sign idioms keep their f64 meaning.

# the code with every funclet call replaced by the funclet's body and the funclets removed (with the
# jump over each); refuses what stays non-straight-line
function inline_funclets(name::String, code::Vector{Instruction})::Vector{Instruction}
    called = Set(i.label for i in code if i isa Call)
    labels = Dict(i.label => k for (k, i) in enumerate(code) if i isa Label)
    control(i) = i isa Union{Label,Branch,BranchIf,Call}
    bodies = Dict{String,Vector{Instruction}}()
    drop = falses(length(code))
    for l in called
        haskey(labels, l) || refuse("$name: a call of $l (math functions are not supported)")
        s = labels[l]
        r = findnext(i -> i isa Branch && i.label == ".ret", code, s + 1)
        r === nothing && refuse("$name: the funclet $l does not return")
        body = code[s+1:r-1]
        any(control, body) && refuse("$name: the funclet $l is not straight-line")
        bodies[l] = body
        drop[s:r] .= true
        # the jump over it: `goto J` just before its label, `J:` just after its return
        if s > 1 && r < length(code) && code[s-1] isa Branch && code[r+1] isa Label && code[s-1].label == code[r+1].label
            drop[s-1] = drop[r+1] = true
        end
    end
    out = Instruction[]
    for (k, i) in enumerate(code)
        drop[k] && continue
        if i isa Call
            i.num_args == 0 || refuse("$name: a call of $(i.label) with arguments")
            append!(out, bodies[i.label])
        else
            push!(out, i)
        end
    end
    any(control, out) && refuse("$name: control flow other than straight-line funclets (MSL has no goto)")
    return out
end

# the kernel `name` of `app` as an MSL function (df64.h before it)
function kernel_msl(name::String, app::SavedApplication)::String
    check_kernel(name, app)
    mir = MIR(inline_funclets(name, app.mir.code), app.mir.consts)
    (declarations, body, counts, _, _) = mixed_body(name, app; lang = :msl, mir)
    out = IOBuffer()
    println(out, "/* the symjit kernel $name in mixed precision (kernel_msl.jl): $(counts[:f32]) f32 and " *
                 "$(counts[:f64]) df64 operations, $(counts[:conversions]) conversions */")
    println(out, "static inline int $name(thread df64 *mem) {")
    foreach(d -> println(out, d), declarations)
    foreach(l -> println(out, l), body)
    println(out, "return 0;")
    println(out, "}")
    return String(take!(out))
end

kernel_msl(name::String, path::AbstractString) = kernel_msl(name, read_sjb(path; lowered = true))
