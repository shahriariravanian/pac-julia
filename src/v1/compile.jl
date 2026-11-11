# compile(path, p::Program): `p` lowered (lower.jl) and, with `optimize` (the default), sorted,
# fused and laid out by liveness (passes.jl), then compiled as a Lowered (../compile.jl).
function compile(path, prog::Program; examples = nothing, optimize::Bool = true, interpreter::Bool = false)
    l = lower(prog)
    return compile(path, optimize ? PacRunner.optimize(l) : l; examples, interpreter)
end
