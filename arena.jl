struct Arena
    scalars::Dict{String, Complex{Float64}}
    vectors::Dict{String, Vector{Complex{Float64}}}

    function Arena()
        new(Dict(), Dict())
    end
end

function set_scalar(arena::Arena, name::String, x::Complex{Float64})
    l = findfirst(isequal('['), name)
    if l != nothing
        v = get_vector(arena, name[1:l-1])
        k = parse(Int, name[l+1:end-1]) + 1
        v[k] = x
    else
        arena.scalars[name] = x
    end
end

set_float(arena::Arena, name::String, x::Float64) = set_scalar(arena, name, complex(x, 0))

function set_vector(arena::Arena, name::String, xs::Vector{Complex{Float64}})
    arena.vectors[name] = xs
end

function get_scalar(arena::Arena, name::String)::Complex{Float64}
    l = findfirst(isequal('['), name)
    if l != nothing
        v = arena.vectors[name[1:l-1]]
        k = parse(Int, name[l+1:end-1]) + 1
        return v[k]
    else
        return arena.scalars[name]
    end
end

function get_float(arena::Arena, name::String)::Float64
    if startswith(name, "re ")
        return real(get_scalar(arena, name[4:end]))
    elseif startswith(name, "im ")
        return imag(get_scalar(arena, name[4:end]))
    else
        return real(arena.scalars[name])
    end
end

function get_vector(arena::Arena, name::String)::Vector{Float64}
    return arena.vectors[name]
end
