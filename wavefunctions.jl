# The external wavefunctions of the source currents, transcribed from RustiCol's
# rusticol-core/src/wavefunctions.rs (f64; the same operations in the same order, so the values are
# bit for bit RustiCol's). The dispatch and the factor follow pacuda's runner (`source_values` and
# `run_sources` in runner/recurrence_runner.cu).
#
# Each wavefunction has a method on plain values, (k, h, ...) with k::Momentum = (E, px, py, pz),
# returning a tuple of its kind (Scalar, Weyl, Dirac, Vec4; types.jl), and one on (source, arena).
# The arena must provide
#   get_vector(arena, "q<i>") -> (E, px, py, pz)   a momentum form
#   get_scale(arena, "s<i>")  -> Complex{Float64}  a scale (the mass is its real part)
#   get_factor(arena, "x<i>") -> Complex{Float64}  a factor

include(joinpath(@__DIR__, "types.jl"))

struct Source
    wavefunction::String
    family::String
    orientation::String
    leg::Union{Int,Nothing}  # main.json's leg (0-based), or null
    momentum::String
    helicity::Helicity
    chirality::Chirality
    mass::String
    factor::String
end

source_momentum(source, arena) = get_vector(arena, source.momentum)
source_mass(source, arena)::Float64 = real(get_scale(arena, source.mass))

# ---- scalar --------------------------------------------------------------------------------------

scalar()::Scalar = (complex(1.0, 0.0),)

# ---- Weyl spinors (ext_quark_weyl_array, ext_antiquark_weyl_array) -------------------------------

function weyl_particle(k::Momentum, h::Helicity, ch::Chirality)::Weyl
    (E, px, py, pz) = k

    if E > 0
        s = px == 0 && py == 0 && pz < 0 ? 0.0 : √max(E + pz, 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * E), 0.0) : complex(h * px / s, -py / s)

        if (h, ch) == (+1, +1)
            return (a, b)
        elseif (h, ch) == (-1, -1)
            return (b, a)
        else
            return (ZERO, ZERO)
        end
    else
        s = px == 0 && py == 0 && pz > 0 ? 0.0 : -√max(-(E + pz), 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * abs(E)), 0.0) : complex(h * px / s, py / s)

        if (h, ch) == (-1, +1)
            return (a, b)
        elseif (h, ch) == (+1, -1)
            return (b, a)
        else
            return (ZERO, ZERO)
        end
    end
end

function weyl_antiparticle(k::Momentum, h::Helicity, ch::Chirality)::Weyl
    (E, px, py, pz) = k

    if E > 0
        s = px == 0 && py == 0 && pz < 0 ? 0.0 : -√max(E + pz, 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * E), 0.0) : complex(-h * px / s, py / s)

        if (h, ch) == (+1, +1)
            return (b, a)
        elseif (h, ch) == (-1, -1)
            return (a, b)
        else
            return (ZERO, ZERO)
        end
    else
        s = px == 0 && py == 0 && pz > 0 ? 0.0 : √max(-(E + pz), 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * abs(E)), 0.0) : complex(-h * px / s, -py / s)

        if (h, ch) == (-1, +1)
            return (b, a)
        elseif (h, ch) == (+1, -1)
            return (a, b)
        else
            return (ZERO, ZERO)
        end
    end
end

weyl_particle(source, arena) = weyl_particle(source_momentum(source, arena), source.helicity, source.chirality)
weyl_antiparticle(source, arena) = weyl_antiparticle(source_momentum(source, arena), source.helicity, source.chirality)

# ---- massless Dirac spinors (ext_quark_dirac, ext_antiquark_dirac) -------------------------------

function quark_dirac(k::Momentum, h::Helicity)::Dirac
    (E, px, py, pz) = k

    if E > 0
        s = px == 0 && py == 0 && pz < 0 ? 0.0 : √max(E + pz, 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * E), 0.0) : complex(h * px / s, -py / s)

        if h == +1
            return (a, b, ZERO, ZERO)
        else
            return (ZERO, ZERO, b, a)
        end
    else
        s = px == 0 && py == 0 && pz > 0 ? 0.0 : -√max(-(E + pz), 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * abs(E)), 0.0) : complex(h * px / s, py / s)

        if h == -1
            return (a, b, ZERO, ZERO)
        else
            return (ZERO, ZERO, b, a)
        end
    end
end

function antiquark_dirac(k::Momentum, h::Helicity)::Dirac
    (E, px, py, pz) = k

    if E > 0
        s = px == 0 && py == 0 && pz < 0 ? 0.0 : -√max(E + pz, 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * E), 0.0) : complex(-h * px / s, py / s)

        if h == -1
            return (ZERO, ZERO, a, b)
        else
            return (b, a, ZERO, ZERO)
        end
    else
        s = px == 0 && py == 0 && pz > 0 ? 0.0 : √max(-(E + pz), 0.0)
        a = complex(s, 0.0)
        b = s == 0 ? complex(-h * √(2 * abs(E)), 0.0) : complex(-h * px / s, -py / s)

        if h == +1
            return (ZERO, ZERO, a, b)
        else
            return (b, a, ZERO, ZERO)
        end
    end
end

quark_dirac(source, arena) = quark_dirac(source_momentum(source, arena), source.helicity)
antiquark_dirac(source, arena) = antiquark_dirac(source_momentum(source, arena), source.helicity)

# ---- Dirac spinors (ext_quark_dirac_massive, ext_antiquark_dirac_massive) ------------------------
# |mass| < 1e-8 falls back to the massless spinors; the runner calls these for every Dirac source.

function quark_dirac_massive(k::Momentum, h::Helicity, mass::Float64)::Dirac
    if abs(mass) < 1.0e-8
        return quark_dirac(k, h)
    end

    (E, px, py, pz) = k
    nsf = E > 0 ? 1 : -1
    nh = nsf * h

    pp = √(px * px + py * py + pz * pz)
    omega1 = √(abs(E) + pp)
    omega2 = mass / omega1

    sf1 = (1 + nsf + (1 - nsf) * nh) * 0.5
    sf2 = (1 + nsf - (1 - nsf) * nh) * 0.5

    (signed_px, signed_py, signed_pz) = E > 0 ? (px, py, pz) : (-px, -py, -pz)
    pp3 = max(pp + signed_pz, 0.0)
    chi1 = pp == 0 ? complex(1.0, 0.0) : complex(√(pp3 * 0.5 / pp), 0.0)
    chi2 = pp3 == 0 || pp == 0 ? complex(-Float64(nh), 0.0) :
           complex(nh * signed_px, -signed_py) / √(2.0 * pp * pp3)

    if nh > 0
        return (chi1 * (sf2 * omega1), chi2 * (sf2 * omega1), chi1 * (sf1 * omega2), chi2 * (sf1 * omega2))
    else
        return (chi2 * (sf2 * omega2), chi1 * (sf2 * omega2), chi2 * (sf1 * omega1), chi1 * (sf1 * omega1))
    end
end

function antiquark_dirac_massive(k::Momentum, h::Helicity, mass::Float64)::Dirac
    if abs(mass) < 1.0e-8
        return antiquark_dirac(k, h)
    end

    (E, px, py, pz) = k
    nsf = E > 0 ? -1 : 1
    nh = nsf * h

    pp = √(px * px + py * py + pz * pz)
    omega1 = √(abs(E) + pp)
    omega2 = mass / omega1

    sf1 = (1 + nsf + (1 - nsf) * nh) * 0.5
    sf2 = (1 + nsf - (1 - nsf) * nh) * 0.5

    (signed_px, signed_py, signed_pz) = E > 0 ? (px, py, pz) : (-px, -py, -pz)
    pp3 = max(pp + signed_pz, 0.0)
    chi1 = pp == 0 ? complex(1.0, 0.0) : complex(√(pp3 * 0.5 / pp), 0.0)
    chi2 = pp3 == 0 || pp == 0 ? complex(-Float64(nh), 0.0) :
           complex(nh * signed_px, signed_py) / √(2.0 * pp * pp3)

    if nh > 0
        return (chi1 * (sf1 * omega2), chi2 * (sf1 * omega2), chi1 * (sf2 * omega1), chi2 * (sf2 * omega1))
    else
        return (chi2 * (sf1 * omega1), chi1 * (sf1 * omega1), chi2 * (sf2 * omega2), chi1 * (sf2 * omega2))
    end
end

quark_dirac_massive(source, arena) =
    quark_dirac_massive(source_momentum(source, arena), source.helicity, source_mass(source, arena))
antiquark_dirac_massive(source, arena) =
    antiquark_dirac_massive(source_momentum(source, arena), source.helicity, source_mass(source, arena))

# ---- massless vectors: gluon, photon (ext_gluon) --------------------------------------------------

function gluon(k::Momentum, h::Helicity)::Vec4
    (E, px, py, pz) = k
    sqh = √0.5

    if E > 0
        hel = Float64(h)
        pp = E
        pt = √(px * px + py * py)
        wf3 = complex(hel * pt / pp * sqh, 0.0)

        if pt != 0
            pzpt = pz / (pp * pt) * sqh * hel
            (wf1, wf2) = (complex(-px * pzpt, -py / pt * sqh), complex(-py * pzpt, px / pt * sqh))
        else
            (wf1, wf2) = (complex(-hel * sqh, 0.0), complex(0.0, copysign(sqh, pz)))
        end
    else
        hel = Float64(-h)
        pp = -E
        pt = √(px * px + py * py)
        wf3 = complex(hel * pt / pp * sqh, 0.0)

        if pt != 0
            pzpt = -pz / (pp * pt) * sqh * hel
            (wf1, wf2) = (complex(px * pzpt, py / pt * sqh), complex(py * pzpt, -px / pt * sqh))
        else
            (wf1, wf2) = (complex(-hel * sqh, 0.0), complex(0.0, -copysign(sqh, pz)))
        end
    end

    return (complex(0.0, 0.0), wf1, wf2, wf3)
end

gluon(source, arena) = gluon(source_momentum(source, arena), source.helicity)

# ---- massive vectors: Z, W (ext_massive_vector) ---------------------------------------------------

function massive_vector(k::Momentum, helicity::Helicity, mass::Float64)::Vec4
    (E, px, py, pz) = k

    if E < 0
        return massive_vector((-E, -px, -py, -pz), -helicity, mass)
    end

    sqh = √0.5
    hel = Float64(helicity)
    nsvahl = Float64(abs(helicity))
    pt2 = px * px + py * py
    pp = min(E, √(pt2 + pz * pz))
    pt = min(pp, √pt2)
    hel0 = 1.0 - abs(hel)

    if pp == 0
        return (complex(0.0, 0.0), complex(-hel * sqh, 0.0), complex(0.0, nsvahl * sqh), complex(hel0, 0.0))
    end

    emp = E / (mass * pp)
    wf0 = complex(hel0 * pp / mass, 0.0)
    wf3 = complex(hel0 * pz * emp + hel * pt / pp * sqh, 0.0)

    if pt != 0
        pzpt = pz / (pp * pt) * sqh * hel
        (wf1, wf2) = (
            complex(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh),
            complex(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh),
        )
    else
        (wf1, wf2) = (complex(-hel * sqh, 0.0), complex(0.0, nsvahl * copysign(sqh, pz)))
    end

    return (wf0, wf1, wf2, wf3)
end

massive_vector(source, arena) =
    massive_vector(source_momentum(source, arena), source.helicity, source_mass(source, arena))

# ---- a source current ------------------------------------------------------------------------------

# The wavefunction of a source as the runner picks it (`source_values`): by family and orientation;
# a Dirac source always through the massive spinors, a vector by whether its mass is exactly 0.
function wavefunction(family::String, orientation::String, k::Momentum, h::Helicity, ch::Chirality, mass::Float64)::CurrentValue
    anti = orientation == "antiparticle"
    if family == "scalar"
        return scalar()
    elseif family == "weyl"
        return anti ? weyl_antiparticle(k, h, ch) : weyl_particle(k, h, ch)
    elseif family == "dirac"
        return anti ? antiquark_dirac_massive(k, h, mass) : quark_dirac_massive(k, h, mass)
    elseif family == "vector"
        return mass == 0 ? gluon(k, h) : massive_vector(k, h, mass)
    else
        error("unknown source family $family")
    end
end

# The value of a source current: factor × its wavefunction (`currents[i].source`); 1, 2 or 4
# components. The complex product is the runner's: (fr·re − fi·im, fr·im + fi·re).
function apply_source(s::Source, arena)::CurrentValue
    w = wavefunction(s.family, s.orientation, source_momentum(s, arena), s.helicity, s.chirality, source_mass(s, arena))
    factor = get_factor(arena, s.factor)
    return map(c -> factor * c, w)
end
