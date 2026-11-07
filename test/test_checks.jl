# The load-time checks of check.jl: every program of a pac2json folder loads (the kinds of its
# currents are counted), and programs made invalid in one place each are rejected with the
# expected message.
# usage: julia test_checks.jl PAC2JSON_DIR
include(joinpath(@__DIR__, "..", "pacrun.jl"))

# the first index of `xs` where `f` holds
first_where(f, xs) = findfirst(f, xs)

function main(dir)
    path = joinpath(dir, "main.json")
    text = read(path, String)
    main = JSON.parse(text)
    kernels = load_kernels(main, path)

    # every program loads; the kinds of the currents
    counts = Dict{CurrentKind,Int}()
    for prog in main.programs
        for c in load_program(prog, kernels).currents
            counts[c.kind] = get(counts, c.kind, 0) + 1
        end
    end
    println("$(length(main.programs)) programs load; current kinds: ",
            join(["$(kind_name(k)) $(counts[k])" for k in sort(collect(keys(counts)))], ", "))

    # (what, change of a program, expected text in the message); a change returns false when the
    # program has nothing to change (the case then takes the next program)
    name_index(prog, name) = first_where(c -> c.name == name, prog.currents)
    cases = [
        ("a vertex on the wrong kinds", prog -> begin
                k = first_where(o -> o.op == "vertex" && o.type == "weyl_vector_a", prog.body)
                k === nothing && return false
                prog.body[k]["type"] = "three_vector"
            end, "is a Weyl spinor"),
        ("an unknown vertex type", prog -> begin
                k = first_where(o -> o.op == "vertex", prog.body)
                prog.body[k]["type"] = "four_vector"
            end, "unknown vertex type four_vector"),
        ("a propagator on the wrong kind", prog -> begin
                k = first_where(o -> o.op == "propagator" && o.type == "vector", prog.body)
                k === nothing && return false
                prog.body[k]["type"] = "weyl_a"
            end, "not a Weyl spinor"),
        ("a source read before it has a value", prog -> begin
                c = prog.currents[first_where(c -> c.initial == "source", prog.currents)]
                c["initial"] = "written"
                delete!(c, "source")
            end, "is read before it has a value"),
        ("a first write with \"+=\"", prog -> begin
                k = first_where(o -> o.op == "vertex" && o.mode == "=", prog.body)
                prog.body[k]["mode"] = "+="
                prog.currents[name_index(prog, prog.body[k].out)]["initial"] = "written"
            end, "is read before it has a value"),
        ("a kernel input beyond its operand", prog -> begin
                isempty(prog.bindings) && return false
                b = prog.bindings[1]
                k = first_where(x -> startswith(x, "re in[1]"), b.inputs)
                k === nothing && return false
                b.inputs[k] = "re in[1][3]"
            end, "the kernel reads component 4"),
        ("a factor out of range", prog -> begin
                k = first_where(o -> o.op == "vertex", prog.body)
                prog.body[k]["factor"] = "x999"
            end, "factor 1000 of"),
        ("a current with the wrong number of components", prog -> begin
                k = first_where(c -> c.initial == "source" && c.components == 2, prog.currents)
                k === nothing && return false
                prog.currents[k]["components"] = 4
            end, "not those of a Weyl spinor"),
    ]
    failed = 0
    for (what, change!, expected) in cases
        k = first_where(1:length(main.programs)) do k
            change!(JSON.parse(text).programs[k]) !== false
        end
        k === nothing && (println("  skipped:  ", rpad(what, 48), "no program to change"); continue)
        prog = JSON.parse(text).programs[k]
        change!(prog)
        result = try
            load_program(prog, kernels)
            "accepted"
        catch e
            e isa ProgramError ? sprint(showerror, e) : "error $(typeof(e)): $(sprint(showerror, e))"
        end
        ok = occursin(expected, result)
        failed += !ok
        println(ok ? "  rejected: " : "  WRONG:    ", rpad(what, 48), "program $k: ", result)
    end
    println(failed == 0 ? "all $(length(cases)) invalid programs rejected" : "$failed of $(length(cases)) cases wrong")
    return failed == 0 ? 0 : 1
end

length(ARGS) == 1 || (println("usage: julia test_checks.jl PAC2JSON_DIR"); exit(2))
exit(main(ARGS[1]))
