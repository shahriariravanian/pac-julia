using PacRunner

programs = load_recurrence_export("zgg.json")
prog = programs[1]
println(programs[1].colour)
println(programs[2].colour)
arena = LArena(prog)

momenta = [
           [500., 0., 0., 500.],
           [ 500., 0., 0., -500.],
           [142.22002372, -53.10090083, 59.0409804, -74.87149502],
           [380.350097, -268.20692462, 256.15852034, -84.34485361],
           [477.42987928,  321.30782545, -315.19950074, 159.21634863]]

helicities = evaluate(prog, arena, momenta)
println(sum(helicities))
