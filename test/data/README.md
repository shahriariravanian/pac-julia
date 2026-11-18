# Test cases

pac2json folders (`main.json`, `examples.json`, `kernels/*.plane.symjit`), without the kernels'
shared libraries: the Julia runner needs them, and `test_compile.jl` and `bench_compile.jl` build them
with `sjb2obj` in a temporary copy (the compiled C needs none: its kernels are C, `kernel_c`). `examples.json` holds RustiCol's outputs (8 points per program).

| Case | Process | Programs | Instructions | Kernels | What it covers |
| --- | --- | --- | --- | --- | --- |
| `uux_zgg` | u u~ > Z g g | 1 | 184 | 1 | the running test |
| `uux_zgggg` | u u~ > Z g g g g | 1 | 1,492 | 1 | a larger program (benchmark) |
| `ppjjj_crossed` | pp > jjj, 6 processes | 21 | 1,062 | 6 | crossing (12 programs), 6 kernels, every vertex and propagator type, `written` currents |

## How `ppjjj_crossed` was made

pac2json records only the processes of an artifact's provider header, the representatives of the
process groups, and those never have a crossing. A process-group member whose legs are permuted
does: RustiCol loads it as its representative's program with an input crossing map.

1. The LC artifact of the mg7 output `ppjjj` (`SubProcesses/A1_pp_jjj/lc`) was copied, and three
   members were added to its `API/umami/provider_p0.h`, as processes 11–13 with their own colour
   ids (from RustiCol's `color_components()` for the member):
   - `mp0_mp0_to_mp0_mp0_mp0_31`, g d~ > d d~ d~, member of `_7`, legs `[0, 1, 3, 4, 2]`;
   - `mp0_mp0_to_mp0_mp0_mp0_13`, g u > d u d~, member of `_8`, legs `[0, 1, 3, 2, 4]`;
   - `mp0_mp0_to_mp0_mp0_mp0_37`, g u~ > d d~ u~, member of `_8`, legs `[0, 1, 4, 3, 2]`.
2. `pac2json LC -o OUT --sjb2obj sjb2obj` recorded 49 programs (12 crossed) and read them back.
3. The programs of `_1` (g g > g g g), `_7`, `_8` and the three members were kept (21 programs),
   `configs.programs` and `configs.artifact` set accordingly, and the JSON written one program per
   line. The members have no `expression` (the artifact does not list them).

All crossing signs here are +1 (permutations); a sign of −1 does not occur in these artifacts.

The `configs.conventions` and `configs.instructions` legends that pac2json writes are removed from
the three `main.json` files; their text is in `CONVENTIONS.md` at the root.

## `ssa/`

symjit-utils' `sjb2ssa` (0.3.0) of the kernels above (`sjb2ssa -q KERNEL.plane.symjit -o ssa/NAME.ssa`):
their lowered MIR in SSA form, for `read_sjb`'s `.ssa` support (the "sjb reader" tests).

## `cpp/`

Three leaves of pyAmpliCol's cpp backend (planc): `pyamplicol generate "g g > g g g" OUT --color-accuracy lc
--execution-mode compiled --backend cpp` with pyAmpliCol 1.0.2 (b3132bc), from
`OUT/processes/ggggg/helicity-sum/compiled/repackaged/`: a stage-1 leaf (propagators: reciprocals, √½),
a stage-3 leaf (tensor currents) and an amplitude leaf. The "direct cpp" tests read them
(`read_direct_cpp`) and evaluate them against the same C++ built with `c++ -O2 -ffp-contract=off`
(pyAmpliCol itself builds with `-O3 -ffast-math`, which reassociates: the last bits differ).

## `planc_gggg/`

g g > g g, LC, pyAmpliCol 1.0.2: `processes/gggg/execution.json` and the C++ of its helicity-sum leaves
from `pyamplicol generate "g g > g g" OUT --color-accuracy lc --execution-mode compiled --backend cpp`
(the leaves `read_compiled` reads, at their paths; the JSON written compact), and
`export_recurrence.json`, the `pyamplicol export-recurrence` of the same process generated in recurrence
mode. The "compiled plan" tests (`compiled_plan`, test/compare_export.jl).

## `planc_uuddg/`

u u~ > d d~ g, LC, pyAmpliCol 1.0.2, as `planc_gggg/` (compiled with `--backend cpp`, its
`processes/uuddg/execution.json` reduced to `helicity_sum_execution`; `model/parameters.json`;
`export_recurrence.json` from recurrence mode), and `model.pyamplicol-model`: the built-in SM's prepared
kernels (`assets/prepared_models/built-in-sm-jit-o2-x86_64.pyamplicol-model`), its 49
`kernels/<id>/application-1.plane.symjit` and the manifest reduced to the kernels' contracts (id, kind,
input contracts, output arity and layout), written as a stored zip. The "model kernels" and
"compiled plan, quarks" tests.
