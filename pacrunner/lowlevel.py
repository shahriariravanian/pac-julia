"""pacrunner's low-level API: PacRunner.jl's exported functions and types, from Python, through juliacall.

    from pacrunner import lowlevel as pr
    programs = pr.load_recurrence_export("export.json")        # a Julia Vector{Lowered}
    pr.umami_libraries("export.json", "metadata.json", "out", device="cuda", precision="mixed")
    pr.compile("folder", programs[0], interpreter=True)
    pr.jl                                                      # the PacRunner module itself

Every name PacRunner.jl exports (but the deprecated v1 path: main.json, pac2json) is a function of
the same name here, documented by its Julia docstring (`help(pr.umami_libraries)` after Julia started;
`pr.doc(name)`). The wrappers only convert what Python cannot spell: the keywords Julia types as a
Symbol (device, precision, kernels) take strings, and paths take any os.PathLike. Results are
juliacall objects (Julia's own: index them, pass them back, convert with juliacall's tools).

Julia starts at the first call (pacrunner._julia: this repository's Julia project, PacRunner loaded).
"""

import os

from . import _julia

# PacRunner.jl's exports (src/PacRunner.jl), but v1's (deprecated: Program, Arena, read_main, load_artifact,
# load_programs, load_examples, lower, install, capture, alpha_s_power); test_lowlevel checks the list
EXPORTS = (
    # RustiCol's recurrence export
    "load_recurrence_export", "recurrence_references", "umami_libraries",
    # UIR
    "Lowered", "LArena", "evaluate", "validate", "sort_runs", "fuse", "reuse_planes", "optimize", "Encoded", "encode",
    "decode",
    # the C
    "compile", "ArenaIndex",
    # the kernels
    "read_sjb", "SavedApplication", "MIR", "complexify", "listing", "kernel_c",
)
V1 = ("Program", "Arena", "read_main", "load_artifact", "load_programs", "load_examples", "lower", "install",
      "capture", "alpha_s_power")

# the keywords of the exported functions that Julia types as a Symbol
SYMBOL_KEYWORDS = frozenset({"device", "precision", "kernels"})


def _arg(x):
    return os.fspath(x) if isinstance(x, os.PathLike) else x


def _wrap(name):
    def call(*args, **kwargs):
        jl = _julia.main()
        f = getattr(_julia.pacrunner(), name)
        args = [_arg(a) for a in args]
        kwargs = {k: jl.Symbol(v) if k in SYMBOL_KEYWORDS and isinstance(v, str) else _arg(v)
                  for k, v in kwargs.items()}
        return f(*args, **kwargs)

    call.__name__ = call.__qualname__ = name
    call.__doc__ = f"PacRunner.{name} (Julia; its docstring: pacrunner.lowlevel.doc({name!r}))."
    return call


for _name in EXPORTS:
    globals()[_name] = _wrap(_name)


def __getattr__(name):
    if name == "jl":
        return _julia.pacrunner()
    if name in V1:
        raise AttributeError(f"PacRunner.{name} is the deprecated v1 path (main.json, pac2json): not in pacrunner; "
                             f"pacrunner.lowlevel.jl.{name} reaches it")
    raise AttributeError(f"module 'pacrunner.lowlevel' has no attribute {name!r}")


def doc(name):
    """the Julia docstring of an exported name, as text"""
    jl = _julia.main()
    return str(jl.seval(f"string(@doc PacRunner.{name})"))


__all__ = list(EXPORTS) + ["doc", "jl", "EXPORTS", "SYMBOL_KEYWORDS"]
