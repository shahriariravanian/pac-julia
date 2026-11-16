"""pacrunner: pac-julia (PacRunner.jl) from Python.

pyAmpliCol's leading-colour recurrence programs, compiled by PacRunner into UMAMI libraries (CPU, CUDA)
for MadGraph7 and MadSpace. Two layers:

  - the high-level API: PacLibrary and MadLibrary (umami_libraries' folders) and their Runtime, the one
    callable for any UMAMI matrix element and PacArtifact (a pyAmpliCol artifact; MadArtifact, MadOutput: to come);
  - the low-level API, pacrunner.lowlevel: PacRunner.jl's exported functions through juliacall.

Setup: Julia installed (`julia` on the PATH, or PACRUNNER_JULIA), this repository cloned and installed
with `python -m pip install -e .`, a MadGraph7 checkout with MadSpace for running libraries:

    import pacrunner
    pacrunner.configure(madgraph7="~/MadGraph7")      # or PACRUNNER_MADGRAPH7
    lib = pacrunner.PacLibrary("OUTPUT/lib/pacrunner/A1_pp_jjj_p0")
    out = lib.runtime(channel=0)(momenta, flavours, random_helicity=r)
    out.matrix_element, out.color_index, out.helicity_index
"""

from ._env import CONFIG, Config, PacRunnerError, configure, repository

__version__ = "0.1.0"

# imported on first use: `python -m pacrunner._madgraph` runs in MadGraph7's python, which may have
# no numpy; the low-level layer starts nothing until one of its functions is called
_LAZY = {"PacArtifact": "._artifacts", "PacLibrary": "._libraries", "MadLibrary": "._libraries", "load_libraries": "._libraries",
         "Runtime": "._runtime", "RuntimeResult": "._runtime", "lowlevel": None}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name] or f".{name}", __name__)
        value = module if _LAZY[name] is None else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pacrunner' has no attribute {name!r}")


__all__ = ["CONFIG", "Config", "PacRunnerError", "configure", "repository", "lowlevel", "PacArtifact", "PacLibrary", "MadLibrary",
           "Runtime", "RuntimeResult", "load_libraries", "__version__"]
