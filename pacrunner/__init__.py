"""pacrunner: pac-julia (PacRunner.jl) from Python.

pyAmpliCol's leading-colour recurrence programs, compiled by PacRunner into UMAMI libraries (CPU, CUDA)
for MadGraph7 and MadSpace. Two layers:

  - the high-level API (PacArtifact, PacLibrary, MadArtifact, MadLibrary, MadOutput, Runtime): from an
    artifact to matrix elements and to a whole MadGraph7 run (to come: plans in the repository);
  - the low-level API, pacrunner.lowlevel: PacRunner.jl's exported functions through juliacall.

Setup: Julia installed (`julia` on the PATH, or PACRUNNER_JULIA), this repository cloned and installed
with `python -m pip install -e .`, a MadGraph7 checkout with MadSpace for running libraries:

    import pacrunner
    pacrunner.configure(madgraph7="~/MadGraph7")      # or PACRUNNER_MADGRAPH7
"""

from ._env import CONFIG, Config, PacRunnerError, configure, repository

__version__ = "0.1.0"


def __getattr__(name):
    # the low-level layer on first use (it starts nothing until one of its functions is called)
    if name == "lowlevel":
        import importlib
        return importlib.import_module(".lowlevel", __name__)
    raise AttributeError(f"module 'pacrunner' has no attribute {name!r}")


__all__ = ["CONFIG", "Config", "PacRunnerError", "configure", "repository", "lowlevel", "__version__"]
