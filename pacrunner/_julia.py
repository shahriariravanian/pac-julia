"""Julia in this process, through juliacall: the first call starts it and loads PacRunner, which
pacrunner/juliapkg.json puts in juliacall's environment (this repository, developed in place; juliapkg
installs its dependencies at the first start: no Julia REPL needed). Only the low-level API
and building libraries need it; loading and running libraries does not."""

import os
import threading

from ._env import PacRunnerError, julia, repository

_lock = threading.Lock()
_pacrunner = None


def booted() -> bool:
    return _pacrunner is not None


def pacrunner():
    """the PacRunner module (Julia's, through juliacall), started once"""
    global _pacrunner
    if _pacrunner is not None:
        return _pacrunner
    with _lock:
        if _pacrunner is None:
            # juliacall: this Julia (not a copy juliapkg would download), Julia's own signal handling
            # (Ctrl-C, its threads), this repository's project instead of juliapkg's
            os.environ.setdefault("PYTHON_JULIAPKG_EXE", julia())
            os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
            try:
                from juliacall import Main as jl
            except ImportError as e:
                raise PacRunnerError("juliacall is missing: `python -m pip install -e .` in the pac-julia "
                                     f"repository installs it ({e})") from e
            # PacRunner is in juliacall's own environment (pacrunner/juliapkg.json: this repository,
            # developed in place), resolved with juliacall's packages: one set of versions
            repository()
            jl.seval("using PacRunner")
            _pacrunner = jl.PacRunner
    return _pacrunner


def main():
    """juliacall's Main (after pacrunner())"""
    pacrunner()
    from juliacall import Main
    return Main
