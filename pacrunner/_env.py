"""Where pacrunner's tools are: this repository (the Julia project), Julia, a MadGraph7 checkout with
its MadSpace (CPU or CUDA build) and pyAmpliCol. Set with pacrunner.configure(...), else from the
environment (PACRUNNER_JULIA, PACRUNNER_MADGRAPH7), else found."""

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


class PacRunnerError(RuntimeError):
    """A pacrunner setup or use error, with what to do about it"""


# the repository: the Julia project (Project.toml, src/PacRunner.jl) is this package's parent folder
# (pacrunner is installed editable, from a clone)
REPOSITORY = Path(__file__).resolve().parent.parent


def repository() -> Path:
    if not (REPOSITORY / "Project.toml").is_file() or not (REPOSITORY / "src" / "PacRunner.jl").is_file():
        raise PacRunnerError(f"{REPOSITORY}: not the pac-julia repository (no Project.toml, src/PacRunner.jl): "
                             "install pacrunner from a clone, `python -m pip install -e .` in it")
    return REPOSITORY


@dataclass
class Config:
    julia: str | None = None            # the julia executable
    madgraph7: str | None = None        # a MadGraph7 checkout (madspace/, pyamplicol/)
    cuda: bool | None = None            # MadSpace's CUDA build (None: when MadGraph7 has one)
    pyamplicol: str | None = None       # a pyAmpliCol installation: a python or an install folder


CONFIG = Config()


def configure(julia=None, madgraph7=None, cuda=None, pyamplicol=None):
    """Where pacrunner's tools are, for this process: `julia` (the executable; default PACRUNNER_JULIA,
    else `julia` on the PATH), `madgraph7` (a MadGraph7 checkout with MadSpace and pyAmpliCol installed;
    default PACRUNNER_MADGRAPH7, or an mg7 output's own), `cuda` (True: MadSpace's CUDA build, False: the
    CPU build; default: the CUDA build when MadGraph7 has one), `pyamplicol` (a pyAmpliCol installation
    for artifacts it produced: a python that imports pyamplicol, or an install folder such as
    MADGRAPH7/pyamplicol/install; default PACRUNNER_PYAMPLICOL, else found). Julia and MadSpace are loaded
    once per process: configure them before their first use. Returns the configuration."""
    from . import _julia
    if julia is not None:
        if _julia.booted():
            raise PacRunnerError("Julia is already running in this process: configure(julia=...) before the first use")
        CONFIG.julia = os.fspath(julia)
    if madgraph7 is not None or cuda is not None:
        if "madspace" in sys.modules:
            raise PacRunnerError("MadSpace is already loaded in this process: configure(madgraph7=..., cuda=...) "
                                 "before the first runtime")
        if madgraph7 is not None:
            CONFIG.madgraph7 = os.fspath(Path(madgraph7).expanduser().resolve())
        if cuda is not None:
            CONFIG.cuda = bool(cuda)
    if pyamplicol is not None:
        CONFIG.pyamplicol = os.fspath(Path(pyamplicol).expanduser())
    return CONFIG


def julia() -> str:
    """the julia executable: configure(julia=...), PACRUNNER_JULIA, or `julia` on the PATH"""
    exe = CONFIG.julia or os.environ.get("PACRUNNER_JULIA") or shutil.which("julia")
    if not exe or not (shutil.which(exe) or os.path.isfile(exe)):
        raise PacRunnerError("no Julia: install it (https://julialang.org/downloads/, e.g. juliaup) and put `julia` "
                             "on the PATH, or set PACRUNNER_JULIA, or pacrunner.configure(julia=...)")
    return shutil.which(exe) or exe


def madgraph7(required=True) -> Path | None:
    """the MadGraph7 checkout: configure(madgraph7=...) or PACRUNNER_MADGRAPH7"""
    path = CONFIG.madgraph7 or os.environ.get("PACRUNNER_MADGRAPH7") or None   # an empty variable: unset
    if path is None:
        if required:
            raise PacRunnerError("no MadGraph7: pacrunner.configure(madgraph7=...) or PACRUNNER_MADGRAPH7 (a MadGraph7 "
                                 "checkout with MadSpace installed: `install madspace` in its bin/madgraph)")
        return None
    path = Path(path).expanduser().resolve()
    if not (path / "bin" / "madgraph").is_file():
        raise PacRunnerError(f"{path}: not a MadGraph7 checkout (no bin/madgraph)")
    return path


def mg7_output_of(path) -> Path | None:
    """the mg7 output holding `path` (the first folder, `path` or above, with bin/generate_events)"""
    d = Path(path).expanduser().resolve()
    for p in (d, *d.parents):
        if (p / "bin" / "generate_events").is_file():
            return p
    return None


def madgraph7_of_output(output) -> Path:
    """the MadGraph7 checkout an mg7 output was made with (its bin/generate_events' sys.path line)"""
    import re
    text = (Path(output) / "bin" / "generate_events").read_text()
    m = re.search(r"sys\.path\.append\('([^']*)'\)", text)
    if m is None:
        raise PacRunnerError(f"{output}: bin/generate_events names no MadGraph7 folder")
    return Path(m.group(1)).resolve()


def madgraph7_for(path=None, required=True) -> Path | None:
    """the MadGraph7 checkout for something at `path`: the configured one, else the one of the mg7 output
    holding `path` (none: None, or the error when `required`)"""
    mg7 = madgraph7(required=False)
    if mg7 is not None:
        return mg7
    output = mg7_output_of(path) if path is not None else None
    if output is not None:
        return madgraph7_of_output(output)
    return madgraph7(required=required)   # none configured: None or the error


def madspace_dir(mg7: Path) -> Path:
    """the MadSpace build to import: install-cuda (configure(cuda=True), or by default when it exists:
    it has the CPU device too), else install"""
    cuda, cpu = mg7 / "madspace" / "install-cuda", mg7 / "madspace" / "install"
    want_cuda = CONFIG.cuda if CONFIG.cuda is not None else (cuda / "madspace").is_dir()
    path = cuda if want_cuda else cpu
    if not (path / "madspace").is_dir():
        what = "a CUDA build of MadSpace (cmake -DENABLE_CUDA=ON)" if want_cuda else "MadSpace (`install madspace`)"
        raise PacRunnerError(f"{mg7}: no {what} in {path}")
    return path


def import_madspace(mg7: Path | None = None):
    """MadSpace, imported once for this process: from the MadGraph7 checkout (`mg7`, or the configured
    one), else the one installed in this python (`import madspace`); Symbolica's lock folder defaulted
    (XDG_RUNTIME_DIR) for pyAmpliCol's libraries"""
    if "madspace" in sys.modules:
        return sys.modules["madspace"]
    mg7 = mg7 or madgraph7(required=False)
    os.environ.setdefault("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime"))
    os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
    if mg7 is not None:
        sys.path.insert(0, os.fspath(madspace_dir(mg7)))
    try:
        import madspace
    except ImportError as e:
        if mg7 is not None:
            raise
        raise PacRunnerError("no MadSpace: install it in this python, or point pacrunner at a MadGraph7 checkout with "
                             "MadSpace installed (pacrunner.configure(madgraph7=...) or PACRUNNER_MADGRAPH7; "
                             "`install madspace` in its bin/madgraph)") from e
    return madspace
