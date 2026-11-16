"""A pyAmpliCol installation: a virtual environment's python, or an install folder (MadGraph7's
pyamplicol/install: the package and bin/rusticol-config in it) with a python. pacrunner runs pyAmpliCol's
commands (export-recurrence) and builds its UMAMI bundles (rusticol-config) with the installation that
produced an artifact: artifact.json's producer.git_revision is the installation's
pyamplicol/_build_info.json source_revision (a newer or older one may not read the artifact: e.g.
"payloads[N].role is unsupported").

Candidates, in order: configure(pyamplicol=...), PACRUNNER_PYAMPLICOL (either a python executable or an
install folder), the MadGraph7 checkout's pyamplicol/install (configured, or the artifact's mg7 output's),
and this python's own pyamplicol."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import _env


@dataclass(frozen=True)
class PyAmpliCol:
    python: str              # the interpreter to run it with
    path: str | None         # a folder to put on PYTHONPATH (an install folder), or None
    package: str             # the pyamplicol package folder
    revision: str | None     # its source revision (_build_info.json)
    version: str | None

    @property
    def rusticol_config(self):
        """rusticol-config of this installation (builds API/umami bundles)"""
        for candidate in ((Path(self.path) / "bin" / "rusticol-config") if self.path else None,
                          Path(self.python).parent / "rusticol-config"):
            if candidate is not None and candidate.is_file():
                return os.fspath(candidate)
        raise _env.PacRunnerError(f"pyAmpliCol at {self.package}: no rusticol-config beside it")

    def env(self):
        env = dict(os.environ)
        if self.path:
            env["PYTHONPATH"] = os.pathsep.join([self.path] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.setdefault("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime"))   # Symbolica's lock folder
        os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
        return env

    def run(self, *args):
        """`python -m pyamplicol ARGS` with this installation; raises with its output on failure"""
        r = subprocess.run([self.python, "-m", "pyamplicol", *map(os.fspath, args)], env=self.env(),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise _env.PacRunnerError(f"pyamplicol {' '.join(map(str, args[:1]))} failed (pyAmpliCol {self.version} "
                                      f"at {self.package}):\n{(r.stderr or r.stdout)[-3000:]}")
        return r.stdout


def _build_info(package):
    try:
        with open(Path(package) / "_build_info.json") as f:
            info = json.load(f)
        return info.get("source_revision"), info.get("version")
    except (OSError, ValueError):
        return None, None


def from_folder(folder, python=None):
    """an install folder (holding pyamplicol/), run with `python` (default: this one)"""
    package = Path(folder) / "pyamplicol"
    if not (package / "__init__.py").is_file():
        return None
    revision, version = _build_info(package)
    return PyAmpliCol(python or sys.executable, os.fspath(Path(folder).resolve()), os.fspath(package.resolve()),
                      revision, version)


def from_python(python):
    """the pyamplicol a python imports (a virtual environment's)"""
    r = subprocess.run([python, "-c", "import os, pyamplicol; print(os.path.dirname(pyamplicol.__file__))"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    package = r.stdout.strip().splitlines()[-1]
    revision, version = _build_info(package)
    return PyAmpliCol(python, None, package, revision, version)


def _from_spec(spec):
    spec = os.path.expanduser(os.fspath(spec))
    return from_folder(spec) if os.path.isdir(spec) else from_python(spec)


def candidates(near=None):
    """the pyAmpliCol installations to consider, in order (`near`: an artifact, whose mg7 output's
    MadGraph7 counts)"""
    found = []
    for spec in (_env.CONFIG.pyamplicol, os.environ.get("PACRUNNER_PYAMPLICOL")):
        if spec:
            p = _from_spec(spec)
            if p is None:
                raise _env.PacRunnerError(f"{spec}: no pyAmpliCol there (a python that imports pyamplicol, or an install "
                                          "folder holding pyamplicol/)")
            found.append(p)
    try:
        mg7 = _env.madgraph7_for(near)
    except _env.PacRunnerError:
        mg7 = None
    if mg7 is not None:
        p = from_folder(mg7 / "pyamplicol" / "install")
        if p is not None:
            found.append(p)
    p = from_python(sys.executable)
    if p is not None:
        found.append(p)
    return found


def producer_of(artifact_json, near=None, strict=True):
    """the installation that produced an artifact (its producer.git_revision); with strict=False the
    first candidate when none matches (a warning is the caller's)"""
    with open(artifact_json) as f:
        producer = json.load(f).get("producer", {})
    revision, version = producer.get("git_revision"), producer.get("version")
    found = candidates(near)
    for p in found:
        if revision is not None and p.revision == revision:
            return p
    if not strict and found:
        return found[0]
    have = "; ".join(f"{p.version} at {p.revision or '?'} ({p.package})" for p in found) or "none found"
    raise _env.PacRunnerError(
        f"{Path(artifact_json).parent}: made by pyAmpliCol {version} at revision {revision}, and no installation of that "
        f"revision is known ({have}). Point pacrunner at it: pacrunner.configure(pyamplicol=PYTHON or INSTALL_FOLDER) "
        "or PACRUNNER_PYAMPLICOL (MadGraph7's is MADGRAPH7/pyamplicol/install), or regenerate the artifact with an "
        "installation you have")
