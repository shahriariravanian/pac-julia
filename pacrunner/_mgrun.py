"""Provider.run and Output.run: generate_events on PacRunner's libraries, by default in a process of
its own (MadGraph7's run is global state, once per process): the output's python (bin/generate_events'
shebang) runs `python -m pacrunner._madgraph` with this repository on its PYTHONPATH, the result back
through a JSON file; Ctrl-C stops it. in_process=True runs pacrunner._madgraph.run here instead."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

from . import _env


class RunResult(NamedTuple):
    cross_section: float     # pb
    error: float             # pb
    events: int              # unweighted events
    run_dir: Path            # OUTPUT/Events/run_N
    output: Path
    wall_seconds: float


def output_python(output):
    """the python of an mg7 output: bin/generate_events' shebang"""
    with open(Path(output) / "bin" / "generate_events") as f:
        line = f.readline()
    if not line.startswith("#!"):
        return sys.executable
    parts = line[2:].split()
    if parts and os.path.basename(parts[0]) == "env" and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else sys.executable


def run_libraries(output, libraries, args=None, in_process=False, log=None, python=None):
    """generate_events of `output` with `libraries` (Providers) in place of their stock libraries (a
    stock Provider: its subprocess stays stock; none of PacRunner's: the output's own run);
    `args` generate_events' arguments (default -f); `log` a file for the run's output (default: this
    process's stdout and stderr); `python` the interpreter (default: the output's). Returns a RunResult."""
    from . import _madgraph
    libraries = list(libraries)
    if not libraries:
        raise _env.PacRunnerError("no libraries to run")
    pairs = [(os.fspath(lib.folder), os.fspath(lib.artifact)) for lib in libraries if not lib.stock]
    if not pairs:
        if in_process:
            raise _env.PacRunnerError("in_process: for PacRunner's providers only (the stock run is bin/generate_events)")
        return run_stock(output, args=args, log=log, python=python)
    start = time.time()
    if in_process:
        r = _madgraph.run(pairs, args=args)
    else:
        runner = [a for lib, art in pairs for a in (lib, art)]
        r = _runner(output, runner, args, log, python, "generate_events on PacRunner's libraries")
    return RunResult(float(r["cross_section"]), float(r["error"]), int(r["events"]), Path(r["run"]),
                     Path(r["output"]), time.time() - start)


def run_stock(output, args=None, log=None, python=None):
    """the output's own generate_events (every subprocess on its stock library), in a process of its own
    (the output's python), with MadSpace as PacRunner's runs choose it (the CUDA build for a run on
    cuda); a RunResult"""
    start = time.time()
    r = _runner(output, ["--stock", os.fspath(Path(output).resolve())], args, log, python, "generate_events")
    return RunResult(float(r["cross_section"]), float(r["error"]), int(r["events"]), Path(r["run"]),
                     Path(r["output"]), time.time() - start)


def _runner(output, runner_args, args, log, python, what):
    """python -m pacrunner._madgraph --json RESULT RUNNER_ARGS -- ARGS in the output (its python); the
    result's dictionary"""
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.json"
        command = [python or output_python(output), "-m", "pacrunner._madgraph", "--json", os.fspath(result),
                   *runner_args, "--", *(args or ["-f"])]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([os.fspath(_env.repository())] +
                                            ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.setdefault("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime"))
        os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
        out = open(log, "w") if log is not None else None
        try:
            p = subprocess.Popen(command, env=env, cwd=os.fspath(output), stdout=out,
                                 stderr=subprocess.STDOUT if out else None)
            try:
                code = p.wait()
            except KeyboardInterrupt:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                raise
        finally:
            if out is not None:
                out.close()
        if code != 0 or not result.is_file():
            where = f" (its output: {log})" if log is not None else ""
            raise _env.PacRunnerError(f"{output}: {what} failed, exit code {code}{where}")
        with open(result) as f:
            return json.load(f)
