"""The tutorial's snippets (tutorial/*.py), run as they are written, each a script in a scratch folder
holding what the tutorial makes first: artifacts/zgg (tutorial/generate_zgg.sh) and processes/zgg
(README: `generate u u~ > z g g [amplicol]`, `output mg7 processes/zgg`).

    PACRUNNER_TEST_TUTORIAL=1 PACRUNNER_PYAMPLICOL=PYTHON PACRUNNER_MADGRAPH7=~/mg7 \
        python -m unittest discover -s test/python -p test_tutorial.py

Optional: not part of the default suite; every test here skips unless PACRUNNER_TEST_TUTORIAL is
set (~3 minutes: it generates the artifact and two or three MadGraph7 outputs).
PACRUNNER_PYAMPLICOL: a pyAmpliCol installation, the python of a virtual environment with its
`pyamplicol` command beside it, or an install folder (MadGraph7's pyamplicol/install); generate_zgg.sh
runs its `pyamplicol`; PACRUNNER_MADGRAPH7: MadGraph7 (bin/madgraph) for processes/zgg. Without
them the tests skip. The GPU snippets need nvcc (and CuPy, or MadSpace's CUDA build, for
cuda_provider). Every snippet of tutorial/ must be one of SNIPPETS: a new one fails until it is.

A snippet's `# should print X` is its expected output (the last line it prints): exact for the
matrix elements. A cross section depends on the run's seed (the default run card draws one), so
the cross-section snippets run on an output with a fixed seed and must print the stock run's cross
section of that output (device cuda: the GPU matrix elements, MadSpace on the CPU), or agree with
it within the errors (cuda_provider: MadSpace on the GPU samples another phase space, and the run
card needs run.device = ["cuda"] and color_accuracy = "lc", which the default card does not have: the
reference is then the stock run of an LC output, not the default full-colour one)."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL = unittest.skipUnless(os.environ.get("PACRUNNER_TEST_TUTORIAL"), "optional: PACRUNNER_TEST_TUTORIAL not set")
TUTORIAL = ROOT / "tutorial"
PYAMPLICOL = os.environ.get("PACRUNNER_PYAMPLICOL")
MADGRAPH7 = os.environ.get("PACRUNNER_MADGRAPH7")
SEED = 7

# snippet -> (what it needs: "artifact" or "output"; device)
SNIPPETS = {
    "pac_stock.py": ("artifact", "cpu"),
    "pac_cpu.py": ("artifact", "cpu"),
    "pac_cuda.py": ("artifact", "cuda"),
    "pac_cuda_provider.py": ("artifact", "cuda_provider"),
    "mad_cpu.py": ("output", "cpu"),
    "cross_section_cuda.py": ("output", "cuda"),
    "cross_section_cuda_provider.py": ("output", "cuda_provider"),
}


def expected(snippet):
    """the snippet's `# should print X`"""
    m = re.search(r"#\s*should print\s+(.+?)\s*$", (TUTORIAL / snippet).read_text(), re.M)
    return m.group(1) if m else None


def environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime"))
    os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
    return env


def run_snippet(snippet, folder):
    """the snippet run in `folder`: its last line of output"""
    r = subprocess.run([sys.executable, str(TUTORIAL / snippet)], cwd=folder, env=environment(),
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise AssertionError(f"{snippet} failed (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if not lines:
        raise AssertionError(f"{snippet} printed nothing:\n{r.stderr[-3000:]}")
    return lines[-1].strip()


def nvcc():
    return os.environ.get("NVCC") or shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)


def has_cupy():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def skip_device(test, device):
    if device in ("cuda", "cuda_provider") and nvcc() is None:
        test.skipTest(f"{device}: no nvcc")


def pyamplicol_command(folder):
    """a `pyamplicol` command of PACRUNNER_PYAMPLICOL: beside its python (a virtual environment), or
    for an install folder (MadGraph7's pyamplicol/install, whose bin/pyamplicol needs the folder on
    PYTHONPATH) a wrapper in `folder` running `python -m pyamplicol` with it"""
    if not PYAMPLICOL:
        return None
    path = Path(PYAMPLICOL)
    if path.is_file():
        command = path.parent / "pyamplicol"
        return command if command.exists() else None
    if not (path / "pyamplicol").is_dir():
        return None
    command = Path(folder) / "bin" / "pyamplicol"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(f'#!/bin/sh\nPYTHONPATH="{path}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{sys.executable}" -m pyamplicol "$@"\n')
    command.chmod(0o755)
    return command


def make_output(folder, device="cpu", color_accuracy=None):
    """processes/zgg in `folder`, as the README makes it, then the run card's seed fixed and its
    device set (cuda: color_accuracy = "lc" too, which PacRunner on a CUDA device needs: the stock
    full-colour ratio library is a CPU one); the output's path"""
    (folder / "processes").mkdir(parents=True, exist_ok=True)   # MadGraph7 creates no parent folder
    commands = folder / "mg7_commands.txt"
    commands.write_text("generate u u~ > z g g [amplicol]\noutput mg7 processes/zgg\n")
    r = subprocess.run([sys.executable, str(Path(MADGRAPH7) / "bin" / "madgraph"), str(commands)], cwd=folder,
                       env=environment(), capture_output=True, text=True, timeout=3600)
    output = folder / "processes" / "zgg"
    if r.returncode != 0 or not (output / "bin" / "generate_events").exists():
        raise AssertionError(f"MadGraph7 made no processes/zgg (exit {r.returncode}):\n{r.stdout[-3000:]}")
    card = output / "Cards" / "run_card.toml"
    text = re.sub(r"(?m)^seed = .*$", f"seed = {SEED}", card.read_text())
    text = re.sub(r'(?m)^device = .*$', f'device = ["{device}"]', text)
    if device == "cuda" or color_accuracy is not None:
        text = re.sub(r'(?m)^color_accuracy = .*$', f'color_accuracy = "{color_accuracy or "lc"}"', text)
    card.write_text(text)
    return output


def stock_cross_section(output):
    """the output's own run (its stock libraries): (cross section, error)"""
    before = set((output / "Events").glob("run_*")) if (output / "Events").exists() else set()
    subprocess.run([sys.executable, "bin/generate_events", "-f"], cwd=output, env=environment(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
    run = max(set((output / "Events").glob("run_*")) - before, key=lambda d: int(d.name.rsplit("_", 1)[1]))
    with open(run / "info.json") as f:
        process = json.load(f)["process"]
    return float(process["mean"]), float(process["error"])


@OPTIONAL
class Snippets(unittest.TestCase):
    def test_every_snippet_is_tested(self):
        found = sorted(p.name for p in TUTORIAL.glob("*.py"))
        self.assertEqual(found, sorted(SNIPPETS), "tutorial/ and SNIPPETS differ: add the new snippets here")

    def test_every_snippet_says_what_it_prints(self):
        for snippet in SNIPPETS:
            self.assertIsNotNone(expected(snippet), f"{snippet}: no `# should print`")


@OPTIONAL
@unittest.skipUnless(PYAMPLICOL, "PACRUNNER_PYAMPLICOL not set")
class ArtifactSnippets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        command = pyamplicol_command(cls.folder / "tools")
        if command is None:
            cls.tmp.cleanup()
            raise unittest.SkipTest(f"no pyamplicol command for {PYAMPLICOL}")
        env = environment()
        env["PATH"] = os.pathsep.join([str(command.parent), env.get("PATH", "")])
        r = subprocess.run(["sh", str(TUTORIAL / "generate_zgg.sh")], cwd=cls.folder, env=env,
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not (cls.folder / "artifacts" / "zgg" / "artifact.json").exists():
            raise AssertionError(f"generate_zgg.sh failed (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, snippet):
        need, device = SNIPPETS[snippet]
        skip_device(self, device)
        if snippet == "pac_cuda_provider.py" and not has_cupy():
            self.skipTest("no CuPy")
        self.assertEqual(run_snippet(snippet, self.folder), expected(snippet))

    def test_pac_stock(self):
        self.check("pac_stock.py")

    def test_pac_cpu(self):
        self.check("pac_cpu.py")

    def test_pac_cuda(self):
        self.check("pac_cuda.py")

    def test_pac_cuda_provider(self):
        self.check("pac_cuda_provider.py")


@OPTIONAL
@unittest.skipUnless(MADGRAPH7, "PACRUNNER_MADGRAPH7 not set")
class OutputSnippets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "cpu"
        cls.output = make_output(cls.folder)
        cls.stock = None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stock_run(self):
        if self.stock is None:
            type(self).stock = stock_cross_section(self.output)
        return self.stock

    def test_mad_cpu(self):
        self.assertEqual(run_snippet("mad_cpu.py", self.folder), expected("mad_cpu.py"))

    def test_cross_section_cuda(self):
        skip_device(self, "cuda")
        xs, _ = self.stock_run()
        # the GPU matrix elements in f64 are the stock library's bits at the card's alpha_s; the
        # default scale is dynamic, and the libraries scale by (alpha_s / alpha_s0)^power where the
        # stock library evaluates at alpha_s: an ulp per event, ~1e-14 on the cross section
        got = float(run_snippet("cross_section_cuda.py", self.folder))
        self.assertLess(abs(got - xs), 1e-12 * abs(xs), f"{got} vs the stock {xs}")

    def test_cross_section_cuda_provider(self):
        skip_device(self, "cuda_provider")
        mg7 = Path(MADGRAPH7)
        if not (mg7 / "madspace" / "install-cuda" / "madspace").is_dir():
            self.skipTest(f"no CUDA build of MadSpace in {mg7 / 'madspace' / 'install-cuda'}")
        # its own output: cuda_provider libraries need run.device = ["cuda"] and color_accuracy = "lc",
        # so the reference is the stock run of an LC output (on the CPU: the stock library is a CPU one)
        xs, error = stock_cross_section(make_output(Path(self.tmp.name) / "cpu_lc", color_accuracy="lc"))
        folder = Path(self.tmp.name) / "cuda_provider"
        make_output(folder, device="cuda")
        got = float(run_snippet("cross_section_cuda_provider.py", folder))
        # MadSpace on the GPU samples another phase space: the stock LC cross section within the errors
        self.assertLess(abs(got - xs), 5 * math.hypot(error, error), f"{got} vs the stock LC {xs} +- {error}")


if __name__ == "__main__":
    unittest.main()
