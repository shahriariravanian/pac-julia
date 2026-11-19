"""pacrunner.Output, SubProcess and Provider.run on a MadGraph7 output of several subprocesses.

    PACRUNNER_MADGRAPH7=~/MadGraph7 PACRUNNER_TEST_MG7_MULTI=OUTPUT python -m unittest discover -s test/python

PACRUNNER_TEST_MG7_MULTI: an mg7 output of two pyAmpliCol processes with color_accuracy = "lc" and
run.device = ["cpu"], e.g. `generate p p > z j [amplicol]`, `add process p p > z j j [amplicol]`,
`output mg7 OUTPUT`, with fixed scales and the param card's aS the run's alpha_s (MadGraph7's
test_mg7_amplicol card: tests/acceptance_tests/test_mg7_amplicol.py _edit_run_card, then aS =
0.13000271085472234): the runs are compared with the stock run bit for bit, and with another alpha_s
per event (a dynamic scale) PacRunner's (alpha_s / alpha_s0)^power scaling differs from the stock
library by an ulp, which changes the unweighting. The tests compile its libraries (lib/pacrunner/, ~20 s the first time) and add
runs to its Events/ (a stock one, then on PacRunner: each a few seconds for this process). The
param-card reader's test needs nothing."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pacrunner
from pacrunner import _mad, _mgrun

OUTPUT = os.environ.get("PACRUNNER_TEST_MG7_MULTI")

CARD = """\
######################################################################
BLOCK MASS #
      5 4.700000e+00 # mb
      23 9.118800e+01 # mz
BLOCK SMINPUTS #
      1 1.325070e+02 # aewm1
      3 1.300027e-01 # as
Block yukawa
    6 1.73d+02 # ymt
DECAY 23 2.441404e+00 # wz
DECAY 25 6.382339e-03
BLOCK QNUMBERS 9000001 # gh
      1 0 # 3 times electric charge
"""


class ParamCard(unittest.TestCase):
    def test_the_card_as_madgraph_reads_it(self):
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
            f.write(CARD)
        refs = {"MZ": ["MASS", [23]], "aS": ["SMINPUTS", [3]], "ymt": ["YUKAWA", [6]], "WZ": ["DECAY", [23]],
                "WH": ["DECAY", [25]], "MT": ["MASS", [6]]}
        self.assertEqual(_mad.card_parameters(f.name, refs),
                         {"MZ": 91.188, "aS": 0.1300027, "ymt": 173.0, "WZ": 2.441404, "WH": 0.006382339})
        os.unlink(f.name)

    def test_folder_names(self):
        self.assertEqual([_mad.library_folder_name(*v) for v in (("cpu", "f64"), ("cuda", "f64"), ("cuda", "mixed"),
                                                                 ("cuda_provider", "mixed"))],
                         ["pacrunner", "pacrunner-cuda", "pacrunner-cuda-mixed", "pacrunner-cuda_provider-mixed"])
        # cuda_full is cuda_provider: the same folder (one build)
        self.assertEqual(_mad.library_folder_name("cuda_full", "mixed"), "pacrunner-cuda_provider-mixed")
        self.assertEqual(pacrunner._env.device_name("cuda_full"), "cuda_provider")


def stock_run(output):
    """generate_events of the output as it is (its stock libraries): (cross section, events.npy)"""
    output = Path(output)
    env = dict(os.environ, XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR", os.path.expanduser("~/.cache/xdg-runtime")))
    subprocess.run([_mgrun.output_python(output), "bin/generate_events", "-f"], cwd=output, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run = max((output / "Events").glob("run_*"), key=lambda d: int(d.name.rsplit("_", 1)[1]))
    with open(run / "info.json") as f:
        return float(json.load(f)["process"]["mean"]), np.load(run / "events.npy")


@unittest.skipUnless(OUTPUT, "PACRUNNER_TEST_MG7_MULTI not set")
class OutputRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = pacrunner.Output(OUTPUT)
        cls.libs = cls.out.compile(device="cpu")
        cls.stock_xs, cls.stock_events = stock_run(OUTPUT)

    def test_layout(self):
        out = self.out
        self.assertGreaterEqual(len(out.names), 2)
        with open(out.path / "SubProcesses" / "subprocesses.json") as f:
            entries = [e for e in json.load(f) if e.get("backend") == "amplicol"]
        self.assertEqual(sum(a.channels for a in out.subprocesses()), len(entries))
        self.assertEqual(out.names, sorted({os.path.basename(e["path"]) for e in entries}))   # the SubProcesses folders
        for a, lib in zip(out.subprocesses(), self.libs):
            self.assertEqual((lib.name, lib.output, lib.channels), (a.library_name, out.path, a.channels))
            self.assertEqual(a.library_name, f"{a.name}_{a.provider}")
            self.assertEqual((lib.subprocess.name, lib.subprocess.provider), (a.name, a.provider))
            self.assertTrue(a.stock_library_path.is_file())
            self.assertIn("aS", a.parameters)
            with open(lib.folder / "pacrunner-build.json") as f:
                self.assertEqual(json.load(f)["parameters"], dict(sorted(a.parameters.items())))

    def test_compile_is_the_stock_library(self):
        rng = np.random.default_rng(3)
        for a, lib in zip(self.out.subprocesses(), self.libs):
            momenta = np.stack([p for _, p in a.artifact.validation_points()])
            for c in range(a.channels):
                flavours = rng.integers(0, a.flavours(c), len(momenta))
                r = rng.random(len(momenta))
                s = a.runtime(c)(momenta, flavours, random_helicity=r)
                o = lib.runtime(c, threads=1)(momenta, flavours, random_helicity=r)
                for name in ("matrix_element", "color_index", "helicity_index"):
                    self.assertTrue(np.array_equal(getattr(s, name).view(np.uint64) if name == "matrix_element"
                                                   else getattr(s, name),
                                                   getattr(o, name).view(np.uint64) if name == "matrix_element"
                                                   else getattr(o, name)), f"{a.name} channel {c}: {name}")

    def test_the_stock_provider(self):
        for a, lib in zip(self.out.subprocesses(), self.libs):
            sp = a.stock()
            self.assertTrue(sp.stock and not lib.stock)
            self.assertEqual((sp.name, sp.output, sp.subprocess, sp.channels), (a.library_name, self.out.path, a, a.channels))
            for c in range(a.channels):
                # MadGraph7's flavours, as its own integrand remaps them
                self.assertEqual(sp.flavor_arguments(c)["flavor_remap"], [f["index"] for f in a.entries[c]["flavors"]])
                mes, tables = sp.integrand_inputs(c)
                self.assertEqual(len(mes), 1)

    def test_run_the_stock_providers(self):
        # the stock provider's run is the output's own: the stock cross section
        r = self.out.subprocesses()[0].stock().run(log=tempfile.mktemp(suffix=".log"))
        self.assertEqual(r.cross_section, self.stock_xs)
        r = self.out.run(providers=[s.stock() for s in self.out.subprocesses()], log=tempfile.mktemp(suffix=".log"))
        self.assertEqual(r.cross_section, self.stock_xs)

    def test_integrand_inputs(self):
        lib = self.libs[0]
        mes, tables = lib.integrand_inputs(0)
        self.assertEqual(len(mes), len(lib._channel(0)["libraries"]))
        self.assertEqual(len(tables["flavor_remap"]), lib.flavours(0))

    def check_events(self, result, ours):
        """the run's events: the stock run's but for the spins of the subprocesses on PacRunner"""
        events = np.load(result.run_dir / "events.npy")
        stock = self.stock_events
        self.assertEqual(len(events), len(stock))
        particles = stock["part5_pdg_id"] != 0   # the larger process's events
        for name in stock.dtype.names:
            same = stock[name] == events[name]
            if name.endswith("_spin"):
                # the helicity draw of a split channel: its subprocess's events only
                self.assertTrue(same[~ours(particles)].all(), name)
            else:
                self.assertTrue(same.all(), name)

    def test_run_all_on_pacrunner(self):
        r = self.out.run(device="cpu", log=tempfile.mktemp(suffix=".log"))
        self.assertEqual(r.cross_section, self.stock_xs)
        self.check_events(r, lambda larger: np.ones_like(larger))

    def test_run_the_first_subprocess_on_pacrunner(self):
        # subprocess 0 on PacRunner, the others stock: the event layout is subprocess 0's
        r = self.libs[0].run(log=tempfile.mktemp(suffix=".log"))
        self.assertEqual(r.cross_section, self.stock_xs)
        self.check_events(r, lambda larger: ~larger)

    def test_run_the_last_subprocess_on_pacrunner(self):
        r = self.libs[-1].run(log=tempfile.mktemp(suffix=".log"))
        self.assertEqual(r.cross_section, self.stock_xs)
        self.check_events(r, lambda larger: larger)


@unittest.skipUnless(OUTPUT, "PACRUNNER_TEST_MG7_MULTI not set")
class OldNames(unittest.TestCase):
    def test_the_0_1_names_still_work(self):
        # aliases for one release
        self.assertIs(pacrunner.MadOutput, pacrunner.Output)
        self.assertIs(pacrunner.MadArtifact, pacrunner.SubProcess)
        self.assertIs(pacrunner.PacArtifact, pacrunner.Artifact)
        self.assertIs(pacrunner.PacLibrary, pacrunner.Provider)
        self.assertIs(pacrunner.MadLibrary, pacrunner.Provider)
        out = pacrunner.Output(OUTPUT)
        s = out.subprocess(out.names[0])
        s.compile(device="cpu")   # its own build (OutputRuns, which also makes it, runs after this class)
        # a stock library's name (subprocess_provider) still names a subprocess, as MadArtifact's did
        self.assertEqual(out.artifact(s.library_name).name, s.name)
        self.assertEqual([a.library_name for a in out.artifacts()], [a.library_name for a in out.subprocesses()])
        self.assertEqual(s.library().folder, s.compiled().folder)
        self.assertEqual(s.compiled().mad_artifact().name, s.name)


if __name__ == "__main__":
    unittest.main()
