"""pacrunner.PacArtifact: information, the producing pyAmpliCol, the stock runtime, export and compile.

    PACRUNNER_MADGRAPH7=~/MadGraph7 PACRUNNER_TEST_ARTIFACT=ARTIFACT PACRUNNER_PYAMPLICOL=PYTHON \
        python -m unittest discover -s test/python

PACRUNNER_TEST_ARTIFACT: a stand-alone LC artifact with its API bundle (`pyamplicol generate ...
--color-accuracy lc --emit-api-bundle`); PACRUNNER_PYAMPLICOL: the pyAmpliCol that generated it (a
python, or an install folder). The compile test starts Julia and builds the libraries (~15 s). The
tests of the producer check and the provider choice need nothing."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pacrunner
from pacrunner import _pyamplicol

ARTIFACT = os.environ.get("PACRUNNER_TEST_ARTIFACT")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and (np.array_equal(a.view(np.uint64), b.view(np.uint64)) if a.dtype == np.float64
                                   else np.array_equal(a, b))


def fake_artifact(revision="0" * 40, providers=("p0",)):
    """an artifact folder with the files PacArtifact reads, none of pyAmpliCol's payloads"""
    d = Path(tempfile.mkdtemp()) / "fake"
    (d / "API" / "umami").mkdir(parents=True)
    (d / "artifact.json").write_text(json.dumps({"artifact_id": "f" * 64, "producer": {"version": "9.9", "git_revision": revision},
                                                 "processes": [{"id": "fake", "expression": "a > b"}]}))
    meta = {"kind": "pyamplicol-umami-metadata", "provider": {"id": "p0", "particle_count": 2, "masses": [0.0, 0.0],
            "color_accuracy": "lc"}, "channels": [{"processes": [{}]}]}
    if len(providers) == 1:
        (d / "API" / "umami" / "metadata.json").write_text(json.dumps(meta))
    else:
        (d / "API" / "umami" / "metadata.json").write_text(json.dumps({"kind": "pyamplicol-umami-provider-index",
            "providers": [{"id": p, "metadata": f"metadata_{p}.json"} for p in providers]}))
        for p in providers:
            (d / "API" / "umami" / f"metadata_{p}.json").write_text(json.dumps(meta))
    return d


class Producer(unittest.TestCase):
    def test_an_unknown_revision_is_refused_with_the_way_out(self):
        a = pacrunner.PacArtifact(fake_artifact())
        with self.assertRaises(pacrunner.PacRunnerError) as e:
            a.pyamplicol()
        self.assertIn("no installation of that revision", str(e.exception))
        self.assertIn("PACRUNNER_PYAMPLICOL", str(e.exception))

    def test_a_provider_must_be_picked(self):
        d = fake_artifact(providers=("p0", "p1"))
        with self.assertRaises(pacrunner.PacRunnerError):
            pacrunner.PacArtifact(d)
        a = pacrunner.PacArtifact(d, provider="p1")
        self.assertEqual((a.provider, a.providers), ("p1", ["p0", "p1"]))

    def test_a_configured_installation_must_exist(self):
        old = pacrunner.CONFIG.pyamplicol
        try:
            pacrunner.configure(pyamplicol=tempfile.mkdtemp())
            with self.assertRaises(pacrunner.PacRunnerError):
                _pyamplicol.candidates()
        finally:
            pacrunner.CONFIG.pyamplicol = old


@unittest.skipUnless(ARTIFACT, "PACRUNNER_TEST_ARTIFACT not set")
class Artifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # a copy: the stock library and the libraries are built beside it
        cls.dir = Path(tempfile.mkdtemp())
        shutil.copytree(ARTIFACT, cls.dir / Path(ARTIFACT).name, symlinks=True)
        cls.a = pacrunner.PacArtifact(cls.dir / Path(ARTIFACT).name)

    def test_information(self):
        a = self.a
        self.assertEqual(a.producer_revision, a.pyamplicol().revision)
        self.assertEqual(len(a.masses), a.particle_count)
        self.assertGreaterEqual(a.channels, 1)
        self.assertTrue(all(a.flavours(c) >= 1 for c in range(a.channels)))
        points = a.validation_points()
        self.assertTrue(points and all(p.shape == (a.particle_count, 4) for _, p in points))
        self.assertIn("alpha_s", a.parameters)

    def test_compile_is_the_stock_library(self):
        a = self.a
        lib = a.compile(self.dir / "libs", device="cpu")
        self.assertEqual((lib.device, lib.channels), ("cpu", a.channels))
        momenta = np.stack([p for _, p in a.validation_points()])
        n = len(momenta)
        r = np.linspace(0.05, 0.95, n)
        for c in range(a.channels):
            stock, ours = a.runtime(c), lib.runtime(c, threads=1)
            for f in range(a.flavours(c)):
                s, o = stock(momenta, f, random_helicity=r), ours(momenta, f, random_helicity=r)
                for name in ("matrix_element", "color_index", "helicity_index"):
                    self.assertTrue(same(getattr(s, name), getattr(o, name)), f"channel {c} flavour {f}: {name}")
                for h in range(len(a.metadata["helicities"])):
                    self.assertTrue(same(stock(momenta, f, helicity=h).matrix_element,
                                         ours(momenta, f, helicity=h).matrix_element), f"helicity {h}")
        # unchanged: the stamp skips the build; force rebuilds
        stamp = (self.dir / "libs" / "pacrunner-build.json").stat().st_mtime_ns
        a.compile(self.dir / "libs", device="cpu")
        self.assertEqual((self.dir / "libs" / "pacrunner-build.json").stat().st_mtime_ns, stamp)
        a.compile(self.dir / "libs", device="cpu", force=True)
        self.assertNotEqual((self.dir / "libs" / "pacrunner-build.json").stat().st_mtime_ns, stamp)


if __name__ == "__main__":
    unittest.main()
