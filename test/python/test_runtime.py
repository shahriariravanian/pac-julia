"""pacrunner.PacLibrary / MadLibrary and their Runtime, against the stock pyAmpliCol library.

    PACRUNNER_MADGRAPH7=~/MadGraph7 PACRUNNER_TEST_MG7_OUTPUT=OUTPUT python -m unittest discover -s test/python

PACRUNNER_TEST_MG7_OUTPUT: an mg7 output whose libraries were built (lib/pacrunner/<name>, the CPU
ones: test/mg7_run.py prepare) and whose stock library exists (lib/libamplicol_<name>.so: after one
generate_events). PACRUNNER_TEST_CUDA_OUTPUT: one with CUDA provider libraries
(lib/pacrunner-cuda_provider/<name>; MadSpace's CUDA build, CuPy), run in a separate process (one
MadSpace build per process). Skipped without them."""

import glob
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

import pacrunner

OUTPUT = os.environ.get("PACRUNNER_TEST_MG7_OUTPUT")
CUDA_OUTPUT = os.environ.get("PACRUNNER_TEST_CUDA_OUTPUT")


def momenta(masses, events, rng):
    p = rng.uniform(-300.0, 300.0, size=(events, len(masses), 3))
    e = np.sqrt((p * p).sum(axis=2) + np.asarray(masses) ** 2)
    return np.concatenate([e[:, :, None], p], axis=2)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and (np.array_equal(a.view(np.uint64), b.view(np.uint64)) if a.dtype == np.float64
                                   else np.array_equal(a, b))


def library_folder(output, kind="pacrunner"):
    found = sorted(glob.glob(os.path.join(output, "lib", kind, "*", "umami_pac.json")))
    if not found:
        raise unittest.SkipTest(f"{output}: no lib/{kind}/*/umami_pac.json")
    return os.path.dirname(found[0])


def stock_runtime(lib, channel, context):
    """the stock library of `lib`'s subprocess (lib/libamplicol_<name>.so) as a pacrunner Runtime"""
    from pacrunner import _runtime
    stock = Path(lib.output) / "lib" / f"libamplicol_{lib.name}.so"
    if not stock.is_file():
        raise unittest.SkipTest(f"{stock}: no stock library (run generate_events once)")
    api = context.load_matrix_element(os.fspath(stock), os.fspath(lib.artifact), lib.parameters)
    return _runtime.matrix_element_runtime(api, lib.particle_count, channel, context, cuda=False,
                                           description="stock")


@unittest.skipUnless(OUTPUT, "PACRUNNER_TEST_MG7_OUTPUT not set")
class Library(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = pacrunner.MadLibrary(library_folder(OUTPUT))
        with open(cls.lib.table["metadata"]) as f:
            meta = json.load(f)
        cls.masses = meta["provider"]["masses"]
        cls.exported = len(meta["helicities"])

    def test_layout(self):
        lib = self.lib
        self.assertEqual(Path(lib.output), Path(OUTPUT).resolve())
        self.assertEqual(lib.channels, len(lib.table["channels"]))
        for c in range(lib.channels):
            args = lib.flavor_arguments(c)
            self.assertEqual(len(args["flavor_remap"]), lib.flavours(c))
            self.assertEqual(set(args["flavor_remap"]), {0})
            libs = lib.table["channels"][c]["libraries"]
            for f in range(lib.flavours(c)):
                self.assertEqual(libs[args["flavor_diff_xs_indices"][f]], lib.library(c, f))

    def test_runtime_is_the_stock_library(self):
        lib = self.lib
        rng = np.random.default_rng(20261116)
        for c in range(lib.channels):
            rt = lib.runtime(c, threads=4)
            stock = stock_runtime(lib, c, rt.context)
            mom = momenta(self.masses, 128, rng)
            flav = rng.integers(0, lib.flavours(c), 128)
            r = rng.uniform(0.0, 1.0, 128)
            a, b = rt(mom, flav, random_helicity=r), stock(mom, flav, random_helicity=r)
            for name in ("matrix_element", "color_index", "helicity_index"):
                self.assertTrue(same(getattr(a, name), getattr(b, name)), f"channel {c}: {name}")
            for h in (0, self.exported // 2, self.exported - 1):
                self.assertTrue(same(rt(mom, flav, helicity=h).matrix_element, stock(mom, flav, helicity=h).matrix_element),
                                f"channel {c}: helicity {h}")
            # the plain sum: the stock library sums over helicity orbits in RustiCol (rounding)
            np.testing.assert_allclose(rt(mom, flav).matrix_element, stock(mom, flav).matrix_element, rtol=1e-12)
            self.assertIsNone(rt(mom, flav).color_index)

    def test_inputs(self):
        rt = self.lib.runtime(0, threads=4)
        mom = momenta(self.masses, 8, np.random.default_rng(1))
        self.assertTrue(same(rt(mom, 0).matrix_element, rt(mom, np.zeros(8, np.int64)).matrix_element))
        with self.assertRaises(ValueError):
            rt(mom[:, :-1], 0)                                     # a particle missing
        with self.assertRaises(ValueError):
            rt(mom, 0, random_helicity=0.5, helicity=0)            # one helicity input at most
        with self.assertRaises(ValueError):
            rt(mom, np.zeros(3, np.int32))                         # one flavour per event

    def test_load_libraries_compatibility(self):
        rt = self.lib.runtime(0, threads=4)
        old = pacrunner.load_libraries(rt.context, self.lib.folder)
        mom = momenta(self.masses, 16, np.random.default_rng(2))
        flav = np.zeros(16, np.int32)
        (me,) = old.runtime(0).call([mom, flav])
        self.assertTrue(same(np.array(me.numpy()), rt(mom, flav).matrix_element))


CUDA_SCRIPT = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[2])
import test_runtime as t, pacrunner
lib = pacrunner.MadLibrary(t.library_folder(sys.argv[1], "pacrunner-cuda_provider"))
meta = json.load(open(lib.table["metadata"]))
rng = np.random.default_rng(3)
ok = True
for c in range(lib.channels):
    rt = lib.runtime(c)
    stock = t.stock_runtime(lib, c, pacrunner._runtime.context("cpu", 4))
    mom = t.momenta(meta["provider"]["masses"], 256, rng)
    flav = rng.integers(0, lib.flavours(c), 256)
    r = rng.uniform(0.0, 1.0, 256)
    a, b = rt(mom, flav, random_helicity=r), stock(mom, flav, random_helicity=r)
    if lib.precision == "f64":
        ok = ok and all(t.same(getattr(a, n), getattr(b, n)) for n in ("matrix_element", "color_index", "helicity_index"))
    else:
        ok = ok and np.allclose(a.matrix_element, b.matrix_element, rtol=1e-4, atol=0) and t.same(a.color_index, b.color_index)
print("CUDA", lib.precision, "ok" if ok else "DIFFERENT")
"""


@unittest.skipUnless(CUDA_OUTPUT, "PACRUNNER_TEST_CUDA_OUTPUT not set")
class CudaLibrary(unittest.TestCase):
    def test_cuda_provider_runtime_is_the_stock_library(self):
        # a process of its own: MadSpace's CUDA build (configure(cuda=True)), CuPy
        env = dict(os.environ)
        r = subprocess.run([sys.executable, "-c", CUDA_SCRIPT, CUDA_OUTPUT, os.path.dirname(__file__)],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout, r.stdout + r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
