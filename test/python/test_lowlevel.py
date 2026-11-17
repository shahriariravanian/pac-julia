"""pacrunner.lowlevel: PacRunner.jl's exports through juliacall (starts Julia: a minute the first time).

    python -m unittest discover -s test/python      (or pytest test/python)

Skipped without Julia (PACRUNNER_JULIA, or julia on the PATH)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pacrunner

ROOT = pacrunner.repository()
EXPORT = ROOT / "test" / "data" / "export" / "zggg_union.json"
HAVE_JULIA = bool(os.environ.get("PACRUNNER_JULIA") or shutil.which("julia"))


@unittest.skipUnless(HAVE_JULIA, "no Julia")
class LowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pacrunner import lowlevel
        cls.pr = lowlevel
        cls.jl = pacrunner._julia.main()

    def test_every_export_but_v1_is_wrapped(self):
        exported = {str(n) for n in self.jl.seval("names(PacRunner)")} - {"PacRunner"}
        self.assertEqual(set(self.pr.EXPORTS), exported - set(self.pr.V1),
                         "lowlevel.EXPORTS differs from PacRunner.jl's exports (but v1)")
        self.assertTrue(set(self.pr.V1) <= exported)
        for name in self.pr.EXPORTS:
            self.assertTrue(callable(getattr(self.pr, name)), name)

    def test_v1_is_not_reexported(self):
        with self.assertRaises(AttributeError):
            self.pr.read_main
        self.assertIsNotNone(self.pr.jl.read_main)   # still reachable through the module

    def test_strings_become_symbols_and_paths_are_accepted(self):
        programs = self.pr.load_recurrence_export(Path(EXPORT), kernels="c")   # PathLike, kernels a Symbol
        self.assertEqual(len(programs), 1)
        with self.assertRaises(Exception):   # a Julia error for a wrong value, not a type error from Python
            self.pr.load_recurrence_export(EXPORT, kernels="nonsense")

    def test_evaluate_is_the_export_reference_bit_for_bit(self):
        programs = self.pr.load_recurrence_export(EXPORT)
        refs = self.pr.recurrence_references(EXPORT)
        program, (momenta, values) = programs[0], refs[0]
        for point, want in zip(momenta, values):
            got = self.pr.evaluate(program, self.pr.LArena(program), point)
            self.assertEqual(list(got), list(want))

    def test_compile_writes_the_folder(self):
        program = self.pr.load_recurrence_export(EXPORT)[0]
        out = tempfile.mkdtemp()
        self.pr.compile(out, self.pr.optimize(program), interpreter=True)
        for f in ("runner.c", "runner.h", "uirvm.c", "Makefile"):
            self.assertTrue(os.path.isfile(os.path.join(out, f)), f)

    def test_doc(self):
        self.assertIn("umami_libraries", self.pr.doc("umami_libraries"))


if __name__ == "__main__":
    unittest.main()
