"""pacrunner._env: discovery and configuration, without Julia or MadSpace.

    python -m unittest discover -s test/python      (or pytest test/python)

The MadSpace tests need a MadGraph7 checkout (PACRUNNER_MADGRAPH7); they are skipped without one."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pacrunner
from pacrunner import _env


class Repository(unittest.TestCase):
    def test_repository_is_the_julia_project(self):
        root = pacrunner.repository()
        self.assertTrue((root / "Project.toml").is_file())
        self.assertTrue((root / "src" / "PacRunner.jl").is_file())
        self.assertEqual(root, Path(pacrunner.__file__).resolve().parent.parent)


class Julia(unittest.TestCase):
    def test_julia_from_the_environment(self):
        with mock.patch.dict(os.environ, {"PACRUNNER_JULIA": sys.executable}), \
                mock.patch.object(_env.CONFIG, "julia", None):
            self.assertEqual(_env.julia(), sys.executable)

    def test_no_julia_is_a_clear_error(self):
        env = {k: v for k, v in os.environ.items() if k != "PACRUNNER_JULIA"}
        env["PATH"] = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(_env.CONFIG, "julia", None):
            with self.assertRaises(pacrunner.PacRunnerError) as e:
                _env.julia()
            self.assertIn("install it", str(e.exception))


class MadGraph7(unittest.TestCase):
    def test_not_a_checkout(self):
        with mock.patch.object(_env.CONFIG, "madgraph7", tempfile.mkdtemp()):
            with self.assertRaises(pacrunner.PacRunnerError):
                _env.madgraph7()

    def test_none_configured(self):
        env = {k: v for k, v in os.environ.items() if k != "PACRUNNER_MADGRAPH7"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(_env.CONFIG, "madgraph7", None):
            self.assertIsNone(_env.madgraph7(required=False))
            with self.assertRaises(pacrunner.PacRunnerError):
                _env.madgraph7()

    def test_madspace_build_choice(self):
        mg7 = Path(tempfile.mkdtemp())
        (mg7 / "bin").mkdir()
        (mg7 / "bin" / "madgraph").write_text("")
        (mg7 / "madspace" / "install" / "madspace").mkdir(parents=True)
        with mock.patch.object(_env.CONFIG, "cuda", None):
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install")   # no CUDA build: the CPU one
            (mg7 / "madspace" / "install-cuda" / "madspace").mkdir(parents=True)
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install-cuda")   # the CUDA build by default
        with mock.patch.object(_env.CONFIG, "cuda", False):
            self.assertEqual(_env.madspace_dir(mg7), mg7 / "madspace" / "install")


MG7 = os.environ.get("PACRUNNER_MADGRAPH7")


@unittest.skipUnless(MG7, "PACRUNNER_MADGRAPH7 not set")
class MadSpace(unittest.TestCase):
    def run_python(self, code):
        # a fresh process: MadSpace is imported once per process
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    def test_import_and_late_configure(self):
        r = self.run_python(
            "import pacrunner; from pacrunner import _env\n"
            "ms = _env.import_madspace()\n"
            "print(ms.cpu_device() is not None)\n"
            "try:\n    pacrunner.configure(cuda=False)\nexcept pacrunner.PacRunnerError as e:\n    print('refused')\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["True", "refused"])


if __name__ == "__main__":
    unittest.main()
