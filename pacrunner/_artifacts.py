"""PacArtifact: a pyAmpliCol artifact folder (`pyamplicol generate`, or the one MadGraph7's [amplicol]
writes in an mg7 output).

    a = PacArtifact("artifacts/zgg")
    a.id, a.producer_version, a.processes, a.particle_count, a.masses, a.channels, a.flavours(0), a.parameters
    process, momenta = a.validation_points()[0]
    stock = a.runtime(channel=0)                                  # the stock UMAMI library (built once)
    lib = a.compile(device="cpu")                                 # export + umami_libraries -> PacLibrary
    stock(momenta[None], 0).matrix_element, lib.runtime(0)(momenta[None], 0).matrix_element

The export and the stock library's build use the pyAmpliCol installation that produced the artifact
(pacrunner._pyamplicol: found by its revision, or configure(pyamplicol=...)). An artifact may hold
several UMAMI providers (API/umami/metadata.json an index): pick one with `provider=`."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import _env, _pyamplicol

STAMP = "pacrunner-build.json"


def _read(path):
    with open(path) as f:
        return json.load(f)


def _git_revision(folder):
    """this repository's revision, '+' when its Julia or C sources differ from it (None without git)"""
    try:
        head = subprocess.run(["git", "-C", os.fspath(folder), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", os.fspath(folder), "status", "--porcelain", "--", "src", "c"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


class PacArtifact:
    """A pyAmpliCol artifact, one of its UMAMI providers.

    path             the artifact folder
    id               its artifact id
    producer         artifact.json's producer (version, git_revision, ...)
    processes        [{"id", "expression", ...}]
    providers        the UMAMI providers' ids; provider: the one used here
    metadata         the provider's UMAMI metadata (API/umami/...)
    particle_count, masses, pdg_ids, color_accuracy, helicity_count, incoming_count, supports_alpha_s
    channels         the provider's UMAMI channels' count; flavours(channel): a channel's flavours
    parameters       the model's external parameters at their defaults ({name: real part})"""

    def __init__(self, path, provider=None):
        self.path = Path(path).expanduser().resolve()
        manifest = self.path / "artifact.json"
        if not manifest.is_file():
            raise _env.PacRunnerError(f"{self.path}: not a pyAmpliCol artifact (no artifact.json)")
        self.manifest = _read(manifest)
        self.id = self.manifest["artifact_id"]
        self.producer = self.manifest.get("producer", {})
        self.processes = self.manifest.get("processes", [])
        entries = self.provider_metadata(self.path)
        self.providers = list(entries)
        if provider is None:
            if len(entries) != 1:
                raise _env.PacRunnerError(f"{self.path}: {len(entries)} UMAMI providers {self.providers}: pick one, "
                                          "PacArtifact(path, provider=...)")
            provider = self.providers[0]
        if str(provider) not in entries:
            raise _env.PacRunnerError(f"{self.path}: no UMAMI provider {provider!r} (there are {self.providers})")
        self.provider = str(provider)
        self.metadata_path = entries[self.provider]
        self.metadata = _read(self.metadata_path)
        p = self.metadata["provider"]
        self.particle_count = int(p["particle_count"])
        self.masses = [float(m) for m in p["masses"]]
        self.color_accuracy = p.get("color_accuracy")
        self.helicity_count = p.get("helicity_count")
        self.incoming_count = p.get("incoming_count", 2)
        self.supports_alpha_s = bool(p.get("supports_alpha_s", False))
        self.pdg_ids = self.metadata.get("pdg_ids")
        self.channels = len(self.metadata["channels"])
        params = self.path / "model" / "parameters.json"
        self.parameters = {k: float(v[0] if isinstance(v, list) else v) for k, v in _read(params)["parameters"].items()} \
            if params.is_file() else {}
        self._stock_apis = {}

    @staticmethod
    def provider_metadata(path):
        """{provider id: its UMAMI metadata path} of the artifact at `path`"""
        umami = Path(path) / "API" / "umami"
        if not (umami / "metadata.json").is_file():
            raise _env.PacRunnerError(f"{path}: no UMAMI bundle (API/umami/metadata.json): generate with the API "
                                      "bundle (pyamplicol generate --emit-api-bundle)")
        index = _read(umami / "metadata.json")
        if index.get("kind") == "pyamplicol-umami-provider-index":
            return {str(p.get("id", p.get("provider"))): umami / p["metadata"] for p in index["providers"]}
        return {str(index["provider"].get("id")): umami / "metadata.json"}

    def __repr__(self):
        names = ", ".join(p.get("expression") or p["id"] for p in self.processes[:3])
        return f"<PacArtifact {self.path} ({names}; provider {self.provider}, {self.color_accuracy}, {self.channels} channels)>"

    @property
    def producer_version(self):
        return self.producer.get("version")

    @property
    def producer_revision(self):
        return self.producer.get("git_revision")

    def flavours(self, channel):
        """the provider's flavours of `channel`"""
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        return len(self.metadata["channels"][channel]["processes"])

    def validation_points(self):
        """the saved phase-space points (API/validation_points.dat): [(process id, momenta[particle, 4])]"""
        path = self.path / "API" / "validation_points.dat"
        points = []
        with open(path) as f:
            if f.readline().strip() != "RUSTICOL_VALIDATION_POINTS_V1":
                raise _env.PacRunnerError(f"{path}: not a RUSTICOL_VALIDATION_POINTS_V1 file")
            for line in f:
                fields = line.split()
                if fields:
                    n = int(fields[1])
                    points.append((fields[0], np.array([float(x) for x in fields[2:2 + 4 * n]]).reshape(n, 4)))
        return points

    # ---- pyAmpliCol

    def pyamplicol(self, strict=True):
        """the pyAmpliCol installation that produced the artifact (pacrunner._pyamplicol)"""
        return _pyamplicol.producer_of(self.path / "artifact.json", near=self.path, strict=strict)

    # ---- the stock library

    @property
    def stock_library_path(self):
        """the stock UMAMI library of the provider (the bundle's Makefile puts it in the sibling
        .pyamplicol-api-build/<artifact>/umami)"""
        build = self.path.parent / ".pyamplicol-api-build" / self.path.name / "umami"
        name = "libumami.so" if len(self.providers) == 1 else f"libumami_{self.provider}.so"
        for suffix in (".so", ".dylib"):
            candidate = build / (Path(name).stem + suffix)
            if candidate.is_file():
                return candidate
        return build / name

    def build_stock_library(self, force=False):
        """the stock library built (`make` in API/umami, the producer's rusticol-config); returns its path"""
        lib = self.stock_library_path
        if lib.is_file() and not force:
            return lib
        producer = self.pyamplicol()
        env = producer.env()
        env["RUSTICOL_CONFIG"] = producer.rusticol_config
        r = subprocess.run(["make", "-s", "-C", os.fspath(self.path / "API" / "umami")], env=env, capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise _env.PacRunnerError(f"{self.path}: building the stock UMAMI library failed:\n{(r.stderr or r.stdout)[-3000:]}")
        lib = self.stock_library_path
        if not lib.is_file():
            raise _env.PacRunnerError(f"{self.path}: make built no {lib}")
        return lib

    def runtime(self, channel=0, threads=None, parameters=None):
        """the stock UMAMI library's matrix element of `channel`: a pacrunner.Runtime (on the CPU; the
        model at `parameters` over the artifact's defaults). Builds the library once."""
        from . import _runtime
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        lib = self.build_stock_library()
        context = _runtime.context("cpu", threads, _env.madgraph7_for(self.path, required=False))
        key = (id(context), json.dumps(parameters or {}, sort_keys=True))
        if key not in self._stock_apis:
            self._stock_apis[key] = context.load_matrix_element(os.fspath(lib), os.fspath(self.path),
                                                                dict(parameters or {}))
        return _runtime.matrix_element_runtime(self._stock_apis[key], self.particle_count, channel, context, cuda=False,
                                               description=f"stock {self.path.name} channel {channel}")

    # ---- the export and PacRunner's libraries

    def export(self, path=None, parameters=None):
        """the recurrence export (`pyamplicol export-recurrence`, with the producing pyAmpliCol) to `path`
        (default: export.json in the default library folder), the model at `parameters` (default: the
        artifact's); returns the path"""
        path = Path(path) if path is not None else self.default_folder() / "export.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        args = ["export-recurrence", self.path, "-o", path]
        if parameters:
            params = path.with_name(path.stem + ".parameters.json")
            params.write_text(json.dumps(dict(parameters), indent=1))
            args += ["--model-parameters", params]
        self.pyamplicol().run(*args)
        return path

    def default_folder(self, device="cpu", precision="f64"):
        """where compile puts the libraries by default: <parent>/.pacrunner-build/<artifact>/<provider>-<device>[-<precision>]"""
        name = f"{self.provider}-{device}" + ("" if precision == "f64" else f"-{precision}")
        return self.path.parent / ".pacrunner-build" / self.path.name / name

    def compile(self, folder=None, device="cpu", precision="f64", parameters=None, interpreter=False, force=False):
        """PacRunner's UMAMI libraries of this provider: the export (export()), then umami_libraries
        (Julia) into `folder` (default: default_folder()); returns the PacLibrary. Skipped when `folder`
        holds libraries of the same artifact, provider, parameters, device, precision and pac-julia
        revision (its pacrunner-build.json), unless `force`. device: "cpu", "cuda" (the GPU, a CPU
        provider), "cuda_provider" (MadSpace's GPU runtime); precision "f64" or "mixed" (the GPU)."""
        from . import lowlevel
        from ._libraries import PacLibrary
        folder = Path(folder) if folder is not None else self.default_folder(device, precision)
        stamp = {"artifact_id": self.id, "provider": self.provider, "device": device, "precision": precision,
                 "interpreter": bool(interpreter), "parameters": dict(sorted((parameters or {}).items())),
                 "pac_julia": _git_revision(_env.repository())}
        if not force and (folder / "umami_pac.json").is_file() and (folder / STAMP).is_file():
            if _read(folder / STAMP) == stamp and stamp["pac_julia"] is not None and not stamp["pac_julia"].endswith("+"):
                return PacLibrary(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / STAMP).unlink(missing_ok=True)
        export = self.export(folder / "export.json", parameters)
        lowlevel.umami_libraries(export, self.metadata_path, folder, device=device, precision=precision,
                                 interpreter=interpreter, quiet=True)
        (folder / STAMP).write_text(json.dumps(stamp, indent=1))
        return PacLibrary(folder)
