"""MadOutput and MadArtifact: a MadGraph7 output (`generate ... [amplicol]`, `output mg7 OUTPUT`) and its
pyAmpliCol subprocesses.

    out = MadOutput("OUTPUT")
    out.names                                   # ["A1_pp_zj_p0", "A2_pp_zjj_p0"]: one per stock library
    m = out.artifact("A1_pp_zj_p0")             # or MadArtifact("OUTPUT") when it has one
    m.processes, m.channels, m.flavours(0), m.parameters, m.artifact   # a PacArtifact
    stock = m.runtime(channel=0)                # lib/libamplicol_<name>.so at the param card's parameters
    lib = m.compile(device="cpu")               # -> MadLibrary in OUTPUT/lib/pacrunner[-device[-precision]]/<name>
    lib.run()                                   # generate_events on these libraries (the others stock)
    out.run(device="cpu")                       # every subprocess compiled (when needed) and run on PacRunner

A MadArtifact is one stock library of the output (`me_path` in SubProcesses/subprocesses.json): one
artifact's UMAMI provider, with one subprocesses.json entry per UMAMI channel."""

import json
import re
from pathlib import Path

from . import _env
from ._artifacts import PacArtifact


def read_param_card(path):
    """{(block, (code, ...)): value} of an SLHA param card: BLOCK sections and DECAY lines, block
    names lower case (as MadGraph7's ParamCard; a "width" block is "decay")"""
    values, block = {}, None
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            head = fields[0].lower()
            if head == "block":
                block = fields[1].lower() if len(fields) > 1 else None
                block = "decay" if block == "width" else block
                continue
            if head == "decay":
                block = None
                if len(fields) >= 3:
                    values[("decay", (int(fields[1]),))] = float(fields[2])
                continue
            if block is None:
                continue
            try:
                codes, value = tuple(int(x) for x in fields[:-1]), float(fields[-1].replace("d", "e").replace("D", "e"))
            except ValueError:
                continue
            values[(block, codes)] = value
    return values


def card_parameters(param_card, references):
    """{name: value} of the param card for a subprocess's parameter references ({name: [block, code]}),
    as MadGraph7's amplicol_parameters gives them to the libraries (missing ones left out)"""
    card = read_param_card(param_card)
    out = {}
    for name, (block, code) in references.items():
        value = card.get((block.lower(), tuple(code)))
        if value is not None:
            out[name] = value
    return out


def library_folder_name(device="cpu", precision="f64"):
    """lib/<this>/<name>: pacrunner (cpu, f64), pacrunner-<device>[-<precision>]"""
    if device == "cpu" and precision == "f64":
        return "pacrunner"
    return f"pacrunner-{device}" + ("" if precision == "f64" else f"-{precision}")


class MadOutput:
    """A MadGraph7 output with pyAmpliCol subprocesses.

    path         the output folder
    madgraph7    the MadGraph7 checkout it was made with
    run_card, param_card   its cards (Cards/run_card.toml, Cards/param_card.dat)
    names        the pyAmpliCol stock libraries' names (lib/libamplicol_<name>.so), one MadArtifact each"""

    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        output = _env.mg7_output_of(path)
        if output is None or not (output / "SubProcesses" / "subprocesses.json").is_file():
            raise _env.PacRunnerError(f"{path}: not a MadGraph7 output (no bin/generate_events and "
                                      "SubProcesses/subprocesses.json): `output mg7 OUTPUT` in MadGraph7 makes one")
        self.path = output
        self.madgraph7 = _env.madgraph7_of_output(output)
        self.run_card = output / "Cards" / "run_card.toml"
        self.param_card = output / "Cards" / "param_card.dat"
        with open(output / "SubProcesses" / "subprocesses.json") as f:
            entries = json.load(f)
        entries = entries.get("subprocesses", entries) if isinstance(entries, dict) else entries
        self.entries = {}
        for e in entries:
            if e.get("backend") == "amplicol":
                self.entries.setdefault(self._name(e["me_path"]), []).append(e)
        if not self.entries:
            raise _env.PacRunnerError(f"{output}: no pyAmpliCol subprocess (generate with [amplicol])")
        self.names = sorted(self.entries)
        self._artifacts = {}

    @staticmethod
    def _name(me_path):
        m = re.fullmatch(r"libamplicol_(.*)\.(so|dylib)", Path(me_path).name)
        if m is None:
            raise _env.PacRunnerError(f"{me_path}: not a pyAmpliCol stock library name (libamplicol_<name>.so)")
        return m.group(1)

    def __repr__(self):
        return f"<MadOutput {self.path} ({', '.join(self.names)})>"

    def artifact(self, name=None):
        """the MadArtifact `name` (default: the only one)"""
        if name is None:
            if len(self.names) != 1:
                raise _env.PacRunnerError(f"{self.path}: {len(self.names)} pyAmpliCol subprocesses {self.names}: pick "
                                          "one by name")
            name = self.names[0]
        if name not in self.entries:
            raise _env.PacRunnerError(f"{self.path}: no pyAmpliCol subprocess {name!r} (there are {self.names})")
        if name not in self._artifacts:
            self._artifacts[name] = MadArtifact(self, name)
        return self._artifacts[name]

    def artifacts(self):
        return [self.artifact(n) for n in self.names]

    def compile(self, device="cpu", precision="f64", interpreter=False, force=False):
        """every subprocess's libraries (MadArtifact.compile): [MadLibrary]"""
        return [a.compile(device, precision, interpreter=interpreter, force=force) for a in self.artifacts()]

    def run(self, libraries=None, device="cpu", precision="f64", args=None, in_process=False, log=None):
        """generate_events on PacRunner's libraries: `libraries` (MadLibrary's of this output; default:
        every subprocess compiled for `device` and `precision`, built when needed); see MadLibrary.run"""
        from . import _mgrun
        if libraries is None:
            libraries = self.compile(device, precision)
        for lib in libraries:
            if lib.output != self.path:
                raise _env.PacRunnerError(f"{lib.folder}: libraries of {lib.output}, not of {self.path}")
        return _mgrun.run_libraries(self.path, libraries, args=args, in_process=in_process, log=log)


class MadArtifact:
    """One pyAmpliCol subprocess (stock library) of a MadGraph7 output.

    output       its MadOutput; name: lib/libamplicol_<name>.so's <name>
    entries      its subprocesses.json entries (one per UMAMI channel)
    artifact     its PacArtifact (SubProcesses/<subprocess>/lc, the library's provider)
    subprocess   the SubProcesses folder's name
    processes, particle_count, masses, channels, flavours(channel): the artifact's
    parameters   the param card's values of the model parameters (what MadGraph7 sets)"""

    def __init__(self, output, name=None):
        if not isinstance(output, MadOutput):
            output = MadOutput(output)
        if name is None or name not in output.entries:
            name = output.artifact(name).name   # the only one, or the error
        self.output, self.name = output, name
        self.entries = sorted(output.entries[name], key=lambda e: e["umami_channel"])
        artifacts = {e["artifact"] for e in self.entries}
        if len(artifacts) != 1:
            raise _env.PacRunnerError(f"{name}: entries of several artifacts {sorted(artifacts)}")
        lc = output.path / artifacts.pop()
        # the stock library <name> is <subprocess>_<provider>: the artifact's provider of that id
        ids = list(PacArtifact.provider_metadata(lc))
        match = ids if len(ids) == 1 else [p for p in ids if name.endswith("_" + p)]
        if len(match) != 1:
            raise _env.PacRunnerError(f"{lc}: no single UMAMI provider for {name} among {ids}")
        self.artifact = PacArtifact(lc, provider=match[0])
        self.subprocess = Path(self.entries[0]["path"]).name
        self.stock_library_path = output.path / self.entries[0]["me_path"]
        if [e["umami_channel"] for e in self.entries] != list(range(self.artifact.channels)):
            raise _env.PacRunnerError(f"{name}: subprocesses.json has UMAMI channels "
                                      f"{[e['umami_channel'] for e in self.entries]}, the artifact {self.artifact.channels}")

    def __repr__(self):
        return f"<MadArtifact {self.name} of {self.output.path} ({self.artifact.channels} channels)>"

    @property
    def processes(self):
        return self.artifact.processes

    @property
    def particle_count(self):
        return self.artifact.particle_count

    @property
    def masses(self):
        return self.artifact.masses

    @property
    def channels(self):
        return self.artifact.channels

    def flavours(self, channel):
        return self.artifact.flavours(channel)

    @property
    def parameters(self):
        """the param card's values of the subprocess's model parameters (read each time: the card may change)"""
        return card_parameters(self.output.param_card, self.entries[0]["parameters"])

    def runtime(self, channel=0, threads=None):
        """the output's stock library (lib/libamplicol_<name>.so) at the param card's parameters, as a
        pacrunner.Runtime on the CPU. MadGraph7 builds the library at the output's first run
        (bin/generate_events)."""
        from . import _runtime
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        if not self.stock_library_path.is_file():
            raise _env.PacRunnerError(f"{self.stock_library_path}: not built yet; MadGraph7 builds it at the output's "
                                      "first run (bin/generate_events -f in the output)")
        context = _runtime.context("cpu", threads, self.output.madgraph7)
        key = (id(context), json.dumps(self.parameters, sort_keys=True))
        apis = self.artifact._stock_apis
        if key not in apis:
            apis[key] = context.load_matrix_element(str(self.stock_library_path), str(self.artifact.path),
                                                    self.parameters)
        return _runtime.matrix_element_runtime(apis[key], self.particle_count, channel, context, cuda=False,
                                               description=f"stock {self.name} channel {channel}")

    def library_folder(self, device="cpu", precision="f64"):
        return self.output.path / "lib" / library_folder_name(device, precision) / self.name

    def compile(self, device="cpu", precision="f64", interpreter=False, force=False):
        """PacRunner's libraries of this subprocess, exported at the param card's parameters (the values
        MadGraph7 sets: a library refuses others), in OUTPUT/lib/pacrunner[-device[-precision]]/<name>;
        returns the MadLibrary. Skipped when built already for the same card (the stamp)."""
        from ._libraries import MadLibrary
        self.artifact.compile(self.library_folder(device, precision), device, precision, parameters=self.parameters,
                              interpreter=interpreter, force=force)
        return MadLibrary(self.library_folder(device, precision))

    def library(self, device="cpu", precision="f64"):
        """the MadLibrary compiled already (no build)"""
        from ._libraries import MadLibrary
        folder = self.library_folder(device, precision)
        if not (folder / "umami_pac.json").is_file():
            raise _env.PacRunnerError(f"{folder}: not compiled (MadArtifact.compile)")
        return MadLibrary(folder)
