"""Output and SubProcess: a MadGraph7 output (`generate ... [amplicol]`, `output mg7 OUTPUT`) and its
pyAmpliCol subprocesses.

    out = Output("OUTPUT")
    out.names                                   # ["A1_pp_zj", "A2_pp_zjj"]: the subprocesses (SubProcesses/<name>)
    s = out.subprocess("A1_pp_zj")              # or SubProcess("OUTPUT") when it has one
    s.processes, s.channels, s.flavours(0), s.parameters, s.artifact    # an Artifact
    stock = s.runtime(channel=0)                # lib/libamplicol_<name>_<provider>.so at the param card's parameters
    provider = s.compile(device="cpu")          # -> Provider in OUTPUT/lib/pacrunner[-device[-precision]]/<name>_<provider>
    provider.run()                              # generate_events on it (the other subprocesses stock)
    out.run(device="cpu")                       # every subprocess compiled (when needed) and run on PacRunner

Nomenclature: an output holds one or more processes (`generate`, `add process`), MadGraph splits them
into subprocesses (SubProcesses/<name>, one pyAmpliCol artifact each); an artifact's provider (`p0`, ...)
is one UMAMI matrix-element library (the stock lib/libamplicol_<name>_<provider>.so), with one
subprocesses.json entry per UMAMI channel; a Provider is PacRunner's build of it."""

import json
import re
from pathlib import Path

from . import _env
from ._artifacts import Artifact


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


class Output:
    """A MadGraph7 output with pyAmpliCol subprocesses.

    path         the output folder
    madgraph7    the MadGraph7 checkout it was made with
    run_card, param_card   its cards (Cards/run_card.toml, Cards/param_card.dat)
    names        its pyAmpliCol subprocesses (SubProcesses/<name>), one SubProcess each"""

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
        # subprocess folder -> stock library name (<name>_<provider>) -> its entries (one per UMAMI channel)
        self.entries = {}
        for e in entries:
            if e.get("backend") == "amplicol":
                self.entries.setdefault(Path(e["path"]).name, {}).setdefault(self._library_name(e["me_path"]), []).append(e)
        if not self.entries:
            raise _env.PacRunnerError(f"{output}: no pyAmpliCol subprocess (generate with [amplicol])")
        self.names = sorted(self.entries)
        self._subprocesses = {}

    @staticmethod
    def _library_name(me_path):
        m = re.fullmatch(r"libamplicol_(.*)\.(so|dylib)", Path(me_path).name)
        if m is None:
            raise _env.PacRunnerError(f"{me_path}: not a pyAmpliCol stock library name (libamplicol_<name>.so)")
        return m.group(1)

    def __repr__(self):
        return f"<Output {self.path} ({', '.join(self.names)})>"

    def _resolve(self, name):
        """(subprocess, stock library name or None) for a subprocess name or a stock library name"""
        if name is None:
            if len(self.names) != 1:
                raise _env.PacRunnerError(f"{self.path}: {len(self.names)} pyAmpliCol subprocesses {self.names}: pick "
                                          "one by name")
            return self.names[0], None
        if name in self.entries:
            return name, None
        for sub, libs in self.entries.items():
            if name in libs:
                return sub, name
        raise _env.PacRunnerError(f"{self.path}: no pyAmpliCol subprocess {name!r} (there are {self.names})")

    def subprocess(self, name=None, provider=None):
        """the SubProcess `name` (SubProcesses/<name>; default: the only one), its provider `provider`
        (default: the only one); a stock library's name (<name>_<provider>) names both"""
        key = (name, provider)
        if key not in self._subprocesses:
            self._subprocesses[key] = SubProcess(self, name, provider)
        return self._subprocesses[key]

    def subprocesses(self):
        """every subprocess, with every provider: [SubProcess]"""
        return [self.subprocess(lib) for n in self.names for lib in sorted(self.entries[n])]

    def compile(self, device="cpu", precision="f64", interpreter=False, force=False):
        """every subprocess's Provider (SubProcess.compile): [Provider]"""
        return [s.compile(device, precision, interpreter=interpreter, force=force) for s in self.subprocesses()]

    def run(self, providers=None, device="cpu", precision="f64", args=None, in_process=False, log=None):
        """generate_events on PacRunner's providers: `providers` (of this output; default: every
        subprocess's, compiled for `device` and `precision` when needed); see Provider.run"""
        from . import _mgrun
        if providers is None:
            providers = self.compile(device, precision)
        for p in providers:
            if p.output != self.path:
                raise _env.PacRunnerError(f"{p.folder}: a provider of {p.output}, not of {self.path}")
        return _mgrun.run_libraries(self.path, providers, args=args, in_process=in_process, log=log)

    # the names before 0.2 (one release)
    def artifact(self, name=None):
        return self.subprocess(name)

    def artifacts(self):
        return self.subprocesses()


class SubProcess:
    """One pyAmpliCol subprocess of a MadGraph7 output (SubProcesses/<name>), with one of its artifact's
    providers.

    output         its Output; name: the SubProcesses folder's name
    provider       the provider's id (p0, ...); providers: all of them
    library_name   the stock library's name: lib/libamplicol_<library_name>.so (<name>_<provider>)
    entries        its subprocesses.json entries (one per UMAMI channel)
    artifact       its Artifact (SubProcesses/<name>/lc, with the provider)
    processes, particle_count, masses, channels, flavours(channel): the artifact's
    parameters     the param card's values of the model parameters (what MadGraph7 sets)"""

    def __init__(self, output, name=None, provider=None):
        if not isinstance(output, Output):
            output = Output(output)
        (name, library_name) = output._resolve(name)
        libs = output.entries[name]
        if library_name is None:
            if provider is not None:
                library_name = next((l for l in libs if l.endswith("_" + str(provider))), None)
                if library_name is None:
                    raise _env.PacRunnerError(f"{name}: no provider {provider!r} (stock libraries {sorted(libs)})")
            elif len(libs) == 1:
                library_name = next(iter(libs))
            else:
                raise _env.PacRunnerError(f"{name}: {len(libs)} providers ({sorted(libs)}): pick one, provider=...")
        self.output, self.name, self.library_name = output, name, library_name
        self.entries = sorted(libs[library_name], key=lambda e: e["umami_channel"])
        artifacts = {e["artifact"] for e in self.entries}
        if len(artifacts) != 1:
            raise _env.PacRunnerError(f"{library_name}: entries of several artifacts {sorted(artifacts)}")
        lc = output.path / artifacts.pop()
        # the stock library <name>_<provider>: the artifact's provider of that id
        ids = list(Artifact.provider_metadata(lc))
        self.providers = ids
        match = ids if len(ids) == 1 else [p for p in ids if library_name.endswith("_" + p)]
        if len(match) != 1:
            raise _env.PacRunnerError(f"{lc}: no single UMAMI provider for {library_name} among {ids}")
        self.provider = match[0]
        self.artifact = Artifact(lc, provider=self.provider)
        self.stock_library_path = output.path / self.entries[0]["me_path"]
        if [e["umami_channel"] for e in self.entries] != list(range(self.artifact.channels)):
            raise _env.PacRunnerError(f"{library_name}: subprocesses.json has UMAMI channels "
                                      f"{[e['umami_channel'] for e in self.entries]}, the artifact {self.artifact.channels}")

    def __repr__(self):
        return (f"<SubProcess {self.name} of {self.output.path} (provider {self.provider}, "
                f"{self.artifact.channels} channels)>")

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

    def stock(self):
        """the stock provider (lib/libamplicol_<library_name>.so, which MadGraph7 builds at the output's
        first run) as a Provider (stock=True), at the param card's parameters, called with MadGraph7's
        flavours (each channel's entry's flavour indices)"""
        from ._libraries import Provider
        if not self.stock_library_path.is_file():
            raise _env.PacRunnerError(f"{self.stock_library_path}: not built yet; MadGraph7 builds it at the output's "
                                      "first run (bin/generate_events -f in the output)")
        params = self.parameters
        key = json.dumps(params, sort_keys=True)
        if getattr(self, "_stock_key", None) != key:
            remap = [[f["index"] for f in e["flavors"]] for e in self.entries]
            self._stock_provider = Provider._stock(self.stock_library_path, self.artifact.path, self.particle_count,
                                                   [len(r) for r in remap], params, name=self.library_name,
                                                   subprocess=self, remap=remap)
            self._stock_key = key
        return self._stock_provider

    def runtime(self, channel=0, threads=None):
        """the stock provider's `channel` as a pacrunner.Runtime (on the CPU): stock().runtime"""
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        return self.stock().runtime(channel, threads)

    def provider_folder(self, device="cpu", precision="f64"):
        """where compile() puts the Provider: OUTPUT/lib/pacrunner[-device[-precision]]/<library_name>"""
        return self.output.path / "lib" / library_folder_name(device, precision) / self.library_name

    def compile(self, device="cpu", precision="f64", interpreter=False, force=False):
        """PacRunner's build of the provider, exported at the param card's parameters (the values
        MadGraph7 sets: a library refuses others), in provider_folder(); returns the Provider. Skipped
        when built already for the same card (the stamp)."""
        from ._libraries import Provider
        self.artifact.compile(self.provider_folder(device, precision), device, precision, parameters=self.parameters,
                              interpreter=interpreter, force=force)
        return Provider(self.provider_folder(device, precision))

    def compiled(self, device="cpu", precision="f64"):
        """the Provider compiled already (no build)"""
        from ._libraries import Provider
        folder = self.provider_folder(device, precision)
        if not (folder / "umami_pac.json").is_file():
            raise _env.PacRunnerError(f"{folder}: not compiled (SubProcess.compile)")
        return Provider(folder)

    # the names before 0.2 (one release)
    def library(self, device="cpu", precision="f64"):
        return self.compiled(device, precision)

    def library_folder(self, device="cpu", precision="f64"):
        return self.provider_folder(device, precision)


# the names before 0.2 (one release)
MadOutput = Output
MadArtifact = SubProcess
