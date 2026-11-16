"""An end-to-end MadGraph7 run (event generation, the cross section) on PacRunner's UMAMI libraries.

Given a pac-julia folder (the output of PacRunner's umami_libraries) and the pyAmpliCol artifact it
was built from, `run` starts MadGraph7's generate_events on the mg7 output that holds the artifact,
with that subprocess's stock pyAmpliCol library replaced by the PacRunner libraries: one MadSpace
matrix element per library, MadSpace splitting each batch by flavour (every library sees flavour 0).
MadGraph7 does the rest as usual (phase space, PDFs, scales, cuts, the survey, unweighting, the
event files), and `run` returns the cross section:

    from pacrunner._madgraph import run              # MadLibrary.run (to come) wraps it
    result = run("OUTPUT/lib/pacrunner/A1_pp_jjj_p0", "OUTPUT/SubProcesses/A1_pp_jjj/lc")
    print(result["cross_section"], "+-", result["error"], "pb;", result["events"])

or from the shell, with the python of the mg7 output (the shebang of OUTPUT/bin/generate_events):

    python -m pacrunner._madgraph LIBRARIES ARTIFACT [generate_events arguments, default -f]

(the repository on PYTHONPATH when pacrunner is not installed in that python)

Several (libraries, artifact) pairs of one output: run([(LIBRARIES, ARTIFACT), ...]); a subprocess
without a pair keeps its stock library.

What it needs:
  - the artifact inside an mg7 output (OUTPUT/SubProcesses/<subprocess>/lc, as MadGraph7's
    `output mg7` writes it with [amplicol]): the run uses that output's cards and phase space;
  - libraries exported with the output's model parameters (the param card: `pyamplicol
    export-recurrence ... --model-parameters`; test/mg7_run.py's prepare does it): a library refuses
    any other value MadGraph7 sets;
  - the device the libraries were built for (umami_pac.json's "device"): cpu, cuda and metal
    libraries are CPU providers (run.device = ["cpu"]); cuda_provider libraries need run.device =
    ["cuda"], color_accuracy = "lc" and MadSpace's CUDA build in MADGRAPH7/madspace/install-cuda;
  - several flavours per UMAMI channel: color_accuracy = "lc" (the full-colour ratio pass receives
    the remapped flavour).

The run is in this process (generate_events' main, with the subprocess class replaced), so call it
once per process."""

import glob
import json
import os
import re
import sys


def mg5_dir(output):
    """MadGraph7's folder, as the output's bin/generate_events appends it to sys.path"""
    text = open(os.path.join(output, "bin", "generate_events")).read()
    return re.search(r"sys\.path\.append\('([^']*)'\)", text).group(1)


def mg7_output(artifact):
    """the mg7 output holding `artifact` (the first parent with bin/generate_events)"""
    d = os.path.abspath(artifact)
    while d != os.path.dirname(d):
        d = os.path.dirname(d)
        if os.path.isfile(os.path.join(d, "bin", "generate_events")):
            return d
    raise ValueError(f"{artifact}: not inside an mg7 output (no bin/generate_events above it); the run needs "
                     "MadGraph7's output of the process (`generate ... [amplicol]`, `output mg7 OUTPUT`)")


def read_table(libraries):
    """umami_pac.json of a pac-julia folder (or the file itself), with its folder"""
    path = libraries if libraries.endswith(".json") else os.path.join(libraries, "umami_pac.json")
    with open(path) as f:
        table = json.load(f)
    if table.get("kind") != "pacrunner-umami-libraries":
        raise ValueError(f"{path}: not the output of PacRunner's umami_libraries")
    return table, os.path.dirname(os.path.abspath(path))


def provider_id(metadata_path):
    with open(metadata_path) as f:
        return json.load(f)["provider"].get("id")


class Pair:
    """one pac-julia folder and its artifact: the table, the provider, the device"""

    def __init__(self, libraries, artifact):
        self.table, self.dir = read_table(libraries)
        self.artifact = os.path.realpath(artifact)
        with open(os.path.join(self.artifact, "artifact.json")) as f:
            artifact_id = json.load(f)["artifact_id"]
        if artifact_id != self.table["artifact_id"]:
            raise ValueError(f"{self.dir}: built from artifact {self.table['artifact_id']}, but {artifact} is "
                             f"{artifact_id}")
        # the provider the libraries were built for, in this artifact (the table's metadata path, or
        # the artifact's own metadata.json beside the same file name if the folders moved)
        meta = self.table["metadata"]
        if not os.path.isfile(meta):
            meta = os.path.join(self.artifact, "API", "umami", os.path.basename(meta))
        self.provider = provider_id(meta)
        self.device = self.table["device"]

    def matches(self, meta):
        """whether a subprocess (its mg7 metadata) is this pair's: the same artifact and provider"""
        if os.path.realpath(meta["artifact"]) != self.artifact:
            return False
        stem = re.fullmatch(r"libamplicol_(.*)\.so", os.path.basename(meta["me_path"])).group(1)
        return self.provider is None or stem.endswith("_" + str(self.provider))


def install(pairs):
    """MadGraph7's AmplicolSubprocess replaced (in madgraph_process, which instantiates it) by
    PacRunnerSubprocess: the subprocesses of `pairs` run on their PacRunner libraries"""
    import madspace as ms
    import madgraph.mg7.madgraph_process as mp
    import madgraph.mg7.subprocess_madmatrix as smm
    from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess, amplicol_parameters

    class PacRunnerSubprocess(AmplicolSubprocess):
        def pac_pair(self):
            found = [p for p in pairs if p.matches(self.meta)]
            return found[0] if found else None

        def load_matrix_elements(self, api_path_formats, member_metas):
            """AmplicolSubprocess.load_matrix_elements (its checks and attributes, the stock ratio
            library), with the PacRunner libraries instead of the stock LC library (not loaded)"""
            pair = self.pac_pair()
            if pair is None:   # not one of ours: the stock library
                self.pac_matrix_elements = None
                return super().load_matrix_elements(api_path_formats, member_metas)
            process = self.process
            table = pair.table
            provider = pair.device == "cuda_provider"
            want = "cuda" if provider else "cpu"
            if any(d != want for d in process.device_types):
                raise RuntimeError(f"PacRunner libraries of device {pair.device} ({pair.dir}) need run.device = "
                                   f"[\"{want}\"], not {process.run_card['run']['device']}")
            lc = process.run_card["amplicol"]["color_accuracy"] == "lc"
            if provider and not lc:
                raise RuntimeError("PacRunner on a CUDA device: color_accuracy = \"lc\" only (the stock "
                                   "full-colour ratio library is a CPU one)")
            channel = table["channels"][self.meta["umami_channel"]]
            libs = channel["libraries"]
            flavors = self.meta["flavors"]
            if len(libs) > 1 and not lc:
                raise RuntimeError("PacRunner libraries: several flavours per UMAMI channel need "
                                   "color_accuracy = \"lc\" (the ratio pass gets the remapped flavour)")
            # MadGraph7's own method, seeing CPU devices (its only refusal we lift) and not loading
            # the stock LC library
            real_load, types = self.load, process.device_types
            self.load = lambda path, artifact, parameters: (
                None if path == self.meta["me_path"] else real_load(path, artifact, parameters))
            process.device_types = ["cpu"] * len(types)
            try:
                super().load_matrix_elements(api_path_formats, member_metas)
            finally:
                process.device_types = types
                del self.load
            # loaded through MadGraph7 (every context, recorded for gridpacks), with the card's
            # parameters: a library refuses a value other than the exported one
            parameters = amplicol_parameters(self.meta, process)
            self.pac_matrix_elements = [
                self.load(os.path.join(pair.dir, table["libraries"][k]["path"]), self.meta["artifact"], parameters)
                for k in libs]
            # per mg7 flavour: its library's position among the channel's
            self.pac_diff_xs = [libs.index(channel["flavor_libraries"][f["index"]]) for f in flavors]
            self.matrix_elements = self.pac_matrix_elements[:1]   # what AmplicolSubprocess reads
            print(f"PacRunner: subprocess {self.subproc_id} channel {self.channel_index}: "
                  f"{len(flavors)} flavours on {len(libs)} libraries ({pair.device})", flush=True)

        def integrand_matrix_elements(self):
            if self.pac_matrix_elements is None:
                return super().integrand_matrix_elements()
            stock = super().integrand_matrix_elements()[0]
            return [ms.MatrixElement(api.index(), self.particle_count, stock.inputs(), stock.outputs(), 1,
                                     True, self.process.me_frame, self.process.incoming_count,
                                     self.channel_index)
                    for api in self.pac_matrix_elements]

        def build_integrands(self, *args, **kwargs):
            if self.pac_matrix_elements is None:
                return super().build_integrands(*args, **kwargs)
            n = len(self.meta["flavors"])
            remap = [0] * n
            diff_xs = self.pac_diff_xs
            subproc = [self.subproc_id] * n
            per_subproc = list(range(n))
            stock_remap = [f["index"] for f in self.meta["flavors"]]
            real = ms.Integrand

            def integrand(*a, **kw):
                a = list(a)
                # positional flavor_remap, flavor_factors, flavor_mirror, flavor_diff_xs_indices,
                # flavor_subproc_indices, flavor_per_subproc_remap (subprocess_madmatrix.build_integrands)
                assert a[21] == stock_remap and a[24] == [] and a[25] == [] and a[26] == [], "Integrand layout"
                a[21] = remap
                if len(self.pac_matrix_elements) > 1:
                    a[24], a[25], a[26] = diff_xs, subproc, per_subproc
                return real(*a, **kw)

            smm.ms.Integrand = integrand
            try:
                return super().build_integrands(*args, **kwargs)
            finally:
                smm.ms.Integrand = real

    mp.AmplicolSubprocess = PacRunnerSubprocess


def latest_run(output):
    runs = sorted(glob.glob(os.path.join(output, "Events", "run_*")), key=lambda d: int(d.rsplit("_", 1)[1]))
    return runs[-1] if runs else None


def run(libraries, artifact=None, args=None):
    """An end-to-end MadGraph7 run on PacRunner's libraries: `libraries` a pac-julia folder and
    `artifact` its pyAmpliCol artifact (inside an mg7 output), or a list of such pairs; `args`
    generate_events' arguments (default ["-f"]). Returns {"cross_section", "error" (pb), "events",
    "run" (the run's folder in OUTPUT/Events), "output"}."""
    pairs = [Pair(lib, art) for lib, art in ([(libraries, artifact)] if artifact is not None else libraries)]
    outputs = {mg7_output(p.artifact) for p in pairs}
    if len(outputs) != 1:
        raise ValueError(f"the artifacts are in {len(outputs)} mg7 outputs: one run is one output")
    output = outputs.pop()
    devices = {p.device for p in pairs}
    if len(devices) != 1:
        raise ValueError(f"libraries of several devices {sorted(devices)}: one run is one device")
    device = devices.pop()
    sys.path.append(mg5_dir(output))
    from madgraph.mg7 import bootstrap
    bootstrap.ensure_madspace(interactive=False)  # MadGraph7's madspace first on sys.path
    if device == "cuda_provider":
        # MadSpace built with CUDA (ENABLE_CUDA) beside the CPU build: madspace/install-cuda
        cuda = os.path.join(mg5_dir(output), "madspace", "install-cuda")
        if not os.path.isdir(os.path.join(cuda, "madspace")):
            raise RuntimeError(f"cuda_provider libraries need a CUDA build of MadSpace in {cuda}")
        sys.path.insert(0, cuda)
    install(pairs)
    from madgraph.mg7.launch import main
    before = latest_run(output)
    cwd, argv = os.getcwd(), sys.argv
    os.chdir(output)
    sys.argv = [os.path.join(output, "bin", "generate_events")] + list(args or ["-f"])
    try:
        main()
    finally:
        os.chdir(cwd)
        sys.argv = argv
    folder = latest_run(output)
    if folder is None or folder == before:
        raise RuntimeError(f"{output}: generate_events made no new run")
    with open(os.path.join(folder, "info.json")) as f:
        process = json.load(f)["process"]
    return {"cross_section": float(process["mean"]), "error": float(process["error"]),
            "events": int(process["count_unweighted"]), "run": folder, "output": output}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    result = run(sys.argv[1], sys.argv[2], sys.argv[3:])
    print(f"PacRunner: cross section {result['cross_section']!r} +- {result['error']:.4g} pb, "
          f"{result['events']} events in {result['run']}")
