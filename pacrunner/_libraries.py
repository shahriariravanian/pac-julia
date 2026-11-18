"""Provider: PacRunner's build of a UMAMI provider, the folder PacRunner's umami_libraries writes.

    provider = Provider("zgg_libs")                 # a folder (or its umami_pac.json)
    provider.device, provider.precision, provider.channels, provider.flavours(0), provider.library_count
    rt = provider.runtime(channel=0)                # a pacrunner.Runtime over all the channel's flavours
    provider.subprocess, provider.run()             # a provider in a MadGraph7 output: its SubProcess, generate_events

A provider (pyAmpliCol's `p0`, ...) is one UMAMI matrix-element interface for its channels and flavours;
pyAmpliCol's own build is the stock library, PacRunner's this folder. umami_libraries writes one library
per (process, colour flow, permutation) a provider uses, each for one flavour (always called
with flavour 0), and umami_pac.json: the libraries and, per UMAMI channel, the library of each of the
provider's flavours. A runtime is one MadSpace function that looks up each event's library, splits the
batch (batch_split_by_index), runs each library with flavour 0 and merges the outputs back
(batch_merge_by_index), as MadSpace's Integrand does; a channel of one library needs no split. So the
split runs where the context runs: on the GPU for cuda_provider libraries.

The device: libraries built for cpu, cuda and metal are CPU providers (a CPU context); cuda_provider
libraries need MadSpace's CUDA build and a CUDA context (made here). The libraries carry the export's
model parameters; `parameters` holds the artifact's external ones at those values."""

import json
import os
from pathlib import Path

from . import _env, _runtime


def exported_parameters(export_path, artifact=None):
    """the model parameters of a recurrence export, {name: real part} (MadSpace's parameters are real):
    with `artifact`, only its model's external parameters (model/parameters.json: the ones a library can
    be given; the export also lists the derived ones)"""
    with open(export_path) as f:
        values = {p["name"]: float(p["value"]["re"]) for p in json.load(f)["parameters"]}
    external = os.path.join(artifact, "model", "parameters.json") if artifact else None
    if external is None or not os.path.isfile(external):
        return values
    with open(external) as f:
        names = json.load(f)["parameters"]
    return {name: values[name] for name in names if name in values}


def read_table(folder):
    """umami_pac.json of an umami_libraries folder (or the file): (table, the folder)"""
    path = Path(folder)
    path = path if path.suffix == ".json" else path / "umami_pac.json"
    if not path.is_file():
        raise _env.PacRunnerError(f"{path}: no umami_pac.json (not an output of PacRunner's umami_libraries)")
    with open(path) as f:
        table = json.load(f)
    if table.get("kind") != "pacrunner-umami-libraries":
        raise _env.PacRunnerError(f"{path}: not the output of PacRunner's umami_libraries")
    return table, path.parent.resolve()


def _with_channel(ME, inputs):
    """the inputs with channel_in (the channel is the MatrixElement's, not a call argument)"""
    inputs = list(inputs)
    if ME.momenta_in not in inputs or ME.flavor_in not in inputs:
        raise ValueError("the inputs need momenta_in and flavor_in")
    return inputs if ME.channel_in in inputs else inputs + [ME.channel_in]


class Provider:
    """PacRunner's build of one UMAMI provider (an umami_libraries folder).

    folder         the folder
    output         the MadGraph7 output it lies in (OUTPUT/lib/pacrunner[-device]/<name>), else None
    name           the folder's name (in an output: the stock library's, lib/libamplicol_<name>.so)
    table          umami_pac.json, as read
    device         what they were built for: "cpu", "cuda", "cuda_provider", "metal"
    precision      "f64" or "mixed"
    particle_count the provider's particles
    channels       the UMAMI channels' count
    library_count  the libraries' count
    artifact       the pyAmpliCol artifact the provider belongs to
    parameters     the artifact's external model parameters at the exported values ({name: real part})"""

    def __init__(self, folder):
        self.table, self.folder = read_table(folder)
        self.device = self.table["device"]
        self.precision = self.table.get("precision", "f64")
        self.particle_count = int(self.table["particle_count"])
        self.channels = len(self.table["channels"])
        self.library_count = len(self.table["libraries"])
        # the metadata is <artifact>/API/umami/metadata.json
        metadata = Path(self.table["metadata"])
        self.artifact = metadata.parent.parent.parent
        export = Path(self.table["export"])
        self.parameters = exported_parameters(export, self.artifact) if export.is_file() else {}
        self.output = _env.mg7_output_of(self.artifact)
        self.name = self.folder.name
        self._apis = {}      # id(context) -> [MatrixElementApi]
        self._subprocess = None

    def __repr__(self):
        return (f"<{type(self).__name__} {self.folder} ({self.device}, {self.precision}: {self.library_count} libraries, "
                f"{self.channels} channels)>")

    # ---- the layout

    def _channel(self, channel):
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        return self.table["channels"][channel]

    def flavours(self, channel):
        """the provider's flavours of `channel`"""
        return len(self._channel(channel)["flavor_libraries"])

    def library(self, channel, flavour):
        """the index of the library of `flavour` in `channel`"""
        return self._channel(channel)["flavor_libraries"][flavour]

    def library_path(self, k):
        return self.folder / self.table["libraries"][k]["path"]

    def flavor_arguments(self, channel):
        """an Integrand's flavour arguments for `channel`'s matrix_elements: flavor_remap (every flavour
        to 0: each library is one flavour) and flavor_diff_xs_indices (flavour f's library's position)"""
        ch = self._channel(channel)
        libs = ch["libraries"]
        return {"flavor_remap": [0] * len(ch["flavor_libraries"]),
                "flavor_diff_xs_indices": [libs.index(k) for k in ch["flavor_libraries"]]}

    # ---- MadSpace

    @property
    def context_device(self):
        """the MadSpace device of these libraries' context: "cuda" for CUDA providers, else "cpu" """
        return "cuda" if self.device == "cuda_provider" else "cpu"

    def context(self, threads=None):
        """the madspace.Context for these libraries (made once per process and thread count)"""
        return _runtime.context(self.context_device, threads, _env.madgraph7_for(self.folder, required=False))

    def apis(self, context, parameters=None, param_card=""):
        """every library loaded into `context` (once per context): [MatrixElementApi]. The libraries do
        not read param_card; a parameter given must be the exported value (or the library refuses it)"""
        key = id(context)
        if key not in self._apis:
            self._apis[key] = [context.load_matrix_element(os.fspath(self.library_path(k)), param_card,
                                                           dict(parameters or {}))
                               for k in range(self.library_count)]
        return self._apis[key]

    def matrix_elements(self, channel, inputs=None, outputs=None, context=None, sample_random_inputs=False,
                        me_frame=(), incoming_count=2):
        """one madspace.MatrixElement per library of `channel` (in the order of the channel's
        "libraries", the order flavor_arguments' indices refer to), the channel fixed; flavour inputs
        must be 0 (default inputs: momenta, flavour; outputs: the matrix element)"""
        ms = _runtime.madspace()
        ME = ms.MatrixElement
        inputs = _with_channel(ME, inputs if inputs is not None else [ME.momenta_in, ME.flavor_in])
        outputs = list(outputs if outputs is not None else [ME.matrix_element_out])
        apis = self.apis(context if context is not None else self.context())
        return [ME(apis[k].index(), self.particle_count, inputs, outputs, diagram_count=1,
                   sample_random_inputs=sample_random_inputs, me_frame=list(me_frame), incoming_count=incoming_count,
                   channel=channel)
                for k in self._channel(channel)["libraries"]]

    def function(self, channel, inputs, outputs, context=None, me_frame=(), incoming_count=2):
        """`channel`'s matrix element over all its flavours as one madspace.Function: the arguments are
        `inputs` but channel_in, the results `outputs`, events in order"""
        ms = _runtime.madspace()
        ME = ms.MatrixElement
        mes = self.matrix_elements(channel, inputs, outputs, context, me_frame=me_frame, incoming_count=incoming_count)
        first = mes[0].function()
        fb = ms.FunctionBuilder(ms.NamedTypes([(f"input{i}", v.type) for i, v in enumerate(first.inputs)]),
                                ms.NamedTypes([(f"output{i}", v.type) for i, v in enumerate(first.outputs)]))
        args = [fb.input(i) for i in range(len(first.inputs))]
        flavour = [i for i in inputs if i != ME.channel_in].index(ME.flavor_in)
        tables = self.flavor_arguments(channel)
        if len(mes) == 1:
            # one library: no split (its gathers and merges cost as much as a fast matrix element)
            args[flavour] = fb.gather_int(args[flavour], ms.Value(tables["flavor_remap"]))   # every flavour: 0
            for j, value in enumerate(mes[0].build_function(fb, args).values()):
                fb.output(j, value)
            return fb.function()
        # each event's library (its position among the channel's), the batch split by it
        position = fb.gather_int(args[flavour], ms.Value(tables["flavor_diff_xs_indices"]))
        parts = fb.batch_split_by_index(position, ms.Value(len(mes)))
        merged = [[] for _ in first.outputs]
        for me, indices in zip(mes, parts):
            part = [fb.batch_gather(indices, a) for a in args]
            part[flavour] = fb.gather_int(part[flavour], ms.Value(tables["flavor_remap"]))   # every flavour: 0
            for j, value in enumerate(me.build_function(fb, part).values()):
                merged[j] += [value, indices]
        for j, values in enumerate(merged):
            fb.output(j, fb.batch_merge_by_index(values))
        return fb.function()

    def runtime(self, channel=0, threads=None):
        """`channel`'s matrix element over all its flavours: a pacrunner.Runtime (threads: the CPU
        context's, default all CPUs; a CUDA context's for cuda_provider libraries, default 1)"""
        self._channel(channel)
        context = self.context(threads)
        return _runtime.Runtime(lambda inputs, outputs: self.function(channel, inputs, outputs, context),
                                context, self.particle_count, self.context_device == "cuda",
                                f"{self.folder.name} channel {channel}")


    # ---- in a MadGraph7 output

    @property
    def subprocess(self):
        """the SubProcess it was compiled for (in a MadGraph7 output), else None"""
        if self.output is None:
            return None
        if self._subprocess is None:
            from ._mad import Output
            self._subprocess = Output(self.output).subprocess(self.name)
        return self._subprocess

    def run(self, args=None, in_process=False, log=None):
        """an end-to-end MadGraph7 run (generate_events: the survey, the events, the cross section) with
        this subprocess on these libraries and the output's other subprocesses on their stock ones.
        In a process of its own by default, the output's python (in_process=True: in this one, once per
        process); `args` generate_events' arguments (default -f); `log` a file for its output (default:
        this process's). The cards are the output's (Cards/run_card.toml: cuda_provider libraries need
        run.device = ["cuda"]). Returns a RunResult (cross_section, error, events, run_dir, output,
        wall_seconds). A provider in a MadGraph7 output only."""
        from . import _mgrun
        if self.output is None:
            raise _env.PacRunnerError(f"{self.folder}: its artifact {self.artifact} is not in a MadGraph7 output: "
                                      "run() needs one (SubProcess.compile builds a provider there)")
        return _mgrun.run_libraries(self.output, [self], args=args, in_process=in_process, log=log)

    def integrand_inputs(self, channel, inputs=None, outputs=None, context=None, me_frame=(), incoming_count=2):
        """the pieces of a madspace.Integrand of your own for `channel`: (matrix_elements, flavour
        arguments), matrix_elements one madspace.MatrixElement per library (sampling its random inputs,
        as the Integrand needs), the arguments {flavor_remap, flavor_diff_xs_indices} to pass with them"""
        return (self.matrix_elements(channel, inputs, outputs, context, sample_random_inputs=True, me_frame=me_frame,
                                     incoming_count=incoming_count),
                self.flavor_arguments(channel))

    # the name before 0.2 (one release)
    def mad_artifact(self):
        return self.subprocess


class _ContextLibraries:
    """load_libraries' result (the earlier utils/libraries.py API, for code written against it): a
    Provider loaded into the caller's context; runtime() a madspace.FunctionRuntime"""

    def __init__(self, context, folder, parameters=None, param_card=""):
        self.library = Provider(folder)
        for name in ("table", "device", "particle_count", "channels", "artifact", "parameters"):
            setattr(self, name, getattr(self.library, name))
        self.context = context
        self.apis = self.library.apis(context, parameters, param_card)

    def flavours(self, channel):
        return self.library.flavours(channel)

    def library_index(self, channel, flavour):
        return self.library.library(channel, flavour)

    def flavor_arguments(self, channel):
        return self.library.flavor_arguments(channel)

    def matrix_elements(self, channel, inputs=None, outputs=None, sample_random_inputs=False, me_frame=(),
                        incoming_count=2):
        return self.library.matrix_elements(channel, inputs, outputs, self.context, sample_random_inputs, me_frame,
                                            incoming_count)

    def function(self, channel=0, inputs=None, outputs=None, me_frame=(), incoming_count=2):
        ME = _runtime.madspace().MatrixElement
        inputs = list(inputs if inputs is not None else [ME.momenta_in, ME.flavor_in])
        outputs = list(outputs if outputs is not None else [ME.matrix_element_out])
        return self.library.function(channel, inputs, outputs, self.context, me_frame, incoming_count)

    def runtime(self, channel=0, inputs=None, outputs=None, me_frame=(), incoming_count=2):
        ms = _runtime.madspace()
        return ms.FunctionRuntime(self.function(channel, inputs, outputs, me_frame, incoming_count), self.context)


def load_libraries(context, out, parameters=None, param_card=""):
    """the libraries of umami_libraries' output `out` loaded into `context` (a madspace.Context you
    made): the earlier utils/libraries.py API (runtime() a madspace.FunctionRuntime). New code:
    Provider(out).runtime(channel) (a pacrunner.Runtime, the context made for the libraries' device)."""
    return _ContextLibraries(context, out, parameters, param_card)


# the names before 0.2 (one release)
PacLibrary = Provider
MadLibrary = Provider
