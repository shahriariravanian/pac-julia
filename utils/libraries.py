"""PacRunner's UMAMI libraries in MadSpace, in a few lines.

umami_libraries (src/umami.jl) writes one UMAMI library per program a provider uses, each for one
flavour (it is always called with flavour 0), and OUT/umami_pac.json: the libraries and, per UMAMI
channel, the library of each of the provider's flavours. This module loads them into a
madspace.Context and hides that layout:

    import madspace as ms
    from libraries import load_libraries          # pac-julia/utils on sys.path

    context = ms.Context(ms.cpu_device(), 1)
    libs = load_libraries(context, OUT)           # OUT: umami_libraries' folder (or its umami_pac.json)
    rt = libs.runtime(channel=0)                  # a madspace.FunctionRuntime: momenta, flavour -> ME
    (me,) = rt.call([momenta, flavours])          # flavours: the provider's flavour of each event

The runtime is one MadSpace function over all the channel's flavours, built as MadSpace's Integrand
builds its flavour splitting: each event's flavour is looked up to its library
(flavor_diff_xs_indices), the batch is split by library (batch_split_by_index), each part runs its
library's matrix element with flavour 0 (flavor_remap), and the outputs are merged back into the
events' order (batch_merge_by_index). So the splitting runs where the context runs: on the GPU for
a CUDA context and :cuda_provider libraries, on the CPU otherwise. `call` takes and returns what any
FunctionRuntime of that context does (numpy arrays on the CPU; CuPy or other DLPack device arrays
on a CUDA device). Other inputs and outputs (random helicity, helicity index, alpha_s; colour and
helicity indices) as for any UMAMI matrix element:

    ME = ms.MatrixElement
    rt = libs.runtime(0, inputs=[ME.momenta_in, ME.flavor_in, ME.random_helicity_in],
                      outputs=[ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out])

libs.function(channel, ...) is the same as a madspace.Function, to build into a larger one. For an
Integrand of your own: libs.matrix_elements(channel, ...) gives one ms.MatrixElement per library of
the channel, and libs.flavor_arguments(channel) its flavor_remap and flavor_diff_xs_indices, as
test/mg7_run.py passes them to MadGraph7's Integrand.

The device: libraries built for :cpu, :cuda and :metal are CPU providers (a CPU context, host
memory); :cuda_provider libraries are CUDA providers (MadSpace's CUDA build, a CUDA context:
ms.Context(ms.cuda_device(), 1)); libs.device says which. The libraries take their model parameters
from the export: `parameters` may be empty, and a parameter given must be the exported value (or the
library refuses it); libs.parameters holds the exported values."""

import json
import os

import madspace as ms

ME = ms.MatrixElement

DEFAULT_INPUTS = (ME.momenta_in, ME.flavor_in)
DEFAULT_OUTPUTS = (ME.matrix_element_out,)


def load_libraries(context, out, parameters=None, param_card=""):
    """Every library of umami_libraries' output `out` (its folder or its umami_pac.json) loaded into
    `context` (a madspace.Context); returns a PacLibraries"""
    return PacLibraries(context, out, parameters, param_card)


class PacLibraries:
    """The libraries of one umami_libraries output, loaded into a context.

    table          umami_pac.json, as read
    apis[k]        library k's MatrixElementApi (Context.load_matrix_element)
    device         the device the libraries were built for ("cpu", "cuda", "cuda_provider", "metal")
    particle_count the provider's particles (momenta per event)
    channels       the UMAMI channels' count
    artifact       the pyAmpliCol artifact the provider belongs to
    parameters     the model's external parameters at the values the libraries were exported with
                   ({name: real part}): e.g. to load the stock library at the same point of the model,
                   context.load_matrix_element(stock_lib, artifact, libs.parameters)"""

    def __init__(self, context, out, parameters=None, param_card=""):
        path = out if out.endswith(".json") else os.path.join(out, "umami_pac.json")
        with open(path) as f:
            self.table = json.load(f)
        if self.table.get("kind") != "pacrunner-umami-libraries":
            raise ValueError(f"{path}: not the output of PacRunner's umami_libraries")
        base = os.path.dirname(os.path.abspath(path))
        self.context = context
        self.device = self.table["device"]
        self.particle_count = int(self.table["particle_count"])
        self.channels = len(self.table["channels"])
        # the metadata is <artifact>/API/umami/metadata.json
        self.artifact = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(self.table["metadata"]))))
        self.parameters = exported_parameters(self.table["export"], self.artifact)
        # param_card is not read by these libraries (they carry the export's parameters)
        self.apis = [context.load_matrix_element(os.path.join(base, lib["path"]), param_card, dict(parameters or {}))
                     for lib in self.table["libraries"]]

    def _channel(self, channel):
        if not 0 <= channel < self.channels:
            raise IndexError(f"UMAMI channel {channel}: there are {self.channels}")
        return self.table["channels"][channel]

    def flavours(self, channel):
        """the provider's flavours of `channel`"""
        return len(self._channel(channel)["flavor_libraries"])

    def library(self, channel, flavour):
        """the index (into apis) of the library of `flavour` in `channel`"""
        return self._channel(channel)["flavor_libraries"][flavour]

    def matrix_elements(self, channel, inputs=DEFAULT_INPUTS, outputs=DEFAULT_OUTPUTS, sample_random_inputs=False,
                        me_frame=(), incoming_count=2):
        """one ms.MatrixElement per library of `channel` (in the order of the channel's "libraries",
        the order flavor_arguments' indices refer to), the channel fixed; flavour inputs must be 0"""
        inputs = _with_channel(inputs)
        return [ME(self.apis[k].index(), self.particle_count, inputs, list(outputs), diagram_count=1,
                   sample_random_inputs=sample_random_inputs, me_frame=list(me_frame), incoming_count=incoming_count,
                   channel=channel)
                for k in self._channel(channel)["libraries"]]

    def flavor_arguments(self, channel):
        """an Integrand's flavour arguments for `channel`'s matrix_elements: flavor_remap (every flavour
        to 0: each library is one flavour) and flavor_diff_xs_indices (flavour f's library's position)"""
        ch = self._channel(channel)
        libs = ch["libraries"]
        return {"flavor_remap": [0] * len(ch["flavor_libraries"]),
                "flavor_diff_xs_indices": [libs.index(k) for k in ch["flavor_libraries"]]}

    def function(self, channel=0, inputs=DEFAULT_INPUTS, outputs=DEFAULT_OUTPUTS, me_frame=(), incoming_count=2):
        """`channel`'s matrix element over all its flavours as one ms.Function: the arguments are
        `inputs` but channel_in (the channel is fixed), the results `outputs`, events in order"""
        mes = self.matrix_elements(channel, inputs, outputs, me_frame=me_frame, incoming_count=incoming_count)
        first = mes[0].function()
        fb = ms.FunctionBuilder(ms.NamedTypes([(f"input{i}", v.type) for i, v in enumerate(first.inputs)]),
                                ms.NamedTypes([(f"output{i}", v.type) for i, v in enumerate(first.outputs)]))
        args = [fb.input(i) for i in range(len(first.inputs))]
        flavour = [i for i in inputs if i != ME.channel_in].index(ME.flavor_in)
        tables = self.flavor_arguments(channel)
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

    def runtime(self, channel=0, inputs=DEFAULT_INPUTS, outputs=DEFAULT_OUTPUTS, me_frame=(), incoming_count=2):
        """`channel`'s matrix element over all its flavours (function()) as an ms.FunctionRuntime of
        the context: call([momenta, flavours, ...]) -> the outputs, as any FunctionRuntime's"""
        return ms.FunctionRuntime(self.function(channel, inputs, outputs, me_frame, incoming_count), self.context)


def exported_parameters(export_path, artifact=None):
    """the model parameters of a recurrence export, {name: real part} (MadSpace's parameters are real):
    with `artifact`, only its model's external parameters (model/parameters.json: the ones a library
    can be given; the export also lists the derived ones)"""
    with open(export_path) as f:
        values = {p["name"]: float(p["value"]["re"]) for p in json.load(f)["parameters"]}
    external = os.path.join(artifact, "model", "parameters.json") if artifact else None
    if external is None or not os.path.isfile(external):
        return values
    with open(external) as f:
        names = json.load(f)["parameters"]
    return {name: values[name] for name in names if name in values}


def _with_channel(inputs):
    """the inputs with channel_in (the channel is the MatrixElement's, not a call argument)"""
    inputs = list(inputs)
    if ME.momenta_in not in inputs or ME.flavor_in not in inputs:
        raise ValueError("the inputs need momenta_in and flavor_in")
    return inputs if ME.channel_in in inputs else inputs + [ME.channel_in]
