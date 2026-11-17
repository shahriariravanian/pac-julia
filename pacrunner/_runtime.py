"""Runtime: one callable for any UMAMI matrix element in MadSpace (the stock pyAmpliCol library or
PacRunner's), numpy in and numpy out, the MadSpace function built for each call's inputs on first use.

    rt = library.runtime(channel=0)
    out = rt(momenta, flavours)                               # the helicity-summed matrix element
    out = rt(momenta, flavours, random_helicity=r)            # the LC pass: + colour and helicity indices
    out = rt(momenta, flavours, helicity=h)                   # one exported helicity
    out.matrix_element, out.color_index, out.helicity_index   # numpy arrays (None when not asked)

Contexts (madspace.Context) are made here, one per (device, threads) and process."""

from typing import NamedTuple

import numpy as np

from . import _env

_contexts = {}


def madspace(mg7=None):
    """MadSpace, imported once for this process (pacrunner._env.import_madspace)"""
    return _env.import_madspace(mg7)


def context(device="cpu", threads=None, mg7=None):
    """the madspace.Context of `device` ("cpu" or "cuda") with `threads` threads (default: the CPUs for
    the CPU, 1 for CUDA), one per process"""
    ms = madspace(mg7)
    if device not in ("cpu", "cuda"):
        raise _env.PacRunnerError(f"device {device!r}: 'cpu' or 'cuda'")
    threads = threads if threads is not None else (-1 if device == "cpu" else 1)
    key = (device, threads)
    if key not in _contexts:
        dev = ms.cpu_device() if device == "cpu" else ms.cuda_device()
        _contexts[key] = ms.Context(dev, threads)
    return _contexts[key]


class RuntimeResult(NamedTuple):
    """a Runtime call's outputs, one value per event (None: not asked for)"""
    matrix_element: object
    color_index: object = None
    helicity_index: object = None


def _events(x, n, dtype, what):
    """an input of one value per event: a scalar repeated, or an array of n"""
    if np.isscalar(x):
        return np.full(n, x, dtype=dtype)
    if _is_device_array(x):
        return x
    a = np.asarray(x, dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"{what}: {a.shape}, expected one value per event ({n},)")
    return a


def _is_device_array(x):
    return hasattr(x, "__cuda_array_interface__")


class Runtime:
    """A matrix element of one UMAMI channel in MadSpace: call(momenta, flavour, ...) -> RuntimeResult.

    momenta[event, particle, (E, px, py, pz)] in the provider's particle order; flavour the provider's
    flavour per event (an int: every event). On a CUDA context numpy inputs are moved to the device
    (CuPy) and the outputs back; device arrays (CuPy) in give device arrays out. `function(inputs,
    outputs)` is the madspace.Function of one input set, `function_runtime(...)` its FunctionRuntime."""

    def __init__(self, build, context, particle_count, cuda, description=""):
        self._build = build              # (inputs, outputs) -> madspace.Function
        self.context = context
        self.particle_count = particle_count
        self.cuda = cuda
        self.description = description
        self._runtimes = {}

    def __repr__(self):
        return f"<pacrunner.Runtime {self.description} ({'cuda' if self.cuda else 'cpu'})>"

    def function(self, inputs, outputs):
        return self._build(list(inputs), list(outputs))

    def function_runtime(self, inputs, outputs):
        """the madspace.FunctionRuntime of these inputs and outputs (built once)"""
        ms = madspace()
        key = (tuple(int(i) for i in inputs), tuple(int(o) for o in outputs))
        if key not in self._runtimes:
            self._runtimes[key] = ms.FunctionRuntime(self.function(inputs, outputs), self.context)
        return self._runtimes[key]

    def __call__(self, momenta, flavour=0, *, random_helicity=None, helicity=None, alpha_s=None):
        ME = madspace().MatrixElement
        if random_helicity is not None and helicity is not None:
            raise ValueError("random_helicity (the LC pass: a helicity drawn) or helicity (one fixed), not both")
        device_in = _is_device_array(momenta)
        if not device_in:
            momenta = np.asarray(momenta, dtype=np.float64)
        n = momenta.shape[0]
        if tuple(momenta.shape[1:]) != (self.particle_count, 4):
            raise ValueError(f"momenta: {tuple(momenta.shape)}, expected (events, {self.particle_count}, 4)")
        inputs, args = [ME.momenta_in, ME.flavor_in], [momenta, _events(flavour, n, np.int32, "flavour")]
        outputs = [ME.matrix_element_out]
        if random_helicity is not None:
            inputs.append(ME.random_helicity_in)
            args.append(_events(random_helicity, n, np.float64, "random_helicity"))
            outputs += [ME.color_index_out, ME.helicity_index_out]
        elif helicity is not None:
            inputs.append(ME.helicity_in)
            args.append(_events(helicity, n, np.int32, "helicity"))
        if alpha_s is not None:
            inputs.append(ME.alpha_s_in)
            args.append(_events(alpha_s, n, np.float64, "alpha_s"))
        values = self._call(self.function_runtime(inputs, outputs), args, device_in)
        return RuntimeResult(*values)

    def _call(self, rt, args, device_in):
        if not self.cuda:
            if any(_is_device_array(a) for a in args):
                raise ValueError("device arrays for a CPU runtime: pass numpy arrays")
            return [np.array(x.numpy()) for x in rt.call(args)]
        try:
            import cupy
        except ImportError as e:
            raise _env.PacRunnerError("a CUDA runtime needs CuPy (`python -m pip install -e .[cuda]`)") from e
        outs = rt.call([cupy.asarray(a) for a in args])
        cupy.cuda.runtime.deviceSynchronize()
        return [cupy.from_dlpack(x) if device_in else cupy.from_dlpack(x).get() for x in outs]


def matrix_element_runtime(api, particle_count, channel, context, cuda, description=""):
    """a Runtime of one loaded UMAMI library as it is (a provider of many flavours, called with each
    event's flavour: the stock library), its channel fixed"""
    ms = madspace()
    ME = ms.MatrixElement

    def build(inputs, outputs):
        return ME(api.index(), particle_count, inputs + [ME.channel_in], outputs, diagram_count=1,
                  sample_random_inputs=False, me_frame=[], incoming_count=2, channel=channel).function()

    return Runtime(build, context, particle_count, cuda, description)
