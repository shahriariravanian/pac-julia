"""pacrunner.load_libraries against the stock pyAmpliCol library, through MadSpace: for
every UMAMI channel, a batch of events of mixed flavours (random on-shell momenta, a random flavour
of the channel per event, random helicity numbers) goes to the stock library (each event with its
flavour) and to libs.runtime (one MadSpace function: the batch split by flavour over the PacRunner
libraries, each with flavour 0, and merged back). Compared bit for bit, in the events' order: the LC
pass (ME, colour and helicity indices) and the fixed-helicity pass (ME at every exported helicity).
The stock library runs at the libraries' exported parameters (libs.parameters). Libraries built in
mixed precision (the table's "precision": CUDA's f32 currents) are compared within MIXED_RTOL
instead: the colour index exactly, the drawn helicity index reported where it differs (a random
number at a boundary of the cumulative helicity sum picks the neighbour), and a fixed helicity's ME
relative to the event's summed ME (f32 currents carry errors on the scale of the whole amplitude: a
small helicity's own relative error can be large).

usage: python test_libraries.py OUT [STOCK_LIB ARTIFACT_DIR] [EVENTS]
  OUT:        umami_libraries' output folder (or its umami_pac.json)
  STOCK_LIB:  the stock library (default: the mg7 output's lib/libamplicol_<name>.so beside
              OUT = <output>/lib/pacrunner[-<device>]/<name>); ARTIFACT_DIR: the artifact (default:
              the libraries')
MadSpace on PYTHONPATH (e.g. MadGraph7/madspace/install). For :cuda_provider libraries (a CUDA
context; the stock library still on the CPU): MadSpace's CUDA build (madspace/install-cuda) and
CuPy, whose arrays carry the inputs to the GPU (a CUDA runtime takes device arrays)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # pacrunner, from the clone
import madspace as ms  # noqa: E402
from pacrunner import load_libraries  # noqa: E402

ME = ms.MatrixElement
MIXED_RTOL = 1e-4   # test_umami.jl's


def momenta(masses, events, rng):
    p = rng.uniform(-300.0, 300.0, size=(events, len(masses), 3))
    e = np.sqrt((p * p).sum(axis=2) + np.asarray(masses) ** 2)
    return np.concatenate([e[:, :, None], p], axis=2)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and (np.array_equal(a.view(np.uint64), b.view(np.uint64)) if a.dtype == np.float64
                                   else np.array_equal(a, b))


def stock_runtime(context, api, n, channel, inputs, outputs):
    me = ME(api.index(), n, inputs + [ME.channel_in], outputs, diagram_count=1, sample_random_inputs=False,
            me_frame=[], incoming_count=2, channel=channel)
    return ms.FunctionRuntime(me.function(), context)


def caller(cuda):
    """call(runtime, args) -> numpy outputs: host arrays, or CuPy device arrays on a CUDA context"""
    if not cuda:
        return lambda rt, args: [np.array(x.numpy()) for x in rt.call(args)]
    import cupy

    def call(rt, args):
        outs = rt.call([cupy.asarray(a) for a in args])
        cupy.cuda.runtime.deviceSynchronize()
        return [cupy.from_dlpack(x).get() for x in outs]
    return call


def main(out, stock_path=None, artifact=None, events=256):
    table_path = out if out.endswith(".json") else os.path.join(out, "umami_pac.json")
    table = json.load(open(table_path))
    meta = json.load(open(table["metadata"]))
    if stock_path is None:
        name = os.path.basename(os.path.dirname(os.path.abspath(table_path)))
        stock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(table_path)))),
                                  f"libamplicol_{name}.so")
    n = int(meta["provider"]["particle_count"])
    masses = meta["provider"]["masses"]
    exported = len(meta["helicities"])
    cuda = table["device"] == "cuda_provider"
    mixed = table.get("precision") == "mixed"
    close = (lambda x, y: np.allclose(x, y, rtol=MIXED_RTOL, atol=0.0)) if mixed else same
    if cuda:
        try:
            import cupy  # noqa: F401
        except ImportError:
            print(f"{out}: CUDA provider libraries need CuPy (device arrays for the CUDA runtime): not run")
            return 2
    cpu = ms.Context(ms.cpu_device(), 1)
    context = ms.Context(ms.cuda_device(), 1) if cuda else cpu
    libs = load_libraries(context, out)
    artifact = artifact or libs.artifact
    stock = cpu.load_matrix_element(stock_path, os.path.abspath(artifact), libs.parameters)
    call_pac, call_stock = caller(cuda), caller(False)
    rng = np.random.default_rng(20261115)
    sampled_in = [ME.momenta_in, ME.flavor_in, ME.random_helicity_in]
    sampled_out = [ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out]
    fixed_in = [ME.momenta_in, ME.flavor_in, ME.helicity_in]
    bad, flavours_seen, worst, fixed_worst, label_differ = [], 0, 0.0, 0.0, 0
    for c in range(libs.channels):
        mom = momenta(masses, events, rng)
        flav = rng.integers(0, libs.flavours(c), events).astype(np.int32)
        flavours_seen += len(np.unique(flav))
        r = rng.uniform(0.0, 1.0, events)
        a = call_stock(stock_runtime(cpu, stock, n, c, sampled_in, sampled_out), [mom, flav, r])
        b = call_pac(libs.runtime(c, sampled_in, sampled_out), [mom, flav, r])
        worst = max(worst, float(np.max(np.abs(a[0] - b[0]) / np.abs(a[0]))))
        if mixed:
            ok = close(a[0], b[0]) and same(a[1], b[1])
            label_differ += int(np.count_nonzero(a[2] != b[2]))
        else:
            ok = all(same(x, y) for x, y in zip(a, b))
        if not ok:
            bad.append(f"channel {c}: sampled pass differs (ME max rel. "
                       f"{np.max(np.abs(a[0] - b[0]) / np.abs(a[0])):.2e})")
        s_fixed = stock_runtime(cpu, stock, n, c, fixed_in, [ME.matrix_element_out])
        pac_fixed = libs.runtime(c, fixed_in, [ME.matrix_element_out])
        for h in range(exported):
            hel = np.full(events, h, np.int32)
            x, y = call_stock(s_fixed, [mom, flav, hel])[0], call_pac(pac_fixed, [mom, flav, hel])[0]
            err = np.abs(x - y) / np.abs(a[0])   # relative to the event's summed ME (the sampled pass's)
            fixed_worst = max(fixed_worst, float(np.max(err)))
            if not (np.all(err <= MIXED_RTOL) if mixed else same(x, y)):
                bad.append(f"channel {c}: fixed helicity {h} differs")
                break
    agree = f"within {MIXED_RTOL:g} (ME max rel. {worst:.1e}; fixed helicities max {fixed_worst:.1e} of the ME; " \
            f"helicity index differs for {label_differ} of {events * libs.channels} events)" if mixed else "bit for bit"
    print(f"{os.path.dirname(os.path.abspath(table_path))} ({libs.device}, {table.get('precision', 'f64')}): "
          f"{libs.channels} channels, {len(libs.apis)} libraries, {events} events of mixed flavours per channel "
          f"({flavours_seen} flavours in all): " + (f"sampled and fixed-helicity passes {agree}" if not bad else "FAILURES"))
    for line in bad[:10]:
        print("  " + line)
    return 0 if not bad else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    events = int(args.pop()) if len(args) in (2, 4) and args[-1].isdigit() else 256
    sys.exit(main(*args, events=events))
