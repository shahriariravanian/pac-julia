"""PacRunner's CUDA provider libraries (umami_libraries(...; device = :cuda_provider)) inside
MadSpace's GPU runtime, against the CPU libraries (device = :cpu) of the same export in MadSpace's
CPU runtime: for every library, the same events (random on-shell momenta, random helicity
numbers) go through a MatrixElement in a CUDA context and in a CPU context. Compared bit for bit
(f64) or within --rtol (mixed):

  sampled   ME, colour index and helicity index (random helicity in: the LC pass);
  fixed     ME at each exported helicity (the reweight pass's helicity input); with --rtol the
            difference is taken relative to the event's summed ME (a small helicity's own relative
            error in mixed precision can reach 1e-3..1e-2 while the sum stays at ~1e-6).

usage: python umami_check_gpu.py CPU_UMAMI_PAC_JSON PROVIDER_UMAMI_PAC_JSON [EVENTS] [--rtol R]
(a CUDA build of MadSpace and CuPy on PYTHONPATH: MadSpace's Python runs a CUDA context on
inputs that are already on the device, which CuPy provides through DLPack)"""

import json
import os
import sys

import numpy as np

import cupy

import madspace as ms

ME = ms.MatrixElement


def momenta(masses, events, rng):
    p = rng.uniform(-300.0, 300.0, size=(events, len(masses), 3))
    e = np.sqrt((p * p).sum(axis=2) + np.asarray(masses) ** 2)
    return np.concatenate([e[:, :, None], p], axis=2)


def runtime(context, api, n, inputs, outputs):
    me = ME(api.index(), n, inputs, outputs, 1, False, [], 2, 0)
    return ms.FunctionRuntime(me.function(), context)


def call(rt, args, gpu=False):
    """The runtime on `args` (numpy arrays; on the device for a CUDA context), its outputs on the host."""
    if gpu:
        outs = rt.call([cupy.asarray(a) for a in args])
        cupy.cuda.runtime.deviceSynchronize()
        return [cupy.from_dlpack(x).get() for x in outs]
    return [np.array(np.from_dlpack(x)) for x in rt.call(args)]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and (
        np.array_equal(a.view(np.uint64), b.view(np.uint64)) if a.dtype == np.float64 else np.array_equal(a, b))


def main(cpu_path, gpu_path, events=4096, rtol=None):
    cpu_table, gpu_table = json.load(open(cpu_path)), json.load(open(gpu_path))
    meta = json.load(open(cpu_table["metadata"]))
    n = int(meta["provider"]["particle_count"])
    masses = meta["provider"]["masses"]
    exported = len(meta["helicities"])
    assert len(cpu_table["libraries"]) == len(gpu_table["libraries"]), "the tables have different libraries"
    cpu = ms.Context(ms.cpu_device(), 1)
    gpu = ms.Context(ms.cuda_device(), 1)
    rng = np.random.default_rng(20261113)
    sampled_in = [ME.momenta_in, ME.flavor_in, ME.random_helicity_in]
    sampled_out = [ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out]
    fixed_in = [ME.momenta_in, ME.flavor_in, ME.helicity_in]

    def agree(a, b):
        if rtol is None or a.dtype != np.float64:
            return same(a, b)
        return np.allclose(a, b, rtol=rtol, atol=0.0)

    counts = {"sampled": 0, "fixed": 0}
    worst = 0.0
    fixed_own = fixed_sum = 0.0   # the fixed pass's largest difference relative to itself, to the sum
    bad = []
    for k, (lc, lg) in enumerate(zip(cpu_table["libraries"], gpu_table["libraries"])):
        a_api = cpu.load_matrix_element(os.path.join(os.path.dirname(cpu_path), lc["path"]), "", {})
        g_api = gpu.load_matrix_element(os.path.join(os.path.dirname(gpu_path), lg["path"]), "", {})
        mom = momenta(masses, events, rng)
        zero = np.zeros(events, np.int32)
        r = rng.uniform(0.0, 1.0, events)
        a = call(runtime(cpu, a_api, n, sampled_in, sampled_out), [mom, zero, r])
        b = call(runtime(gpu, g_api, n, sampled_in, sampled_out), [mom, zero, r], gpu=True)
        worst = max(worst, float(np.max(np.abs(a[0] - b[0]) / np.abs(a[0]))))
        if rtol is None:
            ok = all(same(x, y) for x, y in zip(a, b))
        else:   # mixed: a helicity drawn near a boundary may differ
            ok = agree(a[0], b[0]) and same(a[1], b[1]) and np.mean(a[2] == b[2]) > 0.99
        counts["sampled"] += ok
        ok or bad.append(f"library {k}: sampled: ME rel. diff {np.max(np.abs(a[0] - b[0]) / np.abs(a[0])):.2e}, "
                         f"colour {np.array_equal(a[1], b[1])}, helicity {np.mean(a[2] == b[2]):.4f} equal")
        cpu_fixed = runtime(cpu, a_api, n, fixed_in, [ME.matrix_element_out])
        gpu_fixed = runtime(gpu, g_api, n, fixed_in, [ME.matrix_element_out])
        ok = True
        total = np.abs(a[0])
        for h in range(exported):
            hel = np.full(events, h, np.int32)
            x, y = call(cpu_fixed, [mom, zero, hel])[0], call(gpu_fixed, [mom, zero, hel], gpu=True)[0]
            d = np.abs(x - y)
            nz = x != 0
            if nz.any():
                fixed_own = max(fixed_own, float(np.max(d[nz] / np.abs(x[nz]))))
            fixed_sum = max(fixed_sum, float(np.max(d / total)))
            ok = ok and (same(x, y) if rtol is None else bool(np.all(d <= rtol * total)))
        counts["fixed"] += ok
        ok or bad.append(f"library {k}: fixed helicity differs")
    libraries = len(cpu_table["libraries"])
    print(f"{os.path.basename(os.path.dirname(gpu_path))}: {libraries} libraries, {events} events each, "
          f"CUDA provider ({gpu_table.get('precision', 'f64')}) vs CPU" + (f", rtol {rtol}" if rtol else ", bit for bit"))
    for name, good in counts.items():
        print(f"  {name}: {good} of {libraries} libraries agree")
    print(f"  largest relative ME difference (sampled): {worst:.1e}; fixed: {fixed_own:.1e} of the helicity's "
          f"value, {fixed_sum:.1e} of the event's sum")
    for line in bad[:10]:
        print("  " + line)
    return 0 if not bad else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    rtol = None
    if "--rtol" in args:
        i = args.index("--rtol")
        rtol = float(args[i + 1])
        del args[i:i + 2]
    sys.exit(main(args[0], args[1], *(int(x) for x in args[2:3]), rtol=rtol))
