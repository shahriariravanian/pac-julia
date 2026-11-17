"""Timing of pacrunner.load_libraries' runtime: does load_libraries cost anything? For every UMAMI channel
of an umami_libraries output, a batch of EVENTS events of mixed flavours (random on-shell momenta,
a random flavour of the channel per event, random helicity numbers; the LC pass: ME, colour and
helicity indices out), timed three ways:

  runtime   libs.runtime(channel): one MadSpace function, the flavour split inside it
  direct    one FunctionRuntime per library (libs.matrix_elements), each on its own events, split
            beforehand (not timed): what MadSpace's Integrand runs, without load_libraries
  stock     the stock pyAmpliCol library on the whole batch (CPU outputs; at the exported parameters)

Each is warmed up, then the best of REPEATS calls; µs per event over all channels. runtime / direct
is load_libraries' overhead (1.00: none).

usage: python bench_libraries.py OUT [--threads T] [--events N] [--repeats R] [--no-stock]
  T: the context's threads (CPU outputs; default 1); a :cuda_provider output runs on a CUDA context
  (MadSpace's CUDA build on PYTHONPATH, CuPy for the device arrays)."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # pacrunner, from the clone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import madspace as ms  # noqa: E402
from pacrunner import load_libraries  # noqa: E402
from test_libraries import momenta, stock_runtime  # noqa: E402

ME = ms.MatrixElement
SAMPLED_IN = [ME.momenta_in, ME.flavor_in, ME.random_helicity_in]
SAMPLED_OUT = [ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out]


def timer(cuda):
    """best(fn, repeats): the best wall time of fn() (synchronized on a CUDA device)"""
    sync = (lambda: __import__("cupy").cuda.runtime.deviceSynchronize()) if cuda else (lambda: None)

    def best(fn, repeats):
        fn(); sync()                                  # warm up (first use: allocations, compilation)
        times = []
        for _ in range(repeats):
            t = time.perf_counter(); fn(); sync(); times.append(time.perf_counter() - t)
        return min(times)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--events", type=int, default=64000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-stock", action="store_true")
    a = ap.parse_args()
    table_path = a.out if a.out.endswith(".json") else os.path.join(a.out, "umami_pac.json")
    table = json.load(open(table_path))
    cuda = table["device"] == "cuda_provider"
    device_arrays = (lambda x: __import__("cupy").asarray(x)) if cuda else (lambda x: x)
    context = ms.Context(ms.cuda_device(), a.threads) if cuda else ms.Context(ms.cpu_device(), a.threads)
    libs = load_libraries(context, a.out)
    meta = json.load(open(table["metadata"]))
    masses = meta["provider"]["masses"]
    best = timer(cuda)
    stock = None
    if not cuda and not a.no_stock:
        name = os.path.basename(os.path.dirname(os.path.abspath(table_path)))
        stock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(table_path)))),
                                  f"libamplicol_{name}.so")
        stock = context.load_matrix_element(stock_path, libs.artifact, libs.parameters)
    rng = np.random.default_rng(20261115)
    total = {"runtime": 0.0, "direct": 0.0, "stock": 0.0}
    for c in range(libs.channels):
        mom = momenta(masses, a.events, rng)
        flav = rng.integers(0, libs.flavours(c), a.events).astype(np.int32)
        r = rng.uniform(0.0, 1.0, a.events)
        args = [device_arrays(x) for x in (mom, flav, r)]
        # runtime: the split inside MadSpace
        rt = libs.runtime(c, SAMPLED_IN, SAMPLED_OUT)
        total["runtime"] += best(lambda: rt.call(args), a.repeats)
        # direct: a runtime per library on its own events (the split done here, untimed)
        position = np.asarray(libs.flavor_arguments(c)["flavor_diff_xs_indices"])[flav]
        parts = []
        for k, me in enumerate(libs.matrix_elements(c, SAMPLED_IN, SAMPLED_OUT)):
            idx = np.nonzero(position == k)[0]
            if len(idx):
                parts.append((ms.FunctionRuntime(me.function(), context),
                              [device_arrays(x) for x in (mom[idx], np.zeros(len(idx), np.int32), r[idx])]))
        total["direct"] += best(lambda: [rt_.call(p) for rt_, p in parts], a.repeats)
        if stock is not None:
            srt = stock_runtime(context, stock, libs.particle_count, c, SAMPLED_IN, SAMPLED_OUT)
            total["stock"] += best(lambda: srt.call(args), a.repeats)
    n = a.events * libs.channels
    us = {k: 1e6 * v / n for k, v in total.items()}
    line = (f"{os.path.basename(os.path.dirname(os.path.abspath(table_path)))} {table['device']} {table.get('precision', 'f64')}"
            f" threads {a.threads}: {libs.channels} channels x {a.events} events, {len(libs.apis)} libraries; "
            f"µs/event runtime {us['runtime']:.3f}, direct {us['direct']:.3f} (runtime/direct {us['runtime'] / us['direct']:.2f})")
    if stock is not None:
        line += f", stock {us['stock']:.3f}"
    print(line, flush=True)


if __name__ == "__main__":
    main()
