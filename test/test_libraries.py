"""utils/libraries.py (load_libraries) against the stock pyAmpliCol library, through MadSpace: for
every UMAMI channel, a batch of events of mixed flavours (random on-shell momenta, a random
flavour of the channel per event, random helicity numbers) goes to the stock library (each event
with its flavour) and to PacRuntime (which splits the batch by flavour over the PacRunner
libraries). Compared bit for bit, in the events' order: the LC pass (ME, colour and helicity
indices) and the fixed-helicity pass (ME at every exported helicity).

usage: python test_libraries.py OUT [STOCK_LIB ARTIFACT_DIR] [EVENTS]
  OUT:        umami_libraries' output folder (or its umami_pac.json)
  STOCK_LIB:  the stock library (default: the mg7 output's lib/libamplicol_<name>.so beside
              OUT = <output>/lib/pacrunner/<name>); ARTIFACT_DIR: the artifact (default: from the
              table's metadata path, .../API/umami/metadata.json)
(MadSpace on PYTHONPATH, e.g. MadGraph7/madspace/install)"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
import madspace as ms  # noqa: E402
from libraries import load_libraries  # noqa: E402

ME = ms.MatrixElement


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
    context = ms.Context(ms.cpu_device(), 1)
    libs = load_libraries(context, out)
    artifact = artifact or libs.artifact
    # the stock library at the model point the libraries were exported at (an mg7 output: the card's)
    stock = context.load_matrix_element(stock_path, os.path.abspath(artifact), libs.parameters)
    rng = np.random.default_rng(20261115)
    sampled_in = [ME.momenta_in, ME.flavor_in, ME.random_helicity_in]
    sampled_out = [ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out]
    fixed_in = [ME.momenta_in, ME.flavor_in, ME.helicity_in]
    bad, flavours_seen = [], 0
    for c in range(libs.channels):
        nf = libs.flavours(c)
        mom = momenta(masses, events, rng)
        flav = rng.integers(0, nf, events).astype(np.int32)
        flavours_seen += len(np.unique(flav))
        r = rng.uniform(0.0, 1.0, events)
        pac = libs.runtime(c, sampled_in, sampled_out)
        s = stock_runtime(context, stock, n, c, sampled_in, sampled_out)
        a = [np.array(x.numpy()) for x in s.call([mom, flav, r])]
        b = pac.call([mom, flav, r])
        if not all(same(x, y) for x, y in zip(a, b)):
            bad.append(f"channel {c}: sampled pass differs (ME max rel. "
                       f"{np.max(np.abs(a[0] - b[0]) / np.abs(a[0])):.2e})")
        pac_fixed = libs.runtime(c, fixed_in, [ME.matrix_element_out])
        s_fixed = stock_runtime(context, stock, n, c, fixed_in, [ME.matrix_element_out])
        for h in range(exported):
            hel = np.full(events, h, np.int32)
            (x,) = [np.array(t.numpy()) for t in s_fixed.call([mom, flav, hel])]
            (y,) = pac_fixed.call([mom, flav, hel])
            if not same(x, y):
                bad.append(f"channel {c}: fixed helicity {h} differs")
                break
    print(f"{os.path.dirname(os.path.abspath(table_path))}: {libs.channels} channels, {len(libs.apis)} libraries, "
          f"{events} events of mixed flavours per channel ({flavours_seen} flavours in all): "
          + ("sampled and fixed-helicity passes bit for bit" if not bad else "FAILURES"))
    for line in bad[:10]:
        print("  " + line)
    return 0 if not bad else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    events = int(args.pop()) if len(args) in (2, 4) and args[-1].isdigit() else 256
    sys.exit(main(*args, events=events))
