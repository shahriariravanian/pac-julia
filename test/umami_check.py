"""PacRunner's UMAMI libraries (umami_libraries) against the stock pyAmpliCol library, through
MadSpace: for every (channel, flavour) of the provider, the same events (random on-shell momenta in
the input leg order, random helicity numbers) go to the stock library (that channel and flavour) and
to the flavour's PacRunner library (flavour 0), each as a MadSpace MatrixElement. Compared bit for bit:

  sampled   ME, colour index and helicity index (the LC pass: random helicity in, indices out);
  summed    ME without index outputs: informational (stock takes this sum over helicity orbits in
            RustiCol, which the export does not describe), the largest relative difference;
  fixed     ME at each exported helicity (the reweight pass's helicity input).

usage: python umami_check.py UMAMI_PAC_JSON STOCK_LIBUMAMI_SO ARTIFACT_DIR [EVENTS]
(MadSpace on PYTHONPATH, e.g. MadGraph7/madspace/install)"""

import json
import os
import sys

import numpy as np

import madspace as ms

ME = ms.MatrixElement


def momenta(masses, events, rng):
    p = rng.uniform(-300.0, 300.0, size=(events, len(masses), 3))
    e = np.sqrt((p * p).sum(axis=2) + np.asarray(masses) ** 2)
    return np.concatenate([e[:, :, None], p], axis=2)


def runtime(context, api, n, channel, inputs, outputs):
    me = ME(api.index(), n, inputs, outputs, 1, False, [], 2, channel)
    return ms.FunctionRuntime(me.function(), context)


def call(rt, args):
    return [np.array(x.numpy()) for x in rt.call(args)]


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and (
        np.array_equal(a.view(np.uint64), b.view(np.uint64)) if a.dtype == np.float64 else np.array_equal(a, b))


def main(table_path, stock_path, artifact, events=64):
    table = json.load(open(table_path))
    base = os.path.dirname(table_path)
    meta = json.load(open(table["metadata"]))
    provider = meta["provider"]
    n = int(provider["particle_count"])
    masses = provider["masses"]
    exported = len(meta["helicities"])
    context = ms.Context(ms.cpu_device(), 1)
    stock = context.load_matrix_element(stock_path, os.path.abspath(artifact), {})
    libs = {}
    rng = np.random.default_rng(20261111)
    sampled_in = [ME.momenta_in, ME.flavor_in, ME.channel_in, ME.random_helicity_in]
    sampled_out = [ME.matrix_element_out, ME.color_index_out, ME.helicity_index_out]
    fixed_in = [ME.momenta_in, ME.flavor_in, ME.channel_in, ME.helicity_in]
    counts = {"sampled": [0, 0], "summed": [0, 0], "fixed": [0, 0]}
    summed_worst = 0.0
    bad = []
    for c, channel in enumerate(table["channels"]):
        rts = {}
        for f, k in enumerate(channel["flavor_libraries"]):
            if k not in libs:
                libs[k] = context.load_matrix_element(os.path.join(base, table["libraries"][k]["path"]), "", {})
            key = ("stock", c)
            if key not in rts:
                rts[key] = [runtime(context, stock, n, c, sampled_in, sampled_out),
                            runtime(context, stock, n, c, sampled_in[:3], [ME.matrix_element_out]),
                            runtime(context, stock, n, c, fixed_in, [ME.matrix_element_out])]
            if ("pac", k) not in rts:
                rts[("pac", k)] = [runtime(context, libs[k], n, c, sampled_in, sampled_out),
                                   runtime(context, libs[k], n, c, sampled_in[:3], [ME.matrix_element_out]),
                                   runtime(context, libs[k], n, c, fixed_in, [ME.matrix_element_out])]
            s, p = rts[key], rts[("pac", k)]
            mom = momenta(masses, events, rng)
            flav = np.full(events, f, np.int32)
            zero = np.zeros(events, np.int32)
            r = rng.uniform(0.0, 1.0, events)
            label = f"channel {c} flavour {f} (library {k})"
            # sampled
            a, b = call(s[0], [mom, flav, r]), call(p[0], [mom, zero, r])
            ok = all(same(x, y) for x, y in zip(a, b))
            counts["sampled"][0] += 1
            counts["sampled"][1] += ok
            ok or bad.append(f"{label}: sampled, ME rel. diff {np.max(np.abs(a[0] - b[0]) / np.abs(a[0])):.2e}, "
                             f"colour {np.array_equal(a[1], b[1])}, helicity {np.array_equal(a[2], b[2])}")
            # summed
            a, b = call(s[1], [mom, flav]), call(p[1], [mom, zero])
            ok = same(a[0], b[0])
            counts["summed"][0] += 1
            counts["summed"][1] += ok
            summed_worst = max(summed_worst, float(np.max(np.abs(a[0] - b[0]) / np.abs(a[0]))))
            # fixed: every exported helicity
            ok = True
            for h in range(exported):
                hel = np.full(events, h, np.int32)
                a, b = call(s[2], [mom, flav, hel]), call(p[2], [mom, zero, hel])
                ok = ok and same(a[0], b[0])
            counts["fixed"][0] += 1
            counts["fixed"][1] += ok
            ok or bad.append(f"{label}: fixed helicity differs")
    flavours = sum(len(ch["flavor_libraries"]) for ch in table["channels"])
    print(f"{os.path.basename(base)}: {len(table['channels'])} channels, {flavours} flavours, "
          f"{len(table['libraries'])} libraries, {events} events each")
    for name, (total, good) in counts.items():
        print(f"  {name}: {good} of {total} flavours bit for bit"
              + (f" (largest relative difference {summed_worst:.1e})" if name == "summed" else ""))
    for line in bad[:10]:
        print("  " + line)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4], *(int(x) for x in sys.argv[4:5])))
