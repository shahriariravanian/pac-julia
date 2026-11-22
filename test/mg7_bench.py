"""mg7_bench.py OUTPUT [--events N] [--batch B] [--modes stock,cpu,metal] [--repeat R] [--no-build]
                     [--no-timer] [--tag=TAG]

An mg7 output's generation on the stock pyAmpliCol library, PacRunner's CPU (f64) libraries and its
Metal libraries: per mode a warm-up run, then R timed runs (same seed). Every run: sigma, the run's
phases (survey, generate, combine: wall and CPU seconds, from info.json), the phase-space points
evaluated, and the matrix element's own time (test/umami_timer.c in front of every library, the
stock one too: calls, events, seconds summed over MadSpace's threads). Writes OUTPUT/bench<TAG>.json.
The timer shims are built and loaded at the run's start (some seconds for many libraries): the whole
run's wall time is for --no-timer runs, the phases' and the matrix element's for either.

Modes: stock (the output's own library), cpu (PacRunner f64), metal (PacRunner on the Apple GPU);
the card is set as for the Linux benchmark: fixed scales, no systematics, no MadNIS, LC colour,
run.device = ["cpu"]. `mg7_bench.py stock-run OUTPUT` is the stock mode's process (internal).
"""
import argparse, glob, json, os, re, subprocess, sys, time

PAC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PAC, "test"))
import mg7_example, mg7_run  # noqa: E402



def last_info(output):
    runs = sorted(glob.glob(os.path.join(output, "Events", "*", "info.json")), key=os.path.getmtime)
    return runs[-1], json.load(open(runs[-1]))


def timer_totals(log):
    calls = events = me = init = 0.0
    libs = 0
    if os.path.exists(log):
        for line in open(log):
            f = line.split()
            if len(f) >= 6:
                libs += 1
                calls += float(f[2]); events += float(f[3]); me += float(f[4]); init += float(f[5])
    return dict(libraries=libs, calls=calls, events=events, me_seconds=me, init_seconds=init)


def run_once(output, mode, label, timer=True):
    log = os.path.join(output, f"bench-{label}.timer")
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, PAC_UMAMI_TIMER=log) if timer else {k: v for k, v in os.environ.items() if k != "PAC_UMAMI_TIMER"}
    if mode == "stock":
        cmd = [sys.executable, os.path.abspath(__file__), "stock-run", output]
    else:
        cmd = [sys.executable, os.path.join(PAC, "test", "mg7_run.py"), "run", output, "--device", mode, "-f"]
    start = time.time()
    with open(os.path.join(output, f"bench-{label}.log"), "w") as out:
        status = subprocess.call(cmd, cwd=output, stdout=out, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - start
    if status:
        raise RuntimeError(f"{label}: exit {status}, see {output}/bench-{label}.log")
    path, info = last_info(output)
    p = info["process"]
    if not p["done"]:
        raise RuntimeError(f"{label}: run {path} not done")
    rt = info["run_times"]
    r = dict(label=label, mode=mode, run=os.path.dirname(path), sigma=p["mean"], error=p["error"],
             points=p["count"], process_wall=wall,
             phases={k: dict(wall=v["wall_time_sec"], cpu=v["cpu_time_sec"]) for k, v in rt.items()},
             me=timer_totals(log))
    r["phase_wall"] = sum(v["wall"] for v in r["phases"].values())
    return r


# ---- the stock mode's process: generate_events with every stock library behind the timer ----

def stock_timed(path):
    log = os.environ.get("PAC_UMAMI_TIMER")
    if not log:
        return path
    stem = re.sub(r"\.(so|dylib)$", "", os.path.basename(path))
    shim = os.path.join(os.path.dirname(path), f"timer_{stem}.so")
    subprocess.check_call(["cc", "-O2", "-std=c11", "-fPIC", "-shared", f"-I{os.path.join(PAC, 'c')}",
                           f'-DREAL_LIB="{os.path.abspath(path)}"', f'-DLOG_FILE="{os.path.abspath(log)}"',
                           f'-DNAME="stock/{stem}"', os.path.join(PAC, "test", "umami_timer.c"), "-o", shim])
    return shim


def stock_run(output):
    output = os.path.abspath(output)
    sys.path.append(mg7_run.mg5_dir(output))
    from madgraph.mg7 import bootstrap
    bootstrap.ensure_madspace(interactive=False)
    from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess
    real_load = AmplicolSubprocess.load
    AmplicolSubprocess.load = lambda self, path, artifact, parameters: real_load(self, stock_timed(path), artifact, parameters)
    from madgraph.mg7.launch import main as launch
    os.chdir(output)
    sys.argv = [os.path.join(output, "bin", "generate_events"), "-f"]
    launch()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--events", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=64000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--modes", default="stock,cpu,metal")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-timer", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    output = os.path.abspath(a.output)
    modes = a.modes.split(",")
    mg7_example.edit_card(output, {"events": a.events, "seed": a.seed, "make_plots": "false",
                                   "color_accuracy": '"lc"', "fixed_ren_scale": "true", "fixed_fact_scale": "true",
                                   "cpu_batch_size": a.batch, "gpu_batch_size": a.batch,
                                   "max_batch_size": max(a.batch, 32000), "device": '["cpu"]'},
                          [("systematics", "enable", "false"), ("madnis", "enable", "false")])
    if not a.no_build:
        for mode in modes:
            if mode == "stock":
                continue
            start = time.time()
            mg7_run.prepare(output, mode)
            print(f"built {mode} libraries in {time.time() - start:.1f} s", flush=True)
    results = []
    for mode in modes:
        print(f"{mode}: warm-up", flush=True)
        warm = run_once(output, mode, f"{mode}-warm{a.tag}", not a.no_timer)
        warm["warmup"] = True
        results.append(warm)
        for k in range(a.repeat):
            r = run_once(output, mode, f"{mode}-{k}{a.tag}", not a.no_timer)
            r["warmup"] = False
            results.append(r)
            me = r["me"]
            print(f"  {mode}: sigma {r['sigma']:.10g} +- {r['error']:.3g}; wall {r['process_wall']:.1f} s "
                  f"(phases {r['phase_wall']:.1f} s), {r['points']} points; ME {me['me_seconds']:.1f} s over "
                  f"{me['events']:.0f} events in {me['libraries']} libraries", flush=True)
    json.dump(dict(events=a.events, batch=a.batch, seed=a.seed, results=results),
              open(os.path.join(output, f"bench{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["stock-run"]:
        stock_run(sys.argv[2])
    else:
        main()
