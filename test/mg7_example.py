#!/usr/bin/env python3
"""An end-to-end example of PacRunner (pac-julia) in MadGraph7: from a process to its cross section.

  1. MadGraph7 generates the process with pyAmpliCol ([amplicol]) and writes an mg7 output;
  2. pyAmpliCol exports the output's recurrence programs (`pyamplicol export-recurrence`), and
     PacRunner, called from Python through juliacall, builds one UMAMI library per program
     (PacRunner.umami_libraries);
  3. MadSpace integrates and generates events on those libraries (test/mg7_run.py swaps them in for
     the stock pyAmpliCol library of the run);
  4. optionally (--stock) the same run on the stock library, for comparison.

usage:
  python mg7_example.py --mg7 MADGRAPH7 [--process 'g g > g g g'] [--output DIR] [--events N]
                        [--device cpu|cuda|cuda_provider] [--precision f64|mixed] [--color lc|full]
                        [--pdf SET] [--batch N] [--seed N] [--stock] [--keep-scales]

Needs:
  - MadGraph7 with pyAmpliCol's recurrence export (MadGraphTeam/MadGraph7 PR #452 or later), with
    MadSpace and pyAmpliCol installed in it (`install madspace`, `install pyamplicol`);
  - Julia (1.10 or later) and this repository; `pip install juliacall` in the python that runs
    MadGraph7 (the first run lets juliacall set up its Julia environment);
  - a C compiler; for --device cuda / cuda_provider the CUDA toolkit (nvcc on the PATH or in
    /usr/local/cuda/bin), and for cuda_provider a CUDA build of MadSpace in
    MADGRAPH7/madspace/install-cuda (cmake -DENABLE_CUDA=ON).

Devices: cpu: the programs as C on MadSpace's CPU threads; cuda: the programs on the GPU, MadSpace
on the CPU (full colour possible); cuda_provider: MadSpace itself on the GPU (leading colour only).
--precision mixed (GPU only): f32 currents with f64 denominators, about 1e-6 on the cross section.

The run card is edited as MadGraph7's own [amplicol] acceptance test edits it (fixed scales at the
Z mass, no systematics, no MadNIS, no plots; --keep-scales keeps the card's scale choice). The
libraries are built for the param card of the output: build them again after changing it.

With f64 the PacRunner libraries give the stock library's matrix elements bit for bit at the
exported alpha_s; at another alpha_s (the PDF set's, in a normal run) they are rescaled by its
power, which agrees to about 1e-11.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAC_JULIA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mg7_run  # noqa: E402  (test/mg7_run.py: the export, and the run on PacRunner's libraries)


def generate(mg7, process, output):
    """Step 1: `generate PROCESS [amplicol]` and `output mg7 OUTPUT` in MadGraph7."""
    sys.path.insert(0, mg7)
    import madgraph.interface.master_interface as interface
    cmd = interface.MasterCmd()
    cmd.no_notification()
    cmd.exec_cmd("set automatic_html_opening False --no_save")
    processes = [p.strip() for p in process.split(";")]
    for k, p in enumerate(processes):
        p = p if "[amplicol]" in p else p + " [amplicol]"
        cmd.exec_cmd(("generate " if k == 0 else "add process ") + p)
    cmd.exec_cmd(f"output mg7 {output}")


def edit_card(output, settings, sections=None):
    """Set `key = value` lines of Cards/run_card.toml (value: the TOML text); `sections`: (section,
    key, value) for keys that several sections have."""
    path = os.path.join(output, "Cards", "run_card.toml")
    card = open(path).read()
    for key, value in settings.items():
        card, n = re.subn(rf"(?m)^{re.escape(key)} = .*$", f"{key} = {value}", card)
        if n == 0:
            raise RuntimeError(f"run_card.toml has no '{key}'")
    for section, key, value in sections or []:
        card, n = re.subn(rf"(\[{re.escape(section)}\]\n(?:(?!\[).*\n)*?{re.escape(key)} = ).*", rf"\g<1>{value}", card, count=1)
        if n == 0:
            raise RuntimeError(f"run_card.toml has no [{section}] {key}")
    open(path, "w").write(card)


def pacrunner():
    """PacRunner in this process, through juliacall (with the Julia on the PATH, if any)."""
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    julia = shutil.which("julia")
    if julia:
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", julia)
    from juliacall import Main as jl
    jl.seval("import Pkg")
    jl.Pkg.activate(PAC_JULIA)     # this repository's project
    jl.Pkg.instantiate()
    jl.seval("using PacRunner")
    return jl


def build_libraries(output, device, precision):
    """Step 2: the recurrence export of each [amplicol] provider and its PacRunner libraries."""
    jl = pacrunner()

    def build(export, metadata, folder, device_, precision_):
        table = jl.PacRunner.umami_libraries(export, metadata, folder, device=jl.Symbol(device_),
                                             precision=jl.Symbol(precision_), quiet=True)
        count = len(json.load(open(str(table)))["libraries"])
        print(f"  {count} libraries in {folder}", flush=True)

    return mg7_run.prepare(output, device, precision=precision, build=build)


def result(output):
    """(cross section, error, events directory) of the output's last run."""
    runs = sorted(glob.glob(os.path.join(output, "Events", "*", "info.json")), key=os.path.getmtime)
    if not runs:
        raise RuntimeError(f"no run in {output}/Events")
    process = json.load(open(runs[-1]))["process"]
    if not process["done"]:
        raise RuntimeError(f"the run {os.path.dirname(runs[-1])} did not finish")
    return float(process["mean"]), float(process["error"]), os.path.dirname(runs[-1])


def launch(output, command, log):
    """A generation in its own process; returns (cross section, error, directory, seconds)."""
    start = time.time()
    with open(log, "w") as stream:
        status = subprocess.call(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
    if status != 0:
        raise RuntimeError(f"{' '.join(command)} failed: see {log}\n" + open(log).read()[-2000:])
    return (*result(output), time.time() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mg7", required=True, help="the MadGraph7 directory")
    parser.add_argument("--process", default="g g > g g g", help="the process ('a ; b' for several)")
    parser.add_argument("--output", default=None, help="the mg7 output directory (default: ./pac_<process>)")
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda_provider"])
    parser.add_argument("--precision", default="f64", choices=["f64", "mixed"])
    parser.add_argument("--color", default="lc", choices=["lc", "full"], help="colour accuracy of the run")
    parser.add_argument("--pdf", default=None, help="the LHAPDF set of both beams (default: the card's)")
    parser.add_argument("--batch", type=int, default=None, help="MadSpace's batch size (default: the card's)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stock", action="store_true", help="also run the stock pyAmpliCol library")
    parser.add_argument("--keep-scales", action="store_true", help="keep the card's scale choice")
    args = parser.parse_args()
    if args.precision != "f64" and args.device == "cpu":
        parser.error("--precision mixed needs --device cuda or cuda_provider")
    if args.color != "lc" and args.device == "cuda_provider":
        parser.error("--device cuda_provider runs leading colour only")
    mg7 = os.path.abspath(args.mg7)
    name = re.sub(r"[^A-Za-z0-9]+", "_", args.process).strip("_")
    output = os.path.abspath(args.output or f"pac_{name}")

    if os.path.isdir(os.path.join(output, "SubProcesses")):
        print(f"1. using the existing output {output}", flush=True)
    else:
        print(f"1. generating {args.process} [amplicol] into {output}", flush=True)
        generate(mg7, args.process, output)
    settings = {"events": args.events, "seed": args.seed, "make_plots": "false", "color_accuracy": f'"{args.color}"'}
    if not args.keep_scales:
        settings.update({"fixed_ren_scale": "true", "fixed_fact_scale": "true"})
    if args.pdf:
        settings.update({"pdf1": f'"{args.pdf}"', "pdf2": f'"{args.pdf}"'})
    if args.batch:
        settings.update({"cpu_batch_size": args.batch, "gpu_batch_size": args.batch, "max_batch_size": max(args.batch, 32000)})
    edit_card(output, settings, [("systematics", "enable", "false"), ("madnis", "enable", "false")])

    print(f"2. recurrence export and PacRunner libraries ({args.device}, {args.precision})", flush=True)
    start = time.time()
    build_libraries(output, args.device, args.precision)
    print(f"  built in {time.time() - start:.1f} s", flush=True)

    stock = None
    if args.stock:
        print("3a. the run on the stock pyAmpliCol library (CPU; the first run builds it)", flush=True)
        edit_card(output, {"device": '["cpu"]'})
        command = [sys.executable, os.path.join(output, "bin", "generate_events"), "-f"]
        launch(output, command, os.path.join(output, "stock-build.log"))
        stock = launch(output, command, os.path.join(output, "stock.log"))

    variant = args.device if args.precision == "f64" else f"{args.device}-{args.precision}"
    print(f"3. the run on PacRunner's libraries ({variant})", flush=True)
    edit_card(output, {"device": '["cuda"]' if args.device == "cuda_provider" else '["cpu"]'})
    ours = launch(output, [sys.executable, os.path.join(HERE, "mg7_run.py"), "run", output, "--device", variant, "-f"],
                  os.path.join(output, "pacrunner.log"))

    print()
    print(f"{args.process}, {args.events} events, colour accuracy {args.color}")
    if stock:
        print(f"  stock      sigma = {stock[0]:.10g} +- {stock[1]:.3g} pb   ({stock[3]:.1f} s)   {stock[2]}")
    print(f"  PacRunner  sigma = {ours[0]:.10g} +- {ours[1]:.3g} pb   ({ours[3]:.1f} s)   {ours[2]}")
    if stock:
        difference = abs(ours[0] - stock[0])
        print(f"  difference: {difference / abs(stock[0]):.2e} relative, {difference / stock[1]:.2g} statistical errors")
        if args.device == "cuda_provider":
            print("  (MadSpace samples differently on the GPU: the runs agree within the statistical error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
