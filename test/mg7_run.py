"""An mg7 run of an output's [amplicol] subprocesses on PacRunner's UMAMI libraries, without
changing MadGraph7: the AmplicolSubprocess class is replaced, in this process only, by one that
loads the libraries of umami_libraries (one per program, flavour 0) and lets MadSpace split each
batch by flavour (one DifferentialCrossSection per library of the UMAMI channel).

  python mg7_run.py prepare OUTPUT [--device cpu|cuda] [--julia JULIA]
      the recurrence export of every SubProcesses/*/lc artifact and, per UMAMI provider, the
      libraries in OUTPUT/lib/pacrunner/<name>_<provider>/umami_pac.json
  python mg7_run.py run OUTPUT [generate_events arguments, default -f]
      bin/generate_events with the replaced subprocess class (stock libraries still built: the
      full-colour pass keeps the stock ratio library)

Run with the python of the output (the shebang of OUTPUT/bin/generate_events). The full-colour
pass gets MadSpace's matrix_element_flavor, the remapped flavour: with several flavours per UMAMI
channel only color_accuracy = "lc" is right for now (checked below)."""

import glob
import json
import os
import re
import subprocess
import sys

PAC_JULIA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mg5_dir(output):
    text = open(os.path.join(output, "bin", "generate_events")).read()
    return re.search(r"sys\.path\.append\('([^']*)'\)", text).group(1)


def provider_metadata(lc):
    """The (provider id, metadata path) of an artifact's providers, as export_amplicol reads them."""
    root = os.path.join(lc, "API", "umami")
    index = json.load(open(os.path.join(root, "metadata.json")))
    if index.get("kind") != "pyamplicol-umami-provider-index":
        return [(None, os.path.join(root, "metadata.json"))]
    return [(p.get("id", p.get("provider")), os.path.join(root, p["metadata"])) for p in index["providers"]]


def table_dir(output, me_path):
    """lib/libamplicol_<name>_<provider>.so -> lib/pacrunner/<name>_<provider>"""
    stem = re.fullmatch(r"libamplicol_(.*)\.so", os.path.basename(me_path)).group(1)
    return os.path.join(output, "lib", "pacrunner", stem)


def card_parameters(output, references):
    """{name: value} of the param card, as subprocess_amplicol.amplicol_parameters reads it."""
    if mg5_dir(output) not in sys.path:
        sys.path.append(mg5_dir(output))
    from madgraph.models.check_param_card import ParamCard
    card = ParamCard(os.path.join(output, "Cards", "param_card.dat"))
    parameters = {}
    for name, (block, code) in references.items():
        try:
            parameters[name] = float(card.get_value(block.lower(), tuple(code)))
        except (KeyError, ValueError):
            continue
    return parameters


def prepare(output, device="cpu", julia="julia"):
    output = os.path.abspath(output)
    entries = json.load(open(os.path.join(output, "SubProcesses", "subprocesses.json")))
    entries = entries.get("subprocesses", entries) if isinstance(entries, dict) else entries
    # one table per (artifact, provider): the entries of a provider share their me_path
    jobs = {}
    for entry in entries:
        if entry.get("backend") != "amplicol":
            continue
        lc = os.path.join(output, entry["artifact"])
        jobs.setdefault(entry["me_path"], (lc, entry["parameters"]))
    for me_path, (lc, references) in sorted(jobs.items()):
        dir_ = table_dir(output, me_path)
        os.makedirs(dir_, exist_ok=True)
        # resolved with the values mg7 gives umami_set_parameter (amplicol_parameters): the param
        # card's, which differ from the model defaults in the last digits
        parameters_path = os.path.join(dir_, "parameters.json")
        json.dump(card_parameters(output, references), open(parameters_path, "w"), indent=1)
        export = os.path.join(dir_, "export.json")
        env = dict(os.environ, PYTHONPATH=os.path.join(mg5_dir(output), "pyamplicol", "install"))
        subprocess.check_call([sys.executable, "-m", "pyamplicol", "export-recurrence", lc, "-o", export,
                               "--model-parameters", parameters_path], env=env)
        providers = provider_metadata(lc)
        stem = os.path.basename(dir_)
        match = [m for pid, m in providers if pid is None or stem.endswith("_" + str(pid))]
        if len(match) != 1:
            raise RuntimeError(f"{me_path}: no single provider metadata among {providers}")
        script = ("using PacRunner; umami_libraries(ARGS[1], ARGS[2], ARGS[3]; device = Symbol(ARGS[4])); "
                  "println(ARGS[3])")
        subprocess.check_call([julia, f"--project={PAC_JULIA}", "-e", script, export, match[0], dir_, device])


def install(output):
    """Replace AmplicolSubprocess (in madgraph_process, which instantiates it) by PacSubprocess."""
    import madspace as ms
    import madgraph.mg7.madgraph_process as mp
    import madgraph.mg7.subprocess_madmatrix as smm
    from madgraph.mg7.subprocess_amplicol import AmplicolSubprocess, amplicol_parameters

    class PacSubprocess(AmplicolSubprocess):
        def load_matrix_elements(self, api_path_formats, member_metas):
            super().load_matrix_elements(api_path_formats, member_metas)  # checks, stock lib, ratio lib
            dir_ = table_dir(output, self.meta["me_path"])
            table = json.load(open(os.path.join(dir_, "umami_pac.json")))
            channel = table["channels"][self.channel_index]
            libs = channel["libraries"]
            flavors = self.meta["flavors"]
            if len(libs) > 1 and self.ratio_matrix_element is not None:
                raise RuntimeError("PacRunner libraries: several flavours per UMAMI channel need "
                                   "color_accuracy = \"lc\" (the ratio pass gets the remapped flavour)")
            parameters = amplicol_parameters(self.meta, self.process)
            self.pac_libraries = libs
            self.pac_matrix_elements = [
                self.load(os.path.join(dir_, table["libraries"][k]["path"]), self.meta["artifact"], parameters)
                for k in libs]
            # per mg7 flavour: its library's position among the channel's
            self.pac_diff_xs = [libs.index(channel["flavor_libraries"][f["index"]]) for f in flavors]
            print(f"PacRunner: subprocess {self.subproc_id} channel {self.channel_index}: "
                  f"{len(flavors)} flavours on {len(libs)} libraries", flush=True)

        def integrand_matrix_elements(self):
            stock = super().integrand_matrix_elements()[0]
            return [ms.MatrixElement(api.index(), self.particle_count, stock.inputs(), stock.outputs(), 1,
                                     True, self.process.me_frame, self.process.incoming_count,
                                     self.channel_index)
                    for api in self.pac_matrix_elements]

        def build_integrands(self, *args, **kwargs):
            n = len(self.meta["flavors"])
            remap = [0] * n
            diff_xs = self.pac_diff_xs
            subproc = [self.subproc_id] * n
            per_subproc = list(range(n))
            stock_remap = [f["index"] for f in self.meta["flavors"]]
            real = ms.Integrand

            def integrand(*a, **kw):
                a = list(a)
                # positional flavor_remap, flavor_factors, flavor_mirror, flavor_diff_xs_indices,
                # flavor_subproc_indices, flavor_per_subproc_remap (subprocess_madmatrix.build_integrands)
                assert a[21] == stock_remap and a[24] == [] and a[25] == [] and a[26] == [], "Integrand layout"
                a[21] = remap
                if len(self.pac_matrix_elements) > 1:
                    a[24], a[25], a[26] = diff_xs, subproc, per_subproc
                return real(*a, **kw)

            smm.ms.Integrand = integrand
            try:
                return super().build_integrands(*args, **kwargs)
            finally:
                smm.ms.Integrand = real

    mp.AmplicolSubprocess = PacSubprocess


def run(output, args):
    output = os.path.abspath(output)
    sys.path.append(mg5_dir(output))
    from madgraph.mg7 import bootstrap
    bootstrap.ensure_madspace(interactive=False)  # MadGraph7's madspace first on sys.path
    install(output)
    from madgraph.mg7.launch import main
    os.chdir(output)
    sys.argv = [os.path.join(output, "bin", "generate_events")] + (args or ["-f"])
    main()


if __name__ == "__main__":
    command, output, *rest = sys.argv[1:]
    if command == "prepare":
        opts = dict(zip(rest[::2], rest[1::2]))
        prepare(output, opts.get("--device", "cpu"), opts.get("--julia", "julia"))
    elif command == "run":
        run(output, rest)
    else:
        sys.exit(__doc__)
