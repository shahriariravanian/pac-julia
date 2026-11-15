"""pyAmpliCol with MadGraph's MadSpace, in four steps, on u u~ > z g g at leading colour:

  1. generate   pyAmpliCol writes the process artifact, with its API bundle (`pyamplicol generate`);
  2. compile    the bundle's UMAMI provider is built into a shared library (`make` in API/umami:
                libumami.so, beside its C driver umami_driver);
  3. load       MadSpace loads the library (`Context.load_matrix_element`; the UMAMI library takes the
                absolute artifact directory as its "parameter card");
  4. evaluate   MadSpace evaluates the matrix element at the artifact's validation point
                (API/validation_points.dat), and the bundle's C driver evaluates the same point.

The two values come from the same library at the same point, so they must agree bit for bit.

usage: python umami_example.py [WORKDIR]     (default: ./umami_example; an existing artifact is reused)

Run it with the pyAmpliCol environment's python (its bin/ holds `pyamplicol` and `rusticol-config`),
MadSpace on PYTHONPATH (e.g. MadGraph7/madspace/install) and Symbolica usable (pyAmpliCol generates
with it; here XDG_RUNTIME_DIR=$HOME/.cache/xdg-runtime):

  PYTHONPATH=~/mg7-pr452/madspace/install ~/pyamplicol-1.0/.venv/bin/python test/umami_example.py"""

import glob
import json
import os
import subprocess
import sys

import numpy as np

import madspace as ms

PROCESS = "u u~ > z g g"
NAME = "zgg"
BIN = os.path.dirname(sys.executable)   # the pyAmpliCol environment's tools


def generate(workdir):
    """1. the artifact of PROCESS (LC), with its API bundle; returns its absolute path"""
    artifact = os.path.abspath(os.path.join(workdir, NAME))
    if not os.path.isfile(os.path.join(artifact, "artifact.json")):
        os.makedirs(workdir, exist_ok=True)
        subprocess.run([os.path.join(BIN, "pyamplicol"), "generate", PROCESS, artifact, "--name", NAME,
                        "--color-accuracy", "lc", "--emit-api-bundle", "--progress", "off"], check=True)
    return artifact


def compile_umami(artifact):
    """2. the UMAMI provider built: (libumami.so, umami_driver) in the sibling .pyamplicol-api-build"""
    umami = os.path.join(artifact, "API", "umami")
    env = dict(os.environ, RUSTICOL_CONFIG=os.path.join(BIN, "rusticol-config"))
    subprocess.run(["make", "-s", "-C", umami], check=True, env=env)
    build = os.path.join(os.path.dirname(artifact), ".pyamplicol-api-build", os.path.basename(artifact), "umami")
    library = sorted(glob.glob(os.path.join(build, "libumami*.so")) + glob.glob(os.path.join(build, "libumami*.dylib")))[0]
    driver = sorted(glob.glob(os.path.join(build, "umami_driver*")))[0]
    return library, driver


def validation_point(artifact):
    """the process's saved point: (particles, momenta[particle][E, px, py, pz])"""
    with open(os.path.join(artifact, "API", "validation_points.dat")) as f:
        assert f.readline().strip() == "RUSTICOL_VALIDATION_POINTS_V1"
        for line in f:
            fields = line.split()
            if fields and fields[0] == NAME:
                n = int(fields[1])
                return n, np.array([float(x) for x in fields[2:2 + 4 * n]]).reshape(n, 4)
    raise RuntimeError(f"no validation point for {NAME}")


def madspace_matrix_element(library, artifact, momenta, channel=0, flavour=0):
    """3. and 4. the library in MadSpace, the matrix element at one point (channel, flavour 0)"""
    context = ms.Context(ms.cpu_device(), 1)
    api = context.load_matrix_element(library, artifact, {})
    ME = ms.MatrixElement
    # by the library's index in the context (the overload taking `api` itself asks the library for
    # metadata the stock provider does not report); the channel is fixed here, not a call argument
    me = ME(api.index(), len(momenta), [ME.momenta_in, ME.flavor_in, ME.channel_in], [ME.matrix_element_out],
            diagram_count=1, sample_random_inputs=False, me_frame=[], incoming_count=2, channel=channel)
    runtime = ms.FunctionRuntime(me.function(), context)
    batch = momenta[None, :, :]                       # one event: (events, particles, 4)
    flavours = np.full(1, flavour, np.int32)
    (result,) = runtime.call([batch, flavours])
    return float(np.array(result.numpy())[0])


def driver_matrix_element(driver, artifact, channel=0, flavour=0):
    """the bundle's C driver at the same point: its JSON line"""
    out = subprocess.run([driver, artifact, str(channel), str(flavour)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])["matrix_element"]


def main(workdir="umami_example"):
    artifact = generate(workdir)
    library, driver = compile_umami(artifact)
    n, momenta = validation_point(artifact)
    print(f"{PROCESS}: artifact {artifact}")
    print(f"  library {library}")
    print(f"  validation point ({n} particles, E px py pz):")
    for p in momenta:
        print("    " + "  ".join(f"{x:22.15f}" for x in p))
    me = madspace_matrix_element(library, artifact, momenta)
    reference = driver_matrix_element(driver, artifact)
    same = np.float64(me).view(np.uint64) == np.float64(reference).view(np.uint64)
    print(f"  MadSpace:     |M|^2 = {me!r}")
    print(f"  umami_driver: |M|^2 = {reference!r}  ({'bit for bit' if same else 'DIFFERENT'})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
