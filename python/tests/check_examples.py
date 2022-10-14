import os
import sys


EXAMPLES_SCRIPTS_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)

EXAMPLES_DATA_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "rascaline", "examples", "data")
)

EXAMPLES_ARGS = {
    "compute-soap.py": [os.path.join(EXAMPLES_DATA_PATH, "water.xyz")],
    "profiling.py": [os.path.join(EXAMPLES_DATA_PATH, "water.xyz")],
}

if __name__ == "__main__":
    # run python example while re-directing all output to /dev/null
    devnull = open(os.devnull, "w")
    sys.stdout = devnull

    for script, args in EXAMPLES_ARGS.items():
        print(f"running {script} {' '.join(args)}", file=sys.stderr)
        with open(os.path.join(EXAMPLES_SCRIPTS_PATH, script), encoding="utf8") as fd:
            sys.argv = [script] + args
            exec(fd.read())

    devnull.close()
