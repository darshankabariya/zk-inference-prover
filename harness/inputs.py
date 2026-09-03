"""Where a script reads measured artifacts from.

Nothing measured lives in this repository.  A script that reads run outputs
takes `--artifacts DIR` (or `ZK_ARTIFACTS`), a directory laid out as the
script's own docstring says; the cost model takes `--inputs DIR` (or
`ZK_COSTMODEL_INPUTS`) for its micro-benchmark outputs.  Both resolve the
command line first and the environment second, so scripts without an argument
parser get the same behaviour as scripts with one.
"""
import os
import pathlib
import sys


def _flag(name, env, what):
    argv = sys.argv[1:]
    if name in argv and argv.index(name) + 1 < len(argv):
        return pathlib.Path(argv[argv.index(name) + 1])
    for a in argv:
        if a.startswith(name + "="):
            return pathlib.Path(a.split("=", 1)[1])
    if os.environ.get(env):
        return pathlib.Path(os.environ[env])
    raise SystemExit(f"{what}: pass {name} DIR or set {env}")


def artifacts_dir():
    return _flag("--artifacts", "ZK_ARTIFACTS", "measured artifacts")


def costmodel_inputs():
    return _flag("--inputs", "ZK_COSTMODEL_INPUTS",
                 "cost-model inputs (commit.json, field.json, logup.json, fri.json and "
                 "the b1..b10 subdirectories the readers name)")


def add_arguments(parser, artifacts=True, inputs=False):
    if artifacts:
        parser.add_argument("--artifacts", metavar="DIR", default=os.environ.get("ZK_ARTIFACTS"),
                            help="directory of measured run outputs (or set ZK_ARTIFACTS)")
    if inputs:
        parser.add_argument("--inputs", metavar="DIR", default=os.environ.get("ZK_COSTMODEL_INPUTS"),
                            help="cost-model micro-benchmark outputs (or set ZK_COSTMODEL_INPUTS)")
