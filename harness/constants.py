"""Read a `constants.json` from a zk-results checkout.

The path is never a default: pass `--constants PATH` on the command line or set
`ZK_CONSTANTS`.  Scripts with an argument parser declare the option through
`add_argument`; scripts without one call `load()` and get the same resolution.
Nothing here writes or re-derives a number.
"""
import json
import os
import sys

ENV = "ZK_CONSTANTS"
_PATH = None
_DOC = None


def _resolve(path):
    if path:
        return path
    argv = sys.argv[1:]
    if "--constants" in argv and argv.index("--constants") + 1 < len(argv):
        return argv[argv.index("--constants") + 1]
    for a in argv:
        if a.startswith("--constants="):
            return a.split("=", 1)[1]
    if os.environ.get(ENV):
        return os.environ[ENV]
    raise SystemExit("constants.json needed: pass --constants PATH (a zk-results checkout's "
                     f"constants.json) or set {ENV}")


def load(path=None):
    """Read the file once; later calls to `get`/`value` use it."""
    global _PATH, _DOC
    _PATH = _resolve(path)
    with open(_PATH) as f:
        _DOC = json.load(f)
    return _DOC


def doc():
    return _DOC if _DOC is not None else load()


def path():
    doc()
    return _PATH


def get(key):
    """One entry: value, unit, card, context, timing_scope, grade, run, artifact, note."""
    c = doc()["constants"]
    if key not in c:
        near = sorted(k for k in c if k.split(".")[0] == key.split(".")[0])[:8]
        raise KeyError(f"{key!r} is not in {_PATH}; same group: {near or 'none'}")
    return c[key]


def value(key):
    """The bare value; a `null` raises rather than becoming a guess."""
    e = get(key)
    if e["value"] is None:
        raise ValueError(f"{key!r} has no value: {e.get('note', '')}")
    return e["value"]


def keys(prefix=""):
    return sorted(k for k in doc()["constants"] if k.startswith(prefix))


def add_argument(parser):
    parser.add_argument("--constants", metavar="PATH", default=os.environ.get(ENV),
                        help=f"constants.json from a zk-results checkout (or set {ENV})")
