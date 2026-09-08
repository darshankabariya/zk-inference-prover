"""Install the generated constraint kernels into the fork (B7a milestone 2b, step 3).

Reads `$ZK_KERNEL_STORE/manifest.json` (written by `zk-prover cudaemit`) and applies three
asserted, idempotent edits to the fork's CUDA tree:

  1. copy the generated `evaluate_zk_*.cu` / `.cuh` and `zk_ops.cuh` into
     `crates/stwo/src/stwo_cuda/cuda/`;
  2. add each `.cu` to the explicit source list in `CMakeLists.txt` (the `.cuh`
     files are picked up by its GLOB_RECURSE, so only the sources need listing);
  3. add an `#include` and a `DISPATCH_EVAL_WITH_BOOLS` case per family to
     `evaluate_constraints.cu`, keyed on the same `cuda_eval_name()` milestone 2a
     already gave every eval -- so the fork's dispatch does not change shape, and
     a family that spans several constraint shapes picks the kernel inside its own
     launcher from the eval blob.

Re-running after a fresh `cudaemit` refreshes the files and leaves the edits
alone. `--undo` removes all three, restoring the pristine `41c2a05` dispatch.

    python3 install_constraint_kernels.py --fork DIR --gen DIR [--undo]
"""
import json
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]


def opt(name, default=None):
    if name in args and args.index(name) + 1 < len(args):
        return args[args.index(name) + 1]
    if default is None:
        sys.exit(f"{name} DIR is required; this script knows no host paths")
    return default


FORK = Path(opt("--fork"))
GEN = Path(opt("--gen"))
UNDO = "--undo" in args
CUDA = FORK / "crates" / "stwo" / "src" / "stwo_cuda" / "cuda"
BEGIN = "        # B7a 2b: generated zk-prover constraint kernels\n"
END = "        # end B7a 2b\n"
INC_BEGIN = "// B7a 2b: generated zk-prover constraint kernels\n"
INC_END = "// end B7a 2b includes\n"
CASE_BEGIN = "        // B7a 2b: generated zk-prover constraint kernels\n"
CASE_END = "        // end B7a 2b cases\n"


def strip_block(s, begin, end):
    if begin not in s:
        return s, False
    i = s.index(begin)
    j = s.index(end) + len(end)
    return s[:i] + s[j:], True


def replace_block(path, begin, end, body, anchor):
    p = CUDA / path
    s = p.read_text()
    s, had = strip_block(s, begin, end)
    if not UNDO:
        n = s.count(anchor)
        assert n == 1, f"{path}: anchor {anchor!r} found {n} times"
        s = s.replace(anchor, begin + body + end + anchor)
    p.write_text(s)
    print(f"  {path}: {'replaced' if had else ('removed' if UNDO else 'added')}")


if UNDO:
    for f in CUDA.glob("evaluate_zk_*"):
        f.unlink()
    (CUDA / "zk_ops.cuh").unlink(missing_ok=True)
    print("  removed generated sources")
    replace_block("CMakeLists.txt", BEGIN, END, "", "        batch_inverse.cu\n")
    replace_block("evaluate_constraints.cu", INC_BEGIN, INC_END, "",
                  "bool g_should_accumulate_host = true;")
    replace_block("evaluate_constraints.cu", CASE_BEGIN, CASE_END, "", "        default:\n")
    print("install_constraint_kernels: undone")
    raise SystemExit(0)

man = json.loads((GEN / "manifest.json").read_text())
fams = [f["family"] for f in man["families"]]
assert fams, "manifest lists no families"

for f in list(GEN.glob("*.cu")) + list(GEN.glob("*.cuh")):
    shutil.copy2(f, CUDA / f.name)
print(f"  copied {len(fams)*2 + 1} generated sources into {CUDA}")

replace_block(
    "CMakeLists.txt", BEGIN, END,
    "".join(f"        evaluate_{f}.cu\n" for f in fams),
    "        batch_inverse.cu\n",
)
replace_block(
    "evaluate_constraints.cu", INC_BEGIN, INC_END,
    "".join(f'#include "evaluate_{f}.cuh"\n' for f in fams),
    "bool g_should_accumulate_host = true;",
)
replace_block(
    "evaluate_constraints.cu", CASE_BEGIN, CASE_END,
    "".join(
        f'        DISPATCH_EVAL_WITH_BOOLS("{f}", "call cuda eval {f}", evaluate_{f});\n'
        for f in fams
    ),
    "        default:\n",
)
lines = sum(f["cu_lines"] for f in man["families"])
print(f"install_constraint_kernels: {len(fams)} families, {man['kernels']} kernels, "
      f"{man['instructions']} instructions, {lines} lines of CUDA")
