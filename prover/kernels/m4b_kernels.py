"""B7a milestone 4 (18.7): install the generated *generation* kernels into the fork.

The counterpart of `m2b_kernels.py`, and simpler than it, because the fork does
not dispatch these: our own crate calls `gen_zk_dispatch` directly, so there is
no case to add to `evaluate_constraints.cu` and no `cuda_eval_name` to key on.
Two asserted, idempotent edits:

  1. copy `gen_zk_*.cu` / `.cuh` (from `zk-prover genemit`) and the hand-written
     `gen_zk_logup.cu` / `.cuh` into `crates/stwo/src/stwo_cuda/cuda/`;
  2. add each `.cu` to the explicit source list in `CMakeLists.txt`.

`--undo` removes both.

    python3 m4b_kernels.py --fork DIR --gen DIR [--src DIR] [--undo]
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
HAND = Path(opt("--src", str(Path(__file__).resolve().parent)))
UNDO = "--undo" in args
CUDA = FORK / "crates" / "stwo" / "src" / "stwo_cuda" / "cuda"
BEGIN = "        # B7a m4: generated zk-prover generation kernels\n"
END = "        # end B7a m4\n"
ANCHOR = "        batch_inverse.cu\n"


def replace_block(path, begin, end, body, anchor):
    p = CUDA / path
    s = p.read_text()
    had = begin in s
    if had:
        i = s.index(begin)
        j = s.index(end) + len(end)
        s = s[:i] + s[j:]
    if not UNDO:
        n = s.count(anchor)
        assert n == 1, f"{path}: anchor {anchor!r} found {n} times"
        s = s.replace(anchor, begin + body + end + anchor)
    p.write_text(s)
    print(f"  {path}: {'replaced' if had else ('removed' if UNDO else 'added')}")


if UNDO:
    for f in list(CUDA.glob("gen_zk_*")):
        f.unlink()
    print("  removed the generation sources")
    replace_block("CMakeLists.txt", BEGIN, END, "", ANCHOR)
    print("m4b_kernels: undone")
    raise SystemExit(0)

man = json.loads((GEN / "gen_manifest.json").read_text())
fams = [f["family"] for f in man["families"]]
assert fams, "gen_manifest lists no families"

srcs = []
for f in sorted(GEN.glob("gen_zk_*.cu")) + sorted(GEN.glob("gen_zk_*.cuh")):
    shutil.copy2(f, CUDA / f.name)
    if f.suffix == ".cu":
        srcs.append(f.name)
# `zk_ops.cuh` is shared with 2b's kernels and both emitters write the same
# `cudagen::OPS_CUH`; refresh it here so a fork installed before `zk_ext`
# existed picks it up.
shutil.copy2(GEN / "zk_ops.cuh", CUDA / "zk_ops.cuh")

for name in ("gen_zk_logup.cu", "gen_zk_logup.cuh"):
    src = HAND / name
    assert src.exists(), f"missing hand-written source {src}"
    shutil.copy2(src, CUDA / name)
    if src.suffix == ".cu":
        srcs.append(name)
srcs.sort()
print(f"  copied {len(srcs)} generation sources into {CUDA}")

replace_block("CMakeLists.txt", BEGIN, END, "".join(f"        {s}\n" for s in srcs), ANCHOR)
lines = sum(f["cu_lines"] for f in man["families"])
print(
    f"m4b_kernels: {len(fams)} families, {man['shapes']} fraction shapes, "
    f"{lines} lines of generated CUDA plus the four hand-written phases"
)
