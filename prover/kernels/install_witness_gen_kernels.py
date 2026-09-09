"""Install the generated witness-generation kernels into the fork (B7a milestone 4, 18.7).

The counterpart of `install_constraint_kernels.py`, and simpler than it,
because the fork does not dispatch these: our own crate calls
`gen_zk_dispatch` directly, so there is no case to add to
`evaluate_constraints.cu` and no `cuda_eval_name` to key on.
Two asserted, idempotent edits:

  1. copy `gen_zk_*.cu` / `.cuh` (from `zk-prover genemit`) and the hand-written
     `gen_zk_logup.cu` / `.cuh` into `crates/stwo/src/stwo_cuda/cuda/`;
  2. add each `.cu` to the explicit source list in `CMakeLists.txt`.

`--undo` removes both. `BEGIN` below is the marker the fork's `CMakeLists.txt`
already carries around its checked-in block, and the two strings move together
or not at all: that is what makes the first install replace that block instead
of adding a second one. `--selftest` holds them together.

    python3 install_witness_gen_kernels.py --fork DIR --gen DIR [--src DIR] [--undo]
    python3 install_witness_gen_kernels.py --selftest
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

args = sys.argv[1:]

BEGIN = "        # B7a m4: generated witness-generation kernels\n"
END = "        # end B7a m4\n"
ANCHOR = "        batch_inverse.cu\n"


def opt(name, default=None):
    if name in args and args.index(name) + 1 < len(args):
        return args[args.index(name) + 1]
    if default is None:
        sys.exit(f"{name} DIR is required; this script knows no host paths")
    return default


def cut(s):
    """The text without its one `# B7a m4` block, and that block's body."""
    i = s.index(BEGIN)
    j = s.index(END, i) + len(END)
    return s[:i] + s[j:], s[i + len(BEGIN):j - len(END)]


def selftest():
    """Install twice and undo once against a scratch copy of the fork's CMakeLists."""
    lists = (Path(__file__).resolve().parents[2] / "nitroo" / "crates" / "stwo" /
             "src" / "stwo_cuda" / "cuda" / "CMakeLists.txt")
    pristine = lists.read_text()
    assert pristine.count(BEGIN) == 1, f"{lists} does not carry the marker {BEGIN!r}"
    assert pristine.count(END) == 1, f"{lists} does not carry the marker {END!r}"
    bare, _ = cut(pristine)
    with tempfile.TemporaryDirectory() as tmp:
        fork = Path(tmp) / "fork"
        cuda = fork / "crates" / "stwo" / "src" / "stwo_cuda" / "cuda"
        cuda.mkdir(parents=True)
        (cuda / "CMakeLists.txt").write_text(pristine)
        gen = Path(tmp) / "gen"
        gen.mkdir()
        (gen / "gen_manifest.json").write_text(
            json.dumps({"families": [{"family": "selftest", "cu_lines": 0}], "shapes": 0}))
        for name in ("gen_zk_dispatch.cu", "gen_zk_dispatch.cuh", "zk_ops.cuh"):
            (gen / name).write_text("")
        run = [sys.executable, str(Path(__file__).resolve()),
               "--fork", str(fork), "--gen", str(gen)]

        subprocess.run(run, check=True, stdout=subprocess.DEVNULL)
        first = (cuda / "CMakeLists.txt").read_text()
        assert first.count(BEGIN) == 1, f"the first install left {first.count(BEGIN)} blocks"
        rest, body = cut(first)
        assert rest == bare, "the first install changed the file outside its block"
        assert body == "        gen_zk_dispatch.cu\n        gen_zk_logup.cu\n", body

        subprocess.run(run, check=True, stdout=subprocess.DEVNULL)
        second = (cuda / "CMakeLists.txt").read_text()
        assert second == first, f"the second install left {second.count(BEGIN)} blocks"

        subprocess.run(run + ["--undo"], check=True, stdout=subprocess.DEVNULL)
        undone = (cuda / "CMakeLists.txt").read_text()
        assert BEGIN not in undone and END not in undone, "--undo left the block standing"
        assert undone == bare, "--undo changed the file outside its block"
        assert not list(cuda.glob("gen_zk_*")), "--undo left generation sources behind"
    print("install_witness_gen_kernels: self-test passed, one block after two installs "
          "and none after --undo")


if "--selftest" in args:
    selftest()
    raise SystemExit(0)


FORK = Path(opt("--fork"))
GEN = Path(opt("--gen"))
HAND = Path(opt("--src", str(Path(__file__).resolve().parent)))
UNDO = "--undo" in args
CUDA = FORK / "crates" / "stwo" / "src" / "stwo_cuda" / "cuda"


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
    print("install_witness_gen_kernels: undone")
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
    f"install_witness_gen_kernels: {len(fams)} families, {man['shapes']} fraction shapes, "
    f"{lines} lines of generated CUDA plus the four hand-written phases"
)
