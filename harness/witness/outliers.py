"""Why the per-token scheme beats static per-tensor W8A8: activation outliers.

For every linear it compares the calibrated per-tensor range against the range an
individual token actually occupies.  Two numbers matter:

  * grid use -- median(per-token max|x|) / calibrated range.  A static scale is
    sized for the worst token in the calibration set, so a typical token only
    reaches that fraction of the +-127 grid; the rest of the int8 codes are
    wasted, costing log2 of the ratio in effective precision.
  * clipped tokens -- share of eval tokens whose max|x| exceeds the calibrated
    range, i.e. whose largest components are clamped outright.

    uv run python b1/outliers.py
"""
import json
import math
import os

import torch

from common import MODEL, OUT, SPECS, chunks, setup, tokens
from model import Model

CHUNKS = 4


def main():
    ranges = json.load(open(os.path.join(OUT, "static_ranges.json")))
    ev, _ = tokens()
    m = Model(MODEL, SPECS["bf16-only"])
    m.calib = {}
    for c in chunks(ev, CHUNKS):
        m.forward(c)
    per_token = {k: torch.cat(v) for k, v in m.calib.items()}

    rows = {}
    for name, amax in per_token.items():
        kind = name.split(".")[-1]
        static = ranges[name]
        rows.setdefault(kind, []).append(
            (amax.median().item() / static, (amax > static).float().mean().item(),
             amax.max().item() / amax.median().item()))

    print(f"{'matmul':<9} {'n':>3} {'grid use':>9} {'lost bits':>10} "
          f"{'clipped tok':>12} {'max/median':>11}")
    out = {}
    for kind, vals in rows.items():
        use = sorted(v[0] for v in vals)[len(vals) // 2]
        clip = sorted(v[1] for v in vals)[len(vals) // 2]
        peak = sorted(v[2] for v in vals)[len(vals) // 2]
        out[kind] = dict(n=len(vals), grid_use=use, lost_bits=-math.log2(use),
                         clipped=clip, peak_ratio=peak)
        print(f"{kind:<9} {len(vals):>3} {use * 100:8.1f}% {-math.log2(use):10.1f} "
              f"{clip * 100:11.2f}% {peak:11.1f}")
    json.dump(out, open(os.path.join(OUT, "outliers.json"), "w"), indent=1)
    print(f"\n(medians over the n layers of each kind, {CHUNKS * 512} eval tokens)")


if __name__ == "__main__":
    setup()
    main()
