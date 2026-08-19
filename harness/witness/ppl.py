"""B1 acceptance 1: perplexity of the attested computation vs fp32 and static W8A8.

One spec per invocation (peak RAM stays at a single model); results are merged
into out/ppl.json.  `--report` prints the table.

    uv run python b1/ppl.py --spec dynamic-int8
    uv run python b1/ppl.py --report
"""
import argparse
import json
import math
import os
import time

import torch

from common import MODEL, OUT, SPECS, chunks, setup, tokens
from model import Model
from quant import INT8_MAX, bf16_ceil

RESULTS = os.path.join(OUT, "ppl.json")
CALIB = os.path.join(OUT, "static_ranges.json")


def calibration_scales():
    """Per-tensor static activation scales, observed on the float model.

    This is the classic W8A8 PTQ baseline: one scale per linear, fixed at
    calibration time from max|x| over a held-out slice disjoint from the eval set.
    """
    if not os.path.exists(CALIB):
        _, cal = tokens()
        m = Model(MODEL, SPECS["bf16-only"])
        m.calib = {}
        for c in chunks(cal):
            m.forward(c)
        ranges = {k: torch.cat(v).max().item() for k, v in m.calib.items()}
        json.dump(ranges, open(CALIB, "w"), indent=1, sort_keys=True)
        del m
    ranges = json.load(open(CALIB))
    return {k: bf16_ceil(torch.tensor(v) / INT8_MAX).view(1, 1) for k, v in ranges.items()}


def run(name, n_chunks):
    spec = SPECS[name]
    scales = calibration_scales() if spec.linear == "static" else None
    t0 = time.time()
    m = Model(MODEL, spec, static_scales=scales)
    ev, _ = tokens()
    nll, cnt = 0.0, 0
    for i, c in enumerate(chunks(ev, n_chunks)):
        logits, _ = m.forward(c)
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        nll += -lp.gather(1, c[1:, None]).sum().item()
        cnt += len(c) - 1
        print(f"  chunk {i + 1:>3}: ppl {math.exp(nll / cnt):8.4f}  ({time.time() - t0:5.1f}s)", flush=True)
    del m
    return dict(spec=spec.asdict(), ppl=math.exp(nll / cnt), nll=nll / cnt,
                tokens=cnt, chunks=n_chunks, seconds=round(time.time() - t0, 1))


def markdown():
    r = json.load(open(RESULTS))
    base = r["fp32"]["ppl"]
    print("| spec | ppl | vs fp32 |")
    print("|---|---|---|")
    for k in [s for s in SPECS if s in r]:
        v = r[k]
        d = "--" if k == "fp32" else f"{(v['ppl'] / base - 1) * 100:+.2f}%"
        print(f"| `{k}` | {v['ppl']:.3f} | {d} |")


def report():
    r = json.load(open(RESULTS))
    base = r["fp32"]["ppl"]
    print(f"{'spec':<12} {'ppl':>9} {'vs fp32':>9}  tokens  spec")
    for k, v in r.items():
        s = v["spec"]
        d = f"{(v['ppl'] / base - 1) * 100:+.2f}%" if k != "fp32" else "--"
        print(f"{k:<12} {v['ppl']:9.4f} {d:>9}  {v['tokens']:>6}  "
              f"lin={s['linear']} nl={s['nonlinear']} red={s['reduce']}"
              f"{' qk=int8' if s['qk_int8'] else ''}"
              f"{' fp32-logits' if s.get('fp32_logits') else ''}"
              f"{' fp32:' + ','.join(s['fp32_matmuls']) if s.get('fp32_matmuls') else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", choices=list(SPECS))
    p.add_argument("--all", action="store_true", help="every spec, one model at a time")
    p.add_argument("--missing", action="store_true", help="only specs not yet in ppl.json")
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--report", action="store_true")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args()
    setup()
    if a.report or a.markdown:
        (markdown if a.markdown else report)()
        raise SystemExit
    done = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    todo = [s for s in SPECS if s not in done] if a.missing else (list(SPECS) if a.all else [a.spec])
    for name in todo:
        print(f"== {name}: {SPECS[name].asdict()}", flush=True)
        out = run(name, a.chunks)
        all_r = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
        all_r[name] = out
        json.dump(all_r, open(RESULTS, "w"), indent=1)
        print(f"   ppl = {out['ppl']:.4f}  ({out['seconds']}s)", flush=True)
