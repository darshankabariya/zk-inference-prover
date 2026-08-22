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

import numpy as np
import torch

from common import (B5_SPECS, B6_SPECS, B65_SPECS, B8_SPECS, MODEL, OUT, SPECS, chunks,
                    setup, tokens)
from model import Model
from quant import INT8_MAX, bf16_ceil, qscale

RESULTS = os.path.join(OUT, "ppl.json")
CALIB = os.path.join(OUT, "static_ranges.json")
ACALIB = os.path.join(OUT, "attn_ranges.json")


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


def attention_ranges():
    """max|k| per (layer, head) and max|v| per (layer, head, channel).

    B5: the integer softmax needs the K and V scales to be constant along the
    context axis (the max shift and the A.V accumulator are integer statements
    only then), and a scale that is recomputed as the cache grows would
    re-quantize columns that are already committed.  So they are calibrated once,
    on the same disjoint slice B1's static W8A8 baseline uses.
    """
    if not os.path.exists(ACALIB):
        _, cal = tokens()
        m = Model(MODEL, SPECS["bf16-only"])
        m.attn_calib = {}
        for c in chunks(cal):
            m.forward(c)
        ranges = {k: torch.stack(v).amax(0).tolist() for k, v in m.attn_calib.items()}
        json.dump(ranges, open(ACALIB, "w"), indent=1, sort_keys=True)
        del m
    return json.load(open(ACALIB))


def attention_scales(spec):
    """Calibrated ranges -> the two scale tensors `Model.kv_scales` expects."""
    qmax_k = (1 << (spec.qk_bits[1] - 1)) - 1
    out = {}
    for k, v in attention_ranges().items():
        t = torch.tensor(v)
        p2 = spec.pow2_scales in ("all", "attn")
        if k.endswith(".k"):                       # (nh,) -> (nh, 1, 1)
            out[k] = qscale(t.view(-1, 1, 1), -1, qmax_k, p2)
        elif k.endswith(".q"):
            qmax_q = (1 << (spec.qk_bits[0] - 1)) - 1
            out[k] = qscale(t.view(-1, 1, 1), -1, qmax_q, p2)
        else:                                      # (nh, hd) -> (nh, 1, hd)
            out[k] = qscale(t.unsqueeze(1), 1, INT8_MAX, p2)
    return out


def run(name, n_chunks, save=True):
    spec = SPECS[name]
    scales = calibration_scales() if spec.linear == "static" else None
    ascales = attention_scales(spec) if spec.int_softmax else None
    t0 = time.time()
    m = Model(MODEL, spec, static_scales=scales, attn_scales=ascales)
    ev, _ = tokens()
    nll, cnt, per_token = 0.0, 0, []
    for i, c in enumerate(chunks(ev, n_chunks)):
        logits, _ = m.forward(c)
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        t = -lp.gather(1, c[1:, None].to(lp.device)).squeeze(1)
        per_token.append(t.cpu())
        nll += t.sum().item()
        cnt += len(c) - 1
        print(f"  chunk {i + 1:>3}: ppl {math.exp(nll / cnt):8.4f}  ({time.time() - t0:5.1f}s)", flush=True)
    del m
    if save:
        # B8: the per-token NLLs, so a difference between two specs gets a paired
        # standard error (the statistical floor under a ppl delta)
        os.makedirs(os.path.join(OUT, "nll"), exist_ok=True)
        np.save(os.path.join(OUT, "nll", f"{name}.npy"), torch.cat(per_token).numpy())
    return dict(spec=spec.asdict(), ppl=math.exp(nll / cnt), nll=nll / cnt,
                tokens=cnt, chunks=n_chunks, seconds=round(time.time() - t0, 1))


def paired(a, b):
    """ppl(b)/ppl(a) - 1 with its paired standard error, from the per-token NLLs."""
    x = np.load(os.path.join(OUT, "nll", f"{a}.npy")).astype(np.float64)
    y = np.load(os.path.join(OUT, "nll", f"{b}.npy")).astype(np.float64)
    d = y - x
    se = d.std(ddof=1) / math.sqrt(len(d))
    return math.exp(d.mean()) - 1, se      # se is on the mean NLL difference (nats)


def markdown(only=None):
    r = json.load(open(RESULTS))
    base, att = r["fp32"]["ppl"], r["dynamic-int8"]["ppl"]
    keys = [s for s in SPECS if s in r and (only is None or s in only)]
    print("| spec | ppl | vs fp32 | vs dynamic-int8 |")
    print("|---|---|---|---|")
    for k in keys:
        v = r[k]
        d = "--" if k == "fp32" else f"{(v['ppl'] / base - 1) * 100:+.2f}%"
        a = "--" if k == "dynamic-int8" else f"{(v['ppl'] / att - 1) * 100:+.2f}%"
        print(f"| `{k}` | {v['ppl']:.3f} | {d} | {a} |")


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
    p.add_argument("--b5", action="store_true", help="only B5's probes (with --markdown/--missing)")
    p.add_argument("--b6", action="store_true", help="only B6's probes (with --markdown/--missing)")
    p.add_argument("--b65", action="store_true",
                   help="only the B6 review's probes (with --markdown/--missing)")
    p.add_argument("--b8", action="store_true",
                   help="only the rows B8 re-measures at 8B (with --markdown/--missing)")
    p.add_argument("--no-save", action="store_true", help="smoke run: do not touch ppl.json")
    a = p.parse_args()
    setup()
    if a.report or a.markdown:
        if a.markdown:
            only = None
            if a.b5:
                only = B5_SPECS + ["fp32", "dynamic-int8", "qk-int8"]
            if a.b6:
                only = B6_SPECS + ["fp32", "dynamic-int8", "int-softmax-12sq"]
            if a.b65:
                only = B65_SPECS + ["fp32", "dynamic-int8", "bf16-dequant",
                                    "pow2-dequant", "int-attnout", "b6-design-bf16"]
            if a.b8:
                only = B8_SPECS
            markdown(only)
        else:
            report()
        raise SystemExit
    done = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    pool = (B5_SPECS if a.b5 else
            (B6_SPECS if a.b6 else
             (B65_SPECS if a.b65 else (B8_SPECS if a.b8 else list(SPECS)))))
    todo = [s for s in pool if s not in done] if a.missing else (pool if a.all else [a.spec])
    for name in todo:
        print(f"== {name}: {SPECS[name].asdict()}", flush=True)
        out = run(name, a.chunks, save=not a.no_save)
        if not a.no_save:
            all_r = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
            all_r[name] = out
            json.dump(all_r, open(RESULTS, "w"), indent=1)
        print(f"   ppl = {out['ppl']:.4f}  ({out['seconds']}s)", flush=True)
