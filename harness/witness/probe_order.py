"""Probe for b1/selftest.py §6: where does batched-vs-sequential divergence
originate, and are the specs reported bit-identical actually identical by
construction, or did one 32-token run just not cross a bf16 rounding boundary?

Part A: dynamic-int8 spec, 32 tokens — record every witness column under both
schedules and report the first mismatching (token, op) in forward-op order.
Part B: for dynamic-int8 / bf16-reduce / qk-int8 / both-pinned, try 3 fresh
48-token slices under 3 schedules (one batch, two half-batches, token-at-a-time).
Any mismatch anywhere means "bit-identical" was empirical, not structural.

Findings (2026-08-19, torch 2.13 CPU, 4 threads; README §5 has the discussion):
  * origin = fp32 Q·Kᵀ, GEMV vs GEMM path (first divergence: token 7, L15.scores);
  * bf16-reduce is NOT schedule-invariant — it fails on all 3 fresh slices;
    its selftest §6 pass was a data-dependent false positive;
  * qk-int8 (and qk-int8 + bf16-reduce) passed every slice and schedule, but
    only empirically: A·V and the fp32 reductions remain unpinned;
  * two half-batches always match the full batch — only the single-token GEMV
    path diverges, i.e. the mismatch is exactly decode vs prefill.
"""
import gc
import os
import sys

import torch
from .common import MODEL, SPECS, chunks, setup, tokens
from .model import Cache, Model, Recorder, Spec


def run_recorded(m, ids, seq):
    m.rec = Recorder()
    if seq:
        cache = Cache(m.L)
        logits = torch.cat([m.forward(ids[i:i + 1], cache)[0] for i in range(len(ids))])
        rec = m.rec.cols                       # key -> list of 1 tensor per call
    else:
        logits, _ = m.forward(ids)
        rec = {k: v[0] for k, v in m.rec.cols.items()}   # key -> one tensor
    m.rec = None
    return logits, rec


def first_divergence(m, ids):
    S = len(ids)
    lb, rb = run_recorded(m, ids, seq=False)
    ls, rs = run_recorded(m, ids, seq=True)
    hits = []
    for i in range(S):
        for k in rb:                            # insertion order == op order
            b, e = rb[k], rs[k][i]
            if b.shape[0] == S:                 # token-major: (S, ...) vs (1, ...)
                bt, st = b[i], e[0]
            else:                               # attention: (nh, S, C) vs (nh, 1, C_i)
                bt, st = b[:, i, :e.shape[-1]], e[:, 0]
            if not torch.equal(bt.float(), st.float()):
                hits.append((i, k, (bt.float() - st.float()).abs().max().item()))
        if hits:
            break
    return hits, bool(torch.equal(lb, ls))


def schedules(m, ids):
    full, _ = m.forward(ids)
    h = len(ids) // 2
    c = Cache(m.L)
    a, c = m.forward(ids[:h], c)
    b, c = m.forward(ids[h:], c)
    halves = torch.cat([a, b])
    c = Cache(m.L)
    seq = torch.cat([m.forward(ids[i:i + 1], c)[0] for i in range(len(ids))])
    return (bool(torch.equal(full, halves)), (full - halves).abs().max().item(),
            bool(torch.equal(full, seq)), (full - seq).abs().max().item())


if __name__ == "__main__":
    setup()
    ev, _ = tokens()
    cs = chunks(ev)

    print("== Part A: first divergent op under `dynamic-int8`, 32 tokens (selftest slice)")
    m = Model(MODEL, SPECS["dynamic-int8"])
    hits, ident = first_divergence(m, cs[0][:32])
    del m
    gc.collect()
    if not hits:
        print("   no divergence found (?)")
    for i, k, d in hits[:10]:
        print(f"   token {i:>2}  {k:<20} max|d| {d:.3e}")
    print(f"   logits bit-identical: {ident}")

    print("== Part B: 3 schedules x 3 fresh 48-token slices, bit-identical logits?")
    probes = {
        "dynamic-int8": SPECS["dynamic-int8"],
        "bf16-reduce": SPECS["bf16-reduce"],
        "qk-int8": SPECS["qk-int8"],
        "pin-both": Spec(linear="dynamic", nonlinear="bf16", reduce="bf16", qk_int8=True),
    }
    for name, spec in probes.items():
        m = Model(MODEL, spec)
        for ci in (1, 2, 3):
            hv, hd, sv, sd = schedules(m, cs[ci][:48])
            print(f"   {name:<12} slice {ci}: halves {'==' if hv else '!='} full "
                  f"(max|d| {hd:.2e}), seq {'==' if sv else '!='} full (max|d| {sd:.2e})")
        del m
        gc.collect()
