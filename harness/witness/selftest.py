"""B1 self-tests: is the reimplementation the right model, and does int8 embed in M31?

  1-2. the rounding and scale rules on their own;
  3. our fp32 forward vs the HuggingFace forward on the same tokens
     -- the reimplementation is faithful, so the quality numbers mean something;
  4. int8 x int8 accumulation: BLAS vs int64 vs M31 field arithmetic, on real
     activations at every real reduction width (H2);
  5. observed |accumulator| vs the M31 signed range, per matmul (H2 headroom);
  6. batched vs token-at-a-time evaluation -- what it takes for the order to be
     genuinely fixed.

    uv run python b1/selftest.py
"""
import gc
import json
import os

import numpy as np
import torch

from common import MODEL, OUT, SPECS, chunks, setup, tokens
from model import Cache, Linear, Model, Recorder
from quant import BF16, INT8_MAX, qscale_per_token, round_hafz

P = (1 << 31) - 1          # M31
M31_MAX = (P - 1) // 2     # largest |v| a single field element carries signed
# B1's default.  A model with local attention needs a slice longer than its
# sliding window before test 3 exercises the window at all -- Gemma-3-270M's is
# 512, so `B1_SELFTEST_TOKENS=640` is the run that actually checks the two mask
# kinds against HuggingFace (briefs/e5a-gemma.md).
N_TOK = int(os.environ.get("B1_SELFTEST_TOKENS", 128))
results = {}


def m31_gemm(qx, qw):
    """Dot products over M31, reducing mod p per product before summing."""
    a = np.mod(qx.astype(np.int64), P).astype(np.uint64)
    b = np.mod(qw.astype(np.int64), P).astype(np.uint64)
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.uint64)
    for k in range(a.shape[1]):
        acc = (acc + (a[:, k:k + 1] * b[k:k + 1, :]) % P) % P
    signed = acc.astype(np.int64)
    return np.where(signed > M31_MAX, signed - P, signed)


def test_primitives():
    """Rounding and scale rules, on their own: no model needed."""
    x = torch.tensor([0.5, -0.5, 1.5, 2.5, -2.5, 0.4999, 127.5])
    hafz = round_hafz(x).tolist()
    want = [1, -1, 2, 3, -3, 0, 128]
    ok = hafz == want
    print(f"1. round-half-away-from-zero {hafz} == {want}: {ok}"
          f"  (torch.round would give {torch.round(x).tolist()})")

    torch.manual_seed(0)
    v = torch.randn(4096, 896) * torch.logspace(-3, 3, 4096).unsqueeze(1)
    sc = qscale_per_token(v)
    is_bf16 = bool((sc.to(BF16).float() == sc).all().item())
    # the point of rounding the scale up: the clamp must never actually bind,
    # since clamping is exactly the outlier clipping that sinks static W8A8.
    raw = round_hafz(v / sc).abs()
    binds = bool((raw > INT8_MAX).any().item())
    naive = round_hafz(v / (v.abs().amax(-1, keepdim=True) / INT8_MAX).to(BF16).float()).abs()
    naive_binds = int((naive > INT8_MAX).sum().item())
    lo = int(raw.amax(-1).min().item())
    print(f"2. scales are bf16 values: {is_bf16}; clamp binds: {binds} "
          f"(round-to-nearest scales would clip {naive_binds} values); "
          f"quietest row still reaches {lo}/127 of the grid")
    results["primitives"] = dict(round_hafz=ok, scales_bf16=is_bf16, clamp_binds=binds,
                                 nearest_scale_clipped=naive_binds, min_row_peak=lo)
    return ok and is_bf16 and not binds and lo >= INT8_MAX - 1


def test_vs_hf(ids):
    from transformers import AutoModelForCausalLM
    m = Model(MODEL, SPECS["fp32"])
    ours, _ = m.forward(ids)
    del m
    gc.collect()          # 11 GB box: never hold two models at once
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    theirs = hf(ids.unsqueeze(0)).logits[0].float()
    del hf
    d = (ours - theirs).abs()
    rel = d.max().item() / theirs.abs().max().item()
    agree = (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()
    print(f"3. vs HuggingFace: max|dlogit| {d.max().item():.3e}  ({rel:.1e} of logit range)  "
          f"argmax agreement {agree * 100:.1f}%")
    results["hf_max_abs_logit_diff"] = d.max().item()
    results["hf_argmax_agreement"] = agree
    return d.max().item() < 5e-2 and agree == 1.0


def test_int_exact(ids):
    m = Model(MODEL, SPECS["dynamic-int8"], track_acc=True)
    m.rec = Recorder()
    m.forward(ids)
    rec = m.rec.stack()
    m.rec = None

    print("4. int8 accumulation, BLAS vs int64 vs M31 (real activations):")
    ok = True
    for lin in [v for v in m.layers[0].values() if isinstance(v, Linear)] + [m.head]:
        qx = rec[f"{lin.name}/qx"][0][:8].numpy().astype(np.int64)
        qw = lin.qw[:, :64].numpy().astype(np.int64)
        blas = (torch.from_numpy(qx).to(lin.qw.dtype) @ lin.qw[:, :64]).numpy().astype(np.int64)
        ref = qx @ qw
        field = m31_gemm(qx, qw)
        good = np.array_equal(blas, ref) and np.array_equal(field, ref)
        ok &= good
        print(f"   {lin.name:<10} K={lin.n_in:<5} max|acc| {np.abs(ref).max():>10,}  "
              f"blas==int64==M31: {good}")
    results["int_exact"] = bool(ok)

    print("5. observed |accumulator| vs the M31 signed range (2^30-1 = 1,073,741,823):")
    by_kind = {}
    for k, v in m.acc_max.items():
        by_kind.setdefault(k.split(".")[-1], []).append(v)
    for kind, vals in by_kind.items():
        mx = max(vals)
        print(f"   {kind:<8} max|acc| {int(mx):>12,}   headroom x{M31_MAX / max(mx, 1):>9,.0f}")
    results["acc_max_by_kind"] = {k: max(v) for k, v in by_kind.items()}
    results["acc_max_overall"] = max(m.acc_max.values())
    del m
    return ok and results["acc_max_overall"] <= M31_MAX


def test_sequential(ids):
    """Batched vs token-at-a-time.

    Weight matmuls are exact integer accumulations, so they cannot differ.  What
    can differ is the *order* BLAS accumulates the fp32 reductions in (RMSNorm
    mean, softmax denominator, the two bf16 attention matmuls), which is
    shape-dependent -- and under bf16 storage an order difference of 1e-7 can
    cross a rounding boundary and then compound through 24 layers.
    """
    print("6. batched vs token-at-a-time, per spec (max|d| on the residual stream):")
    out = {}
    for name in ("fp32", "dynamic-int8", "bf16-reduce", "qk-int8"):
        m = Model(MODEL, SPECS[name])
        m.rec = Recorder()
        batched, _ = m.forward(ids)
        rb = m.rec.stack()
        m.rec, cache = Recorder(), Cache(m.L)
        seq = torch.cat([m.forward(ids[i:i + 1], cache)[0] for i in range(len(ids))])
        rs = m.rec.stack()
        per_layer = [(rb[f"L{i}.x"][0].float() - rs[f"L{i}.x"].reshape(len(ids), -1).float())
                     .abs().max().item() for i in range(m.L)]
        agree = (batched.argmax(-1) == seq.argmax(-1)).float().mean().item()
        out[name] = dict(bit_identical=bool((batched == seq).all().item()),
                         logit_max_abs=(batched - seq).abs().max().item(),
                         argmax_agreement=agree, layer0=per_layer[0], layer_last=per_layer[-1])
        print(f"   {name:<11} L0 {per_layer[0]:.2e} -> L{m.L - 1} {per_layer[-1]:.2e}, "
              f"logits {out[name]['logit_max_abs']:.2e}, argmax agree {agree * 100:.1f}%, "
              f"bit-identical {out[name]['bit_identical']}")
        m.rec = None
        del m
    results["seq_vs_batched"] = out
    # fp32 must agree to BLAS noise; bf16 storage is allowed to diverge, but the
    # prediction must not: that would mean a logic difference, not rounding.
    return out["fp32"]["logit_max_abs"] < 1e-2 and out["dynamic-int8"]["argmax_agreement"] > 0.95


if __name__ == "__main__":
    setup()
    ev, _ = tokens()
    ids = chunks(ev)[0][:N_TOK]
    passed = [test_primitives(), test_vs_hf(ids), test_int_exact(ids),
              test_sequential(ids[:32])]
    results["all_passed"] = all(passed)
    json.dump(results, open(os.path.join(OUT, "selftest.json"), "w"), indent=1)
    print(f"\nall passed: {all(passed)}")
