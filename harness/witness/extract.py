"""Dump the layer witness B6.5 composes into one proof.

Forked from `b6/extract.py`.  Re-runs B1's forward pass under the composed
layout's spec (`b65-design`: power-of-two linear scales on q/k/v/o, the bf16
chain on gate/up/down, the integer softmax at int12 x int11, and the attention
output rounded straight to int8 with a power-of-two `s_o`) and writes everything
the composed proof's regions read:

  * **the attention block**, in `b5/extract.py`'s format (the S queries of the
    last prefill chunk against the C-key prefix) -- the packed bands load a band
    of heads from it;
  * **the matmul accumulators and scales**, as in B6 -- the dequant regions;
  * **new against B6**: the quantized weights `qw_*.i8` (the preprocessed side of
    the r^T W walk), the quantized activations `qx_*.i8` (the walk's committed
    side), the bf16 input patterns `x_*.u16` (the requantize / max-argument
    witnesses), and the attention-output fold's scale exponents `jo.i32` (the
    per-token power-of-two `s_o`) and `jv.i32` (the per-(head, channel)
    power-of-two V scales).

    uv run python b65/extract.py --tokens 1536 --chunk 512 --layer 12
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "b1"))

from common import MODEL, SPECS, setup, tokens  # noqa: E402
from model import BF16, Cache, Model, Recorder  # noqa: E402
from quant import round_hafz  # noqa: E402

OUT = os.path.join(HERE, "out", "witness")
MATS = ["q", "k", "v", "o", "gate", "up", "down"]


class LastChunk(Recorder):
    """Keep only the newest value of each column: the last prefill chunk."""

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def add(self, key, t):
        if key in self.keep:
            self.cols[key] = t.detach().clone()


def write(name, arr, digests):
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(OUT, name))
    digests[name] = dict(values=int(arr.size), dtype=str(arr.dtype), shape=list(arr.shape),
                         sha256=hashlib.sha256(arr.tobytes()).hexdigest()[:16])
    print(f"  {name:<14} {arr.size:>12,} values ({arr.nbytes / 2 ** 20:6.1f} MiB)  {arr.shape}")


def bits_for(v):
    return int(v).bit_length()


def bf16_parts(t):
    """(mantissa field, biased exponent) of a bf16-valued tensor."""
    bits = t.to(BF16).view(torch.int16).numpy().astype(np.uint16)
    return (bits & 0x7F).astype(np.int32), ((bits >> 7) & 0xFF).astype(np.int32)


def pow2_exponent(t):
    """`j` such that the value is exactly `2^j`; NaN-safe, and -1000 if it is not."""
    v = t.double().numpy()
    with np.errstate(divide="ignore"):
        j = np.log2(np.abs(v))
    exact = np.isfinite(j) & (np.abs(j - np.round(j)) < 1e-9)
    return np.where(exact, np.round(j), -1000).astype(np.int32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=1536)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--layer", type=int, default=12)
    # B7b: several layers from one forward pass, each to `<out>_L<k>`; with one
    # layer the output directory is unchanged
    p.add_argument("--layers", default=None,
                   help="comma-separated layer indices (overrides --layer)")
    p.add_argument("--out", default=OUT, help="output directory (base, for --layers)")
    p.add_argument("--spec", default="b65-design")
    p.add_argument("--mats", default=",".join(MATS))
    a = p.parse_args()
    setup()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
    OUT_BASE = a.out
    spec = SPECS[a.spec]
    assert spec.int_softmax, f"{a.spec} is not an integer-softmax spec"
    assert spec.attn_out_int8 and spec.attn_out_pow2, \
        f"{a.spec} does not use the provable int8 attention-output fold"
    mats = a.mats.split(",")

    from ppl import attention_scales  # noqa: E402

    m = Model(MODEL, spec, attn_scales=attention_scales(spec), track_acc=True)
    m.clamped = {}
    keep = set()
    for L in layers:
        keep |= {f"L{L}.qq", f"L{L}.kk", f"L{L}.vv", f"L{L}.qkacc",
                 f"L{L}.expnum/in", f"L{L}.expnum/out", f"L{L}.sqk",
                 f"L{L}.attnq", f"L{L}.s_o", f"L{L}.zz"}
        keep |= {f"L{L}.{n}/acc" for n in mats} | {f"L{L}.{n}/s_x" for n in mats}
        keep |= {f"L{L}.{n}/qx" for n in mats} | {f"L{L}.{n}/x" for n in mats}
    m.rec = LastChunk(keep)
    ev, _ = tokens()
    ids = ev[: a.tokens]
    cache = Cache(m.L)
    t0 = time.time()
    for i in range(0, len(ids), a.chunk):
        m.forward(ids[i:i + a.chunk], cache)
        print(f"  prefilled {min(i + a.chunk, len(ids))}/{len(ids)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    for L in layers:
        OUT = OUT_BASE if len(layers) == 1 else f"{OUT_BASE}_L{L}"
        os.makedirs(OUT, exist_ok=True)
        print(f"layer {L} -> {OUT}")
        digests = {}

        # ---- the attention block, in b5/extract.py's format ---------------------
        sc = m.rec.cols[f"L{L}.qkacc"].to(torch.int64)          # (nh, S, C)
        nh, S, C = sc.shape
        hd = m.hd
        # The decode rectangle: key c is visible to query t iff c <= (C - S) + t, so the
        # C - S prefix is fully visible and the S x S corner is the only masked part.
        P = C - S
        assert P > 0 and (P & (P - 1)) == 0, f"prefix {P} must be a power of two"
        sc = sc[:, :, :P].contiguous()
        mm = sc.amax(-1)
        dd = mm.unsqueeze(-1) - sc
        sqk = m.rec.cols[f"L{L}.sqk"].double()
        xx = (-dd.double() * sqk).float().to(BF16)
        nmax = (1 << spec.num_bits) - 1
        num = round_hafz(nmax * torch.exp(xx.double())).clamp(0, nmax).to(torch.int64)
        zz = num.sum(-1)

        write("qq.i16", m.rec.cols[f"L{L}.qq"].numpy().astype(np.int16), digests)
        write("kk.i16", m.rec.cols[f"L{L}.kk"][:, :P].numpy().astype(np.int16), digests)
        write("vv.i8", m.rec.cols[f"L{L}.vv"][:, :P].numpy().astype(np.int8), digests)
        write("sc.i32", sc.numpy().astype(np.int32), digests)
        write("dd.i32", dd.numpy().astype(np.int32), digests)
        write("num.i32", num.numpy().astype(np.int32), digests)
        write("xx.u16", xx.contiguous().view(torch.int16).numpy().view(np.uint16), digests)
        write("mm.i32", mm.numpy().astype(np.int32), digests)
        write("zz.i32", zz.numpy().astype(np.int32), digests)

        # ---- the attention-output fold's scales ---------------------------------
        # The composed attnout region proves the *rectangle* sub-computation (the
        # same statement the bands prove; the causal corner stays outside every AIR
        # in the project, B9's job), so its accumulators and denominators are the
        # recomputed `out`/`zz` above.  The scales are the forward pass's: a
        # power-of-two `s_o` per token and power-of-two V scales per (head, channel).
        s_o = m.rec.cols[f"L{L}.s_o"].float()                    # (S, 1)
        jo = pow2_exponent(s_o.reshape(-1))
        assert (jo != -1000).all(), "s_o is not a power of two -- wrong spec?"
        write("jo.i32", jo, digests)
        s_v = attention_scales(spec)[f"L{L}.v"]                  # (nh, 1, hd)
        jv = pow2_exponent(s_v.reshape(nh, hd))
        assert (jv != -1000).all(), "V scales are not powers of two -- wrong spec?"
        write("jv.i32", jv.reshape(-1), digests)
        # the fold's exponent e = jv[h, ch] - jo[t]: its range decides the int8
        # gadget's num/den split (num = 2^max(e,0), den = Z * 2^max(-e,0))
        e_min = int(jv.min()) - int(jo.max())
        e_max = int(jv.max()) - int(jo.min())

        # ---- the accumulators the dequant rounds, and the walk's operands -------
        accs = {}
        for name in mats:
            key = f"L{L}.{name}/acc"
            if key not in m.rec.cols:
                print(f"  (no accumulator recorded for {name})")
                continue
            acc = m.rec.cols[key]
            lin = m.layers[L][name]
            s_x = m.rec.cols[f"L{L}.{name}/s_x"]
            write(f"acc_{name}.i32", acc.numpy().astype(np.int32), digests)
            write(f"qx_{name}.i8", m.rec.cols[f"L{L}.{name}/qx"].numpy().astype(np.int8),
                  digests)
            write(f"qw_{name}.i8", lin.qw.numpy().astype(np.int8), digests)
            if f"L{L}.{name}/x" in m.rec.cols:
                write(f"x_{name}.u16",
                      m.rec.cols[f"L{L}.{name}/x"].contiguous().view(torch.int16)
                      .numpy().view(np.uint16), digests)
            jx, jw = pow2_exponent(s_x.reshape(-1)), pow2_exponent(lin.s_w.reshape(-1))
            mx, ex = bf16_parts(s_x.reshape(-1))
            mw, ew = bf16_parts(lin.s_w.reshape(-1))
            write(f"jx_{name}.i32", jx, digests)
            write(f"jw_{name}.i32", jw, digests)
            write(f"sx_{name}.i32", np.stack([mx, ex], -1), digests)
            write(f"sw_{name}.i32", np.stack([mw, ew], -1), digests)
            av = acc.abs().amax().item()
            accs[name] = dict(
                n_in=int(lin.n_in), n_out=int(lin.n_out), tokens=int(acc.shape[0]),
                abs_max=int(av), bits=bits_for(av),
                kmax_needed=max(0, bits_for(av) - 9),
                worst_case=int(127 * 127 * lin.n_in),
                kmax_worst_case=max(0, bits_for(127 * 127 * lin.n_in) - 9),
                scales_pow2=bool((jx != -1000).all() and (jw != -1000).all()),
                zero_frac=float((acc == 0).double().mean()),
            )
            print(f"  {name:<5} |acc| <= {av:>12,.0f} ({bits_for(av)} bits) "
                  f"kmax {accs[name]['kmax_needed']} "
                  f"(worst case {accs[name]['kmax_worst_case']}), "
                  f"pow2 scales {accs[name]['scales_pow2']}")

        frac = (-torch.log2(sqk)).reshape(nh, -1)
        fpi = frac[:, 0].tolist()
        stats = dict(
            heads=nh, s=S, c=P, hd=hd, layer=L, spec=a.spec, qk_bits=list(spec.qk_bits),
            num_bits=spec.num_bits, model=MODEL,
            d_max=int(dd.max()), d_bits=bits_for(dd.max()),
            d_fits_16bit=bool(dd.max() < (1 << 16)),
            z_max=int(zz.max()), z_worst_case=int(nmax * P),
            numerator_zero_frac=float((num == 0).double().mean()),
            frac_per_head=fpi,
            frac_shared_per_head=bool(frac.shape[1] == 1 or bool((frac == frac[:, :1]).all())),
            frac_is_integer=bool(((frac - frac.round()).abs() < 1e-9).all()),
            frac_spread=float(max(fpi) - min(fpi)),
            frac_integer_separated=bool(
                all(abs((max(fpi) - f) - round(max(fpi) - f)) < 1e-9 for f in fpi)),
            master_table_bits=int(bits_for(dd.max()) + round(max(fpi) - min(fpi))),
            attn_out=dict(jo_min=int(jo.min()), jo_max=int(jo.max()),
                          jv_min=int(jv.min()), jv_max=int(jv.max()),
                          e_min=e_min, e_max=e_max),
            accumulators=accs,
            clamped={k: dict(over=v[0], total=v[1], frac=v[0] / v[1])
                     for k, v in (m.clamped or {}).items()},
            torch=torch.__version__, files=digests,
        )
        json.dump(stats, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=1))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
