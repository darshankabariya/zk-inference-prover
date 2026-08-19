"""B1 deliverable: the witness a prover would have to attest to, plus its manifest.

The sequence is decoded one token at a time (sequential KV append), so every step
is a decode step and the witness is laid out exactly as the circuit would walk it.

Dumped per matmul, per layer, per token: the int8 activations qx, the exact int32
accumulator acc, and the bf16 activation scale; per nonlinear: bf16 input/output
bit patterns; per attention: scores and probabilities (ragged -- the context grows
by one key per token).  The manifest is the "circuit config": dims, op list with
the witness values each op contributes per token, and the content digest used to
check that regeneration is bit-identical.

    uv run python b1/gen_witness.py                 # generate
    uv run python b1/gen_witness.py --check         # regenerate, compare digests
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import MODEL, OUT, SPECS, THREADS, digest, setup, tokens
from model import Cache, Model, Recorder

WITNESS = os.path.join(OUT, "witness.npz")
MANIFEST = os.path.join(OUT, "manifest.json")


def to_np(t):
    """bf16 columns are dumped as their uint16 bit patterns."""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def pack(rec):
    """One array per witness column.

    Columns whose width grows with the context (attention probabilities) are
    stored flat with an offsets vector rather than padded.
    """
    out = {}
    for k, vals in rec.cols.items():
        arrs = [to_np(v) for v in vals]
        if all(a.shape == arrs[0].shape for a in arrs):
            s = np.stack(arrs)
            out[k] = s.reshape((s.shape[0],) + s.shape[2:]) if s.shape[1] == 1 else s
        else:
            out[k] = np.concatenate([a.reshape(-1) for a in arrs])
            out[k + "@offsets"] = np.cumsum([0] + [a.size for a in arrs]).astype(np.int64)
            out[k + "@context"] = np.array([a.shape[-1] for a in arrs], dtype=np.int64)
    return out


KINDS = {"/qx": "int8 activations", "/acc": "int32 accumulators", "/s_x": "int8 scales",
         ".silu": "bf16 lookups", ".exp": "bf16 lookups", ".recip": "bf16 lookups",
         ".rsqrt": "bf16 lookups", ".probs": "attention", ".scores": "attention",
         ".x": "residual stream"}


def by_kind(cols):
    """Witness values grouped by what the prover would commit them as."""
    out = {}
    for k, v in cols.items():
        if k.endswith("@offsets") or k.endswith("@context"):
            continue
        kind = next((n for s, n in KINDS.items() if s in k), "other")
        out[kind] = out.get(kind, 0) + int(v.size)
    return out


def circuit_config(m, cols, steps):
    """The op list a circuit has to instantiate, with the witness values each one
    contributes per token (measured from the dump, not modelled)."""
    def values(prefix):
        n = sum(v.size for k, v in cols.items()
                if k.startswith(prefix) and not k.endswith(("@offsets", "@context")))
        return round(n / steps, 1)

    ops = []
    for i, l in enumerate(m.layers):
        for n in ("norm1", "norm2"):
            ops.append(dict(name=f"L{i}.{n}", kind="rmsnorm", width=m.d,
                            lookups=["rsqrt"], values=values(f"L{i}.{n}")))
        for n in ("q", "k", "v", "o", "gate", "up", "down"):
            lin = l[n]
            ops.append(dict(name=lin.name, kind="matmul", n_in=lin.n_in, n_out=lin.n_out,
                            bias=lin.bias is not None, values=values(lin.name + "/")))
        ops.append(dict(name=f"L{i}.rope", kind="elementwise",
                        width=(m.nh + m.nkv) * m.hd, values=0.0,
                        note="cos/sin are preprocessed constants; output is not re-dumped"))
        ops.append(dict(name=f"L{i}.attn", kind="attention", heads=m.nh, head_dim=m.hd,
                        lookups=["exp", "recip"], values_per_key=4 * m.nh,
                        values=values(f"L{i}.scores") + values(f"L{i}.exp") + values(f"L{i}.probs"),
                        note="scores, exp input, exp output and probs: 4 values per head "
                             "per key; a prover can derive the shifted exp input and commit 3"))
        ops.append(dict(name=f"L{i}.silu", kind="lookup", width=l["gate"].n_out,
                        table="bf16 unary, 2^16 entries", values=values(f"L{i}.silu")))
        ops.append(dict(name=f"L{i}.x", kind="residual", width=m.d, values=values(f"L{i}.x")))
    ops.append(dict(name="final_norm", kind="rmsnorm", width=m.d, lookups=["rsqrt"],
                    values=values("final_norm")))
    ops.append(dict(name="lm_head", kind="matmul", n_in=m.head.n_in, n_out=m.head.n_out,
                    bias=False, values=values("lm_head/")))
    return ops


def generate(n_prompt, n_decode):
    m = Model(MODEL, SPECS["dynamic-int8"], track_acc=True)
    ev, _ = tokens()
    ids = list(ev[:n_prompt].tolist())
    m.rec, cache = Recorder(), Cache(m.L)
    t0 = time.time()
    for step in range(n_prompt + n_decode):
        logits, _ = m.forward(torch.tensor([ids[step]]), cache)
        nxt = int(logits[-1].argmax())
        if step >= len(ids) - 1:
            ids.append(nxt)
    cols = pack(m.rec)
    dt = time.time() - t0
    man = dict(
        model=MODEL, spec=SPECS["dynamic-int8"].asdict(),
        dims=dict(hidden=m.d, layers=m.L, heads=m.nh, kv_heads=m.nkv, head_dim=m.hd,
                  ffn=m.layers[0]["gate"].n_out, vocab=m.head.n_out,
                  rms_eps=float(m.cfg.rms_norm_eps),
                  rope_theta=float(m.cfg.rope_parameters["rope_theta"])),
        sequence=dict(prompt_tokens=n_prompt, decoded_tokens=n_decode,
                      steps=n_prompt + n_decode, token_ids=ids),
        quantization=dict(weights="per-output-channel symmetric int8",
                          activations="per-token dynamic symmetric int8",
                          rounding="half away from zero, clamp +-127",
                          scales="bf16, rounded up so the tensor max never clips",
                          accumulator="exact int32, one M31 element per dot product"),
        ops=circuit_config(m, cols, n_prompt + n_decode),
        columns=sorted({k: dict(dtype=str(v.dtype), shape=list(v.shape))
                        for k, v in cols.items()}.items()),
        acc_max=dict(overall=max(m.acc_max.values()),
                     m31_signed_max=(1 << 30) - 1),
        witness=dict(columns=len(cols), values=int(sum(v.size for v in cols.values())),
                     bytes=int(sum(v.nbytes for v in cols.values())),
                     values_by_kind=by_kind(cols), seconds=round(dt, 1)),
        env=dict(torch=torch.__version__, threads=THREADS, deterministic=True),
        digest=digest(cols))
    return cols, man


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=int, default=32)
    p.add_argument("--decode", type=int, default=32)
    p.add_argument("--check", action="store_true", help="regenerate and compare digests")
    a = p.parse_args()
    setup()
    cols, man = generate(a.prompt, a.decode)
    n = man["witness"]
    steps = man["sequence"]["steps"]
    print(f"witness: {n['columns']} columns, {n['values'] / 1e6:.1f}M values, "
          f"{n['bytes'] / 2 ** 20:.1f} MiB for {steps} tokens "
          f"({n['bytes'] / steps / 2 ** 10:.0f} KiB/token, {n['values'] / steps / 1e3:.0f}k values/token)")
    for k, v in sorted(n["values_by_kind"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:<20} {v / steps / 1e3:8.1f}k values/token  ({v / n['values'] * 100:4.1f}%)")
    print(f"digest: {man['digest']}")
    if a.check:
        old = json.load(open(MANIFEST))["digest"]
        ok = old == man["digest"]
        print(f"stored: {old}\nbit-identical regeneration: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)
    np.savez(WITNESS, **cols)
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print(f"wrote {WITNESS} ({os.path.getsize(WITNESS) / 2 ** 20:.1f} MiB) and {MANIFEST}")
