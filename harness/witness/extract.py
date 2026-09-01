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

**Stream DD (E11): the block axis.**  The forward loop below already prefills the
whole concatenation in `--chunk` blocks, and `LastChunk.add` *assigns*, so every
recorded column is computed for every block and then overwritten by the next.
Dumping a prompt block is therefore a loop inversion and not a new harness: the
dump body moved inside the forward loop, the `P = C - S` assertion relaxed to a
whole number of blocks, and a block axis on the output directory.  `--blocks`
selects which blocks are written; its default, `last`, is exactly the pre-DD
behaviour, byte for byte, and is what the identity control checks.

    uv run python b65/extract.py --tokens 4608 --chunk 512 --layers 0,1 \\
        --blocks all --corner --slim --share out/w/shared --out out/w/witness
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
from quant import pow2_ceil, round_hafz  # noqa: E402

OUT = os.path.join(HERE, "out", "witness")
MATS = ["q", "k", "v", "o", "gate", "up", "down"]

# B10 step 1: the attention block's three *derived* (nh, S, C) columns.
# `AttnWitness::load` reads only `qq.i16`, `kk.i16`, `vv.i8` and `num.i32` from a
# dump -- the scores, the `d = max - sc` gaps and the bf16 exp-table inputs are
# what the AIR recomputes and commits itself, and nothing in the prover opens
# them.  They are kept for diagnosis, and at C=16K they are 2.50 GB of a 4.03 GB
# witness, which is the difference between the model's 32 layers fitting on a
# disk and not.  `--slim` hashes them into the manifest and does not write them.
# `mm` and `zz` stay: they are (nh, S), 64 KiB, and the model-wide shape table
# reads `zz` (`b10/gpu/shapes32.py` computes `ao_rem_bits` from it).
DERIVED = ("sc.i32", "dd.i32", "xx.u16", "scc.i32", "ddc.i32", "xxc.u16")
SLIM = False
# Stream DD: the second disk lever.  `num.i32` is 84.2% of the per-prefix-key
# term (nh*S*4 of nh*hd*3 + nh*S*4 = 65,536 of 77,824 bytes at 8B) and the
# prover reads it for one thing only -- `torch_num`, which becomes
# `AttnWitness::torch_mismatch` and feeds no constraint (`attn_witness.rs:111,
# 851-853`; `attn_run.rs:104` reports it).  `--slim-num` hashes it into the
# manifest and does not write it, exactly as `--slim` does for the derived
# columns.  It is OFF by default and it needs a prover that reads the file as
# optional: today `Half::read` opens it unconditionally, so a `--slim-num` dump
# does not load.  See `briefs/dd-extract.md`'s Record for the policy call.
NUMFILES = ("num.i32", "numc.i32")
SLIM_NUM = False
# Stream DD: the first disk lever.  These files are a function of the *layer*
# and not of the token block -- the quantized weights, their scales, the two
# norm weights and the offline V scales -- so one copy serves every block.
# `--share DIR` writes them once under `DIR/L<k>/` and leaves a *relative*
# symlink in each block's dump (the pattern at `b10/out/x/ctl_witness_L1/`).
# The manifest's sha256 is taken from the array in memory either way, so a
# shared dump hashes identically to a copied one.
SHARE = None


def is_shared(name):
    """Is this file the same for every token block of one layer?"""
    return (name.startswith(("qw_", "jw_", "sw_"))
            or name in ("jv.i32", "w1.u16", "w2.u16"))


class LastChunk(Recorder):
    """Keep only the newest value of each column: the newest prefill chunk.

    Stream DD: this *assignment* (against `Recorder.add`'s append) is the whole
    reason the block axis is free.  The recorder holds exactly one block, so the
    dump body reads block `b` if it runs after block `b`'s forward call and
    block `B-1` if it runs after the loop -- and nothing else has to change.
    """

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def add(self, key, t):
        if key in self.keep:
            # B9: B8 runs the forward pass on the GPU, so a recorded column is a
            # cuda tensor and every `.numpy()` below would fail; the dump is host
            # bytes either way, so it comes across here, once, per chunk.
            self.cols[key] = t.detach().to("cpu").clone()


def write(name, arr, digests):
    arr = np.ascontiguousarray(arr)
    held = (SLIM and name in DERIVED) or (SLIM_NUM and name in NUMFILES)
    link = None
    if not held and SHARE is not None and is_shared(name):
        # one copy per layer, symlinked into this block's dump.  Written only
        # when it is missing or the wrong length, so a re-run of one block does
        # not rewrite 218 MB of weights and a half-written file is not reused.
        src = os.path.join(SHARE, name)
        os.makedirs(SHARE, exist_ok=True)
        if not os.path.exists(src) or os.path.getsize(src) != arr.nbytes:
            arr.tofile(src)
        dst = os.path.join(OUT, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        link = os.path.relpath(src, OUT)
        os.symlink(link, dst)
    elif not held:
        arr.tofile(os.path.join(OUT, name))
    digests[name] = dict(values=int(arr.size), dtype=str(arr.dtype), shape=list(arr.shape),
                         sha256=hashlib.sha256(arr.tobytes()).hexdigest()[:16])
    if held:
        digests[name]["written"] = False
    if link:
        digests[name]["link"] = link
    note = ("  [derived, hashed only]" if held
            else f"  [-> {link}]" if link else "")
    print(f"  {name:<14} {arr.size:>12,} values ({arr.nbytes / 2 ** 20:6.1f} MiB)  {arr.shape}"
          f"{note}")


def parse_blocks(spec, n):
    """Which token blocks to dump.  `last` is the pre-DD behaviour."""
    if spec in ("", "last"):
        return {n - 1}
    if spec == "all":
        return set(range(n))
    if spec == "prompt":                       # every block but the decoded one
        return set(range(n - 1))
    want = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            want |= set(range(int(lo), int(hi) + 1))
        else:
            want.add(int(part))
    bad = sorted(b for b in want if not 0 <= b < n)
    assert not bad, f"--blocks names {bad}, outside 0..{n - 1}"
    return want


def bits_for(v):
    return int(v).bit_length()


def bf16_parts(t):
    """(mantissa field, biased exponent) of a bf16-valued tensor."""
    # B9: on the B8 path the model's own tensors live on the GPU
    bits = t.detach().to("cpu").to(BF16).view(torch.int16).numpy().astype(np.uint16)
    return (bits & 0x7F).astype(np.int32), ((bits >> 7) & 0xFF).astype(np.int32)


def _pow2_mat(spec, name):
    """Does this spec quantize `name` with power-of-two scales? (`model.py`'s rule)"""
    return spec.pow2_scales in ("all", "linear") or name in spec.pow2_matmuls


def pow2_exponent(t):
    """`j` such that the value is exactly `2^j`; NaN-safe, and -1000 if it is not."""
    v = t.detach().to("cpu").double().numpy()
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
    # B10 step 1: the sequence's witness is a real *decode*, so the ids are the
    # prompt plus `b10/gpu/gen_ids.py`'s greedy continuation rather than the
    # eval corpus's next 512 tokens (B8 step 1: decode is bit-identical to
    # prefill, so the extractor still prefills the concatenation).
    p.add_argument("--ids", default=None,
                   help="token ids (.npy) to run instead of the corpus's first --tokens")
    p.add_argument("--slim", action="store_true",
                   help="hash the derived attention columns into the manifest, do not write them")
    # ---- stream DD (E11): the block axis and the two disk levers -------------
    p.add_argument("--blocks", default="last",
                   help="which token blocks to dump: 'last' (the pre-DD behaviour, and "
                        "the only value that leaves the output paths unchanged), 'all', "
                        "'prompt' (every block but the decoded one), or a list/range "
                        "like '1-8' or '0,3,4'")
    p.add_argument("--share", default=None,
                   help="write the block-invariant files (qw_*, jw_*, sw_*, jv, w1, w2) "
                        "once under DIR/L<k>/ and symlink them into every block's dump; "
                        "177.5 -> 121.6 GB on a 4096+512 sequence at 32 layers")
    p.add_argument("--slim-num", action="store_true",
                   help="hash num.i32/numc.i32 into the manifest and do not write them "
                        "(121.6 -> ~74 GB, and it needs a prover that reads them as "
                        "optional -- see the module docstring)")
    p.add_argument("--glue", action="store_true",
                   help="also dump B10 step 2's wiring columns (residual stream, "
                        "norm weights and rsqrt/silu lookups)")
    p.add_argument("--head", action="store_true",
                   help="also dump B10 step 4's `lm_head` chunk (the accumulators, the "
                        "quantized weights, the logits' order keys and the emitted tokens)")
    p.add_argument("--corner", action="store_true",
                   help="also dump B10 step 3's causal corner (the S decoded tokens "
                        "against each other): one joint row maximum, the total "
                        "denominator, and the corner's own keys and values")
    a = p.parse_args()
    SLIM = a.slim
    SLIM_NUM = a.slim_num
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
        # B10 step 1: `expnum/in` and `expnum/out` are (nh, S, C) and nothing
        # below reads them -- `num` is recomputed from `xx` here.  At C=16K they
        # are 1.5 GB of host memory per layer, and holding 32 layers from one
        # forward pass is what makes the model's 32 dumps one pass and not eight.
        keep |= {f"L{L}.qq", f"L{L}.kk", f"L{L}.vv", f"L{L}.qkacc", f"L{L}.sqk",
                 f"L{L}.attnq", f"L{L}.s_o", f"L{L}.zz"}
        keep |= {f"L{L}.{n}/acc" for n in mats} | {f"L{L}.{n}/s_x" for n in mats}
        keep |= {f"L{L}.{n}/qx" for n in mats} | {f"L{L}.{n}/x" for n in mats}
        # B10 step 2: the glue's own columns -- the residual stream at the three
        # points the two adds join, each add's second operand, the two norms'
        # rsqrt lookups and the SwiGLU's.  `--glue` gates them because they are
        # 5 (S, d) + 2 (S, dff) bf16 columns, which at C=16K is another 1.2 GB a
        # layer and nothing in steps 0-1 reads them.
        if a.glue:
            keep |= {f"L{L}.xin", f"L{L}.xmid", f"L{L}.x", f"L{L}.o_out", f"L{L}.d_out",
                     f"L{L}.norm1.rsqrt/in", f"L{L}.norm1.rsqrt/out",
                     f"L{L}.norm2.rsqrt/in", f"L{L}.norm2.rsqrt/out",
                     f"L{L}.silu/in", f"L{L}.silu/out"}
    if a.head:
        # B10 step 4: the head is a `Linear` like any other, but it has no layer
        # prefix -- `m.head = Linear(..., "lm_head", ...)` -- so its recorder keys
        # are bare.
        keep |= {"lm_head/acc", "lm_head/qx", "lm_head/s_x"}
    m.rec = LastChunk(keep)
    if a.ids:
        ids = torch.from_numpy(np.load(a.ids).astype(np.int64))
        assert len(ids) == a.tokens, f"--ids has {len(ids)} tokens, --tokens says {a.tokens}"
        print(f"  ids from {a.ids}: {len(ids)} tokens, the last {a.chunk} decoded")
    else:
        ev, _ = tokens()
        ids = ev[: a.tokens]
    cache = Cache(m.L)
    t0 = time.time()
    logits = None

    nblocks = (len(ids) + a.chunk - 1) // a.chunk
    # Stream DD: the block axis is only well defined when every block is the
    # same width -- a ragged last block would give it a different `S` from the
    # one its own prefix is a whole number of, and every driver in the tree
    # already passes a multiple.
    assert len(ids) % a.chunk == 0, \
        f"--tokens {len(ids)} is not a whole number of --chunk {a.chunk} blocks"
    want = parse_blocks(a.blocks, nblocks)
    # `last` leaves the output paths exactly where they were, which is what the
    # identity control (re-extraction reproduces the existing dump byte for
    # byte) compares against.  Any other value tags the block into the path.
    tag_blocks = a.blocks not in ("", "last")
    print(f"  {nblocks} blocks of {a.chunk}; dumping {sorted(want)}"
          f"{' (block-tagged paths)' if tag_blocks else ''}")

    def dump_blocks(bi, base):
        """Dump every requested layer's witness for token block `bi`.

        Stream DD's loop inversion.  `bi` is the block index and `base` the
        position of its first token, so the recorder holds the S queries of
        positions `base .. base + S - 1` against the whole `base + S` key
        prefix.  Called from inside the forward loop; called after it for the
        last block only, it is the pre-DD dump.
        """
        global OUT, SHARE

        for L in layers:
            # Stream DD: the block axis, and it is entirely in the *name*.  The
            # prover knows no directory grammar (`grep -rn "witness_L" prover/`
            # returns nothing; every path is `format!("{dir}/<literal>")`), so a
            # block is a directory and nothing else.  `_B<b>` goes before the
            # `_L<k>` the shape table already walks, which is what lets
            # `shapes32.py <base>_B<b>_L 32` read one block unchanged.
            base_dir = f"{OUT_BASE}_B{bi}" if tag_blocks else OUT_BASE
            OUT = base_dir if len(layers) == 1 else f"{base_dir}_L{L}"
            SHARE = os.path.join(a.share, f"L{L}") if a.share else None
            os.makedirs(OUT, exist_ok=True)
            print(f"block {bi} layer {L} -> {OUT}")
            digests = {}

            # ---- the attention block, in b5/extract.py's format ---------------------
            sc_all = m.rec.cols[f"L{L}.qkacc"].to(torch.int64)      # (nh, S, C)
            nh, S, C = sc_all.shape
            hd = m.hd
            # The block's rectangle: key c is visible to query t iff c <= (C - S) + t,
            # so the C - S prefix is fully visible and the S x S corner is the only
            # masked part.  Stream DD: `P` is the *block index* times `S` -- block 3
            # of a 512-token sweep has a 1,536-key prefix -- so the pre-DD
            # power-of-two assertion was a restriction to the last block of a
            # power-of-two prompt and nothing deeper.  What the rest of the body
            # actually needs is that the prefix is a whole number of blocks.
            P = C - S
            assert P >= 0 and P % S == 0, \
                f"prefix {P} must be a whole number of {S}-token blocks"
            assert P or a.corner, \
                "block 0 has no prefix at all: it is corner-only and needs --corner"
            sc = sc_all[:, :, :P].contiguous()
            # B10 step 3: the causal corner.  Key k of it *is* decoded token k, so token
            # t attends to it exactly when `k <= t`; `vis[t, k]` is that mask, and it is
            # a function of the position alone, which is why the AIR can preprocess it.
            scc = sc_all[:, :, P:].contiguous() if a.corner else None
            vis = (torch.arange(S).unsqueeze(0) <= torch.arange(S).unsqueeze(1)) if a.corner \
                else None

            # ONE row maximum for the pair.  `m_t` is a property of the token, not of a
            # half: the softmax is shift-invariant only when the same maximum is taken
            # against every key of the row, and the two halves' numerators would
            # otherwise be on two different scales (B10 rule / README 6.2).
            # `amax` has no identity on an empty axis, so block 0's maximum comes
            # from the corner alone.  Everything downstream of `mm` is written
            # against a zero-width prefix without further help: `dd`, `num` and
            # `xx` are empty, `zz` is the corner's sum, and the A.V einsum over a
            # zero-length key axis is the zero it should be.
            mm = (sc.amax(-1) if P else
                  torch.full((nh, S), torch.iinfo(torch.int64).min, dtype=torch.int64))
            if a.corner:
                mm = torch.maximum(mm, scc.masked_fill(~vis, torch.iinfo(torch.int64).min)
                                   .amax(-1))
            sqk = m.rec.cols[f"L{L}.sqk"].double()
            nmax = (1 << spec.num_bits) - 1

            def numer(d):
                """The spec's numerator for a shift array, and the bf16 exp input."""
                x = (-d.double() * sqk).float().to(BF16)
                n = round_hafz(nmax * torch.exp(x.double())).clamp(0, nmax).to(torch.int64)
                return x, n

            dd = mm.unsqueeze(-1) - sc
            xx, num = numer(dd)
            zz = num.sum(-1)
            if a.corner:
                # the true shift, which the AIR commits for *every* slot -- Freivalds is
                # over the whole rectangle -- and which is negative wherever a masked
                # key outscores the maximum the token was shifted by
                ddc = mm.unsqueeze(-1) - scc
                zero = torch.zeros_like(ddc)
                xxc, numc = numer(torch.where(vis, ddc, zero))
                numc = torch.where(vis, numc, torch.zeros_like(numc))
                zz = zz + numc.sum(-1)

            write("qq.i16", m.rec.cols[f"L{L}.qq"].numpy().astype(np.int16), digests)
            write("kk.i16", m.rec.cols[f"L{L}.kk"][:, :P].numpy().astype(np.int16), digests)
            write("vv.i8", m.rec.cols[f"L{L}.vv"][:, :P].numpy().astype(np.int8), digests)
            write("sc.i32", sc.numpy().astype(np.int32), digests)
            write("dd.i32", dd.numpy().astype(np.int32), digests)
            write("num.i32", num.numpy().astype(np.int32), digests)
            write("xx.u16", xx.contiguous().view(torch.int16).numpy().view(np.uint16), digests)
            write("mm.i32", mm.numpy().astype(np.int32), digests)
            write("zz.i32", zz.numpy().astype(np.int32), digests)
            if a.corner:
                write("kkc.i16", m.rec.cols[f"L{L}.kk"][:, P:].numpy().astype(np.int16), digests)
                write("vvc.i8", m.rec.cols[f"L{L}.vv"][:, P:].numpy().astype(np.int8), digests)
                write("scc.i32", scc.numpy().astype(np.int32), digests)
                write("ddc.i32", ddc.numpy().astype(np.int32), digests)
                write("numc.i32", numc.numpy().astype(np.int32), digests)
                write("xxc.u16", xxc.contiguous().view(torch.int16).numpy().view(np.uint16),
                      digests)

            # ---- the attention-output fold's scales ---------------------------------
            # The composed attnout region proves the *rectangle* sub-computation (the
            # same statement the bands prove; the causal corner stays outside every AIR
            # in the project, B9's job), so its accumulators and denominators are the
            # recomputed `out`/`zz` above.  The V scales are the forward pass's,
            # power-of-two per (head, channel).
            s_v = attention_scales(spec)[f"L{L}.v"]                  # (nh, 1, hd)
            jv = pow2_exponent(s_v.reshape(nh, hd))
            assert (jv != -1000).all(), "V scales are not powers of two -- wrong spec?"
            write("jv.i32", jv.reshape(-1), digests)

            # B9 step 1b: so must the *output* scale be the rectangle's.  Taking `s_o`
            # from the forward pass -- which attended to the prefix *and* the causal
            # corner -- calibrates it on a different quantity: the corner's numerators
            # enlarge `Z` more than they enlarge `A.V` on some rows, so dropping them
            # makes `A.V s_v / Z` *larger* than the scale was cut for, the quotient
            # leaves the fold's 2^`ao_q_bits` (q, nz) table, its LogUp sum does not
            # cancel and the honest proof is rejected.  That is 1 to 662 outputs of
            # ~33 M on 14 of Qwen-0.5B's 24 layers, with quotients up to 370 against a
            # 127 window.  The gadget's completeness precondition is
            # `max |A.V s_v / Z| <= 127 s_o` over the rows it proves, so the scale is
            # recalibrated on those rows here.  `s_o_fwd` is kept in the manifest so
            # the two are comparable; nothing else in the dump moves.
            out_rect = torch.einsum("hsc,hcd->hsd", num.double(),
                                    m.rec.cols[f"L{L}.vv"][:, :P].double())
            if a.corner:
                # with the corner proved, the AIR's attention output is the forward
                # pass's: both halves, causally masked
                out_rect = out_rect + torch.einsum("hsc,hcd->hsd", numc.double(),
                                                   m.rec.cols[f"L{L}.vv"][:, P:].double())
            real_rect = out_rect * s_v.double().reshape(nh, 1, hd) / zz.double().unsqueeze(-1)
            flat_rect = real_rect.transpose(0, 1).reshape(S, nh * hd)
            # stream Z: the fold's target width is the spec's, not int8's -- the
            # scale calibrates against `qmax_o` and the whole dump declares it
            qmax_o = (1 << (spec.attn_out_bits - 1)) - 1
            s_o_rect = pow2_ceil(flat_rect.abs().amax(-1, keepdim=True).clamp_min(1e-30)
                                 / float(qmax_o))
            s_o_fwd = m.rec.cols[f"L{L}.s_o"].float()                # (S, 1)
            jo = pow2_exponent(s_o_rect.reshape(-1).float())
            jo_fwd = pow2_exponent(s_o_fwd.reshape(-1))
            assert (jo != -1000).all(), "the rectangle's s_o is not a power of two"
            assert (jo_fwd != -1000).all(), "s_o is not a power of two -- wrong spec?"
            write("jo.i32", jo, digests)
            # the quotient the fold will build, so the precondition is checked here and
            # not four hours later inside a prover
            q_rect = round_hafz(flat_rect.abs() / s_o_rect.double())
            assert int(q_rect.max()) <= qmax_o, \
                f"fold quotient {int(q_rect.max())} > {qmax_o}"
            jo_bumped = int((jo != jo_fwd).sum())
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
                # stream Z: the widened attention output makes o's quantized input
                # int16; the file name carries the storage width so a loader cannot
                # misread it
                if name == "o" and spec.attn_out_bits > 8:
                    write(f"qx_{name}.i16",
                          m.rec.cols[f"L{L}.{name}/qx"].numpy().astype(np.int16), digests)
                else:
                    write(f"qx_{name}.i8",
                          m.rec.cols[f"L{L}.{name}/qx"].numpy().astype(np.int8), digests)
                write(f"qw_{name}.i8", lin.qw.detach().to("cpu").numpy().astype(np.int8),
                      digests)
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

            # ---- stream Z: the fold's own identity, at the witness -------------------
            # `y-delay-*` carries v's output as the exact product and the A.V quantize
            # is the one rounding.  The AIR will prove exactly
            # `round_hafz(acc s_x s_w / s_v)`, so that quantity is recomputed here from
            # the dumped arrays alone and byte-compared against the recorder's own
            # `vvc` (the S decoded tokens' values are the corner's) -- the identity
            # gate's witness half, checked at extraction and not hours later.  The
            # no-clamp precondition rides along: a clamped site has no honest witness.
            fold_stats = {}
            if "v" in spec.delay_matmuls:
                assert a.corner, "the v fold's identity check reads the corner's vvc"
                accv = m.rec.cols[f"L{L}.v/acc"].double()               # (S, nkv*hd)
                linv = m.layers[L]["v"]
                # the fold's statement is `round_hafz(acc s_x s_w / s_v)` and nothing
                # else -- a v_proj bias (Qwen) sits between the product and the
                # quantize, and no bias-free gadget can state that site
                assert linv.bias is None, \
                    "v_proj carries a bias -- the fold cannot state this model's v site"
                sxv = m.rec.cols[f"L{L}.v/s_x"].double()                # (S, 1)
                swv = linv.s_w.detach().to("cpu").double().reshape(1, -1)
                rep = nh * hd // linv.n_out
                svq = s_v.detach().to("cpu").double().reshape(nh // rep, rep, hd)
                assert bool((svq == svq[:, :1]).all()), "s_v disagrees inside a GQA group"
                sv_flat = svq[:, 0].reshape(1, -1)                      # (1, nkv*hd)
                qv = round_hafz(accv * sxv * swv / sv_flat)
                # the AIR proves the true rounding and the clamp beside it, so the
                # completeness precondition is the quotient *domain*, not 127
                q_dom = (1 << (spec.attn_out_bits - 1)) - 1
                assert int(qv.abs().max()) <= q_dom, \
                    f"v fold quotient {int(qv.abs().max())} past the {q_dom} domain"
                clamped = int((qv.abs() > 127).sum())
                q_true_max = int(qv.abs().max())
                qv = qv.clamp(-127, 127)
                vvc = m.rec.cols[f"L{L}.vv"][:, P:].double()            # (nh, S, hd)
                vvc_kv = vvc.reshape(nh // rep, rep, S, hd)[:, 0]       # (nkv, S, hd)
                vvc_flat = vvc_kv.permute(1, 0, 2).reshape(S, -1)       # (S, nkv*hd)
                assert bool((qv == vvc_flat).all()), \
                    "v fold identity: clamp(round_hafz(acc s_x s_w / s_v)) != vvc"
                gt = 134 - bf16_parts(m.rec.cols[f"L{L}.v/s_x"].reshape(-1))[1]
                gc = 134 - bf16_parts(linv.s_w.reshape(-1))[1] \
                    + pow2_exponent(sv_flat.reshape(-1).float())
                assert int(gt.min() + gc.min()) >= 14, "fold divisor below 2^14"
                fold_stats = dict(
                    q_max=q_true_max, clamped=clamped,
                    gt_min=int(gt.min()), gt_max=int(gt.max()),
                    gc_min=int(gc.min()), gc_max=int(gc.max()),
                    fold_rem_bits=int(gt.max() + gc.max() - 13),
                    fold_alpha=int(max(14 - gc.min(), 0)),
                )
                print(f"  v fold identity holds on {qv.numel():,} sites "
                      f"({clamped} clamped); rem {fold_stats['fold_rem_bits']} bits, "
                      f"alpha {fold_stats['fold_alpha']}")

            # ---- B10 step 2: the glue's columns -------------------------------------
            # Every one is a bf16 *pattern* column, which is what the wiring family
            # commits: the residual stream in, mid and out, the two adds' second
            # operands, the two norm weights (preprocessed -- they are weights), the
            # two rsqrt lookups and the SwiGLU's.  `eps` and `1/d` are bf16 scalars and
            # go in the manifest.
            glue = {}
            if a.glue:
                def pat(t):
                    return (t.detach().to("cpu").contiguous().to(BF16)
                            .view(torch.int16).numpy().view(np.uint16))

                for key, name in (("xin", "xin"), ("xmid", "xmid"), ("x", "xout"),
                                  ("o_out", "o_out"), ("d_out", "d_out"),
                                  ("norm1.rsqrt/in", "n1_in"), ("norm1.rsqrt/out", "n1_out"),
                                  ("norm2.rsqrt/in", "n2_in"), ("norm2.rsqrt/out", "n2_out"),
                                  ("silu/in", "silu_in"), ("silu/out", "silu_out")):
                    write(f"{name}.u16", pat(m.rec.cols[f"L{L}.{key}"]), digests)
                write("w1.u16", pat(m.layers[L]["norm1"]), digests)
                write("w2.u16", pat(m.layers[L]["norm2"]), digests)
                glue = dict(d=int(m.d), dff=int(m.layers[L]["gate"].n_out),
                            eps=int(pat(m.eps.reshape(1))[0]),
                            inv_d=int(pat(m.inv_d.reshape(1))[0]),
                            reduce=spec.reduce, rms_norm_eps=float(m.eps.item()))

            # ---- B10 step 4: the `lm_head` chunk -------------------------------------
            # The head is a `Linear` like every other matmul, recorded under a bare
            # `lm_head/...` key because it has no layer.  What is new here is the
            # *token* side: the order key the argmax region compares, and which token
            # each row emits.
            head = {}
            if a.head:
                hacc = m.rec.cols["lm_head/acc"].to(torch.int64)          # (S, vocab)
                hsx = m.rec.cols["lm_head/s_x"]
                lin = m.head
                # the files carry the *mat* name, so `Layer::load` reads the head with
                # the same code every other matmul goes through
                write("acc_lm_head.i32", hacc.numpy().astype(np.int32), digests)
                write("qx_lm_head.i8", m.rec.cols["lm_head/qx"].numpy().astype(np.int8), digests)
                write("qw_lm_head.i8", lin.qw.detach().to("cpu").numpy().astype(np.int8), digests)
                jxh = pow2_exponent(hsx.reshape(-1))
                jwh = pow2_exponent(lin.s_w.reshape(-1))
                write("jx_lm_head.i32", jxh, digests)
                write("jw_lm_head.i32", jwh, digests)
                # and the bf16 scale pairs, which the chain rung reads and the
                # power-of-two one ignores -- the dump does not know which it will be
                # composed under, only which the spec says (`deq_variants`)
                mxh, exh = bf16_parts(hsx.reshape(-1))
                mwh, ewh = bf16_parts(lin.s_w.reshape(-1))
                write("sx_lm_head.i32", np.stack([mxh, exh], -1), digests)
                write("sw_lm_head.i32", np.stack([mwh, ewh], -1), digests)
                # B1's own logits for these rows, and the **order key** the argmax
                # compares: bf16 magnitude order is pattern order, but logits are
                # signed, so negatives are flipped into a single ascending 16-bit key.
                lg = logits.detach().to("cpu").to(BF16)
                pat = lg.view(torch.int16).numpy().view(np.uint16).astype(np.int64)
                sign, mag = pat >> 15, pat & 0x7FFF
                key = np.where(sign == 1, 0x7FFF - mag, 0x8000 + mag)
                write("key_head.u16", key.astype(np.uint16), digests)
                tokh = lg.float().argmax(-1).numpy().astype(np.int64)
                write("tok_head.i32", tokh.astype(np.int32), digests)

                # Which tokens do these rows emit?  This block covers positions
                # `base .. base + S - 1`, and row `t` predicts the token at position
                # `base + t + 1` -- so the rows emit the window *shifted by one* from
                # the tokens the layer rectangle is extracted at.  Both alignments
                # are counted rather than assumed, because getting this wrong is
                # silent: the head would prove a real argmax of the wrong row.
                # Stream DD: block-relative.  On the last block `base = N - S` and
                # these are the pre-DD slices to the element.  The comparison stays
                # at `S - 1` rows on every block so the number means one thing:
                # row `S - 1` of the last block predicts a position that is not in
                # `ids` at all.
                idn = ids.numpy().astype(np.int64)
                nxt = int((tokh[:-1] == idn[base + 1:base + S]).sum())
                same = int((tokh == idn[base:base + S]).sum())
                accs["lm_head"] = dict(
                    n_in=int(lin.n_in), n_out=int(lin.n_out), tokens=int(hacc.shape[0]),
                    abs_max=int(hacc.abs().amax()), bits=bits_for(int(hacc.abs().amax())),
                    kmax_needed=max(0, bits_for(int(hacc.abs().amax())) - 9),
                    worst_case=int(127 * 127 * lin.n_in),
                    kmax_worst_case=max(0, bits_for(127 * 127 * lin.n_in) - 9),
                    scales_pow2=bool((jxh != -1000).all() and (jwh != -1000).all()),
                    zero_frac=float((hacc == 0).double().mean()),
                )
                head = dict(
                    vocab=int(lin.n_out), d_in=int(lin.n_in), tokens=int(hacc.shape[0]),
                    tied_embedding=bool(getattr(m, "tied", False)),
                    abs_max=int(hacc.abs().amax()), bits=bits_for(int(hacc.abs().amax())),
                    kmax_needed=max(0, bits_for(int(hacc.abs().amax())) - 9),
                    scales_pow2=bool((jxh != -1000).all() and (jwh != -1000).all()),
                    zero_frac=float((hacc == 0).double().mean()),
                    # the alignment, both ways round
                    emits_next_token=nxt, emits_next_of=int(S - 1),
                    emits_same_position=same, emits_same_of=int(S),
                    # what the tie has to carry (README 9.2 measured this separately)
                    ties=int((lg.float() == lg.float().amax(-1, keepdim=True)).sum(-1).gt(1).sum()),
                )
                print(f"  head  vocab {lin.n_out:,} |acc| <= {int(hacc.abs().amax()):,} "
                      f"({head['bits']} bits) pow2 {head['scales_pow2']}")
                print(f"        rows emit the *next* token on {nxt}/{S - 1}; "
                      f"the same position on {same}/{S}")

            frac = (-torch.log2(sqk)).reshape(nh, -1)
            fpi = frac[:, 0].tolist()
            # B10 step 3: the master table is sized by the largest shift the prover
            # *looks up*, so the corner's visible half counts and its masked half does
            # not -- a masked shift is taken against a maximum over keys it is not one
            # of, can be negative, and is never a table row.
            d_max = int(dd.max()) if dd.numel() else 0
            corner_stats = {}
            if a.corner:
                dvis = ddc[vis.expand_as(ddc)]
                d_max = max(d_max, int(dvis.max()))
                corner_stats = dict(
                    keys=int(S),
                    masked_frac=float((~vis).double().mean()),
                    d_max_visible=int(dvis.max()), d_min_visible=int(dvis.min()),
                    d_min_masked=int(ddc[~vis.expand_as(ddc)].min()),
                    numerator_zero_frac=float((numc[vis.expand_as(numc)] == 0).double().mean()),
                    z_corner_max=int(numc.sum(-1).max()),
                    z_corner_min=int(numc.sum(-1).min()),
                    # which half the row maximum came from, per token: the reason
                    # `Z >= nmax` is a statement about the total and not about a band
                    max_from_corner=int((scc.masked_fill(~vis, torch.iinfo(torch.int64).min)
                                         .amax(-1) > sc.amax(-1)).sum()),
                    rows=int(nh * S))
            stats = dict(
                heads=nh, s=S, c=P, hd=hd, layer=L, spec=a.spec, qk_bits=list(spec.qk_bits),
                # Stream DD: which token block this is.  `c` already says it (a
                # block's prefix is `block * s`), but a driver that walks a
                # sequence should not have to divide to find out, and `blocks`
                # is what says the dump is one of a set rather than a decode.
                block=int(bi), blocks=int(nblocks), base_pos=int(base),
                seq_tokens=int(len(ids)), chunk=int(a.chunk),
                slim_num=bool(SLIM_NUM), shared=bool(a.share),
                num_bits=spec.num_bits, model=MODEL,
                corner=bool(a.corner), corner_stats=corner_stats,
                d_max=d_max, d_bits=bits_for(d_max),
                d_fits_16bit=bool(d_max < (1 << 16)),
                z_max=int(zz.max()), z_worst_case=int(nmax * (P + S if a.corner else P)),
                numerator_zero_frac=(float((num == 0).double().mean()) if num.numel()
                                     else 0.0),
                frac_per_head=fpi,
                frac_shared_per_head=bool(frac.shape[1] == 1 or bool((frac == frac[:, :1]).all())),
                frac_is_integer=bool(((frac - frac.round()).abs() < 1e-9).all()),
                frac_spread=float(max(fpi) - min(fpi)),
                frac_integer_separated=bool(
                    all(abs((max(fpi) - f) - round(max(fpi) - f)) < 1e-9 for f in fpi)),
                master_table_bits=int(bits_for(d_max) + round(max(fpi) - min(fpi))),
                attn_out=dict(jo_min=int(jo.min()), jo_max=int(jo.max()),
                              jv_min=int(jv.min()), jv_max=int(jv.max()),
                              e_min=e_min, e_max=e_max,
                              # B9 step 1b: the fold's scale, calibrated on the
                              # rectangle the AIR proves rather than on the forward
                              # pass's masked output
                              jo_fwd_min=int(jo_fwd.min()), jo_fwd_max=int(jo_fwd.max()),
                              jo_tokens_bumped=jo_bumped, jo_tokens=int(S),
                              fold_q_max=int(q_rect.max())),
                accumulators=accs,
                # B10 step 0: which dequant gadget each matmul needs is a property of
                # the *spec*, not of the prover's build.  `b65-design` puts the
                # power-of-two dequant on q/k/v/o and the bf16 chain on gate/up/down;
                # `b10-design` (= `b6-design-p2o` + away + 9 bits) puts the chain on
                # all seven, which is the layout B8 licensed and the denominator
                # charges.  The witness says so, so a prover cannot compose it under
                # the other layout by accident.
                # B10 step 4: `lm_head` is declared beside the seven, because the head
                # chunk's dequant is 79% of it and the two rungs differ by 52% -- a
                # prover must not have to guess (rule 1).  Note that `b10-design` puts
                # it on the *chain*, while `b2/cost_model.py` prices the design B8
                # licensed (power-of-two on the head, +0.02% at 8B).  That divergence
                # is named in `B10.md` §3; the manifest states what the dump is.
                deq_variants={**{n: ("fold" if n in spec.delay_matmuls
                                     else "pow2" if _pow2_mat(spec, n) else "chain")
                                 for n in mats},
                              **({"lm_head": ("pow2" if _pow2_mat(spec, "lm_head") else "chain")}
                                 if a.head else {})},
                # stream Z: the fold's target width and its measured divisor range --
                # the witness is the knob, so the prover reads both from here
                attn_out_bits=spec.attn_out_bits,
                fold=fold_stats,
                dequant_round=spec.dequant_round,
                clamped={k: dict(over=v[0], total=v[1], frac=v[0] / v[1])
                         for k, v in (m.clamped or {}).items()},
                ids_from=a.ids, slim=bool(SLIM),
                glue=glue, head=head,
                torch=torch.__version__, files=digests,
            )
            json.dump(stats, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
            print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=1))
            # B10 step 1: a layer's recorded columns are ~1.4 GB at C=16K and its
            # working set another ~10 GB, so both go before the next layer starts --
            # 32 layers out of one forward pass against a 128 GiB cgroup.
            del sc, dd, num, xx, mm, zz, out_rect, real_rect, flat_rect, glue
            for key in [k for k in m.rec.cols if k.startswith(f"L{L}.")]:
                del m.rec.cols[key]

    for bi, i in enumerate(range(0, len(ids), a.chunk)):
        logits, cache = m.forward(ids[i:i + a.chunk], cache)
        print(f"  prefilled {min(i + a.chunk, len(ids))}/{len(ids)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # The inversion: the dump body used to run once, here, after the loop.
        # `LastChunk` holds exactly this block, so running it here dumps this
        # block and running it after the loop dumps the last -- same code.
        if bi in want:
            dump_blocks(bi, i)

    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
