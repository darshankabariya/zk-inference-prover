"""Shared setup for B1: determinism, corpus slices, spec presets."""
import hashlib
import os
from dataclasses import replace

import numpy as np
import torch

from model import Spec

HERE = os.path.dirname(os.path.abspath(__file__))
# B8 points the same code at an 8B model on a GPU: the model, the output
# directory and the device come from the environment, with B1's defaults.
MODEL = os.environ.get("B1_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT = os.environ.get("B1_OUT", os.path.join(HERE, "out"))
DEVICE = os.environ.get("B1_DEVICE", "cpu")
TEXT = os.path.join(HERE, os.pardir, "text.txt")
THREADS = int(os.environ.get("B1_THREADS", 4))
CHUNK = 512

# The three specs B1 has to compare, plus probes that isolate one choice each.
SPECS = {
    "fp32":        Spec(linear="fp32",    nonlinear="fp32"),
    "dynamic-int8":  Spec(linear="dynamic", nonlinear="bf16"),
    "static":      Spec(linear="static",  nonlinear="bf16"),
    # B9's rider: the INT8 baseline that is actually deployed.  Static per-tensor
    # W8A8 (`static`) is the weakest INT8 there is; weight-only per-channel int8
    # with float activations is what a "public INT8 variant" normally means, and
    # it is the row the claim "beats a public INT8 variant" has to be read against.
    "w8a16":       Spec(linear="w8a16",   nonlinear="bf16"),
    # decomposition of the dynamic-int8 spec's cost
    "int8-only":   Spec(linear="dynamic", nonlinear="fp32"),
    "bf16-only":   Spec(linear="fp32",    nonlinear="bf16"),
    "fp32-logits": Spec(linear="dynamic", nonlinear="bf16", fp32_logits=True),
    # which matmul carries the int8 loss: hold one kind out of int8 at a time
    "keep-head":   Spec(linear="dynamic", nonlinear="bf16", fp32_matmuls=("lm_head",)),
    "keep-down":   Spec(linear="dynamic", nonlinear="bf16", fp32_matmuls=("down",)),
    "keep-o":      Spec(linear="dynamic", nonlinear="bf16", fp32_matmuls=("o",)),
    "keep-qkv":    Spec(linear="dynamic", nonlinear="bf16", fp32_matmuls=("q", "k", "v")),
    "keep-gate-up": Spec(linear="dynamic", nonlinear="bf16", fp32_matmuls=("gate", "up")),
    # design probes for B4/B5
    "bf16-reduce": Spec(linear="dynamic", nonlinear="bf16", reduce="bf16"),
    "bf16-norm":   Spec(linear="dynamic", nonlinear="bf16", reduce="bf16-norm"),
    "bf16-softmax": Spec(linear="dynamic", nonlinear="bf16", reduce="bf16-softmax"),
    "qk-int8":     Spec(linear="dynamic", nonlinear="bf16", qk_int8=True),
    # --- B5 probes -----------------------------------------------------------
    # the free knob: the M31 window is per matmul and Q.K^T reduces over hd = 64,
    # so int12 x int11 costs the same committed cell as int8 x int8
    "qk-int12":    Spec(linear="dynamic", nonlinear="bf16", qk_int8=True, qk_bits=(12, 11)),
    # power-of-two scales everywhere: every dequant multiply becomes an exponent
    # add, and the exp-table input becomes a bit window of the score
    "pow2-scales": Spec(linear="dynamic", nonlinear="bf16", pow2_scales="all"),
    # the integer softmax, in the shape the multiplicity ladder forces
    "int-softmax":      Spec(linear="dynamic", nonlinear="bf16", int_softmax=True),
    "int-softmax-pow2": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             pow2_scales="all"),
    # the lever that matters: pow2 only in the attention path, which is what turns
    # the exp-table input into a bit window of the score
    "int-softmax-p2qk": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             pow2_scales="attn"),
    "int-softmax-12p2": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             qk_bits=(12, 11), pow2_scales="attn"),
    "int-softmax-12":   Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             qk_bits=(12, 11)),
    # diagnostics: is the loss the 8-bit numerator, or the calibrated K/V scales?
    "int-softmax-n16":  Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             num_bits=16),
    "int-softmax-dyn":  Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             attn_scale_mode="cache"),
    # one Q scale per (layer, head) as well: the exp-table exponent becomes a
    # public constant and the table can be indexed by the integer shift itself
    "int-softmax-12sq": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                             qk_bits=(12, 11), pow2_scales="attn", q_scale_mode="head"),
    # --- B6 probes -----------------------------------------------------------
    # The dequant's numeric price, isolated.  B1 forms `acc s_x s_w` in fp32 and
    # rounds once; a gadget with general bf16 scales can only prove a chain of
    # three correctly-rounded steps, so this is what the `chain` variant computes.
    "bf16-dequant": Spec(linear="dynamic", nonlinear="bf16", dequant="bf16-chain"),
    # Power-of-two weight and activation scales, which collapse the dequant to one
    # rounding plus an exponent add: 15 cells per accumulator instead of 26, and
    # numerically identical to B1's fp32-then-round because the scale is exact.
    "pow2-dequant": Spec(linear="dynamic", nonlinear="bf16", pow2_scales="linear"),
    # The attention output rounded straight to int8, on top of B5's design point:
    # deletes the bf16 waypoint, the reciprocal lookup and the requantize.
    "int-attnout":  Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                         qk_bits=(12, 11), pow2_scales="attn", q_scale_mode="head",
                         attn_out_int8=True),
    # Everything B6's AIR pins, in one spec: B5's integer softmax at int12 x int11,
    # power-of-two scales everywhere (so every dequant is a single rounding), and
    # the attention output straight to int8.
    "b6-design":    Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                         qk_bits=(12, 11), pow2_scales="all", q_scale_mode="head",
                         attn_out_int8=True),
    # the same without power-of-two linear scales, i.e. paying the 26-cell chain
    "b6-design-bf16": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                           qk_bits=(12, 11), pow2_scales="attn", q_scale_mode="head",
                           attn_out_int8=True, dequant="bf16-chain"),
    # --- B6.5 probes (B6 review) ----------------------------------------------
    # The int8 attention-output gadget's no-wrap window needs `den = Z`, i.e. a
    # power-of-two output scale; `int-attnout` and `b6-design-bf16` measured a
    # general bf16 `s_o` the 9-cell gadget cannot prove at C = 16K.  This is the
    # provable spec: `b6-design-bf16` with `s_o` rounded up to a power of two.
    "b6-design-p2o": Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                          qk_bits=(12, 11), pow2_scales="attn", q_scale_mode="head",
                          attn_out_int8=True, dequant="bf16-chain",
                          attn_out_pow2=True),
    # Round::Away on the chain: no sticky bit, one fewer witness column and half
    # the table rows (25.12 against 26.12 cells) -- open gap 5's missing number.
    "bf16-dequant-away": Spec(linear="dynamic", nonlinear="bf16",
                              dequant="bf16-chain", dequant_round="away"),
    # Where does pow2-dequant's +2.29% live?  Power-of-two linear scales for one
    # matmul group at a time; the union of the four is `pow2-dequant`.  Whichever
    # groups come out free can take the 15-cell dequant instead of the 26-cell
    # chain in the composed layer.
    "pow2-qkvo":   Spec(linear="dynamic", nonlinear="bf16",
                        pow2_matmuls=("q", "k", "v", "o")),
    "pow2-gateup": Spec(linear="dynamic", nonlinear="bf16",
                        pow2_matmuls=("gate", "up")),
    "pow2-down":   Spec(linear="dynamic", nonlinear="bf16", pow2_matmuls=("down",)),
    "pow2-head":   Spec(linear="dynamic", nonlinear="bf16", pow2_matmuls=("lm_head",)),
    # --- B6.5: the composed layer's numeric spec -------------------------------
    # The mixed dequant layout the B6 review's per-matmul attribution licenses:
    # `b6-design-p2o` (integer softmax at int12 x int11, power-of-two attention
    # scales, one Q scale per head, the attention output rounded straight to int8
    # with a power-of-two `s_o`, the 26-cell chain elsewhere) plus power-of-two
    # *linear* scales on exactly the matmuls where they are free (q/k/v/o, -0.43%),
    # keeping general bf16 scales on gate/up/down where they are not.
    "b65-design":  Spec(linear="dynamic", nonlinear="bf16", int_softmax=True,
                        qk_bits=(12, 11), pow2_scales="attn", q_scale_mode="head",
                        attn_out_int8=True, attn_out_pow2=True, dequant="bf16-chain",
                        pow2_matmuls=("q", "k", "v", "o")),
}

# --- B8 probes ---------------------------------------------------------------
# At 8B the 8-bit numerator is the largest single term in `b6-design-p2o`'s cost
# (`int-softmax-n16` erases it), so the design point with a wider numerator.  The
# width is pinned by the M31 window on the A.V accumulator, n_bits + 7 + log2(C)
# <= 30: 9 bits at C = 16K, 11 at 4K, 13 at 1K.
for _n in (9, 10, 12, 16):
    SPECS[f"b6-design-p2o-n{_n}"] = replace(SPECS["b6-design-p2o"], num_bits=_n)

# --- B10: the one spec that is both scored and proved ------------------------
# B9 left the quality row and the throughput row on two different systems: the
# chunk that proved was extracted under `b65-design` (B6.5's mixed layout, the
# power-of-two dequant on q/k/v/o) while the layout B8 licensed and the
# denominator charges is the bf16 chain on all seven layer matmuls.  `b10-design`
# is that layout, frozen with the two knobs B9 priced: ties-away rounding on the
# chain (one fewer witness column, half the table rows) and the 9-bit numerator
# (0 cells per accumulator, a few shared-table rows).  Everything B10 proves and
# everything B10 scores is this spec.
SPECS["b10-design"] = replace(SPECS["b6-design-p2o"],
                              dequant_round="away", num_bits=9)

# B10 step 2: the two knobs the *glue* decides, and owes a row for.  Proving the
# RMSNorm pins both.  The layer's rounding mode stops being the dequant's choice
# -- `store()` is ties-to-even and one shared `NORM` table is built in one mode,
# so the whole layer rounds even -- and the reduction's *order* stops being B1's
# fp32 accumulation, which no bf16-add gadget reproduces, in favour of B1's own
# balanced tree.  `b10-glue` is `b10-design` with both, i.e. the spec the AIR
# actually proves; `b10-design-even` isolates the mode so the two are separable.
SPECS["b10-design-even"] = replace(SPECS["b10-design"], dequant_round="even")
SPECS["b10-glue"] = replace(SPECS["b10-design"], dequant_round="even",
                            reduce="bf16-norm")

# B10's own list: one spec, scored against B8's fp32 and `b6-design-p2o` rows
B10_SPECS = ["b10-design", "b10-design-even", "b10-glue"]

# B5's probes, kept apart so B1's own table stays what B1 measured
B5_SPECS = ["qk-int12", "pow2-scales", "int-softmax", "int-softmax-12",
            "int-softmax-p2qk", "int-softmax-12p2", "int-softmax-pow2",
            "int-softmax-n16", "int-softmax-dyn", "int-softmax-12sq"]

# B6's probes, same discipline
B6_SPECS = ["bf16-dequant", "pow2-dequant", "int-attnout", "b6-design",
            "b6-design-bf16"]

# the B6 review's probes, run before B6.5 freezes the composed layout,
# plus the composed layout's own spec (the one B6.5's AIR pins)
B65_SPECS = ["b6-design-p2o", "bf16-dequant-away", "pow2-qkvo", "pow2-gateup",
             "pow2-down", "pow2-head", "b65-design"]

# B8: the rows re-measured at 8B -- the three the plan names, the decomposition
# of the attested spec, the B6 ladder around `b6-design-p2o`, and the two riders
B8_SPECS = ["fp32", "dynamic-int8", "static", "int8-only", "bf16-only",
            "bf16-dequant", "bf16-dequant-away", "pow2-dequant",
            "b6-design", "b6-design-bf16", "b6-design-p2o", "b65-design",
            "pow2-qkvo", "pow2-gateup", "pow2-down", "pow2-head",
            "b6-design-p2o-n9", "b6-design-p2o-n10", "b6-design-p2o-n12", "b6-design-p2o-n16",
            # B9's rider (plan, B9 step 3): the strong INT8 comparator
            "w8a16"]


def setup():
    if DEVICE.startswith("cuda"):
        # deterministic cuBLAS needs a fixed workspace; and the fp32 matmuls
        # must stay fp32 (the exact-integer paths rely on it)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)


def tokens():
    """Eval tokens and a disjoint calibration slice of the same corpus."""
    cache = os.path.join(OUT, "tokens.npz")
    if not os.path.exists(cache):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        text = open(TEXT, encoding="utf-8").read()
        ev = np.array(tok(text[10_000:200_000]).input_ids[:40 * CHUNK], dtype=np.int64)
        cal = np.array(tok(text[400_000:440_000]).input_ids[:4 * CHUNK], dtype=np.int64)
        np.savez(cache, eval=ev, calib=cal)
    z = np.load(cache)
    return torch.from_numpy(z["eval"]), torch.from_numpy(z["calib"])


def chunks(ids, n=None):
    out = [ids[i * CHUNK:(i + 1) * CHUNK] for i in range(len(ids) // CHUNK)]
    return out if n is None else out[:n]


def digest(arrays):
    """Content hash of a witness: sha256 over every array's bytes, key-sorted."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
