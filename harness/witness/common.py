"""Shared setup for B1: determinism, corpus slices, spec presets."""
import hashlib
import os

import numpy as np
import torch

from model import Spec

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TEXT = os.path.join(HERE, os.pardir, "text.txt")
THREADS = 4
CHUNK = 512

# The three specs B1 has to compare, plus probes that isolate one choice each.
SPECS = {
    "fp32":        Spec(linear="fp32",    nonlinear="fp32"),
    "dynamic-int8":  Spec(linear="dynamic", nonlinear="bf16"),
    "static":      Spec(linear="static",  nonlinear="bf16"),
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
}


def setup():
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
