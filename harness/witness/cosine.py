"""B1 acceptance (E5b): cosine similarity of the logits against fp32.

DeepProve's Table 4 reports cosine similarity of a quantized model's logits
against its float baseline (Gemma-3-270M at 8 bits: 0.2035).  `ppl.py` never
forms this statistic -- it only ever runs one spec per process, so a
comparison spec's logits are never held next to fp32's.  This script runs
the `fp32` model and one comparison spec together, over the same tokens, and
reports the per-position cosine similarity of the full logit vector (all
vocab entries, not just the next-token slice `ppl.py` scores), mean +/- SE
over every evaluated position.

Each 512-token chunk is still one independent context (a fresh `Cache`, same
grouping `ppl.py` scores), but is fed to both models in smaller sub-batches
that extend that cache (`selftest.py` already checks batched vs
token-at-a-time is bit-identical) -- this bounds the peak size of the
per-call (S, vocab) logits tensor instead of forming it at S=512 for two
262144-vocab models at once.

    uv run python b1/cosine.py --spec static --chunks 40
    uv run python b1/cosine.py --spec b10-final --chunks 40
"""
import argparse
import gc
import json
import math
import os
import resource
import time

import numpy as np
import torch

from .. import provenance
from .common import (CHUNK, DEVICE, MODEL, OUT, SPECS, TEXT, THREADS, chunks,
                    setup, tokens)
from .model import Cache, Model
from .ppl import attention_scales, calibration_scales

RESULTS = os.path.join(OUT, "cosine.json")
SUBBATCH = int(os.environ.get("ZK_COSINE_SUBBATCH", 64))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(name, n_chunks):
    spec = SPECS[name]
    scales = calibration_scales() if spec.linear == "static" else None
    ascales = attention_scales(spec) if spec.int_softmax else None
    t0 = time.time()
    base = Model(MODEL, SPECS["fp32"])
    cmp = Model(MODEL, spec, static_scales=scales, attn_scales=ascales)
    ev, _ = tokens()
    cos_parts = []
    for i, c in enumerate(chunks(ev, n_chunks)):
        cache_b, cache_c = Cache(base.L), Cache(cmp.L)
        chunk_cos = []
        for j in range(0, len(c), SUBBATCH):
            sub = c[j:j + SUBBATCH]
            lb, cache_b = base.forward(sub, cache=cache_b)
            lc, cache_c = cmp.forward(sub, cache=cache_c)
            cs = torch.nn.functional.cosine_similarity(lb, lc, dim=-1).clone()
            chunk_cos.append(cs.cpu().numpy().astype(np.float64))
            del lb, lc, cs
        del cache_b, cache_c
        gc.collect()
        cos_parts.append(np.concatenate(chunk_cos))
        running = np.concatenate(cos_parts)
        print(f"  chunk {i + 1:>3}: cos mean {running.mean():.6f}  "
              f"min {running.min():.6f}  rss {rss_mb():7.0f}MB  "
              f"({time.time() - t0:5.1f}s)", flush=True)
    del base, cmp
    gc.collect()
    cos = np.concatenate(cos_parts)
    assert len(cos) == n_chunks * CHUNK
    mean = cos.mean()
    se = cos.std(ddof=1) / math.sqrt(len(cos))
    return dict(spec=spec.asdict(), cosine_mean=float(mean), cosine_se=float(se),
                cosine_min=float(cos.min()), tokens=len(cos), chunks=n_chunks,
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, choices=[s for s in SPECS if s != "fp32"])
    p.add_argument("--chunks", type=int, default=40)
    a = p.parse_args()
    setup()
    out = run(a.spec, a.chunks)
    print(f"   cosine({a.spec}, fp32) = {out['cosine_mean']:.6f} +/- {out['cosine_se']:.6f}  "
          f"(min {out['cosine_min']:.6f})")
    res = provenance.rows(json.load(open(RESULTS))) if os.path.exists(RESULTS) else {}
    res[a.spec] = out
    json.dump({"provenance": provenance.block(
                   model=MODEL, device=DEVICE, threads=THREADS,
                   module="harness.witness.cosine",
                   specs_written=[a.spec], specs_in_file=sorted(res),
                   chunks=a.chunks, corpus=provenance.digest(TEXT)),
               "result": res}, open(RESULTS, "w"), indent=1)
