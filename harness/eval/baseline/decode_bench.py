"""E2h: the unproved decode-throughput harness.

Measures tokens/second for unproved autoregressive decode, in the shape our
proved numbers are quoted in, so that `unproved / proved` is an overhead
multiple and not a category error.  This is the denominator of the correction to
the community's 500,000x figure (Baker et al., arXiv:2507.15916 fn. 170); the
numerator is frozen in `experiments/paper/constants.json` and is never
re-derived here.

Two rows, because the belief being corrected mixes them:
  single-stream  -- batch 1, greedy, 512 tokens: the like-for-like latency row.
  batched        -- a batch sweep: the production-serving row that gives $/token.
Two backends, for the same reason: `hf` (what a non-specialist gets) and `vllm`
(what the 668 tok/s in the belief actually means).

Every like-for-like dimension is checked in `e2/lfl.py` and the verdict is
written into the run's own JSON.  A dimension that cannot be matched is recorded
with the direction it pushes the answer.

Usage (see e2/run.sh for the exact box invocation):

    uv run python e2/decode_bench.py --backend hf --context 4096 --batch 1
    uv run python e2/decode_bench.py --selftest        # no model, no GPU
    uv run python e2/decode_bench.py --probe           # backend availability
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backends as B  # noqa: E402
import lfl  # noqa: E402

SCHEMA = "e2h/decode-throughput/2"
OUT = Path(os.environ.get("E2_OUT", HERE / "out"))

# What the unproved seconds contain, written into every run so the ratio can be
# audited without re-running anything.  The proved side's four boundaries live in
# `constants.json`'s `_boundaries`; this is the FIFTH thing a reader needs, and
# it is the unproved side's own boundary.
BOUNDARY = {
    "name": "unproved decode wall, weights resident",
    "includes": [
        "the forward passes that produce the measured tokens",
        "sampling/argmax, KV-cache growth, the engine's own scheduling",
    ],
    "excludes": [
        "process start, weight load and any compile/CUDA-graph capture "
        "(a warmup pass precedes every timed rep)",
        "tokenization of the prompt (built once, outside the timed region)",
        "the prefill -- for the decode-only rate ONLY; it is measured by a "
        "separate max_new_tokens=1 call and subtracted. The e11 rate includes it.",
    ],
    "comparable_to": (
        "the proved side's per-chunk boundaries. NOTE: none of the four proved "
        "boundaries contains an inference pass -- `whole wall` is trace gen + "
        "stwo + serialization, and even `loop_secs`, which adds the witness READ "
        "and the layer build, reads a witness that a prior extraction produced. "
        "So every ratio here is prover-seconds against inference-seconds, which "
        "is exactly the quantity fn. 170 computed (zkLLM's 803 s against 668 "
        "tok/s). An end-to-end deployment that must both infer and prove costs "
        "(1 + ratio), not (ratio) -- immaterial at x25-40, and stated anyway."),
}


def device_block(device):
    try:
        import torch
    except ImportError:
        return {"kind": "unknown", "name": "unknown", "count": 0, "torch": None}
    if device.startswith("cuda") and torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        return {"kind": "cuda", "name": p.name, "count": torch.cuda.device_count(),
                "total_mem_mib": p.total_memory // (1 << 20),
                "capability": f"{p.major}.{p.minor}",
                "torch": torch.__version__, "cuda": torch.version.cuda}
    return {"kind": "cpu", "name": platform.processor() or platform.machine(),
            "count": os.cpu_count(), "total_mem_mib": None, "capability": None,
            "torch": torch.__version__, "cuda": None,
            "threads": torch.get_num_threads()}


def _stats(per_rep, key):
    v = [r[key] for r in per_rep]
    return {"median": statistics.median(v), "min": min(v), "max": max(v),
            "spread_pct": (max(v) - min(v)) / statistics.median(v) * 100 if v else None}


def run_one(a, be, context, batch, purpose):
    prompts = B.build_prompts(be.tok, context, batch)
    prompt_sha = B._sha(prompts)
    per_rep, gen_sha = B.measure(be, prompts, a.new_tokens, a.reps, a.warmup)

    run = {
        "model": {"id": a.model, "revision": a.revision, **be.info()},
        "tokenizer": {"from_repo": a.model, "class": type(be.tok).__name__,
                      "vocab_size": len(be.tok)},
        "dtype": a.dtype,
        "device": device_block(a.device),
        "context_keys": context,
        "prompt_tokens": context,
        "new_tokens": a.new_tokens,
        "total_sequence": context + a.new_tokens,
        "batch_size": batch,
        "decoding": {"greedy": True, "do_sample": False, "num_beams": 1,
                     "eos_suppressed": True},
        "reps": a.reps, "warmup": a.warmup,
        "prompt_sha256_16": prompt_sha,
        "generated_sha256_16": gen_sha,
        "synthetic": bool(getattr(be, "synthetic", False)),
    }

    def _agg(st):
        return {k: (v * batch if isinstance(v, float) and k != "spread_pct" else v)
                for k, v in st.items()}

    dec = _stats(per_rep, "tok_s_decode_per_stream")
    e11 = _stats(per_rep, "tok_s_e11_all_positions_per_stream")
    measured = {
        "per_rep": per_rep,
        "prefill_s": _stats(per_rep, "prefill_s"),
        "decode_s": _stats(per_rep, "decode_s"),
        "total_s": _stats(per_rep, "total_s"),
        # --- counting convention 1: decode only, prefill excluded (v1) -------
        "tok_s_decode_per_stream": dec,
        "tok_s_decode_aggregate": _agg(dec),
        # --- counting convention 2: every position of the concatenated pass --
        #     (E11 / DeepProve TPM). Prefill INCLUDED on both sides.
        "tok_s_e11_all_positions_per_stream": e11,
        "tok_s_e11_all_positions_aggregate": _agg(e11),
        "e11_positions_per_stream": context + a.new_tokens,
        # --- reported, used by neither ladder row ----------------------------
        "tok_s_wall_per_stream": _stats(per_rep, "tok_s_wall_per_stream"),
        "ttft_s": _stats(per_rep, "prefill_s"),
        "counting_conventions": {
            "decode_only": "(new_tokens - 1) / decode_s. The v1 comparison: our "
                           "proved seconds cover only the 512 decoded positions "
                           "(scope.prefill_status -- the prefix is declared, not "
                           "proven), so the baseline excludes prefill too. "
                           "Excluding prefill makes the unproved rate LARGER, so "
                           "this choice makes the reported overhead LARGER.",
            "e11": "(prompt_tokens + new_tokens) / total_s. Every position of the "
                   "concatenated prompt+output pass over that pass's whole wall -- "
                   "the convention DeepProve's TPM uses and the one our own "
                   "statement needs once E11 proves the prompt. At C=4096 it "
                   "counts 4608 positions, not 512, so the unproved rate is much "
                   "larger and the multiple grows with C.",
        },
        "ladder_uses": "tok_s_decode_per_stream.median for the single-stream "
                       "latency row; tok_s_decode_aggregate.median for the batched "
                       "$/token row; tok_s_e11_all_positions_*.median for the E11 "
                       "arm",
    }
    return {
        "schema": SCHEMA,
        "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "backend": {"name": be.name, "label": be.label, "version": be.version,
                    "available": True, "reason": None,
                    "synthetic": bool(getattr(be, "synthetic", False))},
        "purpose": purpose,
        "boundary": BOUNDARY,
        "run": run,
        "measured": measured,
        "like_for_like": lfl.check(run, context=context, batch_size=batch, purpose=purpose),
    }


def selftest():
    """The like-for-like checker has teeth: prove it on synthetic run blocks,
    with no model and no GPU."""
    base = {
        "model": {"id": "unsloth/Llama-3.1-8B-Instruct",
                  "revision": "4699cc75b550f9c6f3173fb80f4703b62d946aa5"},
        "tokenizer": {"from_repo": "unsloth/Llama-3.1-8B-Instruct",
                      "class": "PreTrainedTokenizerFast", "vocab_size": 128256},
        "dtype": "bfloat16",
        "device": {"kind": "cuda", "name": "NVIDIA H200", "count": 1},
        "context_keys": 4096, "new_tokens": 512,
        "decoding": {"greedy": True},
    }
    cases = [
        ("the 8B single-stream row the box must produce", base, 1, "single-stream",
         "like-for-like with stated mismatches", True),
        ("the same row batched (production, not latency)", base, 32, "batched",
         "like-for-like with stated mismatches", True),
        ("this CPU validation run", {**base,
            "model": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "main"},
            "tokenizer": {"from_repo": "Qwen/Qwen2.5-0.5B-Instruct",
                          "class": "PreTrainedTokenizerFast", "vocab_size": 151665},
            "device": {"kind": "cpu", "name": "x86_64", "count": 8},
            "context_keys": 4096, "new_tokens": 64},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
        ("right card, wrong token count", {**base, "new_tokens": 128}, 1, "single-stream",
         "NOT-LIKE-FOR-LIKE", False),
        ("right everything, sampled not greedy", {**base, "decoding": {"greedy": False}},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
        ("right everything, H100 not H200", {**base,
            "device": {"kind": "cuda", "name": "NVIDIA H100 80GB HBM3", "count": 1}},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
        ("right everything, but SYNTHETIC seconds", {**base, "synthetic": True},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
    ]
    bad = 0
    for label, run, batch, purpose, want_verdict, want_quotable in cases:
        r = lfl.check(run, context=run["context_keys"] if run["context_keys"] in lfl.CONTEXTS
                      else 4096, batch_size=batch, purpose=purpose)
        ok = r["verdict"] == want_verdict and r["quotable"] == want_quotable
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:46s} -> {r['verdict']} "
              f"(quotable={r['quotable']})")
        for b in r["blocking"]:
            print(f"          blocking: {b}")
    print()
    print("proved denominators the ladder will divide into:")
    for c in lfl.CONTEXTS:
        for row in lfl.proved_rows(c):
            v = "NULL -- " + row["blocked_by"] if row["tok_s"] is None else f"{row['tok_s']:.3f} tok/s"
            print(f"  C={c:<6d} {row['boundary']:<11s} {'headless' if row['headless'] else 'full  '}"
                  f"  {v}")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="hf", choices=sorted(B.BACKENDS))
    p.add_argument("--model", default=os.environ.get("B1_MODEL", "unsloth/Llama-3.1-8B-Instruct"))
    p.add_argument("--revision", default=os.environ.get("E2_REVISION", "main"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=os.environ.get("B1_DEVICE", "cuda"))
    p.add_argument("--attn", default=None, help="attn_implementation for the hf backend")
    p.add_argument("--context", type=int, action="append",
                   help="committed KV prefix C; repeatable. default 4096 and 16384")
    p.add_argument("--new-tokens", type=int, default=512)
    p.add_argument("--batch", type=int, action="append",
                   help="batch size; repeatable. default 1")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    a = p.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if a.probe:
        print(json.dumps({n: B.probe(n) for n in sorted(B.BACKENDS)}, indent=2))
        sys.exit(0)

    contexts = a.context or list(lfl.CONTEXTS)
    batches = a.batch or [1]
    OUT.mkdir(parents=True, exist_ok=True)

    kw = {}
    if a.backend == "vllm":
        kw = {"max_model_len": a.max_model_len or (max(contexts) + a.new_tokens),
              "gpu_memory_utilization": a.gpu_memory_utilization}
    try:
        be = B.BACKENDS[a.backend](a.model, a.revision, a.dtype, a.device, a.attn, **kw)
    except B.Unavailable as e:
        rec = {"schema": SCHEMA, "backend": {"name": a.backend, "available": False,
                                             "reason": str(e)},
               "measured": None,
               "note": "Backend unavailable. NO number is produced; do not substitute one."}
        path = OUT / f"{a.backend}_unavailable.json"
        path.write_text(json.dumps(rec, indent=2))
        print(f"backend {a.backend} unavailable: {e}", file=sys.stderr)
        print(f"wrote {path}")
        sys.exit(3)

    tag = a.tag or a.model.split("/")[-1].lower().replace(".", "")
    results = []
    for c in contexts:
        for b in batches:
            purpose = "single-stream" if b == 1 else "batched"
            try:
                rec = run_one(a, be, c, b, purpose)
            except Exception as e:  # OOM is a RESULT: record where the sweep stopped
                oom = "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__
                rec = {"schema": SCHEMA, "purpose": purpose,
                       "backend": {"name": a.backend, "available": True},
                       "run": {"context_keys": c, "batch_size": b,
                               "new_tokens": a.new_tokens, "dtype": a.dtype,
                               "synthetic": bool(getattr(be, "synthetic", False))},
                       "measured": None,
                       "failure": {"oom": oom, "type": type(e).__name__, "message": str(e)[:2000]},
                       "note": "This (context, batch) did not run. The sweep's last "
                               "successful batch is the one to quote."}
                print(f"C={c} batch={b}: FAILED ({type(e).__name__}) -- recorded, not skipped",
                      file=sys.stderr)
            path = OUT / f"{tag}_{a.backend}_c{c}_b{b}_n{a.new_tokens}.json"
            path.write_text(json.dumps(rec, indent=2))
            results.append((path, rec))
            m = rec.get("measured")
            if m:
                lf = rec["like_for_like"]
                print(f"C={c:<6d} batch={b:<4d} {a.backend:<5s} "
                      f"per-stream {m['tok_s_decode_per_stream']['median']:9.3f} tok/s   "
                      f"aggregate {m['tok_s_decode_aggregate']['median']:10.3f} tok/s   "
                      f"ttft {m['ttft_s']['median']:7.3f}s   [{lf['verdict']}]")
            print(f"  -> {path}")
            if rec.get("failure", {}).get("oom"):
                print("  batch sweep stopped: out of memory", file=sys.stderr)
                break
    return results


if __name__ == "__main__":
    main()
