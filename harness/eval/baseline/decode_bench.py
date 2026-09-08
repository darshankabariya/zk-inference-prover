"""The unproved baseline: how fast the same model runs when nothing is proved.

Three numbers per point, each positions per second over the positions IT covers
(the decision of 2026-09-08 in zk-results, `docs/decisions/
2026-09-08-cost-per-proved-position-not-an-overhead-multiple.md`):

  prefill      the prompt's positions over the prefill's wall-clock time
  decode       the generated positions over the decode's wall-clock time
  end to end   both, over the whole pass's wall-clock time

None of the three is an "overhead" denominator, and no headline rate subtracts
the prompt out. The paper quotes a cost per proved position and the reader
applies their own mix of prompt and generated positions, so the comparison --
when a constants file is given -- puts the proved cost per position beside the
unproved cost per position of each of the three, and never one multiple.

Two rows, because the belief being corrected mixes them:
  single-stream  -- batch 1, greedy: the like-for-like latency row.
  batched        -- a batch sweep: the production-serving row.
Two backends, for the same reason: `hf` (what a non-specialist gets) and `vllm`
(what a production serving figure actually means).

Every like-for-like dimension is checked in `lfl.py` and the verdict is written
into the run's own JSON. A dimension that cannot be matched is recorded with the
direction it pushes the answer. The record is written whether or not the
comparison could be made: the measurement is this run's product, and the
comparison is a later reading of it against numbers this repository does not
hold.

    uv run python -m harness.eval.baseline.decode_bench --backend hf --context 4096 --batch 1
    uv run python -m harness.eval.baseline.decode_bench --selftest   # no model, no GPU
    uv run python -m harness.eval.baseline.decode_bench --probe      # backend availability
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

from harness import constants

from . import backends as B
from . import lfl

from harness import provenance

HERE = Path(__file__).resolve().parent

SCHEMA = "e2h/decode-throughput/3"
OUT = Path(os.environ.get("ZK_BASELINE_OUT", "out"))

# What the unproved seconds contain, written into every run so a reading can be
# audited without re-running anything.  The proved side's timing scopes are in
# `constants.json`'s `_timing_scopes`; this is the unproved side's own.
TIMING_SCOPE = {
    "name": "unproved inference, weights resident",
    "parts": {
        "prefill": "one `max_new_tokens=1` call: the whole prompt read, and the "
                   "first generated position emitted",
        "decode": "the rest of the full call, by difference: the remaining "
                  "generated positions, one step each",
        "end_to_end": "the full call on its own: the prompt prefilled and every "
                      "generated position decoded, as one wall-clock time",
    },
    "includes": [
        "the forward passes that produce the measured positions",
        "sampling/argmax, KV-cache growth, the engine's own scheduling",
    ],
    "excludes": [
        "process start, weight load and any compile/CUDA-graph capture "
        "(the warmup passes run before the timed repetitions and make the same "
        "two calls, generating the same token count, so no timed repetition is "
        "the first full-length decode of the process)",
        "tokenization of the prompt (built once, outside the timed region)",
    ],
    "comparable_to": (
        "the proved side's timing scopes, with one thing said out loud: none of "
        "them contains an inference pass. `whole wall` is trace generation, "
        "interaction generation, stwo and serialization; `loop`, which adds the "
        "witness read and the layer build, still reads a witness a prior "
        "extraction produced. So a proved cost per position is prover-seconds "
        "and an unproved one is inference-seconds, and a deployment that must "
        "both infer and prove pays the sum of the two."),
}

# What the three rates count, written into every record.
COUNTING = (
    "Each rate is positions per second over the positions it covers: the prompt "
    "over the prefill's wall-clock time, the generated positions over the "
    "decode's, and both over the whole pass's. The parts fall one position short "
    "of the end-to-end count, because the first generated position is produced "
    "inside the prefill call and is charged to neither part. None of the three "
    "is an overhead denominator: the paper quotes a cost per proved position and "
    "the reader applies their own mix of prompt and generated positions. See the "
    "decisions of 2026-09-08 in zk-results on the unproved baseline and on cost "
    "per proved position.")


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

    measured = {
        "per_rep": per_rep,
        # The positions each rate is over, per stream.
        "positions": {"prefill": context, "decode": a.new_tokens - 1,
                      "end_to_end": context + a.new_tokens},
        "prefill_s": _stats(per_rep, "prefill_s"),
        "decode_s": _stats(per_rep, "decode_s"),
        "wall_s": _stats(per_rep, "wall_s"),
        "counting": COUNTING,
    }
    for part in lfl.PARTS:
        st = _stats(per_rep, f"pos_s_{part}")
        measured[f"pos_s_{part}_per_stream"] = st
        measured[f"pos_s_{part}_aggregate"] = _agg(st)
    return {
        "schema": SCHEMA,
        "recorded_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "backend": {"name": be.name, "label": be.label, "version": be.version,
                    "available": True, "reason": None,
                    "synthetic": bool(getattr(be, "synthetic", False))},
        "purpose": purpose,
        "timing_scope": TIMING_SCOPE,
        "run": run,
        "measured": measured,
    }


def comparison(rec, *, context, batch, purpose, have_constants):
    """The like-for-like check and the cost per position, when constants exist.

    The record is written either way.  A measurement is this run's product; a
    comparison is a reading of it against numbers this repository does not hold,
    so a missing constants file, a missing key and a null value are all recorded
    as the reason no comparison was made, and none of them costs the run.
    """
    if not have_constants:
        return {"made": False,
                "why": "no --constants PATH (a zk-results constants.json) and no "
                       "ZK_CONSTANTS; the measurement stands on its own"}
    if not rec.get("measured"):
        return {"made": False, "why": "this point produced no number"}
    try:
        return {"made": True,
                "like_for_like": lfl.check(rec["run"], context=context,
                                           batch_size=batch, purpose=purpose),
                "cost_per_position": lfl.compare(rec["measured"], context=context)}
    except (KeyError, ValueError, OSError, SystemExit) as e:
        return {"made": False, "why": f"{type(e).__name__}: {e}"}


def selftest_rates():
    """The three rates, on the mock backend: no model, no card, no constants."""
    be = B.MockBackend("MOCK/none", "main", "bfloat16", "cpu", None)
    n_prompt, n_new = 64, 8
    per_rep, _ = B.measure(be, [[1] * n_prompt], n_new, reps=1, warmup=0)
    r = per_rep[0]
    cases = [
        ("the decode's time is the full call's minus the prefill call's",
         abs(r["wall_s"] - (r["prefill_s"] + r["decode_s"])) < 1e-9),
        ("prefill: the prompt's positions over the prefill's wall-clock time",
         abs(r["pos_s_prefill"] - n_prompt / r["prefill_s"]) < 1e-6),
        ("decode: the generated positions over the decode's wall-clock time",
         abs(r["pos_s_decode"] - (n_new - 1) / r["decode_s"]) < 1e-6),
        ("end to end: prompt plus generated over the whole wall-clock time",
         abs(r["pos_s_end_to_end"] - (n_prompt + n_new) / r["wall_s"]) < 1e-6),
        ("no rate subtracts the prompt out of the positions it counts",
         r["pos_s_end_to_end"] > r["pos_s_decode"]),
    ]
    bad = 0
    for label, ok in cases:
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label}")
    return bad


def selftest_warmup():
    """The warmup pass makes the same calls a timed repetition makes.

    A warmup that generated fewer tokens than the timed repetitions left the
    first timed repetition the first full-length decode of the process, and
    every point then reported that cold pass as its spread (zk-prover #106).
    So the shape of the calls is checked, not asserted in prose.
    """
    class Recorder(B.MockBackend):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = []

        def generate(self, prompts, n_new):
            self.calls.append(n_new)
            return super().generate(prompts, n_new)

    n_prompt, n_new, reps, warmup = 32, 16, 2, 1
    be = Recorder("MOCK/none", "main", "bfloat16", "cpu", None)
    B.measure(be, [[1] * n_prompt], n_new, reps=reps, warmup=warmup)
    rep = [1, n_new]
    cases = [
        ("the warmup generates the token count the timed repetitions generate",
         be.calls[:len(rep)] == rep),
        ("no timed repetition is the first full-length decode of the process",
         be.calls.count(n_new) == reps + warmup),
        ("the warmup is one untimed repetition and nothing else",
         be.calls == rep * (warmup + reps)),
    ]
    bad = 0
    for label, ok in cases:
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label}")
    return bad


def selftest_check(card):
    """The like-for-like checker has teeth, on synthetic run blocks."""
    base = {
        "model": {"id": "unsloth/Llama-3.1-8B-Instruct",
                  "revision": "4699cc75b550f9c6f3173fb80f4703b62d946aa5"},
        "tokenizer": {"from_repo": "unsloth/Llama-3.1-8B-Instruct",
                      "class": "PreTrainedTokenizerFast", "vocab_size": 128256},
        "dtype": "bfloat16",
        "device": {"kind": "cuda", "name": f"NVIDIA {card}", "count": 1},
        "context_keys": 4096, "new_tokens": 512,
        "decoding": {"greedy": True},
    }
    other = "H100" if "h100" not in card.lower() else "H200"
    cases = [
        ("the 8B single-stream row the host must produce", base, 1, "single-stream",
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
        (f"right everything, {other} not the proved {card}", {**base,
            "device": {"kind": "cuda", "name": f"NVIDIA {other} 80GB HBM3", "count": 1}},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
        ("right everything, but SYNTHETIC seconds", {**base, "synthetic": True},
         1, "single-stream", "NOT-LIKE-FOR-LIKE", False),
    ]
    bad = 0
    for label, run, batch, purpose, want_verdict, want_quotable in cases:
        r = lfl.check(run, context=4096, batch_size=batch, purpose=purpose)
        ok = r["verdict"] == want_verdict and r["quotable"] == want_quotable
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'}  {label:46s} -> {r['verdict']} "
              f"(quotable={r['quotable']})")
        for b in r["blocking"]:
            print(f"          blocking: {b}")
    return bad


def selftest(have_constants):
    bad = selftest_rates()
    print()
    bad += selftest_warmup()
    print()
    if not have_constants:
        print("no --constants PATH (a zk-results constants.json) and no ZK_CONSTANTS: "
              "the like-for-like cases and the proved rows were NOT checked.")
        return 1 if bad else 0
    rows = {c: lfl.proved_rows(c) for c in lfl.CONTEXTS}
    resolved = [r for c in lfl.CONTEXTS for r in rows[c] if r["blocked_by"] is None]
    if not resolved:
        print("this constants file carries no proved row at any measured context; "
              "the checker's cases cannot be stated against one.")
        return 1
    bad += selftest_check(resolved[0]["card"])
    print()
    print("the proved rows a cost per position is read beside:")
    for c in lfl.CONTEXTS:
        for row in rows[c]:
            v = ("blocked -- " + row["blocked_by"] if row["blocked_by"]
                 else f"{row['pos_s']:.5f} positions/s = "
                      f"{row['secs_per_position']:.4f} s/position")
            print(f"  C={c:<6d} {row.get('timing_scope', '?'):<11s} {v}")
    return 1 if bad else 0


def _provenance(a, extra=None):
    """The block every record here carries.

    A decode benchmark makes no proof: it times the unproved forward pass the
    proved one is compared against.  So the block says `kind` `harness`, carries
    no binding table and no proof digest, and names what it did run -- the
    model, its revision, the backend, the arithmetic and the device.
    """
    return provenance.block(
        model=a.model, device=a.device,
        threads=int(os.environ.get("ZK_THREADS", 0) or 0),
        module="harness.eval.baseline.decode_bench",
        decode_backend=a.backend, dtype=a.dtype, attention=a.attn,
        model_revision_requested=a.revision, **(extra or {}))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="hf", choices=sorted(B.BACKENDS))
    p.add_argument("--model", default=os.environ.get("ZK_MODEL", "unsloth/Llama-3.1-8B-Instruct"))
    p.add_argument("--revision", default=os.environ.get("ZK_MODEL_REVISION", "main"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=os.environ.get("ZK_DEVICE", "cuda"))
    p.add_argument("--attn", default=None, help="attn_implementation for the hf backend")
    p.add_argument("--context", type=int, action="append",
                   help="committed KV prefix C; repeatable. default 4096 and 16384")
    p.add_argument("--new-tokens", type=int, default=512)
    p.add_argument("--batch", type=int, action="append",
                   help="batch size; repeatable. default 1")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed repetitions before the timed ones; each makes the "
                        "same two calls a timed repetition makes")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true",
                   help="the three rates and the like-for-like checker, with no model "
                        "and no card; the checker's half needs --constants")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    constants.add_argument(p)
    a = p.parse_args()
    # The constants file is read once, here, and only if one was named.  A
    # measurement is this run's product and does not need it; the comparison
    # does, and says so in the record when there is none.
    have_constants = bool(a.constants)
    if have_constants:
        constants.load(a.constants)

    if a.selftest:
        sys.exit(selftest(have_constants))
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
        rec["comparison"] = {"made": False, "why": "the backend did not run"}
        rec["provenance"] = _provenance(a)
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
            rec["comparison"] = comparison(rec, context=c, batch=b, purpose=purpose,
                                           have_constants=have_constants)
            rec["provenance"] = _provenance(a)
            path = OUT / f"{tag}_{a.backend}_c{c}_b{b}_n{a.new_tokens}.json"
            path.write_text(json.dumps(rec, indent=2))
            results.append((path, rec))
            m = rec.get("measured")
            if m:
                cmp_ = rec["comparison"]
                verdict = (cmp_["like_for_like"]["verdict"] if cmp_["made"]
                           else "no comparison: " + cmp_["why"])
                print(f"C={c:<6d} batch={b:<4d} {a.backend:<5s} positions/s per stream: "
                      f"prefill {m['pos_s_prefill_per_stream']['median']:11.2f}  "
                      f"decode {m['pos_s_decode_per_stream']['median']:9.3f}  "
                      f"end to end {m['pos_s_end_to_end_per_stream']['median']:11.3f}   "
                      f"[{verdict}]")
            print(f"  -> {path}")
            if rec.get("failure", {}).get("oom"):
                print("  batch sweep stopped: out of memory", file=sys.stderr)
                break
    return results


if __name__ == "__main__":
    main()
