"""E2h: the like-for-like checker, and the proved reference rows it checks against.

The denominator of an overhead multiple has to match its numerator.  This module
holds that discipline in code rather than in prose: every run config is checked,
dimension by dimension, against the proved row it will be divided into, and the
verdict travels in the run's own JSON.  A dimension that CANNOT be matched is
not silently dropped -- it is recorded with the direction it pushes the answer,
so a reader can see whether the mismatch flatters us or costs us.

Proved numbers come from a zk-results `constants.json` (`--constants PATH`).
Nothing here re-derives one.
"""

from harness.constants import get, value

# The four boundaries the freeze names, in order of how much of the prover they
# contain.  `loop_secs` is NOT `whole wall`: whole wall has never counted the
# witness read or the layer build, and loop_secs does.  Any ratio table must say
# which of these its denominator is.
BOUNDARIES = ("commit", "stwo total", "whole wall", "loop_secs")

# The proved rows an unproved measurement can be divided into.  Each names the
# freeze key, the boundary, and whether it covers all 33 chunks or only the 32
# layer chunks (`headless`) -- a headless denominator against a whole-model
# numerator would overstate our throughput, so it is flagged, not hidden.
PROVED_ROWS = {
    4096: [
        {"key": "tok_s.seq5.c4096.commit", "boundary": "commit", "headless": False},
        {"key": "tok_s.seq5.c4096.stwo", "boundary": "stwo total", "headless": False},
        {"key": "tok_s.seq5.c4096.whole", "boundary": "whole wall", "headless": False},
        {"key": "loop.h200.c4096.pipelined_defaults", "boundary": "loop_secs",
         "headless": True, "secs_for_512": True},
    ],
    16384: [
        {"key": "tok_s.seq5.c16384.commit", "boundary": "commit", "headless": False},
        {"key": "tok_s.seq5.c16384.stwo", "boundary": "stwo total", "headless": False},
        {"key": "tok_s.seq5.c16384.whole", "boundary": "whole wall", "headless": False},
        {"key": "loop.h200.c16384", "boundary": "loop_secs",
         "headless": True, "secs_for_512": True},
    ],
}

# The like-for-like reference: the row the paper's single-stream sentence divides
# into (final-framing section 4 quotes the whole wall).  Its card and context are
# what every dimension below is checked against.
REFERENCE_KEY = {4096: "tok_s.seq5.c4096.whole", 16384: "tok_s.seq5.c16384.whole"}

CONTEXTS = tuple(PROVED_ROWS)


def proved_tok_s(row):
    """tok/s for one proved row, from the freeze.  Raises on a deliberate null."""
    if row.get("secs_for_512"):
        return value("spec.decoded_tokens") / value(row["key"])
    return value(row["key"])


def proved_rows(context, *, allow_null=True):
    """Every proved denominator at `context`, resolved.  A null stays a null and
    names the key that has to be filled -- never a substituted guess."""
    out = []
    for row in PROVED_ROWS[context]:
        e = get(row["key"])
        r = dict(row, card=e["card"], note=e["note"], grade=e["grade"], source=e["source"])
        if e["value"] is None:
            if not allow_null:
                raise ValueError(f"{row['key']} is null: {e['note']}")
            r["tok_s"] = None
            r["blocked_by"] = row["key"]
        else:
            r["tok_s"] = proved_tok_s(row)
            r["blocked_by"] = None
        out.append(r)
    return out


# --- the dimensions -------------------------------------------------------

# direction: what a mismatch on this dimension does to the overhead multiple we
# report.  "flatters_us" = the true multiple is LARGER than the one this run
# produces; "penalizes_us" = smaller; "different-comparison" = the row is a valid
# number but answers another question; "invalid" = the row must not be quoted.
MATCH, STATED, MISMATCH, UNKNOWN = "match", "stated-mismatch", "mismatch", "unknown"


def _d(name, ours, proved, verdict, direction, note):
    return {"name": name, "ours": ours, "proved": proved, "verdict": verdict,
            "direction": direction, "note": note}


def check(run, *, context, batch_size, purpose):
    """`run` is the recorded run block of a bench JSON.  `purpose` is
    "single-stream" (the like-for-like latency row) or "batched" (the production
    row that feeds $/token and is NOT like-for-like on batch, by design)."""
    # A context the freeze has no proved row at (anything but 4096 / 16384) can
    # still be measured -- it just has no denominator, and the check says so
    # rather than quietly borrowing one.
    no_reference = context not in REFERENCE_KEY
    ref_key = REFERENCE_KEY.get(context, REFERENCE_KEY[4096])
    ref = get(ref_key)
    dims = []

    # 1. model
    want_model = value("spec.model")
    ours_model = run["model"]["id"]
    ok = ours_model.split("/")[-1] == want_model
    dims.append(_d("model", f'{ours_model}@{run["model"]["revision"]}', want_model,
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   get("spec.model")["note"]))

    # 2. tokenizer -- same repo as the weights, or we are not decoding the same
    #    token stream the proved run decoded.
    tk = run["tokenizer"]
    ok = tk["from_repo"] == ours_model and tk["from_repo"].split("/")[-1] == want_model
    dims.append(_d("tokenizer", f'{tk["from_repo"]} ({tk["class"]}, vocab {tk["vocab_size"]})',
                   f"the tokenizer of {want_model}", MATCH if ok else MISMATCH,
                   "neutral" if ok else "invalid",
                   "Recorded so a reader can check the token stream is the same one; "
                   "tok/s is meaningless across two tokenizations."))

    # 3. card
    want_card = ref["card"]
    dev = run["device"]
    ours_card = dev["name"]
    ok = dev["kind"] == "cuda" and want_card.lower() in ours_card.lower()
    dims.append(_d("card", f'{ours_card} x{dev["count"]}', want_card,
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "The proved row is one GPU. A ratio across two different cards "
                   "is not an overhead multiple."))

    # 4. context
    ok = (not no_reference) and run["context_keys"] == context == ref["context"]
    dims.append(_d("context_keys", run["context_keys"],
                   f"no proved row at C={context}; the freeze has 4096 and 16384 only"
                   if no_reference else ref["context"],
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "C is the committed KV prefix (prover `--ctx`); the S=512 decoded "
                   "positions attend over it plus the diagonal block, so the unproved "
                   "run prompts with exactly C tokens and decodes 512 more."))

    # 5. generated tokens
    want_n = value("spec.decoded_tokens")
    ok = run["new_tokens"] == want_n
    dims.append(_d("new_tokens", run["new_tokens"], want_n,
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "Both sides divide by the same token count. EOS is suppressed so "
                   "exactly this many are produced."))

    # 6. decoding rule
    ok = run["decoding"]["greedy"]
    dims.append(_d("decoding", "greedy (do_sample=False, num_beams=1)",
                   value("scope.sampling_status"), MATCH if ok else MISMATCH,
                   "neutral" if ok else "invalid",
                   get("scope.sampling_status")["note"]))

    # 7. batch
    if purpose == "single-stream":
        ok = batch_size == 1
        dims.append(_d("batch_size", batch_size, 1, MATCH if ok else MISMATCH,
                       "neutral" if ok else "invalid",
                       "The proved row is one sequence per proof, batch 1. This is the "
                       "like-for-like latency comparison."))
    else:
        dims.append(_d("batch_size", batch_size, 1, STATED, "different-comparison",
                       "DELIBERATE mismatch: the production-serving row. Batched decode "
                       "is not comparable to our batch-1 latency; it exists for the "
                       "$/token comparison, which is the currency the 500,000x claim "
                       "is really about. Never quote this row as a latency overhead."))

    # 8. arithmetic
    prec = run["dtype"]
    dims.append(_d("arithmetic", prec, value("spec.name") + " (int8 dynamic linear, bf16 nonlinear)",
                   STATED, "flatters_us" if prec in ("bfloat16", "float16", "float32") else "unknown",
                   "We prove a quantized arithmetic; the baseline runs the deployed one. "
                   "A bf16/fp16 baseline is SLOWER than an int8/fp8 serving stack, so this "
                   "mismatch makes the overhead multiple we report SMALLER than the truth. "
                   "Pass --dtype to run a faster baseline arm and the multiple grows."))

    # 9. what the token count covers
    dims.append(_d("positions_counted", "512 decoded positions, prefill excluded",
                   f'512 decoded positions; prompt {value("scope.prefill_status")}',
                   MATCH, "neutral",
                   "The v1 comparison. The proved seconds never include the prompt, so "
                   "the unproved rate excludes prefill too (a separate max_new_tokens=1 "
                   "call, subtracted). The SAME run also records an `e11` rate over all "
                   "prompt+output positions -- the convention DeepProve's TPM uses -- and "
                   "the ladder reports it as a separate arm under a named assumption. "
                   "tok_s_wall (512 positions but prefill included) is recorded and used "
                   "by neither."))

    # 10. synthetic -- the mock backend must never reach a table.
    if run.get("synthetic"):
        dims.append(_d("synthetic", "SYNTHETIC (mock backend)", "a real measurement",
                       MISMATCH, "invalid",
                       "These seconds came from an arithmetic model, not from running a "
                       "model on a card. The run exists to exercise the harness. It is "
                       "blocking here and `ladder.py` drops it before it picks."))

    invalid = [d for d in dims if d["direction"] == "invalid"]
    stated = [d for d in dims if d["verdict"] == STATED]
    if invalid:
        verdict, quotable = "NOT-LIKE-FOR-LIKE", False
    elif stated:
        verdict, quotable = "like-for-like with stated mismatches", True
    else:
        verdict, quotable = "like-for-like", True

    return {
        "purpose": purpose,
        "reference": {
            "key": None if no_reference else ref_key,
            "value": None if no_reference else ref["value"], "unit": ref["unit"],
            "card": ref["card"], "context": ref["context"], "boundary": ref["boundary"],
            "grade": ref["grade"], "source": ref["source"],
        },
        "dimensions": dims,
        "verdict": verdict,
        "quotable": quotable,
        "blocking": [f'{d["name"]}: {d["ours"]!r} vs proved {d["proved"]!r}' for d in invalid],
        "stated_mismatches": [f'{d["name"]}: {d["note"]}' for d in stated],
    }
