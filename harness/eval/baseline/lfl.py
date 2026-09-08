"""The unproved baseline's comparison: cost per position, and the like-for-like check.

The paper quotes a cost per proved position and not one "times slower" multiple
(the decision of 2026-09-08 in zk-results, `docs/decisions/
2026-09-08-cost-per-proved-position-not-an-overhead-multiple.md`), because
overhead is a property of the workload and not of the prover: the prover costs
the same per position whether that position was prompted or generated, while
the unproved cost of the two differs by two orders of magnitude.  So `compare`
puts the proved seconds per position beside the unproved seconds per position
of each of the three measured rates, and divides nothing across the two sides.

`check` is the other half: the denominator of any reading has to match its
numerator, so every dimension of a measured run is checked against the proved
rows it will be read beside, and the verdict travels in the run's own JSON.  A
dimension that CANNOT be matched is not silently dropped -- it is recorded with
the direction it pushes the answer.

Proved numbers come from a zk-results `constants.json` (`--constants PATH`).
Nothing here re-derives one.
"""

from harness.constants import get

# The proved rows an unproved measurement is read beside, by name.  Every one is
# positions per second over the same positions the unproved end-to-end rate
# counts -- the prompt's prefill and the decoded tokens -- so the two sides
# divide the same positions, which is what the decision of 2026-09-08 on the
# unproved baseline requires.  The timing scope of each is in its own entry.
PROVED_ROWS = {
    4096: ("pos_s.headline.c4096.whole_wall", "pos_s.headline.c4096.loop"),
    16384: ("pos_s.headline.c16384.whole_wall", "pos_s.headline.c16384.loop"),
}

# How many positions those rows are over, per context.
POSITIONS_KEY = {c: f"positions.headline.c{c}" for c in PROVED_ROWS}

CONTEXTS = tuple(PROVED_ROWS)

# What the proved rows are rows OF: the fixed configuration every recorded
# sequence run is made under.  These are not measured numbers and they are not
# in the constants file -- they are the statement's own configuration, and a
# baseline that does not match them is not the same computation.
STATEMENT = {
    "model": "Llama-3.1-8B-Instruct",
    "decoded_tokens": 512,
    "decoding": "greedy, no sampling, EOS suppressed",
    "spec": "y-delay-vw12 (int8 dynamic linear, bf16 nonlinear)",
}

# The three measured rates, and the positions each is over.
PARTS = ("prefill", "decode", "end_to_end")


def proved_rows(context):
    """Every proved row at `context`, resolved.

    A key the constants file does not carry and a key it carries as null are
    both recorded as blocked, and both name the key -- never a substituted
    guess.
    """
    out = []
    for key in PROVED_ROWS.get(context, ()):
        try:
            e = get(key)
        except KeyError as exc:
            out.append({"key": key, "pos_s": None, "secs_per_position": None,
                        "blocked_by": f"{key} is not in the constants file ({exc})"})
            continue
        row = {"key": key, "unit": e["unit"], "card": e["card"],
               "context": e["context"], "timing_scope": e["timing_scope"],
               "grade": e["grade"], "note": e["note"],
               "pos_s": e["value"], "secs_per_position": None, "blocked_by": None}
        if e["value"] is None:
            row["blocked_by"] = f"{key} has no value: {e.get('note', '')}"
        else:
            row["secs_per_position"] = 1.0 / e["value"]
        out.append(row)
    return out


def proved_positions(context):
    """How many positions the proved rows at `context` are over, or None."""
    key = POSITIONS_KEY.get(context)
    if key is None:
        return {"key": None, "positions": None,
                "blocked_by": f"no proved row at C={context}; "
                              f"the constants file has {sorted(CONTEXTS)}"}
    try:
        e = get(key)
    except KeyError as exc:
        return {"key": key, "positions": None,
                "blocked_by": f"{key} is not in the constants file ({exc})"}
    return {"key": key, "positions": e["value"], "note": e["note"],
            "blocked_by": None if e["value"] is not None else f"{key} has no value"}


def compare(measured, *, context):
    """The proved cost per position, beside the unproved cost per position.

    Seconds per position on both sides, three unproved rows and one proved row
    per timing scope, and nothing divided across the two.  A single multiple
    would hold for one mix of prompt and generated positions only; a reader
    multiplies the proved cost per position by their own mix and reads the
    unproved cost of that same mix off the three rows here.
    """
    unproved = {}
    for part in PARTS:
        st = (measured or {}).get(f"pos_s_{part}_per_stream")
        if not st:
            continue
        r = st["median"]
        unproved[part] = {"positions": measured["positions"][part],
                          "positions_per_s": r,
                          "secs_per_position": 1.0 / r if r else None}
    return {
        "unit": "seconds per position, per stream",
        "unproved": unproved,
        "proved": proved_rows(context),
        "proved_positions": proved_positions(context),
        "no_multiple": (
            "No ratio is taken across the two sides. The prover costs the same "
            "per position whether the position was prompted or generated; the "
            "unproved cost of the two differs by two orders of magnitude. One "
            "multiple would therefore be true of one mix of prompt and generated "
            "positions and of no other. See the decision of 2026-09-08 in "
            "zk-results, docs/decisions/"
            "2026-09-08-cost-per-proved-position-not-an-overhead-multiple.md."),
    }


# --- the dimensions -------------------------------------------------------

# direction: what a mismatch on this dimension does to a reading of these
# numbers.  "flatters_us" = the true unproved cost is HIGHER than this run
# reports, so we look worse than the truth; "penalizes_us" = the reverse;
# "different-comparison" = the row is a valid number but answers another
# question; "invalid" = the row must not be quoted.
MATCH, STATED, MISMATCH, UNKNOWN = "match", "stated-mismatch", "mismatch", "unknown"


def _d(name, ours, proved, verdict, direction, note):
    return {"name": name, "ours": ours, "proved": proved, "verdict": verdict,
            "direction": direction, "note": note}


def check(run, *, context, batch_size, purpose):
    """`run` is the recorded run block of a bench JSON.  `purpose` is
    "single-stream" (the like-for-like latency row) or "batched" (the production
    row that feeds the dollar figures and is NOT like-for-like on batch, by
    design)."""
    # A context the constants file has no proved row at can still be measured --
    # it just has nothing to be read beside, and the check says so rather than
    # quietly borrowing another context's row.
    rows = proved_rows(context)
    resolved = [r for r in rows if r["blocked_by"] is None]
    no_reference = not resolved
    ref = resolved[0] if resolved else None
    positions = proved_positions(context)
    dims = []

    # 1. model
    want_model = STATEMENT["model"]
    ours_model = run["model"]["id"]
    ok = ours_model.split("/")[-1] == want_model
    dims.append(_d("model", f'{ours_model}@{run["model"]["revision"]}', want_model,
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "The model the proved rows are rows of. A rate measured on "
                   "another model is a number about another model."))

    # 2. tokenizer -- same repo as the weights, or we are not running the same
    #    token stream the proved run ran.
    tk = run["tokenizer"]
    ok = tk["from_repo"] == ours_model and tk["from_repo"].split("/")[-1] == want_model
    dims.append(_d("tokenizer", f'{tk["from_repo"]} ({tk["class"]}, vocab {tk["vocab_size"]})',
                   f"the tokenizer of {want_model}", MATCH if ok else MISMATCH,
                   "neutral" if ok else "invalid",
                   "Recorded so a reader can check the token stream is the same one; "
                   "positions per second is meaningless across two tokenizations."))

    # 3. card
    want_card = ref["card"] if ref else None
    dev = run["device"]
    ours_card = dev["name"]
    ok = bool(want_card) and dev["kind"] == "cuda" and want_card.lower() in ours_card.lower()
    dims.append(_d("card", f'{ours_card} x{dev["count"]}', want_card or "no proved row to name one",
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "The proved rows are one GPU. Two costs per position measured on "
                   "two different cards are not about the same machine."))

    # 4. context
    ok = (not no_reference) and run["context_keys"] == context == ref["context"]
    dims.append(_d("context_keys", run["context_keys"],
                   f"no proved row at C={context}; the constants file has "
                   f"{sorted(CONTEXTS)}" if no_reference else ref["context"],
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "C is the committed KV prefix (prover `--ctx`). The proved rows "
                   "cover the C prompt positions and the decoded ones, so the "
                   "unproved run prompts with exactly C tokens and decodes the same "
                   "number more."))

    # 5. generated tokens
    want_n = STATEMENT["decoded_tokens"]
    ok = run["new_tokens"] == want_n
    dims.append(_d("new_tokens", run["new_tokens"], want_n,
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "The decoded positions the statement covers. EOS is suppressed so "
                   "exactly this many are produced."))

    # 6. decoding rule
    ok = run["decoding"]["greedy"]
    dims.append(_d("decoding", "greedy (do_sample=False, num_beams=1)",
                   STATEMENT["decoding"], MATCH if ok else MISMATCH,
                   "neutral" if ok else "invalid",
                   "The proved sequence is the model's own greedy continuation; a "
                   "sampled baseline decodes a different token stream."))

    # 7. batch
    if purpose == "single-stream":
        ok = batch_size == 1
        dims.append(_d("batch_size", batch_size, 1, MATCH if ok else MISMATCH,
                       "neutral" if ok else "invalid",
                       "The proved rows are one sequence per proof, batch 1. This is "
                       "the like-for-like latency comparison."))
    else:
        dims.append(_d("batch_size", batch_size, 1, STATED, "different-comparison",
                       "DELIBERATE mismatch: the production-serving row. Batched "
                       "decode is not comparable to a batch-1 latency; it exists for "
                       "the cost-per-position comparison at serving concurrency. "
                       "Never read this row as a latency figure."))

    # 8. arithmetic
    prec = run["dtype"]
    dims.append(_d("arithmetic", prec, STATEMENT["spec"], STATED,
                   "flatters_us" if prec in ("bfloat16", "float16", "float32") else "unknown",
                   "We prove a quantized arithmetic; the baseline runs the deployed "
                   "one. A bf16/fp16 baseline is SLOWER than an int8/fp8 serving "
                   "stack, so this mismatch makes the unproved cost per position we "
                   "report LARGER than the truth. Pass --dtype to run a faster "
                   "baseline and the unproved cost falls."))

    # 9. what the positions cover
    ours_positions = run["context_keys"] + run["new_tokens"]
    want_positions = positions["positions"]
    ok = want_positions is not None and ours_positions == want_positions
    dims.append(_d("positions", f"{run['context_keys']} prompt + {run['new_tokens']} "
                                f"generated = {ours_positions}, end to end",
                   want_positions if want_positions is not None else positions["blocked_by"],
                   MATCH if ok else MISMATCH, "neutral" if ok else "invalid",
                   "Both sides count the prompt. The end-to-end rate is over these "
                   "positions and the whole pass's wall-clock time; the prefill and "
                   "decode rates are over their own parts, and no headline rate "
                   "subtracts the prompt out. See the decisions of 2026-09-08 in "
                   "zk-results on the unproved baseline and on cost per proved "
                   "position."))

    # 10. synthetic -- the mock backend must never reach a table.
    if run.get("synthetic"):
        dims.append(_d("synthetic", "SYNTHETIC (mock backend)", "a real measurement",
                       MISMATCH, "invalid",
                       "These seconds came from an arithmetic model, not from running "
                       "a model on a card. The run exists to exercise the harness."))

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
        "reference": {"rows": rows, "positions": positions},
        "dimensions": dims,
        "verdict": verdict,
        "quotable": quotable,
        "blocking": [f'{d["name"]}: {d["ours"]!r} vs proved {d["proved"]!r}' for d in invalid],
        "stated_mismatches": [f'{d["name"]}: {d["note"]}' for d in stated],
    }
