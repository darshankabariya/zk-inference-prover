"""E2h: the overhead-multiple ladder.

Turns measured unproved tok/s (from `decode_bench.py`) plus the frozen proved
tok/s (from a zk-results `constants.json`, `--constants PATH`) into the table the paper's
section 4 needs: one row per BOUNDARY, because "the overhead" is not one number
and the four boundaries the freeze names differ by 6x among themselves.

    commit      the commitment path only
    stwo total  commit + prove_ex
    whole wall  trace gen + interaction gen + stwo + serialization
                (has NEVER counted the witness read or the layer build)
    loop_secs   the one-process chunk loop's own wall, which DOES count them.
                A FOURTH boundary, not comparable to whole wall, and the only
                H200 loop rows on record are HEADLESS (32 layer chunks).

THREE arms, from the same runs, because the denominator forks twice:

  latency   single-stream decode-only unproved / proved.  Like-for-like on every
            dimension the freeze names.  This is the cell section 4 wants.
  e11       all positions of the concatenated prompt+output pass on both sides.
            The convention DeepProve's TPM uses and the one our own statement
            needs once E11 proves the prompt.  Carries a NAMED assumption (see
            `E11_ASSUMPTION`) because no E11 proved row exists yet.
  $/token   batched-serving unproved against our batch-1 proof.  The currency
            the 500,000x claim is really in; deliberately NOT like-for-like on
            batch, and the row says so.

Both backends are reported side by side -- a hf-only baseline would flatter us
in exactly the way we accuse fn. 170 of.  The headline cell takes the FASTER
backend, which is the conservative direction: a faster denominator makes the
overhead we report LARGER, not smaller.

Dollars come only from a `*_reference_*` cost key of the freeze (its own rule);
`--usd-per-gpu-hour` overrides it and stamps the whole dollar block
`from_freeze: false`.

    uv run python e2/ladder.py --out e2/out/ladder.json
    uv run python e2/ladder.py --markdown
    uv run python e2/ladder.py --emit-constants   # rows ready for the freeze
"""

import argparse
import json
import sys
from pathlib import Path

from harness import constants
from harness.constants import get, value

from . import lfl

HERE = Path(__file__).resolve().parent

SCHEMA = "e2h/overhead-ladder/2"
RUN_SCHEMAS = ("e2h/decode-throughput/1", "e2h/decode-throughput/2")
# The two a publishable table must have. `mock` is neither, and never counts.
BACKENDS_WANTED = ("hf", "vllm")

# Both sides of every ratio in this file are prover-seconds or inference-seconds
# on ONE card.  What follows is in neither, and a reader is owed the list.
WHAT_NEITHER_SIDE_COUNTS = [
    "the witness extraction. The proved side's `loop_secs` reads a witness that "
    "a prior extraction step produced; that step is in no boundary and has no "
    "unproved counterpart.",
    "model / weight load, and on the proved side the kernel-store emit and the "
    "prover binary's own start-up.",
    "verification. Every number here is prover-side.",
    "the inference pass itself, on the PROVED side: none of the four boundaries "
    "contains one. So an overhead multiple U/P is prover-seconds over "
    "inference-seconds -- precisely what fn. 170 computed (zkLLM's 803 s over "
    "668 tok/s). A deployment that must infer AND prove costs (1 + U/P) x the "
    "unproved time; at x25-40 that unit is immaterial, and it is stated anyway.",
]

E11_ASSUMPTION = (
    "REGIME 1, the same convention stream E7 used in notes/sufficiency.md: every "
    "proved position is charged one measured C=4096 decoded token, so the proved "
    "rate in positions/second equals the frozen tok/s row unchanged. This is "
    "CONSERVATIVE for E11 -- a prompt position inside a causal rectangle attends "
    "to ~N/2 keys, not a full C band -- and it is an assumption, not a "
    "measurement: no E11 proved row exists. The unproved side needs no assumption; "
    "it is measured directly as (prompt + output) / wall of the concatenated pass."
)


def usd_per_mtok(tok_s, usd_per_h):
    if tok_s is None or usd_per_h is None:
        return None
    return usd_per_h / (tok_s * 3600.0) * 1e6


def load_runs(outdir, allow_synthetic=False):
    """Every bench artifact in `outdir`.  Synthetic ones are DROPPED here, before
    anything can pick them -- `lfl` also marks them blocking, but a table must
    not depend on one guard."""
    runs, dropped = [], []
    for p in sorted(Path(outdir).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("schema") not in RUN_SCHEMAS:
            continue
        if d.get("run", {}).get("synthetic") and not allow_synthetic:
            dropped.append(str(p))
            continue
        runs.append((p, d))
    return runs, dropped


METRICS = {
    # arm            per-stream key                            aggregate key
    "decode_only": ("tok_s_decode_per_stream", "tok_s_decode_aggregate"),
    "e11": ("tok_s_e11_all_positions_per_stream", "tok_s_e11_all_positions_aggregate"),
}


def _one(runs, context, purpose, metric, backend=None):
    """The unproved number for one (context, purpose, metric[, backend]).

    Single-stream: batch 1.  Batched: the best AGGREGATE rate over the sweep and
    the batch it came from.  Preference, strongest first: a quotable
    (like-for-like) run; failing that a run with the canonical 512-token count;
    failing that whatever exists, with the caveat block saying so.  A 16-token
    probe must never outrank a real run just because short decodes are faster.
    """
    per_stream_key, agg_key = METRICS[metric]
    cands = [(p, d) for p, d in runs
             if d.get("measured") and d["run"]["context_keys"] == context
             and d["purpose"] == purpose
             and (backend is None or d["backend"]["name"] == backend)]
    if purpose == "single-stream":
        cands = [(p, d) for p, d in cands if d["run"]["batch_size"] == 1]
        key = per_stream_key
    else:
        key = agg_key
    cands = [(p, d) for p, d in cands if key in d["measured"]]
    if not cands:
        return None
    want_n = value("spec.decoded_tokens")
    quotable = [(p, d) for p, d in cands if d["like_for_like"]["quotable"]]
    right_n = [(p, d) for p, d in cands if d["run"]["new_tokens"] == want_n]
    pool = quotable or right_n or cands
    p, d = max(pool, key=lambda pd: pd[1]["measured"][key]["median"])
    return {
        "artifact": str(p),
        "backend": d["backend"]["name"],
        "backend_label": d["backend"].get("label"),
        "backend_version": d["backend"].get("version"),
        "metric": metric,
        "which": key,
        "tok_s": d["measured"][key]["median"],
        "spread_pct": d["measured"][key].get("spread_pct"),
        "reps": d["run"].get("reps"),
        "batch_size": d["run"]["batch_size"],
        "card": d["run"]["device"]["name"],
        "dtype": d["run"]["dtype"],
        "model": d["run"]["model"]["id"],
        "revision": d["run"]["model"].get("revision"),
        "new_tokens": d["run"]["new_tokens"],
        "prompt_tokens": d["run"].get("prompt_tokens"),
        "quotable": d["like_for_like"]["quotable"],
        "lfl_verdict": d["like_for_like"]["verdict"],
        "blocking": d["like_for_like"]["blocking"],
        "stated_mismatches": d["like_for_like"]["stated_mismatches"],
    }


def select(runs, context, purpose, metric):
    """Both backends side by side, plus the one the headline cell uses.

    The chosen cell is the FASTER backend.  That is deliberate and it is the
    conservative direction: overhead = unproved / proved, so a faster unproved
    denominator makes the multiple we report LARGER.  Quoting only `hf` would
    flatter us in exactly the way this paper accuses fn. 170 of doing.
    """
    present = sorted({d["backend"]["name"] for _, d in runs})
    by_backend = {}
    for name in present:
        r = _one(runs, context, purpose, metric, backend=name)
        if r:
            by_backend[name] = r
    if not by_backend:
        return None
    chosen = max(by_backend.values(), key=lambda r: r["tok_s"])
    return {
        "by_backend": by_backend,
        "chosen": chosen,
        "choice_rule": "the faster of the backends present; a faster unproved "
                       "denominator makes the reported overhead LARGER, so this is "
                       "the conservative choice",
        "backends_seen": sorted(by_backend),
        "backends_missing": [b for b in BACKENDS_WANTED if b not in by_backend],
    }


def build(outdir, usd_override=None, allow_synthetic=False):
    runs, dropped = load_runs(outdir, allow_synthetic)
    rate_e = get("cost.h200_reference_usd_per_hour")
    if usd_override is not None:
        usd_per_h, from_freeze = usd_override, False
        rate_note = ("CLI override, NOT from the freeze. The freeze's own rule is that "
                     "dollar figures come from a *_reference_* cost key. Any figure in "
                     "this block is provisional.")
    else:
        usd_per_h, from_freeze = rate_e["value"], True
        rate_note = rate_e["note"]

    doc = {
        "schema": SCHEMA,
        "proved_source": constants.path(),
        "unproved_source": str(outdir),
        "what_neither_side_counts": WHAT_NEITHER_SIDE_COUNTS,
        "e11_assumption": E11_ASSUMPTION,
        "cost": {"h200_usd_per_gpu_hour": usd_per_h, "from_freeze": from_freeze,
                 "key": "cost.h200_reference_usd_per_hour",
                 "grade": rate_e["grade"], "source": rate_e["source"],
                 "blocking_key": None if usd_per_h is not None
                                 else "cost.h200_reference_usd_per_hour",
                 "note": rate_note},
        "contexts": {},
        "caveats": [],
    }
    if dropped:
        doc["caveats"].append(
            f"{len(dropped)} SYNTHETIC run(s) in {outdir} were dropped before "
            f"selection and appear in no cell: {dropped}")
    doc["boundary_definitions"] = dict(constants.doc().get("_timing_scopes", {}))
    doc["boundary_definitions"]["unproved decode wall"] = (
        "the unproved side's own boundary: forward passes with weights resident, "
        "warmup and prompt tokenization outside the timed region. See a run "
        "artifact's `boundary` block.")

    for c in lfl.CONTEXTS:
        single = select(runs, c, "single-stream", "decode_only")
        e11 = select(runs, c, "single-stream", "e11")
        batched = select(runs, c, "batched", "decode_only")
        rows = []
        for pr in lfl.proved_rows(c):
            proved = pr["tok_s"]
            row = {
                "boundary": pr["boundary"],
                "proved_key": pr["key"],
                "proved_tok_s": proved,
                "proved_scope": "32 layer chunks (HEADLESS)" if pr["headless"]
                                else "33 chunks (head included)",
                "proved_card": pr["card"],
                "proved_grade": pr["grade"],
                "proved_source": pr["source"],
                "blocked_by": pr["blocked_by"],
                "proved_usd_per_mtok": usd_per_mtok(proved, usd_per_h),
                "latency_overhead_x": None,
                "e11_overhead_x": None,
                "usd_overhead_x": None,
            }
            if proved:
                if single:
                    row["latency_overhead_x"] = single["chosen"]["tok_s"] / proved
                if e11:
                    row["e11_overhead_x"] = e11["chosen"]["tok_s"] / proved
                    row["e11_overhead_x_note"] = (
                        f"{e11['chosen']['tok_s']:.1f} positions/s unproved over "
                        f"{proved:.3f} proved positions/s under the regime-1 "
                        "assumption; see `e11_assumption`.")
                if batched:
                    row["unproved_batched_usd_per_mtok"] = usd_per_mtok(
                        batched["chosen"]["tok_s"], usd_per_h)
                    # Proved and unproved sit on the SAME card, so the $/token
                    # multiple is the throughput multiple and needs no rate at
                    # all; only the ABSOLUTE $/Mtok cells need the cost key.
                    row["usd_overhead_x"] = batched["chosen"]["tok_s"] / proved
                    row["usd_overhead_x_note"] = (
                        "rate-independent: same card on both sides, so the $/token "
                        "ratio equals the aggregate-throughput ratio. Numerator is "
                        f"batch {batched['chosen']['batch_size']} serving, denominator "
                        "is our batch-1 proof: a $/token comparison, NOT a latency one.")
            rows.append(row)
        doc["contexts"][str(c)] = {
            "unproved_single_stream": single,
            "unproved_e11": e11,
            "unproved_batched": batched,
            "rows": rows,
        }
        if single is None:
            doc["caveats"].append(f"C={c}: no single-stream unproved run in {outdir}.")
        else:
            if not single["chosen"]["quotable"]:
                doc["caveats"].append(
                    f"C={c}: the single-stream run is {single['chosen']['lfl_verdict']} "
                    f"({'; '.join(single['chosen']['blocking'])}). The latency column is "
                    "NOT publishable.")
            if single["backends_missing"]:
                doc["caveats"].append(
                    f"C={c}: backend(s) {single['backends_missing']} missing. The belief "
                    "being corrected quotes an OPTIMIZED batched figure, so a hf-only "
                    "baseline flatters us; run the vllm arm before publishing.")
        if batched is None:
            doc["caveats"].append(f"C={c}: no batched unproved run; the $/token column is empty.")
    if usd_per_h is None:
        doc["caveats"].append(
            "Every absolute dollar cell is null: cost.h200_reference_usd_per_hour is "
            "null in the freeze. The MULTIPLES are unaffected -- same card on both "
            "sides makes them rate-independent.")
    return doc


# --- the freeze hand-off ---------------------------------------------------

def constants_entries(doc):
    """The constants.json rows this ladder's artifacts justify, ready to paste.

    Emitted rather than written: `experiments/paper/constants.json` is edited by
    one integrator, and a measurement stream hands over rows, it does not merge
    them.  Every row carries grade, source (the artifact path) and a note.
    """
    out = {}
    rate = doc["cost"]

    # A freeze row may only come from a run that passed the like-for-like check.
    # If any selected run did not -- a synthetic one, the wrong card, the wrong
    # token count -- every key is prefixed and every grade nulled, so what comes
    # out cannot be pasted into `constants.json` by accident.
    chosen = [blk[k]["chosen"] for blk in doc["contexts"].values()
              for k in ("unproved_single_stream", "unproved_e11", "unproved_batched")
              if blk.get(k)]
    bad = [r for r in chosen if not r["quotable"]]
    prefix = "" if (chosen and not bad) else "NOT_QUOTABLE."
    if not chosen:
        return {"_WARNING": "No unproved runs selected; there is nothing to freeze."}
    if bad:
        out["_WARNING"] = (
            "NOT PUBLISHABLE. " + str(len(bad)) + " selected run(s) failed the "
            "like-for-like check (" + "; ".join(sorted({b["lfl_verdict"] for b in bad}))
            + "). Every key below is prefixed `NOT_QUOTABLE.` and every grade is "
              "null so it cannot reach constants.json by accident. Reasons: "
            + " | ".join(sorted({b for r in bad for b in r["blocking"]})))
    card = chosen[0]["card"] if bad else "H200"

    def entry(v, unit, ctx, boundary, grade, source, note):
        return {"value": v, "unit": unit, "card": card, "context": ctx,
                "boundary": boundary, "grade": None if bad else grade,
                "source": source, "note": note}

    out["baseline.e2h_protocol"] = entry(
        "greedy, EOS suppressed, exactly 512 new tokens on a C-token prompt; TTFT "
        "from a max_new_tokens=1 call and decode seconds by subtraction; prompts "
        "distinct per batch element; prefix caching off",
        "statement", "n/a", "unproved decode wall", "measured",
        "experiments/e2/decode_bench.py; artifacts in experiments/e2/out/",
        "The measurement protocol every `tok_s.unproved_8b_h200.*` row was produced "
        "by. Both counting conventions (decode-only and E11 all-positions) come from "
        "the same runs.")

    for c, blk in doc["contexts"].items():
        for arm, suffix, purpose in (("decode_only", "b1", "unproved_single_stream"),
                                     ("e11", "b1_e11", "unproved_e11")):
            sel = blk[purpose]
            if not sel:
                continue
            for be, r in sel["by_backend"].items():
                out[f"tok_s.unproved_8b_h200.{be}.c{c}.{suffix}"] = entry(
                    round(r["tok_s"], 3), "tok/s", int(c), "unproved decode wall",
                    "measured", r["artifact"],
                    f"{r['backend_label']}; batch 1, {r['dtype']}, {r['model']}"
                    f"@{r['revision']}, {r['card']}; "
                    + ("decode only, prefill excluded (v1 counting)." if arm == "decode_only"
                       else f"ALL {r['prompt_tokens']} + {r['new_tokens']} positions of the "
                            "concatenated pass over that pass's wall (E11 counting)."))
        bl = blk["unproved_batched"]
        if bl:
            for be, r in bl["by_backend"].items():
                out[f"tok_s.unproved_8b_h200.{be}.c{c}.batched_peak"] = entry(
                    round(r["tok_s"], 1), "tok/s aggregate", int(c),
                    "unproved decode wall", "measured", r["artifact"],
                    f"Peak AGGREGATE decode rate over the batch sweep, at batch "
                    f"{r['batch_size']}. NOT a latency figure.")
                out[f"batch.unproved_8b_h200.{be}.c{c}.at_peak"] = entry(
                    r["batch_size"], "sequences", int(c), "unproved decode wall",
                    "measured", r["artifact"],
                    "The batch the peak aggregate rate came from. For vLLM this is "
                    "OFFERED concurrency, scheduled by the engine.")
        short = {"commit": "commit", "stwo total": "stwo", "whole wall": "whole",
                 "loop_secs": "loop"}
        for r in blk["rows"]:
            sb = short[r["boundary"]]
            for arm, key, src in (("single_stream", "latency_overhead_x", "unproved_single_stream"),
                                  ("e11", "e11_overhead_x", "unproved_e11"),
                                  ("usd_per_token", "usd_overhead_x", "unproved_batched")):
                v = r[key]
                if v is None:
                    continue
                sel = blk[src]
                note = (f"{sel['chosen']['backend']} unproved "
                        f"{sel['chosen']['tok_s']:.3f} / proved {r['proved_tok_s']:.3f} "
                        f"({r['proved_key']}, {r['proved_scope']}).")
                if arm == "e11":
                    note += " ASSUMPTION: " + E11_ASSUMPTION
                if arm == "usd_per_token":
                    note += (" Rate-independent: same card both sides. NOT a latency "
                             "multiple.")
                out[f"overhead.8b.c{c}.{arm}.{sb}"] = entry(
                    round(v, 1), "x", int(c), r["boundary"], "derived",
                    f"{sel['chosen']['artifact']}; {r['proved_key']}", note)
            if r["proved_usd_per_mtok"] is not None:
                out[f"usd_per_mtok.proved_8b.c{c}.{sb}"] = entry(
                    round(r["proved_usd_per_mtok"], 2), "USD/Mtok", int(c),
                    r["boundary"], "derived",
                    f"{r['proved_key']} / 3600 * cost.h200_reference_usd_per_hour",
                    f"At ${rate['h200_usd_per_gpu_hour']}/GPU-h "
                    f"(from_freeze={rate['from_freeze']}). {r['proved_scope']}.")
        if bl and rate["h200_usd_per_gpu_hour"] is not None:
            r0 = bl["chosen"]
            out[f"usd_per_mtok.unproved_8b.c{c}.batched"] = entry(
                round(usd_per_mtok(r0["tok_s"], rate["h200_usd_per_gpu_hour"]), 4),
                "USD/Mtok", int(c), "unproved decode wall", "derived", r0["artifact"],
                f"Peak aggregate serving at batch {r0['batch_size']}, "
                f"${rate['h200_usd_per_gpu_hour']}/GPU-h "
                f"(from_freeze={rate['from_freeze']}).")
    if prefix:
        out = {(k if k.startswith("_") else prefix + k): v for k, v in out.items()}
    return out


def markdown(doc):
    L = []
    L.append("### The overhead ladder, by boundary\n")
    rate = doc["cost"]
    L.append(f"H200 reference rate: "
             f"{'$%.2f/GPU-h' % rate['h200_usd_per_gpu_hour'] if rate['h200_usd_per_gpu_hour'] else '**null**'}"
             f" (from_freeze={rate['from_freeze']})\n")
    for c, blk in doc["contexts"].items():
        s, e, b = blk["unproved_single_stream"], blk["unproved_e11"], blk["unproved_batched"]
        L.append(f"\n**C={c}**")
        if s:
            for be, r in sorted(s["by_backend"].items()):
                mark = " ← headline" if r is s["chosen"] else ""
                L.append(f"- unproved single-stream, `{be}`: **{r['tok_s']:.2f} tok/s** "
                         f"(batch {r['batch_size']}, {r['dtype']}, {r['card']}, "
                         f"{r['model']}) — {r['lfl_verdict']}{mark}")
        else:
            L.append("- unproved single-stream: **not measured**")
        if e:
            L.append(f"- unproved E11 all-positions: **{e['chosen']['tok_s']:.1f} pos/s** "
                     f"({e['chosen']['backend']}, "
                     f"{e['chosen']['prompt_tokens']}+{e['chosen']['new_tokens']} positions)")
        if b:
            L.append(f"- unproved batched peak: **{b['chosen']['tok_s']:.1f} tok/s aggregate** "
                     f"at batch {b['chosen']['batch_size']} ({b['chosen']['backend']})")
        else:
            L.append("- unproved batched peak: **not measured**")
        L.append("")
        L.append("| boundary | proved tok/s | scope | latency × | E11 × | proved $/Mtok "
                 "| batched $/Mtok | $/token × |")
        L.append("|---|---|---|---|---|---|---|---|")
        rate_missing = f"null ({rate['blocking_key']})" if rate["blocking_key"] else "null"
        for r in blk["rows"]:
            proved_missing = f"null ({r['blocked_by']})" if r["blocked_by"] else "null"

            def f(x, spec, missing):
                return spec.format(x) if isinstance(x, (int, float)) else missing
            L.append(f"| {r['boundary']} | {f(r['proved_tok_s'], '{:.3f}', proved_missing)} | "
                     f"{r['proved_scope']} | "
                     f"{f(r['latency_overhead_x'], '×{:.1f}', proved_missing)} | "
                     f"{f(r['e11_overhead_x'], '×{:.1f}', proved_missing)} | "
                     f"{f(r['proved_usd_per_mtok'], '${:.2f}', rate_missing)} | "
                     f"{f(r.get('unproved_batched_usd_per_mtok'), '${:.4f}', rate_missing)} | "
                     f"{f(r['usd_overhead_x'], '×{:.0f}', proved_missing)} |")
    L.append("\n**The E11 column rests on an assumption:** " + doc["e11_assumption"])
    L.append("\n**What neither side counts:**")
    for w in doc["what_neither_side_counts"]:
        L.append(f"- {w}")
    if doc["caveats"]:
        L.append("\n**Caveats the harness raised itself:**")
        for cv in doc["caveats"]:
            L.append(f"- {cv}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", required=True, help="directory of decode_bench run JSONs")
    constants.add_argument(p)
    p.add_argument("--out", default=None, help="write the ladder JSON here")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--emit-constants", action="store_true",
                   help="print the constants.json rows these artifacts justify")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="DO NOT USE for a table: include mock-backend runs, to "
                        "exercise this file where no card exists")
    p.add_argument("--usd-per-gpu-hour", type=float, default=None,
                   help="override the frozen H200 rate; stamps from_freeze=false")
    a = p.parse_args()
    constants.load(a.constants)
    doc = build(a.outdir, a.usd_per_gpu_hour, a.allow_synthetic)
    if a.allow_synthetic:
        doc["WARNING"] = ("--allow-synthetic: this table may contain SYNTHETIC seconds "
                          "from the mock backend. Not publishable.")
        print("WARNING: --allow-synthetic; this table is not publishable",
              file=sys.stderr)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=2))
        # stderr, so `--markdown > table.md` produces a clean table
        print(f"wrote {a.out}", file=sys.stderr)
    if a.emit_constants:
        print(json.dumps(constants_entries(doc), indent=2))
    elif a.markdown or not a.out:
        print(markdown(doc))


if __name__ == "__main__":
    main()
