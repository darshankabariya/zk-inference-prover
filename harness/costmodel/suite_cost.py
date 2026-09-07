"""Stream E7 -- what it costs to prove a full evaluation suite.

    uv run python -m harness.costmodel.suite_cost --constants PATH --artifacts DIR

Every rate, cost and cells constant comes from a zk-results `constants.json`
(`--constants PATH`); nothing is inlined here.  The only input from outside the freeze is B8's per-item JSONL,
and that is used only to re-derive the token counts that ARE frozen -- so a
change to either side shows up as a mismatch in `check.py`.

Two accountings, both reported:

  regime 1  per-position: every proved position is charged one measured
            C=4096 decoded token at the named boundary.  The paper's unit,
            and the only one made of measurements alone.
  regime 2  per-proof: one proof per eval item, priced in cells from stream
            DD's exact (S, C) band model plus a shared-table floor solved
            against our own measured composed chunk, converted at the
            measured G cells/s of the named boundary.  [projected]

Dollar figures use the `*_reference_*` cost keys only (freeze rule 4).
"""

import glob
import json
import os
import statistics
import sys

from harness import inputs
from harness.constants import get, value as v

# The per-item task rows: `--artifacts DIR` holds `tasks/`, `e5t/tasks/` and
# `e5v/tasks/` as the eval harness wrote them.
ART = str(inputs.artifacts_dir())
B8 = os.path.join(ART, "tasks")
E5T = os.path.join(ART, "e5t", "tasks")
E5V = os.path.join(ART, "e5v", "tasks")
SEC_PER_H = 3600.0
SPECS = ("fp32", "dynamic-int8", "b6-design", "b6-design-p2o", "b65-design", "static")
TASKS = ("gpqa", "mmlu_pro", "ifeval")
# `spec.name`.  The paper's default spec since 2026-09-02 (notes/eval-plan.md
# section 1.2 addendum); stream E5v scored the same corpus on it.  It was
# `b10-final` before that, and E5t's rows for the two `b10` specs are still read
# below as the retired values.
FROZEN_SPEC = "y-delay-vw12"
SPEC_ROWS = {"b10-design": E5T, "b10-final": E5T, "y-delay-vw12": E5V}


# -- token accounting, straight off B8's per-item rows ---------------------
def rows(task, spec):
    p = os.path.join(B8, f"{task}.{spec}.0of1.jsonl")
    return [json.loads(line) for line in open(p) if line.strip()]


def scored_ifeval(spec):
    """The per-item IFEval rows for one scored spec, across its shards.

    E5t's directory for the two `b10` specs, E5v's for `y-delay-vw12`.  The
    suite is priced under the spec the paper ships, so the generated-token count
    is the EXACT sum of `n_gen` over these 541 rows -- not a rounded mean times
    an item count, which is what `suite.ifeval.generated_tokens` held before
    E5t's artifacts existed.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(SPEC_ROWS[spec], f"ifeval.{spec}.*of*.jsonl"))):
        rows += [json.loads(ln) for ln in open(f) if ln.strip()]
    assert len(rows) == 541, f"ifeval.{spec}: {len(rows)} rows, expected 541"
    return rows


def accounting(spec="b6-design-p2o"):
    out = {}
    for task in TASKS:
        r = rows(task, spec)
        out[task] = dict(
            items=len(r),
            prompt=sum(x["n_prompt"] for x in r),
            prompt_mean=statistics.mean(x["n_prompt"] for x in r),
            prompt_median=statistics.median(x["n_prompt"] for x in r),
            prompt_max=max(x["n_prompt"] for x in r),
            gen=sum(x.get("n_gen", 0) for x in r),
            gen_mean=statistics.mean(x.get("n_gen", 0) for x in r),
            gen_max=max(x.get("n_gen", 0) for x in r),
            # decode-only (v1): 1 proved position for a multiple-choice item
            # (the answer position, on an unchecked prefix); n_gen for IFEval.
            decode=len(r) if task != "ifeval" else sum(x["n_gen"] for x in r),
            # E11: every position of the concatenated prompt+output.
            e11=sum(x["n_prompt"] + x.get("n_gen", 0) for x in r),
            lengths=[(x["n_prompt"], x.get("n_gen", 0)) for x in r],
        )
    return out


# -- the four boundaries ---------------------------------------------------
def boundaries():
    """(name, card, tok/s, $/GPU-h, note), in the order the paper uses."""
    h200 = v("cost.h200_reference_usd_per_hour")
    h100 = v("cost.h100_reference_usd_per_hour")
    S = v("spec.decoded_tokens")
    return [
        ("commit", "H200", v("tok_s.seq5.c4096.commit"), h200, "33 chunks"),
        ("stwo total", "H200", v("tok_s.seq5.c4096.stwo"), h200, "33 chunks"),
        ("whole wall", "H200", v("tok_s.seq5.c4096.whole"), h200, "33 chunks"),
        ("loop_secs", "H200", S / v("loop.h200.c4096.pipelined_defaults"), h200,
         "32 layer chunks, pipelined, HEADLESS"),
        ("loop_secs", "H100", S / v("loop.h100.c4096.pipelined_defaults"), h100,
         "32 layer chunks, pipelined, HEADLESS"),
    ]


def usd_per_position(tok_s, usd_h):
    return usd_h / SEC_PER_H / tok_s


# -- regime 2: cells for one proof of one item ----------------------------
def cell_model():
    """The whole layer chunk in (S, C), composed from frozen pieces.

        chunk_fresh(S, C) = fixed + per_token*S + band_fresh(S, C)
        band_fresh(S, C)  = 140*S*C + 24,960*C + 12,688*S

    The band half is stream DD's exact closed form
    (`attention_grid.band_model_fresh_coeffs`) -- structural, not fitted, and
    reproduced to the cell on eight artifacts including composed 8B chunks at
    C != S.  It REPLACES the retired `326 M + 1.37 M*S + 72.1*S^2` fit, whose
    S^2 coefficient (72.083) ran +12.6% against the true 64 because it charged
    the band family's whole linear term to the quadratic.

    `per_token` is `design.prefill_chunk_remainder[1]`, DD's two-point fit of
    everything the band family does not cover.  Its `[0]` partner is the same
    floor for B9's SYNTHETIC probe chunk; our composed layer statement also
    carries the glue, both residual adds and the committed residual streams,
    so `fixed` is solved separately against our own measured chunk at C=4096.
    Predicting C=16384 from it lands at -0.63% -- a held-out check, whose
    residual is the composed chunk's own small C-growth outside the family.
    """
    S = v("spec.decoded_tokens")
    sc, bc, bs = v("attention_grid.band_model_fresh_coeffs")
    fixed_b9, per_token = v("design.prefill_chunk_remainder")
    return dict(
        L=32, S=S, sc=sc, bc=bc, bs=bs,
        per_token=per_token,
        fixed=v("attention_grid.chunk_model_fixed_composed"),
        fixed_b9=fixed_b9,
        head_per_token=v("cells.c4096.fresh_head") / S,
        per_layer_c4096=v("cells.c4096.fresh_layers32") / 32,
        per_layer_c16384=v("cells.c16384.fresh_layers32") / 32,
    )


def band_fresh(m, S, C):
    """The attention band family's fresh cells: S query tokens over C keys."""
    return m["sc"] * S * C + m["bc"] * C + m["bs"] * S


def cells_decode_item(m, n_prompt, n_gen, fixed=None):
    """One proof of one item under the v1 decode-only statement: n_gen decoded
    positions (1 for a multiple-choice item) over an UNCHECKED band of n_prompt
    keys, plus the corner in which those positions attend to each other.  The
    per-chunk shared tables are paid once per layer, per item."""
    Sn = max(n_gen, 1)
    f = m["fixed"] if fixed is None else fixed
    per_layer = (f + m["per_token"] * Sn
                 + band_fresh(m, Sn, n_prompt) + band_fresh(m, Sn, Sn))
    return m["L"] * per_layer + m["head_per_token"] * Sn


def cells_e11_item(m, n_prompt, n_gen, fixed=None):
    """One proof of one item under E11: one causal rectangle over the whole
    concatenated prompt+output, N = n_prompt + n_gen positions attending over
    its own N keys.  One block: every item in this suite is under the S~1K
    block size, so `design.e11_chunking_net_pct` does not bite here."""
    N = n_prompt + n_gen
    f = m["fixed"] if fixed is None else fixed
    per_layer = f + m["per_token"] * N + band_fresh(m, N, N)
    return m["L"] * per_layer + m["head_per_token"] * N


def g_cells_rates():
    """G fresh cells/s at each boundary (check.py's convention)."""
    fresh32 = v("cells.c4096.fresh_layers32")
    h200 = v("cost.h200_reference_usd_per_hour")
    h100 = v("cost.h100_reference_usd_per_hour")
    return [
        ("commit", "H200", v("rate.h200.c4096.commit_33chunks"), h200),
        ("stwo total", "H200", v("rate.h200.c4096.stwo_33chunks"), h200),
        ("whole wall", "H200", v("rate.h200.c4096.whole_33chunks"), h200),
        ("loop_secs", "H200", fresh32 / 1e9 / v("loop.h200.c4096.pipelined_defaults"), h200),
        ("loop_secs", "H100", fresh32 / 1e9 / v("loop.h100.c4096.pipelined_defaults"), h100),
    ]


def main():
    acc = accounting()
    accs = {s: accounting(s) for s in SPECS}

    print("=" * 86)
    print("1. TOKEN ACCOUNTING (B8 per-item rows; prompt counts are spec-independent)")
    print("=" * 86)
    print(f"{'task':10s} {'items':>6s} {'prompt tok':>11s} {'mean':>7s} {'med':>6s} "
          f"{'max':>6s} {'gen tok':>9s} {'mean':>7s} {'max':>5s}")
    tot = dict(items=0, prompt=0, gen=0, decode=0, e11=0)
    for t in TASKS:
        a = acc[t]
        print(f"{t:10s} {a['items']:6d} {a['prompt']:11,d} {a['prompt_mean']:7.1f} "
              f"{a['prompt_median']:6.0f} {a['prompt_max']:6d} {a['gen']:9,d} "
              f"{a['gen_mean']:7.1f} {a['gen_max']:5d}")
        for k in tot:
            tot[k] += a[k]
    print(f"{'TOTAL':10s} {tot['items']:6d} {tot['prompt']:11,d} "
          f"{'':7s} {'':6s} {'':6s} {tot['gen']:9,d}")
    print()
    print(f"{'task':10s} {'decode-only pos':>16s} {'E11 pos':>12s} {'ratio':>8s}")
    for t in TASKS:
        a = acc[t]
        print(f"{t:10s} {a['decode']:16,d} {a['e11']:12,d} {a['e11'] / a['decode']:8.2f}x")
    print(f"{'SUITE':10s} {tot['decode']:16,d} {tot['e11']:12,d} "
          f"{tot['e11'] / tot['decode']:8.2f}x")
    print()
    print("spec spread on IFEval generated tokens (prompt tokens identical everywhere):")
    for s in SPECS:
        a = accs[s]["ifeval"]
        print(f"   {s:16s} {a['gen']:9,d}  (mean {a['gen_mean']:.2f})")
    scored = {s: sum(r["n_gen"] for r in scored_ifeval(s)) for s in SPEC_ROWS}
    for s, g in scored.items():
        d = "e5t" if SPEC_ROWS[s] is E5T else "e5v"
        print(f"   {s:16s} {g:9,d}  (mean {g / 541:.3f}, "
              f"b8/out/{d}/tasks/ifeval.{s}.*of3.jsonl -- EXACT sum)")
    frozen_gen = scored[FROZEN_SPEC]
    print(f"   two retired entries: 382 x 541 = {382 * 541:,d} -- b10-design's mean "
          f"ROUNDED to the integer, {382 * 541 / scored['b10-design'] - 1:+.3%} against")
    print(f"   b10-design's own exact sum -- and b10-final's {scored['b10-final']:,d}, "
          f"which the suite was priced on until 2026-09-02.")
    print(f"   The paper's spec {FROZEN_SPEC} is "
          f"{frozen_gen / scored['b10-final'] - 1:+.3%} against b10-final and "
          f"{frozen_gen / (382 * 541) - 1:+.3%} against the rounded mean.")
    print()
    longest = max(max(p + g for p, g in acc[t]["lengths"]) for t in TASKS)
    print(f"longest sequence anywhere in the suite: {longest:,d} tokens "
          f"-> the whole suite fits inside C=4096")
    print()

    # headline uses the FROZEN spec's own exact IFEval generation (stream E5t)
    dec = (tot["items"] - acc["ifeval"]["items"]) + frozen_gen
    e11 = tot["prompt"] + frozen_gen
    print(f"HEADLINE POSITIONS ({FROZEN_SPEC} IFEval generation, exact sum):")
    print(f"   decode-only {dec:,d}    E11 {e11:,d}    ratio {e11 / dec:.3f}x")
    print()

    print("=" * 86)
    print("2. REGIME 1 -- per-position, at the four boundaries  [derived from measured]")
    print("=" * 86)
    print(f"{'boundary':12s} {'card':5s} {'tok/s':>8s} {'$/GPU-h':>8s} {'$/pos':>11s} "
          f"{'$ decode':>10s} {'$ E11':>10s} {'GPU-h dec':>10s} {'GPU-h E11':>10s}")
    reg1 = {}
    for name, card, tps, usdh, note in boundaries():
        pp = usd_per_position(tps, usdh)
        reg1[(name, card)] = (tps, pp, dec * pp, e11 * pp)
        print(f"{name:12s} {card:5s} {tps:8.3f} {usdh:8.2f} {pp:11.3e} "
              f"{dec * pp:10.2f} {e11 * pp:10.2f} {dec / tps / 3600:10.2f} "
              f"{e11 / tps / 3600:10.2f}")
    hf = v("tok_s.seq5.c4096.whole_headless") / v("tok_s.seq5.c4096.whole")
    print(f"   loop_secs rows are HEADLESS (32 layer chunks). Whole-wall headless/with-head")
    print(f"   is {hf:.4f}x, so a head-inclusive loop_secs is ~{hf:.3f}x dearer [projected]:")
    for card in ("H200", "H100"):
        _, pp, d, e = reg1[("loop_secs", card)]
        print(f"      loop_secs+head {card}   $ decode {d * hf:8.2f}   $ E11 {e * hf:8.2f}")
    print()

    print("=" * 86)
    print("3. REGIME 2 -- one proof per item, priced in cells  [projected]")
    print("=" * 86)
    m = cell_model()
    held = (m["fixed"] + m["per_token"] * m["S"] + band_fresh(m, m["S"], 16384)
            + band_fresh(m, m["S"], m["S"]))
    print(f"   band model (frozen, exact)        {m['sc']}*S*C + {m['bc']:,}*C + {m['bs']:,}*S")
    print(f"   per_token (prefill_chunk_remainder)  {m['per_token']:,.0f} cells/token/layer")
    print(f"   fixed, solved on our C=4096 chunk {m['fixed'] / 1e6:,.1f} M   "
          f"(b9 probe's own floor {m['fixed_b9'] / 1e6:,.1f} M, "
          f"{m['fixed'] / m['fixed_b9'] - 1:+.1%})")
    print(f"   HELD OUT at C=16384: model {held:,.0f}  measured "
          f"{m['per_layer_c16384']:,.0f}   {held / m['per_layer_c16384'] - 1:+.4%}")
    print(f"   band alpha_fresh {m['sc'] / 32:.3f} + its C-term aliased at S=512 "
          f"{m['bc'] / (32 * m['S']):.4f} = "
          f"{v('attention_grid.composed_slope_explained_by_band_model'):.4f}")
    print(f"   against the frozen composed slope "
          f"{v('attention_grid.composed_chunk_fresh_slope')} -- so 5.898 is NOT a per-entry price")
    print()
    cd = {t: sum(cells_decode_item(m, p, g if t == "ifeval" else 1)
                 for p, g in acc[t]["lengths"]) for t in TASKS}
    ce = {t: sum(cells_e11_item(m, p, g) for p, g in acc[t]["lengths"]) for t in TASKS}
    print(f"{'task':10s} {'decode G cells':>16s} {'E11 G cells':>14s} {'ratio':>8s}")
    for t in TASKS:
        print(f"{t:10s} {cd[t] / 1e9:16,.1f} {ce[t] / 1e9:14,.1f} {ce[t] / cd[t]:8.2f}x")
    CD, CE = sum(cd.values()), sum(ce.values())
    print(f"{'SUITE':10s} {CD / 1e9:16,.1f} {CE / 1e9:14,.1f} {CE / CD:8.2f}x")
    print()
    print(f"{'boundary':12s} {'card':5s} {'G cells/s':>10s} {'$ decode':>10s} {'$ E11':>10s}")
    for name, card, rate, usdh in g_cells_rates():
        f = usdh / SEC_PER_H / (rate * 1e9)
        print(f"{name:12s} {card:5s} {rate:10.4f} {CD * f:10.2f} {CE * f:10.2f}")
    print()
    per_tok = v("cells_per_token.8b.c4096.fresh_all")
    print(f"   regime 1's position counts in the same currency: decode "
          f"{dec * per_tok / 1e9:,.1f} G, E11 {e11 * per_tok / 1e9:,.1f} G")
    print(f"   (at {per_tok / 1e6:.1f} M fresh cells per measured decoded token, C=4096)")
    print()

    print("=" * 86)
    print("4. SUB-SAMPLING")
    print("=" * 86)
    for name, card in (("commit", "H200"), ("stwo total", "H200"),
                       ("whole wall", "H200"), ("loop_secs", "H200")):
        _, _, d, e = reg1[(name, card)]
        print(f"{name:12s} {card}  p=1% ${d * 0.01:7.2f} / ${e * 0.01:7.2f}"
              f"   p=10% ${d * 0.10:7.2f} / ${e * 0.10:7.2f}"
              f"   p=100% ${d:7.2f} / ${e:7.2f}   (decode / E11)")
    print()
    n = tot["items"]
    print(f"   Detection, elementary: the auditor demands proofs for a random "
          f"p-fraction of the {n:,d} items;")
    print("   a provider who substituted a cheaper model on a fraction f of them is")
    print("   caught with probability 1-(1-p)^(f*n).")
    ps = (0.001, 0.005, 0.01, 0.05, 0.10)
    print("   " + f"{'f \\ p':>8s}" + "".join(f"{p:>9.1%}" for p in ps))
    for f in (0.01, 0.05, 0.10, 0.25):
        print("   " + f"{f:>8.0%}" + "".join(f"{1 - (1 - p) ** (f * n):>9.1%}" for p in ps))
    print()

    print("=" * 86)
    print("5. THE UNPROVED COMPARISON -- BLOCKED on tok_s.unproved_8b_h200")
    print("=" * 86)
    e = get("tok_s.unproved_8b_h200")
    print(f"   note: {e['note']}")
    print("   The unproved run does every position of every item either way, so its")
    print("   cost is the E11 position count at the unproved rate T:")
    print("      $_unproved(T) = e11_positions / (T * 3600) * $/GPU-h")
    print("   latency multiple = T / R_boundary;  dollar multiple = $_proved / $_unproved")
    print(f"   {'T (tok/s)':>10s} {'$ unproved':>12s}  " +
          "".join(f"{b[0][:10] + '/' + b[1]:>18s}" for b in boundaries()[:4]))
    for T in (100, 125, 150, 200):
        usd = e11 / (T * 3600) * v("cost.h200_reference_usd_per_hour")
        mult = "".join(f"{T / b[2]:>18.1f}" for b in boundaries()[:4])
        print(f"   {T:>10d} {usd:>12.3f}  {mult}   <- latency multiple")
    print()
    print(f"   dollar multiple against the SAME single-stream unproved run "
          f"(T=125 illustration):")
    usd125 = e11 / (125 * 3600) * v("cost.h200_reference_usd_per_hour")
    for name, card in (("commit", "H200"), ("stwo total", "H200"),
                       ("whole wall", "H200"), ("loop_secs", "H200")):
        _, _, d, e = reg1[(name, card)]
        print(f"      {name:12s} {card}   decode-only x{d / usd125:6.1f}   E11 x{e / usd125:6.1f}")
    print("   The T column is NOT frozen. notes/final-framing.md section 4 carries")
    print("   ~125-200 tok/s as a [derived] band awaiting E2; it is an illustration,")
    print("   not a measurement, and no single value may be quoted from it.")
    print("   Against BATCHED serving the dollar multiple is larger again; section 4")
    print("   puts it at ~10^3 at the whole wall. Also blocked on E2.")
    print()

    print("=" * 86)
    print("6. THE OTHER TWO COSTS OF AN AUDIT: BYTES AND VERIFY TIME")
    print("=" * 86)
    mib = v("proof.seq5.c4096.mib_per_token")
    vsec = v("verify.seq5.c4096.sequence_secs") / v("spec.decoded_tokens")
    for label, npos in (("decode-only", dec), ("E11", e11)):
        print(f"   {label:12s} {npos:9,d} positions -> "
              f"{npos * mib / 1024:8,.1f} GiB of proof, "
              f"{npos * vsec / 3600:6.2f} CPU-h to verify")
    print(f"   ({mib:.4f} MiB and {vsec * 1000:.2f} ms per position, seq5 C=4096, 33 chunks,")
    print(f"    no aggregation layer -- scope.aggregation_status is "
          f"'{v('scope.aggregation_status')}'.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
