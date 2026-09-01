"""Stream E7 -- what it costs to prove a full evaluation suite.

    cd experiments/paper && uv run --project .. python sufficiency.py

Every rate, cost and cells constant comes from `constants.json`; nothing is
inlined here.  The only input from outside the freeze is B8's per-item JSONL,
and that is used only to re-derive the token counts that ARE frozen -- so a
change to either side shows up as a mismatch in `check.py`.

Two accountings, both reported:

  regime 1  per-position: every proved position is charged one measured
            C=4096 decoded token at the named boundary.  The paper's unit,
            and the only one made of measurements alone.
  regime 2  per-proof: one proof per eval item, priced in cells from B9's
            causal-rectangle fit plus the composed grid slope, converted at
            the measured G cells/s of the named boundary.  [projected]

Dollar figures use the `*_reference_*` cost keys only (freeze rule 4).
"""

import json
import os
import statistics
import sys

from constants import get, value as v

HERE = os.path.dirname(os.path.abspath(__file__))
B8 = os.path.join(HERE, os.pardir, "b8", "out", "tasks")
SEC_PER_H = 3600.0
SPECS = ("fp32", "dynamic-int8", "b6-design", "b6-design-p2o", "b65-design", "static")
TASKS = ("gpqa", "mmlu_pro", "ifeval")


# -- token accounting, straight off B8's per-item rows ---------------------
def rows(task, spec):
    p = os.path.join(B8, f"{task}.{spec}.0of1.jsonl")
    return [json.loads(line) for line in open(p) if line.strip()]


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
            # (the answer position, on a declared prefix); n_gen for IFEval.
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
    """Coefficients, each traceable to a frozen entry.

    B9's measured causal-rectangle fit, per LAYER chunk of S positions:
        fresh(S) = A + B*S + Q*S^2            A=326e6, B=1.37e6, Q=72.1
    The composed decode chunk (our own measurement), per LAYER chunk of
    S=512 decoded tokens over a declared band of C keys:
        fresh(C) = I + G*heads*C*S            G = composed_chunk_fresh_slope_32layers
    I is solved from the measured C=4096 point.  The cross-check is that the
    two models agree where they should: I (a decode chunk with no band left,
    i.e. the corner alone) against the fit at S=512.
    """
    S = v("spec.decoded_tokens")
    L = 32
    heads = 32
    G = v("attention_grid.composed_chunk_fresh_slope_32layers")
    per_layer_c4096 = v("cells.c4096.fresh_layers32") / L
    per_layer_c16384 = v("cells.c16384.fresh_layers32") / L
    slope_check = (per_layer_c16384 - per_layer_c4096) / ((16384 - 4096) * heads * S)
    I = per_layer_c4096 - G * heads * 4096 * S
    A, B, Q = 326e6, 1.37e6, 72.1        # B9's fit, as quoted in the freeze
    head_per_token = v("cells.c4096.fresh_head") / S
    return dict(L=L, heads=heads, G=G, I=I, A=A, B=B, Q=Q, S=S,
                slope_check=slope_check, per_layer_c4096=per_layer_c4096,
                head_per_token=head_per_token, fit_at_512=A + B * S + Q * S * S)


def cells_decode_item(m, n_prompt, n_gen):
    """One proof of one item under the v1 decode-only statement: n_gen decoded
    positions (1 for a multiple-choice item) over a DECLARED band of n_prompt
    keys.  The per-chunk shared tables are paid once per layer, per item."""
    Sn = max(n_gen, 1)
    per_layer = (m["A"] + m["B"] * Sn + m["G"] * m["heads"] * n_prompt * Sn
                 + m["Q"] * Sn * Sn)
    return m["L"] * per_layer + m["head_per_token"] * Sn


def cells_e11_item(m, n_prompt, n_gen):
    """One proof of one item under E11: one causal rectangle over the whole
    concatenated prompt+output, N = n_prompt + n_gen positions."""
    N = n_prompt + n_gen
    per_layer = m["A"] + m["B"] * N + m["Q"] * N * N
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
    b10_mean = 382
    b10_gen = b10_mean * acc["ifeval"]["items"]
    print(f"   {'b10-design':16s} {b10_gen:9,d}  (mean {b10_mean}, "
          f"experiments/b10/out/tasks.md; rounded to the integer, so +/-0.13%)")
    print()
    longest = max(max(p + g for p, g in acc[t]["lengths"]) for t in TASKS)
    print(f"longest sequence anywhere in the suite: {longest:,d} tokens "
          f"-> the whole suite fits inside C=4096")
    print()

    # headline uses b10-design's IFEval generation: the freeze's task-row spec
    dec = (tot["items"] - acc["ifeval"]["items"]) + b10_gen
    e11 = tot["prompt"] + b10_gen
    print(f"HEADLINE POSITIONS (b10-design IFEval generation):")
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
    print(f"   composed slope, from the freeze   {m['G']:.4f} cells/(head,key,token)")
    print(f"   re-solved from the two contexts   {m['slope_check']:.4f}   (cross-check)")
    print(f"   decode chunk at C=4096, per layer {m['per_layer_c4096'] / 1e9:.4f} G")
    print(f"   its band-free intercept I         {m['I'] / 1e9:.4f} G")
    print(f"   B9's fit at S=512                 {m['fit_at_512'] / 1e9:.4f} G   "
          f"(the two agree to {abs(m['fit_at_512'] - m['I']) / m['I'] * 100:.1f}%)")
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
