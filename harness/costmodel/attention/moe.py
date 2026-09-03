"""B9 step 2, probe 2: the MoE negative control (H6).

H6 says MoE breaks the batching: *confirmed* if the MoE variant's per-proof
weight cost is x E_experts and/or it needs a routing argument, *falsified* if MoE
proves as cheaply as dense.

The measurement is the **walk region** of the FFN at the 8B shape, at the token
batch a top-k router leaves each expert (S k / E), read out of `compose` runs on
the SIMD reference (`b9/out/moe_s*.json`).  The question the runs answer is not
"how much does the walk cost" -- B3 measured that -- but **which part of it a
smaller token batch makes cheaper**.  The answer is: the fresh cells, which are a
sixth of it, and not the preprocessed weight trace, which is the rest and is
exactly invariant.  That is the whole of H6 -- the weight term is per *weight*,
so partitioning the tokens across E experts multiplies it by E and amortises
nothing.

    uv run experiments/b9/moe.py
"""
import json
import pathlib

from harness import inputs

HERE = pathlib.Path(__file__).parent
# `--artifacts DIR`: the `moe_s*.json` probe outputs; `moe.json` is written beside them.
OUT = inputs.artifacts_dir()

# the 8B FFN, as `B7_SYNTH_DIMS` gives it to the runs
D, DFF = 4096, 14336
FFN_PARAMS = 2 * D * DFF + DFF * D          # gate, up, down
S, TOPK = 512, 2
EXPERTS = (1, 2, 4, 8)


def load(s):
    p = OUT / f"moe_s{s}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["result"]["point"]
    fc = d["family_cells"]
    groups = {g["group"]: g for g in fc["walk_weights"]}
    ffn = ["gateup", "down"]
    return dict(
        s=d["params"]["s"], accepts=d["accepts"], off_table=d["off_table"],
        walks=fc["walks"], fresh=d["cells"]["fresh"], pre=d["cells"]["preprocessed"],
        groups=groups,
        ffn_weights=sum(groups[g]["weights"] for g in ffn if g in groups),
        ffn_pre=sum(groups[g]["pre_cells"] for g in ffn if g in groups),
        attn_weights=sum(groups[g]["weights"] for g in groups if g not in ffn),
        attn_pre=sum(groups[g]["pre_cells"] for g in groups if g not in ffn),
        mats={m["name"]: m["cells"] for m in fc["mats"]},
    )


rows = [r for r in (load(s) for s in (512, 256, 128, 64)) if r]
if not rows:
    raise SystemExit("no b9/out/moe_s*.json; run `sh b9/gpu/b9_run.sh moe` on the box")


def g(x, unit=""):
    for sfx, d in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.2f} {sfx}{unit}"
    return f"{x:.2f} {unit}"


print("=" * 92)
print("B9 probe 2 -- the MoE negative control (H6), the FFN of the 8B shape")
print("=" * 92)
print(f"FFN per expert: gate/up {D}x{DFF} + down {DFF}x{D} = {g(FFN_PARAMS)}weights")
print(f"the walk region, measured at the per-expert token batch a top-{TOPK} router leaves:")
print()
print(f"  {'tokens':>7} {'= S k/E for E':>14} {'walk fresh':>12} {'FFN walk pre':>13} "
      f"{'FFN mat cells':>14} {'accepts':>8}")
for r in rows:
    e = S * TOPK / r["s"] if r["s"] else 0
    ffn_mats = sum(v for k, v in r["mats"].items() if k in ("gate", "up", "down"))
    print(f"  {r['s']:>7} {e:>14.0f} {g(r['walks']):>12} {g(r['ffn_pre']):>13} "
          f"{g(ffn_mats):>14} {str(r['accepts']):>8}")

base = next(r for r in rows if r["s"] == S)
small = min(rows, key=lambda r: r["s"])
print()
print(f"  Falling {base['s']}->{small['s']} tokens ({base['s'] // small['s']}x), the walk's fresh cells move "
      f"{small['walks'] / base['walks'] - 1:+.1%} -- they are")
print(f"  laid over rows the token batch chunks -- while its **preprocessed weight trace")
print(f"  moves {small['ffn_pre'] / base['ffn_pre'] - 1:+.1%}**: {g(base['ffn_pre'])}cells either way, {base['ffn_pre'] / base['walks']:.0f}x the fresh side.  That")
print("  invariant term is the weights themselves, read once per proof, and it is what")
print("  H6 is about: an expert whose batch shrinks does not get cheaper to walk -- it")
print("  gets *more* expensive per token it serves.")

# --- the model ---------------------------------------------------------------
print()
print("what that means per proof and per token, at S=512:")
print(f"  {'E':>3} {'per-expert tokens':>18} {'walk fresh':>12} {'walk pre':>11} "
      f"{'FFN accs':>10} {'FFN cells/token':>16} {'vs dense':>9}")
dense_ffn_mats = sum(v for k, v in base["mats"].items() if k in ("gate", "up", "down"))
dense_walk = base["walks"]
dense_pre = base["ffn_pre"]
dense_total = dense_walk + dense_pre + dense_ffn_mats
for e in EXPERTS:
    per_expert = max(1, S * TOPK // e) if e > 1 else S
    r = min(rows, key=lambda x: abs(x["s"] - per_expert))
    # every expert the sequence touches is walked in full; at S=512 with top-2 a
    # sequence touches all E of them for any E a layer would use
    walk = e * r["walks"] if e > 1 else dense_walk
    pre = e * r["ffn_pre"] if e > 1 else dense_pre
    # the accumulators: each token still passes through exactly k experts, so the
    # dequant side is x k, not x E
    accs = dense_ffn_mats * (TOPK if e > 1 else 1)
    total = walk + pre + accs
    print(f"  {e:>3} {per_expert:>18} {g(walk):>12} {g(pre):>11} {g(accs):>10} "
          f"{g(total / S):>16} {total / dense_total:>8.2f}x")

# --- the routing argument, priced by analogy ---------------------------------
# Nobody has built it.  Two pieces, each with a measured analogue:
#   (a) selection: every token's FFN rows have to be tied to the instance of the
#       expert it was routed to -- a LogUp multiset over (token, expert) pairs.
#       The nearest measured thing is B4's packed pair table, 2 cells per lookup
#       plus the committed expert id: ~3 cells per (token, chosen expert).
#   (b) top-k: the router's argmax over E logits per token.  The nearest measured
#       thing is B6.5's max argument, which costs 11.57 - 8.21 = 3.36 cells per
#       value on top of the requantize it rides along with.
SEL_CELLS = 3.0
MAX_CELLS = 3.36
print()
print("the routing-selection argument, priced by analogy (nobody has built it):")
print(f"  selection  a LogUp multiset tying each token's rows to its expert's instance:")
print(f"             ~{SEL_CELLS:.0f} cells per (token, chosen expert) -- B4's packed pair "
      f"table, 2 cells/lookup, plus the committed expert id")
print(f"  top-k      an argmax over E router logits per token: ~{MAX_CELLS:.2f} cells per "
      f"comparison -- B6.5's max argument (11.57 - 8.21 measured)")
for e in EXPERTS[1:]:
    cells = S * (TOPK * SEL_CELLS + e * MAX_CELLS)
    print(f"    E={e}: {g(cells)}cells per proof = {g(cells / S)}cells/token "
          f"({cells / dense_total * 100:.4f}% of the dense FFN's {g(dense_total)})")
print("  -- small beside the walk, and the point is that it is a *new gadget*: no AIR")
print("     in this project ties a committed row to a routed instance.")

print()
ratios = []
for e in EXPERTS:
    per_expert = max(1, S * TOPK // e) if e > 1 else S
    r = min(rows, key=lambda x: abs(x["s"] - per_expert))
    walk = e * r["walks"] if e > 1 else dense_walk
    pre = e * r["ffn_pre"] if e > 1 else dense_pre
    accs = dense_ffn_mats * (TOPK if e > 1 else 1)
    ratios.append((e, (walk + pre + accs) / dense_total))
d = dict(ratios)
print("H6, in one paragraph.")
print("**Confirmed**, on its own clause: the per-proof weight cost is x E_experts, and a")
print("routing argument is needed on top.  What a Freivalds proof pays for a matmul is")
print(f"dominated by the weight trace it walks -- {g(base['ffn_pre'])}preprocessed cells for the 8B")
print(f"FFN against {g(base['walks'])}fresh -- and that term is per *weight*, so it is exactly")
print(f"invariant ({small['ffn_pre'] / base['ffn_pre'] - 1:+.1%}) when an expert's token batch falls {base['s']}->{small['s']}.  A")
print(f"sequence of {S} tokens with top-{TOPK} routing touches every expert a layer would use,")
print("so the proof walks all E weight matrices in full: the weight term is x E while")
print("the accumulators grow only x k.  Per token at S=512 the FFN costs " +
      ", ".join(f"x{v:.2f} at E={e}" for e, v in ratios) + ".")
print("The comparison that matters is against a dense model with the same *active*")
print(f"parameters, which pays x{TOPK}: at E=8 MoE pays x{d[8]:.2f}, i.e. **x{d[8] / TOPK:.2f} of pure proving")
print("overhead for the same FLOPs per token**.  And the scheme still lacks the gadget:")
print("no AIR in this project ties a committed row to a routed instance.  Priced by")
print("analogy the argument is cheap (<0.01% of the FFN), but 'cheap once built' is not")
print("'built'.  H6 is a negative control and it fires: **MoE does not prove as cheaply")
print("as dense.**  The batching that makes the dense scheme work -- one weight pass")
print("amortised over S tokens -- is what expert partitioning takes away.")

(OUT / "moe.json").write_text(json.dumps(dict(
    shape=dict(d=D, dff=DFF, ffn_params=FFN_PARAMS, s=S, topk=TOPK),
    runs=[{k: v for k, v in r.items() if k != "groups"} for r in rows],
    routing=dict(sel_cells=SEL_CELLS, max_cells=MAX_CELLS),
), indent=1))
print()
print(f"wrote {OUT / 'moe.json'}")
