"""B9 step 2, probe 3: the prefill probe (H4's second face).

Decode proves one token against C cached keys; prefill proves S tokens against S
keys with a causal mask, and **the mask is the term no AIR in this project has**.
`b65`'s grid runs its row axis over `(instance, key)` and its column axis over
tokens, exactly tiled, so a mask that is a function of `(instance, key, token)`
cannot be expressed by the tiling: the AIR proves the whole `S x C` rectangle and
a masked entry is an honest zero that still costs a committed cell.  At `C = S`
that is half the rectangle.

Measured: a prompt-processing pass at `C = S` on the 8B shape, one layer, on the
SIMD reference (`b9/out/prefill_s*.json`) -- the naive layout, because it is the
only one built.  Modelled: the two-region split (unmasked bulk + triangular tail)
a mask-aware layout would use, and what the corner costs in each.

Every "per token" below is per *model* token, i.e. the measured one-layer chunk
times L; the runs prove one layer.

    uv run experiments/b9/prefill.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"

NH, HD, L8B = 32, 128, 32
SS = (512, 1024, 2048)
GRID_TARGET = 64          # `--scores-per-row`, the tiling granularity of the grid

# B9 1.6, measured: the whole wall on the real 8B chunk (2.415 G cells, C=16K)
GPU_WALL_CELLS_S = 1.28e9
# B10 step 0, amendment to this probe: the *target* rate, i.e. what 53 tok/s at
# 27B/16K needs (B9 1.1).  Reading the overhead ratio at the measured rate mixes
# two different misses into one number -- the decode throughput gap, which H4's
# first face already counts, and prefill's own structure.  At the target rate the
# same cell counts give prefill's own ratio, which is what H4's second face is
# about: prefill k / decode k.
TARGET_CELLS_S = 15.95e9
# B1's forward pass under `b65-design` on this H100, no recorder: b8's ppl run
# does 40 chunks of 512 in 15.8 s (b8/out/ppl.log).  Slow -- it is a Python loop
# over layers -- but it is the same code every other number in this project uses.
B1_PREFILL_TOK_S = 40 * 512 / 15.8
# a production prefill: 2 N FLOPs per token at 40% MFU of an H100's ~400 TFLOP/s
MFU_FLOPS = 0.40 * 400e12
# B10 step 0: E6c's own baseline was 2 N / 1 POPS -- a full petaop of dense int8,
# no MFU discount -- which is x6.3 cheaper per token than the 40%-MFU one above.
# Half the distance between this probe's ratio and E6c's 50-150x band is that
# choice, and it belongs beside the ratio.
E6C_OPS = 1e15
# B9 step 0's denominator, printed by b2/cost_model.py
CM27 = {4096: 188.22e6, 16384: 300.97e6}
L27, NH27 = 62, 32
ALPHA_CM = 4.0            # the cost model's cells per (head, key, token)


def load(s):
    p = OUT / f"prefill_s{s}.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = json.loads(p.read_text())["result"]["point"]
    fc = d["family_cells"]
    return dict(s=s, accepts=d["accepts"], off_table=d["off_table"],
                fresh=d["cells"]["fresh"], pre=d["cells"]["preprocessed"],
                bands=fc["bands"], dequant=fc["dequant"], inter=fc["interaction"],
                walks=fc["walks"],
                leads=sorted({(l["chan"], l["key"], l["out"]) for l in fc["band_leads"]}),
                wall=d["phases"]["wall_secs"])


rows = [r for r in (load(s) for s in SS) if r]
missing = [s for s in SS if not load(s)]
if not rows:
    raise SystemExit("no b9/out/prefill_s*.json; run the prefill step on the box")


def g(x, unit=""):
    for sfx, d in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.2f} {sfx}{unit}"
    return f"{x:.2f} {unit}"


print("=" * 96)
print("B9 probe 3 -- the prefill probe (H4's second face), C = S on the 8B shape")
print("=" * 96)
print("the naive layout, as built: the whole S x C rectangle committed, half of it inert")
print("at C = S.  One layer of Llama-3.1-8B's shape (32 heads x 128), SIMD reference.")
print()
print(f"  {'S = C':>6} {'chunk (fresh)':>14} {'grid':>10} {'grid share':>11} "
      f"{'dequant':>10} {'lead rows':>11} {'accepts':>8}")
for r in rows:
    print(f"  {r['s']:>6} {g(r['fresh']):>14} {g(r['bands']):>10} "
          f"{r['bands'] / r['fresh'] * 100:>10.1f}% {g(r['dequant']):>10} "
          f"{str(r['leads']):>11} {str(r['accepts']):>8}")
for s in missing:
    print(f"  {s:>6} {'-- killed at 133.8 GB RSS: the box holds a 128 GiB host cap':>62}")

# --- the corner, in both layouts ---------------------------------------------
# The split: tokens in blocks of B.  For block i the keys below it are a full
# rectangle (no mask), the keys inside it are a B x B diagonal block proven whole
# (the corner), and the keys above it are never committed.
#     bulk = G (S^2/2 - S B/2)/S^2      tail = G S B / S^2      -> G (1/2 + B/2S)
print()
print("the causal corner, per (head, key, token), naive against the two-region split")
print(f"  split: tokens in blocks of B = {GRID_TARGET} (the grid's own tiling granularity);")
print("  keys below a block are a mask-free rectangle, keys inside it are a B x B block")
print("  proven whole, keys above it are never committed.")
print()
print(f"  {'S = C':>6} {'naive /rect':>12} {'naive /useful':>14} {'split /useful':>14} "
      f"{'split/naive':>12} {'grid saved':>11} {'of the chunk':>13}")
marg = None
if len(rows) >= 2:
    a, b = rows[-2], rows[-1]
    marg = ((b["bands"] - a["bands"])
            / (NH * (b["s"] ** 2 - a["s"] ** 2)))
for r in rows:
    S = r["s"]
    per_rect = r["bands"] / (NH * S * S)
    useful = NH * S * (S + 1) / 2
    B = min(GRID_TARGET, S)
    ratio = 0.5 + B / (2 * S)
    saved = r["bands"] * (1 - ratio)
    print(f"  {S:>6} {per_rect:>12.2f} {r['bands'] / useful:>14.2f} "
          f"{r['bands'] * ratio / useful:>14.2f} {ratio:>12.3f} {g(saved):>11} "
          f"{saved / r['fresh'] * 100:>12.1f}%")
if marg:
    print(f"\n  marginal grid cost, {rows[-2]['s']}->{rows[-1]['s']}: {marg:.2f} committed cells per "
          f"(head, key, token)")
    print(f"  (the cost model charges {ALPHA_CM:.2f}, which also counts the exp lookup's "
          f"interaction cells)")

# --- the finding --------------------------------------------------------------
big = rows[-1]
print()
print("**The corner is a correctness gap, not a cost gap, at these S.**")
print(f"  At C = S = {big['s']} the grid is {big['bands'] / big['fresh'] * 100:.1f}% of the chunk, so halving it saves")
print(f"  {(1 - (0.5 + GRID_TARGET / (2 * big['s']))) * big['bands'] / big['fresh'] * 100:.1f}%.  A prefill chunk is dominated by the per-token matmul work, not by")
print("  attention: the grid grows as S^2 and everything else as S, so the grid only")
print("  takes over at large S.  From the two measured points the crossover is")
if len(rows) >= 2:
    a, b = rows[-2], rows[-1]
    # fresh = F + q S + p S^2, with the S^2 term read off the grid family
    p = (b["bands"] - a["bands"]) / (b["s"] ** 2 - a["s"] ** 2)
    q = ((b["fresh"] - b["bands"]) - (a["fresh"] - a["bands"])) / (b["s"] - a["s"])
    fix = (a["fresh"] - a["bands"]) - q * a["s"]
    cross = (q + (q * q + 4 * p * fix) ** 0.5) / (2 * p)
    print(f"  S ~ {cross / 1000:.0f} k -- fresh = {g(fix)}+ {q:,.0f} S + {p:.1f} S^2, so the grid passes")
    print("  half the chunk only around the context the *decode* path already runs at.")

# --- the overhead ratio -------------------------------------------------------
print()
print("H4's second face: proven prefill per token against plain prefill")
print(f"  a chunk is one layer, so a model token is x{L8B} (Llama-3.1-8B)")
b1_ms = 1e3 / B1_PREFILL_TOK_S
prod8_ms = 2 * 8.03e9 / MFU_FLOPS * 1e3
print(f"  plain prefill baselines: B1's forward pass on this H100 "
      f"{B1_PREFILL_TOK_S:,.0f} tok/s = {b1_ms:.2f} ms/token;")
print(f"                           a production prefill at 40% MFU = {prod8_ms:.3f} ms/token")
print()
print(f"  {'S = C':>6} {'model cells/token':>18} {'prover ms/token':>16} "
      f"{'vs B1 fwd':>10} {'vs production':>14}")
for r in rows:
    per_model = r["fresh"] / r["s"] * L8B
    ms = per_model / GPU_WALL_CELLS_S * 1e3
    print(f"  {r['s']:>6} {g(per_model):>18} {ms:>15.1f}m {ms / b1_ms:>9.0f}x "
          f"{ms / prod8_ms:>13,.0f}x")

print()
print("extrapolated to 27B, on B9's own denominator (b2/cost_model.py):")
prod27_ms = 2 * 27e9 / MFU_FLOPS * 1e3
print(f"  a production 27B prefill at 40% MFU is {prod27_ms:.3f} ms/token")
print(f"  {'S = C':>6} {'naive cells/token':>18} {'split':>10} {'naive ms':>10} "
      f"{'split ms':>10} {'naive x':>9} {'split x':>9}")
for S, tot in sorted(CM27.items()):
    grid = ALPHA_CM * L27 * NH27 * S
    B = min(GRID_TARGET, S)
    split = tot - grid * (1 - (0.5 + B / (2 * S)))
    n_ms, s_ms = tot / GPU_WALL_CELLS_S * 1e3, split / GPU_WALL_CELLS_S * 1e3
    print(f"  {S:>6} {g(tot):>18} {g(split):>10} {n_ms:>9.0f}m {s_ms:>9.0f}m "
          f"{n_ms / prod27_ms:>8,.0f}x {s_ms / prod27_ms:>8,.0f}x")

# --- B10 step 0: the same counts at the target rate ---------------------------
print()
print("the same cell counts at the *target* rate (15.95 G cells/s = 53 tok/s at 27B/16K)")
print("  -- i.e. prefill's own overhead, with the decode throughput gap taken out.  H4's")
print("  first face already counts that gap; charging it twice is what puts this probe")
print("  three to five times above E6c's band.")
e6c27_ms = 2 * 27e9 / E6C_OPS * 1e3
print(f"  a production 27B prefill: {prod27_ms:.3f} ms/token at 40% MFU of 400 TFLOP/s,")
print(f"                            {e6c27_ms:.3f} ms/token at E6c's own 1 POPS "
      f"(x{prod27_ms / e6c27_ms:.1f} cheaper)")
print()
print(f"  {'S = C':>6} {'naive ms':>10} {'split ms':>10} {'naive x':>9} {'split x':>9} "
      f"{'naive x (E6c base)':>19}")
target27 = {}
for S, tot in sorted(CM27.items()):
    grid = ALPHA_CM * L27 * NH27 * S
    B = min(GRID_TARGET, S)
    split = tot - grid * (1 - (0.5 + B / (2 * S)))
    n_ms, s_ms = tot / TARGET_CELLS_S * 1e3, split / TARGET_CELLS_S * 1e3
    target27[S] = dict(naive_ms=n_ms, split_ms=s_ms, naive_x=n_ms / prod27_ms,
                       split_x=s_ms / prod27_ms, naive_x_e6c=n_ms / e6c27_ms)
    print(f"  {S:>6} {n_ms:>9.1f}m {s_ms:>9.1f}m {n_ms / prod27_ms:>8,.0f}x "
          f"{s_ms / prod27_ms:>8,.0f}x {n_ms / e6c27_ms:>18,.0f}x")
print()
print("  Read this way the structural claim is the opposite of a miss: at the rate the")
print("  scheme is aimed at, proven 27B prefill is x35 (4K) to x56 (16K) a production")
print("  prefill -- at or below the low end of E6c's 50-150 -- and E6c's own, cheaper")
print("  baseline puts it above the band again.  What survives either baseline is the")
print("  *ratio*: prefill k / decode k ~ 0.8-1.0, because prefill costs a full decode")
print("  token's trace per token.  That is E6c's flat-k assumption **confirmed**, not a")
print("  term E6c missed; the absolute overhead is the decode gap H4's first face")
print("  already reports, and it is reached iff the decode target is.")

print()
print("the verdict, in one paragraph.")
print("The corner's price is settled and it is small: at C = S the attention grid is")
print(f"{rows[-1]['bands'] / rows[-1]['fresh'] * 100:.1f}% of a prefill chunk, so the two-region split -- bulk rectangle plus a")
print(f"B x B diagonal block, {0.5 + GRID_TARGET / (2 * rows[-1]['s']):.3f} of the naive rectangle at S = {rows[-1]['s']} -- is worth")
print(f"{(1 - (0.5 + GRID_TARGET / (2 * rows[-1]['s']))) * rows[-1]['bands'] / rows[-1]['fresh'] * 100:.1f}% of the chunk.  The mask is a **soundness and completeness** gap (no AIR")
print("here has it, and 1.4 shows its absence already leaking into the fold's scale),")
print("not a throughput one, until S reaches the context where the grid dominates.")
print("**At the measured rate the overhead ratio does not land in E6c's 50-150x band.**")
print("Against B1's own forward pass on this box it is ~60x at 8B, inside the band;")
print("against a production prefill at 40% MFU it is ~450x at 8B and ~550-700x")
print("extrapolated to 27B, three to five times above it.  B10 step 0's amendment:")
print("that number charges the decode throughput gap (x12.5, H4's first face) a second")
print("time.  At the target rate the same counts give x35-x56 at 27B, at or below the")
print("band's low end, and E6c's own 1-POPS baseline moves it back above -- so the")
print("absolute ratio is a baseline choice.  The term that moved is not attention: it")
print("is that **prefill in this scheme costs a full decode-token's trace per token**.")
print("The weight walk")
print("amortises over the batch, which is what 'x100 on input' assumed; the per-token")
print("terms -- the dequant at 33% of the budget, the accumulators, the requantize -- do")
print("not amortise at all, and they are the budget.  Prefill is not cheap here because")
print("prefill was never the expensive half.")
print()
print("Rule 13 did not fire: every grid tiles exactly at both S (lead rows 0/0/0), so")
print("the inert-lead-row paths stay unexercised and no `nomask`/`noactive`/`noident`")
print("control was needed.  No GPU throughput is quoted for a C = S shape -- it has not")
print("been through the emit chain; the prover rate above is B9 1.6's measured whole")
print("wall on the real 8B decode chunk applied to a prefill cell count.")

OUT.mkdir(exist_ok=True)
(OUT / "prefill.json").write_text(json.dumps(dict(
    nh=NH, hd=HD, layers_8b=L8B, grid_target=GRID_TARGET,
    gpu_wall_cells_s=GPU_WALL_CELLS_S, b1_prefill_tok_s=B1_PREFILL_TOK_S,
    mfu_flops=MFU_FLOPS, cost_model_27b=CM27, missing=missing, rows=rows,
    # B10 step 0: the ratio reading -- the same counts at the rate the
    # denominator is defined at, and E6c's own baseline beside the MFU one
    target_cells_s=TARGET_CELLS_S, e6c_ops=E6C_OPS,
    e6c_prod27_ms=2 * 27e9 / E6C_OPS * 1e3, prod27_ms=prod27_ms,
    target_27b=target27), indent=1))
print()
print(f"wrote {OUT / 'prefill.json'}")
