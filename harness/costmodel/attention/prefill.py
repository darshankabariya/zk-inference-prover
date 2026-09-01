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
import sys

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"

NH, HD, L8B = 32, 128, 32
SS = (256, 512, 1024, 2048)
GRID_TARGET = 64          # `--scores-per-row`, the tiling granularity of the grid

# Every two-point reading this probe published names the pair it was taken on.
# Stream DD showed `bands` is exact at all four S under `64 S^2 + 12,416 S`, so
# `p = d(bands)/d(S^2)` is a SECANT and moves with the pair it is read between:
# 80.167 (256->512), 72.083 (512->1024), 68.042 (1024->2048), all reproduced by
# the identity `64 + 12,416/(S1+S2)`.  Reading it off the two largest rows would
# silently re-price a published line every time the S list grows, so the pair is
# pinned here and labelled for what it is: the SUPERSEDED single-parameter fit.
INCUMBENT_PAIR = (512, 1024)   # the retired `bands = p S^2` reading, and its 2.25
REMAINDER_PAIR = (512, 1024)   # `design.prefill_chunk_remainder`, frozen on this pair
HEADLINE_S = 1024              # the S every published sentence in this probe quotes

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


def artifact(s):
    """This probe's own C = S run at S, or stream DD's re-run of the same shape.

    The probe's S=2048 run was killed at 127.60 GiB RSS, 0.40 GiB under the box's
    128 GiB cap (`out/prefill_s2048.err`), so `out/prefill_s2048.json` is empty
    and stays empty.  DD re-ran the whole C = S sweep on a 150 GiB container and
    added S=256; its artifacts are `out/dd_s{S}.json` and they agree with this
    probe's to the cell at both S the two harnesses share.
    """
    for name in (f"prefill_s{s}.json", f"dd_s{s}.json"):
        q = OUT / name
        if q.exists() and q.stat().st_size:
            return q
    return None


def load(s):
    p = artifact(s)
    if p is None:
        return None
    d = json.loads(p.read_text())["result"]["point"]
    fc = d["family_cells"]
    return dict(s=s, src=p.name, accepts=d["accepts"], off_table=d["off_table"],
                fresh=d["cells"]["fresh"], pre=d["cells"]["preprocessed"],
                bands=fc["bands"], dequant=fc["dequant"], inter=fc["interaction"],
                walks=fc["walks"],
                leads=sorted({(l["chan"], l["key"], l["out"]) for l in fc["band_leads"]}),
                wall=d["phases"]["wall_secs"])


rows = [r for r in (load(s) for s in SS) if r]
by_s = {r["s"]: r for r in rows}
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
      f"{'dequant':>10} {'lead rows':>11} {'accepts':>8} {'artifact':>18}")
for r in rows:
    print(f"  {r['s']:>6} {g(r['fresh']):>14} {g(r['bands']):>10} "
          f"{r['bands'] / r['fresh'] * 100:>10.1f}% {g(r['dequant']):>10} "
          f"{str(r['leads']):>11} {str(r['accepts']):>8} {r['src']:>18}")
for s in missing:
    print(f"  {s:>6} {'-- no artifact':>62}")
# GNU `time` reports "Maximum resident set size" in **KiB**: this probe's own
# s2048 run's 133,799,684 is 127.60 GiB, not "133.8 GB".  It died 0.40 GiB under
# the box's 128 GiB cap, at composition-commit; DD's re-run on a 150 GiB
# container is the S=2048 row above.

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
if all(s in by_s for s in INCUMBENT_PAIR):
    a, b = (by_s[s] for s in INCUMBENT_PAIR)
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
    print(f"\n  marginal grid cost, {INCUMBENT_PAIR[0]}->{INCUMBENT_PAIR[1]} (the PUBLISHED pair, "
          f"pinned): {marg:.2f} committed cells")
    print("  per (head, key, token) -- a secant, not a coefficient; see the S^2 section below")
    print(f"  (the cost model charges {ALPHA_CM:.2f}, which also counts the exp lookup's "
          f"interaction cells)")

# --- the finding --------------------------------------------------------------
big = by_s[HEADLINE_S]
print()
print("**The corner is a correctness gap, not a cost gap, at these S.**")
print(f"  At C = S = {big['s']} the grid is {big['bands'] / big['fresh'] * 100:.1f}% of the chunk, so halving it saves")
print(f"  {(1 - (0.5 + GRID_TARGET / (2 * big['s']))) * big['bands'] / big['fresh'] * 100:.1f}%.  A prefill chunk is dominated by the per-token matmul work, not by")
print("  attention: the grid grows as S^2 and everything else as S, so the grid only")
print(f"  takes over at large S.  From the pinned {INCUMBENT_PAIR[0]}->{INCUMBENT_PAIR[1]} "
      "pair the crossover is")
if all(s in by_s for s in INCUMBENT_PAIR):
    a, b = (by_s[s] for s in INCUMBENT_PAIR)
    # fresh = F + q S + p S^2, with the S^2 term read off the grid family.
    # PINNED to INCUMBENT_PAIR: this is the superseded two-point reading, kept
    # so the correction below has the published number to be measured against.
    p = (b["bands"] - a["bands"]) / (b["s"] ** 2 - a["s"] ** 2)
    q = ((b["fresh"] - b["bands"]) - (a["fresh"] - a["bands"])) / (b["s"] - a["s"])
    fix = (a["fresh"] - a["bands"]) - q * a["s"]
    cross = (q + (q * q + 4 * p * fix) ** 0.5) / (2 * p)
    print(f"  S ~ {cross / 1000:.0f} k -- fresh = {g(fix)}+ {q:,.0f} S + {p:.1f} S^2, so the grid passes")
    print("  half the chunk only around the context the *decode* path already runs at.")
    print("  -- but see the next section: that `p` is 12.6% high and the crossover moves.")

# --- the S^2 coefficient: two models, only one of them a model ----------------
# Stream DD, 2026-09-01.  The fit above reads `p` off the bands family alone as
# d(bands)/d(S^2), i.e. it assumes `bands = p S^2` with no lower-order term.  The
# data already refutes that: `bands / (NH S^2)` is 2.758 at S=512 and 2.379 at
# S=1024, so a two-point marginal absorbs a linear term into the quadratic
# coefficient.  Both models hit both measured points, so fit quality cannot
# discriminate between them.  The mechanism can.
#
# Every coefficient below is an *identity of the layout*, read off
# `prover/src/attn_trace.rs:1167-1172` (the committed base columns) and `:1173-1181`
# (their LogUp/Horner interaction) with the layouts of `prover/src/layout.rs`.
# With `grid_target = 64`, `chan_target = hd = 128`, `rw = 16` and S, C, hd and
# `insts` all powers of two, every region tiles exactly (`lead_rows` 0/0/0 in both
# artifacts) and the products are exact -- nothing here is fitted:
#
#   region                           committed base           interaction
#   channel: keys     (`Cells::kv`)  pk*rows  = hd*NH*C       2*hd*NH*C + 384*C
#   channel: queries                 pq*rows  = hd*NH*S       2*hd*NH*S
#   grid: scores + numerators        2q*rows  = 2*NH*S*C      (12 + 64)*S*C
#   grid: V values    (`Cells::kv`)  pv*rows  = hd*NH*C       2*hd*NH*C
#   out (A.V)                        moved to the `attnout` region -- not in `bands`
#   per-token: max, denom, 2 limbs   4*rw*rrows = 4*NH*S      8.5*rw*rrows = 272*S
#
# `bands` in `family_cells` is BASE ONLY -- `compose.rs:3216` sums each band
# region's committed column domains -- which is why the fresh figures below are
# larger than the `grid share` column at the top of this probe.
if len(rows) < 2:
    raise SystemExit("the structural section needs both the S=512 and S=1024 artifacts")
NH_HD = NH * HD                      # 4096: one channel column per head, per token


def band_base(s, c, rows_region=True):
    """Committed base cells of the band family at (S, C).  Exact, not fitted."""
    return (2 * NH * s * c                          #   64 S C  scores + numerators
            + 2 * NH_HD * c                         # 8192 C    K and V = `Cells::kv`
            + NH_HD * s                             # 4096 S    queries
            + (4 * NH * s if rows_region else 0))   #  128 S    max, denom, two limbs


def band_inter(s, c, rows_region=True):
    """The same family's LogUp/Horner interaction cells.  Also exact."""
    return (76 * s * c                              # 12 Horner + 64 exp/numerator LogUp
            + (2 * NH_HD + 384 + 8192) * c          # 16,768 C: V range, chan Horner, K range
            + 2 * NH_HD * s                         #  8,192 S: query range LogUp
            + (272 * s if rows_region else 0))


def band_fresh(s, c, rows_region=True):
    return band_base(s, c, rows_region) + band_inter(s, c, rows_region)


def bands_of(s, c, corner):
    """A composed chunk's bands: the prefix bands, plus an S x S corner that
    carries no per-token region (one max and one denominator per token, not per
    band -- `attn_trace.rs`'s `with_rows = !a.corner`)."""
    return (band_base(s, c) + (band_base(s, s, False) if corner else 0),
            band_inter(s, c) + (band_inter(s, s, False) if corner else 0))


gamma, delta = 2 * NH, 2 * NH_HD + NH_HD + 4 * NH
print()
print("the S^2 coefficient: `bands = p S^2` against `bands = gamma S C + ... `")
print("  bands / (NH S^2) is "
      + " / ".join(f"{r['bands'] / (NH * r['s'] ** 2):.3f}" for r in rows)
      + " at S = " + " / ".join(str(r["s"]) for r in rows) + " -- not constant,")
print("  so `bands` is not purely quadratic and the marginal read above absorbs a linear term.")
print(f"  The secants themselves move the same way: "
      + " / ".join(
          f"{(by_s[y]['bands'] - by_s[x]['bands']) / (y * y - x * x):.3f}"
          for x, y in zip([r['s'] for r in rows], [r['s'] for r in rows][1:]))
      + f" for the {len(rows) - 1} adjacent pairs,")
print("  which is the identity 64 + 12,416/(S1+S2) -- so `p` is a property of the PAIR.")
print()
print(f"  incumbent   bands = {p:.3f} S^2        (SUPERSEDED; the S={INCUMBENT_PAIR[0]}->{INCUMBENT_PAIR[1]} "
      f"secant, pinned)")
print(f"  structural  bands = {gamma} S C + {2 * NH_HD:,} C + {NH_HD + 4 * NH:,} S")
print(f"              at C = S that is {gamma} S^2 + {delta:,} S, hitting all {len(rows)} points "
      "exactly with")
print(f"              gamma = {gamma} = 2*NH (alpha = {gamma / NH:.3f} base cells per head-key-token:")
print(f"              exactly two columns, the score and its numerator) and")
print(f"              delta = {delta:,} = 2*hd*NH ({2 * NH_HD:,}: K and V, `Cells::kv`)")
print(f"                                  + hd*NH ({NH_HD:,}: the query columns)")
print(f"                                  + 4*NH ({4 * NH}: the per-token region).")
print(f"  the incumbent p is {p / gamma - 1:+.1%} against gamma, because it charges delta*S to S^2.")
print()
print("  the control -- the closed form against every artifact that records a band count:")
CHECKS = [(f"b9 prefill  S=C={s:<6d}", artifact(s), s, s, False) for s in SS]
SEQ5 = HERE.parent / "b10/out/seq5"
CHECKS += [(f"seq5 8B L{L:<2d} S=512 C={c:<5d}", SEQ5 / f"seq_c{c}_L{L}.json", 512, c, True)
           for L in (0, 16, 31) for c in (4096, 16384)]
print(f"  {'artifact':25s} {'base predicted':>15} {'base measured':>15} "
      f"{'inter predicted':>16} {'inter measured':>15}")
band_checks = []
for name, path, s_, c_, corner in CHECKS:
    if path is None or not path.exists() or path.stat().st_size == 0:
        print(f"  {name:25s} {'-- artifact absent or empty, check skipped':>47}")
        continue
    fc = json.loads(path.read_text())["result"]["point"]["family_cells"]
    pb, pi = bands_of(s_, c_, corner)
    gb = fc["bands"]
    gi = fc.get("interaction_by_family", {}).get("bands", {}).get("cells")
    band_checks.append(dict(artifact=str(path), s=s_, c=c_, corner=corner, base_pred=pb,
                            base_got=gb, inter_pred=pi, inter_got=gi,
                            ok=pb == gb and (gi is None or pi == gi)))
    print(f"  {name:25s} {pb:>15,} {gb:>15,}{'' if pb == gb else ' MISMATCH'} "
          f"{pi:>16,} {(f'{gi:,}' if gi is not None else '(not recorded)'):>15}"
          f"{'' if gi is None or pi == gi else ' MISMATCH'}")
print(f"  {sum(c['ok'] for c in band_checks)}/{len(band_checks)} exact to the cell, across two "
      "harnesses, two cards, four S at C = S and three contexts.")
print()
print("  **The paper should use the structural model, and this is not a choice of fit.**")
print("  The linear term is the K/V commitment (`Cells::kv`: hd*NH per context key, twice)")
print("  plus the query columns and the per-token region.  Its C-proportional half and its")
print("  S-proportional half are indistinguishable at C = S -- and are separated exactly by")
print("  the composed chunks at C = 4K and 16K above, on the REAL 8B witness.  A model in S")
print("  alone cannot express chunked prefill at all, where every chunk has its own C.")

# --- the whole chunk, in (S, C) ----------------------------------------------
# `rest` is everything the band family does not cover: the mats, the dequant, the
# walks, the attnout region and the tables.  Every one of them is per-token or
# fixed -- none carries a C, because a chunk computes S new keys whatever its
# prefix -- so a linear fit of `rest` in S is structurally the right shape.  It
# is still a two-point fit, and that is where the remaining uncertainty lives.
rest = {r["s"]: r["fresh"] - band_fresh(r["s"], r["s"]) for r in rows}
_rlo, _rhi = REMAINDER_PAIR          # PINNED: `design.prefill_chunk_remainder`
q_star = (rest[_rhi] - rest[_rlo]) / (_rhi - _rlo)
f_star = rest[_rlo] - q_star * _rlo
BAND_SC, BAND_C, BAND_S = 140, 24960, 12688      # band FRESH cells: base + interaction


def fresh_of(s, c):
    """Fresh cells of one prefill layer chunk at (S, C)."""
    return f_star + q_star * s + band_fresh(s, c)


print()
print("the whole chunk, in (S, C) rather than in S alone")
print(f"  fresh(S, C) = {g(f_star)}+ {q_star:,.0f} S  +  band_fresh(S, C)")
print(f"  band_fresh  = {BAND_SC} S C + {BAND_C:,} C + {BAND_S:,} S   (base + interaction)")
for r in rows:
    got = fresh_of(r["s"], r["s"])
    tag = "fitted" if r["s"] in REMAINDER_PAIR else "HELD OUT"
    print(f"    S = C = {r['s']:>4}: model {got:>15,.0f}   measured {r['fresh']:>15,}   "
          f"{got / r['fresh'] - 1:+.4%}   {tag}")
print(f"  The linear fit is pinned to S = {_rlo} and {_rhi} (the pair "
      "`design.prefill_chunk_remainder` was")
print("  frozen on); the other two S are hold-outs and are NOT refitted.")
print(f"  the band family in FRESH cells is {BAND_SC / NH:.3f} per (head, key, token), against "
      f"{ALPHA_CM:.2f} in the")
print("  cost model and 5.898 for a composed chunk's context slope; the base-only "
      f"{gamma / NH:.3f} is")
print("  the figure comparable to B10's corner 2.750.  NOTE that the `grid share` column at")
print(f"  the top of this probe divides a BASE count by a FRESH one: fresh against fresh the")
print(f"  band family is {band_fresh(HEADLINE_S, HEADLINE_S) / by_s[HEADLINE_S]['fresh'] * 100:.1f}% "
      f"of the S={HEADLINE_S} chunk, not "
      f"{by_s[HEADLINE_S]['bands'] / by_s[HEADLINE_S]['fresh'] * 100:.1f}%.")


def cross_of(a2, a1, a0):
    return (-a1 + (a1 * a1 - 4 * a2 * a0) ** 0.5) / (2 * a2)


x_str = cross_of(gamma, delta - q, -fix)                          # base-only, like for like
x_fresh = cross_of(BAND_SC, BAND_S + BAND_C - q_star, -f_star)    # fresh against fresh
print()
print("the crossover -- where the attention family passes half the chunk")
print(f"  incumbent   p S^2 = rest                     S ~ {cross:>8,.0f}")
print(f"  structural  gamma S^2 + delta S = rest       S ~ {x_str:>8,.0f}   {x_str / cross - 1:+.1%}")
print(f"  fresh against fresh (the one to quote)       S ~ {x_fresh:>8,.0f}   {x_fresh / cross - 1:+.1%}")
print("  The first two compare a BASE band count against a fresh remainder, which is the top")
print("  table's own inconsistency carried forward; the third counts the family's own")
print("  interaction and lands well inside the context the decode path already runs at.")

# --- chunking, and E11's multiple ---------------------------------------------
# A prompt of N tokens proven in n = N/S chunks: chunk i covers S new tokens
# against C = (i+1) S keys, its own block plus every earlier one.  The published
# figure counts only the fixed term paid n times; the (S, C) model also sees the
# two terms an S-only model cannot -- the prefix K/V RE-COMMITTED by every later
# chunk, and the half of the causal rectangle the split does NOT commit.
E11_S = 1024
print()
print(f"chunking a prompt of N tokens into blocks of S = {E11_S:,}, per layer chunk")
print(f"  {'N':>6} {'one shot':>10} {'chunked':>10} {'fixed x n':>10} {'prefix K/V':>11} "
      f"{'grid saved':>11} {'net':>8} {'published':>10}")
chunk_rows = {}
for n_tok, pub in ((4096, 13.7), (16384, 11.6)):
    n = n_tok // E11_S
    one = fresh_of(n_tok, n_tok)
    tot = sum(fresh_of(E11_S, (i + 1) * E11_S) for i in range(n))
    tri = n * (n + 1) / 2
    fixed, kv = (n - 1) * f_star, BAND_C * E11_S * tri - BAND_C * n_tok
    grid = BAND_SC * E11_S * E11_S * tri - BAND_SC * n_tok * n_tok
    chunk_rows[n_tok] = dict(chunks=n, one_shot=one, chunked=tot, fixed_pct=fixed / one * 100,
                             kv_pct=kv / one * 100, grid_pct=grid / one * 100,
                             net_pct=(tot / one - 1) * 100, published_pct=pub)
    print(f"  {n_tok:>6} {g(one):>10} {g(tot):>10} {fixed / one * 100:>9.1f}% "
          f"{kv / one * 100:>10.1f}% {grid / one * 100:>10.1f}% {(tot / one - 1) * 100:>7.1f}% "
          f"{pub:>9.1f}%")
print("  The published +13.7 / +11.6 is the `fixed x n` column alone, and it reproduces to")
print("  the decimal under the incumbent fit: it is the right term and the wrong total.")
print("  Chunking also re-commits the prefix K/V once per later chunk, and it SAVES the half")
print("  of the causal rectangle a one-shot proof commits inert.  At 16K the saving wins.")

try:
    sys.path.insert(0, str(HERE.parent))
    from paper.constants import value as frozen
    DECODE = {4096: frozen("cells.c4096.fresh_33chunks"),
              16384: frozen("cells.c16384.fresh_33chunks")}
except Exception as exc:                     # experiments/paper is optional here
    DECODE, e11 = {}, {}
    print(f"\n  (E11 multiple skipped -- {type(exc).__name__}: {exc})")
e11 = {}
if DECODE:
    print()
    print("E11's cells multiple: proving the prompt too, against seq5's MEASURED 512-token")
    print("decode sequence (H200, 33 chunks, fresh cells).  32 layer chunks; head excluded.")
    print(f"  {'C = N':>6} {'prompt cells':>13} {'x decode':>9} {'prompt+output':>14} "
          f"{'x decode':>9} {'square-chunk read':>18}")
    for n_tok in sorted(DECODE):
        n = n_tok // E11_S
        prompt = sum(fresh_of(E11_S, (i + 1) * E11_S) for i in range(n)) * L8B
        square = fresh_of(E11_S, E11_S) * n * L8B      # what the S-only fit could express
        e11[n_tok] = dict(prompt_cells=prompt, x_prompt=prompt / DECODE[n_tok],
                          x_total=(prompt + DECODE[n_tok]) / DECODE[n_tok],
                          x_square=square / DECODE[n_tok], decode_cells=DECODE[n_tok])
        print(f"  {n_tok:>6} {g(prompt):>13} {prompt / DECODE[n_tok]:>8.2f}x "
              f"{g(prompt + DECODE[n_tok]):>14} "
              f"{(prompt + DECODE[n_tok]) / DECODE[n_tok]:>8.2f}x {square / DECODE[n_tok]:>17.2f}x")
    print("  The published ~x4.5-5.5 at C=4K is the last column, and that column plus one: it")
    print("  prices every prompt chunk as its own S = C = 1024 square, which is not a proof of")
    print("  the sequence -- the fourth chunk must attend over all 4,096 keys, not its own")
    print("  1,024.  With the real geometry it is x5.2 (prompt) / x6.2 (prompt+output) at 4K.")
    print("  A head chunk per 512 prompt positions is NOT counted here, and was not counted in")
    print("  the published band either; at 4K it would add ~8x `cells.c4096.fresh_head`.")

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
print(f"{big['bands'] / big['fresh'] * 100:.1f}% of a prefill chunk, so the two-region split -- bulk rectangle plus a")
print(f"B x B diagonal block, {0.5 + GRID_TARGET / (2 * big['s']):.3f} of the naive rectangle at S = {big['s']} -- is worth")
print(f"{(1 - (0.5 + GRID_TARGET / (2 * big['s']))) * big['bands'] / big['fresh'] * 100:.1f}% of the chunk.  The mask is a **soundness and completeness** gap (no AIR")
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
    target_27b=target27,
    # stream DD, 2026-09-01: the structural band model and what it moves
    band_model=dict(
        base_form="2*NH*S*C + 2*hd*NH*C + hd*NH*S + 4*NH*S",
        inter_form="76*S*C + 16768*C + 8464*S",
        s_c_base=gamma, s_c_fresh=BAND_SC, c_base=2 * NH_HD, s_base=NH_HD + 4 * NH,
        alpha_base=gamma / NH, alpha_fresh=BAND_SC / NH,
        delta_at_c_eq_s=delta, incumbent_p=p, incumbent_pair=list(INCUMBENT_PAIR),
        checks=band_checks),
    chunk_model=dict(fixed=f_star, per_token=q_star, band_sc=BAND_SC,
                     band_c=BAND_C, band_s=BAND_S, fitted_pair=list(REMAINDER_PAIR)),
    crossover=dict(incumbent=cross, structural_base=x_str, structural_fresh=x_fresh),
    chunking=chunk_rows, e11=e11, e11_chunk_s=E11_S), indent=1))
print()
print(f"wrote {OUT / 'prefill.json'}")
