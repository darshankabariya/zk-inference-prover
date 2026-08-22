# B9 (M3) — the denominator, the precondition, and the negative controls

The plan's three probes read on the post-B8 cost model and, where a real witness
is needed, on a real **Llama-3.1-8B** layer rather than B7b's synthetic 7B proxy —
plus the three things B8 left on the table that gate them. The running state is
`B9.md`; the plan's B9 section is the brief. One section per session, the state
block first, tables over prose.

```
b9/naive.py      probe 1: the naive baseline, closed on measured constants
b9/gpu/b9_patch.py  the two layer fixes, as an asserted patch for the derived trees
b9/gpu/b9_diag.py   `off_table` attributed to the table that lost the row
b9/gpu/b9_run.sh    the steps on the box
b9/out/          the runs: num*.json (the numerator), layers24/b9_layer_L*.json,
                 ref8b.json / prove8b.json (the real 8B layer), naive.json
b2/cost_model.py the denominator (step 0); b8/ the W8A16 rider (step 3)
```

## 1. The denominator, both layer-failure modes, and a real 8B layer (2026-08-22)

state of B9: **done** — step 0 (the denominator, the numerator's price in the AIR,
B7b's ladder re-read); step 1 (both failure modes fixed, **all 24 Qwen-0.5B layers
accept**, rungs 1–2 green, the master table priced per spread, **a real
Llama-3.1-8B layer at S=512 C=16K proved at a 2.415 G chunk**); step 2's naive
baseline; step 3's W8A16 rider (ppl + GPQA-D + MMLU-Pro + IFEval at 8B).

state of B9: **remaining** — step 2's MoE probe (H6) and prefill probe (H4's second
face); rung 3 on the new witnesses.

state of B9: **headline** — the plan's step-1 diagnosis was **half right**: the
master exponent is a band-wide constant where it must be layer-wide (12 of 24
layers), but the *second* mode is not the dequant's exponent underflow the brief
named — it is the **attention-output fold's quotient**, 128–370 against a 127
window, because the witness proves the causal *rectangle* while its output scale
was calibrated on the forward pass's masked output. Both fixes are witness-side:
no column moves, no AIR changes, no kernel regeneration. The denominator is now one
number — **300.97 M cells/token at 27B/16K, 15.95 G cells/s for 53 tok/s** — and
the ladder against it, **read on the real 8B chunk rather than B7b's synthetic
proxy**, is commit ×4.2, stwo ×7.2, whole wall ×12.5 = **4.24 tok/s, ×12.5 off
53**.

### 1.1 Step 0 — the denominator

B8's attribution retires B6.5's mixed dequant layout: power-of-two linear scales
are free on q/k/v/o at 0.5B (−0.43%) and cost **+3.67%** perplexity at 8B, while
`lm_head` goes the other way (+1.10% → +0.02%). So `b2/cost_model.py` now charges

* the **chain at `Round::Away`, 25.12 cells**, on all seven layer matmuls
  (3.841 M accumulators/token at 27B), and
* the **15.06-cell power-of-two dequant on `lm_head` alone** (262 k), which sits
  outside the layer proof anyway.

| | cells/token, 27B | for 53 tok/s |
|---|---|---|
| B6.5's mixed-layout planning range | 262.32 – 307.71 M | 13.90 – 16.31 G cells/s |
| **B8's layout, the default** | **300.97 M** (191.02 M at C=4K) | **15.95 G cells/s** |

The range is retired: there is one layout, not a band. The dequant is 33% of the
per-token budget, the attention grid 43%.

**B7b's ladder, re-read** (the three rates are constants of B7b's run at its
largest chunk, 2.217 G cells; what moved is the denominator):

| boundary | measured | vs 15.95 G | 27B/16K tok/s | vs 53 | verdict |
|---|---|---|---|---|---|
| commit path | 4.45 G cells/s | **×3.6** | 14.79 | ×3.6 | ≤5×, confirmed |
| stwo total | 2.89 G | ×5.5 | 9.60 | ×5.5 | at the 5× line |
| whole wall | 1.46 G | **×10.9** | **4.85** | **×11** | 5–100×, engineering |

Every rung moved by under 4% and **no verdict moves**. `B7A.md` §8 carries the
paragraph.

### 1.2 Step 0 — what a 9-bit numerator costs the AIR as built

B8 §2 item 2 read the 9-bit numerator as free at C=16K from
`n_bits + 7 + log2(C) ≤ 30`. That is the right inequality against the wrong
ceiling. B5's AIR range-checks V into [−128, 127] and the committed denominator
`Z` into `z_lo + z_hi = 22` bits (`attn_air.rs` `RowEval`), so the circuit
certifies `|A·V| < 128·2²² = 2²⁹` — exactly **B4's offset limb window**, which
splits 2³⁰ with a +2²⁹ offset and carries only 2²⁹ of magnitude.

| numerator bits | nmax | Z bits | z_hi | \|A·V\| < | B4 limb 2²⁹ | sign-bit 2³⁰ |
|---|---|---|---|---|---|---|
| 8 | 255 | 22 | 6 | 2²⁹ | fits | fits |
| 9 | 511 | 23 | 7 | 2³⁰ | **OVER** | fits |
| 10 | 1023 | 24 | 8 | 2³¹ | OVER | **OVER** |

Nine bits **leaves B4's window** and lands in M31's own, floor(p/2) = 2³⁰ − 1.
That window is reachable only because both readings commit an explicit sign bit
and read the magnitude — B6's chain, and the attention-output fold's `sg`
(`walk_air.rs`: `mag = acc − 2·acc·sg`). B4's limb decomposition alone would not
carry it. Ten bits is out of the field at C=16K.

In cells, measured (`compose --numbits 9 --z-hi-bits 7` against the same run at 8;
`b9/out/num*.json`): **no column moves** — 7,987 columns on the real 0.5B layer and
7,409 on the C=16K synthetic, identical in both pairs — and **0 cells per
accumulator**. The whole cost is shared-table rows at 5 fresh cells each (1 base
multiplicity + 4 interaction, plus 1 preprocessed): the denominator's high limb
2⁶ → 2⁷ rows and the fold's remainder table (`ao_rem_bits`, data-derived) one bit
wider.

| shape | 8-bit | 9-bit | Δ | `ao_rem_bits` |
|---|---|---|---|---|
| real Qwen-0.5B layer, C=1K | 281.04 M | 282.35 M | **+1.31 M (+0.47%)** | 18 → 19 |
| synthetic, C=16K | 1.014 G | 1.035 G | **+20.97 M (+2.07%)** | 22 → 23 |

The synthetic is the blind worst case — its numerators are dense, so `Z` sits at
`nmax·C`. On the real 8B witness below, **99.72% of the numerators are zero** and
`z_max` is 6,000 against a blind 4,177,920, so a real 16K witness sits at the
0.47% end. Both runs accept with `off_table` 0 and `ref_mismatch` 0. `b8/B8.md`
§2 item 2 is amended with this.

### 1.3 Step 1a — the master exponent is layer-wide

`Attn::build` took `F` = the largest `frac` over the band's *own* heads
(`attn_witness.rs`), so which table a band builds depends on how the heads were
split into bands, and `Layer::load`'s "every band's table is identical" holds only
where every band happens to contain a head at the layer-wide maximum. On
Qwen-0.5B that is 12 layers failing `bands disagree on the exp table`.

The forward pass already writes every head's exponent into the manifest
(`frac_per_head`, `frac_shared_per_head`), so the fix reads the layer-wide maximum
rather than deriving it, and hands it to every band; each then scales its shifts
up to it by a public power of two and reads one table of `d_bits + spread` rows —
`master_table_bits`, already in the manifest. **After the fix, `bands disagree` is
gone on all 24 layers**, and the exp table has headroom: L11's largest scaled key
is 2,438,788 against 4,194,304 rows.

Alone, this took the 24 layers from **4 accepts to 10**. It also revealed that
6 of the 12 "disagree" layers carried the second mode as well.

### 1.4 Step 1b — the second mode is the fold's quotient, not the dequant's exponent

The brief named the dequant's bf16 output exponent underflowing (`eo < 0`,
`layer.rs:756/774`). `off_table` is one counter over every lookup table a composed
layer builds, so it cannot tell the two apart; `b9/gpu/b9_diag.py` attributes it.
The answer, on the SIMD reference (the CUDA tree fills the grid on the device, so
host-side counters see nothing there):

```
L11  off_table 662 by table: {"ao_qnz": (662, 128, 370)}
L4   off_table  90 by table: {"ao_qnz":  (90, 128, 130)}
L5   off_table   1 by table: {"ao_qnz":   (1, 129, 129)}
```

Every lost row is **`ao_qnz`** — the attention-output fold's quotient leaving its
2⁷-row `(q, nz)` pair table, with values 128 to 370 against a 127 window. The
dequant's exponent never leaves its table, and neither does the exp table.

The cause is in the witness, not the gadget. `b65/extract.py` dumps the decode
rectangle — key `c` is visible to query `t` iff `c ≤ (C − S) + t`, so the AIR
proves the fully-visible `C − S` prefix and **the causal corner stays outside every
AIR in the project** (the extractor says so in its own comment). But it took `s_o`,
the fold's per-token output scale, from the *forward pass*, which attended to the
prefix **and** the corner. The corner's numerators enlarge `Z` more than they
enlarge `A·V` on some rows, so dropping them makes `A·V·s_v/Z` **larger** than the
scale was cut for, and `round(·/s_o)` runs past 127.

The fix is one scale: calibrate `s_o` on the rectangle the AIR proves. The
gadget's completeness precondition is `max |A·V·s_v/Z| ≤ 127·s_o` over the rows it
proves, and the extractor now asserts it (`fold_q_max ≤ 127`) instead of a prover
discovering it an hour later. It moved the scale on **22 of 24** Qwen layers
(0–481 tokens of 512 each) and on **74 of 512** tokens of the 8B layer. Nothing
else in the dump moves, and no column, constraint or kernel changes.

**This is a completeness precondition of the same kind as the exp table's**, and
it belongs beside it: a sub-computation proof has to be given scales cut for the
sub-computation.

### 1.5 All 24 layers, and what the shared master table costs

| | before B9 | after 1a | after 1a + 1b |
|---|---|---|---|
| accept | 4 | 10 | **24 of 24** |
| `bands disagree on the exp table` | 12 | 0 | 0 |
| `off_table` (all `ao_qnz`) | 8 | 14 | 0 |

Rung 2 holds on every one (`root_ok`, `stark_ok`, `sums_cancel` and all five
identities true, `ref_mismatch` 0). **Rung 1** holds across the whole spread range
— the SIMD reference's `sha256(proof_bytes)` equals the CUDA tree's on L4
(spread 0), L12 (1), L9 (2), L0 (3) and L1 (5), with the same fresh cell counts.

The master table's price is exactly linear in its rows, and the rows are
exponential in the spread. Across the 24 layers (same shape otherwise):

`fresh = 237.8 M + 5.010 cells per exp-table row`

| spread | table | rows | table cells | fresh | table share |
|---|---|---|---|---|---|
| 0 | 2²¹ | 2.1 M | 10.5 M | 248.3 M | 4.2% |
| 1 | 2²² | 4.2 M | 21.0 M | 258.8–259.4 M | 8.1% |
| 2 | 2²³ | 8.4 M | 42.0 M | 279.7–281.0 M | 15.0% |
| 2 (d_bits 22) | 2²⁴ | 16.8 M | 84.1 M | 321.7 M | 26.1% |
| 5 | 2²⁶ | 67.1 M | 336.2 M | 573.4–574.6 M | **58.6%** |

At spread 5 the shared table **more than doubles** the layer proof. (The plan
guessed ~7 cells per row and ~10% of a 2.2 G chunk at 2²⁵ rows; measured, 5.01
cells per row and 6.9% of the 2.415 G 8B chunk.)

**One shape per model is the wrong trade for this model.** `--ao-rem-bits` now
joins `--kmax` and `--table-bits`, so a model's layers *can* be pinned to one
shape — Qwen-0.5B's model-wide worst case is `--table-bits 26 --ao-rem-bits 18
--kmax 18`. But every layer then pays the deepest layer's table:

| | 24 layers |
|---|---|
| each at its own `master_table_bits` | **7.002 G cells** |
| all at the model-wide 2²⁶ | **13.777 G cells** (**×1.97**) |

The plan's "a few table rows against one emit chain" holds only where the spread
is uniform. Here the spread runs 0–5 and the table is exponential in it, so the
one-shape choice costs **+97% of the model's whole trace** and per-layer shapes
(a second emit chain) are the right call. A model whose spread is uniform — and
B8 measures Llama-3.1-8B at spread 1: 15, 2: 12, 3: 5, a range of 2 rather than 5
— pays ×2² = ×4 on the table alone, which on the 8B layer below is 6.9% → ~24%.

### 1.6 A real Llama-3.1-8B layer, S=512, C=16K

`b65/extract.py` under `b8/env.sh` (`--tokens 16896 --chunk 512 --layer 16`), the
`b65-design` spec, on the H100: a 4.1 GB witness in 70 s. The recorder now copies
each column to the host as it is kept — on the B8 path the forward pass is on the
GPU and every `.numpy()` below it failed otherwise.

| | Llama-3.1-8B layer 16 | B7b's synthetic 7B proxy |
|---|---|---|
| shape | 32 heads × 128, S=512, C=16384 | 28 × 128, S=512, C=16384 |
| bands | 16, 8, 8 | 16, 8, 4 |
| `frac_spread` / master table | **2** / 2²⁵ | — |
| `d_bits`, `kmax`, `ao_rem_bits` | 23, 19, 14 | 18, 17 |
| numerators that are zero | **99.72%** | — |
| `z_max` vs blind `nmax·C` | 6,000 vs 4,177,920 (**×700 tighter**) | — |
| **fresh cells** | **2.415 G** | 2.217 G |
| preprocessed | 449.5 M | — |
| columns | 6,866 | 7,987 (0.5B) |
| proof | 4.79 MiB | 3.79 MiB |

It **accepts**: `off_table` 0, `ref_mismatch` 0, `sums_cancel` true, `root_ok`,
`stark_ok`, `av_ok`, `den_link_ok`, all three `qk_ok`/`den_ok` and all seven
`walk_ok` true. SIMD reference digest `6ad759f57e4b97bc…`, 67.1 s of prove wall on
16 CPU threads at 84.9 GB peak RSS. The master table is 167.8 M cells, **6.9% of
the chunk**.

**On the GPU.** A shape B7 never proved runs the emit chain first (B7A rule):
`cudaemit` + `genemit` for the 8B IR shapes (42 s), into the fork and
`libstwo_cuda` rebuilt (69 s), the shape tables into the proving tree (3 min).
Then `compose-chunks --chunks 3`, read as B7b reads it — the median of chunks ≥ 1,
the per-process CUDA context (0.503 s) out of every number (rule 6):

| boundary | 8B, 2.415 G chunk | vs 15.95 G | 27B/16K tok/s | B7b's synthetic 7B, 2.217 G |
|---|---|---|---|---|
| commit path | **3.80 G cells/s** | ×4.2 | 12.63 | 4.45 G (×3.6) |
| stwo total | **2.22 G** | ×7.2 | 7.38 | 2.89 G (×5.5) |
| **whole wall** | **1.28 G** | **×12.5** | **4.24** | 1.46 G (×10.9) |

All three chunks give **one digest**, and it is the SIMD reference's
(`6ad759f57e4b97bc`) — **rung 1 holds on the real 8B layer**. 4a is **0 bytes of
trace columns in 0 columns**: every committed column is born on the device. The
witness H2D (2.49 GB in 0.352 s) is the only host traffic left, 19% of the wall.
Peak device memory **79.3 GB** — a 2.4 G chunk is at the 80 GB edge, which is the
practical ceiling on chunk size here. Every grid tiles exactly (lead rows
`0/0/0` on all three bands), so rule 13's `nomask`/`noactive`/`noident` paths
stay unexercised and no number here depends on them.

**The real chunk is 12–23% slower per cell than the synthetic proxy** on every
boundary, so the headline moves: **4.85 → 4.24 tok/s at 27B/16K, ×11 → ×12.5 off
53**. The verdicts are unchanged in kind — the commit path is still ≤5×
(**×4.2**, confirmed) and the whole wall still inside 5–100× (an engineering gap)
— with one exception worth naming: **stwo total is no longer at the 5× line**
(×5.5 on the proxy, **×7.2** on the real chunk). This row replaces the synthetic
proxy in every later tok/s statement.

### 1.7 Step 2, probe 1 — the naive baseline

Closed on measured constants, nothing run (`b9/naive.py`, `b9/out/naive.json`).
One committed cell per MAC wire, at 27B/C=16K:

| | naive | ours |
|---|---|---|
| cells/token | **35.33 G** | 300.97 M (**×117**) |
| trace/token | 141 GB | 1.20 GB |
| for 53 tok/s, written | **7.49 TB/s** (B2's 7.5, reproduced) | 63 GB/s |
| for 53 tok/s, touched on the prove path (60 B/cell) | 112 TB/s = **×34 an H100's HBM** | — |
| on B7b's measured commit path | ×421 short | ×3.6 |
| on B7b's whole wall | 0.041 tok/s | 4.85 |

B3's remark folded in: the weight pass is the term that collapses — the rᵀW walk
commits 0.090 fresh cells per weight against one per MAC, and in-circuit multiplies
drop 31× at S=32, 305× at S=512, 639–1365× at S=4096, the ratio growing with S
because only the weight pass is fixed. B4's: what is left is not free — reading an
accumulator as an integer rather than a field element is 25.12 cells with B6's
rounding, 33% of the per-token trace, for a statement a field-only argument gets
wrong by exactly p. **H-verdict: three to four orders of magnitude is not an
engineering gap.** Every other number in this block exists only because the
Freivalds identity took this one off the table first.

### 1.8 Step 3 — the W8A16 rider

the published system measured against a "public INT8 variant" that beat them on IFEval.
Static per-tensor W8A8 (`static`) is the weakest INT8 there is; the comparator a
deployment actually uses is **weight-only per-channel int8 with float activations**.
`Spec(linear="w8a16")` quantizes and dequantizes each weight once, at load, so the
matmul is the float one and no activation is ever quantized.

| spec | ppl | vs fp32 | vs `dynamic-int8` | GPQA-D | MMLU-Pro | IFEval prompt-strict | inst-strict |
|---|---|---|---|---|---|---|---|
| `fp32` | 4.408 | — | −4.78% | 33.8% | 38.9% | 74.5% | 81.9% |
| **`w8a16`** | **4.397** | **−0.26%** | **−5.04%** | **34.3%** | **39.2%** | 73.4% | 80.3% |
| `dynamic-int8` | 4.630 | +5.03% | — | 32.3% | 38.7% | 74.7% | 81.7% |
| `b6-design-p2o` | 4.796 | +8.81% | +3.60% | 32.8% | 38.5% | 75.6% | 82.5% |
| `static` | 65.990 | +1397% | +1325% | 25.8% | 12.2% | 28.1% | 39.4% |

W8A16 is **within noise of fp32 on every measure** — perplexity −0.26% (below the
0.6% noise floor), GPQA-D and MMLU-Pro half a point *above* it, IFEval a point
below — and it agrees with fp32 on 93.9% / 96.1% of the multiple-choice answers
against `dynamic-int8`'s 85.9% / 86.2%. Which is the point: **the honest statement is
that the scheme matches the strong INT8 baseline on the task evals and costs
+5.0% (attested) to +8.8% (provable) perplexity against it**, not that it beats a
strawman. Static W8A8's ×15 collapse says nothing about INT8; it says the
per-token dynamic scale is load-bearing, as B1 §3 said. `b8/README.md` §2 carries
the paragraph.

### 1.9 What this session changed, and where

| change | file | why |
|---|---|---|
| B8's layout as the default, `attn_window()`, the ladder re-read | `b2/cost_model.py` | step 0 |
| `Attn::build`/`load` take a `master` exponent | `b65/bench/src/attn_witness.rs` | 1a |
| `Layer::load` reads the layer-wide maximum | `b65/bench/src/layer.rs` | 1a |
| `--ao-rem-bits` | `b65/bench/src/main.rs` | one shape per model |
| the fold's `s_o` calibrated on the rectangle; `.cpu()` on the B8 path | `b65/extract.py` | 1b, 1.6 |
| the same edits as an asserted patch for the derived trees | `b9/gpu/b9_patch.py` | the box |
| `off_table` attributed to its table | `b9/gpu/b9_diag.py` | 1b |
| `Spec(linear="w8a16")` | `b1/model.py`, `b1/common.py` | step 3 |

B6.5's own results are unaffected: `master = None` is the old behaviour, and a
layer that accepted before contained a head at the layer-wide maximum in every
band, so its `F` — and its digest — are unchanged.
