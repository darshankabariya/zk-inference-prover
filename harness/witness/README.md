# B1 (M0) — Witness generator: deterministic quantized inference

Produces the exact computation a prover would have to attest to for
`Qwen/Qwen2.5-0.5B-Instruct`, and measures what that computation costs in quality.

Reproduce with `sh b1/run_all.sh` (~35 min on 4 CPU cores; peak ~4 GB RAM).

```
b1/quant.py        bf16 rounding, int8 quantization, exact integer GEMM
b1/model.py        the explicit forward pass (no transformers at inference time)
b1/common.py       determinism setup, corpus slices, the spec table
b1/selftest.py     faithfulness, H2 (M31 embedding), determinism
b1/probe_order.py  §5 follow-up: divergence origin + schedule stress test
b1/ppl.py          quality sweep over 20,480 tokens
b1/outliers.py     why static per-tensor W8A8 loses
b1/gen_witness.py  the witness dump + manifest ("circuit config")
b1/out/            results (JSON, committed) and the witness (npz, gitignored)
```

## What ran

* Model `Qwen/Qwen2.5-0.5B-Instruct`: L=24, d=896, d_ff=4864, 14 heads / 2 KV
  heads, head_dim=64, RMSNorm + SwiGLU + RoPE + GQA, tied embeddings — the same
  op set as Gemma, at a size this box can run.
* Corpus: `experiments/text.txt` (Pride and Prejudice), 40 chunks × 512 = 20,480
  tokens (20,440 next-token predictions) for evaluation, and a disjoint
  2,048-token slice for static calibration. Not held out from the model's
  training data — only relative comparisons between schemes are meaningful here.
* torch 2.13 CPU, 4 threads, `torch.use_deterministic_algorithms(True)`, seeds fixed.

## The attested computation

The forward pass is written out op by op so nothing hides in a library fusion.
The numeric contract (`quant.py`):

* **Weight matmuls are int8**: per-output-channel symmetric weight scales, per-token
  dynamic symmetric activation scales, round-half-away-from-zero, clamp to ±127.
  Scales are themselves bf16 values, rounded **up** so the clamp cannot bind. The
  accumulator is an exact int32; the dequant `acc·s_x·s_w (+bias)` is evaluated in
  fp32 and rounded once to bf16.
* **Everything else is bf16**: every value flowing between ops is a bf16 value and
  each elementwise op is correctly rounded — i.e. exactly one 2^16-entry LogUp
  table (`rsqrt`, `silu`, `exp`, `reciprocal`) or one small-table FP op.
* **Reductions** (RMSNorm mean, softmax denominator) accumulate in fp32 and round
  once — the v1 soundness gap B4 names. `reduce="bf16"` swaps in a balanced tree of
  correctly-rounded bf16 adds, which a v2 FP-add gadget could prove directly. The
  two bf16 attention matmuls always accumulate in fp32.
* **Fixed evaluation order**: sequential KV append, row max subtracted before `exp`,
  one reciprocal per softmax row rather than a division per element, RoPE tables
  precomputed in fp64 and rounded once to bf16 as preprocessed constants.

Two deliberate deviations from the plan's spec: activation *scales* are bf16 rather
than fp32 (one value per token per matrix, and it keeps the dequant inside the FP
gadget), and attention's Q·Kᵀ / A·V stay bf16 in the headline spec — quantizing
them is B5's job, and `qk-int8` below is the first half of that probe.

## Results

### 1. The reimplementation is the model

Against the HuggingFace forward on the same 128 tokens, in fp32:
max |Δlogit| = **4.5e-5** (1.8e-6 of the logit range), argmax agreement **100%**.
The numbers below therefore measure quantization, not a broken model.

### 2. Quality (H1) — 20,440 predictions

| spec | ppl | vs fp32 |
|---|---|---|
| `fp32` | 21.855 | -- |
| `dynamic-int8` | 22.835 | +4.49% |
| `static` | 110.777 | +406.88% |
| `int8-only` | 22.629 | +3.54% |
| `bf16-only` | 21.914 | +0.27% |
| `fp32-logits` | 22.828 | +4.45% |
| `keep-head` | 22.603 | +3.42% |
| `keep-down` | 22.507 | +2.98% |
| `keep-o` | 22.756 | +4.12% |
| `keep-qkv` | 22.624 | +3.52% |
| `keep-gate-up` | 22.552 | +3.19% |
| `bf16-reduce` | 22.758 | +4.13% |
| `bf16-norm` | 22.782 | +4.24% |
| `bf16-softmax` | 22.678 | +3.77% |
| `qk-int8` | 23.879 | +9.26% |

`dynamic-int8` is the full contract; `static` is the classic W8A8 PTQ baseline (same
int8 weights, one calibrated activation scale per tensor); the rest hold one choice
out of int8/bf16 at a time.

* **The attested spec costs +4.5% perplexity** — it misses B1's +2% bar by ~2.2×.
* **The standard baseline is 91× worse.** Static per-tensor W8A8 costs **+407%**.
  This is the leg of the theory that was actually in question, and it is confirmed
  emphatically at 0.5B, reproducing `quant_ppl.py`'s direction at 6× the scale.
* **The bf16 half of the theory is nearly free**: +0.27% for putting every
  non-matmul value and every nonlinearity in bf16. Rounding the *logits* to bf16 is
  a non-issue (`fp32-logits` differs from `dynamic-int8` by 0.04%). The loss is the
  int8 matmuls (`int8-only`, +3.54%), and the two effects are roughly additive.
* **The loss is spread across matmuls, not concentrated.** Holding one kind out of
  int8 recovers, of the 4.49 points: `down` 1.51, `gate`+`up` 1.30, `lm_head` 1.07,
  `q`/`k`/`v` 0.97, `o` 0.37. No single matrix is the culprit, so there is no cheap
  "keep one matrix in bf16" fix — though `down_proj` is the worst offender, which
  §3 explains.

### 3. Why static loses — and which matmul is hardest

For each linear, `outliers.py` compares the calibrated per-tensor range against the
range an individual token actually occupies:

| matmul | grid use | bits lost | clipped tokens | max/median |
|---|---|---|---|---|
| `q`/`k`/`v` | 45.2% | 1.1 | 0.00% | 2.2 |
| `o` | 42.9% | 1.2 | 0.00% | 2.2 |
| `gate`/`up` | 53.0% | 0.9 | 0.00% | 2.0 |
| `down` | **19.6%** | **2.4** | 0.00% | 5.4 |
| `lm_head` | 72.0% | 0.5 | 0.05% | 1.4 |

Static W8A8 does **not** fail by clipping here — calibration captured the range
(0.00% of tokens exceed it). It fails by *wasting resolution*: the one scale is
sized for the loudest token, so a typical token reaches only 20–72% of the ±127
grid and is effectively quantized to 5.6–7.5 bits. `down_proj`, whose input is the
SwiGLU product, is the worst (5.4× spread between its loudest and median token) —
and independently the largest single contributor to the int8 loss in §2. Two
different measurements agreeing on the same matrix is the useful part.

### 4. int8 embeds exactly in M31 (H2)

On real activations, at every reduction width the model uses, BLAS == int64 == M31
field arithmetic (products reduced mod p before summing), bit for bit — all eight
matmul kinds pass. Observed accumulator maxima over 128 tokens:

| matmul | K | max abs accumulator | headroom to 2^30−1 |
|---|---|---|---|
| `q` | 896 | 174,814 | ×6,142 |
| `k` | 896 | 174,058 | ×6,169 |
| `v` | 896 | 115,840 | ×9,269 |
| `o` | 896 | 213,819 | ×5,022 |
| `gate` | 896 | 234,612 | ×4,577 |
| `up` | 896 | 189,928 | ×5,653 |
| `down` | 4864 | 342,986 | ×3,131 |
| `lm_head` | 896 | 113,801 | ×9,435 |

`budget.py` gives the worst-case bound (127²·K fits a signed M31 element for
K ≤ 66,572); B1 adds the empirical half on a real model. The widest matmul here
(K=4864) has 13.7× headroom even adversarially, and the accumulators actually
observed sit ~3,100× below the field limit. **H2 holds with no decomposition — one
field element per dot product, and the margin is not tight.**

### 5. Determinism, and what it takes

Regenerating the witness in a fresh process reproduces it bit for bit
(digest `af039357…`, **PASS**) — B1's second acceptance criterion.

The stronger check — batched vs token-at-a-time evaluation of the same sequence —
exposes something worth carrying forward:

| spec | residual stream L0 → L23 | logits | argmax | bit-identical (this slice) |
|---|---|---|---|---|
| `fp32` | 1.95e-3 → 1.56e-2 | 2.90e-4 | 100% | no |
| `dynamic-int8` | 0 → 3.00 | 2.25 | 96.9% | no |
| `bf16-reduce` | 0 → 0 | 0 | 100% | yes — **false positive, see below** |
| `qk-int8` | 0 → 0 | 0 | 100% | yes (empirically) |

The int8 matmuls cannot differ — integer accumulation is order-independent. The
origin is the fp32-accumulated Q·Kᵀ: BLAS contracts it in a different order for a
1-row GEMV than for a 32-row GEMM, and under bf16 storage a sub-ulp order
difference occasionally crosses a rounding boundary and compounds through 24
layers (exactly 0 at layer 0, 3.0 by layer 23, flipping 1 of 32 predicted
tokens). `probe_order.py` localizes the first divergence to exactly that op
(token 7, the layer-15 scores) and adds the test power this single-slice table
lacks — 3 fresh 48-token slices × 3 schedules (full batch, two half-batches,
token-at-a-time):

* **`bf16-reduce`'s "yes" above is a false positive.** On fresh slices it
  diverges every time (logit diffs up to 2.6). Pinning the RMSNorm/softmax
  reductions does nothing for Q·Kᵀ; the selftest slice passed by luck, because
  divergence needs a data-dependent rounding-boundary crossing.
* `qk-int8` (alone or combined with `bf16-reduce`) is bit-identical on every
  slice and schedule tried — but *empirically*, not by construction: A·V and the
  fp32 reductions are still unpinned BLAS/torch ops that happen to be
  schedule-invariant in torch-CPU at 4 threads. A GPU runtime (split-K kernels,
  atomics) voids that.
* Two half-batches always match the full batch; only the single-token path
  diverges. The schedule mismatch is literally decode vs prefill.

So a prover and a runtime that batch differently do not compute the same thing
until **every** non-integer accumulation — attention matmuls included — is made
exact or given a pinned, provable order by construction. "Bit-identical on a
test run" is a necessary check, never the guarantee. That is a constraint on B5
and B6, not a detail.

### 6. The witness

64 decode steps (32 prompt + 32 greedy-decoded), one token at a time:
**1013 columns, 64.2M values, 163 MiB — 1,003k values and 2,609 KiB per token.**

| kind | values/token | share |
|---|---|---|
| int32 accumulators | 456.1k | 45.5% |
| bf16 lookup in/out | 256.1k | 25.5% |
| int8 activations | 246.7k | 24.6% |
| attention (scores, probs) | 21.8k | 2.2% |
| residual stream | 21.5k | 2.1% |
| activation scales | 0.2k | 0.03% |

Attention looks negligible only because the context is ≤64 here. It costs 4 values
per head per key (scores, exp in, exp out, probs; a prover would commit 3 and derive
the shifted exp input) — i.e. α·L·nh·C with **α = 3–4**, squarely inside B5's
predicted [2, 6]. Everything else is flat at ~958k values/token, so at C=8192
attention alone would be ~11M values/token, **92% of the witness**. B5's
"attention dominates at long context" prediction is already visible in the shape.

`out/manifest.json` is the circuit config: dims, the full op list with the witness
values each op contributes per token, the quantization rules, the accumulator bound
and the content digest. The dump is deliberately redundant — each op records its own
inputs and outputs so a gadget can be tested standalone (q/k/v each store their own
copy of the same `qx`, ~4.5% of the total).

## Acceptance verdict

| criterion | result | verdict |
|---|---|---|
| quantized vs fp32 within +2% ppl on ~20k tokens | +4.49% | **missed** (2.2× budget) |
| clearly beats static per-tensor W8A8 | +4.49% vs +406.88% | **met**, by 91× |
| witness regeneration bit-identical | digest matches | **met** |

**Two of three.** The quality bar is missed at 0.5B. That does not falsify H1 — the
plan's falsifier is "≫5% at 7–9B scale", small models are the hard case for
post-training quantization, and a perplexity percentage is a harsher metric than the
GPQA-D/IFEval scores the published system reports. But it is not a confirmation either. B8
remains the real test, and it should be run knowing that the damage at 0.5B is
+4.5%, spread evenly across matmuls, and driven by resolution loss on the widest
activations rather than by clipping.

## What this hands to later blocks

* **B3** — real shapes and real data. `manifest.json` gives the FFN pair
  (896→4864, 4864→896) the plan names; `witness.npz` supplies genuine int8 qX/qY
  and exact int32 accumulators to feed the Freivalds gadget instead of random
  tensors, so its corruption tests run against the real distribution.
* **B4** — bf16 reductions in a *balanced tree* cost nothing: +4.13% (both), +4.24%
  (RMSNorm only), +3.77% (softmax only), all within ±0.7 points of the +4.49%
  fp32-accumulation spec and all slightly below it. So the v2 FP-add gadget is
  quality-neutral, and B4 v1's "commit the reduction sum" shortcut buys only gadget
  simplicity, not accuracy. The witness already carries the four unary tables B4
  needs (`rsqrt`, `silu`, `exp`, `reciprocal`) with inputs and outputs.
* **B5** — two warnings. Quantizing Q·Kᵀ to int8 costs **+4.8 points on top** of the
  attested spec (9.26% total), the most expensive single choice measured here — it
  buys schedule-invariance for Q·Kᵀ (§5) at a real quality price. The untested
  middle option is a pinned-order bf16 tree for the attention accumulations
  themselves (B1 measured trees only for the norm/softmax reductions, where they
  were quality-neutral). And A·V has a structural
  problem B1 could not paper over: its reduction runs along the *sequence* axis, so
  per-token V scales do not factor out of the sum the way per-token Q and K scales
  do out of Q·Kᵀ. B5 must choose per-channel-over-context or per-block V scales, and
  neither commits cleanly as the KV cache grows.
* **B6** — every non-integer accumulation (attention matmuls included) must be
  exact or pinned by construction, or prover and runtime disagree; decode-vs-prefill
  equivalence on ≥3 disjoint slices is the acceptance test, with `probe_order.py`
  as the template (§5).
* **B8** — the scheme to replicate at 8–9B, and the numbers to beat: +4.5% ppl at
  0.5B, with static W8A8 at +407% as the control.
