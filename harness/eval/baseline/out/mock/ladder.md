### The overhead ladder, by boundary

H200 reference rate: $4.59/GPU-h (from_freeze=True)


**C=4096**
- unproved single-stream, `mock`: **820.57 tok/s** (batch 1, bfloat16, x86_64, MOCK/none) — NOT-LIKE-FOR-LIKE ← headline
- unproved E11 all-positions: **7369.1 pos/s** (mock, 4096+512 positions)
- unproved batched peak: **3002.6 tok/s aggregate** at batch 32 (mock)

| boundary | proved tok/s | scope | latency × | E11 × | proved $/Mtok | batched $/Mtok | $/token × |
|---|---|---|---|---|---|---|---|
| commit | 30.306 | 33 chunks (head included) | ×27.1 | ×243.2 | $42.07 | $0.4246 | ×99 |
| stwo total | 16.566 | 33 chunks (head included) | ×49.5 | ×444.8 | $76.96 | $0.4246 | ×181 |
| whole wall | 4.991 | 33 chunks (head included) | ×164.4 | ×1476.5 | $255.46 | $0.4246 | ×602 |
| loop_secs | 8.277 | 32 layer chunks (HEADLESS) | ×99.1 | ×890.3 | $154.05 | $0.4246 | ×363 |

**C=16384**
- unproved single-stream, `mock`: **545.90 tok/s** (batch 1, bfloat16, x86_64, MOCK/none) — NOT-LIKE-FOR-LIKE ← headline
- unproved E11 all-positions: **17973.5 pos/s** (mock, 16384+512 positions)
- unproved batched peak: **1587.8 tok/s aggregate** at batch 8 (mock)

| boundary | proved tok/s | scope | latency × | E11 × | proved $/Mtok | batched $/Mtok | $/token × |
|---|---|---|---|---|---|---|---|
| commit | 20.856 | 33 chunks (head included) | ×26.2 | ×861.8 | $61.13 | $0.8030 | ×76 |
| stwo total | 12.289 | 33 chunks (head included) | ×44.4 | ×1462.6 | $103.75 | $0.8030 | ×129 |
| whole wall | 4.192 | 33 chunks (head included) | ×130.2 | ×4287.6 | $304.15 | $0.8030 | ×379 |
| loop_secs | null (loop.h200.c16384) | 32 layer chunks (HEADLESS) | null (loop.h200.c16384) | null (loop.h200.c16384) | null | null | null (loop.h200.c16384) |

**The E11 column rests on an assumption:** REGIME 1, the same convention stream E7 used in notes/sufficiency.md: every proved position is charged one measured C=4096 decoded token, so the proved rate in positions/second equals the frozen tok/s row unchanged. This is CONSERVATIVE for E11 -- a prompt position inside a causal rectangle attends to ~N/2 keys, not a full C band -- and it is an assumption, not a measurement: no E11 proved row exists. The unproved side needs no assumption; it is measured directly as (prompt + output) / wall of the concatenated pass.

**What neither side counts:**
- the witness extraction. The proved side's `loop_secs` reads a witness that a prior extraction step produced; that step is in no boundary and has no unproved counterpart.
- model / weight load, and on the proved side the kernel-store emit and the prover binary's own start-up.
- verification. Every number here is prover-side.
- the inference pass itself, on the PROVED side: none of the four boundaries contains one. So an overhead multiple U/P is prover-seconds over inference-seconds -- precisely what fn. 170 computed (zkLLM's 803 s over 668 tok/s). A deployment that must infer AND prove costs (1 + U/P) x the unproved time; at x25-40 that unit is immaterial, and it is stated anyway.

**Caveats the harness raised itself:**
- C=4096: the single-stream run is NOT-LIKE-FOR-LIKE (model: 'MOCK/none@main' vs proved 'Llama-3.1-8B-Instruct'; tokenizer: 'MOCK/none (_Tok, vocab 32000)' vs proved 'the tokenizer of Llama-3.1-8B-Instruct'; card: 'x86_64 x4' vs proved 'H200'; synthetic: 'SYNTHETIC (mock backend)' vs proved 'a real measurement'). The latency column is NOT publishable.
- C=4096: backend(s) ['hf', 'vllm'] missing. The belief being corrected quotes an OPTIMIZED batched figure, so a hf-only baseline flatters us; run the vllm arm before publishing.
- C=16384: the single-stream run is NOT-LIKE-FOR-LIKE (model: 'MOCK/none@main' vs proved 'Llama-3.1-8B-Instruct'; tokenizer: 'MOCK/none (_Tok, vocab 32000)' vs proved 'the tokenizer of Llama-3.1-8B-Instruct'; card: 'x86_64 x4' vs proved 'H200'; synthetic: 'SYNTHETIC (mock backend)' vs proved 'a real measurement'). The latency column is NOT publishable.
- C=16384: backend(s) ['hf', 'vllm'] missing. The belief being corrected quotes an OPTIMIZED batched figure, so a hf-only baseline flatters us; run the vllm arm before publishing.
