### The overhead ladder, by boundary

H200 reference rate: $4.59/GPU-h (from_freeze=True)


**C=4096**
- unproved single-stream: **not measured**
- unproved batched peak: **not measured**

| boundary | proved tok/s | scope | latency × | E11 × | proved $/Mtok | batched $/Mtok | $/token × |
|---|---|---|---|---|---|---|---|
| commit | 30.306 | 33 chunks (head included) | null | null | $42.07 | null | null |
| stwo total | 16.566 | 33 chunks (head included) | null | null | $76.96 | null | null |
| whole wall | 4.991 | 33 chunks (head included) | null | null | $255.46 | null | null |
| loop_secs | 8.277 | 32 layer chunks (HEADLESS) | null | null | $154.05 | null | null |

**C=16384**
- unproved single-stream: **not measured**
- unproved batched peak: **not measured**

| boundary | proved tok/s | scope | latency × | E11 × | proved $/Mtok | batched $/Mtok | $/token × |
|---|---|---|---|---|---|---|---|
| commit | 20.856 | 33 chunks (head included) | null | null | $61.13 | null | null |
| stwo total | 12.289 | 33 chunks (head included) | null | null | $103.75 | null | null |
| whole wall | 4.192 | 33 chunks (head included) | null | null | $304.15 | null | null |
| loop_secs | null (loop.h200.c16384) | 32 layer chunks (HEADLESS) | null (loop.h200.c16384) | null (loop.h200.c16384) | null | null | null (loop.h200.c16384) |

**The E11 column rests on an assumption:** REGIME 1, the same convention stream E7 used in notes/sufficiency.md: every proved position is charged one measured C=4096 decoded token, so the proved rate in positions/second equals the frozen tok/s row unchanged. This is CONSERVATIVE for E11 -- a prompt position inside a causal rectangle attends to ~N/2 keys, not a full C band -- and it is an assumption, not a measurement: no E11 proved row exists. The unproved side needs no assumption; it is measured directly as (prompt + output) / wall of the concatenated pass.

**What neither side counts:**
- the witness extraction. The proved side's `loop_secs` reads a witness that a prior extraction step produced; that step is in no boundary and has no unproved counterpart.
- model / weight load, and on the proved side the kernel-store emit and the prover binary's own start-up.
- verification. Every number here is prover-side.
- the inference pass itself, on the PROVED side: none of the four boundaries contains one. So an overhead multiple U/P is prover-seconds over inference-seconds -- precisely what fn. 170 computed (zkLLM's 803 s over 668 tok/s). A deployment that must infer AND prove costs (1 + U/P) x the unproved time; at x25-40 that unit is immaterial, and it is stated anyway.

**Caveats the harness raised itself:**
- C=4096: no single-stream unproved run in e2/out.
- C=4096: no batched unproved run; the $/token column is empty.
- C=16384: no single-stream unproved run in e2/out.
- C=16384: no batched unproved run; the $/token column is empty.
