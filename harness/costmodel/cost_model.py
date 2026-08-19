"""B2 cost model: measured stwo constants -> per-token prover budget, checked
against a published 53 tok/s target for a Gemma-3-27B-class model.

Every constant is read out of experiments/b2/out/*.json, i.e. out of an actual
run. The CPU->H100 speedup is never assumed: the model reports the trace
commitment throughput a prover would need for 53 tok/s and how far one CPU is
from it. Closing that gap is B7's job.

    uv run experiments/b2/cost_model.py
"""
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).parent / "out"
B1 = pathlib.Path(__file__).parent.parent / "b1" / "out"


def load(name, required=True):
    p = OUT / f"{name}.json"
    if not p.exists():
        if required:
            sys.exit(f"missing {p}; run experiments/b2/run_all.sh first")
        return None
    return json.loads(p.read_text())


commit = load("commit")
commit_host = load("commit_host", required=False)
field = load("field")
logup = load("logup")
fri = load("fri")

ENV, SEC = commit["env"], commit["security"]


def wide(run):
    """Points with enough columns that Merkle's per-row cost has amortised."""
    return [dict(p, env=run["env"]["os"]) for p in run["result"]["points"] if p["n_cols"] >= 16]


def peak(pts):
    return max(pts, key=lambda p: p["base_cells_per_sec"]["total"])


def largest(pts):
    return max(pts, key=lambda p: p["base_cells"])


def rate(kernel, where="all_threads"):
    for k in field["result"][where]:
        if k["kernel"] == kernel:
            return k["ops_per_sec"]
    raise KeyError(kernel)


# Headline constants come from the containerised run only, so the whole model
# sits in one environment. The host run (bigger traces than the 8 GiB VM can
# hold) is reported separately below.
CP = peak(wide(commit))
COMMIT_CELLS_S = CP["base_cells_per_sec"]["total"]

# End-to-end prover throughput: a whole proof, not just the commit path. Still
# climbing at the largest shape measured, so this is a floor, not a plateau.
fri_cols = load("fri_cols", required=False)
E2E = max((p for r in (fri, fri_cols) if r for p in r["result"]["points"]),
          key=lambda p: p["trace_cells_per_sec"])
E2E_CELLS_S = E2E["trace_cells_per_sec"]
M31_MUL_S = rate("m31_mul_simd")
QM31_MUL_S = rate("qm31_mul_simd")

# LogUp marginal cost: total cells per lookup minus the 2 (in, out) trace cells,
# which a witness has to commit anyway. Measured, per batch size.
LOGUP_MARGINAL = {}
for p in logup["result"]["points"]:
    LOGUP_MARGINAL[p["logup_batch_size"]] = p["cells"]["per_lookup"] - 2
LOGUP_NS = {
    b: min(p["prove_secs"] / p["n_lookups"] * 1e9
           for p in logup["result"]["points"] if p["logup_batch_size"] == b)
    for b in LOGUP_MARGINAL
}

# ---------------------------------------------------------------- the model --
# Committed cells per decoded token, in the shape B1's witness actually has.
#   acc   one int32 accumulator per matmul output              (B1: 45.5% of cells)
#   act   one int8 value per *distinct* matmul input           (B1: 24.6%)
#   nl    bf16 lookup in/out for SiLU and the two norms        (B1: 25.5%)
#   attn  3 columns per (layer, head, key): score, exp, prob   (B1: alpha = 3-4)
#   logup the LogUp interaction column each lookup drags along (B2, measured)


def cells_per_token(cfg, ctx, batch):
    L, d, nh, nkv, hd, dff, vocab = cfg
    acc = L * ((nh + 2 * nkv) * hd + d + 2 * dff + d) + vocab
    act = L * (2 * d + nh * hd + dff) + d
    nl = L * (2 * dff + 4 * d)
    attn = 3 * L * nh * ctx
    n_lookups = L * nh * ctx + L * (2 * dff + 2 * (2 * d + 1)) + vocab
    logup = n_lookups * LOGUP_MARGINAL[batch]
    return dict(acc=acc, act=act, nl=nl, attn=attn, logup=logup, n_lookups=n_lookups,
                flat=acc + act + nl, total=acc + act + nl + attn + logup)


def ops_per_token(cfg, ctx, seq):
    """In-circuit M31 multiplies, the E6b Freivalds model (freivalds.py)."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    params = vocab * d + L * (d * nh * hd + 2 * d * nkv * hd + nh * hd * d + 3 * d * dff)
    mats = [(d, (nh + 2 * nkv) * hd), (nh * hd, d), (d, 2 * dff), (dff, d)]
    io = sum(a + b for a, b in mats)
    ext = 4  # QM31 = 4 x M31
    weight_pass = ext * params / seq
    per_tok_io = ext * L * io
    attn = 3 * ext * L * nh * ctx
    lookups = L * nh * ctx + L * (2 * dff + 2 * (2 * d + 1)) + vocab
    return dict(total=weight_pass + per_tok_io + attn + lookups, weight_pass=weight_pass,
                params=params, naive_cells=params + L * 2 * nh * hd * ctx)


QWEN = (24, 896, 14, 2, 64, 4864, 151936)      # B1's model
GEMMA = (62, 5376, 32, 16, 128, 21504, 262144)  # the 27B proxy shape
TARGET_TOKS, TARGET_CTX, SEQ = 53.0, 16384, 512


def g(x, unit=""):
    for s, d in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.2f} {s}{unit}"
    return f"{x:.2f} {unit}"


print("=" * 78)
print("B2 cost model - measured stwo constants -> per-token prover budget")
print("=" * 78)
print(f"platform: {ENV['arch']}/{ENV['os']}, {ENV['threads']} threads, "
      f"{ENV['simd_n_lanes']} SIMD lanes")
print(f"security: blowup {SEC['blowup_factor']}, {SEC['n_queries']} FRI queries, "
      f"{SEC['pow_bits']} PoW bits -> {SEC['security_bits_conjectured']} bits conjectured, "
      f"{SEC['hash']}")
print()
print("measured constants")
print(f"  trace commit  2^{CP['log_size']} rows x {CP['n_cols']} cols   "
      f"{g(COMMIT_CELLS_S)}cells/s   (commit path only, peak over the wide traces)")
print(f"  whole proof   2^{E2E['log_size']} rows x {E2E['n_cols']} cols   "
      f"{g(E2E_CELLS_S)}cells/s   (prove() end to end, still rising with size)")
print(f"  M31 mul       SIMD, {ENV['threads']} threads              {g(M31_MUL_S)}op/s")
print(f"  QM31 mul      SIMD, {ENV['threads']} threads              {g(QM31_MUL_S)}op/s")
for b in sorted(LOGUP_MARGINAL):
    print(f"  LogUp batch {b} into 2^{logup['result']['log_table']} table      "
          f"{LOGUP_MARGINAL[b]:.0f} extra cells/lookup, {LOGUP_NS[b]:.0f} ns/lookup")
print()

# --- validate the cells model against B1's actual witness ---------------------
man = B1 / "manifest.json"
if man.exists():
    w = json.loads(man.read_text())["witness"]
    steps = json.loads(man.read_text())["sequence"]["steps"]
    kinds = w["values_by_kind"]
    b1_flat = sum(v for k, v in kinds.items() if k != "attention") / steps
    model_flat = cells_per_token(QWEN, 0, 1)["flat"]
    print(f"model check vs B1's witness (Qwen2.5-0.5B, context-independent cells/token):")
    print(f"  B1 measured {b1_flat:,.0f}   model {model_flat:,.0f}   "
          f"ratio {model_flat / b1_flat:.3f}")
    print()

BATCH = min(LOGUP_MARGINAL, key=LOGUP_MARGINAL.get)  # cheapest measured batching
print(f"Gemma-3-27B proxy, S={SEQ}, LogUp batch {BATCH} -- the largest blowup 2 allows, "
      f"since batch b needs degree-(b+1) constraints")
for ctx in (4096, TARGET_CTX):
    c = cells_per_token(GEMMA, ctx, BATCH)
    o = ops_per_token(GEMMA, ctx, SEQ)
    t_commit = c["total"] / COMMIT_CELLS_S
    t_e2e = c["total"] / E2E_CELLS_S
    t_field = o["total"] / M31_MUL_S
    print()
    print(f"  ctx={ctx}")
    print(f"    committed cells/token  {g(c['total'])}  "
          f"(attention {c['attn'] / c['total'] * 100:.0f}%, "
          f"LogUp columns {c['logup'] / c['total'] * 100:.0f}%, "
          f"layer I/O {c['flat'] / c['total'] * 100:.0f}%)")
    print(f"    in-circuit M31 mul/tok {g(o['total'])}  "
          f"(weight pass {o['weight_pass'] / o['total'] * 100:.0f}%)")
    print(f"    on this box            commit-limited {1 / (t_commit + t_field):5.2f} tok/s"
          f"   |  measured end-to-end rate {1 / (t_e2e + t_field):5.2f} tok/s")

c = cells_per_token(GEMMA, TARGET_CTX, BATCH)
o = ops_per_token(GEMMA, TARGET_CTX, SEQ)
need_cells = c["total"] * TARGET_TOKS
need_ops = o["total"] * TARGET_TOKS
print()
print(f"what {TARGET_TOKS:.0f} tok/s at ctx={TARGET_CTX} requires:")
print(f"  trace commitment   {g(need_cells)}cells/s")
print(f"    vs commit path only   {g(COMMIT_CELLS_S)}cells/s  ->  "
      f"x{need_cells / COMMIT_CELLS_S:,.0f} short")
print(f"    vs whole-proof rate   {g(E2E_CELLS_S)}cells/s  ->  "
      f"x{need_cells / E2E_CELLS_S:,.0f} short")
print(f"  in-circuit field   {g(need_ops)}M31 mul/s  "
      f"measured {g(M31_MUL_S)}op/s  ->  "
      + (f"x{need_ops / M31_MUL_S:,.1f} short" if need_ops > M31_MUL_S
         else f"already {M31_MUL_S / need_ops:,.1f}x headroom"))
print(f"  commit-every-MAC   {g(o['naive_cells'] * TARGET_TOKS)}cells/s "
      f"= {o['naive_cells'] * TARGET_TOKS * 4 / 1e12:,.1f} TB/s of trace  ->  impossible")
print()
print("proof size / verification, measured at blowup 2 and "
      f"{SEC['n_queries']} queries:")
for p in fri["result"]["points"]:
    print(f"  2^{p['log_size']} rows x {p['n_cols']} cols   "
          f"{p['proof_bytes'] / 1024:6.1f} KiB   verify {p['verify_secs'] * 1e3:5.2f} ms   "
          f"{p['n_fri_query_positions']} query positions, "
          f"{p['n_fri_inner_layers']} FRI layers")


if commit_host:
    hp, hl = peak(wide(commit_host)), largest(wide(commit_host))
    print()
    print("host run (same binary and toolchain, outside the 8 GiB VM), for the trace")
    print("sizes the container cannot hold:")
    print(f"  peak     2^{hp['log_size']} x {hp['n_cols']}   {g(hp['base_cells_per_sec']['total'])}cells/s"
          f"  ->  x{need_cells / hp['base_cells_per_sec']['total']:,.0f} short of {TARGET_TOKS:.0f} tok/s")
    print(f"  largest  2^{hl['log_size']} x {hl['n_cols']}   {g(hl['base_cells_per_sec']['total'])}cells/s"
          f"  ->  x{need_cells / hl['base_cells_per_sec']['total']:,.0f} short")
    same = [(d, h) for d in wide(commit) for h in wide(commit_host)
            if d["log_size"] == h["log_size"] and d["n_cols"] == h["n_cols"]]
    if same:
        ratios = [h["base_cells_per_sec"]["total"] / d["base_cells_per_sec"]["total"]
                  for d, h in same]
        print(f"  VM overhead on the {len(same)} overlapping points: host is "
              f"{min(ratios):.2f}-{max(ratios):.2f}x the container")
