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
B3 = pathlib.Path(__file__).parent.parent / "b3" / "out"
B4 = pathlib.Path(__file__).parent.parent / "b4" / "out"
B5 = pathlib.Path(__file__).parent.parent / "b5" / "out"
B6 = pathlib.Path(__file__).parent.parent / "b6" / "out"
B65 = pathlib.Path(__file__).parent.parent / "b65" / "out"


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

# --- B3's two missing terms, measured ----------------------------------------
# (1) the r^T W walk. Its three QM31 accumulator columns cost 12 M31 cells per
#     trace row, and a row carries `p` weights, so the walk commits 12/p fresh
#     cells per weight -- a knob, not a constant. B3's width sweep is read here
#     and the design point is the one that minimises fresh cells subject to the
#     proof staying small (b3/README.md section 2).
# (2) the byte-range check on every committed int8 activation, which B2's model
#     charged nothing for: one LogUp lookup per activation cell.
def b3_constants():
    f = B3 / "width.json"
    if not f.exists():
        return None
    pts = json.loads(f.read_text())["result"]["points"]
    best = min(pts, key=lambda p: (p["proof_bytes"], p["cells"]["fresh_per_token"]))
    return dict(
        cells_per_weight=best["cells"]["cells_per_weight"],
        weights_per_row=best["layout_x"]["weights_per_row"],
        proof_bytes=best["proof_bytes"],
        shape=best["shape"], s=best["s"],
        range_all=[(p["cells"]["cells_per_weight"], p["cells"]["fresh_per_token"])
                   for p in pts],
        # cells of *preprocessed* trace the walk reads per proof, per weight
        preprocessed_per_weight=best["cells"]["preprocessed_per_weight"],
    )


B3C = b3_constants()
WEIGHT_WALK_CELLS_PER_WEIGHT = B3C["cells_per_weight"] if B3C else 0.0
PREPROCESSED_PER_WEIGHT = B3C["preprocessed_per_weight"] if B3C else 0.0


# --- B4's terms, measured -----------------------------------------------------
# (1) the dequant's limb decomposition. B3 proves qY only modulo p; reading an
#     accumulator as an integer costs one committed limb plus two range-check
#     lookups per accumulator, in the cheapest *sound* variant (limb bound 2^30,
#     below p, with the sign absorbed by a 2^29 offset).
# (2) the int8 range check, again: B3 spent one lookup per activation, B4 checks
#     two activations with one arity-2 lookup into a 2^16 pair table.
def b4_constants():
    fl, fr = B4 / "limbs.json", B4 / "range.json"
    if not (fl.exists() and fr.exists()):
        return None
    limbs = json.loads(fl.read_text())["result"]["points"]
    sound = [p for p in limbs if p["spec"]["sound"]
             and p["spec"]["max_k"] >= 21504]          # 27B's widest matmul
    best = min(sound, key=lambda p: (p["cells_per_acc_added"],
                                     p["cells_per_acc_with_table"]))
    rng = json.loads(fr.read_text())["result"]["points"]
    per_mode = {}
    for p in rng:
        per_mode.setdefault(p["mode"], p["cells_per_byte_marginal"])
    return dict(
        cells_per_acc=best["cells_per_acc_added"],
        limb_variant=best["spec"]["name"],
        max_k=best["spec"]["max_k"],
        range_marginal=per_mode.get("pair16", 2.0),
        range_marginal_b3=per_mode.get("byte8", 2.0),
        # measured cells per bf16 application, witness included
        lookup_cells=min(p["cells_per_lookup"]
                         for p in json.loads((B4 / "unary.json").read_text())
                         ["result"]["points"]) if (B4 / "unary.json").exists() else None,
    )


B4C = b4_constants()
LIMB_CELLS_PER_ACC = B4C["cells_per_acc"] if B4C else 0.0


# --- B5's terms, measured ------------------------------------------------------
# (1) alpha: committed cells per (layer, head, key) per decoded token.  B2's model
#     charged 3 witness columns plus the exp lookup's 2 interaction cells = 5,
#     which is exactly what B5's v1 layout measures.  The integer softmax commits
#     the *shift* instead of the score and looks it up directly, so the same
#     lookup is the range check and one column disappears: 4.
# (2) the KV cache and the queries have to be committed as quantized integers --
#     they are matmul outputs, so no column in B2's model held them -- and each
#     needs a range check, at 2 cells for the wider Q/K and 1 for a packed int8 V.
# (3) the per-token softmax scalars: the row maximum, the denominator, and the
#     two-limb range check that makes the A.V window explicit.
def b5_constants():
    f = B5 / "block.json"
    if not f.exists():
        return None
    try:
        pts = json.loads(f.read_text())["result"]["points"]
    except (ValueError, KeyError):        # a run in progress leaves it empty
        return None
    per = {}
    for v in dict.fromkeys(p["variant"] for p in pts):
        ps = sorted([p for p in pts if p["variant"] == v], key=lambda p: p["ctx"])
        # marginal cells per (head, key) per token, from the largest two contexts
        a, b = ps[-2], ps[-1]
        per[v] = dict(
            alpha_grid=b["alpha_grid"],
            alpha_marginal=(b["cells"]["fresh_per_token"] - a["cells"]["fresh_per_token"])
            / (b["ctx"] - a["ctx"]),
            ctx=b["ctx"], s=b["s"], hd=b["hd"],
        )
    return per


B5C = b5_constants()


# --- B6's terms, measured -----------------------------------------------------
# (1) the dequant.  B4 stopped at the integer reading (5 cells/accumulator) and
#     scoped `bf16(acc s_x s_w)` out; B6 builds it.  The normalisation table's own
#     bound *is* an injective integer reading below 2^30, so the dequant
#     **replaces** B4's limb decomposition rather than adding to it -- the model
#     charges one or the other, never both.
# (2) the attention output straight to int8.  Its consumer (`o_proj`) requantises
#     anyway, so the scale, the softmax division and the requantize fold into one
#     rounding: 9 cells against a bf16 dequant plus a reciprocal lookup plus a
#     requantize.  B1 scores it at -0.29% perplexity, i.e. free.
# (3) the packing.  Committed columns and proof size are ~independent of L*nh once
#     instances share one column set, so the whole-model proof is bands x one
#     band's columns, with popcount(L*nh) bands.
def b6_constants():
    fd, fp = B6 / "dequant.json", B6 / "pack.json"
    if not fd.exists():
        return None
    try:
        pts = json.loads(fd.read_text())["result"]["points"]
    except (ValueError, KeyError):
        return None
    by = {}
    for p in pts:
        by[(p["spec"]["variant"], p["spec"]["round"])] = p["cells_per_acc_marginal"]
    # Prefer the host sweep at the *design point* (`intexp`, the alpha = 4.00 variant
    # the model charges) over the containerised `intsplit6` one, which exists only
    # because one master exp table at 2^23 rows does not fit 7.7 GiB.
    hp = B6 / "pack_intexp_host.json"
    pack = None
    for f, label in ((hp, "intexp (host)"), (fp, "intsplit6 (container)")):
        if not f.exists():
            continue
        try:
            pp = json.loads(f.read_text())["result"]["points"]
        except (ValueError, KeyError, IndexError):
            continue
        extra = sorted(
            (json.loads(g.read_text())["result"]["points"][0]
             for g in B6.glob("pack_host_*.json")),
            key=lambda p: p["insts"]) if label.endswith("(container)") else []
        pp = pp + extra
        pack = dict(
            variant=label,
            columns=pp[0]["columns"]["total"],
            insts=[p["insts"] for p in pp],
            proof_bytes=[p["proof_bytes"] for p in pp],
            per_inst=[p["proof_bytes_per_instance"] for p in pp],
            alpha_grid=pp[0]["alpha_grid"],
            alpha_block=[p["alpha"] for p in pp],
        )
        break
    return dict(
        limbs=by.get(("limbs", "even")),
        # the design point for a matmul whose consumer is a bf16 table or the
        # residual stream: general bf16 scales, three roundings, no quality cost
        chain=by.get(("chain", "even")),
        # B8 blessed `Round::Away` (-0.45% +- 0.26 at 8B): no sticky bit, one
        # fewer witness column and half the table rows
        chain_away=by.get(("chain", "away")),
        # the cheaper alternative: power-of-two scales, one rounding, +2.29% ppl
        pow2=by.get(("pow2", "even")),
        int8=by.get(("int8", "even")),
        pack=pack,
    )


B6C = b6_constants()


# --- B6.5's terms, measured ----------------------------------------------------
# The requantize `q = round(x / s_x)` and the scale argument `s_x = max|x|/127`
# were the largest unmeasured per-token terms left (B4 section 8 named both; B6 gap 2).
# B6.5 prices them as one gadget: a (q, nz) pair table, one division-table lookup
# keyed by the exponent difference for q >= 1, one pattern comparison for q = 0,
# and a committed max-pattern with a selector and a ceil-divide table for the
# scale.  The marginal below includes the committed q cell, which `act` already
# charges, so the model charges (marginal - 1) per distinct matmul input.
def b65_constants():
    pts = []
    for f in ("requant_q", "requant_gate", "requant_down"):
        p = B65 / f"{f}.json"
        if p.exists():
            try:
                pts += json.loads(p.read_text())["result"]["points"]
            except (ValueError, KeyError):
                pass
    if not pts:
        return None
    full = [p for p in pts if p["with_max"]]
    out = dict(
        requant_max=max(p["cells_per_value_marginal"] for p in full) if full else None,
        requant_only=max(p["cells_per_value_marginal"] for p in pts if not p["with_max"]),
        by_variant={(p["variant"], p["with_max"]): p["cells_per_value_marginal"]
                    for p in pts},
    )
    c = B65 / "compose_host.json"
    if not c.exists():
        c = B65 / "compose.json"
    if c.exists():
        try:
            pt = json.loads(c.read_text())["result"]["point"]
            out["compose"] = dict(
                fresh=pt["cells"]["fresh"], proof_bytes=pt["proof_bytes"],
                columns=pt["columns"]["total"], accepts=pt["accepts"])
        except (ValueError, KeyError):
            pass
    return out


B65C = b65_constants()
REQUANT_CELLS = (B65C["requant_max"] - 1.0) if B65C and B65C.get("requant_max") else 0.0
# `dequant` is the per-accumulator cost of reading the integer *and* rounding it;
# it subsumes B4's limb term, so `cells_per_token` charges this instead of that.
DEQUANT_CELLS_PER_ACC = (B6C or {}).get("chain") or 0.0
DEQUANT_AWAY_PER_ACC = (B6C or {}).get("chain_away") or DEQUANT_CELLS_PER_ACC
DEQUANT_POW2_PER_ACC = (B6C or {}).get("pow2") or 0.0
ATTNOUT_CELLS_PER_ACC = (B6C or {}).get("int8") or 0.0

# --- B8's layout, the default since B9 step 0 ---------------------------------
# B6.5's mixed layout rested on a 0.5B attribution that **inverts** at 8B
# (b8/B8.md section 2 item 1): power-of-two linear scales are free on q/k/v/o at
# 0.5B (-0.43%) and cost +3.67% at 8B, while `lm_head` goes the other way (+1.10%
# -> +0.02%).  So the chain runs on all seven *layer* matmuls and the 15-cell
# power-of-two dequant is charged on `lm_head` alone -- which sits outside the
# layer proof anyway.  `Round::Away` is blessed (-0.45% +- 0.26 at 8B), so the
# chain is 25.12 cells, not 26.12.  The mixed-layout planning range this replaced
# (270-308 M cells/token, 14.3-15.8 G cells/s -- b7/README section 3) is retired:
# the model prints one number.
LAYER_DEQUANT_PER_ACC = DEQUANT_AWAY_PER_ACC
HEAD_DEQUANT_PER_ACC = DEQUANT_POW2_PER_ACC

# --- the numerator width, and what the AIR as built can carry ------------------
# B5's AIR range-checks V into [-128, 127] and the committed denominator Z into
# `z_lo_bits + z_hi_bits` (b5/README section 4; `attn_air.rs` RowEval), so the
# circuit certifies |A.V| < 128 * 2^z_bits.  The numerator width sets Z's blind
# worst case (`nmax * C`) and therefore both bounds.  Two ceilings sit above it:
#   * 2^29 -- B4's *offset* limb variant, whose 2^30 window is split by a +2^29
#     offset and so carries only 2^29 of magnitude;
#   * 2^30 -- M31's own injectivity edge, floor(p/2) = 2^30 - 1, which B6's chain
#     and the attention-output fold both reach because each commits an explicit
#     sign bit and reads the magnitude (`walk_air.rs`: `mag = acc - 2 acc sg`).
# So a numerator wider than 8 bits at C = 16K is not a layout change and costs no
# cell per accumulator; it is a *bound* change that leaves B4's window and lands
# in the sign-bit one.
Z_LO_BITS = 16


def attn_window(num_bits, ctx, z_lo_bits=Z_LO_BITS):
    nmax = (1 << num_bits) - 1
    z_bits = (nmax * ctx).bit_length()          # the AIR's range check on Z
    av = 128 << z_bits                          # |A.V| < 128 * 2^z_bits
    return dict(num_bits=num_bits, nmax=nmax, ctx=ctx, z_bits=z_bits,
                z_hi_bits=max(0, z_bits - z_lo_bits), av_bound=av,
                fits_limb_2p29=av <= 1 << 29,   # B4's offset variant
                fits_sign_2p30=av <= 1 << 30,   # the sign-bit reading, floor(p/2)
                blind_sum=nmax * ctx)
# the integer softmax is the design point; v1 is what B2's model already assumed
ALPHA = (B5C["intexp"]["alpha_grid"] if B5C and "intexp" in B5C else 5.0)
ALPHA_V1 = (B5C["bf16x"]["alpha_grid"] if B5C and "bf16x" in B5C else 5.0)

# ---------------------------------------------------------------- the model --
# Committed cells per decoded token, in the shape B1's witness actually has.
#   acc   one int32 accumulator per matmul output              (B1: 45.5% of cells)
#   act   one int8 value per *distinct* matmul input           (B1: 24.6%)
#   nl    bf16 lookup in/out for SiLU and the two norms        (B1: 25.5%)
#   attn  3 columns per (layer, head, key): score, exp, prob   (B1: alpha = 3-4)
#   logup the LogUp interaction column each lookup drags along (B2, measured)


def cells_per_token(cfg, ctx, batch, seq=None, alpha=None, dequant=None,
                    head_dequant=None):
    L, d, nh, nkv, hd, dff, vocab = cfg
    acc_layer = L * ((nh + 2 * nkv) * hd + d + 2 * dff + d)
    acc = acc_layer + vocab
    act = L * (2 * d + nh * hd + dff) + d
    nl = L * (2 * dff + 4 * d)
    # B5: the whole (token, key) grid at the measured alpha, lookup included
    attn = (alpha if alpha is not None else ALPHA) * L * nh * ctx
    n_lookups = L * (2 * dff + 2 * (2 * d + 1)) + vocab
    logup = n_lookups * LOGUP_MARGINAL[batch]
    # B3: the weight walk's accumulators, and a range check per int8 activation.
    params = n_params(cfg)
    walk = WEIGHT_WALK_CELLS_PER_WEIGHT * params / seq if seq else 0.0
    # B4: the packed pair table halves B3's per-activation range check
    rng_cells = B4C["range_marginal"] if B4C else LOGUP_MARGINAL[batch]
    rng = act * rng_cells if seq else 0.0
    # B4 read the accumulator as an integer and stopped there; B6 rounds it too, and
    # its normalisation *is* the integer reading, so this replaces B4's limb term
    # rather than adding to it.  `dequant` picks which rung of B6's ladder: the
    # general-scale chain (no quality cost, 26 cells) or power-of-two scales
    # (+2.29% perplexity, 15 cells).
    # B8's layout (B9 step 0): the chain on every layer matmul, the power-of-two
    # dequant on `lm_head` alone.  `dequant`/`head_dequant` override either rung.
    per_acc = dequant if dequant is not None else (
        LAYER_DEQUANT_PER_ACC or LIMB_CELLS_PER_ACC)
    head_acc = head_dequant if head_dequant is not None else (
        per_acc if dequant is not None else (HEAD_DEQUANT_PER_ACC or per_acc))
    limbs = (acc_layer * per_acc + vocab * head_acc) if seq else 0.0
    # B6: the attention output's own dequant, folded straight to int8 because
    # `o_proj` requantises it anyway -- one rounding per output element
    attn_out = L * nh * hd * (ATTNOUT_CELLS_PER_ACC or 0.0) if seq else 0.0
    # B6.5: the requantize + max argument, once per *distinct* matmul input
    # (q/k/v share one, gate/up share one; o's input is already int8 via the fold)
    requant = (L * (2 * d + dff) + d) * REQUANT_CELLS if seq else 0.0
    # B5: the committed quantized KV cache and queries, and their range checks
    # (2 cells for int11/int12 K and Q, 1 for a pair-packed int8 V), plus the
    # per-token softmax scalars (max, denominator, its limb and two lookups)
    kv_vals = 2.0 * L * nkv * hd * ctx / seq if seq else 0.0
    q_vals = L * nh * hd
    kv = kv_vals + q_vals + (1.5 * kv_vals + 2.0 * q_vals if seq else 0.0)
    softmax_rows = 5.0 * L * nh
    total = (acc + act + nl + attn + logup + walk + rng + limbs + kv + softmax_rows
             + attn_out + requant)
    return dict(acc=acc, act=act, nl=nl, attn=attn, logup=logup, n_lookups=n_lookups,
                walk=walk, range=rng, limbs=limbs, kv=kv, softmax_rows=softmax_rows,
                attn_out=attn_out, requant=requant, dequant_per_acc=per_acc,
                head_per_acc=head_acc, n_acc=acc, n_acc_layer=acc_layer,
                flat=acc + act + nl, total=total,
                # committed offline, but read once per proof by the walk
                preprocessed=PREPROCESSED_PER_WEIGHT * params / seq if seq else 0.0)


def n_params(cfg):
    L, d, nh, nkv, hd, dff, vocab = cfg
    return vocab * d + L * (d * nh * hd + 2 * d * nkv * hd + nh * hd * d + 3 * d * dff)


def ops_per_token(cfg, ctx, seq):
    """In-circuit M31 multiplies, the E6b Freivalds model (freivalds.py)."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    params = n_params(cfg)
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
if B3C:
    print(f"  r^T W walk    B3, {B3C['weights_per_row']} weights/row     "
          f"{WEIGHT_WALK_CELLS_PER_WEIGHT:.3f} fresh cells/weight "
          f"(12/p; {PREPROCESSED_PER_WEIGHT:.2f} preprocessed cells/weight read per proof)")
else:
    print("  r^T W walk    NOT MEASURED -- run experiments/b3/run_all.sh; "
          "the walk is charged 0 cells below")
if B4C:
    print(f"  limb decomp   B4, variant {B4C['limb_variant']:<9}      "
          f"{B4C['cells_per_acc']:.2f} cells/accumulator "
          f"(sound up to K = {B4C['max_k']:,})")
    print(f"  int8 range    B4, packed pair table    "
          f"{B4C['range_marginal']:.2f} cells/activation "
          f"(B3's one-lookup-per-byte: {B4C['range_marginal_b3']:.2f})")
    if B4C["lookup_cells"]:
        print(f"  bf16 lookup   B4, 2^16 table          "
              f"{B4C['lookup_cells']:.2f} cells/application, witness included")
else:
    print("  limb decomp   NOT MEASURED -- run experiments/b4/run_all.sh; "
          "the dequant is charged 0 cells below")
if B5C:
    for v, c in sorted(B5C.items()):
        print(f"  attention     B5, {v:<8}            "
              f"{c['alpha_grid']:.2f} cells/(head,key,token) on the grid, "
              f"{c['alpha_marginal']:.2f} marginal (one head, S={c['s']}, hd={c['hd']})")
else:
    print("  attention     NOT MEASURED -- run experiments/b5/run_all.sh; the model "
          "charges B2's assumed 5 cells/key below")
if B6C:
    print(f"  dequant       B6, general bf16 scales  "
          f"{B6C['chain']:.2f} cells/accumulator "
          f"(power-of-two scales: {B6C['pow2']:.2f}; B4's integer reading alone: "
          f"{B6C['limbs']:.2f})")
    print(f"  attn output   B6, straight to int8     "
          f"{B6C['int8']:.2f} cells/element "
          f"(vs a bf16 dequant + reciprocal + requantize)")
    if B6C["pack"]:
        pk = B6C["pack"]
        print(f"  packing       B6, {pk['insts'][0]}-{pk['insts'][-1]} instances, "
              f"{pk['variant']}")
        print(f"                {pk['columns']} committed columns, flat; proof "
              f"{pk['proof_bytes'][0] / 1024:.0f}-{max(pk['proof_bytes']) / 1024:.0f} KiB, "
              f"i.e. {pk['per_inst'][0] / 1024:.0f} -> {pk['per_inst'][-1] / 1024:.1f} KiB "
              f"per instance; grid alpha {pk['alpha_grid']:.2f}, block alpha "
              f"{pk['alpha_block'][0]:.2f} -> {pk['alpha_block'][-1]:.2f} as the shared "
              f"tables amortise")
else:
    print("  dequant       NOT MEASURED -- run experiments/b6/run_all.sh; the model "
          "falls back to B4's limb term")
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
    c = cells_per_token(GEMMA, ctx, BATCH, SEQ)
    o = ops_per_token(GEMMA, ctx, SEQ)
    t_commit = c["total"] / COMMIT_CELLS_S
    t_e2e = c["total"] / E2E_CELLS_S
    t_field = o["total"] / M31_MUL_S
    print()
    print(f"  ctx={ctx}")
    print(f"    committed cells/token  {g(c['total'])}  "
          f"(attention {c['attn'] / c['total'] * 100:.0f}%, "
          f"LogUp columns {c['logup'] / c['total'] * 100:.0f}%, "
          f"layer I/O {c['flat'] / c['total'] * 100:.0f}%, "
          f"weight walk {c['walk'] / c['total'] * 100:.0f}%, "
          f"int8 range {c['range'] / c['total'] * 100:.0f}%, "
          f"dequant {c['limbs'] / c['total'] * 100:.0f}%, "
          f"requant {c['requant'] / c['total'] * 100:.0f}%, "
          f"KV cache {c['kv'] / c['total'] * 100:.0f}%)")
    print(f"    + preprocessed         {g(c['preprocessed'])}cells/token of weight trace, "
          f"committed offline but read once per proof (B3)")
    print(f"    in-circuit M31 mul/tok {g(o['total'])}  "
          f"(weight pass {o['weight_pass'] / o['total'] * 100:.0f}%)")
    print(f"    on this box            commit-limited {1 / (t_commit + t_field):5.2f} tok/s"
          f"   |  measured end-to-end rate {1 / (t_e2e + t_field):5.2f} tok/s")

c = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ)
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
if B3C:
    print()
    print(f"B3's weight walk, folded in (measured on {B3C['shape']} at S={B3C['s']}):")
    print(f"  {WEIGHT_WALK_CELLS_PER_WEIGHT:.3f} fresh cells/weight at "
          f"{B3C['weights_per_row']} weights/row  ->  "
          f"{g(c['walk'])}cells/token, {c['walk'] / c['total'] * 100:.1f}% of the total")
    print("  the whole measured trade-off (cells/weight, fresh cells/token):")
    print("    " + "  ".join(f"{cw:.3f}->{g(fpt)}" for cw, fpt in B3C["range_all"]))
    print(f"  int8 byte-range checks add {g(c['range'])}cells/token "
          f"({c['range'] / c['total'] * 100:.1f}%), also new")
if B4C:
    print()
    print(f"B4's terms, folded in:")
    # B4's own number, not the dequant that superseded it: the model charges one or
    # the other, and `c` above already charges B6's.
    b4c = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, dequant=LIMB_CELLS_PER_ACC)
    print(f"  limb decomposition {LIMB_CELLS_PER_ACC:.2f} cells/accumulator "
          f"({B4C['limb_variant']})  ->  {g(b4c['limbs'])}cells/token, "
          f"{b4c['limbs'] / b4c['total'] * 100:.1f}% of what the model was before B6"
          + (f" -- superseded by B6's {DEQUANT_CELLS_PER_ACC:.2f}, which proves the "
             f"rounding as well" if B6C else ""))
    print(f"  packed int8 range check {B4C['range_marginal']:.0f} instead of "
          f"{B4C['range_marginal_b3']:.0f} cells/activation  ->  {g(c['range'])}cells/token "
          f"(B3 charged {g(c['act'] * B4C['range_marginal_b3'])})")
    print(f"  the 2^30 limb bound caps a dot product at K = {B4C['max_k']:,} terms "
          f"(p/2 alone would allow 66,572); this model's widest matmul is "
          f"dff = {GEMMA[5]:,}")
    if B6C:
        print("  B6's normalisation recovers K <= 66,572 for free: the sign is an "
              "explicit committed bit there, so the whole 2^30 window is magnitude")
    print(f"  plan's threshold was 0.25 cells/weight for the 181 M picture to stand: "
          f"{'stands' if WEIGHT_WALK_CELLS_PER_WEIGHT <= 0.25 else 'DOES NOT stand'}")
    print(f"  on top, {g(c['preprocessed'])}cells/token of preprocessed weight trace are "
          f"read per proof")
    print(f"  ({(c['preprocessed'] + c['total']) / c['total'] - 1:.0%} more cells to walk than "
          f"the fresh count if a preprocessed cell costs the same as a fresh one; its "
          f"interpolate/LDE/Merkle share is offline, only quotients and OODS are not)")
if B6C:
    print()
    print("B6's terms, folded in:")
    ch = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, dequant=B6C["chain"])
    p2 = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, dequant=B6C["pow2"])
    b4 = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, dequant=B6C["limbs"])
    print(f"  {ch['n_acc'] / 1e6:.2f} M accumulators/token, each one read as an integer "
          f"*and* rounded to bf16")
    print(f"    general bf16 scales  {B6C['chain']:.2f} cells/acc  ->  "
          f"{g(ch['limbs'])}cells/token, {ch['limbs'] / ch['total'] * 100:.0f}% of "
          f"{g(ch['total'])}  (B1: -0.48% perplexity, i.e. free)")
    print(f"    power-of-two scales  {B6C['pow2']:.2f} cells/acc  ->  "
          f"{g(p2['limbs'])}cells/token, {p2['limbs'] / p2['total'] * 100:.0f}% of "
          f"{g(p2['total'])}  (B1: +2.29% perplexity)")
    print(f"    B4's integer reading alone was {B6C['limbs']:.2f}  ->  {g(b4['total'])}, "
          f"so the rounding B4 scoped out is {ch['total'] / b4['total'] - 1:+.0%} of the "
          f"whole per-token budget")
    print(f"  attention output straight to int8: {g(ch['attn_out'])}cells/token at "
          f"{B6C['int8']:.2f} cells/element")
    print()
    print("B8's layout, the default above (B9 step 0):")
    print(f"  {c['n_acc_layer'] / 1e6:.2f} M layer accumulators at "
          f"{c['dequant_per_acc']:.2f} cells (the chain, Round::Away) and "
          f"{GEMMA[6] / 1e3:.0f} k `lm_head` accumulators at {c['head_per_acc']:.2f} "
          f"(power-of-two)")
    print(f"    -> {g(c['limbs'])}cells/token against {g(ch['limbs'])} chain-everywhere: "
          f"{g(c['total'])} per token, {c['total'] / ch['total'] - 1:+.1%}")
    mix = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, dequant=B6C["pow2"],
                          head_dequant=B6C["pow2"])
    print(f"    B6.5's mixed layout spanned {g(mix['total'])}-{g(ch['total'])} per token "
          f"({g(mix['total'] * TARGET_TOKS)}-{g(ch['total'] * TARGET_TOKS)}cells/s for "
          f"{TARGET_TOKS:.0f} tok/s); B8's attribution retires that range -- pow2 on the "
          f"layer costs +3.67% perplexity at 8B, so there is one layout, not a band")
    print()
    print("the numerator width, and the window the AIR as built certifies:")
    print(f"  {'bits':>4} {'nmax':>5} {'Z bits':>7} {'z_hi':>5} {'|A.V| <':>10}  "
          f"B4 limb 2^29   sign-bit 2^30")
    for nb in (8, 9, 10, 11):
        w = attn_window(nb, TARGET_CTX)
        print(f"  {nb:>4} {w['nmax']:>5} {w['z_bits']:>7} {w['z_hi_bits']:>5} "
              f"{'2^' + str(w['av_bound'].bit_length() - 1):>10}  "
              f"{'fits' if w['fits_limb_2p29'] else 'OVER':>12}   "
              f"{'fits' if w['fits_sign_2p30'] else 'OVER':>13}")
    print("  8 bits is the last width B4's offset limb variant covers; 9 fits only the")
    print("  sign-bit reading (B6's chain, and the fold's committed `sg`), and 10 leaves")
    print("  M31 altogether at C = 16K.  The cell price of the extra bit is zero per")
    print("  accumulator: no column moves, only two shared range tables widen by one bit")
    print("  (the denominator's high limb 2^6 -> 2^7 rows, the fold's remainder table x2).")
if B65C:
    print()
    print("B6.5's terms, folded in:")
    print(f"  requantize + max argument: {B65C['requant_max']:.2f} cells/value measured "
          f"({B65C['requant_only']:.2f} without the max argument), charged at "
          f"{REQUANT_CELLS:.2f} (the committed q is already in `act`)")
    print(f"    -> {g(c['requant'])}cells/token, "
          f"{c['requant'] / c['total'] * 100:.1f}% of {g(c['total'])} "
          f"-- inside the plan's +5-15% guess")
    for (v, m), cells in sorted(B65C["by_variant"].items()):
        print(f"    {v:<5} {'with' if m else 'no  '} max argument  {cells:.2f} cells/value")
    if B65C.get("compose"):
        cp = B65C["compose"]
        print(f"  the composed layer proof: {g(cp['fresh'])}fresh cells, "
              f"{cp['columns']} columns, {cp['proof_bytes'] / 2**20:.2f} MiB, "
              f"accepted by the serialized-statement verifier alone")
if B5C:
    print()
    print("B5's terms, folded in:")
    v1 = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ, alpha=ALPHA_V1)
    print(f"  attention grid at {ALPHA:.2f} cells/(head,key,token) (integer softmax) "
          f"-> {g(c['attn'])}cells/token, {c['attn'] / c['total'] * 100:.0f}% of the total")
    print(f"  v1 (committed bf16 exp input) is {ALPHA_V1:.2f} -> {g(v1['attn'])}, "
          f"i.e. {v1['total'] / c['total'] - 1:+.1%} on the whole per-token budget")
    print(f"  committed KV cache + queries and their range checks: {g(c['kv'])}cells/token "
          f"({c['kv'] / c['total'] * 100:.1f}%), a term no earlier model had")
    print(f"  per-token softmax scalars (max, denominator, its limbs): "
          f"{g(c['softmax_rows'])}cells/token")
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

# --- B7b's ladder, re-read against this denominator (B9 step 0) ----------------
# The three boundaries B7a section 5 defines, measured on the H100 at B7b's
# largest chunk (2.217 G cells, the 7B shape at C = 16K; b7/README 26.5).  They
# are constants of that run, not of this model -- what moves here is the
# denominator they are divided into.
B7B_CHUNK_CELLS = 2.217e9
B7B = [("commit path", 4.45e9), ("stwo total (commit + prove_ex)", 2.89e9),
       ("whole wall (generation + stwo + serialize)", 1.46e9)]
print()
print(f"B7b's ladder at 27B/ctx={TARGET_CTX}, re-read against {g(c['total'])}cells/token")
print(f"({g(need_cells)}cells/s for {TARGET_TOKS:.0f} tok/s, measured at B7b's "
      f"{B7B_CHUNK_CELLS / 1e9:.3f} G chunk):")
for name, rate_ in B7B:
    print(f"  {name:<42} {rate_ / 1e9:.2f} G cells/s   x{need_cells / rate_:.1f}   "
          f"{rate_ / c['total']:5.2f} tok/s  (x{TARGET_TOKS / (rate_ / c['total']):.0f} "
          f"off {TARGET_TOKS:.0f})")
