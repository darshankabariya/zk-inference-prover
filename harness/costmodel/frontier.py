"""E1 -- the frontier price table.

The paper's thesis is that the cost of verified inference is a *computable
function* of a model's architecture and a card's measured hash rate.  This module
is that computation: it takes the B2 cost model (`cost_model.py`, extended by this
stream with experts / top-k / sequence-batch terms), calibrates it against the one
sequence this project has actually measured end to end, and prices seven frontier
shapes at both contexts, at all three boundaries, on both cards.

Three rules it keeps, and they are the point:

 1. **No measured constant is written down here.**  Every number the table stands
    on is read out of a checked-in artifact, or -- for the dollar rates, which are
    market facts and not measurements -- taken from
    `experiments/paper/constants.json` (stream W0) if it exists and named as a
    required input if it does not.  A missing rate prints `n/a`; it never
    defaults, and it never falls back to zero.
 2. **Every architectural number has a source**, and every shape carries a
    self-check: the model's own parameter count against the published total and
    active counts.  A shape that fails its check is marked `[check]` and the
    failure is printed, not smoothed.
 3. **The 8B row the model was calibrated on is checked, not tuned.**  The
    calibration has exactly one free parameter and two measured points, so it has
    one residual degree of freedom, and that residual is printed.

    uv run experiments/b2/frontier.py
    uv run experiments/b2/frontier.py --seq-batch 32
"""
import argparse
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE))

import cost_model as cm                                          # noqa: E402

TOKENS = 512            # the measured sequence's decoded tokens (S per chunk)
REPO = ROOT.parent


def rel(p):
    """Paths in the printed table are repo-relative: the table is a committed
    artifact and a worktree path in it is noise."""
    try:
        return str(pathlib.Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- constants --
# What this module needs that it cannot compute.  Stream W0 is producing
# `experiments/paper/constants.json`; until it lands, the entries below carry the
# same fields it will (value, unit, grade, source) so that the swap is a lookup
# change and nothing else.  `value: None` means *required and not yet known* --
# every column that needs it prints `n/a`.
CONSTANTS_JSON = ROOT / "paper" / "constants.json"

NEEDED = {
    # dollars.  The paper prices GPU time at citable reference market rates, not
    # at what we paid: sponsored access is neither reproducible nor comparable for
    # a reader, and a zero denominator would make $/Mtok meaningless.  So the
    # `*_reference_*` keys are the only ones this module ever reads.
    "h100_reference_usd_per_hour": dict(
        value=3.39, unit="USD/GPU-hour", grade="[web]",
        source="Novita on-demand H100, given by the user 2026-08-31"),
    "h200_reference_usd_per_hour": dict(
        value=None, unit="USD/GPU-hour", grade="[pending]",
        source="not yet chosen; the H200 we measure on is sponsored (Amodo) and "
               "its actual rate is not a publishable reference rate"),
}


def load_constants():
    """`experiments/paper/constants.json` if it exists, else the local table.

    Only `*_reference_*` dollar keys are read.  `*_actual_*` keys exist in W0's
    file marked not-for-publication and this module must never consume them."""
    table = {k: dict(v, origin="frontier.py (W0's file absent)") for k, v in NEEDED.items()}
    if CONSTANTS_JSON.exists():
        doc = json.loads(CONSTANTS_JSON.read_text())
        for k in NEEDED:
            if k in doc:
                e = doc[k] if isinstance(doc[k], dict) else dict(value=doc[k])
                table[k] = dict(NEEDED[k], **e, origin=rel(CONSTANTS_JSON))
    return table


# ------------------------------------------------------- measured artifacts --
# The card constants.  Not typed in: parsed out of the two sequence runs this
# project has recorded at all three boundaries.  Each carries its card, its
# context, its chunk count and its boundary, per the house rule.
H200_SUMMARY = ROOT / "b10" / "out" / "seq5" / "n_summary_c{ctx}.json"
H100_LOOP = ROOT / "b10" / "out" / "w" / "fin_pipe_c4096.json"

SUMMED = re.compile(
    r"=== \(3\) the sequence summed, C=(?P<ctx>\d+) ===(?P<body>.*?)(?====|\Z)", re.S)
ROW = re.compile(r"^\s*(?P<k>[a-z ]+?):\s+(?P<v>[\d,\.]+)", re.M)
BOUND = re.compile(r"^\s{2}(?P<name>commit|stwo total|whole wall)\s+"
                   r"(?P<secs>[\d\.]+)s\s+(?P<toks>[\d\.]+)\s*$", re.M)
CHUNK = re.compile(r"^\s*(?P<chunk>\S+)\s+(?P<tbl>\d+)\s+(?P<acc>yes|no)\s+"
                   r"(?P<fresh>[\d,]+)\s+(?P<commit>[\d\.]+)s\s+(?P<stwo>[\d\.]+)s\s+"
                   r"(?P<whole>[\d\.]+)s\s+(?P<peak>\d+)\s+(?P<proof>[\d\.]+)", re.M)


def h200_card():
    """H200, seq5, 33 chunks (32 layers + head), S=512, defaults regime, empty
    family env.  `briefs/n-alldev.md` Record (5); artifacts in b10/out/seq5/."""
    cards = {}
    for ctx in (4096, 16384):
        p = pathlib.Path(str(H200_SUMMARY).format(ctx=ctx))
        if not p.exists():
            sys.exit(f"missing {p}")
        text = p.read_text()
        m = SUMMED.search(text)
        body = m.group("body")
        vals = {k.strip(): v for k, v in ROW.findall(body)}
        fresh = float(vals["fresh cells"].replace(",", ""))
        proof = float(vals["proof bytes"].replace(",", ""))
        rates = {b.group("name"): dict(secs=float(b.group("secs")),
                                       tok_s=float(b.group("toks")),
                                       cells_s=fresh / float(b.group("secs")))
                 for b in BOUND.finditer(body)}
        chunks = [(float(c.group("fresh").replace(",", "")), int(c.group("peak")))
                  for c in CHUNK.finditer(text)]
        big, peak = max(chunks)      # the widest chunk, and *its* card peak
        cards[ctx] = dict(fresh=fresh, cells_per_token=fresh / TOKENS, proof_bytes=proof,
                          mib_per_token=proof / 2 ** 20 / TOKENS,
                          chunks=int(vals["chunks accepted"]), boundaries=rates,
                          peak_mib=peak, peak_chunk_cells=big, source=rel(p))
    return dict(card="H200", tokens=TOKENS, per_ctx=cards,
                note="33 chunks (32 layers + the head), one process per chunk, "
                     "not pipelined; stream N's seq5")


def h100_card():
    """H100, stream W's final pipelined loop, 32 layer chunks (no head), C=4096.
    `briefs/w-genbuild.md` Record (5); artifact b10/out/w/fin_pipe_c4096.json."""
    if not H100_LOOP.exists():
        sys.exit(f"missing {H100_LOOP}")
    res = json.loads(H100_LOOP.read_text())["result"]
    ch, s = res["chunks"], res["summary"]
    fresh = sum(c["cells"]["fresh"] for c in ch)
    proof = sum(c["proof_bytes"] for c in ch)
    secs = dict(commit=s["commit_sum"], **{"stwo total": s["stwo_sum"],
                                           "whole wall": s["whole_sum"],
                                           "pipelined loop": s["loop_secs"]})
    return dict(card="H100", tokens=TOKENS, per_ctx={4096: dict(
        fresh=fresh, cells_per_token=fresh / TOKENS, proof_bytes=proof,
        mib_per_token=proof / 2 ** 20 / TOKENS, chunks=len(ch),
        boundaries={k: dict(secs=v, tok_s=TOKENS / v, cells_s=fresh / v)
                    for k, v in secs.items()},
        peak_mib=max(c.get("mem_after_prove", 0) for c in ch) / 2 ** 20,
        peak_chunk_cells=max(c["cells"]["fresh"] for c in ch),
        source=rel(H100_LOOP))},
        note="32 layer chunks, no head; the fourth boundary `pipelined loop` is "
             "wider than `whole wall` (it counts the witness read) and smaller "
             "(the prove hides behind the next chunk's generation)")


# ------------------------------------------------------------- the shapes ----
# Every field below is read out of the model's own `config.json`, fetched
# 2026-08-31.  `check_total` / `check_active` are the vendor's published counts and
# the table prints the model's own arithmetic against them.
def mla(d, nh, q_lora, kv_lora, qk_nope=128, qk_rope=64, v_head=128):
    """DeepSeek-V3-family multi-head latent attention, as parameters and as
    committed accumulators per layer.  The cost model's GQA formulas do not
    describe MLA: the q and kv sides are low-rank pairs, so both the weight count
    and the accumulator count differ, and the cache is one latent (kv_lora +
    qk_rope) rather than a K and a V."""
    q_out, kv_a_out, kv_b_out = nh * (qk_nope + qk_rope), kv_lora + qk_rope, nh * (qk_nope + v_head)
    params = (d * q_lora + q_lora * q_out + d * kv_a_out + kv_lora * kv_b_out
              + nh * v_head * d)
    accs = q_lora + q_out + kv_a_out + kv_b_out + d
    # what the model caches per key per layer, expressed as the GQA term
    # `2 * nkv * hd` the cost model charges, so that nkv is an *effective* number
    return dict(attn_params_per_layer=params, attn_accs_per_layer=accs,
                nkv_eff=kv_a_out / (2.0 * v_head))


def group(layers, dff, experts=1, topk=1, linear=False, **kw):
    return dict(layers=layers, dff=dff, experts=experts, topk=topk, linear=linear, **kw)


DS3 = mla(7168, 128, 1536, 512)
K2 = mla(7168, 64, 1536, 512)
K3 = mla(7168, 96, 1536, 512)

SHAPES = [
    dict(name="Llama-3.1-8B", tag="calibration", d=4096, nh=32, nkv=8, hd=128,
         vocab=128256, groups=[group(32, 14336)],
         check_total=8.03e9, check_active=8.03e9,
         source="config.json (meta-llama/Meta-Llama-3.1-8B); the shape every "
                "measured number in this repo was taken on"),
    dict(name="Llama-3.1-405B", tag="dense", d=16384, nh=128, nkv=8, hd=128,
         vocab=128256, groups=[group(126, 53248)],
         check_total=405.85e9, check_active=405.85e9,
         source="[web] config.json, nvidia/Llama-3.1-405B-Instruct-FP8, fetched "
                "2026-08-31: L=126 d=16384 nh=128 nkv=8 dff=53248 vocab=128256"),
    dict(name="1T dense", tag="dense, hypothetical", d=20480, nh=160, nkv=16, hd=128,
         vocab=128256, groups=[group(160, None)], solve_dff_for=1.0e12,
         check_total=1.0e12, check_active=1.0e12,
         source="[hypothetical] defined here, not a real model: Llama-405B's "
                "aspect ratio scaled to 1.0 T parameters (dff solved)"),
    dict(name="DeepSeek-V3", tag="MoE 256/8", d=7168, nh=128, nkv=DS3["nkv_eff"], hd=128,
         vocab=129280,
         groups=[group(3, 18432, **{k: DS3[k] for k in ("attn_params_per_layer",
                                                        "attn_accs_per_layer")}),
                 group(58, 2048, experts=257, topk=9,
                       **{k: DS3[k] for k in ("attn_params_per_layer",
                                              "attn_accs_per_layer")})],
         check_total=671e9, check_active=37e9,
         source="[web] config.json, deepseek-ai/DeepSeek-V3, fetched 2026-08-31: "
                "L=61 d=7168 nh=128 MLA(kv_lora 512, q_lora 1536) "
                "moe_intermediate 2048, 256 routed + 1 shared, top-8, "
                "first_k_dense_replace 3 (intermediate 18432), vocab 129280"),
    dict(name="Kimi K2", tag="MoE 384/8", d=7168, nh=64, nkv=K2["nkv_eff"], hd=128,
         vocab=163840,
         groups=[group(1, 18432, **{k: K2[k] for k in ("attn_params_per_layer",
                                                       "attn_accs_per_layer")}),
                 group(60, 2048, experts=385, topk=9,
                       **{k: K2[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")})],
         check_total=1.04e12, check_active=32e9,
         source="[web] config.json, moonshotai/Kimi-K2-Instruct, fetched "
                "2026-08-31: L=61 d=7168 nh=64 MLA(kv_lora 512, q_lora 1536) "
                "moe_intermediate 2048, 384 routed + 1 shared, top-8, "
                "first_k_dense_replace 1, vocab 163840"),
    dict(name="Kimi K3", tag="MoE 896/16 [check]", d=7168, nh=96, nkv=K3["nkv_eff"], hd=128,
         vocab=163840,
         groups=[group(24, 3072, experts=898, topk=18,
                       **{k: K3[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")}),
                 group(69, 3072, experts=898, topk=18, linear=True,
                       **{k: K3[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")})],
         check_total=2.8e12, check_active=104e9,
         source="[web, CHECK] config.json, moonshotai/Kimi-K3, fetched 2026-08-31: "
                "L=93 (24 full-attention + 69 KDA linear-attention) d=7168 nh=96 "
                "moe_intermediate 3072, 896 routed + 2 shared, top-16, "
                "vocab 163840.  Two things this row cannot confirm: the expert "
                "geometry does not reproduce the published 2.8 T (see the check "
                "column), and the 69 KDA layers' proving cost is priced at a zero "
                "attention grid, which is a floor and not a model."),
    dict(name="Qwen3-235B-A22B", tag="MoE 128/8", d=4096, nh=64, nkv=4, hd=128,
         vocab=151936, groups=[group(94, 1536, experts=128, topk=8)],
         check_total=235e9, check_active=22e9,
         source="[web] config.json, Qwen/Qwen3-235B-A22B, fetched 2026-08-31: "
                "L=94 d=4096 nh=64 nkv=4 hd=128 moe_intermediate 1536, "
                "128 experts top-8, no shared expert, vocab 151936"),
    dict(name="Llama-4 Maverick", tag="MoE 128/1", d=5120, nh=40, nkv=8, hd=128,
         vocab=202048,
         groups=[group(24, 16384), group(24, 8192, experts=129, topk=2)],
         check_total=400e9, check_active=17e9,
         source="[web] config.json (text_config), Llama-4-Maverick-17B-128E-"
                "Instruct, fetched 2026-08-31: L=48 d=5120 nh=40 nkv=8 hd=128, "
                "interleave_moe_layer_step 2 -> 24 dense layers at "
                "intermediate_size_mlp 16384 and 24 MoE layers at "
                "intermediate_size 8192, 128 routed + 1 shared, top-1, "
                "vocab 202048"),
]


def solve_dff(shape, target):
    """The hypothetical dense shape's FFN width, solved for a parameter target."""
    d, nh, nkv, hd, vocab = shape["d"], shape["nh"], shape["nkv"], shape["hd"], shape["vocab"]
    L = shape["groups"][0]["layers"]
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    dff = (target - vocab * d - L * attn) / (L * 3 * d)
    return int(round(dff / 1024) * 1024)


for _s in SHAPES:
    if _s.get("solve_dff_for"):
        _s["groups"][0]["dff"] = solve_dff(_s, _s["solve_dff_for"])


def cfg_of(shape, grp):
    return (grp["layers"], shape["d"], shape["nh"], shape["nkv"], shape["hd"],
            grp["dff"], shape["vocab"])


def params_of(shape, walked=False, embedding=False):
    """Weights the *proof* sees: the `lm_head`, the attention, and either every
    expert (`walked=True`) or the top-k a token actually uses.

    `embedding=True` adds the input embedding table.  The proof never walks it --
    it is a lookup, not a matmul, and the cost model has always excluded it -- but
    a vendor's published parameter count includes it, so the shape checks add it
    back to compare like with like."""
    tot = shape["vocab"] * shape["d"] * (2 if embedding else 1)
    for grp in shape["groups"]:
        tot += cm.n_params(cfg_of(shape, grp), experts=grp["experts"],
                           topk=grp["topk"], walked=walked, head=False,
                           attn_params_per_layer=grp.get("attn_params_per_layer"))
    return tot


# ------------------------------------------------------------ calibration ----
def measured_8b(h200):
    return {c: h200["per_ctx"][c]["cells_per_token"] for c in h200["per_ctx"]}


def model_cells(shape, ctx, seq=TOKENS, seq_batch=1, alpha=None, batch=None,
                with_head=True):
    """The whole per-token budget for a shape, summed group by group: the cost
    model's `cells_per_token` plus B10 step 2's glue, with the model-level terms
    (the head's accumulators, its lookups, the token tie, the final norm) charged
    to exactly one group."""
    batch = cm.BATCH if batch is None else batch
    out = dict(total=0.0, preprocessed=0.0, glue=0.0, parts={})
    for i, grp in enumerate(shape["groups"]):
        cfg = cfg_of(shape, grp)
        head = with_head and (i == 0)
        c = cm.cells_per_token(
            cfg, 0 if grp["linear"] else ctx, batch, seq, alpha=alpha,
            experts=grp["experts"], topk=grp["topk"], seq_batch=seq_batch, head=head,
            attn_accs_per_layer=grp.get("attn_accs_per_layer"),
            attn_params_per_layer=grp.get("attn_params_per_layer"))
        gl = cm.glue_cells(cfg, "tree", topk=grp["topk"], final_norm=head)
        out["total"] += c["total"] + gl["net"]
        out["preprocessed"] += c["preprocessed"]
        out["glue"] += gl["net"]
        for k, v in c.items():
            if isinstance(v, (int, float)) and k not in ("dequant_per_acc", "head_per_acc",
                                                         "tokens_per_proof"):
                out["parts"][k] = out["parts"].get(k, 0.0) + v
    return out


def calibrate(h200, shape, batch=None):
    """One free parameter, two measured points, one residual degree of freedom.

    `alpha` -- the committed cells per (head, key, token) of the attention grid --
    is the only term the composed sequence measures differently from the standalone
    block AIR B5 read it off (the composed chunk's LogUp interaction columns for
    the grid are not in B5's `alpha_grid`).  Solve it from the *slope* between the
    two measured contexts, then check the *level* at each.  Nothing else is
    touched: this is a constant replaced by its correctly-measured counterpart, not
    a fit to the totals."""
    meas = measured_8b(h200)
    lo, hi = sorted(meas)
    a0, a1 = 0.0, 1.0
    def slope(a):
        return (model_cells(shape, hi, alpha=a, batch=batch)["total"]
                - model_cells(shape, lo, alpha=a, batch=batch)["total"])
    m0, m1 = slope(a0), slope(a1)
    alpha = (meas[hi] - meas[lo] - m0) / (m1 - m0)
    resid = {c: model_cells(shape, c, alpha=alpha, batch=batch)["total"] / meas[c] - 1.0
             for c in meas}
    raw = {c: model_cells(shape, c, batch=batch)["total"] / meas[c] - 1.0 for c in meas}
    return dict(alpha=alpha, alpha_b5=cm.ALPHA, residual=resid, raw_residual=raw,
                measured=meas)


# ------------------------------------------------------- E11: prompt+output --
def prefill_fit():
    """B9 probe 3's measured fit, re-derived here from the two runs rather than
    quoted: fresh ~ F + q S + p S^2 per *layer* chunk at C = S, 8B shape.

    **Two measured points, three coefficients, zero residual degrees of freedom.**
    The S=2048 run was killed at 133.8 GB host RSS and its JSON is empty, so the
    fit cannot be checked against anything it was not built from.  Every number
    that rides on it carries this caveat."""
    pts = []
    for s in (512, 1024, 2048):
        p = ROOT / "b9" / "out" / f"prefill_s{s}.json"
        if p.exists() and p.stat().st_size:
            d = json.loads(p.read_text())["result"]["point"]
            pts.append(dict(s=s, fresh=d["cells"]["fresh"], bands=d["family_cells"]["bands"]))
    if len(pts) < 2:
        return None
    a, b = pts[-2], pts[-1]
    p = (b["bands"] - a["bands"]) / (b["s"] ** 2 - a["s"] ** 2)
    q = ((b["fresh"] - b["bands"]) - (a["fresh"] - a["bands"])) / (b["s"] - a["s"])
    f = (a["fresh"] - a["bands"]) - q * a["s"]
    return dict(fixed=f, per_token=q, per_pair=p, points=[x["s"] for x in pts],
                missing=[s for s in (512, 1024, 2048)
                         if s not in [x["s"] for x in pts]],
                source=rel(ROOT / "b9" / "out" / "prefill_s*.json"))


def e11_factor(fit, n_positions, n_output, s_chunk, layers=32):
    """The full prompt+output statement (E11, decided 2026-08-31), as a multiple of
    today's decode-only statement, from B9's measured fit at the 8B shape.

    Today: `n_output` positions proven, each against a declared C-key prefix.
    E11:   all `n_positions` proven from the embedding up, as a chunked rectangle
           of layer x token-block chunks of `s_chunk` tokens.

    The chunking overhead is the shared-table term `fixed`, paid N/S times instead
    of once."""
    per_chunk = fit["fixed"] + fit["per_token"] * s_chunk + fit["per_pair"] * s_chunk ** 2
    chunks = n_positions / s_chunk
    cells = per_chunk * chunks * layers
    # the chunking overhead, defined exactly as section 9 defines it: the shared
    # table term paid N/S times instead of once, as a share of the chunked total.
    # (Comparing against a *monolith* instead is a different and misleading number
    # -- the monolith's S^2 grid term is quadratic in N, so at large N chunking
    # would score as a saving.)
    over = fit["fixed"] * (chunks - 1) * layers / cells
    monolith = (fit["fixed"] + fit["per_token"] * n_positions
                + fit["per_pair"] * n_positions ** 2) * layers
    return dict(cells=cells, per_position=cells / n_positions,
                chunk_overhead=over, vs_monolith=cells / monolith - 1.0,
                s_chunk=s_chunk, n_positions=n_positions, n_output=n_output)


# ------------------------------------------------------------------ table ----
def g(x, unit=""):
    if x is None:
        return "n/a"
    for s, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.2f} {s}{unit}"
    return f"{x:.2f} {unit}"


def usd_per_mtok(tok_s, usd_per_hour):
    if tok_s is None or not usd_per_hour:
        return None
    return 1e6 / tok_s / 3600.0 * usd_per_hour


def price(shape, cards, alpha, contexts=(4096, 16384), seq=TOKENS, seq_batch=1,
          usd=None, target_tok_s=20.0, mib_per_chunk=None, batch=None):
    """The reusable per-shape row.  Everything the table prints comes from here."""
    usd = usd or {}
    row = dict(name=shape["name"], tag=shape["tag"],
               params_total=params_of(shape, walked=True, embedding=True),
               params_active=params_of(shape, embedding=True),
               params_walked=params_of(shape, walked=True),
               check_total=shape.get("check_total"), check_active=shape.get("check_active"),
               source=shape["source"], seq=seq, seq_batch=seq_batch, contexts={})
    n_chunks = sum(grp["layers"] for grp in shape["groups"]) + 1
    for ctx in contexts:
        c = model_cells(shape, ctx, seq=seq, seq_batch=seq_batch, alpha=alpha,
                        batch=batch)
        entry = dict(cells_per_token=c["total"], preprocessed_per_token=c["preprocessed"],
                     glue_per_token=c["glue"], parts=c["parts"], chunks=n_chunks,
                     cards={})
        # proof bytes, two ways, because they disagree and neither is measured at
        # frontier scale (see the record)
        entry["mib_per_token_chunks"] = (mib_per_chunk or 0.0) * n_chunks
        entry["mib_per_token_params"] = 0.15 * row["params_active"] / 1e9
        for name, card in cards.items():
            if ctx not in card["per_ctx"]:
                continue
            ref = card["per_ctx"][ctx]
            out = {}
            for b, m in ref["boundaries"].items():
                tok_s = m["cells_s"] / c["total"]
                out[b] = dict(cells_s=m["cells_s"], tok_s=tok_s,
                              cards_for_target=math.ceil(target_tok_s / tok_s),
                              usd_per_mtok=usd_per_mtok(tok_s, usd.get(name)))
            entry["cards"][name] = out
        row["contexts"][ctx] = entry
    return row


def dense_twin(shape):
    """The same shape with every expert count collapsed to its top-k: a dense model
    with *identical active parameters*.  This is the comparison B9's `x2.35 pure
    overhead` is stated against, and the only fair one -- against a dense model of
    the same *total* size an MoE is of course cheaper."""
    twin = dict(shape)
    twin["groups"] = [dict(grp, experts=grp["topk"]) for grp in shape["groups"]]
    return twin


def moe_ratio(shape, ctx, alpha, seq=TOKENS, seq_batch=1, ref_batch=1, batch=None):
    """MoE at batch B against its dense twin at batch `ref_batch`.

    With `ref_batch=1` the reference is fixed, which is what section 6.4's counter
    claims: at `B ~ E/k` the batched MoE proof costs what the unbatched dense proof
    costs, because each expert is again amortising its walk over S tokens."""
    dense = dense_twin(shape)
    a = model_cells(shape, ctx, seq=seq, seq_batch=seq_batch, alpha=alpha, batch=batch)
    b = model_cells(dense, ctx, seq=seq, seq_batch=ref_batch, alpha=alpha, batch=batch)
    return dict(moe=a["total"], dense_active=b["total"], ratio=a["total"] / b["total"],
                moe_pre=a["preprocessed"], dense_pre=b["preprocessed"],
                ratio_with_pre=(a["total"] + a["preprocessed"])
                / (b["total"] + b["preprocessed"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=TOKENS,
                    help="tokens per chunk S (the measured sequence's is 512)")
    ap.add_argument("--seq-batch", type=int, default=1,
                    help="sequences per proof B -- E6a's knob, NOT the LogUp "
                         "lookup-batch size, which is --logup-batch")
    ap.add_argument("--logup-batch", type=int, default=None,
                    help="LogUp lookups sharing one interaction column (b2/out/"
                         "logup.json); nothing to do with --seq-batch")
    ap.add_argument("--target", type=float, default=20.0)
    ap.add_argument("--json", default=str(OUT / "frontier.json"))
    args = ap.parse_args(argv)

    consts = load_constants()
    batch = args.logup_batch if args.logup_batch is not None else cm.BATCH
    if batch not in cm.LOGUP_MARGINAL:
        sys.exit(f"--logup-batch {batch} not measured; have {sorted(cm.LOGUP_MARGINAL)}")
    usd = {"H100": consts["h100_reference_usd_per_hour"]["value"],
           "H200": consts["h200_reference_usd_per_hour"]["value"]}
    h200, h100 = h200_card(), h100_card()
    cards = {"H200": h200, "H100": h100}
    llama8b = next(s for s in SHAPES if s["name"] == "Llama-3.1-8B")
    cal = calibrate(h200, llama8b, batch=batch)
    alpha = cal["alpha"]
    fit = prefill_fit()
    mib_per_chunk = (h200["per_ctx"][4096]["proof_bytes"] / 2 ** 20
                     / h200["per_ctx"][4096]["chunks"] / TOKENS)

    print("=" * 100)
    print("E1 -- the frontier price table")
    print("=" * 100)

    # --- the ledger ----------------------------------------------------------
    print("\nthe constants this table stands on (nothing below is typed in twice)")
    src = "experiments/paper/constants.json" if CONSTANTS_JSON.exists() else "not present"
    print(f"  W0's constants file: {src}")
    for k, v in consts.items():
        print(f"    {k:<34} {str(v['value']):>8} {v['unit']:<14} {v['grade']:<10} {v['source']}")
    for name, card in cards.items():
        for ctx, ref in sorted(card["per_ctx"].items()):
            bs = "  ".join(f"{b} {m['cells_s']/1e9:.3f} G/s" for b, m in ref["boundaries"].items())
            print(f"    {name} C={ctx:<6} {ref['chunks']:>2} chunks  "
                  f"{ref['cells_per_token']/1e6:7.2f} M cells/token   {bs}")
            print(f"    {'':<14} {ref['source']}")
        print(f"    {'':<14} {card['note']}")

    # --- calibration ---------------------------------------------------------
    print("\nthe calibration: one free parameter, two measured points, one residual dof")
    print("  alpha (committed cells per head/key/token of the attention grid)")
    print(f"    B5's standalone block AIR      {cal['alpha_b5']:.3f}   (`b5/out/block.json`, "
          "`alpha_grid`, the constant the model shipped with)")
    print(f"    solved from the sequence slope {alpha:.3f}   (seq5's two contexts, "
          "the composed chunk's own grid + its interaction columns)")
    print(f"  {'context':>8} {'measured M/token':>18} {'model @ B5 alpha':>18} "
          f"{'model @ solved':>16} {'residual':>10}")
    for ctx in sorted(cal["measured"]):
        m = cal["measured"][ctx]
        print(f"  {ctx:>8} {m/1e6:>18.2f} {m*(1+cal['raw_residual'][ctx])/1e6:>18.2f} "
              f"{m*(1+cal['residual'][ctx])/1e6:>16.2f} {cal['residual'][ctx]:>+9.2%}")
    print(f"  the model as it shipped is {min(cal['raw_residual'].values()):+.1%} to "
          f"{max(cal['raw_residual'].values()):+.1%} against the measured sequence;")
    print(f"  with the one constant replaced it is {min(cal['residual'].values()):+.2%} to "
          f"{max(cal['residual'].values()):+.2%}, and that is a *check*, not a fit:")
    print("  the slope fixes alpha and the two levels are then predicted, not matched.")

    # --- the shapes ----------------------------------------------------------
    print("\n(1) the shapes, and each one's own parameter check")
    print("  the check: the model's own arithmetic over the fetched config, plus the input")
    print("  embedding (which the proof never walks), against the vendor's published counts.")
    print(f"  {'model':<18} {'L':>4} {'d':>6} {'nh':>4} {'dff':>7} {'E':>5} {'k':>3} "
          f"{'total (model)':>14} {'published':>10} {'d':>7} "
          f"{'active (model)':>15} {'published':>10} {'d':>7}")
    rows = []
    for shape in SHAPES:
        L = sum(grp["layers"] for grp in shape["groups"])
        moe = [grp for grp in shape["groups"] if grp["experts"] > 1]
        E = moe[0]["experts"] if moe else 1
        k = moe[0]["topk"] if moe else 1
        dff = moe[0]["dff"] if moe else shape["groups"][0]["dff"]
        tot = params_of(shape, walked=True, embedding=True)
        act = params_of(shape, embedding=True)
        dt = tot / shape["check_total"] - 1 if shape.get("check_total") else None
        da = act / shape["check_active"] - 1 if shape.get("check_active") else None
        bad = (dt is not None and abs(dt) > 0.05) or (da is not None and abs(da) > 0.05)
        print(f"  {shape['name']:<18} {L:>4} {shape['d']:>6} {shape['nh']:>4} {dff:>7} "
              f"{E:>5} {k:>3} {g(tot):>14} {g(shape['check_total']):>10} {dt:>+6.1%} "
              f"{g(act):>15} {g(shape['check_active']):>10} {da:>+6.1%}"
              + ("  [check]" if bad else ""))
        rows.append(dict(shape=shape, total=tot, active=act, d_total=dt, d_active=da,
                         flagged=bad))
    print("  E and k include the shared expert where the architecture has one "
          "(always walked, always active).")
    for r in rows:
        if r["flagged"]:
            print(f"  [check] {r['shape']['name']}: total {r['d_total']:+.0%}, "
                  f"active {r['d_active']:+.0%} -- the fetched config does not reproduce "
                  "the published counts,")
            print("          so every number in this row's line is a shape the vendor has "
                  "not confirmed.")
    print("\n  where each shape comes from:")
    for shape in SHAPES:
        print(f"    {shape['name']:<18} {shape['source']}")

    # --- cells and throughput ------------------------------------------------
    priced = [price(s, cards, alpha, seq=args.seq, seq_batch=args.seq_batch, usd=usd,
                    target_tok_s=args.target, mib_per_chunk=mib_per_chunk, batch=batch)
              for s in SHAPES]

    print(f"\n(2) committed cells per token, S={args.seq} tokens/chunk, B={args.seq_batch} "
          "sequence(s)/proof")
    print(f"  {'model':<18} {'C=4K fresh':>12} {'C=16K fresh':>12} {'C=4K pre':>10} "
          f"{'C=16K pre':>10} {'walk %':>7} {'grid %':>7} {'chunks':>7}")
    for r in priced:
        a, b = r["contexts"][4096], r["contexts"][16384]
        print(f"  {r['name']:<18} {g(a['cells_per_token']):>12} {g(b['cells_per_token']):>12} "
              f"{g(a['preprocessed_per_token']):>10} {g(b['preprocessed_per_token']):>10} "
              f"{a['parts']['walk']/a['cells_per_token']*100:>6.1f}% "
              f"{a['parts']['attn']/a['cells_per_token']*100:>6.1f}% {a['chunks']:>7}")
    print("  `pre` is the preprocessed weight trace, committed offline but read once per")
    print("  proof.  It is NOT in the fresh count the measured rates divide, and it is")
    print("  where MoE's x E lands hardest -- see (4).")

    print("\n(2b) proof bytes: two estimates that disagree, and neither is measured above 8B")
    print(f"  {'model':<18} {'by chunk count':>15} {'by 0.15 MiB/B-active':>22} {'ratio':>8}")
    for r in priced:
        e = r["contexts"][4096]
        a, b = e["mib_per_token_chunks"], e["mib_per_token_params"]
        print(f"  {r['name']:<18} {a:>15.2f} {b:>22.2f} {b / a:>7.1f}x")
    print("  Left: (L+1) chunks x the measured 8B per-chunk proof size "
          f"({mib_per_chunk * TOKENS:.2f} MiB a chunk).")
    print("  Right: section 6.3's rule of thumb, 0.15 MiB per token per B active params,")
    print("  which is the same 8B measurement divided by 8.03 B and then scaled linearly.")
    print("  They cannot both be right.  Proof size is queries x columns x log(rows), and")
    print("  B6's packing measured columns *flat* from 1 to 512 instances -- a bigger model")
    print("  puts its extra work in rows, which the proof pays for logarithmically.  So the")
    print("  left column is the better structural argument and section 6.3's ~61 MiB/token")
    print("  at 405B is very likely an order of magnitude too high.  Neither is measured at")
    print("  frontier scale; the left column also holds the per-chunk size fixed, so it is")
    print("  a floor.  **Open: this needs one measured point at a wider shape.**")

    for cardname in ("H200", "H100"):
        card = cards[cardname]
        rate = usd[cardname]
        print(f"\n(3{'a' if cardname == 'H200' else 'b'}) {cardname}: per-card tok/s at "
              f"every boundary, cards for {args.target:.0f} tok/s, $/Mtok at "
              f"{('$%.2f/GPU-h' % rate) if rate else 'n/a (no reference rate)'}")
        for ctx in sorted(card["per_ctx"]):
            print(f"  C={ctx}")
            bnames = list(card["per_ctx"][ctx]["boundaries"])
            hdr = "".join(f"{b:>17}" for b in bnames)
            print(f"  {'model':<18}{hdr}   {'MiB/token':>10} {'$/Mtok (wall)':>14}")
            for r in priced:
                e = r["contexts"][ctx]
                cells = e["cards"].get(cardname)
                if not cells:
                    continue
                cs = ""
                for b in bnames:
                    m = cells[b]
                    cs += f"{m['tok_s']:>9.3f}/{m['cards_for_target']:<7}"
                wall = cells.get("pipelined loop") or cells.get("whole wall")
                dollars = wall["usd_per_mtok"]
                print(f"  {r['name']:<18}{cs}   {e['mib_per_token_chunks']:>10.2f} "
                      + (f"{dollars:>14,.0f}" if dollars else f"{'n/a':>14}"))
            print("   (each cell is tok/s on one card / cards needed for the target)")

    # --- MoE and the batch knob ----------------------------------------------
    print("\n(4) MoE against a dense model with the same *active* parameters, C=4K")
    print("  the mechanism (B9 probe 2): a proof walks every expert in full, and the walk")
    print("  is per weight -- exactly invariant to how many tokens used it.  So the weight")
    print("  side is x E while the accumulators are only x k.  B batches sequences into one")
    print("  proof; at B ~ E/k every expert sees S tokens again and the walk amortises")
    print("  exactly as a dense matmul's does at B = 1.")
    moes = [s for s in SHAPES if any(grp["experts"] > 1 for grp in s["groups"])]
    BS = (1, 4, 16, 64, 256)
    print("  the reference is the dense twin at B=1, held fixed, so the column reads")
    print("  'what the MoE proof costs against the dense proof the paper already prices'.")
    print(f"  {'model':<18} {'E/k':>6} " + "".join(f"{'B=%d' % b:>10}" for b in BS)
          + f"{'B=E/k':>10}")
    moe_rows = {}
    for s in moes:
        grp = next(grp for grp in s["groups"] if grp["experts"] > 1)
        ek = grp["experts"] / grp["topk"]
        cells = [moe_ratio(s, 4096, alpha, seq=args.seq, seq_batch=B, batch=batch)
                 for B in BS]
        same = [moe_ratio(s, 4096, alpha, seq=args.seq, seq_batch=B, ref_batch=B,
                          batch=batch) for B in BS]
        at_ek = moe_ratio(s, 4096, alpha, seq=args.seq, seq_batch=max(1, round(ek)),
                          batch=batch)
        moe_rows[s["name"]] = dict(ek=ek, sweep=dict(zip(BS, cells)),
                                   same_b=dict(zip(BS, same)), at_ek=at_ek)
        print(f"  {s['name']:<18} {ek:>6.1f} "
              + "".join(f"{c['ratio']:>9.3f}x" for c in cells)
              + f"{at_ek['ratio']:>9.3f}x")
    print("  It crosses 1.00 at B ~ E/k, which is the whole of section 6.4's counter, and")
    print("  keeps going: past B = E/k a batched MoE proof is *cheaper* than the unbatched")
    print("  dense proof the paper prices today, because the walk it inflated has gone.")
    print("\n  the same rows counting the preprocessed weight trace, which the measured")
    print("  boundary rates do NOT count:")
    for s in moes:
        r = moe_rows[s["name"]]
        print(f"  {s['name']:<18} {r['ek']:>6.1f} "
              + "".join(f"{r['sweep'][B]['ratio_with_pre']:>9.3f}x" for B in BS)
              + f"{r['at_ek']['ratio_with_pre']:>9.3f}x")
    print("\n  and against the dense twin at the *same* B -- the structural penalty, which")
    print("  batching shrinks but never removes:")
    for s in moes:
        r = moe_rows[s["name"]]
        print(f"  {s['name']:<18} {r['ek']:>6.1f} "
              + "".join(f"{r['same_b'][B]['ratio']:>9.3f}x" for B in BS))

    print("\n(5) what B buys in absolute terms, C=4K, whole-wall boundary on the H200")
    print(f"  {'model':<18} " + "".join(f"{'B=%d' % b:>14}" for b in (1, 4, 16, 64, 256)))
    batch_rows = {}
    for s in SHAPES:
        vals = []
        for B in (1, 4, 16, 64, 256):
            c = model_cells(s, 4096, seq=args.seq, seq_batch=B, alpha=alpha,
                            batch=batch)["total"]
            vals.append(h200["per_ctx"][4096]["boundaries"]["whole wall"]["cells_s"] / c)
        batch_rows[s["name"]] = vals
        print(f"  {s['name']:<18} " + "".join(f"{v:>13.3f}" for v in vals))
    print("  tok/s per card; B sequences are proven at once, so this is throughput, not")
    print("  latency, and the wall per proof grows with B.")

    # --- E11 -----------------------------------------------------------------
    e11 = None
    if fit:
        print("\n(6) the full prompt+output statement (E11, decided 2026-08-31)")
        print("  B9's measured fit, one layer chunk at C = S, 8B shape:")
        print(f"    fresh ~ {g(fit['fixed'])}+ {fit['per_token']/1e6:.2f} M x S "
              f"+ {fit['per_pair']:.1f} x S^2")
        print(f"    fitted on S = {fit['points']}; "
              f"S = {fit['missing']} MISSING (killed at 133.8 GB host RSS, empty JSON).")
        print("    **Two points, three coefficients: zero residual degrees of freedom.**")
        print("    Everything in this section inherits that; it is a sized direction, not")
        print("    a measurement, and the paper must say so wherever the multiple appears.")
        e11 = {}
        print(f"\n  {'N positions':>12} {'S chunk':>8} {'chunks':>7} "
              f"{'chunking overhead':>18} {'x today at that C':>18}")
        for ctx in (4096, 16384):
            for sc in (512, 1024):
                f = e11_factor(fit, ctx, TOKENS, sc)
                base = h200["per_ctx"][ctx]["fresh"]
                e11[(ctx, sc)] = dict(factor=f["cells"] / base,
                                      chunk_overhead=f["chunk_overhead"],
                                      vs_monolith=f["vs_monolith"], cells=f["cells"])
                print(f"  {ctx:>12} {sc:>8} {ctx // sc:>7} "
                      f"{f['chunk_overhead']:>17.1%} {f['cells'] / base:>17.2f}x")
        print("  Section 9 quotes +13.7% (4K) / +11.6% (16K) for the chunking overhead: the")
        print("  4K figure is the N=4096 / S=1024 row above (this module computes 13.6%),")
        print("  and the 16K one lands between the S=1024 and S=2048 chunkings, at an S the")
        print("  record does not name.")
        print("  Reading: proving every position of an N-token prompt+output rectangle,")
        print("  instead of 512 decoded positions against a declared prefix, multiplies the")
        print("  committed cells by the figures above -- so every tok/s in (3) divides and")
        print("  every $/Mtok multiplies by them, at the same N.")
        print(f"\n  {'model':<18} {'C=4K x':>8} {'cells/token':>12} {'H100 tok/s':>11} "
              f"{'$/Mtok':>10}   (E11 at N=4096, S=1024, whole wall)")
        fac = e11[(4096, 1024)]["factor"]
        for r in priced:
            e = r["contexts"][4096]
            c = e["cells_per_token"] * fac
            h = h100["per_ctx"][4096]["boundaries"]["pipelined loop"]["cells_s"] / c
            d = usd_per_mtok(h, usd["H100"])
            print(f"  {r['name']:<18} {fac:>7.2f}x {g(c):>12} {h:>11.4f} "
                  + (f"{d:>10,.0f}" if d else f"{'n/a':>10}"))
        print("  The multiple is transported from the 8B fit, not recomputed per shape:")
        print("  B9's prefill runs exist only at the 8B shape, one layer, the naive")
        print("  (unmasked) grid.  Per-shape it would move with nh/L and with the causal")
        print("  split, both unmeasured.")

    # --- the memory wall -----------------------------------------------------
    ref = h200["per_ctx"][4096]
    gib_per_gcell = ref["peak_mib"] / 1024.0 / (ref["peak_chunk_cells"] / 1e9)
    print(f"\n(7) does a layer chunk still fit?  measured: the biggest seq5 chunk is "
          f"{ref['peak_chunk_cells']/1e9:.2f} G cells")
    print(f"  at {ref['peak_mib']:,} MiB of card, i.e. ~{gib_per_gcell:.1f} GiB per G cell "
          "(one point, C=4K, 8B -- approximate)")
    print(f"  {'model':<18} {'G cells/layer chunk':>20} {'GiB implied':>12} "
          f"{'80 GB':>7} {'141 GB':>7}   at S=512")
    fits = {}
    for s in SHAPES:
        per_layer = 0.0
        for grp in s["groups"]:
            c = model_cells(dict(s, groups=[grp]), 4096, seq=args.seq,
                            seq_batch=args.seq_batch, alpha=alpha, batch=batch,
                            with_head=False)
            per_layer = max(per_layer, c["total"] * args.seq / grp["layers"])
        gib = per_layer / 1e9 * gib_per_gcell
        fits[s["name"]] = dict(cells=per_layer, gib=gib)
        print(f"  {s['name']:<18} {per_layer/1e9:>20.2f} {gib:>12.1f} "
              f"{'yes' if gib < 80 else 'NO':>7} {'yes' if gib < 141 else 'NO':>7}")
    print("  A chunk that does not fit forces a smaller S, which raises the walk term")
    print("  (~1/S) and multiplies the proof bytes -- the frontier rows above assume the")
    print("  measured S=512 and are optimistic wherever this column says NO.")

    OUT.mkdir(exist_ok=True)
    doc = dict(
        constants=consts, cards={k: v for k, v in cards.items()},
        calibration=cal, alpha=alpha, seq=args.seq, seq_batch=args.seq_batch,
        prefill_fit=fit,
        shapes=[dict(name=s["name"], tag=s["tag"], source=s["source"],
                     groups=s["groups"], d=s["d"], nh=s["nh"], nkv=s["nkv"],
                     hd=s["hd"], vocab=s["vocab"],
                     params_total=params_of(s, walked=True), params_active=params_of(s),
                     check_total=s.get("check_total"), check_active=s.get("check_active"))
                for s in SHAPES],
        rows=priced,
        moe={k: dict(ek=v["ek"],
                     sweep={str(b): dict(ratio=c["ratio"], ratio_with_pre=c["ratio_with_pre"],
                                         moe=c["moe"], dense_active=c["dense_active"],
                                         ratio_same_b=v["same_b"][b]["ratio"],
                                         ratio_same_b_with_pre=v["same_b"][b]["ratio_with_pre"])
                            for b, c in v["sweep"].items()},
                     at_ek=dict(ratio=v["at_ek"]["ratio"],
                                ratio_with_pre=v["at_ek"]["ratio_with_pre"]))
             for k, v in moe_rows.items()},
        batch_tok_s=batch_rows,
        e11={f"{c}/{s}": v for (c, s), v in (e11 or {}).items()},
        chunk_fit=fits)
    pathlib.Path(args.json).write_text(json.dumps(doc, indent=1, default=str))
    print(f"\nwrote {rel(args.json)}")
    return doc


if __name__ == "__main__":
    main()
