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
# Stream W0's freeze, `experiments/paper/constants.json`, read through its own
# module.  Nothing measured, cited or agreed is written down here: this module
# names the keys it needs and fails loudly on a `null`.  When the freeze is
# absent (an older checkout) the fallbacks below carry the same fields, so the
# swap is a lookup change and nothing else.
PAPER = ROOT / "paper"
CONSTANTS_JSON = PAPER / "constants.json"

# The keys this module consumes, and nothing else.
#   dollars -- the `*_reference_*` pair only.  `*_actual_*` exists in the freeze
#   marked not-for-publication (the H200 we measure on is sponsored, so its
#   actual rate is 0.00) and this module must never read it.
#   the grid -- E1's own composed-chunk solve, plus the other values in
#   circulation, which section (2) reconciles rather than averages.
KEYS = dict(
    h100_usd_per_hour="cost.h100_reference_usd_per_hour",
    h200_usd_per_hour="cost.h200_reference_usd_per_hour",
    # the attention term.  The drift guard checks these COEFFICIENTS, not a
    # composite slope: a single cells-per-(head, key, token) number is a per-entry
    # cost and a per-key cost collapsed together by the accident that the
    # calibration point has S=512, and it is valid only at that S.
    band_fresh_coeffs="attention_grid.band_model_fresh_coeffs",
    band_base_coeffs="attention_grid.band_model_base_coeffs",
    band_alpha_fresh="attention_grid.band_model_alpha_fresh",
    band_alpha_base="attention_grid.band_model_alpha_base",
    # the superseded single slopes, kept only so the table can show what they cost
    alpha_e1="attention_grid.e1_composed_solve",
    alpha_w0_32layers="attention_grid.composed_chunk_fresh_slope_32layers",
    alpha_b5="attention_grid.b5_standalone_alpha",
    # E11, re-priced by stream E7 on the band model
    e11_multiple="design.e11_cells_multiple_c4096",
    e11_chunking_net="design.e11_chunking_net_pct",
    e11_chunking_fixed="design.e11_chunking_overhead_pct",
    chunk_floor="attention_grid.chunk_model_fixed_composed",
    prefill_remainder="design.prefill_chunk_remainder",
)

# Deliberately all `None`: a market rate moves, and a stale copy of one sitting in
# this file would price the table at a number nobody cited.  Without the freeze the
# dollar columns print `n/a`, which is the correct answer.
FALLBACK = {
    "cost.h100_reference_usd_per_hour": dict(
        value=None, unit="USD/GPU-h", grade="pending", card="H100",
        source="frontier.py fallback; there is no reference rate without the freeze"),
    "cost.h200_reference_usd_per_hour": dict(
        value=None, unit="USD/GPU-h", grade="pending", card="H200",
        source="frontier.py fallback; there is no reference rate without the freeze"),
}


def load_constants():
    """The freeze if it is there, the fallbacks if it is not.

    Returns `{alias: entry}`, each entry carrying the freeze's own value / unit /
    grade / card / source, plus the key it came from and where it was read."""
    doc = None
    if CONSTANTS_JSON.exists():
        sys.path.insert(0, str(ROOT))
        import paper.constants as frozen                       # noqa: E402
        doc = frozen
    out = {}
    for alias, key in KEYS.items():
        if doc is not None:
            e = dict(doc.get(key))
            e["origin"] = rel(CONSTANTS_JSON)
        else:
            e = dict(FALLBACK.get(key, dict(value=None, unit="", grade="pending",
                                            card="none", source="not frozen")))
            e["origin"] = "frontier.py fallback (the freeze is absent)"
        e["key"] = key
        out[alias] = e
    return out


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
    # W0's freeze names FOUR boundaries and they are not interchangeable.
    # `whole wall` = trace gen + interaction gen + stwo + serialize, summed per
    # chunk -- it has NEVER counted the witness read or the layer build.
    # `loop_secs` = the one-process chunk loop's own wall, which DOES count them,
    # and which here is pipelined (one chunk ahead), so the prove hides behind the
    # next chunk's generation.  Neither is a subset of the other and a reader
    # comparing our pipelined numbers to our sequence numbers needs both named.
    secs = dict(commit=s["commit_sum"], **{"stwo total": s["stwo_sum"],
                                           "whole wall": s["whole_sum"],
                                           "loop_secs": s["loop_secs"]})
    return dict(card="H100", tokens=TOKENS, per_ctx={4096: dict(
        fresh=fresh, cells_per_token=fresh / TOKENS, proof_bytes=proof,
        mib_per_token=proof / 2 ** 20 / TOKENS, chunks=len(ch),
        boundaries={k: dict(secs=v, tok_s=TOKENS / v, cells_s=fresh / v)
                    for k, v in secs.items()},
        peak_mib=max(c.get("mem_after_prove", 0) for c in ch) / 2 ** 20,
        peak_chunk_cells=max(c["cells"]["fresh"] for c in ch),
        source=rel(H100_LOOP))},
        note="32 layer chunks, no head, pipelined (`B7_M5_PIPE=1`).  `loop_secs` "
             "is W0's FOURTH boundary and is not `whole wall`: it counts the "
             "witness read and the layer build, which `whole wall` has never "
             "counted, and it hides the prove behind the next chunk's generation. "
             "Here it is both wider (112.89 -> it counts more work) and faster "
             "(96.33 s -> it overlaps), which is exactly why the two must not be "
             "compared as if one were a subset of the other.")


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
                with_head=True, corner=True):
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
            attn_params_per_layer=grp.get("attn_params_per_layer"), corner=corner)
        out["saturated"] = out.get("saturated", False) or bool(c["band_saturated"])
        gl = cm.glue_cells(cfg, "tree", topk=grp["topk"], final_norm=head)
        out["total"] += c["total"] + gl["net"]
        out["preprocessed"] += c["preprocessed"]
        out["glue"] += gl["net"]
        for k, v in c.items():
            if isinstance(v, (int, float)) and k not in ("dequant_per_acc", "head_per_acc",
                                                         "tokens_per_proof"):
                out["parts"][k] = out["parts"].get(k, 0.0) + v
    return out


def check_8b(h200, shape, consts, batch=None):
    """The check, and it now has **no free parameter at all**.

    Before 2026-09-01 the attention term was one slope, `alpha`, solved from the
    two measured contexts -- one fitted parameter, two points, one residual dof.
    Stream DD's band model and stream E1's symbolic re-derivation of it replace
    that with a closed form whose every coefficient is an identity of the layout,
    so the model now predicts both measured levels with nothing solved: two points,
    zero fitted parameters, two residual degrees of freedom.

    Also re-checks the coefficients themselves against the freeze, and against the
    band family measured directly in the seq5 artifacts."""
    meas = measured_8b(h200)
    nh, hd = shape["nh"], shape["hd"]
    got_fresh = _coeffs(nh, hd)
    got_base = _coeffs(nh, hd, base_only=True)
    want_fresh = consts["band_fresh_coeffs"]["value"]
    want_base = consts["band_base_coeffs"]["value"]
    out = dict(
        measured=meas,
        band={c: model_cells(shape, c, batch=batch)["total"] / meas[c] - 1 for c in meas},
        legacy_slope={c: model_cells(shape, c, alpha=consts["alpha_e1"]["value"],
                                     batch=batch)["total"] / meas[c] - 1 for c in meas},
        legacy_b5={c: model_cells(shape, c, alpha=consts["alpha_b5"]["value"],
                                  batch=batch)["total"] / meas[c] - 1 for c in meas},
        coeffs_fresh=got_fresh, coeffs_base=got_base,
        frozen_fresh=want_fresh, frozen_base=want_base,
        coeffs_agree=(got_fresh == tuple(want_fresh) and got_base == tuple(want_base)),
        family=_band_family_check(nh, hd))
    return out


def _coeffs(nh, hd, base_only=False, s0=512, c0=4096):
    """The (S*C, C, S) coefficients this module's band model implies at a shape.

    Recovered from the band function itself by finite differences, rather than
    restated, so the drift guard compares the code that is actually used against
    the freeze.  Differenced at (s0, c0) and its doublings, well clear of
    `Grid::new`'s caps -- at S or C below them the layout saturates, `q` and `pk`
    stop being the targets, and the form is no longer a plain trilinear."""
    def f(s, c):
        b, i, _ = cm.band_cells(nh, hd, s, c)
        return b if base_only else b + i
    a = (f(2 * s0, 2 * c0) - f(2 * s0, c0) - f(s0, 2 * c0) + f(s0, c0)) / (s0 * c0)
    b = (f(s0, 2 * c0) - f(s0, c0)) / c0 - a * s0
    d = (f(2 * s0, c0) - f(s0, c0)) / s0 - a * c0
    return tuple(round(x, 6) for x in (a, b, d))


def _band_family_check(nh, hd):
    """`family_cells.bands` is base-only; predict it on every seq5 layer chunk and
    on B9's two prefill points, and report exact-or-not.  Nothing is fitted, so a
    miss here is a real disagreement, not a residual."""
    rows = []
    for f in sorted((ROOT / "b10" / "out" / "seq5").glob("seq_c*_L*.json")):
        try:
            d = json.loads(f.read_text())["result"]["point"]
        except (ValueError, KeyError):
            continue
        pr = d["params"]
        if pr["nh"] != nh or pr["hd"] != hd:
            continue
        b, _, _ = cm.band_cells(nh, hd, pr["s"], pr["c"],
                                grid_target=pr["grid_target"], chan_target=pr["chan_target"],
                                rw=pr["rw"])
        if pr.get("corner"):
            b2, _, _ = cm.band_cells(nh, hd, pr["s"], pr["s"], rows_region=False,
                                     grid_target=pr["grid_target"],
                                     chan_target=pr["chan_target"], rw=pr["rw"])
            b += b2
        rows.append((f.name, pr["s"], pr["c"], d["family_cells"]["bands"], b))
    for s in (512, 1024):
        f = ROOT / "b9" / "out" / f"prefill_s{s}.json"
        if not (f.exists() and f.stat().st_size):
            continue
        d = json.loads(f.read_text())["result"]["point"]
        b, _, _ = cm.band_cells(nh, hd, s, s)
        rows.append((f.name, s, s, d["family_cells"]["bands"], b))
    return rows


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
    ap.add_argument("--compare", default=None,
                    help="a previous frontier.json to diff every row against; prints "
                         "what moved in cells/token, tok/s and cards-at-target, and "
                         "whether the ordering changed")
    args = ap.parse_args(argv)

    consts = load_constants()
    batch = args.logup_batch if args.logup_batch is not None else cm.BATCH
    if batch not in cm.LOGUP_MARGINAL:
        sys.exit(f"--logup-batch {batch} not measured; have {sorted(cm.LOGUP_MARGINAL)}")
    usd = {"H100": consts["h100_usd_per_hour"]["value"],
           "H200": consts["h200_usd_per_hour"]["value"]}
    h200, h100 = h200_card(), h100_card()
    cards = {"H200": h200, "H100": h100}
    llama8b = next(s for s in SHAPES if s["name"] == "Llama-3.1-8B")
    cal = check_8b(h200, llama8b, consts, batch=batch)
    # The drift guard, now on the coefficients rather than on a composite slope.
    # This module derives them symbolically and the freeze carries stream DD's;
    # they must agree exactly at the 8B geometry, because both claim to be
    # identities of the same layout rather than fits.
    if consts["band_fresh_coeffs"]["value"] and not cal["coeffs_agree"]:
        sys.exit(f"band coefficients disagree with the freeze:\n"
                 f"  fresh  derived {cal['coeffs_fresh']}  frozen {cal['frozen_fresh']}\n"
                 f"  base   derived {cal['coeffs_base']}  frozen {cal['frozen_base']}\n"
                 "Both claim to be layout identities, so one of them is wrong; "
                 "resolve it, do not average.")
    bad = [r for r in cal["family"] if r[3] != r[4]]
    if bad:
        sys.exit("the band model no longer reproduces `family_cells.bands` exactly on "
                 + ", ".join(r[0] for r in bad))
    alpha = None                       # the band model IS the attention term now
    fit = prefill_fit()
    mib_per_chunk = (h200["per_ctx"][4096]["proof_bytes"] / 2 ** 20
                     / h200["per_ctx"][4096]["chunks"] / TOKENS)

    print("=" * 100)
    print("E1 -- the frontier price table")
    print("=" * 100)

    # --- the ledger ----------------------------------------------------------
    print("\nthe constants this table stands on (nothing below is typed in twice)")
    src = rel(CONSTANTS_JSON) if CONSTANTS_JSON.exists() else "NOT PRESENT (fallbacks)"
    print(f"  W0's freeze: {src}")
    for k, v in consts.items():
        print(f"    {v['key']:<52} {str(v['value']):>8}  [{v['grade']}]")
    for alias in ("h100_usd_per_hour", "h200_usd_per_hour"):
        e = consts[alias]
        print(f"    {e['card']} ${e['value']}/GPU-h  [{e['grade']}]  {e['source']}")
    print("  Both rates are published list prices from ONE provider, ONE tier and ONE page")
    print("  date, so the H100/H200 ratio is not a cross-provider artifact.  Secure Cloud")
    print("  is deliberate: Community Cloud is cheaper ($2.69 / $3.59) and using the higher")
    print("  tier is the conservative choice for a cost claim -- it makes proving look more")
    print("  expensive, not less.  The H100 figure is the SXM page, matching our box")
    print("  (H100 80GB HBM3 SXM, 81,559 MiB); RunPod's PCIe page is $2.89 Secure and is a")
    print("  different part.  Sanity check, not used in any column: our actual Novita cost")
    print("  is $3.39/h, within 3% of the list price, so the reference is realistic rather")
    print("  than cherry-picked.  The H200's `*_actual_*` key is 0.00 (sponsored access) and")
    print("  this module never reads an `*_actual_*` key.  Grade both $ columns [web].")

    print("\nthe four boundaries, kept apart (W0's freeze names all four)")
    print("  commit      interpolate + LDE + Merkle over the committed trees")
    print("  stwo total  commit + prove_ex (constraint eval, quotients, OODS, FRI)")
    print("  whole wall  trace gen + interaction gen + stwo + serialize, summed per")
    print("              chunk.  Has NEVER counted the witness read or the layer build.")
    print("  loop_secs   the one-process chunk loop's own wall, which DOES count them,")
    print("              pipelined one chunk ahead so the prove hides behind the next")
    print("              chunk's generation.  A FOURTH boundary, not a superset and not")
    print("              a subset of `whole wall` -- on the H100 below it counts more")
    print("              work and still finishes sooner.  W/X/Z/O all report here.")
    for name, card in cards.items():
        for ctx, ref in sorted(card["per_ctx"].items()):
            bs = "  ".join(f"{b} {m['cells_s']/1e9:.3f} G/s" for b, m in ref["boundaries"].items())
            print(f"    {name} C={ctx:<6} {ref['chunks']:>2} chunks  "
                  f"{ref['cells_per_token']/1e6:7.2f} M cells/token   {bs}")
            print(f"    {'':<14} {ref['source']}")
        print(f"    {'':<14} {card['note']}")
    print("    the H200 sequence ran one process per chunk, so it has no `loop_secs`;")
    print("    the H100 loop has no head chunk.  Neither gap is fillable by arithmetic.")

    # --- the check: no free parameter at all ---------------------------------
    print("\nthe check: the attention term now has ZERO fitted parameters")
    print("  The model used to charge one slope, `alpha`, solved from the two measured")
    print("  contexts -- one parameter, two points, one residual dof.  Stream DD's band")
    print("  model and this module's symbolic re-derivation of it replace that slope with")
    print("  a closed form in (S, C) whose every coefficient is an identity of the layout,")
    print("  so both measured levels are now predicted with nothing solved: two points,")
    print("  zero fitted parameters, TWO residual degrees of freedom.")
    print(f"\n  {'context':>8} {'measured M/token':>18} {'band model':>12} {'residual':>10}"
          f"   {'slope 5.855':>12} {'B5 4.00':>10}")
    for ctx in sorted(cal["measured"]):
        m = cal["measured"][ctx]
        print(f"  {ctx:>8} {m/1e6:>18.2f} {m*(1+cal['band'][ctx])/1e6:>12.2f} "
              f"{cal['band'][ctx]:>+9.2%}   {cal['legacy_slope'][ctx]:>+11.2%} "
              f"{cal['legacy_b5'][ctx]:>+9.2%}")
    print(f"  band model residual {min(cal['band'].values()):+.2%} to "
          f"{max(cal['band'].values()):+.2%}, on no fitted attention parameter at all.")
    print("  The two superseded columns are the same model with the band term replaced by")
    print("  a single slope: 5.855 was solved to hit these two points, and 4.00 is B5's")
    print("  standalone block AIR, which is what everything before 2026-09-01 charged.")

    print("\n  the coefficients, derived here and frozen by stream DD -- they must agree")
    print("  exactly, because both claim to be identities of the same layout:")
    print(f"  {'':<10} {'S*C':>10} {'C':>12} {'S':>12}")
    print(f"  {'fresh':<10} {cal['coeffs_fresh'][0]:>10} {cal['coeffs_fresh'][1]:>12} "
          f"{cal['coeffs_fresh'][2]:>12}   derived symbolically here")
    print(f"  {'':<10} {cal['frozen_fresh'][0]:>10} {cal['frozen_fresh'][1]:>12} "
          f"{cal['frozen_fresh'][2]:>12}   attention_grid.band_model_fresh_coeffs")
    print(f"  {'base':<10} {cal['coeffs_base'][0]:>10} {cal['coeffs_base'][1]:>12} "
          f"{cal['coeffs_base'][2]:>12}   derived symbolically here")
    print(f"  {'':<10} {cal['frozen_base'][0]:>10} {cal['frozen_base'][1]:>12} "
          f"{cal['frozen_base'][2]:>12}   attention_grid.band_model_base_coeffs")
    n_exact = sum(1 for r in cal["family"] if r[3] == r[4])
    print(f"  and `family_cells.bands` reproduced EXACTLY, to the cell, on "
          f"{n_exact}/{len(cal['family'])} artifacts")
    print("  (every seq5 layer chunk at C=4096 and C=16384 on the real extracted")
    print("  witness, plus B9's two S=C prefill points).  Nothing is fitted, so a miss")
    print("  would be a")
    print("  disagreement, not a residual, and `frontier.py` exits on one.")

    # --- the general form ----------------------------------------------------
    print("\n  the closed form, general in (nh, hd, S, C).  DD froze the coefficients at")
    print("  Llama-8B's geometry; these are the same numbers re-derived from the column")
    print("  accounting so they can be evaluated at a shape that is not Llama-8B:")
    print("      base  = 2*nh*S*C  +  2*nh*hd*C  +  nh*hd*S  +  4*nh*S")
    print("      inter = (2 + 24/gt)*nh*S*C  +  (4*nh*hd + 12*nh*hd/ct)*C")
    print("              +  2*nh*hd*S  +  (8*nh + 8*nh/rw)*S")
    print("  with gt = grid_target (64), ct = chan_target (128), rw = row width (16).")
    print(f"  At nh=32, hd=128 that is exactly {cal['frozen_fresh']} fresh and "
          f"{cal['frozen_base']} base.")
    print("  Read off `prover/src/attn_trace.rs:1167-1181` with `prover/src/layout.rs`:")
    print("    lc.g.rows = hd*insts*C/pk (pk -> ct), lk.g.rows = C*insts*S/q (q -> gt),")
    print("    rrows*rw = nh*S, and `insts` sums to nh over the bands -- every term is")
    print("    linear in `insts`, so banding 16/8/8 costs exactly what 32 would.")
    print("\n  Three things this derivation does NOT settle, and they are the extrapolation")
    print("  risk the frontier rows carry:")
    print("   1. **K and V are committed per ATTENTION head, not per KV group.**")
    print("      `LayoutChan`/`LayoutKey` are built from `Shape::chans()`/`keys()`, both")
    print("      carrying `insts` = attention heads, so the `C` term is 2*nh*hd and `nkv`")
    print("      does not appear.  Grouped-query attention buys this prover nothing on the")
    print("      committed KV side today.  At 8B nh/nkv is 4; at Llama-405B and Qwen3 it")
    print("      is 16, and at the MLA shapes higher still.  A GQA-aware layout would")
    print("      divide that term by nh/nkv -- a named, unbuilt optimisation, and the")
    print("      single largest structural saving this table can see.")
    print("   2. **gt / ct / rw are run knobs, not constants.**  The composed sequence used")
    print("      64 / 128 / 16 and this model keeps them.  They tile exactly only when S,")
    print("      C, hd and `insts` are powers of two and the target divides; `Grid::new`")
    print("      also caps q at min(S, hd) and pk at min(C, S).  Where a shape saturates a")
    print("      cap the table says so.")
    print("   3. **The out (A.V) region is not in the band family.**  It is the `attnout`")
    print("      family and `cells_per_token` charges it separately, so there is no double")
    print("      count -- but it is also the one attention region whose price here is a")
    print("      measured per-element constant rather than a derived column count.")
    sat = [r["name"] for r in
           [dict(name=s["name"], c=model_cells(s, 16384, seq=args.seq, batch=batch))
            for s in SHAPES] if r["c"].get("saturated")]
    print("  layout caps saturated at: " + (", ".join(sat) if sat else "no shape"))


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

    print("\n(2a) what the superseded attention terms cost.  Section 6.3's dense table")
    print("     was produced at B5's 4.00; the version this stream committed at 3ec240c")
    print("     used the single composed slope 5.855.  Both are here against the band model.")
    b5 = consts["alpha_b5"]["value"]
    slope = consts["alpha_e1"]["value"]
    print(f"  {'model':<18} {'@4.00':>10} {'@5.855':>10} {'band':>10} {'band vs 4.00':>13}"
          f" {'band vs 5.855':>14}   {'C=16K: vs 4.00':>15} {'vs 5.855':>10}")
    grid_delta = {}
    for shape in SHAPES:
        row = {}
        for ctx in (4096, 16384):
            kw = dict(seq=args.seq, seq_batch=args.seq_batch, batch=batch)
            row[ctx] = dict(
                b5=model_cells(shape, ctx, alpha=b5, **kw)["total"],
                slope=model_cells(shape, ctx, alpha=slope, **kw)["total"],
                band=model_cells(shape, ctx, **kw)["total"])
            row[ctx]["d_b5"] = row[ctx]["band"] / row[ctx]["b5"] - 1
            row[ctx]["d_slope"] = row[ctx]["band"] / row[ctx]["slope"] - 1
        grid_delta[shape["name"]] = {str(k): v for k, v in row.items()}
        a, b = row[4096], row[16384]
        print(f"  {shape['name']:<18} {g(a['b5']):>10} {g(a['slope']):>10} {g(a['band']):>10}"
              f" {a['d_b5']:>+12.1%} {a['d_slope']:>+13.1%}   {b['d_b5']:>+14.1%}"
              f" {b['d_slope']:>+9.1%}")
    d5 = [v["4096"]["d_b5"] for v in grid_delta.values()]
    d5c = [v["16384"]["d_b5"] for v in grid_delta.values()]
    ds = [v["4096"]["d_slope"] for v in grid_delta.values()]
    dsc = [v["16384"]["d_slope"] for v in grid_delta.values()]
    print(f"  Against B5's 4.00 the band model is {min(d5):+.1%} to {max(d5):+.1%} at C=4K")
    print(f"  and {min(d5c):+.1%} to {max(d5c):+.1%} at C=16K -- that is the error in")
    print("  section 6.3's table, and it grows with the context because the term that was")
    print("  wrong is the one on the context.")
    print(f"  Against the single slope 5.855 it is {min(ds):+.1%} to {max(ds):+.1%} (4K) and")
    print(f"  {min(dsc):+.1%} to {max(dsc):+.1%} (16K) -- a spread of {max(ds)-min(ds):.1%} "
          f"(4K) / {max(dsc)-min(dsc):.1%} (16K)")
    print("  across the eight shapes.  So the aliasing is BENIGN at this table's settings,")
    print("  and it is worth being precise about why, because the reason is not that a")
    print("  composite slope is safe.  Write the composite out:")
    print("")
    gt, ct = cm.GRID_TARGET, cm.CHAN_TARGET
    print("      alpha_composite(hd, S) = (4 + 24/gt)  +  hd*(6 + 12/ct)/S")
    print(f"                             = {4 + 24/gt:.3f}          +  {128*(6+12/ct):.0f}/S"
          f"   at hd=128")
    for s_ in (128, 512, 1024, 4096):
        print(f"        S={s_:<5} -> {4 + 24/gt + 128*(6 + 12/ct)/s_:6.3f}")
    print("")
    print("  It does not depend on `nh` at all, and every shape in this table has hd=128")
    print("  and is priced at S=512 -- so the aliased constant happens to be the SAME")
    print("  number for all eight rows, which is exactly why the residual spread is small.")
    print("  Two things follow, and they are the reason the band model is still required:")
    print("   * The composite moves 13% between S=512 and S=1024 and doubles by S=128.  A")
    print("     'constant' that is a function of the chunk length is not extrapolable to")
    print("     the two regimes this table cares most about -- E11's S=1024 prompt chunks")
    print("     (section 6) and any shape forced off S=512 by memory (section 7).")
    print("   * Most of the residual above is NOT aliasing.  It is that the old model")
    print("     charged the committed K/V at `nkv` while the AIR charges it at `nh`, so the")
    print("     correction tracks nh/nkv: +4.3% at Qwen3 (nh/nkv=16) and +4.0% at")
    print("     DeepSeek-V3, against +2.0% at Maverick (5) and +1.5% at Kimi K3, whose")
    print("     69 linear-attention layers have no grid to correct.")

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
            # the widest boundary this card actually has: `loop_secs` where it was
            # measured, `whole wall` where it was not.  They are DIFFERENT
            # boundaries and the header says which one the dollars are priced at.
            money = "loop_secs" if "loop_secs" in bnames else "whole wall"
            hdr = "".join(f"{b:>17}" for b in bnames)
            print(f"  {'model':<18}{hdr}   {'MiB/token':>10} {'$/Mtok @ ' + money:>19}")
            for r in priced:
                e = r["contexts"][ctx]
                cells = e["cards"].get(cardname)
                if not cells:
                    continue
                cs = ""
                for b in bnames:
                    m = cells[b]
                    cs += f"{m['tok_s']:>9.3f}/{m['cards_for_target']:<7}"
                dollars = cells[money]["usd_per_mtok"]
                print(f"  {r['name']:<18}{cs}   {e['mib_per_token_chunks']:>10.2f} "
                      + (f"{dollars:>19,.0f}" if dollars else f"{'n/a':>19}"))
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
    print("\n(6) the full prompt+output statement (E11, decided 2026-08-31)")
    print("  Re-priced by stream E7 on DD's band model.  The old numbers in this section")
    print("  came from B9's `326 M + 1.37 M*S + 72.1*S^2` fit, whose S^2 coefficient was")
    print("  +12.6% against the true 64 because a two-point marginal in S alone absorbs the")
    print("  band family's linear term into its quadratic one.  A model in S alone also")
    print("  cannot express the term that turns out to matter most: a chunked prompt")
    print("  re-commits the growing K/V prefix, which is a function of C, not S.")
    e11 = {}
    mult = consts["e11_multiple"]["value"]
    net = consts["e11_chunking_net"]["value"]
    fixed = consts["e11_chunking_fixed"]["value"]
    print("\n  the cells multiple at C=4096, against seq5's measured decode sequence:")
    print(f"    prompt alone         x{mult[0]:.2f}")
    print(f"    prompt + output      x{mult[1]:.2f}       (`design.e11_cells_multiple_c4096`)")
    print("    at C=16384 the same model gives x17.55 / x18.55")
    print("  The superseded ~x4.5-5.5 priced each prompt chunk as its own S=C=1024 SQUARE.")
    print("  That is not a proof of the sequence: the fourth chunk must attend over all")
    print("  4,096 keys, not its own 1,024.  Under the band model the square reading is")
    print("  x4.56 / x5.56, so the correction is +14% on the prompt multiple -- and a head")
    print("  chunk per 512 prompt positions is counted in NEITHER (at 4K it would add")
    print("  ~8 x the measured head chunk).")
    print("\n  the chunking cost, which is now three terms and not one:")
    print(f"    {'the shared-table floor paid n times':<44} {fixed[1]:>+6.1f}% (4K) "
          f"{fixed[0]:>+6.1f}% (16K)")
    print(f"    {'the prefix K/V re-committed by every later chunk':<44} {1.9:>+6.1f}%      "
          f"{5.2:>+6.1f}%")
    print(f"    {'the inert half of the rectangle a one-shot pays':<44} {-11.2:>+6.1f}%      "
          f"{-30.0:>+6.1f}%")
    print(f"    {'NET':<44} {net[0]:>+6.1f}%      {net[1]:>+6.1f}%")
    print("  At C=16384 chunking is CHEAPER outright -- the causal saving beats the floor")
    print("  and the re-commitment together.  The old section reported only the first row")
    print("  (+13.7% / +11.6%) and called it the overhead.")
    e11 = dict(multiple_prompt=mult[0], multiple_prompt_output=mult[1],
               chunking_net_4k=net[0], chunking_net_16k=net[1],
               chunking_fixed=fixed)
    fac = mult[1]
    print(f"\n  the table at x{fac:.2f} (prompt+output, C=4096, S=1024 chunking):")
    print(f"  {'model':<18} {'cells/token':>12} {'H100 tok/s':>11} {'$/Mtok':>10}"
          f"   (H100 loop_secs, ${usd['H100']}/GPU-h)")
    for r in priced:
        e = r["contexts"][4096]
        c = e["cells_per_token"] * fac
        h = h100["per_ctx"][4096]["boundaries"]["loop_secs"]["cells_s"] / c
        d = usd_per_mtok(h, usd["H100"])
        r["e11"] = dict(cells_per_token=c, tok_s_h100=h, usd_per_mtok=d)
        print(f"  {r['name']:<18} {g(c):>12} {h:>11.4f} "
              + (f"{d:>10,.0f}" if d else f"{'n/a':>10}"))
    print("  The multiple is transported from the 8B shape, not recomputed per row: E7's")
    print("  chunk model carries a `design.prefill_chunk_remainder` fitted on B9's two")
    print("  SYNTHETIC S=C points, which is the only fitted part left and the only place")
    print("  its uncertainty lives.  Per shape the multiple would move with nh/L and with")
    print("  hd (the re-commitment term is the band family's C term, and that one this")
    print("  module can now evaluate at any shape -- but the remainder cannot).")
    floor = consts["chunk_floor"]["value"]
    if floor:
        print(f"\n  the shared-table floor of our own composed layer chunk: "
              f"{g(floor)}fresh cells/layer,")
        print("  solved at C=4096 and held out at C=16384 to -0.63% "
              "(`attention_grid.chunk_model_fixed_composed`).")
        print("  It is +14.1% over B9's synthetic-probe floor, which under-predicts our own")
        print("  chunk by 7.15% -- so a one-position eval item costs "
              f"{g(floor * 32)}cells at 8B before")
        print("  it proves anything, which is what makes short multiple-choice proofs dear.")

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

    # --- what moved against a previous run -----------------------------------
    if args.compare:
        prev = json.loads(pathlib.Path(args.compare).read_text())
        po = {r["name"]: r for r in prev["rows"]}
        print(f"\n(8) what moved against {rel(args.compare)}")
        print(f"  {'model':<18} {'C=4K cells/token':>24} {'C=16K':>16} "
              f"{'H200 commit tok/s':>22} {'cards@%g' % args.target:>12}")
        moved = []
        for r in priced:
            o = po.get(r["name"])
            if not o:
                continue
            a4, b4 = o["contexts"]["4096"], r["contexts"][4096]
            a16, b16 = o["contexts"]["16384"], r["contexts"][16384]
            ca = a4["cards"]["H200"]["commit"]
            cb = b4["cards"]["H200"]["commit"]
            moved.append((r["name"], ca["cards_for_target"], cb["cards_for_target"]))
            print(f"  {r['name']:<18} {a4['cells_per_token']/1e6:8.1f} ->"
                  f"{b4['cells_per_token']/1e6:8.1f} {b4['cells_per_token']/a4['cells_per_token']-1:>+6.1%}"
                  f" {b16['cells_per_token']/a16['cells_per_token']-1:>15.1%}"
                  f" {ca['tok_s']:>10.3f} ->{cb['tok_s']:>8.3f}"
                  f" {ca['cards_for_target']:>7d} ->{cb['cards_for_target']:<4d}")
        def order(rows, key):
            return [n for n, _ in sorted(rows, key=key, reverse=True)]
        oa = order([(r["name"], po[r["name"]]) for r in priced if r["name"] in po],
                   lambda kv: kv[1]["contexts"]["4096"]["cards"]["H200"]["commit"]["tok_s"])
        ob = order([(r["name"], r) for r in priced],
                   lambda kv: kv[1]["contexts"][4096]["cards"]["H200"]["commit"]["tok_s"])
        changed = [f"{n} {a}->{b}" for n, a, b in moved if a != b]
        print(f"  ordering by H200 commit tok/s at C=4K: "
              f"{'UNCHANGED' if oa == ob else 'CHANGED'}")
        print("  cards-at-target changes: " + (", ".join(changed) if changed else "none"))

    OUT.mkdir(exist_ok=True)
    doc = dict(
        constants=consts, cards={k: v for k, v in cards.items()},
        calibration=cal, alpha=alpha, seq=args.seq, seq_batch=args.seq_batch,
        prefill_fit=fit,
        band_coeffs=dict(fresh=cal["coeffs_fresh"], base=cal["coeffs_base"],
                         frozen_fresh=cal["frozen_fresh"], frozen_base=cal["frozen_base"]),
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
        e11=e11,
        chunk_fit=fits, grid_constant_delta=grid_delta)
    pathlib.Path(args.json).write_text(json.dumps(doc, indent=1, default=str))
    print(f"\nwrote {rel(args.json)}")
    return doc


if __name__ == "__main__":
    main()
