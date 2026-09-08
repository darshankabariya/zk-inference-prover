"""E4 -- the unfitted-scale prediction, registered BEFORE it is measured.

    uv run python -m harness.eval.predict.predict --constants PATH \
        --artifacts DIR --inputs DIR          # print the registration
    uv run python -m harness.eval.predict.predict ... --json   # and rewrite
                                              # out/registration.json

What this is.  `experiments/b2/cost_model.py` was built from gadget marginals
measured at exactly two model scales: Qwen2.5-0.5B (B1-B6.5) and Llama-3.1-8B
(B8-B10).  Nothing in it has ever been checked against a third.  This script
freezes what the model says a *third* scale costs, with a pass/fail bracket
fixed in advance, so that the later measurement is a test and not a fit.

The rule the plan attaches to E4 (eval-plan.md section 4, row 9) is in bold:
**"Must be registered before it is measured."**  At the moment this file was
committed no witness, no proof and no cell count existed at any of the three
target scales, on either box -- see the `provenance` block in the emitted JSON.

Nothing here re-derives a frozen constant.  Measured anchors are read through
`harness.constants.value()`; the cost model's own constants come from the
checked-in B2-B10 JSON exactly as every other caller sees them.
"""
import argparse
import hashlib
import json
import os
import sys

from harness import constants
from harness import inputs
from harness.constants import get, value
from harness.costmodel import cost_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))

S = 512                     # decoded tokens per chunk: the frozen spec's chunking
B = cm.BATCH                # the LogUp lookup-batch size the model charges (2)

# --------------------------------------------------------------------------
# The target shapes.  Chosen for one reason: `experiments/b1/model.py` runs
# them with no new architecture (an explicit Qwen2 forward pass, the same code
# B8 pointed at Llama-3.1-8B through `ZK_MODEL`), and `hd` is a power of two,
# which the model asserts.  Gemma needs E5a's embedding normalizer, QK-norm and
# dual-theta local/global attention first, so it is not an E4 candidate.
#
# 7-tuple order is `cost_model`'s: (L, d, nh, nkv, hd, dff, vocab).
# Every field is transcribed from the model's own published config.json.
# --------------------------------------------------------------------------
SHAPES = {
    "Qwen2.5-14B-Instruct": dict(
        cfg=(48, 5120, 40, 8, 128, 13824, 152064), tied=False,
        role="primary", hf="Qwen/Qwen2.5-14B-Instruct",
        why="above both fitted scales, so the claim is extrapolation and not "
            "interpolation; 29.5 GB of bf16 weights and a layer chunk ~1.3x the "
            "8B's, both comfortable on the H200"),
    "Qwen2.5-3B-Instruct": dict(
        cfg=(36, 2048, 16, 2, 128, 11008, 151936), tied=True,
        role="fallback-small", hf="Qwen/Qwen2.5-3B-Instruct",
        why="if the H200 window is under an hour: interpolates between the two "
            "fitted scales rather than extrapolating, so it is a weaker test -- "
            "registered so that falling back is not a post-hoc choice"),
    "Qwen2.5-32B-Instruct": dict(
        cfg=(64, 5120, 40, 8, 128, 27648, 152064), tied=False,
        role="fallback-large", hf="Qwen/Qwen2.5-32B-Instruct",
        why="the strongest test if the window is generous, and the shape closest "
            "to the paper's 27B frontier row; C=4096 only -- a 16K layer chunk "
            "extrapolates past the card"),
}

CONTEXTS = (4096, 16384)


_SEQ5 = {}


def _seq5(ctx, layer=16):
    """One measured 8B layer chunk, for scaling only (memory, ratios).

    Not a frozen constant -- `constants.json` freezes the 32-layer sums and the
    per-token figures, not L16 alone -- so it is read from the artifact and
    cited as such.  Nothing measured lives in this repository, so the artifact
    comes from `--artifacts DIR` (or `ZK_ARTIFACTS`), the directory the rest of
    the cost model already reads run outputs from, laid out as
    `<DIR>/seq5/seq_c<C>_L<k>.json`.  It is read when a caller asks for it and
    not when this module is imported.
    """
    if (ctx, layer) not in _SEQ5:
        f = inputs.artifacts_dir() / "seq5" / f"seq_c{ctx}_L{layer}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}: the measured 8B layer chunk at C={ctx}, "
                             "layer 16, which the shape ratios are stated against")
        _SEQ5[(ctx, layer)] = json.loads(f.read_text())["result"]["point"]
    return _SEQ5[(ctx, layer)]


def ref8b():
    """The 8B L16 chunk's fresh cells at both contexts: the scaling reference."""
    return {c: _seq5(c)["cells"]["fresh"] for c in CONTEXTS}


REF8B_SOURCE = "<--artifacts>/seq5/seq_c{4096,16384}_L16.json (result.point.cells.fresh)"

# --------------------------------------------------------------------------
# The predictor.  Two halves, deliberately kept apart because they have very
# different standing:
#
#   key tiles   the attention key tile family.  A CLOSED FORM read off the layout
#           (`prover/src/attn_trace.rs` x `layout.rs`), not a fit -- see
#           `attention_grid.{base,fresh}_cell_coeffs`.  It reproduces the
#           measured `family_cells.key_tiles` of all 32 seq5 layer chunks EXACTLY,
#           at both contexts.  So the E4 claim on this half is exact, not
#           bracketed.
#
#   rest    everything else in a layer chunk: the dequant, the Freivalds check, the
#           requantize, the int8 range checks, the attention-output fold and
#           the residual stream.  Pure arithmetic in the 7-tuple, from the B2-B10 gadget
#           marginals.  This is the half that can be wrong at a new scale.
#
# `rest` is the cost model's layers-only per-token total minus its `attn` term,
# plus the residual stream's NET cost (`residual_stream_cells(...)['net']` --
# total minus the stub the composed statement replaces).  Charging the residual
# stream's gross total instead over-predicts the 8B chunk by 11.6%; charging no
# residual stream at all under-predicts it by 14.3%.  The net is the one that
# matches what the prover builds.
# --------------------------------------------------------------------------


def key_tiles_per_chunk(nh, hd, s, c, base_slope=2.0):
    """`cm.key_tile_cells` + diagonal block, with the ONE coefficient caveat 1 disputes exposed.

    `base_slope` is the key tile family's committed BASE cells per (head, key, token)
    -- the `2 * nh * s * c` score-and-numerator term of `cm.key_tile_cells`, i.e. the
    64 in `attention_grid.key_tile_model_base_coeffs` divided by NH=32.  The refuted
    `key tiles = p * S^2` reading of B9's two prefill points puts it at
    72.083 / 32 = 2.2526 instead.  Everything else is untouched: this is a
    sensitivity, not a second model.
    """
    b, i, sat1 = cm.key_tile_cells(nh, hd, s, c)
    cb, ci, sat2 = cm.key_tile_cells(nh, hd, s, s, rows_component=False)
    extra = (base_slope - 2.0) * nh * (s * c + s * s)
    return b + cb + extra, i + ci, bool(sat1 or sat2)


def predict(cfg, ctx, seq=S, base_slope=2.0):
    """Layers-only fresh cells per decoded token, split key tiles / rest."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    r = cm.cells_per_token(cfg, ctx, B, seq=seq, head=False)
    gl = cm.residual_stream_cells(cfg, "tree")
    bb, bi, sat = key_tiles_per_chunk(nh, hd, seq, ctx, base_slope=base_slope)
    key_tiles = L * (bb + bi) / seq
    rest = r["total"] - r["attn"] + gl["net"]
    return dict(key_tiles=key_tiles, rest=rest, total=key_tiles + rest,
                key_tiles_base_per_chunk=bb, key_tiles_interaction_per_chunk=bi,
                per_layer_chunk=(key_tiles + rest) * seq / L,
                saturated=sat, residual_stream_net=gl["net"], bare=r["total"])


def head_chunk(cfg, seq=S):
    return cm.lm_head_chunk(cfg, seq)


# --------------------------------------------------------------------------
# Anchors: what the same predictor says about the two scales it WAS built from.
# This is the only honest basis for a bracket, so it is computed here rather
# than asserted.
# --------------------------------------------------------------------------
def anchors():
    out = {}
    for ctx, key in ((4096, "cells_per_token.8b.c4096.fresh_layers"),
                     (16384, "cells_per_token.8b.c16384.fresh_layers")):
        p = predict(cm.LLAMA8B, ctx)
        m = value(key)
        out[f"llama8b_c{ctx}"] = dict(
            scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=m,
            ratio=p["total"] / m, source=get(key)["source"], grade=get(key)["grade"],
            card=get(key)["card"])
    # Qwen2.5-0.5B: B6.5's composed layer, a DIFFERENT statement -- no
    # residual-stream family, no diagonal block key tile, and q/k/v/o on the pow2 dequant
    # rung while gate/up/down are on the chain.  Evaluated the same way, as a weak anchor.
    L, d, nh, nkv, hd, dff, vocab = cm.QWEN
    r = cm.cells_per_token(cm.QWEN, 1024, B, seq=S, head=False, diagonal_block=False)
    mixed = L * (((nh + 2 * nkv) * hd + d) * cm.HEAD_DEQUANT_PER_ACC
                 + (2 * dff + d) * cm.LAYER_DEQUANT_PER_ACC)
    tot = (r["total"] - r["limbs"] + mixed) * S / L
    meas = cm.B65C["compose"]["fresh"]
    out["qwen0p5b_c1024"] = dict(
        scale="Qwen2.5-0.5B", ctx=1024, predicted=tot, measured=meas,
        ratio=tot / meas, source="experiments/b65/out/compose_host.json "
        "(result.point.cells.fresh)", grade="measured", card="none (host, macOS)",
        note="weak anchor: no residual-stream family, no diagonal block key tile, mixed dequant rungs, "
             "and a prover generation older than the seq5 statement")
    return out


# --------------------------------------------------------------------------
# The pre-committed bracket.  R's prize-floor / floor / outer pattern.
# --------------------------------------------------------------------------
BRACKET = dict(
    quantity="rho = measured / predicted, on layers-only fresh cells per decoded "
             "token at the registered (shape, S, C)",
    prize=[0.97, 1.03],
    pass_=[0.90, 1.10],
    marginal=[0.80, 1.25],
    rule=(
        "PRIZE  rho in [0.97, 1.03] -- the model predicts a never-fitted scale to "
        "3%. The extrapolation licence is claimed without qualification and the "
        "section 6.3 frontier table keeps its [projected] grade.\n"
        "PASS   rho in [0.90, 1.10] -- roughly twice the worst residual the same "
        "predictor shows at the two scales it WAS built from (-5.0% at 0.5B, "
        "+4.2% at 8B). The licence is claimed with the measured residual quoted "
        "beside it.\n"
        "MARGINAL rho in [0.80, 0.90) or (1.10, 1.25] -- the licence is claimed "
        "only with an explicit +/-25% key tile printed on every frontier row.\n"
        "FAIL   rho < 0.80 or rho > 1.25 -- the cost model is REFUTED as an "
        "extrapolator. Section 6.3 is regraded from [projected] to "
        "[illustrative], the 'you didn't build it' mitigation falls back to "
        "publishing the model runnable, and E4 is written up as a negative "
        "result. No re-fit may be presented as if it were this prediction."),
    protocol_widening=dict(
        A_all_layers=dict(
            what="extract and prove every layer at the registered C, the seq5 "
                 "recipe at a new shape; compare layers-only cells/token",
            widen_pct=0.0),
        B_even_subset=dict(
            what=">= 8 EVENLY SPACED layers (L0, L*1/8, ... by a rule fixed "
                 "before the run), compare the mean scaled to L layers",
            widen_pct=1.0,
            basis="on the 8B's own 32 chunks, the worst 8-layer evenly-spaced "
                  "subsample mean is 0.92% off the full mean at C=4096 and 0.41% "
                  "at C=16384, over every phase offset"),
        C_single_layer=dict(
            what="one layer only -- the minimum that still tests anything",
            widen_pct_upper=9.0, widen_pct_lower=1.5,
            basis="a single 8B chunk runs -1.18% to +8.59% around the 32-layer "
                  "mean at C=4096, ENTIRELY from the per-layer master exp table. "
                  "Every other family is identical to the cell across all 32 "
                  "layers, and the whole chunk is the exact identity "
                  "`cells.fresh == key tiles(S, C) + 1,006,949,376 + 5 * 2^log_table`, "
                  "verified on all 32 chunks at BOTH contexts (`log_table` 22..25, "
                  "`interaction_by_family.table` 56.0-173.4 Mcells). The run MUST "
                  "report `params.log_table` and "
                  "`interaction_by_family.table.cells` for the layer it proved."),
    ),
    also=(
        "SEPARATE and EXACT: the key tiles family. `family_cells.key tiles` (base) and "
        "`interaction_by_family.key tiles.cells` must equal the registered integers "
        "TO THE CELL, as they do on all 32 seq5 layer chunks at both contexts. "
        "Any nonzero deviation is a finding in its own right and must be "
        "reported even if rho passes -- it means the layout identities in "
        "`key_tile_cells` do not hold at a shape whose `nh` and `d` are not powers "
        "of two, which is the one thing the closed form's own docstring flags."),
    invalidation=(
        "The comparison is void, not failed, if the run does not reproduce the "
        "registered configuration: spec `b10-final` (round even, all seven "
        "matmuls on the `chain` dequant rung), S=512, diagonal block key tile ON, "
        "grid_target 64 / chan_target 128 / rw 16, k_bits 11 / q_bits 12, "
        "bounds on, and a REAL extracted witness (not `ZK_SYNTHETIC_DIMS`). "
        "A void run is re-run, never re-bracketed."),
)


def selfcheck():
    """The predictor must reproduce the 8B chunk's key tiles family TO THE CELL.

    This is what licenses registering the key tiles half with no bracket at all: the
    same code path, at the shape it was derived from, against the artifact.
    """
    out = []
    for ctx in CONTEXTS:
        pt = _seq5(ctx)
        nh, hd = pt["params"]["nh"], pt["params"]["hd"]
        assert (nh, hd) == (cm.LLAMA8B[2], cm.LLAMA8B[4]), (nh, hd)
        bb, bi, sat = key_tiles_per_chunk(nh, hd, S, ctx)
        mb = pt["family_cells"]["key_tiles"]
        mi = pt["family_cells"]["interaction_by_family"]["key_tiles"]["cells"]
        assert int(bb) == mb, (ctx, bb, mb)
        assert int(round(bi)) == mi, (ctx, bi, mi)
        # and the local reimplementation must equal cost_model's own key_tile_fresh
        bf, _ = cm.key_tile_fresh(nh, hd, S, ctx, diagonal_block=True)
        assert abs(bf - (bb + bi)) < 1e-6, (ctx, bf, bb + bi)
        out.append(dict(ctx=ctx, key_tiles_base=mb, key_tiles_interaction=mi, exact=True,
                        saturated=sat))
    return out


def canonical(block):
    return json.dumps(block, sort_keys=True, separators=(",", ":")).encode()


def build():
    reg = {}
    ref8b_cells = ref8b()
    for name, meta in SHAPES.items():
        cfg = meta["cfg"]
        L, d, nh, nkv, hd, dff, vocab = cfg
        n = cm.n_params(cfg) + (0 if meta["tied"] else vocab * d)
        entry = dict(hf=meta["hf"], role=meta["role"], why=meta["why"],
                     cfg=list(cfg), tied_embeddings=meta["tied"],
                     params_total=n, params_b=round(n / 1e9, 3),
                     weights_gib_bf16=round(n * 2 / 2 ** 30, 2), contexts={})
        for ctx in CONTEXTS:
            if meta["role"] == "fallback-large" and ctx == 16384:
                continue
            p = predict(cfg, ctx)
            # caveat 1: what the refuted `p * S^2` reading's base slope would do
            hi = predict(cfg, ctx, base_slope=72.083 / 32.0)
            ref8 = ref8b_cells[ctx]
            entry["contexts"][str(ctx)] = dict(
                cells_per_token_layers=p["total"],
                cells_per_token_layers_key_tiles=p["key_tiles"],
                cells_per_token_layers_rest=p["rest"],
                per_layer_chunk_fresh=p["per_layer_chunk"],
                key_tiles_family_base_per_chunk=int(p["key_tiles_base_per_chunk"]),
                key_tiles_family_interaction_per_chunk=int(round(p["key_tiles_interaction_per_chunk"])),
                key_tiles_family_fresh_per_chunk=int(round(p["key_tiles_base_per_chunk"]
                                                       + p["key_tiles_interaction_per_chunk"])),
                key_tile_layout_saturated=p["saturated"],
                chunk_over_8b_l16_chunk=p["per_layer_chunk"] / ref8,
                sensitivity_base_slope_2p2526=dict(
                    cells_per_token_layers=hi["total"],
                    shift_pct=100.0 * (hi["total"] / p["total"] - 1.0)),
            )
        h = head_chunk(cfg)
        entry["head_chunk"] = dict(fresh=h["fresh"], per_token=h["per_token"],
                                   preprocessed=h["preprocessed"])
        reg[name] = entry
    return reg


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def provenance():
    return dict(
        inputs_sha256={
            "harness/costmodel/cost_model.py": _sha(cm.__file__),
            "constants.json": _sha(constants.path()),
        },
        registered_utc_date="2026-09-01",
        registered_by="stream E4 (desk lane)",
        no_measurement_exists=(
            "At registration there is NO witness, NO proof and NO cell count for "
            "Qwen2.5-3B / 14B / 32B anywhere in this repo or on either box. The "
            "only scales ever extracted are Qwen2.5-0.5B (experiments/b65/out/) "
            "and Llama-3.1-8B (experiments/b8, b9, b10). The H200 queue at "
            "registration time is bring-up -> E8 seq6 -> E2 measurement -> E4 "
            "extraction, and E8 is the item in progress; the H100 is on E11's "
            "third fit point. Neither box has been asked for any number in this "
            "file."),
        predictor=dict(
            engine="experiments/b2/cost_model.py",
            entry_points=["cells_per_token(cfg, ctx, batch, seq, head=False)",
                          "residual_stream_cells(cfg, 'tree')", "key_tile_cells", "key_tile_fresh",
                          "lm_head_chunk"],
            not_the_engine="experiments/budget.py is an early one-shape proxy "
                           "(eval-plan.md section 3) and is not used here",
            attention_term="the KEY_TILE MODEL (alpha=None). The legacy single-slope "
                           "alpha=4.00 path is NOT used -- section 7.5(e)'s ruling "
                           "is that the paper uses the composed-chunk value.",
            lookup_batch=B, seq_per_chunk=S,
        ),
    )


def frozen_inputs():
    keys = [
        "cells_per_token.8b.c4096.fresh_layers",
        "cells_per_token.8b.c16384.fresh_layers",
        "cells.c4096.fresh_layers32",
        "cells.c16384.fresh_layers32",
        "attention_grid.key_tile_model_base_coeffs",
        "attention_grid.key_tile_model_fresh_coeffs",
        "attention_grid.key_tile_model_alpha_fresh",
        "attention_grid.composed_chunk_fresh_slope",
        "attention_grid.composed_chunk_fresh_slope_32layers",
        "attention_grid.b5_standalone_alpha",
        "model_constants.attention_cells_per_head_key_token",
        "model_constants.dequant_chain_cells_per_acc",
        "model_constants.dequant_pow2_cells_per_acc",
        "model_constants.attnout_int8_cells_per_acc",
        "model_constants.weight_freivalds_cells_per_weight",
        "model_constants.logup_marginal_cells_per_lookup",
        "model_constants.nonlinear_cells_per_application",
        "model_constants.default_logup_lookup_batch",
        "rate.h200.c4096.commit_layers32",
        "rate.h200.c16384.commit_layers32",
        "spec.name", "spec.decoded_tokens", "spec.round_mode",
    ]
    out = {}
    for k in keys:
        e = get(k)
        out[k] = dict(value=e["value"], grade=e["grade"], card=e["card"],
                      context=e["context"], boundary=e["boundary"])
    return out


def implied_tok_s(reg):
    """Secondary, and conditional on a rate that is NOT part of the prediction."""
    out = {}
    for name, e in reg.items():
        row = {}
        for ctx, key in (("4096", "rate.h200.c4096.commit_layers32"),
                         ("16384", "rate.h200.c16384.commit_layers32")):
            if ctx not in e["contexts"]:
                continue
            rate = value(key) * 1e9
            row[ctx] = dict(tok_s_commit_boundary=rate / e["contexts"][ctx]["cells_per_token_layers"],
                            rate_key=key, rate=value(key), card=get(key)["card"],
                            boundary=get(key)["boundary"])
        out[name] = row
    return out


def integrator_notes(reg):
    """Two things E4 found on the way past, for the coordinator -- not predictions.

    (1) section 7.5(e): the frontier rows really are still on the 4.00.
    (2) the card footprint the run-book has to plan around.
    """
    g = cm.GEMMA
    rows = {}
    for ctx, key in ((4096, "cells_per_token.27b.c4096.model"),
                     (16384, "cells_per_token.27b.c16384.model")):
        alpha4 = cm.cells_per_token(g, ctx, B, seq=S, alpha=4.0)["total"]
        key_tile = cm.cells_per_token(g, ctx, B, seq=S)["total"]
        rows[str(ctx)] = dict(frozen=value(key), rerun_alpha_4p00=alpha4,
                              rerun_key_tile_model=key_tile,
                              key_tile_over_frozen=key_tile / value(key),
                              understated_pct=100.0 * (key_tile / value(key) - 1.0))
    mem = {}
    for name, e in reg.items():
        mem[name] = {ctx: dict(chunk_over_8b_l16=c["chunk_over_8b_l16_chunk"])
                     for ctx, c in e["contexts"].items()}
    return dict(
        frontier_rows_still_on_4p00=dict(
            finding="Every `cells_per_token.27b.*` row in paper/constants.json "
                    "reproduces EXACTLY when the cost model is re-run with the "
                    "legacy alpha=4.00 path, and not otherwise. Re-running the "
                    "same shape on the key tile model -- the composed-chunk value "
                    "W0's freeze ruled the paper must use (eval-plan.md 7.5(e)) "
                    "-- raises them as below. E4 reports this and does not patch "
                    "it: `paper/constants.json` is owned elsewhere this session.",
            grades_still_wrong=[
                dict(key="attention_grid.b5_standalone_alpha",
                     grade=get("attention_grid.b5_standalone_alpha")["grade"]),
                dict(key="model_constants.attention_cells_per_head_key_token",
                     grade=get("model_constants.attention_cells_per_head_key_token")["grade"]),
            ],
            rows=rows),
        card_footprint=dict(
            basis="seq5's single 8B layer chunk peaked 76.8 GiB on the card at "
                  "C=16384 (eval-plan.md section 8). E4's chunks scale roughly "
                  "with fresh cells; the H200 has 141 GB and the C=16384 arm must "
                  "run SERIAL, never pipelined two-ahead.",
            ratios=mem),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="rewrite the registration file")
    constants.add_argument(ap)
    ap.add_argument("--inputs", metavar="DIR", help="cost-model inputs (read at import; or set ZK_COSTMODEL_INPUTS)")
    inputs.add_arguments(ap)
    a = ap.parse_args()
    constants.load(a.constants)

    chk = selfcheck()
    reg = build()
    anc = anchors()
    # The falsifiable content, hashed ALONE.  This must not move when an
    # unrelated constant is edited, or the hash cannot answer the only question
    # a later reader asks of it: is this the prediction that was registered?
    content = dict(schema="e4-registration/1", shapes=reg, bracket=BRACKET,
                   anchors=anc, key_tiles_selfcheck_8b=chk,
                   frozen_inputs=frozen_inputs())
    content_digest = hashlib.sha256(canonical(content)).hexdigest()
    block = dict(content, provenance=provenance())
    digest = hashlib.sha256(canonical(block)).hexdigest()
    doc = dict(block, prediction_sha256=content_digest,
               prediction_block_sha256=digest,
               implied_tok_s_secondary=implied_tok_s(reg),
               integrator_notes=integrator_notes(reg),
               reference_8b_chunk=dict(cells=ref8b(), source=REF8B_SOURCE))

    print("=" * 78)
    print("E4 -- unfitted-scale prediction, registered before measurement")
    print("=" * 78)
    print()
    print("key tiles self-check against the 8B artifacts (must be exact to the cell):")
    for c in chk:
        print(f"  C={c['ctx']:<6} base {c['key_tiles_base']:>13,}  interaction "
              f"{c['key_tiles_interaction']:>13,}  exact={c['exact']}  saturated={c['saturated']}")
    print()
    print("the predictor's residual at the two scales it WAS built from:")
    for k, v in anc.items():
        print(f"  {v['scale']:<16} C={v['ctx']:<6} predicted {v['predicted']:>15,.0f}  "
              f"measured {v['measured']:>15,}  ratio {v['ratio']:.4f}  "
              f"({100*(v['ratio']-1):+.2f}%)")
    print()
    for name, e in reg.items():
        print(f"{name}  [{e['role']}]  L={e['cfg'][0]} d={e['cfg'][1]} nh={e['cfg'][2]} "
              f"nkv={e['cfg'][3]} hd={e['cfg'][4]} dff={e['cfg'][5]} vocab={e['cfg'][6]}")
        for ctx, c in e["contexts"].items():
            print(f"    C={ctx:<6} cells/token (layers) {c['cells_per_token_layers']:>15,.0f}"
                  f"   = key tiles {c['cells_per_token_layers_key_tiles']:>14,.0f}"
                  f" + rest {c['cells_per_token_layers_rest']:>13,.0f}")
            print(f"    {'':<8} one layer chunk, fresh {c['per_layer_chunk_fresh']:>13,.0f}"
                  f"   key tiles family/chunk: base {c['key_tiles_family_base_per_chunk']:,}"
                  f"  interaction {c['key_tiles_family_interaction_per_chunk']:,}")
            print(f"    {'':<8} caveat-1 sensitivity (base slope 2.0 -> 2.2526): "
                  f"{c['sensitivity_base_slope_2p2526']['shift_pct']:+.2f}%"
                  f"   saturated={c['key_tile_layout_saturated']}"
                  f"   chunk vs 8B L16 x{c['chunk_over_8b_l16_chunk']:.3f}")
        print()
    _n = integrator_notes(reg)["frontier_rows_still_on_4p00"]["rows"]
    print("section 7.5(e) check -- the frozen 27B frontier rows:")
    for ctx, r in _n.items():
        print(f"  C={ctx:<6} frozen {r['frozen']:>13,}  = alpha 4.00 rerun "
              f"{r['rerun_alpha_4p00']:>13,.0f}   key tile model {r['rerun_key_tile_model']:>13,.0f}"
              f"  ({r['understated_pct']:+.2f}% understated)")
    print()
    print(f"prediction sha256 (content only): {content_digest}")
    print(f"prediction block sha256 (incl. provenance): {digest}")
    if a.json:
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        p = os.path.join(HERE, "out", "registration.json")
        with open(p, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
