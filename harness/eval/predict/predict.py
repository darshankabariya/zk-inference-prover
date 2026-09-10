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

Nothing here re-derives a frozen constant, and nothing here falls back to a
number.  The anchor that carries the weight is the 8B layer block zk-results
records per proved position, one row per context, with a measured 8B layer
chunk read from `--artifacts` beside it; a constants key that this constants
file does not name is recorded as absent and nothing stands in for it.  The
cost model's own inputs come through `cost_model.PROVENANCE`, constants first
and the micro-benchmark outputs second, and the whole record is frozen with the
prediction.
"""
import argparse
import hashlib
import json
import os
import sys

from harness import constants
from harness import inputs
from harness.constants import get
from harness.costmodel import cost_model as cm
from harness.costmodel import key_tile

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
        _SEQ5[(ctx, layer)] = json.loads(f.read_text())
    return _SEQ5[(ctx, layer)]["result"]["point"]


def _seq5_source(ctx, layer=16):
    """Which run, block and segment the reference chunk was derived from.

    The chunk carries its own `derived_from` block -- the run id, the slice
    output, the segment tag and the proof digest -- so the registration records
    where the number came from instead of naming a file path."""
    _seq5(ctx, layer)
    d = dict(_SEQ5[(ctx, layer)].get("derived_from") or {})
    d.pop("file", None)
    return d


def ref8b():
    """The 8B L16 chunk's fresh cells at both contexts: the scaling reference."""
    return {c: _seq5(c)["cells"]["fresh"] for c in CONTEXTS}


REF8B_SOURCE = "<--artifacts>/seq5/seq_c{4096,16384}_L16.json (result.point.cells.fresh)"

# --------------------------------------------------------------------------
# The predictor.  Two halves, deliberately kept apart because they have very
# different standing:
#
#   key tiles   the attention key tile family.  A CLOSED FORM read off the layout
#           (`prover/src/attn_trace.rs` x `layout.rs` x `compose.rs`'s `Plan`),
#           not a fit -- see `attention_grid.{base,fresh}_cell_coeffs`.  It
#           reproduces the measured `family_cells.key_tiles` of every recorded
#           layer chunk EXACTLY, at both contexts, base and interaction, with
#           and without the five reductions
#           (`harness.costmodel.key_tile --selftest`).  So the E4 claim on this
#           half is exact, not bracketed.
#
#           The family is `C/S` prefix key tiles of S keys each plus the S x S
#           diagonal block, with one per-token component across the lot, and it
#           costs `nh*hd*(C - S)` base cells per chunk more than one untiled
#           prefix key tile would (issue #141).  `selfcheck` prices the statement
#           the measured chunk names; `predict` prices `STATEMENT` below, the
#           statement the run this is registered against proves.
#
#   rest    everything else in a layer chunk: the dequant, the Freivalds check, the
#           requantize, the q/k quantize, the int8 range checks, the
#           attention-output fold, RoPE, the lookup tables, the boundary trees
#           and the residual stream.  Two of those are closed forms of the
#           layout like the key tile family (the boundary trees and the lookup
#           tables); the rest is arithmetic in the 7-tuple times a gadget rung.
#           This is the half that can be wrong at a new scale.
#
#           Both halves are priced at the SAME statement, and `cells_per_token`
#           is told which one: `--gkr-dequant`, `--gkr-residual-stream`,
#           `--gkr-freivalds` and `--gkr-attnout` each prove a family's LogUp
#           running sums instead of committing them, and the driver passes all
#           four beside `--gkr-key-tiles`.  What the model does NOT carry is
#           `gkrlink`, the reductions' own link -- 12,529,664 cells on both
#           recorded 8B layer chunks, with no form in the shape -- so this half
#           is short by about that much per chunk and `cost_model.PROVENANCE`
#           records it as unpriced.
#
# `rest` is the cost model's layers-only per-token total minus its `attn` term,
# plus the residual stream's NET cost (`residual_stream_cells(...)['net']` --
# total minus the stub the composed statement replaces).
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# THE STATEMENT THE REGISTRATION PRICES.  `tools/drivers/predict-14b.sh` in
# zk-results proves every layer of the target shape with `--diagonal-block
# --gkr-key-tiles --gkr-dequant --gkr-residual-stream --gkr-freivalds
# --gkr-attnout`, the same configuration as both headline runs, so that is the
# statement the key tile half is priced at.  The closed form's own defaults are
# a different statement -- the five reductions off -- and would register a
# larger interaction at the same shape, so it is asked for this one explicitly
# and never left to default.
#
# `kv_boundary` and `bind` are not flags: the prover derives `kv_boundary =
# diagonal_block && !head` (`main.rs`) and publishes the root-equality claims
# whenever the segment has `Params::rope`, which every layer segment does.
# --------------------------------------------------------------------------
STATEMENT = dict(diagonal_block=True, kv_boundary=True, bind=True, gkr=True)

# The same statement for the other half: every one of the prover's five batched
# reductions, which is what the driver passes.  `STATEMENT` is the key tile
# family's own configuration and `gkr=True` in it is `--gkr-key-tiles` alone;
# this is the set `cells_per_token` and `residual_stream_cells` are told about.
STATEMENT_REDUCTIONS = cm.REDUCTIONS

STATEMENT_FLAGS = dict(
    diagonal_block="--diagonal-block",
    kv_boundary="not a flag: `kv_boundary = diagonal_block && !head` (`main.rs`), "
                "so every layer segment of the run has it",
    bind="not a flag: `Params::rope`, which every layer segment has and no "
         "output head does",
    gkr="--gkr-key-tiles",
)

# What the driver passes beside `--gkr-key-tiles`, and what the model does with
# it.  Named here so a reader of the registration is not left to infer it from
# the integers.
STATEMENT_NOT_PRICED = dict(
    flags="--gkr-dequant --gkr-residual-stream --gkr-freivalds --gkr-attnout",
    effect="each proves the LogUp running sums of the family it names instead "
           "of committing them, and moves what it does commit into `gkrlink`. "
           "`cells_per_token(gkr=...)` prices all four; `gkrlink` itself is the "
           "one family the model does not price, and "
           "`cost_model.PROVENANCE['gkrlink_cells_per_chunk']` says so",
)


def key_tiles_per_chunk(nh, hd, s, c, base_slope=2.0, **kw):
    """`cm.key_tile_family`, with the ONE coefficient caveat 1 disputes exposed.

    The family is the prefix tiled into key blocks of `s` keys plus the diagonal
    block, not one prefix key tile over `c` keys; `**kw` is the statement's own
    configuration (`diagonal_block`, `kv_boundary`, `bind`, `gkr`), which
    `selfcheck` reads off the measured chunk and `predict` takes from
    `STATEMENT`.  There is no default here worth having: the caller says which
    statement it is pricing or the integers mean nothing.

    `base_slope` is the key tile family's committed BASE cells per (head, key, token)
    -- the `2 * nh * s * c` score-and-numerator term of `cm.key_tile_cells`, i.e. the
    64 in `attention_grid.key_tile_model_base_coeffs` divided by NH=32.  The refuted
    `key tiles = p * S^2` reading of B9's two prefill points puts it at
    72.083 / 32 = 2.2526 instead.  Everything else is untouched: this is a
    sensitivity, not a second model.  The family commits that term over `c + s`
    keys however the prefix is tiled, so the shift is unchanged.
    """
    b, i, sat = cm.key_tile_family(nh, hd, s, c, **kw)
    extra = (base_slope - 2.0) * nh * (s * c + s * s)
    return b + extra, i, sat


def predict(cfg, ctx, seq=S, base_slope=2.0, statement=None, reductions=None):
    """Layers-only fresh cells per decoded token, split key tiles / rest.

    `statement` is the key tile configuration being priced, `STATEMENT` by
    default.  The cost model's own `attn` term is subtracted whole and replaced
    by `key_tiles_per_chunk`, so it is that argument and nothing else that says
    which statement the key tile half of the prediction is for.

    `reductions` is the same statement for the other half -- which of the
    prover's five batched reductions the run proves -- and it defaults to
    `STATEMENT_REDUCTIONS`, all five, which is what the driver passes.  The two
    arguments have to agree: pricing the key tile family at a reduced statement
    and everything else at an un-reduced one is not a statement any run proves.
    """
    st = STATEMENT if statement is None else statement
    red = STATEMENT_REDUCTIONS if reductions is None else reductions
    L, d, nh, nkv, hd, dff, vocab = cfg
    r = cm.cells_per_token(cfg, ctx, B, seq=seq, head=False, gkr=red)
    gl = cm.residual_stream_cells(cfg, "tree", gkr=red)
    bb, bi, sat = key_tiles_per_chunk(nh, hd, seq, ctx, base_slope=base_slope, **st)
    key_tiles = L * (bb + bi) / seq
    rest = r["total"] - r["attn"] + gl["net"]
    return dict(key_tiles=key_tiles, rest=rest, total=key_tiles + rest,
                key_tiles_base_per_chunk=bb, key_tiles_interaction_per_chunk=bi,
                per_layer_chunk=(key_tiles + rest) * seq / L,
                saturated=sat, residual_stream_net=gl["net"], bare=r["total"],
                statement=dict(st), reductions=tuple(red),
                reductions_not_priced=r["gkr_not_priced"],
                rest_rows={k: r[k] for k in ("acc", "act", "nl", "logup", "freivalds",
                                             "range", "limbs", "attn_out", "requant",
                                             "rope", "quant", "tables", "boundary")})


def head_chunk(cfg, seq=S):
    return cm.lm_head_chunk(cfg, seq)


# --------------------------------------------------------------------------
# Anchors: what the same predictor says about the two scales it WAS built from.
# This is the only honest basis for a bracket, so it is computed here rather
# than asserted.
#
# The anchor that carries the weight is the RECORDED LAYER BLOCK, one
# constants row per context: the fresh cells of the model's 32 layer segments
# of one token block over the 512 positions that block proves.  The block is
# the one whose prefix is the whole context, so both sides of the comparison
# are 32 layers at C over a block of S, and neither side carries an output head
# segment: it is the quantity `predict` states, position for position.  The
# other `cells_per_position.headline.*` rows of the same run are NOT that
# quantity -- they divide every segment of the run, output heads and shorter
# prefixes included, by every position the run proves -- and are not read here.
#
# One difference between the two sides is named rather than adjusted away: the
# measured block's layer segment 0 carries the embedding lookup and one more
# cell per attention-output accumulator, which the model prices as no layer's
# cost.  `layer_chunks.json` holds the same 32 segments, so both the size of
# that difference and the recorded figure itself are recomputed here, to the
# cell, instead of transcribed.
#
# The second anchor per context is the DERIVED REFERENCE CHUNK: one 8B layer
# chunk out of that same block, read from `--artifacts`, scaled to 32 layers.
# It is what a constants file older than the row above leaves, and it splits
# the residual into the key tile half and the rest, which the row cannot.
# --------------------------------------------------------------------------
FROZEN_8B_KEYS = {4096: "cells_per_position.headline.c4096.fresh_layers_block",
                  16384: "cells_per_position.headline.c16384.fresh_layers_block"}
FROZEN_8B_ABSENT = (
    "this constants file does not name it. A constants file older than the "
    "record of this row has only the `cells_per_position.headline.*` rows over "
    "the whole decode sequence, which are a different quantity -- every segment "
    "of the run, output heads and shorter prefixes included, over every "
    "position it proves -- and none of them stands in for it. The registration "
    "is anchored on the derived reference chunk beside this instead.")


def entry_provenance(e):
    """Where a constants entry came from, in the fields zk-results names.

    That file's entries carry `run` and `artifact`; none of them has a `source`
    field and its own check refuses one that does. An entry that names neither
    -- an assumption, or a figure read from a public source -- says so here
    rather than being given a path it does not have."""
    run, artifact = e.get("run"), e.get("artifact")
    if run and artifact:
        return f"runs/{run}/{artifact}"
    return run or artifact or "the constants file, which names no run for this entry"


def recorded_layer_block(ctx):
    """The measured 8B layer block, out of this repository's copy of its segments.

    `layer_chunks.json` holds every segment of the token block each recorded
    row divides, so the row can be recomputed here to the cell instead of
    trusted, and the embedding lookup that separates the two sides of the
    comparison can be sized without transcribing a number."""
    ch = [q for q in key_tile.chunks() if not q["head"] and q["c"] == ctx]
    if not ch:
        return None
    ident = key_tile.chunk_identity(ch)
    exact = len(ident["layer"]) == 1 and len(ident["embedding"]) == 1
    fresh, s = sum(q["fresh"] for q in ch), ch[0]["s"]
    return dict(layers=len(ch), positions=s, fresh=fresh, per_position=fresh / s,
                embedding_lookup_per_position=(ident["embedding"][0] - ident["layer"][0]) / s
                if exact else None,
                run=ch[0]["run"], artifact=ch[0]["output"],
                block=ch[0]["segment"].split("L")[0],
                note="the same 32 segments the recorded row divides, copied into "
                     "this repository; `embedding_lookup_per_position` is what "
                     "layer segment 0 builds above the other 31, which is the "
                     "embedding lookup and one more cell per attention-output "
                     "accumulator")


def anchors():
    out = {}
    for ctx in CONTEXTS:
        key = FROZEN_8B_KEYS[ctx]
        p = predict(cm.LLAMA8B, ctx)
        e = constants.find(key)
        if e is None:
            out[f"llama8b_c{ctx}"] = dict(
                scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=None,
                ratio=None, key=key, source="absent", grade="absent",
                statement=dict(STATEMENT), note=FROZEN_8B_ABSENT)
        else:
            m = e["value"]
            rec = recorded_layer_block(ctx)
            emb = rec and rec["embedding_lookup_per_position"]
            out[f"llama8b_c{ctx}"] = dict(
                scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=m,
                ratio=p["total"] / m, key=key, source=entry_provenance(e),
                run=e.get("run"), artifact=e.get("artifact"),
                grade=e["grade"], card=e["card"], statement=dict(STATEMENT),
                recorded_chunks=rec,
                measured_matches_recorded_chunks=(rec is not None
                                                  and rec["fresh"] == m * rec["positions"]),
                embedding_lookup_per_position=emb,
                ratio_without_the_embedding_layer=None if not emb else p["total"] / (m - emb),
                note="both sides are the fresh cells of the model's 32 layer segments "
                     "over the 512 positions one token block proves, at this context "
                     "and at the statement this registration prices, with no output "
                     "head segment on either. The measured block is the one whose "
                     "prefix is the whole context: the earlier blocks of that sequence "
                     "read a shorter prefix and cost less, which is the comparison the "
                     "prediction does not make. Its layer segment 0 carries the "
                     "embedding lookup and one more cell per attention-output "
                     "accumulator, which the model prices as no layer's cost -- that "
                     "is `embedding_lookup_per_position`, and the ratio beside it is "
                     "the same comparison without it")
        # The second anchor: one layer chunk out of the same block, scaled to
        # the model's 32.  It splits the residual into the key tile half, which
        # is exact to the cell, and the rest, which the recorded row cannot.
        pt = _seq5(ctx)
        mref = pt["cells"]["fresh"] * cm.LLAMA8B[0] / S
        mkt = (pt["family_cells"]["key_tiles"]
               + pt["family_cells"]["interaction_by_family"]["key_tiles"]["cells"])
        pkt = cm.LLAMA8B[0] * (p["key_tiles_base_per_chunk"]
                               + p["key_tiles_interaction_per_chunk"]) / S
        mrest = mref - mkt * cm.LLAMA8B[0] / S
        out[f"llama8b_c{ctx}_registered_statement"] = dict(
            scale="Llama-3.1-8B", ctx=ctx, predicted=p["total"], measured=mref,
            ratio=p["total"] / mref, source=REF8B_SOURCE, grade="measured",
            card="n/a (a cell count is the same on either card)",
            statement=dict(STATEMENT),
            key_tiles_exact=int(round(pkt)) == int(round(mkt * cm.LLAMA8B[0] / S)),
            rest_ratio=p["rest"] / mrest,
            rest_residual_per_chunk=(p["rest"] - mrest) * S / cm.LLAMA8B[0],
            chunk=_seq5_source(ctx),
            note="one layer chunk per context, scaled to 32 layers, at exactly the "
                 "statement this registration prices. The key tile half of this "
                 "ratio is exact to the cell; every cell of the difference is in "
                 "`rest`, and the largest single term the model does not price is "
                 "`gkrlink`, the reductions' own link")
    # Qwen2.5-0.5B: B6.5's composed layer, a DIFFERENT statement -- no
    # residual-stream family, no diagonal block key tile, none of the five reductions,
    # and q/k/v/o on the pow2 dequant
    # rung while gate/up/down are on the chain.  Evaluated the same way, as a weak anchor.
    #
    # Four families that prover generation did not build are taken back off the
    # predicted side, because charging a chunk for a family it has no entry for
    # is not a residual of the model: its `cells.fresh` is exactly its base plus
    # its interaction, so it commits no boundary tree at all, and its
    # `family_cells` names no lookup-table, quantize or residual-stream family.
    L, d, nh, nkv, hd, dff, vocab = cm.QWEN
    r = cm.cells_per_token(cm.QWEN, 1024, B, seq=S, head=False, diagonal_block=False)
    mixed = L * (((nh + 2 * nkv) * hd + d) * cm.HEAD_DEQUANT_PER_ACC
                 + (2 * dff + d) * cm.LAYER_DEQUANT_PER_ACC)
    not_built = r["boundary"] + r["tables"] + r["quant"] + r["rope"]
    tot = (r["total"] - r["limbs"] + mixed - not_built) * S / L
    meas = (cm.B65C or {}).get("compose", {}).get("fresh")
    if meas is None:
        out["qwen0p5b_c1024"] = dict(
            scale="Qwen2.5-0.5B", ctx=1024, predicted=tot, measured=None, ratio=None,
            source="b65/compose_host.json under --inputs DIR", grade="absent",
            note="the composed 0.5B layer is in no constants file and no key names "
                 "it; without the inputs directory this anchor is not evaluated")
        return out
    out["qwen0p5b_c1024"] = dict(
        scale="Qwen2.5-0.5B", ctx=1024, predicted=tot, measured=meas,
        ratio=tot / meas, source="experiments/b65/out/compose_host.json "
        "(result.point.cells.fresh)", grade="measured", card="none (host, macOS)",
        statement=dict(diagonal_block=False, kv_boundary=False, bind=True, gkr=False),
        statement_note="what `cells_per_token` prices here, not what B6.5 proved: "
                       "that prover generation published no root-equality claim, "
                       "so `bind` is charged against a chunk that has none",
        families_not_built=dict(
            boundary=r["boundary"] * S / L, tables=r["tables"] * S / L,
            quant=r["quant"] * S / L, rope=r["rope"] * S / L,
            why="this chunk's `cells.fresh` is exactly its base plus its "
                "interaction, so it commits no boundary tree, and its "
                "`family_cells` names no lookup-table, quantize or "
                "residual-stream family. Charged, they are most of the chunk."),
        note="weak anchor: no residual-stream family, no diagonal block key tile, mixed dequant rungs, "
             "and a prover generation older than the seq5 statement")
    return out


# --------------------------------------------------------------------------
# What one layer chunk is, and how far one layer is from the mean of them.
# Both are computed off `harness/costmodel/layer_chunks.json` -- every chunk of
# one token block of each headline run, copied into the repository -- so they
# move with the runs and never with a sentence somebody forgot to edit.
# --------------------------------------------------------------------------
def chunk_identity():
    """The whole-chunk identity, solved on the recorded chunks.

    A composed layer chunk's fresh cells are its key tile family, its boundary
    trees, its lookup tables and a rest the shape fixes:

        cells.fresh == key_tile_family(nh, hd, S, C) + boundary_cells(...)
                       + 5 * (sum of 2^log_rows over the chunk's tables) + rest

    and `rest` is one integer on every recorded non-embedding layer chunk at
    both contexts.  A single-layer run can check it on its own chunk, which is
    what protocol C leans on.
    """
    ident = key_tile.chunk_identity()
    return dict(
        form="cells.fresh == key_tile_family(nh, hd, S, C) + boundary_cells(nh, hd, d, S, C)"
             " + 5 * sum_t 2^log_rows(t) + rest",
        rest=ident["layer"][0] if len(ident["layer"]) == 1 else ident["layer"],
        rest_embedding_layer=(ident["embedding"][0] if len(ident["embedding"]) == 1
                              else ident["embedding"]),
        exact_to_the_cell=len(ident["layer"]) == 1,
        chunks=sum(1 for q in key_tile.chunks() if not q["head"] and not q["embedding"]),
        note="8B, S=512, spec b10-final, both contexts. The embedding layer's rest is "
             "higher by its embedding lookup and one more cell per attention-output "
             "accumulator. `boundary_cells` is the term no `family_cells` entry names: "
             "what `cells.fresh` has above `base + interaction`.",
        checked_by="harness.costmodel.key_tile --selftest")


def layer_spread():
    """How far one layer chunk is from the mean of a context's 32, measured.

    This is the basis the protocol widening was set from and it is recomputed
    here rather than quoted, at both contexts, for one layer and for the worst
    evenly-spaced subsample of eight over every phase offset.  The registered
    widening percentages are NOT derived from it: they are what they were.
    """
    out = {}
    for ctx in CONTEXTS:
        ch = sorted((q for q in key_tile.chunks()
                     if not q["head"] and q["c"] == ctx),
                    key=lambda q: int(q["segment"].split("L")[1]))
        if not ch:
            continue
        fresh = [q["fresh"] for q in ch]
        n = len(fresh)
        mean = sum(fresh) / n
        worst8 = max(abs(sum(fresh[o::4]) / len(fresh[o::4]) / mean - 1) for o in range(4))
        no0 = fresh[1:]
        out[str(ctx)] = dict(
            layers=n, mean_fresh_per_chunk=mean,
            single_layer_pct=[100 * (min(fresh) / mean - 1), 100 * (max(fresh) / mean - 1)],
            single_layer_pct_without_the_embedding_layer=[
                100 * (min(no0) / mean - 1), 100 * (max(no0) / mean - 1)],
            worst_even_8_subsample_pct=100 * worst8,
            widest_layer=ch[fresh.index(max(fresh))]["segment"],
            narrowest_layer=ch[fresh.index(min(fresh))]["segment"],
            run=ch[0]["run"],
            note="the widest layer is the embedding one: it carries the embedding "
                 "lookup, the largest master exponent table of the 32 and one more "
                 "cell per attention-output accumulator. Without it the spread is "
                 "the row above.")
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
        "PASS   rho in [0.90, 1.10] -- the range as registered, unchanged. What "
        "the same predictor shows at the two scales it WAS built from is not "
        "quoted here: it moves whenever the model does, and a number in this "
        "sentence went stale once already. It is computed into the `anchors` "
        "block of this document, per scale and per context, with the statement "
        "each measured side was proved at beside it. The range is NOT rebased "
        "on any of them. The licence is claimed with the measured residual "
        "quoted beside it.\n"
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
            basis="the percentages here are the ones registered and are NOT "
                  "re-derived; `layer_spread` in this document is what the two "
                  "headline runs' own 32 chunks per context say today, computed "
                  "rather than quoted"),
        C_single_layer=dict(
            what="one layer only -- the minimum that still tests anything",
            widen_pct_upper=9.0, widen_pct_lower=1.5,
            basis="the percentages here are the ones registered and are NOT "
                  "re-derived; `layer_spread` in this document is what the two "
                  "headline runs say today, and the run MUST report "
                  "`params.log_table` and `interaction_by_family.table.cells` "
                  "for the layer it proved. The whole chunk is the exact "
                  "identity `chunk_identity` states, which a single-layer run "
                  "can check on its own chunk."),
    ),
    also=(
        "SEPARATE and EXACT: the key tiles family. `family_cells.key_tiles` (base) and "
        "`family_cells.interaction_by_family.key_tiles.cells` must equal the registered "
        "integers TO THE CELL, as they do on every recorded layer chunk at both contexts "
        "(`harness.costmodel.key_tile --selftest`). The registered interaction is "
        "the statement WITH the five reductions -- `--diagonal-block --gkr-key-tiles "
        "--gkr-dequant --gkr-residual-stream --gkr-freivalds --gkr-attnout`, the "
        "configuration `tools/drivers/predict-14b.sh` proves every layer under and "
        "the one both headline runs used. A run WITHOUT `--gkr-key-tiles` commits the "
        "LogUp running sums instead, reports a LARGER interaction, and is compared "
        "against `key_tile_family(..., gkr=False)` at the same shape. The base is the "
        "same integer either way. Any nonzero deviation is a finding in its own right "
        "and must be reported even if rho passes -- it means the layout identities "
        "in `key_tile_cells` do not hold at a shape whose `nh` and `d` are not "
        "powers of two, which is the one thing the closed form's own docstring "
        "flags.\n"
        "SEPARATE and EXACT, the second one: the BOUNDARY TREES. "
        "`cells.fresh - cells.base - cells.interaction` must equal "
        "`2*d*S + 2*nh*hd*(2*C + S)` on every layer chunk and `d*S` on every "
        "output head chunk, as it does on all 68 recorded 8B chunks of the two "
        "headline runs (`harness.costmodel.key_tile --selftest`). No "
        "`family_cells` entry names this family, so a run that does not report "
        "`cells.base` and `cells.interaction` beside `cells.fresh` cannot be "
        "checked against it. Like the key tiles, a deviation is a finding in "
        "its own right and is reported whatever rho does; unlike a "
        "configuration miss, it does not void the run."),
    invalidation=(
        "The comparison is void, not failed, if the run does not reproduce the "
        "registered configuration: spec `b10-final` (round even, all seven "
        "matmuls on the `chain` dequant rung), S=512, diagonal block key tile ON, "
        "grid_target 64 / chan_target 128 / rw 16, k_bits 11 / q_bits 12, "
        "bounds on, and a REAL extracted witness (not `ZK_SYNTHETIC_DIMS`). "
        "A void run is re-run, never re-bracketed."),
)


def registered_statement():
    """What the registered key tile integers price, said in the document itself.

    Four settings and the driver flags they correspond to, so that a reader of
    `registration.json` can see which statement the integers assume without
    reading this file.  It is part of the hashed content: a registration that
    priced another statement is another registration.
    """
    return dict(
        STATEMENT,
        driver="tools/drivers/predict-14b.sh (zk-results), which proves every "
               "layer of the target shape with these flags",
        driver_flags="--diagonal-block --gkr-key-tiles --gkr-dequant "
                     "--gkr-residual-stream --gkr-freivalds --gkr-attnout",
        flags=dict(STATEMENT_FLAGS),
        key_tiles_half="priced at exactly this statement: "
                       "`key_tile_family(nh, hd, S, C, **STATEMENT)`",
        rest_half=dict(STATEMENT_NOT_PRICED),
    )


def statement(pt):
    """The key tile configuration the measured chunk was proved under, off its own
    `params`: whether the prefix is tiled into key blocks, whether the chunk binds
    its K/Q/V to the boundary roots, and whether the five reductions were on.

    Read rather than assumed, because the closed form prices a different family
    under each and the chunk is the only thing that says which it is.
    """
    return dict(grid_target=pt["params"]["grid_target"],
                chan_target=pt["params"]["chan_target"], rw=pt["params"]["rw"],
                diagonal_block=bool(pt["params"]["diagonal_block"]),
                kv_boundary=bool(pt["params"].get("kv_boundary")),
                bind=pt["params"].get("rope") is not None,
                gkr=bool(pt["params"].get("gkr_key_tiles")))


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
        st = statement(pt)
        bb, bi, sat = key_tiles_per_chunk(nh, hd, S, ctx, **st)
        mb = pt["family_cells"]["key_tiles"]
        mi = pt["family_cells"]["interaction_by_family"]["key_tiles"]["cells"]
        assert int(bb) == mb, (ctx, bb, mb)
        assert int(round(bi)) == mi, (ctx, bi, mi)
        # and the local reimplementation must equal cost_model's own key_tile_fresh
        bf, _ = cm.key_tile_fresh(nh, hd, S, ctx, **st)
        assert abs(bf - (bb + bi)) < 1e-6, (ctx, bf, bb + bi)
        out.append(dict(ctx=ctx, key_tiles_base=mb, key_tiles_interaction=mi, exact=True,
                        saturated=sat, statement=st))
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
        # the same digests the frozen inputs carry, so the two cannot drift
        inputs_sha256=dict(frozen_inputs()["code_sha256"],
                           **{"constants.json": _sha(constants.path())}),
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
            entry_points=["cells_per_token(cfg, ctx, batch, seq, head=False, gkr=...)",
                          "residual_stream_cells(cfg, 'tree', gkr=...)",
                          "key_tile_family", "key_tile_cells", "key_tile_fresh",
                          "boundary_cells", "lookup_table_cells", "lm_head_chunk"],
            reductions_priced=list(STATEMENT_REDUCTIONS),
            not_the_engine="experiments/budget.py is an early one-shape proxy "
                           "(eval-plan.md section 3) and is not used here",
            attention_term="the KEY_TILE MODEL (alpha=None). The legacy single-slope "
                           "alpha=4.00 path is NOT used -- section 7.5(e)'s ruling "
                           "is that the paper uses the composed-chunk value.",
            lookup_batch=B, seq_per_chunk=S,
        ),
    )


def frozen_inputs():
    """What this prediction actually read, frozen with it.

    It used to be a list of constants keys, and most of them were in no
    constants file: three of them named the key tile model's fitted
    coefficients, which this predictor does not read at all -- the key tile
    half is a closed form, so its coefficients are an output of the layout and
    not an input to the prediction.  What the prediction does read is three
    things, and all three are here:

      * every cell cost the model charged, with the source each came from --
        `cost_model.PROVENANCE`, which #153 made the model's own record of
        constants-first reading with an archive fallback.  A cost the model
        could not price is in it too, marked absent;
      * the statement priced, which is what makes the integers mean anything;
      * the code that computed it, by digest, and the two measured chunks it
        was anchored on, by run and segment.

    A key that IS in the constants file and was read still appears, under
    `constants_entries`, with its grade -- that is the graded provenance the
    freeze was for -- and one that is not is not silently dropped: the
    `cell_costs` entry for it says `absent` and names the key.
    """
    cells = {n: dict(e) for n, e in sorted(cm.PROVENANCE.items())}
    # `boundary` is what an older constants file called the timing scope; a
    # cell count has neither, so both are optional here and neither defaults.
    def entry(k):
        e = get(k)
        return {f: e[f] for f in ("value", "grade", "card", "context", "unit",
                                  "timing_scope", "boundary", "run", "artifact")
                if f in e}

    from_constants = {e["key"]: entry(e["key"])
                      for e in cells.values() if e["source"] == "constants" and e["key"]}
    unpriced = sorted(n for n, e in cells.items() if e["source"] == "absent")
    return dict(
        cell_costs=cells,
        constants_entries=from_constants,
        constants_file_sha256=_sha(constants.path()),
        n_from_constants=len(from_constants), n_inputs=len(cells),
        not_priced=unpriced,
        statement=registered_statement(),
        reference_chunks={str(c): _seq5_source(c) for c in CONTEXTS},
        code_sha256={
            "harness/costmodel/cost_model.py": _sha(cm.__file__),
            "harness/costmodel/key_tile.py": _sha(key_tile.__file__),
            "harness/costmodel/layer_chunks.json": _sha(str(key_tile.CHUNKS)),
            "harness/costmodel/composed_rungs.json": _sha(str(cm.COMPOSED_RUNGS)),
            "harness/eval/predict/predict.py": _sha(os.path.join(HERE, "predict.py")),
        })


# The card rate the secondary tok/s column would need, and the key that would
# hold it.  Neither is part of the prediction: a cell count is the same on
# either card, and the rate is a card measurement the registration is not
# making.  zk-results holds `rate.headline.*` for the H100 and no H200 rate at
# all, so on that constants file this column is not printed rather than
# computed from something else.
RATE_KEYS = {"4096": "rate.h200.c4096.commit_layers32",
             "16384": "rate.h200.c16384.commit_layers32"}


def implied_tok_s(reg):
    """Secondary, and conditional on a rate that is NOT part of the prediction."""
    out = {}
    for name, e in reg.items():
        row = {}
        for ctx, key in RATE_KEYS.items():
            if ctx not in e["contexts"]:
                continue
            r = constants.find(key)
            if r is None:
                row[ctx] = dict(tok_s_commit_boundary=None, rate_key=key, rate=None,
                                note="this constants file does not name the H200 "
                                     "commit rate, so no tok/s is implied here")
                continue
            rate = r["value"] * 1e9
            row[ctx] = dict(tok_s_commit_boundary=rate / e["contexts"][ctx]["cells_per_token_layers"],
                            rate_key=key, rate=r["value"], card=r["card"],
                            boundary=r["boundary"])
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
        kt = cm.cells_per_token(g, ctx, B, seq=S, gkr=STATEMENT_REDUCTIONS)["total"]
        e = constants.find(key)
        if e is None:
            rows[str(ctx)] = dict(frozen=None, key=key, rerun_alpha_4p00=alpha4,
                                  rerun_key_tile_model=kt,
                                  note="this constants file does not name the frozen "
                                       "27B row, so there is nothing to compare the "
                                       "re-run against")
            continue
        rows[str(ctx)] = dict(frozen=e["value"], rerun_alpha_4p00=alpha4,
                              rerun_key_tile_model=kt,
                              key_tile_over_frozen=kt / e["value"],
                              understated_pct=100.0 * (kt / e["value"] - 1.0))
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
                dict(key=k, grade=(constants.find(k) or {}).get("grade", "absent"))
                for k in ("attention_grid.b5_standalone_alpha",
                          "model_constants.attention_cells_per_head_key_token")
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
    content = dict(schema="e4-registration/1", statement=registered_statement(),
                   shapes=reg, bracket=BRACKET,
                   anchors=anc, key_tiles_selfcheck_8b=chk,
                   chunk_identity=chunk_identity(), layer_spread=layer_spread(),
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
    st = registered_statement()
    print("the statement priced:", " ".join(f"{k}={STATEMENT[k]}" for k in sorted(STATEMENT)))
    print(f"  the run it is registered against: {st['driver_flags']}")
    print(f"  not priced: {STATEMENT_NOT_PRICED['flags']} -- {STATEMENT_NOT_PRICED['effect']}")
    print()
    print("key tiles self-check against the 8B artifacts (must be exact to the cell):")
    for c in chk:
        print(f"  C={c['ctx']:<6} base {c['key_tiles_base']:>13,}  interaction "
              f"{c['key_tiles_interaction']:>13,}  exact={c['exact']}  saturated={c['saturated']}")
    print()
    print("the predictor's residual at the two scales it WAS built from:")
    for k, v in anc.items():
        if v["measured"] is None:
            print(f"  {k:<34} predicted {v['predicted']:>15,.0f}  "
                  f"measured    not in this constants file ({v.get('key') or v['source']})")
            continue
        print(f"  {k:<34} predicted {v['predicted']:>15,.0f}  "
              f"measured {v['measured']:>15,.0f}  ratio {v['ratio']:.4f}  "
              f"({100*(v['ratio']-1):+.2f}%)")
        print(f"  {'':<34} measured side, {v['grade']}: {v['source']}")
        if v.get("embedding_lookup_per_position"):
            print(f"  {'':<34} the embedding lookup in it, which the model prices as "
                  f"no layer's cost: {v['embedding_lookup_per_position']:,.0f} a position, "
                  f"ratio without it {v['ratio_without_the_embedding_layer']:.4f}"
                  f"   recomputed from this repository's copy of the 32 segments: "
                  f"{v['measured_matches_recorded_chunks']}")
    print()
    ci = chunk_identity()
    print(f"one layer chunk, to the cell on {ci['chunks']} recorded chunks at both contexts:")
    print(f"  {ci['form']}")
    print(f"  rest = {ci['rest']:,}   (the embedding layer: {ci['rest_embedding_layer']:,})")
    ls = layer_spread()
    print("how far one layer chunk is from the mean of its 32, measured:")
    for ctx, v in ls.items():
        lo, hi = v["single_layer_pct"]
        nlo, nhi = v["single_layer_pct_without_the_embedding_layer"]
        print(f"  C={ctx:<6} one layer {lo:+.2f}% .. {hi:+.2f}% "
              f"(without the embedding layer {nlo:+.2f}% .. {nhi:+.2f}%), "
              f"worst even 8 of 32 {v['worst_even_8_subsample_pct']:.2f}%")
    w = BRACKET["protocol_widening"]
    print(f"  registered widening, unchanged: protocol B {w['B_even_subset']['widen_pct']}%, "
          f"protocol C -{w['C_single_layer']['widen_pct_lower']}% / "
          f"+{w['C_single_layer']['widen_pct_upper']}%")
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
        if r["frozen"] is None:
            print(f"  C={ctx:<6} not in this constants file ({r['key']}); the re-runs are "
                  f"alpha 4.00 {r['rerun_alpha_4p00']:,.0f} and key tile model "
                  f"{r['rerun_key_tile_model']:,.0f}")
            continue
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
