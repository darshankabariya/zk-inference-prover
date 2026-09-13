"""E1 -- the frontier price table.

The paper's thesis is that the cost of verified inference is a *computable
function* of a model's architecture and a card's measured hash rate.  This module
is that computation: it takes the B2 cost model (`cost_model.py`, extended by this
stream with experts / top-k / sequence-batch terms), calibrates it against the two
sequences zk-results has measured end to end, and prices nine frontier shapes at
both contexts, at all four boundaries, on both cards.

What the table prices is the SOUND STATEMENT: a prompt and the output the model
writes from it, proved as one statement in token blocks, per proved position.
That is what zk-results' two headline runs prove and what the paper quotes
(`docs/decisions/2026-09-08-cost-per-proved-position-not-an-overhead-multiple.md`),
and it is not the decode-only statement the private archive's sequence proved.
The two differ by more than a third at the long context, so a rate measured on
one of them and a cell count taken on the other do not divide.

Three rules it keeps, and they are the point:

 1. **No measured constant is written down here.**  Every number the table stands
    on is read out of zk-results' `constants.json` (`--constants PATH`) where a key
    names it, and out of a checked-in artifact under `--artifacts DIR` where none
    does yet; which of the two each one came from is printed and written into the
    output.  A missing rate prints `n/a`; it never defaults, and it never falls
    back to zero.
 2. **Every architectural number has a source**, and every shape carries a
    self-check: the model's own parameter count against the published total and
    active counts.  A shape that fails its check is marked `[check]` and the
    failure is printed, not smoothed.
 3. **The 8B row the model was calibrated on is checked, not tuned.**  The
    calibration has no free parameter at all and two measured points, so it has
    two residual degrees of freedom, and that residual is printed and recorded
    (`frontier.calibration.residual_pct`).

    uv run python -m harness.costmodel.frontier --constants PATH
    uv run python -m harness.costmodel.frontier --constants PATH --artifacts DIR \
        --inputs DIR --seq-batch 32
"""
import argparse
import json
import math
import pathlib
import re
import sys

from harness import constants, inputs, provenance
from . import cost_model as cm
from . import exact_cells as ec
from .attention import latent as la
from . import key_tile

HERE = pathlib.Path(__file__).parent
# The measured run outputs this module reads, laid out as the readers below name
# them (`b10/out/seq5/...`, `b9/out/...`, `b12/.../out/...`).  Optional: every
# reader tests its own file, and the two sequence runs -- the only inputs no
# constants key can replace today -- say so by name when the directory is absent.
ROOT = inputs.artifacts_dir(optional=True)
OUT = pathlib.Path(".")

TOKENS = 512            # the measured sequence's decoded tokens (S per chunk)
REPO = ROOT


def art(*parts):
    """A path under `--artifacts DIR`, or None when no directory was given."""
    return None if ROOT is None else ROOT.joinpath(*parts)


def rel(p):
    """Paths in the printed table are repo-relative: the table is a committed
    artifact and a worktree path in it is noise."""
    if REPO is None:
        return str(p)
    try:
        return str(pathlib.Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- constants --
# Read from a zk-results `constants.json` (`--constants PATH`).  Nothing measured,
# cited or agreed is written down here: this module names the keys it needs, fails
# loudly on a `null` in one a priced row depends on, and says `not recorded` where
# a key only a diagnostic reads is not in the file yet.

# The keys a priced row depends on, and nothing else: every `usd_per_mtok` in
# the table is a rate times a time, so a missing rate is a missing row and this
# module fails on it.  The `*_reference_*` pair only; an `*_actual_*` key must
# never be read.
KEYS = dict(
    h100_usd_per_hour="cost.h100_reference_usd_per_hour",
    h200_usd_per_hour="cost.h200_reference_usd_per_hour",
)

# The keys a diagnostic reads and no priced row depends on.  Each one that the
# constants file does not name yet prints `not recorded` where it would have been
# read, and the table around it is printed without it.  Nothing here defaults and
# nothing here falls back to zero.
OPTIONAL = dict(
    # the attention term's base coefficients, measured by `prefill-sweep-8b`.  The
    # drift guard checks these COEFFICIENTS, not a composite slope: a single
    # cells-per-(head, key, token) number is a per-entry cost and a per-key cost
    # collapsed together by the accident that the calibration point has S=512, and
    # it is valid only at that S.  There is no `fresh_cell_coeffs` beside them:
    # with the five reductions on, most of the key tiles' interaction sits in the
    # reduction rather than in their own columns, so solving for it over the
    # measured points returns a negative per-token cost, and a negative number
    # cannot be a cost.  The interaction side is checked instead by the family
    # reproducing every recorded segment's committed cells exactly.
    key_tile_base_coeffs="attention_grid.base_cell_coeffs",
    # the superseded single slopes, read only so section (2a) can show what they cost
    alpha_e1="attention_grid.e1_composed_solve",
    alpha_b5="attention_grid.b5_standalone_alpha",
    # E11, re-priced by stream E7 on the key tile model
    e11_multiple="design.e11_cells_multiple_c4096",
    e11_chunking_net="design.e11_chunking_net_pct",
    e11_chunking_fixed="design.e11_chunking_overhead_pct",
    chunk_floor="attention_grid.chunk_model_fixed_composed",
    # the fresh cells of the 32 layer segments of ONE token block, per position:
    # the block whose prefix is the whole context, no output head segment in it.
    # It is what the exact count is calibrated against, because it is the same
    # segments the count builds `params` blocks for.
    block_c4096="cells_per_position.headline.c4096.fresh_layers_block",
    block_c16384="cells_per_position.headline.c16384.fresh_layers_block",
)


def load_constants():
    """`{alias: entry}`, each entry carrying the constants file's own value / unit /
    grade / card, plus the key it came from and where it was read.

    A required key that is not in the file is an error.  An optional one that is
    not gets a `null` value and `recorded=False`, so the diagnostic that would
    have read it prints `not recorded` instead of exiting."""
    out = {}
    for alias, key in KEYS.items():
        e = dict(constants.get(key))
        e.update(origin=constants.path(), key=key, recorded=True)
        out[alias] = e
    for alias, key in OPTIONAL.items():
        e = constants.find(key)
        out[alias] = (dict(e, origin=constants.path(), key=key, recorded=True) if e
                      else dict(value=None, grade="not recorded", origin=constants.path(),
                                key=key, recorded=False))
    return out


# ----------------------------------------------------------- the statement --
# The sound statement, and the unit every priced row is in.
#
# zk-results' two headline runs prove a prompt and the output the model writes
# from it as ONE statement: `positions.headline.c{ctx}` positions in token blocks
# of S, each block's layer segments attending over the blocks before it, and two
# output head segments beside them, `segments.headline.c{ctx}` proofs in all.  A
# position is the unit the paper quotes.  So the statement's own numbers -- how
# many positions, in how many blocks, with how many head segments -- are read out
# of the constants file here and transported to the other shapes, and nothing
# about it is typed in.
#
# The five batched GKR reductions are on, which is the paper's configuration
# (zk-results' decision of 2026-09-03), so the model prices each family at what
# the composed prover builds it at and not at its standalone rung.
STATEMENT_GKR = cm.REDUCTIONS
CONTEXTS = (4096, 16384)
# The four timing scopes: the name the constants file's keys use, and the name
# this module has printed since W0's freeze.  All four are kept apart on purpose
# and a recorded row names the one it was taken at.
SCOPES = (("commit", "commit"), ("stwo_total", "stwo total"),
          ("whole_wall", "whole wall"), ("loop", "loop"))


def headline_card():
    """The measured side, out of zk-results' `constants.json`: the card the two
    headline runs were proved on, at four scopes and two contexts.

    Every number is a key -- the cells the sequence committed, the positions it
    proved, the segments it took, the rate at each scope, the proof bytes a
    position cost and the peak the card held -- so a run made from this side alone
    reads nothing outside a repository and is recordable.  Returns None when the
    constants file names none of it, so that the archive layout below can still
    answer and say that it did."""
    per_ctx, keys = {}, []
    for ctx in CONTEXTS:
        cells = constants.find(f"cells.headline.c{ctx}.fresh")
        pos = constants.find(f"positions.headline.c{ctx}")
        segs = constants.find(f"segments.headline.c{ctx}")
        if cells is None or pos is None or segs is None:
            continue
        read = [f"cells.headline.c{ctx}.fresh", f"positions.headline.c{ctx}",
                f"segments.headline.c{ctx}"]
        boundaries = {}
        for scope, name in SCOPES:
            e = constants.find(f"rate.headline.c{ctx}.{scope}")
            if e is None:
                continue
            # the file records a rate in Mcell/s; this module divides cells
            boundaries[name] = dict(cells_s=e["value"] * 1e6, scope=scope,
                                    key=f"rate.headline.c{ctx}.{scope}",
                                    secs=cells["value"] / (e["value"] * 1e6))
            read.append(f"rate.headline.c{ctx}.{scope}")
        if not boundaries:
            continue
        proof = constants.find(f"proof.headline.c{ctx}.bytes_per_position")
        seg_bytes = constants.find(f"proof.headline.c{ctx}.bytes_per_segment_median")
        peak = constants.find(f"mem.headline.c{ctx}.peak_mib")
        for e, k in ((proof, f"proof.headline.c{ctx}.bytes_per_position"),
                     (seg_bytes, f"proof.headline.c{ctx}.bytes_per_segment_median"),
                     (peak, f"mem.headline.c{ctx}.peak_mib")):
            if e is not None:
                read.append(k)
        per_ctx[ctx] = dict(
            fresh=cells["value"], positions=pos["value"], segments=segs["value"],
            cells_per_position=cells["value"] / pos["value"],
            mib_per_position=None if proof is None else proof["value"] / 2 ** 20,
            mib_per_segment=None if seg_bytes is None else seg_bytes["value"] / 2 ** 20,
            peak_mib=None if peak is None else peak["value"],
            boundaries=boundaries, source=constants.path(), keys=read,
            run=cells.get("run"), card=cells["card"])
        keys += read
    if not per_ctx:
        return None
    # the card is the entry's own, not a name this module carries
    cards = {v["card"] for v in per_ctx.values()}
    if len(cards) != 1:
        sys.exit(f"the recorded sequences are on more than one card, {sorted(cards)}; "
                 "this table prices one card at a time")
    return dict(card=cards.pop(), source="constants", per_ctx=per_ctx, keys=keys,
                statement="sound", tokens=None,
                note="the prompt and the output proved as one statement, per proved "
                     "position, the five reductions on; read from the constants file "
                     "and from nothing else, so a row priced on it is recordable")


# ------------------------------------------------------- measured artifacts --
# The archive's two sequences, and they are the FALLBACK.  Each is parsed out of a
# run output in `--artifacts DIR` -- a directory that is in no repository -- and
# each proves the decode-only statement: `TOKENS` decoded tokens, every one of
# them against a full C-key prefix, at a prover that predates the tiled prefix,
# the root-equality claims and the five reductions.  A row priced on one of them
# is a row about a different statement and this module marks it as one.
H200_SUMMARY = art("b10", "out", "seq5", "n_summary_c{ctx}.json")
H100_LOOP = art("b10", "out", "w", "fin_pipe_c4096.json")

NO_SEQUENCE = (
    "no measured card rate: the constants file names none of\n"
    "  cells.headline.c{4096,16384}.fresh, positions.headline.c{4096,16384},\n"
    "  segments.headline.c{4096,16384}, rate.headline.c{4096,16384}.{commit,"
    "stwo_total,whole_wall,loop}\n"
    "and the artifacts directory holds neither archive sequence:\n"
    "  b10/out/seq5/n_summary_c{4096,16384}.json -- H200, 33 chunks, decode-only\n"
    "  b10/out/w/fin_pipe_c4096.json             -- H100, 32 layer chunks, pipelined\n"
    "Pass --constants PATH at a zk-results checkout that records the headline runs, "
    "or --artifacts DIR.")

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
    family env.  `briefs/n-alldev.md` Record (5); artifacts in b10/out/seq5/.

    None when the artifacts directory does not hold it: it is the fallback, and a
    reader that has the constants file's own card does not need it."""
    if H200_SUMMARY is None:
        return None
    cards = {}
    for ctx in (4096, 16384):
        p = pathlib.Path(str(H200_SUMMARY).format(ctx=ctx))
        if not p.exists():
            return None
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
    return dict(card="H200", tokens=TOKENS, per_ctx=cards, source="archive",
                statement="decode-only", keys=[],
                note="33 chunks (32 layers + the head), one process per chunk, "
                     "not pipelined; stream N's seq5.  The DECODE-ONLY statement: "
                     "512 tokens each against a full C-key prefix, proved before "
                     "the tiled prefix, the root-equality claims and the five "
                     "reductions existed.  Read from the archive, so nothing "
                     "priced on it is recordable")


def h100_card():
    """H100, stream W's final pipelined loop, 32 layer chunks (no head), C=4096.
    `briefs/w-genbuild.md` Record (5); artifact b10/out/w/fin_pipe_c4096.json.

    None when the artifacts directory does not hold it, like `h200_card`."""
    if H100_LOOP is None or not H100_LOOP.exists():
        return None
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
        source="archive", statement="decode-only", keys=[],
        note="32 layer chunks, no head, pipelined (`ZK_PIPELINE=1`).  `loop_secs` "
             "is W0's FOURTH boundary and is not `whole wall`: it counts the "
             "witness read and the layer build, which `whole wall` has never "
             "counted, and it hides the prove behind the next chunk's generation. "
             "Here it is both wider (112.89 -> it counts more work) and faster "
             "(96.33 s -> it overlaps), which is exactly why the two must not be "
             "compared as if one were a subset of the other.")


# ------------------------------------------------------------- the shapes ----
# Every field below is read out of the model's own published `config.json`.  The
# owner's ruling of 2026-09-11 on what a projection may assume applies here too:
# the fewer assumptions the better, and a model's own architecture is the way to
# have fewer.  So no dimension in this table is solved or approximated except
# where the row says so, and each row names the repository its config was read
# from and the date.
def mla(d, nh, q_lora, kv_lora, qk_nope=128, qk_rope=64, v_head=128, rope=True):
    """DeepSeek-V3-family multi-head latent attention, as parameters and as
    committed accumulators per layer.  The cost model's GQA formulas do not
    describe MLA: the q and kv sides are low-rank pairs, so both the weight count
    and the accumulator count differ, and the cache is one latent (kv_lora +
    qk_rope) rather than a K and a V.

    `cfg` beside them is the five published fields themselves, which is what the
    counted path builds a `params` block from, and both counts below are that
    module's own so the two paths read one formula (`attention/latent.py`).
    `rope` is `False` where a config says its attention applies no rotation."""
    cfg = dict(q_lora_rank=q_lora, kv_lora_rank=kv_lora, qk_nope_head_dim=qk_nope,
               qk_rope_head_dim=qk_rope, v_head_dim=v_head)
    # the fitted path takes the materialised form's accumulators, which is the
    # form a counted row records; the weights are the same in both
    z = la.dims(cfg, nh, "materialised")
    # what the model caches per key per layer, expressed as the GQA term
    # `2 * nkv * hd` the cost model charges, so that nkv is an *effective* number
    return dict(attn_params_per_layer=la.attn_params(z, d),
                attn_accs_per_layer=la.attn_accs(z, d),
                nkv_eff=z["latent"] / (2.0 * v_head), cfg=cfg, rope=rope)


def group(layers, dff=0, routed=0, per_token=0, shared=0, moe_dff=0,
          routed_n_in=None, linear=False, **kw):
    """One run of layers that are all alike, from the model's published config.

    Every field is a key of that config and nothing is solved: `dff` is
    `intermediate_size` on a dense layer and zero on a mixture layer, `routed` is
    `n_routed_experts` / `num_experts` / `num_local_experts`, `per_token` is
    `num_experts_per_tok`, `shared` is `n_shared_experts` and `moe_dff` is
    `moe_intermediate_size`.  `routed_n_in` is the width a routed expert reads
    where the config says it is not the model's (`routed_expert_hidden_size`).

    `experts` and `topk` beside them are the folded pair the older per-unit path
    reads -- the shared experts counted as routed ones that every token uses --
    and they are derived here rather than written down twice."""
    width = moe_dff if routed else dff
    experts = (routed + shared) if routed else 1
    topk = (per_token + shared) if routed else 1
    return dict(layers=layers, dff=width, experts=experts, topk=topk, linear=linear,
                dense_dff=dff, routed=routed, per_token=per_token, shared=shared,
                moe_dff=moe_dff, routed_n_in=routed_n_in, **kw)


# Multi-head latent attention, from the three configs' own `q_lora_rank`,
# `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim` and `v_head_dim`.
DS3 = mla(7168, 128, 1536, 512)
K2 = mla(7168, 64, 1536, 512)
# `mla_use_nope` is true in Kimi K3's config: the widths are K2's, the rotation
# is not there, and the position is the linear-attention layers' business
K3 = mla(7168, 96, 1536, 512, rope=False)

# Every field below is read out of the model's own published `config.json`,
# fetched 2026-09-11 (2026-09-10 for the 70B, which has not moved).  Values only:
# `source` names the repository the file was read from and the date, and a shape
# the vendor's published totals do not reproduce is marked `[check]` and printed
# rather than adjusted.  `check_total` / `check_active` are the vendor's counts.
SHAPES = [
    dict(name="Llama-3.1-8B", tag="calibration", d=4096, nh=32, nkv=8, hd=128,
         vocab=128256, groups=[group(32, dff=14336)],
         check_total=8.03e9, check_active=8.03e9,
         source="config.json, meta-llama/Meta-Llama-3.1-8B, fetched 2026-09-11 "
                "from the ungated mirror unsloth/Meta-Llama-3.1-8B, which "
                "agrees: L=32 d=4096 nh=32 nkv=8 head_dim=128 dff=14336 "
                "vocab=128256, untied embeddings; the shape every measured "
                "number in this repository was taken on"),
    dict(name="Llama-3.1-70B", tag="dense", d=8192, nh=64, nkv=8, hd=128,
         vocab=128256, groups=[group(80, dff=28672)],
         check_total=70.55e9, check_active=70.55e9,
         source="[web] config.json, nvidia/Llama-3.1-70B-Instruct-FP8 and "
                "unsloth/Meta-Llama-3.1-70B, which agree, fetched 2026-09-10: "
                "L=80 d=8192 nh=64 nkv=8 dff=28672 vocab=128256, untied "
                "embeddings; head_dim=128. The published count is 70,553,706,496"),
    dict(name="Llama-3.1-405B", tag="dense", d=16384, nh=128, nkv=8, hd=128,
         vocab=128256, groups=[group(126, dff=53248)],
         check_total=405.85e9, check_active=405.85e9,
         source="[web] config.json, NousResearch/Hermes-3-Llama-3.1-405B, which "
                "is the 405B architecture and is not gated, fetched 2026-09-11: "
                "L=126 d=16384 nh=128 nkv=8 dff=53248 vocab=128256, untied "
                "embeddings. A 4-bit repack of the same model "
                "(hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4) states "
                "num_key_value_heads 16; the base config and the model card both "
                "say 8 and that is what is used"),
    dict(name="1T dense", tag="dense, hypothetical", d=20480, nh=160, nkv=16, hd=128,
         vocab=128256, groups=[group(160, dff=None)], solve_dff_for=1.0e12,
         check_total=1.0e12, check_active=1.0e12, solved="dff",
         source="[hypothetical] defined here, not a real model, and the ONE row "
                "in this table with no published config: Llama-405B's aspect "
                "ratio scaled to 1.0 T parameters with the feed-forward width "
                "solved for that target"),
    dict(name="DeepSeek-V3", tag="MoE 256/8", d=7168, nh=128, nkv=DS3["nkv_eff"], hd=128,
         vocab=129280,
         latent=DS3["cfg"], latent_rope=DS3["rope"],
         groups=[group(3, dff=18432, **{k: DS3[k] for k in ("attn_params_per_layer",
                                                            "attn_accs_per_layer")}),
                 group(58, routed=256, per_token=8, shared=1, moe_dff=2048,
                       **{k: DS3[k] for k in ("attn_params_per_layer",
                                              "attn_accs_per_layer")})],
         check_total=671e9, check_active=37e9,
         source="[web] config.json, deepseek-ai/DeepSeek-V3, fetched 2026-09-11: "
                "num_hidden_layers 61, hidden_size 7168, num_attention_heads 128, "
                "vocab_size 129280, MLA q_lora_rank 1536 kv_lora_rank 512 "
                "qk_nope_head_dim 128 qk_rope_head_dim 64 v_head_dim 128, "
                "first_k_dense_replace 3 at intermediate_size 18432, "
                "moe_intermediate_size 2048, n_routed_experts 256, "
                "n_shared_experts 1, num_experts_per_tok 8, moe_layer_freq 1, "
                "untied embeddings"),
    dict(name="Kimi K2", tag="MoE 384/8", d=7168, nh=64, nkv=K2["nkv_eff"], hd=128,
         vocab=163840,
         latent=K2["cfg"], latent_rope=K2["rope"],
         groups=[group(1, dff=18432, **{k: K2[k] for k in ("attn_params_per_layer",
                                                           "attn_accs_per_layer")}),
                 group(60, routed=384, per_token=8, shared=1, moe_dff=2048,
                       **{k: K2[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")})],
         check_total=1.04e12, check_active=32e9,
         source="[web] config.json, moonshotai/Kimi-K2-Instruct, fetched "
                "2026-09-11: num_hidden_layers 61, hidden_size 7168, "
                "num_attention_heads 64, vocab_size 163840, MLA q_lora_rank 1536 "
                "kv_lora_rank 512 qk_nope_head_dim 128 qk_rope_head_dim 64 "
                "v_head_dim 128, first_k_dense_replace 1 at intermediate_size "
                "18432, moe_intermediate_size 2048, n_routed_experts 384, "
                "n_shared_experts 1, num_experts_per_tok 8, untied embeddings"),
    dict(name="Kimi K3", tag="MoE 896/16", d=7168, nh=96, nkv=K3["nkv_eff"], hd=128,
         vocab=163840,
         latent=K3["cfg"], latent_rope=K3["rope"],
         groups=[group(1, dff=33792, linear=True,
                       **{k: K3[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")}),
                 group(24, routed=896, per_token=16, shared=2, moe_dff=3072,
                       routed_n_in=3584,
                       **{k: K3[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")}),
                 group(68, routed=896, per_token=16, shared=2, moe_dff=3072,
                       routed_n_in=3584, linear=True,
                       **{k: K3[k] for k in ("attn_params_per_layer",
                                             "attn_accs_per_layer")})],
         check_total=2.8e12, check_active=104e9,
         source="[web] config.json, moonshotai/Kimi-K3 (text_config), fetched "
                "2026-09-11: num_hidden_layers 93, hidden_size 7168, "
                "num_attention_heads 96, vocab_size 163840, MLA q_lora_rank 1536 "
                "kv_lora_rank 512 qk_nope_head_dim 128 qk_rope_head_dim 64 "
                "v_head_dim 128 -- K2's attention exactly -- and mla_use_nope "
                "true, so the widths are there and the rotation is not: the "
                "counted row builds no rotation for this shape and the position "
                "is the linear layers' business.  first_k_dense_replace 1 at "
                "intermediate_size "
                "33792, moe_intermediate_size 3072, num_experts 896, "
                "num_shared_experts 2, num_experts_per_token 16, "
                "routed_expert_hidden_size 3584 -- the routed experts read a "
                "projection of the residual and not the model's own width, which "
                "is what makes the published 2.8 T come out. "
                "linear_attn_config names 24 full-attention layers and the rest "
                "Kimi Delta Attention, whose proving cost is priced at a zero "
                "attention grid here: a floor and not a model.  The active count "
                "below is short because those layers' own projections are not in "
                "the config in a form this table reads, and the latent-attention "
                "form is charged for them instead"),
    dict(name="Qwen3-235B-A22B", tag="MoE 128/8", d=4096, nh=64, nkv=4, hd=128,
         vocab=151936,
         groups=[group(94, routed=128, per_token=8, shared=0, moe_dff=1536)],
         check_total=235e9, check_active=22e9,
         source="[web] config.json, Qwen/Qwen3-235B-A22B, fetched 2026-09-11: "
                "num_hidden_layers 94, hidden_size 4096, num_attention_heads 64, "
                "num_key_value_heads 4, head_dim 128, vocab_size 151936, "
                "moe_intermediate_size 1536, num_experts 128, "
                "num_experts_per_tok 8, decoder_sparse_step 1 and "
                "mlp_only_layers empty (so every layer is a mixture), no "
                "shared-expert key at all, untied embeddings"),
    dict(name="Llama-4 Maverick", tag="MoE 128/1", d=5120, nh=40, nkv=8, hd=128,
         vocab=202048,
         groups=[group(24, dff=16384),
                 group(24, routed=128, per_token=1, shared=1, moe_dff=8192)],
         check_total=400e9, check_active=17e9,
         source="[web] config.json (text_config), "
                "unsloth/Llama-4-Maverick-17B-128E-Instruct, an ungated mirror of "
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct, fetched "
                "2026-09-11: num_hidden_layers 48, hidden_size 5120, "
                "num_attention_heads 40, num_key_value_heads 8, head_dim 128, "
                "vocab_size 202048, moe_layers the 24 odd indices at "
                "intermediate_size 8192 and the 24 even ones at "
                "intermediate_size_mlp 16384, num_local_experts 128, "
                "num_experts_per_tok 1, untied embeddings. The config names no "
                "shared-expert key; `Llama4TextMoe` carries one at "
                "intermediate_size, and that is where the shared expert here "
                "comes from"),
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
        _w = solve_dff(_s, _s["solve_dff_for"])
        _s["groups"][0].update(dff=_w, dense_dff=_w)
# The weights each run of layers holds, from the published fields and not from
# one folded width: `all` is every expert, which is what a proof covers, and
# `token` is the shared experts plus the top-k a token uses.  Kimi K3's routed
# experts read `routed_expert_hidden_size` and its shared ones read the model's
# own width, which one folded width cannot say.
for _s in SHAPES:
    for _g in _s["groups"]:
        _d, _w = _s["d"], _g["moe_dff"]
        _rn = _g["routed_n_in"] or _d
        _base = 3 * _d * (_g["dense_dff"] or 0) + 3 * _d * _w * _g["shared"]
        _g["ffn_all"] = _base + 3 * _rn * _w * _g["routed"]
        _g["ffn_token"] = _base + 3 * _rn * _w * _g["per_token"]


def attn_params(shape, grp):
    """One layer's attention weights: the shape's own where a group overrides
    them (the latent-attention rows), the GQA form otherwise."""
    if grp.get("attn_params_per_layer") is not None:
        return grp["attn_params_per_layer"]
    d, nh, nkv, hd = shape["d"], shape["nh"], shape["nkv"], shape["hd"]
    return d * nh * hd + 2 * d * nkv * hd + nh * hd * d


def cfg_of(shape, grp):
    return (grp["layers"], shape["d"], shape["nh"], shape["nkv"], shape["hd"],
            grp["dff"], shape["vocab"])


def params_of(shape, freivalds=False, embedding=False):
    """Weights the *proof* sees: the `lm_head`, the attention, and either every
    expert (`freivalds=True`) or the top-k a token actually uses.

    `embedding=True` adds the input embedding table.  The proof never covers it --
    it is a lookup, not a matmul, and the cost model has always excluded it -- but
    a vendor's published parameter count includes it, so the shape checks add it
    back to compare like with like."""
    tot = shape["vocab"] * shape["d"] * (2 if embedding else 1)
    for grp in shape["groups"]:
        tot += grp["layers"] * (attn_params(shape, grp)
                                + (grp["ffn_all"] if freivalds else grp["ffn_token"]))
    return tot


# ------------------------------------------------------------ calibration ----
def measured_8b(card):
    """What one unit of the card's own statement measured: a proved position where
    the card is the constants file's headline card, a decoded token where it is one
    of the archive's decode-only sequences.  The two are not the same quantity and
    the caller says which it is comparing against."""
    per = card["per_ctx"]
    if all("cells_per_position" in v for v in per.values()):
        return {c: per[c]["cells_per_position"] for c in per}
    return {c: per[c]["cells_per_token"] for c in per}


def statement_shape(card, shape, s_chunk=TOKENS):
    """How the sound statement is laid out, read off the run that recorded it: the
    block length, how many positions the output is, and how many output head
    segments the plan carries beside the layer segments.

    Nothing here is typed in.  The head count is `segments - blocks * layers` on the
    run that recorded both, at the shape that run was made on, and it is checked at
    the second context rather than assumed to hold there.  These three numbers are
    what a shape with no measured run of its own inherits."""
    layers = sum(grp["layers"] for grp in shape["groups"])
    got = None
    for ctx in sorted(card.get("per_ctx", {})):
        ref = card["per_ctx"][ctx]
        if not ref.get("positions") or not ref.get("segments"):
            continue
        positions, segments = int(ref["positions"]), int(ref["segments"])
        blocks, rem = divmod(positions, s_chunk)
        if rem:
            sys.exit(f"positions.headline.c{ctx} is {positions}, which is not a whole "
                     f"number of {s_chunk}-position blocks: the statement's block length "
                     f"and the recorded sequence's disagree.  Pass --seq S.")
        heads = segments - blocks * layers
        this = dict(s_chunk=s_chunk, n_output=positions - ctx, heads=heads,
                    source="constants", read_at=ctx,
                    keys=[f"positions.headline.c{ctx}", f"segments.headline.c{ctx}"])
        if got is None:
            got = this
            continue
        if (heads, this["n_output"]) != (got["heads"], got["n_output"]):
            sys.exit(f"the two recorded sequences do not prove the same statement: "
                     f"C={got['read_at']} is {got['n_output']} output positions and "
                     f"{got['heads']} head segments, C={ctx} is {this['n_output']} and "
                     f"{heads}.  One of them is not the statement this table prices.")
        got["keys"] += this["keys"]
    if got is None:
        # no run recorded it: the archive's sequences prove the decode-only
        # statement and say nothing about a prompt, so this is the plan the two
        # headline runs are known to have proved, named as an assumption.
        return dict(s_chunk=s_chunk, n_output=s_chunk, heads=2, source="assumed",
                    read_at=None, keys=[])
    return got


def statement_of(shape, ctx, shp):
    """The sound statement at one context and one shape: how many positions are
    proved, in how many blocks, over how many layer segments and head segments.

    The layout comes from `statement_shape` -- the run that recorded it -- and only
    the layer count is the shape's own.  That transport is the extrapolation every
    other row in this table makes and it is named here rather than hidden."""
    layers = sum(grp["layers"] for grp in shape["groups"])
    positions = ctx + shp["n_output"]
    blocks, rem = divmod(positions, shp["s_chunk"])
    if rem:
        sys.exit(f"a prompt of {ctx} and an output of {shp['n_output']} is not a whole "
                 f"number of {shp['s_chunk']}-position blocks")
    return dict(positions=positions, blocks=blocks, s_chunk=shp["s_chunk"],
                heads=shp["heads"], layers=layers,
                segments=blocks * layers + shp["heads"], ctx=ctx,
                n_output=shp["n_output"], source=shp["source"], keys=list(shp["keys"]),
                gkr=tuple(STATEMENT_GKR))


# --------------------------------------------------- the exact cell count ----
# The owner's ruling of 2026-09-11 (`docs/decisions/2026-09-11-cells-are-counted-
# from-the-layout-wherever-a-shape-can-be-stated.md`): wherever a shape can be
# stated as a `params` block, the cells are COUNTED from the layout and not
# priced from per-unit constants.  So the cell side of every row below builds
# one `params` block per segment -- the layer segment, the segment of the stack
# that carries the embedding lookup, the output head -- and calls `exact_cells`,
# which reproduces all 1,872 recorded chunks to the cell.
#
# Two things stay as they were, and both are named in the output:
#
#  * the MoE routing inputs, which the prover does not build at all.  They are
#    still `route_selection_cells` and `route_topk_cells` out of `cost_model.py`,
#    priced by analogy, and the ledger says `analogy` beside them.
#  * the witness-calibrated fields of a `params` block, which a caller who has a
#    model and a context but no witness cannot know.  They are ONE named
#    assumption block, `WITNESS_ASSUMPTIONS`, with a stated size in cells; an
#    upper bound is printed beside every point estimate and `result.assumptions`
#    carries the block so zk-results can record each field as a named constant.
CELLS_MODES = ("exact", "fitted")
CELLS = "exact"

# The fields a `params` block carries that a model and a context do not: the
# calibration a run took off its own witness, and two knobs of the prover's own
# command line.  The numbers are the MODAL value of the 64 layer and first
# segments of the two recorded 8B headline runs, and `check_assumptions` reads
# the same modes out of `exact_chunks.json` and exits if the two ever part, so
# nothing here is a number somebody typed once.
WITNESS_ASSUMPTIONS = dict(
    # the master exponent table, `5 * 2^log_table` cells and the largest of them
    log_table=23,
    # the normalisation table's binades and the fraction table's rows
    kmax=19,
    # the attention output's measured exponent ranges: three range tables and
    # the exponent window `ao_jv - ao_jo`
    ao_acc_hi_bits=5, ao_acc_lo_bits=16, ao_rem_bits=20, ao_q_bits=11,
    ao_jo_min=-12, ao_jo_max=-10, ao_jv_min=-7, ao_jv_max=-5,
    # the fold's own remainder range table
    fold_rem_bits=15,
    # The output head is a different plan and its own recorded segments agree on
    # a narrower remainder window than a layer's.
    head=dict(ao_rem_bits=15),
    # `--head-groups`, a run knob: the two headline runs proved 32 attention
    # heads as 16, 8, 8, so a shape's heads are a half and two quarters
    head_groups="half-quarter-quarter",
    # `Variant::parse` and the rounding: every composed segment of the two
    # headline runs is the integer-exponent softmax rounded to even
    variant="intexp", round="even",
)
# What each field sizes, for the assumption block a driver records.
ASSUMPTION_SIZES = dict(
    log_table="log_table", kmax="kmax", ao_acc_hi_bits="ao_ranges",
    ao_rem_bits="ao_ranges", ao_jo_min="ao_ranges", ao_jo_max="ao_ranges",
    ao_jv_min="ao_ranges", ao_jv_max="ao_ranges",
    ao_acc_lo_bits="ao_acc_lo_bits", ao_q_bits="ao_q_bits",
    fold_rem_bits="fold_rem_bits")


def witness_fields(kind="layer"):
    """The assumption block for one segment kind, as `exact_cells` reads it."""
    out = {f: WITNESS_ASSUMPTIONS[f] for f in ec.WITNESS_FIELDS}
    if kind == "head":
        out.update(WITNESS_ASSUMPTIONS["head"])
    return out


def check_assumptions():
    """`WITNESS_ASSUMPTIONS` against the modes of the recorded headline runs.

    Returns `{field: (value, run, segment, n, of)}` for the layer side and the
    head side, and exits where the block above is no longer what the runs say."""
    modes = dict(layer=ec.modal_witness(("layer", "first segment")),
                 head=ec.modal_witness("output head"))
    bad = []
    for kind, want in (("layer", witness_fields()), ("head", witness_fields("head"))):
        for f, v in want.items():
            got = modes[kind][f][0]
            if got != v:
                bad.append(f"    {kind} {f}: this module assumes {v}, the recorded "
                           f"runs' mode is {got} ({modes[kind][f][1]} "
                           f"{modes[kind][f][2]})")
    if bad:
        sys.exit("the witness assumption block is no longer the recorded runs' modal "
                 "value:\n" + "\n".join(bad)
                 + "\nEdit `WITNESS_ASSUMPTIONS`, or say why the assumption is not "
                   "the mode.")
    return modes


def assumption_rows(shape, seq=TOKENS, ctx=4096):
    """`WITNESS_ASSUMPTIONS` as rows a driver can record, one per field.

    Each row says what the field is, what this table assumes it is, which
    recorded run and segment that value is the mode of, what it sizes in cells on
    one 8B layer segment and what that is as a share of the segment -- and the
    bound the upper column is computed at.  A projected row names these in its
    `inputs`, so a reader of the constants file can see the assumption rather
    than find it inside a residual."""
    grp = shape["groups"][0]
    geom = dict(d=shape["d"], nh=shape["nh"], nkv=shape["nkv"], hd=shape["hd"],
                dff=grp["dff"] * grp["topk"], vocab=shape["vocab"], s=seq, c=ctx)
    p = ec.segment_params("layer", witness=witness_fields(), gkr=STATEMENT_GKR,
                          head_groups=ec.head_groups_for(
                              shape["nh"], WITNESS_ASSUMPTIONS["head_groups"]),
                          **geom)
    terms = ec.witness_terms(p)
    fresh = ec.exact_cells(p)["fresh"]
    modes = check_assumptions()
    bounds = ec.witness_bounds()
    rng = ec.witness_range()
    rows = []
    for f in ec.WITNESS_FIELDS:
        value, run, segment, n, of = modes["layer"][f]
        cells, share = terms.get(ASSUMPTION_SIZES.get(f, ""), (None, None))
        rows.append(dict(
            field=f, value=value, grade="derived", bound=bounds[f],
            recorded_values=rng[f], modal_of=[n, of],
            source_run=run, source_segment=segment,
            head_value=witness_fields("head")[f],
            sizes=ASSUMPTION_SIZES.get(f), cells=cells, share_of_segment=share,
            note=f"the mode of {f} over the {of} layer and first segments of the "
                 f"two recorded 8B headline runs, {n} of them; read off "
                 f"{run} {segment}"))
    # the two that are not a witness field at all: a flag of the prover's own
    # command line, and the softmax the recorded runs were proved with
    rows.append(dict(
        field="head_groups", value=WITNESS_ASSUMPTIONS["head_groups"],
        grade="asserted", bound=None, recorded_values=["16,8,8 at 32 heads"],
        source_run=ec.HEADLINE_RUNS[0], source_segment="every segment",
        sizes="the three key grids' heights", cells=None,
        share_of_segment=_head_group_share(shape, geom),
        note="the prover's --head-groups, a run knob: both headline runs proved "
             "32 attention heads as 16, 8, 8, so a shape's heads are assumed to "
             "be a half and two quarters.  `share_of_segment` is what one group "
             "of all the heads would cost instead"))
    kv = ec.kv_sharing(p)
    rows.append(dict(
        field="kv_per_head", value="per attention head", grade="asserted", bound=None,
        recorded_values=[f"{kv['nh']} heads over {kv['nkv']} kv"],
        source_run=ec.HEADLINE_RUNS[0], source_segment="every segment",
        sizes="the key cache, the value cache and the boundary trees that hand "
              "them on", cells=kv["saving"], share_of_segment=kv["saving_of_fresh"],
        note=f"`LayoutChan` and `LayoutKey` are built from `Shape::chans()` and "
             f"`Shape::keys()`, both carrying `insts` = attention heads, and "
             f"`boundary.rs`'s `kv_sides` hands on a cache of that width, so a key "
             f"and a value are committed once per attention head and not once per "
             f"key/value head.  On this segment that term is "
             f"{kv['as_built']['total']:,} cells, {kv['share_of_fresh']:.2%} of it; "
             f"a prover that shared them across the {kv['nh'] // kv['nkv']} heads of "
             f"a group would commit {kv['shared']['total']:,}, which is "
             f"{kv['saving']:,} cells and {kv['saving_of_fresh']:.2%} of the segment "
             f"less.  Not built: no AIR here shares a key tile's K and V columns "
             f"between two heads' score grids.  Counted per shape in section (2c)"))
    for f in ("variant", "round"):
        rows.append(dict(
            field=f, value=WITNESS_ASSUMPTIONS[f], grade="asserted", bound=None,
            recorded_values=sorted({q["params"][f] for q in ec.headline_chunks()}),
            source_run=ec.HEADLINE_RUNS[0], source_segment="every segment",
            sizes=("an auxiliary column per score and a range table"
                   if f == "variant" else "the sticky bit of every rounding"),
            cells=None, share_of_segment=None,
            note="every composed segment of the two headline runs carries this "
                 "value; a shape proved at another one is a different statement"))
    # and the two a mixture row adds, on the row every one of whose layers is a
    # mixture layer, so they are read off a segment with nothing else in it
    moe_ref = next((s for s in SHAPES if not refusal(s)
                    and all(g["routed"] for g in s["groups"])), None)
    moe_rows = [] if moe_ref is None else moe_assumption_rows(moe_ref, seq=seq, ctx=ctx)
    rows += moe_rows
    # and the four a latent-attention row adds, on the first row that has one
    lat_ref = next((s for s in SHAPES if not refusal(s) and s.get("latent")), None)
    if lat_ref is not None:
        rows += latent_assumption_rows(lat_ref, seq=seq, ctx=ctx)
    return dict(segment_fresh=fresh, context=ctx, seq=seq, rows=rows,
                moe_reference=None if moe_ref is None else moe_ref["name"],
                latent_reference=None if lat_ref is None else lat_ref["name"])


def moe_assumption_rows(shape, seq=TOKENS, ctx=4096, gkr=STATEMENT_GKR):
    """The two assumptions a counted mixture row makes, and what each is worth.

    Everything else about a mixture layer is the published config and the
    prover's layout.  These two are not: how the tokens are spread over the
    experts, and the selection argument on top of the router's logits, which no
    AIR in this project builds."""
    grp = next((g for g in shape["groups"] if g["routed"]), None)
    if grp is None:
        return []
    geom = group_geom(shape, grp, seq, ctx)
    # what the ASSUMPTION is worth, so the row is counted at the uniform routing
    # even where `--routing FILE` names this shape and the priced row is not
    moe = moe_of(shape, grp, seq)
    moe = dict(moe, distribution=None,
               touched=max(1, int(round(moe["touched_expected"]))))
    wit = witness_fields()

    def at(**kw):
        return segment_cells("layer", geom, wit, gkr, True, False,
                             moe=dict(moe, **kw))["fresh"]
    whole = at()
    # the other end of the routing assumption: the same token slots on as few
    # experts as they fit on, `topk` experts of `seq` tokens each
    concentrated = at(distribution=[(seq, grp["per_token"])])
    no_router = at(router=False)
    topk = grp["per_token"] * grp["routed"] * (cm.MAXARG_CELLS or 0.0) * seq
    row = f"{shape['name']}, one mixture layer segment at C={ctx:,}, S={seq}"
    return [dict(
        field="moe_routing", value="uniform", grade="asserted", bound=None,
        recorded_values=["no run of any of these models has been proved"],
        source_run="none", source_segment=row,
        sizes="how many tokens each routed matmul covers, and so its rows",
        cells=abs(whole - concentrated), share_of_segment=abs(whole - concentrated) / whole,
        note=f"every token's top-k is taken uniform over the {grp['routed']} "
             f"experts, so a segment of {seq} tokens touches "
             f"{moe['touched_expected']:.1f} of them and each routed matmul covers "
             f"{ec.routed_blocks(seq, grp['per_token'], moe['touched'])[0]} tokens. "
             f"`cells` is what the assumption is worth: the same token slots "
             f"concentrated on {grp['per_token']} experts of {seq} tokens each cost "
             f"{concentrated:,} against {whole:,}.  Routing statistics from an "
             f"actual run of the model remove the assumption: pass them as "
             f"--routing FILE and the row is counted at them"),
        dict(
        field="moe_router", value="gate matmul counted, top-k bounded",
        grade="bound", bound=None,
        recorded_values=["no AIR in this project builds a selection argument"],
        source_run="none", source_segment=row,
        sizes="the router's own matmul and the selection on its logits",
        cells=(whole - no_router) + topk,
        share_of_segment=((whole - no_router) + topk) / (whole + topk),
        note=f"the gate logits are a matmul of {grp['routed']} outputs over every "
             f"token and are COUNTED like any other ({whole - no_router:,} cells "
             f"a segment).  The top-k selection on them is not: no AIR here builds "
             f"one, so it is an upper bound, {grp['per_token']} argmax passes over "
             f"{grp['routed']} logits a token at the built gadget's measured "
             f"marginal ({cm.MAXARG_CELLS} cells a comparison, "
             f"`model_constants.argmax_marginal_cells_per_comparison`), "
             f"{topk:,.0f} cells a segment")]


def _wrap(text, width):
    """A note, broken at spaces, for a printed block that is read and not parsed."""
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _head_group_share(shape, geom):
    """What grouping all the heads into one key tile costs against the rule, as a
    fraction of the segment: the price of the head-group assumption."""
    def at(rule):
        p = ec.segment_params("layer", witness=witness_fields(), gkr=STATEMENT_GKR,
                              head_groups=ec.head_groups_for(shape["nh"], rule),
                              **geom)
        return ec.exact_cells(p)["fresh"]
    try:
        return at("one-group") / at(WITNESS_ASSUMPTIONS["head_groups"]) - 1.0
    except ec.Refused:
        return None


class Refused(Exception):
    """A shape whose segment is not a `params` block, with the reason."""


def refusal(shape):
    """Why the exact count cannot price this shape, or None.

    A shape is refused rather than smoothed: a row the layout cannot state is a
    row this table has no number for, and printing the fitted one under the same
    heading would hide exactly the extrapolation the count is here to remove.

    Multi-head latent attention was refused here until #183: the q and kv sides
    are low-rank pairs and the cache is one latent, so the layer's matmuls are
    five and not four and the key grids read a width no field of a GQA shape
    states.  Those widths are now read off each config and counted, both ways
    (`attention/latent.py`), and what is left is named in the assumption block."""
    if not shape.get("latent") and any(g.get("attn_accs_per_layer")
                                       for g in shape["groups"]):
        return ("multi-head latent attention with no published widths: the cache "
                "is one latent and its width is the five fields of a config, "
                "which this shape does not carry")
    try:
        exact_model_cells(shape, 4096, seq=TOKENS, gkr=STATEMENT_GKR)
    except (ec.Refused, Refused) as e:
        return str(e)
    return None


# Which of the two latent-attention layouts a counted row uses.  Both are
# projections -- no AIR in this project proves either -- and the table prints
# them side by side; this is the one the row records.
#
# It is the materialised one, for two reasons.  It is the layout the key tile
# family ALREADY builds: a K and a V per head, at two widths the config states,
# and nothing in it is a component no recorded segment shows.  And it is the
# cheaper of the two at every shape and both contexts, so it is what a prover
# choosing between them would build -- the absorbed form saves the cache and
# pays for it three times over in a score grid of `kv_lora + qk_rope` channels,
# an attention output of `kv_lora`, and two more matmuls a token.  Flipping it
# is decision 1 of #183.
LATENT_FORM = "materialised"


def latent_of(shape, form=None):
    """This shape's latent attention at one of the two forms, or None where its
    attention is the ordinary one.  The widths are the config's own five fields
    and the head count, and nothing else."""
    cfg = shape.get("latent")
    if not cfg:
        return None
    z = la.dims(cfg, shape["nh"], form or LATENT_FORM)
    z["rope"] = bool(shape.get("latent_rope", True))
    return z


_SEGMENTS = {}
# Per-expert token counts measured on a real run of a model, keyed by the row's
# slug: `{slug: [[tokens, matmuls], ...]}`.  `--routing FILE` fills it, and a row
# it names is counted at that distribution instead of at the uniform one, which
# is how the routing assumption is removed rather than restated.
ROUTING = {}


def moe_of(shape, grp, seq):
    """The mixture block of one group's `params`, or None where the group is dense.

    `touched` is how many of the routed experts a block of `seq` tokens reaches
    under uniform routing, which is how many routed matmuls the segment builds
    and whose weights its preprocessed trace holds."""
    if not grp["routed"]:
        return None
    touched = ec.expected_distinct_experts(grp["routed"], grp["per_token"], seq)
    dist = ROUTING.get(slug(shape["name"]))
    # A measured routing says how many experts the block touched, and that is the
    # count the segment's preprocessed trace holds the weights of (`moe_weights`).
    # The uniform answer stays beside it as `touched_expected`, which is what the
    # assumption row prices and what the table prints the measurement against.
    return dict(experts=grp["routed"], topk=grp["per_token"], moe_dff=grp["moe_dff"],
                n_in=grp["routed_n_in"],
                touched=(sum(int(n) for _, n in dist) if dist
                         else max(1, int(round(touched)))),
                touched_expected=touched, router=True, distribution=dist)


def group_geom(shape, grp, seq, ctx):
    """The `segment_params` arguments of one group's layer segment.

    The feed-forward side is split the way the config splits it: `dff` is what
    every token goes through -- a dense layer's own width, or the shared experts'
    -- and the routed experts are the `moe` block beside it."""
    return dict(d=shape["d"], nh=shape["nh"], nkv=shape["nkv"], hd=shape["hd"],
                dff=(grp["dense_dff"] or 0) + grp["shared"] * grp["moe_dff"],
                vocab=shape["vocab"], s=seq, c=0 if grp["linear"] else ctx)


def segment_cells(kind, geom, witness, gkr, diagonal_block, bound, moe=None,
                  kv_shared=False, latent=None):
    """One segment's committed cells, counted from its `params` block.  Memoised:
    a statement prices the same segment once per (kind, shape, prefix).

    `kv_shared=True` counts the same segment with its keys and values committed
    once per key/value head instead of once per attention head, which is the
    unbuilt layout section (2c) prices."""
    key = (kind, tuple(sorted(geom.items())), tuple(sorted(witness.items())),
           tuple(gkr), diagonal_block, bound, kv_shared,
           json.dumps(moe, sort_keys=True, default=str),
           json.dumps(latent, sort_keys=True, default=str))
    if key not in _SEGMENTS:
        p = ec.segment_params(kind, witness=witness, gkr=gkr, moe=moe,
                              latent=latent,
                              head_groups=ec.head_groups_for(
                                  geom["nh"], WITNESS_ASSUMPTIONS["head_groups"]),
                              **geom)
        p["diagonal_block"] = diagonal_block
        _SEGMENTS[key] = ec.cells_of(p, bound=bound, kv_shared=kv_shared)
    return _SEGMENTS[key]


def kv_term(shape, ctx, seq=TOKENS):
    """What one layer segment of this shape commits to hold its key and value
    cache, counted per ATTENTION head as the prover builds it and per KEY/VALUE
    head as the sharing allows (`exact_cells.kv_sharing`).

    Read off the first group whose layers have an attention grid at all: a run of
    linear-attention layers is priced at no grid, and it is not the segment this
    term is about.
    """
    grp = next((g for g in shape["groups"] if not g["linear"]), shape["groups"][0])
    geom = group_geom(shape, grp, seq, ctx)
    p = ec.segment_params("layer", witness=witness_fields(), gkr=STATEMENT_GKR,
                          moe=moe_of(shape, grp, seq), latent=latent_of(shape),
                          head_groups=ec.head_groups_for(
                              geom["nh"], WITNESS_ASSUMPTIONS["head_groups"]),
                          **geom)
    return ec.kv_sharing(p)


# what each layout does, in one clause, for the notes that name them
LATENT_WHAT = {
    "materialised": "expands the keys and the values per head before the "
                    "attention, as a prover with no latent layout builds it",
    "absorbed": "folds `kv_b` into the query and the output projection and caches "
                "the latent itself, as the published implementations decode",
}


def other_form(form):
    """The latent-attention layout a row does NOT count."""
    return la.FORMS[1] if form == la.FORMS[0] else la.FORMS[0]


def latent_segment(shape, ctx, seq=TOKENS, form=None, **kw):
    """One layer segment of a latent-attention shape, as a `params` block."""
    grp = next((g for g in shape["groups"] if not g["linear"]), shape["groups"][0])
    geom = group_geom(shape, grp, seq, ctx)
    return ec.segment_params("layer", witness=witness_fields(), gkr=STATEMENT_GKR,
                             moe=moe_of(shape, grp, seq),
                             latent=latent_of(shape, form),
                             head_groups=ec.head_groups_for(
                                 geom["nh"], WITNESS_ASSUMPTIONS["head_groups"]),
                             **dict(geom, **kw))


def latent_term(shape, ctx, seq=TOKENS):
    """What one layer segment of a latent-attention shape commits to hold its
    keys and its values, at both forms (`exact_cells.cache_cells`).

    The materialised form holds a K of `qk_nope + qk_rope` and a V of `v_head`
    for every head; the absorbed form holds one latent of `kv_lora + qk_rope` for
    every head at once, and pays for it in a wider score grid and a wider
    attention output, which are in `fresh` and not in this term."""
    out = {}
    for form in la.FORMS:
        p = latent_segment(shape, ctx, seq, form)
        z, cache = latent_of(shape, form), ec.cache_cells(p)
        fresh = ec.exact_cells(p)["fresh"]
        out[form] = dict(cache, fresh=fresh, hd_qk=z["hd_qk"], hd_v=z["hd_v"],
                         nh=z["nh"],
                         cache_per_key=z["cache_per_key"],
                         share_of_fresh=cache["total"] / fresh)
    return out


def latent_assumption_rows(shape, seq=TOKENS, ctx=4096):
    """The four assumptions a counted latent-attention row makes, and what each
    is worth on one layer segment of it.

    Everything else about these three shapes is the published config and the
    prover's own layout.  These four are not: which of the two layouts the row
    counts, how many copies of a shared cache a segment holds, the program the
    two low-rank norms build, and where the low-rank chain is requantized."""
    p = latent_segment(shape, ctx, seq)
    fresh = ec.exact_cells(p)["fresh"]
    z = latent_of(shape)
    row = f"{shape['name']}, one layer segment at C={ctx:,}, S={seq}"
    alt = other_form(z["form"])
    alt_fresh = ec.exact_cells(latent_segment(shape, ctx, seq, alt))["fresh"]
    groups = len(p["head_groups"])
    cache = ec.cache_cells(p)
    copies = (groups - 1) * (cache["keys"] + cache["values"]) // groups
    # what the two low-rank norms cost, and what the chain's own seams cost,
    # each as the segment without them
    no_norm = ec.exact_cells(dict(
        p, residual_stream=dict(p["residual_stream"], norms=[])))["fresh"]
    dense_seams = [s for s in p["seams"] if not isinstance(s, (tuple, list))
                   or s[0] == "qkv"]
    no_seams = ec.exact_cells(dict(p, seams=dense_seams))["fresh"]
    rows = [dict(
        field="mla_form", value=z["form"], grade="asserted", bound=None,
        recorded_values=["no AIR in this project proves either"],
        source_run="none", source_segment=row,
        sizes="where the low-rank pair `kv_b` is applied, and so what a key costs "
              "the cache and how wide the score grid is",
        cells=abs(alt_fresh - fresh), share_of_segment=abs(alt_fresh - fresh) / fresh,
        note=f"the row counts the {z['form']} layout, which {LATENT_WHAT[z['form']]}: "
             f"the cache is {z['cache_per_key']:,} cells a key and the score grid "
             f"runs over {z['hd_qk']} channels a head.  The {alt} layout "
             f"{LATENT_WHAT[alt]}, and the same segment counted that way is "
             f"{alt_fresh:,} cells against {fresh:,}.  The model's weights are the "
             f"same either way -- `W_UK` and `W_UV` together ARE `kv_b` -- so this "
             f"is a choice of layout and not of architecture, and section (2d) "
             f"prices the whole statement both ways"),
        dict(
        field="mla_cache_per_tile", value="once per key tile", grade="asserted",
        bound=None, recorded_values=[f"{groups} head groups"],
        source_run="none", source_segment=row,
        sizes="how many copies of a shared cache one segment commits",
        cells=copies if z["cache_shared"] else 0,
        share_of_segment=(copies if z["cache_shared"] else 0) / fresh,
        note=f"under the absorbed layout a key tile is a component and commits the "
             f"columns it reads, so the {groups} head groups each hold the latent of "
             f"every key in that tile: {groups} copies.  One channel component and "
             f"one grid shared between the {groups} grids would commit {copies:,} "
             f"cells fewer, and the split cache component also carries the "
             f"accumulator set of the component it splits from -- 12 and 24 columns "
             f"over its own rows, the one column group here no recorded segment "
             f"shows."
             + (f"  The row records the {z['form']} layout, which holds a K and a V "
                f"per head and shares nothing, so on the recorded row this "
                f"assumption costs no cells at all."
                if not z["cache_shared"] else "")),
        dict(
        field="mla_latent_norm", value="the output head's plain norm at the rank's "
                                       "width", grade="asserted", bound=None,
        recorded_values=["`q_a_layernorm` and `kv_a_layernorm` are config fields; "
                         "the program they build is not"],
        source_run=ec.HEADLINE_RUNS[0], source_segment=row,
        sizes="the wiring family of the two low-rank norms",
        cells=fresh - no_norm, share_of_segment=(fresh - no_norm) / fresh,
        note=f"each low-rank read is normed before the matmul that follows it.  The "
             f"program counted here is the one the OUTPUT HEAD's final norm already "
             f"builds -- a squares tree, its second-level reduction and a scale -- "
             f"at {z['q_lora_rank']} and {z['kv_lora_rank']} channels instead of the "
             f"model's width, so no new program is assumed, only that these two "
             f"norms are that one"),
        dict(
        field="mla_seams", value="one requantize seam per rank the chain passes",
        grade="asserted", bound=None,
        recorded_values=["a dense layer requantizes at qkv, gateup and down"],
        source_run=ec.HEADLINE_RUNS[0], source_segment=row,
        sizes="the requantize seams between the low-rank matmuls",
        cells=fresh - no_seams, share_of_segment=(fresh - no_seams) / fresh,
        note="a low-rank pair is two matmuls, and the second reads what the first "
             "wrote, so the rank is requantized between them exactly as the "
             "residual is requantized before the projections.  The seams counted "
             "here are "
             + ", ".join(f"`{s[0]}` at {s[1]:,}" for s in p["seams"]
                         if isinstance(s, (tuple, list)))
             + "; a layout that carried the rank in the dequantized form instead "
               "would not build them")]
    return rows


def moe_weights(shape, grp, moe):
    """The weights one layer of this group holds that a proof over it covers: the
    attention, the dense or shared feed-forward side, and the routed experts the
    segment TOUCHES -- not every expert the model has, because a segment that
    never reads an expert's weights does not commit them."""
    d, w = shape["d"], grp["moe_dff"]
    base = attn_params(shape, grp) + 3 * d * ((grp["dense_dff"] or 0) + grp["shared"] * w)
    if not moe:
        return base
    return base + 3 * (grp["routed_n_in"] or d) * w * moe["touched"]


def exact_model_cells(shape, ctx, seq=TOKENS, seq_batch=1, alpha=None, batch=None,
                      with_head=True, diagonal_block=True, gkr=(), bound=False,
                      kv_shared=False, latent_form=None):
    """`model_cells`, counted from the layout: one `params` block per segment.

    The unit is the same -- committed cells per decoded token, the layer segments
    of the whole stack plus the output head, each amortised over the `seq`
    positions its segment covers.

    A mixture layer is counted and not twinned.  Its shared experts are a matmul
    over every token; each routed expert is a matmul over the tokens routed to it
    alone, laid out as `MatLayout::choose` lays out that many tokens, so the row
    padding a small token count pays is in the number; the router's own matmul of
    `experts` logits is counted beside them.  What is NOT counted from a layout is
    the top-k selection, which no AIR here builds: it is a stated bound, below."""
    if alpha is not None:
        raise Refused("a single attention slope is not a layout; `--cells fitted` "
                      "is what prices it")
    if seq_batch != 1:
        raise Refused(f"B = {seq_batch} sequences a proof: a batched proof commits "
                      f"B key caches and B wiring families, and one `params` block "
                      f"describes one segment of one sequence")
    lat = latent_of(shape, latent_form)
    out = dict(total=0.0, preprocessed=0.0, residual_stream=0.0, parts={},
               gkr=tuple(gkr), saturated=False, cells="exact", bound=bound,
               kv_shared=kv_shared, segments={}, moe={},
               latent=None if lat is None else lat["form"])
    first = True
    for grp in shape["groups"]:
        geom = group_geom(shape, grp, seq, ctx)
        moe = moe_of(shape, grp, seq)
        counts = []
        layers = grp["layers"]
        if first and layers:
            counts.append(("first", 1))
            layers -= 1
            first = False
        counts.append(("layer", layers))
        for kind, n in counts:
            if not n:
                continue
            got = segment_cells(kind, geom, witness_fields(kind), gkr,
                                diagonal_block, bound, moe=moe, kv_shared=kv_shared,
                                latent=lat)
            out["segments"][f"{kind} x{n}"] = got["fresh"]
            out["total"] += n * got["fresh"] / seq
            out["residual_stream"] += n * sum(got["components"]["residual_stream"]) / seq
            for name, (b, inter) in got["components"].items():
                out["parts"][name] = out["parts"].get(name, 0.0) + n * (b + inter) / seq
            out["parts"]["boundary"] = (out["parts"].get("boundary", 0.0)
                                        + n * got["boundary"] / seq)
        # the weight trace the Freivalds check reads, committed offline.  It is
        # not a committed cell of any segment, so the exact count says nothing
        # about it and this is `cost_model.py`'s per-weight figure as before.
        w = grp["layers"] * moe_weights(shape, grp, moe)
        if with_head and grp is shape["groups"][0]:
            w += shape["vocab"] * shape["d"]
        out["preprocessed"] += (cm.PREPROCESSED_PER_WEIGHT or 0.0) * w / seq
        if moe:
            # the top-k selection over the router's `experts` logits, which no AIR
            # in this project builds.  Priced as `topk` argmax passes over the
            # logits at the built gadget's measured marginal, which is an upper
            # bound on a selection argument and is labelled one.
            out["parts"]["router_topk"] = out["parts"].get("router_topk", 0.0) + (
                grp["layers"] * grp["per_token"] * grp["routed"]
                * (cm.MAXARG_CELLS or 0.0))
            out["moe"][f"{grp['layers']}x{grp['routed']}/{grp['per_token']}"] = dict(
                touched=moe["touched"], touched_expected=moe["touched_expected"],
                experts=grp["routed"], topk=grp["per_token"], shared=grp["shared"],
                moe_dff=grp["moe_dff"], routed_n_in=grp["routed_n_in"] or shape["d"],
                distribution=moe.get("distribution") or "uniform",
                weights_per_layer=moe_weights(shape, grp, moe))
    if with_head:
        grp = shape["groups"][0]
        geom = dict(group_geom(shape, grp, seq, ctx), c=ctx)
        got = segment_cells("head", geom, witness_fields("head"), gkr,
                            diagonal_block, bound, moe=moe_of(shape, grp, seq),
                            kv_shared=kv_shared, latent=lat)
        out["segments"]["head"] = got["fresh"]
        out["total"] += got["fresh"] / seq
        for name, (b, inter) in got["components"].items():
            out["parts"][name] = out["parts"].get(name, 0.0) + (b + inter) / seq
        out["parts"]["boundary"] = out["parts"].get("boundary", 0.0) + got["boundary"] / seq
    out["total"] += out["parts"].get("router_topk", 0.0)
    # the two names the table's own columns read
    out["parts"]["attn"] = out["parts"].get("key_tiles", 0.0)
    out["parts"].setdefault("freivalds", 0.0)
    return out


def head_chunk(shape, seq, seq_batch=1, batch=None, alpha=None, gkr=STATEMENT_GKR,
               cells=None, bound=False, ctx=None, kv_shared=False,
               latent_form=None):
    """One output head segment, at any shape: the model-level terms -- the head's
    accumulators, its lookups, its Freivalds check, the token binding and the final
    norm -- over the `seq` positions the segment covers.

    Taken as the difference between the budget with the head charged and the budget
    without it, so the head is priced by the function that prices the layers and no
    second form of it is written down here.

    Under the fitted count it does not depend on the context, because that form of
    the head chunk performs no attention.  Under the exact count it does: the head
    segment the two headline runs proved carries a whole layer beside the head, key
    tiles and all, so `ctx` is the prefix that segment attends over and the caller
    says which block the head sits on."""
    kw = dict(seq=seq, seq_batch=seq_batch, batch=batch, alpha=alpha, gkr=gkr,
              cells=cells, bound=bound, kv_shared=kv_shared,
              latent_form=latent_form)
    c = seq if ctx is None else ctx
    return seq * (model_cells(shape, c, with_head=True, **kw)["total"]
                  - model_cells(shape, c, with_head=False, **kw)["total"])


def sound_cells(shape, st, seq_batch=1, batch=None, alpha=None, gkr=STATEMENT_GKR,
                cells=None, bound=False, kv_shared=False, latent_form=None):
    """What one proved position costs under the sound statement, modelled.

    Block `b` of the statement attends over the `b * S` positions before it, so the
    layer segments are priced at that prefix and not at the full context, and the
    head segments are added whole.  This is NOT `model_cells` divided by anything:
    `model_cells` prices one decoded token against a full C-key prefix, which is the
    decode-only statement, and the two differ by more than a third at C=16,384."""
    per_block, total = [], 0.0
    saturated = False
    parts = {}
    kw = dict(seq_batch=seq_batch, batch=batch, alpha=alpha, gkr=gkr, cells=cells,
              bound=bound, kv_shared=kv_shared, latent_form=latent_form)
    for b in range(st["blocks"]):
        c = model_cells(shape, b * st["s_chunk"], seq=st["s_chunk"], with_head=False,
                        **kw)
        per_block.append(c["total"])
        total += st["s_chunk"] * c["total"]
        saturated = saturated or bool(c.get("saturated"))
        for k, v in c["parts"].items():
            parts[k] = parts.get(k, 0.0) + st["s_chunk"] * v
    layers = total
    if (cells or CELLS) == "exact":
        # the head segments sit over the LAST blocks of the statement -- the last
        # prompt block and the generated one (the decision of 2026-09-08) -- and
        # each one attends over the prefix its own block does
        heads = [head_chunk(shape, st["s_chunk"], ctx=(st["blocks"] - k) * st["s_chunk"],
                            **kw) for k in range(1, st["heads"] + 1)]
        total += sum(heads)
        head = sum(heads) / max(1, len(heads))
    else:
        head = head_chunk(shape, st["s_chunk"], **kw)
        total += st["heads"] * head
    return dict(cells=total, per_position=total / st["positions"],
                layers_cells=layers, layers_per_position=layers / st["positions"],
                head_chunk=head, heads=st["heads"], per_block=per_block,
                parts={k: v / st["positions"] for k, v in parts.items()},
                saturated=saturated, statement=st)


def model_cells(shape, ctx, seq=TOKENS, seq_batch=1, alpha=None, batch=None,
                with_head=True, diagonal_block=True, gkr=(), cells=None, bound=False,
                kv_shared=False, latent_form=None):
    """The whole per-token budget for a shape, summed group by group.

    `cells="exact"` -- the default, and the owner's ruling -- counts every segment
    from its own `params` block (`exact_cells`).  `cells="fitted"` is the older
    path: the cost model's `cells_per_token` plus B10 step 2's residual stream,
    with the model-level terms (the head's accumulators, its lookups, the token
    binding, the final norm) charged to exactly one group.  Both are kept so that
    the two can be printed side by side; nothing falls back from one to the other."""
    if (cells or CELLS) == "exact":
        return exact_model_cells(shape, ctx, seq=seq, seq_batch=seq_batch,
                                 alpha=alpha, batch=batch, with_head=with_head,
                                 diagonal_block=diagonal_block, gkr=gkr, bound=bound,
                                 kv_shared=kv_shared, latent_form=latent_form)
    if kv_shared:
        raise Refused("what the key/value sharing would save is counted from the "
                      "layout and the fitted path has no layout: it charges the "
                      "cache through a per-unit constant read off one segment")
    if latent_form:
        raise Refused("which of the two latent-attention layouts a row counts is a "
                      "property of the counted path; the fitted one reads one "
                      "accumulator count off the config and builds no layout")
    batch = cm.BATCH if batch is None else batch
    out = dict(total=0.0, preprocessed=0.0, residual_stream=0.0, parts={},
               gkr=tuple(gkr))
    for i, grp in enumerate(shape["groups"]):
        cfg = cfg_of(shape, grp)
        head = with_head and (i == 0)
        c = cm.cells_per_token(
            cfg, 0 if grp["linear"] else ctx, batch, seq, alpha=alpha,
            experts=grp["experts"], topk=grp["topk"], seq_batch=seq_batch, head=head,
            attn_accs_per_layer=grp.get("attn_accs_per_layer"),
            attn_params_per_layer=grp.get("attn_params_per_layer"),
            diagonal_block=diagonal_block, gkr=gkr)
        out["saturated"] = out.get("saturated", False) or bool(c["key_tile_saturated"])
        gl = cm.residual_stream_cells(cfg, "tree", topk=grp["topk"], final_norm=head,
                                      batch=batch, gkr=gkr)
        out["total"] += c["total"] + gl["net"]
        out["preprocessed"] += c["preprocessed"]
        out["residual_stream"] += gl["net"]
        for k, v in c.items():
            if isinstance(v, (int, float)) and k not in ("dequant_per_acc", "head_per_acc",
                                                         "tokens_per_proof"):
                out["parts"][k] = out["parts"].get(k, 0.0) + v
    return out


def exact_block_check(shape, consts, seq=TOKENS):
    """The exact count of one token block's layer segments against the recorded one.

    This is the calibration the owner's ruling asks for, and it compares the same
    thing on both sides: the 32 layer segments of the block whose prefix is the
    whole context, over the 512 positions that block proves, with no output head
    segment on either side (`cells_per_position.headline.c{ctx}.fresh_layers_block`).

    Three rows per context, and the distance between them is the whole of what
    the assumption costs: the block counted at `WITNESS_ASSUMPTIONS`, the same
    block counted at each segment's OWN recorded witness fields -- which has to
    be exact, because then nothing is assumed -- and the block at the bound."""
    out = {}
    for ctx in CONTEXTS:
        e = consts.get(f"block_c{ctx}")
        rec = [p for p in ec.headline_chunks(("layer", "first segment"))
               if p["params"]["c"] == ctx]
        if not rec:
            continue
        kw = dict(seq=seq, with_head=False, gkr=STATEMENT_GKR, cells="exact")
        assumed = model_cells(shape, ctx, **kw)["total"]
        at_bound = model_cells(shape, ctx, bound=True, **kw)["total"]
        own = sum(p["fresh"] for p in rec) / seq
        row = dict(ctx=ctx, segments=len(rec), key=OPTIONAL[f"block_c{ctx}"],
                   assumed=assumed, at_bound=at_bound, recorded_fields=own,
                   recorded=None if e is None else e["value"],
                   run=None if e is None else e.get("run"))
        for name, v in (("assumed", assumed), ("at_bound", at_bound),
                        ("recorded_fields", own)):
            row[f"{name}_pct"] = (None if row["recorded"] in (None, 0)
                                  else 100.0 * (v / row["recorded"] - 1.0))
        out[ctx] = row
    return out


def check_8b(card, shape, consts, batch=None, shp=None):
    """The check, and it now has **no free parameter at all**.

    Before 2026-09-01 the attention term was one slope, `alpha`, solved from the
    two measured contexts -- one fitted parameter, two points, one residual dof.
    Stream DD's key tile model and stream E1's symbolic re-derivation of it replace
    that with a closed form whose every coefficient is an identity of the layout,
    so the model now predicts both measured levels with nothing solved: two points,
    zero fitted parameters, two residual degrees of freedom.

    **What the two levels are.**  Where the card is the constants file's headline
    card the comparison is per PROVED POSITION over the sound statement: the model
    prices the same blocks, the same prefixes and the same output head segments the
    run proved, at the same five reductions.  Where it is one of the archive's
    sequences the comparison is per DECODED TOKEN against a full C-key prefix, which
    is a different statement, and `statement` says which one each residual is on.

    Also re-checks the base coefficients themselves against the measured triple,
    and the whole family against every recorded layer segment's own key tiles."""
    meas = measured_8b(card)
    sound = card.get("statement") == "sound"
    nh, hd = shape["nh"], shape["hd"]
    got_fresh = _coeffs(nh, hd)
    got_base = _base_coeffs(nh, hd)
    want_base = consts["key_tile_base_coeffs"]["value"]
    st = {c: statement_of(shape, c, shp) for c in meas} if sound else {}

    def predict(ctx, alpha=None):
        if sound:
            return sound_cells(shape, st[ctx], batch=batch, alpha=alpha)["per_position"]
        return model_cells(shape, ctx, alpha=alpha, batch=batch)["total"]

    def legacy(alias):
        a = consts[alias]["value"]
        return None if a is None else {c: predict(c, a) / meas[c] - 1 for c in meas}

    pred = {c: predict(c) for c in meas}
    resid = {c: pred[c] / meas[c] - 1 for c in meas}
    out = dict(
        measured=meas, predicted=pred,
        key_tile=resid,
        residual_pct={c: 100.0 * r for c, r in resid.items()},
        statement={c: st[c] for c in meas} if sound else None,
        unit="cells per proved position" if sound else "cells per decoded token",
        card=card["card"], source=card.get("source"),
        keys={c: card["per_ctx"][c].get("keys", []) for c in meas},
        legacy_slope=legacy("alpha_e1"), legacy_b5=legacy("alpha_b5"),
        coeffs_fresh=got_fresh, coeffs_base=got_base,
        frozen_base=want_base,
        coeffs_agree=(want_base is None or got_base == tuple(want_base)),
        cells=CELLS, assumptions=None if CELLS != "exact" else check_assumptions(),
        block=exact_block_check(shape, consts, seq=shp["s_chunk"] if shp else TOKENS),
        family=_key_tile_family_check(nh, hd))
    return out


def _coeffs(nh, hd, base_only=False, s0=512, c0=4096):
    """The (S*C, C, S) coefficients this module's key tile model implies at a shape.

    Recovered from the key tile function itself by finite differences, rather than
    restated, so the drift guard compares the code that is actually used against
    the freeze.  Differenced at (s0, c0) and its doublings, well clear of
    `Grid::new`'s caps -- at S or C below them the layout saturates, `q` and `pk`
    stop being the targets, and the form is no longer a plain trilinear.

    On the FAMILY, which is what `cells_per_token` charges: the prefix tiled into
    key blocks of S keys plus the diagonal block, at the prover's defaults.  One
    untiled prefix key tile costs `nh*hd*(C - S)` base cells less and is not what any
    recorded layer chunk built (issue #141)."""
    def f(s, c):
        b, i, _ = cm.key_tile_family(nh, hd, s, c)
        return b if base_only else b + i
    a = (f(2 * s0, 2 * c0) - f(2 * s0, c0) - f(s0, 2 * c0) + f(s0, c0)) / (s0 * c0)
    b = (f(s0, 2 * c0) - f(s0, c0)) / c0 - a * s0
    d = (f(2 * s0, c0) - f(s0, c0)) / s0 - a * c0
    return tuple(round(x, 6) for x in (a, b, d))


def _base_coeffs(nh, hd, s0=512, c0=4096):
    """The base triple in the parametrisation `attention_grid.base_cell_coeffs`
    names, so that the drift guard compares like with like.

    The measured entry states the family's base cells as `a S (C + S) + b (C + S)
    + d S`: the diagonal block is the same key tile at (S, S) with the per-token
    component dropped, so it costs what a prefix of S more keys costs, and the
    two S-shaped terms group into `C + S` rather than standing apart.  `_coeffs`
    above differences the same function as a plain trilinear in (S, C) instead,
    which folds `a S^2 + b S` into its S coefficient and so reports a different
    third number at the point it differences.  Both describe the same family;
    only this one is the triple the constant carries."""
    def f(s, c):
        b, i, _ = cm.key_tile_family(nh, hd, s, c)
        return b
    a = (f(2 * s0, 2 * c0) - f(2 * s0, c0) - f(s0, 2 * c0) + f(s0, c0)) / (s0 * c0)
    b = (f(s0, 2 * c0) - f(s0, c0)) / c0 - a * s0
    d = (f(s0, c0) - a * s0 * (c0 + s0) - b * (c0 + s0)) / s0
    return tuple(round(x, 6) for x in (a, b, d))


def _key_tile_family_check(nh, hd):
    """`family_cells.key_tiles` is base-only; predict it on every recorded layer
    segment and on B9's two prefill points, and report exact-or-not.  Nothing is
    fitted, so a miss here is a real disagreement, not a residual.

    The recorded segments are `key_tile_segments.json`, the file
    `key_tile --selftest` checks and the one this repository carries so the check
    runs with no artifacts directory: every 8B layer segment of the two recorded
    sequences, at both contexts and with the five reductions on and off.  An
    artifacts directory adds the seq5 layer chunks and B9's prefill points on top,
    and a run output that does not carry the family count under this name -- an
    older vocabulary -- is skipped rather than read as a zero."""
    rows = []
    for pt in key_tile.segments():
        if pt["nh"] != nh or pt["hd"] != hd:
            continue
        b, _ = key_tile.evaluate(pt)
        name = f"{pt['run']}/{pt['output']}"
        if pt.get("segment"):
            name += f":{pt['segment']}"
        rows.append((name, pt["s"], pt["c"], pt["key_tiles"], b))
    seq5 = art("b10", "out", "seq5")
    chunks = sorted(seq5.glob("seq_c*_L*.json")) if seq5 and seq5.is_dir() else []
    for f in chunks:
        try:
            d = json.loads(f.read_text())["result"]["point"]
        except (ValueError, KeyError):
            continue
        if "key_tiles" not in d.get("family_cells", {}):
            continue      # an older vocabulary; not this quantity under this name
        pr = d["params"]
        if pr["nh"] != nh or pr["hd"] != hd:
            continue
        # the statement's own configuration: whether the prefix is tiled into key
        # blocks, and whether the chunk binds its K/Q/V to the boundary roots
        b, _, _ = cm.key_tile_family(nh, hd, pr["s"], pr["c"],
                                 grid_target=pr["grid_target"], chan_target=pr["chan_target"],
                                 rw=pr["rw"], diagonal_block=bool(pr.get("diagonal_block")),
                                 kv_boundary=bool(pr.get("kv_boundary")),
                                 bind=pr.get("rope") is not None)
        rows.append((f.name, pr["s"], pr["c"], d["family_cells"]["key_tiles"], b))
    for s in (512, 1024):
        f = art("b9", "out", f"prefill_s{s}.json")
        if not (f and f.exists() and f.stat().st_size):
            continue
        d = json.loads(f.read_text())["result"]["point"]
        if "key_tiles" not in d.get("family_cells", {}):
            continue
        pr = d.get("params") or {}
        b, _, _ = cm.key_tile_family(nh, hd, s, s,
                                 diagonal_block=bool(pr.get("diagonal_block")),
                                 kv_boundary=bool(pr.get("kv_boundary")),
                                 bind=pr.get("rope") is not None)
        rows.append((f.name, s, s, d["family_cells"]["key_tiles"], b))
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
        p = art("b9", "out", f"prefill_s{s}.json")
        if p and p.exists() and p.stat().st_size:
            d = json.loads(p.read_text())["result"]["point"]
            if "key_tiles" not in d.get("family_cells", {}):
                continue      # an older vocabulary; not this quantity under this name
            pts.append(dict(s=s, fresh=d["cells"]["fresh"],
                            key_tiles=d["family_cells"]["key_tiles"]))
    if len(pts) < 2:
        return None
    a, b = pts[-2], pts[-1]
    p = (b["key_tiles"] - a["key_tiles"]) / (b["s"] ** 2 - a["s"] ** 2)
    q = ((b["fresh"] - b["key_tiles"]) - (a["fresh"] - a["key_tiles"])) / (b["s"] - a["s"])
    f = (a["fresh"] - a["key_tiles"]) - q * a["s"]
    return dict(fixed=f, per_token=q, per_pair=p, points=[x["s"] for x in pts],
                missing=[s for s in (512, 1024, 2048)
                         if s not in [x["s"] for x in pts]],
                source=rel(art("b9", "out", "prefill_s*.json")))


def e11_factor(fit, n_positions, n_output, s_chunk, layers=32):
    """The full prompt+output statement (E11, decided 2026-08-31), as a multiple of
    today's decode-only statement, from B9's measured fit at the 8B shape.

    Today: `n_output` positions proven, each against an unchecked C-key prefix.
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


# ------------------------------------------- what a chunk holds on the card --
# Stream GG (`briefs/gg-livemem.md`) read the live bytes inside the fork's device
# memory pool at twelve named marks per chunk, on the H100 at 8B, C=4096, S=512.
# It retired the single slope this module used to price the memory wall with
# (~22.6 GiB per G fresh cell, read off ONE chunk) and replaced it with four
# terms.  Nothing below is typed in, per this module's first rule: GG's two runs
# differ only in the shared exponent table's height (`params.log_table` 23 and
# 25), so a two-point solve separates every term that scales with the tallest
# column from every term that does not, and the coefficients fall out of the two
# committed artifacts.
GG_DIR = art("b12", "livemem", "out", "h100")
GG_RUNS = ("r1_L16_defaults.json", "r3_L0_defaults.json")


def gg_point(path):
    """One of GG's runs, reduced to the facts the memory model needs.

    `pool_used_high` is the live peak the pool ever held, which GG located at
    `prove_values` (the FRI transient) in every run; `driver_used - pool_reserved`
    at that same mark is what the driver holds OUTSIDE the pool."""
    p = json.loads(pathlib.Path(path).read_text())["result"]["point"]
    marks = p["mem_marks"]
    peak = max(marks, key=lambda m: m["pool_used_high"])
    witness_gen = next(m for m in marks if m["mark"] == "after_witness_gen")
    fam = p["family_cells"]
    return dict(
        h=p["params"]["log_table"], cells=p["cells"],
        # the weights this chunk's layer covers, and the preprocessed trace
        # they actually commit (`pre_cells`), group by group
        freivalds_weights=sum(w["weights"] for w in fam["freivalds_weights"]),
        freivalds_pre=sum(w["pre_cells"] for w in fam["freivalds_weights"]),
        # trees 0 and 1, as far as they are resident on the device before commit
        devcopy_cells=p["upload"]["devcopy_bytes"] // 4,
        witness_gen_resident=witness_gen["pool_used"] - p["upload"]["devcopy_bytes"],
        peak=peak["pool_used_high"], peak_mark=peak["mark"],
        driver=peak["driver_used"] - peak["pool_reserved"],
        slack=peak["pool_reserved_high"] - peak["pool_used_high"],
        card_total=marks[0]["driver_total"], source=rel(path))


def memory_coeffs():
    """GG's four terms, solved from the two runs rather than restated.

    The two runs are the same layer shape at two table heights, so for any
    quantity `x` measured at both, `(x25 - x16) / (2**25 - 2**23)` is how many
    columns of the tallest height it carries and the remainder is its fixed part.
    That is the whole solve; the only judgement in it is which fixed parts are
    held constant when the shape changes, and section (7b) prints that."""
    if not (GG_DIR and all((GG_DIR / f).exists() for f in GG_RUNS)):
        return None
    a, b = (gg_point(GG_DIR / f) for f in GG_RUNS)
    if a["h"] == b["h"]:
        return None
    dh = 2 ** b["h"] - 2 ** a["h"]

    def cols(key, sub=None):
        xa = a[key][sub] if sub else a[key]
        xb = b[key][sub] if sub else b[key]
        return (xb - xa) / dh, xa

    fresh_cols, fresh_a = cols("cells", "fresh")
    pre_cols, pre_a = cols("cells", "preprocessed")
    base_cols, base_a = cols("cells", "base")
    inter_cols, inter_a = cols("cells", "interaction")
    dev_cols, dev_a = cols("devcopy_cells")
    witness_gen_bytes_cols, witness_gen_a = cols("witness_gen_resident")
    peak_cols, peak_a = cols("peak")
    # the three cell terms, each at 12 B / 4 B, leave the height term
    cell_b, copy_b = 12.0, 4.0
    # the height term is `height_b * 2**(h+1)`, so its slope in 2**h is twice it
    height_b = (peak_cols - cell_b * (fresh_cols + pre_cols)
                - copy_b * dev_cols - witness_gen_bytes_cols) / 2.0
    resid = peak_a - (cell_b * (fresh_a + pre_a) + copy_b * dev_a + witness_gen_a
                      + height_b * 2 ** (a["h"] + 1))
    bounds = a["cells"]["fresh"] - a["cells"]["base"] - a["cells"]["interaction"]
    return dict(
        runs=[a, b], cell_bytes=cell_b, copy_bytes=copy_b, height_bytes=height_b,
        # the boundary trees' own Merkle, plus whatever the decomposition does not
        # name; GG's brackets put the second part at a few kB
        residual_bytes=resid, bounds_cells=bounds,
        fresh_h_cols=fresh_cols, pre_h_cols=pre_cols, base_h_cols=base_cols,
        inter_h_cols=inter_cols, dev_h_cols=dev_cols, witness_gen_h_bytes=witness_gen_bytes_cols,
        witness_gen_fixed=witness_gen_a - witness_gen_bytes_cols * 2 ** a["h"],
        # the Freivalds check's OWN preprocessed trace, per covered weight, as the
        # composed chunk commits it.  B3's `preprocessed_per_weight` is a standalone
        # width-sweep slope and does not carry the composed chunk's padding.
        pre_per_weight=a["freivalds_pre"] / a["freivalds_weights"],
        b3_pre_per_weight=cm.PREPROCESSED_PER_WEIGHT,
        # everything in tree 0 that is not the Freivalds check: the shared lookup
        # tables and the per-component preprocessed columns
        pre_shared_fixed=pre_a - a["freivalds_pre"] - pre_cols * 2 ** a["h"],
        pre_host_built=pre_a - (dev_a - a["cells"]["base"]),
        # base's share of the height-independent trace, used to split a modelled
        # fresh count into the base half the device witness generations clone and the
        # interaction half they do not
        base_share=((base_a - base_cols * 2 ** a["h"])
                    / (base_a + inter_a - (base_cols + inter_cols) * 2 ** a["h"])),
        driver_bytes=max(a["driver"], b["driver"]),
        slack_bytes=max(a["slack"], b["slack"]),
        card_total=a["card_total"],
        weights_per_freivalds_row=(cm.WEIGHT_FREIVALDS_CELLS_PER_WEIGHT
                                   and 12.0 / cm.WEIGHT_FREIVALDS_CELLS_PER_WEIGHT))


def chunk_memory(mc, fresh, weights, h, measured=None):
    """One layer chunk's live peak on the card, in bytes, at table height `h`.

    `fresh` is the chunk's fresh cells as the cost model gives them (which is the
    h = 23 convention, since that is the height every seq5 artifact was taken at);
    `weights` is what the chunk's Freivalds check covers -- for an MoE layer,
    EVERY expert, which is why the preprocessed term is the one that moves the
    MoE rows.

    `measured=<point>` replaces the modelled cell counts with that run's own, so
    the same function serves the calibration check and the frontier rows."""
    h_cells = 2 ** h
    if measured is not None:
        F, P = measured["cells"]["fresh"], measured["cells"]["preprocessed"]
        D = measured["devcopy_cells"]
    else:
        F = fresh + mc["fresh_h_cols"] * (h_cells - 2 ** 23)
        P = mc["pre_per_weight"] * weights + mc["pre_shared_fixed"] \
            + mc["pre_h_cols"] * h_cells
        base_fixed = mc["base_share"] * (fresh - mc["boundary_cells"]
                                         - mc["fresh_h_cols"] * 2 ** 23)
        D = base_fixed + mc["base_h_cols"] * h_cells + (P - mc["pre_host_built"])
    witness_gen = mc["witness_gen_fixed"] + mc["witness_gen_h_bytes"] * h_cells
    cells_b = mc["cell_bytes"] * (F + P)
    copies_b = mc["copy_bytes"] * D
    height_b = mc["height_bytes"] * 2 * h_cells
    pool = cells_b + copies_b + witness_gen + height_b + mc["residual_bytes"]
    return dict(fresh=F, pre=P, dev=D, cells_bytes=cells_b, copies_bytes=copies_b,
                witness_gen_bytes=witness_gen, height_bytes=height_b, pool=pool,
                card=pool + mc["driver_bytes"] + mc["slack_bytes"])


def head_cross_check(mc, ref):
    """The one chunk the retired slope was actually read from, priced by the model.

    seq5's `lm_head` chunk is a different chunk on a different card, and none of
    it entered GG's solve, so it is a real held-out point.  It is informational
    and not gated: its peak is an `nvidia-smi` sample every 2 s (so a floor on a
    transient, which GG showed can be 2.5 GiB tall), and its witness generation residents were
    never marked, so the layer chunk's are used for them."""
    p = ROOT / "b10" / "out" / "seq5" / "seq_c4096_head.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["result"]["point"]
    h = d["params"]["log_table"]
    fam = d["family_cells"]
    got = chunk_memory(mc, None, None, h, measured=dict(
        cells=d["cells"], devcopy_cells=d["upload"]["devcopy_bytes"] // 4))
    card = got["card"]
    return dict(source=rel(p), h=h, cells=d["cells"],
                freivalds_weights=sum(w["weights"] for w in fam["freivalds_weights"]),
                freivalds_pre=sum(w["pre_cells"] for w in fam["freivalds_weights"]),
                measured_mib=ref["peak_mib"], model_mib=card / 2 ** 20,
                delta=card / 2 ** 20 / ref["peak_mib"] - 1.0)


def freivalds_matrices(shape, grp):
    """The weights of each matrix one layer's Freivalds check covers, biggest first.

    Only the biggest matters here: a Freivalds group's row count is the next power of
    two above `weights / p` for the widest matrix in it (B3's design point `p`),
    and that row count is the only thing in a layer chunk that could out-grow the
    shared exponent table and take over `h`."""
    d, nh, nkv, hd = shape["d"], shape["nh"], shape["nkv"], shape["hd"]
    if grp.get("attn_params_per_layer"):
        # MLA: the low-rank pair is not `(nh + 2 nkv) hd`, and this module does not
        # carry its per-matrix split.  Charging the whole attention block as one
        # matrix is an over-estimate, which is the safe direction for a bound.
        mats = [grp["attn_params_per_layer"]]
    else:
        mats = [d * nh * hd, d * nkv * hd, nh * hd * d]
    mats += [d * grp["dff"], grp["dff"] * d]         # per expert, not per layer
    return sorted(mats, reverse=True)


def freivalds_height(mc, shape, grp):
    """`ceil(log2(widest matrix / p))` -- the tallest column the check builds."""
    p = mc["weights_per_freivalds_row"]
    if not p:
        return None
    return math.ceil(math.log2(max(freivalds_matrices(shape, grp)) / p))


# ------------------------------------- what LDE-only residency changes (HH) --
# Stream HH (`briefs/hh-ldeonly.md`) kept every committed column on the trace
# domain at 4 B per cell and rebuilt its blown-up evaluations in each of the four
# phases that read them, behind `ZK_LDE_ONLY`.  The proof does not move.  Its
# gate-1 runs are GG's two shapes again -- the same L16 (h=23) and L0 (h=25)
# chunks on the same H100 -- with the switch off (A1, A3) and on (A2, A4), and
# A1/A3 reproduce GG's R1/R3 peaks to the byte, so the same two-point solve that
# gives (7b) its coefficients gives this one its residency model.  Nothing below
# is typed in either.
HH_DIR = art("b12", "ldeonly", "out", "h100")
HH_ON = ("a2_L16_on.json", "a4_L0_on.json")
HH_OFF = ("a1_L16_off.json", "a3_L0_off.json")
HH_LOOP = ("p1_pipe_off_c4096.json", "p2_pipe_on_c4096.json")
# Two facts about the box rather than about the proof, needed only by variant
# (iii): the pinned staging pool measured 30.16 GB/s (`briefs/l-tracegen.md`),
# and `/sys/fs/cgroup/memory.max` on the H100 container reads 150.0 GiB
# (`briefs/dd-extract.md`), which is what a host-resident coefficient set has.
PCIE_PINNED_BPS = 30.16e9
H100_HOST_CGROUP = 161061273600


def hh_point(path):
    """One of HH's runs, reduced to the facts the LDE-only model needs.

    Beside the peak this keeps every level mark, because the levels are the
    held-out check: they are what the coefficients must reproduce at the height
    they were not read at, and none of them enters the two-point solve."""
    p = json.loads(pathlib.Path(path).read_text())["result"]["point"]
    marks = {m["mark"]: m for m in p["mem_marks"]}
    peak = max(p["mem_marks"], key=lambda m: m["pool_used_high"])
    c = p["cells"]
    return dict(
        h=p["params"]["log_table"], cells=c,
        committed=c["fresh"] + c["preprocessed"],
        boundary_cells=c["fresh"] - c["base"] - c["interaction"],
        devcopy_cells=p["upload"]["devcopy_bytes"] // 4,
        witness_gen=marks["after_witness_gen"]["pool_used"], marks=marks,
        peak=peak["pool_used_high"], peak_mark=peak["mark"],
        prove=p["prove_secs"],
        phase={k: v["secs"] for k, v in p["phases"]["phases"].items()},
        source=rel(path))


def _lde_terms(r):
    """The residency coefficients, read off ONE run's mark deltas.

    Nothing here is typed in and nothing here comes from the other run.  The two
    trace trees of different sizes give `delta = c x cells + M x 2^(h+1)` twice,
    which solves the per-cell cost `c` and the Merkle cost `M` together; the
    third tree is then a check on that solve (`merkle_check`).  The twiddles, the
    composition accumulator and the composition tree are pure height terms, the
    boundary trees' Merkle is a constant because their height is fixed at 2^14,
    and the transient above `composition_commit` is the quotient column and FRI.
    `_lde_levels` replays the lot at the OTHER height, where none of it was
    read."""
    n = 2.0 ** (r["h"] + 1)
    m, c = r["marks"], r["cells"]
    d_pre = m["commit_pre"]["pool_used"] - m["after_twiddles"]["pool_used"]
    d_base = m["commit_base"]["pool_used"] - m["commit_pre"]["pool_used"]
    d_inter = m["commit_inter"]["pool_used"] - m["commit_base"]["pool_used"]
    cell_b = (d_base - d_pre) / (c["base"] - c["preprocessed"])
    merkle = (d_pre - cell_b * c["preprocessed"]) / n
    return dict(
        cell_bytes=cell_b, merkle=merkle,
        merkle_check=d_inter - (cell_b * c["interaction"] + merkle * n),
        twiddles=(m["after_twiddles"]["pool_used"] - r["witness_gen"]) / n,
        composition=(m["composition"]["pool_used"]
                     - m["commit_boundary"]["pool_used"]) / n,
        comp_tree=(m["composition_commit"]["pool_used"]
                   - m["composition"]["pool_used"]) / n,
        boundary_merkle=(m["commit_boundary"]["pool_used"] - m["commit_inter"]["pool_used"]
                       - cell_b * r["boundary_cells"]),
        fri=(r["peak"] - m["composition_commit"]["pool_used"]) / n)


def _lde_levels(r, co):
    """Replay `co`'s coefficients against run `r`'s own cells and witness-generation level."""
    n = 2.0 ** (r["h"] + 1)
    c, out = r["cells"], {}
    lv = r["witness_gen"] + co["twiddles"] * n
    out["after_twiddles"] = lv
    for k, cc in (("commit_pre", c["preprocessed"]), ("commit_base", c["base"]),
                  ("commit_inter", c["interaction"])):
        lv += co["cell_bytes"] * cc + co["merkle"] * n
        out[k] = lv
    lv += co["cell_bytes"] * r["boundary_cells"] + co["boundary_merkle"]
    out["commit_boundary"] = lv
    lv += co["composition"] * n
    out["composition"] = lv
    lv += co["comp_tree"] * n
    out["composition_commit"] = lv
    return out


def ldeonly_coeffs(mc):
    """HH's peak, solved from its two switched-on runs the way (7b) solves GG's.

        peak = 4 B x (fresh + preprocessed) + the copies (7b) already prices
             + X(h),   X(h) = a + b x 2^(h+1)

    Two measured points, two unknowns, so the solve itself checks nothing.  What
    checks it is `held_out`: the coefficients of the h=23 run's mark deltas,
    replayed against the h=25 run's own cells, must reproduce all six of ITS
    level marks and its peak, and no level mark is in the solve."""
    if not (HH_DIR and all((HH_DIR / f).exists() for f in HH_ON + HH_OFF + HH_LOOP)):
        return None
    a, b = (hh_point(HH_DIR / f) for f in HH_ON)
    off = [hh_point(HH_DIR / f) for f in HH_OFF]
    if a["h"] == b["h"]:
        return None

    def X(r):
        # the copies term is (7b)'s `4 B x D + other witness_gen residents`, which is
        # exactly what the run's own `after_witness_gen` level reads
        return r["peak"] - 4.0 * r["committed"] - r["witness_gen"]

    hb = (X(b) - X(a)) / (2 ** (b["h"] + 1) - 2 ** (a["h"] + 1))
    ha = X(a) - hb * 2 ** (a["h"] + 1)

    # what a and b are, in the marks.  Read the height terms off each run and
    # cross-check them against the 472 B of the retained-evaluation model.
    ta, tb = _lde_terms(a), _lde_terms(b)
    resident_n = 3 * ta["merkle"] + ta["twiddles"] + ta["composition"] + ta["comp_tree"]
    # The transient above `composition_commit` is the phase the switch changes.
    # At h=25 it is the FRI's own 160 x 2^(h+1) to 6 kB, and at h=23 it is 2.69x
    # that.  What reconciles the two heights is ONE rebuilt log-size group at 8 B
    # per trace cell (a rebuilt column is the blown-up one), with the transient
    # the LARGER of it and the FRI's: at h=23 the group wins, at h=25 the FRI
    # does, and the group is never charged on top of the FRI.  HH's record reads
    # the h=23 excess instead as a smaller group standing beside the quotient
    # accumulator; that reading does not reconcile with h=25, where a group of
    # its share would have pushed the transient 4.96 GiB above the 160 it
    # measured.  The share below is one measurement at one shape either way.
    grp_bytes = a["peak"] - a["marks"]["composition_commit"]["pool_used"]
    grp_share = grp_bytes / 8.0 / a["committed"]
    held = []
    for r, src in ((b, a), (a, b)):
        pred = _lde_levels(r, _lde_terms(src))
        for k, v in pred.items():
            held.append(dict(run=r["source"], h=r["h"], mark=k, from_h=src["h"],
                             measured=r["marks"][k]["pool_used"], model=v,
                             delta=r["marks"][k]["pool_used"] - v))
    # the same X, written so that the group scales with the shape instead of
    # sitting inside a fitted constant.  It reproduces both measured X to a few
    # kB with one share, which the a + b x 2^(h+1) form cannot do off the 8B
    # shape, and it is what the wide rows are also priced under.
    def X_scaled(h, cells):
        n = 2 ** (h + 1)
        return (resident_n * n + ta["boundary_merkle"]
                + max(tb["fri"] * n, 8.0 * grp_share * cells))

    struct = {r["h"]: dict(model=X_scaled(r["h"], r["committed"]),
                           measured=X(r),
                           delta=X(r) - X_scaled(r["h"], r["committed"]))
              for r in (a, b)}
    return dict(
        runs=[a, b], off_runs=off, a_bytes=ha, b_bytes=hb,
        X={r["h"]: X(r) for r in (a, b)},
        terms=ta, terms_hi=tb, resident_n=resident_n, fri_n=tb["fri"],
        height_n=resident_n + tb["fri"], retained_n=mc["height_bytes"],
        group_bytes=grp_bytes, group_cells=grp_bytes / 8.0, group_share=grp_share,
        struct=struct, held_out=held,
        held_worst=max(abs(x["delta"]) for x in held),
        driver_bytes=mc["driver_bytes"], slack_bytes=mc["slack_bytes"])


def ldeonly_memory(mc, lo, part, h):
    """One layer chunk under LDE-only, in bytes, from `chunk_memory`'s own split.

    Three variants, in the order they would have to be built:
      (i)   as built -- the switch as stream HH measured it;
      (ii)  the witness generations' trace-domain copies dropped once the tree is committed,
            designed and not built: they are redundant with the coefficients;
      (iii) the coefficients host-resident and streamed per log-size group,
            designed and not built: the card then holds only the height terms
            and whatever group is in flight."""
    n = 2 ** (h + 1)
    cells = part["fresh"] + part["pre"]
    cells4 = lo["terms"]["cell_bytes"] * cells
    copies = part["copies_bytes"] + part["witness_gen_bytes"]
    fixed = lo["driver_bytes"] + lo["slack_bytes"]
    X = lo["a_bytes"] + lo["b_bytes"] * n
    group = lo["group_share"] * cells
    resident = lo["resident_n"] * n + lo["terms"]["boundary_merkle"]
    transient = max(lo["fri_n"] * n, 8.0 * group)
    # (iii) prices the group in flight explicitly: 4 B of coefficients crossing
    # the bus beside the 8 B of evaluations rebuilt from them.  At the shape and
    # height the coefficients were read at that is X(h) + 4 B x group, to 192 B.
    return dict(
        cells_bytes=cells4, copies_bytes=copies, height_bytes=X, group_cells=group,
        as_built=cells4 + copies + X + fixed,
        no_copies=cells4 + X + fixed,
        host_coeffs=resident + transient + 4.0 * group + fixed,
        host_ram=cells4, pcie_secs=3.0 * cells4 / PCIE_PINNED_BPS,
        # the same two, with the rebuilt group scaled to the shape rather than
        # held at the 8B chunk's own size inside the fitted `a`
        as_built_scaled=cells4 + copies + resident + transient + fixed,
        no_copies_scaled=cells4 + resident + transient + fixed)


def ldeonly_wall(lo):
    """The three rebuild passes, and what a batched rebuild would cost instead.

    The commit builds a whole log-size group in one `ntt_n2b_columns`; the
    rebuild calls `get_evaluation_on_domain` per column.  `lde_extend` is the
    commit's own batched extension and it does not move between the arms, so it
    is the batched rate for the same column set, measured in the same run."""
    a, b = lo["runs"]                       # A2 (h=23), A4 (h=25)
    o1, o3 = lo["off_runs"]                 # A1 (h=23), A3 (h=25)
    passes = ("constraint_eval", "fri_quotients", "other")   # composition, quotients, decommit
    per = []
    for on, off, tag in ((a, o1, "L16"), (b, o3, "L0")):
        d = {k: on["phase"][k] - off["phase"][k] for k in on["phase"]}
        rebuild = sum(d[k] for k in passes)
        batched = off["phase"]["lde_extend"]
        per.append(dict(chunk=tag, h=on["h"], off=off["prove"], on=on["prove"],
                        delta=on["prove"] - off["prove"], rebuild=rebuild,
                        by_pass={k: d[k] for k in passes}, other={
                            k: d[k] for k in ("merkle", "fri_commit", "oods_eval")},
                        batched=batched, ratio=rebuild / 3.0 / batched,
                        projected=off["prove"] + 3.0 * batched,
                        cells=on["committed"]))
    loops = {}
    for f, tag in zip(HH_LOOP, ("off", "on")):
        d = json.loads((HH_DIR / f).read_text())["result"]
        ch, s = d["chunks"], d["summary"]
        loops[tag] = dict(
            loop=s["loop_secs"], n=s["n"],
            gen=sum(c["build_secs"] + c["generate_secs"] for c in ch),
            prove=sum(c["prove_secs"] for c in ch),
            lde=sum(c["phases"]["phases"]["lde_extend"]["secs"] for c in ch),
            cells=sum(c["cells"]["fresh"] + c["cells"]["preprocessed"] for c in ch),
            source=rel(HH_DIR / f))
    p1 = loops["off"]
    proj = p1["prove"] + 3.0 * p1["lde"]
    over = p1["loop"] - max(p1["gen"], p1["prove"])
    p1["projected_prove"] = proj
    p1["slack"] = p1["gen"] - p1["prove"]
    p1["overhead"] = over
    p1["projected_loop"] = max(p1["gen"], proj) + over
    p1["projected_delta"] = p1["projected_loop"] / p1["loop"] - 1.0
    p1["gen_per_cell"] = p1["gen"] / p1["cells"]
    return dict(per_chunk=per, loops=loops,
                prove_per_cell=per[0]["on"] / per[0]["cells"],
                projected_per_cell=per[0]["projected"] / per[0]["cells"])


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


# A model's name as one segment of a constants key: lower case, and everything
# that is not a letter or a digit becomes a dash.  `Llama-3.1-8B` carries dots and
# a dotted key would split it into two levels of a namespace that has no such
# levels, so the dots go the same way the spaces do.
def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# The target every `cards_for_20` row is stated against, in PROVED POSITIONS per
# second (the paper's unit), and the key that would replace it.  The number is in
# the key's name, so a constants file that names another target renames the row.
TARGET_KEY = "design.frontier_target_positions_per_s"


def price(shape, cards, alpha, contexts=(4096, 16384), seq=TOKENS, seq_batch=1,
          usd=None, target_tok_s=20.0, mib_per_chunk=None, batch=None,
          shp=None, cells=None):
    """The reusable per-shape row.  Everything the table prints comes from here.

    Two quantities per context, and they are not interchangeable.  `cells_per_token`
    is the decode-only statement -- one decoded token against a full C-key prefix --
    and every section that compares this table against an older one reads it.
    `cells_per_position` is the sound statement, per proved position, and it is what
    the cards-at-target and the dollars columns are computed from, because that is
    what the measured rates were measured on and what the paper quotes."""
    usd = usd or {}
    row = dict(name=shape["name"], slug=slug(shape["name"]), tag=shape["tag"],
               params_total=params_of(shape, freivalds=True, embedding=True),
               params_active=params_of(shape, embedding=True),
               params_freivalds=params_of(shape, freivalds=True),
               check_total=shape.get("check_total"), check_active=shape.get("check_active"),
               source=shape["source"], seq=seq, seq_batch=seq_batch, contexts={})
    n_chunks = sum(grp["layers"] for grp in shape["groups"]) + 1
    exact = (cells or CELLS) == "exact"
    for ctx in contexts:
        kw = dict(seq=seq, seq_batch=seq_batch, alpha=alpha, batch=batch, cells=cells)
        c = model_cells(shape, ctx, **kw)
        st = statement_of(shape, ctx, shp)
        sc = sound_cells(shape, st, seq_batch=seq_batch, batch=batch, alpha=alpha,
                         cells=cells)
        # the assumption's other end: the same statement with every
        # witness-calibrated field at the bound of `exact_cells.witness_bounds`
        hi = (sound_cells(shape, st, seq_batch=seq_batch, batch=batch, alpha=alpha,
                          cells=cells, bound=True) if exact else None)
        # and the same statement with the keys and values committed once per
        # key/value head: the layout nobody has built, priced (section 2c).  A
        # latent attention holds no K and no V per key/value head at all, so
        # there is nothing there to share and the row carries the two latent
        # forms instead (section 2d).
        lat = latent_of(shape) if exact else None
        kv = (sound_cells(shape, st, seq_batch=seq_batch, batch=batch, alpha=alpha,
                          cells=cells, kv_shared=True)
              if exact and lat is None else None)
        alt_form = None if lat is None else other_form(lat["form"])
        alt = (sound_cells(shape, st, seq_batch=seq_batch, batch=batch, alpha=alpha,
                           cells=cells, latent_form=alt_form)
               if alt_form else None)
        entry = dict(cells_per_token=c["total"], preprocessed_per_token=c["preprocessed"],
                     residual_stream_per_token=c["residual_stream"], parts=c["parts"], chunks=n_chunks,
                     cells_per_position=sc["per_position"],
                     cells_per_position_bound=None if hi is None else hi["per_position"],
                     cells_per_position_kv_shared=(None if kv is None
                                                   else kv["per_position"]),
                     kv_shared_saving_pct=(None if kv is None else 100.0 *
                                           (kv["per_position"] / sc["per_position"] - 1.0)),
                     kv=None if not exact or lat else kv_term(shape, ctx, seq=seq),
                     latent=None if lat is None else lat["form"],
                     latent_alt=alt_form,
                     latent_term=(None if lat is None
                                  else latent_term(shape, ctx, seq=seq)),
                     cells_per_position_latent_alt=(None if alt is None
                                                    else alt["per_position"]),
                     latent_alt_pct=(None if alt is None else 100.0 *
                                     (alt["per_position"] / sc["per_position"] - 1.0)),
                     cells_per_token_latent_alt=(
                         None if alt_form is None else
                         model_cells(shape, ctx, latent_form=alt_form,
                                     **kw)["total"]),
                     cells_per_token_bound=(None if not exact else
                                            model_cells(shape, ctx, bound=True,
                                                        **kw)["total"]),
                     cells_per_token_kv_shared=(None if not exact or lat else
                                                model_cells(shape, ctx, kv_shared=True,
                                                            **kw)["total"]),
                     cells=CELLS if cells is None else cells,
                     layers_per_position=sc["layers_per_position"],
                     head_chunk_cells=sc["head_chunk"], statement=st,
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
                pos_s = m["cells_s"] / sc["per_position"]
                lo_s = None if hi is None else m["cells_s"] / hi["per_position"]
                kv_s = None if kv is None else m["cells_s"] / kv["per_position"]
                alt_s = None if alt is None else m["cells_s"] / alt["per_position"]
                out[b] = dict(cells_s=m["cells_s"], tok_s=tok_s, pos_s=pos_s,
                              scope=m.get("scope"), rate_key=m.get("key"),
                              rate_source=card.get("source"),
                              rate_statement=card.get("statement"),
                              cards_for_target=math.ceil(target_tok_s / pos_s),
                              cards_for_target_tok_s=math.ceil(target_tok_s / tok_s),
                              usd_per_mposition=usd_per_mtok(pos_s, usd.get(name)),
                              usd_per_mtok=usd_per_mtok(tok_s, usd.get(name)),
                              pos_s_bound=lo_s,
                              cards_for_target_bound=(None if lo_s is None else
                                                      math.ceil(target_tok_s / lo_s)),
                              usd_per_mposition_bound=usd_per_mtok(lo_s, usd.get(name)),
                              pos_s_kv_shared=kv_s,
                              cards_for_target_kv_shared=(None if kv_s is None else
                                                          math.ceil(target_tok_s / kv_s)),
                              usd_per_mposition_kv_shared=usd_per_mtok(kv_s,
                                                                       usd.get(name)),
                              pos_s_latent_alt=alt_s,
                              cards_for_target_latent_alt=(
                                  None if alt_s is None else
                                  math.ceil(target_tok_s / alt_s)),
                              usd_per_mposition_latent_alt=usd_per_mtok(
                                  alt_s, usd.get(name)))
            entry["cards"][name] = out
        row["contexts"][ctx] = entry
    return row


def dense_twin(shape):
    """The same shape with every mixture layer collapsed to one dense feed-forward
    block of the width a token actually goes through: a dense model with
    *identical active parameters*.  This is the comparison B9's `x2.35 pure
    overhead` is stated against, and the only fair one -- against a dense model of
    the same *total* size a mixture is of course cheaper."""
    twin = dict(shape)
    groups = []
    for grp in shape["groups"]:
        if not grp["routed"]:
            groups.append(dict(grp))
            continue
        w = (grp["dense_dff"] or 0) + (grp["shared"] + grp["per_token"]) * grp["moe_dff"]
        t = dict(grp, dense_dff=w, dff=w, routed=0, per_token=0, shared=0,
                 moe_dff=0, routed_n_in=None, experts=1, topk=1,
                 ffn_all=3 * shape["d"] * w, ffn_token=3 * shape["d"] * w)
        groups.append(t)
    twin["groups"] = groups
    return twin


def moe_ratio(shape, ctx, alpha, seq=TOKENS, seq_batch=1, ref_batch=1, batch=None,
              gkr=STATEMENT_GKR, cells=None):
    """MoE at batch B against its dense twin at batch `ref_batch`.

    With `ref_batch=1` the reference is fixed, which is what section 6.4's counter
    claims: at `B ~ E/k` the batched MoE proof costs what the unbatched dense proof
    costs, because each expert is again amortising its Freivalds check over S
    tokens."""
    dense = dense_twin(shape)
    a = model_cells(shape, ctx, seq=seq, seq_batch=seq_batch, alpha=alpha, batch=batch,
                    gkr=gkr, cells=cells)
    b = model_cells(dense, ctx, seq=seq, seq_batch=ref_batch, alpha=alpha, batch=batch,
                    gkr=gkr, cells=cells)
    return dict(moe=a["total"], dense_active=b["total"], ratio=a["total"] / b["total"],
                moe_pre=a["preprocessed"], dense_pre=b["preprocessed"],
                ratio_with_pre=(a["total"] + a["preprocessed"])
                / (b["total"] + b["preprocessed"]))




# ------------------------------------------------- what a driver can record --
# The rows above, as constants entries: the key, the value, the unit, the card,
# the context and the timing scope, the grade -- always `projected`, because
# nothing here is measured -- and the `inputs` naming every constants key the
# model read to produce it.  A driver in zk-results turns this list into entries
# and records nothing else; which of them it MAY record is `recordable`, and a row
# that read anything outside the constants file is not one of them.


def read_from_constants():
    """Every cell cost the model charges that a key in the constants file backs."""
    return sorted({e["key"] for e in cm.PROVENANCE.values()
                   if e["source"] == "constants" and e["key"]})


# The key a recorded assumption would carry in zk-results.  Proposed, not
# recorded: no constants file names one of these yet, and the pull request asks
# the owner to settle the names before a driver writes them.
ASSUMPTION_KEY = "assumption.cost_model.{field}"


# The two a counted mixture row adds: how the tokens are spread over the experts,
# and the selection argument on the router's logits, which is a stated bound.
MOE_ASSUMPTIONS = ("moe_routing", "moe_router")
# The four a counted latent-attention row adds: which of the two layouts it
# counts, how many copies of a shared cache a segment holds, the program the two
# low-rank norms build, and where the low-rank chain is requantized.
LATENT_ASSUMPTIONS = ("mla_form", "mla_cache_per_tile", "mla_latent_norm",
                      "mla_seams")


def assumption_keys(moe=True, latent=True):
    fields = list(ec.WITNESS_FIELDS) + ["head_groups", "kv_per_head", "variant", "round"]
    return [ASSUMPTION_KEY.format(field=f)
            for f in fields + (list(MOE_ASSUMPTIONS) if moe else [])
            + (list(LATENT_ASSUMPTIONS) if latent else [])]


def missing_assumption_keys():
    """The assumption block's keys that this constants file does not hold.

    A projected row names them in its `inputs`, so until zk-results records them
    the run is not recordable and this says exactly which are wanted."""
    return [k for k in assumption_keys() if constants.find(k) is None]


def cells_inputs():
    """Every input the cell side of a priced row read, as constants keys.

    Under the fitted count that is the per-unit cell costs.  Under the exact
    count the cell side reads no cell cost at all -- it counts columns -- and
    what it reads instead is the assumption block, plus the one per-weight figure
    the preprocessed column still charges."""
    if CELLS != "exact":
        return read_from_constants()
    pre = [e["key"] for e in cm.PROVENANCE.values()
           if e["key"] == "model_constants.weight_freivalds_preprocessed_cells_per_weight"]
    return sorted(set(pre + assumption_keys()))


# What the COUNTED path does not charge, although the fitted one does: the two
# routing inputs priced by analogy.  A counted mixture layer builds the router's
# matmul like any other matmul and states the selection on top of it as a bound,
# so neither analogy is read.
NOT_CHARGED_EXACT = ("route_selection_cells", "route_topk_cells")


def read_elsewhere():
    """Every input the model charges that the constants file did not give it, by
    where it came from: `inputs` is the micro-benchmark directory, `composed` this
    repository's own re-fit of the composed rungs, `analogy` a number nothing
    measures anywhere, `absent` one nothing prices at all."""
    out = {}
    for name, e in sorted(cm.PROVENANCE.items()):
        if e["source"] == "constants" or not e.get("charged", True):
            continue
        if CELLS == "exact" and name in NOT_CHARGED_EXACT:
            continue
        out.setdefault(e["source"], []).append(
            dict(input=name, key=e["key"], file=e["file"], value=e["value"]))
    return out


def entry(key, value, unit, card, context, scope, inputs, note, from_constants=True):
    """One row, ready to be an entry of zk-results' `constants.json`.

    `entry` is the entry verbatim -- the seven fields every entry carries plus the
    `inputs` a projected one must name -- and `from_constants` says whether every
    number behind it came from that same file.  A driver records `entry` and reads
    the two fields beside it to decide whether it may."""
    return dict(key=key, recordable=None, from_constants=from_constants, entry=dict(
        value=value, unit=unit, card=card, context=context, timing_scope=scope,
        grade="projected", inputs=sorted(set(inputs)), note=note))


def projected_rows(priced, moe_rows, cal, cards, usd, target, seq, seq_batch,
                   cells_keys=None):
    """#88's rows, ready to be recorded, each naming what it was computed from."""
    cells_keys = read_from_constants() if cells_keys is None else list(cells_keys)
    rows = []
    for r in priced:
        for ctx, e in sorted(r["contexts"].items()):
            st = e["statement"]
            for cname, per_b in sorted(e["cards"].items()):
                card = cards[cname]
                usd_key = f"cost.{cname.lower()}_reference_usd_per_hour"
                for bname, m in per_b.items():
                    scope = m.get("scope") or bname.replace(" ", "_")
                    base = f"frontier.{r['slug']}.{cname.lower()}.c{ctx}.{scope}"
                    inputs = cells_keys + list(st["keys"])
                    if m.get("rate_key"):
                        inputs = inputs + [m["rate_key"]]
                    where = ("the rate is a key of this file"
                             if m.get("rate_source") == "constants" else
                             f"the rate is NOT from this file: it is the archive's "
                             f"{m.get('rate_statement')} sequence on this card")
                    common = (f"{r['name']} at {ctx:,} context, {st['positions']:,} proved "
                              f"positions in {st['blocks']} blocks of {st['s_chunk']} and "
                              f"{st['heads']} output head segments, the five reductions on. "
                              f"{e['cells_per_position']:,.0f} modelled cells a proved "
                              f"position at {m['cells_s'] / 1e6:,.1f} Mcell/s, "
                              f"{m['pos_s']:.4f} proved positions a second on one card. "
                              f"{where}.")
                    ours = m.get("rate_source") == "constants"
                    rows.append(entry(
                        f"{base}.cards_for_20", m["cards_for_target"], "cards",
                        card["card"], ctx, bname,
                        inputs, f"Cards to reach {target:g} proved positions a second. "
                                + common, from_constants=ours))
                    rows.append(entry(
                        f"{base}.usd_per_mposition", m["usd_per_mposition"],
                        "USD/1000000 positions", card["card"], ctx, bname,
                        inputs + [usd_key],
                        f"Dollars a million proved positions at "
                        f"{usd.get(cname)} USD/GPU-h. " + common, from_constants=ours))
                    if m.get("usd_per_mposition_kv_shared") is not None:
                        rows.append(entry(
                            f"{base}.usd_per_mposition_kv_shared",
                            m["usd_per_mposition_kv_shared"],
                            "USD/1000000 positions", card["card"], ctx, bname,
                            inputs + [usd_key],
                            f"The same row for a prover that committed each key and "
                            f"each value once per KEY/VALUE head rather than once per "
                            f"attention head, which this shape shares "
                            f"{e['kv']['nh'] // e['kv']['nkv']} ways. Not built: no "
                            f"AIR in this project shares a key tile's K and V columns "
                            f"between two heads' score grids, and the row beside this "
                            f"one is what the prover commits today. "
                            f"{e['cells_per_position_kv_shared']:,.0f} modelled cells "
                            f"a proved position against {e['cells_per_position']:,.0f}. "
                            + common, from_constants=ours))
                    if m.get("usd_per_mposition_latent_alt") is not None:
                        lt = e["latent_term"][e["latent_alt"]]
                        rows.append(entry(
                            f"{base}.usd_per_mposition_mla_{e['latent_alt']}",
                            m["usd_per_mposition_latent_alt"],
                            "USD/1000000 positions", card["card"], ctx, bname,
                            inputs + [usd_key],
                            f"The same row at the OTHER latent-attention layout. The "
                            f"row beside this one counts the {e['latent']} form; this "
                            f"one counts the {e['latent_alt']} form, whose cache is "
                            f"{lt['cache_per_key']:,} cells a key and whose score "
                            f"grid runs over {lt['hd_qk']} channels a head. Neither "
                            f"is built: no AIR in this project proves a latent "
                            f"attention, and the model's weights are the same either "
                            f"way. {e['cells_per_position_latent_alt']:,.0f} modelled "
                            f"cells a proved position against "
                            f"{e['cells_per_position']:,.0f}. "
                            + common, from_constants=ours))
    # one derived row per shape and context: the saving is a ratio of two cell
    # counts, so no card and no timing scope enters it and every dollar column
    # above moves by exactly this per cent.
    for r in priced:
        for ctx, e in sorted(r["contexts"].items()):
            if e.get("kv_shared_saving_pct") is None:
                continue
            kv = e["kv"]
            rows.append(entry(
                f"frontier.{r['slug']}.c{ctx}.kv_shared_saving_pct",
                e["kv_shared_saving_pct"], "per cent", "none", ctx, "n/a",
                cells_keys + list(e["statement"]["keys"]),
                f"{r['name']} at {ctx:,} context: what committing each key and each "
                f"value once per key/value head instead of once per attention head "
                f"takes off a proved position. {kv['nh']} attention heads over "
                f"{kv['nkv']} key/value heads; on one layer segment at the full "
                f"prefix the two caches and the boundary trees that hand them on are "
                f"{kv['share_of_fresh']:.2%} of the segment and the sharing leaves "
                f"{kv['saving_of_fresh']:.2%} of it uncommitted. Not built, and the "
                f"figure is a floor: the interaction columns those columns carry are "
                f"in the batched reduction and are counted by nothing here."))
    # and one derived row per latent-attention shape and context: what the other
    # layout costs, as a ratio of two cell counts and so of no card and no scope
    for r in priced:
        for ctx, e in sorted(r["contexts"].items()):
            if e.get("latent_alt_pct") is None:
                continue
            lt = e["latent_term"]
            rows.append(entry(
                f"frontier.{r['slug']}.c{ctx}.mla_{e['latent_alt']}_pct",
                e["latent_alt_pct"], "per cent", "none", ctx, "n/a",
                cells_keys + list(e["statement"]["keys"]),
                f"{r['name']} at {ctx:,} context: what counting its attention at the "
                f"{e['latent_alt']} layout costs a proved position against the "
                f"{e['latent']} one, which is what the row records. On one layer "
                f"segment at the full prefix the cache and the boundary trees that "
                f"hand it on are {lt[e['latent']]['share_of_fresh']:.2%} of the "
                f"segment at the {e['latent']} layout and "
                f"{lt[e['latent_alt']]['share_of_fresh']:.2%} at the "
                f"{e['latent_alt']} one. Neither layout is built, and the model's "
                f"weights are the same either way: it is a choice of where the "
                f"low-rank pair is applied."))
    for name, v in sorted(moe_rows.items()):
        s = slug(name)
        r = v["sweep"][1]
        for suffix, val, what in (
                ("fresh_over_dense", r["ratio"],
                 "committed cells against the dense model with the same active "
                 "parameters, at one sequence a proof"),
                ("preprocessed_over_fresh", r["ratio_with_pre"],
                 "the same comparison counting the preprocessed weight trace, which "
                 "the measured rates do not count")):
            how = ("" if CELLS != "exact" else
                   " Counted from the layout, where the segment prices the experts a "
                   "token evaluates and the committed side of the Freivalds check "
                   "does not scale with the expert count; what a prover covering "
                   "every expert in full would commit for it is a routed layout "
                   "nobody has built and is counted by nothing here.")
            rows.append(entry(
                f"moe.{s}.{suffix}", val, "fraction", "none", 4096, "n/a", cells_keys,
                f"{name}, {v['ek']:.1f} experts a token's top-k: {what}. Modelled at "
                f"S={seq} and B={seq_batch}, the five reductions on; no measured rate "
                f"enters it and no card." + how))
    for ctx, pct in sorted(cal.get("residual_pct", {}).items()):
        rows.append(entry(
            f"frontier.calibration.c{ctx}.residual_pct", pct, "per cent",
            cal["card"], ctx, "n/a",
            cells_keys + cal["keys"].get(ctx, []),
            f"The model's prediction for the sound statement against what the run "
            f"measured, per proved position: {cal['predicted'][ctx]:,.0f} modelled "
            f"against {cal['measured'][ctx]:,.0f}. No fitted parameter: two measured "
            f"levels, two residual degrees of freedom."))
    if cal.get("residual_pct"):
        worst = max(cal["residual_pct"], key=lambda c: abs(cal["residual_pct"][c]))
        rows.append(entry(
            "frontier.calibration.residual_pct", cal["residual_pct"][worst], "per cent",
            cal["card"], worst, "n/a",
            cells_keys + [k for ks in cal["keys"].values() for k in ks],
            f"The calibration residual the whole table stands on: the worst of the two "
            f"contexts, which is C={worst:,}. The per-context rows beside it carry both."))
    return rows


def _e11_section(priced, h100, usd, mult, net, fixed):
    """Section (6)'s tables, printed only where all three constants are recorded.
    Nothing a priced row's cards-at-target or dollars-per-token depends on is
    computed here: what this adds to a row is the row's own `e11` block."""
    print("\n  the cells multiple at C=4096, against seq5's measured decode sequence:")
    print(f"    prompt alone         x{mult[0]:.2f}")
    print(f"    prompt + output      x{mult[1]:.2f}       (`{OPTIONAL['e11_multiple']}`)")
    print("    at C=16384 the same model gives x17.55 / x18.55")
    print("  The superseded ~x4.5-5.5 priced each prompt chunk as its own S=C=1024 SQUARE.")
    print("  That is not a proof of the sequence: the fourth chunk must attend over all")
    print("  4,096 keys, not its own 1,024.  Under the key tile model the square reading is")
    print("  x4.56 / x5.56, so the correction is +14% on the prompt multiple -- and a head")
    print("  chunk per 512 prompt positions is counted in NEITHER (at 4K it would add")
    print("  ~8 x the measured lm_head chunk).")
    print("\n  the chunking cost, which is now three terms and not one:")
    print(f"    {'the shared-table floor paid n times':<44} {fixed[1]:>+6.1f}% (4K) "
          f"{fixed[0]:>+6.1f}% (16K)")
    print(f"    {'the prefix K/V re-committed by every later chunk':<44} {1.9:>+6.1f}%      "
          f"{5.2:>+6.1f}%")
    print(f"    {'the unconstrained half of the block a one-shot pays':<44} "
          f"{-11.2:>+6.1f}%      {-30.0:>+6.1f}%")
    print(f"    {'NET':<44} {net[0]:>+6.1f}%      {net[1]:>+6.1f}%")
    print("  At C=16384 chunking is CHEAPER outright -- the causal saving beats the floor")
    print("  and the re-commitment together.  The old section reported only the first row")
    print("  (+13.7% / +11.6%) and called it the overhead.")
    fac = mult[1]
    print(f"\n  the table at x{fac:.2f} (prompt+output, C=4096, S=1024 chunking):")
    print(f"  {'model':<18} {'cells/token':>12} {'tok/s':>11} {'$/Mtok':>10}"
          f"   ({h100['card']} chunk loop, ${usd.get(h100['card'])}/GPU-h)")
    for r in priced:
        e = r["contexts"][4096]
        c = e["cells_per_token"] * fac
        bnd = h100["per_ctx"][4096]["boundaries"]
        h = bnd[next(b for b in ("loop", "loop_secs", "whole wall") if b in bnd)]["cells_s"] / c
        d = usd_per_mtok(h, usd.get(h100["card"]))
        r["e11"] = dict(cells_per_token=c, tok_s_h100=h, usd_per_mtok=d)
        print(f"  {r['name']:<18} {g(c):>12} {h:>11.4f} "
              + (f"{d:>10,.0f}" if d else f"{'n/a':>10}"))
    print("  The multiple is transported from the 8B shape, not recomputed per row: E7's")
    print("  chunk model carries a remainder fitted on B9's two SYNTHETIC S=C points,")
    print("  which is the only fitted part left and the only place its uncertainty lives.")
    print("  Per shape the multiple would move with nh/L and with hd (the re-commitment")
    print("  term is the key tile family's C term, and that one this module can now")
    print("  evaluate at any shape -- but the remainder cannot).")


def main(argv=None):
    global CELLS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=TOKENS,
                    help="tokens per chunk S (the measured sequence's is 512)")
    ap.add_argument("--seq-batch", type=int, default=1,
                    help="sequences per proof B -- E6a's knob, NOT the LogUp "
                         "lookup-batch size, which is --logup-batch")
    ap.add_argument("--logup-batch", type=int, default=None,
                    help="LogUp lookups sharing one interaction column (b2/out/"
                         "logup.json); nothing to do with --seq-batch")
    ap.add_argument("--target", type=float, default=None,
                    help="the throughput a `cards_for_20` row is stated against, in "
                         "PROVED POSITIONS per second (the paper's unit).  Default: "
                         f"the constants file's `{TARGET_KEY}`, or 20 where it names "
                         "none")
    ap.add_argument("--routing", metavar="FILE", default=None,
                    help="per-expert token counts measured on a real run, "
                         "`{slug: [[tokens, matmuls], ...]}`; a mixture row the "
                         "file names is counted at that distribution instead of "
                         "at the uniform one, which is what removes the routing "
                         "assumption")
    ap.add_argument("--cells", choices=CELLS_MODES, default=CELLS,
                    help="how the cell side of every row is counted: `exact` "
                         "builds a `params` block per segment and calls "
                         "`exact_cells` (the default, and the owner's ruling of "
                         "2026-09-11); `fitted` is the older per-unit-constant "
                         "path, kept so the two can be compared")
    ap.add_argument("--json", default="frontier.json")
    constants.add_argument(ap)
    inputs.add_arguments(ap, artifacts=True, inputs=True)
    ap.add_argument("--compare", default=None,
                    help="a previous frontier.json to diff every row against; prints "
                         "what moved in cells/token, tok/s and cards-at-target, and "
                         "whether the ordering changed")
    args = ap.parse_args(argv)
    constants.load(args.constants)
    CELLS = args.cells
    if args.routing:
        ROUTING.update(json.loads(pathlib.Path(args.routing).read_text()))
        bad = sorted(set(ROUTING) - {slug(s["name"]) for s in SHAPES})
        if bad:
            sys.exit(f"--routing {args.routing} names rows this table has no shape "
                     f"for: {', '.join(bad)}")
    if CELLS == "exact" and args.seq_batch != 1:
        sys.exit(f"--cells exact --seq-batch {args.seq_batch}: a batched proof "
                 f"commits B key caches and B wiring families, and one `params` "
                 f"block describes one segment of one sequence.  Price the batch "
                 f"sweep with --cells fitted, or leave B at 1.")
    assumptions = check_assumptions() if CELLS == "exact" else None
    refusals = ({s["name"]: refusal(s) for s in SHAPES} if CELLS == "exact"
                else {s["name"]: None for s in SHAPES})
    priceable = [s for s in SHAPES if not refusals[s["name"]]]
    if not priceable:
        sys.exit("no shape in this table can be stated as a `params` block, so "
                 "there is nothing to count:\n"
                 + "\n".join(f"  {n}: {w}" for n, w in refusals.items() if w))

    consts = load_constants()
    target_key = constants.find(TARGET_KEY)
    target = (args.target if args.target is not None else
              target_key["value"] if target_key else 20.0)
    # Rule 1 of this module: a number this table stands on never defaults and
    # never falls back to zero.  A cell cost with no key and no file is a term
    # `cells_per_token` would multiply by a count and charge nothing for, so the
    # table is not printed at all.
    unpriced = [n for n, e in cm.PROVENANCE.items()
                if e["source"] == "absent" and e.get("charged", True)]
    if unpriced:
        # Two different states, and they need two different things done about them:
        # an input a key already names is waiting for a run in zk-results to record
        # it, and an input no key names is waiting for a name first.
        named = [n for n in unpriced if cm.PROVENANCE[n]["key"]]
        unnamed = [n for n in unpriced if not cm.PROVENANCE[n]["key"]]
        msg = ["these cell costs have no source, and the model charges no term it "
               "cannot price:"]
        if named:
            msg.append("  a key names it and this constants file does not hold it -- "
                       "record it in zk-results:")
            msg += [f"    {n:<40} {cm.PROVENANCE[n]['key']}" for n in named]
        if unnamed:
            msg.append("  no key names it at all -- it needs a name before it can be "
                       "recorded:")
            msg += [f"    {n:<40} today: {cm.PROVENANCE[n]['file'] or 'no file'} "
                    f"under --inputs DIR" for n in unnamed]
        msg.append("Record a key for each in zk-results, or pass --inputs DIR to price "
                   "them from the")
        msg.append("micro-benchmark outputs -- which is a directory in no repository, so "
                   "a table priced")
        msg.append("that way is printed and is not recordable.")
        sys.exit("\n".join(msg))
    batch = args.logup_batch if args.logup_batch is not None else cm.BATCH
    if batch not in cm.LOGUP_MARGINAL:
        sys.exit(f"the LogUp marginal at batch {batch} is not there, so no cell count can "
                 f"be charged for a lookup.\n"
                 f"  measured batches: {sorted(cm.LOGUP_MARGINAL) or 'none'}\n"
                 f"  the key that would back it: "
                 f"model_constants.logup_marginal_cells_per_lookup (zk-prover #140)\n"
                 f"  the file that holds it today: logup.json under --inputs DIR")
    usd = {"H100": consts["h100_usd_per_hour"]["value"],
           "H200": consts["h200_usd_per_hour"]["value"]}
    # The measured side, constants first.  The headline card is the sound
    # statement's, per proved position, and every number in it is a key; the two
    # archive sequences are the fallback, they prove the decode-only statement, and
    # a row priced on one of them is marked as one and is not recordable.
    headline = headline_card()
    arch200, arch100 = h200_card(), h100_card()
    cards = {}
    if headline:
        cards[headline["card"]] = headline
    for c in (arch200, arch100):
        if c and c["card"] not in cards:
            cards[c["card"]] = c
    if not cards:
        sys.exit(NO_SEQUENCE)
    priced_card = headline or cards[sorted(cards)[0]]
    llama8b = next(s for s in SHAPES if s["name"] == "Llama-3.1-8B")
    shp = statement_shape(priced_card, llama8b, s_chunk=args.seq)
    cal = check_8b(priced_card, llama8b, consts, batch=batch, shp=shp)
    # The statement the model prices has to be the statement the run proved: the
    # positions and the segments it comes back with are the run's own.
    if cal.get("statement"):
        for ctx, st in cal["statement"].items():
            ref = priced_card["per_ctx"][ctx]
            if (st["positions"], st["segments"]) != (ref["positions"], ref["segments"]):
                sys.exit(
                    f"the statement this table prices is not the one the run at "
                    f"C={ctx} proved: {st['positions']} positions in {st['segments']} "
                    f"segments against the recorded {ref['positions']} and "
                    f"{ref['segments']}.")
    # The drift guard, now on the coefficients rather than on a composite slope.
    # This module derives them symbolically and `prefill-sweep-8b` measured them;
    # they must agree exactly at the 8B geometry, because both claim to be
    # identities of the same layout rather than fits.
    if not cal["coeffs_agree"]:
        sys.exit(f"the key tile base coefficients disagree with the measurement:\n"
                 f"  derived  {cal['coeffs_base']}\n"
                 f"  measured {tuple(cal['frozen_base'])}  "
                 f"({OPTIONAL['key_tile_base_coeffs']})\n"
                 "Both claim to be layout identities, so one of them is wrong; "
                 "resolve it, do not average.")
    bad = [r for r in cal["family"] if r[3] != r[4]]
    if bad:
        sys.exit("the key tile model no longer reproduces `family_cells.key_tiles` exactly on "
                 + ", ".join(r[0] for r in bad))
    alpha = None                       # the key tile model IS the attention term now
    fit = prefill_fit()
    # the measured proof size of one segment, per position in the block it covers.
    # From the constants file where a key names it, from the archive sequence where
    # none does; None where neither has it and section (2b) says so.
    seg_mib = priced_card["per_ctx"].get(4096, {}).get("mib_per_segment")
    mib_per_chunk = (seg_mib / args.seq if seg_mib else
                     (arch200["per_ctx"][4096]["proof_bytes"] / 2 ** 20
                      / arch200["per_ctx"][4096]["chunks"] / TOKENS) if arch200 else None)

    print("=" * 100)
    print("E1 -- the frontier price table")
    print("=" * 100)

    # --- the ledger ----------------------------------------------------------
    print("\nthe constants this table stands on (nothing below is typed in twice)")
    src = constants.path()
    print(f"  W0's freeze: {src}")
    for v in consts.values():
        print(f"    {v['key']:<52} {str(v['value']):>8}  [{v['grade']}]")
    print(f"\n  how the cells are counted: {CELLS}")
    if CELLS == "exact":
        print("    Every segment below is counted from its own `params` block "
              "(`exact_cells.py`),")
        print("    which reproduces all 1,872 recorded chunks in zk-results to the "
              "cell.  No cell")
        print("    cost is read for it and none is fitted.  The list below is what "
              "the FITTED")
        print("    count would charge, printed so a reader can see what the exact "
              "count replaced;")
        print("    the only entries a row here still reads are the preprocessed "
              "weight trace's")
        print("    and the two routing inputs, which the prover does not build at all.")
    else:
        print("    Every segment below is priced from the per-unit constants in the "
              "list, read off")
        print("    measured runs.  `--cells exact` counts the same segments from "
              "their layout.")
    print("\n  the cell costs the model charges, and where each one came from")
    print(f"    {'input':<40} {'value':>14}  source")
    for name, e in cm.PROVENANCE.items():
        where = (e["key"] if e["source"] == "constants"
                 else f"{e['source']}: {e['file'] or 'no file and no key'}")
        print(f"    {name:<40} {str(e['value']):>14}  {where}")
    backed = [n for n, e in cm.PROVENANCE.items() if e["source"] == "constants"]
    elsewhere = read_elsewhere()
    print(f"    {len(backed)} of {len(cm.PROVENANCE)} are backed by a key in the constants "
          f"file.  Every other one the model")
    print("    charges is named here with where it came from, because a row that reads "
          "any of them")
    print("    is a row this repository cannot record:")
    WHERE = dict(inputs="the micro-benchmark directory under --inputs DIR",
                 composed="this repository's own re-fit, harness/costmodel/composed_rungs.json",
                 analogy="nothing measures it anywhere",
                 absent="nothing prices it at all")
    for where, items in sorted(elsewhere.items()):
        print(f"      {where} ({WHERE.get(where, where)}), {len(items)}:")
        for it in items:
            print(f"        {it['input']:<40} {it['key'] or 'no key names it yet'}")
    if not elsewhere:
        print("      none: every input the model charges is a key of the constants file.")
    if CELLS == "exact":
        arows = assumption_rows(llama8b, seq=args.seq)
        print("\n  the assumption, and it is the whole of what an exact count still "
              "assumes")
        print("    A `params` block is not an architecture: some of its fields are "
              "the calibration")
        print("    a run took off its own witness, and a caller with a model and a "
              "context but no")
        print("    witness has to assume them.  Each one below is the modal value of "
              "the 64 layer")
        print("    and first segments of the two recorded 8B headline runs, and the "
              "bound beside it")
        print("    is the widest value those runs carry -- which is what the `bound` "
              "columns price.")
        print(f"    {'field':<16} {'assumed':>20} {'bound':>7} {'recorded':>18} "
              f"{'sizes, cells':>14} {'of a segment':>13}  read off")
        own = set(MOE_ASSUMPTIONS) | set(LATENT_ASSUMPTIONS)
        for r in [q for q in arows["rows"] if q["field"] not in own]:
            sh = r["share_of_segment"]
            vals = ", ".join(str(v) for v in r["recorded_values"][:4])
            print(f"    {r['field']:<16} {str(r['value']):>20} "
                  f"{str(r['bound']) if r['bound'] is not None else '-':>7} "
                  f"{vals:>18} "
                  + (f"{r['cells']:>14,}" if r["cells"] else f"{'-':>14}")
                  + (f" {sh:>12.4%}" if sh is not None else f" {'-':>12}")
                  + f"  {r['source_run'][:10]} {r['source_segment']}")
        moe_rows = [q for q in arows["rows"] if q["field"] in MOE_ASSUMPTIONS]
        if moe_rows:
            print(f"\n    and the two a counted MIXTURE row adds, on "
                  f"{arows['moe_reference']}, one layer segment:")
            for r in moe_rows:
                print(f"      {r['field']:<12} {r['value']:<38} [{r['grade']}]  "
                      f"{r['cells']:>14,.0f} cells  {r['share_of_segment']:>8.2%}")
                for line in _wrap(r["note"], 88):
                    print(f"        {line}")
        lat_rows = [q for q in arows["rows"] if q["field"] in LATENT_ASSUMPTIONS]
        if lat_rows:
            print(f"\n    and the four a counted LATENT-ATTENTION row adds, on "
                  f"{arows['latent_reference']}, one layer segment:")
            for r in lat_rows:
                print(f"      {r['field']:<19} {str(r['value']):<31} [{r['grade']}]  "
                      f"{(r['cells'] or 0):>14,.0f} cells  "
                      f"{r['share_of_segment']:>8.2%}")
                for line in _wrap(r["note"], 88):
                    print(f"        {line}")
        print(f"    the five `ao_*` range fields size ONE group of tables together, "
              f"so their cells")
        print(f"    column is that group's and is not added up.  A driver in "
              f"zk-results records")
        print(f"    each row as a named constant ({ASSUMPTION_KEY.format(field='<field>')}"
              f") and every")
        print(f"    projected row names them in its `inputs`.")
    for alias in ("h100_usd_per_hour", "h200_usd_per_hour"):
        e = consts[alias]
        print(f"    {e['card']} ${e['value']}/GPU-h  [{e['grade']}]  "
              f"{e.get('source') or e.get('note', '')}")
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

    print("\n  the measured side, and the statement it was measured on")
    for name, card in cards.items():
        where = ("the constants file" if card.get("source") == "constants"
                 else "the private archive, which is in no repository")
        print(f"    {name:<6} {card.get('statement', 'unknown'):<12} from {where}")
        for ctx in sorted(card["per_ctx"]):
            ks = card["per_ctx"][ctx].get("keys") or []
            print(f"      C={ctx:<6} " + (", ".join(ks) if ks else card["per_ctx"][ctx]["source"]))
    print(f"    the statement's own layout ({shp['s_chunk']}-position blocks, "
          f"{shp['n_output']} output positions,")
    print(f"    {shp['heads']} output head segments) is "
          + (f"read from {', '.join(shp['keys'])}"
             if shp["source"] == "constants" else
             "assumed: no key records it"))

    print("\nthe four boundaries, kept apart (W0's freeze names all four)")
    print("  commit      interpolate + LDE + Merkle over the committed trees")
    print("  stwo total  commit + prove_ex (constraint eval, quotients, OODS, FRI)")
    print("  whole wall  trace gen + interaction gen + stwo + serialize, summed per")
    print("              chunk.  Has NEVER counted the witness read or the layer build.")
    print("  loop        the chunk loop's own wall, which DOES count them.  A FOURTH")
    print("              boundary, not a superset and not a subset of `whole wall`: it")
    print("              counts more work and, where the loop is pipelined, still")
    print("              finishes sooner.  `rate.headline.*.loop` is this one, and the")
    print("              archive's sequence calls it `loop_secs`.")
    for name, card in cards.items():
        for ctx, ref in sorted(card["per_ctx"].items()):
            bs = "  ".join(f"{b} {m['cells_s']/1e9:.3f} G/s" for b, m in ref["boundaries"].items())
            if "cells_per_position" in ref:
                what = (f"{ref['segments']:>4} segments  "
                        f"{ref['cells_per_position']/1e6:7.2f} M cells/position")
            else:
                what = (f"{ref['chunks']:>4} chunks    "
                        f"{ref['cells_per_token']/1e6:7.2f} M cells/token  ")
            print(f"    {name} C={ctx:<6} {what}   {bs}")
            print(f"    {'':<14} {ref['source']}")
        print(f"    {'':<14} {card['note']}")

    # --- the check: no free parameter at all ---------------------------------
    print("\nthe check: the attention term now has ZERO fitted parameters")
    print("  The model used to charge one slope, `alpha`, solved from the two measured")
    print("  contexts -- one parameter, two points, one residual dof.  Stream DD's key tile")
    print("  model and this module's symbolic re-derivation of it replace that slope with")
    print("  a closed form in (S, C) whose every coefficient is an identity of the layout,")
    print("  so both measured levels are now predicted with nothing solved: two points,")
    print("  zero fitted parameters, TWO residual degrees of freedom.")
    if cal.get("statement"):
        st0 = cal["statement"][min(cal["statement"])]
        print(f"\n  What is compared, and it is the whole of the campaign's ruling: the "
              f"model's own")
        print(f"  price for the SOUND STATEMENT -- the prompt and the output proved as one "
              f"statement,")
        print(f"  in blocks of {st0['s_chunk']}, each block's layer segments attending over "
              f"the blocks before")
        print(f"  it, {st0['heads']} output head segments beside them, the five reductions "
              f"on -- against what")
        print(f"  the two headline runs measured, per proved position.  The statement's own "
              f"numbers")
        print(f"  are read from {', '.join(st0['keys']) or 'no key'} and the layer count is "
              f"the shape's.")
        print("  The archive's sequences prove a DIFFERENT statement (512 decoded tokens, "
              "each")
        print("  against a full C-key prefix, before the tiled prefix and the reductions "
              "existed), and")
        print("  a rate measured on one statement over cells modelled on another does not "
              "divide.")

    def legacy_col(which, ctx, width):
        v = cal[which]
        return f"{v[ctx]:>+{width}.2%}" if v else f"{'not recorded':>{width + 1}}"

    unit = "M/position" if cal.get("statement") else "M/token"
    print(f"\n  {'context':>8} {'positions':>10} {'measured ' + unit:>20} "
          f"{'key tile model':>14} {'residual':>10}   {'single slope':>12} {'B5 alpha':>12}")
    for ctx in sorted(cal["measured"]):
        m, q = cal["measured"][ctx], cal["predicted"][ctx]
        n = cal["statement"][ctx]["positions"] if cal.get("statement") else TOKENS
        print(f"  {ctx:>8} {n:>10,} {m/1e6:>20.2f} {q/1e6:>14.2f} "
              f"{cal['key_tile'][ctx]:>+9.2%}   {legacy_col('legacy_slope', ctx, 11)} "
              f"{legacy_col('legacy_b5', ctx, 11)}")
    print(f"  key tile model residual {min(cal['key_tile'].values()):+.2%} to "
          f"{max(cal['key_tile'].values()):+.2%}, on no fitted attention parameter at all.")
    print(f"  recorded as `frontier.calibration.residual_pct` (worst of the two) and "
          f"per context beside it.")
    print("  The two superseded columns are the same model with the key tile term replaced")
    print("  by a single slope, each read from the key named beside it; where the constants")
    print("  file does not name one the column says so.")

    if cal.get("block"):
        print("\n  the calibration the exact count is answerable to, and it compares "
              "the same")
        print("  segments on both sides: the 32 layer segments of the block whose "
              "prefix is the")
        print("  whole context, over the 512 positions that block proves, with no "
              "output head")
        print("  segment on either side.")
        print(f"  {'context':>8} {'recorded /position':>19} {'counted':>16} "
              f"{'residual':>9} {'at the bound':>16} {'':>9}   at each segment's own "
              f"recorded fields")
        for ctx, r in sorted(cal["block"].items()):
            if r["recorded"] is None:
                print(f"  {ctx:>8} {'not recorded':>19}   ({r['key']})")
                continue
            print(f"  {ctx:>8} {r['recorded']:>19,.0f} {r['assumed']:>16,.0f} "
                  f"{r['assumed_pct']:>+8.2f}% {r['at_bound']:>16,.0f} "
                  f"{r['at_bound_pct']:>+8.2f}%   {r['recorded_fields_pct']:>+.4f}%")
        print("  The last column is the whole point: give the count each segment's own "
              "recorded")
        print("  witness fields and it is EXACT, to the cell, on both blocks.  The "
              "residual beside")
        print("  it is what the assumption costs and nothing else -- it is not a "
              "modelling error.")
    print("\n  the coefficients, derived here and measured by `prefill-sweep-8b` -- they")
    print("  must agree exactly, because both claim to be identities of the same layout:")
    print(f"  {'':<10} {'S*C':>10} {'C':>12} {'S':>12}")
    print(f"  {'base':<10} {cal['coeffs_base'][0]:>10} {cal['coeffs_base'][1]:>12} "
          f"{cal['coeffs_base'][2]:>12}   derived symbolically here")
    if cal["frozen_base"]:
        print(f"  {'':<10} {cal['frozen_base'][0]:>10} {cal['frozen_base'][1]:>12} "
              f"{cal['frozen_base'][2]:>12}   {OPTIONAL['key_tile_base_coeffs']}")
    else:
        print(f"  {'':<10} {'not recorded':>36}   {OPTIONAL['key_tile_base_coeffs']}")
    print(f"  {'fresh':<10} {cal['coeffs_fresh'][0]:>10} {cal['coeffs_fresh'][1]:>12} "
          f"{cal['coeffs_fresh'][2]:>12}   derived symbolically here, as a plain trilinear")
    print("  There is no measured fresh triple to check it against, and there will not be")
    print("  one at this configuration: with the five reductions on, most of the key tiles'")
    print("  interaction sits in the reduction rather than in their own columns, so solving")
    print("  for it over the measured points returns a negative per-token cost.  What checks")
    print("  the interaction side instead is the family below, to the cell.")
    n_exact = sum(1 for r in cal["family"] if r[3] == r[4])
    print(f"  and `family_cells.key_tiles` reproduced EXACTLY, to the cell, on "
          f"{n_exact}/{len(cal['family'])} recorded points")
    print("  (`key_tile_segments.json`: every 8B layer segment of the two recorded")
    print("  sequences, at both contexts and with the five reductions on and off, plus")
    print("  whatever the artifacts directory adds).  Nothing is fitted, so a miss would")
    print("  be a disagreement, not a residual, and this module exits on one.")

    # --- the general form ----------------------------------------------------
    print("\n  the closed form, general in (nh, hd, S, C).  DD froze the coefficients at")
    print("  Llama-8B's geometry; these are the same numbers re-derived from the column")
    print("  accounting so they can be evaluated at a shape that is not Llama-8B:")
    print("      base  = 2*nh*S*C  +  2*nh*hd*C  +  nh*hd*S  +  4*nh*S")
    print("      inter = (2 + 24/gt)*nh*S*C  +  (4*nh*hd + 12*nh*hd/ct)*C")
    print("              +  2*nh*hd*S  +  (8*nh + 8*nh/rw)*S")
    print("  with gt = grid_target (64), ct = chan_target (128), rw = row width (16).")
    print(f"  At nh=32, hd=128 the derived triples are {cal['coeffs_fresh']} fresh and "
          f"{cal['coeffs_base']} base.")
    print("  Read off `prover/src/attn_trace.rs:1167-1181` with `prover/src/layout.rs`:")
    print("    lc.g.rows = hd*insts*C/pk (pk -> ct), lk.g.rows = C*insts*S/q (q -> gt),")
    print("    rrows*rw = nh*S, and `insts` sums to nh over the key tiles -- every term is")
    print("    linear in `insts`, so grouping the heads 16/8/8 costs exactly what 32 would.")
    print("\n  Three things this derivation does NOT settle, and they are the extrapolation")
    print("  risk the frontier rows carry:")
    print("   1. **K and V are committed per ATTENTION head, not per KV group.**")
    print("      `LayoutChan`/`LayoutKey` are built from `Shape::chans()`/`keys()`, both")
    print("      carrying `insts` = attention heads, so the `C` term is 2*nh*hd and `nkv`")
    print("      does not appear, and the boundary trees hand on a cache of the same")
    print("      width.  At 8B nh/nkv is 4; at Llama-405B and Qwen3 it is 16.  The three")
    print("      latent-attention shapes hold no K and no V per key/value head at all"
          + (" --" if CELLS == "exact" else "."))
    if CELLS == "exact":
        print("      what a key costs their cache, both ways, is section (2d).")
    if CELLS == "exact":
        kv8 = {ctx: kv_term(llama8b, ctx, seq=args.seq) for ctx in CONTEXTS}
        print(f"      What that is worth is now COUNTED and not named: on one 8B layer "
              f"segment")
        print(f"      the two caches and the boundary trees that hand them on are "
              f"{kv8[4096]['share_of_fresh']:.1%} of the")
        print(f"      segment at C=4,096 and {kv8[16384]['share_of_fresh']:.1%} at "
              f"C=16,384, and committing them once per key/value")
        print(f"      head instead would leave {kv8[4096]['saving_of_fresh']:.1%} and "
              f"{kv8[16384]['saving_of_fresh']:.1%} of the segment uncommitted.  Per proved")
        print("      position over the whole statement, shape by shape, that is section (2c).")
        print("      Not built: no AIR in this project shares a key tile's K and V columns")
        print("      between two heads' score grids, and what this table prices is what the")
        print("      prover commits today.")
    else:
        print("      Grouped query attention buys this prover nothing on the committed KV")
        print("      side today.  What a layout that used it would save is counted under")
        print("      `--cells exact`, shape by shape, and this path does not count it: it")
        print("      charges the cache through a per-unit constant read off one segment.")
    print("   2. **gt / ct / rw are run knobs, not constants.**  The composed sequence used")
    print("      64 / 128 / 16 and this model keeps them.  They tile exactly only when S,")
    print("      C, hd and `insts` are powers of two and the target divides; `Grid::new`")
    print("      also caps q at min(S, hd) and pk at min(C, S).  Where a shape saturates a")
    print("      cap the table says so.")
    print("   3. **The out (A.V) component is not in the key tile family.**  It is the `attnout`")
    print("      family and `cells_per_token` charges it separately, so there is no double")
    print("      count -- but it is also the one attention component whose price here is a")
    print("      measured per-element constant rather than a derived column count.")
    if CELLS == "exact":
        print("  Under the exact count this is not a question the table asks: the "
              "grids are")
        print("  built rather than closed-form, so a layout that does not tile is "
              "counted as it")
        print("  tiles and nothing is reported as saturated.")
    else:
        sat = [r["name"] for r in
               [dict(name=s["name"], c=model_cells(s, 16384, seq=args.seq, batch=batch))
                for s in priceable] if r["c"].get("saturated")]
        print("  layout caps saturated at: " + (", ".join(sat) if sat else "no shape"))


    # --- the shapes ----------------------------------------------------------
    print("\n(1) the shapes, and each one's own parameter check")
    print("  the check: the model's own arithmetic over the fetched config, plus the input")
    print("  embedding (which the proof never covers), against the vendor's published counts.")
    print(f"  {'model':<18} {'L':>4} {'d':>6} {'nh':>4} {'dff':>7} {'E':>5} {'k':>3} "
          f"{'sh':>3} {'total (model)':>14} {'published':>10} {'d':>7} "
          f"{'active (model)':>15} {'published':>10} {'d':>7}")
    rows = []
    for shape in SHAPES:
        L = sum(grp["layers"] for grp in shape["groups"])
        moe = [grp for grp in shape["groups"] if grp["routed"]]
        E = moe[0]["routed"] if moe else 0
        k = moe[0]["per_token"] if moe else 0
        sh = moe[0]["shared"] if moe else 0
        dff = moe[0]["moe_dff"] if moe else shape["groups"][0]["dense_dff"]
        tot = params_of(shape, freivalds=True, embedding=True)
        act = params_of(shape, embedding=True)
        dt = tot / shape["check_total"] - 1 if shape.get("check_total") else None
        da = act / shape["check_active"] - 1 if shape.get("check_active") else None
        bad = (dt is not None and abs(dt) > 0.05) or (da is not None and abs(da) > 0.05)
        print(f"  {shape['name']:<18} {L:>4} {shape['d']:>6} {shape['nh']:>4} {dff:>7} "
              f"{E:>5} {k:>3} {sh:>3} {g(tot):>14} {g(shape['check_total']):>10} {dt:>+6.1%} "
              f"{g(act):>15} {g(shape['check_active']):>10} {da:>+6.1%}"
              + ("  [check]" if bad else ""))
        rows.append(dict(shape=shape, total=tot, active=act, d_total=dt, d_active=da,
                         flagged=bad))
    print("  E is the ROUTED expert count, k the experts a token routes to and sh "
          "the shared experts,")
    print("  each as its own published field; `dff` is the expert width on a "
          "mixture row and the")
    print("  feed-forward width on a dense one.  Every one of them is a key of the "
          "model's own")
    print("  config.json, named in the source line below, and none is solved -- "
          "except the")
    print("  hypothetical row, which says so.")
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
                    target_tok_s=target, mib_per_chunk=mib_per_chunk, batch=batch,
                    shp=shp)
              for s in priceable]
    if any(refusals.values()):
        print("\n  refused, and not priced below: a shape whose segment is not a "
              "`params` block")
        print("  has no exact cell count, and this table prints the refusal rather "
              "than the fitted")
        print("  number under the same heading.")
        for name, why in refusals.items():
            if why:
                print(f"    {name:<18} {why}")

    print(f"\n(2) committed cells, S={args.seq} positions/block, B={args.seq_batch} "
          "sequence(s)/proof, at both statements")
    print("  per PROVED POSITION is the sound statement and is what every priced row "
          "below divides;")
    print("  per DECODED TOKEN is one token against a full C-key prefix, which is what "
          "the archive's")
    print("  sequences prove and what sections (2a) and (6) compare against.")
    print(f"  {'model':<18} {'C=4K /pos':>11} {'bound':>10} {'C=16K /pos':>11} "
          f"{'bound':>10} {'C=4K /token':>12} "
          f"{'C=16K /token':>12} {'C=4K pre':>10} {'freivalds %':>11} {'grid %':>7} "
          f"{'segments':>9}")
    for r in priced:
        a, b = r["contexts"][4096], r["contexts"][16384]
        print(f"  {r['name']:<18} {g(a['cells_per_position']):>11} "
              f"{g(a['cells_per_position_bound']):>10} "
              f"{g(b['cells_per_position']):>11} "
              f"{g(b['cells_per_position_bound']):>10} {g(a['cells_per_token']):>12} "
              f"{g(b['cells_per_token']):>12} {g(a['preprocessed_per_token']):>10} "
              f"{a['parts']['freivalds']/a['cells_per_token']*100:>10.1f}% "
              f"{a['parts']['attn']/a['cells_per_token']*100:>6.1f}% "
              f"{a['statement']['segments']:>9,}")
    if CELLS == "exact":
        print("  `bound` is the same statement with every witness-calibrated field of "
              "the `params`")
        print("  block at the widest value the two recorded runs carry.  It is a "
              "bound over the")
        print("  evidence and not a theorem: the prover's automatic path caps the "
              "exponent table")
        print("  at 2^22 and every value above that is the per-layer calibration a "
              "model manifest")
        print("  publishes, so the ceiling a projection can state is the widest one "
              "a run has")
        print("  published.")
    print("  `pre` is the preprocessed weight trace, committed offline but read once per")
    print("  proof.  It is NOT in the fresh count the measured rates divide, and it is")
    print("  where MoE's x E lands hardest -- see (4).")
    print("  A proved position costs LESS than a decoded token at the same context "
          "because most")
    print("  of the sequence's positions are in earlier blocks, which attend over a "
          "shorter prefix.")

    print("\n(2a) what the superseded attention terms cost.  Section 6.3's dense table")
    print("     was produced at B5's standalone block AIR; the version this work item")
    print("     committed at 3ec240c used a single composed slope solved to hit the two")
    print("     measured points.  Both are here against the key tile model.")
    b5 = consts["alpha_b5"]["value"]
    slope = consts["alpha_e1"]["value"]
    grid_delta = {}
    superseded = b5 is not None and slope is not None
    if superseded:
        print(f"  {'model':<18} {'@B5':>10} {'@slope':>10} {'key_tile':>10} "
              f"{'key tile vs B5':>13} {'key tile vs slope':>14}   "
              f"{'C=16K: vs B5':>15} {'vs slope':>10}")
    else:
        for alias in ("alpha_b5", "alpha_e1"):
            e = consts[alias]
            print(f"  {e['key']:<52} "
                  + ("not recorded" if not e["recorded"] else str(e["value"])))
        print("  What a superseded slope cost is exactly those two numbers and there is")
        print("  nothing to substitute for one, so the table is not printed.  No priced")
        print("  row depends on them.")
    for shape in (SHAPES if superseded else []):
        row = {}
        for ctx in (4096, 16384):
            # the superseded terms are fitted slopes, so this section is the
            # fitted path on both sides whatever `--cells` says
            kw = dict(seq=args.seq, seq_batch=args.seq_batch, batch=batch,
                      cells="fitted")
            row[ctx] = dict(
                b5=model_cells(shape, ctx, alpha=b5, **kw)["total"],
                slope=model_cells(shape, ctx, alpha=slope, **kw)["total"],
                key_tile=model_cells(shape, ctx, **kw)["total"])
            row[ctx]["d_b5"] = row[ctx]["key_tile"] / row[ctx]["b5"] - 1
            row[ctx]["d_slope"] = row[ctx]["key_tile"] / row[ctx]["slope"] - 1
        grid_delta[shape["name"]] = {str(k): v for k, v in row.items()}
        a, b = row[4096], row[16384]
        print(f"  {shape['name']:<18} {g(a['b5']):>10} {g(a['slope']):>10} {g(a['key_tile']):>10}"
              f" {a['d_b5']:>+12.1%} {a['d_slope']:>+13.1%}   {b['d_b5']:>+14.1%}"
              f" {b['d_slope']:>+9.1%}")
    if grid_delta:
        d5 = [v["4096"]["d_b5"] for v in grid_delta.values()]
        d5c = [v["16384"]["d_b5"] for v in grid_delta.values()]
        ds = [v["4096"]["d_slope"] for v in grid_delta.values()]
        dsc = [v["16384"]["d_slope"] for v in grid_delta.values()]
        print(f"  Against B5's {b5} the key tile model is {min(d5):+.1%} to {max(d5):+.1%} "
              "at C=4K")
        print(f"  and {min(d5c):+.1%} to {max(d5c):+.1%} at C=16K -- that is the error in")
        print("  section 6.3's table, and it grows with the context because the term that")
        print("  was wrong is the one on the context.")
        print(f"  Against the single slope {slope} it is {min(ds):+.1%} to {max(ds):+.1%} "
              f"(4K) and")
        print(f"  {min(dsc):+.1%} to {max(dsc):+.1%} (16K) -- a spread of "
              f"{max(ds)-min(ds):.1%} (4K) / {max(dsc)-min(dsc):.1%} (16K)")
        print("  across the eight shapes.  So the aliasing is BENIGN at this table's")
        print("  settings, and it is worth being precise about why, because the reason is")
        print("  not that a composite slope is safe.  Write the composite out:")
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
    print("  Two things follow, and they are the reason the key tile model is still required:")
    print("   * The composite moves 13% between S=512 and S=1024 and doubles by S=128.  A")
    print("     'constant' that is a function of the chunk length is not extrapolable to")
    print("     the two regimes this table cares most about -- E11's S=1024 prompt chunks")
    print("     (section 6) and any shape forced off S=512 by memory (section 7).")
    print("   * Most of the residual above is NOT aliasing.  It is that the old model")
    print("     charged the committed K/V at `nkv` while the AIR charges it at `nh`, so the")
    print("     correction tracks nh/nkv: +4.3% at Qwen3 (nh/nkv=16) and +4.0% at")
    print("     DeepSeek-V3, against +2.0% at Maverick (5) and +1.5% at Kimi K3, whose")
    print("     69 linear-attention layers have no grid to correct.  Those three ratios")
    print("     are the old model's own reading of an attention it describes as GQA.")
    if CELLS == "exact":
        print("     What their attention actually holds is section (2d).")

    print("\n(2b) proof bytes: two estimates that disagree, and neither is measured above 8B")
    if not mib_per_chunk:
        print("  not answered: neither `proof.headline.c4096.bytes_per_segment_median` nor "
              "the archive")
        print("  sequence gives one segment's proof size, so the left column has nothing "
              "to scale.")
    print(f"  {'model':<18} {'by chunk count':>15} {'by 0.15 MiB/B-active':>22} {'ratio':>8}")
    for r in (priced if mib_per_chunk else []):
        e = r["contexts"][4096]
        a, b = e["mib_per_token_chunks"], e["mib_per_token_params"]
        print(f"  {r['name']:<18} {a:>15.2f} {b:>22.2f} {b / a:>7.1f}x")
    print("  Left: (L+1) segments x the measured 8B per-segment proof size "
          + (f"({mib_per_chunk * args.seq:.2f} MiB a segment)."
             if mib_per_chunk else "(not recorded)."))
    print("  Right: section 6.3's rule of thumb, 0.15 MiB per token per B active params,")
    print("  which is the same 8B measurement divided by 8.03 B and then scaled linearly.")
    print("  They cannot both be right.  Proof size is queries x columns x log(rows), and")
    print("  B6's packing measured columns *flat* from 1 to 512 instances -- a bigger model")
    print("  puts its extra work in rows, which the proof pays for logarithmically.  So the")
    print("  left column is the better structural argument and section 6.3's ~61 MiB/token")
    print("  at 405B is very likely an order of magnitude too high.  Neither is measured at")
    print("  frontier scale; the left column also holds the per-chunk size fixed, so it is")
    print("  a floor.  **Open: this needs one measured point at a wider shape.**")

    if CELLS == "exact":
        print("\n(2c) the keys and values, counted per attention head and per "
              "key/value head")
        print("  The AIR commits one K and one V column set per ATTENTION head: "
              "`LayoutChan` and")
        print("  `LayoutKey` are built from `Shape::chans()` and `Shape::keys()`, "
              "both carrying")
        print("  `insts` = attention heads, and the boundary trees hand on a cache "
              "of that same")
        print("  width.  Every shape below shares its keys and values across heads, "
              "so each one is")
        print("  committed `nh/nkv` times over.  The right-hand columns count the "
              "same three terms")
        print("  at the key/value head count instead -- the layout is BUILT at that "
              "instance count,")
        print("  not divided by anything.  No AIR in this project builds it: this "
              "prices it.")
        print("  A latent attention holds neither a K nor a V per key/value head, "
              "so the three")
        print("  latent rows are not here: what their cache costs, both ways, is "
              "section (2d).")
        gqa = [r for r in priced if r["contexts"][4096]["kv"]]
        print(f"  {'model':<18} {'nh':>4} {'nkv':>4} {'nh/nkv':>7} "
              f"{'C=4K /pos':>11} {'kv-shared':>11} {'saving':>8} "
              f"{'C=16K /pos':>11} {'kv-shared':>11} {'saving':>8}")
        for r in gqa:
            a, b = r["contexts"][4096], r["contexts"][16384]
            kv = a["kv"]
            print(f"  {r['name']:<18} {kv['nh']:>4} {kv['nkv']:>4} "
                  f"{kv['nh'] // kv['nkv']:>6}x "
                  f"{g(a['cells_per_position']):>11} "
                  f"{g(a['cells_per_position_kv_shared']):>11} "
                  f"{a['kv_shared_saving_pct']:>+7.1f}% "
                  f"{g(b['cells_per_position']):>11} "
                  f"{g(b['cells_per_position_kv_shared']):>11} "
                  f"{b['kv_shared_saving_pct']:>+7.1f}%")
        print("\n  the term itself, on ONE layer segment at the full prefix: the key "
              "cache in the")
        print("  channel component, the value cache in the key grid, and the "
              "boundary trees that")
        print("  hand both on.  `of segment` is their share of that segment's "
              "committed cells.")
        print(f"  {'model':<18} {'C':>7} {'keys':>10} {'values':>10} "
              f"{'boundary':>10} {'as built':>10} {'of segment':>11} {'shared':>10} "
              f"{'saving':>10} {'of segment':>11}")
        for r in gqa:
            for ctx in CONTEXTS:
                kv = r["contexts"][ctx]["kv"]
                print(f"  {r['name']:<18} {ctx:>7,} {g(kv['as_built']['keys']):>10} "
                      f"{g(kv['as_built']['values']):>10} "
                      f"{g(kv['as_built']['boundary']):>10} "
                      f"{g(kv['as_built']['total']):>10} "
                      f"{kv['share_of_fresh']:>10.2%} "
                      f"{g(kv['shared']['total']):>10} {g(kv['saving']):>10} "
                      f"{kv['saving_of_fresh']:>10.2%}")
        bnd = priced_card["per_ctx"][4096]["boundaries"]
        bname = next(b for b in ("loop", "loop_secs", "whole wall", *bnd) if b in bnd)
        rate = usd.get(priced_card["card"])
        print(f"\n  what that is worth on the priced card, at the {bname} boundary"
              + (f" and ${rate:.2f}/GPU-h:" if rate else ":"))
        at = f"cards@{target:g}"
        print(f"  {'model':<18} " + " ".join(
            f"{h:>12}" for h in ("C=4K $/Mpos", "kv-shared", at, "kv-shared",
                                 "C=16K $/Mpos", "kv-shared", at, "kv-shared")))
        for r in gqa:
            cells = []
            for ctx in CONTEXTS:
                m = r["contexts"][ctx]["cards"].get(priced_card["card"], {}).get(bname)
                cells += ([m["usd_per_mposition"], m["usd_per_mposition_kv_shared"],
                           m["cards_for_target"], m["cards_for_target_kv_shared"]]
                          if m else [None] * 4)
            print(f"  {r['name']:<18} "
                  + " ".join(f"{x:>12,.0f}" if isinstance(x, (int, float)) else f"{'n/a':>12}"
                             for x in cells))
        print("  Every dollar column moves by exactly the cells' own fraction, so "
              "the other three")
        print("  timing scopes move by the same per cent.  What is NOT in the saving: "
              "the interaction")
        print("  columns those same key and value columns carry.  With the five "
              "reductions on -- the")
        print("  statement both headline runs prove -- their LogUp running sums are "
              "in the batched")
        print("  reduction, and what the link pays is one price per distinct trace "
              "height, which a")
        print("  layout nobody has built decides.  So each saving above is a floor.")
        print("  The projections the q, k and v matmuls pay are already per key/value "
              "head, at the")
        print("  narrower output width the shape gives them; this term is the "
              "attention grid's own")
        print("  cache and the boundary trees, and it is the whole of what the "
              "sharing would move.")

    lat_rows = ([r for r in priced if r["contexts"][4096]["latent"]]
                if CELLS == "exact" else [])
    if lat_rows:
        print("\n(2d) multi-head latent attention, counted both ways")
        print("  These three shapes project no key and no value per head.  Both "
              "sides of the")
        print("  attention go through a low rank -- the queries through "
              "`q_lora_rank`, the keys and")
        print("  values through `kv_lora_rank` -- and the position is carried by a "
              "decoupled key of")
        print("  `qk_rope_head_dim` computed once per KEY rather than once per head. "
              " So what is")
        print("  committed per key is the latent plus that key, not `2 nh hd`, and "
              "the layer's")
        print("  matmuls are five and not four.  Every width below is one of the "
              "five fields of the")
        print("  model's own config.json, named in its source line above.")
        print("  The two ways are where the low-rank pair `kv_b` is applied, and "
              "the model's")
        print("  weights are the SAME either way -- `W_UK` and `W_UV` together ARE "
              "`kv_b`:")
        print("    materialised  the keys and values are expanded per head before "
              "the attention,")
        print("                  which is what a prover with no latent layout "
              "builds;")
        print("    absorbed      `kv_b` is folded into the query and the output "
              "projection, and the")
        print("                  cache IS the latent, one for every head at once.")
        print("  NEITHER is built: no AIR in this project proves a latent "
              "attention.  The row")
        print(f"  records the {LATENT_FORM} one (`assumption.cost_model.mla_form`) "
              "and prints both.")
        print(f"  {'model':<18} {'n_h':>4} {'hd_qk':>6} {'hd_v':>5} "
              f"{'cache/key':>10} {'C=4K /pos':>11} {'other way':>11} {'move':>8} "
              f"{'C=16K /pos':>11} {'other way':>11} {'move':>8}")
        for r in lat_rows:
            a, b = r["contexts"][4096], r["contexts"][16384]
            lt = a["latent_term"][a["latent"]]
            print(f"  {r['name']:<18} "
                  f"{lt['nh']:>4} {lt['hd_qk']:>6} "
                  f"{lt['hd_v']:>5} {lt['cache_per_key']:>10,} "
                  f"{g(a['cells_per_position']):>11} "
                  f"{g(a['cells_per_position_latent_alt']):>11} "
                  f"{a['latent_alt_pct']:>+7.1f}% "
                  f"{g(b['cells_per_position']):>11} "
                  f"{g(b['cells_per_position_latent_alt']):>11} "
                  f"{b['latent_alt_pct']:>+7.1f}%")
        print("\n  the cache itself, on ONE layer segment at the full prefix: the "
              "key side in the")
        print("  channel component, the value side in the key grid, and the "
              "boundary trees that")
        print("  hand both on.  `of segment` is their share of that segment's "
              "committed cells.")
        print(f"  {'model':<18} {'C':>7} {'form':<13} {'keys':>10} {'values':>10} "
              f"{'boundary':>11} {'total':>11} {'of segment':>11} {'segment':>11}")
        for r in lat_rows:
            for ctx in CONTEXTS:
                for form in la.FORMS:
                    lt = r["contexts"][ctx]["latent_term"][form]
                    mark = "*" if form == r["contexts"][ctx]["latent"] else " "
                    print(f"  {r['name']:<18} {ctx:>7,} {form + mark:<13} "
                          f"{g(lt['keys']):>10} {g(lt['values']):>10} "
                          f"{g(lt['boundary']):>11} {g(lt['total']):>11} "
                          f"{lt['share_of_fresh']:>10.2%} {g(lt['fresh']):>11}")
        print("  * the form the row records.")
        bnd = priced_card["per_ctx"][4096]["boundaries"]
        bname = next(b for b in ("loop", "loop_secs", "whole wall", *bnd) if b in bnd)
        rate = usd.get(priced_card["card"])
        print(f"\n  what that is worth on the priced card, at the {bname} boundary"
              + (f" and ${rate:.2f}/GPU-h:" if rate else ":"))
        at = f"cards@{target:g}"
        print(f"  {'model':<18} " + " ".join(
            f"{h:>12}" for h in ("C=4K $/Mpos", "other way", at, "other way",
                                 "C=16K $/Mpos", "other way", at, "other way")))
        for r in lat_rows:
            cells = []
            for ctx in CONTEXTS:
                m = r["contexts"][ctx]["cards"].get(priced_card["card"], {}).get(bname)
                cells += ([m["usd_per_mposition"], m["usd_per_mposition_latent_alt"],
                           m["cards_for_target"], m["cards_for_target_latent_alt"]]
                          if m else [None] * 4)
            print(f"  {r['name']:<18} "
                  + " ".join(f"{x:>12,.0f}" if isinstance(x, (int, float)) else f"{'n/a':>12}"
                             for x in cells))
        print("  What is NOT counted either way: a selection over the cache, a "
              "paged cache, and")
        print("  the top-k the mixture side needs, which section (2b)'s assumption "
              "block states.")
        print("  What the two forms do not move at all is the weight trace: the "
              "preprocessed")
        print("  column is the same number in both, because the two are the same "
              "weights applied")
        print("  in a different order.")

    for n_card, cardname in enumerate(sorted(cards, key=lambda c: c != priced_card["card"])):
        card = cards[cardname]
        rate = usd.get(cardname)
        print(f"\n(3{'ab'[n_card] if n_card < 2 else n_card}) {cardname}: per-card PROVED "
              f"POSITIONS/s at every boundary, cards for {target:g} positions/s, "
              f"$/M positions at "
              f"{('$%.2f/GPU-h' % rate) if rate else 'n/a (no reference rate)'}")
        print(f"  the statement is the sound one and the unit is a proved position, not a "
              f"decoded token.")
        if card.get("source") != "constants":
            print(f"  This card's rates come from the archive's {card.get('statement')} "
                  "sequence, not from a key of")
            print("  the constants file, so every row below is a rate measured on one "
                  "statement over a cell")
            print("  count modelled on another, and none of them is recordable.")
        for ctx in sorted(card["per_ctx"]):
            print(f"  C={ctx}")
            bnames = list(card["per_ctx"][ctx]["boundaries"])
            # the widest boundary this card actually has: the chunk loop's own wall
            # where it was measured, `whole wall` where it was not.  They are
            # DIFFERENT boundaries and the header says which the dollars are at.
            money = next((b for b in ("loop", "loop_secs", "whole wall") if b in bnames),
                         bnames[-1])
            hdr = "".join(f"{b:>17}" for b in bnames)
            print(f"  {'model':<18}{hdr}   {'MiB/pos':>10} {'$/Mpos @ ' + money:>19}"
                  f"{'at the bound':>16}")
            for r in priced:
                e = r["contexts"][ctx]
                cells = e["cards"].get(cardname)
                if not cells:
                    continue
                cs = ""
                for b in bnames:
                    m = cells[b]
                    cs += f"{m['pos_s']:>9.3f}/{m['cards_for_target']:<7}"
                dollars = cells[money]["usd_per_mposition"]
                hi = cells[money].get("usd_per_mposition_bound")
                mib = e["mib_per_token_chunks"]
                print(f"  {r['name']:<18}{cs}   "
                      + (f"{mib:>10.2f} " if mib else f"{'n/a':>10} ")
                      + (f"{dollars:>19,.0f}" if dollars else f"{'n/a':>19}")
                      + (f"{hi:>16,.0f}" if hi else f"{'-':>16}"))
            print("   (each cell is proved positions/s on one card / cards needed for "
                  "the target)")

    # --- MoE and the batch knob ----------------------------------------------
    print("\n(4) MoE against a dense model with the same *active* parameters, C=4K")
    if CELLS != "exact":
        print("  the mechanism (B9 probe 2): a proof covers every expert in full, and "
              "the")
        print("  Freivalds check is per weight -- exactly invariant to how many tokens "
              "used")
        print("  it.  So the weight side is x E while the accumulators are only x k.  B")
        print("  batches sequences into one proof; at B ~ E/k every expert sees S "
              "tokens")
        print("  again and the Freivalds check amortises exactly as a dense matmul's "
              "does")
        print("  at B = 1.")
    else:
        print("  the mechanism, counted rather than asserted: a proof covers every "
              "expert the")
        print("  segment touches, and what it COMMITS for each is that expert's own "
              "matmul over")
        print("  the few tokens routed to it -- which pads -- and the activations its "
              "Freivalds")
        print("  check reads.  The expert count lands on the preprocessed weight "
              "trace, which the")
        print("  measured rates do not count and which the second table below carries.")
    moes = [s for s in priceable if any(grp["experts"] > 1 for grp in s["groups"])]
    BS = (1, 4, 16, 64, 256)
    print("  the reference is the dense twin at B=1, held fixed, so the column reads")
    print("  'what the MoE proof costs against the dense proof the paper already prices'.")
    print(f"  {'model':<18} {'E/k':>6} " + "".join(f"{'B=%d' % b:>10}" for b in BS)
          + f"{'B=E/k':>10}")
    moe_rows = {}
    for s in moes:
        grp = next(grp for grp in s["groups"] if grp["experts"] > 1)
        ek = grp["experts"] / grp["topk"]
        # B = 1 is counted the way the rest of the table is; every B above it is
        # the fitted model, because a batched proof is not one `params` block
        def at(B, ref_batch=1):
            return moe_ratio(s, 4096, alpha, seq=args.seq, seq_batch=B,
                             ref_batch=ref_batch, batch=batch,
                             cells=None if B == 1 else "fitted")
        cells = [at(B) for B in BS]
        same = [at(B, ref_batch=B) for B in BS]
        at_ek = at(max(1, round(ek)))
        moe_rows[s["name"]] = dict(ek=ek, sweep=dict(zip(BS, cells)),
                                   same_b=dict(zip(BS, same)), at_ek=at_ek,
                                   counted=(model_cells(s, 4096, seq=args.seq,
                                                        batch=batch).get("moe")
                                            if CELLS == "exact" else None))
        print(f"  {s['name']:<18} {ek:>6.1f} "
              + "".join(f"{c['ratio']:>9.3f}x" for c in cells)
              + f"{at_ek['ratio']:>9.3f}x")
    if CELLS == "exact":
        print("\n  WHAT A COUNTED MIXTURE SEGMENT IS, and it is not a twin of a dense "
              "one:")
        print("   * the shared experts are a matmul over every token of the block;")
        print("   * each ROUTED expert is a matmul over the tokens routed to it "
              "alone, laid out")
        print("     the way `MatLayout::choose` and `FreivaldsLayout::choose` lay "
              "out that many")
        print("     tokens -- so the row padding a block of a few tokens pays is in "
              "the number, and")
        print("     it is most of why the counted column is above 1.00 at B = 1;")
        print("   * the router's own matmul of `experts` logits is counted like any "
              "other matmul,")
        print("     and only the top-k selection on top of it is a bound;")
        print("   * the preprocessed trace holds the weights of every expert the "
              "segment TOUCHES,")
        print("     which under uniform routing is `E * (1 - (1 - k/E)^S)` and is "
              "not always E.")
        print("  The B = 1 column is counted; every column beside it is the fitted "
              "model, because")
        print("  a proof over B sequences commits B key caches and is not one "
              "`params` block, so")
        print("  the step from the first column to the second is a change of method "
              "as well.")
        for name, v in sorted(moe_rows.items()):
            for k2, d2 in sorted((v.get("counted") or {}).items()):
                uniform = isinstance(d2["distribution"], str)
                how = d2["distribution"] if uniform else "measured"
                # the count the segment was built at, which a measurement replaces
                n = d2["touched_expected"] if uniform else d2["touched"]
                print(f"    {name:<18} {k2}: touches {n:.1f} of "
                      f"{d2['experts']} experts a block, {how} routing, "
                      f"{d2['shared']} shared, expert width {d2['moe_dff']:,} read "
                      f"at {d2['routed_n_in']:,}")
    print("  It crosses 1.00 at B ~ E/k, which is the whole of section 6.4's counter, and")
    print("  keeps going: past B = E/k a batched MoE proof is *cheaper* than the unbatched")
    print("  dense proof the paper prices today, because the Freivalds check it inflated has gone.")
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

    print(f"\n(5) what B buys in absolute terms, C=4K, whole-wall boundary on the "
          f"{priced_card['card']}")
    print(f"  {'model':<18} " + "".join(f"{'B=%d' % b:>14}" for b in (1, 4, 16, 64, 256)))
    batch_rows = {}
    cells_s = priced_card["per_ctx"][4096]["boundaries"]["whole wall"]["cells_s"]
    for s in priceable:
        vals = []
        for B in (1, 4, 16, 64, 256):
            c = sound_cells(s, statement_of(s, 4096, shp), seq_batch=B, batch=batch,
                            alpha=alpha,
                            cells=None if B == 1 else "fitted")["per_position"]
            vals.append(cells_s / c)
        batch_rows[s["name"]] = vals
        print(f"  {s['name']:<18} " + "".join(f"{v:>13.3f}" for v in vals))
    print("  proved positions/s per card; B sequences are proven at once, so this is")
    print("  throughput, not latency, and the wall per proof grows with B.")
    if CELLS == "exact":
        print("  B = 1 is counted and every column beside it is the fitted model, so "
              "the step from")
        print("  the first column to the second is a change of method and not what "
              "batching buys.")

    # --- E11 -----------------------------------------------------------------
    print("\n(6) the full prompt+output statement (E11, decided 2026-08-31)")
    print("  Re-priced by stream E7 on DD's key tile model.  The old numbers in this section")
    print("  came from B9's `326 M + 1.37 M*S + 72.1*S^2` fit, whose S^2 coefficient was")
    print("  +12.6% against the true 64 because a two-point marginal in S alone absorbs the")
    print("  key tile family's linear term into its quadratic one.  A model in S alone also")
    print("  cannot express the term that turns out to matter most: a chunked prompt")
    print("  re-commits the growing K/V prefix, which is a function of C, not S.")
    e11 = None
    mult = consts["e11_multiple"]["value"]
    net = consts["e11_chunking_net"]["value"]
    fixed = consts["e11_chunking_fixed"]["value"]
    if mult is None or net is None or fixed is None:
        print("\n  not recorded.  This section is three numbers and the table that rides on")
        print("  them, and the constants file names none of the three:")
        for alias in ("e11_multiple", "e11_chunking_net", "e11_chunking_fixed"):
            e = consts[alias]
            print(f"    {e['key']:<44} "
                  + ("not recorded" if not e["recorded"] else str(e["value"])))
        print("  No priced row depends on them: section (3)'s rows carry no E11 block and")
        print("  their cards-at-target and dollars-per-token columns are unchanged.")
    else:
        _e11_section(priced, priced_card, usd, mult, net, fixed)
        e11 = dict(multiple_prompt=mult[0], multiple_prompt_output=mult[1],
                   chunking_net_4k=net[0], chunking_net_16k=net[1],
                   chunking_fixed=fixed)
    floor = consts["chunk_floor"]["value"]
    if floor:
        print(f"\n  the shared-table floor of our own composed layer chunk: "
              f"{g(floor)}fresh cells/layer,")
        print("  solved at C=4096 and held out at C=16384 to -0.63% "
              f"(`{OPTIONAL['chunk_floor']}`).")
        print("  It is +14.1% over B9's synthetic-probe floor, which under-predicts our own")
        print("  chunk by 7.15% -- so a one-position eval item costs "
              f"{g(floor * 32)}cells at 8B before")
        print("  it proves anything, which is what makes short multiple-choice proofs dear.")
    else:
        print(f"\n  the composed layer chunk's shared-table floor: not recorded "
              f"(`{OPTIONAL['chunk_floor']}`).")

    # --- the memory wall -----------------------------------------------------
    # One slope over one chunk, and it needs a chunk's own peak beside its own cell
    # count.  No constants key carries a single segment's fresh cells, so this
    # section is the archive's or it is not printed; (7b) below supersedes it and
    # no priced row depends on either.
    ref = arch200["per_ctx"][4096] if arch200 else None
    fits = {}
    if ref is None:
        print("\n(7) does a layer chunk still fit?  not answered here: it needs one "
              "chunk's own peak")
        print("  beside that same chunk's own fresh cells, and the constants file "
              "carries a sequence's")
        print("  cells and a run's peak but not one segment's of each.  Pass "
              "--artifacts DIR, or read (7b).")
    else:
        gib_per_gcell = ref["peak_mib"] / 1024.0 / (ref["peak_chunk_cells"] / 1e9)
        print(f"\n(7) does a layer chunk still fit?  measured: the biggest seq5 chunk is "
              f"{ref['peak_chunk_cells']/1e9:.2f} G cells")
        print(f"  at {ref['peak_mib']:,} MiB of card, i.e. ~{gib_per_gcell:.1f} GiB per G cell "
              "(one point, C=4K, 8B -- approximate)")
        print(f"  {'model':<18} {'G cells/layer chunk':>20} {'GiB implied':>12} "
              f"{'80 GB':>7} {'141 GB':>7}   at S=512")
    for s in (priceable if ref else []):
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
    if ref:
        print("  A chunk that does not fit forces a smaller S, which raises the Freivalds term")
        print("  (~1/S) and multiplies the proof bytes -- the frontier rows above assume the")
        print("  measured S=512 and are optimistic wherever this column says NO.")

    # --- the memory wall, re-priced on stream GG's coefficients --------------
    mc = memory_coeffs()
    mem = None
    if mc:
        GIB = 2 ** 30
        # card sizes.  The budget is the vendor total, as section (7) used it; the
        # measured figure beside it is what `cudaMemGetInfo` leaves a context on our
        # own H100, which is 0.47 GiB tighter.
        CARDS = (("80 GB", 81559 / 1024.0), ("141 GB", 143771 / 1024.0))
        a, b = mc["runs"]
        print("\n(7b) the same question, re-priced with stream GG's coefficients")
        print(f"  read from {a['source']} and {b['source']} "
              f"(H100, 8B, C=4096, S=512, inside the fork's device pool).")
        print("  Four terms, not one slope over fresh cells:")
        print(f"    {mc['cell_bytes']:.0f} B x (fresh + preprocessed cells)   "
              "every tree, base and interaction alike")
        print(f"    {mc['copy_bytes']:.0f} B x device-resident cells of trees 0 and 1"
              "   base cells plus the tables the witness generations clone")
        print(f"    {mc['height_bytes']:.0f} B x 2^(h+1)                      "
              "h = log rows of the tallest column")
        print(f"    + {mc['witness_gen_fixed'] / GIB:.3f} GiB + {mc['witness_gen_h_bytes']:.0f} B x 2^h "
              f"of other witness generation residents, + {mc['residual_bytes'] / 2 ** 20:.2f} MiB "
              "of boundary-tree Merkle")
        print(f"    + {mc['driver_bytes'] / GIB:.3f} GiB the driver holds outside the pool"
              f", + {mc['slack_bytes'] / 2 ** 20:.0f} MiB of pool slack")
        print(f"  the Freivalds check commits {mc['pre_per_weight']:.4f} preprocessed cells per "
              f"covered weight in the composed chunk,")
        print(f"  against B3's standalone-sweep {mc['b3_pre_per_weight']:.4f} that this "
              "module's `pre` column uses")
        print(f"  ({mc['pre_per_weight'] / mc['b3_pre_per_weight'] - 1:+.1%}: the composed "
              "Freivalds check pads each group to a power-of-two row count).")
        print(f"  Held constant when the shape changes, all read at 8B: "
              f"{g(mc['pre_shared_fixed'])} shared")
        print(f"  preprocessed cells (the lookup tables and the per-component columns), "
              f"{g(mc['pre_host_built'])} of")
        print(f"  them host-built, and base = {mc['base_share']:.4f} of the "
              "height-independent trace.")

        print("\n  the gate: the model must reproduce BOTH measured peaks within 5%")
        print(f"  {'chunk':<8} {'h':>3} {'measured pool peak':>19} {'model':>12} "
              f"{'delta':>8}")
        gate = []
        cal8b = next(s for s in SHAPES if s["name"] == "Llama-3.1-8B")
        cal_grp = cal8b["groups"][0]
        cal_fresh = model_cells(dict(cal8b, groups=[cal_grp]), 4096, seq=args.seq,
                                seq_batch=args.seq_batch, alpha=alpha, batch=batch,
                                with_head=False)["total"] * args.seq / cal_grp["layers"]
        cal_w = cm.n_params(cfg_of(cal8b, cal_grp), experts=cal_grp["experts"],
                            topk=cal_grp["topk"], freivalds=True, head=False,
                            attn_params_per_layer=cal_grp.get("attn_params_per_layer")
                            ) / cal_grp["layers"]
        for run, tag in zip(mc["runs"], ("L16", "L0")):
            got = chunk_memory(mc, cal_fresh, cal_w, run["h"])["pool"]
            d = got / run["peak"] - 1.0
            gate.append(dict(chunk=tag, h=run["h"], measured=run["peak"], model=got,
                             delta=d, source=run["source"]))
            print(f"  {tag:<8} {run['h']:>3} {run['peak'] / GIB:>15.3f} GiB "
                  f"{got / GIB:>8.3f} GiB {d:>+8.2%}")
        worst = max(abs(r["delta"]) for r in gate)
        if worst > 0.05:
            sys.exit(f"the memory model misses a measured peak by {worst:.2%} (> 5%): "
                     "it is not calibrated and no frontier row may be printed from it.")
        print(f"  worst {worst:+.2%}, inside the 5% gate.  The residual is the cost "
              "model's own")
        print("  fresh-cell residual showing through; the coefficients themselves "
              "decompose both")
        print("  measured peaks to a few kB (stream GG, `## Record`).")
        head = head_cross_check(mc, ref) if ref else None
        if head:
            print(f"  held out, and in no part of the solve: the `lm_head` chunk the "
                  f"retired slope was")
            print(f"  read from ({rel(head['source'])}, H200, its own cells) prices at "
                  f"{head['model_mib']:,.0f} MiB")
            print(f"  against the {head['measured_mib']:,.0f} MiB seq5 sampled, "
                  f"{head['delta']:+.2%}.  Its Freivalds check commits")
            print(f"  {head['freivalds_pre'] / head['freivalds_weights']:.4f} preprocessed cells "
                  f"per weight against the layer chunk's "
                  f"{mc['pre_per_weight']:.4f}: one wide")
            print("  matrix pads better than a layer's four groups, so the frontier "
                  "rows below,")
            print("  which use the layer figure, are high on the preprocessed term "
                  "rather than low.")

        print(f"\n  one layer chunk at S={args.seq}, C=4096, all [projected]. "
              "h is NOT known for a")
        print("  shape whose keys were never extracted, so both bracket ends are "
              "priced.")
        print(f"  {'model':<18} {'G fresh':>8} {'G pre':>8} {'GiB h=23':>9} "
              f"{'GiB h=25':>9} " + " ".join(f"{c + ' @' + str(h):>10}"
                                             for c, _ in CARDS for h in (23, 25)))
        mem = {}
        for s in priceable:
            best = None
            for grp in s["groups"]:
                fresh = model_cells(dict(s, groups=[grp]), 4096, seq=args.seq,
                                    seq_batch=args.seq_batch, alpha=alpha, batch=batch,
                                    with_head=False)["total"] * args.seq / grp["layers"]
                w = cm.n_params(cfg_of(s, grp), experts=grp["experts"],
                                topk=grp["topk"], freivalds=True, head=False,
                                attn_params_per_layer=grp.get("attn_params_per_layer")
                                ) / grp["layers"]
                at = {h: chunk_memory(mc, fresh, w, h) for h in (23, 25)}
                row = dict(fresh=fresh, weights=w, at=at,
                           freivalds_h=freivalds_height(mc, s, grp))
                if best is None or at[23]["card"] > best["at"][23]["card"]:
                    best = row
            best["fits"] = {name: {h: best["at"][h]["card"] / GIB < gib
                                   for h in (23, 25)} for name, gib in CARDS}
            mem[s["name"]] = best
            cells = " ".join(
                f"{('yes' if best['fits'][c][h] else 'NO'):>10}"
                for c, _ in CARDS for h in (23, 25))
            print(f"  {s['name']:<18} {best['at'][23]['fresh'] / 1e9:>8.2f} "
                  f"{best['at'][23]['pre'] / 1e9:>8.2f} "
                  f"{best['at'][23]['card'] / GIB:>9.1f} "
                  f"{best['at'][25]['card'] / GIB:>9.1f} {cells}")
        tall = [n for n, r in mem.items() if r["freivalds_h"] and r["freivalds_h"] > 25]
        print(f"  budgets: {' and '.join(f'{c} = {gib:.1f} GiB' for c, gib in CARDS)}"
              " (vendor totals); our own H100 leaves a")
        print(f"  context {mc['card_total'] / GIB:.2f} GiB, so read a row inside "
              "0.5 GiB of the 80 GB line as at it.")
        print(f"  tallest Freivalds column over the eight shapes: 2^"
              f"{max(r['freivalds_h'] for r in mem.values())}, so the exponent table "
              "still sets h everywhere")
        print("  " + (f"except {', '.join(tall)}." if tall
                      else "and the h = 23..25 bracket holds for every row."))
        print("  The preprocessed term is per CHUNK, so an MoE layer chunk carries "
              "every expert's")
        print("  Freivalds trace; splitting such a layer into per-expert chunks divides it, "
              "and this")
        print("  table prices section (7)'s layer-chunk convention, not the smallest "
              "feasible chunk.")

    # --- the same question under stream HH's LDE-only residency --------------
    lo = ldeonly_coeffs(mc) if mc else None
    ldeo, wall = None, None
    if lo:
        a, b = lo["runs"]
        ta = lo["terms"]
        wall = ldeonly_wall(lo)
        print("\n(7c) the same question again, with stream HH's LDE-only residency")
        print(f"  read from {a['source']} and {b['source']}")
        print("  (H100, 8B, C=4096, S=512, `ZK_LDE_ONLY=1`, the proof byte-identical "
              "to the off runs).")
        print("  Every committed column is kept on the trace domain and its blown-up")
        print("  evaluations are rebuilt in each phase that reads them, so the 12 B "
              "cell term")
        print(f"  of (7b) becomes {ta['cell_bytes']:.4f} B and the height term changes "
              "shape.  Solved the same way:")
        print("    peak = %.0f B x (fresh + preprocessed) + the copies (7b) prices "
              "+ X(h)" % ta["cell_bytes"])
        print(f"    X(h) = {lo['a_bytes'] / GIB:.4f} GiB + "
              f"{lo['b_bytes']:.2f} B x 2^(h+1)      two points, two unknowns")
        print(f"  measured X: {lo['X'][a['h']] / GIB:.4f} GiB at h={a['h']}, "
              f"{lo['X'][b['h']] / GIB:.4f} GiB at h={b['h']}.")
        print("  What the two are, in the marks.  Read off the runs' own deltas, "
              "per 2^(h+1):")
        print(f"    Merkle {3 * ta['merkle']:.0f} (three trace trees at "
              f"{ta['merkle']:.0f}), twiddles {ta['twiddles']:.0f}, composition "
              f"accumulator {ta['composition']:.0f},")
        print(f"    composition tree {ta['comp_tree']:.0f}   ->  "
              f"{lo['resident_n']:.0f} B resident, plus the transient above "
              f"`composition_commit`,")
        print(f"    which at h={b['h']} is the quotient column and FRI at "
              f"{round(lo['fri_n'])} x 2^(h+1) to "
              f"{b['peak'] - b['marks']['composition_commit']['pool_used'] - round(lo['fri_n']) * 2 ** (b['h'] + 1):,.0f} B."
              f"  Together {lo['height_n']:.0f} B")
        print(f"  against the retained model's {lo['retained_n']:.0f} B (7b): the "
              f"{lo['retained_n'] - lo['height_n']:.0f} B difference is the composition "
              "tree's own")
        print("  blown-up evaluations, the only height-scaled thing the switch drops -- "
              "the cell")
        print("  term carries the rest, which is the whole point of the variant.")
        print(f"  At h={a['h']} the transient is {lo['group_bytes'] / GIB:.4f} GiB, "
              f"x{lo['group_bytes'] / (lo['fri_n'] * 2 ** (a['h'] + 1)):.2f} the FRI's, "
              "and what reconciles the two")
        print("  heights is ONE rebuilt log-size group at 8 B per trace cell, the "
              "transient being")
        print(f"  the LARGER of it and the FRI's: {lo['group_cells'] / 1e6:,.1f} M cells, "
              f"{lo['group_share']:.2%} of the chunk's committed")
        print(f"  cells, wins at h={a['h']} and loses at h={b['h']}.  So `a` is not a "
              "constant and `b` is not")
        print(f"  {lo['height_n']:.0f} B: the fit spreads that one group over the "
              f"interval and reads {lo['a_bytes'] / GIB:.3f} GiB and")
        print(f"  {lo['b_bytes']:.0f} B.  Inside h={a['h']}..{b['h']} it is exact at "
              "both ends, which is where this table prices;")
        print("  outside it, or off this shape, use the group form.  Both reproduce "
              "both measured")
        print("  X: " + ", ".join(
            f"h={h} {v['delta']:+,.0f} B" for h, v in sorted(lo["struct"].items()))
              + " for the group form, exactly for the fit.")

        print("\n  held out, and in no part of the peak solve: the LEVEL marks.  The "
              "coefficients")
        print("  above come from ONE run's commit deltas -- two trees of different "
              "sizes solve the")
        print("  per-cell and the Merkle cost together, the third tree checks them "
              f"({ta['merkle_check']:+,.0f} B; the")
        print("  Merkle reads 64 B less a constant 32 B per tree, which GG's record "
              "also found).")
        print("  Replayed against the other run's own cells they must reproduce every "
              "level it")
        print(f"  reached, at a height they were never read at.  h={b['h']} from the "
              f"h={a['h']} run's deltas:")
        print(f"  {'mark':<20} {'measured':>16} {'model':>16} {'delta':>8}")
        for x in lo["held_out"]:
            if x["h"] == b["h"]:
                print(f"  {x['mark']:<20} {x['measured']:>16,} {x['model']:>16,.0f} "
                      f"{x['delta']:>+8,.0f}")
        print(f"  worst over both directions and all {len(lo['held_out'])} levels: "
              f"{lo['held_worst']:,.0f} B, "
              f"{lo['held_worst'] / lo['runs'][0]['peak']:.1e} of the peak.")

        print("\n  the gate, as (7b) has one: the model must reproduce BOTH of HH's "
              "measured peaks")
        print("  within 5%, at the calibration shape's own modelled cells.")
        print(f"  {'chunk':<8} {'h':>3} {'measured pool peak':>19} {'model':>12} "
              f"{'delta':>8}")
        lgate = []
        for run, tag in zip(lo["runs"], ("L16", "L0")):
            part = mem["Llama-3.1-8B"]["at"][run["h"]]
            got = ldeonly_memory(mc, lo, part, run["h"])
            pool = got["as_built"] - lo["driver_bytes"] - lo["slack_bytes"]
            d = pool / run["peak"] - 1.0
            lgate.append(dict(chunk=tag, h=run["h"], measured=run["peak"], model=pool,
                              delta=d, source=run["source"]))
            print(f"  {tag:<8} {run['h']:>3} {run['peak'] / GIB:>15.3f} GiB "
                  f"{pool / GIB:>8.3f} GiB {d:>+8.2%}")
        lworst = max(abs(r["delta"]) for r in lgate)
        if lworst > 0.05:
            sys.exit(f"the LDE-only model misses a measured peak by {lworst:.2%} "
                     "(> 5%): no frontier row may be printed from it.")
        print(f"  worst {lworst:+.2%}, inside the 5% gate, and the same residual (7b) "
              "carries: the")
        print("  cost model's own fresh-cell figure, not the residency coefficients, "
              "which decompose")
        print("  both peaks to a few hundred bytes.")

        print(f"\n  one layer chunk at S={args.seq}, C=4096, all [projected], three "
              "variants:")
        print("   (i)   as built      -- HH's switch, measured on this card")
        print("   (ii)  copies dropped -- DESIGNED, NOT BUILT: the witness generations' trace-domain "
              "copies are")
        print("         redundant with the coefficients once the tree is committed")
        print("   (iii) host coeffs   -- DESIGNED, NOT BUILT: the coefficients live in "
              "host RAM and")
        print("         are streamed to the card per log-size group, once per rebuild "
              "pass")
        ldeo = {}
        for s in SHAPES:
            r = mem[s["name"]]
            ldeo[s["name"]] = {h: ldeonly_memory(mc, lo, r["at"][h], h) for h in (23, 25)}
            for h in (23, 25):
                ldeo[s["name"]][h]["fits"] = {
                    name: {k: ldeo[s["name"]][h][k] / GIB < gib
                           for k in ("as_built", "no_copies", "host_coeffs",
                                     "as_built_scaled", "no_copies_scaled")}
                    for name, gib in CARDS}
        for key, title in (("as_built", "(i) as built"),
                           ("no_copies", "(ii) copies dropped after commit"),
                           ("host_coeffs", "(iii) host-resident coefficients")):
            print(f"\n  {title}")
            print(f"  {'model':<18} {'GiB h=23':>9} {'GiB h=25':>9} "
                  + " ".join(f"{c + ' @' + str(h):>10}" for c, _ in CARDS for h in (23, 25)))
            for s in SHAPES:
                row = ldeo[s["name"]]
                cells = " ".join(
                    f"{('yes' if row[h]['fits'][name][key] else 'NO'):>10}"
                    for name, _ in CARDS for h in (23, 25))
                print(f"  {s['name']:<18} {row[23][key] / GIB:>9.1f} "
                      f"{row[25][key] / GIB:>9.1f} {cells}")
        print(f"\n  budgets as (7b): {' and '.join(f'{c} = {gib:.1f} GiB' for c, gib in CARDS)}"
              f", each including the {lo['driver_bytes'] / GIB:.3f} GiB")
        print(f"  the driver holds outside the pool and the "
              f"{lo['slack_bytes'] / 2 ** 20:.0f} MiB of pool slack.")
        print("  (i) and (ii) carry the fitted X(h), so their group in flight is the "
              "8B chunk's")
        print("  own; (iii) prices the group explicitly and therefore scales it with "
              "the shape.")
        print("  The group is not derivable per shape from (7b)'s family split, which "
              "carries no")
        print(f"  heights, so it is the measured {lo['group_share']:.2%} share of the "
              "chunk's own committed cells.")
        print("  The exponent table's two key columns at 2^h x 4 B "
              f"({2 * 2 ** 23 * 4 / GIB:.3f} GiB at h=23,")
        print(f"  {2 * 2 ** 25 * 4 / GIB:.3f} GiB at h=25) are a floor on it and are far "
              "below, so they bound nothing useful.")

        print("\n  (iii) also moves the problem to the host and to the bus:")
        print(f"  {'model':<18} {'host GiB':>9} {'150 GiB cgroup':>15} "
              f"{'PCIe s/chunk':>13} {'x3 passes GB':>13}")
        for s in SHAPES:
            row = ldeo[s["name"]][23]
            print(f"  {s['name']:<18} {row['host_ram'] / GIB:>9.1f} "
                  f"{('yes' if row['host_ram'] < H100_HOST_CGROUP else 'NO'):>15} "
                  f"{row['pcie_secs']:>13.2f} {3 * row['host_ram'] / 1e9:>13.1f}")
        print(f"  at {PCIE_PINNED_BPS / 1e9:.2f} GB/s pinned (`briefs/l-tracegen.md`) "
              "and the H100 container's")
        print(f"  {H100_HOST_CGROUP / GIB:.1f} GiB `memory.max` (`briefs/dd-extract.md`); "
              "the host must also hold the witness")
        print("  build, so the cgroup column is a ceiling on the coefficients alone.")

        print("\n  the same (i) and (ii), with the rebuilt group scaled to the shape "
              "instead of held")
        print("  at the 8B chunk's own size inside the fitted `a`.  This is the "
              "conservative reading")
        cal8 = ldeo["Llama-3.1-8B"]
        print("  and it is the one the verdict is stated from.  At the calibration "
              "shape the two")
        print(f"  agree to {max(abs(cal8[h]['as_built_scaled'] - cal8[h]['as_built']) for h in (23, 25)) / GIB:.2f} GiB, "
              "which is this module's own cell residual and not a modelling")
        print("  difference; every wider row is higher under it.")
        print(f"  {'model':<18} {'(i) h=23':>9} {'(i) h=25':>9} {'(ii) h=23':>9} "
              f"{'(ii) h=25':>9} {'80 GB':>10} {'141 GB':>10}")
        moved = []
        for s in SHAPES:
            row = ldeo[s["name"]]
            best = []
            for name, gib in CARDS:
                ok = [k for k, v in (("(i)", "as_built_scaled"), ("(ii)", "no_copies_scaled"))
                      if min(row[23][v], row[25][v]) / GIB < gib]
                best.append(",".join(ok) if ok else "none")
            for key, sc in (("as_built", "as_built_scaled"),
                            ("no_copies", "no_copies_scaled")):
                for name, gib in CARDS:
                    for h in (23, 25):
                        was, now = row[h][key] / GIB < gib, row[h][sc] / GIB < gib
                        if was != now:
                            moved.append(f"{s['name']} {key} {name} @{h} "
                                         f"{'yes->NO' if was else 'NO->yes'}")
            print(f"  {s['name']:<18} {row[23]['as_built_scaled'] / GIB:>9.1f} "
                  f"{row[25]['as_built_scaled'] / GIB:>9.1f} "
                  f"{row[23]['no_copies_scaled'] / GIB:>9.1f} "
                  f"{row[25]['no_copies_scaled'] / GIB:>9.1f} "
                  f"{best[0]:>10} {best[1]:>10}")
        print("  the last two columns name the variants that fit that card at some h; "
              "`none` means")
        print("  neither (i) nor (ii) does, and the row needs (iii) or a smaller chunk.")
        print(f"  {len(moved)} verdict cells move against the fitted table"
              + (", all of them yes->NO:" if all("yes->NO" in m for m in moved)
                 else ":"))
        for i in range(0, len(moved), 2):
            print("    " + "; ".join(moved[i:i + 2]))

        # --- the wall ------------------------------------------------------
        p1, p2 = wall["loops"]["off"], wall["loops"]["on"]
        c16 = wall["per_chunk"][0]
        print("\n  the wall, which is where this variant is paid for.  Three rebuild "
              "passes, and")
        print("  the same three at the commit's own batched rate [projected]:")
        print(f"  {'pass':<22} {'L16 off':>9} {'L16 on':>9} {'delta':>9} {'L0 delta':>9}")
        names = (("constraint_eval", "composition"), ("fri_quotients", "FRI quotients"),
                 ("other", "unattributed rest"))
        for k, lbl in names:
            print(f"  {lbl:<22} {lo['off_runs'][0]['phase'][k]:>9.3f} "
                  f"{lo['runs'][0]['phase'][k]:>9.3f} "
                  f"{wall['per_chunk'][0]['by_pass'][k]:>+9.3f} "
                  f"{wall['per_chunk'][1]['by_pass'][k]:>+9.3f}")
        print("  the composition and the FRI quotients are named phases; the third "
              "pass is the")
        print("  decommitment, which has no counter of its own and shows in the "
              "unattributed rest.")
        for w in wall["per_chunk"]:
            print(f"  {w['chunk']} (h={w['h']}): prove_secs {w['off']:.3f} -> "
                  f"{w['on']:.3f} ({w['delta']:+.3f}), of which the three passes are "
                  f"{w['rebuild']:+.3f};")
            print(f"    `lde_extend`, the commit's own batched extension over the same "
                  f"columns, is {w['batched']:.3f} s")
            print(f"    and does not move between the arms, so a rebuilt pass costs "
                  f"x{w['ratio']:.2f} a built one.")
            print(f"    batched rebuild [projected]: {w['off']:.3f} + 3 x "
                  f"{w['batched']:.3f} = {w['projected']:.3f} s against "
                  f"{w['on']:.3f} as built.")
        print(f"  W's pipelined {p1['n']}-chunk loop at C=4096, from {p1['source']}")
        print(f"  and {p2['source']}:")
        print(f"    generation thread {p1['gen']:.3f} s, prove thread {p1['prove']:.3f} s, "
              f"loop {p1['loop']:.3f} s")
        print(f"    -- generation-bound, with {p1['slack']:.3f} s "
              f"({p1['slack'] / p1['n']:.3f} s per chunk) of slack on the prove thread.")
        print(f"    as built the prove thread is {p2['prove']:.3f} s and the loop "
              f"{p2['loop']:.3f} s, {p2['loop'] / p1['loop'] - 1:+.1%}.")
        print(f"    batched [projected]: the prove thread becomes {p1['projected_prove']:.3f} s, "
              f"{p1['projected_prove'] - p1['prove']:+.3f} s,")
        print(f"    which is {p1['projected_prove'] / p1['n'] - p1['prove'] / p1['n']:+.3f} s "
              f"per chunk against {p1['slack'] / p1['n']:.3f} s of slack, so the loop turns "
              "prove-bound by")
        print(f"    {p1['projected_prove'] - p1['gen']:.3f} s and reads "
              f"{p1['projected_loop']:.3f} s, {p1['projected_delta']:+.2%}.  The "
              "red-team's \"hidden behind")
        print("    generation\" survives only in that weak form, and only if the "
              "batched rebuild is built.")
        big_row = mem["Llama-3.1-405B"]["at"][23]
        big_cells = big_row["fresh"] + big_row["pre"]
        ratio = big_cells / c16["cells"]
        print(f"  scaled to the 405B row by cells ({big_cells / 1e9:.2f} G against "
              f"{c16['cells'] / 1e9:.2f} G, x{ratio:.2f}):")
        print(f"    prove {wall['prove_per_cell'] * 1e9:.4f} ns/cell as built -> "
              f"{wall['prove_per_cell'] * big_cells:.1f} s per chunk, "
              f"{wall['projected_per_cell'] * 1e9:.4f} ns/cell batched -> "
              f"{wall['projected_per_cell'] * big_cells:.1f} s;")
        print(f"    generation {p1['gen_per_cell'] * 1e9:.4f} ns/cell (the loop's own "
              f"rate) -> {p1['gen_per_cell'] * big_cells:.1f} s per chunk.")
        print(f"    so the batched ratio is x{wall['projected_per_cell'] * big_cells / (p1['gen_per_cell'] * big_cells):.3f} "
              f"at 405B against x{p1['projected_prove'] / p1['gen']:.3f} at 8B, and the "
              "as-built ratio")
        print(f"    x{wall['prove_per_cell'] / p1['gen_per_cell']:.3f} against "
              f"x{p2['prove'] / p1['gen']:.3f}: the same ratio holds, because both "
              "threads scale with cells")
        print("    at these two shapes.  The 405B generation rate is the 8B loop's, "
              "extrapolated;")
        print("    no wide witness exists and nothing here measures one.")

    # --- what moved against a previous run -----------------------------------
    if args.compare:
        prev = provenance.rows(json.loads(pathlib.Path(args.compare).read_text()))
        po = {r["name"]: r for r in prev["rows"]}
        print(f"\n(8) what moved against {rel(args.compare)}")
        # A file written before this module priced positions carries only the
        # per-token side, so the comparison falls back to it and says so.
        first = next(iter(po.values()), {}).get("contexts", {}).get("4096", {})
        unit = "cells_per_position" if "cells_per_position" in first else "cells_per_token"
        rate = "pos_s" if unit == "cells_per_position" else "tok_s"
        print(f"  {'model':<18} {'C=4K ' + unit:>26} {'C=16K':>14} "
              f"{'commit ' + rate:>22} {'cards@%g' % target:>12}")
        moved = []
        for r in priced:
            o = po.get(r["name"])
            if not o:
                continue
            a4, b4 = o["contexts"]["4096"], r["contexts"][4096]
            a16, b16 = o["contexts"]["16384"], r["contexts"][16384]
            cardname = next((c for c in b4["cards"] if c in a4["cards"]), None)
            if cardname is None:
                continue
            ca = a4["cards"][cardname]["commit"]
            cb = b4["cards"][cardname]["commit"]
            moved.append((r["name"], ca["cards_for_target"], cb["cards_for_target"]))
            print(f"  {r['name']:<18} {a4[unit]/1e6:10.1f} ->"
                  f"{b4[unit]/1e6:8.1f} {b4[unit]/a4[unit]-1:>+6.1%}"
                  f" {b16[unit]/a16[unit]-1:>13.1%}"
                  f" {ca[rate]:>10.3f} ->{cb[rate]:>8.3f}"
                  f" {ca['cards_for_target']:>7d} ->{cb['cards_for_target']:<4d}")
        def order(rows, key):
            return [n for n, _ in sorted(rows, key=key, reverse=True)]
        cardname = next(iter(priced[0]["contexts"][4096]["cards"]))
        # the ordering is compared over the rows BOTH runs carry: a run that
        # prices a shape the other refused has a longer list, and a longer list
        # is not a row that moved
        both = [r for r in priced if r["name"] in po]
        oa = order([(r["name"], po[r["name"]]) for r in both],
                   lambda kv: kv[1]["contexts"]["4096"]["cards"][cardname]["commit"][rate])
        ob = order([(r["name"], r) for r in both],
                   lambda kv: kv[1]["contexts"][4096]["cards"][cardname]["commit"][rate])
        changed = [f"{n} {a}->{b}" for n, a, b in moved if a != b]
        print(f"  ordering by {cardname} commit {rate} at C=4K: "
              f"{'UNCHANGED' if oa == ob else 'CHANGED'}"
              f" ({len(both)} row(s) in both runs)")
        added = [r["name"] for r in priced if r["name"] not in po]
        if added:
            print("  priced here and not there: " + ", ".join(added))
        print("  cards-at-target changes: " + (", ".join(changed) if changed else "none"))

    # --- what a driver may record --------------------------------------------
    to_record = projected_rows(priced, moe_rows, cal, cards, usd, target, args.seq,
                               args.seq_batch, cells_keys=cells_inputs())
    elsewhere = read_elsewhere()
    # A row may be recorded when every input behind it is a key of this constants
    # file: the cell costs the model charged, the statement's own layout, and the
    # rate it divided by.  One input from anywhere else and the whole run is a
    # projection over a number that is in no repository, so none of it is recorded.
    missing = missing_assumption_keys() if CELLS == "exact" else []
    file_only = not elsewhere and shp["source"] == "constants" and not missing
    for r in to_record:
        r["recordable"] = bool(file_only and r["entry"]["inputs"]
                               and r["from_constants"])
    ok = [r for r in to_record if r["recordable"]]
    rows = to_record
    print(f"\n(9) what this run may record: {len(ok)} of {len(rows)} rows")
    print(f"  {'key':<58} {'value':>14}  unit")
    for r in rows:
        v = r["entry"]["value"]
        print(f"  {r['key']:<58} "
              + (f"{v:>14,.4f}" if isinstance(v, float) else f"{str(v):>14}")
              + f"  {r['entry']['unit']}"
              + ("" if r["recordable"] else "   [not recordable]"))
    recordable = bool(ok) and len(ok) == len(rows)
    if not recordable:
        print("\n  NOT RECORDABLE, and exactly why:")
        for where, items in sorted(elsewhere.items()):
            print(f"    {len(items)} input(s) from {where}: "
                  + ", ".join(i["input"] for i in items))
            for it in items:
                if it["key"]:
                    print(f"      {it['key']} would back {it['input']} and this file "
                          "does not name it")
        if shp["source"] != "constants":
            print("    the statement's own layout is assumed and no key records it")
        if missing:
            print(f"    {len(missing)} assumption key(s) this constants file does "
                  f"not hold yet, which every")
            print("    projected row names in its `inputs`:")
            for k in missing:
                print(f"      {k}")
        print("  The gate counts every cell cost this model can charge, including the "
              "standalone")
        print("  rungs the five reductions replace: it is deliberately over-strict, "
              "because the")
        print("  model does not record which inputs one statement read.  Record those "
              "keys in")
        print("  zk-results and run this again; nothing here is recorded from a "
              "directory that is")
        print("  in no repository.")

    OUT.mkdir(exist_ok=True)
    result = dict(
        constants=consts,
        # Where every input came from, so a reader of this file can tell a
        # number that a run in zk-results backs from one that only a directory
        # outside any repository has.
        provenance=dict(
            constants_file=constants.path(),
            cost_model={k: dict(v) for k, v in cm.PROVENANCE.items()},
            required_keys=dict(KEYS), optional_keys=dict(OPTIONAL),
            not_recorded=[v["key"] for v in consts.values() if not v.get("recorded", True)],
            sequence={f"{name}.c{ctx}": dict(card=card["card"], source=r["source"],
                                             context=ctx, read_from=card.get("source"),
                                             statement=card.get("statement"),
                                             keys=r.get("keys", []))
                      for name, card in cards.items()
                      for ctx, r in sorted(card["per_ctx"].items())}),
        cards={k: v for k, v in cards.items()},
        calibration=cal, alpha=alpha, seq=args.seq, seq_batch=args.seq_batch,
        prefill_fit=fit,
        key_tile_coeffs=dict(fresh=cal["coeffs_fresh"], base=cal["coeffs_base"],
                             measured_base=cal["frozen_base"]),
        shapes=[dict(name=s["name"], tag=s["tag"], source=s["source"],
                     groups=s["groups"], d=s["d"], nh=s["nh"], nkv=s["nkv"],
                     hd=s["hd"], vocab=s["vocab"],
                     params_total=params_of(s, freivalds=True), params_active=params_of(s),
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
        chunk_fit=fits, grid_constant_delta=grid_delta,
        memory=(None if not mc else dict(
            coeffs={k: v for k, v in mc.items() if k != "runs"},
            runs=[{k: v for k, v in r.items() if k != "cells"} | dict(cells=r["cells"])
                  for r in mc["runs"]],
            calibration=gate, head_cross_check=head,
            cards={c: gib for c, gib in CARDS},
            rows={n: dict(fresh=r["at"][23]["fresh"], preprocessed=r["at"][23]["pre"],
                          device_resident=r["at"][23]["dev"], weights=r["weights"],
                          freivalds_log_rows=r["freivalds_h"],
                          at={str(h): dict(pool=r["at"][h]["pool"],
                                           card=r["at"][h]["card"],
                                           cells_bytes=r["at"][h]["cells_bytes"],
                                           copies_bytes=r["at"][h]["copies_bytes"],
                                           height_bytes=r["at"][h]["height_bytes"])
                              for h in (23, 25)},
                          fits={c: {str(h): r["fits"][c][h] for h in (23, 25)}
                                for c, _ in CARDS})
                  for n, r in mem.items()})),
        memory_ldeonly=(None if not lo else dict(
            coeffs={k: v for k, v in lo.items()
                    if k not in ("runs", "off_runs", "held_out")},
            runs=[{k: v for k, v in r.items() if k != "marks"}
                  for r in lo["runs"] + lo["off_runs"]],
            held_out=lo["held_out"], wall=wall,
            cards={c: gib for c, gib in CARDS},
            host_cgroup_bytes=H100_HOST_CGROUP, pcie_bytes_per_s=PCIE_PINNED_BPS,
            rows={n: {str(h): v for h, v in r.items()} for n, r in ldeo.items()})))
    result.update(
        statement=shp, target_positions_per_s=target, projected=rows,
        recordable=recordable, priced_card=priced_card["card"],
        # how the cell side of every row was counted, what it assumed where a
        # `params` block carries a field a model does not, and what it refused
        cells=CELLS,
        assumptions=(None if CELLS != "exact" else dict(
            assumption_rows(llama8b, seq=args.seq),
            key=ASSUMPTION_KEY, modes={k: {f: list(v) for f, v in m.items()}
                                       for k, m in assumptions.items()},
            bound=ec.witness_bounds(), block=cal.get("block"),
            analogy=[dict(input=n, key=e["key"], value=e["value"])
                     for n, e in sorted(cm.PROVENANCE.items())
                     if e["source"] == "analogy"])),
        refused={n: w for n, w in refusals.items() if w})
    doc = dict(
        # The block every run JSON in this project carries, so that a driver in
        # zk-results can write a manifest from this file the way it writes one from
        # a prover output.  This run makes no proof and touches no card: it reads a
        # constants file and prints arithmetic, so it says `harness`, it names no
        # GPU even where the machine has one, and it binds nothing.
        provenance=provenance.block(
            model=None, device="none", threads=None, module=__spec__.name,
            gpu=None, driver_version=None, cuda_version=None,
            constants_file=constants.path(),
            artifacts_dir=None if ROOT is None else str(ROOT),
            inputs_dir=None if cm.OUT is None else str(cm.OUT),
            statement="sound" if shp["source"] == "constants" else "assumed",
            recordable=recordable),
        result=result)
    pathlib.Path(args.json).write_text(json.dumps(doc, indent=1, default=str))
    print(f"\nwrote {rel(args.json)}")
    return doc


if __name__ == "__main__":
    main()
