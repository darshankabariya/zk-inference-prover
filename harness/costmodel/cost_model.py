"""B2 cost model: measured stwo constants -> per-token prover budget, checked
against a published 53 tok/s target for a Gemma-3-27B-class model.

Nothing measured lives in this repository.  Every constant below is read from a
zk-results `constants.json` (`--constants PATH`) where a key names it, and from
the micro-benchmark run outputs in `--inputs DIR` (or `ZK_COSTMODEL_INPUTS`)
where no key does yet; `PROVENANCE` records which source each one came from and
says `absent` for the ones neither has.  Both directories are optional: without
`--inputs` the model keeps every cell cost a key backs and charges nothing for
the rest, and the rates the four bench files carry print `not measured here`.
The CPU->H100 speedup is never assumed: the model reports the trace
commitment throughput a prover would need for 53 tok/s and how far one CPU is
from it. Closing that gap is B7's job.

    uv run python -m harness.costmodel.cost_model --constants PATH --inputs DIR
"""
import json
import pathlib

from harness import constants, inputs
from harness.costmodel.key_tile import (  # noqa: F401  (re-exported)
    CHAN_TARGET, GRID_TARGET, ROW_WIDTH, key_tile_cells, key_tile_family,
    key_tile_fresh)

# The micro-benchmark outputs this model reads, all under one directory: the
# stwo benches at its root and the witness-side probes in named subdirectories.
# The directory is optional, so every reader below tests its own file first and
# every quantity it would have produced can be `None`.
OUT = inputs.costmodel_inputs(optional=True)


def _sub(name):
    return None if OUT is None else OUT / name


B1, B3, B4, B5, B6, B65, B10 = (_sub(n) for n in
                                ("b1", "b3", "b4", "b5", "b6", "b65", "b10"))


def load(name):
    if OUT is None:
        return None
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------- where each input came from -----
# `PROVENANCE[name]` is one input's `value`, the `source` it was read from
# (`constants`, `inputs`, `stand-in`, `analogy` or `absent`), the constants
# `key` that backs it or `None` where no key does yet, and the `file` under
# `--inputs DIR` that the fallback read.  A caller writes the whole record into
# its own output, so a reader of a projection can see which numbers came from a
# run in zk-results and which from a directory that is not in any repository.
PROVENANCE = {}


def cell(name, key, measured=None, file=None, note="", charged=True):
    """One cell-cost input, constants first: the value the constants file holds
    under `key`, the micro-benchmark outputs' `measured` where the file has no
    such key, and None where neither has it.

    `charged=False` marks an input the model reads for a comparison it prints and
    does not multiply by a count, so a caller that refuses to price a term it
    cannot price knows not to count this one."""
    e = constants.find(key) if key else None
    if e is not None:
        PROVENANCE[name] = dict(value=e["value"], source="constants", key=key,
                                grade=e.get("grade"), run=e.get("run"), file=file,
                                note=note or e.get("note", ""), charged=charged)
        return e["value"]
    PROVENANCE[name] = dict(value=measured, key=key, file=file, note=note, charged=charged,
                            source="inputs" if measured is not None else "absent")
    return measured


commit = load("commit")
commit_host = load("commit_host")
field = load("field")
logup = load("logup")
fri = load("fri")

ENV = commit["env"] if commit else None
SEC = commit["security"] if commit else None


def wide(run):
    """Points with enough columns that Merkle's per-row cost has amortised."""
    return [dict(p, env=run["env"]["os"]) for p in run["result"]["points"] if p["n_cols"] >= 16]


def peak(pts):
    return max(pts, key=lambda p: p["base_cells_per_sec"]["total"])


def largest(pts):
    return max(pts, key=lambda p: p["base_cells"])


def rate(kernel, where="all_threads"):
    if not field:
        return None
    for k in field["result"][where]:
        if k["kernel"] == kernel:
            return k["ops_per_sec"]
    raise KeyError(kernel)


# The four rates below are the only quantities the bench files at the root of
# `--inputs DIR` carry, and nothing but this module's own report reads them: no
# constants key backs any of them, and without the directory each is None and
# the report says `not measured here`.
#
# Headline constants come from the containerised run only, so the whole model
# sits in one environment. The host run (bigger traces than the 8 GiB VM can
# hold) is reported separately below.
CP = peak(wide(commit)) if commit else None
COMMIT_CELLS_S = CP["base_cells_per_sec"]["total"] if CP else None

# End-to-end prover throughput: a whole proof, not just the commit path. Still
# climbing at the largest shape measured, so this is a floor, not a plateau.
fri_cols = load("fri_cols")
_e2e = [p for r in (fri, fri_cols) if r for p in r["result"]["points"]]
E2E = max(_e2e, key=lambda p: p["trace_cells_per_sec"]) if _e2e else None
E2E_CELLS_S = E2E["trace_cells_per_sec"] if E2E else None
M31_MUL_S = rate("m31_mul_simd")
QM31_MUL_S = rate("qm31_mul_simd")

# The LogUp lookup-batch size the model charges: 2, the largest a blowup of 2
# allows, since a batch of b needs degree-(b+1) constraints. That is a design
# point of the proof system and not a measurement.
LOGUP_BATCH = 2

# LogUp marginal cost: total cells per lookup minus the 2 (in, out) trace cells,
# which a witness has to commit anyway. Measured, per batch size.
LOGUP_MARGINAL = {}
LOGUP_NS = {}
if logup:
    for p in logup["result"]["points"]:
        LOGUP_MARGINAL[p["logup_batch_size"]] = p["cells"]["per_lookup"] - 2
    LOGUP_NS = {
        b: min(p["prove_secs"] / p["n_lookups"] * 1e9
               for p in logup["result"]["points"] if p["logup_batch_size"] == b)
        for b in LOGUP_MARGINAL
    }
_logup_marginal = cell("logup_marginal_cells_per_lookup",
                       "model_constants.logup_marginal_cells_per_lookup",
                       LOGUP_MARGINAL.get(LOGUP_BATCH), "logup.json")
if _logup_marginal is not None:
    LOGUP_MARGINAL[LOGUP_BATCH] = _logup_marginal

# --- B3's two missing terms, measured ----------------------------------------
# (1) the r^T W Freivalds check. Its three QM31 accumulator columns cost 12 M31
#     cells per trace row, and a row carries `p` weights, so the check commits
#     12/p fresh
#     cells per weight -- a knob, not a constant. B3's width sweep is read here
#     and the design point is the one that minimises fresh cells subject to the
#     proof staying small (b3/README.md section 2).
# (2) the byte-range check on every committed int8 activation, which B2's model
#     charged nothing for: one LogUp lookup per activation cell.
def b3_constants():
    f = B3 and B3 / "width.json"
    if not (f and f.exists()):
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
        # cells of *preprocessed* trace the check reads per proof, per weight
        preprocessed_per_weight=best["cells"]["preprocessed_per_weight"],
    )


B3C = b3_constants()
WEIGHT_FREIVALDS_CELLS_PER_WEIGHT = cell(
    "weight_freivalds_cells_per_weight", "model_constants.weight_freivalds_cells_per_weight",
    B3C["cells_per_weight"] if B3C else None, "b3/width.json") or 0.0
# The preprocessed side of the same layout choice.  No key names it yet, so this
# one term still forces the inputs directory even when every other cell cost is
# backed by a constant.
PREPROCESSED_PER_WEIGHT = cell(
    "preprocessed_cells_per_weight", None,
    B3C["preprocessed_per_weight"] if B3C else None, "b3/width.json") or 0.0


# --- B4's terms, measured -----------------------------------------------------
# (1) the dequant's limb decomposition. B3 proves qY only modulo p; reading an
#     accumulator as an integer costs one committed limb plus two range-check
#     lookups per accumulator, in the cheapest *sound* variant (limb bound 2^30,
#     below p, with the sign absorbed by a 2^29 offset).
# (2) the int8 range check, again: B3 spent one lookup per activation, B4 checks
#     two activations with one arity-2 lookup into a 2^16 pair table.
def b4_constants():
    if B4 is None:
        return None
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
LIMB_CELLS_PER_ACC = cell("limb_cells_per_acc", None,
                          B4C["cells_per_acc"] if B4C else None, "b4/limbs.json",
                          charged=False) or 0.0
# the packed pair table's range check on every committed int8 activation
RANGE_CELLS_PER_ACTIVATION = cell("int8_range_cells_per_activation", None,
                                  B4C["range_marginal"] if B4C else None, "b4/range.json")


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
    f = B5 and B5 / "block.json"
    if not (f and f.exists()):
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
#     instances share one column set, so the whole-model proof is key tiles x one
#     key tile's columns, with popcount(L*nh) key tiles.
def b6_constants():
    if B6 is None:
        return None
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
        # the normalisation alone -- `y = bf16(acc)`, no scale.  B10's bf16 add
        # reuses exactly this half, so the model reads it rather than guessing.
        round_away=by.get(("round", "away")),
        # the cheaper alternative: power-of-two scales, one rounding, +2.29% ppl
        pow2=by.get(("pow2", "even")),
        pow2_away=by.get(("pow2", "away")),
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
    if B65 is None:
        return None
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
# the requantize with its scale argument, less the committed quantized value
# `cells_per_token` already counts on every distinct matmul input
_requant = cell("requant_cells_per_value", "model_constants.requant_cells_per_value",
                B65C.get("requant_max") if B65C else None, "b65/requant_*.json")
REQUANT_CELLS = (_requant - 1.0) if _requant else 0.0
# B10 step 4: greedy decoding's argmax is B6.5's **max argument** run over the
# vocabulary instead of over a matmul input's row -- the same gadget, the same
# per-value price.  `requant_max - requant_only` is what the argument costs on
# top of the requantize it was measured inside (b65/out/requant_*.json): one
# 15-bit range lookup on `pm - pat_i`, one selector bit, and the constraint
# `sel_i (pm - pat_i) = 0` that pins the maximum to an attained element.
#
# What step 4 adds on top of the gadget is the *binding*: `sum_i sel_i * i` against
# the committed next token, which is one accumulator over a column the argument
# already commits and costs no cell per vocabulary entry.  So the whole token binding
# is priced by this one number.
#
# Two numbers price this gadget and they disagree, so the model carries both.
# The B6.5 *inference* above (3.36) reads the argument's price off a requantize
# it was measured inside; the gadget B10 actually built and proved is dumped in
# `b10/out/argmax_v8k.json` and costs more.  The projection uses the built one:
# an inference from a neighbouring gadget is not a measurement of this one.
def b10_argmax_constants():
    """The built argmax gadget, per comparison, from `b10/out/argmax_v8k.json`."""
    f = B10 and B10 / "argmax_v8k.json"
    if not (f and f.exists()):
        return None
    try:
        res = json.loads(f.read_text())["result"]
    except (ValueError, KeyError):
        return None
    # the strict point; `lax` drops a constraint and is not what the model prices
    pts = [p for p in res["points"] if not p.get("lax")]
    if not pts:
        return None
    p = pts[0]
    cmps = p["vocab"] * p["tokens"]
    # the gadget's own fixed side: 10 QM31 interaction columns at its 2^13 rows,
    # i.e. the table half of its LogUp batches, which does not grow with the
    # vocabulary.  This one shape cannot *prove* that split -- it is the residual
    # that reconciles the file's whole-gadget figure with the marginal the B10
    # review quotes -- and `argmax_v16k.json` / `argmax_vqwen.json` would pin it.
    fixed = 10 * 4 * (1 << p["layout"]["log_rows"])
    return dict(total=p["cells_per_comparison"],
                base=p["base_per_comparison"],
                marginal=(p["cells"]["fresh"] - fixed) / cmps,
                fixed=fixed, comparisons=cmps, vocab=p["vocab"])


B10A = b10_argmax_constants()
MAXARG_CELLS_B65 = ((B65C["requant_max"] - B65C["requant_only"])
                    if B65C and B65C.get("requant_max") else 0.0)
# `model_constants.argmax_cells_per_comparison` does NOT back this: that key is
# the whole gadget over its comparisons, the fixed table half included, and the
# marginal below subtracts that half using the run's own row count, which the
# key does not carry.  Recording the marginal beside it would back this input.
MAXARG_CELLS_BUILT = cell("argmax_marginal_cells_per_comparison", None,
                          B10A["marginal"] if B10A else None, "b10/argmax_v8k.json",
                          "the whole-gadget key is not this quantity") or 0.0
MAXARG_CELLS = MAXARG_CELLS_BUILT or MAXARG_CELLS_B65
# E1 / B9 probe 2 (`b9/moe.py`, and `b9/out/moe.json`'s `routing`): the MoE
# routing-selection argument, priced **by analogy** because nobody has built it.
# No AIR in this project bindings a committed row to a routed instance, so neither
# number below is a measurement of this gadget -- each is the nearest gadget that
# was measured, and the model prints them as such.
#   (a) selection: a LogUp multiset tying each token's FFN rows to the instance of
#       the expert it was routed to -- B4's packed pair table at 2 cells/lookup
#       plus the committed expert id.
#   (b) top-k: an argmax over the router's `E` logits per token -- B6.5's max
#       argument, `requant_max - requant_only`.
ROUTE_SEL_CELLS = 3.0
ROUTE_MAX_CELLS = MAXARG_CELLS_B65 or 3.36
for _n, _v in (("route_selection_cells", ROUTE_SEL_CELLS),
               ("route_topk_cells", ROUTE_MAX_CELLS)):
    PROVENANCE[_n] = dict(value=_v, source="analogy", key=None, file=None, charged=True,
                          note="nobody has built this gadget; the nearest gadget that "
                               "was measured stands in for it")
# `dequant` is the per-accumulator cost of reading the integer *and* rounding it;
# it subsumes B4's limb term, so `cells_per_token` charges this instead of that.
# The two keys are measured with the round half to even; the model charges the
# rounding away from zero on every layer matmul, and no key names that one, so
# where the inputs directory is missing the even figure stands in and the model
# is one cell an accumulator high.
DEQUANT_CELLS_PER_ACC = cell("dequant_chain_cells_per_acc",
                             "model_constants.dequant_chain_cells_per_acc",
                             (B6C or {}).get("chain"), "b6/dequant.json",
                             charged=False) or 0.0
DEQUANT_AWAY_PER_ACC = cell("dequant_chain_away_cells_per_acc", None,
                            (B6C or {}).get("chain_away"), "b6/dequant.json") or 0.0
DEQUANT_POW2_PER_ACC = cell("dequant_pow2_cells_per_acc",
                            "model_constants.dequant_pow2_cells_per_acc",
                            (B6C or {}).get("pow2"), "b6/dequant.json",
                            charged=False) or 0.0
DEQUANT_POW2_AWAY_PER_ACC = cell("dequant_pow2_away_cells_per_acc", None,
                                 (B6C or {}).get("pow2_away"), "b6/dequant.json") or 0.0


def _stands_in(name, value, said):
    """Nothing names this input, so a neighbouring one is charged instead; the
    record says which, because the difference is a real cost and not a rounding.
    A stand-in that is itself absent is absent, not a zero."""
    PROVENANCE[name].update(value=value or None, note=said,
                            source="stand-in" if value else "absent")
    return value


if not DEQUANT_AWAY_PER_ACC:
    DEQUANT_AWAY_PER_ACC = _stands_in(
        "dequant_chain_away_cells_per_acc", DEQUANT_CELLS_PER_ACC,
        "the round-half-to-even chain stands in, one cell an accumulator high")
if not DEQUANT_POW2_AWAY_PER_ACC:
    DEQUANT_POW2_AWAY_PER_ACC = _stands_in(
        "dequant_pow2_away_cells_per_acc", DEQUANT_POW2_PER_ACC,
        "the round-half-to-even power-of-two figure stands in, one cell high")
ATTNOUT_CELLS_PER_ACC = cell("attnout_int8_cells_per_acc",
                             "model_constants.attnout_int8_cells_per_acc",
                             (B6C or {}).get("int8"), "b6/dequant.json") or 0.0

# --- B10 step 2: the residual stream between the components ---------------------------------
# B6.5 open gap 1: the composition bindings every matmul's accumulators to its own
# committed activations, and nothing bindings one component's output to the next
# component's input.  The wiring is four gadgets, and every one of them is bf16
# elementwise arithmetic:
#
#   (a) RMSNorm      square (unary), the sum over `d`, `* 1/d`, `+ eps`,
#                    `rsqrt` (unary), `* r`, `* w`
#   (b) SwiGLU       one `silu` unary table and one bf16 product per `dff` channel
#   (c) two residual adds, `d` bf16 adds each
#   (d) the requantize's placement -- B6.5's gadget, already charged as `requant`;
#       wiring it between a dequant's output and a Freivalds check's input adds
#       no cell.
#
# Three primitives price all four, and two of the three are already measured:
#
#   * a **unary** bf16 op is exactly a 2^16-entry table -- B4's `silu`/`exp`
#     points, 4.00 cells per application packed 4+ per row;
#   * a **bf16 multiply** is B6's mantissa-product table plus an exponent add:
#     the chain dequant is the pow2 one plus exactly two of them, so the model
#     reads (chain - pow2) / 2 rather than assuming a number;
#   * a **bf16 add** is the gadget B4 section 9 scoped and refused to improvise
#     (it guessed 12-15 cells).  B10 builds it: the operands are ordered by
#     magnitude pattern (B6.5's max argument, one 15-bit range lookup and a
#     selector), aligned by the exponent difference into one integer, and that
#     integer is normalised by the *same* `NORM`/`FRAC` tables the dequant
#     already commits -- so the add is B6's `round` variant plus an alignment.
#     `b65/out/residual_stream.json` carries the measured marginal; until it exists the
#     model prints the construction's own column count and says so.  Both the add
#     and the multiply range-check their own output against `VAL` (README 3.7's
#     named gap: without it the gadget can prove a value bf16 cannot represent),
#     which is +2 cells on each and is in the measured numbers below.
def b10_residual_stream_constants():
    f = B65 and B65 / "residual_stream.json"
    if not (f and f.exists()):
        return None
    try:
        pts = json.loads(f.read_text())["result"]["points"]
    except (ValueError, KeyError):
        return None
    res = json.loads(f.read_text())["result"]
    prim = res.get("primitives", {})
    return dict(
        add=prim.get("add"), mul=prim.get("mul"), unary=prim.get("unary"),
        value=prim.get("value", 2.0),
        drift=res.get("norm_reduction_drift"),
        by_gadget={p["gadget"]: p for p in pts},
    )


B10G = b10_residual_stream_constants()
# the bf16 multiply, read off B6's ladder: `chain` is `pow2` plus two of them
_mul_ladder = ((DEQUANT_AWAY_PER_ACC - DEQUANT_POW2_AWAY_PER_ACC) / 2.0
               if DEQUANT_AWAY_PER_ACC and DEQUANT_POW2_AWAY_PER_ACC else 0.0)
MUL_CELLS = cell("residual_stream_mul_cells_per_op",
                 "model_constants.residual_stream_mul_cells_per_op",
                 (B10G or {}).get("mul") or _mul_ladder or None,
                 "b65/residual_stream.json") or 0.0
UNARY_CELLS = cell("unary_cells_per_application", None,
                   (B10G or {}).get("unary") or (B4C or {}).get("lookup_cells") or None,
                   "b4/unary.json") or 0.0
# the add, from its construction: B6's `round` variant (the normalisation, one
# `NORM` and one `FRAC` lookup, 11.06 cells/away) minus the committed integer it
# no longer needs (the aligned sum is an expression, not a column), plus the
# alignment -- the order selector, the 15-bit magnitude compare, the committed
# `2^Delta` and its pair-table lookup, and the far-branch selector and range
# lookup that make the gadget complete when the exponents are further apart than
# the round bit can see.
ROUND_CELLS = (B6C or {}).get("round_away") or 0.0
# `round/away` is 8 committed columns (acc, sg, m2, ph, r, m7, ef, nz) and two
# lookups, so the LogUp share a gadget of this shape actually realises is
# (11.06 - 8) / 2 = 1.53 cells per lookup -- the number to price the add's extra
# lookups with, not the standalone 2.00.
ROUND_COLS = 8.0
LOGUP_IN_GADGET = (ROUND_CELLS - ROUND_COLS) / 2.0 if ROUND_CELLS else 0.0
# the add against `round`: -1 column (the aligned sum is an expression, not a
# committed integer), +3 columns (the order selector, the committed 2^Delta, the
# far-branch selector) and +3 lookups (the 15-bit magnitude compare that orders
# the operands, the (Delta, 2^Delta) pair table, and the far-branch range check)
ADD_CELLS_MODEL = ((ROUND_CELLS - 1.0) + 3.0 + 3.0 * LOGUP_IN_GADGET
                   if ROUND_CELLS else None)
ADD_CELLS = cell("residual_stream_add_cells_per_op",
                 "model_constants.residual_stream_add_cells_per_op",
                 (B10G or {}).get("add"), "b65/residual_stream.json")
if ADD_CELLS is None and ADD_CELLS_MODEL is not None:
    ADD_CELLS = _stands_in("residual_stream_add_cells_per_op", ADD_CELLS_MODEL,
                           "nothing measures the add; the construction's own column "
                           "and lookup count stands in")
ADD_CELLS = ADD_CELLS or 0.0

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
# B10 step 4: at the spec's *rounding*, not at ties-to-even.  `b10-design` rounds
# away, and B6 measured every rung at both modes -- the power-of-two one is
# **14.0625** away against 15.0625 even, exactly as the chain is 25.125 against
# 26.125.  Charging the head at the even figure was a cell per accumulator too
# many, and on the `lm_head` chunk that is 65.7 M cells at 8B.  Confirmed on the real
# head accumulators, `b10/out/head_deq_*.json`.
HEAD_DEQUANT_PER_ACC = DEQUANT_POW2_AWAY_PER_ACC

# --- the numerator width, and what the AIR as built can carry ------------------
# B5's AIR range-checks V into [-128, 127] and the committed denominator Z into
# `z_lo_bits + z_hi_bits` (b5/README section 4; `attn_air.rs` RowEval), so the
# circuit certifies |A.V| < 128 * 2^z_bits.  The numerator width sets Z's blind
# worst case (`nmax * C`) and therefore both bounds.  Two ceilings sit above it:
#   * 2^29 -- B4's *offset* limb variant, whose 2^30 window is split by a +2^29
#     offset and so carries only 2^29 of magnitude;
#   * 2^30 -- M31's own injectivity edge, floor(p/2) = 2^30 - 1, which B6's chain
#     and the attention-output fold both reach because each commits an explicit
#     sign bit and reads the magnitude (`freivalds.rs`: `mag = acc - 2 acc sg`).
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
ALPHA = cell("attention_cells_per_head_key_token",
             "model_constants.attention_cells_per_head_key_token",
             B5C["intexp"]["alpha_grid"] if B5C and "intexp" in B5C else None,
             "b5/block.json") or 5.0
ALPHA_V1 = (B5C["bf16x"]["alpha_grid"] if B5C and "bf16x" in B5C else 5.0)

# --- the attention key tile family, as a closed form in (S, C) ---------------------
# `key_tile.py`, imported above.  It lives in its own module because its
# self-test -- the form against every recorded layer segment, to the cell -- has
# to run where this file's micro-benchmark inputs do not exist:
#
#     uv run python -m harness.costmodel.key_tile --selftest
#
# `key_tile_family` is what a composed segment costs and what `cells_per_token`
# charges; `key_tile_cells` is one key tile of it, and a caller that hands it the
# whole context gets one untiled prefix key tile and not the family.


# ---------------------------------------------------------------- the model --
# Committed cells per decoded token, in the shape B1's witness actually has.
#   acc   one int32 accumulator per matmul output              (B1: 45.5% of cells)
#   act   one int8 value per *distinct* matmul input           (B1: 24.6%)
#   nl    bf16 lookup in/out for SiLU and the two norms        (B1: 25.5%)
#   attn  3 columns per (layer, head, key): score, exp, prob   (B1: alpha = 3-4)
#   logup the LogUp interaction column each lookup drags along (B2, measured)


# --- B10 step 2: the residual stream, priced ---------------------------------------------
# The reduction is the whole question.  B1 accumulates the RMSNorm mean in fp32
# and rounds once (`reduce="fp32"`), an order no bf16-add gadget reproduces, so
# proving the norm *pins the reduction's order* -- a spec change with a quality
# row behind it, and B1 already carries the tree (`reduce="bf16-norm"`).  All
# three candidate orders are the same `n - 1` correctly-rounded adds and so cost
# the same cells; they differ in how many chained components the AIR needs and in
# how far the result can drift from the exact sum.  A chained component is a running
# sum with a reset selector (the requantize's `nofirst`/`last` shape), so every
# order below is mask-addressed -- no wiring lookup, which is what a general tree
# would cost (+3 lookups per node).
def sum_order(n, order):
    pad = 1 << (n - 1).bit_length()
    if order == "tree":            # B1's `tree_sum`: pad with +0, log2 levels
        # `x + 0` is exact, so the AIR prunes every slot the pad would fill and
        # proves `n - 1` adds however far `pad` reaches past `n` (B10 step 2);
        # the *depth*, and so the drift, is still the padded tree's
        return dict(adds=n - 1, components=pad.bit_length() - 1, err_ulps=pad.bit_length() - 1)
    if order == "two-level":       # blocks of ~sqrt(n), then the block sums
        b = 1 << ((n - 1).bit_length() // 2)
        c = -(-n // b)
        return dict(adds=n - 1, components=2, err_ulps=b + c - 2)
    if order == "running":         # one component, one chain
        return dict(adds=n - 1, components=1, err_ulps=n - 1)
    raise SystemExit(f"unknown sum order {order}")


def residual_stream_cells(cfg, order="tree", topk=1, final_norm=True):
    """B6.5 open gap 1, per decoded token: what the wiring between the components
    costs, per gadget.  `d`-wide terms are the residual stream's width, `dff`-wide
    the SwiGLU's.

    `topk` (E1): with a top-k MoE router a token passes through `topk` experts, so
    the SwiGLU is `topk` times as wide.  The norms and the residual adds are on the
    residual stream and do not move.  `final_norm=False` drops the model-level norm
    so that a two-group model (dense layers + MoE layers) can be summed without
    counting it twice."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    dff = topk * dff
    so = sum_order(d, order)
    # (a) RMSNorm, twice per layer plus the final one.  Per channel: the square
    # (one unary table), its share of the reduction, and the two multiplies
    # (`x * r`, then `* w`).  Per token: `* 1/d`, `+ eps`, `rsqrt`.
    # The composed AIR merges the reduction's first levels with the residual add
    # into one component per `(residue, token)` row, so the stream is committed once
    # and read by both -- which is a saving on the *leaves*, not on the ops, and
    # the table below counts ops.
    norms = 2 * L + (1 if final_norm else 0)
    sq = norms * d * UNARY_CELLS
    red = norms * so["adds"] * ADD_CELLS
    scale = norms * 2 * d * MUL_CELLS
    scalars = norms * (MUL_CELLS + ADD_CELLS + UNARY_CELLS)
    # (b) SwiGLU: one `silu` table and one bf16 product per dff channel
    swiglu = L * dff * (UNARY_CELLS + MUL_CELLS)
    # (c) the two residual adds
    resid = L * 2 * d * ADD_CELLS
    # (d) the requantize's placement.  B6.5 priced the gadget and `cells_per_token`
    # already charges it on every distinct matmul input; wiring it between a
    # dequant's output and a Freivalds check's input is AIR work at zero marginal
    # cell.
    requant_place = 0.0
    total = sq + red + scale + scalars + swiglu + resid + requant_place
    # what the stub this replaces charged: B1's witness columns for the two norms
    # and the SiLU, plus their lookups' LogUp share
    stub = (L * (2 * dff + 4 * d)
            + (L * (2 * dff + 2 * (2 * d + 1))) * LOGUP_MARGINAL[min(LOGUP_MARGINAL)])
    return dict(order=order, adds=so["adds"], components=so["components"],
                err_ulps=so["err_ulps"],
                norm_square=sq, norm_reduce=red, norm_scale=scale,
                norm_scalars=scalars, norm=sq + red + scale + scalars,
                swiglu=swiglu, resid=resid, requant_place=requant_place,
                total=total, stub=stub, net=total - stub)


def cells_per_token(cfg, ctx, batch, seq=None, alpha=None, dequant=None,
                    head_dequant=None, experts=1, topk=1, seq_batch=1,
                    head=True, attn_accs_per_layer=None,
                    attn_params_per_layer=None, diagonal_block=True):
    """Committed (fresh) cells per decoded token.

    `batch` is the **LogUp lookup-batch size** -- how many lookups share one
    interaction column, measured in `b2/out/logup.json`.  It is NOT a sequence
    batch.  E1's sequence-batch knob is `seq_batch`, below; the two are unrelated
    and the names must stay apart (E1/E6a).

    E1's three additions:

      * `experts` / `topk` -- a mixture of experts.  A proof covers **every** expert
        the chunk's tokens touch, in full, because Freivalds' weight-side cost is
        per weight and exactly invariant to how many tokens used it (B9 probe 2:
        218.63 M preprocessed cells for the 8B FFN whether the expert served 512
        tokens or 64).  So the FFN's *weight* terms scale with `experts` while its
        *per-token* terms scale with `topk`, and against a dense model with the
        same active parameters the weight side is x experts/topk.
      * `seq_batch` (B) -- how many decoded **sequences** one proof covers.  The
        per-proof terms (the Freivalds check, the preprocessed read) amortise over
        `seq * seq_batch` tokens rather than `seq`.  This is the MoE counter: at
        `B ~ experts/topk` each expert sees `seq` tokens again, so its Freivalds
        check is amortised exactly as a dense matmul's is at B = 1.
      * `attn_accs_per_layer` / `attn_params_per_layer` -- overrides for
        attention shapes the GQA formulas do not describe (MLA's q/kv low-rank
        pair, which commits a different accumulator count and a different weight
        count from `(nh + 2 nkv) hd`).

    `head=False` drops the model-level terms (the `lm_head`'s accumulators, its
    lookups and the token binding) so that a model whose layers are not all alike can
    be summed group by group without counting the head more than once.
    """
    L, d, nh, nkv, hd, dff, vocab = cfg
    if not head:
        vocab = 0
    # the FFN, per token: `topk` experts of width `dff` are actually evaluated
    dff_active = topk * dff
    tokens = (seq * seq_batch) if seq else None
    attn_accs = (attn_accs_per_layer if attn_accs_per_layer is not None
                 else (nh + 2 * nkv) * hd + d)
    acc_layer = L * (attn_accs + 2 * dff_active + topk * d)
    acc = acc_layer + vocab
    act = L * (2 * d + nh * hd + dff_active) + d
    nl = L * (2 * dff_active + 4 * d)
    # The attention key tile family.  Two regimes, and the default is the key tile model.
    #
    # `alpha=<x>` keeps the LEGACY single-slope term, `x * L * nh * C`, which is
    # what every figure before 2026-09-01 used.  It is retained only so the two
    # can be printed side by side: a single cells-per-(head, key, token) number is
    # *not* a price per grid entry.  Stream E7 showed the composed chunk's slope
    # decomposes as 5.898 = 4.375 (the S*C term, per entry) + 1.5234 (the C-only
    # K/V commitment, divided by S), so any single slope is only valid at the S it
    # was solved at -- S=512 here -- and mis-scales in `hd`, `nh` and C together.
    #
    # The key tile model charges the two separately and subsumes what used to be three
    # terms: `attn`, `kv` (the committed KV cache and queries) and `softmax_rows`.
    # Charging any of those beside it double-counts.
    legacy_alpha = alpha is not None
    if legacy_alpha:
        attn = alpha * L * nh * ctx
        key_tile_saturated = False
    else:
        # the whole family of one layer chunk: the prefix tiled into key blocks of
        # `seq` keys, the diagonal block, and stream NN's root-equality claims.  The
        # LogUp running sums are charged, which is the prover's default; a run with
        # `--gkr-key-tiles` moves them into a family this model does not carry.
        per_layer, key_tile_saturated = key_tile_fresh(nh, hd, seq or 1, ctx, diagonal_block=diagonal_block)
        attn = L * per_layer / (seq or 1)
    n_lookups = L * (2 * dff_active + 2 * (2 * d + 1)) + vocab
    logup = n_lookups * LOGUP_MARGINAL[batch]
    # B3: the weight Freivalds check's accumulators, and a range check per int8
    # activation.  The check is a per-*proof* cost, so it divides by the tokens
    # the proof covers
    # -- `seq * seq_batch`, not `seq`.
    params = n_params(cfg, experts=experts, freivalds=True, head=head,
                      attn_params_per_layer=attn_params_per_layer)
    freivalds = WEIGHT_FREIVALDS_CELLS_PER_WEIGHT * params / tokens if seq else 0.0
    # B4: the packed pair table halves B3's per-activation range check
    rng_cells = (RANGE_CELLS_PER_ACTIVATION if RANGE_CELLS_PER_ACTIVATION is not None
                 else LOGUP_MARGINAL[batch])
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
    requant = (L * (2 * d + dff_active) + d) * REQUANT_CELLS if seq else 0.0
    # B10 step 4: the token binding.  Proving that the emitted token *is* the argmax
    # means showing no other logit is larger, which is one comparison per
    # vocabulary entry per token -- the max argument's own per-value price.  The
    # `lm_head` Freivalds check and its dequant are already charged (`freivalds`
    # counts `vocab * d` weights through `n_params`, `limbs` counts `vocab`
    # accumulators); this is the term neither of them covers.
    argmax = vocab * MAXARG_CELLS if seq else 0.0
    # B5: the committed quantized KV cache and queries, and their range checks
    # (2 cells for int11/int12 K and Q, 1 for a pair-packed int8 V), plus the
    # per-token softmax scalars (max, denominator, its limb and two lookups)
    # the KV cache is written once per token and read by every later token in the
    # chunk, so it amortises over the chunk's own tokens (`seq`) and not over the
    # sequences a batched proof carries -- each sequence has its own cache.
    if legacy_alpha:
        kv_vals = 2.0 * L * nkv * hd * ctx / seq if seq else 0.0
        q_vals = L * nh * hd
        kv = kv_vals + q_vals + (1.5 * kv_vals + 2.0 * q_vals if seq else 0.0)
        softmax_rows = 5.0 * L * nh
    else:
        # both are inside `attn` now: the key tile family's `C` term IS the committed
        # K and V (`Cells::kv`) with their range checks, its `S` term carries the
        # queries, and its per-token row component carries the max, the denominator
        # and their two limbs.
        kv = softmax_rows = 0.0
    # E1 / B9 probe 2: the routing-selection gadget, priced by analogy and built by
    # nobody.  Per token per layer: one LogUp multiset entry tying the token's rows
    # to each chosen expert's instance (`ROUTE_SEL_CELLS`), and an argmax over the
    # router's `experts` logits (`ROUTE_MAX_CELLS`, B6.5's max argument).
    route = (L * (topk * ROUTE_SEL_CELLS + experts * ROUTE_MAX_CELLS)
             if experts > 1 else 0.0)
    total = (acc + act + nl + attn + logup + freivalds + rng + limbs + kv + softmax_rows
             + attn_out + requant + argmax + route)
    return dict(acc=acc, act=act, nl=nl, attn=attn, logup=logup, n_lookups=n_lookups,
                freivalds=freivalds, range=rng, limbs=limbs, kv=kv, softmax_rows=softmax_rows,
                attn_out=attn_out, requant=requant, argmax=argmax, route=route,
                key_tiles=attn, key_tile_saturated=key_tile_saturated, legacy_alpha=legacy_alpha,
                dequant_per_acc=per_acc,
                head_per_acc=head_acc, n_acc=acc, n_acc_layer=acc_layer,
                flat=acc + act + nl, total=total, freivalds_params=params,
                tokens_per_proof=tokens,
                # committed offline, but read once per proof by the Freivalds check
                preprocessed=PREPROCESSED_PER_WEIGHT * params / tokens if seq else 0.0)


def lm_head_chunk(cfg, seq, seq_batch=1):
    """B10 step 4: what the `lm_head` chunk costs, split the way it is built.

    One chunk after the final norm, covering `seq` decoded tokens:

      * the **Freivalds check** over `vocab x d` weights -- once per chunk, not
        per token, which is the whole reason the head is its own chunk;
      * the **dequant** of `vocab x seq` accumulators, at whichever rung the head
        ends up on -- `HEAD_DEQUANT_PER_ACC`, a knob, not a settled number.  B8's
        attribution puts the head on `Pow2` and every layer matmul on the chain;
        the dumps declare something else and disagree with each other
        (`b9/out/prove8b.json` has q/k/v/o on `pow2` and gate/up/down on `chain`,
        `b10/out/loop32_*.json` has all seven on `chain`), and neither declares
        the head at all -- it is not in either chunk.  So this docstring asserts
        no rung: the spec decision is pending, and until it lands the number here
        is whatever `HEAD_DEQUANT_PER_ACC` is set to.  The whole span is one cell
        per accumulator between the two `Pow2` roundings and 11.06 between `Pow2`
        and the chain -- at 27B, 0.26 M and 2.90 Mcells/token respectively;
      * the **argmax**, one comparison per vocabulary entry per token, at the
        built gadget's measured marginal (`MAXARG_CELLS`);
      * the **token binding** itself, `sum_i sel_i * i` against the committed next
        token -- one accumulator over a column the argmax already commits, so it
        is free per entry and is not a line here.

    Returned per chunk, not per token: the head is amortised over `seq` and
    reading it per token is how the Freivalds term gets mis-stated.
    """
    _, d, _, _, _, _, vocab = cfg
    weights = vocab * d
    accs = vocab * seq * seq_batch
    freivalds = weights * WEIGHT_FREIVALDS_CELLS_PER_WEIGHT
    pre = weights * PREPROCESSED_PER_WEIGHT
    deq = accs * HEAD_DEQUANT_PER_ACC
    arg = accs * MAXARG_CELLS
    return dict(weights=weights, accumulators=accs, freivalds=freivalds, preprocessed=pre,
                dequant=deq, argmax=arg, fresh=freivalds + deq + arg,
                per_token=(freivalds + deq + arg) / (seq * seq_batch))


def n_params(cfg, experts=1, topk=1, freivalds=False, head=True,
             attn_params_per_layer=None):
    """Weights in the shape.  `freivalds=True` returns what one *proof* covers -- every
    expert in full, because the Freivalds check is per weight and does not amortise
    over the tokens that used it (B9 probe 2).  `freivalds=False` returns the **active**
    parameters, `topk` experts per token, which is the number a serving system
    quotes and the one a like-for-like dense comparison uses."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    attn = (attn_params_per_layer if attn_params_per_layer is not None
            else d * nh * hd + 2 * d * nkv * hd + nh * hd * d)
    n_ffn = experts if freivalds else topk
    return (vocab * d if head else 0) + L * (attn + n_ffn * 3 * d * dff)


def ops_per_token(cfg, ctx, seq, experts=1, topk=1, seq_batch=1):
    """In-circuit M31 multiplies, the E6b Freivalds model (freivalds.py)."""
    L, d, nh, nkv, hd, dff, vocab = cfg
    params = n_params(cfg, experts=experts, freivalds=True)
    dff_active = topk * dff
    mats = [(d, (nh + 2 * nkv) * hd), (nh * hd, d), (d, 2 * dff_active), (dff_active, d)]
    io = sum(a + b for a, b in mats)
    ext = 4  # QM31 = 4 x M31
    weight_pass = ext * params / (seq * seq_batch)
    per_tok_io = ext * L * io
    attn = 3 * ext * L * nh * ctx
    lookups = L * nh * ctx + L * (2 * dff_active + 2 * (2 * d + 1)) + vocab
    return dict(total=weight_pass + per_tok_io + attn + lookups, weight_pass=weight_pass,
                params=params, naive_cells=params + L * 2 * nh * hd * ctx)


# the LogUp lookup-batch size the model charges: the design point above where it
# is priced, and the cheapest measured batching otherwise.  This is the `batch`
# argument of `cells_per_token`, and it has nothing to do with batching
# *sequences* into one proof, which is `seq_batch` (E1/E6a).
BATCH = (LOGUP_BATCH if LOGUP_BATCH in LOGUP_MARGINAL
         else min(LOGUP_MARGINAL, key=LOGUP_MARGINAL.get) if LOGUP_MARGINAL
         else LOGUP_BATCH)

QWEN = (24, 896, 14, 2, 64, 4864, 151936)      # B1's model
LLAMA8B = (32, 4096, 32, 8, 128, 14336, 128256)  # B8/B9/B10's rung
GEMMA = (62, 5376, 32, 16, 128, 21504, 262144)  # the 27B proxy shape
TARGET_TOKS, TARGET_CTX, SEQ = 53.0, 16384, 512


def g(x, unit=""):
    if x is None:
        return "not measured here "
    for s, d in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= d:
            return f"{x / d:.2f} {s}{unit}"
    return f"{x:.2f} {unit}"


def _source_line(name):
    """`value (where it came from)` for one recorded input, for the report."""
    e = PROVENANCE[name]
    where = (e["key"] if e["source"] == "constants"
             else f"{e['source']}, {e['file']}" if e["file"] else e["source"])
    return f"{'not priced' if e['value'] is None else e['value']} ({where})"


def report():
    """The B2 script's own report.  Guarded so that `frontier.py` (E1) and any
    other caller can import this module for its constants and its model without
    printing 200 lines first."""
    print("=" * 78)
    print("B2 cost model - measured stwo constants -> per-token prover budget")
    print("=" * 78)
    if ENV:
        print(f"platform: {ENV['arch']}/{ENV['os']}, {ENV['threads']} threads, "
              f"{ENV['simd_n_lanes']} SIMD lanes")
        print(f"security: blowup {SEC['blowup_factor']}, {SEC['n_queries']} FRI queries, "
              f"{SEC['pow_bits']} PoW bits -> {SEC['security_bits_conjectured']} bits "
              f"conjectured, {SEC['hash']}")
    else:
        print("platform and security: not measured here -- pass --inputs DIR for the four "
              "bench files")
    print()
    print("measured constants")
    if CP:
        print(f"  trace commit  2^{CP['log_size']} rows x {CP['n_cols']} cols   "
              f"{g(COMMIT_CELLS_S)}cells/s   (commit path only, peak over the wide traces)")
    else:
        print("  trace commit  not measured here (commit.json); no constants key backs it")
    if E2E:
        print(f"  whole proof   2^{E2E['log_size']} rows x {E2E['n_cols']} cols   "
              f"{g(E2E_CELLS_S)}cells/s   (prove() end to end, still rising with size)")
    else:
        print("  whole proof   not measured here (fri.json); no constants key backs it")
    if M31_MUL_S:
        print(f"  M31 mul       SIMD, {ENV['threads']} threads              {g(M31_MUL_S)}op/s")
        print(f"  QM31 mul      SIMD, {ENV['threads']} threads              {g(QM31_MUL_S)}op/s")
    else:
        print("  field rates   not measured here (field.json); no constants key backs them")
    for b in sorted(LOGUP_MARGINAL):
        ns = (f", {LOGUP_NS[b]:.0f} ns/lookup" if b in LOGUP_NS
              else ", time not measured here")
        table = (f" into 2^{logup['result']['log_table']} table" if logup else "")
        print(f"  LogUp batch {b}{table}      "
              f"{LOGUP_MARGINAL[b]:.0f} extra cells/lookup{ns}")
    if not LOGUP_MARGINAL:
        print("  LogUp         NOT MEASURED -- no logup.json and no "
              "`model_constants.logup_marginal_cells_per_lookup`; nothing below can be "
              "charged for a lookup")
    if B3C:
        print(f"  Freivalds     B3, {B3C['weights_per_row']} weights/row     "
              f"{WEIGHT_FREIVALDS_CELLS_PER_WEIGHT:.3f} fresh cells/weight "
              f"(12/p; {PREPROCESSED_PER_WEIGHT:.2f} preprocessed cells/weight read per proof)")
    else:
        print(f"  Freivalds     {_source_line('weight_freivalds_cells_per_weight')} "
              "fresh cells/weight")
        print(f"                {_source_line('preprocessed_cells_per_weight')} "
              "preprocessed cells/weight read per proof")
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
        print(f"  int8 range    {_source_line('int8_range_cells_per_activation')} "
              "cells/activation")
        print(f"  bf16 lookup   {_source_line('unary_cells_per_application')} "
              "cells/application")
    if B5C:
        for v, c in sorted(B5C.items()):
            print(f"  attention     B5, {v:<8}            "
                  f"{c['alpha_grid']:.2f} cells/(head,key,token) on the grid, "
                  f"{c['alpha_marginal']:.2f} marginal (one head, S={c['s']}, hd={c['hd']})")
    else:
        print(f"  attention     {_source_line('attention_cells_per_head_key_token')} "
              "cells/(head,key,token) on the grid -- the superseded single-slope form")
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
        for _label, _n in (("dequant chain", "dequant_chain_cells_per_acc"),
                           ("dequant pow2 ", "dequant_pow2_cells_per_acc"),
                           ("attn output  ", "attnout_int8_cells_per_acc")):
            print(f"  {_label} {_source_line(_n)} cells/accumulator")
    print()

    # --- validate the cells model against B1's actual witness ---------------------
    man = B1 and B1 / "manifest.json"
    if man and man.exists():
        w = json.loads(man.read_text())["witness"]
        steps = json.loads(man.read_text())["sequence"]["steps"]
        kinds = w["values_by_kind"]
        b1_flat = sum(v for k, v in kinds.items() if k != "attention") / steps
        model_flat = cells_per_token(QWEN, 0, 1)["flat"]
        print(f"model check vs B1's witness (Qwen2.5-0.5B, context-independent cells/token):")
        print(f"  B1 measured {b1_flat:,.0f}   model {model_flat:,.0f}   "
              f"ratio {model_flat / b1_flat:.3f}")
        print()

    print(f"Gemma-3-27B proxy, S={SEQ}, LogUp batch {BATCH} -- the largest blowup 2 allows, "
          f"since batch b needs degree-(b+1) constraints")
    for ctx in (4096, TARGET_CTX):
        c = cells_per_token(GEMMA, ctx, BATCH, SEQ)
        o = ops_per_token(GEMMA, ctx, SEQ)
        print()
        print(f"  ctx={ctx}")
        print(f"    committed cells/token  {g(c['total'])}  "
              f"(attention {c['attn'] / c['total'] * 100:.0f}%, "
              f"LogUp columns {c['logup'] / c['total'] * 100:.0f}%, "
              f"layer I/O {c['flat'] / c['total'] * 100:.0f}%, "
              f"weight Freivalds check {c['freivalds'] / c['total'] * 100:.0f}%, "
              f"int8 range {c['range'] / c['total'] * 100:.0f}%, "
              f"dequant {c['limbs'] / c['total'] * 100:.0f}%, "
              f"requant {c['requant'] / c['total'] * 100:.0f}%, "
              f"KV cache {c['kv'] / c['total'] * 100:.0f}%)")
        print(f"    + preprocessed         {g(c['preprocessed'])}cells/token of weight trace, "
              f"committed offline but read once per proof (B3)")
        print(f"    in-circuit M31 mul/tok {g(o['total'])}  "
              f"(weight pass {o['weight_pass'] / o['total'] * 100:.0f}%)")
        if COMMIT_CELLS_S and E2E_CELLS_S and M31_MUL_S:
            t_field = o["total"] / M31_MUL_S
            t_commit = c["total"] / COMMIT_CELLS_S + t_field
            t_e2e = c["total"] / E2E_CELLS_S + t_field
            print(f"    on this host           commit-limited {1 / t_commit:5.2f} tok/s"
                  f"   |  measured end-to-end rate {1 / t_e2e:5.2f} tok/s")
        else:
            print("    on this host           not measured here: the four bench files at "
                  "the root of --inputs DIR carry the only rates")

    c = cells_per_token(GEMMA, TARGET_CTX, BATCH, SEQ)
    o = ops_per_token(GEMMA, TARGET_CTX, SEQ)
    need_cells = c["total"] * TARGET_TOKS
    need_ops = o["total"] * TARGET_TOKS
    print()
    print(f"what {TARGET_TOKS:.0f} tok/s at ctx={TARGET_CTX} requires:")
    print(f"  trace commitment   {g(need_cells)}cells/s")
    print(f"    vs commit path only   {g(COMMIT_CELLS_S)}cells/s"
          + (f"  ->  x{need_cells / COMMIT_CELLS_S:,.0f} short" if COMMIT_CELLS_S else ""))
    print(f"    vs whole-proof rate   {g(E2E_CELLS_S)}cells/s"
          + (f"  ->  x{need_cells / E2E_CELLS_S:,.0f} short" if E2E_CELLS_S else ""))
    print(f"  in-circuit field   {g(need_ops)}M31 mul/s  "
          f"measured {g(M31_MUL_S)}op/s"
          + ("" if not M31_MUL_S else
             f"  ->  x{need_ops / M31_MUL_S:,.1f} short" if need_ops > M31_MUL_S
             else f"  ->  already {M31_MUL_S / need_ops:,.1f}x headroom"))
    print(f"  commit-every-MAC   {g(o['naive_cells'] * TARGET_TOKS)}cells/s "
          f"= {o['naive_cells'] * TARGET_TOKS * 4 / 1e12:,.1f} TB/s of trace  ->  impossible")
    if B3C:
        print()
        print(f"B3's weight Freivalds check, folded in "
              f"(measured on {B3C['shape']} at S={B3C['s']}):")
        print(f"  {WEIGHT_FREIVALDS_CELLS_PER_WEIGHT:.3f} fresh cells/weight at "
              f"{B3C['weights_per_row']} weights/row  ->  "
              f"{g(c['freivalds'])}cells/token, {c['freivalds'] / c['total'] * 100:.1f}% of the total")
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
              f"{'stands' if WEIGHT_FREIVALDS_CELLS_PER_WEIGHT <= 0.25 else 'DOES NOT stand'}")
        print(f"  on top, {g(c['preprocessed'])}cells/token of preprocessed weight trace are "
              f"read per proof")
        print(f"  ({(c['preprocessed'] + c['total']) / c['total'] - 1:.0%} more cells to check than "
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
              f"layer costs +3.67% perplexity at 8B, so there is one layout, not a key tile")
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

    # --- B10 step 2: the residual stream's price table ------------------------
    print()
    print("B10 step 2 -- the residual stream between the components (B6.5 open gap 1), priced")
    _add_src = PROVENANCE["residual_stream_add_cells_per_op"]
    SRC = (f"{_add_src['key']}, measured" if _add_src["source"] == "constants"
           else "b65/out/residual_stream.json, measured" if _add_src["source"] == "inputs"
           else "from the construction")
    print(f"  bf16 unary (2^16 table)  {UNARY_CELLS:.2f} cells/application  ({SRC}; B4's "
          f"packed {(B4C or {}).get('lookup_cells', 0):.2f} in a pattern layout)")
    print(f"  bf16 multiply            {MUL_CELLS:.2f} cells/product       ({SRC}; B6's "
          f"ladder implies (chain {DEQUANT_AWAY_PER_ACC:.2f} - pow2 "
          f"{DEQUANT_POW2_AWAY_PER_ACC:.2f}) / 2 = "
          f"{(DEQUANT_AWAY_PER_ACC - DEQUANT_POW2_AWAY_PER_ACC) / 2:.2f})")
    _says = (f"the construction says {ADD_CELLS_MODEL:.2f}, " if ADD_CELLS_MODEL
             else "the construction is not priced here, ")
    print(f"  bf16 add                 {ADD_CELLS:.2f} cells/add           ({SRC}; "
          f"{_says}B4 section 9 guessed 12-15 and built none)")
    if B10G and B10G.get("value"):
        print(f"  one committed bf16 value {B10G['value']:.2f} cells                (`sm`, `ef`: "
              "two columns, which is what keeps every constraint degree 2)")
    print()
    print("  the RMSNorm reduction: three orders, the same adds, different shapes")
    print(f"  {'order':<11} {'adds (d=4096)':>14} {'components':>8} {'worst drift':>12}")
    for o in ("tree", "two-level", "running"):
        so = sum_order(4096, o)
        print(f"  {o:<11} {so['adds']:>14,} {so['components']:>8} "
              f"{so['err_ulps'] / 512.0:>11.1%}")
    print("  (drift = worst-case relative error of the reduction, err_ulps x 2^-9; B1's")
    print("   `reduce=\"bf16-norm\"` is the tree, and B1 scores it at 22.758 against the")
    print("   base spec's 22.835 at 0.5B -- the fp32 sum `b10-design` specifies is not")
    print("   reproducible by any bf16-add gadget, so proving the norm pins the order)")
    for dr in (B10G or {}).get("drift") or []:
        print(f"  measured on Qwen-0.5B layer 12, {dr['norm']}, d={dr['d']}: of {dr['tokens']} "
              f"real tokens the tree's `ms + eps`")
        print(f"  is bit-identical to B1's fp32 reduction on {dr['identical']} and differs on "
              f"{dr['differ']}, by at most")
        print(f"  {dr['worst_ulps']} ulp ({dr['worst_relative']:.2%} relative) -- the whole "
              "numeric content of the spec change;")
        print(f"  and the composed two-level components reproduce `tree_sum`'s own root on "
              f"{dr['two_level_matches_tree_sum']} of {dr['two_level_checked']}")
    print()
    CHUNKS = {}
    if B10 and (B10 / "prove8b.json").exists():
        try:
            pt = json.loads((B10 / "prove8b.json").read_text())["result"]["point"]
            CHUNKS["Llama-3.1-8B"] = (pt["cells"]["fresh"], pt["params"]["s"], pt["params"]["c"])
        except (ValueError, KeyError):
            pass
    if B65C and B65C.get("compose"):
        CHUNKS["Qwen2.5-0.5B"] = (B65C["compose"]["fresh"], SEQ, 1024)
    RESIDUAL_STREAM_MODELS = [("Qwen2.5-0.5B", QWEN), ("Llama-3.1-8B", LLAMA8B), ("Gemma-3-27B", GEMMA)]
    print("  cells per decoded token, per gadget (the tree order), and per layer chunk")
    print(f"  {'model':<13} {'gadget':<21} {'cells/token':>12} {'per layer':>11} "
          f"{'chunk S=512':>12} {'% of chunk':>10}")
    for name, cfg in RESIDUAL_STREAM_MODELS:
        gl = residual_stream_cells(cfg, "tree")
        L, d, dff = cfg[0], cfg[1], cfg[5]
        chunk = CHUNKS.get(name)
        rows = [("(a) RMSNorm x2", gl["norm"]), ("      of it, the sum", gl["norm_reduce"]),
                ("(b) SwiGLU", gl["swiglu"]), ("(c) residual adds", gl["resid"]),
                ("(d) requantize place", gl["requant_place"]),
                ("residual stream, total", gl["total"]), ("the stub it replaces", gl["stub"]),
                ("residual stream, net", gl["net"])]
        print(f"  {name:<13} {f'L={L} d={d} dff={dff}':<21}"
              + (f" {'':>12} {'':>11} {g(chunk[0]):>12}   the chunk, S={chunk[1]} C={chunk[2]}"
                 if chunk else ""))
        for label, v in rows:
            per_layer = v / L
            per_chunk = per_layer * SEQ
            pc = f"{per_chunk / chunk[0] * 100:9.1f}%" if chunk else f"{'--':>10}"
            print(f"  {'':<13} {label:<21} {g(v):>12} {g(per_layer):>11} "
                  f"{g(per_chunk):>12} {pc:>10}")
    gl27 = residual_stream_cells(GEMMA, "tree")
    print()
    print(f"  at 27B the residual stream is {g(gl27['net'])}cells/token net of the stub, i.e. "
          f"{gl27['net'] / c['total']:+.1%} on {g(c['total'])} -> {g(c['total'] + gl27['net'])}, "
          f"and {TARGET_TOKS:.0f} tok/s needs "
          f"{g((c['total'] + gl27['net']) * TARGET_TOKS)}cells/s")
    print("  the requantize's placement is the one gadget with no cell of its own: "
          f"{g(c['requant'])}cells/token")
    print(f"  are already charged for it ({REQUANT_CELLS:.2f}/value on every distinct matmul "
          "input), and wiring it")
    print("  between a dequant's output and a Freivalds check's input is AIR work, "
          "not trace")

    # --- B10 step 4: the `lm_head` chunk and the token binding -------------------------
    print()
    print("B10 step 4 -- the `lm_head` chunk, one per 512-token continuation:")
    print(f"  the argmax is priced at {MAXARG_CELLS:.3f} cells/comparison -- the gadget B10 "
          f"built and proved")
    if B10A:
        print(f"  (b10/out/argmax_v8k.json, vocab {B10A['vocab']:,}: {B10A['total']:.3f} "
              f"fresh cells/comparison whole,")
        print(f"   {MAXARG_CELLS_BUILT:.3f} net of the gadget's own fixed table side, "
              f"{B10A['base']:.3f} of it base)")
    print(f"  the other candidate, not used: {MAXARG_CELLS_B65:.3f} cells/comparison, inferred "
          f"from B6.5's")
    print("  requant_max - requant_only (b65/out/requant_*.json) -- the same *shape* of gadget,"
          " run over")
    print("  the vocabulary instead of a matmul input's row, but not this build of it")
    print("  (the binding `sum_i sel_i * i` rides free on a column the argument already commits)")
    print()
    print(f"  {'model':<10} {'vocab':>9} {'freivalds':>10} {'dequant':>10} {'argmax':>10} "
          f"{'fresh':>10} {'preproc':>10}")
    for name, cfg in (("Qwen-0.5B", QWEN), ("Llama-8B", LLAMA8B), ("Gemma-27B", GEMMA)):
        h = lm_head_chunk(cfg, SEQ)
        print(f"  {name:<10} {cfg[6]:>9,} {g(h['freivalds']):>10} {g(h['dequant']):>10} "
              f"{g(h['argmax']):>10} {g(h['fresh']):>10} {g(h['preprocessed']):>10}")
    h8 = lm_head_chunk(LLAMA8B, SEQ)
    print(f"  at 8B the dequant is {h8['dequant'] / h8['fresh']:.0%} of the chunk and the "
          f"argmax {h8['argmax'] / h8['fresh']:.0%};")
    print("  comparing the *accumulators* before the dequant would remove most of the "
          "first, and is")
    print("  the plan's own rider -- it is sound only where the per-channel scales "
          "are shared, and")
    print("  B1 decodes on the rounded logits, so it has to match those.")
    hg = lm_head_chunk(GEMMA, SEQ)
    print(f"  against the per-token budget: the head is {g(hg['per_token'])}cells/token at "
          f"27B, {hg['per_token'] / c['total']:.1%} of {g(c['total'])}")

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
    if fri:
        print("proof size / verification, measured at blowup 2 and "
              f"{SEC['n_queries']} queries:")
        for p in fri["result"]["points"]:
            print(f"  2^{p['log_size']} rows x {p['n_cols']} cols   "
                  f"{p['proof_bytes'] / 1024:6.1f} KiB   verify "
                  f"{p['verify_secs'] * 1e3:5.2f} ms   "
                  f"{p['n_fri_query_positions']} query positions, "
                  f"{p['n_fri_inner_layers']} FRI layers")
    else:
        print("proof size / verification: not measured here (fri.json)")


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



if __name__ == "__main__":
    report()
