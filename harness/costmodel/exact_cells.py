"""A composed segment's committed cells, counted from its `params` block alone.

    uv run python -m harness.costmodel.exact_cells --selftest

`cost_model.py` prices a segment from per-unit constants read off measured runs
-- cells per accumulator, per weight, per lookup -- and the frontier calibrates
the total against the recorded sequence.  This module does not: it rebuilds the
trace layout the prover builds, component by component, and multiplies columns
by rows.  Nothing here is fitted and nothing here is measured; every line is an
identity of `prover/src`, named in the comment above it.

`key_tile.py` solved three pieces of this already: the attention key tile
family, the boundary trees and the lookup tables.  The boundary form is
imported from it unchanged.  The other two are rebuilt here -- the tables
because the set a segment registers and the height of each is derivable rather
than a fixed list, and the key tiles because `key_tile.py`'s closed form holds
only where the three grids tile exactly, which the output head's C = 3,584 does
not.  What is new is everything else: the dequantize components, the attention
output, the Freivalds checks, the residual stream, the rotation's quantize
components, the requantize seams, their own tables, the GKR link, the argmax
and the embedding gadget.  That is the whole chunk, which is why `--selftest`
compares against `cells.base`, `cells.interaction` and `cells.fresh` rather
than against one family.

WHAT IT IS EXACT IN.  A `params` block, not an architecture.  Some of its
fields are not the model's shape at all: they are the calibration a run took
off its own witness (`kmax`, `log_table`, `ao_jo_min/max`, `ao_jv_min/max`,
`ao_acc_hi_bits`, `ao_acc_lo_bits`, `ao_rem_bits`, `ao_q_bits`, `fold_alpha`,
`fold_rem_bits`), and three groups of them move the count: see `witness_terms`.
A caller who has a model and a context but no witness has to assume them, and
the largest of the three -- the master exponent table at `2^log_table` -- is
between 1.5 and 11 per cent of a recorded chunk.  `head_groups` is a run knob
rather than a shape, and it decides the grids' heights: a `params` block
without it is refused rather than guessed at, and so is the `bf16x` attention
variant, which no recorded segment proves.

    exact_cells(params) -> {"base": ..., "interaction": ..., "boundary": ...,
                            "fresh": ..., "components": {name: (base, inter)}}
"""
import argparse
import json
import pathlib
import sys

from harness.costmodel.key_tile import boundary_cells

HERE = pathlib.Path(__file__).parent
CHUNKS = HERE / "exact_chunks.json"

# stwo's SimdBackend needs N_LANES = 16 rows (`common.rs`, `MIN_LOG_ROWS`).
MIN_LOG_ROWS = 4
# A QM31 column is four M31 cells: every interaction column below is counted in
# QM31 and multiplied by this at the end (`plogup.rs`, `devfrac.rs`).
M31_PER_QM31 = 4


def log_rows_for(n, per_row=1):
    """`common.rs`, `log_rows_for`: rows for `n` items at `per_row` each, at or
    above the SIMD floor."""
    rows = max(1, -(-n // max(1, per_row)))
    return max(MIN_LOG_ROWS, (rows - 1).bit_length())


# ------------------------------------------------------------- the layouts --
# Every family below tiles its values into rows the same way: a width that
# divides the value count, a row count rounded up to a power of two, and the
# slack carried as inert lead rows.  The four choosers differ only in what they
# score and what they are allowed to pad.


def mat_layout(n_out, tokens, target):
    """`layer.rs`, `MatLayout::choose_or_padded`: the exact tiling when one
    exists, the padded one otherwise.  Returns (per_row, chunks, log_rows)."""
    best = None
    for pr in range(1, min(n_out, 4 * max(target, 1)) + 1):
        if n_out % pr:
            continue
        chunks = n_out // pr
        rows = tokens * chunks
        if rows & (rows - 1) or rows < 16:
            continue
        score = abs(pr - target)
        if best is None or score < best[0]:
            best = (score, (pr, chunks, rows.bit_length() - 1))
    if best is not None:
        return best[1]
    return _mat_layout_padded(n_out, tokens, target)


def _mat_layout_padded(n_out, tokens, target):
    """`layer.rs`, `MatLayout::choose_padded`: a vocabulary does not tile a
    power of two at any width, so the row count is rounded up and a candidate
    whose lead exceeds a quarter of its rows is refused."""
    max_lead_frac = 0.25
    best = None
    for pr in range(1, n_out + 1):
        if n_out % pr:
            continue
        chunks = n_out // pr
        active = tokens * chunks
        log_rows = max(MIN_LOG_ROWS, (active - 1).bit_length())
        lead = (1 << log_rows) - active
        if lead / (1 << log_rows) > max_lead_frac:
            continue
        score = abs(pr - target)
        if best is None or score < best[0]:
            best = (score, (pr, chunks, log_rows), lead)
        if pr > target and best is not None and best[2] == 0:
            break
    assert best is not None, f"no layout for n_out={n_out}"
    return best[1]


def ao_layout(nh, s, hd, target):
    """`layer.rs`, `AoLayout::choose`: the largest divisor of `hd` at most the
    target, one row per (head, token, chunk), inert lead first."""
    pr = max(1, min(target, hd))
    while hd % pr:
        pr -= 1
    chunks = hd // pr
    active = nh * s * chunks
    return pr, chunks, max(MIN_LOG_ROWS, (active - 1).bit_length())


def q_layout(n, hd, tokens, target):
    """`rope.rs`, `q_layout`: a quantize width divides both the head width and
    the matmul's output width; an exact tiling beats a padded one."""
    best = None
    for pr in range(1, hd + 1):
        if hd % pr or n % pr:
            continue
        chunks = n // pr
        active = tokens * chunks
        log_rows = log_rows_for(active)
        lead = (1 << log_rows) - active
        key = (1 if lead > 0 else 0, abs(pr - target))
        if best is None or key < best[0]:
            best = (key, (pr, chunks, log_rows))
    assert best is not None, f"no quantize layout for n={n}, hd={hd}"
    return best[1]


def freivalds_layout(n_in, n_out_max, tokens, target):
    """`layer.rs`, `FreivaldsLayout::choose`: one row group per input channel,
    `pk` weights and `pq` activations per row, the chunk count scored on `pk`
    and the least padded winning among equals.  Returns (pk, pq, log_rows)."""
    best = None
    for chunks in range(1, n_out_max + 1):
        pk = -(-n_out_max // chunks)
        pq = -(-tokens // chunks)
        rows = n_in * chunks
        log_rows = max(MIN_LOG_ROWS, (rows - 1).bit_length())
        # `Grid::new` caps the chunk count at `rows / groups`; a candidate the
        # cap would cut back is not the layout it asked for, and is dropped
        if (1 << log_rows) // n_in < chunks:
            continue
        pad = (1 << log_rows) - rows
        key = (abs(pk - target), pad)
        if best is None or key < best[0]:
            best = (key, (pk, pq, log_rows))
    assert best is not None, "no Freivalds layout"
    return best[1]


# `layer.rs`, `FREIVALDS_GROUPS`: the matmuls sharing one committed input.
FREIVALDS_GROUPS = [("qkv", ("q", "k", "v")), ("o", ("o",)), ("gateup", ("gate", "up")),
                    ("down", ("down",)), ("head", ("lm_head",))]


# ------------------------------------------------------- the dequantize family --
# `dequant.rs`, `Map::new`: a row carries the variant's shared columns and then
# one slot per accumulator.  The slot is the whole statement of one
# dequantization, so its width is a property of the variant and of the rounding
# alone -- not of the data, not of `kmax`, and not of the matmul's shape.


def deq_columns(variant, sticky, per_row):
    """`dequant.rs`, `Map::new`: committed columns of one dequantize component."""
    if variant == "fold":
        shared, per_slot = 2, 9          # flo fhi sg fa fb qm qc fe fdp
    elif variant == "chain":
        # acc sg m2 ph [st] r m7 ef nz, then two (mantissa, carry, exponent)
        shared, per_slot = 2, 8 + int(sticky) + 6
    elif variant == "pow2":
        shared, per_slot = 1, 8 + int(sticky) + 1
    elif variant == "int8":
        shared, per_slot = 1, 5
    elif variant == "limbs":
        shared, per_slot = 0, 2
    else:
        raise ValueError(f"unknown dequant variant {variant!r}")
    return shared + per_row * per_slot


def deq_looks(variant, per_row):
    """`dequant.rs`, `look_units`: how many lookups one row's slots draw.  Only
    the count matters for cells; whether a lookup is at unit multiplicity does
    not."""
    per_slot = {"fold": 8, "pow2": 3, "chain": 6}[variant]
    return per_slot * per_row


# ----------------------------------------------------------- the residual stream --
# `compose.rs`, `ResidualStreamPlan::new` and `residual_stream.rs`, `Prog`.  A
# component's committed width is `leaves_width + slots * slot_width`, and both
# are counted off the program the plan builds -- an add slot is 13 columns (14
# with the sticky bit round-to-even needs), a multiply slot is 4, a leaf is two
# columns or four when a unary lookup reads it.
ADD_SLOT, MUL_SLOT, LEAF, UN_LEAF = 13, 4, 2, 4


def _tree_slots(n):
    """`Prog::tree`: `n - 1` real adds however far the padding reaches, because
    `x + 0` is pruned rather than proved."""
    return n - 1


def residual_stream_components(p):
    """The wiring family's components: (tag, columns, log_rows, bindings,
    lookups per row).  `head_norm` is the output head's three; every layer
    segment builds eleven.

    The lookup count is `gkrlink.rs`'s own `residual_stream_look_count`: one
    `VAL` lookup per committed leaf, and five per add slot or two per multiply
    slot."""
    g = p["residual_stream"]
    sticky = p["round"] == "even"
    add, mul = ADD_SLOT + int(sticky), MUL_SLOT
    a, d, s = g["a"], g["d"], p["s"]
    nreal = d // a                       # `Tiling::new`: committed leaves per row
    lr1 = log_rows_for(a * s)            # `rows1 = tokens * a`
    lr2 = log_rows_for(s)
    out = []

    def looks(leaves, slots, kind):
        return leaves + (5 if kind == "add" else 2) * slots

    def square_tree(n):
        # `Prog::tree` under the square lookup: the norm's level-1 reduction
        slots = _tree_slots(n)
        return n * UN_LEAF + slots * add, looks(n, slots, "add")

    def norm_resid():
        # `Prog::norm_resid`: the squares' tree and the residual add on one row
        slots = _tree_slots(nreal) + nreal
        cols = nreal * UN_LEAF + nreal * LEAF + slots * add
        return cols, looks(2 * nreal, slots, "add")

    def tree_plain(n):
        # `Prog::tree` with no unary: the second-level reduction over `a` rows
        slots = _tree_slots(n)
        return n * LEAF + slots * add, looks(n, slots, "add")

    def norm_scale(n):
        # `Prog::norm_scale`: `h = (x * r) * w`, `r` one committed leaf per row
        # under the `NormR` lookup, `w` preprocessed
        return n * LEAF + UN_LEAF + 2 * n * mul, looks(n + 1, 2 * n, "mul")

    def pairs_silu(n):
        # `Prog::pairs`: the SwiGLU product, group 0 under the silu lookup
        return n * UN_LEAF + n * LEAF + n * mul, looks(2 * n, n, "mul")

    if g.get("head_norm"):
        out.append(("gnf", *square_tree(nreal), lr1, 2))
        out.append(("gnfb", *tree_plain(a), lr2, 2))
        out.append(("ntf", *norm_scale(nreal), lr1, 3))
        return [(t, c, lr, b, lk) for t, c, lk, lr, b in out]

    # a layer: two norms, each a level-1 reduction folded with a residual add,
    # a level-2 reduction and a scale; then the SwiGLU and the two rotations
    for n1, n2 in (("gn1", "gn1b"), ("gn2", "gn2b")):
        # `In(0)`, `Out(0)`, `Out(1)`, and the `o` / `down` output claim
        out.append((n1, *norm_resid(), lr1, 4))
        out.append((n2, *tree_plain(a), lr2, 2))
    for tag in ("nt1", "nt2"):
        out.append((tag, *norm_scale(nreal), lr1, 3))
    t2_nreal = g["dff"] // a
    out.append(("gsw", *pairs_silu(t2_nreal), log_rows_for(a * s), 3))
    rope = p.get("rope")
    if rope:
        hd = p["hd"]
        for name, heads in (("q", p["nh"]), ("k", rope["nkv"])):
            lr = log_rows_for(s * heads)
            # `Prog::rope_mul`: `hd` leaves, `2 hd` multiply slots
            out.append((f"rp{name}m", hd * LEAF + 2 * hd * mul,
                        looks(hd, 2 * hd, "mul"), lr, 2))
            # `Prog::rope_add`: the multiplies' `2 hd` outputs, `hd` add slots
            out.append((f"rp{name}a", 2 * hd * LEAF + hd * add,
                        looks(2 * hd, hd, "add"), lr, 2))
    return [(t, c, lr, b, lk) for t, c, lk, lr, b in out]


# ------------------------------------------------------------- the lookup tables --
# The bf16 tables' row counts, in closed form (`bf16.rs`, `residual_stream.rs`,
# `requant.rs`, `rope.rs`).  Only `norm` and `frac` depend on anything the run
# calibrates; the rest are properties of the encoding.
TT_RANGE = 9
POW_ROWS = 13 + 260                      # `pow_table`: DMAX + 1 exponents, UMIN zeros
VAL_ROWS = 1 + 254 * 128 * 2             # `val_table`: every legal (sm, ef)
SMUL_ROWS = 257 * 257                    # `smul_table`: signed significand pairs
MUL_LOG_ROWS = 14                        # `bf16::mul_table`: 128 x 128, fixed


def norm_rows(kmax, sticky):
    """`bf16::norm_table`: the zero row, the 255 exactly-representable
    magnitudes, and 256 significands per binade (twice with the sticky bit)."""
    return 1 + 255 + (kmax + 1) * 256 * (1 + int(sticky))


def frac_rows(kmax):
    """`Tables::new`: one row per (ph, r) pair, `ph = 2^k` for `k <= kmax`."""
    return 1 + sum(1 << k for k in range(kmax + 1))


def _rq_table_rows(bf16=True):
    """`requant.rs`, `rq_table`: the division table's admissible rows."""
    n = 0
    for tt in range(TT_RANGE):
        for ms in (range(128) if bf16 else (0,)):
            sden = 128 + ms if bf16 else 1
            for m7 in range(128):
                m = 128 + m7
                q = (((m << tt) + sden) // (2 * sden) if bf16
                     else ((m << (tt + 1)) + 512) // 1024)
                if 1 <= q <= 127:
                    n += 1
    return n


def _qdiv_table_rows(bits):
    """`rope.rs`, `qdiv_table`: the same shape at the quantize side's width."""
    qmax = (1 << (bits - 1)) - 1
    n = 0
    for tt in range(TT_RANGE + bits - 8):
        for m7 in range(128):
            q = (((128 + m7) << (tt + 1)) + 512) // 1024
            if 1 <= q <= qmax:
                n += 1
    return n


def embed_layout(d, tokens, vocab, target):
    """`embed.rs`, `EmbLayout::choose`: the smallest power-of-two block count
    that brings a block at or under the target.  Returns (pe, k, log_use,
    log_tab)."""
    k = 1
    while d % k or d // k > target:
        k *= 2
        assert k <= d, f"no block count divides d = {d} at target {target}"
    log_v = log_rows_for(vocab)
    return d // k, k, log_rows_for(tokens * k), log_v + (k.bit_length() - 1)


def range_widths(p):
    """`compose.rs`, `Plan::new`: the range tables a segment registers.  Four of
    the eight base widths are the run's own calibration of its witness, which is
    where an architecture-only count stops being possible -- see
    `witness_terms`."""
    w = [p["k_bits"], p["q_bits"], 8, p["z_lo_bits"], p["z_hi_bits"], p["ao_rem_bits"],
         p["ao_acc_lo_bits"], p["ao_acc_hi_bits"]]
    name, k = attn_variant(p)
    if name == "intsplit":
        w.append(k)
    if any(m["variant"] == "fold" for m in p["mats"]):
        w += [13, 16, p["fold_rem_bits"], 7, p["ao_q_bits"]]
    if p["ao_q_bits"] > 7:
        w.append(p["ao_q_bits"] + 1)
    if p.get("residual_stream"):
        w.append(15)
    return sorted(set(w))


def e_window(p):
    """`compose.rs`, `Plan::e_window`: the exponent window the attention output's
    `EPOW` table covers, `e = jv - jo` over the ranges the run measured."""
    return p["ao_jv_min"] - p["ao_jo_max"], p["ao_jv_max"] - p["ao_jo_min"]


def table_rows(p):
    """Every lookup table's row count, by name.  One base multiplicity column
    and one QM31 interaction column each, so a table of `2^r` rows costs
    `5 * 2^r` and nothing else."""
    sticky = p["round"] == "even"
    rows = {"exp": 1 << p["log_table"],
            "norm": 1 << log_rows_for(norm_rows(p["kmax"], sticky)),
            "frac": 1 << log_rows_for(frac_rows(p["kmax"])),
            "mul": 1 << MUL_LOG_ROWS,
            "qnz": 1 << p["ao_q_bits"]}
    for w in range_widths(p):
        rows[f"range{w}"] = 1 << w
    lo, hi = e_window(p)
    rows["epow"] = 1 << log_rows_for(hi - lo + 1)
    g = p.get("residual_stream")
    if g:
        # `residual_stream_table_kinds`: pow, smul, val, and one per unary kind
        rows["residual_stream0"] = 1 << log_rows_for(POW_ROWS)
        rows["residual_stream1"] = 1 << log_rows_for(SMUL_ROWS)
        log_val = log_rows_for(VAL_ROWS)
        rows["residual_stream2"] = 1 << log_val
        n_un = 2 if g.get("head_norm") else 3      # Square, NormR, and the silu
        for i in range(n_un):
            rows[f"residual_stream{3 + i}"] = 1 << log_val
    return rows


def requant_table_rows(p):
    """`compose.rs`'s requantize table block: per scale variant a division table
    and a scale table, then the sign table, then the quantize family's own
    division tables and its `(q, nz)` pair table."""
    rows = {}
    if not p.get("residual_stream"):
        return rows
    variants = sorted({"pow2" if _seam_variant(p, name) == "pow2" else "bf16"
                       for name in _seam_names(p)})
    for v in variants:
        rows[f"rq_{v}"] = 1 << log_rows_for(_rq_table_rows(v == "bf16"))
        rows[f"scl_{v}"] = 1 << 15
    if variants:
        rows["rqv"] = 1 << 9
    widths = _quant_widths(p)
    for w in widths:
        rows[f"qdiv{w}"] = 1 << log_rows_for(_qdiv_table_rows(w))
    if widths:
        rows["qnz_q"] = 1 << (max(widths) - 1)
    return rows


V_BITS = 8       # `rope.rs`, `V_BITS`: the value side's quantize width


def _quant_widths(p):
    return sorted({bits for _, _, bits in _quant_sides(p)})


def _quant_sides(p):
    """`compose.rs`: q, k and v get a quantize component, except a `v` whose
    dequantize is the fold -- the fold IS the quantize."""
    rope = p.get("rope")
    if not rope or not p.get("residual_stream"):
        return []
    out = []
    for name, heads, bits in (("q", p["nh"], p["q_bits"]),
                              ("k", rope["nkv"], p["k_bits"]),
                              ("v", rope["nkv"], V_BITS)):
        m = _mat(p, name)
        if m is None:
            continue
        if name == "v" and m["variant"] == "fold":
            continue
        out.append((name, m, bits))
    return out


def _mat(p, name):
    for m in p["mats"]:
        if m["name"] == name:
            return m
    return None


def _seam_names(p):
    g = p.get("residual_stream") or {}
    names = ["head"] if g.get("head_norm") else ["qkv", "gateup", "down"]
    return [n for n in names if any(g2 == n for g2, _ in _freivalds_groups(p))]


def _seam_variant(p, name):
    mats = dict(FREIVALDS_GROUPS)[name]
    for mn in mats:
        m = _mat(p, mn)
        if m is not None:
            return m["variant"]
    return "chain"


def _seam_width(p, name):
    g = p["residual_stream"]
    return g["dff"] if name == "down" else g["d"]


def _freivalds_groups(p):
    """The groups this segment builds: those whose matmuls it carries."""
    out = []
    for name, members in FREIVALDS_GROUPS:
        idx = [m for m in p["mats"] if m["name"] in members]
        if idx:
            out.append((name, idx))
    return out


# ------------------------------------------------------------------ the count --
def exact_cells(p):
    """Every committed cell of one composed segment, from its `params` block.

    Returns a dict with `base`, `interaction`, `boundary` and `fresh` totals and
    a `components` map from family name to (base, interaction).  Compare it with
    a recorded chunk's `cells`: `base` is `cells.base`, `interaction` is
    `cells.interaction`, and `fresh` is `cells.fresh` -- base plus interaction
    plus the boundary trees, which no `family_cells` entry names."""
    comp = {}
    s, c, hd, nh = p["s"], p["c"], p["hd"], p["nh"]
    sticky = p["round"] == "even"
    head = bool((p.get("residual_stream") or {}).get("head_norm"))

    # ---- the attention key tiles
    comp["key_tiles"] = key_tile_cells(p)

    # ---- the dequantize components, one per matmul
    deq_b = deq_i = 0
    mat_lays = {}
    for m in p["mats"]:
        pr, _, lr = mat_layout(m["n_out"], m["tokens"], p["deq_target"])
        mat_lays[m["name"]] = (pr, lr)
        deq_b += deq_columns(m["variant"], sticky, pr) << lr
        # `mat_interaction`: the two Horner accumulators `z` and `h`, the output
        # claim the wiring family reads, and -- on a matmul a requantize seam
        # feeds, and only if it is not folded -- the x-side scale's fold
        qm = 2 + int(_has_output_claim(p, m["name"])) + int(_has_scale_claim(p, m))
        if not p.get("gkr_dequant"):
            qm += -(-deq_looks(m["variant"], pr) // 2)
        deq_i += (M31_PER_QM31 * qm) << lr
    comp["dequant"] = (deq_b, deq_i)

    # ---- the attention output
    ao_pr, _, ao_lr = ao_layout(nh, s, hd, p["ao_target"])
    # `layer.rs`, `ao_component`: the row's `den` and `joff`, then per slot
    # acc, sg, qm, nz2, denp, d, num, pj, alo -- and `mag` tenth under `acc_mag`
    ao_cols = 2 + ao_pr * (9 + int(p.get("ao_acc_mag", False)))
    ao_qm = 5 if p.get("gkr_attnout") else 5 + -(-7 * ao_pr // 2)
    comp["attnout"] = (ao_cols << ao_lr, (M31_PER_QM31 * ao_qm) << ao_lr)

    # ---- the Freivalds checks, one per group of matmuls sharing an input
    fv_b = fv_i = 0
    fv_lays = []
    for name, mats in _freivalds_groups(p):
        n_in = mats[0]["n_in"]
        pk, pq, lr = freivalds_layout(n_in, max(m["n_out"] for m in mats),
                                      mats[0]["tokens"], p["freivalds_p"])
        fv_lays.append((name, pk, pq, lr, len(mats)))
        fv_b += pq << lr
        # one accumulator pair per matmul, plus the group's own `z` and its
        # claim over the committed activations
        qm = 2 * len(mats) + 2
        if not p.get("gkr_freivalds"):
            qm += -(-pq // 2)
        fv_i += (M31_PER_QM31 * qm) << lr
    comp["freivalds"] = (fv_b, fv_i)

    # ---- the residual stream
    rs_b = rs_i = 0
    rs_comps = residual_stream_components(p) if p.get("residual_stream") else []
    for _tag, cols, lr, bindings, looks in rs_comps:
        rs_b += cols << lr
        qm = bindings + (0 if p.get("gkr_residual_stream") else -(-looks // 2))
        rs_i += (M31_PER_QM31 * qm) << lr
    comp["residual_stream"] = (rs_b, rs_i)

    # ---- the rotation's quantize components
    q_b = q_i = 0
    for _name, m, _bits in _quant_sides(p):
        pr, _, lr = q_layout(m["n_out"], hd, m["tokens"], p["deq_target"])
        # `rope.rs`, `Q_PER_SLOT`: ef, m7, q, nz, sm, sg, cl
        q_b += (7 * pr) << lr
        # five lookups per slot, batched in pairs, plus the two Horner claims
        q_i += (M31_PER_QM31 * (-(-5 * pr // 2) + 2)) << lr
    comp["quant"] = (q_b, q_i)

    # ---- the requantize seams
    rq_b = rq_i = 0
    for name in _seam_names(p):
        n = _seam_width(p, name)
        pr, _, lr = mat_layout(n, s, p["deq_target"])
        # `requant.rs`: the row's max pattern, the scale (one column for a
        # power-of-two scale, two for a bf16 one), seven columns per slot, and
        # the selector sum last
        n_scale = 1 if _seam_variant(p, name) == "pow2" else 2
        rq_b += (2 + n_scale + 7 * pr) << lr
        # five lookups per slot, batched in pairs, and four Horner claims
        rq_i += (M31_PER_QM31 * (-(-5 * pr // 2) + 4)) << lr
    comp["requant"] = (rq_b, rq_i)

    # ---- the output head's argmax, which `family_cells` does not name (#159)
    am_b = am_i = 0
    if p.get("prove_argmax"):
        m = _mat(p, "lm_head")
        pr, _, lr = mat_layout(m["n_out"], m["tokens"], p["deq_target"])
        # `argmax.rs`, `component`: the row maximum and its token, then a key
        # and an indicator per slot, then the running selector sum
        am_b = (2 * pr + 3) << lr
        # the binding pair, and one range lookup per slot batched in pairs
        am_i = (M31_PER_QM31 * (2 + -(-pr // 2))) << lr
    comp["argmax"] = (am_b, am_i)

    # ---- the embedding gadget on the first segment of the stack
    em_b = em_i = 0
    if p.get("embed_vocab"):
        pe, k, log_use, log_tab = embed_layout(
            (p["residual_stream"] or {})["d"], s, p["embed_vocab"], p["embed_per_row"])
        # `embed.rs`: the token id and an (sm, ef) pair per channel on the use
        # side; one multiplicity column on the table side
        em_b = ((1 + 2 * pe) << log_use) + (1 << log_tab)
        # `1 + pe` lookups per use row batched in pairs, two Horner claims, and
        # the table's own side
        em_i = (M31_PER_QM31 * (-(-(1 + pe) // 2) + 2)) << log_use
        em_i += M31_PER_QM31 << log_tab
    comp["embed"] = (em_b, em_i)

    # ---- the tables: one base multiplicity column and one QM31 column each
    rows = sum(table_rows(p).values())
    comp["tables"] = (rows, M31_PER_QM31 * rows)
    rq_rows = sum(requant_table_rows(p).values())
    comp["requant_tables"] = (rq_rows, M31_PER_QM31 * rq_rows)

    # ---- the GKR link: 16 M31 columns per distinct height it carries
    heights = link_heights(p, mat_lays, fv_lays, rs_comps, ao_lr)
    comp["gkrlink"] = (0, 16 * sum(1 << h for h in heights))

    base = sum(b for b, _ in comp.values())
    inter = sum(i for _, i in comp.values())
    boundary = boundary_cells(nh, hd, (p.get("residual_stream") or {}).get("d", 0),
                              s, c, head=head)
    return {"components": comp, "base": base, "interaction": inter,
            "boundary": boundary, "fresh": base + inter + boundary,
            "table_rows": rows}


def _has_output_claim(p, name):
    """`compose.rs`, `Plan::has_output_claim`: the wiring family publishes one
    for every matmul whose output a component reads, and the head publishes the
    logits claim."""
    g = p.get("residual_stream")
    if g and not g.get("head_norm"):
        if name in ("o", "down", "gate", "up"):
            return True
        if p.get("rope") and name in ("q", "k", "v"):
            return True
    return bool(p.get("logits_claim")) and name == "lm_head"


def _has_scale_claim(p, m):
    """`compose.rs`: the x-side scale's fold, on the matmuls a requantize seam
    feeds -- and never on a folded one, whose second shared column is not the
    exponent the claim would compare."""
    if m["variant"] == "fold":
        return False
    for name in _seam_names(p):
        if m["name"] in dict(FREIVALDS_GROUPS)[name]:
            return True
    return False


def link_heights(p, mat_lays, fv_lays, rs_comps, ao_lr):
    """`gkr_key_tile.rs`, `sites_of`: the distinct trace heights the batched
    reduction reads, which is what the link pays 16 committed columns for."""
    hs = set()
    if p.get("gkr_key_tiles") or p.get("gkr_bands"):
        for insts, c_tile, rows_comp, _diag in _key_tile_shapes(p):
            hs.add(chan_layout(p, insts, c_tile)["log_rows"])
            hs.add(key_layout(p, insts, c_tile)["log_rows"])
            if rows_comp:
                hs.add(row_layout(insts * p["s"], p["rw"])[1])
    if p.get("gkr_dequant"):
        hs.update(lr for _, lr in mat_lays.values())
    if p.get("gkr_residual_stream"):
        hs.update(lr for _, _, lr, _, _ in rs_comps)
    if p.get("gkr_freivalds"):
        hs.update(lr for _, _, _, lr, _ in fv_lays)
    if p.get("gkr_attnout"):
        hs.add(ao_lr)
    return sorted(hs)


def attn_variant(p):
    """`attn_witness.rs`, `Variant::parse`: `intexp`, `intsplit<k>` or `bf16x`.
    Returns (name, k) with `k` only for `intsplit`."""
    v = p["variant"]
    if v in ("intexp", "bf16x"):
        if v == "bf16x":
            raise SystemExit(
                "exact_cells: the `bf16x` attention variant commits no row maximum, "
                "so its per-token component is a different width; no recorded "
                "composed segment uses it and this module does not describe it.")
        return v, None
    if v.startswith("intsplit"):
        return "intsplit", int(v[len("intsplit"):])
    raise SystemExit(f"exact_cells: unknown attention variant {v!r}")


def key_tile_cells(p):
    """The key tile family of one segment, from the three grids themselves.

    `key_tile.py`'s closed form is this count where nothing is padded -- `hd`,
    `S` and `C` powers of two and both chunk counts dividing what they are read
    against.  Here the grids are built, so a context that is not a power of two
    (the output head's `C = 3584`, whose key grid carries 32,768 inert rows)
    counts right instead of counting `saturated`."""
    base = inter = 0
    bind = p.get("rope") is not None
    gkr = p.get("gkr_key_tiles", False) or p.get("gkr_bands", False)
    split = attn_variant(p)[0] == "intsplit"
    for insts, c_tile, rows_comp, diagonal in _key_tile_shapes(p):
        lc = chan_layout(p, insts, c_tile)
        lk = key_layout(p, insts, c_tile)
        # `attn_trace.rs`, `commit_base`: the channel component commits the key
        # cache and the queries; the grid commits the score, its numerator, the
        # `intsplit` variant's auxiliary column per score, and the value cache
        n_aux = lk["q"] if split else 0
        base += (lc["pk"] + lc["pq"]) << lc["log_rows"]
        base += (2 * lk["q"] + n_aux + lk["pv"]) << lk["log_rows"]
        # `attn_trace.rs`, `interaction_phase`: three Horner accumulator pairs on
        # the channel component, six on the grid, and stream NN's root-equality
        # claims -- Q on every bound key tile, K and V on the diagonal block
        ci = 12 + (M31_PER_QM31 * (2 if diagonal else 1) if bind else 0)
        gi = 24 + (M31_PER_QM31 if bind and diagonal else 0)
        if not gkr:
            # the LogUp running sums, batched in pairs
            ci += M31_PER_QM31 * -(-(lc["pk"] + lc["pq"]) // 2)
            gi += M31_PER_QM31 * -(-(lk["q"] + n_aux + lk["pv"]) // 2)
        inter += ci << lc["log_rows"]
        inter += gi << lk["log_rows"]
        if rows_comp:
            # the per-token component: one row maximum, one denominator and its
            # two limbs per token, `rw` tokens per row
            rw, lr = row_layout(insts * p["s"], p["rw"])
            base += (4 * rw) << lr
            ri = 8 + (0 if gkr else M31_PER_QM31 * 2 * rw)
            inter += ri << lr
    return base, inter


def row_layout(toks, rw_target):
    """`layout.rs`, `Layouts::choose`: the per-token component tiles exactly, so
    the tokens per row come down until the row count lands on the SIMD floor
    rather than padding up to it.  Returns (rw, log_rows)."""
    floor = 1 << MIN_LOG_ROWS
    rw = max(1, min(rw_target, toks // floor))
    rows = max(floor, -(-toks // rw))
    assert rw * rows == toks, f"the per-token component does not tile: rw={rw}"
    return rw, max(MIN_LOG_ROWS, (rows - 1).bit_length())


def _grid_chunks(groups, log_rows, cap):
    """`layout.rs`, `Grid::new`: `chunks` rows per group, capped, with the slack
    carried as inert lead rows."""
    rows = 1 << log_rows
    if rows < groups or groups == 0:
        return None
    return min(rows // groups, max(cap, 1))


def chan_layout(p, insts, c_tile):
    """`layout.rs`, `LayoutChan::choose`: one row group per (instance, channel),
    the height whose keys per row is nearest `chan_target`."""
    best = None
    for lr in range(MIN_LOG_ROWS, 27):
        chunks = _grid_chunks(p["hd"] * insts, lr, min(c_tile, p["s"]))
        if chunks is None:
            continue
        pk = -(-c_tile // chunks)
        key = (abs(pk - p["chan_target"]), 1 << lr)
        if best is None or key < best[0]:
            best = (key, {"log_rows": lr, "chunks": chunks, "pk": pk,
                          "pq": -(-p["s"] // chunks)})
    assert best is not None, "no channel layout"
    return best[1]


def key_layout(p, insts, c_tile):
    """`layout.rs`, `LayoutKey::choose`: one row group per (instance, key),
    scored on the chunk count, and the token axis has to tile exactly."""
    want = -(-p["s"] // p["grid_target"])
    best = None
    for lr in range(MIN_LOG_ROWS, 27):
        chunks = _grid_chunks(c_tile * insts, lr, min(p["s"], p["hd"]))
        if chunks is None or p["s"] % chunks:
            continue
        key = (abs(chunks - want), 1 << lr)
        if best is None or key < best[0]:
            best = (key, {"log_rows": lr, "chunks": chunks,
                          "q": -(-p["s"] // chunks), "pv": -(-p["hd"] // chunks)})
    assert best is not None, "no key layout"
    return best[1]


def _key_tile_shapes(p):
    """The segment's key tiles: `c / s` prefix tiles of `s` keys per head group
    (stream AA's `--kv-boundary` tiling) and the diagonal block, with the
    per-token component riding on the group's first tile."""
    groups = p.get("head_groups")
    if groups is None:
        raise SystemExit(
            "exact_cells: this params block carries no `head_groups`, and how the "
            "attention heads are grouped into key tiles decides the grids' heights "
            "and therefore what the GKR link pays for.  It is the prover's "
            "`--head-groups`, a run knob; run outputs before 2026-09-08 do not "
            "record it.")
    out = []
    for insts in groups:
        n_pre = 0 if not p["c"] else (p["c"] // p["s"] if p.get("kv_boundary") else 1)
        sub_c = p["s"] if p.get("kv_boundary") else p["c"]
        for k in range(n_pre):
            out.append((insts, sub_c, k == 0, False))
        if p.get("diagonal_block"):
            out.append((insts, p["s"], n_pre == 0, True))
    return out




# --------------------------------------------- what is not the architecture --
def witness_terms(p):
    """The terms of the count that a `params` block carries but a model and a
    context do not: the run measured them on its own witness.  Returns each
    one's cells and its share of `fresh`, so a caller who has to ASSUME them can
    see what the assumption is worth."""
    total = exact_cells(p)["fresh"]
    sticky = p["round"] == "even"
    out = {}
    # the master exponent table: sized to cover the scores this witness produced
    out["log_table"] = 5 * (1 << p["log_table"])
    # `kmax` sizes the normalisation table and the fraction table
    out["kmax"] = 5 * ((1 << log_rows_for(norm_rows(p["kmax"], sticky)))
                       + (1 << log_rows_for(frac_rows(p["kmax"]))))
    # the attention output's measured exponent ranges size three range tables
    # and the exponent window
    lo, hi = e_window(p)
    out["ao_ranges"] = 5 * ((1 << p["ao_acc_hi_bits"]) + (1 << p["ao_rem_bits"])
                            + (1 << log_rows_for(hi - lo + 1)))
    return {k: (v, v / total) for k, v in out.items()}


# ------------------------------------------------------------- the self-test --
def chunks():
    d = json.loads(CHUNKS.read_text())
    return [dict(p, params=dict(d["params"], **p["params"])) for p in d["points"]]


def selftest(argv_points=None):
    """The form against every recorded chunk, to the cell."""
    pts = chunks() if argv_points is None else argv_points
    bad = []
    for p in pts:
        got = exact_cells(p["params"])
        want = (p["base"], p["interaction"], p["fresh"])
        have = (got["base"], got["interaction"], got["fresh"])
        if have != want:
            bad.append((p, want, have, got))
    for p, want, have, got in bad[:12]:
        print(f"  {p['run']} {p['segment']}: base {have[0]:,} vs {want[0]:,} "
              f"({have[0] - want[0]:+,}), interaction {have[1]:,} vs {want[1]:,} "
              f"({have[1] - want[1]:+,})")
        for k, (b, i) in sorted(got["components"].items()):
            rec = p.get("family", {}).get(k)
            if rec is not None and (b, i) != tuple(rec):
                print(f"      {k}: base {b:,} vs {rec[0]:,}, "
                      f"interaction {i:,} vs {rec[1]:,}")
    if bad:
        print(f"exact cells: {len(bad)} of {len(pts)} recorded chunks do not reproduce")
        return 1
    # the negative controls: the comparison above has to reject each recorded
    # number moved by one cell, or "exact" means nothing
    for p in pts:
        got = exact_cells(p["params"])
        for field in ("base", "interaction", "fresh"):
            if got[field] == p[field] + 1:
                print(f"exact cells: the comparison accepts a one-cell difference "
                      f"in `{field}` at {p['segment']}")
                return 1
    def spread(f):
        v = sorted({f(p["params"]) for p in pts})
        return f"{v[0]:,} to {v[-1]:,}" if len(v) > 1 else f"{v[0]:,}"

    kinds = sorted({segment_kind(p["params"]) for p in pts})
    red = sorted({tuple(sorted(k for k in p["params"] if k.startswith("gkr_")
                               and p["params"][k])) for p in pts})
    print(f"exact cells: {len(pts)} recorded segments, exact to the cell")
    print(f"  C = {spread(lambda p: p['c'])}, S = {spread(lambda p: p['s'])}, "
          f"hd = {spread(lambda p: p['hd'])}, nh = {spread(lambda p: p['nh'])}")
    print(f"  segment kinds {', '.join(kinds)}; attention variants "
          f"{', '.join(sorted({p['params']['variant'] for p in pts}))}; roundings "
          f"{', '.join(sorted({p['params']['round'] for p in pts}))}")
    print(f"  {len(red)} reduction configurations, from none of the five to all five")
    print("  a one-cell difference is a failure: checked")
    return 0


def segment_kind(p):
    """What this segment is: the output head, the first segment of a stack (the
    one that carries the embedding lookup), or a layer."""
    if (p.get("residual_stream") or {}).get("head_norm"):
        return "output head"
    return "first segment" if p.get("embed_vocab") else "layer"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="check the count against the recorded chunks (the default)")
    ap.add_argument("--witness-share", action="store_true",
                    help="print what the witness-derived terms cost on each chunk")
    a = ap.parse_args(argv)
    if a.witness_share:
        for p in chunks():
            w = witness_terms(p["params"])
            print(f"{p['run']} {p['segment']}: " + ", ".join(
                f"{k} {v:,} ({f:.2%})" for k, (v, f) in w.items()))
        return 0
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
