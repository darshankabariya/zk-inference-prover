"""The closed forms a chunk's cell count is exact in -- the attention key tile
family, the boundary trees and the lookup tables -- and their self-test.

    uv run python -m harness.costmodel.key_tile --selftest

Kept out of `cost_model.py` so that the self-test runs with nothing: the cost
model reads its micro-benchmark inputs at import time, and these forms read no
measured artifact at all.  `cost_model` re-exports every name below, so
`cm.key_tile_cells`, `cm.key_tile_fresh`, `cm.boundary_cells` and
`cm.lookup_table_cells` are the same objects for their callers.

The self-test evaluates the key tile form against `key_tile_segments.json`, the
measured `family_cells.key_tiles` of every recorded layer segment, and the
boundary form and the whole-chunk identity against `layer_chunks.json`, every
chunk of one token block of each headline run.  Both compare to the cell: a
one-cell difference is a failure and there is no tolerance to widen.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
SEGMENTS = HERE / "key_tile_segments.json"
CHUNKS = HERE / "layer_chunks.json"

# Stream DD (2026-09-01) replaced B9's `p S^2` fit with a two-variable model of the
# key tile family, and stream E1 re-derived its coefficients symbolically so they can
# be evaluated at a shape that is not Llama-8B's.  Every line below is an identity
# of the layout, read off `prover/src/attn_trace.rs` (`commit_base`,
# `interaction_phase`) against `prover/src/layout.rs` and `compose.rs`'s `Plan`;
# nothing here is fitted.
#
# ONE KEY TILE over `c` keys, `nh` heads of width `hd`, `s` decoded tokens:
#
#   component                        base                     interaction
#   channel: keys   `Cells::kv`   pk*rows   = nh*hd*C      2*nh*hd*C  (K range LogUp)
#                                                          + 12*nh*hd/ct*C (Horner)
#   channel: queries              pq*rows   = nh*hd*S      2*nh*hd*S  (query range)
#   grid: score + numerator       2q*rows   = 2*nh*S*C     2*nh*S*C   (exp/numerator)
#                                                          + 24*nh/gt*S*C (Horner)
#   grid: V values  `Cells::kv`   pv*rows   = nh*hd*C      2*nh*hd*C  (V range LogUp)
#   per-token: max, denom, 2 limbs 4*nh*S                  (8*nh + 8*nh/rw)*S
#   out (A.V)                     the `attnout` family, NOT `key_tiles` -- charged
#                                 separately by `cells_per_token`'s `attn_out`
#   root equality (`bind`)        no committed cell        4*nh*hd/ct*C per claim
#                                                          + 4*nh/gt*S*C for V
#
# The layout identities the row counts come from, with `insts` summing to `nh`
# over the key tiles (every term is linear in `insts`, so how the heads are grouped into key tiles is
# cost-neutral -- 16/8/8 costs what 32 would):
#   lc.g.rows = hd*insts*C/pk,  pk -> chan_target   (LayoutChan::choose)
#   lk.g.rows =  C*insts*S/q,   q  -> grid_target   (LayoutKey::choose)
#   rrows*rw  = nh*S                                (the per-token row component)
#
# THE FAMILY of one composed layer segment is not one prefix key tile.  Stream AA's
# `--kv-boundary` tiles the unchecked prefix into blocks of S keys, so that a
# consumer's sub-key-tile is the same `Shape` as the producer's diagonal block and the
# two column lists coincide by construction (`compose.rs`, `Plan::new`):
#
#   * `C/S` prefix key tiles of S keys each, plus the S x S diagonal block;
#   * the group's ONE per-token component -- one row maximum and one denominator
#     per token, not per key block -- sits on its FIRST key tile alone
#     (`Plan::rows_key_tile`), so `C/S` of the `C/S + 1` key tiles carry none;
#   * block 0 has no prefix at all: `C = 0`, no prefix key tile, and the diagonal block
#     carries the per-token component.
#
# `--kv-boundary` is on whenever the diagonal block is and the segment is not the
# output head (`cli.rs`: `kv_boundary = diagonal_block && !head`), which is every
# layer segment this prover generation records.  Against the untiled form the
# difference is exactly `nh*hd*(C - S)` of base cells per layer segment: a third
# `nh*hd` column set per key block on the channel side, less the one the untiled
# diagonal block no longer duplicates.
#
# Two more things the composed statement charges that a standalone key tile does not:
#
#   * `bind` -- stream NN's root-equality claims, one folded column each: Q on
#     every key tile, K and V on the diagonal block, whose keys are this block's own
#     tokens.  Present whenever the segment has `Params::rope`, i.e. on every
#     layer segment and on no output head.
#   * `gkr` -- with `--gkr-key-tiles` the key tiles' batched reduction proves the
#     LogUp running sums and the committed running-sum columns are not built at
#     all.  Off by default in the prover, so off by default here; the reduction
#     moves those cells into the `gkrlink` family, which this model does not carry.
#
# Two things that are NOT derived and would move at another shape:
#   * `grid_target` / `chan_target` / `rw` are run knobs, not constants.  The
#     composed sequence used 64 / 128 / 16 and this model keeps them; a prover run
#     at a frontier shape would `choose()` against the same targets, but the
#     achieved `pk`/`q` are `div_ceil` results and only tile exactly when S, C, hd
#     and `insts` are powers of two and the target divides.  Away from that the
#     real cost is a few percent above this.
#   * `Grid::new` caps `q` at `min(S, hd)` and `pk` at `min(C, S)`.  Below those
#     caps the targets are met; above them the layout saturates and these
#     expressions stop being exact.  `key_tile_cells` reports when it saturates,
#     and reports a shape whose grids do not tile exactly by the same flag.
#
# The `intexp` variant only: `intsplit` commits an aux column per score and range
# checks it, `bf16x` commits no row maximum, and neither is priced here.
GRID_TARGET, CHAN_TARGET, ROW_WIDTH = 64, 128, 16


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def key_tile_cells(nh, hd, s, c, grid_target=GRID_TARGET, chan_target=CHAN_TARGET,
                   rw=ROW_WIDTH, rows_component=True, diagonal_block=False,
                   bind=True, gkr=False):
    """Committed cells of ONE key tile over `c` keys, for `nh` heads of width `hd`
    and `s` decoded tokens.  Returns (base, interaction, saturated).

    `key_tile_family` assembles the segment's whole family out of this; a caller
    that wants the family must not call this once with the whole context.

    `rows_component=False` drops the per-token max/denominator component, which is what
    every key tile but the group's first does (`attn_trace.rs`, `with_rows`): they
    share one max and one denominator per token.

    `diagonal_block=True` says this key tile's keys are the segment's own tokens, so it
    publishes stream NN's K and V root-equality claims beside the Q claim every
    bound key tile publishes.  `bind=False` drops all three, which is what a segment
    without `Params::rope` -- the output head -- builds.

    `gkr=True` drops the committed LogUp running-sum columns, which is what
    `--gkr-key-tiles` does: the key tiles' batched reduction proves those sums instead.

    **`hd` here is the ATTENTION head width and `nh` the number of ATTENTION
    heads.**  The AIR commits one K and one V column set per attention head, not
    per KV group -- `LayoutChan`/`LayoutKey` are built from `Shape::chans()` and
    `Shape::keys()`, both of which carry `insts` = attention heads.  So grouped
    query attention buys this prover nothing on the committed KV side today; a
    GQA-aware layout would divide the `C` term by `nh/nkv`, and at a frontier
    shape that ratio is 8-32.  That is a named, unbuilt optimisation, not a
    modelling choice."""
    q = max(1, min(grid_target, s, hd))     # scores per grid row, capped by Grid::new
    pk = max(1, min(chan_target, c, s))     # keys per channel row, same cap
    row_w = min(rw, max(1, nh * s))
    # a layer with no attention grid at all (a linear-attention block, priced
    # here at C = 0) is not a saturated layout, it is an absent one
    saturated = bool(s and c) and ((q != grid_target) or (pk != chan_target)
                                   or not _tiles(nh, hd, s, c, q, pk, row_w))
    base = (2 * nh * s * c                       # score + numerator
            + 2 * nh * hd * c                    # K (channel component) and V (grid)
            + nh * hd * s                        # the queries
            + (4 * nh * s if rows_component else 0))  # max, denominator, two limbs
    inter = ((24.0 * nh / q) * s * c             # the grid's six accumulator columns
             + (12.0 * nh * hd / pk) * c         # the channel component's three
             + ((8.0 * nh / row_w) * s if rows_component else 0))  # the per-token pair
    if bind:                                     # stream NN's root-equality claims
        inter += (4.0 * nh * hd / pk) * c * (2 if diagonal_block else 1)   # Q, and K
        if diagonal_block:
            inter += (4.0 * nh / q) * s * c                                # and V
    if not gkr:                                  # the LogUp running sums
        inter += (2 * nh * s * c                 # exp / numerator LogUp
                  + 2 * nh * hd * c              # V range LogUp
                  + 2 * nh * hd * c              # K range LogUp
                  + 2 * nh * hd * s              # query range LogUp
                  + (8.0 * nh * s if rows_component else 0))
    return base, inter, saturated


def _tiles(nh, hd, s, c, q, pk, row_w):
    """Do the three grids tile exactly at this shape?

    `Grid::new` pads every component up to a power of two and `pick` takes the
    `div_ceil` closest to the target, so the expressions above are identities
    only where nothing is padded and nothing is rounded up: `hd`, `S` and `C`
    powers of two, both chunk counts dividing what they are read against, and
    neither above `Grid::new`'s cap.  `insts` is a power of two by the layout's
    own assertion, and `nh` need not be: every term is linear in `insts`."""
    if not (_pow2(hd) and _pow2(s) and _pow2(c)):
        return False
    if c % pk or s % q:
        return False
    kc, kk = c // pk, s // q                     # the two chunk counts
    return (kc <= min(c, s) and s % kc == 0      # the channel component's queries per row
            and kk <= min(s, hd) and hd % kk == 0    # the grid's V values per row
            and (nh * s) % row_w == 0)


def key_tile_family(nh, hd, s, c, diagonal_block=True, kv_boundary=None, **kw):
    """The whole key tile family of ONE composed segment.  Returns (base, interaction,
    saturated).

    With `kv_boundary` the prefix is tiled into `c // s` key tiles of `s` keys each
    (stream AA); without it the prefix is one key tile over `c` keys.  It defaults to
    the diagonal block's own setting, which is what the prover derives it from
    (`cli.rs`: `kv_boundary = diagonal_block && !head`).  The group's one
    per-token component rides on the first key tile of the family."""
    if kv_boundary is None:
        kv_boundary = diagonal_block
    n_pre = 0 if not c else (c // s if kv_boundary else 1)
    sub_c = s if kv_boundary else c
    base, inter = 0, 0.0
    # `--kv-boundary` asserts `C % S == 0`; a shape that does not divide is not
    # one this form describes
    saturated = bool(c) and kv_boundary and bool(c % s)
    for k in range(n_pre):
        b, i, sat = key_tile_cells(nh, hd, s, sub_c, rows_component=(k == 0),
                                   diagonal_block=False, **kw)
        base, inter, saturated = base + b, inter + i, saturated or sat
    if diagonal_block:
        b, i, sat = key_tile_cells(nh, hd, s, s, rows_component=(n_pre == 0),
                                   diagonal_block=True, **kw)
        base, inter, saturated = base + b, inter + i, saturated or sat
    return base, inter, saturated


# ------------------------------------------------------- the boundary trees --
# A chunk's `cells.fresh` is its base tree plus its interaction tree plus its
# BOUNDARY trees, and no `family_cells` entry names the last of the three: they
# are what `fresh` has above `base + interaction` (`prover/src/common.rs`,
# `Cells::fresh`; `prover/src/boundary.rs`, `sides`/`kv_sides`).  A boundary
# tree commits one side of what a chunk hands to the chunk that reads it, so
# the layout fixes the count exactly:
#
#   * the residual stream, `d` wide over `s` tokens, on the way in and on the
#     way out                                                        2*d*s
#   * the keys and the values a layer chunk publishes, `nh*hd` wide per key
#     block: the OUT side covers this chunk's own `c/s + 1` key blocks (the
#     tiled prefix and the diagonal block) and the IN side the `c/s` blocks it
#     read, which is `2*nh*hd*(2*c + s)` over both K and V
#
# An output head chunk publishes no keys and no values and reads the residual
# stream without handing one on, so it commits the one input side alone.
def boundary_cells(nh, hd, d, s, c, head=False, kv_per_key=None):
    """The boundary trees of ONE composed chunk, in cells.  An identity of the
    layout, exact on every recorded chunk (`--selftest`), and the only term
    besides the key tile family that grows with the context.

    `kv_per_key` is what one key costs the two caches, `2 nh hd` where a head
    holds a whole K and a whole V.  A latent attention holds neither: its cache
    is one latent per key whatever the head count, so the caller states the
    width and this form stays the one it was."""
    if head:
        return d * s
    kv = 2 * nh * hd if kv_per_key is None else kv_per_key
    return 2 * d * s + kv * (2 * c + s)


# --------------------------------------------------------- the lookup tables --
# Every lookup table a chunk uses costs the same two columns whatever it holds:
# one multiplicity column in the base trace, one QM31 column in the interaction
# trace, both one entry per table ROW.  A QM31 cell is four M31 cells, so a
# table of `2^r` rows costs `5 * 2^r` and nothing else -- an identity of LogUp,
# not a measurement, and the `interaction_by_family.table` of every recorded
# chunk is exactly four times its `family_cells.tables` multiplicities.
#
# The set below is the `b10-final` spec's own, read off the recorded chunks'
# `family_cells.tables`: nineteen tables whose rows the spec fixes, plus the
# master exponent table, whose rows are the chunk's `params.log_table` (22 to
# 25 over the recorded layers, 23 the modal one and 25 on layer 0) and which is
# most of the total on its own.  A twentieth, small range table (`range4`,
# `range5` or `range6`, 16 to 64 rows) appears on some layers and not others,
# which is why this form is exact to 320 cells and not to the cell.
LOG_TABLE = 23
FIXED_TABLE_ROWS = ((14, "norm"), (20, "frac"), (14, "mul"), (11, "qnz"),
                    (7, "range7"), (8, "range8"), (11, "range11"), (12, "range12"),
                    (13, "range13"), (15, "range15"), (16, "range16"), (20, "range20"),
                    (4, "epow"), (9, "residual_stream0"), (17, "residual_stream1"),
                    (16, "residual_stream2"), (16, "residual_stream3"),
                    (16, "residual_stream4"), (16, "residual_stream5"))
CELLS_PER_TABLE_ROW = 5


def lookup_table_cells(log_table=LOG_TABLE, tables=FIXED_TABLE_ROWS):
    """What one chunk's lookup tables cost, in cells: five per table row over the
    master exponent table at `2^log_table` rows and the spec's fixed tables."""
    rows = (1 << log_table) + sum(1 << r for r, _ in tables)
    return CELLS_PER_TABLE_ROW * rows


def key_tile_fresh(nh, hd, s, c, diagonal_block=True, **kw):
    """The key tile family's fresh cells (base + interaction) of one composed segment."""
    b, i, sat = key_tile_family(nh, hd, s, c, diagonal_block=diagonal_block, **kw)
    return b + i, sat


# ------------------------------------------------------------- the self-test --
def segments():
    """The measured points the form is checked against, and the shape they share."""
    d = json.loads(SEGMENTS.read_text())
    return [dict(d["shape"], **p) for p in d["points"]]


def evaluate(p):
    """(predicted base, predicted interaction) at one recorded point."""
    b, i, _ = key_tile_family(
        p["nh"], p["hd"], p["s"], p["c"],
        grid_target=p["grid_target"], chan_target=p["chan_target"], rw=p["rw"],
        diagonal_block=p["diagonal_block"], kv_boundary=p["kv_boundary"],
        bind=p["rope"], gkr=p["gkr_key_tiles"])
    return int(b), int(round(i))


def reproduces(p):
    """Does the form give this recorded point's two cell counts exactly?  Equality,
    with no tolerance: one cell apart is not reproduced."""
    return evaluate(p) == (p["key_tiles"], p["key_tiles_interaction"])


def chunks():
    """The recorded chunks the boundary form and the identity are checked against."""
    d = json.loads(CHUNKS.read_text())
    return [dict(d["shape"], **p) for p in d["points"]]


def chunk_identity(pts=None):
    """The whole-chunk identity, solved on the recorded chunks.

    A composed layer chunk's fresh cells are its key tile family, its boundary
    trees, its lookup tables and a REST that the shape fixes and the context
    does not touch:

        cells.fresh == key_tile_family(nh, hd, S, C) + boundary_cells(...)
                       + 5 * table_rows + rest

    `rest` comes out at one integer on every recorded layer chunk of both
    headline runs at both contexts, and at one other integer on the two layer-0
    chunks, which carry the embedding lookup and one more cell per
    attention-output accumulator.  Both are returned rather than asserted:
    solving them here is what lets a caller quote them without transcribing a
    number.
    """
    out = {}
    for p in (chunks() if pts is None else pts):
        if p["head"]:
            continue
        b, i, _ = key_tile_family(
            p["nh"], p["hd"], p["s"], p["c"],
            grid_target=p["grid_target"], chan_target=p["chan_target"], rw=p["rw"],
            diagonal_block=p["diagonal_block"], kv_boundary=p["kv_boundary"],
            bind=p["rope"], gkr=p["gkr_key_tiles"])
        rest = (p["fresh"] - (int(b) + int(round(i)))
                - boundary_cells(p["nh"], p["hd"], p["d"], p["s"], p["c"])
                - CELLS_PER_TABLE_ROW * p["table_rows"])
        out.setdefault("embedding" if p["embedding"] else "layer", set()).add(rest)
    return dict(layer=sorted(out.get("layer", ())),
                embedding=sorted(out.get("embedding", ())))


def boundary_selftest():
    """The boundary form, the lookup-table form and the identity, to the cell."""
    pts = chunks()
    bad = [p for p in pts
           if (p["fresh"] - p["base"] - p["interaction"])
           != boundary_cells(p["nh"], p["hd"], p["d"], p["s"], p["c"], head=p["head"])]
    for p in bad:
        print(f"  {p['run']} {p['segment']} C={p['c']} head={p['head']}: boundary "
              f"{p['fresh'] - p['base'] - p['interaction']:,} vs "
              f"{boundary_cells(p['nh'], p['hd'], p['d'], p['s'], p['c'], head=p['head']):,}")
    if bad:
        print(f"boundary form: {len(bad)} of {len(pts)} recorded chunks do not reproduce")
        return 1
    # the negative control: a one-cell difference has to be rejected
    for p in pts:
        q = dict(p, fresh=p["fresh"] + 1)
        if (q["fresh"] - q["base"] - q["interaction"]) == boundary_cells(
                q["nh"], q["hd"], q["d"], q["s"], q["c"], head=q["head"]):
            print(f"boundary form: the comparison accepts a one-cell difference "
                  f"at {p['segment']}")
            return 1
    heads = sum(1 for p in pts if p["head"])
    ctxs = sorted({p["c"] for p in pts if not p["head"]})
    print(f"boundary form: {len(pts)} recorded chunks ({len(pts) - heads} layer, "
          f"{heads} output head), exact to the cell")
    print(f"  contexts C = {ctxs[0]:,} and {ctxs[-1]:,}, d = {pts[0]['d']:,}, "
          f"S = {pts[0]['s']}")
    # the lookup tables, against the same chunks' own table rows
    off = sorted({CELLS_PER_TABLE_ROW * p["table_rows"] - lookup_table_cells(p["log_table"])
                  for p in pts if not p["head"]})
    if off[-1] > 320 or off[0] < 0:
        print(f"lookup-table form: off by {off[0]:,} to {off[-1]:,} cells, not 0 to 320")
        return 1
    print(f"lookup-table form: the spec's fixed tables plus 2^log_table are "
          f"{off[0]} to {off[-1]} cells under every chunk's own table rows "
          f"(the small per-layer range table)")
    # the whole-chunk identity
    ident = chunk_identity(pts)
    if len(ident["layer"]) != 1 or len(ident["embedding"]) != 1:
        print(f"whole-chunk identity: rest is not single-valued: {ident}")
        return 1
    lay, emb = ident["layer"][0], ident["embedding"][0]
    n_lay = sum(1 for p in pts if not p["head"] and not p["embedding"])
    print(f"whole-chunk identity: fresh == key tiles + boundary + 5 * table rows + "
          f"{lay:,} on {n_lay} layer chunks, exact to the cell")
    print(f"  the two layer-0 chunks come out at {emb:,}, {emb - lay:,} more: the "
          f"embedding lookup and one more cell per attention-output accumulator")
    return 0


def selftest():
    """The form against every recorded layer segment, to the cell."""
    pts = segments()
    bad = [p for p in pts if not reproduces(p)]
    n_seg = sum(p["segments"] for p in pts)
    for p in bad:
        b, i = evaluate(p)
        print(f"  {p['run']} {p['output']} S={p['s']} C={p['c']} "
              f"gkr={p['gkr_key_tiles']}: base {b:,} vs {p['key_tiles']:,}, "
              f"interaction {i:,} vs {p['key_tiles_interaction']:,}")
    if bad:
        print(f"key tile form: {len(bad)} of {len(pts)} recorded points do not reproduce")
        return 1
    # the negative controls: the comparison above has to reject each measured
    # number moved by one cell, on every point, or "exact" means nothing
    for p in pts:
        for field in ("key_tiles", "key_tiles_interaction"):
            if reproduces(dict(p, **{field: p[field] + 1})):
                print(f"key tile form: the comparison accepts a one-cell difference "
                      f"in `{field}` at C={p['c']}")
                return 1
    ctxs = sorted({p["c"] for p in pts})
    red = sorted({p["gkr_key_tiles"] for p in pts})
    print(f"key tile form: {len(pts)} recorded points, {n_seg:,} segments, exact to the cell")
    print(f"  contexts C = {ctxs[0]:,} to {ctxs[-1]:,}, S in {sorted({p['s'] for p in pts})}, "
          f"the five reductions {'on and off' if len(red) == 2 else red[0]}")
    print("  a one-cell difference is a failure: checked")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="check the forms against the recorded segments (the default)")
    ap.parse_args(argv)
    return selftest() or boundary_selftest()


if __name__ == "__main__":
    sys.exit(main())
