"""The attention key tile family, as a closed form in (S, C), and its self-test.

    uv run python -m harness.costmodel.key_tile --selftest

Kept out of `cost_model.py` so that the self-test runs with nothing: the cost
model reads its micro-benchmark inputs at import time, and this form reads no
measured artifact at all.  `cost_model` re-exports every name below, so
`cm.key_tile_cells` and `cm.key_tile_fresh` are unchanged for their callers.

The self-test evaluates the form against `key_tile_segments.json`, the measured
`family_cells.key_tiles` of every recorded layer segment, and compares to the
cell: a one-cell difference is a failure and there is no tolerance to widen.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
SEGMENTS = HERE / "key_tile_segments.json"

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
# output head (`main.rs`: `kv_boundary = diagonal_block && !head`), which is every
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
    (`main.rs`: `kv_boundary = diagonal_block && !head`).  The group's one
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
                    help="check the form against the recorded segments (the default)")
    ap.parse_args(argv)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
