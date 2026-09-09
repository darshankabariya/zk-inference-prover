# The verifier derives the shape root instead of reading it

Date: 2026-09-09. Status: decided by the owner (2026-09-09).

## Decision

`verify_composed` rebuilds the shape tree from the statement's own parameters
and the per-layer calibration the model manifest publishes, and compares the
root it derives with the `pre_root` the statement declares. A mismatch is a
rejection naming both roots. The model root is not derived: it commits weights
the verifier does not hold, so it stays a comparison against the published
manifest.

The calibration is the verifier's input, given as `--verify-calibration FILE`,
and `boundary-grid --verify` demands it as it demands `--prompt`. A run given
none derives nothing and says so: the shape root is then asserted by the prover
and checked by nothing, which is what a single proof's verification did before
this.

## Why

The shape tree carries the columns the AIR reads as facts rather than as the
prover's values -- every selector, the causal mask, the marker for the rows a
key grid really holds, the range tables' indices. A verifier that read the root
off the statement accepted one proof against whatever shape that statement
named, and the mask invariant the attention component rests on held only because
`boundary-grid --expect-pre-roots` compared the declaration with a table derived
elsewhere. A run that verified one proof alone made no such comparison.

The derivation became affordable in two steps. The shape tree was shown to be a
function of exactly the statement and the per-layer calibration, and cached
under that pair; then the loader was given a form that builds a segment from
those two alone, with no dump on disk. What was left was to make the comparison.

## Consequence

A verifier pays one preprocessed build per `(layer, shape)` in a process: a grid
of 290 segments over 32 layers pays about 32 derivations, not 290, and each is
the cost `expect-root --shape-only` already had. `--expect-pre-roots` becomes a
second opinion rather than the only one -- it compares the same statements
against a table derived outside the run.

Nothing about proving moves: no column, no constraint, no committed value, and
no recorded digest.
