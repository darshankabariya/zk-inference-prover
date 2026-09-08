# The rotation gadget pads tokens times heads to a power of two

Date: 2026-09-08. Status: decided by the owner (2026-09-08).

## Decision

The prover supports models whose attention head count has an odd factor. The
rotation gadget, which today requires the block length times the head count to
be a power of two and refuses a layer otherwise, pads its rows to the next
power of two with rows that the constraints hold trivially on and that no
other component reads.

## Why

The prover is meant to prove more models than the one the paper measures. A
head count with a factor of five (Qwen2.5-14B has 40 query heads) can never
tile a power of two at any block length, so without padding a whole model
class is out of reach (zk-prover #73), and the registered-prediction campaign
has no shape to run on. Padding costs committed cells in proportion to the
gap to the next power of two (64 rows for 40 heads: 60% more rotation rows on
that model, none on the 8B) and changes nothing on a model that already tiles.

## Consequence

The gadget's row count becomes a statement parameter derived from the shape,
not an assertion; the binding table names the padded rows as unconstrained
padding, checked to be zero. The recorded digests do not move: every recorded
shape tiles exactly and takes the same path as before. The change is its own
issue with the digest gate on both backends.
