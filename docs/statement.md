# What the proof binds

A proof made by this prover is about one **segment**: one transformer layer
applied to one block of `S` token positions with a context of `C` earlier
positions, or the language-model head applied to one block. A sequence is
proved as a grid of such segments, and `boundary-grid` checks that the segments
fit together. This page says, edge by edge, what one segment's proof binds and
what an unbound edge would let a dishonest prover do.

Every run prints its own copy of this table as `provenance.binding` (folded
over every proof the run made) and as `result.point.binding` per proof. The
table below is that report for a layer segment under the default flags with a
witness that carries everything the defaults need. The `state` column has
three values: **bound** (the verifier compares the producer's claim with the
consumer's, or both sides are columns of one AIR component), **unbound** (the
consumer reads a value nothing binds to its producer) and **absent** (neither
side is in this statement).

| # | edge | default | turned off by |
|---|---|---|---|
| 1 | residual stream into the first norm's reduction | bound | `--unsound-skip-residual-stream` |
| 2 | first norm's output into the q, k, v matmul inputs | bound | `--unsound-skip-residual-stream` |
| 3 | q, k, v outputs into the attention block | bound | `--unsound-skip-residual-stream`, or a dump without rotation parameters |
| 4 | attention scores and values into the attention output | bound | nothing: it is the attention component itself |
| 5 | attention output into the o matmul input | bound | `--unsound-skip-residual-stream` |
| 6 | o output into the first residual add | bound | `--unsound-skip-residual-stream` |
| 7 | first residual add into the second norm's reduction | bound | `--unsound-skip-residual-stream` |
| 8 | second norm's output into the gate and up matmul inputs | bound | `--unsound-skip-residual-stream` |
| 9 | gate and up outputs into the SwiGLU product | bound | `--unsound-skip-residual-stream` |
| 10 | SwiGLU output into the down matmul input | bound | `--unsound-skip-residual-stream` |
| 11 | down output into the second residual add | bound | `--unsound-skip-residual-stream` |
| 12 | each matmul's input times the pinned weights into its accumulator | bound | nothing: it is the Freivalds check itself |
| 13 | residual stream out of this segment into the next layer's segment | bound | `--unsound-skip-layer-boundary` |
| 14 | keys and values out of this token block into the next block | bound | `--unsound-skip-kv-boundary`, or a dump without its own block |
| 15 | embedding row into the first segment's residual stream | bound on layer 0, absent elsewhere | `--unsound-skip-embedding` |
| 16 | logits into the emitted token | bound on the head segment, absent elsewhere | `--unsound-skip-argmax`, or weakened past bound by `--unsound-argmax-lax` |

A head segment has no attention and no MLP of its own, so edges 3 to 11 and 14
are absent there. Absent says this segment states none of those seams, not that
its witness holds no such matmuls: a head dump carries its layer's eight
matmuls, and the head segment proves each of them against the pinned weights
(edge 12) while binding the input of none and reading the output of none. The
one requantize seam it states is the final norm's output into the head matmul.
Its edge 1 is the final norm's reduction, its edge 2 that seam, and its edge 13
the residual it consumed, committed as the in side of the last layer link.

## What each bound edge means

**The residual stream and the norms (1, 7).** The two RMSNorm reductions read
the residual stream the segment committed, not a copy: the norm's reduction
tree and the residual add are bound by an equality the verifier checks from
the claims. Unbound, the prover could normalise any vector it liked and
present the result as the norm of the residual. Each reduction's second level
publishes its answer, the mean of squares, and that list of equalities is the
only thing that reads it: one entry says the value the norm took the reciprocal
square root of is that answer, and a proof is accepted only if every entry
holds. An answer no entry named would be a value the prover could choose, so
the plan refuses to state one. The lookup tables this arithmetic reads are
padded to a power of two by repeating their last real row, so a row past a
table's real ones carries the same key and the same value as a real one; that a
proof puts no use on such a row is asserted by the prover and checked by
nothing, and it needs no check, because a use there would be answered with the
value of the row it copies.

**The requantize seams (2, 8, 10).** Every matmul consumes an int8
activation. The seam proves that activation is the requantization of the
producer's bf16 output under a per-token scale pinned to that output's own
maximum, and that the Freivalds check multiplied exactly those integers.
Unbound, the prover could feed a matmul any activation, and the matmul's
proof would faithfully multiply the wrong input.

**The dequant outputs (6, 9, 11).** A matmul's accumulator is dequantized to
bf16 and the component that consumes it (a residual add, the SwiGLU product)
reads the same values: the dequant publishes an output claim and the consumer
publishes the claim over what it read, and the two are compared. Unbound, the
prover could add any vector to the residual stream, or gate any product.

**The attention seams (3, 5).** Edge 3 says the q, k, v that the attention
component reads are the projections' outputs, rotated and rounded exactly as
the model does it: the rotation is its own component built from the
statement's rotation parameters, the per-head power-of-two quantize is
another, and the key tiles' columns are bound to their outputs. Edge 5 says
the o matmul's input is the attention output's requantization. Unbound, the
attention would be over values of the prover's choosing while the projections
and the o matmul each prove the wrong thing correctly.

The rotation holds one row per (token, head) and the quantize one per block of
channels inside a head, so both of their row counts have the head count as a
factor. A model whose head count has an odd factor -- Qwen2.5-14B has 40 query
heads -- therefore gives row counts that are a power of two at no block length.
Each component rounds its row count up to the next power of two and puts the
difference at the front as **unconstrained padding**:
rows whose every committed value is zero, on which the rotation is a product
and a sum of zeros and the quantize rounds a zero to a zero, so no constraint
of either component says anything about what they hold. What checks them is the
comparison the real rows are already bound by. Each component folds all of its
rows, the padded ones included, into one weighted sum, and that sum is compared
with the producer's -- the q and k matmuls' dequantized outputs on the way in,
the key tiles' own columns on the way out -- and the producer's runs over the
real values alone. A padded row carries weights no real value carries, so
anything but zero in one changes one side of that comparison and nothing else,
and the comparison fails. Nothing else in the segment reads a padded row. On a
model whose head count is already a power of two there is no padding and the
proof is the one it always was; at 40 heads the rotation holds 64 rows per token
for 40 real ones, and 37.5% of those two components' committed cells is
padding.

**The attention component (4) and the Freivalds checks (12).** These are not
seams but the components themselves: the softmax identities and the
accumulation are constraints of one AIR, and each matmul is a Freivalds check
against the committed weights. They have no skip; a statement without them
is not a statement about a transformer. The causal mask and the marker for the
rows a key grid really holds are both derived from the public parameters rather
than chosen by the prover, and the component subtracts the one from the other,
which is the same as gating on both only while the mask is clear on every row of
unconstrained padding. The code that builds the two columns asserts that, over
exactly those rows and with the predicate that writes them, on the one path that
builds them for either backend.

**The shape tree.** Those two columns are not the prover's, and neither is any
other column the AIR reads as a fact: every selector, the marker for the rows a
key grid really holds, the range tables' row indices, the rotation's angles.
They are committed as a tree of their own, and what says they are the columns
the public parameters give is that the verifier **builds that tree itself** and
compares its root with the root the statement declares. It needs two things to
do it and holds both: the statement's own parameters, and the per-layer
calibration the model manifest publishes beside the model commitment. No weight
enters the shape tree, so no dump is needed. A verifier given no calibration
derives nothing, and then the shape root is asserted by the prover and checked
by nothing -- a run says which of the two it was. The other preprocessed tree is
the model's, and its root is not derivable by anyone but the publisher: it
commits the weights, and the verifier compares it with the published manifest.

**The boundaries (13, 14).** The residual stream leaving a segment and the
keys and values a block writes are committed as their own trees, and
`boundary-grid` compares each such root with the root the consuming segment
committed for its input. Unbound, the segments could each be honest while
the sequence they form is not: a layer fed a residual nobody produced, or a
block attending to a cache nobody wrote.

**The ends (15, 16).** The embedding gadget binds a layer-0 segment's
residual stream to the rows of the embedding matrix that its published ids
select; the argmax binds the token a head segment emits to the **lowest-indexed**
maximum of the logits it computed, which is the rule the model itself decodes
by. Unbound, the sequence could start from any vector, or emit any token, and
`boundary-grid --verify --prompt` would have nothing to compare the published
ids with.

Edge 16 has a second, weaker form. `--unsound-argmax-lax` proves that the
emitted token attains the maximum instead of that it is the first token to
attain it. Where two logits are co-maximal -- 15 of 512 positions at a context
of 4096 on the real continuations -- the weaker statement admits either of them
and the strict one admits only the first, so a lax proof holds for a token the
model would not have emitted. The table marks edge 16 unbound while it is on
and says so in the row, and `provenance.unsound_skips` lists `argmax-lax`: the
run is a control, exactly as one with a skip is.

## The unbound configuration

Every skip flag names one of these edges, and `--unsound-argmax-lax` weakens
edge 16 rather than dropping it. A run whose `provenance.unsound_skips`
is not empty is a control or a measurement of cost, never a statement about a
forward pass, and `zk-results` refuses to tag a release that cites one. The
digests recorded before the sound defaults reproduce with the skips set, and
`prover/README.md` lists both configurations.
