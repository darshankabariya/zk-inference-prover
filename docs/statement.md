# What the proof binds

A proof made by this prover is about one **segment**: one transformer layer
applied to one block of `S` token positions with a context of `C` earlier
positions, or the language-model head applied to one block. A sequence is
proved as a grid of such segments, and `bounds-grid` checks that the segments
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
| 16 | logits into the emitted token | bound on the head segment, absent elsewhere | `--unsound-skip-argmax` |

A head segment has no attention and no MLP, so edges 3 to 11 and 14 are
absent there; its edge 1 is the final norm's reduction, its edge 2 the norm's
output into the head matmul, and its edge 13 the residual it consumed,
committed as the in side of the last layer link.

## What each bound edge means

**The residual stream and the norms (1, 7).** The two RMSNorm reductions read
the residual stream the segment committed, not a copy: the norm's reduction
tree and the residual add are bound by an equality the verifier checks from
the claims. Unbound, the prover could normalise any vector it liked and
present the result as the norm of the residual.

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

**The attention component (4) and the Freivalds checks (12).** These are not
seams but the components themselves: the softmax identities and the
accumulation are constraints of one AIR, and each matmul is a Freivalds check
against the committed weights. They have no skip; a statement without them
is not a statement about a transformer.

**The boundaries (13, 14).** The residual stream leaving a segment and the
keys and values a block writes are committed as their own trees, and
`bounds-grid` compares each such root with the root the consuming segment
committed for its input. Unbound, the segments could each be honest while
the sequence they form is not: a layer fed a residual nobody produced, or a
block attending to a cache nobody wrote.

**The ends (15, 16).** The embedding gadget binds a layer-0 segment's
residual stream to the rows of the embedding matrix that its published ids
select; the argmax binds the token a head segment emits to the maximum of the
logits it computed. Unbound, the sequence could start from any vector, or emit
any token, and `bounds-grid --verify --prompt` would have nothing to compare
the published ids with.

## The unbound configuration

Every skip flag names one of these edges. A run whose `provenance.unsound_skips`
is not empty is a control or a measurement of cost, never a statement about a
forward pass, and `zk-results` refuses to tag a release that cites one. The
digests recorded before the sound defaults reproduce with the skips set, and
`prover/README.md` lists both configurations.
