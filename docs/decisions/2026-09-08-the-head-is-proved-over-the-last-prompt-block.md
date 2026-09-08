# The output head is proved over the last prompt block as well

Date: 2026-09-08. Status: decided by the owner (2026-09-08).

## Decision

An end-to-end statement carries a head segment over the last block of the
prompt as well as over the generated block, so that every emitted token is
bound to a proved argmax (edge 16). The first generated token's id is the
argmax of the logits at the last prompt position; with a head segment over
the generated block only, that one id was compared against nothing.

## Why

Binding the emitted token is what makes the statement end to end rather than
a statement about layers. One unbound id out of 512 is a soundness gap, not a
rounding of the claim: a prover could emit any first token and prove the rest
honestly. The cost is one more head segment per statement, about a minute of
proving out of hours, and one more head dump.

## Consequence

The zk-results plans name both blocks for the head; the link pin and the grid's
`--verify --prompt` check cover every generated position. The run at context
16,384 in flight on 2026-09-08 carries the generated block's head only and is
recorded with the gap stated in one sentence; a head segment over its last
prompt block is added as the run's last step where the driver's restart path
allows it, else the campaign is run again. The recorded digests do not move:
nothing in the prover changes.
