# Cells are counted from the layout wherever a shape can be stated

Date: 2026-09-11. Status: decided by the owner (2026-09-11).

## Decision

The exact formula is used wherever we can use it. Wherever a shape and a
context can be stated as a `params` block, the cost model and the frontier
table COUNT that segment's committed cells from the prover's own trace layout
(`harness/costmodel/exact_cells.py`) instead of pricing them from per-unit
constants read off a measured segment.

Three things are left, and each is named in the output rather than folded into
a residual:

1. The fitted per-unit constants stay for what the prover has no layout for.
   Today that is the mixture-of-experts routing inputs alone --
   `route_selection_cells` and `route_topk_cells` -- which no AIR in this
   project builds and which are priced by analogy with the nearest gadget that
   was measured. The output says `analogy` beside them.
2. The fields of a `params` block that a caller with a model and a context
   cannot know are ONE named assumption block with a stated size in cells. They
   are the calibration a run took off its own witness -- the master exponent
   table's height, the normalisation table's binades and the attention output's
   measured exponent ranges -- plus two knobs of the prover's command line, the
   head grouping and the attention variant.
3. A shape whose segment is not a `params` block is REFUSED with the reason.
   The latent-attention shapes are refused today: their layer has five matmuls
   and not four and their cache is one latent, so the key grids read a width
   the prover has no layout for. A refusal is printed where the row would have
   been; the fitted number is not printed under the same heading.

## Why

A per-unit constant is a ratio measured at one shape, and the frontier table's
whole purpose is to extrapolate to shapes no run has proved. zk-prover #168
computed the same per-unit quantities exactly at four other recorded segments
and they are not constants:

| per-unit constant | at `B8L16` / `B32L16` / `B8L0` | at the output head | at the c512 prefill shape |
| --- | --: | --: | --: |
| `dequant_chain_composed_cells_per_acc` | 16.059524 (+0.00%) | 11.250514 (-29.94%) | 13.196429 (-17.83%) |
| `attnout_composed_cells_per_acc` | 10.375 (+0.00%), but 11.375 (+9.64%) on `B8L0` | 10.375 (+0.00%) | 10.375 (+0.00%) |
| `quant_composed_cells_per_value` | 17.5 (+0.00%) | no quantize component | 26.25 (+50.00%) |
| `requant_composed_cells_per_value` | 18.437544 (+0.00%) | 17.118896 (-7.15%) | 23.450521 (+27.19%) |
| `weight_freivalds_composed_cells_per_weight` | 0.201923 (+0.00%) | 0.155148 (-23.16%) | 2.383929 (+1080.61%) |

The last row is the clearest. What a segment COMMITS for the Freivalds check is
the activations, `pq * 2^log_rows`; the weights are preprocessed. A per-weight
figure is therefore a ratio of two quantities that do not scale together: right
at the 8B layer segment it was read off, and an order of magnitude out at a
128-wide one.

The count that replaces them is not a better fit. It reproduces `cells.base`,
`cells.interaction` and `cells.fresh` to the cell on all 1,872 recorded chunks
in zk-results -- both headline runs, the older run at context 4,096, the memory
probe and every configuration of the reduction ablation -- and on every
single-segment run beside them, over S from 32 to 2,048, C from 128 to 16,384,
both attention variants and both roundings. Nothing in it is fitted and nothing
in it reads a measured input.

## What stays an assumption, and what it costs

A `params` block is not an architecture. Several of its fields are the
calibration a run took off its own witness, and three groups of them move the
cell count:

| what | what it sizes | on a recorded segment |
| --- | --- | --- |
| `log_table` | the master exponent table, `5 * 2^log_table` cells | 0.95% to 10.89% of a segment |
| `kmax` | the normalisation table and the fraction table | 0.22% to 0.39% |
| `ao_acc_hi_bits`, `ao_rem_bits`, `ao_jo_*`, `ao_jv_*` | three range tables and the exponent window | 0.01% to 0.38% |

`log_table` is the one that matters. It is 22, 23, 24 or 25 over the layer
segments of a single token block, and it is chosen so the table covers the
scores the witness actually produced -- a property of the activations, not of
the shape. Everything else in the segment is the architecture.

The assumption is each field at the modal value of the recorded 8B headline
runs. Priced against the 32 layer segments of one token block, that is +0.09%
at context 4,096 and -0.53% at context 16,384; given each segment's own
recorded fields the same count is exact, to the cell, on both blocks. So the
residual a projection carries is the price of a stated assumption and not a
modelling error, which is the whole point of the change.

## How a projection presents it

A projected row is not allowed to hide the assumption inside its number.

- The assumption is one block in the output (`result.assumptions`), one entry
  per field: the value, the run and segment its mode is read from, what it
  sizes in cells and what that is as a share of a segment.
- zk-results records each entry as its own constant, graded `derived` where it
  is the mode of measured values and `asserted` where it is a choice, and every
  projected row names those keys in its `inputs`.
- Every row prints an UPPER BOUND beside its point estimate: the same statement
  with every witness-calibrated field at the widest value the recorded runs
  carry. That is a bound over the evidence and not a theorem, and the output
  says so -- the prover's automatic path caps the exponent table at `2^22` and
  every value above it is the per-layer calibration a model manifest publishes.
- The calibration a run reports is the exact count of one token block's layer
  segments against `cells_per_position.headline.c{4096,16384}.fresh_layers_block`,
  which compares the same segments on both sides.

## Consequence

`harness/costmodel/frontier.py` counts by default and keeps the fitted path
behind `--cells fitted`, so the two can be compared and neither falls back to
the other. `harness/costmodel/cost_model.py` gains the same count under
`cells="exact"`; its default stays fitted because
`harness/eval/predict/predict.py` is a registration, and whether a registered
prediction moves to the counted number is the owner's to decide.

Two numbers move because the count is right where the constant was not. The
committed side of the Freivalds check does not scale with the expert count, so
a mixture-of-experts proof costs what its dense twin costs on the committed
side and the expert count lands on the preprocessed trace, which the measured
rates do not count. And the key tiles are committed per attention head rather
than per key/value group, which the count carries at every shape rather than at
the one it was read off.
