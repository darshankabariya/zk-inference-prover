# harness

Witness generation (`witness/`), quality evaluation (`eval/`) and the cost
model (`costmodel/`), as one uv package. Run every script as a module from this
directory, after `uv sync`:

    uv run python -m harness.witness.selftest

Every self-test runs on the CPU, and CI runs the six: `witness.selftest` on the
small model, `eval.selftest --cpu` on a model with no weights (the task loaders,
the decode the task runner drives, the scoring and the two report writers),
`witness.extract --selftest`, `eval.baseline.decode_bench --selftest`,
`costmodel.key_tile --selftest` and `costmodel.exact_cells --selftest` on
neither. `eval.selftest` without `--cpu` is the model's own row and needs a card
and the 8B weights.

`costmodel.key_tile --selftest` is the cost model's own: it evaluates the
attention key tile closed form against the measured cell counts of every
recorded layer segment and compares to the cell. Those counts are copied
into `costmodel/key_tile_segments.json`, so the check reads no measured
directory and no other repository, and a one-cell difference is a failure with
no tolerance to widen. The form lives in `costmodel/key_tile.py` rather than in
`cost_model.py` because `cost_model.py` reads its micro-benchmark inputs at
import time and the check has to run where those do not exist.

`costmodel.exact_cells --selftest` is the same check over a whole segment.
`costmodel/exact_cells.py` rebuilds the trace layout component by component
from a run's `params` block and multiplies columns by rows, so it reproduces a
recorded segment's `cells.base`, `cells.interaction` and `cells.fresh` without
a fitted per-unit constant anywhere. Its recorded points are in
`costmodel/exact_chunks.json` on the same terms, and it reads the same
micro-benchmark inputs the key tile form does: none.

The witness package also holds the statement's token generator
(`gen_ids`, a greedy continuation under the proved variant, and `shift_ids`),
the shape table (`shapes`) and the plan writer (`plan`) the results drivers
call at a pinned commit.

A dump names the ids file it was extracted from by digest, in its manifest's
`ids_from`, and the decode that wrote that file names the same digest in its own
note. An extraction into a directory whose manifest names other ids is refused
before the model is loaded: the directory has to be removed, or the dump written
under another `--out`. Re-extracting over the same ids is the ordinary resume
the drivers do, and a manifest is written only when its dump is finished.

A dump also names itself and the calibration it was quantized under, which is
what a record needs to say that a moved proof digest is a moved dump and not a
moved prover (issue #130). `dump_sha256` is the roll-up over the per-array
digests the manifest already lists under `files`: over the arrays alone, so a
`--slim` or a `--share` dump has the identity of the full copy of it, and over
what the manifest claims about them rather than a re-reading of 4.2 GB. The
prover computes the same number over every dump it opens and writes it into its
run JSON, so the two ends of a recorded row name one dump.

`attn_ranges` is the digest of `out/attn_ranges.json`, the K, V and Q ranges of
a forward pass over the calibration slice. Its K half becomes `jk.i32`, the
per-head power-of-two K scale exponents, so it decides the integer scores, the
shape table's `z_max` and the model tree's root -- and it is computed where it
is absent and reused where it is present, which makes it a per-host input that
no commit pins. `--attn-ranges FILE` extracts under a calibration a record
names instead of under whichever one this host has; it is read and never
written, so a file that is not there is a refusal rather than a fresh
calibration under that name.

Scripts that read measured numbers take `--constants PATH` (a zk-results
`constants.json`) and `--artifacts DIR` or `--inputs DIR` for run outputs.
Nothing measured lives in this repository. The cost model reads each of its
cell costs from the constants file where a key names it and from `--inputs DIR`
where none does yet, records which of the two under `provenance` in what it
writes, and prices no term that neither has rather than charging it zero. `witness/text.txt` is the fixed
evaluation corpus (Pride and Prejudice, Project Gutenberg #1342).

## What the extractor refuses

The attention output's rounding identity multiplies `2 mag num`, and it is an
identity over the integers only while that product stays under the field's
`2^31 - 1`. The prover states that as `1 + lo + hi + e_hi <= 31` over the
accumulator's two limbs and the exponent window, with `q_bits + rem_bits <= 31`
beside it, and refuses a statement that does not fit. Both are decided by the
dump's own accumulator and per-head exponents, so `witness/extract.py` evaluates
them as it writes each layer and refuses a dump that would not compose --
naming the block, the layer and the numbers -- instead of leaving that to a
proving run hours later. Each layer prints what its two windows come to and
what is left of the field, the margin: zero margin is a layer that composes and
admits nothing wider, which is where layer 0 sits under the paper's
quantization. A plan that pins wider model-wide widths than a dump's own can
only spend more of the window, so a dump refused here is refused under every
plan. Each layer's manifest records the three numbers the first of the two
windows is decided by -- the accumulator's bit length and the two exponent
extremes -- so the shape table and the plan writer evaluate that window too,
under the pins they write. `witness/extract.py --selftest` checks the
arithmetic, with no model and no card, against the shape the prover pins its own
window on, and checks that the manifest block comes back as the same window.

## What the shape table and the plan writer refuse

`--ao-rem-bits` is one number for a whole chunk loop. The shape table takes it
over every layer of every token block and the drivers put that worst case on the
loop's command line, so a layer is proved at the model-wide value and not at its
own -- and the second of the two bounds above, `q_bits + rem_bits` against 30 in
the plain form and 31 in the `denp2` form, has `q_bits` from the layer's own fold
width. A layer can therefore be pushed past that cap by a pin another layer
needed, although its own divisor range fits inside it. `witness/shapes.py`
evaluates the bound under the pin it writes and `witness/plan.py` under the pin
each line will resolve -- the shared one, unless the line carries one of its own
-- and both refuse, with the layer named and the numbers, rather than leaving
that to the loop's first chunk hours later. The layer's own value is legal, so
the refusal names what admits it: `plan.py --ao-rem-per-line` writes each layer's
own value on that layer's line, and a line's arguments are read after the shared
ones. Every table and every plan also prints the tightest window and its margin.
The recorded 4,096-token set sits at `11 + 20 = 31` of 31, which composes and
admits nothing wider, and its plan is unchanged line for line.

The other of the two bounds, `1 + lo + hi + e_hi`, moves with no pin: it is the
dump's own accumulator and its own exponents. A row that carries those three
numbers is evaluated on it here as well, so a shape that does not compose is
refused before a plan file is written rather than by the loop's first chunk, and
the refusal names what would have to move instead -- a per-line pin recovers no
bit of that window. A row read off a dump extracted before the accumulator's
width was recorded states none, and then that window is **not checked**: a table
and a plan say how many rows that was rather than checking them against a
guessed accumulator, which would refuse a set that composes or admit one that
does not. The recorded 4,096-token table's 288 rows are all of that kind; a
re-extracted set carries the width and gets both windows.

The head line is checked the same way, and only in the plan writer. A head
chunk proves the attention output's rounding exactly as a layer chunk does: its
dump is the layer dump of the layer it was extracted at with a head block added,
so the segment carries that layer's attention columns and the prover builds the
attention-output component and its interaction for it unconditionally. Both
bounds therefore apply to the head line, and the manifest the plan writer
already opens for the head's master table width states every number they need --
the fold's target width, the accumulator's bit length and the two exponent
extremes. So the head line is evaluated at the pin it will resolve, which is its
own `--head-flags` value or the shared one when it carries none, and a head line
over either cap is refused with the numbers beside the layers'. The shape table
has no row for it and should not have one: it goes through a block's layer dumps, and
the pin it writes is the shared one that the head line overrides. A head
manifest that states neither the fold width nor the exponents is not a dump's
and is refused rather than skipped; one written before the accumulator's width
was recorded leaves that window **not checked**, as a layer row of the same age
does. The recorded head configuration sits at `7 + 23 = 30` of 30, which
composes and admits nothing wider.

`witness/shapes.py --selftest` and `witness/plan.py --selftest` check the
arithmetic and every refusal, with no model and no card.

## The unproved baseline

`eval/baseline` measures how fast the same model runs when nothing is proved,
in two backends, and records three numbers per point, each positions per second
over the positions it covers: the prompt's prefill, the decode of the generated
positions, and the two end to end over the whole pass's wall-clock time. None of
the three is an overhead denominator. The paper quotes a cost per proved
position and the reader applies their own mix of prompt and generated positions,
so `--constants PATH` makes the record carry the proved cost per position beside
the unproved cost per position of each of the three, and never one multiple.

The record is written whether or not that comparison could be made. The
measurement is the run's product and carries a provenance block either way; the
comparison is a later reading of it against numbers this repository does not
hold, and a missing constants file, a missing key and a null value are each
recorded as the reason there is none.

    uv run python -m harness.eval.baseline.decode_bench --selftest
    uv run python -m harness.eval.baseline.decode_bench --backend hf \
        --context 4096 --batch 1 --constants PATH

## torch on a GPU host

`uv sync` installs the CPU build of torch. That is the default everywhere: the
container, CI and the selftest. A host with a CUDA 12.9 driver installs the
CUDA build instead, from a second index:

    uv sync --frozen --no-group cpu --group cuda
    uv run --no-group cpu --group cuda python -m harness.witness.extract ...

The two groups conflict, so exactly one build of torch is installed at a time,
and every `uv run` that must reach the card repeats the two flags: without them
uv syncs the environment back to the CPU build. `ZK_DEVICE=cuda` needs the CUDA
build; on the CPU build torch raises `Torch not compiled with CUDA enabled`.
The lockfile pins both builds, so neither `uv sync` resolves anything.

## The vLLM baseline's environment

The two backends do not share one environment. The `hf` backend runs in this
one. vLLM cannot share it (it pins its own CUDA torch), so it gets a separate
virtual environment that is never committed:
`uv venv .venv-vllm && uv pip install --python .venv-vllm vllm`, then run
`decode_bench --backend vllm` with `.venv-vllm/bin/python -m harness.eval.baseline.decode_bench`.
