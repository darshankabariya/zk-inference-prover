# harness

Witness generation (`witness/`), quality evaluation (`eval/`) and the cost
model (`costmodel/`), as one uv package. Run every script as a module from this
directory, after `uv sync`:

    uv run python -m harness.witness.selftest

Every self-test runs on the CPU, and CI runs the four: `witness.selftest` on the
small model, `eval.selftest --cpu` on a model with no weights (the task loaders,
the decode the task runner drives, the scoring and the two report writers),
`witness.extract --selftest` and `eval.baseline.decode_bench --selftest` on
neither. `eval.selftest` without `--cpu` is the model's own row and needs a card
and the 8B weights.

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

Scripts that read measured numbers take `--constants PATH` (a zk-results
`constants.json`) and `--artifacts DIR` or `--inputs DIR` for run outputs.
Nothing measured lives in this repository. `witness/text.txt` is the fixed
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
plan. `witness/extract.py --selftest` checks the arithmetic, with no model and
no card, against the shape the prover pins its own window on.

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
