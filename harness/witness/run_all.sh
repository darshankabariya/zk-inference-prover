#!/bin/sh
# Everything the witness package measures, in order, on the small default model.
# ~20 min on 4 CPU cores.  Run from anywhere:  sh witness/run_all.sh
set -e
cd "$(dirname "$0")/.."

uv run python -m harness.witness.selftest                              # faithfulness, headroom, determinism
uv run python -m harness.witness.ppl --all --chunks 40                 # quality, 20k tokens
uv run python -m harness.witness.outliers                              # why static W8A8 loses
uv run python -m harness.witness.gen_witness --prompt 32 --decode 32   # the witness + manifest
uv run python -m harness.witness.gen_witness --check                   # bit-identical regeneration
uv run python -m harness.witness.ppl --report
