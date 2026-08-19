#!/bin/sh
# Everything B1 measures, in order.  ~20 min on 4 CPU cores.
# Run from anywhere:  sh b1/run_all.sh
set -e
cd "$(dirname "$0")/.."

uv run python b1/selftest.py                              # faithfulness, H2, determinism
uv run python b1/ppl.py --all --chunks 40                 # quality, 20k tokens
uv run python b1/outliers.py                              # why static W8A8 loses
uv run python b1/gen_witness.py --prompt 32 --decode 32   # the witness + manifest
uv run python b1/gen_witness.py --check                   # bit-identical regeneration
uv run python b1/ppl.py --report
