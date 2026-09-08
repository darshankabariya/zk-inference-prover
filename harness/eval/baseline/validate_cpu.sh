#!/bin/sh
# E2h's desk validation: the harness end to end with no GPU anywhere.  It proves
# the mechanics, the JSON schema and the like-for-like checks -- NOT a number
# anyone may quote.  Every run it produces is stamped NOT-LIKE-FOR-LIKE on model
# and card, which is the point.
#
#   sh eval/baseline/validate_cpu.sh CONSTANTS.json   # from harness/. ~3 minutes, ~2 GB RSS.
#
# The coordinating machine is ~11.9 GB / 4 cores with several agents on it and
# has been OOM-killed by exactly this kind of job.  So the default is DELIBERATELY
# small: two tiny models, short contexts, and a mock backend for everything that
# only scale would otherwise exercise.  `ZK_BASELINE_VALIDATE=heavy` runs the old
# Qwen-0.5B-at-C=4096 path instead; it wants a machine with room.
set -e
cd "$(dirname "$0")/../.."
CONSTANTS=${1:?path to a zk-results constants.json}
OUT=${ZK_BASELINE_VALIDATE_OUT:-validate-out}
mkdir -p "$OUT"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-uv run python}
BENCH="$PY -m harness.eval.baseline.decode_bench --constants $CONSTANTS"

# 1. the checker has teeth, with no model and no GPU at all
$BENCH --selftest
$BENCH --probe

# 2. the real HF path, on models small enough to be safe here.  distilgpt2 runs
#    the canonical 512-token count; Qwen exercises a second tokenizer family.
ZK_BASELINE_OUT=$OUT $BENCH --backend hf --model distilgpt2 \
  --revision main --dtype float32 --device cpu --tag distilgpt2 \
  --context 256 --new-tokens 32 --batch 1 --batch 2 --reps 2 --warmup 1
ZK_BASELINE_OUT=$OUT $BENCH --backend hf --model distilgpt2 \
  --revision main --dtype float32 --device cpu --tag distilgpt2 \
  --context 256 --new-tokens 512 --batch 1 --reps 1 --warmup 1
ZK_BASELINE_OUT=$OUT $BENCH --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision main --dtype float32 --device cpu --tag qwen05b \
  --context 1024 --new-tokens 64 --batch 1 --reps 1 --warmup 1

if [ "$ZK_BASELINE_VALIDATE" = "heavy" ]; then
  ZK_BASELINE_OUT=$OUT $BENCH --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --revision main --dtype float32 --device cpu --tag qwen05b \
    --context 4096 --new-tokens 512 --batch 1 --reps 2 --warmup 1
  ZK_BASELINE_OUT=$OUT $BENCH --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --revision main --dtype float32 --device cpu --tag qwen05b \
    --context 16384 --new-tokens 512 --batch 1 --reps 1 --warmup 0
fi

# 3. the parts only an 8B box run would otherwise reach -- the 512-token count at
#    both real contexts, the batch sweep and its OOM stop, both counting
#    conventions -- on the SYNTHETIC backend.  These artifacts live in $OUT/mock/,
#    away from the real-model runs; every one of them is stamped `synthetic: true`,
#    which is blocking on the like-for-like check, so none of them can be read as
#    a measurement.
ZK_BASELINE_OUT=$OUT/mock $BENCH --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 \
  --context 4096 --context 16384 --batch 1 --reps 2 --warmup 1
ZK_BASELINE_OUT=$OUT/mock $BENCH --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 --context 4096 \
  --batch 2 --batch 4 --batch 8 --batch 16 --batch 32 --batch 64 --batch 128 \
  --reps 1 --warmup 0
ZK_BASELINE_OUT=$OUT/mock $BENCH --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 --context 16384 \
  --batch 2 --batch 4 --batch 8 --batch 16 --batch 32 --reps 1 --warmup 0

echo "E2VAL DONE"
