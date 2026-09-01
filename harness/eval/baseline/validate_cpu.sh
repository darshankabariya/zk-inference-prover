#!/bin/sh
# E2h's desk validation: the harness end to end with no GPU anywhere.  It proves
# the mechanics, the JSON schema and the like-for-like checks -- NOT a number
# anyone may quote.  Every run it produces is stamped NOT-LIKE-FOR-LIKE on model
# and card, which is the point.
#
#   sh e2/validate_cpu.sh          # from experiments/. ~3 minutes, ~2 GB RSS.
#
# The coordinating machine is ~11.9 GB / 4 cores with several agents on it and
# has been OOM-killed by exactly this kind of job.  So the default is DELIBERATELY
# small: two tiny models, short contexts, and a mock backend for everything that
# only scale would otherwise exercise.  `E2_VALIDATE=heavy` runs the old
# Qwen-0.5B-at-C=4096 path instead; it wants a machine with room.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-uv run python}

# 1. the checker has teeth, with no model and no GPU at all
$PY e2/decode_bench.py --selftest
$PY e2/decode_bench.py --probe

# 2. the real HF path, on models small enough to be safe here.  distilgpt2 runs
#    the canonical 512-token count; Qwen exercises a second tokenizer family.
E2_OUT=e2/out $PY e2/decode_bench.py --backend hf --model distilgpt2 \
  --revision main --dtype float32 --device cpu --tag distilgpt2 \
  --context 256 --new-tokens 32 --batch 1 --batch 2 --reps 2 --warmup 1
E2_OUT=e2/out $PY e2/decode_bench.py --backend hf --model distilgpt2 \
  --revision main --dtype float32 --device cpu --tag distilgpt2 \
  --context 256 --new-tokens 512 --batch 1 --reps 1 --warmup 1
E2_OUT=e2/out $PY e2/decode_bench.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision main --dtype float32 --device cpu --tag qwen05b \
  --context 1024 --new-tokens 64 --batch 1 --reps 1 --warmup 1

if [ "$E2_VALIDATE" = "heavy" ]; then
  E2_OUT=e2/out $PY e2/decode_bench.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --revision main --dtype float32 --device cpu --tag qwen05b \
    --context 4096 --new-tokens 512 --batch 1 --reps 2 --warmup 1
  E2_OUT=e2/out $PY e2/decode_bench.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
    --revision main --dtype float32 --device cpu --tag qwen05b \
    --context 16384 --new-tokens 512 --batch 1 --reps 1 --warmup 0
fi

# 3. the parts only an 8B box run would otherwise reach -- the 512-token count at
#    both real contexts, the batch sweep and its OOM stop, both counting
#    conventions, the whole ladder -- on the SYNTHETIC backend.  These artifacts
#    live in e2/out/mock/ and the ladder drops them by default; see step 5.
E2_OUT=e2/out/mock $PY e2/decode_bench.py --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 \
  --context 4096 --context 16384 --batch 1 --reps 2 --warmup 1
E2_OUT=e2/out/mock $PY e2/decode_bench.py --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 --context 4096 \
  --batch 2 --batch 4 --batch 8 --batch 16 --batch 32 --batch 64 --batch 128 \
  --reps 1 --warmup 0
E2_OUT=e2/out/mock $PY e2/decode_bench.py --backend mock --model MOCK/none \
  --device cpu --dtype bfloat16 --tag mock --new-tokens 512 --context 16384 \
  --batch 2 --batch 4 --batch 8 --batch 16 --batch 32 --reps 1 --warmup 0

# 4. the real-model ladder: no run in e2/out is at a frozen context on an H200,
#    so every cell that needs one must come out null and say why.
$PY e2/ladder.py --outdir e2/out --out e2/out/ladder.json --markdown \
  | tee e2/out/ladder.md

# 5. the ladder driven all the way through, on synthetic seconds.  The header
#    says NOT PUBLISHABLE and the freeze rows come out `NOT_QUOTABLE.`-prefixed
#    with null grades -- that refusal is what is being validated.
$PY e2/ladder.py --outdir e2/out/mock --allow-synthetic \
  --out e2/out/mock/ladder.json --markdown | tee e2/out/mock/ladder.md
$PY e2/ladder.py --outdir e2/out/mock --allow-synthetic --emit-constants \
  > e2/out/mock/constants_rows.json

echo "E2VAL DONE"
