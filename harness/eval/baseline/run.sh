#!/bin/sh
# E2h on the box: the unproved 8B decode-throughput baseline.
#
# The proved side is already frozen (experiments/paper/constants.json, seq5,
# H200, 33 chunks, 512 tokens, greedy, batch 1).  This measures the DENOMINATOR
# on the SAME card, at the SAME two contexts, with the SAME token count, in two
# backends -- and then prints the ladder.
#
#   sh e2/run.sh preflight   # 0: card, interpreter, weights, backends. Free.
#   sh e2/run.sh hf          # 1: the transformers baseline
#   sh e2/run.sh vllm        # 2: the production baseline (its own venv)
#   sh e2/run.sh ladder      # 3: the table + the freeze rows
#   sh e2/run.sh verify      # 4: did every expected artifact land, quotable?
#   sh e2/run.sh all         # 0-4 in order
#
# RUN IT DETACHED.  A foreground ssh dies with the session and takes the job:
#
#   setsid nohup sh e2/run.sh all > e2/out/run.log 2>&1 < /dev/null &
#   ...then poll, in short separate ssh calls:
#   tail -5 e2/out/e2.status
#
# Do NOT trust this script's exit code alone -- `b8/run.sh`'s `tee` pipeline
# once reported ok while every step died.  `verify` re-reads the artifacts and
# is the thing to believe.
#
# Nothing here installs anything system-wide.
set -e
cd "$(dirname "$0")/.."

# --- interpreter ------------------------------------------------------------
# The H200 has no `uv` and its bare `python3` has no torch; the interpreter is
# ~/b8/.venv/bin/python (briefs/README.md, lane-4 operational notes).  The desk
# has uv and no venv.  Resolve, do not assume.
if [ -n "$PYTHON" ]; then
  PY="$PYTHON"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
elif [ -x "$HOME/b8/.venv/bin/python" ]; then
  PY="$HOME/b8/.venv/bin/python"
else
  PY="python3"
fi
VLLM_PY=${VLLM_PYTHON:-.venv-vllm/bin/python}

. ./b8/env.sh                       # B1_MODEL = unsloth/Llama-3.1-8B-Instruct
export TOKENIZERS_PARALLELISM=false
E2_REVISION=${E2_REVISION:-4699cc75b550f9c6f3173fb80f4703b62d946aa5}
export E2_REVISION
OUT=${E2_OUT:-e2/out}
export E2_OUT="$OUT"
mkdir -p "$OUT"
STATUS="$OUT/e2.status"

say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$STATUS"; }

BENCH="e2/decode_bench.py --model $B1_MODEL --revision $E2_REVISION \
--device cuda --dtype bfloat16 --new-tokens 512"
# The batch sweep. It stops itself at the first OOM and records where.
SWEEP=${E2_SWEEP:-"--batch 2 --batch 4 --batch 8 --batch 16 --batch 32 --batch 64 --batch 128"}

case ${1:-all} in
  preflight)
    say "preflight: interpreter = $PY"
    rc=0
    $PY -c "import torch,transformers,sys; \
p=torch.cuda.get_device_properties(0); \
print('card        ', p.name, p.total_memory>>20, 'MiB'); \
print('torch       ', torch.__version__, 'cuda', torch.version.cuda); \
print('transformers', transformers.__version__); \
sys.exit(0 if 'H200' in p.name else 7)" > "$OUT/preflight.txt" 2>&1 || rc=$?
    tee -a "$STATUS" < "$OUT/preflight.txt"
    if [ "$rc" -ne 0 ]; then
      say "preflight FAILED (rc=$rc): no H200. E2's measurement is pinned to the card"
      say "  the headline runs on (eval-plan section 7); an H100 run is not quotable"
      say "  and the harness will say so. E2_ALLOW_ANY_CARD=1 proceeds anyway."
      [ -n "$E2_ALLOW_ANY_CARD" ] || exit 7
    fi
    $PY e2/decode_bench.py --probe > "$OUT/probe.json" 2>&1 || true
    tee -a "$STATUS" < "$OUT/probe.json"
    say "preflight: ok"
    ;;
  hf)
    say "hf: start"
    $PY $BENCH --backend hf --context 4096  --batch 1 --reps 3 --warmup 1
    $PY $BENCH --backend hf --context 16384 --batch 1 --reps 3 --warmup 1
    $PY $BENCH --backend hf --context 4096  $SWEEP --reps 1 --warmup 1
    $PY $BENCH --backend hf --context 16384 $SWEEP --reps 1 --warmup 1
    say "hf: done"
    ;;
  vllmenv)
    # vLLM pulls CUDA torch and ~27 nvidia-* wheels: it cannot co-exist with
    # experiments/pyproject.toml's CPU torch pin, so it gets its OWN venv and is
    # NOT a project dependency. Nothing system-wide either way.
    # `--default-index` MATTERS: run from experiments/, uv reads this project's
    # `[tool.uv.index] pytorch-cpu` pin, no modern vLLM wheel can satisfy its
    # CUDA torch requirement, and the resolver backtracks to the vllm==0.1.3
    # sdist and tries to BUILD it (measured on the desk, 2026-09-01).
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 .venv-vllm
      uv pip install --python .venv-vllm --default-index https://pypi.org/simple vllm
    else
      python3 -m venv .venv-vllm
      .venv-vllm/bin/pip install --upgrade pip
      .venv-vllm/bin/pip install vllm
    fi
    say "vllmenv: $($VLLM_PY -c 'import vllm;print(vllm.__version__)')"
    ;;
  vllm)
    say "vllm: start"
    $VLLM_PY $BENCH --backend vllm --context 4096  --batch 1 --reps 3 --warmup 1
    $VLLM_PY $BENCH --backend vllm --context 16384 --batch 1 --reps 3 --warmup 1
    $VLLM_PY $BENCH --backend vllm --context 4096  $SWEEP --reps 1 --warmup 1
    $VLLM_PY $BENCH --backend vllm --context 16384 $SWEEP --reps 1 --warmup 1
    say "vllm: done"
    ;;
  ladder)
    $PY e2/ladder.py --outdir "$OUT" --out "$OUT/ladder.json" --markdown \
      | tee "$OUT/ladder.md"
    $PY e2/ladder.py --outdir "$OUT" --emit-constants > "$OUT/constants_rows.json"
    say "ladder: $OUT/ladder.json, $OUT/ladder.md, $OUT/constants_rows.json"
    ;;
  verify)
    # Believe this, not the exit codes above.  And do NOT pipe it into `tee`:
    # that is precisely how b8/run.sh reported `ok` six times while every step
    # died -- the pipeline's status is tee's.  Redirect, capture, then print.
    rc=0
    $PY - "$OUT" > "$OUT/verify.txt" 2>&1 <<'PYVERIFY' || rc=$?
import json, sys, glob, os
out = sys.argv[1]
want = [(be, c) for be in ("hf", "vllm") for c in (4096, 16384)]
bad = 0
for be, c in want:
    hits = [p for p in glob.glob(os.path.join(out, "*.json"))
            if f"_{be}_c{c}_b1_n512" in os.path.basename(p)]
    if not hits:
        print(f"MISSING  {be} C={c} batch 1"); bad += 1; continue
    d = json.load(open(hits[0]))
    lf = d["like_for_like"]
    m = d["measured"]
    ok = lf["quotable"] and not d["run"]["synthetic"]
    print(f"{'ok      ' if ok else 'NOT-OK  '}{be} C={c} b1  "
          f"{m['tok_s_decode_per_stream']['median']:8.3f} tok/s decode  "
          f"{m['tok_s_e11_all_positions_per_stream']['median']:9.3f} pos/s e11  "
          f"[{lf['verdict']}]")
    for b in lf["blocking"]:
        print("           blocking:", b)
    bad += not ok
    sweep = [p for p in glob.glob(os.path.join(out, "*.json"))
             if f"_{be}_c{c}_b" in os.path.basename(p) and "_b1_" not in os.path.basename(p)]
    if not sweep:
        print(f"MISSING  {be} C={c} batch sweep"); bad += 1
sys.exit(1 if bad else 0)
PYVERIFY
    tee -a "$STATUS" < "$OUT/verify.txt"
    say "verify: exit $rc  (0 = every expected artifact landed and is quotable)"
    exit $rc
    ;;
  all)
    sh e2/run.sh preflight
    sh e2/run.sh hf
    sh e2/run.sh vllm
    sh e2/run.sh ladder
    sh e2/run.sh verify
    ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
