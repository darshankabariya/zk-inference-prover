#!/bin/sh
# E4 (briefs/e4-predict.md section 7), box side, H200.  The run-book, no decisions:
#
#   sh e4_box.sh download     # the 14B bf16 weights into $HF_HOME (~29.5 GB)
#   sh e4_box.sh shape        # step 2: the loaded config's 7-tuple == the registered one
#   sh e4_box.sh extract      # step 3: 48 layers, b10-final, --chunk 512 --glue --corner --slim, C=4096
#   sh e4_box.sh manifests    # step 5: every dump names the real model and the registered shape
#   sh e4_box.sh prove        # step 4: one composed layer chunk per layer, reference arm (SimdBackend)
#   sh e4_box.sh all          # download shape extract manifests prove, in that order; stops on the first failure
#   sh e4_box.sh bg all       # the same, detached; poll ~/e4/e4.status and ~/e4/e4.done
set -u

ROOT=$HOME/e4
B7=$HOME/b7
B8=$HOME/b8
Z=$HOME/z
EX=$B8/experiments
PY=$B8/.venv/bin/python
HF=$B8/.venv/bin/hf
MODEL=Qwen/Qwen2.5-14B-Instruct
SHAPE=qwen2p5_14b
CTX=4096
S=512
NLAY=48
W=$B8/w14b/c${CTX}_final          # the dumps: witness_L<k>
OUT=$ROOT/out
LOG=$ROOT/logs
STATUS=$ROOT/e4.status
EXBATCH=${E4_EXBATCH:-6}

export HF_HOME=$HOME/hf
export B8_DATA=$B8/data
export B1_MODEL=$MODEL
export B1_OUT=$ROOT/b1out          # a FRESH cache: tokens.npz / attn_ranges.json for this tokenizer and model
export B1_DEVICE=cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
# the synthetic-fallback trap (section 7 step 5): never set here
unset B7_SYNTH_DIMS

# the prover's env: the first 20 lines of ~/z/boxenv.sh, verbatim in effect
export B7_ROOT=$B7 B8_ROOT=$B8
export RUSTUP_HOME=$B7/rustup CARGO_HOME=$B7/cargo
for d in "$HOME/cuda-12.9" /usr/local/cuda-12.9 /usr/local/cuda; do
  [ -x "$d/bin/nvcc" ] && CUDA_HOME=$d && break
done
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_PATH=$CUDA_HOME CUDACXX=$CUDA_HOME/bin/nvcc
[ -x "$HOME/gcc14/bin/g++" ] && export CUDAHOSTCXX=$HOME/gcc14/bin/g++
export PATH=$B8/.venv/bin:$CARGO_HOME/bin:$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export B7_GEN=$Z/gen
export CARGO_TARGET_DIR=$Z/z-simd
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-16}
SIMD=$Z/z-simd/release/b65

# The design point: o_h200.sh's EVENPT + --tight-table (dd_box.sh's note), the
# band split for nh = 40 (section 7 step 4), bounds on.  No --kmax / --ao-rem-bits
# pins: each layer takes its own data-derived values (recorded per chunk).
PT="--s $S --ctx $CTX --hd 128 --bands 32,8 --round even --numbits 9 --z-hi-bits 7 \
--variants intexp --tight-table --bounds"

mkdir -p $ROOT $OUT $LOG $B1_OUT
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $STATUS; echo "$*"; }
fail() { st "e4: $1 FAIL ${2:-}"; exit 1; }
pass() { st "e4: $1 PASS ${2:-}"; }

# ---------------------------------------------------------------- download --
download() {
  st "download: $MODEL -> $HF_HOME (disk before: $(df -h / | tail -1 | awk '{print $4}') avail)"
  $HF download $MODEL > $LOG/download.log 2>&1
  rc=$?
  [ $rc -eq 0 ] || fail download "hf download rc=$rc, see $LOG/download.log"
  snap=$(ls -d $HF_HOME/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/*/ | head -1)
  st "download: snapshot $snap"
  st "download: $(du -shL $snap | awk '{print $1}') in the snapshot; disk after: $(df -h / | tail -1 | awk '{print $4}') avail"
  ls -lL $snap > $LOG/snapshot_ls.txt
  (cd $snap && sha256sum -L config.json model.safetensors.index.json) > $LOG/snapshot_sha.txt 2>&1
  pass download
}

# ------------------------------------------------------------------- shape --
shape() {
  st "shape: reading the loaded config's 7-tuple"
  $PY - <<'EOF' > $OUT/shape.json 2> $LOG/shape.err
import json, os
from transformers import AutoConfig
m = os.environ["B1_MODEL"]
c = AutoConfig.from_pretrained(m)
hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
t = [c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.num_key_value_heads,
     hd, c.intermediate_size, c.vocab_size]
reg = [48, 5120, 40, 8, 128, 13824, 152064]
print(json.dumps(dict(model=m, tuple=t, registered=reg, equal=(t == reg),
                      tie_word_embeddings=bool(getattr(c, "tie_word_embeddings", False)),
                      model_type=c.model_type, torch_dtype=str(getattr(c, "torch_dtype", None)),
                      rope_theta=getattr(c, "rope_theta", None),
                      max_position_embeddings=c.max_position_embeddings,
                      transformers_version=getattr(c, "transformers_version", None)), indent=1))
EOF
  rc=$?
  [ $rc -eq 0 ] || fail shape "AutoConfig rc=$rc, see $LOG/shape.err"
  cat $OUT/shape.json | tee -a $STATUS
  eq=$($PY -c "import json;print(json.load(open('$OUT/shape.json'))['equal'])")
  [ "$eq" = "True" ] || fail shape "7-tuple differs from the registered one -- do NOT adjust; re-run predict.py with the actual tuple on the desk"
  pass shape "7-tuple == (48, 5120, 40, 8, 128, 13824, 152064)"
}

# ----------------------------------------------------------------- extract --
batch() {   # batch LO HI  -> extractor rc
  lo=$1; hi=$2
  ls_=$($PY -c "print(','.join(str(i) for i in range($lo, $hi)))")
  st "extract: layers $ls_ (batch $EXBATCH)"
  ( cd $EX && . ./b8/env.sh && \
    env | grep -E '^(B1_|B8_|HF_HOME|B7_SYNTH)' && \
    $PY b65/extract.py --tokens $((CTX + S)) --chunk $S --layers "$ls_" \
      --spec b10-final --glue --corner --slim \
      --out $W/witness ) >> $LOG/extract.log 2>&1
  rc=$?
  st "extract: layers $ls_ rc=$rc"
  return $rc
}

extract() {
  mkdir -p $W
  st "extract: $MODEL C=$CTX S=$S, spec b10-final, --glue --corner --slim, corpus ids (B1_OUT=$B1_OUT), B7_SYNTH_DIMS=${B7_SYNTH_DIMS:-<unset>}"
  : > $LOG/extract.log
  n=$EXBATCH; b=0
  while [ $b -lt $NLAY ]; do
    e=$((b + n)); [ $e -gt $NLAY ] && e=$NLAY
    if batch $b $e; then
      b=$e
    else
      if [ $n -le 1 ]; then fail extract "layer $b fails at batch 1, see $LOG/extract.log"; fi
      n=$((n / 2))
      st "extract: retrying from layer $b at batch $n"
    fi
  done
  # the shape table over the 48 dumps (the seq5 pattern), for the record
  ( cd $EX && $PY b10/gpu/shapes32.py $W/witness_L $NLAY > $LOG/shapes.log 2>&1 ) || st "extract: shapes32.py rc=$? (informational)"
  st "extract: done, $(du -sh $W | awk '{print $1}') in $W; disk after: $(df -h / | tail -1 | awk '{print $4}') avail"
  pass extract
}

# --------------------------------------------------------------- manifests --
manifests() {
  $PY - "$W" "$NLAY" "$MODEL" <<'EOF' > $OUT/manifests.json 2> $LOG/manifests.err
import json, sys, os
w, n, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows, bad = [], []
for k in range(n):
    p = f"{w}/witness_L{k}/manifest.json"
    if not os.path.exists(p):
        bad.append(f"L{k}: no manifest"); continue
    m = json.load(open(p))
    r = dict(layer=k, model=m.get("model"), heads=m.get("heads"), s=m.get("s"), c=m.get("c"),
             hd=m.get("hd"), spec=m.get("spec"), corner=m.get("corner"),
             num_bits=m.get("num_bits"), qk_bits=m.get("qk_bits"),
             master_table_bits=m.get("master_table_bits"),
             deq_variants=m.get("deq_variants"),
             mats={a: (b.get("n_in"), b.get("n_out"), b.get("tokens")) for a, b in m.get("accumulators", {}).items()},
             glue=("glue" in m) or any(x.startswith("glue") for x in m))
    rows.append(r)
    want = dict(model=model, heads=40, s=512, c=4096, hd=128, spec="b10-final", corner=True)
    for kk, vv in want.items():
        if r[kk] != vv: bad.append(f"L{k}: {kk}={r[kk]!r} != {vv!r}")
print(json.dumps(dict(ok=not bad, bad=bad, rows=rows), indent=1))
sys.exit(1 if bad else 0)
EOF
  rc=$?
  [ $rc -eq 0 ] || fail manifests "see $OUT/manifests.json / $LOG/manifests.err"
  pass manifests "$NLAY dumps name $MODEL, heads 40, s 512, c 4096, hd 128, b10-final, corner"
}

# ------------------------------------------------------------------- prove --
prove_one() {   # prove_one K
  k=$1
  wd=$W/witness_L$k
  [ -d "$wd" ] || fail prove "no witness at $wd"
  st "prove: L$k start ($PT --witness $wd)"
  /usr/bin/time -f 'time: wall %e s, user %U s, sys %S s, maxrss %M KiB' -o $OUT/L$k.time \
    env RAYON_NUM_THREADS=$RAYON_NUM_THREADS $SIMD compose $PT --witness $wd \
      > $OUT/L$k.json 2> $OUT/L$k.err
  rc=$?
  acc=$(grep -o '"accepts": [a-z]*' $OUT/L$k.json 2>/dev/null | head -1)
  fresh=$($PY -c "import json;print(json.load(open('$OUT/L$k.json'))['result']['point']['cells']['fresh'])" 2>/dev/null)
  sha=$($PY -c "import json;print(json.load(open('$OUT/L$k.json'))['result']['point']['proof_sha256'][:16])" 2>/dev/null)
  rss=$($PY -c "import json;print(json.load(open('$OUT/L$k.json'))['result']['point']['peak_rss_bytes'])" 2>/dev/null)
  tt=$(grep '^tight-table' $OUT/L$k.err | head -1)
  st "prove: L$k rc=$rc $acc fresh=$fresh digest=$sha peak_rss_bytes=$rss | $tt | $(cat $OUT/L$k.time 2>/dev/null)"
  [ $rc -eq 0 ] || fail prove "L$k rc=$rc, see $OUT/L$k.err"
  [ "$acc" = '"accepts": true' ] || fail prove "L$k did not accept"
  [ -n "$fresh" ] && [ "$fresh" -gt 0 ] || fail prove "L$k cells.fresh not > 0 (the vacuous-OK trap)"
}

prove() {
  st "prove: reference arm $SIMD ($(md5sum $SIMD | cut -c1-32)), RAYON_NUM_THREADS=$RAYON_NUM_THREADS, B7_SYNTH_DIMS=${B7_SYNTH_DIMS:-<unset>}"
  lo=${1:-0}; hi=${2:-$NLAY}
  k=$lo
  while [ $k -lt $hi ]; do
    if [ -s $OUT/L$k.json ] && grep -q '"accepts": true' $OUT/L$k.json; then
      st "prove: L$k already accepted, skipping"
    else
      prove_one $k
    fi
    k=$((k + 1))
  done
  pass prove "$lo..$((hi - 1))"
}

all() {
  download && shape && extract && manifests && prove
  rc=$?
  st "all: rc=$rc"
  echo $rc > $ROOT/e4.done
  return $rc
}

case "${1:-}" in
  download|shape|extract|manifests|all) "$1" ;;
  prove) shift; prove "$@" ;;
  bg) shift; rm -f $ROOT/e4.done
      setsid nohup sh "$0" "$@" > $LOG/bg_$1.out 2>&1 < /dev/null &
      echo "detached: $!" ;;
  *) echo "usage: $0 {download|shape|extract|manifests|prove [LO HI]|all|bg STEP}" >&2; exit 2 ;;
esac
