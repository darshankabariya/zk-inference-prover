#!/bin/sh
# B9 step 1 on the H100: the two layer fixes, all 24 Qwen-0.5B layers, and the
# real Llama-3.1-8B layer at S=512, C=16K.
#
# Runs ON the box, after B7b's `m5_run.sh` chain has built the trees.
#
#   sh b9_run.sh patch      # b9_patch.py + b9_diag.py onto all three trees, rebuilt
#   sh b9_run.sh extract    # the 24 Qwen layers, with the fold's rectangle scale
#   sh b9_run.sh layers     # all 24, each at its own master_table_bits
#   sh b9_run.sh rungs      # rung 1: the SIMD reference's digest for five spreads
#   sh b9_run.sh emit8b     # the 8B shape's fraction and constraint kernels
#   sh b9_run.sh install8b  # ... into the fork, libstwo_cuda rebuilt
#   sh b9_run.sh shapes8b   # ... the shape tables into the proving tree
#   sh b9_run.sh prove8b    # the real 8B layer, CUDA, with the three boundaries
#   sh b9_run.sh ref8b      # the same layer on the SIMD reference (rung 1)
#
# No `set -e`: a step that fails is recorded and the chain goes on.
ROOT=${B7_ROOT:-/root/b7}
GEN=${B7_GEN:-$ROOT/gen}
OUT=${B9_OUT:-$ROOT/out/b9}
STEP=${1:-all}
export RUSTUP_HOME=$ROOT/rustup CARGO_HOME=$ROOT/cargo PATH=$ROOT/cargo/bin:$PATH
export B7_GEN=$GEN
mkdir -p $OUT
B=$ROOT/b65_gpu/target/release/b65
R=$ROOT/b65_ref/target/release/b65
DEV="B7_M4_FILL=1 B7_M4_WFILL=1 B7_M4_TABDEV=1 B7_M4_ACC=1 B7_M4_PRE=1"
ALL="$DEV B7_M5_IRCACHE=1 B7_M5_PRECACHE=1"
FAMS=b65_chan_eval,b65_key_eval,b65_row_eval,b65_attn_out_eval,b65_dequant_eval,b65_walk_eval,b65_exp_table_eval,b65_table_eval

# Llama-3.1-8B's layer 16, as `b65/extract.py` dumps it under `b8/env.sh`
W8B=${B9_W8B:-/root/b8/witness8b/witness}
DIMS8B="q:4096x4096,k:4096x1024,v:4096x1024,o:4096x4096,gate:4096x14336,up:4096x14336,down:14336x4096"
SHAPE8B="--variants intexp --table-bits 25 --s 512 --ctx 16384 --hd 128 --bands 16,8,8 --nin 4096 --nout 14336"
st() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/status; }

for s in $STEP; do
case $s in
patch|all)
  st "patch"
  for t in gpu ref probe; do
    [ -d $ROOT/b65_$t ] || continue
    python3 $ROOT/scout/b9_patch.py $ROOT/b65_$t
    python3 $ROOT/scout/b9_diag.py $ROOT/b65_$t
  done
  for t in gpu ref probe; do
    [ -d $ROOT/b65_$t ] || continue
    ( cd $ROOT/b65_$t && cargo build --release ) > $OUT/build_$t.log 2>&1 \
      && echo "b65_$t OK" || { echo "b65_$t FAILED"; tail -n 20 $OUT/build_$t.log; }
  done
  st "patch done"
  ;;
esac
case $s in
extract)
  # The 24 Qwen-0.5B layers again: the fold's output scale is now the
  # rectangle's, so the extractor asserts the gadget's completeness precondition
  # (`fold_q_max <= 127`) instead of a prover discovering it two hours later.
  st "extract: 24 layers"
  ( cd $ROOT/py && HF_HOME=$ROOT/hf OMP_NUM_THREADS=16 python3 b65/extract.py \
      --tokens 1536 --chunk 512 --out $ROOT/py/b65/out/witness \
      --layers $(seq -s, 0 23) ) > $OUT/extract24.log 2>&1; st "rc=$?"
  ;;
esac
case $s in
layers|all)
  # Every layer at its *own* master table: `master_table_bits` is
  # `d_bits + spread`, and the spread runs 0..5 across the 24.
  st "layers: 24 single composes"
  : > $OUT/layers24.txt
  for d in $(ls -d $ROOT/py/b65/out/witness_L* | sort -t L -k 2 -n); do
    L=$(basename $d | sed 's/witness_L//')
    tb=$(python3 -c "import json;print(json.load(open('$d/manifest.json'))['master_table_bits'])")
    env $DEV B7_M5_IRCACHE=1 RAYON_NUM_THREADS=16 $B compose --variants intexp \
        --table-bits $tb --witness $d > $OUT/b9_layer_L$L.json 2> $OUT/b9_layer_L$L.err \
        && rc=0 || rc=$?
    python3 $ROOT/scout/m5_layer.py $OUT/b9_layer_L$L.json $L $tb $rc >> $OUT/layers24.txt
  done
  cat $OUT/layers24.txt
  st "layers done"
  ;;
esac
case $s in
rungs|all)
  # Rung 1 across the whole spread range: 0 (L4), 1 (L12), 2 (L9), 3 (L0), 5 (L1).
  st "rungs: SIMD reference"
  for L in 4 12 9 0 1; do
    d=$ROOT/py/b65/out/witness_L$L
    tb=$(python3 -c "import json;print(json.load(open('$d/manifest.json'))['master_table_bits'])")
    RAYON_NUM_THREADS=16 $R compose --variants intexp --table-bits $tb --witness $d \
        > $OUT/rung1_ref_L$L.json 2> $OUT/rung1_ref_L$L.err; st "ref L$L rc=$?"
  done
  st "rungs done"
  ;;
esac
case $s in
emit8b)
  # A new shape runs the emit chain first (B7A rule; B7b 26.3): the constraint
  # and fraction kernels are generated per IR shape, and the 8B layer is one.
  st "emit8b: generator tree"
  python3 $ROOT/scout/m5_probe.py $ROOT/b65_probe > $OUT/probe_patch.log 2>&1
  python3 $ROOT/scout/b9_patch.py $ROOT/b65_probe >> $OUT/probe_patch.log 2>&1
  ( cd $ROOT/b65_probe && cargo build --release ) > $OUT/probe_build.log 2>&1 || \
      { tail -n 20 $OUT/probe_build.log; st "probe build failed"; continue; }
  P=$ROOT/b65_probe/target/release/b65
  st "emit8b: cudaemit"
  env B7_SYNTH_DIMS=$DIMS8B $P cudaemit $SHAPE8B --out $GEN --block 256 --min-blocks 2 \
      > $OUT/emit8b_c.json 2>> $OUT/emit8b.log; st "cudaemit rc=$?"
  st "emit8b: genemit"
  env B7_SYNTH_DIMS=$DIMS8B $P genemit $SHAPE8B --families $FAMS --out $GEN \
      > $OUT/emit8b_g.json 2>> $OUT/emit8b.log; st "genemit rc=$?"
  tail -n 5 $OUT/emit8b.log
  ;;
esac
case $s in
install8b)
  st "install8b"
  mkdir -p $ROOT/scout/m4b
  cp $ROOT/gpu/m4b/* $ROOT/scout/m4b/ 2>/dev/null || true
  python3 $ROOT/scout/m2b_kernels.py --fork $ROOT/nitroo --gen $GEN > $OUT/install8b.log 2>&1
  python3 $ROOT/scout/m4b_kernels.py --fork $ROOT/nitroo --gen $GEN --src $ROOT/scout/m4b \
      >> $OUT/install8b.log 2>&1
  ( cd $ROOT/nitroo && /usr/bin/time -f "libstwo_cuda rebuild: %e s" \
      cargo build --release -p stwo ) >> $OUT/install8b.log 2>&1
  tail -n 3 $OUT/install8b.log
  st "install8b done"
  ;;
esac
case $s in
shapes8b)
  st "shapes8b"
  cp $GEN/cudakernels.rs $GEN/gencudakernels.rs $ROOT/b65_gpu/src/
  ( cd $ROOT/b65_gpu && cargo build --release ) > $OUT/shapes8b.log 2>&1
  tail -n 2 $OUT/shapes8b.log
  st "shapes8b done"
  ;;
esac
case $s in
prove8b)
  # The headline row: one real Llama-3.1-8B layer, S=512, C=16K, all three
  # boundaries with the chunk size stated (B7A section 5).
  st "prove8b"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 \
      > $OUT/prove8b.mem.samples 2>/dev/null &
  mpid=$!
  env $ALL RAYON_NUM_THREADS=16 $B compose --variants intexp --table-bits 25 \
      --bands 16,8,8 --witness $W8B > $OUT/prove8b.json 2> $OUT/prove8b.err; rc=$?
  kill $mpid 2>/dev/null
  sort -n $OUT/prove8b.mem.samples | tail -n 1 > $OUT/prove8b.mem
  st "prove8b rc=$rc mem=$(cat $OUT/prove8b.mem) MiB"
  ;;
esac
case $s in
ref8b)
  st "ref8b"
  RAYON_NUM_THREADS=16 $R compose --variants intexp --table-bits 25 \
      --bands 16,8,8 --witness $W8B > $OUT/ref8b.json 2> $OUT/ref8b.err; st "rc=$?"
  ;;
esac
done
