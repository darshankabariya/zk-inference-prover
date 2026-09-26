#!/bin/bash
# on-card.sh <name>  -- the digest gate, run on the GPU box; the tree is already
# in ~/verify/<name>/src with its commit in .source-commit.
#
# Builds the SIMD backend and runs the pinned synthetic digests on it; parses the
# shapes out of prover/tests/synthetic_digest.rs, emits the constraint and
# witness-generation kernels for each, installs them into the vendored fork,
# builds the CUDA backend and runs the same digests on the device.
#
# The card decides the architecture: CUDAARCHS comes from nvidia-smi and the
# built library is read back for that architecture before the CUDA test runs.
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CUDAHOSTCXX=/usr/bin/g++ CARGO_TERM_COLOR=never RUSTUP_AUTO_INSTALL=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
# /tmp on these rentals is the container's overlay root; rustc cannot write its
# archives there ("Bad address"). Keep temp files under the home directory.
mkdir -p $HOME/tmp; export TMPDIR=$HOME/tmp
set -u
NAME=${1:?the tree name under ~/verify}; R=$HOME/verify/$NAME; SRC=$R/src; GEN=$R/gen; LOGS=$R/logs
# Every phase records its own status. A red phase must not reach VERIFY_DONE: the
# caller reads the last line and the exit code, so both have to say the same thing.
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }
die()  { echo "FATAL: $*"; echo "== VERIFY_FAIL $NAME"; exit 1; }
rm -rf $GEN; mkdir -p $GEN $LOGS

# The architecture is the card's own: compute capability 8.9 becomes CUDAARCHS=89
# and sm_89. Pinning it by hand is how a gate ends up building for a card that is
# not in the box. CUDAARCHS is only advisory here -- the vendored CMake detects the
# card before project(), so it largely ignores the variable; the cuobjdump read-back
# of the built archive below is what actually guarantees this card's code is in it.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
case $CAP in
  [0-9]*.[0-9]*) ;;
  *) die "nvidia-smi reported no compute capability (got \"$CAP\")" ;;
esac
CUDAARCHS=$(echo "$CAP" | tr -d .); export CUDAARCHS
SM=sm_$CUDAARCHS

# never share the card with a timed campaign run: before every heavy phase, wait
# until no run's status says running. NOIDLE=1 skips the wait.
idle() { [ "${NOIDLE:-0}" = 1 ] && return 0; while ls $HOME/runs/*.status >/dev/null 2>&1 && tail -qn1 $HOME/runs/*.status | grep -q '^running'; do sleep 60; done; }

CG=/sys/fs/cgroup
say() { echo "== $* ($(date -u +%H:%M:%S))"; }
T0=$(date +%s); LAST=$T0
phase() { local n=$(date +%s); echo "== phase time ${1}: $((n-LAST))s (total $((n-T0))s)"; LAST=$n; }

# this container is memory-capped well below what `free` reports; sample the
# cgroup so a near-miss is visible and an exit 137 can be called an OOM kill.
echo "== memory.max $(cat $CG/memory.max 2>/dev/null)"
echo "== memory.events before"; cat $CG/memory.events 2>/dev/null
( PEAK=0; while :; do c=$(cat $CG/memory.current 2>/dev/null || echo 0); [ "$c" -gt "$PEAK" ] && { PEAK=$c; echo $PEAK > $R/mem.peak; }; sleep 5; done ) &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT

idle
cd $SRC || exit 1
say "$NAME source $(cat .source-commit)"
rustc --version; cargo --version; cmake --version | head -1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
say "CUDAARCHS=$CUDAARCHS (compute capability $CAP)"

idle; say "simd build"
CARGO_TARGET_DIR=$R/t-simd cargo build --release --locked > $LOGS/simd-build.log 2>&1
SB=$?; echo "  exit $SB"; [ $SB -eq 0 ] || fail "simd build (exit $SB)"
grep -E '^error|Finished' $LOGS/simd-build.log | tail -3
phase simd-build

say "simd test"
CARGO_TARGET_DIR=$R/t-simd cargo test --release --locked > $LOGS/simd-test.log 2>&1
ST=$?; echo "  exit $ST"; [ $ST -eq 0 ] || fail "simd test (exit $ST)"
grep -E 'test result|FAILED|panicked|digest moved|^error' $LOGS/simd-test.log
phase simd-test
SIMD=$R/t-simd/release/zk-prover

# the synthetic shapes and flag names as this tree's digest test spells them
python3 - "$SRC/prover/tests/synthetic_digest.rs" > $R/cases.txt <<'PY'
import re,sys
s=open(sys.argv[1]).read()
def arr(name):
    m=re.search(r'const %s: &\[&str\] = &\[(.*?)\];'%name, s, re.S); return re.findall(r'"([^"]*)"', m.group(1))
SYN=arr('SYN')
consts={m.group(1): re.findall(r'"([^"]*)"', m.group(2)) for m in re.finditer(r'const ([A-Z_0-9]+): &\[&str\] =\s*&\[(.*?)\];', s, re.S)}
strs={m.group(1): m.group(2) for m in re.finditer(r'const ([A-Z_0-9]+): &str = "([^"]*)";', s)}
for m in re.finditer(r'flags: &\[(.*?)\]', s, re.S):
    if '"' not in m.group(1): continue
    out=[]
    for q, ident in re.findall(r'"([^"]*)"|([A-Z_][A-Z_0-9]*)', m.group(1)):
        if ident: out.append(strs[ident])
        else: out+= consts[q] if q in consts else [q]
    print(' '.join(SYN[1:]+out))
# tests that build their own command with args_of(&[...]) outside the CASES table
for m in re.finditer(r'args_of\(&\[(.*?)\]\)', s, re.S):
    if '"' not in m.group(1): continue
    out=[]
    for q, ident in re.findall(r'"([^"]*)"|([A-Z_][A-Z_0-9]*)', m.group(1)):
        if ident: out.append(strs.get(ident, ''))
        else: out+= consts[q] if q in consts else [q]
    print(' '.join(SYN[1:]+[o for o in out if o]))
PY
say "$(wc -l < $R/cases.txt) synthetic cases"
# An empty list is not a pass: the shapes are parsed out of the test source, so a
# rename there would otherwise leave the gate running no digests and still green.
[ -s $R/cases.txt ] || die "no synthetic cases parsed from prover/tests/synthetic_digest.rs"

# The kernel groups to generate. #205 gives the tree a `Family` list to read;
# until it lands the list is spelled out here and can be overridden with FAM=.
FAM=${FAM:-zk_chan_eval,zk_key_eval,zk_row_eval,zk_attn_out_eval,zk_dequant_eval,zk_freivalds_eval,zk_exp_table_eval,zk_table_eval,zk_residual_stream_eval,zk_residual_stream_table_eval}
say "kernel groups: $FAM"
EMITFAIL=0
while read -r c; do
  $SIMD cudaemit $c --out $GEN --block 256 --min-blocks 2 > /dev/null 2> $GEN/emit.err || { echo "cudaemit failed: $c"; tail -5 $GEN/emit.err; EMITFAIL=1; }
  $SIMD genemit $c --families "$FAM" --out $GEN > /dev/null 2> $GEN/gen.err || { echo "genemit failed: $c"; tail -5 $GEN/gen.err; EMITFAIL=1; }
done < $R/cases.txt
echo "== emit failures: $EMITFAIL"
[ $EMITFAIL -eq 0 ] || fail "kernel emit failed for at least one case"
phase emit

K=$SRC/prover/kernels
# `| tail -3` makes the pipeline's status tail's, which is always 0; read the
# installer's own status out of PIPESTATUS, on the line right after the pipeline.
python3 $K/install_constraint_kernels.py --fork $SRC/nitroo --gen $GEN 2>&1 | tail -3
[ ${PIPESTATUS[0]} -eq 0 ] || die "install_constraint_kernels.py failed"
python3 $K/install_witness_gen_kernels.py --fork $SRC/nitroo --gen $GEN --src $K 2>&1 | tail -3
[ ${PIPESTATUS[0]} -eq 0 ] || die "install_witness_gen_kernels.py failed"
find $SRC/nitroo/crates/stwo/src/stwo_cuda/cuda \( -name '*.cu' -o -name '*.cuh' \) -exec touch {} +
# A failed copy leaves the previous tree's kernels in place and the CUDA test then
# gates the wrong code, so this cannot be allowed to pass quietly.
cp $GEN/cudakernels.rs $GEN/gencudakernels.rs $SRC/prover/src/ || die "could not install the generated kernel sources"
phase install

idle; say "cuda build"
CARGO_TARGET_DIR=$R/t-cuda cargo build --release --locked -p zk-prover --features cuda > $LOGS/cuda-build.log 2>&1
CB=$?; echo "  exit $CB"; grep -E '^error|Finished' $LOGS/cuda-build.log | tail -3
phase cuda-build

# The fork's CMake auto-detects the card and falls back to "89;100;120" silently
# when detection fails, so the archive must be read rather than trusted.
say "$SM check"
# The || cannot hang off the pipeline: its status is sort's, which always succeeds.
CMLINE=$(grep -h -E 'Auto-detected GPU architecture|Could not detect GPU' $R/t-cuda/release/build/stwo-*/output 2>/dev/null | sort -u)
echo "${CMLINE:-  (no CMake architecture line in the build script output)}"
LIB=$(find $R/t-cuda/release/build -path '*/out/*' -name libstwo_cuda.a 2>/dev/null | head -1)
if [ -z "$LIB" ]; then die "no libstwo_cuda.a under $R/t-cuda/release/build/stwo-*/out/"; fi
echo "  lib $LIB ($(du -h $LIB | cut -f1))"
ARCHS=$(cuobjdump --list-elf "$LIB" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
echo "  cuobjdump --list-elf: ${ARCHS:-<none>}"
case " $ARCHS " in
  *" $SM "*) echo "  $SM present" ;;
  *) die "the vendored library carries no $SM code (${ARCHS:-none}); the CUDA test would not run this card's kernels" ;;
esac
[ $CB -eq 0 ] || die "the cuda build failed (exit $CB)"
phase arch-check

idle; say "cuda test"
CARGO_TARGET_DIR=$R/t-cuda cargo test --release --locked -p zk-prover --features cuda > $LOGS/cuda-test.log 2>&1
CT=$?; echo "  exit $CT"; [ $CT -eq 0 ] || fail "cuda test (exit $CT)"
grep -E 'test result|FAILED|panicked|digest moved|^error' $LOGS/cuda-test.log
phase cuda-test

say "memory.events after"; cat $CG/memory.events 2>/dev/null
say "memory.current peak $(cat $R/mem.peak 2>/dev/null) bytes"
say "disk"; du -sh $R/t-simd $R/t-cuda $GEN 2>/dev/null
if [ $FAIL -ne 0 ]; then say "VERIFY_FAIL $NAME (total $(( $(date +%s) - T0 ))s)"; exit 1; fi
say "VERIFY_DONE $NAME (total $(( $(date +%s) - T0 ))s)"
