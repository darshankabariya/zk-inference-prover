#!/bin/sh
# DD (box half): the prefill sweep that pins B9 probe 3's S^2 coefficient.
#
# B9 1.10 fitted `fresh ~ 326 M + 1.37 M*S + 72.1*S^2` from two points, S=512
# and S=1024, and eval-plan §2 showed the fit is 11% unidentifiable: two points,
# two unknowns, and `bands = p S^2` and `bands = gamma S^2 + delta S` both hit
# both points exactly while disagreeing by 11% on the coefficient that carries
# E11's headline.  More points separate them.  The 2026-08-22 sweep was typed
# into a remote shell and never checked in (eval-plan §5.1), which is why this
# file exists: the sweep is now a script, not a shell history.
#
# Runs ON the box.  The probe reads HOST-side counters, so it must run on the
# **SimdBackend reference arm** (B9 rule 2: the CUDA tree fills the grid on the
# device, where those counters see nothing).  That is the whole bring-up: no
# `--features cuda`, no kernel emit, no extracted witness -- the layer is
# synthetic (`--nin 4096 --nout 14336`, no `--witness`), exactly as the two
# survivors were.  A bare box needs `setup` and `build` and nothing else.
#
#   sh dd_prefill.sh setup    # rustup + the tree's pinned toolchain, into $ROOT
#   sh dd_prefill.sh build    # cargo build --release, SimdBackend
#   sh dd_prefill.sh sweep    # the points, data-derived `ao_rem_bits`
#   sh dd_prefill.sh pinned   # the same points at one common `--ao-rem-bits`
#   sh dd_prefill.sh edge     # the two shapes past the affordable range
#   sh dd_prefill.sh controls # B9 rule 13: nomask / noactive / noident
#   sh dd_prefill.sh all      # setup build sweep controls
#
# Geometry -- which S tile exactly -- is `b9/dd_fit.py --geom`, pure Python off
# the box, checked against the `family_cells.band_leads` every point records.
#
# Long steps go under `setsid nohup ... &` and are polled through $OUT/status:
# an ssh foreground job dies with the session (CLAUDE.md, "The boxes").
#
# No `set -e`: a point that fails is recorded (`rc=`) and the sweep goes on --
# an OOM at the largest S is a measurement, not an abort.
ROOT=${DD_ROOT:-/root/dd}
SRC=${DD_SRC:-$ROOT/src}
OUT=${DD_OUT:-$ROOT/out}
STEP=${1:-all}
CMAKEV=${DD_CMAKEV:-3.31.6}
export RUSTUP_HOME=$ROOT/rustup CARGO_HOME=$ROOT/cargo
export PATH=$ROOT/cargo/bin:$ROOT/cmake/bin:/usr/local/cuda/bin:$PATH
mkdir -p $OUT
B=$SRC/prover/target/release/b65

# The two survivors' configuration, held fixed in every respect the sweep can
# hold fixed (`b9/out/prefill_s512.json`, `prefill_s1024.json`):
#   variant intexp, hd 128, bands 16,8,8 (nh = 32), the 8B matmul shape,
#   grid_target 64, chan_target 128, rw 16, deq 16, ao 16, and C = S.
# `--table-bits 26` is NOT a free choice: both artifacts record `log_table` 26,
# and the derived sizing (`--table-bits 0`) gives 22 on this tree, which leaves
# 12,364 keys off the table and the honest proof REJECTED (`sums_cancel` false).
# 26 reproduces both survivors to the cell, and it is most of the peak RSS --
# the exp table's four interaction columns alone are 4 * 2^26 = 268 M cells.
# `kmax` / `ao_rem_bits` stay data-derived, which is what `sweep` measures;
# `pinned` is the same family with `ao_rem_bits` held at one value, because it
# is the one shape parameter that moved across the original runs (17 -> 18).
SHAPE="--variants intexp --table-bits 26 --hd 128 --bands 16,8,8 --nin 4096 --nout 14336"
# ... and the seven matmuls by name.  `--nin/--nout` alone give every matmul
# 4096x14336, which is NOT the 8B layer and not what the survivors proved: their
# `params.mats` is q/o 4096x4096, k/v 4096x1024, gate/up 4096x14336, down
# 14336x4096, which comes from `B7_SYNTH_DIMS` (b9/gpu/b9_run.sh's `DIMS8B`,
# read in `layer.rs:753`).  Leaving it unset silently proves a different layer.
DIMS8B="q:4096x4096,k:4096x1024,v:4096x1024,o:4096x4096,gate:4096x14336,up:4096x14336,down:14336x4096"
export B7_SYNTH_DIMS=${DD_DIMS:-$DIMS8B}
# 16 rayon threads, not this box's 22: `env.threads` is 16 in both artifacts,
# and the thread count moves both the wall and the peak RSS.
THREADS=${DD_THREADS:-16}
# 256/512/1024/2048 tile exactly (S a power of two, and C = S, hd, insts with
# it).  1536 is in the list to be REFUSED, on the record: `attn_trace.rs:320`
# asserts the grid's token axis tiles (`q * chunks == S`, 74*21 = 1554 != 1536),
# so eval-plan §5.1's "accept the padding at 1536" is not an option that exists
# -- the AIR will not build the shape.  See `dd_fit.py --geom`.
SS=${DD_S:-"256 512 1024 1536 2048"}
AOPIN=${DD_AOPIN:-20}
st() { echo "$(date -u +%H:%M:%S) $*" >> $OUT/status; }

# The outside-the-process peak RSS, in KiB -- the field the old S=2048 kill was
# read off, and misread as GB once already.  GNU `time` is not in this image, so
# the same two numbers come out of `resource.getrusage(RUSAGE_CHILDREN)`, which
# is exactly where GNU `time` reads them; the format is GNU's so the same
# one-liners keep working.  The JSON carries the prover's own `peak_rss_bytes`
# beside it, and a kill leaves only this file.
TIMER=$OUT/rusage.py
cat > $TIMER <<'PY'
import resource, subprocess, sys, time
out, t = sys.argv[1], time.time()
rc = subprocess.call(sys.argv[2:])
r = resource.getrusage(resource.RUSAGE_CHILDREN)
open(out, "w").write("\tMaximum resident set size (kbytes): %d\n"
                     "\tElapsed (wall clock) time (s): %.3f\n"
                     "\tExit status: %d\n" % (r.ru_maxrss, time.time() - t, rc))
sys.exit(rc if 0 <= rc < 126 else 128 - rc)
PY

# One point.  $1 tag, $2 S, rest extra flags.
point() {
  tag=$1; s=$2; shift 2
  st "point $tag S=C=$s $*"
  python3 $TIMER $OUT/$tag.time \
    env RAYON_NUM_THREADS=$THREADS $B compose $SHAPE --s $s --ctx $s "$@" \
    > $OUT/$tag.json 2> $OUT/$tag.err
  rc=$?
  st "point $tag rc=$rc rss=$(grep 'Maximum resident' $OUT/$tag.time | tr -dc 0-9) KiB"
  return $rc
}

for step in $STEP; do
case $step in
setup|all)
  st "setup"
  # rustup into $ROOT, not into the image: nothing installed globally, and a
  # second lane on this box gets its own.  `prover/rust-toolchain.toml` pins the
  # channel, so `--default-toolchain none` plus the first cargo call in that
  # directory installs exactly the pin and nothing else.
  if [ ! -x $ROOT/cargo/bin/cargo ]; then
    curl -sSf https://sh.rustup.rs -o $OUT/rustup-init.sh || { st "rustup fetch failed"; continue; }
    sh $OUT/rustup-init.sh -y --no-modify-path --profile minimal \
        --default-toolchain none > $OUT/rustup.log 2>&1 || { st "rustup failed"; continue; }
  fi
  # `cmake`, also into $ROOT and not into the image.  The reference arm needs it
  # even though it never runs a kernel: the vendored fork's `crates/stwo/build.rs`
  # builds `libstwo_cuda.a` through `cmake::Config` unconditionally, with no
  # feature gate, so a SimdBackend build still compiles the whole `.cu` tree.
  # (It wants `nvcc` for the same reason; a CUDA image already has that.)  3.x,
  # because the CMakeLists asks for 3.22 and cmake 4 dropped the <3.5 shim.
  if [ ! -x $ROOT/cmake/bin/cmake ] && ! command -v cmake >/dev/null 2>&1; then
    mkdir -p $ROOT/cmake
    curl -sSfL "https://github.com/Kitware/CMake/releases/download/v$CMAKEV/cmake-$CMAKEV-linux-x86_64.tar.gz" \
        -o $OUT/cmake.tgz || { st "cmake fetch failed"; continue; }
    tar xzf $OUT/cmake.tgz -C $ROOT/cmake --strip-components=1 || { st "cmake unpack failed"; continue; }
  fi
  ( cd $SRC/prover && cargo --version ) >> $OUT/rustup.log 2>&1
  st "setup done: $(cd $SRC/prover && cargo --version 2>&1 | head -1)"
  ;;
esac
case $step in
build|all)
  st "build (SimdBackend, no cuda feature)"
  ( cd $SRC/prover && cargo build --release ) > $OUT/build.log 2>&1
  rc=$?
  [ $rc -eq 0 ] || tail -n 25 $OUT/build.log
  st "build rc=$rc"
  ;;
esac
case $step in
sweep|all)
  # S=512 and S=1024 are in the list as REPRODUCTION CONTROLS, not as new
  # points: the archived artifacts were made by B9's `b65_ref` tree on
  # 2026-08-22 and this is `prover/` after streams B10 and B11 A-Z.  If the two
  # survivors do not come back cell for cell, the third point is not in their
  # family and the fit has to be done on this sweep alone.  Either way it is
  # measured, not assumed.
  st "sweep: $SS"
  for s in $SS; do point s$s $s; done
  st "sweep done"
  ;;
esac
case $step in
pinned)
  # The same points with `ao_rem_bits` pinned, so the family is parameter
  # identical.  The pin is above every point's own data-derived value, which
  # the composition asserts, and it touches the attnout region and the `o` walk
  # table -- not the band family, so `bands` must come back bit-identical to
  # `sweep`'s.  That equality is itself the check.
  st "pinned: $SS at --ao-rem-bits $AOPIN"
  for s in $SS; do point p$s $s --ao-rem-bits $AOPIN; done
  st "pinned done"
  ;;
esac
case $step in
edge)
  # The two shapes past the affordable range, run to find the boundary rather
  # than to be quoted: 3072 does NOT tile (odd factor 3, exactly as 1536) and
  # 4096 does.  Both are expected to reach the container's memory cap; the
  # `.time` file records where.
  st "edge: ${DD_EDGE_S:-3072 4096}"
  for s in ${DD_EDGE_S:-3072 4096}; do point e$s $s; done
  st "edge done"
  ;;
esac
case $step in
controls|all)
  # B9 rule 13 (B7A §5.13): `nomask`, `noactive` and `noident` guard the inert
  # lead rows of the band grids, and they have fired NOWHERE -- every shape
  # proven so far tiles exactly, so the paths they mutate have never run.
  #
  # Two separate things this step establishes:
  #  1. whether the sweep's shapes leave an inert lead row at all -- read off
  #     `family_cells.band_leads` in each point's own JSON, and predicted by
  #     `dd_fit.py --geom`;
  #  2. whether the three controls can be armed on the arm the probe must run
  #     on.  They are `B7_M4_ACCMUT` / `B7_M4_PREMUT` on the DEVICE fills
  #     (`devacc.rs`, `devpre.rs`), and `util.rs::m4b_default` holds every
  #     device mode at 0 without the `cuda` feature, deliberately, so the
  #     reference arm never calls an FFI stub.  Arm them anyway and record what
  #     happens: a control that cannot be armed on this arm is a finding, not a
  #     silence.
  # Run at TWO shapes, because a control that fires at neither says nothing:
  #   S = C = 256  tiles exactly, `band_leads` 0/0/0 -- no inert row to mutate
  #   S = C = 240  composes AND leaves 1024/512/512 inert lead rows (6.25% of
  #                every band grid), which is rule 13's precondition, met
  # 240 rather than §5.1's 1536 because 1536 does not compose at all (the token
  # axis assertion at `attn_trace.rs:320`).  The two conditions are different
  # and 1536 fails the wrong one: `Grid::lead` is about the KEY row-group axis,
  # the assertion is about the TOKEN axis, and a shape needs `chunks | S` for
  # the second.  `dd_fit.py`'s `composable()` scans for shapes meeting both;
  # 240 is the cheapest with a large lead fraction.
  # Each mutation is preceded by its own HONEST mirror -- mode 2 with no
  # mutation -- because a panic from a device path that cannot run on this build
  # is not a control firing, and only the honest run tells the two apart.
  st "controls"
  : > $OUT/controls.txt
  for mut in "B7_M4_ACC=2" \
             "B7_M4_ACC=2 B7_M4_ACCMUT=nomask:grid" \
             "B7_M4_PRE=2" \
             "B7_M4_PRE=2 B7_M4_PREMUT=noactive:chan" \
             "B7_M4_PRE=2 B7_M4_PREMUT=noactive:key" \
             "B7_M4_PRE=2 B7_M4_PREMUT=noident:key"; do
    for s in ${DD_CTRL_S:-256 240}; do
      tag=ctl_$(echo "$mut" | tr -dc 'a-z0-9')_s$s
      st "control $mut at S=$s"
      env RAYON_NUM_THREADS=$THREADS $mut $B compose $SHAPE --s $s --ctx $s \
          > $OUT/$tag.json 2> $OUT/$tag.err
      rc=$?
      echo "$mut S=$s: rc=$rc $(grep -m1 -A1 'panicked at' $OUT/$tag.err | \
          tail -1 | cut -c1-160)" >> $OUT/controls.txt
      echo "  accepts=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['result']['point']['accepts'])" \
          $OUT/$tag.json 2>/dev/null || echo '(no json)')" >> $OUT/controls.txt
    done
  done
  cat $OUT/controls.txt
  st "controls done"
  ;;
esac
done
