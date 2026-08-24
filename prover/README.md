# `prover/` — the B6.5 pipeline, one crate for both arms

This is the single source of truth for the prover. It used to live at
`experiments/b65/bench` (the SIMD reference) with a separately tracked CUDA
port at `experiments/b65/gpu`, and before that the CUDA side was derived on
the boxes by a chain of Python patch scripts. Both of those are gone: the two
arms are now one crate, and the difference between them is one cargo feature.

```sh
cargo build --release                     # the SIMD reference (was b65_ref)
cargo build --release --features cuda     # the CUDA prover    (was b65_gpu)
```

Source changes land here as ordinary commits. Do not derive trees with patch
scripts, and do not edit trees on a box without bringing the change back here —
the box copies are builds of this crate, not places where source lives.

## What the `cuda` feature changes

Almost nothing, deliberately. Both arms build against the vendored
NitrooZK-stwo fork in `../nitroo` (see its README for provenance), so they
share the stwo version, the Merkle channel, the span accounting, the proof
digest in the JSON output, and every line of witness and trace construction.
The feature only moves commitment and proving onto the fork's `CudaBackend`:

- `common.rs` defines `pub type Backend`, which the feature switches between
  `SimdBackend` and `CudaBackend`. `Session`, `twiddles()` and the two `prove`
  entry points are written against the alias.
- With the feature on, columns cross to the device at the single
  `extend_evals` call in `Session::commit`, through `device::upload_all`, and
  the upload is timed into the JSON report's `upload` block.
- Every `FrameworkEval` carries a `cuda_eval_name()` prefixed `b65_`. The
  names deliberately match nothing in the fork's kernel switch, so constraint
  evaluation currently takes the fork's scalar CPU fallback. Making those
  kernels real again (the B7 emit chain) is what restores device-speed
  constraint evaluation; see "what is left" below.

This is the state milestone 2a called "no kernel work": correct on the device,
not yet fast on it.

## The regression test

The digest is the oracle for "this change broke nothing". Any change that is
not supposed to alter the proof must reproduce, on both arms, the recorded
`proof_sha256` starting `6c3fb61524dee2ec` for 8B L16 at S=512, C=16K
(`experiments/b10/out/prove8b.json` and `ref8b.json`, reproduced on the H200
on both arms, 2026-08-24):

```sh
env RAYON_NUM_THREADS=16 ./target/release/b65 compose \
    --round away --numbits 9 --z-hi-bits 7 --variants intexp \
    --table-bits 25 --bands 16,8,8 --witness $B8_ROOT/witness8b_b10/witness
```

**Standing item for the next box session:** this crate was assembled and
type-checked off-box (the exact construction is recorded in the consolidation
commits), but no GPU has run it yet. Before any device work builds on it, run
the digest command above on both arms and confirm `6c3fb61524dee2ec`.

To type-check off-box on a machine without CUDA: temporarily replace
`../nitroo/crates/stwo/build.rs` with `fn main() {}` (checking does not link,
so the CMake build of the kernels is not needed), run
`cargo check` / `cargo check --features cuda`, and restore the file. Do not
commit the stub.

## Building on a box

The boxes consume this repo; they own nothing. Ship `prover/` and `nitroo/`
(tar over ssh — the boxes have no rsync), keep them side by side so the
relative path dependency in `Cargo.toml` resolves, and build with the box's
pinned toolchain (`rust-toolchain.toml` here selects it automatically). Two
box facts recorded the hard way, both in `briefs/a-h200.md`:

- The fork's `build.rs` hardcodes `/usr/local/cuda/lib64`; the H200 resolves
  it with a root symlink to the tenant's CUDA 12.9 install.
- nvcc 12.9 needs gcc 14 or older; `CUDAHOSTCXX` must point at one.

## What is left: the device chain (performance, not correctness)

Milestones 2b–5 move constraint evaluation, generation, tables, accumulators
and preprocessed columns onto the device. Their content still lives in the
frozen patch scripts under `experiments/b7/gpu/` (see the README there) and
has to be brought into this crate as commits, milestone by milestone, with the
digest checked after each. The audit of what applies cleanly and what drifted
is in `briefs/a-h200.md` ("the chain, repaired as far as it can be"): about
127 edits apply as written, and the genuine work is roughly 23 anchors in
m4e–m4h plus re-deriving the `B7_M4_SCAN` fast path against today's `at_opt`
layout.

One design decision from that work is already made and should be kept when
m4d lands: **corner bands decline the device weight-fill path** —

```rust
let m4d = if with_rows { crate::util::m4b_mode("B7_M4_WFILL") } else { 0 };
```

because `devwfill.rs` has no notion of the corner/prefix distinction, and a
guessed kernel there would file a silently wrong fill. The cells this gives up
are only the S x S corner; the C − S prefix band still fills on the device.
Teaching the fill the distinction is a performance question, not a
correctness one.

## What a corner band is

A **band** is one rectangular block of the attention matrix the prover handles
as a unit. A chunk is `S = 512` freshly decoded tokens sitting on a committed
KV cache, and each token's attention row splits into two blocks:

- the **prefix band** — the `S` queries against the `C - S` *cached* keys. A
  full rectangle: every new token legitimately attends to every cached key, so
  there is nothing to mask.
- the **corner band** — the `S` queries against the `S` *new* keys, the
  decoded tokens attending to one another. The triangle at the end of the row:
  token `t` may attend to `1..t` but not `t+1..S`. So it carries a causal
  mask, one preprocessed column per token slot (`attn_air.rs`), affordable
  only because it lives on the corner's own `S x S` rows.

They are not independent proofs: softmax runs over the whole row, so the
denominators add, `Z = Z_prefix + Z_corner`. And there is one row maximum and
one denominator per **token**, not per band, so only the prefix band commits
that per-token region and the corner rides on it (`attn_trace.rs`). That is
what the `with_rows` argument means — it is passed as `!a.corner`. Hence: a
corner band keeps its preprocessed columns (its interaction phase reads its
own mask columns back — the exp lookup's multiplicity is `1 - msk`), and a
corner band must **not** commit a per-token region, or it commits columns the
AIR does not expect.

## History

The crate's past lives, for anyone reading old records: `b65src` was the sync
of `experiments/b65/bench` on a box; `b65_ref` and `b65_gpu` were the two
trees `m2a_port.sh` derived from it; `b65base` was a previous-commit identity
control whose recipe was never recorded. Records written before 2026-08-24
refer to those names. The full port provenance (fork commit, the recorded
Merkle-channel deviation, the `line_fold_step` rename) is in the m2a scripts'
git history and summarized in `../nitroo/README.md`.
