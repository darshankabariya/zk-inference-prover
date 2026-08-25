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
on both arms, 2026-08-24). The opt-in statement flags (`--norm-tails`,
`--swiglu`, `--deq-ties`, `--head-key`/`--head-argmax` from stream E;
`--bounds` from stream H; `--tight-table` from stream J) must all be absent
for the oracle to apply — each has held it with the flag off, on both arms:

```sh
env RAYON_NUM_THREADS=16 ./target/release/b65 compose \
    --round away --numbits 9 --z-hi-bits 7 --variants intexp \
    --table-bits 25 --bands 16,8,8 --witness $B8_ROOT/witness8b_b10/witness
```

**Verified on the H200, 2026-08-24**, on exactly this tree and the vendored
fork, three ways: the reference arm (54 s), the cuda arm with every device
switch off (220 s, the scalar constraint fallback as expected before an
emit-install), and the cuda arm with `B7_M4_FILL=2 B7_M4_WFILL=2 B7_M4_ACC=2`
— the check mode that builds host and device sides and asserts every cell
(all three bands, attnout and all seven matmul fills reported their
multiplicity tables and f64 cross-checks matching). All three runs reproduced
`proof_sha256 = 6c3fb61524dee2ec…` with `accepts: true`. The `PRE`/`TABDEV`
switches and kernel-speed constraint evaluation still wait on the emit-install
(below).

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

## The device chain is in (milestones 2b–5)

The whole B7 chain — generated constraint kernels' plumbing (2b), the device
Merkle and FRI work's crate side (3), instrumentation and the by-reference
tree lists (4), the device fills, tables, accumulators and preprocessed
columns (4b–4h), and the chunked loop with its preprocessed-tree cache (5) —
now lives in this crate as source, replayed against today's code and folded
in on 2026-08-24. The device paths sit behind the same runtime switches as
before (`B7_M4_FILL`, `B7_M4_WFILL`, `B7_M4_TABDEV`, `B7_M4_ACC`,
`B7_M4_PRE`, `B7_M5_*`); with everything off, both arms walk the reference
path. Mode 2 of each switch builds host and device sides and asserts every
cell — that is the per-family check the digest then seals.

Two things the chain does *not* include, on purpose: the `B7_M4_SCAN`
parallel-carry fast path (its arithmetic predates `at_opt` and is wrong under
it; no recorded number ever used it) and `b9_diag`'s off-table diagnostics (a
settled question). Details in `experiments/b7/gpu/README.md`.

**The emit has been run (stream G, H100, 2026-08-25).** The per-shape kernel
*emit* — `b65 cudaemit` / `b65 genemit` plus the two installers
(`experiments/b7/gpu/m2b_kernels.py`, `m4b_kernels.py`) — ran against the
vendored fork for the first time, over ten design points: 54 constraint
shapes and 46 generation shapes in ten families, of which **47 constraint
shapes are installed**. Records and the driver: `briefs/g-emit.md`,
`experiments/b10/gpu/g_emit.sh`.

Post-emit, **every device switch runs**, at both modes, and none of them moves
a byte: `FILL`, `WFILL`, `ACC`, `TABDEV` and `PRE` were each invoked at mode 1
and mode 2 and all ten reproduced the same `proof_sha256` with `accepts:
true`. `B7_M4_PRE` still requires `B7_M4_TABDEV` (the shared tables are one
list — the guard in `prove_layer` says so loudly). The pre-emit truth this
paragraph used to record — only `ACC` running at mode 1, `FILL`/`WFILL`
panicking in `devfrac.rs` for want of generation kernels — was correct then
and is obsolete now; it is what the emit-install fixed.

Two caveats worth keeping. First, the **glue family stays on the scalar CPU
fallback**: its generated constraint kernels are ~10,400 instructions of
straight-line code each and ptxas does not compile them (`-O3` and `-O1` both
exceeded a 30-minute cap; `-O0` finishes in 636 s but spills a 27 kB stack
frame). They are pruned from the install and carried by
`B7_KERNEL_FALLBACK=1`, which announces the fallback per component. Second,
note that with an *empty* shape table the fallback is silent, not announced —
`blob_lower` returns `None` before reaching the announcement
(`cudagen.rs:1899-1903`).

One design decision to keep: **corner bands decline every device path** —
the fill (`!a.corner` threads through `commit_base_any` as `with_rows`), the
preprocessed columns, and the per-token accumulators. The device kernels have
no notion of the corner/prefix distinction, and a guessed kernel would file a
silently wrong fill; the cells given up are only the S x S corner.

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
