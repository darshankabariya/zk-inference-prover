# Vendored: NitrooZK-stwo

This directory is a vendored copy of
`https://github.com/AntChainOpenLabs/NitrooZK-stwo` at commit
`41c2a05bdb1fc3e3762bc7804108e8bc269bb6fc`, the pin every recorded proof was
built against. It is committed as source so that the whole prover — our crate
in `../prover` plus the fork it builds on — is reproducible from this repo
alone, with no patch scripts and no reliance on any host's checkout.

Four changes are applied on top of the pristine import, each as its own commit
so the delta to upstream stays readable in `git log -- nitroo/`:

1. **B7a milestone 2b** (formerly a patch script): `RelationEntry`
   accessors for the kernel generator, `FrameworkEval::cuda_eval_blob()`
   defaulting to `None`, and its use in the CUDA `ComponentProver`. All in
   `crates/constraint-framework`.
2. **B7a milestone 3** (formerly a patch script): the
   `reduce_hashes_to_m31` Merkle reduction as a device kernel instead of a
   host round-trip, and FRI decommitment reads batched through
   `batch_at_multi`. All in `crates/stwo`.
3. **The pre-emit stand-in's missing feed dispatcher** (2026-09-01):
   `gen_zk_feed_dispatch`, prototyped in `cuda/gen_zk_dispatch.cuh` and defined
   as an aborting fallthrough in `cuda/gen_zk_dispatch.cu`, matching what the
   generator emits for zero kernel groups. An earlier change added that symbol to the
   emitter and to `../prover/src/gkr_key_tile.rs` but not to the stand-in, so from
   that change until this commit the CUDA backend did **not link** on any tree with no
   installed kernel store. Digest-neutral by construction: the only call path
   reaches a kernel group with no kernel, which already aborted. Gated on the
   two synthetic reference digests of that day on both backends (both are
   history now; `../prover/README.md` lists the current pins); the 8B
   regression was not run for this change.
4. **The out-of-domain sampling kernel above 65,535 polynomials**
   (2026-09-08): `batch_eval_at_points` in `crates/stwo` launches the
   polynomial axis in slices of at most 65,535 polynomials instead of one grid
   whose y dimension carries them all, which a device caps at that many blocks.
   Each slice passes the index of its first polynomial and the two kernels add
   it back, so every index is the one a single launch computed and the results
   keep their order. Before this, sampling a tree wider than 65,535 columns
   stopped in `cuda/eval_at_point.cu` with an invalid configuration argument;
   the 8B reference segment without the five GKR reductions is 93,184 columns
   wide. Digest-neutral by construction: the arithmetic and the order of the
   results are unchanged, and the SIMD backend does not run this kernel. Gated
   on the pinned digests of `../prover/README.md` on the SIMD backend; the CUDA
   backend was built and its digests were not run for this change, because the
   machine that made it has no GPU.

4. **The polynomial commitment benchmark's missing argument** (2026-09-08):
   `crates/stwo/benches/pcs.rs` passes `lde_only` to `CommitmentTreeProver::new`,
   which grew that argument when a commitment tree learned to drop a column's
   blown-up evaluations once its root is mixed. Without it the benchmark did not
   compile, so `cargo bench -p stwo --bench pcs --features prover` measured
   nothing at all. The value is `false`, which is what the prover does unless
   `ZK_LDE_ONLY` is set, and which keeps the benchmark a measurement of the
   commitment itself rather than of the commitment and the drop together.
   Digest-neutral by construction: no proof reads a benchmark.

The patch scripts themselves are retired; their docstrings survive in the two
commit messages' file diffs and in git history.

Two quirks worth knowing before building it:

- `crates/stwo/build.rs` runs the CMake build of the CUDA kernels
  unconditionally on Linux and hardcodes `/usr/local/cuda/lib64` for
  `-lcudart`. Without root, a symlink from `/usr/local/cuda` to the
  local CUDA install resolves it.
- The CMakeLists auto-detects the GPU architecture and falls back to
  `"89;100;120"` if detection fails — which silently produces an archive that
  runs nothing on an sm_90 card. After any CMake or driver change, check the
  objects are sm_90.

Do not edit this tree casually: it is a build input, and the recorded digests
were produced with exactly this state. Changes belong in their own commits
with the digest regression (`../prover/README.md`) run on both backends.
