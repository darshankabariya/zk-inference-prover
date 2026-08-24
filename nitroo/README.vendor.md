# Vendored: NitrooZK-stwo

This directory is a vendored copy of
`https://github.com/AntChainOpenLabs/NitrooZK-stwo` at commit
`41c2a05bdb1fc3e3762bc7804108e8bc269bb6fc`, the pin every recorded proof was
built against. It is committed as source so that the whole prover — our crate
in `../prover` plus the fork it builds on — is reproducible from this repo
alone, with no patch scripts and no reliance on a box's checkout.

Two changes are applied on top of the pristine import, each as its own commit
so the delta to upstream stays readable in `git log -- nitroo/`:

1. **B7a milestone 2b** (was `experiments/b7/gpu/m2b_fork.py`): `RelationEntry`
   accessors for the kernel generator, `FrameworkEval::cuda_eval_blob()`
   defaulting to `None`, and its use in the CUDA `ComponentProver`. All in
   `crates/constraint-framework`.
2. **B7a milestone 3** (was `experiments/b7/gpu/m3_fork.py`): the
   `reduce_hashes_to_m31` Merkle reduction as a device kernel instead of a
   host round-trip, and FRI decommitment reads batched through
   `batch_at_multi`. All in `crates/stwo`.

The patch scripts themselves are retired; their docstrings survive in the two
commit messages' file diffs and in git history.

Two quirks worth knowing before building it:

- `crates/stwo/build.rs` runs the CMake build of the CUDA kernels
  unconditionally on Linux and hardcodes `/usr/local/cuda/lib64` for
  `-lcudart`. On a box without root, a symlink from `/usr/local/cuda` to the
  local CUDA install resolves it (that is how the H200 is set up; see
  `briefs/a-h200.md`).
- The CMakeLists auto-detects the GPU architecture and falls back to
  `"89;100;120"` if detection fails — which silently produces an archive that
  runs nothing on an sm_90 card. After any CMake or driver change, check the
  objects are sm_90.

Do not edit this tree casually: it is a build input, and the recorded digests
were produced with exactly this state. Changes belong in their own commits
with the digest regression (`../prover/README.md`) run on both arms.
