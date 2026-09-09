# prover/kernels

The generated device kernels and how they get into the CUDA backend. Without
this step the CUDA backend is correct but evaluates constraints on the CPU.

1. `zk-prover cudaemit --out STORE` and `zk-prover genemit --out STORE` (the
   SIMD binary) write the constraint and witness-generation kernels for one
   segment shape into the store directory, once per distinct shape in a plan.
   `--out` is required: neither command has a default store.
2. `install_constraint_kernels.py --fork ../nitroo --gen STORE` copies the
   constraint kernels into the vendored fork's CUDA tree and adds them to its
   source list and dispatch; `install_witness_gen_kernels.py --fork ../nitroo
   --gen STORE` does the same for the generation kernels together with the
   hand-written `gen_zk_logup.cu`.
3. `cudakernels.rs` and `gencudakernels.rs` from the store replace the stubs in
   `prover/src/`, and the CUDA backend is rebuilt.

Both scripts are idempotent and take `--undo`. The `# B7a m4` marker in
`install_witness_gen_kernels.py` is the one the fork's `CMakeLists.txt` carries
around its checked-in block, so the first install replaces that block and every
later install replaces its own; `--selftest` installs twice and undoes once
against a scratch copy of that file and fails if the two strings ever part.
`--undo` leaves no block at all and removes every `gen_zk_*` from the fork's
CUDA tree, the four stand-ins the fork checks in among them: `git checkout
nitroo/` puts those back.

An installed tree differs from its commit, so a run made from one reports a
dirty build; the results repository records that together with the store's
contents. Digests never move with an install: a missing kernel fails a run
rather than changing a proof.
