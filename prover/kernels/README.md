# prover/kernels

The generated device kernels and how they get into the CUDA backend. Without
this step the CUDA backend is correct but evaluates constraints on the CPU.

1. `zk-prover cudaemit` and `zk-prover genemit` (the SIMD binary) write the constraint and
   witness-generation kernels for one segment shape into a store directory,
   once per distinct shape in a plan.
2. `m2b_kernels.py --fork ../nitroo --gen STORE` copies the constraint kernels
   into the vendored fork's CUDA tree and adds them to its source list and
   dispatch; `m4b_kernels.py --fork ../nitroo --gen STORE` does the same for the
   generation kernels together with the hand-written `gen_b65_logup.cu`.
3. `cudakernels.rs` and `gencudakernels.rs` from the store replace the stubs in
   `prover/src/`, and the CUDA backend is rebuilt.

Both scripts are idempotent and take `--undo`. An installed tree differs from
its commit, so a run made from one reports a dirty build; the results
repository records that together with the store's contents. Digests never
move with an install: a missing kernel fails a run rather than changing a proof.
