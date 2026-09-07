// Placeholder, replaced by `zk-prover cudaemit`'s output (see gpu/m2b_kernels.sh).
//
// With an empty table `cuda_eval_blob` returns None for every component, so the
// crate builds and proves exactly as milestone 2a did -- on the fork's scalar
// CPU fallback -- until the kernels have been generated and compiled in.

pub struct Shape {
    pub family: &'static str,
    pub id: u32,
    pub fingerprint: u32,
    pub n_scalars: usize,
    pub example: &'static str,
}

pub static SHAPES: &[Shape] = &[];

pub fn shape_of(family: &str, fingerprint: u32) -> Option<u32> {
    SHAPES
        .iter()
        .find(|s| s.family == family && s.fingerprint == fingerprint)
        .map(|s| s.id)
}
