//! B7a milestone 2a — the SIMD -> device boundary.
//!
//! Milestone 2a keeps every trace and interaction column exactly where B6.5
//! builds it (on the host, `SimdBackend`, `LogupTraceGenerator`) and moves it
//! to the device at the one place the commitment scheme needs it. Milestone 4
//! is what deletes this file.
//!
//! The upload is instrumented because README section 10.2 found the fork's
//! host-to-device path is pageable and runs at 1.4-7 GB/s against a ~55 GB/s
//! link, and at our design point there are 1.12 GB of fresh cells to move. A
//! number here is what tells 2b/4 whether that still matters.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};
use stwo::core::fields::m31::M31;
use stwo::prover::backend::cuda::CudaBackend;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Column as _;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;

/// A column as B6.5 builds it: on the host, SIMD-packed.
pub type HostEval = CircleEvaluation<SimdBackend, M31, BitReversedOrder>;
/// The same column on the device.
pub type DevEval = CircleEvaluation<CudaBackend, M31, BitReversedOrder>;

static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_COLS: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    UPLOAD_NS.store(0, Ordering::Relaxed);
    UPLOAD_BYTES.store(0, Ordering::Relaxed);
    UPLOAD_COLS.store(0, Ordering::Relaxed);
}

/// Move one column to the device.
///
/// `into_cpu_vec` reinterprets the SIMD column's backing store as a
/// `Vec<M31>` without copying, so what is timed here is the transfer, not a
/// host-side repack.
pub fn upload(e: HostEval) -> DevEval {
    let domain = e.domain;
    let host: Vec<M31> = e.values.into_cpu_vec();
    let bytes = (host.len() * std::mem::size_of::<M31>()) as u64;
    let t = Instant::now();
    let dev = BaseFieldVec::from_vec(host);
    UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
    UPLOAD_COLS.fetch_add(1, Ordering::Relaxed);
    CircleEvaluation::new(domain, dev)
}

pub fn upload_all(evals: Vec<HostEval>) -> Vec<DevEval> {
    evals.into_iter().map(upload).collect()
}

pub fn report() -> Value {
    let ns = UPLOAD_NS.load(Ordering::Relaxed);
    let bytes = UPLOAD_BYTES.load(Ordering::Relaxed);
    let secs = ns as f64 / 1e9;
    json!({
        "columns": UPLOAD_COLS.load(Ordering::Relaxed),
        "bytes": bytes,
        "secs": secs,
        "gb_per_s": if secs > 0.0 { (bytes as f64 / 1e9) / secs } else { 0.0 },
    })
}

/// Section 5 rung 1's digest: the GPU proof's bytes must equal the SIMD
/// proof's, and this is what the two runs are compared on. Hashing the same
/// `serde_json` serialisation that `verify_composed` consumes, so the digest
/// covers exactly the artifact the verifier is handed.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
