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
/// B7a milestone 4 (19.6): what the device fill uploads instead of columns --
/// the accumulators, the scales and the shared tables. Counted apart, because
/// 4a is a claim about *trace columns* and this is not one.
static WITNESS_NS: AtomicU64 = AtomicU64::new(0);
static WITNESS_BYTES: AtomicU64 = AtomicU64::new(0);
static WITNESS_UPS: AtomicU64 = AtomicU64::new(0);

/// B7a milestone 4 (section 3 row 1): the download that hands a device-filled
/// region's columns back to its *interaction*, which still runs on the host.
/// Section 3's row 2 -- the other ten families' LogUp kernels -- is what
/// deletes this, and until then it is reported and not hidden.
static MIRROR_NS: AtomicU64 = AtomicU64::new(0);
static MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);
static MIRROR_COLS: AtomicU64 = AtomicU64::new(0);

pub fn note_mirror(bytes: u64, secs: f64) {
    MIRROR_NS.fetch_add((secs * 1e9) as u64, Ordering::Relaxed);
    MIRROR_BYTES.fetch_add(bytes, Ordering::Relaxed);
    MIRROR_COLS.fetch_add(1, Ordering::Relaxed);
}

pub fn note_witness(bytes: u64, secs: f64) {
    WITNESS_NS.fetch_add((secs * 1e9) as u64, Ordering::Relaxed);
    WITNESS_BYTES.fetch_add(bytes, Ordering::Relaxed);
    WITNESS_UPS.fetch_add(1, Ordering::Relaxed);
}

pub fn reset() {
    UPLOAD_NS.store(0, Ordering::Relaxed);
    UPLOAD_BYTES.store(0, Ordering::Relaxed);
    UPLOAD_COLS.store(0, Ordering::Relaxed);
    DEVCOPY_NS.store(0, Ordering::Relaxed);
    DEVCOPY_BYTES.store(0, Ordering::Relaxed);
    DEVCOPY_COLS.store(0, Ordering::Relaxed);
    WITNESS_NS.store(0, Ordering::Relaxed);
    WITNESS_BYTES.store(0, Ordering::Relaxed);
    WITNESS_UPS.store(0, Ordering::Relaxed);
    MIRROR_NS.store(0, Ordering::Relaxed);
    MIRROR_BYTES.store(0, Ordering::Relaxed);
    MIRROR_COLS.store(0, Ordering::Relaxed);
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

/// Move one column to the device without taking it.
///
/// B7a milestone 4 (18.5): the caller keeps its column because the interaction
/// phase reads it. `BaseColumn::as_slice` is the same `[M31]` `into_cpu_vec`
/// would have handed over, so this is the same single `cudaMemcpy`.
#[cfg(feature = "cuda")]
pub fn upload_ref(e: &HostEval) -> DevEval {
    let host: &[M31] = e.values.as_slice();
    let bytes = (host.len() * std::mem::size_of::<M31>()) as u64;
    let t = Instant::now();
    let dev = unsafe {
        let ptr = stwo::stwo_cuda::bindings::copy_uint32_t_vec_from_host_to_device(
            host.as_ptr() as *const u32,
            host.len() as u32,
        );
        BaseFieldVec::new(ptr, host.len())
    };
    UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
    UPLOAD_COLS.fetch_add(1, Ordering::Relaxed);
    CircleEvaluation::new(e.domain, dev)
}

#[cfg(feature = "cuda")]
pub fn upload_all_ref(evals: &[&HostEval]) -> Vec<DevEval> {
    evals.iter().map(|e| upload_ref(e)).collect()
}

/// B7a milestone 4 (section 3 row 1): the same upload, as a bare device
/// column. The shared tables' columns are uploaded once here and then handed to
/// *both* the commitment and the table families' fraction kernels, so these
/// bytes are `upload_ref`'s bytes moved earlier and never a second copy.
pub fn upload_col(e: &HostEval) -> BaseFieldVec {
    let host: &[M31] = e.values.as_slice();
    let bytes = (host.len() * std::mem::size_of::<M31>()) as u64;
    let t = Instant::now();
    let dev = unsafe {
        let ptr = stwo::stwo_cuda::bindings::copy_uint32_t_vec_from_host_to_device(
            host.as_ptr() as *const u32,
            host.len() as u32,
        );
        BaseFieldVec::new(ptr, host.len())
    };
    UPLOAD_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
    UPLOAD_COLS.fetch_add(1, Ordering::Relaxed);
    dev
}

pub fn upload_all(evals: Vec<HostEval>) -> Vec<DevEval> {
    evals.into_iter().map(upload).collect()
}

/// B7a milestone 4 (19.6): the device-to-device copies `take_all` makes for
/// the columns the interaction still has to read. Not an upload, and on its
/// own line so it cannot hide inside one.
static DEVCOPY_NS: AtomicU64 = AtomicU64::new(0);
static DEVCOPY_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVCOPY_COLS: AtomicU64 = AtomicU64::new(0);

/// A column the commitment scheme is about to take, wherever it lives.
///
/// B7a milestone 4 (19.6): the dequant fill writes its columns straight onto
/// the device in committed order, and the shared tables' columns are still
/// built on the host, so one tree holds both kinds. Borrowed for trees 0 and 1,
/// whose columns the interaction phase reads again; owned for tree 2, whose
/// columns nothing reads after the commit.
pub enum AnyCol<'a> {
    Host(&'a HostEval),
    Dev(&'a BaseFieldVec, u32),
    OwnedHost(HostEval),
    OwnedDev(BaseFieldVec, u32),
}

impl AnyCol<'_> {
    pub fn log_size(&self) -> u32 {
        match self {
            AnyCol::Host(e) => e.domain.log_size(),
            AnyCol::OwnedHost(e) => e.domain.log_size(),
            AnyCol::Dev(_, l) | AnyCol::OwnedDev(_, l) => *l,
        }
    }
    pub fn size(&self) -> usize {
        1usize << self.log_size()
    }
    /// The host column, for the callers that read a committed column again --
    /// the shared tables' interaction sides, which are never on the device. A
    /// device column panics here, which is the loud version of a bug that
    /// would otherwise be a wrong proof.
    pub fn host(&self) -> &HostEval {
        match self {
            AnyCol::Host(e) => e,
            AnyCol::OwnedHost(e) => e,
            _ => panic!("a device column has no host copy"),
        }
    }
}

/// B7a milestone 4 (section 3 row 1): a tree's columns being assembled in AIR
/// order, of both kinds.
///
/// An interaction phase interleaves host accumulator columns with LogUp columns
/// that may be on either side of the link, and `Op::Tree`'s column order is part
/// of the transcript -- so one sink takes both and the order is the order the
/// calls happen in, exactly as it was when every column was a host `Eval`.
#[derive(Default)]
pub struct ColSink(pub Vec<AnyCol<'static>>);

impl ColSink {
    pub fn new() -> Self {
        ColSink(vec![])
    }
    /// Host columns, by value -- the accumulator columns every phase builds.
    pub fn extend(&mut self, v: Vec<HostEval>) {
        self.0.extend(v.into_iter().map(AnyCol::OwnedHost));
    }
    /// Columns of either kind -- what `devfrac::frac_cols` hands back.
    pub fn extend_any(&mut self, v: Vec<AnyCol<'static>>) {
        self.0.extend(v);
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn into_inner(self) -> Vec<AnyCol<'static>> {
        self.0
    }
}

/// Columns the host owns, on their way into a tree.
pub fn host_cols(v: Vec<HostEval>) -> Vec<AnyCol<'static>> {
    v.into_iter().map(AnyCol::OwnedHost).collect()
}

/// Hand a tree its columns.
///
/// A host column is uploaded, exactly as milestone 2a did. A device column is
/// already there -- but `CudaBackend::interpolate_columns` runs the NTT *in
/// place* on the buffer it is given, so a borrowed one is copied device to
/// device (device bandwidth, against the 26 GB/s of the upload it replaces)
/// and an owned one is moved straight in.
#[cfg(feature = "cuda")]
pub fn take_all(cols: Vec<AnyCol>) -> Vec<DevEval> {
    let domain_of = |l: u32| stwo::core::poly::circle::CanonicCoset::new(l).circle_domain();
    cols.into_iter()
        .map(|c| match c {
            AnyCol::Host(e) => upload_ref(e),
            AnyCol::OwnedHost(e) => upload(e),
            AnyCol::Dev(d, l) => {
                let t = Instant::now();
                let copy = d.clone();
                DEVCOPY_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                DEVCOPY_BYTES.fetch_add((d.size * 4) as u64, Ordering::Relaxed);
                DEVCOPY_COLS.fetch_add(1, Ordering::Relaxed);
                CircleEvaluation::new(domain_of(l), copy)
            }
            AnyCol::OwnedDev(d, l) => CircleEvaluation::new(domain_of(l), d),
        })
        .collect()
}

/// Without the cuda feature the prover runs on the host, so a device column
/// comes back here. That download is this tree's alone, which is why it is
/// the check and never the measurement.
#[cfg(not(feature = "cuda"))]
pub fn take_all(cols: Vec<AnyCol>) -> Vec<HostEval> {
    let to_host = |d: &BaseFieldVec, l: u32| {
        let t = Instant::now();
        let v = d.to_vec();
        DEVCOPY_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        DEVCOPY_BYTES.fetch_add((d.size * 4) as u64, Ordering::Relaxed);
        DEVCOPY_COLS.fetch_add(1, Ordering::Relaxed);
        CircleEvaluation::new(
            stwo::core::poly::circle::CanonicCoset::new(l).circle_domain(),
            stwo::prover::backend::simd::column::BaseColumn::from_iter(v),
        )
    };
    cols.into_iter()
        .map(|c| match c {
            AnyCol::Host(e) => e.clone(),
            AnyCol::OwnedHost(e) => e,
            AnyCol::Dev(d, l) => to_host(d, l),
            AnyCol::OwnedDev(d, l) => to_host(&d, l),
        })
        .collect()
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
        "witness_uploads": WITNESS_UPS.load(Ordering::Relaxed),
        "witness_bytes": WITNESS_BYTES.load(Ordering::Relaxed),
        "witness_secs": WITNESS_NS.load(Ordering::Relaxed) as f64 / 1e9,
        "mirror_batches": MIRROR_COLS.load(Ordering::Relaxed),
        "mirror_bytes": MIRROR_BYTES.load(Ordering::Relaxed),
        "mirror_secs": MIRROR_NS.load(Ordering::Relaxed) as f64 / 1e9,
        "devcopy_columns": DEVCOPY_COLS.load(Ordering::Relaxed),
        "devcopy_bytes": DEVCOPY_BYTES.load(Ordering::Relaxed),
        "devcopy_secs": DEVCOPY_NS.load(Ordering::Relaxed) as f64 / 1e9,
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
