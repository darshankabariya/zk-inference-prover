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

// ------------------------------------------------ the pinned staging pool --

/// B11 stream L: one page-locked buffer that the witness packs write into.
///
/// Stream F measured every upload in this crate at **6.95 GB/s on a link that
/// delivers 45.5 GB/s out of page-locked host memory**, and established that
/// the whole gap is the source being pageable (`briefs/f-wfill.md` §1). It also
/// established the shape of the fix, which is the part that is easy to get
/// wrong: **pinning pays exactly where a pass over the data already happens.**
/// A pack that narrows an `i64` witness array into `u32` words has to write its
/// output somewhere, and writing it into pinned memory instead of a fresh
/// `Vec` is free — F measured the pack unchanged at 0.0767 s and the copy
/// falling 0.2230 → 0.0474 s. An array that needs *no* pack is the opposite
/// case: staging it costs a host `memcpy` that eats everything the faster DMA
/// saves (F measured 0.0545 against 0.0548 s — no gain at all).
///
/// So this is a buffer the pack targets, never a buffer things are copied
/// through. One allocation per process, grown to the largest array any family
/// asks for and then reused by every family and every chunk: `cudaHostAlloc`
/// pins physical pages for as long as they are held and is far too expensive
/// to do per array, which is why the pool is a pool and not an allocation.
///
/// `B7_M4_POOL=1` turns it on; with it off every caller takes exactly the
/// pageable path it took before, so the default is still the proven one.
#[cfg(feature = "cuda")]
mod pinned {
    use std::ffi::c_void;
    use std::sync::Mutex;

    unsafe extern "C" {
        fn cudaHostAlloc(p: *mut *mut c_void, size: usize, flags: u32) -> i32;
        fn cudaFreeHost(p: *mut c_void) -> i32;
        fn cudaDeviceSynchronize() -> i32;
    }

    /// Wait for the copy out of the pool to land.
    ///
    /// Load-bearing, not hygiene: the fork's copy is `cudaMemcpyAsync` on
    /// stream 0, and out of *page-locked* memory that call really does return
    /// while the DMA is in flight (out of pageable memory it staged through a
    /// driver buffer first, which is part of why it was slow). The pool is
    /// handed straight to the next pack, so without this the next pack would
    /// overwrite bytes the card has not read yet.
    pub fn sync() {
        unsafe { cudaDeviceSynchronize() };
    }

    /// `cudaHostAllocDefault` — page-locked and not mapped into the device's
    /// address space. Mapping it would make the kernels read over the link
    /// instead of copying, which is a different design and not this one.
    const HOST_ALLOC_DEFAULT: u32 = 0;

    struct Buf {
        ptr: *mut u32,
        words: usize,
    }

    // Safety: the pointer is only ever reached through the mutex below, and one
    // page-locked host allocation is valid for every thread in the process.
    unsafe impl Send for Buf {}

    static POOL: Mutex<Option<Buf>> = Mutex::new(None);
    /// The high-water mark, so the record can say what the pool actually cost.
    static PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    pub fn enabled() -> bool {
        crate::util::m4b_mode("B7_M4_POOL") != 0
    }

    /// `B7_M4_POOLMUT=1`: the pool's deliberate defect, aimed at the staged
    /// bytes themselves. Off unless set, and it never runs in a timed loop.
    pub fn mutate() -> bool {
        crate::util::m4b_mode("B7_M4_POOLMUT") != 0
    }

    pub fn peak_bytes() -> usize {
        PEAK.load(std::sync::atomic::Ordering::Relaxed) * 4
    }

    /// Fill `words` u32s of page-locked memory with `pack` and hand back the
    /// pointer to copy from, for the duration of `then`.
    ///
    /// Returns `None` when the pool is not available — it is turned off, the
    /// allocation failed, or another thread is inside it. Every caller then
    /// takes its old pageable path, so a contended or memory-tight run is
    /// slower and never wrong.
    pub fn with<T>(
        words: usize,
        pack: impl FnOnce(&mut [u32]),
        then: impl FnOnce(*const u32, usize) -> T,
    ) -> Option<T> {
        if !enabled() || words == 0 {
            return None;
        }
        // `try_lock`, not `lock`: the fills are sequential today (one band, then
        // the next), so contention means an assumption has changed and the right
        // answer is the old path rather than a wait or a second pinned buffer.
        let mut g = POOL.try_lock().ok()?;
        if g.as_ref().is_none_or(|b| b.words < words) {
            if let Some(b) = g.take() {
                unsafe { cudaFreeHost(b.ptr as *mut c_void) };
            }
            let mut p: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { cudaHostAlloc(&mut p, words * 4, HOST_ALLOC_DEFAULT) };
            if rc != 0 || p.is_null() {
                eprintln!(
                    "pool: cudaHostAlloc({} MB) failed rc={rc}; \
staying on the pageable path",
                    words * 4 / 1_000_000
                );
                return None;
            }
            PEAK.fetch_max(words, std::sync::atomic::Ordering::Relaxed);
            *g = Some(Buf { ptr: p as *mut u32, words });
        }
        let b = g.as_mut().unwrap();
        // Safety: `b.words >= words` was just ensured, the allocation is live
        // for as long as the pool holds it, and the mutex guard makes this the
        // only reference to it.
        let slice = unsafe { std::slice::from_raw_parts_mut(b.ptr, words) };
        pack(slice);
        // B11 stream L: `B7_M4_POOLMUT=1` is the pool's deliberate defect.
        //
        // A staging pool is a new place for a witness array to be wrong, and a
        // control that has never fired is not a control (B7A section 5 rule 5).
        // This changes one word of one staged array after the pack has written
        // it and before the copy reads it -- the exact failure the pool makes
        // possible, and the narrowest version of it. The band and Freivalds mirrors
        // must panic under it; if they do not, they are not covering the pool.
        if mutate() {
            let k = words / 2;
            slice[k] = slice[k].wrapping_add(1);
        }
        Some(then(b.ptr as *const u32, words))
    }
}

/// Is the pinned staging pool on for this run?
#[cfg(feature = "cuda")]
pub fn pool_enabled() -> bool {
    pinned::enabled()
}

#[cfg(not(feature = "cuda"))]
pub fn pool_enabled() -> bool {
    false
}

/// The largest the pool ever grew, in bytes — reported so the record can price
/// what pinning cost in host memory as well as what it bought in bandwidth.
#[cfg(feature = "cuda")]
pub fn pool_peak_bytes() -> usize {
    pinned::peak_bytes()
}

#[cfg(not(feature = "cuda"))]
pub fn pool_peak_bytes() -> usize {
    0
}

/// Pack `words` u32s straight into page-locked memory and upload them.
///
/// The one entry point the fills use. Hands back the device handle and the two
/// halves' seconds — the pack and the copy — so the caller's `_pack` / `_h2d`
/// timers keep the meaning they had on the pageable path and the two are
/// comparable across the A/B. `None` means the pool was not available and the
/// caller must take its own pageable path — see `pinned::with`.
#[cfg(feature = "cuda")]
pub fn staged_upload(
    words: usize,
    pack: impl FnOnce(&mut [u32]),
) -> Option<(stwo::stwo_cuda::base_field_vec::Uint32Vec, f64, f64)> {
    let mut packed = 0.0f64;
    let timed = |dst: &mut [u32]| {
        let t = Instant::now();
        pack(dst);
        packed = t.elapsed().as_secs_f64();
    };
    let (dev, copy) = pinned::with(words, timed, |ptr, n| {
        let t = Instant::now();
        let dev = unsafe {
            stwo::stwo_cuda::bindings::copy_uint32_t_vec_from_host_to_device(ptr, n as u32)
        };
        pinned::sync();
        (
            stwo::stwo_cuda::base_field_vec::Uint32Vec::new(dev, n),
            t.elapsed().as_secs_f64(),
        )
    })?;
    Some((dev, packed, copy))
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

/// How many bytes of the card are in use right now, as the driver reports it.
///
/// B11 stream O. Peak memory has been sampled from outside with `nvidia-smi`
/// every two seconds, which says what a whole run reached but cannot say when.
/// The overlap question needs a number at an exact line instead: how much a
/// chunk is holding once it has been generated and before anything has been
/// proved, because that -- not the prove-time peak -- is what a second chunk
/// in flight would add. `cudaMemGetInfo` is what the fork already calls, so
/// this is the same number the driver would give the sampler, read where it
/// means something.
#[cfg(feature = "cuda")]
pub fn mem_used_bytes() -> u64 {
    let (free, total) = stwo::stwo_cuda::get_cuda_memory_info();
    total.saturating_sub(free) as u64
}

/// The reference arm has no card to ask.
#[cfg(not(feature = "cuda"))]
pub fn mem_used_bytes() -> u64 {
    0
}

// ------------------------------------------------ the live-bytes marks --

/// B12 stream GG: one mark inside the allocator at a phase boundary.
///
/// Every peak in the record is `nvidia-smi`'s high-water mark of a process
/// whose pool never gives pages back (o-overlap §9), so the memory model has
/// been checked against totals and never against one of its terms.  The
/// fork's pool keeps the two numbers that separate them -- what is allocated
/// from it now (`pool_used`) and what it holds from the driver
/// (`pool_reserved`, the number the sampler sees) -- and this reads both at a
/// named line, after `cudaDeviceSynchronize`: the pool's frees are
/// stream-ordered, so a mark taken before the stream drains reads too high.
///
/// One JSON line to stderr (`[MEM] {...}`; stdout is the run's JSON), the same
/// object kept for `mem_marks` in the point, then the pool's high-water marks
/// are reset so the next interval's high is its own.
#[cfg(feature = "cuda")]
mod marks {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use serde_json::{Value, json};

    unsafe extern "C" {
        fn cudaDeviceSynchronize() -> i32;
    }

    static MARKS: Mutex<Vec<Value>> = Mutex::new(Vec::new());
    static T0: OnceLock<Instant> = OnceLock::new();

    // B12 stream GG, the C=16K half: under `compose-chunks --plan` the marks of
    // two chunks interleave on one stderr, and under `B7_M5_PIPE=1` they
    // interleave in time as well -- chunk k+1's `after_fill` lands between
    // chunk k's `commit_inter` and its `prove_values`. The tag is the plan
    // line's `--ser-tag`, set per thread the way `m4_set_chunk` is, so a mark
    // says which chunk made it and the reporter can split the run.
    thread_local! {
        static TAG: std::cell::RefCell<Option<String>> =
            const { std::cell::RefCell::new(None) };
    }

    pub fn set_tag(tag: &str) {
        TAG.with(|t| *t.borrow_mut() = Some(tag.to_string()));
    }

    pub fn mark(label: &str, tree: Option<Value>) {
        unsafe { cudaDeviceSynchronize() };
        let t0 = T0.get_or_init(Instant::now);
        let pool = stwo::stwo_cuda::mem_pool_stats();
        let (free, total) = stwo::stwo_cuda::get_cuda_memory_info();
        let mut g = MARKS.lock().unwrap();
        let mut o = json!({
            "mark": label,
            "seq": g.len(),
            "t_secs": t0.elapsed().as_secs_f64(),
            "pool_used": pool.map(|p| p[0]),
            "pool_used_high": pool.map(|p| p[1]),
            "pool_reserved": pool.map(|p| p[2]),
            "pool_reserved_high": pool.map(|p| p[3]),
            "driver_used": total.saturating_sub(free) as u64,
            "driver_total": total as u64,
        });
        if let Some(t) = tree {
            o["tree"] = t;
        }
        if let Some(tag) = TAG.with(|t| t.borrow().clone()) {
            o["tag"] = json!(tag);
        }
        eprintln!("[MEM] {o}");
        g.push(o);
        stwo::stwo_cuda::mem_pool_reset_high();
    }

    pub fn all() -> Vec<Value> {
        MARKS.lock().unwrap().clone()
    }
}

/// One mark, at `label`.
#[cfg(feature = "cuda")]
pub fn mem_mark(label: &str) {
    marks::mark(label, None);
}

/// A commit mark, carrying the tree it closes: its index, column and cell
/// counts and the log rows of its tallest column.
#[cfg(feature = "cuda")]
pub fn mem_mark_tree(label: &str, idx: usize, cols: usize, cells: usize, max_log: u32) {
    marks::mark(
        label,
        Some(json!({"idx": idx, "cols": cols, "cells": cells, "max_log": max_log})),
    );
}

/// Every mark so far, for the run's JSON; `None` when there are none, so the
/// field is absent rather than empty.
#[cfg(feature = "cuda")]
pub fn mem_marks_json() -> Option<Value> {
    let v = marks::all();
    (!v.is_empty()).then_some(Value::Array(v))
}

/// Hand the fork's `prove_ex` its three marks.  Once per process.
#[cfg(feature = "cuda")]
pub fn install_mem_hook() {
    let _ = stwo::prover::MEM_HOOK.set(mem_mark);
}

/// Name the chunk whose marks this thread is about to take: the plan line's
/// `--ser-tag`. Thread-local, like `util::m4_set_chunk`, so the pipelined
/// loop's generation and prove threads tag their own.
#[cfg(feature = "cuda")]
pub fn set_mem_tag(tag: &str) {
    marks::set_tag(tag);
}

/// The reference arm marks nothing: no card, no pool, no field in the JSON.
#[cfg(not(feature = "cuda"))]
pub fn mem_mark(_label: &str) {}

#[cfg(not(feature = "cuda"))]
pub fn mem_mark_tree(_label: &str, _idx: usize, _cols: usize, _cells: usize, _max_log: u32) {}

#[cfg(not(feature = "cuda"))]
pub fn mem_marks_json() -> Option<Value> {
    None
}

#[cfg(not(feature = "cuda"))]
pub fn install_mem_hook() {}

#[cfg(not(feature = "cuda"))]
pub fn set_mem_tag(_tag: &str) {}

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

    /// The same, by value, for the standalone probe paths that hand their
    /// columns straight to a `Session`. B11 stream M: the residual stream's rung is one
    /// of those, and it is a SimdBackend path -- it has no device fill to take
    /// and no device column to hand over -- so a device column here is the same
    /// loud bug as above.
    pub fn into_host(self) -> HostEval {
        match self {
            AnyCol::OwnedHost(e) => e,
            AnyCol::Host(e) => e.clone(),
            _ => panic!("a device column has no host copy"),
        }
    }

    /// The same, downloading a device column rather than refusing it: for a
    /// reader that needs the cells on this side of the bus and knows it is
    /// paying for them.  Open item 18: the one such reader is the K/V boundary
    /// tamper (`boundary::kv_cols`), which bumps one cell of a side whose bands
    /// filled on the card; the honest commit takes the device references
    /// instead and never comes here.
    ///
    /// `common::eval_of` takes values in coset order and permutes them into the
    /// committed one; a device column is already in committed order, so this
    /// reinterprets rather than permutes -- the construction `take_all` makes
    /// without the cuda feature and `gkrband`'s controls make with it.  Counted
    /// as a mirror download, which is what it is.
    pub fn download(self) -> HostEval {
        let down = |d: &BaseFieldVec, l: u32| {
            let t = Instant::now();
            let v = d.to_vec();
            note_mirror((d.size * 4) as u64, t.elapsed().as_secs_f64());
            CircleEvaluation::new(
                stwo::core::poly::circle::CanonicCoset::new(l).circle_domain(),
                stwo::prover::backend::simd::column::BaseColumn::from_iter(v),
            )
        };
        match self {
            AnyCol::OwnedHost(e) => e,
            AnyCol::Host(e) => e.clone(),
            AnyCol::Dev(d, l) => down(d, l),
            AnyCol::OwnedDev(d, l) => down(&d, l),
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
        // B11 stream L: whether the packs wrote into page-locked memory, and
        // how much of it that cost -- pinning is host memory taken out of
        // circulation for the process's life, so its size belongs in the record
        // beside the bandwidth it bought.
        "pool_enabled": pool_enabled(),
        "pool_peak_bytes": pool_peak_bytes(),
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
