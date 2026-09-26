//! B7a milestone 4 (section 3 row 1): the plumbing every device witness
//! generation shares.
//!
//! The generations themselves are each gadget's own since #197 --
//! `gadgets::attention::device` for the key tiles, `gadgets::attn_out::device`
//! for the attention output, `gadgets::freivalds::device` for the weight checks
//! and `gadgets::dequant::device` for the dequantization. What is here is what
//! all of them need: the FFI shapes, the timed uploads, the packers that narrow
//! an `i64` or `i32` witness array into the words a kernel reads, and the column
//! and counter comparisons the `=2` paths assert with.
//!
//! Each group of components sits behind its own three-value switch (0 host,
//! 1 device, 2 both and compare; every default is 1 on the CUDA backend and the
//! switches exist for controls):
//!
//! | module | components | switch |
//! |---|---|---|
//! | `attention`, `attn_out`, `freivalds` | the Freivalds checks of the weight matmuls, the attention output, the key tiles | `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` (the key tiles also under `ZK_DEVICE_WITNESS_GEN_KEY_TILE`) |
//! | `dev_dequant_witness_gen` | the dequantization components | `ZK_DEVICE_WITNESS_GEN_DEQUANT` |
//! | `residual_stream` | the residual stream | `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM` |
//!
//! The negative-test switches that corrupt a generated witness are the same
//! three names under `ZK_MUT_WITNESS_GEN_*`.
//!
//! 20.6's row 1, and 20.1's design three more times. What it moves, per family:
//!
//! | | host today | here |
//! |---|---|---|
//! | Freivalds checks | `freivalds_component` over 2^15-2^16 rows, 0.532 s | two kernels, one thread per committed row |
//! | key tiles | `commit_base` over three grids and a token block, 0.401 s | four kernels per key tile |
//! | attnout | `ao_component`, 0.042 s | one kernel |
//! | all three | `eval_of` on 59 M cells | nothing: the thread owns the output index |
//!
//! **What is uploaded is the witness, and for two of the three it is much
//! smaller than the columns it replaces.** The Freivalds checks upload `qx` and `qw` as
//! *int8*, four to a word, because that is how the witness is quantized in the
//! first place: 34 MB against 168 MB of committed and preprocessed cells. The
//! attnout uploads 3.7 MB against 17.7 MB. The key tiles are the honest exception
//! -- 46 MB of shift and numerator against 68 MB of columns -- and that is
//! written down rather than averaged away.
//!
//! `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` says which path runs, on the same three-value pattern
//! `ZK_DEVICE_WITNESS_GEN_DEQUANT` uses: 0 the host witness generations, 1 the device witness generations, 2 both with every
//! cell asserted equal.

use rayon::prelude::*;
use stwo::core::fields::m31::M31;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::common::Eval;

// Every family's entry points, under the spellings the callers this issue does
// not own still say: `compose.rs`, `devacc.rs`, `devfrac.rs`, `devpre.rs` and
// `util.rs` name them all as `crate::dev_attention_freivalds_witness_gen::X`,
// which `lib.rs` keeps pointing here.
pub use crate::gadgets::attention::device::{DevKeyTile, commit_base_dev, diagonal_block_mode,
                                            exp_mult_on_host, key_tile_mode,
                                            key_tile_witness_gen_mode};
pub use crate::gadgets::attn_out::device::{DevAo, ao_component_dev};
pub use crate::gadgets::freivalds::device::{DevFreivalds, freivalds_component_dev};

// ---------------------------------------------------------------- the FFI shapes --

/// Mirrors `ZkGrid` in `zk_attention_freivalds_witness_gen.cuh`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FGrid {
    pub log_rows: u32,
    pub rows: u32,
    pub groups: u32,
    pub chunks: u32,
    pub lead: u32,
    pub descending: u32,
}

impl FGrid {
    pub(crate) fn of(g: &crate::layout::Grid) -> Self {
        Self {
            log_rows: g.log_rows,
            rows: g.rows as u32,
            groups: g.groups as u32,
            chunks: g.chunks as u32,
            lead: g.lead as u32,
            descending: u32::from(g.descending),
        }
    }
}

/// Mirrors `ZkMult`: one counter array with its length. A null pointer is the
/// host's "this witness generation does not bump that table at all", and the kernel skips the
/// call rather than counting an off-table bump.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FMult {
    pub m: *mut u32,
    pub n: u32,
}

impl FMult {
    pub(crate) fn none() -> Self {
        Self { m: std::ptr::null_mut(), n: 0 }
    }
    pub(crate) fn of(v: &Uint32Vec) -> Self {
        Self { m: v.device_ptr as *mut u32, n: v.size as u32 }
    }
}

/// Which family a witness generation belongs to, so a control can be aimed at one of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fam {
    Freivalds,
    Ao,
    KeyTile,
}

impl Fam {
    fn name(self) -> &'static str {
        match self {
            Fam::Freivalds => "freivalds",
            Fam::Ao => "ao",
            Fam::KeyTile => "key_tile",
        }
    }
}

/// `ZK_MUT_WITNESS_GEN_ATTENTION_FREIVALDS`: one deliberate defect in the device witness generations, so mode 2's
/// check has negative controls. The names are the kernel's.
///
/// **The `:family` suffix matters, and 19.6 is why.** The key tiles are checked
/// before the attnout and the Freivalds checks, so an unscoped mutation always panics at
/// the *key tiles* and the run says nothing about whether the other two families'
/// checks would have caught the same defect. `ZK_MUT_WITNESS_GEN_ATTENTION_FREIVALDS=chan:freivalds` aims it
/// at one family, so each check is shown to bite on its own rather than
/// inheriting the key tiles' evidence.
pub fn witness_gen_mut(fam: Fam) -> u32 {
    let raw = std::env::var("ZK_MUT_WITNESS_GEN_ATTENTION_FREIVALDS").ok();
    let raw = raw.as_deref().unwrap_or("");
    let (name, want) = match raw.split_once(':') {
        Some((n, f)) => (n, Some(f)),
        None => (raw, None),
    };
    if let Some(f) = want {
        assert!(["freivalds", "ao", "key_tile"].contains(&f),
                "unknown ZK_MUT_WITNESS_GEN_ATTENTION_FREIVALDS family {f:?}");
        if f != fam.name() {
            return 0;
        }
    }
    match name {
        "" | "none" => 0,
        "noperm" => 1,
        "chan" => 2,
        "nobump" => 3,
        "descend" => 4,
        "inst" => 5,
        "aoref" => 6,
        // B11 stream N, lever 2: the two ways a diagonal block key tile's witness generation could be
        // wrong that no earlier control reaches. `nomask` is the kernel's (a
        // masked slot bumping the exp table); `notok` is this side's (a diagonal block
        // key tile building the per-token component it does not own).
        "nomask" => 7,
        "notok" => 8,
        other => panic!("unknown ZK_MUT_WITNESS_GEN_ATTENTION_FREIVALDS {other:?}"),
    }
}

// --------------------------------------------------------------- the uploads --

fn up_u32(v: Vec<u32>) -> Uint32Vec {
    let bytes = (v.len() * 4) as u64;
    let t = std::time::Instant::now();
    let d = Uint32Vec::from_vec(v);
    crate::device::note_witness(bytes, t.elapsed().as_secs_f64());
    d
}

/// B11 stream F: the same upload with its two halves timed apart.
///
/// `gen/witness_gen_*_up` is a *host pack* — a rayon pass that narrows an `i64` or
/// `i32` witness array into a fresh `Vec<u32>` — followed by a `cudaMemcpy` of
/// that vector. The two answer different questions (a device-side generation
/// path deletes the first and shrinks the second), so each family's `_up` timer
/// keeps its meaning and gains a `_pack` / `_h2d` pair beside it, plus the
/// bytes each array moved under `host_bytes`. Purely additive.
pub(crate) fn timed_up(
    pack_tag: &'static str,
    h2d_tag: &'static str,
    byte_tag: &'static str,
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    let t = std::time::Instant::now();
    let v = f();
    let pack = t.elapsed().as_secs_f64();
    crate::util::add_phase(pack_tag, pack);
    let bytes = (v.len() * 4) as u64;
    let t = std::time::Instant::now();
    let d = Uint32Vec::from_vec(v);
    let el = t.elapsed().as_secs_f64();
    crate::device::note_witness(bytes, el);
    crate::util::add_phase(h2d_tag, el);
    crate::util::add_bytes(h2d_tag, bytes);
    crate::util::add_bytes(byte_tag, bytes);
    // Per array as well as per family. Which arrays a device-side generation
    // path stops uploading is the whole question, and the seconds it leaves
    // behind have to be measured in the loop rather than divided out of an
    // aggregate rate -- so the per-array line is the *copy*, which is what such
    // a path still pays; its pack, which it does not, is in `_pack`.
    let _ = pack;
    crate::util::add_phase(byte_tag, el);
    d
}

/// B11 stream L: the same upload with the pack aimed at the pinned pool.
///
/// `into` writes the array's `words` u32s into a page-locked slice and `f` is
/// the pageable path unchanged, taken whenever the pool is off or unavailable
/// (`device::staged_upload` returns `None`). The two must produce identical
/// bytes — they are two loop shells over one narrowing, which is why the
/// `word_*` helpers exist — and the digest gate is what proves it.
///
/// The timers keep their names and their meaning on both paths: `_pack` is the
/// narrowing, `_h2d` the copy. What moves between the arms is `_h2d`, and that
/// is the whole measurement.
#[cfg(feature = "cuda")]
pub(crate) fn timed_up_into(
    pack_tag: &'static str,
    h2d_tag: &'static str,
    byte_tag: &'static str,
    words: usize,
    into: impl FnOnce(&mut [u32]),
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    let Some((d, pack, copy)) = crate::device::staged_upload(words, into) else {
        return timed_up(pack_tag, h2d_tag, byte_tag, f);
    };
    let bytes = (words * 4) as u64;
    crate::util::add_phase(pack_tag, pack);
    crate::device::note_witness(bytes, copy);
    crate::util::add_phase(h2d_tag, copy);
    crate::util::add_bytes(h2d_tag, bytes);
    crate::util::add_bytes(byte_tag, bytes);
    crate::util::add_phase(byte_tag, copy);
    d
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn timed_up_into(
    pack_tag: &'static str,
    h2d_tag: &'static str,
    byte_tag: &'static str,
    _words: usize,
    _into: impl FnOnce(&mut [u32]),
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    timed_up(pack_tag, h2d_tag, byte_tag, f)
}

// B11 stream L: each narrowing is written once, as the word it produces, and
// then driven by two loop shells -- one that collects a fresh `Vec<u32>` (the
// pageable path, unchanged) and one that writes into a slice of the pinned
// staging pool. Two shells over one kernel of arithmetic rather than two
// implementations of the arithmetic: a pack that disagreed with itself between
// the two paths would move the digest, and there is nothing to disagree about
// if the narrowing exists once.
/// int8, four to a word. The kernel reads the bytes back through a
/// `signed char*`, which is exact because the packing is the little-endian byte
/// order the device already uses; `cudaMalloc` alignment covers the cast.
#[inline]
fn word_i8(v: &[i32], w: usize, bad: &std::sync::atomic::AtomicBool) -> u32 {
    let mut acc = 0u32;
    for k in 0..4 {
        let i = 4 * w + k;
        if i < v.len() {
            let x = v[i];
            if !(-128..=127).contains(&x) {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            acc |= u32::from(x as i8 as u8) << (8 * k);
        }
    }
    acc
}

pub(crate) fn pack_i8(v: &[i32], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out: Vec<u32> =
        (0..v.len().div_ceil(4)).into_par_iter().map(|w| word_i8(v, w, &bad)).collect();
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit int8, so the packed upload would truncate it"
    );
    out
}

pub(crate) fn pack_i8_into(dst: &mut [u32], v: &[i32], what: &str) {
    let bad = std::sync::atomic::AtomicBool::new(false);
    dst.par_iter_mut().enumerate().for_each(|(w, d)| *d = word_i8(v, w, &bad));
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit int8, so the packed upload would truncate it"
    );
}

/// u16, two to a word, with the same checked narrowing.
#[inline]
fn word_u16(v: &[i64], w: usize, bad: &std::sync::atomic::AtomicBool) -> u32 {
    let mut acc = 0u32;
    for k in 0..2 {
        let i = 2 * w + k;
        if i < v.len() {
            let x = v[i];
            if !(0..=65535).contains(&x) {
                bad.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            acc |= ((x as u16) as u32) << (16 * k);
        }
    }
    acc
}

pub(crate) fn pack_u16(v: &[i64], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out: Vec<u32> =
        (0..v.len().div_ceil(2)).into_par_iter().map(|w| word_u16(v, w, &bad)).collect();
    assert!(!bad.load(std::sync::atomic::Ordering::Relaxed), "{what}: a value does not fit u16");
    out
}

pub(crate) fn pack_u16_into(dst: &mut [u32], v: &[i64], what: &str) {
    let bad = std::sync::atomic::AtomicBool::new(false);
    dst.par_iter_mut().enumerate().for_each(|(w, d)| *d = word_u16(v, w, &bad));
    assert!(!bad.load(std::sync::atomic::Ordering::Relaxed), "{what}: a value does not fit u16");
}

pub(crate) fn pack_u16_raw(v: &[u16]) -> Vec<u32> {
    let n = v.len().div_ceil(2);
    (0..n)
        .into_par_iter()
        .map(|w| {
            let mut acc = 0u32;
            for k in 0..2 {
                let i = 2 * w + k;
                if i < v.len() {
                    acc |= u32::from(v[i]) << (16 * k);
                }
            }
            acc
        })
        .collect()
}

/// i64, two words each, little end first.
pub(crate) fn pack_i64(v: &[i64]) -> Vec<u32> {
    v.par_iter()
        .flat_map_iter(|x| {
            let u = *x as u64;
            [u as u32, (u >> 32) as u32]
        })
        .collect()
}

/// i32, checked. `dd` is a *shift*, bounded by the score range, so the window is
/// generous -- and a witness that broke it would otherwise be silently
/// truncated instead of loudly out of range.
#[inline]
fn word_i32(x: i64, bad: &std::sync::atomic::AtomicBool) -> u32 {
    if x < i32::MIN as i64 || x > i32::MAX as i64 {
        bad.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    x as i32 as u32
}

pub(crate) fn pack_i32(v: &[i64], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out = v.par_iter().map(|x| word_i32(*x, &bad)).collect();
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit i32"
    );
    out
}

pub(crate) fn pack_i32_into(dst: &mut [u32], v: &[i64], what: &str) {
    let bad = std::sync::atomic::AtomicBool::new(false);
    dst.par_iter_mut().zip(v.par_iter()).for_each(|(d, x)| *d = word_i32(*x, &bad));
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit i32"
    );
}

pub(crate) fn pack_i32_from_i32(v: &[i32]) -> Vec<u32> {
    v.par_iter().map(|x| *x as u32).collect()
}

pub(crate) fn pack_i32_from_i32_into(dst: &mut [u32], v: &[i32]) {
    dst.par_iter_mut().zip(v.par_iter()).for_each(|(d, x)| *d = *x as u32);
}

pub(crate) fn zeroes(n: usize) -> Uint32Vec {
    Uint32Vec::new_zeroes(n.max(1))
}

pub(crate) fn uninit(n: usize, rows: usize) -> Vec<BaseFieldVec> {
    (0..n).map(|_| BaseFieldVec::new_uninitialized(rows)).collect()
}

pub(crate) fn ptrs(v: &[BaseFieldVec]) -> Vec<*mut u32> {
    v.iter().map(|c| c.device_ptr as *mut u32).collect()
}

/// Merge a device counter array into the host's. In parallel because the exp
/// table is `1 << log_table` entries -- 8.4 M at the design point, three times
/// over -- and a serial merge would be a measurable share of what the key tile
/// kernel just saved.
/// `a - b`, entry by entry: what the *device* added on top of what the host
/// left, so the two deltas can be compared without a second snapshot type.
pub(crate) fn sub_slice(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

pub(crate) fn add_into(dst: &mut [u32], src: &[u32]) {
    assert_eq!(dst.len(), src.len(), "multiplicity counter length");
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d += *s);
}

/// Assert the M31 columns a device witness generation produced against the host's, cell for
/// cell, naming the column and the cell on a mismatch.
///
/// Not a sample: a witness generation that is wrong on one cell of one column is wrong, and
/// the point of checking here rather than against a proof digest is that the
/// panic says where.
/// B11 stream L: `ZK_DEVICE_WITNESS_GEN_DIFF=1` turns the first-mismatch panic into a
/// survey.
///
/// The default is right for a control -- a witness generation that is wrong anywhere is
/// wrong, and stopping at the first cell keeps the panic short. But when the
/// mirror does bite, the *shape* of the disagreement is the diagnosis: one
/// column or every column, one row or all of them, a whole component or a single
/// slot. Under the variable every column is compared and a compact table is
/// printed before the panic, so one run answers that instead of one bisect per
/// column.
fn diff_survey() -> bool {
    crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_DIFF") != 0
}

pub fn check_cols(what: &str, host: &[Eval], got: &[BaseFieldVec]) {
    use stwo::prover::backend::Column as _;
    assert_eq!(host.len(), got.len(), "{what}: column count");
    if diff_survey() {
        return survey_cols(what, host, got);
    }
    for (i, (a, b)) in host.iter().zip(got).enumerate() {
        let (a, b) = (a.values.to_cpu(), b.to_vec());
        assert_eq!(a.len(), b.len(), "{what}: column {i} length");
        if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
            panic!("{what}: column {i} differs at cell {k}: host {:?}, device {:?}", a[k], b[k]);
        }
    }
}

/// Every mismatching column of one check, as a table, then the panic.
fn survey_cols(what: &str, host: &[Eval], got: &[BaseFieldVec]) {
    use stwo::prover::backend::Column as _;
    let mut bad: Vec<(usize, usize, usize, usize, M31, M31)> = vec![];
    let mut rows = 0usize;
    for (i, (a, b)) in host.iter().zip(got).enumerate() {
        let (a, b) = (a.values.to_cpu(), b.to_vec());
        assert_eq!(a.len(), b.len(), "{what}: column {i} length");
        rows = a.len();
        let n = (0..a.len()).filter(|k| a[*k] != b[*k]).count();
        if n > 0 {
            let k = (0..a.len()).find(|k| a[*k] != b[*k]).unwrap();
            bad.push((i, n, a.len(), k, a[k], b[k]));
        }
    }
    if bad.is_empty() {
        return;
    }
    eprintln!(
        "witness_gen-diff: {what} -- {} of {} columns differ ({} rows each)",
        bad.len(),
        host.len(),
        rows
    );
    for (i, n, len, k, h, d) in bad.iter().take(64) {
        eprintln!(
            "witness_gen-diff:   column {i:5} {n:9} of {len} cells differ; \
first at row {k}: host {h:?}, device {d:?}"
        );
    }
    if bad.len() > 64 {
        eprintln!("witness_gen-diff:   ... and {} more columns", bad.len() - 64);
    }
    let cols: Vec<usize> = bad.iter().map(|x| x.0).collect();
    let all = bad.iter().all(|x| x.1 == x.2);
    eprintln!(
        "witness_gen-diff: {what} -- columns {:?}{}, every cell wrong in all of them: {}",
        &cols[..cols.len().min(32)],
        if cols.len() > 32 { " (truncated)" } else { "" },
        all
    );
    let (i, n, len, k, h, d) = bad[0];
    panic!("{what}: column {i} differs at cell {k}: host {h:?}, device {d:?} \
({n} of {len} cells; {} columns in all -- see the witness_gen-diff lines)", bad.len());
}

/// The multiplicity deltas: what the host's bumps added against what the
/// device's scatter-add added, entry by entry.
pub fn check_delta(what: &str, host_before: &[u32], host_after: &[u32], dev: &[u32]) {
    let n = host_after.len();
    assert_eq!(dev.len(), n, "{what}: counter length");
    for k in 0..n {
        let h = host_after[k] - host_before[k];
        if h != dev[k] {
            panic!("{what}[{k}] bumped {h} times by the host and {} by the device", dev[k]);
        }
    }
}

/// The M31 cells of a host column, for the callers that need them beside the
/// device's.
pub fn host_cells(e: &Eval) -> Vec<M31> {
    use stwo::prover::backend::Column as _;
    e.values.to_cpu()
}
