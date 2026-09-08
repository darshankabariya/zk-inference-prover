//! B7a milestone 4 (section 3 row 1): the Freivalds check, attnout and key tile witness generations, on the
//! device.
//!
//! Three modules generate witnesses on the device, one per group of components,
//! each behind its own three-value switch (0 host, 1 device, 2 both and compare;
//! every default is 1 on the CUDA backend and the switches exist for controls):
//!
//! | module | components | switch |
//! |---|---|---|
//! | this one | the Freivalds checks of the weight matmuls, the attention output, the key tiles | `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` (the key tiles also under `ZK_DEVICE_WITNESS_GEN_KEY_TILE`) |
//! | `dev_dequant_witness_gen` | the dequantization components | `ZK_DEVICE_WITNESS_GEN_DEQUANT` |
//! | `dev_residual_stream_witness_gen` | the residual stream | `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM` |
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
//! **The one thing that is not a transcription.** The grid component's numerator
//! column commits `num[g]`, which on an honest witness equals `table[key(g)]`;
//! a kernel that recomputed it from the table would be smaller and would
//! silently repair `Corrupt::Numerator`, turning a rung-3 rejection into an
//! acceptance. So `num` is uploaded, narrowed to u16 and checked.
//!
//! `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` says which path runs, on the same three-value pattern
//! `ZK_DEVICE_WITNESS_GEN_DEQUANT` uses: 0 the host witness generations, 1 the device witness generations, 2 both with every
//! cell asserted equal.

use rayon::prelude::*;
use stwo::core::fields::m31::M31;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::attn_trace::{Cfg, Layouts};
use crate::attn_witness::{Attn, Variant};
use crate::common::Eval;
use crate::layer::{AoLayout, AttnOut, FreivaldsLayout, Layer, Tables};

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
    fn of(g: &crate::layout::Grid) -> Self {
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
    fn none() -> Self {
        Self { m: std::ptr::null_mut(), n: 0 }
    }
    fn of(v: &Uint32Vec) -> Self {
        Self { m: v.device_ptr as *mut u32, n: v.size as u32 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FFreivaldsCfg {
    pub g: FGrid,
    pub pq: u32,
    pub pk: u32,
    pub n_in: u32,
    pub tokens: u32,
    pub n_mats: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAoCfg {
    pub nh: u32,
    pub s: u32,
    pub hd: u32,
    pub per_row: u32,
    pub chunks: u32,
    pub log_rows: u32,
    pub lead: u32,
    pub do_ref: u32,
    pub mutate: u32,
    /// stream JJ (E11): the exponent window's two offsets and the accumulator
    /// window's two limb widths
    pub jv_min: i32,
    pub jo_max: i32,
    pub acc_lo_bits: u32,
    pub acc_hi_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FKeyTileCfg {
    pub gc: FGrid,
    pub gk: FGrid,
    pub go: FGrid,
    pub pk: u32,
    pub pq: u32,
    pub kq: u32,
    pub pv: u32,
    pub oq: u32,
    pub log_row: u32,
    pub rows_tok: u32,
    pub rw: u32,
    pub s: u32,
    pub c: u32,
    pub hd: u32,
    pub insts: u32,
    pub toks: u32,
    pub variant: u32,
    pub split_k: u32,
    pub off_k: i32,
    pub off_q: i32,
    pub range_check: u32,
    pub z_lo_bits: u32,
    pub z_hi_bits: u32,
    pub z_min: i64,
    /// B11 stream N, lever 2: the diagonal block's mask, which the grid kernel
    /// needs for the exp multiplicity and for nothing else.
    pub diagonal_block: u32,
    pub mutate: u32,
}

unsafe extern "C" {
    fn zk_witness_gen_freivalds(
        qx: *const i8,
        qw: *const *const i8,
        n_out: *const u32,
        base_cols: *const *mut u32,
        n_base: u32,
        pre_cols: *const *mut u32,
        n_pre: u32,
        m_r8: FMult,
        counts: *mut u32,
        cfg: FFreivaldsCfg,
    );
    fn zk_witness_gen_attnout(
        acc: *const u32,
        den: *const u32,
        jo: *const i32,
        jv: *const i32,
        base_cols: *const *mut u32,
        n_base: u32,
        pre_cols: *const *mut u32,
        n_pre: u32,
        m_qnz: FMult,
        m_rem: FMult,
        m_acc_lo: FMult,
        m_acc_hi: FMult,
        q_out: *mut i32,
        want_out: *mut u32,
        counts: *mut u32,
        cfg: FAoCfg,
    );
    fn zk_witness_gen_key_tile_chan(
        kk: *const i32,
        qq: *const i32,
        cols: *const *mut u32,
        m_k: FMult,
        m_q: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_grid(
        dd: *const i32,
        num: *const u16,
        xx: *const u16,
        sc: *const u32,
        delta: *const u32,
        vv: *const i32,
        table0: i64,
        cols: *const *mut u32,
        n_aux: u32,
        m_exp: FMult,
        m_aux: FMult,
        m_v: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_out(
        out: *const u32,
        cols: *const *mut u32,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
    fn zk_witness_gen_key_tile_tok(
        mm: *const u32,
        zz: *const u32,
        cols: *const *mut u32,
        with_max: u32,
        m_lo: FMult,
        m_hi: FMult,
        counts: *mut u32,
        cfg: FKeyTileCfg,
    );
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

/// B11 stream L: the key tile witness generation's own switch, defaulting to `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS`.
///
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` turns on three device generation paths at once -- the Freivalds check
/// components, the attnout, and every non-diagonal block key tile. Stream L measured them
/// apart on the sequence's own witness and they are not in the same state: the
/// Freivalds check and attnout witness generations reproduce the host cell for cell, while the key tile witness generation
/// disagrees with the host on the grid component and the proof is then rejected
/// (`off_table` bumps that the host never makes). One switch cannot express
/// that, so the key tile gate becomes its own three-value variable which falls back
/// to `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` when it is not set. `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1 ZK_DEVICE_WITNESS_GEN_KEY_TILE=0` is
/// the configuration that takes the Freivalds checks' win and leaves the key tiles on the host
/// until the witness generation is fixed.
pub fn key_tile_mode() -> u32 {
    match std::env::var("ZK_DEVICE_WITNESS_GEN_KEY_TILE") {
        Ok(v) => v.parse().unwrap_or(0),
        Err(_) => crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS"),
    }
}

/// B11 stream M: which witness generation *this* key tile takes, switch and layout together.
///
/// A diagonal block key tile commits no per-token component and `commit_base_dev` has no
/// notion of that layout, so it generates its witness on the host whatever the switch says
/// (`attn_trace::commit_base_any`). That override used to live inline at the
/// one place that picked the path, and the consequence -- that a diagonal-block-bearing
/// witness therefore leaves multiplicity bumps on the host -- had to be
/// rediscovered by whoever else needed it. `devpre::witness_gen_dev` needs it, so the
/// decision lives here now and both callers read the same function: this one
/// picks the path, and `compose::generate` asks it of every key tile before the
/// first witness generation runs.
pub fn key_tile_witness_gen_mode(a: &Attn) -> u32 {
    if a.diagonal_block { diagonal_block_mode() } else { key_tile_mode() }
}

/// `ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK`: the diagonal block key tiles' own witness generation switch, defaulting **off**.
///
/// B11 stream N, lever 2. Until this stream a diagonal block key tile built on the host
/// whatever any switch said, because `commit_base_dev` had no notion of a
/// layout with no per-token component and would have committed columns the AIR
/// does not expect. It has that notion now (`commit_base_dev` takes
/// `with_rows`, `devpre::key_tile_dev` with it), but a path is born off and
/// graduates on its gate: this is the switch that opens it, and the mirror
/// runs under `ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK=2` on a diagonal-block-bearing witness.
///
/// It falls back to `key_tile_mode` only once it has gated -- until then, not even
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1` reaches the diagonal block.
pub fn diagonal_block_mode() -> u32 {
    std::env::var("ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// B11 stream L: must the exp table's multiplicity column be the host's?
///
/// `ZK_DEVICE_TABLES=1` builds `m_exp` on the card, by summing the per-key-tile exp
/// counters `commit_base_dev` left there. That is right exactly when *every*
/// key tile's bumps are on the card -- and a diagonal block key tile always generates its witness on the host,
/// because it commits no per-token component and the device witness generation has no notion of
/// its layout. So on a diagonal-block-bearing witness the bumps are split across the
/// two sides and neither half alone is the column.
///
/// Only the two uniform states were ever run. All key tiles on the host is the
/// sequence as stream K left it; all key tiles on the device is a rectangle, which
/// has no diagonal block. The sequence's own chunks are the mixed state, and taking the
/// device half alone there silently loses the diagonal block key tiles' exp lookups -- a
/// LogUp sum that does not cancel, with every other check passing.
///
/// `a.zz_total` is the local signal that a diagonal block rides with this key tile: it is
/// set on the prefix half of a pair and on nothing else (`layer.rs:352`). When
/// it is, the device key tile merges its counters into the host array as it does for
/// every other table, and the host column is the one committed. `compose.rs`
/// asks `layer.key_tiles.iter().any(|b| b.diagonal_block)`, which is the same question
/// about the same layer, and the two answers have to agree or the column and
/// its counters come from different places.
pub fn exp_mult_on_host(a: &Attn) -> bool {
    // B11 stream N, lever 2: `a.zz_total.is_some()` says a diagonal block key tile rides
    // with this one, and the reason that forced this key tile's counter back to the
    // host was that the diagonal block's own bumps were made *there*. Once the diagonal block
    // generates its witness on the card that reason is gone and both halves are device
    // counters, so the condition gains the diagonal block's mode.
    //
    // This predicate and `compose::generate`'s `host_diagonal_block` are twins and must
    // say the same thing. Drift between them is not loud where it happens: the
    // proof is rejected later by `sums_cancel` with every other check green.
    crate::devtab::mode() != 1 || (a.zz_total.is_some() && diagonal_block_mode() != 1)
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
fn timed_up(
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
fn timed_up_into(
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
fn timed_up_into(
    pack_tag: &'static str,
    h2d_tag: &'static str,
    byte_tag: &'static str,
    _words: usize,
    _into: impl FnOnce(&mut [u32]),
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    timed_up(pack_tag, h2d_tag, byte_tag, f)
}

/// One key tile array, packed and uploaded, with its own byte line.
fn key_tile_up(byte_tag: &'static str, f: impl FnOnce() -> Vec<u32>) -> Uint32Vec {
    timed_up("gen/witness_gen_key_tile_pack", "gen/witness_gen_key_tile_h2d", byte_tag, f)
}

/// The same, with the pack able to target the pinned pool.
fn key_tile_up_into(
    byte_tag: &'static str,
    words: usize,
    into: impl FnOnce(&mut [u32]),
    f: impl FnOnce() -> Vec<u32>,
) -> Uint32Vec {
    timed_up_into("gen/witness_gen_key_tile_pack", "gen/witness_gen_key_tile_h2d", byte_tag, words, into, f)
}

// ------------------------------------------------ the key tile dump (stream F) --

static KEY_TILE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn raw<T>(v: &[T]) -> &[u8] {
    // Safety: writing the exact bytes of a POD slice; the reader is told the
    // element type and count by the shape JSON beside it.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// B11 stream F: this key tile's witness arrays, raw, for the standalone probe.
///
/// `ZK_KEY_TILE_DUMP=DIR` writes one subdirectory per key tile -- the arrays
/// `commit_base_dev` uploads, plus the exp table and a shape JSON -- so
/// `experiments/b11/probe/key_tile_up.cu` can price a device-side generation path
/// against the *real* values and not only against the right sizes: it
/// recomputes `dd`, `num`, `zz` and `out` on the card and compares them with
/// what is here. Off unless the variable is set (~2.4 GB for a 16-instance
/// key tile at C=16K), and it never runs in a timed loop.
fn key_tile_dump(a: &Attn, with_out: bool) {
    let Ok(dir) = std::env::var("ZK_KEY_TILE_DUMP") else { return };
    let b = KEY_TILE_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let d = std::path::Path::new(&dir).join(format!("key_tile{b}"));
    std::fs::create_dir_all(&d).expect("key_tile_dump directory");
    let put = |name: &str, bytes: &[u8]| {
        std::fs::write(d.join(name), bytes).unwrap_or_else(|e| panic!("key_tile_dump {name}: {e}"));
    };
    let sh = a.shape;
    put("kk.i32", raw(&a.kk));
    put("qq.i32", raw(&a.qq));
    put("vv.i32", raw(&a.vv));
    put("mm.i64", raw(&a.mm));
    put("dd.i64", raw(&a.dd));
    put("num.i64", raw(&a.num));
    put("zz.i64", raw(&a.zz));
    put("out.i64", raw(&a.out));
    put("delta.i64", raw(&a.delta));
    put("table.u32", raw(&a.table));
    let j = serde_json::json!({
        "s": sh.s, "c": sh.c, "hd": sh.hd, "insts": sh.insts,
        "log_table": a.log_table, "table_len": a.table.len(),
        "nmax": a.nmax, "diagonal_block": a.diagonal_block, "with_out": with_out,
        "variant": format!("{:?}", a.variant),
        "off_table": a.off_table,
        "delta": a.delta,
        "bytes": {
            "kk": std::mem::size_of_val(&a.kk[..]), "qq": std::mem::size_of_val(&a.qq[..]),
            "vv": std::mem::size_of_val(&a.vv[..]), "dd": std::mem::size_of_val(&a.dd[..]),
            "num": std::mem::size_of_val(&a.num[..]), "out": std::mem::size_of_val(&a.out[..]),
        },
    });
    put("shape.json", serde_json::to_string_pretty(&j).unwrap().as_bytes());
    eprintln!("key_tile_dump: key tile {b} -> {}", d.display());
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

fn pack_i8(v: &[i32], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out: Vec<u32> =
        (0..v.len().div_ceil(4)).into_par_iter().map(|w| word_i8(v, w, &bad)).collect();
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit int8, so the packed upload would truncate it"
    );
    out
}

fn pack_i8_into(dst: &mut [u32], v: &[i32], what: &str) {
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

fn pack_u16(v: &[i64], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out: Vec<u32> =
        (0..v.len().div_ceil(2)).into_par_iter().map(|w| word_u16(v, w, &bad)).collect();
    assert!(!bad.load(std::sync::atomic::Ordering::Relaxed), "{what}: a value does not fit u16");
    out
}

fn pack_u16_into(dst: &mut [u32], v: &[i64], what: &str) {
    let bad = std::sync::atomic::AtomicBool::new(false);
    dst.par_iter_mut().enumerate().for_each(|(w, d)| *d = word_u16(v, w, &bad));
    assert!(!bad.load(std::sync::atomic::Ordering::Relaxed), "{what}: a value does not fit u16");
}

fn pack_u16_raw(v: &[u16]) -> Vec<u32> {
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
fn pack_i64(v: &[i64]) -> Vec<u32> {
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

fn pack_i32(v: &[i64], what: &str) -> Vec<u32> {
    let bad = std::sync::atomic::AtomicBool::new(false);
    let out = v.par_iter().map(|x| word_i32(*x, &bad)).collect();
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit i32"
    );
    out
}

fn pack_i32_into(dst: &mut [u32], v: &[i64], what: &str) {
    let bad = std::sync::atomic::AtomicBool::new(false);
    dst.par_iter_mut().zip(v.par_iter()).for_each(|(d, x)| *d = word_i32(*x, &bad));
    assert!(
        !bad.load(std::sync::atomic::Ordering::Relaxed),
        "{what}: a value does not fit i32"
    );
}

fn pack_i32_from_i32(v: &[i32]) -> Vec<u32> {
    v.par_iter().map(|x| *x as u32).collect()
}

fn pack_i32_from_i32_into(dst: &mut [u32], v: &[i32]) {
    dst.par_iter_mut().zip(v.par_iter()).for_each(|(d, x)| *d = *x as u32);
}

fn zeroes(n: usize) -> Uint32Vec {
    Uint32Vec::new_zeroes(n.max(1))
}

fn uninit(n: usize, rows: usize) -> Vec<BaseFieldVec> {
    (0..n).map(|_| BaseFieldVec::new_uninitialized(rows)).collect()
}

fn ptrs(v: &[BaseFieldVec]) -> Vec<*mut u32> {
    v.iter().map(|c| c.device_ptr as *mut u32).collect()
}

/// Merge a device counter array into the host's. In parallel because the exp
/// table is `1 << log_table` entries -- 8.4 M at the design point, three times
/// over -- and a serial merge would be a measurable share of what the key tile
/// kernel just saved.
fn add_into(dst: &mut [u32], src: &[u32]) {
    assert_eq!(dst.len(), src.len(), "multiplicity counter length");
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d += *s);
}

// ====================================================== the Freivalds checks ==

/// One Freivalds group's columns, on the device, in committed order.
pub struct DevFreivalds {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    pub log_rows: u32,
    /// B7a milestone 4 (section 3 row 1): the 8-bit range counter, left where
    /// the scatter-add put it.
    pub m_r8: Uint32Vec,
}

/// Build one Freivalds group's component on the device.
///
/// `tables` is read for the 8-bit range table's length and *written* for its
/// multiplicities: the counters come back and are added in, which is the same
/// arithmetic the host's serial bumps do and in an order that cannot matter,
/// because counting is commutative.
pub fn freivalds_component_dev(
    layer: &Layer,
    mat_idx: &[usize],
    lay: &FreivaldsLayout,
    tables: &mut Tables,
) -> DevFreivalds {
    let m0 = &layer.mats[mat_idx[0]];
    let rows = 1usize << lay.g.log_rows;
    let t_up = std::time::Instant::now();

    // The quantised weights are the Freivalds check's bytes (stream F: 231.7 MB a chunk,
    // 218.10 of it `qw`), and their pack is a four-to-a-word narrowing -- the
    // case the pinned pool is for.
    let d_qx = timed_up_into("gen/witness_gen_freivalds_pack", "gen/witness_gen_freivalds_h2d", "freivalds/qx",
                             m0.qx.len().div_ceil(4),
                             |d| pack_i8_into(d, &m0.qx, "freivalds qx"),
                             || pack_i8(&m0.qx, "freivalds qx"));
    let d_qw: Vec<Uint32Vec> = mat_idx
        .iter()
        .map(|i| {
            let qw = &layer.mats[*i].qw;
            timed_up_into("gen/witness_gen_freivalds_pack", "gen/witness_gen_freivalds_h2d", "freivalds/qw",
                          qw.len().div_ceil(4),
                          |d| pack_i8_into(d, qw, "freivalds qw"),
                          || pack_i8(qw, "freivalds qw"))
        })
        .collect();
    let qw_ptrs: Vec<*const i8> = d_qw.iter().map(|d| d.device_ptr as *const i8).collect();
    let n_out: Vec<u32> = mat_idx.iter().map(|i| layer.mats[*i].n_out as u32).collect();
    crate::util::add_phase("gen/witness_gen_freivalds_up", t_up.elapsed().as_secs_f64());

    let t_alloc = std::time::Instant::now();
    let base = uninit(lay.pq, rows);
    let pre = uninit(mat_idx.len() * lay.pk + 4, rows);
    let base_p = ptrs(&base);
    let pre_p = ptrs(&pre);
    let r8 = tables.ridx(8);
    let m_r8 = zeroes(tables.m_r[r8].len());
    let counts = zeroes(3);
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    let cfg = FFreivaldsCfg {
        g: FGrid::of(&lay.g),
        pq: lay.pq as u32,
        pk: lay.pk as u32,
        n_in: m0.n_in as u32,
        tokens: m0.tokens as u32,
        n_mats: mat_idx.len() as u32,
        mutate: witness_gen_mut(Fam::Freivalds),
    };

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_freivalds(
            d_qx.device_ptr as *const i8,
            qw_ptrs.as_ptr(),
            n_out.as_ptr(),
            base_p.as_ptr(),
            base_p.len() as u32,
            pre_p.as_ptr(),
            pre_p.len() as u32,
            FMult::of(&m_r8),
            counts.device_ptr as *mut u32,
            cfg,
        );
    }
    crate::util::add_phase("gen/witness_gen_freivalds_kernel", t_k.elapsed().as_secs_f64());

    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    if crate::devpre::merge_counters() {
        add_into(&mut tables.m_r[r8], &m_r8.to_vec());
    }
    tables.off_table += got[0] as usize;
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevFreivalds { pre, base, log_rows: lay.g.log_rows, m_r8 }
}

// =============================================================== the attnout ==

pub struct DevAo {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    pub log_rows: u32,
    /// B7a milestone 4 (section 3 row 1): the `q != 0` and remainder counters.
    pub m_qnz: Uint32Vec,
    pub m_rem: Uint32Vec,
    /// stream JJ (E11): the accumulator window's two limb counters
    pub m_acc_lo: Uint32Vec,
    pub m_acc_hi: Uint32Vec,
    pub ref_bad: usize,
    pub q_out: Vec<i64>,
    /// the f64 reference's own answer per (coset row, slot), for `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=2`
    pub want: Option<Vec<u32>>,
}

#[allow(clippy::too_many_arguments)]
pub fn ao_component_dev(
    ao: &AttnOut,
    lay: &AoLayout,
    q_bits: u32,
    rem_bits: u32,
    acc_bits: (u32, u32),
    jv_min: i32,
    jo_max: i32,
    tables: &mut Tables,
    with_want: bool,
) -> DevAo {
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    let t_up = std::time::Instant::now();
    let d_acc = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/acc",
                         || pack_i64(&ao.acc));
    let d_den = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/den",
                         || pack_i64(&ao.den));
    let d_jo = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/jo",
                        || pack_i32_from_i32(&ao.jo));
    let d_jv = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/jv",
                        || pack_i32_from_i32(&ao.jv));
    crate::util::add_phase("gen/witness_gen_ao_up", t_up.elapsed().as_secs_f64());

    let t_alloc = std::time::Instant::now();
    let base = uninit(2 + 9 * pr, rows);
    let pre = uninit(pr + 6, rows);
    let base_p = ptrs(&base);
    let pre_p = ptrs(&pre);
    let ri = tables.ridx(rem_bits);
    let (rlo, rhi) = (tables.ridx(acc_bits.0), tables.ridx(acc_bits.1));
    let m_qnz = zeroes(tables.m_qnz.len());
    let m_rem = zeroes(tables.m_r[ri].len());
    let m_acc_lo = zeroes(tables.m_r[rlo].len());
    let m_acc_hi = zeroes(tables.m_r[rhi].len());
    let counts = zeroes(3);
    let q_out = zeroes(ao.nh * ao.s * ao.hd);
    let want = with_want.then(|| zeroes(rows * pr));
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    assert_eq!(tables.m_qnz.len(), 1usize << q_bits, "attnout qnz table width");
    let cfg = FAoCfg {
        nh: ao.nh as u32,
        s: ao.s as u32,
        hd: ao.hd as u32,
        per_row: pr as u32,
        chunks: lay.chunks as u32,
        log_rows: lay.log_rows,
        lead: lay.lead as u32,
        // `ao_component` has no switch for its f64 reference: it computes it on
        // every active site, and so does this.
        do_ref: 1,
        mutate: witness_gen_mut(Fam::Ao),
        // stream JJ (E11): the exponent window's offsets and the accumulator
        // window's two limb widths.  `e = jv - jo` is what the kernel derives
        // `num` and `pj` from; the key it commits is `(jv - jv_min) + (jo_max
        // - jo)`, which is `e - e_lo`.
        jv_min,
        jo_max,
        acc_lo_bits: acc_bits.0,
        acc_hi_bits: acc_bits.1,
    };

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_attnout(
            d_acc.device_ptr,
            d_den.device_ptr,
            d_jo.device_ptr as *const i32,
            d_jv.device_ptr as *const i32,
            base_p.as_ptr(),
            base_p.len() as u32,
            pre_p.as_ptr(),
            pre_p.len() as u32,
            FMult::of(&m_qnz),
            FMult::of(&m_rem),
            FMult::of(&m_acc_lo),
            FMult::of(&m_acc_hi),
            q_out.device_ptr as *mut i32,
            want.as_ref().map(|d| d.device_ptr as *mut u32).unwrap_or(std::ptr::null_mut()),
            counts.device_ptr as *mut u32,
            cfg,
        );
    }
    crate::util::add_phase("gen/witness_gen_ao_kernel", t_k.elapsed().as_secs_f64());

    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    assert_eq!(
        got[2], 0,
        "attnout: {} sites with a non-positive denominator or outside the no-wrap window",
        got[2]
    );
    if crate::devpre::merge_counters() {
        add_into(&mut tables.m_qnz, &m_qnz.to_vec());
        add_into(&mut tables.m_r[ri], &m_rem.to_vec());
        add_into(&mut tables.m_r[rlo], &m_acc_lo.to_vec());
        add_into(&mut tables.m_r[rhi], &m_acc_hi.to_vec());
    }
    // The `EPOW` window's multiplicities never leave the host: `e = jv - jo` is
    // a function of `ao.jv`, `ao.jo` and the layout, so the whole column is one
    // cheap loop over rows and no counter has to come back off the card.
    for j in 0..rows {
        match lay.at(ao.nh, ao.s, j) {
            None => {
                for _ in 0..pr {
                    tables.bump_ek(tables.e_lo, "ao_epow_lead");
                }
            }
            Some((h, t, b)) => {
                for s in 0..pr {
                    let e = ao.jv[h * ao.hd + b * pr + s] - ao.jo[t];
                    tables.bump_ek(e, "ao_epow");
                }
            }
        }
    }
    tables.off_table += got[0] as usize;
    let q_out: Vec<i64> = q_out.to_vec().into_iter().map(|v| v as i32 as i64).collect();
    let want = want.map(|d| d.to_vec());
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevAo { pre, base, log_rows: lay.log_rows, m_qnz, m_rem, m_acc_lo, m_acc_hi,
            ref_bad: got[1] as usize, q_out, want }
}

// ================================================================= the key tiles ==

/// One key tile's committed columns, on the device, with the log size of each --
/// a key tile is four sub-components on three grids plus a token block, so unlike the
/// other two families its columns are not all the same height.
pub struct DevKeyTile {
    pub base: Vec<BaseFieldVec>,
    pub logs: Vec<u32>,
    pub at: Vec<usize>,
    /// B7a milestone 4 (section 3 row 1): this key tile's exp-table counter, left
    /// where the scatter-add put it. `zk_table_witness_gen` sums one per key tile into
    /// the `m_exp` column, so with `ZK_DEVICE_TABLES=1` the 33.5 MB download and
    /// the parallel merge below never happen.
    pub m_exp: Uint32Vec,
    /// B7a milestone 4 (section 3 row 1): the five range counters this key tile's
    /// three witness generations bumped -- the key, query, value and the two `z` limbs -- and
    /// `intsplit`'s auxiliary one where the variant has it. `zk_pre_witness_gen` sums
    /// one per key tile into each range table's multiplicity column.
    pub m_k: Uint32Vec,
    pub m_q: Uint32Vec,
    pub m_v: Uint32Vec,
    pub m_aux: Option<Uint32Vec>,
    pub m_lo: Uint32Vec,
    pub m_hi: Uint32Vec,
}

/// Build one key tile's committed columns on the device.
///
/// `mult_exp` and `mult_r` are the caller's, exactly as `commit_base` takes
/// them: 20.6 called this the awkward part, and it is a counter array, which is
/// what the scatter-add already handles.
/// One key tile's committed base columns, built on the device.
///
/// B11 stream N, lever 2 (a): `with_rows` is `!a.diagonal_block`, and it says whether
/// this key tile commits the **per-token component** -- the row maxima, the
/// denominator and its low limb. A diagonal block key tile does not: there is one maximum
/// and one denominator per *token*, not per key tile, so the diagonal block rides on the
/// prefix key tile's. That distinction was the whole of why a diagonal block key tile could
/// not generate its witness on the card. The witness generation had no notion of it and would have committed
/// a per-token component the AIR does not expect there, so `commit_base_any` sent
/// every diagonal block to the host whatever the switch said; now the witness generation knows, and
/// `dev_attention_freivalds_witness_gen::diagonal_block_mode` is what opens the path.
pub fn commit_base_dev(
    a: &Attn,
    l: &Layouts,
    cfg: &Cfg,
    widths: &[u32],
    mult_exp: &mut [u32],
    mult_r: &mut [Vec<u32>],
    off_table: &mut usize,
    with_out: bool,
    merge_m_exp: bool,
    with_rows: bool,
) -> DevKeyTile {
    let sh = a.shape;
    // the host's two layout assertions, kept here because they are about the
    // layout and not about where the witness generation runs
    assert_eq!(
        l.key.q * l.key.g.chunks,
        sh.s,
        "grid layout leaves padded token slots: q={} chunks={} S={}",
        l.key.q,
        l.key.g.chunks,
        sh.s
    );
    assert_eq!(l.out.q * l.out.g.chunks, sh.s, "output layout leaves padded token slots");
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let ridx = |b: u32| widths.iter().position(|w| *w == b).unwrap();
    let rrows = 1usize << l.log_row;
    let with_max = a.variant != Variant::Bf16x;
    let n_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };

    key_tile_dump(a, with_out);

    // B11 stream N, lever 2: `notok` is this lever's own negative control -- a
    // diagonal block key tile that builds the per-token component anyway, which is precisely
    // what `commit_base_any` used to send every diagonal block to the host to avoid.
    // It has to be decided here, before the uploads, because the two arrays
    // that component reads are skipped with it.
    let with_rows = with_rows || witness_gen_mut(Fam::KeyTile) == 8;

    let t_up = std::time::Instant::now();
    // The four arrays that are 99.99% of the key tile's bytes (stream F's table:
    // dd 1,073.74 MB, num 536.87, kk 268.44, vv 268.44, qq 8.39) go through the
    // pinned pool when it is on. `mm`/`zz`/`delta` are 0.26 MB together and are
    // left on the pageable path: they would each grow the pool to no measurable
    // end.
    let d_kk = key_tile_up_into("key_tile/kk", a.kk.len(), |d| pack_i32_from_i32_into(d, &a.kk),
                            || pack_i32_from_i32(&a.kk));
    let d_qq = key_tile_up_into("key_tile/qq", a.qq.len(), |d| pack_i32_from_i32_into(d, &a.qq),
                            || pack_i32_from_i32(&a.qq));
    let d_vv = key_tile_up_into("key_tile/vv", a.vv.len(), |d| pack_i32_from_i32_into(d, &a.vv),
                            || pack_i32_from_i32(&a.vv));
    let d_dd = key_tile_up_into("key_tile/dd", a.dd.len(), |d| pack_i32_into(d, &a.dd, "key tile dd"),
                            || pack_i32(&a.dd, "key tile dd"));
    let d_num = key_tile_up_into("key_tile/num", a.num.len().div_ceil(2),
                             |d| pack_u16_into(d, &a.num, "key tile num"),
                             || pack_u16(&a.num, "key tile num"));
    let d_delta = key_tile_up("key_tile/delta", || pack_i64(&a.delta));
    let d_mm = with_rows.then(|| key_tile_up("key_tile/mm", || pack_i64(&a.mm)));
    // B11 stream L: `zz_row()`, not `zz`.
    //
    // This is the whole of what b-port called the prefix key tiles' witness-generation kernels
    // "predating the diagonal block's key discipline", and it is on this side of the
    // link rather than in the kernel. When a diagonal block key tile rides with this one the
    // per-token component belongs to the *pair*, and the denominator it commits is
    // the pair's total -- `Z >= nmax` holds of the sum and of neither half,
    // because the key whose shift is zero may be in either (`layer.rs:352`).
    // `commit_base` has always read `a.zz_row()` for it; `commit_base_dev`
    // uploaded `a.zz`, so on any diagonal-block-bearing witness the device built a
    // denominator missing the diagonal block's keys -- 48 of 352 columns wrong on a
    // C=4K chunk (the 3 x rw per-token columns; the 16 maxima were right), and
    // `off_table` bumps the host never makes, hence the rejection. Rectangles
    // have no diagonal block, `zz_row()` is `zz` there, and that is why stream G's
    // switch matrix and stream F's whole loop saw nothing.
    // B11 stream N: both of these feed the per-token component alone, so a diagonal block
    // key tile uploads neither.
    let d_zz = with_rows.then(|| key_tile_up("key_tile/zz", || pack_i64(a.zz_row())));
    // only the bf16x variant reads these two, and only it pays for them
    let bf16x = a.variant == Variant::Bf16x;
    let d_xx = bf16x.then(|| key_tile_up("key_tile/xx", || pack_u16_raw(&a.xx)));
    let d_sc = bf16x.then(|| key_tile_up("key_tile/sc", || pack_i64(&a.sc)));
    let d_out = with_out.then(|| key_tile_up("key_tile/out", || pack_i64(&a.out)));
    crate::util::add_phase("gen/witness_gen_key_tile_up", t_up.elapsed().as_secs_f64());

    let (variant, split_k) = match a.variant {
        Variant::Bf16x => (0u32, 0u32),
        Variant::IntExp => (1u32, 0u32),
        Variant::IntSplit(k) => (2u32, k),
    };
    let bcfg = FKeyTileCfg {
        gc: FGrid::of(&lc.g),
        gk: FGrid::of(&lk.g),
        go: FGrid::of(&lo.g),
        pk: lc.pk as u32,
        pq: lc.pq as u32,
        kq: lk.q as u32,
        pv: lk.pv as u32,
        oq: lo.q as u32,
        log_row: l.log_row,
        rows_tok: rrows as u32,
        rw: l.rw as u32,
        s: sh.s as u32,
        c: sh.c as u32,
        hd: sh.hd as u32,
        insts: sh.insts as u32,
        toks: sh.toks() as u32,
        variant,
        split_k,
        off_k: a.off_k(),
        off_q: a.off_q(),
        range_check: u32::from(cfg.range_check),
        z_lo_bits: cfg.z_lo_bits,
        z_hi_bits: cfg.z_hi_bits,
        z_min: cfg.z_min as i64,
        diagonal_block: u32::from(a.diagonal_block),
        mutate: witness_gen_mut(Fam::KeyTile),
    };

    let t_alloc = std::time::Instant::now();
    let mut base: Vec<BaseFieldVec> = vec![];
    let mut logs: Vec<u32> = vec![];
    let mut at = vec![];
    let counts = zeroes(3);

    // ---- the channel component ------------------------------------------------
    at.push(base.len());
    let chan = uninit(lc.pk + lc.pq, lc.g.rows);
    let chan_p = ptrs(&chan);
    let (mk, mq) = if cfg.range_check {
        (zeroes(mult_r[ridx(a.k_bits)].len()), zeroes(mult_r[ridx(a.q_bits)].len()))
    } else {
        (zeroes(1), zeroes(1))
    };
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_key_tile_chan(
            d_kk.device_ptr as *const i32,
            d_qq.device_ptr as *const i32,
            chan_p.as_ptr(),
            if cfg.range_check { FMult::of(&mk) } else { FMult::none() },
            if cfg.range_check { FMult::of(&mq) } else { FMult::none() },
            counts.device_ptr as *mut u32,
            bcfg,
        );
    }
    logs.extend(vec![lc.g.log_rows; chan.len()]);
    base.extend(chan);

    // ---- the grid component ---------------------------------------------------
    at.push(base.len());
    let grid = uninit(2 * lk.q + n_aux + lk.pv, lk.g.rows);
    let grid_p = ptrs(&grid);
    let m_exp = zeroes(mult_exp.len());
    let m_aux = match a.variant {
        Variant::IntSplit(k) => zeroes(mult_r[ridx(k)].len()),
        _ => zeroes(1),
    };
    let m_v = zeroes(mult_r[ridx(8)].len());
    unsafe {
        zk_witness_gen_key_tile_grid(
            d_dd.device_ptr as *const i32,
            d_num.device_ptr as *const u16,
            d_xx.as_ref().map(|d| d.device_ptr as *const u16).unwrap_or(std::ptr::null()),
            d_sc.as_ref().map(|d| d.device_ptr).unwrap_or(std::ptr::null()),
            d_delta.device_ptr,
            d_vv.device_ptr as *const i32,
            a.table[0] as i64,
            grid_p.as_ptr(),
            n_aux as u32,
            FMult::of(&m_exp),
            if matches!(a.variant, Variant::IntSplit(_)) { FMult::of(&m_aux) } else { FMult::none() },
            if cfg.range_check { FMult::of(&m_v) } else { FMult::none() },
            counts.device_ptr as *mut u32,
            bcfg,
        );
    }
    logs.extend(vec![lk.g.log_rows; grid.len()]);
    base.extend(grid);

    // ---- the A.V outputs ---------------------------------------------------
    at.push(base.len());
    if with_out {
        let outc = uninit(lo.q, lo.g.rows);
        let out_p = ptrs(&outc);
        unsafe {
            zk_witness_gen_key_tile_out(
                d_out.as_ref().unwrap().device_ptr,
                out_p.as_ptr(),
                counts.device_ptr as *mut u32,
                bcfg,
            );
        }
        logs.extend(vec![lo.g.log_rows; outc.len()]);
        base.extend(outc);
    }

    // ---- the per-token component ----------------------------------------------
    // B11 stream N, lever 2 (a): absent on a diagonal block key tile. `at` still gains both
    // ends of it, so `at[3] == at[4]` there and every index after the outputs
    // keeps its meaning -- the host's `commit_base` pushes the same empty span
    // for the same reason.
    at.push(base.len());
    let (m_lo, m_hi) = if with_rows {
        let n_tok = if with_max { l.rw } else { 0 } + 3 * l.rw;
        let tok = uninit(n_tok, rrows);
        let tok_p = ptrs(&tok);
        let m_lo = zeroes(mult_r[ridx(cfg.z_lo_bits)].len());
        let m_hi = zeroes(mult_r[ridx(cfg.z_hi_bits)].len());
        unsafe {
            zk_witness_gen_key_tile_tok(
                d_mm.as_ref().unwrap().device_ptr,
                d_zz.as_ref().unwrap().device_ptr,
                tok_p.as_ptr(),
                u32::from(with_max),
                FMult::of(&m_lo),
                FMult::of(&m_hi),
                counts.device_ptr as *mut u32,
                bcfg,
            );
        }
        logs.extend(vec![l.log_row; tok.len()]);
        base.extend(tok);
        (m_lo, m_hi)
    } else {
        // The two `z` range counters this component would have bumped, at their
        // real width and all zero. A diagonal block key tile makes no `z` lookups at all,
        // but `zk_pre_witness_gen` sums one counter *per key tile* into each range
        // table's multiplicity column and asserts every one of them is the
        // table's width -- so the shape has to be there even when the content
        // is nothing.
        (zeroes(mult_r[ridx(cfg.z_lo_bits)].len()), zeroes(mult_r[ridx(cfg.z_hi_bits)].len()))
    };
    at.push(base.len());
    crate::util::add_phase("gen/witness_gen_key_tile_kernel", t_k.elapsed().as_secs_f64());

    // ---- the counters, back to the host ------------------------------------
    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    // 21.4's "counters: D2H and merge" line, for the widest counter there is:
    // 33.5 MB back and a parallel add, per key tile, for an array whose only reader
    // was `base_mults[0]`. The table witness generation sums the device copies instead.
    if merge_m_exp {
        add_into(mult_exp, &m_exp.to_vec());
    }
    if crate::devpre::merge_counters() {
        if cfg.range_check {
            add_into(&mut mult_r[ridx(a.k_bits)], &mk.to_vec());
            add_into(&mut mult_r[ridx(a.q_bits)], &mq.to_vec());
            add_into(&mut mult_r[ridx(8)], &m_v.to_vec());
        }
        if let Variant::IntSplit(k) = a.variant {
            add_into(&mut mult_r[ridx(k)], &m_aux.to_vec());
        }
        // B11 stream N: a diagonal block key tile has no per-token component, so it made no
        // `z` range bumps to merge.
        if with_rows {
            add_into(&mut mult_r[ridx(cfg.z_lo_bits)], &m_lo.to_vec());
            add_into(&mut mult_r[ridx(cfg.z_hi_bits)], &m_hi.to_vec());
        }
    }
    *off_table += got[0] as usize;
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevKeyTile { base, logs, at, m_exp, m_k: mk, m_q: mq, m_v,
              m_aux: matches!(a.variant, Variant::IntSplit(_)).then_some(m_aux),
              m_lo, m_hi }
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
