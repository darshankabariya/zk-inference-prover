//! B7a milestone 4 (section 3 row 1): 4a's last 79 columns.
//!
//! 24.4 left sub-bar 4a at **0.016 GB in 79 uploads** and named both halves.
//! This file is their witness generation, and it closes the sub-bar.
//!
//! | half | columns | bytes at the synthetic point | what they are |
//! |---|---|---|---|
//! | the shared tables | 31 | 11,849,728 | 13 rows of host-built table, 8 closed forms of the row index, 10 multiplicities |
//! | the key tile selectors | 48 | 3,818,752 | `commit_pre`'s ACTIVE / NOFIRST / LAST / ISTART, three sub-components per key tile |
//!
//! **The 13 that looked like data are closed forms.** 23.1 left the 31 alone
//! with one sentence -- "their contents are host-built tables ... not closed
//! forms, so moving them would trade an upload for an upload" -- and that is
//! true of the *rows* and false of the *columns*. `bf16::norm_table`,
//! `Tables::new`'s fraction table and `bf16::mul_table` are each a function of
//! the row index and nothing else: a leading-zero count, a shift, and one
//! rounding step apiece. The only float anywhere in them is `from_f32(M2 as
//! f32)` for `M2 < 256`, and eight significand bits fit bf16's eight, so that
//! conversion is exact and the device does the same bit surgery the host does.
//!
//! **The 10 multiplicities are 23.1's counter trick, nine more times.** Every
//! one of them is already a device counter that a witness generation scatter-added into:
//! `zk_witness_gen_dequant` bumps `m_norm`, `m_frac`, `m_mul` and the 8-bit range,
//! `zk_witness_gen_freivalds` the 8-bit range, `zk_witness_gen_attnout` `m_qnz` and the
//! remainder range, `zk_witness_gen_key_tile_chan` the key and query ranges,
//! `zk_witness_gen_key_tile_grid` `m_exp`, the auxiliary range and the value range, and
//! `zk_witness_gen_key_tile_tok` the two `z` ranges. They came back to the host only to
//! be merged and re-uploaded, which is 21.4's line and 23.1's fix -- so the
//! sums happen where the counters already are and `gen/witness_gen_counters` and
//! `gen/deq_dev_counters` go with them.
//!
//! **Section 5 rule 8, and the probe that makes it bite.** Rung 3's seventh
//! probe is `ProverTamper::norm_row`: one row of `tables.norm` flipped inside
//! the prover and the witness rebuilt to match, so that *only* the pinned
//! preprocessed root gives it away -- the normalisation table's version of the
//! sixth probe, which 23.1 answered for the exp table. A kernel that computed
//! the closed form and stopped would repair it silently. So the host diffs its
//! actual `norm`, `frac` and `mul` tables against the closed form (three rayon
//! passes, no allocation) and uploads the rows that differ: empty on an honest
//! run, one row on the probe. The pass is in the **measured** path and not a
//! check mode, which is what makes the device column equal to the host's table
//! by construction on any witness rather than by argument on this one. And note
//! which way the asymmetry runs: a host closed form that is *wrong* only ever
//! uploads rows that did not need uploading, where a kernel closed form that is
//! wrong is caught cell for cell by `ZK_DEVICE_PREPROCESSED=2`.
//!
//! **The selectors are the flags `zk_acc.cu` already computes.** Since 24.1
//! every accumulator kernel has derived ACTIVE / NOFIRST / LAST / ISTART per
//! coset row from the component's layout -- which is exactly what `commit_pre`'s
//! 48 columns say. What it could not do is compute them at the right time:
//! `commit_pre` runs before the challenges are drawn and the accumulators run
//! after. `zk_flags_key_tile` is that computation lifted out into a kernel of its
//! own, called here at witness-generation time; the array outlives the challenge draw, this
//! file turns it into columns, and `zk_acc_chan` / `_grid` / `_tok` read it
//! rather than recompute it. The AIR's selectors and the accumulator's
//! segmentation are now one array instead of two transcriptions that agree.
//!
//! `ZK_DEVICE_PREPROCESSED` says which path runs, on the three-value pattern every witness generation in
//! this series uses: 0 the host builds all 79, 1 the device does and they are
//! never on the host, 2 both with every cell asserted equal.

use std::sync::{Mutex, OnceLock};

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::prover::backend::Column as _;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::attn_trace::{Layouts, m31_of};
use crate::attn_witness::{Attn, Variant};
use crate::bf16::{EXP_BIAS, Round};
use crate::common::Eval;
use crate::devacc::FGridA;
use crate::layer::Tables;

// ---------------------------------------------------------------- the FFI shapes --

/// Mirrors `ZkPreCfg` in `zk_pre_witness_gen.cuh`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FPreCfg {
    pub kind: u32,
    pub log_rows: u32,
    pub rows: u32,
    pub real_rows: u32,
    pub kmax: u32,
    pub round: u32,
    pub sticky: u32,
    pub n_cols: u32,
    pub n_counters: u32,
    pub mutate: u32,
    pub at: u32,
}

/// Mirrors `ZkSelCfg`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FSelCfg {
    pub shape: u32,
    pub g: FGridA,
    pub ctx: u32,
    pub n_insts: u32,
    pub with_scale: u32,
    pub n_cols: u32,
    pub fam: u32,
    pub mutate: u32,
    pub at: u32,
}

/// Mirrors `ZkAccFlags` in `zk_acc.cuh`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccFlags {
    pub shape: u32,
    pub g: FGridA,
    pub modulus: u32,
    pub rw: u32,
}

unsafe extern "C" {
    fn zk_witness_gen_pre_table(counters: *const *const u32, patch_idx: *const u32,
                          patch_val: *const u32, n_patch: u32, cols: *const *mut u32,
                          m_col: *mut u32, cfg: FPreCfg);
    fn zk_witness_gen_sel(flags: *const u32, cols: *const *mut u32, scale: *const u32,
                    cfg: FSelCfg);
    fn zk_flags_key_tile(flags: *mut u32, cfg: FAccFlags);
}

// `ZK_PRE_KIND_*`.
const KIND_NORM: u32 = 0;
const KIND_FRAC: u32 = 1;
const KIND_MUL: u32 = 2;
const KIND_QNZ: u32 = 3;
const KIND_RANGE: u32 = 4;

// `ZK_PRE_AT_*`.
const AT_CHAN: u32 = 5;
const AT_KEY: u32 = 6;
const AT_ROW: u32 = 7;
const AT_ALL: u32 = 8;

// `ZK_ACC_SHAPE_*`.
const SHAPE_CHAN: u32 = 0;
const SHAPE_KEY: u32 = 1;
const SHAPE_TOK: u32 = 2;

// ------------------------------------------------------------------ the switch --

/// `ZK_DEVICE_PREPROCESSED`: 0 the host, 1 the device, 2 both with every cell asserted.
pub fn mode() -> u32 {
    crate::util::path_mode("ZK_DEVICE_PREPROCESSED")
}

/// `ZK_MUT_PREPROCESSED`, `name` or `name:target`. Unaimed, every defect shows at the
/// first column compared and the run then says nothing about the other seven
/// targets (section 5 rule 5), so the suffix names the family it is aimed at.
pub fn premut() -> (u32, u32) {
    let raw = std::env::var("ZK_MUT_PREPROCESSED").unwrap_or_default();
    let (name, tgt) = match raw.split_once(':') {
        Some((n, t)) => (n.to_string(), t.to_string()),
        None => (raw, String::new()),
    };
    let m = match name.as_str() {
        "" | "none" => 0,
        "noperm" => 1,
        "nosticky" => 2,
        "noround" => 3,
        "norenorm" => 4,
        "nopad" => 5,
        "nokwin" => 6,
        "onecounter" => 7,
        "nolast" => 8,
        "nofirstrow" => 9,
        "noactive" => 10,
        "noistart" => 11,
        "noinv" => 12,
        "noident" => 13,
        "nopatch" => 14,
        other => panic!("unknown ZK_MUT_PREPROCESSED {other:?}"),
    };
    let t = match tgt.as_str() {
        "" | "all" => AT_ALL,
        "norm" => 0,
        "frac" => 1,
        "mul" => 2,
        "qnz" => 3,
        "range" => 4,
        "chan" => AT_CHAN,
        "key" => AT_KEY,
        "row" => AT_ROW,
        other => panic!("unknown ZK_MUT_PREPROCESSED target {other:?}"),
    };
    (m, t)
}

/// B11 stream M: does some witness generation in this chunk bump a *host* multiplicity
/// counter, for a reason no switch states?
///
/// One key tile can be on the host while the switch says device -- a diagonal block key tile,
/// which generates its witness on the host whatever `ZK_DEVICE_WITNESS_GEN_KEY_TILE` says because the device
/// witness generation has no notion of its layout. `witness_gen_dev` has to know, and it is asked
/// from inside the witness generations themselves (`merge_counters` is read by
/// `mat_component_dev`, `freivalds_component_dev` and `commit_base_dev`), none of which
/// has the layer in hand. So the layer's answer is recorded once, before the
/// first witness generation runs, and read from here.
///
/// Keyed by chunk exactly as the phase timers are, so a pipelined loop -- chunk
/// k proving while chunk k+1 generates -- cannot have one chunk read another's
/// answer. A run that never records one is a run with no key tiles to ask about.
static HOST_WITNESS_GEN: OnceLock<Mutex<std::collections::BTreeMap<u64, bool>>> = OnceLock::new();

fn host_witness_gen_map() -> &'static Mutex<std::collections::BTreeMap<u64, bool>> {
    HOST_WITNESS_GEN.get_or_init(|| Mutex::new(Default::default()))
}

/// Record this chunk's answer. Called once, before the first witness generation.
pub fn note_host_witness_gen(v: bool) {
    host_witness_gen_map().lock().unwrap().insert(crate::util::current_chunk(), v);
}

fn host_witness_gen() -> bool {
    host_witness_gen_map().lock().unwrap().get(&crate::util::current_chunk()).copied().unwrap_or(false)
}

/// Did every witness generation that bumps one of the ten multiplicity counters run on the
/// device? Mode 2 of a witness generation builds both ways and returns the *host* columns
/// (20.2's shape), so only mode 1 leaves a counter where the kernel can sum it.
pub fn witness_gen_dev() -> bool {
    // B11 stream L: the key tile witness generation has its own switch now, and a host-built key tile
    // bumps the host multiplicity arrays. So "every witness generation that bumps a counter
    // ran on the device" has to read the key tile gate too -- otherwise
    // `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1 ZK_DEVICE_WITNESS_GEN_KEY_TILE=0` would claim the counters are on the
    // device while three key tiles' worth of bumps sit on the host.
    //
    // B11 stream M: the switches are not the whole answer. A diagonal block key tile is on
    // the host with every switch at 1, so `ZK_DEVICE_PREPROCESSED=1` on a sequence chunk
    // used to build the nine multiplicity columns on the card from the device
    // counters alone and drop the diagonal block key tiles' chan and key bumps -- accepted
    // by every other check and rejected by `sums_cancel`. Stream L had already
    // met this for the tenth counter, `m_exp`, and fixed it there alone
    // (`dev_attention_freivalds_witness_gen::exp_mult_on_host`); this is the same predicate asked of all
    // ten.
    crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_DEQUANT") == 1
        && crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS") == 1
        && crate::dev_attention_freivalds_witness_gen::key_tile_mode() == 1
        && !host_witness_gen()
}

/// The ten multiplicity columns are the device's exactly when every counter
/// that feeds one is there. With a generate its witness on the host they stay host columns and
/// 4a uploads them -- the honest accounting, not a special case.
pub fn mult_dev() -> bool {
    mode() >= 1 && witness_gen_dev()
}

/// Does a witness generation still owe the host its counters? 21.4's "D2H and merge" line,
/// for the nine counters 23.1 left behind: it disappears exactly when the
/// kernel sums them instead.
pub fn merge_counters() -> bool {
    !(mode() == 1 && witness_gen_dev())
}

fn round_code(r: Round) -> u32 {
    match r {
        Round::Even => 0,
        Round::Away => 1,
        Round::Zero => 2,
    }
}

fn up_u32(v: Vec<u32>) -> Uint32Vec {
    let bytes = (v.len() * 4) as u64;
    let t = std::time::Instant::now();
    let d = Uint32Vec::from_vec(v);
    crate::device::note_witness(bytes, t.elapsed().as_secs_f64());
    d
}

// ------------------------------------------------------- the three closed forms --

/// `bf16::norm_table`'s row `d`: `(m2, ph, st, m7, expfield, nz)`.
fn norm_closed(d: usize, sticky: bool, round: Round) -> [u32; 6] {
    if d == 0 {
        return [0; 6];
    }
    if d < 256 {
        let (_, ef, m7) = crate::bf16::decode(crate::bf16::from_f32(d as f32));
        return [d as u32, 1, 0, m7, ef, 1];
    }
    let per_k = if sticky { 512 } else { 256 };
    let idx = d - 256;
    let k = (idx / per_k) as u32;
    let rem = idx % per_k;
    let m2 = (256 + if sticky { rem / 2 } else { rem }) as u32;
    let st = if sticky { (rem % 2) as u32 } else { 0 };
    let (m, rb) = (m2 >> 1, m2 & 1);
    let up = round.carry(m, rb, st);
    let (msig, extra) = if m + up == 256 { (128, 1) } else { (m + up, 0) };
    [m2, 1 << k, st, msig - 128, k + 1 + extra + EXP_BIAS, 1]
}

/// The fraction table's row `d`: `(ph, r, st)`, with `2^k <= d < 2^(k+1)`.
fn frac_closed(d: usize, sticky: bool) -> [u32; 3] {
    if d == 0 {
        return [0; 3];
    }
    let k = usize::BITS - 1 - d.leading_zeros();
    let r = d - (1usize << k);
    [1u32 << k, r as u32, u32::from(sticky && r != 0)]
}

/// `bf16::mul_table`'s row `d`: `(a, b, m7, carry)`.
fn mul_closed(d: usize, round: Round) -> [u32; 4] {
    let (a, b) = ((d >> 7) as u32, (d & 127) as u32);
    let p = (128 + a) * (128 + b);
    let carry = (p >> 15) & 1;
    let shift = 7 + carry;
    let m = p >> shift;
    let rb = (p >> (shift - 1)) & 1;
    let st = u32::from(p & ((1 << (shift - 1)) - 1) != 0);
    let up = round.carry(m, rb, st);
    let (msig, extra) = if m + up == 256 { (128, 1) } else { (m + up, 0) };
    [a, b, msig - 128, carry + extra]
}

/// The rows where this run's own table is not the closed form.
///
/// Empty on an honest run; one entry for `ProverTamper::norm_row`. A long list
/// means the closed form is wrong for this design point, and that is a panic
/// and not a silent upload of the table it was meant to replace.
fn patch_of(what: &str, n: usize, w: usize,
            row: impl Fn(usize) -> Option<Vec<u32>> + Sync) -> (Vec<u32>, Vec<u32>) {
    const MAX: usize = 4096;
    let bad: Vec<(u32, Vec<u32>)> = (0..n)
        .into_par_iter()
        .filter_map(|i| row(i).map(|v| (i as u32, v)))
        .collect();
    assert!(
        bad.len() <= MAX,
        "the {what} table differs from its closed form in {} rows: zk_pre_witness_gen \
         expects the closed form plus at most a probe's tamper",
        bad.len()
    );
    let mut idx = Vec::with_capacity(bad.len());
    let mut val = Vec::with_capacity(bad.len() * w);
    for (i, v) in bad {
        idx.push(i);
        val.extend(v);
    }
    (idx, val)
}

// ------------------------------------------------------------ the shared tables --

/// One shared table's slot in the two flat lists.
struct TabShape {
    kind: u32,
    log_rows: u32,
    real_rows: usize,
    n_cols: usize,
}

/// The 21 preprocessed columns and the 10 multiplicity columns, on the device.
pub struct DevPreTables {
    pub pre: Vec<BaseFieldVec>,
    /// empty when some witness generation that bumps one of these tables ran on the host, so
    /// there is a counter missing and the host keeps all ten -- the honest
    /// accounting, not a special case
    pub mult: Vec<BaseFieldVec>,
    pub patched: usize,
}

/// Build them.
///
/// `counters` is one list per table in the order the lists are laid out --
/// norm, frac, mul, qnz, then one per range width -- holding the device counter
/// arrays every witness generation that bumps that table scatter-added into. `want_mult` is
/// false when any of those witness generations ran on the host instead.
pub fn tables_dev(t: &Tables, log_norm: u32, log_frac: u32, log_mul: u32, q_bits: u32,
                  widths: &[u32], counters: &[Vec<&Uint32Vec>], want_mult: bool)
                  -> DevPreTables {
    let (mutate, at) = premut();
    let sticky = t.round.needs_sticky();
    let mut shapes = vec![
        TabShape { kind: KIND_NORM, log_rows: log_norm, real_rows: t.norm.len(), n_cols: 6 },
        TabShape { kind: KIND_FRAC, log_rows: log_frac, real_rows: t.frac.len(), n_cols: 3 },
        TabShape { kind: KIND_MUL, log_rows: log_mul, real_rows: t.mul.len(), n_cols: 4 },
        TabShape { kind: KIND_QNZ, log_rows: q_bits, real_rows: 1 << q_bits, n_cols: 2 },
    ];
    for w in widths {
        shapes.push(TabShape { kind: KIND_RANGE, log_rows: *w, real_rows: 1usize << w,
                               n_cols: 1 });
    }
    assert_eq!(counters.len(), shapes.len(), "one counter list per shared table");

    // Section 5 rule 8: the rows where this run's table is not the closed form.
    // `qnz` and the ranges have no host table at all -- they are `0..2^w`
    // generated inline -- so there is nothing there to diff.
    let t_d = std::time::Instant::now();
    let norm_p = patch_of("normalisation", t.norm.len(), 6, |i| {
        let r = &t.norm[i];
        let got = [r.m2, r.ph, r.st, r.m7, r.expfield, r.nz];
        (norm_closed(i, sticky, t.round) != got).then(|| got.to_vec())
    });
    let frac_p = patch_of("fraction", t.frac.len(), 3, |i| {
        let x = t.frac[i];
        let got = [x.0, x.1, x.2];
        (frac_closed(i, sticky) != got).then(|| got.to_vec())
    });
    let mul_p = patch_of("multiply", t.mul.len(), 4, |i| {
        let r = t.mul[i];
        let got = [r.a, r.b, r.m7, r.carry];
        (mul_closed(i, t.round) != got).then(|| got.to_vec())
    });
    crate::util::add_phase("gen/pre_diff", t_d.elapsed().as_secs_f64());
    let patched = norm_p.0.len() + frac_p.0.len() + mul_p.0.len();

    let t_u = std::time::Instant::now();
    let patches: Vec<(Uint32Vec, Uint32Vec, u32)> = [norm_p, frac_p, mul_p]
        .into_iter()
        .map(|(i, v)| {
            let n = i.len() as u32;
            (up_u32(if i.is_empty() { vec![0] } else { i }),
             up_u32(if v.is_empty() { vec![0] } else { v }), n)
        })
        .collect();
    crate::util::add_phase("gen/pre_up", t_u.elapsed().as_secs_f64());

    let t_a = std::time::Instant::now();
    let pre: Vec<Vec<BaseFieldVec>> = shapes
        .iter()
        .map(|s| {
            (0..s.n_cols).map(|_| BaseFieldVec::new_uninitialized(1usize << s.log_rows))
                .collect()
        })
        .collect();
    let mult: Vec<BaseFieldVec> = if want_mult {
        shapes.iter()
            .map(|s| BaseFieldVec::new_uninitialized(1usize << s.log_rows))
            .collect()
    } else {
        vec![]
    };
    crate::util::add_phase("gen/pre_alloc", t_a.elapsed().as_secs_f64());

    let t_k = std::time::Instant::now();
    for (k, s) in shapes.iter().enumerate() {
        let rows = 1usize << s.log_rows;
        let cptr: Vec<*const u32> = counters[k]
            .iter()
            .map(|c| {
                assert_eq!(c.size, rows, "a multiplicity counter is not 1 << log_rows wide");
                c.device_ptr
            })
            .collect();
        assert!(!want_mult || !cptr.is_empty(),
                "shared table {k} has no device counter to sum");
        let cols: Vec<*mut u32> = pre[k].iter().map(|c| c.device_ptr as *mut u32).collect();
        let cfg = FPreCfg {
            kind: s.kind,
            log_rows: s.log_rows,
            rows: rows as u32,
            real_rows: s.real_rows as u32,
            kmax: t.kmax,
            round: round_code(t.round),
            sticky: u32::from(sticky),
            n_cols: s.n_cols as u32,
            n_counters: cptr.len() as u32,
            mutate,
            at,
        };
        let (pi, pv, pn) = match k {
            0 => (patches[0].0.device_ptr, patches[0].1.device_ptr, patches[0].2),
            1 => (patches[1].0.device_ptr, patches[1].1.device_ptr, patches[1].2),
            2 => (patches[2].0.device_ptr, patches[2].1.device_ptr, patches[2].2),
            _ => (std::ptr::null(), std::ptr::null(), 0),
        };
        unsafe {
            zk_witness_gen_pre_table(
                cptr.as_ptr(),
                pi,
                pv,
                pn,
                cols.as_ptr(),
                mult.get(k).map(|c| c.device_ptr as *mut u32).unwrap_or(std::ptr::null_mut()),
                cfg,
            );
        }
    }
    crate::util::add_phase("gen/pre_kernel", t_k.elapsed().as_secs_f64());
    eprintln!(
        "pre-dev: {} shared tables -- {} preprocessed and {} multiplicity columns, \
         patch {} rows, kernels {:.4} s",
        shapes.len(),
        pre.iter().map(|v| v.len()).sum::<usize>(),
        mult.len(),
        patched,
        t_k.elapsed().as_secs_f64()
    );
    DevPreTables { pre: pre.into_iter().flatten().collect(), mult, patched }
}

// ------------------------------------------------------------ the key tile selectors --

/// One key tile's three sub-components: the flags, and the selector columns built out
/// of them.
///
/// The flags outlive the challenge draw on purpose. `commit_pre` runs before it
/// and the accumulators after, and this is the array that lets both read one
/// computation instead of two.
pub struct DevPreKeyTile {
    pub flags: Vec<Uint32Vec>,
    /// the 16 selector columns, or empty when the host still builds them
    pub cols: Vec<BaseFieldVec>,
    pub logs: Vec<u32>,
}

impl DevPreKeyTile {
    pub fn chan(&self) -> &Uint32Vec {
        &self.flags[0]
    }
    pub fn key(&self) -> &Uint32Vec {
        &self.flags[1]
    }
    /// B11 stream N: absent on a diagonal block key tile, which has no per-token component.
    /// Every caller is already inside a `with_rows` arm, so reaching this on a
    /// diagonal block is a bug in the caller and says so rather than indexing past the
    /// end.
    pub fn row(&self) -> &Uint32Vec {
        assert_eq!(self.flags.len(), 3,
                   "a diagonal block key_tile has no per-token flags: this caller is outside a \
                    `with_rows` arm");
        &self.flags[2]
    }
}

/// The per-instance key scale and its inverse, as one array: `1/delta` first,
/// `delta` after, which is what makes `noinv` a control rather than an argument.
fn scale_of(a: &Attn) -> Vec<u32> {
    let inv: Vec<u32> = a
        .delta
        .iter()
        .map(|d| {
            if a.variant == Variant::Bf16x {
                return M31::one().0;
            }
            let m = m31_of(*d);
            assert!(!m.is_zero(), "an exp-table key scale is zero mod p");
            m.inverse().0
        })
        .collect();
    let mut v = inv;
    v.extend(a.delta.iter().map(|d| m31_of(*d).0));
    v
}

/// Build one key tile's flags and, when the device owns them, its 16 selectors.
pub fn key_tile_dev(a: &Attn, l: &Layouts, want_cols: bool, with_rows: bool) -> DevPreKeyTile {
    let (mutate, at) = premut();
    let sh = a.shape;
    let rrows = 1usize << l.log_row;
    // the per-token component is not a `Grid`, so it borrows the shape for its two
    // size fields and nothing else
    let tokg = FGridA { log_rows: l.log_row, rows: rrows as u32, groups: 1, chunks: 1,
                        lead: 0, descending: 0 };
    // B11 stream N, lever 2 (b): `SHAPE_TOK` was carried unconditionally, and a
    // diagonal block key tile has no rows for it. `commit_pre` has always skipped those
    // three selector columns on a diagonal block (`attn_trace.rs:258`), so a diagonal block
    // key tile's device selectors have to skip them too or the two disagree by
    // three columns. The *flags* are cut with them: nothing segments a component
    // that does not exist.
    // Stream CC (E11): a ninth key selector, `act`, when the grid leaves inert
    // lead rows -- the AIR's numerator-fold guard.  `commit_pre` appends it to
    // the end of the grid's block and a device key tile is never a diagonal block key tile, so
    // on this side it is slot 8 of the key group and the two lists still line up
    // column for column (`check_run` is what says so).
    let n_key = 8 + usize::from(crate::attn_air::act_id(0, a.diagonal_block, l.key.g.lead)
                                .is_some());
    let shapes: &[(u32, FGridA, u32, u32, usize, u32)] = if with_rows {
        &[
            (SHAPE_CHAN, FGridA::of(&l.chan.g), sh.hd as u32, 0, 5, AT_CHAN),
            (SHAPE_KEY, FGridA::of(&l.key.g), sh.c as u32, 0, n_key, AT_KEY),
            (SHAPE_TOK, tokg, sh.s as u32, l.rw as u32, 3, AT_ROW),
        ]
    } else {
        &[
            (SHAPE_CHAN, FGridA::of(&l.chan.g), sh.hd as u32, 0, 5, AT_CHAN),
            (SHAPE_KEY, FGridA::of(&l.key.g), sh.c as u32, 0, n_key, AT_KEY),
        ]
    };

    let t_f = std::time::Instant::now();
    let flags: Vec<Uint32Vec> = shapes
        .iter()
        .map(|(shape, g, modulus, rw, _, _)| {
            let f = Uint32Vec::new_zeroes(g.rows as usize);
            unsafe {
                zk_flags_key_tile(f.device_ptr as *mut u32,
                               FAccFlags { shape: *shape, g: *g, modulus: *modulus,
                                           rw: *rw });
            }
            f
        })
        .collect();
    crate::util::add_phase("gen/pre_flags", t_f.elapsed().as_secs_f64());

    let mut cols: Vec<BaseFieldVec> = vec![];
    let mut logs: Vec<u32> = vec![];
    if !want_cols {
        return DevPreKeyTile { flags, cols, logs };
    }

    let t_s = std::time::Instant::now();
    let d_scale = up_u32(scale_of(a));
    for (k, (shape, g, _, _, n_cols, fam)) in shapes.iter().enumerate() {
        let mine: Vec<BaseFieldVec> = (0..*n_cols)
            .map(|_| BaseFieldVec::new_uninitialized(g.rows as usize))
            .collect();
        let p: Vec<*mut u32> = mine.iter().map(|c| c.device_ptr as *mut u32).collect();
        let cfg = FSelCfg {
            shape: *shape,
            g: *g,
            ctx: sh.c as u32,
            n_insts: sh.insts as u32,
            with_scale: u32::from(a.variant != Variant::Bf16x),
            n_cols: *n_cols as u32,
            fam: *fam,
            mutate,
            at,
        };
        unsafe {
            zk_witness_gen_sel(flags[k].device_ptr, p.as_ptr(), d_scale.device_ptr, cfg);
        }
        logs.extend(std::iter::repeat(g.log_rows).take(*n_cols));
        cols.extend(mine);
    }
    crate::util::add_phase("gen/pre_sel", t_s.elapsed().as_secs_f64());
    DevPreKeyTile { flags, cols, logs }
}

// --------------------------------------------------------------------- the check --

/// Assert a run of device columns against the host's, cell for cell, naming the
/// column and the cell on a mismatch.
///
/// Not a sample: a witness generation that is wrong on one cell of one column is wrong, and
/// the point of checking here rather than against a proof digest is that the
/// panic says where.
pub fn check_run(what: &str, host: &[&Eval], got: &[BaseFieldVec]) {
    assert_eq!(host.len(), got.len(), "{what}: column count");
    let mut cells = 0usize;
    for (i, (a, b)) in host.iter().zip(got).enumerate() {
        let (a, b) = (a.values.to_cpu(), b.to_vec());
        assert_eq!(a.len(), b.len(), "{what}: column {i} length");
        if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
            panic!("{what}: column {i} differs at cell {k}: host {:?}, device {:?}",
                   a[k], b[k]);
        }
        cells += a.len();
    }
    eprintln!("pre-dev: {what} -- {} columns, {cells} cells match", host.len());
}
