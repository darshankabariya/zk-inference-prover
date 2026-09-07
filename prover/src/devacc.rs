//! B7a milestone 4 (section 3 row 1): the Horner accumulators, on the device.
//!
//! 23.4 left 4a at 0.066 GB in 351 columns. **272 of those columns are these**
//! -- 0.050 GB, three quarters of everything left, and the last committed
//! columns in the composed proof that are still born on the host:
//!
//! | region | columns | what they are |
//! |---|---|---|
//! | 3 key tiles x channel | `u`, `v`, `acc` | Q.K^T's inner products and their folded sum |
//! | 3 key tiles x grid | `zs`, `hs`, `zn`, `wv`, `av`, `sn` | the score, numerator and value Horners |
//! | 3 key tiles x per-token | `am`, `az` | the row maxima and denominators |
//! | 7 dequant regions | `z`, `h` | the Freivalds claimed side |
//! | attnout | `z`, `h`, `dz` | the output and denominator claims |
//! | 4 Freivalds groups | `v`, `us[m]`, `accs[m]` | `r^T W` per matmul |
//!
//! **They are one primitive, six times.** Every one is `x[j] = a[j] x[j-1] +
//! b[j]` down the *coset* rows of a region, with `b[j]` an inner Horner over
//! that row's own committed columns and `a[j]` a challenge power the layout
//! picks. Affine maps compose associatively, so a parallel scan gives the same
//! field element the host's serial carry gives -- exactly, because M31 is a
//! field and reassociation is not an approximation. A group reset is `a = 0`
//! and an unscanned lane is `a = 0` for every row, so neither is a special case.
//!
//! **What the kernels read.** The host builds `b[j]` out of the witness
//! (`a.kk`, `m.acc`, `m0.qx`, `ao.acc`); the kernels build it out of the
//! columns 20-22's fills wrote those witnesses into, which are already on the
//! device in committed order. That is the AIR's own route -- the constraint
//! reads the same columns -- so nothing is uploaded, nothing is recomputed, and
//! a probe that corrupts a witness corrupts the column and the accumulator
//! carries it (section 5 rule 8 has nothing to repair here).
//!
//! The committed column is not always the witness value, and the four places
//! where it is not are the four the controls are aimed at:
//!
//!  * the channel region commits `kk + off_k` and `qq + off_q` (`nooffset`),
//!    the grid commits `vv + 128` and the Freivalds checks `qx + 128`;
//!  * the grid's key column commits the shift **scaled** to the master exp
//!    table's exponent, `d delta`, while the score accumulator wants `d`
//!    (`nodelta` -- the same preprocessed inverse the AIR divides by);
//!  * at `intsplit`'s design point it commits that scaled shift **split**,
//!    `(d delta) >> k` in the key column and the low `k` bits in the auxiliary
//!    one, so the accumulator reassembles `idx 2^k + aux` first (`nosplit`).
//!    Rung 3 is the only place that point is ever proved, and it is what the
//!    probe-point check caught (section 5 rule 11);
//!  * the grid's numerator column commits `table[0]` on an inert lead row,
//!    where the host's sum has no term at all (`nomask`).
//!
//! Stream CC (E11): `nomask` had never had anything to catch. Every grid proved
//! before E11 tiled exactly, so there was no inert lead row on any shape it was
//! ever run at, and `m4g_run.sh:164` and `dd_prefill.sh:183` both record that as
//! a vacuous control rather than a pass. It is not vacuous now: `KeyEval`'s
//! numerator fold carries an `act` factor, so on a padded grid the honest device
//! sum is the one the AIR wants and `nomask` is a real divergence from it. Run it
//! at a shape with `key_tile_leads.key > 0` or it says nothing.
//!
//! `B7_M4_ACC` says which path runs, on the three-value pattern every fill in
//! this series uses: 0 the host builds them, 1 the device does and they are
//! never on the host, 2 both with every cell asserted equal. Mode 2 is a check
//! with teeth even with every fill on the device, because the host side reads
//! the *witness* and the device side reads the *columns*: it closes the
//! witness -> fill -> column -> accumulator chain end to end.

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};
use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;

use crate::attn_air::Powers;
use crate::attn_trace::m31_of;
use crate::attn_witness::{Attn, Variant};
use crate::common::Eval;
use crate::device::AnyCol;
use crate::layout::{Grid, LayoutChan, LayoutKey};

// ---------------------------------------------------------------- the FFI shapes --

/// Mirrors `B65AccGrid` in `b65_acc.cuh`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FGridA {
    pub log_rows: u32,
    pub rows: u32,
    pub groups: u32,
    pub chunks: u32,
    pub lead: u32,
    pub descending: u32,
}

impl FGridA {
    pub fn of(g: &Grid) -> Self {
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

/// Mirrors `B65AccS2`: one stage-2 lane.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FAccS2 {
    pub dst: u32,
    pub src0: u32,
    pub src1: u32,
    pub sel_i: u32,
    pub m_i: [u32; 4],
    pub m_l: [u32; 4],
    pub m_n: [u32; 4],
}

/// Mirrors `B65AccResidualStreamBind`: one of the residual-stream region's Horner bindings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccResidualStreamBinding {
    pub src: u32,
    pub off: u32,
    pub width: u32,
    pub stride: u32,
    pub w_off: u32,
    pub pad: u32,
    pub step: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccChan {
    pub g: FGridA,
    pub pk: u32,
    pub pq: u32,
    pub hd: u32,
    pub off_k: u32,
    pub off_q: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccKey {
    pub g: FGridA,
    pub q: u32,
    pub pv: u32,
    pub ctx: u32,
    pub n_insts: u32,
    pub off_v: u32,
    pub two_k: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccTok {
    pub log_rows: u32,
    pub rows: u32,
    pub rw: u32,
    pub s_inst: u32,
    pub with_max: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccFreivalds {
    pub g: FGridA,
    pub pq: u32,
    pub pk: u32,
    pub n_mats: u32,
    pub off_x: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccDeq {
    pub log_rows: u32,
    pub rows: u32,
    pub per_row: u32,
    pub chunks: u32,
    pub mutate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAccAo {
    pub log_rows: u32,
    pub rows: u32,
    pub per_row: u32,
    pub chunks: u32,
    pub lead: u32,
    pub nh: u32,
    pub s: u32,
    pub mutate: u32,
}

unsafe extern "C" {
    fn b65_acc_scan(a: *mut u32, b: *mut u32, rows: u32, lanes: u32, mutate: u32);
    fn b65_acc_stage2(flags: *const u32, a: *mut u32, b: *mut u32, lanes: *const FAccS2,
                      n_lanes: u32, rows: u32, mutate: u32);
    fn b65_acc_split(v: *const u32, out: *const *mut u32, lane: *const u32, n_out: u32,
                     rows: u32, log_rows: u32, mutate: u32);
    fn b65_acc_last(v: *const u32, host: *mut u32, rows: u32, lanes: u32);
    fn b65_acc_chan(cols: *const *const u32, r_pow: *const u32, sq_pow: *const u32,
                    r_step: [u32; 4], sq_step: [u32; 4], flags: *const u32, a: *mut u32,
                    b: *mut u32, cfg: FAccChan);
    fn b65_acc_grid(key_cols: *const *const u32, num_cols: *const *const u32,
                    aux_cols: *const *const u32, val_cols: *const *const u32,
                    sg_pow: *const u32, rv_pow: *const u32, sg_step: [u32; 4],
                    rv_step: [u32; 4], inv_delta: *const u32, flags: *const u32,
                    a: *mut u32, b: *mut u32, cfg: FAccKey);
    fn b65_acc_tok(max_cols: *const *const u32, z_cols: *const *const u32,
                   sr_pow: *const u32, sr_step: [u32; 4], mu: [u32; 4],
                   flags: *const u32, a: *mut u32, b: *mut u32, cfg: FAccTok);
    fn b65_acc_freivalds(base_cols: *const *const u32, pre_cols: *const *const u32,
                         s_pow: *const u32, rv_pow: *const u32, s_step: [u32; 4],
                         rv_step: [u32; 4], flags: *mut u32, a: *mut u32, b: *mut u32,
                         cfg: FAccFreivalds);
    fn b65_acc_deq(acc_cols: *const *const u32, rv_pow: *const u32, rv_step: [u32; 4],
                   flags: *mut u32, a: *mut u32, b: *mut u32, cfg: FAccDeq);
    fn b65_acc_ao(den_col: *const u32, acc_cols: *const *const u32, rv_pow: *const u32,
                  rv_step: [u32; 4], flags: *mut u32, a: *mut u32, b: *mut u32,
                  cfg: FAccAo);
    fn b65_acc_residual_stream_bind(in_nat: *const i32, out_nat: *const i32, weights: *const u32,
                                    n_weights: u32, lam: [u32; 4], specs: *const FAccResidualStreamBinding,
                                    n_specs: u32, a: *mut u32, b: *mut u32, rows: u32, mutate: u32);
}

/// `B65A_ISTART`, the only flag a stage-2 lane ever selects on.
const SEL_ISTART: u32 = 8;
/// `B65A_ONE`: a stage-2 lane with one factor.
const ONE_SRC: u32 = u32::MAX;

// ------------------------------------------------------------------ the switches --

/// `B7_M4_ACC`: 0 the host, 1 the device, 2 both with every cell asserted.
pub fn mode() -> u32 {
    crate::util::m4b_mode("B7_M4_ACC")
}

/// Which region an accumulator belongs to, so a control can be aimed at one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg {
    Chan,
    Grid,
    Tok,
    Freivalds,
    Deq,
    Ao,
    /// B11 stream N: the residual-stream region's Horner bindings, the seventh region and the
    /// only one whose values are not committed columns.
    ResidualStreamBinding,
}

impl Reg {
    fn name(self) -> &'static str {
        match self {
            Reg::Chan => "chan",
            Reg::Grid => "grid",
            Reg::Tok => "tok",
            Reg::Freivalds => "freivalds",
            Reg::Deq => "deq",
            Reg::Ao => "ao",
            Reg::ResidualStreamBinding => "residual_stream_bind",
        }
    }
}

/// `B7_M4_ACCMUT`, `name` or `name:region`.
///
/// **The suffix is section 5 rule 5, again.** The key tiles are built first and the
/// channel region first of all, so an unaimed defect always panics at `chan
/// key_tile0` and the run says nothing about whether the grid, the per-token block,
/// the Freivalds checks, the dequant or the attnout would have caught the same thing.
pub fn accmut(reg: Reg) -> u32 {
    let raw = std::env::var("B7_M4_ACCMUT").unwrap_or_default();
    let (name, want) = match raw.split_once(':') {
        Some((n, r)) => (n.to_string(), Some(r.to_string())),
        None => (raw, None),
    };
    if let Some(r) = want.as_deref() {
        assert!(["chan", "grid", "tok", "freivalds", "deq", "ao", "residual_stream_bind"].contains(&r),
                "unknown B7_M4_ACCMUT region {r:?}");
        if r != reg.name() {
            return 0;
        }
    }
    match name.as_str() {
        "" | "none" => 0,
        "noperm" => 1,
        "nocarry" => 2,
        "noreset" => 3,
        "noistart" => 4,
        "nooffset" => 5,
        "nodelta" => 6,
        "nomask" => 7,
        "everyrow" => 8,
        "nosplit" => 9,
        // B11 stream N: the bindings' three factors, one control each.
        "notiestep" => 10,
        "notiew" => 11,
        "notielam" => 12,
        other => panic!("unknown B7_M4_ACCMUT {other:?}"),
    }
}

// -------------------------------------------------------------------- the scaffold --

fn q4(v: SecureField) -> [u32; 4] {
    v.to_m31_array().map(|x| x.0)
}

fn qone() -> [u32; 4] {
    q4(SecureField::one())
}

fn pows(v: &[SecureField]) -> Vec<[u32; 4]> {
    v.iter().map(|x| q4(*x)).collect()
}

fn ptrs(cols: &[BaseFieldVec]) -> Vec<*const u32> {
    cols.iter().map(|c| c.device_ptr).collect()
}

/// One region's lanes: `(a, b)` per lane in coset row order, the per-row layout
/// flags, and the scan that closes them.
struct Acc {
    rows: usize,
    log_rows: u32,
    lanes: usize,
    a: SecureFieldVec,
    b: SecureFieldVec,
    /// B7a milestone 4 (section 3 row 1): the three key tile shapes' flags belong
    /// to `devpre` now -- one array per sub-region, written by
    /// `b65_flags_key_tile` at fill time, read both by the selector columns
    /// `commit_pre` used to build on the host and by the segmentation here. The
    /// Freivalds check, dequant and attnout regions have no preprocessed selectors left on
    /// the host, so they still own theirs. `own` keeps a borrowed array's owner
    /// honest: it outlives this `Acc` because it is the caller's.
    own: Option<Uint32Vec>,
    fptr: *const u32,
}

impl Acc {
    fn new(log_rows: u32, lanes: usize) -> Self {
        let rows = 1usize << log_rows;
        let f = Uint32Vec::new_zeroes(rows);
        let fptr = f.device_ptr;
        Self {
            rows,
            log_rows,
            lanes,
            a: SecureFieldVec::new_uninitialized(lanes * rows),
            b: SecureFieldVec::new_uninitialized(lanes * rows),
            own: Some(f),
            fptr,
        }
    }

    /// The same, over flags somebody else wrote -- the three key tile shapes.
    fn borrowed(log_rows: u32, lanes: usize, f: &Uint32Vec) -> Self {
        let rows = 1usize << log_rows;
        assert_eq!(f.size, rows, "the flags array is not this region's height");
        Self {
            rows,
            log_rows,
            lanes,
            a: SecureFieldVec::new_uninitialized(lanes * rows),
            b: SecureFieldVec::new_uninitialized(lanes * rows),
            own: None,
            fptr: f.device_ptr,
        }
    }

    fn ap(&self) -> *mut u32 {
        self.a.device_ptr as *mut u32
    }

    fn bp(&self) -> *mut u32 {
        self.b.device_ptr as *mut u32
    }

    fn fp(&self) -> *mut u32 {
        self.fptr as *mut u32
    }

    fn fc(&self) -> *const u32 {
        self.fptr
    }

    /// The inclusive affine scan over lanes `[lo, lo + n)`. A QM31 is four
    /// `u32`, which is what the lane offset is counted in.
    fn scan(&self, lo: usize, n: usize, mutate: u32) {
        let off = 4 * lo * self.rows;
        unsafe {
            b65_acc_scan(self.ap().add(off), self.bp().add(off), self.rows as u32,
                         n as u32, mutate);
        }
    }

    fn stage2(&self, lanes: &[FAccS2], mutate: u32) {
        unsafe {
            b65_acc_stage2(self.fc(), self.ap(), self.bp(), lanes.as_ptr(),
                           lanes.len() as u32, self.rows as u32, mutate);
        }
    }

    /// The committed columns, four M31 per lane, in the order `order` names.
    fn split(&self, order: &[u32], mutate: u32) -> Vec<BaseFieldVec> {
        let out: Vec<BaseFieldVec> =
            (0..4 * order.len()).map(|_| BaseFieldVec::new_uninitialized(self.rows)).collect();
        let p: Vec<*mut u32> = out.iter().map(|c| c.device_ptr as *mut u32).collect();
        unsafe {
            b65_acc_split(self.b.device_ptr, p.as_ptr(), order.as_ptr(),
                          order.len() as u32, self.rows as u32, self.log_rows, mutate);
        }
        out
    }

    /// Every lane's last coset row: the claimed sums, and the only thing that
    /// comes back across the link.
    fn last(&self) -> Vec<SecureField> {
        let mut host = vec![SecureField::zero(); self.lanes];
        unsafe {
            b65_acc_last(self.b.device_ptr, host.as_mut_ptr() as *mut u32,
                         self.rows as u32, self.lanes as u32);
        }
        host
    }
}

/// What one region's accumulators came to: the committed columns in AIR order,
/// and the claims in the order the caller asked for them.
pub struct DevAccCols {
    pub cols: Vec<BaseFieldVec>,
    pub claims: Vec<SecureField>,
    pub log_rows: u32,
}

impl DevAccCols {
    /// The columns, on their way into a tree. They were computed where they
    /// will be committed, so they are moved and never copied.
    pub fn into_any(self) -> Vec<AnyCol<'static>> {
        let log_rows = self.log_rows;
        self.cols.into_iter().map(|c| AnyCol::OwnedDev(c, log_rows)).collect()
    }
}

fn note(reg: Reg, tag: &str, n_cols: usize, rows: usize, secs: f64) {
    crate::util::m4_add("int/acc_kernel", secs);
    if mode() == 2 {
        eprintln!("acc-dev: {} {tag} -- {n_cols} columns, {} cells, kernel {secs:.4} s",
                  reg.name(), n_cols * rows);
    }
}

/// One region's accumulators, on their way into the tree.
///
/// `host` is `Some` only in mode 2, and then it is what the device columns are
/// compared against and what the proof is built from -- so mode 2's digest is
/// mode 0's digest, and a difference is a panic that names the cell rather than
/// a hash that does not.
pub fn place(tag: &str, got: DevAccCols, host: Option<(Vec<Eval>, Vec<SecureField>)>,
             sink: &mut crate::device::ColSink) -> Vec<SecureField> {
    match host {
        Some((h, claims)) => {
            check(tag, &h, &claims, &got);
            sink.extend(h);
            claims
        }
        None => {
            let claims = got.claims.clone();
            sink.extend_any(got.into_any());
            claims
        }
    }
}

/// Assert one region's device accumulators against the host's, cell for cell,
/// and the claims exactly.
///
/// Not a sample: an accumulator that is wrong on one cell is wrong, and the
/// point of checking here rather than against a proof digest is that the panic
/// says which column and which cell.
pub fn check(tag: &str, host: &[Eval], want: &[SecureField], got: &DevAccCols) {
    crate::devwfill::check_cols(tag, host, &got.cols);
    assert_eq!(want.len(), got.claims.len(), "{tag}: claim count");
    for (i, (a, b)) in want.iter().zip(&got.claims).enumerate() {
        assert_eq!(a, b, "{tag}: claim {i} differs: host {a:?}, device {b:?}");
    }
    let cells: usize = got.cols.iter().map(|c| c.size).sum();
    eprintln!("acc-dev: {tag} -- {} columns, {cells} cells and {} claims match",
              got.cols.len(), want.len());
}

// ========================================================== the key tile channel ==

/// `u`, `v` and `acc`: Q.K^T's two inner products and the sum of their product.
///
/// `at` is `Committed::at`, so the channel region's columns are
/// `base[at[0] .. at[1]]` -- the keys then the queries, exactly as
/// `commit_base` pushes them.
pub fn chan(base: &[BaseFieldVec], at: &[usize], lc: &LayoutChan, a: &Attn,
            pow: &Powers, flags: &Uint32Vec) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Chan);
    let cols = ptrs(&base[at[0]..at[1]]);
    assert_eq!(cols.len(), lc.pk + lc.pq, "the channel region's column count");
    assert_eq!(pow.r_pow.len(), lc.pk, "r_pow width");
    assert_eq!(pow.sq_pow.len(), lc.pq, "sq_pow width");
    let acc = Acc::borrowed(lc.g.log_rows, 3, flags);
    let (rp, sq) = (pows(&pow.r_pow), pows(&pow.sq_pow));
    let cfg = FAccChan {
        g: FGridA::of(&lc.g),
        pk: lc.pk as u32,
        pq: lc.pq as u32,
        hd: a.shape.hd as u32,
        off_k: m31_of(a.off_k() as i64).0,
        off_q: m31_of(a.off_q() as i64).0,
        mutate,
    };
    unsafe {
        b65_acc_chan(cols.as_ptr(), rp.as_ptr() as *const u32, sq.as_ptr() as *const u32,
                     q4(pow.r_step), q4(pow.sq_step), acc.fc(), acc.ap(), acc.bp(), cfg);
    }
    acc.scan(0, 2, mutate);
    // `acc[j] = acc[j-1] * (istart ? r^C : 1) + (last ? u[j] v[j] : 0)`.
    acc.stage2(&[FAccS2 { dst: 2, src0: 0, src1: 1, sel_i: SEL_ISTART,
                          m_i: q4(pow.r_c), m_l: qone(), m_n: qone() }], mutate);
    acc.scan(2, 1, mutate);
    let claims = acc.last();
    let cols = acc.split(&[0, 1, 2], mutate);
    note(Reg::Chan, "key_tile", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims: vec![claims[2]], log_rows: lc.g.log_rows }
}

// ============================================================= the key tile grid ==

/// `zs`, `hs`, `zn`, `wv`, `av`, `sn`, in the order the host commits them.
///
/// `inv_delta` is the master exp table's key scale, inverted, per instance --
/// the same preprocessed constant `commit_pre` builds and the AIR divides by
/// inside the score accumulator, because the key column commits `d delta` and
/// the accumulator wants `d`.
pub fn grid(base: &[BaseFieldVec], at: &[usize], lk: &LayoutKey, a: &Attn,
            n_aux: usize, pow: &Powers, flags: &Uint32Vec) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Grid);
    let b0 = at[1];
    assert_eq!(at[2] - at[1], 2 * lk.q + n_aux + lk.pv, "the grid region's column count");
    let key = ptrs(&base[b0..b0 + lk.q]);
    let num = ptrs(&base[b0 + lk.q..b0 + 2 * lk.q]);
    // `intsplit(k)` commits `(d delta) >> k` in the key column and the low `k`
    // bits in the auxiliary one, so the score accumulator has to reassemble
    // `d delta = idx 2^k + aux` before dividing `delta` out -- and rung 3's is
    // the only design point where it is ever proved (section 5 rule 11).
    let aux: Vec<*const u32> = match a.variant {
        Variant::IntSplit(_) => {
            assert_eq!(n_aux, lk.q, "intsplit commits one auxiliary column per score");
            ptrs(&base[b0 + 2 * lk.q..b0 + 3 * lk.q])
        }
        _ => vec![],
    };
    let two_k = match a.variant {
        Variant::IntSplit(k) => m31_of(1i64 << k).0,
        _ => M31::one().0,
    };
    let val = ptrs(&base[b0 + 2 * lk.q + n_aux..b0 + 2 * lk.q + n_aux + lk.pv]);
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
    assert_eq!(inv.len(), a.shape.insts, "one key scale per instance");
    let acc = Acc::borrowed(lk.g.log_rows, 6, flags);
    let (sg, rv) = (pows(&pow.sg_pow), pows(&pow.rv_pow));
    let cfg = FAccKey {
        g: FGridA::of(&lk.g),
        q: lk.q as u32,
        pv: lk.pv as u32,
        ctx: a.shape.c as u32,
        n_insts: a.shape.insts as u32,
        off_v: m31_of(128).0,
        two_k,
        mutate,
    };
    unsafe {
        b65_acc_grid(key.as_ptr(), num.as_ptr(),
                     if aux.is_empty() { std::ptr::null() } else { aux.as_ptr() },
                     val.as_ptr(), sg.as_ptr() as *const u32, rv.as_ptr() as *const u32,
                     q4(pow.sg_step), q4(pow.rv_step), inv.as_ptr(), acc.fc(), acc.ap(),
                     acc.bp(), cfg);
    }
    acc.scan(0, 3, mutate);
    acc.stage2(
        &[
            // `hs`: a Horner in r over the group ends, and nothing else.
            FAccS2 { dst: 3, src0: 0, src1: ONE_SRC, sel_i: 0,
                     m_i: qone(), m_l: q4(pow.r), m_n: qone() },
            // `av`: the sum of `wv zn`, folded by rv^hd per instance.
            FAccS2 { dst: 4, src0: 2, src1: 1, sel_i: SEL_ISTART,
                     m_i: q4(pow.rv_hd), m_l: qone(), m_n: qone() },
            // `sn`: the sum of `zn`, folded by r^C per instance.
            FAccS2 { dst: 5, src0: 1, src1: ONE_SRC, sel_i: SEL_ISTART,
                     m_i: q4(pow.r_c), m_l: qone(), m_n: qone() },
        ],
        mutate,
    );
    acc.scan(3, 3, mutate);
    let claims = acc.last();
    let cols = acc.split(&[0, 3, 1, 2, 4, 5], mutate);
    note(Reg::Grid, "key_tile", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims: vec![claims[3], claims[4], claims[5]],
                 log_rows: lk.g.log_rows }
}

// ======================================================== the per-token block ==

/// `am` and `az`: the row maxima and the softmax denominators.
pub fn tok(base: &[BaseFieldVec], at: &[usize], log_row: u32, rw: usize, a: &Attn,
           pow: &Powers, flags: &Uint32Vec) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Tok);
    let with_max = a.variant != Variant::Bf16x;
    let b0 = at[3];
    let z0 = b0 + if with_max { rw } else { 0 };
    let max = ptrs(&base[b0..b0 + if with_max { rw } else { 1 }]);
    let z = ptrs(&base[z0..z0 + rw]);
    let acc = Acc::borrowed(log_row, 2, flags);
    let sr = pows(&pow.sr_pow);
    let cfg = FAccTok {
        log_rows: log_row,
        rows: 1u32 << log_row,
        rw: rw as u32,
        s_inst: a.shape.s as u32,
        with_max: u32::from(with_max),
        mutate,
    };
    unsafe {
        b65_acc_tok(max.as_ptr(), z.as_ptr(), sr.as_ptr() as *const u32, q4(pow.sr_step),
                    q4(pow.mu), acc.fc(), acc.ap(), acc.bp(), cfg);
    }
    acc.scan(0, 2, mutate);
    let claims = acc.last();
    let cols = acc.split(&[0, 1], mutate);
    note(Reg::Tok, "key_tile", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims: vec![claims[0], claims[1]], log_rows: log_row }
}

// ====================================================== the Freivalds checks ==

/// `v`, then `us[m]` and `accs[m]` per matmul, in the order the host commits
/// them. The activations are the committed side and the weights the
/// preprocessed one, both already on the device from 21's Freivalds fill.
pub fn freivalds(dev: &crate::devwfill::DevFreivalds, lay: &crate::layer::FreivaldsLayout,
                n_mats: usize, rv: SecureField, s_ch: SecureField) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Freivalds);
    assert_eq!(dev.base.len(), lay.pq, "the Freivalds region's committed column count");
    assert_eq!(dev.pre.len(), n_mats * lay.pk + 4, "the Freivalds region's weight columns");
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, lay.pk);
    let (s_pow, s_step) = crate::attn_air::powers(s_ch, lay.pq);
    let base = ptrs(&dev.base);
    let pre = ptrs(&dev.pre[..n_mats * lay.pk]);
    let acc = Acc::new(lay.g.log_rows, 1 + 2 * n_mats);
    let (sp, rp) = (pows(&s_pow), pows(&rv_pow));
    let cfg = FAccFreivalds {
        g: FGridA::of(&lay.g),
        pq: lay.pq as u32,
        pk: lay.pk as u32,
        n_mats: n_mats as u32,
        off_x: m31_of(128).0,
        mutate,
    };
    unsafe {
        b65_acc_freivalds(base.as_ptr(), pre.as_ptr(), sp.as_ptr() as *const u32,
                          rp.as_ptr() as *const u32, q4(s_step), q4(rv_step), acc.fp(),
                          acc.ap(), acc.bp(), cfg);
    }
    acc.scan(0, 1 + n_mats, mutate);
    // `accs[m][j] = accs[m][j-1] + (last ? us[m][j] v[j] : 0)`: a plain sum, no
    // per-instance weight, because a Freivalds group is one instance.
    let s2: Vec<FAccS2> = (0..n_mats)
        .map(|m| FAccS2 { dst: (1 + n_mats + m) as u32, src0: (1 + m) as u32, src1: 0,
                          sel_i: 0, m_i: qone(), m_l: qone(), m_n: qone() })
        .collect();
    acc.stage2(&s2, mutate);
    acc.scan(1 + n_mats, n_mats, mutate);
    let claims = acc.last();
    let mut order = vec![0u32];
    for m in 0..n_mats {
        order.push((1 + m) as u32);
        order.push((1 + n_mats + m) as u32);
    }
    let cols = acc.split(&order, mutate);
    note(Reg::Freivalds, "group", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols {
        cols,
        claims: (0..n_mats).map(|m| claims[1 + n_mats + m]).collect(),
        log_rows: lay.g.log_rows,
    }
}

// =============================================================== the dequant ==

/// `z` and `h`: the matmul region's Freivalds claimed side, over the same
/// committed accumulator columns 20's dequant fill wrote.
pub fn deq(dev: &crate::devfill::DevFill, lay: &crate::layer::MatLayout,
           map: &crate::dequant::Map, f_acc: usize, rv: SecureField,
           s_ch: SecureField) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Deq);
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, lay.per_row);
    let cols: Vec<*const u32> =
        (0..lay.per_row).map(|s| dev.base[map.at(s, f_acc)].device_ptr).collect();
    let acc = Acc::new(lay.log_rows, 2);
    let rp = pows(&rv_pow);
    let cfg = FAccDeq {
        log_rows: lay.log_rows,
        rows: 1u32 << lay.log_rows,
        per_row: lay.per_row as u32,
        chunks: lay.chunks as u32,
        mutate,
    };
    unsafe {
        b65_acc_deq(cols.as_ptr(), rp.as_ptr() as *const u32, q4(rv_step), acc.fp(),
                    acc.ap(), acc.bp(), cfg);
    }
    acc.scan(0, 1, mutate);
    acc.stage2(&[FAccS2 { dst: 1, src0: 0, src1: ONE_SRC, sel_i: 0,
                          m_i: qone(), m_l: q4(s_ch), m_n: qone() }], mutate);
    acc.scan(1, 1, mutate);
    let claims = acc.last();
    let cols = acc.split(&[0, 1], mutate);
    note(Reg::Deq, "region", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims: vec![claims[1]], log_rows: lay.log_rows }
}

// =============================================================== the attnout ==

/// `z`, `h` and `dz`.
///
/// `den` is the region's shared first column and it is not scanned: a lane with
/// `a = 0` at every row comes out of the scan holding itself, so stage 2 reads
/// it exactly like a scanned one and there is no second kind of source.
#[allow(clippy::too_many_arguments)]
pub fn ao(dev: &crate::devwfill::DevAo, lay: &crate::layer::AoLayout, nh: usize,
          s: usize, rv: SecureField, s_ch: SecureField, mu_out: SecureField,
          mu_den: SecureField) -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::Ao);
    let pr = lay.per_row;
    // stream JJ (E11): the base layout is `2 + 9 per_row` -- `den`, `joff`,
    // then `acc, sg, qm, nz2, denp, d, num, pj, alo` per slot -- and this fold
    // reads `den` and each slot's `acc`.  It was `1 + 6 per_row`; the two must
    // move together (`layer::ao_region`, `devwfill::ao_region_dev`,
    // `gkrlink`'s `c0`).
    // stream TT (E11), log row 73: the magnitude form's layout is `2 + 10 pr`,
    // but it never reaches here -- `ao_region` fills it on the host on both
    // backends, so `dev` is `None` and this fold is the host's.
    assert_eq!(dev.base.len(), 2 + 9 * pr, "the attnout region's column count");
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, pr);
    let cols: Vec<*const u32> =
        (0..pr).map(|k| dev.base[2 + 9 * k].device_ptr).collect();
    let acc = Acc::new(lay.log_rows, 4);
    let rp = pows(&rv_pow);
    let cfg = FAccAo {
        log_rows: lay.log_rows,
        rows: 1u32 << lay.log_rows,
        per_row: pr as u32,
        chunks: lay.chunks as u32,
        lead: lay.lead as u32,
        nh: nh as u32,
        s: s as u32,
        mutate,
    };
    unsafe {
        b65_acc_ao(dev.base[0].device_ptr, cols.as_ptr(), rp.as_ptr() as *const u32,
                   q4(rv_step), acc.fp(), acc.ap(), acc.bp(), cfg);
    }
    acc.scan(0, 2, mutate);
    acc.stage2(
        &[
            FAccS2 { dst: 2, src0: 0, src1: ONE_SRC, sel_i: SEL_ISTART,
                     m_i: q4(mu_out), m_l: q4(s_ch), m_n: qone() },
            FAccS2 { dst: 3, src0: 1, src1: ONE_SRC, sel_i: SEL_ISTART,
                     m_i: q4(mu_den), m_l: q4(s_ch), m_n: qone() },
        ],
        mutate,
    );
    acc.scan(2, 2, mutate);
    let claims = acc.last();
    let cols = acc.split(&[0, 2, 3], mutate);
    note(Reg::Ao, "region", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims: vec![claims[2], claims[3]], log_rows: lay.log_rows }
}

// ============================================================== the residual-stream bindings ==

/// `B7_M4_RESIDUAL_STREAM_BIND`: 0 the host, 1 the device, 2 both with every cell asserted.
///
/// B11 stream N. Born off, like every path before it: this switch joins the
/// default table in `util::m4b_default` in the same commit that lands its
/// digest gate, and not before.
pub fn binding_mode() -> u32 {
    crate::util::m4b_mode("B7_M4_RESIDUAL_STREAM_BIND")
}

/// One residual-stream region's Horner bindings, on their way into the tree.
///
/// The seventh region of `b65_acc.cuh`'s table and the plainest of them: a binding
/// has no segmentation at all, so there are no flags and no stage 2 -- one
/// stage-1 lane per binding and then the shared affine scan, which is the same
/// primitive as every other region's and therefore the same field element the
/// host's serial fold produces.
///
/// `in_nat` and `out_nat` are the arrays `b65_residual_stream_fill.cu` already leaves on the
/// card, in coset row order. Before this stream they were copied back across
/// the link for the host to fold, which is the whole of `gen/residual_stream_fill_d2h`.
///
/// The caller owns the order: `specs` is `Binding::specs` as the AIR reads it, and
/// the columns come back four per binding in that order, which is exactly what
/// `attn_trace::secure_cols_pub` produces on the host backend.
pub fn residual_stream_bind(in_nat: *const i32, out_nat: *const i32, log_rows: u32,
                            specs: &[FAccResidualStreamBinding], weights: &[SecureField], lam: SecureField)
            -> DevAccCols {
    let t = std::time::Instant::now();
    let mutate = accmut(Reg::ResidualStreamBinding);
    let acc = Acc::new(log_rows, specs.len());
    let w = pows(weights);
    unsafe {
        b65_acc_residual_stream_bind(in_nat, out_nat, w.as_ptr() as *const u32, w.len() as u32,
                                     q4(lam), specs.as_ptr(), specs.len() as u32, acc.ap(), acc.bp(),
                                     acc.rows as u32, mutate);
    }
    acc.scan(0, specs.len(), mutate);
    let claims = acc.last();
    let order: Vec<u32> = (0..specs.len() as u32).collect();
    let cols = acc.split(&order, mutate);
    note(Reg::ResidualStreamBinding, "residual_stream", cols.len(), acc.rows, t.elapsed().as_secs_f64());
    DevAccCols { cols, claims, log_rows }
}
