//! Trace generation for the attention block, packed over `insts` instances.
//!
//! Every region's reduction axis is `(instance, group)`, so the witness accessors
//! split a global group index into `(instance, local)` and everything else is B5's.
//! The three regions whose outer accumulator is a plain sum multiply by the
//! per-instance weight at an instance-start row; the two that are Horners already
//! carry it.
//!
//! Every column is built in *coset* (natural row) order and permuted once with
//! `bit_reverse_coset_to_circle_domain_order`, because that is the order a
//! `[-1, 0]` mask means -- B3's trap, inherited verbatim.
//!
//! The LogUp columns are the part that has to match the AIR exactly: the
//! constraint framework pairs consecutive `add_to_relation` calls, so the
//! generator builds the same ordered list of lookups and pairs it the same way.
//!
//! **Parallelism.**  B3, B4 and B5 all reported that our own trace and interaction
//! generation was a third to a half of wall time, single-threaded, and all three
//! left it that way; B6 measured it at 58% and fixed it.  The trick is that a
//! Horner accumulator is only *sequentially* dependent through its carry: the
//! per-row inner sums -- `sum_i s^i score_i` over the row's slots, which is where
//! the O(rows x slots) work is -- are independent, so they are computed with rayon
//! and the carry pass over them is O(rows) and stays serial.

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::Relation;

use crate::attn_air::{ExpRel, Powers, RangeRel};
use crate::common::{Eval, eval_of, m31_u32};
use crate::layout::{Cells, LayoutChan, LayoutKey, LayoutOut, Shape};
use crate::attn_witness::{Attn, Variant};

pub fn m31_of(v: i64) -> M31 {
    let p = P as i64;
    M31::from_u32_unchecked((((v % p) + p) % p) as u32)
}

/// Split a QM31 accumulator into its four M31 columns.
///
/// Each of the four is an independent pass with its own bit-reversal, so they go in
/// parallel: 13 accumulators x 4 columns is the second-largest block of our own
/// interaction work after the inner sums.
pub fn secure_cols(log_size: u32, vals: &[SecureField]) -> Vec<Eval> {
    (0..4)
        .into_par_iter()
        .map(|i| eval_of(log_size, vals.iter().map(|v| v.to_m31_array()[i]).collect()))
        .collect()
}

/// One lookup, as the AIR adds it: which relation, and which columns carry it.
#[derive(Clone)]
pub enum Look {
    /// `(index, numerator)` against the exp table
    Exp(usize, usize),
    /// one value against a range table
    Range(usize, usize),
    /// `(index, numerator)` against the exp table, at the multiplicity a
    /// preprocessed column carries: `1 - mask`.  B10 step 3's masked grid slots
    /// are looked up at zero, which is how "this token never attended to that
    /// key" reaches the LogUp argument.  The third index is into `pre`.
    ExpVis(usize, usize, usize),
}

/// Build one component's LogUp columns, pairing consecutive lookups exactly as
/// `finalize_logup_in_pairs` does -- via the parallel builder (B6 section 7's
/// named remainder; bit-identical to `LogupTraceGenerator`).
pub fn logup_cols(
    log_rows: u32,
    looks: &[Look],
    cols: &[Eval],
    pre: &[Eval],
    exp: &ExpRel,
    ranges: &[RangeRel],
) -> (Vec<Eval>, SecureField) {
    let (n_cols, f) = crate::plogup::paired(looks.len(),
                                            |i, vr| look_frac(&looks[i], vr, cols, pre, exp,
                                                              ranges));
    crate::plogup::build(log_rows, n_cols, f)
}

/// One lookup's numerator and denominator at one packed row.
///
/// B11 stream Q: this used to be a closure inside `logup_cols`. It is a free
/// function now because the GKR arm proves the *same* fractions this returns
/// without ever pairing them into columns, and the two arms have to be reading
/// one definition. Nothing about the committed path changes: `logup_cols` calls
/// it through `paired` exactly where the closure used to be called.
pub fn look_frac(l: &Look, vec_row: usize, cols: &[Eval], pre: &[Eval], exp: &ExpRel,
                 ranges: &[RangeRel]) -> (PackedQM31, PackedQM31) {
    let one = PackedQM31::broadcast(SecureField::one());
    match l {
        Look::Exp(a, b) => {
            (one, exp.combine(&[cols[*a].data[vec_row], cols[*b].data[vec_row]]))
        }
        Look::Range(r, a) => (one, ranges[*r].combine(&[cols[*a].data[vec_row]])),
        Look::ExpVis(a, b, m) => {
            let vis = one - PackedQM31::from(pre[*m].data[vec_row]);
            (vis, exp.combine(&[cols[*a].data[vec_row], cols[*b].data[vec_row]]))
        }
    }
}

/// Whether a lookup's numerator is one, which decides whether its GKR instance
/// needs to carry numerators at all.
pub fn look_is_unit(l: &Look) -> bool {
    !matches!(l, Look::ExpVis(..))
}

/// A table's own side: `-multiplicity / (z - combine(row))`.
pub fn table_col<R: Relation<PackedM31, PackedQM31> + Sync>(
    log_rows: u32,
    rel: &R,
    row: &[&Eval],
    mult: &Eval,
) -> (Vec<Eval>, SecureField) {
    crate::plogup::build(log_rows, 1, |_, vec_row| {
        let vals: Vec<_> = row.iter().map(|c| c.data[vec_row]).collect();
        (-PackedQM31::from(mult.data[vec_row]), rel.combine(&vals))
    })
}

pub struct Cfg {
    /// target keys per row in the channel region
    pub chan_target: usize,
    /// target scores per row in the grid region
    pub grid_target: usize,
    /// query tokens per row in the per-token region
    pub rw: usize,
    pub range_check: bool,
    /// low limb width of the denominator range check
    pub z_lo_bits: u32,
    pub z_hi_bits: u32,
    /// lower bound on the denominator (the full-scale numerator an honest shift
    /// always produces)
    pub z_min: u32,
}

pub struct Layouts {
    pub chan: LayoutChan,
    pub key: LayoutKey,
    pub out: LayoutOut,
    pub log_row: u32,
    pub rw: usize,
}

impl Layouts {
    pub fn choose(sh: &Shape, cfg: &Cfg) -> Self {
        // The per-token region must tile exactly: pick the tokens per row so the row
        // count lands on the SIMD floor rather than padding up to it.  With S and
        // `insts` powers of two this always divides.
        let floor = 1usize << crate::common::MIN_LOG_ROWS;
        let rw = cfg.rw.min(sh.toks() / floor).max(1);
        let rows = sh.toks().div_ceil(rw).max(floor);
        assert_eq!(rw * rows, sh.toks(), "per-token region does not tile: rw={rw} rows={rows}");
        Self {
            chan: LayoutChan::choose(sh, cfg.chan_target),
            key: LayoutKey::choose(sh, cfg.grid_target),
            out: LayoutOut::choose(sh, cfg.grid_target.min(sh.s)),
            log_row: (usize::BITS - (rows - 1).leading_zeros()).max(crate::common::MIN_LOG_ROWS),
            rw,
        }
    }
    pub fn max_log(&self) -> u32 {
        self.chan.g.log_rows.max(self.key.g.log_rows).max(self.out.g.log_rows).max(self.log_row)
    }
}

/// Widths of the range tables the block uses, in the order the relations are drawn.
pub fn range_widths(a: &Attn, cfg: &Cfg) -> Vec<u32> {
    // The tables are always present -- they are a few thousand cells and a
    // component with zero multiplicity still cancels -- so `--no-range` measures
    // what the *lookups* cost, not what dropping a table would save.
    let mut w = vec![a.k_bits, a.q_bits, 8];
    if let Variant::IntSplit(k) = a.variant {
        w.push(k); // the remainder the table's granularity leaves behind
    }
    w.push(cfg.z_lo_bits);
    w.push(cfg.z_hi_bits);
    w.sort_unstable();
    w.dedup();
    w
}

pub struct Committed {
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// first base column of each component
    pub at: Vec<usize>,
    /// committed values that no table can absorb
    pub off_table: usize,
}

/// The per-band selector columns (tree 0), without the shared tables: the
/// composed proof commits several bands' selectors alongside one master table
/// set, so tables are split into `pre_tables`.
pub fn commit_pre(a: &Attn, l: &Layouts, with_out: bool, with_rows: bool)
                  -> Vec<Eval> {
    let sh = a.shape;
    let sel = |log_rows: u32, f: &dyn Fn(usize) -> bool| {
        eval_of(log_rows, (0..1usize << log_rows).map(|j| m31_u32(f(j) as u32)).collect())
    };
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let mut pre = vec![];
    let cch = |j: usize| lc.g.row(j).map(|(_, c)| c);
    pre.push(sel(lc.g.log_rows, &|j| matches!(cch(j), Some(c) if c + 1 < lc.g.chunks)));
    pre.push(sel(lc.g.log_rows, &|j| matches!(cch(j), Some(0))));
    pre.push(sel(lc.g.log_rows, &|j| j > 0));
    pre.push(sel(lc.g.log_rows, &|j| j + 1 == lc.g.rows));
    // instance start: the first row of an instance, where `acc` picks up r^C
    pre.push(sel(lc.g.log_rows, &|j| j > 0 && chan_istart(lc, sh.hd, j)));

    let kch = |j: usize| lk.g.row(j).map(|(_, c)| c);
    let klast = |j: usize| matches!(kch(j), Some(0));
    pre.push(sel(lk.g.log_rows, &|j| matches!(kch(j), Some(c) if c + 1 < lk.g.chunks)));
    pre.push(sel(lk.g.log_rows, &|j| klast(j)));
    pre.push(sel(lk.g.log_rows, &|j| j > 0));
    pre.push(sel(lk.g.log_rows, &|j| j + 1 == lk.g.rows));
    pre.push(sel(lk.g.log_rows, &|j| j > 0 && !klast(j)));
    pre.push(sel(lk.g.log_rows, &|j| j > 0 && klast(j)));
    // instance start: where `av` picks up rv^hd and `sn` picks up r^C
    pre.push(sel(lk.g.log_rows, &|j| j > 0 && key_istart(lk, sh.c, j)));
    // the master exp table's key scale, inverted, per instance -- so the committed
    // scaled shift reads back as the shift inside the score accumulator
    pre.push(eval_of(lk.g.log_rows, (0..lk.g.rows)
        .map(|j| match lk.g.row(j) {
            Some((gk, _)) if a.variant != Variant::Bf16x => {
                m31_of(a.delta[split(gk, sh.c).0]).inverse()
            }
            _ => M31::one(),
        })
        .collect()));
    if a.corner {
        // `iend`: the row where the key Horner crosses *between* instances, which
        // is the one that completes an instance's **highest** local key -- keys
        // descend, so that is the first key of the instance and the accumulator
        // holds only later instances when it is reached.  On it the multiplier is
        // `kappa` instead of `r'`, and the corner's per-instance weight becomes the
        // prefix's `r^C` (attn_air's CORNER_SEL).  On the instance's *lowest* key
        // the jump would also multiply that instance's own later keys, which is
        // `rho r'^-S` off per key.
        // `j > 0`, like every other selector here: at row 0 the `[-1, 0]` mask
        // wraps to the *last* row of the circle domain, so a multiplier that fires
        // there folds the end of the trace into its start.
        pre.push(sel(lk.g.log_rows, &|j| {
            j > 0 && matches!(lk.g.row(j), Some((g, c)) if c == 0 && g % sh.c == sh.c - 1)
        }));
        // the causal mask, one column per token slot: key `k` is in token `t`'s
        // future exactly when `k > t`
        for i in 0..lk.q {
            pre.push(sel(lk.g.log_rows, &|j| {
                matches!(lk.g.row(j), Some((g, chunk))
                         if split(g, sh.c).1 > chunk * lk.q + i)
            }));
        }
    }
    // Stream CC (E11): the inert-lead-row selector, last of the grid's block and
    // committed only when the grid has such rows.  Placed after the mask columns
    // so `msk0` -- `CHAN_SEL.len() + CORNER_SEL.len()`, in two other files -- is
    // the index it always was.  On a prefix band there are no mask columns, so it
    // lands at slot 8 of the key group, which is where `devpre`'s `b65p_sel`
    // writes it.
    if crate::attn_air::act_id(0, a.corner, lk.g.lead).is_some() {
        pre.push(sel(lk.g.log_rows, &|j| lk.g.row(j).is_some()));
    }

    if with_out {
        let och = |j: usize| lo.g.row(j).map(|(_, c)| c);
        let oend = |j: usize| matches!(och(j), Some(0));
        pre.push(sel(lo.g.log_rows, &|j| matches!(och(j), Some(c) if c + 1 < lo.g.chunks)));
        pre.push(sel(lo.g.log_rows, &|j| oend(j)));
        pre.push(sel(lo.g.log_rows, &|j| j > 0 && !oend(j)));
        pre.push(sel(lo.g.log_rows, &|j| j > 0 && oend(j)));
        pre.push(sel(lo.g.log_rows, &|j| j + 1 == lo.g.rows));
    }

    // B10 step 3: a corner band has no per-token region.  There is one row maximum
    // and one denominator per *token*, not per band, so the pair shares the prefix
    // band's -- which is also the only column the `Z >= nmax` bound can sit on.
    if with_rows {
        let rrows = 1usize << l.log_row;
        pre.push(sel(l.log_row, &|j| j > 0));
        pre.push(sel(l.log_row, &|j| j + 1 == rrows));
        // instance start: where `am` and `az` swap s^rw for mu
        pre.push(sel(l.log_row, &|j| j > 0 && tok_istart(rrows, l.rw, sh.s, j)));
    }
    pre
}

/// The shared tables (tree 0): the exp table and one range table per width.
pub fn pre_tables(a: &Attn, widths: &[u32]) -> Vec<Eval> {
    let tbl_rows = 1usize << a.log_table;
    let mut pre = vec![];
    pre.push(eval_of(a.log_table, (0..tbl_rows).map(|i| m31_u32(i as u32)).collect()));
    pre.push(eval_of(a.log_table, a.table.iter().map(|v| m31_u32(*v)).collect()));
    for w in widths {
        pre.push(eval_of(*w, (0..1u32 << w).map(m31_u32).collect()));
    }
    pre
}

pub static EXP_OFF_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static EXP_OFF_MIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MAX);
pub static EXP_OFF_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
pub static EXP_KEY_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static EXP_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn bump_slice(m: &mut [u32], v: i64, off: &mut usize) {
    if v >= 0 && (v as usize) < m.len() {
        m[v as usize] += 1;
    } else {
        *off += 1;
    }
}

/// The witness columns (tree 1), bumping lookup multiplicities into vectors the
/// caller owns -- shared across every region of a composed proof.
pub fn commit_base(a: &Attn, l: &Layouts, cfg: &Cfg, widths: &[u32],
                   mult_exp: &mut [u32], mult_r: &mut [Vec<u32>], off_table: &mut usize,
                   with_out: bool, with_rows: bool) -> (Vec<Eval>, Vec<usize>) {
    let sh = a.shape;
    // A padded *token* slot in the grid region would have to commit a numerator
    // that is both a table output and zero, and the table's zero row is not the
    // one a zero shift maps to -- so the layout has to tile the token axis
    // exactly (asserted out loud).
    assert_eq!(l.key.q * l.key.g.chunks, sh.s,
               "grid layout leaves padded token slots: q={} chunks={} S={}",
               l.key.q, l.key.g.chunks, sh.s);
    assert_eq!(l.out.q * l.out.g.chunks, sh.s,
               "output layout leaves padded token slots");
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let ridx = |b: u32| widths.iter().position(|w| *w == b).unwrap();

    let mut base = vec![];
    let mut at = vec![];

    // channel region: keys then queries, offset into the range tables
    at.push(base.len());
    for p in 0..lc.pk {
        let mut col = vec![M31::zero(); lc.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = match lc.g.row(j) {
                Some((g, chunk)) => {
                    let (inst, ch) = split(g, sh.hd);
                    let c = chunk * lc.pk + p;
                    if c < sh.c {
                        a.kk[(inst * sh.c + c) * sh.hd + ch] + a.off_k()
                    } else {
                        a.off_k()
                    }
                }
                None => a.off_k(),
            };
            if cfg.range_check {
                bump_slice(&mut mult_r[ridx(a.k_bits)], v as i64, off_table);
            }
            *slot = m31_of(v as i64);
        }
        base.push(eval_of(lc.g.log_rows, col));
    }
    for p in 0..lc.pq {
        let mut col = vec![M31::zero(); lc.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = match lc.g.row(j) {
                Some((g, chunk)) => {
                    let (inst, ch) = split(g, sh.hd);
                    let t = chunk * lc.pq + p;
                    if t < sh.s {
                        a.qq[(inst * sh.s + t) * sh.hd + ch] + a.off_q()
                    } else {
                        a.off_q()
                    }
                }
                None => a.off_q(),
            };
            if cfg.range_check {
                bump_slice(&mut mult_r[ridx(a.q_bits)], v as i64, off_table);
            }
            *slot = m31_of(v as i64);
        }
        base.push(eval_of(lc.g.log_rows, col));
    }

    // grid region: index, numerator, aux, values
    at.push(base.len());
    let grid_at = |j: usize, i: usize| -> Option<usize> {
        lk.g.row(j).and_then(|(g, chunk)| {
            let (inst, c) = split(g, sh.c);
            let t = chunk * lk.q + i;
            (t < sh.s).then_some((inst * sh.s + t) * sh.c + c)
        })
    };
    for i in 0..lk.q {
        let mut col = vec![M31::zero(); lk.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            // padding takes index 0, a real table row, so multiplicities stay exact.
            // A *masked* slot is not a lookup at all -- the AIR reads it at
            // multiplicity zero -- so it bumps nothing and cannot be off-table.
            let g = grid_at(j, i);
            if g.is_none_or(|g| a.visible(g)) {
                let key = g.map(|g| a.table_key(g)).unwrap_or(0);
                use std::sync::atomic::Ordering::Relaxed;
                EXP_KEY_MAX.fetch_max(key, Relaxed);
                EXP_ROWS.store(mult_exp.len(), Relaxed);
                let before = *off_table;
                bump_slice(mult_exp, key, off_table);
                if *off_table != before {
                    EXP_OFF_N.fetch_add(1, Relaxed);
                    EXP_OFF_MIN.fetch_min(key, Relaxed);
                    EXP_OFF_MAX.fetch_max(key, Relaxed);
                }
            }
            *slot = m31_of(grid_at(j, i).map(|g| a.col_idx(g)).unwrap_or(0));
        }
        base.push(eval_of(lk.g.log_rows, col));
    }
    for i in 0..lk.q {
        let mut col = vec![M31::zero(); lk.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            *slot = m31_of(match grid_at(j, i) {
                Some(g) => a.num[g],
                None => a.table[0] as i64,
            });
        }
        base.push(eval_of(lk.g.log_rows, col));
    }
    let n_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };
    for i in 0..n_aux {
        let mut col = vec![M31::zero(); lk.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = grid_at(j, i).map(|g| a.aux_of(g)).unwrap_or(0);
            if let Variant::IntSplit(k) = a.variant {
                bump_slice(&mut mult_r[ridx(k)], v, off_table);
            }
            *slot = m31_of(v);
        }
        base.push(eval_of(lk.g.log_rows, col));
    }
    for p in 0..lk.pv {
        let mut col = vec![M31::zero(); lk.g.rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = match lk.g.row(j) {
                Some((g, chunk)) => {
                    let (inst, c) = split(g, sh.c);
                    let ch = chunk * lk.pv + p;
                    if ch < sh.hd { a.vv[(inst * sh.c + c) * sh.hd + ch] + 128 } else { 128 }
                }
                None => 128,
            };
            if cfg.range_check {
                bump_slice(&mut mult_r[ridx(8)], v as i64, off_table);
            }
            *slot = m31_of(v as i64);
        }
        base.push(eval_of(lk.g.log_rows, col));
    }

    // A.V outputs (the composed proof moves these into the attnout region)
    at.push(base.len());
    if with_out {
        for i in 0..lo.q {
            let mut col = vec![M31::zero(); lo.g.rows];
            for (j, slot) in col.iter_mut().enumerate() {
                *slot = match lo.g.row(j) {
                    Some((g, chunk)) => {
                        let (inst, ch) = split(g, sh.hd);
                        let t = chunk * lo.q + i;
                        if t < sh.s {
                            m31_of(a.out[(inst * sh.s + t) * sh.hd + ch])
                        } else {
                            M31::zero()
                        }
                    }
                    None => M31::zero(),
                };
            }
            base.push(eval_of(lo.g.log_rows, col));
        }
    }

    // per-token region: maxima, denominators, and the denominator's low limb.
    // The denominator it commits is `zz_row()`: the pair's total where a corner
    // rides with this band, this band's own otherwise.
    at.push(base.len());
    if with_rows {
        let rrows = 1usize << l.log_row;
        let tok = |j: usize, i: usize| tok_of(rrows, l.rw, sh.toks(), j, i);
        let with_max = a.variant != Variant::Bf16x;
        let zz = a.zz_row();
        if with_max {
            for i in 0..l.rw {
                let col = (0..rrows)
                    .map(|j| m31_of(tok(j, i).map(|t| a.mm[t]).unwrap_or(0)))
                    .collect();
                base.push(eval_of(l.log_row, col));
            }
        }
        for i in 0..l.rw {
            let col = (0..rrows).map(|j| m31_of(tok(j, i).map(|t| zz[t]).unwrap_or(0))).collect();
            base.push(eval_of(l.log_row, col));
        }
        let lo_mask = (1i64 << cfg.z_lo_bits) - 1;
        for off in [0i64, cfg.z_min as i64] {
            for i in 0..l.rw {
                let mut col = vec![M31::zero(); rrows];
                for (j, slot) in col.iter_mut().enumerate() {
                    let z = tok(j, i).map(|t| zz[t]).unwrap_or(0) - off;
                    let lo = z & lo_mask;
                    bump_slice(&mut mult_r[ridx(cfg.z_lo_bits)], lo, off_table);
                    bump_slice(&mut mult_r[ridx(cfg.z_hi_bits)], (z - lo) >> cfg.z_lo_bits,
                               off_table);
                    *slot = m31_of(lo);
                }
                base.push(eval_of(l.log_row, col));
            }
        }
    }
    at.push(base.len());
    (base, at)
}

/// B7a milestone 4 (section 3 row 1): `commit_base`, wherever it runs.
///
/// Returns the host columns (the mirror under mode 1, because
/// `interaction_phase` reads `com.base`), the component offsets, and the device
/// columns when there are any.
#[allow(clippy::too_many_arguments)]
pub fn commit_base_any(a: &Attn, l: &Layouts, cfg: &Cfg, widths: &[u32],
                       mult_exp: &mut [u32], mult_r: &mut [Vec<u32>],
                       off_table: &mut usize, with_out: bool, with_rows: bool)
                       -> (Vec<Eval>, Vec<usize>, Option<crate::devwfill::DevBand>) {
    // `with_rows` postdates this milestone: it is `!a.corner`, and it says
    // whether this band commits the per-token region (the row maxima, the
    // denominator and its low limb).  A *corner* band does not -- there is one
    // maximum and one denominator per token, not per band, so the corner rides
    // on the prefix band's -- and `commit_base_dev` has no notion of any of
    // this: `devwfill.rs` mentions neither `with_rows` nor `corner`.  A device
    // fill that always built the per-token region would commit columns the AIR
    // does not expect on a corner band.
    //
    // So corner bands take the host path, whatever the switch says.  That is
    // correct by construction rather than by a guess at the kernel, and it
    // costs the device fill only the S x S corner -- the cells are in the
    // C - S prefix band, which still fills on the device.  Teaching the fill
    // the distinction is its own piece of work; until then this cannot be
    // silently wrong.
    //
    // B11 stream L: the switch read here is the band fill's own
    // (`B7_M4_WFILL_BAND`, defaulting to `B7_M4_WFILL`), because the band fill
    // and the walk fill are not in the same state -- see `devwfill::band_mode`.
    //
    // B11 stream M: and the corner override above is now that function's twin,
    // `devwfill::band_fill_mode`, because `devpre::fills_dev` has to reach the
    // same answer before any fill has run.
    // Stream AA: `with_rows` used to *be* `!a.corner`, because the only band
    // that rode on another's per-token region was the causal corner.  With
    // `--kv-bind` the declared prefix is tiled into sub-bands of `S` keys and
    // the group's one region lives on its first sub-band, so a band with
    // `corner == false` and `with_rows == false` is now a legal and ordinary
    // thing.  The implication that survives is the one that matters here, and
    // it is asserted instead: a corner never carries the region.
    debug_assert!(!(a.corner && with_rows), "a corner band cannot carry the per-token region");
    // ... and the device fill still has no notion of any of this
    // (`devwfill.rs` mentions neither `with_rows` nor `corner`), so a band that
    // does not carry the region takes the host path whatever the switch says --
    // which for a corner band is the answer `band_fill_mode` already gives.
    let m4d = if with_rows { crate::devwfill::band_fill_mode(a) } else { 0 };
    if m4d == 1 {
        // B7a milestone 4 (section 3 row 1): with the tables' columns on the
        // device the exp counter has no host reader left, so it stays there --
        // unless a corner band rides with this one, in which case it does have
        // one. See `devwfill::exp_mult_on_host`; `compose.rs` decides which
        // column is committed from the same predicate, and the two must agree.
        let dev = crate::devwfill::commit_base_dev(a, l, cfg, widths, mult_exp, mult_r,
                                                   off_table, with_out,
                                                   crate::devwfill::exp_mult_on_host(a),
                                                   with_rows);
        let at = dev.at.clone();
        return (vec![], at, Some(dev));
    }
    if m4d != 2 {
        let (base, at) = commit_base(a, l, cfg, widths, mult_exp, mult_r, off_table,
                                     with_out, with_rows);
        return (base, at, None);
    }
    // mode 2: both, with every cell asserted equal before any column is used
    let before_exp = mult_exp.to_vec();
    let before_r: Vec<Vec<u32>> = mult_r.to_vec();
    let before_off = *off_table;
    let (base, at) = commit_base(a, l, cfg, widths, mult_exp, mult_r, off_table, with_out,
                                 with_rows);
    let after_exp = mult_exp.to_vec();
    let after_r: Vec<Vec<u32>> = mult_r.to_vec();
    let after_off = *off_table;
    let dev = crate::devwfill::commit_base_dev(a, l, cfg, widths, mult_exp, mult_r,
                                               off_table, with_out, true, with_rows);
    let label = format!("band {} 2^{}", a.shape.label(), l.key.g.log_rows);
    // B11 stream L: the region offsets, printed before the comparison rather
    // than after it.  `check_cols` names a column by its index in the whole
    // base vector, and which *region* that index falls in is the first thing a
    // mismatch has to be attributed to -- but the `at` assert below only runs
    // if the columns already matched, so on a real mismatch the offsets were
    // never printed.  They are four numbers; printing them costs nothing and
    // turns "column 304" into "the grid region's column 40".
    eprintln!(
        "wfill-dev: {label} -- {} base columns, regions at {:?} \
(chan {}+{}, grid 2*{}+{}+{}, out {}, tok {})",
        dev.base.len(), dev.at,
        l.chan.pk, l.chan.pq, l.key.q,
        if a.variant == Variant::IntExp { 0 } else { l.key.q }, l.key.pv,
        if with_out { l.out.q } else { 0 },
        (if a.variant != Variant::Bf16x { l.rw } else { 0 }) + 3 * l.rw
    );
    crate::devwfill::check_cols(&format!("{label} base"), &base, &dev.base);
    assert_eq!(at, dev.at, "{label}: the component offsets differ");
    let delta = |x: &[u32], y: &[u32]| -> Vec<u32> {
        x.iter().zip(y).map(|(p, q)| p - q).collect()
    };
    crate::devwfill::check_delta(&format!("{label} m_exp"), &before_exp, &after_exp,
                                 &delta(mult_exp, &after_exp));
    for (t, w) in widths.iter().enumerate() {
        crate::devwfill::check_delta(&format!("{label} m_r[{w}]"), &before_r[t], &after_r[t],
                                     &delta(&mult_r[t], &after_r[t]));
    }
    assert_eq!(*off_table - after_off, after_off - before_off,
               "{label}: off-table bumps differ");
    // the device's own bumps are the check's, not the proof's
    mult_exp.copy_from_slice(&after_exp);
    for (t, v) in mult_r.iter_mut().enumerate() {
        v.copy_from_slice(&after_r[t]);
    }
    *off_table = after_off;
    eprintln!("wfill-dev: {label} -- {} columns, {} cells, {} multiplicity tables \
and the component offsets all match",
              base.len(), dev.base.iter().map(|c| c.size).sum::<usize>(), widths.len() + 1);
    (base, at, None)
}

/// One multiplicity column per table, in table order.
pub fn mult_cols(a: &Attn, widths: &[u32], mult_exp: &[u32], mult_r: &[Vec<u32>]) -> Vec<Eval> {
    let mut cols = vec![eval_of(a.log_table, mult_exp.iter().map(|m| m31_u32(*m)).collect())];
    for (t, w) in widths.iter().enumerate() {
        cols.push(eval_of(*w, mult_r[t].iter().map(|m| m31_u32(*m)).collect()));
    }
    cols
}

/// Everything committed before the challenges are drawn (the standalone path).
pub fn commit_phase(a: &Attn, l: &Layouts, cfg: &Cfg) -> Committed {
    let widths = range_widths(a, cfg);
    let mut mult_exp = vec![0u32; 1usize << a.log_table];
    let mut mult_r: Vec<Vec<u32>> = widths.iter().map(|w| vec![0u32; 1usize << w]).collect();
    let mut off_table = 0usize;
    let mut pre = commit_pre(a, l, true, true);
    pre.extend(pre_tables(a, &widths));
    let (mut base, at) = commit_base(a, l, cfg, &widths, &mut mult_exp, &mut mult_r,
                                     &mut off_table, true, true);
    base.extend(mult_cols(a, &widths, &mult_exp, &mult_r));
    Committed { pre, base, at, off_table }
}

pub struct Interaction {
    /// B7a milestone 4 (section 3 row 1): the accumulator columns are always
    /// the host's; a LogUp column is wherever its region's columns were.
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed_chan: SecureField,
    pub claimed_hs: SecureField,
    pub claimed_av: SecureField,
    pub claimed_sn: SecureField,
    pub claimed_out: SecureField,
    pub claimed_am: SecureField,
    pub claimed_az: SecureField,
    pub use_sums: Vec<SecureField>,
    pub table_sums: Vec<SecureField>,
}

/// Everything committed after the challenges: the Horner accumulators and the
/// LogUp columns, in component order.  `widths` orders `ranges`; the composed
/// proof passes its global width list, the standalone path `range_widths`.
#[allow(clippy::too_many_arguments)]
pub fn interaction_phase(a: &Attn, l: &Layouts, cfg: &Cfg, pow: &Powers, exp: &ExpRel,
                         ranges: &[RangeRel], com: &Committed, widths: &[u32],
                         with_out: bool, with_rows: bool,
                         air: Option<&crate::devfrac::BandAir<'_>>,
                         mut gkr: Option<&mut crate::gkrband::Collect>)
                         -> Interaction {
    let sh = a.shape;
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let ridx = |b: u32| widths.iter().position(|w| *w == b).unwrap();
    let mut cols = crate::device::ColSink::new();
    let mut use_sums = vec![];
    // B11 stream S, lever 1: when the collector owns the sums, the committed
    // LogUp running-sum columns are not built at all and nothing is pushed
    // onto `use_sums` -- the batched GKR proves the same sums and the verifier
    // feeds them to `sums_cancel` where the claimed sums stood.
    let gkr_owns = gkr.as_deref().is_some_and(|g| g.replace);

    // B7a milestone 4 (section 3 row 1): the accumulators, wherever they are.
    // The kernels read this region's *own committed columns*, so the device
    // route exists exactly when the fill left them there -- and `B7_M4_ACC=2`
    // runs both, which is a check on the whole witness -> fill -> column ->
    // accumulator chain, because the host side reads the witness instead.
    let m4g = air.and_then(|x| x.dev).filter(|_| crate::devacc::mode() >= 1);
    let m4g_host = m4g.is_none() || crate::devacc::mode() == 2;
    // B7a milestone 4 (section 3 row 1): the per-row layout flags, written once
    // at fill time by `b65_flags_band` and read here. Before this row each of
    // the three kernels below recomputed them; now the AIR's selector columns
    // and this segmentation are the same array.
    let m4g_flags = air.and_then(|x| x.pre);
    assert!(m4g.is_none() || m4g_flags.is_some(),
            "the device accumulators need this band's flags: `devpre::band_dev` did not run");
    assert!(m4g.is_none() || !with_out,
            "the device accumulators have no A.V output region: \
             the composed proof moves it into the attnout region");
    let n_chan = if m4g_host { lc.g.rows } else { 0 };

    // ---- channel region ----------------------------------------------------
    let (mut u, mut v, mut acc) = (vec![SecureField::zero(); n_chan],
                                   vec![SecureField::zero(); n_chan],
                                   vec![SecureField::zero(); n_chan]);
    // the inner sums, in parallel: everything but the carry
    let chan_inner: Vec<(SecureField, SecureField, bool, bool, bool)> = (0..n_chan)
        .into_par_iter()
        .map(|j| {
        let (mut kf, mut qf) = (SecureField::zero(), SecureField::zero());
        let (mut nofirst, mut last) = (false, false);
        let mut istart = false;
        if let Some((g, chunk)) = lc.g.row(j) {
            let (inst, ch) = split(g, sh.hd);
            nofirst = chunk + 1 < lc.g.chunks;
            last = chunk == 0;
            istart = j > 0 && chan_istart(lc, sh.hd, j);
            for p in 0..lc.pk {
                let c = chunk * lc.pk + p;
                let val = if c < sh.c { a.kk[(inst * sh.c + c) * sh.hd + ch] } else { 0 };
                if val != 0 {
                    kf += pow.r_pow[p] * SecureField::from(m31_of(val as i64));
                }
            }
            for p in 0..lc.pq {
                let t = chunk * lc.pq + p;
                let val = if t < sh.s { a.qq[(inst * sh.s + t) * sh.hd + ch] } else { 0 };
                if val != 0 {
                    qf += pow.sq_pow[p] * SecureField::from(m31_of(val as i64));
                }
            }
        }
            (kf, qf, nofirst, last, istart)
        })
        .collect();
    for j in 0..n_chan {
        let (kf, qf, nofirst, last, istart) = chan_inner[j];
        // An inert row commits the offset in every slot, so its contribution is
        // exactly zero once the constraint subtracts `sum_p r^p * offset`.
        u[j] = if nofirst { u[j - 1] * pow.r_step } else { SecureField::zero() } + kf;
        v[j] = if nofirst { v[j - 1] * pow.sq_step } else { SecureField::zero() } + qf;
        acc[j] = if j > 0 {
            acc[j - 1] * if istart { pow.r_c } else { SecureField::one() }
        } else {
            SecureField::zero()
        };
        if last {
            acc[j] += u[j] * v[j];
        }
    }
    let claimed_chan = match m4g {
        Some(d) => {
            let got = crate::devacc::chan(&d.base, &d.at, lc, a, pow,
                                           m4g_flags.unwrap().chan());
            let h = m4g_host.then(|| {
                let mut e = secure_cols(lc.g.log_rows, &u);
                e.extend(secure_cols(lc.g.log_rows, &v));
                e.extend(secure_cols(lc.g.log_rows, &acc));
                (e, vec![acc[lc.g.rows - 1]])
            });
            crate::devacc::place(&format!("chan band 2^{}", lc.g.log_rows), got, h,
                                 &mut cols)[0]
        }
        None => {
            cols.extend(secure_cols(lc.g.log_rows, &u));
            cols.extend(secure_cols(lc.g.log_rows, &v));
            cols.extend(secure_cols(lc.g.log_rows, &acc));
            acc[lc.g.rows - 1]
        }
    };
    if cfg.range_check {
        let base0 = com.at[0];
        let mut looks = vec![];
        for p in 0..lc.pk {
            looks.push(Look::Range(ridx(a.k_bits), base0 + p));
        }
        for p in 0..lc.pq {
            looks.push(Look::Range(ridx(a.q_bits), base0 + lc.pk + p));
        }
        if let Some(g) = gkr.as_deref_mut() {
            // B11 stream T: where this band's fill left its columns on the
            // device there is no host column to read, so the feed computes the
            // same fractions there instead. A host-filled band -- a corner --
            // takes the host builder, unchanged.
            crate::gkrband::feed_site(g, air, 0, lc.g.log_rows, &looks, &com.at,
                                      &com.base, &com.pre, exp, ranges);
        }
        if !gkr_owns {
            let (c, s) = crate::devfrac::band_cols(
                air, 0, lc.g.log_rows, &com.at, &com.base,
                || logup_cols(lc.g.log_rows, &looks, &com.base, &com.pre, exp, ranges));
            cols.extend_any(c);
            use_sums.push(s);
        }
    }

    // ---- grid region -------------------------------------------------------
    // Stream CC (E11): the host arm's aimed corruption for the inert-lead-row
    // guard, the twin of `b65_acc.cu`'s `nomask:grid` on the arm that has no
    // card.  See the inert-row branch of `grid_inner` below.  Off unless
    // `B12_GRIDMUT=padnf`.
    //
    // This replaces `B65_PAD_NF=1`, which was the same branch pointing the other
    // way: before the AIR carried `act` it made a padded shape *accept*, and it
    // is recorded that way in `briefs/cc-maskgrid.md` and in
    // `notes/design-notes.md`.  It now makes a padded shape *reject*, so it does
    // not keep the old name.
    let pad_nf = match std::env::var("B12_GRIDMUT").as_deref() {
        Ok("padnf") => true,
        Ok("") | Err(_) => false,
        Ok(other) => panic!("unknown B12_GRIDMUT {other:?}"),
    };
    let n = if m4g_host { lk.g.rows } else { 0 };
    let (mut zs, mut hs) = (vec![SecureField::zero(); n], vec![SecureField::zero(); n]);
    let (mut zn, mut wv) = (vec![SecureField::zero(); n], vec![SecureField::zero(); n]);
    let (mut av, mut sn) = (vec![SecureField::zero(); n], vec![SecureField::zero(); n]);
    let grid_inner: Vec<(SecureField, SecureField, SecureField, bool, bool, bool, bool)> = (0..n)
        .into_par_iter()
        .map(|j| {
        let (mut sf, mut nf, mut vf) =
            (SecureField::zero(), SecureField::zero(), SecureField::zero());
        let (mut nofirst, mut last) = (false, false);
        let (mut istart, mut iend) = (false, false);
        if let Some((gk, chunk)) = lk.g.row(j) {
            let (inst, c) = split(gk, sh.c);
            nofirst = chunk + 1 < lk.g.chunks;
            last = chunk == 0;
            istart = j > 0 && key_istart(lk, sh.c, j);
            iend = a.corner && j > 0 && last && c + 1 == sh.c;
            for i in 0..lk.q {
                let t = chunk * lk.q + i;
                if t < sh.s {
                    let g = (inst * sh.s + t) * sh.c + c;
                    // the score fold keeps every slot; the numerator fold drops the
                    // masked ones, exactly as the AIR's `1 - msk` factor does.  An
                    // honest corner witness has `num = 0` there anyway -- the mask
                    // is written out here so the two agree by construction and not
                    // by coincidence (B10 rule 21).
                    sf += pow.sg_pow[i] * SecureField::from(m31_of(a.score_col(g)));
                    if a.num[g] != 0 && a.visible(g) {
                        nf += pow.sg_pow[i] * SecureField::from(m31_of(a.num[g]));
                    }
                }
            }
            for p in 0..lk.pv {
                let ch = chunk * lk.pv + p;
                let val = if ch < sh.hd { a.vv[(inst * sh.c + c) * sh.hd + ch] } else { 0 };
                if val != 0 {
                    vf += pow.rv_pow[p] * SecureField::from(m31_of(val as i64));
                }
            }
        } else if pad_nf {
            // Stream CC (E11), the CORRUPTION: an INERT lead row's numerator pad
            // enters the sum.  `commit_base` pads the numerator column with
            // `a.table[0]` -- index 0 is a real table row, which is what keeps
            // the exp lookup's multiplicities exact -- and the honest builder
            // contributes no term at all here, which is what `KeyEval`'s `act`
            // factor now says.  Adding the term back is the one way this guard
            // could be missing, and the proof must be rejected for it.
            //
            // It is genuinely aimed: it is confined to `lead` rows out of
            // `g.rows`, it moves no committed cell, and it changes `zn` only on
            // rows that no accumulator reads -- `last` is zero on an inert row,
            // so this `zn` reaches neither `av` nor `sn`, and `nofirst` is zero
            // on the first real row of every group, so it does not carry
            // forward.  The *only* thing left for it to break is the constraint
            // the guard adds.  On a grid that tiles exactly there is no such row
            // and the mutation is vacuous, which is a finding and not a pass
            // (B7a section 5 rule 5).
            let pad = SecureField::from(m31_of(a.table[0] as i64));
            for i in 0..lk.q {
                nf += pow.sg_pow[i] * pad;
            }
        }
            (sf, nf, vf, nofirst, last, istart, iend)
        })
        .collect();
    for j in 0..n {
        let (sf, nf, vf, nofirst, last, istart, iend) = grid_inner[j];
        zs[j] = if nofirst { zs[j - 1] * pow.sg_step } else { SecureField::zero() } + sf;
        zn[j] = if nofirst { zn[j - 1] * pow.sg_step } else { SecureField::zero() } + nf;
        wv[j] = if nofirst { wv[j - 1] * pow.rv_step } else { SecureField::zero() } + vf;
        av[j] = if j > 0 {
            av[j - 1] * if istart { pow.rv_hd } else { SecureField::one() }
        } else {
            SecureField::zero()
        };
        sn[j] = if j > 0 {
            // the denominator's per-instance weight is the *prefix's* `r^C` on both
            // grids, because `sn_prefix + sn_corner` is what `az` is read against
            sn[j - 1] * if istart { pow.sn_c } else { SecureField::one() }
        } else {
            SecureField::zero()
        };
        if last {
            av[j] += wv[j] * zn[j];
            sn[j] += zn[j];
        }
        hs[j] = if j == 0 {
            SecureField::zero()
        } else if last {
            // `kappa` on the row that closes an instance's key block, `r` otherwise;
            // on the prefix grid the two are equal and this is B6's Horner unchanged
            hs[j - 1] * if iend { pow.kappa } else { pow.r }
        } else {
            hs[j - 1]
        };
        if last {
            hs[j] += zs[j];
        }
    }
    let m4g_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };
    let (claimed_hs, claimed_av, claimed_sn) = match m4g {
        Some(d) => {
            let got = crate::devacc::grid(&d.base, &d.at, lk, a, m4g_aux, pow,
                                           m4g_flags.unwrap().key());
            let h = m4g_host.then(|| {
                let mut e = vec![];
                for c in [&zs, &hs, &zn, &wv, &av, &sn] {
                    e.extend(secure_cols(lk.g.log_rows, c));
                }
                (e, vec![hs[n - 1], av[n - 1], sn[n - 1]])
            });
            let c = crate::devacc::place(&format!("grid band 2^{}", lk.g.log_rows), got,
                                         h, &mut cols);
            (c[0], c[1], c[2])
        }
        None => {
            for c in [&zs, &hs, &zn, &wv, &av, &sn] {
                cols.extend(secure_cols(lk.g.log_rows, c));
            }
            (hs[n - 1], av[n - 1], sn[n - 1])
        }
    };
    {
        let b = com.at[1];
        let n_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };
        let mut looks = vec![];
        // where the corner's mask columns sit in `pre` (commit_pre's order)
        let msk0 = crate::attn_air::CHAN_SEL.len() + crate::attn_air::CORNER_SEL.len();
        for i in 0..lk.q {
            let key = if a.variant == Variant::Bf16x { b + 2 * lk.q + i } else { b + i };
            looks.push(if a.corner {
                Look::ExpVis(key, b + lk.q + i, msk0 + i)
            } else {
                Look::Exp(key, b + lk.q + i)
            });
        }
        if let Variant::IntSplit(k) = a.variant {
            for i in 0..lk.q {
                looks.push(Look::Range(ridx(k), b + 2 * lk.q + i));
            }
        }
        if cfg.range_check {
            for p in 0..lk.pv {
                looks.push(Look::Range(ridx(8), b + 2 * lk.q + n_aux + p));
            }
        }
        if let Some(g) = gkr.as_deref_mut() {
            // B11 stream T: where this band's fill left its columns on the
            // device there is no host column to read, so the feed computes the
            // same fractions there instead. A host-filled band -- a corner --
            // takes the host builder, unchanged.
            crate::gkrband::feed_site(g, air, 1, lk.g.log_rows, &looks, &com.at,
                                      &com.base, &com.pre, exp, ranges);
        }
        if !gkr_owns {
            let (c, s) = crate::devfrac::band_cols(
                air, 1, lk.g.log_rows, &com.at, &com.base,
                || logup_cols(lk.g.log_rows, &looks, &com.base, &com.pre, exp, ranges));
            cols.extend_any(c);
            use_sums.push(s);
        }
    }

    // ---- A.V outputs (absent when the attnout region carries the claim) -----
    let (mut z, mut h) = (vec![SecureField::zero(); if with_out { lo.g.rows } else { 0 }],
                          vec![SecureField::zero(); if with_out { lo.g.rows } else { 0 }]);
    let out_inner: Vec<(SecureField, bool, bool)> = (0..if with_out { lo.g.rows } else { 0 })
        .into_par_iter()
        .map(|j| {
        let mut yf = SecureField::zero();
        let (mut nofirst, mut cend) = (false, false);
        if let Some((g, chunk)) = lo.g.row(j) {
            let (inst, ch) = split(g, sh.hd);
            nofirst = chunk + 1 < lo.g.chunks;
            cend = chunk == 0;
            for i in 0..lo.q {
                let t = chunk * lo.q + i;
                let idx = (inst * sh.s + t) * sh.hd + ch;
                if t < sh.s && a.out[idx] != 0 {
                    yf += pow.so_pow[i] * SecureField::from(m31_of(a.out[idx]));
                }
            }
        }
            (yf, nofirst, cend)
        })
        .collect();
    for j in 0..out_inner.len() {
        let (yf, nofirst, cend) = out_inner[j];
        z[j] = if nofirst { z[j - 1] * pow.so_step } else { SecureField::zero() } + yf;
        h[j] = if j == 0 {
            SecureField::zero()
        } else if cend {
            h[j - 1] * pow.rv
        } else {
            h[j - 1]
        };
        if cend {
            h[j] += z[j];
        }
    }
    let claimed_out = if with_out { h[lo.g.rows - 1] } else { SecureField::zero() };
    if with_out {
        cols.extend(secure_cols(lo.g.log_rows, &z));
        cols.extend(secure_cols(lo.g.log_rows, &h));
    }

    // ---- per-token region --------------------------------------------------
    // Absent on a corner band: the pair has one per-token region, the prefix
    // band's, and it carries the row maximum both halves shift by and the total
    // denominator both halves sum to.
    let rrows = if with_rows { 1usize << l.log_row } else { 0 };
    let n_tok = if m4g_host { rrows } else { 0 };
    let (mut am, mut az) = (vec![SecureField::zero(); n_tok], vec![SecureField::zero(); n_tok]);
    let with_max = a.variant != Variant::Bf16x;
    let zz = a.zz_row();
    for j in 0..n_tok {
        let (mut mf, mut zf) = (SecureField::zero(), SecureField::zero());
        for i in 0..l.rw {
            if let Some(t) = tok_of(rrows, l.rw, sh.toks(), j, i) {
                if with_max {
                    mf += pow.sr_pow[i] * SecureField::from(m31_of(a.mm[t]));
                }
                zf += pow.sr_pow[i] * SecureField::from(m31_of(zz[t]));
            }
        }
        // s^rw inside an instance, mu at an instance start: what turns the token
        // region's natural per-instance weight s^S into the r^C the rest carries
        let step = if j > 0 && tok_istart(rrows, l.rw, sh.s, j) { pow.mu } else { pow.sr_step };
        am[j] = if j > 0 { am[j - 1] * step } else { SecureField::zero() } + mf;
        az[j] = if j > 0 { az[j - 1] * step } else { SecureField::zero() } + zf;
    }
    let (claimed_am, claimed_az) = if !with_rows {
        // A corner band has no per-token region, so there is nothing to build
        // on either side of the device link.
        (SecureField::zero(), SecureField::zero())
    } else {
        match m4g {
            Some(d) => {
                let got = crate::devacc::tok(&d.base, &d.at, l.log_row, l.rw, a, pow,
                                          m4g_flags.unwrap().row());
                let h = m4g_host.then(|| {
                    let mut e = secure_cols(l.log_row, &am);
                    e.extend(secure_cols(l.log_row, &az));
                    (e, vec![am[rrows - 1], az[rrows - 1]])
                });
                let c = crate::devacc::place(&format!("tok band 2^{}", l.log_row), got, h,
                                             &mut cols);
                (c[0], c[1])
            }
            None => {
                cols.extend(secure_cols(l.log_row, &am));
                cols.extend(secure_cols(l.log_row, &az));
                (am[rrows - 1], az[rrows - 1])
            }
        }
    };
    if with_rows {
    {
        // The high limb is derived by constraint, so it is not a committed column;
        // the generator rebuilds it from the witness to feed the lookup.
        let b = com.at[3];
        let z_at = b + if with_max { l.rw } else { 0 };
        let lo_at = z_at + l.rw;
        let lo_mask = (1i64 << cfg.z_lo_bits) - 1;
        let mut all: Vec<Eval> = com.base.clone();
        let hi_base = all.len();
        for off in [0i64, cfg.z_min as i64] {
            for i in 0..l.rw {
                let col = (0..rrows)
                    .map(|j| {
                        let z = tok_of(rrows, l.rw, sh.toks(), j, i).map(|t| zz[t]).unwrap_or(0)
                            - off;
                        m31_of((z - (z & lo_mask)) >> cfg.z_lo_bits)
                    })
                    .collect();
                all.push(eval_of(l.log_row, col));
            }
        }
        let mut looks = vec![];
        for i in 0..l.rw {
            looks.push(Look::Range(ridx(cfg.z_lo_bits), lo_at + i));
            looks.push(Look::Range(ridx(cfg.z_hi_bits), hi_base + i));
            looks.push(Look::Range(ridx(cfg.z_lo_bits), lo_at + l.rw + i));
            looks.push(Look::Range(ridx(cfg.z_hi_bits), hi_base + l.rw + i));
        }
        if let Some(g) = gkr.as_deref_mut() {
            // B11 stream T: where this band's fill left its columns on the
            // device there is no host column to read, so the feed computes the
            // same fractions there instead. A host-filled band -- a corner --
            // takes the host builder, unchanged.
            crate::gkrband::feed_site(g, air, 3, l.log_row, &looks, &com.at,
                                      &all, &com.pre, exp, ranges);
        }
        if !gkr_owns {
            let (c, s) = crate::devfrac::band_cols(
                air, 3, l.log_row, &com.at, &com.base,
                || logup_cols(l.log_row, &looks, &all, &com.pre, exp, ranges));
            cols.extend_any(c);
            use_sums.push(s);
        }
    }
    }

    Interaction { cols: cols.into_inner(), claimed_chan, claimed_hs, claimed_av,
                  claimed_sn, claimed_out, claimed_am, claimed_az, use_sums,
                  table_sums: vec![] }
}

/// The tables' own LogUp sides, one column per table, in table order.
pub fn table_sides(a: &Attn, widths: &[u32], exp: &ExpRel, ranges: &[RangeRel],
                   pre_exp: [&Eval; 2], pre_ranges: &[&Eval], mults: &[&Eval])
                   -> (Vec<Eval>, Vec<SecureField>) {
    let mut cols = vec![];
    let mut table_sums = vec![];
    let (c, s) = table_col(a.log_table, exp, &[pre_exp[0], pre_exp[1]], mults[0]);
    cols.extend(c);
    table_sums.push(s);
    for (t, w) in widths.iter().enumerate() {
        let _ = w;
        let (c, s) = table_col(widths[t], &ranges[t], &[pre_ranges[t]], mults[1 + t]);
        cols.extend(c);
        table_sums.push(s);
    }
    (cols, table_sums)
}

/// Index of the exp table's first preprocessed column.
pub fn pre_exp_x(a: &Attn, l: &Layouts) -> usize {
    crate::attn_air::CHAN_SEL.len()
        + if a.corner {
            crate::attn_air::CORNER_SEL.len() + l.key.q
        } else {
            crate::attn_air::KEY_SEL.len()
        }
        // stream CC (E11): the inert-lead-row selector, when the grid has one
        + usize::from(crate::attn_air::act_id(0, a.corner, l.key.g.lead).is_some())
        + crate::attn_air::OUT_SEL.len()
        + crate::attn_air::ROW_SEL.len()
}

/// Is row `j` the first row of an instance in the channel region?
///
/// Groups run descending, so an instance's first *processed* channel is its highest
/// one, and a group's first row is its highest chunk.
pub fn chan_istart(lc: &LayoutChan, hd: usize, j: usize) -> bool {
    matches!(lc.g.row(j), Some((g, c)) if c + 1 == lc.g.chunks && g % hd == hd - 1)
}

/// Is row `j` the first row of an instance in the grid?  Keys run descending.
pub fn key_istart(lk: &LayoutKey, ctx: usize, j: usize) -> bool {
    matches!(lk.g.row(j), Some((g, c)) if c + 1 == lk.g.chunks && g % ctx == ctx - 1)
}

/// Is row `j` the first row of an instance in the per-token region?
///
/// The region carries no group structure: the global token index descends with the
/// row, so an instance starts wherever the row's highest token is `S - 1` mod `S`.
pub fn tok_istart(rows: usize, rw: usize, s: usize, j: usize) -> bool {
    ((rows - 1 - j) * rw + rw) % s == 0
}

/// `(instance, local)` of a packed group index.
pub fn split(g: usize, n: usize) -> (usize, usize) {
    (g / n, g % n)
}

/// Query token held by slot `i` of row `j` in the per-token region, which lays the
/// groups out in descending order so a Horner assigns `s^t`.
pub fn tok_of(rows: usize, rw: usize, s: usize, j: usize, i: usize) -> Option<usize> {
    let t = (rows - 1 - j) * rw + i;
    (t < s).then_some(t)
}

/// Cell accounting from the layouts alone (asserted against the built traces).
pub fn cells(a: &Attn, l: &Layouts, cfg: &Cfg) -> Cells {
    let (lc, lk, lo) = (&l.chan, &l.key, &l.out);
    let widths = range_widths(a, cfg);
    let rrows = 1usize << l.log_row;
    let tbl = 1usize << a.log_table;
    let with_max = a.variant != Variant::Bf16x;
    let n_aux = if a.variant == Variant::IntExp { 0 } else { lk.q };
    let n_split = if matches!(a.variant, Variant::IntSplit(_)) { lk.q } else { 0 };
    let range_tbl: usize = widths.iter().map(|w| 1usize << w).sum();

    let key_sel = if a.corner {
        crate::attn_air::CORNER_SEL.len() + lk.q       // + one mask column per slot
    } else {
        crate::attn_air::KEY_SEL.len()
    // stream CC (E11): + the inert-lead-row selector, when the grid has one
    } + usize::from(crate::attn_air::act_id(0, a.corner, lk.g.lead).is_some());
    let pre = crate::attn_air::CHAN_SEL.len() * lc.g.rows
        + key_sel * lk.g.rows
        + crate::attn_air::OUT_SEL.len() * lo.g.rows
        + crate::attn_air::ROW_SEL.len() * rrows
        + 2 * tbl + range_tbl;
    let base = (lc.pk + lc.pq) * lc.g.rows
        + (2 * lk.q + n_aux + lk.pv) * lk.g.rows
        + lo.q * lo.g.rows
        + ((if with_max { 1 } else { 0 }) + 3) * l.rw * rrows
        + tbl + range_tbl;
    // LogUp columns: one QM31 column per *pair* of consecutive lookups, in the
    // order the AIR adds them, so the pairing is what the eval does
    let n_chan_look = if cfg.range_check { lc.pk + lc.pq } else { 0 };
    let n_grid_look = lk.q + n_split + if cfg.range_check { lk.pv } else { 0 };
    let chan_logup = 4 * n_chan_look.div_ceil(2) * lc.g.rows;
    let grid_logup = 4 * n_grid_look.div_ceil(2) * lk.g.rows;
    let row_logup = 8 * l.rw * rrows;
    let inter = (12 * lc.g.rows + chan_logup)
        + (24 * lk.g.rows + grid_logup)
        + 8 * lo.g.rows
        + (8 * rrows + row_logup)
        + 4 * tbl + 4 * range_tbl;
    // the exp lookups' share of the grid's LogUp columns, and the rest
    let exp_logup = 4 * (lk.q + n_split).div_ceil(2) * lk.g.rows;
    Cells {
        preprocessed: pre,
        base,
        interaction: inter,
        // everything that scales with the S x C grid
        grid: (2 * lk.q + n_aux) * lk.g.rows + exp_logup,
        kv: lc.pk * lc.g.rows + lk.pv * lk.g.rows,
        exp: exp_logup + 5 * tbl,
        range: chan_logup + grid_logup.saturating_sub(exp_logup) + row_logup + 5 * range_tbl,
    }
}

/// Public alias for the region builders outside this module.
pub fn secure_cols_pub(log_size: u32, vals: &[SecureField]) -> Vec<Eval> {
    secure_cols(log_size, vals)
}
