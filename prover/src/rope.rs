//! Stream NN: what stands between the q/k/v projections and the attention.
//!
//! `b1/model.py` does two things after the projections that no region proved:
//!
//!   q = rotate(q, cos, sin)   k = rotate(k, cos, sin)      # RoPE, bf16
//!   qq = quantize(q, s_q)     qk = quantize(k, s_k)        # round-half-away
//!   qv = quantize(v, s_v)                                  # v is not rotated
//!
//! so the bands' committed Q, K and V are *not* the q/k/v dequant outputs and
//! the seam between the layer's matmuls and its attention was a hole
//! (`notes/witness-slack.md` section 1e, `briefs/jj-layerwire-out.md` section 3).
//!
//! The rotation is bf16 arithmetic and is proved by two glue `Prog`s per side
//! (`Prog::rope_mul`, `Prog::rope_add`), with `cos` and the sign-folded `sin` as
//! `Src::Pre` operands -- preprocessed, and therefore a **function of the block
//! index**, because `rope(pos)` reads the token's absolute position
//! `block * S + t`.  This module owns the other half: the quantize, and the
//! consumer-side weights that read the bands' own columns.
//!
//! `QuantEval` is `requant.rs`'s power-of-two requantize with the scale moved
//! from a committed per-token column to a **preprocessed per-slot** one, which
//! is the shape B1's offline `s_k` / `s_v` actually have: per (head, channel),
//! constant down the tokens.  The sign table and the 15-bit range are
//! `requant.rs`'s own, so the identity proved here is the identity stream II
//! already measured on a different scale geometry; the division table and the
//! `(q, nz)` pair table are **this family's**, one per side and sized by that
//! side's `qk_bits` (stream NN-3 -- see "the region's tables" below).

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX, Relation, RelationEntry,
};

use crate::attn_air::RangeRel;
use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::dequant::{FracRel, PairRel, id};
use crate::layer::MatLayout;
use crate::requant::{TT_POW2, TT_RANGE, rqv_at};

pub fn m31_of(v: i64) -> M31 {
    crate::attn_trace::m31_of(v)
}

// ------------------------------------------------------------------- angles ---

/// `rope(pos)` of `b1/model.py`, for one absolute position: `hd` bf16 patterns
/// of `cos` and `hd` of `sin`, with channel `j` reading angle `j mod (hd/2)`
/// (`cat([ang, ang], -1)`).  The angle itself is fp64 and the cosine is taken in
/// fp64, then narrowed to fp32 and stored as bf16 -- `store(ang.cos().float())`.
pub fn rope_at(inv: &[f64], hd: usize, pos: usize) -> (Vec<u16>, Vec<u16>) {
    let h = hd / 2;
    assert_eq!(inv.len(), h, "inv_freq is hd/2 long");
    let mut cos = vec![0u16; hd];
    let mut sin = vec![0u16; hd];
    for j in 0..hd {
        let ang = pos as f64 * inv[j % h];
        cos[j] = crate::glue::canon(bf16::from_f32(ang.cos() as f32));
        sin[j] = crate::glue::canon(bf16::from_f32(ang.sin() as f32));
    }
    (cos, sin)
}

/// The sign folded into the table: `rotate` builds `r = cat([-t[h:], t[:h]])`
/// and multiplies it by `sin`, so channel `j < h` reads `-t[j + h] sin[j]` and
/// channel `j >= h` reads `+t[j - h] sin[j]`.  Negating the *table* leaves the
/// program a pure permutation of committed leaves, which is what a glue `Prog`
/// can express with `Src::Leaf` alone.
pub fn sgn_sin(sin: &[u16], hd: usize) -> Vec<u16> {
    let h = hd / 2;
    (0..hd)
        .map(|j| {
            if j < h { crate::glue::canon(sin[j] ^ 0x8000) } else { crate::glue::canon(sin[j]) }
        })
        .collect()
}

/// A different legal bf16 value, for a control that must actually move a cell.
///
/// `p ^ 1` is the usual bend, and it is *silent* on a pattern the `(sm, ef)`
/// representation collapses: `split` sends every subnormal and every
/// infinity/NaN to `(0, 0)`, so flipping the low mantissa bit of a zero
/// produces a subnormal and commits the identical pair.  At absolute position 0
/// every `sin` is exactly that zero, so a probe aimed at the angle table's first
/// rows measured nothing -- which is the shape `drop-big` had at layer 0 and the
/// reason this exists rather than a bare `^ 1`.
pub fn bend_pat(p: u16) -> u16 {
    let p = crate::glue::canon(p);
    let q = crate::glue::canon(p ^ 1);
    if crate::glue::split(q) != crate::glue::split(p) {
        q
    } else {
        // 1.0, which no zero and no subnormal can be mistaken for
        crate::glue::join(128, 127)
    }
}

/// The `Prog::rope_mul` leaf a slot's second operand reads: `2 j` is
/// `cos[j]`, `2 j + 1` is `sgnsin[j]`.  One row of the region's preprocessed
/// block, for the token at absolute position `pos`.
pub fn pre_row(inv: &[f64], hd: usize, pos: usize) -> Vec<u16> {
    let (cos, sin) = rope_at(inv, hd, pos);
    let sgn = sgn_sin(&sin, hd);
    (0..hd).flat_map(|j| [cos[j], sgn[j]]).collect()
}

/// `b1/model.py`'s `inv_freq_of`, in fp64 and branch for branch.
///
/// The statement carries `theta`, the branch name and that branch's own
/// constants, and **this function is the table** -- so tree 0's angle columns
/// are a pure function of `Fixed.params` and a verifier never reads a `cos` or
/// a `sin` off a dump.  Stream LL's `pre-split` classifies them `needs: Params,
/// varies: Block`, which is what keeps the weights-tree design's margin; a
/// table read off the dump would be `needs: Model` and cost most of it.
///
/// `scale` is the branch's constants in the order `RopeParams::rope_scale`
/// documents: `llama3` reads four, `proportional` two, `default` none.
pub fn inv_freq_of(hd: usize, theta: f64, kind: &str, scale: &[f64]) -> Vec<f64> {
    let base: Vec<f64> =
        (0..hd / 2).map(|i| 1.0 / theta.powf(2.0 * i as f64 / hd as f64)).collect();
    match kind {
        "default" => base,
        // `llama3_inv_freq`: transformers' `_compute_llama3_parameters`
        "llama3" => {
            assert_eq!(scale.len(), 4,
                       "rope_type llama3 reads factor, low, high, original_max");
            let (factor, lo, hi, old) = (scale[0], scale[1], scale[2], scale[3]);
            base.iter()
                .map(|inv| {
                    let wavelen = 2.0 * std::f64::consts::PI / inv;
                    let scaled = if wavelen > old / lo { inv / factor } else { *inv };
                    let smooth = (old / wavelen - lo) / (hi - lo);
                    if wavelen >= old / hi && wavelen <= old / lo {
                        (1.0 - smooth) * scaled / factor + smooth * scaled
                    } else {
                        scaled
                    }
                })
                .collect()
        }
        // `proportional_inv_freq`: Gemma-4's global RoPE, where the pairs past
        // `partial_rotary_factor` are NoPE -- `inv_freq` exactly zero, so they
        // still go through the rotation algebra with cos 1 and sin 0
        "proportional" => {
            assert_eq!(scale.len(), 2,
                       "rope_type proportional reads partial_rotary_factor, factor");
            let (prf, factor) = (scale[0], scale[1]);
            let pairs = (prf * hd as f64 / 2.0) as usize;
            (0..hd / 2)
                .map(|i| {
                    if i < pairs {
                        1.0 / theta.powf(2.0 * i as f64 / hd as f64) / factor
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        other => panic!("rope_type {other:?} is not implemented"),
    }
}


// ------------------------------------------------------ the region's tables ---
//
// Stream NN-3.  The quantize regions used to borrow the attention output's
// `(q, nz)` pair table and `requant.rs`'s power-of-two division table, both of
// which stop at 127: `rq_table(Pow2)` keeps only quotients in `[1, 127]` and the
// shared `qnz` table is `Params::ao_q_bits` tall.  This model's `qk_bits` are
// `(12, 11)`, so on a real 8B chunk 35.9% of the `q` sites and 20.7% of the `k`
// sites had no row and the chunk did not prove.  The regions now own their
// tables and each side's is filtered to **that side's own domain**, which is the
// part that matters: a table wider than the side's `qmax` would silently prove
// the unclamped quotient at a site B1 clamps, turning an incompleteness into a
// disagreement with the model nothing in the statement could see.

/// The `v` side's width.  `b1/model.py` writes the cache's values as `int8`
/// (`vvc.i8`) whatever `qk_bits` says, and `Params` carries no `v_bits`; the
/// constant is here so that a model which changes it changes one line.
pub const V_BITS: u32 = 8;

/// The largest integer this side's width admits, as `b1/model.py`'s `quantize`
/// clamps it: `int8` gives 127, `qk_bits 12` gives 2047.
pub fn qmax_of(bits: u32) -> i64 {
    (1i64 << (bits - 1)) - 1
}

/// How many exponent differences the side's table carries.  `q = M 2^(tt - 9)`
/// with `M` in `[128, 255]`, so one more quotient bit keeps exactly one more
/// `tt`: eight bits is `requant.rs`'s own `TT_RANGE`, twelve is four rows of
/// `tt` more.  The AIR's `tt` offset (`134 - TT_POW2`) is untouched -- only the
/// top of the window moves.
pub fn qtt_range(bits: u32) -> i64 {
    TT_RANGE as i64 + (bits as i64 - 8)
}

/// `(tt, m7) -> q`, for the quotients this side can hold.  At `bits = 8` this is
/// `rq_table(Pow2)`'s three columns row for row.
pub fn qdiv_table(bits: u32) -> Vec<(u32, u32, u32)> {
    assert!((8..=16).contains(&bits), "a quantize side is 8 to 16 bits, not {bits}");
    let qmax = qmax_of(bits) as u64;
    let mut rows = vec![];
    for tt in 0..qtt_range(bits) as u32 {
        for m7 in 0..128u32 {
            let m = 128 + m7 as u64;
            // q = round_half_up(M 2^(tt-9)) = (M 2^(tt+1) + 512) / 1024
            let q = ((m << (tt + 1)) + 512) / 1024;
            if (1..=qmax).contains(&q) {
                rows.push((tt, m7, q as u32));
            }
        }
    }
    rows
}

/// The true quotient at a site, whatever its size -- used to tell a site B1
/// *clamps* from a site that is off the table for any other reason.
pub fn qdiv_ref(tt: i64, m7: u32) -> Option<i64> {
    if !(0..64).contains(&tt) {
        return None;
    }
    let m = 128 + m7 as u128;
    Some((((m << (tt + 1)) + 512) / 1024) as i64)
}

/// Stream NN-3, the clamp.  `b1/model.py`'s `quantize` clamps at `qmax`, so a
/// site whose true quotient is past it has no row in this side's division
/// table and no honest witness without this: `round_half_away(v)` is over
/// `qmax` exactly when `v >= qmax + 0.5`, and `|x| >= (qmax + 0.5) 2^j` is a
/// comparison of the committed bf16 *pattern* against a threshold -- the same
/// shape the `nz = 0` branch already proves, so it costs one committed boolean
/// and one range lookup and never needs the true quotient at all.
///
/// This returns that threshold at `j = 0`, as `ef * 128 + m7`.  Multiplying by
/// `2^j` is exact in bf16 (it adds `j` to `ef`), so the per-slot threshold is
/// `scale * 128 + clamp_pat(bits)` and stays degree 1 in the preprocessed
/// scale, exactly as `path` does.
pub fn clamp_pat(bits: u32) -> i64 {
    let v = qmax_of(bits) as f64 + 0.5;
    let e = v.log2().floor() as i64;
    let m = v / (e as f64).exp2();
    // the smallest significand at or above `m`, and the carry when there is none
    let mut m7 = (m * 128.0).ceil() as i64 - 128;
    let mut ef = e + 127;
    if m7 > 127 {
        m7 = 0;
        ef += 1;
    }
    ef * 128 + m7
}

pub const QDIV_COLS: usize = 3;

pub fn qdiv_col(bits: u32, k: usize) -> PreProcessedColumnId {
    id(format!("b65_qdiv{bits}_{k}"))
}
pub fn qdiv_log_rows(bits: u32) -> u32 {
    crate::common::log_rows_for(qdiv_table(bits).len(), 1)
}
/// Padded by repeating the last real row, the rule every table in this repo
/// pads by: a repeated row admits no triple the real rows do not.
pub fn qdiv_tab_cols(bits: u32) -> Vec<Eval> {
    let rows = qdiv_table(bits);
    let lr = crate::common::log_rows_for(rows.len(), 1);
    (0..QDIV_COLS)
        .map(|k| {
            eval_of(lr, (0..1usize << lr)
                .map(|i| {
                    let r = rows[i.min(rows.len() - 1)];
                    m31_u32([r.0, r.1, r.2][k])
                })
                .collect())
        })
        .collect()
}

/// The quantize family's own `(q, nz)` pair table, one per chunk at the widest
/// side's width.  It is not the attention output's: widening *that* one would
/// move `b65_attn_out_eval`'s fingerprint and every digest this repo has
/// recorded.  Its width is not load-bearing for soundness -- the division table
/// is what fixes the admissible `q` -- so one table at the maximum serves all
/// three regions.
pub fn qnz_log_rows(bits: u32) -> u32 {
    bits - 1
}
pub fn qnz_col(bits: u32, k: usize) -> PreProcessedColumnId {
    id(format!("b65_qqnz{bits}_{k}"))
}
pub fn qnz_tab_cols(bits: u32) -> Vec<Eval> {
    let lr = qnz_log_rows(bits);
    vec![
        eval_of(lr, (0..1u32 << lr).map(m31_u32).collect()),
        eval_of(lr, (0..1u32 << lr).map(|q| m31_u32(u32::from(q != 0))).collect()),
    ]
}

// ------------------------------------------------------------------ the AIR ---

pub const Q_SEL: [&str; 3] = ["nn_q_gt0", "nn_q_end", "nn_q_hstart"];

pub fn q_sel(tag: &str, name: &str) -> PreProcessedColumnId {
    id(format!("{name}_{tag}"))
}
/// The preprocessed power-of-two exponent of slot `sl`: `s = 2^j`, per
/// (head, channel), constant down the tokens and therefore one column per slot.
pub fn q_scale_col(tag: &str, sl: usize) -> PreProcessedColumnId {
    id(format!("nn_q_scale{sl}_{tag}"))
}

/// The two Horner claims one quantize region publishes.
///
/// Both fold the region's own rows under the **token-major** canonical position
/// `p = t n + c`, which is the order `MatLayout` walks its rows in -- so a
/// single row step serves, and the producing region's `Out(0)` tie and the
/// consuming band's fold are two sums over the same positions.
#[derive(Clone)]
pub struct QTie {
    /// the bf16 side, per slot: `w^(M - 1 - sl)` over the producer's own stream
    /// of `tokens x n` values
    pub w_in: Vec<SecureField>,
    /// its row step, `w^(-per_row)`
    pub step_in: SecureField,
    /// the integer side, per slot.  The band holds one column per **query**
    /// head, and grouped-query attention replicates one kv head across `rep` of
    /// them, so a producer value carries the sum of the `rep` weights its copies
    /// have on the band side -- `sum_r w^(-r hd)` times `w^(N - 1 - sl)` with
    /// `N = tokens nh hd`.  With the sum in the weight the equality still forces
    /// each copy separately: the exponents are distinct over
    /// (query head, channel, token), so the difference is a polynomial in `w`
    /// whose coefficients were fixed before `w` was drawn.
    pub w_out: Vec<SecureField>,
    /// its ordinary row step
    pub step_out: SecureField,
    /// and the step on a row that begins a new kv head, where the expanded
    /// channel index jumps by `(rep - 1) hd`.  Equal to `step_out` at `rep = 1`.
    pub step_out_h: SecureField,
    pub lam: SecureField,
    /// the bf16 side: `sm + lam ef`, what the rotation (or the v dequant) produced
    pub claimed_in: SecureField,
    /// the integer side: `q (1 - 2 sg)`, what the bands' columns have to hold
    pub claimed_out: SecureField,
}

/// The quantize: `qx = round_half_away(x / 2^j)` for a bf16 `x` and a
/// preprocessed per-slot exponent `j`, with the sign read off the signed
/// significand by one 257-row lookup.
#[derive(Clone)]
pub struct QuantEval {
    pub tag: String,
    pub log_rows: u32,
    pub per_row: usize,
    pub chunks: usize,
    /// stream NN-3: this side's integer width -- it fixes `qmax` and the
    /// clamp threshold, both of which the AIR reads as constants
    pub bits: u32,
    pub qnz: PairRel,
    pub rq3: FracRel,
    pub rqv: FracRel,
    pub r15: RangeRel,
    pub tie: Option<QTie>,
}

impl FrameworkEval for QuantEval {
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }
    fn cuda_eval_name(&self) -> &'static str { "b65_quant_eval" }
    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let gt0 = eval.get_preprocessed_column(q_sel(&self.tag, Q_SEL[0]));
        let end = eval.get_preprocessed_column(q_sel(&self.tag, Q_SEL[1]));
        let hstart = eval.get_preprocessed_column(q_sel(&self.tag, Q_SEL[2]));
        let c128 = M31::from_u32_unchecked(128);
        let two = M31::from_u32_unchecked(2);
        // read before the committed columns, so the preprocessed id order is
        // the one `Plan::pre_ids` writes
        let scale: Vec<E::F> =
            (0..self.per_row).map(|sl| eval.get_preprocessed_column(q_scale_col(&self.tag, sl)))
                .collect();

        let mut in_vals: Vec<(E::F, E::F)> = vec![];
        let mut out_vals: Vec<E::EF> = vec![];
        let cpat = M31::from_u32_unchecked(clamp_pat(self.bits) as u32);
        let qmax = M31::from_u32_unchecked(qmax_of(self.bits) as u32);
        for sl in 0..self.per_row {
            let ef = eval.next_trace_mask();
            let m7 = eval.next_trace_mask();
            let q = eval.next_trace_mask();
            let nz = eval.next_trace_mask();
            let sm = eval.next_trace_mask();
            let sg = eval.next_trace_mask();
            // stream NN-3: `cl` is "B1 clamped this site".  A clamped site has
            // no division-table row -- the table stops at this side's `qmax` --
            // so what the AIR proves instead is that the value was big enough
            // to clamp, and that the committed integer is `qmax`.
            let cl = eval.next_trace_mask();
            let pat = ef.clone() * c128 + m7.clone();
            eval.add_constraint(cl.clone() * (E::F::one() - cl.clone()));
            // a clamped site is a nonzero one, which is what lets the division
            // lookup's multiplicity stay degree 1 below (`nz - cl` is `[nz and
            // not cl]` exactly when `cl` implies `nz`)
            eval.add_constraint(cl.clone() * (E::F::one() - nz.clone()));
            eval.add_constraint(cl.clone() * (q.clone() - E::F::from(qmax)));
            // (q, nz) pins nz = [q != 0]
            eval.add_to_relation(RelationEntry::new(&self.qnz, E::EF::one(),
                                                    &[q.clone(), nz.clone()]));
            // nz = 1 and not clamped: this side's own power-of-two division
            // table, keyed by the exponent difference.  `scale[sl]` is
            // preprocessed, so `tt` stays degree 1 in the committed fields.
            let tt = ef.clone() - scale[sl].clone()
                - E::F::from(M31::from_u32_unchecked((134 - TT_POW2) as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.rq3, E::EF::from(nz.clone() - cl.clone()),
                &[tt, m7.clone(), q.clone()]));
            // nz = 0: |x| < s/2, a pattern comparison
            let path = (scale[sl].clone() + E::F::from(M31::from_u32_unchecked(126))) * c128;
            eval.add_to_relation(RelationEntry::new(
                &self.r15, E::EF::one() - E::EF::from(nz.clone()),
                &[path - E::F::one() - pat.clone()]));
            // cl = 1: |x| >= (qmax + 1/2) 2^j, the same comparison the other
            // way round
            let cpath = scale[sl].clone() * c128 + E::F::from(cpat);
            eval.add_to_relation(RelationEntry::new(
                &self.r15, E::EF::from(cl.clone()), &[pat - cpath]));
            // the signed significand, with `m7` and the sign read off it
            eval.add_to_relation(RelationEntry::new(
                &self.rqv, E::EF::one(), &[sm.clone(), m7.clone(), sg.clone()]));
            in_vals.push((sm, ef));
            out_vals.push(E::EF::from(q.clone()) - E::EF::from(q * sg * two));
        }
        // ---- the two sides of the seam ---------------------------------------
        // One row step, not two: the canonical position is token-major, so it
        // falls by `per_row` from one `MatLayout` row to the next whether or not
        // the row crosses a token.  `gt0` is `[j > 0]` and stops the `[-1, 0]`
        // mask wrapping the last row into the first.
        if let Some(t) = &self.tie {
            let step_in = E::EF::from(gt0.clone()) * t.step_in;
            let [ui_prev, ui] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut fi = E::EF::zero();
            for (k, (sm, ef)) in in_vals.iter().enumerate() {
                fi += (E::EF::from(sm.clone()) + E::EF::from(ef.clone()) * t.lam) * t.w_in[k];
            }
            eval.add_constraint(ui.clone() - ui_prev * step_in - fi);
            eval.add_constraint((ui - t.claimed_in) * end.clone());

            // `hstart` is a subset of `gt0`, so this stays degree 1
            let step_out = E::EF::from(gt0.clone()) * t.step_out
                + E::EF::from(hstart.clone()) * (t.step_out_h - t.step_out);
            let [uo_prev, uo] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut fo = E::EF::zero();
            for (k, v) in out_vals.iter().enumerate() {
                fo += v.clone() * t.w_out[k];
            }
            eval.add_constraint(uo.clone() - uo_prev * step_out - fo);
            eval.add_constraint((uo - t.claimed_out) * end);
        }
        eval.finalize_logup_in_pairs();
        eval
    }
}

// ------------------------------------------------------------------ witness ---

#[derive(Clone, Copy, Debug)]
pub enum QTamper {
    /// one committed bf16 input moved to another legal pattern, with the whole
    /// row recomputed around it: every constraint still holds and only the
    /// input tie can see it
    In(usize),
    /// one committed quotient moved, the input left alone: the division table
    /// rejects, which is the control that says the lookup is load-bearing
    Out(usize),
    /// one preprocessed scale exponent moved: `pre_root` is what rejects
    Scale(usize),
}

impl QTamper {
    pub fn name(&self) -> &'static str {
        match self {
            QTamper::In(_) => "one quantize input moved to another legal bf16",
            QTamper::Out(_) => "one quantize output moved off its own table row",
            QTamper::Scale(_) => "one preprocessed quantize exponent moved by one",
        }
    }
}

pub struct QRegion {
    pub tag: String,
    pub lay: MatLayout,
    pub tokens: usize,
    pub n: usize,
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// natural row order, `per_row` per row: the `(sm, ef)` the input side folds
    pub in_nat: Vec<(i64, i64)>,
    /// the same, for the signed integer the output side folds
    pub out_nat: Vec<i64>,
    pub off_table: usize,
    /// how many of the derived integers equal the band's committed value
    pub ref_ok: usize,
    pub ref_n: usize,
    /// Stream NN-3: sites whose true quotient is past **this side's own**
    /// `qmax` -- B1 clamps them and the region has no row for them, so a nonzero
    /// count is an incompleteness and not a disagreement.  Before NN-3 the
    /// counter read "past the shared table's window", which at `qk_bits (12,
    /// 11)` was overwhelmingly "the region is narrower than the model" instead.
    pub clamped: usize,
    /// this side's integer width, and the two tables it owns
    pub bits: u32,
    /// Stream NN-3: the per-channel scale exponents this region *committed* to
    /// tree 0 -- the plan's own, or the bent copy under a `Scale` probe.  The
    /// `s_v` tie compares the `v` region's against the attention output's.
    pub je: Vec<i64>,
    pub m_rq: Vec<u32>,
    pub m_qnz: Vec<u32>,
    pub m_rqv: Vec<u32>,
    pub log_rq: u32,
    pub fired: usize,
}

impl QRegion {
    pub fn base_cells(&self) -> usize {
        self.base.len() << self.lay.log_rows
    }
    pub fn pre_cells(&self) -> usize {
        self.pre.len() << self.lay.log_rows
    }
}

pub const Q_PER_SLOT: usize = 7; // ef, m7, q, nz, sm, sg, cl

/// Build one quantize region.  `x` is the bf16 stream in `(token, channel)`
/// order -- the rotation's output for q and k, the dequant's for v -- `jj` the
/// per-channel exponent (`n` of them, constant down the tokens) and `want` the
/// integers the band committed, which the region only *checks* against.
#[allow(clippy::too_many_arguments)]
pub fn q_region(tag: &str, x: &[u16], jj: &[i64], want: Option<&[i64]>, tokens: usize,
                n: usize, hd: usize, bits: u32, nz_bits: u32, lay: MatLayout,
                tables: &mut crate::layer::Tables, tamper: Option<QTamper>) -> QRegion {
    assert_eq!(lay.lead, 0, "a quantize region needs an exact tiling ({tag})");
    assert_eq!(jj.len(), n, "{tag}: one exponent per channel");
    assert_eq!(hd % lay.per_row, 0,
               "{tag}: a quantize row must sit inside one head ({} does not divide {hd})",
               lay.per_row);
    assert!(nz_bits >= bits, "{tag}: the (q, nz) table is the widest side's");
    let pr = lay.per_row;
    let rows = 1usize << lay.log_rows;
    // stream NN-3: this side's own division table, filtered to its own domain
    let (qmax, tt_top) = (qmax_of(bits), qtt_range(bits));
    let cpat = clamp_pat(bits);
    let rq = qdiv_table(bits);
    let log_rq = crate::common::log_rows_for(rq.len(), 1);
    let rq_at = |tt: i64, m7: u32| -> Option<usize> {
        if !(0..tt_top).contains(&tt) {
            return None;
        }
        rq.binary_search_by_key(&(tt as u32, m7), |r| (r.0, r.1)).ok()
    };
    let n_cols = pr * Q_PER_SLOT;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut m_rq = vec![0u32; 1 << log_rq];
    let mut m_qnz = vec![0u32; 1usize << qnz_log_rows(nz_bits)];
    let mut m_rqv = vec![0u32; 512];
    let (mut off_table, mut fired, mut clamped) = (0usize, 0usize, 0usize);
    let (mut ref_ok, mut ref_n) = (0usize, 0usize);

    // the committed exponents, bent by the `Scale` probe.  It moves a
    // *preprocessed* cell, so every constraint in the region still holds with
    // the quotients rebuilt around it and only `pre_root` can reject.
    let mut je = jj.to_vec();
    if let Some(QTamper::Scale(t)) = tamper {
        let at = t % n;
        je[at] += 1;
        fired += 1;
    }
    // The input probe has to be *quiet on the output*: it moves the bf16 value
    // this region rounded and must leave the integer it produced alone, or
    // `nn_out_ok` sees it too and the control has not isolated the input tie it
    // was written for.  A one-ulp move of the mantissa usually rounds to the
    // same quotient; this scans forward from the asked-for site for one that
    // does, and says so rather than silently taking a coupled site.
    let quot = |pat: u16, j: i64| -> Option<u32> {
        let (sm, ef) = crate::glue::split(crate::glue::canon(pat));
        let m7 = if sm == 0 { 0 } else { sm.unsigned_abs() as u32 - 128 };
        let path = (j + 126) * 128;
        if (ef as i64 * 128 + m7 as i64) < path {
            return Some(0);
        }
        rq_at(ef as i64 - j - (134 - TT_POW2), m7).map(|ri| rq[ri].2)
    };
    let bend_at: Option<usize> = match tamper {
        Some(QTamper::In(t)) => {
            let n_all = tokens * n;
            let start = t % n_all;
            let hit = (0..n_all).map(|d| (start + d) % n_all).find(|i| {
                let (p, c) = (crate::glue::canon(x[*i]), *i % n);
                let b = bend_pat(p);
                b != p && quot(b, je[c]).is_some() && quot(b, je[c]) == quot(p, je[c])
            });
            if hit.is_none() {
                eprintln!("{tag}: no quotient-preserving input bend exists on this data; \
the probe would move the output too");
            }
            hit
        }
        _ => None,
    };
    let bend = |i: usize| -> u16 {
        let p = crate::glue::canon(x[i]);
        if bend_at == Some(i) { bend_pat(p) } else { p }
    };

    let ridx15 = tables.ridx(15);
    let mut m_r15 = std::mem::take(&mut tables.m_r[ridx15]);
    let mut in_nat: Vec<(i64, i64)> = Vec::with_capacity(rows * pr);
    let mut out_nat: Vec<i64> = Vec::with_capacity(rows * pr);
    for j in 0..rows {
        let (t, b) = lay.at(tokens, j);
        for sl in 0..pr {
            let c = b * pr + sl;
            let gi = t * n + c;
            let (sm, ef) = crate::glue::split(bend(gi));
            let m7 = if sm == 0 { 0 } else { sm.unsigned_abs() as u32 - 128 };
            let sg = u32::from(sm < 0);
            let pat = ef as i64 * 128 + m7 as i64;
            let path = (je[c] + 126) * 128;
            let nz = pat >= path;
            let (mut q, nzc) = (0u32, u32::from(nz));
            let mut clc = 0u32;
            if nz {
                let tt = ef as i64 - je[c] - (134 - TT_POW2);
                match rq_at(tt, m7) {
                    Some(ri) => {
                        q = rq[ri].2;
                        m_rq[ri] += 1;
                    }
                    None => {
                        // stream NN-3: the table is this side's own, so "no row"
                        // is "B1 clamped here" exactly when the true quotient is
                        // past this side's `qmax`.  `nz` says |x| >= s/2, so the
                        // quotient is at least 1 and a missing row means it is
                        // past `qmax`: either `tt` is past the window (where the
                        // smallest quotient is already 2^(bits-1)) or it is the
                        // window's top row.  Such a site is B1's clamp, and the
                        // region proves it by the pattern comparison rather than
                        // by a table row it does not have.
                        if tt >= tt_top || qdiv_ref(tt, m7).is_some_and(|t| t > qmax) {
                            let d = pat - (je[c] * 128 + cpat);
                            if (0..1 << 15).contains(&d) {
                                m_r15[d as usize] += 1;
                                q = qmax as u32;
                                clc = 1;
                                clamped += 1;
                            } else {
                                off_table += 1;
                            }
                        } else {
                            off_table += 1
                        }
                    }
                }
            } else {
                let d = path - 1 - pat;
                if (0..1 << 15).contains(&d) {
                    m_r15[d as usize] += 1;
                } else {
                    off_table += 1;
                }
            }
            if let Some(QTamper::Out(tt)) = tamper {
                if tt % (tokens * n) == gi && clc == 0 {
                    q = if (q as i64) < qmax { q + 1 } else { q - 1 };
                    fired += 1;
                }
            }
            if (q as usize) < m_qnz.len() {
                m_qnz[q as usize] += 1;
            }
            match rqv_at(sm) {
                Some(k) => m_rqv[k] += 1,
                None => off_table += 1,
            }
            let c0 = sl * Q_PER_SLOT;
            cols[c0][j] = m31_u32(ef);
            cols[c0 + 1][j] = m31_u32(m7);
            cols[c0 + 2][j] = m31_u32(q);
            cols[c0 + 3][j] = m31_u32(nzc);
            cols[c0 + 4][j] = m31_of(sm);
            cols[c0 + 5][j] = m31_u32(sg);
            cols[c0 + 6][j] = m31_u32(clc);
            let signed = if sg == 1 { -(q as i64) } else { q as i64 };
            in_nat.push((sm, ef as i64));
            out_nat.push(signed);
            if let Some(wv) = want {
                ref_n += 1;
                if signed == wv[gi] {
                    ref_ok += 1;
                }
            }
        }
    }
    tables.m_r[ridx15] = m_r15;
    tables.off_table += off_table;

    // tree 0: two selectors and one exponent column per slot, in the order
    // `QuantEval::evaluate` and `Plan::pre_ids` both read them
    let sel = |f: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
    };
    let m = (hd / pr).max(1);
    let mut pre = vec![sel(&|j| j > 0), sel(&|j| j + 1 == rows), sel(&|j| j > 0 && j % m == 0)];
    for sl in 0..pr {
        pre.push(eval_of(lay.log_rows, (0..rows)
            .map(|j| {
                let (_, b) = lay.at(tokens, j);
                m31_of(je[b * pr + sl])
            })
            .collect()));
    }
    if bend_at.is_some() {
        fired += 1;
    }
    if tamper.is_some() && fired == 0 {
        eprintln!("{tag}: the {} probe did not fire -- this region is the honest one",
                  tamper.unwrap().name());
    }
    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    QRegion { tag: tag.to_string(), lay, tokens, n, pre, base, in_nat, out_nat,
              off_table, ref_ok, ref_n, clamped, bits, je, m_rq, m_qnz, m_rqv, log_rq, fired }
}

/// One quantize region's interaction columns: the two tie accumulators, then
/// the LogUp fractions in the AIR's own lookup order.
pub fn q_interaction(r: &QRegion, tie: &QTie, hd: usize, qnz: &PairRel, rq3: &FracRel,
                     r15: &RangeRel, rqv: &FracRel)
                     -> (Vec<Eval>, SecureField, SecureField, SecureField) {
    use stwo::prover::backend::simd::m31::PackedM31;
    use stwo::prover::backend::simd::qm31::PackedQM31;
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    let m = (hd / pr).max(1);
    let mut cols: Vec<Eval> = vec![];
    let mut claims = [SecureField::zero(); 2];
    for side in 0..2 {
        let mut h = vec![SecureField::zero(); rows];
        for j in 0..rows {
            let mut f = SecureField::zero();
            for k in 0..pr {
                let idx = j * pr + k;
                let (val, wk) = if side == 0 {
                    let (sm, ef) = r.in_nat[idx];
                    (SecureField::from(m31_of(sm)) + SecureField::from(m31_of(ef)) * tie.lam,
                     tie.w_in[k])
                } else {
                    (SecureField::from(m31_of(r.out_nat[idx])), tie.w_out[k])
                };
                f += val * wk;
            }
            let step = if side == 0 {
                tie.step_in
            } else if j % m == 0 {
                tie.step_out_h
            } else {
                tie.step_out
            };
            h[j] = if j == 0 { f } else { h[j - 1] * step + f };
        }
        claims[side] = h[rows - 1];
        cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &h));
    }
    let n_looks_slot = 5; // qnz, division, sub-half, clamp, RQV
    let n_looks = pr * n_looks_slot;
    let one = PackedQM31::broadcast(SecureField::one());
    let getb = |c: usize, vr: usize| r.base[c].data[vr];
    let getp = |c: usize, vr: usize| r.pre[c].data[vr];
    let c128p = PackedM31::broadcast(m31_u32(128));
    let c126 = PackedM31::broadcast(m31_u32(126));
    let c141 = PackedM31::broadcast(m31_u32((134 - TT_POW2) as u32));
    let onep = PackedM31::broadcast(M31::one());
    let cpatp = PackedM31::broadcast(m31_u32(clamp_pat(r.bits) as u32));
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        let (sl, which) = (i / n_looks_slot, i % n_looks_slot);
        let c0 = sl * Q_PER_SLOT;
        let (ef, m7, q, nz) =
            (getb(c0, vr), getb(c0 + 1, vr), getb(c0 + 2, vr), getb(c0 + 3, vr));
        let cl = getb(c0 + 6, vr);
        // the preprocessed block is `[gt0, end, hstart]` then one exponent per slot
        let sc = getp(3 + sl, vr);
        let pat = ef * c128p + m7;
        let path = (sc + c126) * c128p;
        match which {
            0 => (one, qnz.combine(&[q, nz])),
            1 => (PackedQM31::from(nz - cl), rq3.combine(&[ef - sc - c141, m7, q])),
            2 => (one - PackedQM31::from(nz), r15.combine(&[path - onep - pat])),
            3 => (PackedQM31::from(cl), r15.combine(&[pat - (sc * c128p + cpatp)])),
            _ => (one, rqv.combine(&[getb(c0 + 4, vr), m7, getb(c0 + 5, vr)])),
        }
    };
    let (n_pairs, fpair) = crate::plogup::paired(n_looks, fr);
    let (lcols, use_sum) = crate::plogup::build(lay.log_rows, n_pairs, fpair);
    cols.extend(lcols);
    (cols, use_sum, claims[0], claims[1])
}

/// The quantize region's tiling.  `per_row` must divide `hd`, so that a row
/// never straddles two heads and the GQA replication weight is a per-slot
/// constant; and the tiling must be exact, because a padded row would commit a
/// bf16 pattern nobody produced and the input tie would carry it.
pub fn q_layout(n: usize, hd: usize, tokens: usize, target: usize) -> MatLayout {
    let mut best: Option<MatLayout> = None;
    for pr in 1..=hd {
        if hd % pr != 0 || n % pr != 0 {
            continue;
        }
        let chunks = n / pr;
        let active = tokens * chunks;
        if !active.is_power_of_two() || active < (1 << crate::layer::MIN_MAT_LOG_ROWS) {
            continue;
        }
        let l = MatLayout { per_row: pr, chunks, log_rows: active.trailing_zeros(), lead: 0 };
        let better = best.as_ref().is_none_or(|b| {
            (pr as isize - target as isize).abs() < (b.per_row as isize - target as isize).abs()
        });
        if better {
            best = Some(l);
        }
    }
    best.unwrap_or_else(|| panic!("no exact quantize tiling for n={n} hd={hd} tokens={tokens}"))
}

/// NN-2: does a quantize region tile this shape at all?  `q_layout` panics when
/// it does not, and the witness generator has to ask before it turns the family
/// on -- a shape whose widths line up but whose rows do not tile leaves the
/// rotation off and reports `nn_open`, exactly as a shape whose widths do not.
pub fn q_tiles(n: usize, hd: usize, tokens: usize) -> bool {
    (1..=hd).any(|pr| {
        hd % pr == 0 && n % pr == 0 && {
            let active = tokens * (n / pr);
            active.is_power_of_two() && active >= (1 << crate::layer::MIN_MAT_LOG_ROWS)
        }
    })
}

/// The reference: `round_half_away(x / 2^j)`, clamped, as B1's `quantize` does
/// it.  Used by the synthetic witness generator to build the bands' own columns
/// and by the desk's B1 check.
pub fn quant_ref(pat: u16, j: i64, qmax: i64) -> i64 {
    let (sm, ef) = crate::glue::split(crate::glue::canon(pat));
    if sm == 0 {
        return 0;
    }
    let mag = sm.unsigned_abs() as f64;
    // |x| = mag * 2^(ef - 134); the quotient is mag * 2^(ef - 134 - j)
    let v = mag * ((ef as i64 - 134 - j) as f64).exp2();
    let q = (v + 0.5).floor() as i64;
    let q = q.min(qmax);
    if sm < 0 { -q } else { q }
}

/// The smallest power of two that quantizes every value of `col` into
/// `[-qmax, qmax]` with no clamp -- B1's `qscale(.., p2=True)` on a slice, which
/// is what the desk's synthetic layer uses to pick an honest `s_k` / `s_v`.
pub fn pow2_scale(col: &[u16], qmax: i64) -> i64 {
    let m = col
        .iter()
        .map(|p| {
            let (sm, ef) = crate::glue::split(crate::glue::canon(*p));
            if sm == 0 { 0.0 } else { sm.unsigned_abs() as f64 * ((ef as i64 - 134) as f64).exp2() }
        })
        .fold(0.0f64, f64::max);
    if m <= 0.0 {
        return -20;
    }
    (m / qmax as f64).log2().ceil() as i64
}

/// bf16 from an f32, as `store` does it -- round to nearest even on the top 16
/// bits.  `bf16::from_f64` is the same function through f64.
pub fn f32_to_bf16(v: f32) -> u16 {
    bf16::from_f32(v)
}
