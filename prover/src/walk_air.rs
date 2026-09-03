//! The two region evals B6.5 adds to compose a layer proof.
//!
//! `WalkEval` is B3's r^T W walk, on B6's ChanEval skeleton: one row group per
//! *input* channel `k`, chunked; `pk` preprocessed weights and `pq` committed
//! activation bytes per row; `v = sum_t s^t qX[k, t]` shared by every matmul of
//! the group (q/k/v share one input, so do gate/up), one `u_m = sum_i rv^i W_m[i, k]`
//! and one `acc_m = sum_k u_m v` per matmul.  The verifier compares each `acc_m`
//! against the dequant region's claimed side `h_m = sum_{t,i} s^t rv^i acc[t, i]`
//! over the *same committed accumulator cells* the dequant rounds — that equality
//! is the Freivalds identity `acc = qX qW` (mod p), with nothing committed twice.
//!
//! `AttnOutEval` replaces the bands' A.V output region *and* rounds the output:
//! one row per (head, token, channel-chunk), the A.V outputs committed here as
//! `acc`, the int8 fold (`q = round(acc * 2^e / Z)`, B6's `int8` variant with the
//! review's two edge fixes) applied slot-wise, and two Horner accumulators that
//! reproduce the claims the bands' identities need:
//!
//!   h  = sum_h rv^(hd*h) sum_t s^t sum_j rv^j acc[h,t,j]   vs  sum_b rv^(hd*off_b) av_b
//!   dz = sum_h r^(C*h)   sum_t s^t den[h,t]                vs  sum_b r^(C*off_b)  az_b
//!
//! Per-slot the scale exponent `e = jv[h,j] - jo[t]` splits into `num = 2^max(e,0)`
//! (preprocessed) and a committed `denp = den * 2^max(-e,0)` pinned by a
//! preprocessed `pj` column, so the error identity stays degree 3.

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX, RelationEntry};

use crate::attn_air::RangeRel;
use crate::dequant::PairRel;
use crate::layout::Grid;

pub fn id(s: impl Into<String>) -> PreProcessedColumnId {
    PreProcessedColumnId { id: s.into() }
}
pub const WALK_SEL: [&str; 4] = ["nofirst", "last", "notzero", "end"];
pub fn walk_sel(tag: &str, name: &str) -> PreProcessedColumnId {
    id(format!("b65_walk{tag}_{name}"))
}
pub fn walk_w(tag: &str, mat: usize, p: usize) -> PreProcessedColumnId {
    id(format!("b65_walk{tag}_w{mat}_{p:04}"))
}
pub const AO_SEL: [&str; 6] = ["nofirst", "last", "m1", "m2", "m3", "end"];
pub fn ao_sel(name: &str) -> PreProcessedColumnId {
    id(format!("b65_ao_{name}"))
}
pub fn ao_chan(kind: &str, slot: usize) -> PreProcessedColumnId {
    id(format!("b65_ao_{kind}{slot:04}"))
}

/// Stream II: the walk's own Horner claim over its committed activations, in
/// the canonical channel-major stream order `q = c * tokens + t`, so that a
/// producing region's claim over the same sequence is comparable to it value
/// for value.
///
/// The walk's grid runs `(input channel, token chunk)` down the rows with the
/// tokens in the slots, so the row step takes two values -- `m_in` inside one
/// channel's chunks and `m_bd` at a channel boundary -- which is the shape
/// `dequant::OSide` already runs at.  Both selectors it needs are ones this
/// region already commits: `nofirst` is `[chunk + 1 < chunks]`, which is true
/// only away from a channel's first row, and `notzero` is `[j > 0]`, so
/// `notzero - nofirst` is exactly "this row begins a channel and is not row 0".
/// No new preprocessed column.
#[derive(Clone)]
pub struct WalkQx {
    /// one weight per token slot
    pub w: Vec<SecureField>,
    pub m_in: SecureField,
    pub m_bd: SecureField,
    pub claimed: SecureField,
}

// ------------------------------------------------------------- the weight walk --

#[derive(Clone)]
pub struct WalkEval {
    pub tag: String,
    pub g: Grid,
    /// weights per row per matmul — B3's packing knob `p`
    pub pk: usize,
    /// activation bytes per row (the token chunk)
    pub pq: usize,
    pub n_mats: usize,
    /// rv^0..rv^{pk-1} and rv^pk
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    /// s^0..s^{pq-1} and s^pq
    pub s_pow: Vec<SecureField>,
    pub s_step: SecureField,
    /// one claimed inner product per matmul
    pub claimed: Vec<SecureField>,
    pub rel_act: RangeRel,
    /// The committed activations' offset: 128 for the int8 walks, 2^11 for the
    /// widened attention output's group (stream Z), whose activations are one
    /// cell in `[1, 2^12)` against a 12-bit range table rather than two limbs.
    pub act_off: u32,
    pub range_check: bool,
    /// B11 stream V, lever 4: this region's range checks are proved by the
    /// batched GKR instead of committed running-sum columns, so the AIR emits
    /// neither the relation entries nor the interaction masks they would
    /// allocate. `range_check` still says whether the checks exist at all; this
    /// says who owns them.
    pub gkr: bool,
    /// Stream II: the activation claim, or `None` on a chunk with no wiring
    /// family -- which is what keeps such a chunk's proof exactly what it was.
    pub qx: Option<WalkQx>,
}

impl FrameworkEval for WalkEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_walk_eval" }

    fn log_size(&self) -> u32 {
        self.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> = WALK_SEL
            .iter()
            .map(|n| eval.get_preprocessed_column(walk_sel(&self.tag, n)))
            .collect();
        let (nofirst, last, notzero, end) =
            (sel[0].clone(), sel[1].clone(), sel[2].clone(), sel[3].clone());
        let ws: Vec<Vec<E::F>> = (0..self.n_mats)
            .map(|m| {
                (0..self.pk)
                    .map(|p| eval.get_preprocessed_column(walk_w(&self.tag, m, p)))
                    .collect()
            })
            .collect();
        let acts: Vec<E::F> = (0..self.pq).map(|_| eval.next_trace_mask()).collect();

        let [v_prev, v] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let mut vf = E::EF::zero();
        for (p, a) in acts.iter().enumerate() {
            vf += a.clone() * self.s_pow[p];
        }
        let off_sum: SecureField =
            self.s_pow.iter().fold(SecureField::zero(), |x, y| x + *y)
                * SecureField::from_u32_unchecked(self.act_off, 0, 0, 0);
        vf = vf - off_sum;
        eval.add_constraint(v.clone() - (v_prev * self.s_step) * nofirst.clone() - vf);

        for m in 0..self.n_mats {
            let [u_prev, u] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let [acc_prev, acc] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut uf = E::EF::zero();
            for (p, w) in ws[m].iter().enumerate() {
                uf += w.clone() * self.rv_pow[p];
            }
            eval.add_constraint(u.clone() - (u_prev * self.rv_step) * nofirst.clone() - uf);
            eval.add_constraint(acc.clone() - acc_prev * notzero.clone() - (u * v.clone()) * last.clone());
            eval.add_constraint((acc - self.claimed[m]) * end.clone());
        }
        // stream II: the activation claim, allocated before the LogUp columns
        // so that the interaction order is ties-then-fractions, as the wiring
        // family's regions already are.  The value folded is `a - act_off`, the
        // signed activation, which is what a producer's requantize publishes.
        if let Some(q) = &self.qx {
            let [x_prev, x] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let off = E::F::from(M31::from_u32_unchecked(self.act_off));
            let mut xf = E::EF::zero();
            for (p, a) in acts.iter().enumerate() {
                xf += (E::EF::from(a.clone()) - E::EF::from(off.clone())) * q.w[p];
            }
            let mult = E::EF::from(nofirst.clone()) * q.m_in
                + (E::EF::from(notzero.clone()) - E::EF::from(nofirst.clone())) * q.m_bd;
            eval.add_constraint(x.clone() - x_prev * mult - xf);
            eval.add_constraint((x - q.claimed) * end.clone());
        }
        if self.range_check && !self.gkr {
            for a in &acts {
                eval.add_to_relation(RelationEntry::new(&self.rel_act, E::EF::one(),
                                                        core::slice::from_ref(a)));
            }
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

// -------------------------------------------------- the attention-output region --

#[derive(Clone)]
pub struct AttnOutEval {
    pub log_rows: u32,
    /// channel slots per row
    pub per_row: usize,
    /// rv^0..rv^{per_row-1} and rv^per_row
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    pub s: SecureField,
    /// rv^hd * s^-(S-1): the head-boundary step of `h`
    pub mu_out: SecureField,
    /// r^C * s^-(S-1): the head-boundary step of `dz`
    pub mu_den: SecureField,
    pub claimed_out: SecureField,
    pub claimed_den: SecureField,
    pub qnz: PairRel,
    pub rel_rem: RangeRel,
    /// B11 stream V, lever 4: as `WalkEval::gkr`.
    pub gkr: bool,
}

impl FrameworkEval for AttnOutEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_attn_out_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let two = M31::from_u32_unchecked(2);
        let sel: Vec<E::F> =
            AO_SEL.iter().map(|n| eval.get_preprocessed_column(ao_sel(*n))).collect();
        let (nofirst, last, m1, m2, m3, end) = (sel[0].clone(), sel[1].clone(), sel[2].clone(),
                                                sel[3].clone(), sel[4].clone(), sel[5].clone());
        let den = eval.next_trace_mask();
        let mut accs: Vec<E::F> = Vec::with_capacity(self.per_row);
        for slot in 0..self.per_row {
            let num = eval.get_preprocessed_column(ao_chan("n", slot));
            let pj = eval.get_preprocessed_column(ao_chan("p", slot));
            let acc = eval.next_trace_mask();
            let sg = eval.next_trace_mask();
            let qm = eval.next_trace_mask();
            let nz2 = eval.next_trace_mask();
            let denp = eval.next_trace_mask();
            let d = eval.next_trace_mask();
            accs.push(acc.clone());
            eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
            // the sign pin (B6 review, gap 7), as in the standalone gadget
            eval.add_constraint(sg.clone() * (E::F::one() - nz2.clone()));
            // denp = den * 2^max(-e, 0), committed so the error identity and the
            // tie-excluding bound stay degree <= 3 / degree 1
            eval.add_constraint(denp.clone() - den.clone() * pj);
            let mag = acc.clone() - (acc * sg) * two;
            eval.add_constraint(
                d.clone() - (mag * num) * two + (qm.clone() * two - E::F::one()) * denp.clone(),
            );
            let dp = denp * two - E::F::one() - d.clone();
            if !self.gkr {
                eval.add_to_relation(RelationEntry::new(&self.qnz, E::EF::one(),
                                                        &[qm, nz2]));
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_rem, E::EF::one(), core::slice::from_ref(&d)));
                eval.add_to_relation(RelationEntry::new(
                    &self.rel_rem, E::EF::one(), core::slice::from_ref(&dp)));
            }
        }

        let [z_prev, z] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [h_prev, h] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [dz_prev, dz] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let mut zf = E::EF::zero();
        for (i, a) in accs.iter().enumerate() {
            zf += a.clone() * self.rv_pow[i];
        }
        eval.add_constraint(z.clone() - (z_prev * self.rv_step) * nofirst - zf);
        let mut mult: E::EF = E::EF::from(m1.clone());
        mult += m2.clone() * self.s;
        mult += m3.clone() * self.mu_out;
        eval.add_constraint(h.clone() - h_prev * mult - z * last.clone());
        let mut dmult: E::EF = E::EF::from(m1);
        dmult += m2 * self.s;
        dmult += m3 * self.mu_den;
        eval.add_constraint(dz.clone() - dz_prev * dmult - E::EF::from(den * last));
        eval.add_constraint((h - self.claimed_out) * end.clone());
        eval.add_constraint((dz - self.claimed_den) * end);
        if !self.gkr {
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}
