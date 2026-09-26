//! The `r^T W` Freivalds check B6.5 adds to compose a layer proof.
//!
//! `FreivaldsEval` is B3's r^T W Freivalds check, on B6's ChanEval skeleton: one row group per
//! *input* channel `k`, chunked; `pk` preprocessed weights and `pq` committed
//! activation bytes per row; `v = sum_t s^t qX[k, t]` shared by every matmul of
//! the group (q/k/v share one input, so do gate/up), one `u_m = sum_i rv^i W_m[i, k]`
//! and one `acc_m = sum_k u_m v` per matmul.  The verifier compares each `acc_m`
//! against the dequant component's claimed side `h_m = sum_{t,i} s^t rv^i acc[t, i]`
//! over the *same committed accumulator cells* the dequant rounds — that equality
//! is the Freivalds identity `acc = qX qW` (mod p), with nothing committed twice.

pub mod claims;
pub mod device;
pub mod layout;
pub mod run;
pub mod trace;

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX,
                                RelationEntry};

use crate::layout::Grid;
use crate::relations::{RangeRel, id};

// The attention-output component is `gadgets/attn_out` since #197, and
// `compose.rs`, `cli.rs` and `gkr_key_tile.rs` still reach its eval and its
// preprocessed ids through this module.
pub use crate::gadgets::attn_out::air::{AO_SEL, AttnOutEval, EPOW_COLS, ao_chan, ao_sel};

pub const FREIVALDS_SEL: [&str; 4] = ["nofirst", "last", "notzero", "end"];
pub fn freivalds_sel(tag: &str, name: &str) -> PreProcessedColumnId {
    id(format!("zk_freivalds{tag}_{name}"))
}
pub fn freivalds_w(tag: &str, mat: usize, p: usize) -> PreProcessedColumnId {
    id(format!("zk_freivalds{tag}_w{mat}_{p:04}"))
}
/// Stream II: the Freivalds check's own Horner claim over its committed activations, in
/// the canonical channel-major stream order `q = c * tokens + t`, so that a
/// producing component's claim over the same sequence is comparable to it value
/// for value.
///
/// The Freivalds check's grid runs `(input channel, token chunk)` down the rows with the
/// tokens in the slots, so the row step takes two values -- `m_in` inside one
/// channel's chunks and `m_bd` at a channel boundary -- which is the shape
/// `dequant::OutputClaim` already runs at.  Both selectors it needs are ones this
/// component already commits: `nofirst` is `[chunk + 1 < chunks]`, which is true
/// only away from a channel's first row, and `notzero` is `[j > 0]`, so
/// `notzero - nofirst` is exactly "this row begins a channel and is not row 0".
/// No new preprocessed column.
#[derive(Clone)]
pub struct FreivaldsQx {
    /// one weight per token slot
    pub w: Vec<SecureField>,
    pub m_in: SecureField,
    pub m_bd: SecureField,
    pub claimed: SecureField,
}

// -------------------------------------------------- the weight Freivalds check --

#[derive(Clone)]
pub struct FreivaldsEval {
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
    /// The committed activations' offset: 128 for the int8 Freivalds checks, 2^11 for the
    /// widened attention output's group (stream Z), whose activations are one
    /// cell in `[1, 2^12)` against a 12-bit range table rather than two limbs.
    pub act_off: u32,
    pub range_check: bool,
    /// B11 stream V, lever 4: this component's range checks are proved by the
    /// batched GKR instead of committed running-sum columns, so the AIR emits
    /// neither the relation entries nor the interaction masks they would
    /// allocate. `range_check` still says whether the checks exist at all; this
    /// says who owns them.
    pub gkr: bool,
    /// Stream II: the activation claim, or `None` on a chunk with no wiring
    /// family -- which is what keeps such a chunk's proof exactly what it was.
    pub qx: Option<FreivaldsQx>,
}

impl FrameworkEval for FreivaldsEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_freivalds_eval" }

    fn log_size(&self) -> u32 {
        self.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> = FREIVALDS_SEL
            .iter()
            .map(|n| eval.get_preprocessed_column(freivalds_sel(&self.tag, n)))
            .collect();
        let (nofirst, last, notzero, end) =
            (sel[0].clone(), sel[1].clone(), sel[2].clone(), sel[3].clone());
        let ws: Vec<Vec<E::F>> = (0..self.n_mats)
            .map(|m| {
                (0..self.pk)
                    .map(|p| eval.get_preprocessed_column(freivalds_w(&self.tag, m, p)))
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
            let claim = eval.claim(self.claimed[m]);
            eval.add_constraint((acc - claim) * end.clone());
        }
        // stream II: the activation claim, allocated before the LogUp columns
        // so that the interaction order is bindings-then-fractions, as the wiring
        // family's components already are.  The value folded is `a - act_off`, the
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
            let claim = eval.claim(q.claimed);
            eval.add_constraint((x - claim) * end.clone());
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

/// The `r^T W` check: that a matmul's committed input times the pinned weights
/// is its committed accumulator.
pub struct Freivalds;

/// Which of the layer's matmul groups the check is over.
pub struct Cfg {
    pub group: String,
    /// the group's matmuls, as indices into the layer's own list
    pub mats: Vec<usize>,
    pub p_target: usize,
}

impl crate::gadgets::Gadget for Freivalds {
    type Witness = crate::layer::Layer;
    type Cfg = Cfg;
    /// What a Freivalds control corrupts is a weight or an activation of the
    /// layer it is handed, not a value beside it.
    type Tamper = ();
    type Outcome = run::Outcome;

    fn run(l: &crate::layer::Layer, cfg: &Cfg, _bad: Option<()>) -> run::Outcome {
        run::run(l, &cfg.group, &cfg.mats, cfg.p_target)
    }

    fn accepts(o: &run::Outcome) -> bool {
        o.accepts()
    }
}
