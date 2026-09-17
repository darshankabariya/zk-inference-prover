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
pub mod columns;
pub mod device;
pub mod layout;
pub mod run;
pub mod trace;

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, RelationEntry};

use crate::gadgets::freivalds::columns::{FreivaldsRow, FreivaldsShape};
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
        // #221: the columns come out of the one declaration in `columns.rs`, in
        // the order it declares them.  Nothing below names a column by position.
        let sh = FreivaldsShape::of(self);
        let FreivaldsRow { pre, base, inter } = FreivaldsRow::read(&mut eval, &sh);
        let act = base.act;

        // what the committed activation bytes carry on top of the signed value
        let off_sum: SecureField =
            self.s_pow.iter().fold(SecureField::zero(), |x, y| x + *y)
                * SecureField::from_u32_unchecked(self.act_off, 0, 0, 0);
        // v: this row's activations folded against s, on top of the previous
        // row's v shifted by s^pq -- away from a channel's first chunk
        eval.add_constraint(inter.v.cur.clone() - (inter.v.prev * self.s_step) * pre.nofirst.clone()
                            - (fold::<E>(&act, &self.s_pow) - off_sum));

        for (m, mat) in inter.mats.into_iter().enumerate() {
            // u_m: this row's weights folded against rv, on the same carry
            eval.add_constraint(mat.u.cur.clone() - (mat.u.prev * self.rv_step) * pre.nofirst.clone()
                                - fold::<E>(&pre.w[m], &self.rv_pow));
            // acc_m: u_m v added once per input channel, on the channel's last row
            eval.add_constraint(mat.acc.cur.clone() - mat.acc.prev * pre.notzero.clone()
                                - (mat.u.cur * inter.v.cur.clone()) * pre.last.clone());
            // acc_m closes on the claim the statement carries
            let claim = eval.claim(self.claimed[m]);
            eval.add_constraint((mat.acc.cur - claim) * pre.end.clone());
        }
        // stream II: the activation claim, allocated before the LogUp columns
        // so that the interaction order is bindings-then-fractions, as the wiring
        // family's components already are.  The value folded is `a - act_off`, the
        // signed activation, which is what a producer's requantize publishes.
        if let Some(q) = &self.qx {
            let x = inter.qx.expect("the row carries qx exactly when the AIR does");
            let off = E::F::from(M31::from_u32_unchecked(self.act_off));
            let mut xf = E::EF::zero();
            for (pp, a) in act.iter().enumerate() {
                xf += (E::EF::from(a.clone()) - E::EF::from(off.clone())) * q.w[pp];
            }
            let mult = E::EF::from(pre.nofirst.clone()) * q.m_in
                + (E::EF::from(pre.notzero.clone()) - E::EF::from(pre.nofirst.clone())) * q.m_bd;
            // the activation claim's own Horner run, at the group's two row steps
            eval.add_constraint(x.cur.clone() - x.prev * mult - xf);
            // and it closes on the claim the wiring family compares against
            let claim = eval.claim(q.claimed);
            eval.add_constraint((x.cur - claim) * pre.end.clone());
        }
        if self.range_check && !self.gkr {
            for a in &act {
                eval.add_to_relation(RelationEntry::new(&self.rel_act, E::EF::one(),
                                                        core::slice::from_ref(a)));
            }
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

/// `sum_p cells[p] * w[p]`: a row's cells folded against fixed challenge powers,
/// which is what every Horner term of this check is.
fn fold<E: EvalAtRow>(cells: &[E::F], w: &[SecureField]) -> E::EF {
    let mut f = E::EF::zero();
    for (p, c) in cells.iter().enumerate() {
        f += c.clone() * w[p];
    }
    f
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

/// The row-level tests are the SIMD backend's, as #221 asks: on the CUDA build
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` and `ZK_GENERATED_LOGUP` default to the
/// device, which has no `corrupt_qx` path and no host fraction builder, so a
/// component built here would be built by a different generator than the one
/// under test.  The digests still run on both backends.
#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use stwo::core::channel::Channel;
    use stwo::core::fields::m31::M31;
    use stwo::core::pcs::TreeVec;
    use stwo::prover::backend::Column;

    use super::*;
    use crate::bf16::Round;
    use crate::columns::constraint_values;
    use crate::common::Eval;
    use crate::gadgets::attention::witness::{Src, Variant};
    use crate::gadgets::freivalds::claims::freivalds_interaction;
    use crate::gadgets::freivalds::columns::{FreivaldsPre, FreivaldsShape, constraint_names};
    use crate::gadgets::freivalds::layout::FREIVALDS_GROUPS;
    use crate::gadgets::freivalds::trace::freivalds_component;
    use crate::layer::{Layer, Tables};
    use crate::relations::{RangeRel, powers};

    const ACT_BITS: u32 = 8;
    const P_TARGET: usize = 8;

    /// A synthetic layer small enough that a whole Freivalds group builds,
    /// closes its accumulators and is checked row by row in well under a second.
    fn small_layer() -> Layer {
        let src = Src { dir: None, head: 0, q_bits: 12, k_bits: 11, nmax: 255, log_table: 8,
                        seed: 7, layout: String::new(), diagonal_block: false, tile: false };
        Layer::synthetic(8, 8, 8, &[1], Variant::parse("intsplit6"), &src, 16, 16, 0x221)
    }

    /// One built component: the AIR, its three trees, and what the controls left
    /// outside the constraints.
    struct Built {
        /// the shape is `FreivaldsShape::of(&air)`, which borrows it
        air: FreivaldsEval,
        pre: Vec<Eval>,
        base: Vec<Eval>,
        inter: Vec<Eval>,
        use_sum: SecureField,
        off_table: usize,
    }

    /// Build one group exactly the way `run.rs` composes it, with the two
    /// controls `trace.rs` takes.
    fn build(group: &str, with_qx: bool, corrupt_qx: Option<usize>,
             oversize_qx: Option<usize>) -> Built {
        let l = small_layer();
        let names = FREIVALDS_GROUPS.iter().find(|(g, _)| *g == group)
            .unwrap_or_else(|| panic!("no Freivalds group {group}")).1;
        let mats: Vec<usize> = l.mats.iter().enumerate()
            .filter(|(_, m)| names.contains(&m.name.as_str())).map(|(i, _)| i).collect();
        let mut tables = Tables::new(vec![0u32; 16], 4, 0, Round::Even, 4, vec![ACT_BITS], 0, 0);
        let r = freivalds_component(&l, group, &mats, P_TARGET, ACT_BITS, &mut tables,
                                    corrupt_qx, oversize_qx);
        let lay = r.lay;
        // fixed challenges: what is under test is the constraints, not the transcript
        let mut ch = stwo::core::channel::Blake2sM31Channel::default();
        ch.mix_u64(0x221);
        let rel_act = RangeRel::draw(&mut ch);
        let f = ch.draw_secure_felts(5);
        let (rv, s_ch) = (f[0], f[1]);
        let (rv_pow, rv_step) = powers(rv, lay.pk);
        let (s_pow, s_step) = powers(s_ch, lay.pq);
        // the activation claim's weights and row steps are the wiring family's;
        // the constraints hold for any of them, so the test draws them
        let qx = |claimed| with_qx.then(|| FreivaldsQx {
            w: powers(f[2], lay.pq).0, m_in: f[3], m_bd: f[4], claimed,
        });
        let air = |claimed: Vec<SecureField>, qx| FreivaldsEval {
            tag: group.to_string(), g: lay.g, pk: lay.pk, pq: lay.pq, n_mats: mats.len(),
            rv_pow: rv_pow.clone(), rv_step, s_pow: s_pow.clone(), s_step, claimed,
            rel_act: rel_act.clone(), act_off: 1 << (ACT_BITS - 1), range_check: true,
            gkr: false, qx,
        };
        let rels = run::rels_of(&rel_act);
        let open = air(vec![SecureField::zero(); mats.len()], qx(SecureField::zero()));
        let it = freivalds_interaction(&l, &r, rv, s_ch, &rels, true, &open, false);
        // the claims the component publishes close the last two constraints
        let air = air(it.claimed.clone(), qx(it.qx_claim));
        Built {
            pre: r.pre.clone(),
            base: r.base.clone(),
            inter: it.cols.into_iter().map(|c| c.into_host()).collect(),
            use_sum: it.use_sum,
            off_table: tables.off_table,
            air,
        }
    }

    /// The (row, constraint name) pairs whose constraint does not hold.
    ///
    /// The preprocessed tree is handed over in the order the AIR *reads* it,
    /// which is the row schema's declaration order and is what a component's own
    /// mask is numbered by; the committed order is the schema's too, and this is
    /// the one place the two meet.
    fn broken(b: &Built) -> Vec<(usize, String)> {
        let sh = FreivaldsShape::of(&b.air);
        let committed = FreivaldsPre::committed_ids(&sh);
        assert_eq!(committed.len(), b.pre.len(), "the committed ids are not the tree");
        // A component's own preprocessed mask is numbered in the order it *reads*
        // its columns, and the tree is committed in another one, so the columns
        // are gathered here the way `FrameworkComponent` gathers them: by id.
        // The read order is the AIR's own, recorded rather than restated.
        let read = b.air
            .evaluate(stwo_constraint_framework::InfoEvaluator::new(
                b.air.log_size(), vec![], b.use_sum))
            .preprocessed_columns;
        let pre: Vec<Vec<M31>> = {
            let by_id: std::collections::HashMap<&str, &Eval> =
                committed.iter().map(|i| i.id.as_str()).zip(b.pre.iter()).collect();
            read.iter().map(|id| by_id[id.id.as_str()].values.to_cpu()).collect()
        };
        let base: Vec<Vec<M31>> = b.base.iter().map(|e| e.values.to_cpu()).collect();
        let inter: Vec<Vec<M31>> = b.inter.iter().map(|e| e.values.to_cpu()).collect();
        assert_eq!((pre.len(), base.len()), (sh.n_pre(), sh.n_base()),
                   "the trees are not the ones the row declares");
        let trace: TreeVec<Vec<&Vec<M31>>> = TreeVec::new(vec![
            pre.iter().collect(),
            vec![],
            base.iter().collect(),
            inter.iter().collect(),
        ]);
        let names = constraint_names(&sh);
        let mut bad = vec![];
        for (row, vals) in constraint_values(&b.air, &trace, b.use_sum).into_iter().enumerate() {
            assert_eq!(vals.len(), names.len(),
                       "the AIR adds {} constraints and the schema names {}",
                       vals.len(), names.len());
            for (v, n) in vals.iter().zip(&names) {
                if !v.is_zero() {
                    bad.push((row, n.clone()));
                }
            }
        }
        bad
    }

    /// A trace built through the schema's writers satisfies every constraint of
    /// the AIR that reads through the schema, on every row of the domain.
    ///
    /// This is the half the pinned digests cannot cover: a digest says the proof
    /// did not change, which is also what it says when the circuit was wrong
    /// before and is wrong in the same way now.  What fails here is a reader and
    /// a writer that disagree about a column's position, a selector true on the
    /// wrong rows, an accumulator carried across the wrong boundary, or a padded
    /// row the constraints do not hold on -- by name, and at the row.
    #[test]
    fn the_honest_trace_satisfies_every_constraint_on_every_row() {
        for group in ["o", "qkv", "gateup"] {
            for qx in [false, true] {
                let b = build(group, qx, None, None);
                assert_eq!(b.off_table, 0,
                           "{group}: an honest activation fell off the range table");
                let bad = broken(&b);
                assert!(bad.is_empty(), "{group} (qx={qx}): {} constraint(s) do not hold, \
                                         first {:?}", bad.len(), &bad[..bad.len().min(4)]);
            }
        }
    }

    /// And each control breaks the constraint it is written for and no other.
    ///
    /// `corrupt_qx` moves one *committed* activation by one and leaves the
    /// interaction's `v` on the honest witness, so the one equality that stops
    /// holding is `v_horner`, on the one row that carries that cell -- and
    /// `qx_horner` with it on a component that carries stream II's claim, which
    /// folds the same committed cells against different weights.
    ///
    /// `oversize_qx` is not a constraint control at all: it puts a committed
    /// activation one past its range table, which is the *table's* business.  The
    /// row sees it only as the same activation disagreeing with `v`, and what it
    /// really broke is the off-table counter.
    #[test]
    fn each_control_breaks_the_constraint_it_targets_and_nothing_else() {
        let idx = 5usize;
        for (control, off_table) in [("corrupt", 0usize), ("oversize", 1usize)] {
            for qx in [false, true] {
                let (c, o) = match control {
                    "corrupt" => (Some(idx), None),
                    _ => (None, Some(idx)),
                };
                let b = build("o", qx, c, o);
                let bad = broken(&b);
                let want: Vec<&str> = if qx { vec!["v_horner", "qx_horner"] }
                                      else { vec!["v_horner"] };
                assert_eq!(bad.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>(), want,
                           "{control}_qx (qx={qx}) broke {bad:?}");
                assert!(bad.windows(2).all(|w| w[0].0 == w[1].0),
                        "{control}_qx (qx={qx}) broke more than one row: {bad:?}");
                assert_eq!(b.off_table, off_table,
                           "{control}_qx (qx={qx}): off-table count");
            }
        }
    }
}
