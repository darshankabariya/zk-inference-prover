//! B11 stream S, lever 3: the link that makes the GKR arm sound.
//!
//! `gkrband` proves the bands' fractional sums, but its `GkrArtifact` ends in
//! *claimed evaluations of the input layers at a random point* -- nothing ties
//! those to the committed base columns. This module closes that gap along the
//! reduction of `experiments/b11/gkr/memo.md` section 4: with the input layer
//! laid out `(look, row)` and every look's numerator and denominator *affine*
//! in committed columns, an instance's input-layer claim collapses to
//!
//!     claim + (verifier scalar)  =  sum_c w_c * col_c~(x_row)
//!
//! -- one affine combination of committed-column MLEs at the row part of the
//! OOD point, with weights `w_c` the verifier computes from `ood_point` and
//! the relation elements alone. Instances of equal height share `x_row`
//! (staggered instances take the *last* coordinates of the batch OOD point),
//! so they fold into **one univariate `mle_eval` IOP per distinct height**
//! under one post-GKR challenge.
//!
//! The univariate IOP itself (eq-column + shifted-prefix-sum constraints, the
//! succinct eq carry quotients) is promoted from the fork's examples crate,
//! `nitroo/crates/examples/src/xor/gkr_lookups/mle_eval.rs`, function for
//! function. Two deviations from the upstream component, both forced by the
//! composed prover running its constraint evaluation through generated CUDA
//! kernels rather than upstream's SIMD-only `ComponentProver`:
//!
//! * the carry-quotient column and the MLE coefficient column are *committed*
//!   (16 M31 columns per height instead of upstream's 8), so the whole
//!   constraint set is an ordinary `FrameworkEval` (`LinkEval`) that rides the
//!   existing kernel machinery;
//! * what upstream's `MleCoeffColumnOracle` did at the OODS point -- read the
//!   coefficient polynomial off the committed masks -- becomes two explicit
//!   *tie* constraints per height in a bespoke component (`LinkTie`): the
//!   committed coefficient column equals the `w_c` combination of the base
//!   columns' own OODS values, and the committed carry-quotient column equals
//!   its succinct analytic evaluation. Both are identically zero for an
//!   honest prover, so `LinkTie`'s prover side contributes zero quotients and
//!   only consumes its accumulator coefficients.

use std::collections::BTreeMap;

use num_traits::{One, Zero};
use stwo::core::circle::{CirclePoint, Coset};
use stwo::core::constraints::{coset_vanishing, point_vanishing};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use stwo::core::ColumnVec;
use stwo::core::air::Component;
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::prover::{DomainEvaluationAccumulator, Trace};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::Column;
use stwo::prover::lookups::gkr_prover::GkrOps;
use stwo::prover::lookups::gkr_verifier::GkrArtifact;
use stwo::prover::lookups::utils::eq;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX};

use crate::attn_witness::Variant as AttnVariant;
use crate::common::Eval;
use crate::gkrband::Inst;

// ------------------------------------------------------------------------
// The univariate IOP, promoted from the fork's examples crate
// (`xor/gkr_lookups/mle_eval.rs`). The math is unchanged; only visibility
// and the generic bounds needed by this crate's evaluators differ.
// ------------------------------------------------------------------------

/// Metadata of one MLE evaluation point (upstream `MleEvalPoint`).
#[derive(Debug, Clone)]
pub struct MleEvalPoint {
    /// `eq({0}^n, p)`
    eq_0_p: SecureField,
    /// `eq({1}^n, p)`
    eq_1_p: SecureField,
    /// index `i` stores `eq(({1}^i, 0), p[0..i+1]) / eq(({0}^i, 1), p[0..i+1])`
    eq_carry_quotients: Vec<SecureField>,
    p: Vec<SecureField>,
}

impl MleEvalPoint {
    /// Panics if the point is empty or has a coordinate that is zero or one
    /// (upstream's completeness caveat; the GKR's OOD point is drawn from the
    /// channel, so hitting one is negligible and loud).
    pub fn new(p: &[SecureField]) -> Self {
        assert!(!p.is_empty());
        let n_variables = p.len();
        let zero = SecureField::zero();
        let one = SecureField::one();
        Self {
            eq_0_p: eq(&vec![zero; n_variables], p),
            eq_1_p: eq(&vec![one; n_variables], p),
            eq_carry_quotients: (0..n_variables)
                .map(|i| {
                    let mut num = vec![one; i + 1];
                    num[i] = zero;
                    let mut den = vec![zero; i + 1];
                    den[i] = one;
                    eq(&num, &p[..i + 1]) / eq(&den, &p[..i + 1])
                })
                .collect(),
            p: p.to_vec(),
        }
    }

    pub fn n_variables(&self) -> usize {
        self.p.len()
    }
}

/// Evaluates the EqEvals-column constraints; returns the eq column at offset 0
/// (upstream `eval_eq_constraints`).
fn eval_eq_constraints<E: EvalAtRow>(
    eq_interaction: usize,
    eval: &mut E,
    mle_eval_point: &MleEvalPoint,
    carry_quotients_col_eval: E::EF,
    is_first: E::F,
    is_second: E::F,
) -> E::EF {
    let [curr, next_next] = eval.next_extension_interaction_mask(eq_interaction, [0, 2]);

    // initial value on half_coset0 and final value on half_coset1; `is_first`
    // and `is_second` are never nonzero at the same row
    let half_coset0_initial_check = (curr.clone() - mle_eval_point.eq_0_p) * is_first;
    let half_coset1_final_check = (curr.clone() - mle_eval_point.eq_1_p) * is_second;
    eval.add_constraint(half_coset0_initial_check + half_coset1_final_check);

    // all the steps
    eval.add_constraint(curr.clone() - next_next * carry_quotients_col_eval);

    curr
}

/// Inclusive-prefix-sum constraints over a column whose values are shifted by
/// `cumulative_sum_shift = cumulative_sum / column_size` (upstream
/// `eval_prefix_sum_constraints`).
fn eval_prefix_sum_constraints<E: EvalAtRow>(
    interaction: usize,
    eval: &mut E,
    row_diff: E::EF,
    cumulative_sum_shift: SecureField,
) {
    let [curr, prev] = eval.next_extension_interaction_mask(interaction, [0, -1]);
    eval.add_constraint(curr - prev - row_diff + cumulative_sum_shift);
}

/// `(half_coset0_carry_quotients, half_coset1_carry_quotients)` (upstream).
fn gen_half_coset_carry_quotients(
    eval_point: &MleEvalPoint,
) -> (Vec<SecureField>, Vec<SecureField>) {
    let last_variable = *eval_point.p.last().unwrap();
    let one = SecureField::one();
    let zero = SecureField::zero();
    let mut half_coset0_carry_quotients = eval_point.eq_carry_quotients.clone();
    *half_coset0_carry_quotients.last_mut().unwrap() *=
        eq(&[one], &[last_variable]) / eq(&[zero], &[last_variable]);
    let half_coset1_carry_quotients =
        half_coset0_carry_quotients.iter().map(|v| v.inverse()).collect();
    (half_coset0_carry_quotients, half_coset1_carry_quotients)
}

/// The succinct eq carry-quotients column as committed values (upstream
/// `gen_carry_quotient_col`).
fn gen_carry_quotient_col(eval_point: &MleEvalPoint) -> SecureColumnByCoords<SimdBackend> {
    let (half0, half1) = gen_half_coset_carry_quotients(eval_point);

    let log_size = eval_point.n_variables() as u32;
    let size = 1usize << log_size;
    let half_coset_size = size / 2;
    let mut col = SecureColumnByCoords::<SimdBackend>::zeros(size);

    for i in 0..half_coset_size {
        let half_coset0_index = coset_index_to_circle_domain_index(i * 2, log_size);
        let half_coset1_index = coset_index_to_circle_domain_index(i * 2 + 1, log_size);
        let half_coset0_index_bit_rev = bit_reverse_index(half_coset0_index, log_size);
        let half_coset1_index_bit_rev = bit_reverse_index(half_coset1_index, log_size);

        let n_trailing_ones = i.trailing_ones() as usize;
        col.set(half_coset0_index_bit_rev, half0[n_trailing_ones]);
        col.set(half_coset1_index_bit_rev, half1[n_trailing_ones]);
    }
    col
}

/// Evaluates the succinct carry-quotients column at `p` (upstream
/// `eval_carry_quotient_col`).
pub fn eval_carry_quotient_col(eval_point: &MleEvalPoint, p: CirclePoint<SecureField>)
                               -> SecureField {
    let n_variables = eval_point.n_variables();
    let log_size = n_variables as u32;
    let coset = CanonicCoset::new(log_size).coset();

    let (half0, half1) = gen_half_coset_carry_quotients(eval_point);

    let mut out = SecureField::zero();
    for variable_i in 0..n_variables.saturating_sub(1) {
        let log_step = variable_i as u32 + 2;
        let offset = (1 << (log_step - 1)) - 2;
        let half_coset0_selector = eval_step_selector_with_offset(coset, offset, log_step, p);
        let half_coset1_selector = eval_step_selector_with_offset(coset, offset + 1, log_step, p);
        out += half_coset0_selector * half0[variable_i];
        out += half_coset1_selector * half1[variable_i];
    }
    let half_coset0_last = eval_is_first(coset, p + coset.step.double().into_ef());
    let half_coset1_first = eval_is_first(coset, p + coset.step.into_ef());
    out += *half0.last().unwrap() * half_coset0_last;
    out += *half1.last().unwrap() * half_coset1_first;
    out
}

/// A polynomial that is `1` every `2^log_step` coset points shifted by
/// `offset` and `0` elsewhere on the coset (upstream).
fn eval_step_selector_with_offset(
    coset: Coset,
    offset: usize,
    log_step: u32,
    p: CirclePoint<SecureField>,
) -> SecureField {
    let offset_step = coset.step.mul(offset as u128);
    eval_step_selector(coset, log_step, p - offset_step.into_ef())
}

/// A polynomial that is `1` every `2^log_step` coset points and `0` elsewhere
/// on the coset (upstream).
fn eval_step_selector(coset: Coset, log_step: u32, p: CirclePoint<SecureField>) -> SecureField {
    if log_step == 0 {
        return SecureField::one();
    }
    // rotate the coset so its first point is the identity element
    let p = p - coset.initial.into_ef();
    let mut vanish_at_log_step: Vec<SecureField> = (0..coset.log_size)
        .scan(p, |p, _| {
            let res = *p;
            *p = p.double();
            Some(res.y)
        })
        .collect();
    vanish_at_log_step.reverse();
    // only the first `log_step` many values are needed
    vanish_at_log_step.truncate(log_step as usize);
    let vanish_at_log_step_inv = SecureField::batch_inverse(&vanish_at_log_step);

    let half_coset_selector_dbl = (vanish_at_log_step[0] * vanish_at_log_step_inv[1]).square();
    let vanish_substep_inv_sum = vanish_at_log_step_inv[1..].iter().sum::<SecureField>();
    (half_coset_selector_dbl + vanish_at_log_step[0] * vanish_substep_inv_sum.double())
        / BaseField::from(1 << (log_step + 1))
}

/// `1` at the first coset point, `0` elsewhere, as a polynomial evaluation
/// (upstream `eval_is_first`).
pub fn eval_is_first(coset: Coset, p: CirclePoint<SecureField>) -> SecureField {
    coset_vanishing(coset, p)
        / (point_vanishing(coset.initial, p) * BaseField::from(1 << coset.log_size))
}

// ------------------------------------------------------------------------
// The affine looks: every band lookup's numerator and denominator as an
// affine form over committed columns, in mask-relative coordinates.
// ------------------------------------------------------------------------

/// One committed (or preprocessed) column, named the way the AIR reads it:
/// by its mask index within the region's component. The committed order and
/// the mask order are the same order -- that is what lets the prover resolve
/// these against `com.base[at[which] + i]` and the verifier against the
/// component's trace location span, and the two cannot disagree without the
/// tie constraint rejecting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RCol {
    /// the channel region's `i`-th trace mask (`ks` then `qs`)
    Chan(usize),
    /// the grid region's `i`-th trace mask (`idx`, `num`, `aux`, `vals`)
    Key(usize),
    /// the per-token region's `i`-th trace mask (`ms`, `zs`, `los`, `los2`)
    Row(usize),
    /// the corner's `i`-th causal-mask column, preprocessed tree
    Msk(usize),
}

/// An affine form `sum coeff * col + k` over one band's columns.
#[derive(Clone, Debug, Default)]
pub struct Affine {
    pub terms: Vec<(RCol, SecureField)>,
    pub k: SecureField,
}

impl Affine {
    fn col(c: RCol, w: SecureField) -> Self {
        Affine { terms: vec![(c, w)], k: SecureField::zero() }
    }
}

/// One lookup: its numerator (`None` = the constant one) and denominator.
#[derive(Clone, Debug)]
pub struct LinkLook {
    pub num: Option<Affine>,
    pub den: Affine,
}

/// One site's looks, in the exact order `interaction_phase` pushes them.
pub struct SiteLooks {
    pub band: usize,
    pub which: usize,
    pub log_rows: u32,
    pub looks: Vec<LinkLook>,
}

fn range_den(rel: &crate::attn_air::RangeRel, slot: Affine) -> Affine {
    let (z, al) = rel.parts();
    let mut den = Affine {
        terms: slot.terms.iter().map(|(c, w)| (*c, *w * al[0])).collect(),
        k: slot.k * al[0] - z,
    };
    den.terms.retain(|(_, w)| !w.is_zero());
    den
}

/// The bands' lookup sites as affine forms, mirroring `gkrband::sites_of`'s
/// site order and `attn_trace::interaction_phase`'s look order exactly. This
/// is the verifier's half of "one definition of the fraction": the collector
/// reads the committed columns, this names them, and the tie constraint is
/// what makes the two the same function.
pub fn link_looks(plan: &crate::compose::Plan, rels: &crate::compose::Rels) -> Vec<SiteLooks> {
    let p = &plan.p;
    let cfg = &plan.cfg;
    let mut out = vec![];
    for (b, corner) in plan.corner.iter().enumerate() {
        let l = &plan.lays[b];
        let (lc, lk) = (&l.chan, &l.key);

        // ---- channel region: the key and query range checks, all unit
        if cfg.range_check {
            let mut looks = vec![];
            for pp in 0..lc.pk {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(p.k_bits),
                                   Affine::col(RCol::Chan(pp), SecureField::one())),
                });
            }
            for pp in 0..lc.pq {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(p.q_bits),
                                   Affine::col(RCol::Chan(lc.pk + pp), SecureField::one())),
                });
            }
            out.push(SiteLooks { band: b, which: 0, log_rows: lc.g.log_rows, looks });
        }

        // ---- grid region: exp lookups, the variant's remainder checks, the
        // V range checks
        let n_aux = match plan.attn_variant {
            AttnVariant::IntExp => 0,
            _ => lk.q,
        };
        let mut looks = vec![];
        let (ez, eal) = rels.exp.parts();
        for i in 0..lk.q {
            let key = match plan.attn_variant {
                AttnVariant::Bf16x => RCol::Key(2 * lk.q + i),
                _ => RCol::Key(i),
            };
            let den = Affine {
                terms: vec![(key, eal[0]), (RCol::Key(lk.q + i), eal[1])],
                k: -ez,
            };
            let num = corner.then(|| Affine {
                terms: vec![(RCol::Msk(i), -SecureField::one())],
                k: SecureField::one(),
            });
            looks.push(LinkLook { num, den });
        }
        if let AttnVariant::IntSplit(kk) = plan.attn_variant {
            for i in 0..lk.q {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(kk),
                                   Affine::col(RCol::Key(2 * lk.q + i), SecureField::one())),
                });
            }
        }
        if cfg.range_check {
            for pp in 0..lk.pv {
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(8),
                                   Affine::col(RCol::Key(2 * lk.q + n_aux + pp),
                                               SecureField::one())),
                });
            }
        }
        out.push(SiteLooks { band: b, which: 1, log_rows: lk.g.log_rows, looks });

        // ---- per-token region: the two denominator bounds' limbs. The high
        // limbs are not committed -- `hi = (Z - lo) * 2^-z_lo_bits` -- so they
        // fold into weights on the committed `zs` and `los` columns, which is
        // memo section 4's first consequence.
        if !*corner {
            let with_max = plan.attn_variant != AttnVariant::Bf16x;
            let ms = if with_max { l.rw } else { 0 };
            let inv = SecureField::from(M31::from_u32_unchecked(1 << cfg.z_lo_bits).inverse());
            let zmin = SecureField::from(M31::from_u32_unchecked(cfg.z_min));
            let mut looks = vec![];
            for i in 0..l.rw {
                let zs = RCol::Row(ms + i);
                let lo = RCol::Row(ms + l.rw + i);
                let lo2 = RCol::Row(ms + 2 * l.rw + i);
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_lo_bits),
                                   Affine::col(lo, SecureField::one())),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_hi_bits), Affine {
                        terms: vec![(zs, inv), (lo, -inv)],
                        k: SecureField::zero(),
                    }),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_lo_bits),
                                   Affine::col(lo2, SecureField::one())),
                });
                looks.push(LinkLook {
                    num: None,
                    den: range_den(rels.range(cfg.z_hi_bits), Affine {
                        terms: vec![(zs, inv), (lo2, -inv)],
                        k: -zmin * inv,
                    }),
                });
            }
            out.push(SiteLooks { band: b, which: 3, log_rows: l.log_row, looks });
        }
    }
    out
}

// ------------------------------------------------------------------------
// Per-height parameters: one folded mle_eval instance per distinct region
// height, derived identically by the prover and the verifier.
// ------------------------------------------------------------------------

/// One folded univariate IOP: everything both sides need about one height.
pub struct HeightLink {
    pub h: u32,
    /// the row part of the OOD point: the last `h` coordinates
    pub point: Vec<SecureField>,
    pub mle_point: MleEvalPoint,
    /// the folded claim, and the shift `claim / 2^h` the prefix column carries
    pub claim: SecureField,
    pub shift: SecureField,
    /// folded weights per referenced column, and the folded constant
    pub weights: Vec<(usize, RCol, SecureField)>,
    pub konst: SecureField,
}

/// eq weights over one instance's looks: position `p`'s bits pair MSB-first
/// with the look part of the instance's OOD point.
fn eq_weights(k: usize, x_look: &[SecureField]) -> Vec<SecureField> {
    assert_eq!(k, x_look.len());
    let mut out = vec![SecureField::one()];
    for x in x_look {
        // prepend the next variable: existing entries take bit 0, new take 1
        let mut next = Vec::with_capacity(out.len() * 2);
        for w in &out {
            next.push(*w * (SecureField::one() - *x));
        }
        for w in &out {
            next.push(*w * *x);
        }
        out = next;
    }
    out
}

/// Derive the per-height fold from the plan, the drawn relations, the GKR
/// artifact and the post-GKR challenge. The verifier calls this to know what
/// the tie constraints assert; the prover calls it to build the very columns
/// those constraints check. `Err` is a rejection, never a panic: the artifact
/// is proof-derived data.
pub fn link_params(
    plan: &crate::compose::Plan,
    rels: &crate::compose::Rels,
    insts: &[Inst],
    artifact: &GkrArtifact,
    alpha: SecureField,
) -> Result<Vec<HeightLink>, String> {
    let looks = link_looks(plan, rels);
    let el = artifact.ood_point.len();
    if artifact.claims_to_verify_by_instance.len() != insts.len() {
        return Err(format!("the GKR artifact carries {} instances, the plan says {}",
                           artifact.claims_to_verify_by_instance.len(), insts.len()));
    }

    struct Acc {
        point: Vec<SecureField>,
        claim: SecureField,
        weights: BTreeMap<(usize, RCol), SecureField>,
        konst: SecureField,
        n: usize,
    }
    let mut by_h: BTreeMap<u32, Acc> = BTreeMap::new();

    for (i, inst) in insts.iter().enumerate() {
        let n_i = inst.n_variables();
        if el < n_i {
            return Err("the GKR OOD point is shorter than an instance".into());
        }
        let x_i = &artifact.ood_point[el - n_i..];
        let k_look = inst.ents.len().ilog2() as usize;
        let (x_look, x_row) = x_i.split_at(k_look);
        let eqw = eq_weights(k_look, x_look);
        let site = &looks[inst.site];
        let h = inst.log_rows;
        debug_assert_eq!(site.log_rows, h);

        let claims = &artifact.claims_to_verify_by_instance[i];
        if claims.len() != 2 {
            return Err(format!("instance {i} carries {} claims, not 2", claims.len()));
        }
        let (num_claim, den_claim) = (claims[0], claims[1]);

        let acc = by_h.entry(h).or_insert_with(|| Acc {
            point: x_row.to_vec(),
            claim: SecureField::zero(),
            weights: BTreeMap::new(),
            konst: SecureField::zero(),
            n: 0,
        });
        debug_assert_eq!(acc.point.as_slice(), x_row);

        fn push(acc: &mut Acc, alpha: SecureField, band: usize, claim: SecureField,
                parts: &mut dyn Iterator<Item = (SecureField, Affine)>) {
            let a = alpha.pow(acc.n as u128);
            acc.n += 1;
            acc.claim += a * claim;
            for (w, aff) in parts {
                for (c, cw) in &aff.terms {
                    *acc.weights.entry((band, *c)).or_insert(SecureField::zero()) +=
                        a * w * *cw;
                }
                acc.konst += a * w * aff.k;
            }
        }

        if inst.unit {
            // a singles instance's numerator MLE is the constant one, so its
            // claim must be exactly one -- checked here, no column involved
            if num_claim != SecureField::one() {
                return Err(format!("instance {i} claims a unit numerator of {num_claim:?}"));
            }
        } else {
            let mut parts = inst.ents.iter().zip(&eqw).map(|(l, w)| {
                (*w, site.looks[*l].num.clone().unwrap_or(Affine {
                    terms: vec![],
                    k: SecureField::one(),
                }))
            });
            push(acc, alpha, site.band, num_claim, &mut parts);
        }
        let mut parts = inst.ents.iter().zip(&eqw)
            .map(|(l, w)| (*w, site.looks[*l].den.clone()));
        push(acc, alpha, site.band, den_claim, &mut parts);
    }

    Ok(by_h
        .into_iter()
        .map(|(h, acc)| {
            let shift = acc.claim / BaseField::from(1u32 << h);
            HeightLink {
                h,
                mle_point: MleEvalPoint::new(&acc.point),
                point: acc.point,
                claim: acc.claim,
                shift,
                weights: acc.weights.into_iter().map(|((b, c), w)| (b, c, w)).collect(),
                konst: acc.konst,
            }
        })
        .collect())
}

/// The distinct link heights of a plan, ascending -- what decides how many
/// `LinkEval` components a statement has before any proof exists (the emit
/// path needs exactly this).
pub fn link_heights(plan: &crate::compose::Plan) -> Vec<u32> {
    let mut hs: Vec<u32> = crate::gkrband::sites_of(plan).iter().map(|s| s.log_rows).collect();
    hs.sort_unstable();
    hs.dedup();
    hs
}

pub fn first_id(h: u32) -> PreProcessedColumnId {
    PreProcessedColumnId { id: format!("s_link_first_{h}") }
}
pub fn second_id(h: u32) -> PreProcessedColumnId {
    PreProcessedColumnId { id: format!("s_link_second_{h}") }
}

/// The two boundary selectors of one height: `1` at the first and second
/// coset rows. Committed to the preprocessed tree only under the flag.
pub fn boundary_cols(h: u32) -> [Eval; 2] {
    let domain = CanonicCoset::new(h).circle_domain();
    let size = 1usize << h;
    [0usize, 1].map(|coset_i| {
        let mut packed = stwo::prover::backend::simd::column::BaseColumn::zeros(size);
        packed.set(bit_reverse_index(coset_index_to_circle_domain_index(coset_i, h), h),
                   M31::one());
        CircleEvaluation::new(domain, packed)
    })
}

// ------------------------------------------------------------------------
// The prover's columns: per height, [q, coeff, eq, prefix] -- 16 M31 columns
// -- in the exact order `LinkEval::evaluate` claims its masks.
// ------------------------------------------------------------------------

fn coords_to_evals(h: u32, c: SecureColumnByCoords<SimdBackend>) -> Vec<Eval> {
    let domain = CanonicCoset::new(h).circle_domain();
    c.columns.into_iter().map(|col| CircleEvaluation::new(domain, col)).collect()
}

/// Build one height's committed columns. `col_of` resolves a referenced
/// column against this chunk's committed data (the prover's ground truth);
/// the tie constraints are what force it to agree with the verifier's own
/// resolution through the component spans.
pub fn build_height_cols<'a>(hl: &HeightLink,
                             col_of: &dyn Fn(usize, &RCol) -> &'a Eval) -> Vec<Eval> {
    let h = hl.h;
    assert!(h >= LOG_N_LANES, "a link height below the SIMD lane count");
    let n_vec = 1usize << (h - LOG_N_LANES);

    // the coefficient column: the folded affine combination, pointwise
    let mut coeff = vec![PackedSecureField::broadcast(hl.konst); n_vec];
    for (band, c, w) in &hl.weights {
        let e: &Eval = col_of(*band, c);
        assert_eq!(e.len(), 1usize << h, "column height disagrees with the fold");
        let w = PackedSecureField::broadcast(*w);
        for (vr, slot) in coeff.iter_mut().enumerate() {
            *slot += w * e.data[vr];
        }
    }
    let coeff_col = SecureColumn { data: coeff, length: 1 << h };

    // the eq column at the row point, and the shifted prefix sum of
    // coeff * eq (upstream `build_trace`, function for function)
    let eq_mle = SimdBackend::gen_eq_evals(&hl.point, SecureField::one());
    assert_eq!(eq_mle.len(), 1usize << h);
    let mut terms: Vec<PackedSecureField> = coeff_col
        .data
        .iter()
        .zip(&eq_mle.data)
        .map(|(a, b)| *a * *b)
        .collect();
    // the prefix-sum boundary is the claim itself: assert it here, loudly,
    // so a weight bug fails the prover and not just the verifier
    let total: SecureField = terms
        .iter()
        .map(|p| p.to_array().iter().fold(SecureField::zero(), |a, b| a + *b))
        .fold(SecureField::zero(), |a, b| a + b);
    assert_eq!(total, hl.claim,
               "the folded coefficient column disagrees with the GKR claim at height {h}");
    let shift = PackedSecureField::broadcast(hl.shift);
    for t in terms.iter_mut() {
        *t -= shift;
    }
    let shifted = SecureColumn { data: terms, length: 1 << h };
    let prefix_coords = shifted
        .into_secure_column_by_coords()
        .columns
        .map(inclusive_prefix_sum);
    let prefix = SecureColumnByCoords { columns: prefix_coords };

    let q = gen_carry_quotient_col(&hl.mle_point);

    let mut out = coords_to_evals(h, q);
    out.extend(coords_to_evals(h, coeff_col.into_secure_column_by_coords()));
    out.extend(coords_to_evals(h, eq_mle.into_evals().into_secure_column_by_coords()));
    out.extend(coords_to_evals(h, prefix));
    out
}

// ------------------------------------------------------------------------
// The framework component: one per height, carrying the eq and prefix-sum
// constraints over the committed [q, coeff, eq, prefix] columns.
// ------------------------------------------------------------------------

#[derive(Clone)]
pub struct LinkEval {
    pub h: u32,
    pub mle_point: MleEvalPoint,
    pub shift: SecureField,
}

impl FrameworkEval for LinkEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    fn cuda_eval_name(&self) -> &'static str { "b65_link_eval" }

    fn log_size(&self) -> u32 {
        self.h
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.h + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // mask claim order defines the committed column order: q, coeff, eq,
        // prefix -- exactly what `build_height_cols` emits
        let [q_cur] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [coeff_cur] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let is_first = eval.get_preprocessed_column(first_id(self.h));
        let is_second = eval.get_preprocessed_column(second_id(self.h));
        let eq_cur = eval_eq_constraints(INTERACTION_TRACE_IDX, &mut eval, &self.mle_point,
                                         q_cur, is_first, is_second);
        let terms = coeff_cur * eq_cur;
        eval_prefix_sum_constraints(INTERACTION_TRACE_IDX, &mut eval, terms, self.shift);
        eval
    }
}

// ------------------------------------------------------------------------
// The tie component: the two OODS equalities per height that upstream's
// oracle and succinct column performed implicitly. Identically zero for an
// honest prover, so the prover side only consumes its accumulator slots.
// ------------------------------------------------------------------------

pub struct TieHeight {
    pub h: u32,
    pub mle_point: MleEvalPoint,
    pub konst: SecureField,
    /// the affine combination's committed side: `(tree, global column, weight)`
    pub terms: Vec<(usize, usize, SecureField)>,
    /// first global tree-2 column of the committed carry-quotient quadruple
    pub q_at: usize,
    /// first global tree-2 column of the committed coefficient quadruple
    pub coeff_at: usize,
}

pub struct LinkTie {
    pub heights: Vec<TieHeight>,
    /// preprocessed columns the ties read (the corner masks), globally indexed
    pub pre_idx: Vec<usize>,
}

impl Component for LinkTie {
    fn n_constraints(&self) -> usize {
        2 * self.heights.len()
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.heights.iter().map(|t| t.h + 1).max().unwrap_or(1)
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::default()
    }

    fn mask_points(
        &self,
        _point: CirclePoint<SecureField>,
        _max_log_degree_bound: u32,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::default()
    }

    fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
        self.pre_idx.clone()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        max_log_degree_bound: u32,
    ) {
        let denom_inverse =
            coset_vanishing(CanonicCoset::new(max_log_degree_bound).coset, point).inverse();
        let ext = |at: usize| {
            SecureField::from_partial_evals([
                mask[INTERACTION_TRACE_IDX][at][0],
                mask[INTERACTION_TRACE_IDX][at + 1][0],
                mask[INTERACTION_TRACE_IDX][at + 2][0],
                mask[INTERACTION_TRACE_IDX][at + 3][0],
            ])
        };
        for t in &self.heights {
            // the committed coefficient column is the `w_c` combination of
            // the base columns -- as polynomials, checked at the OODS point
            let mut tie_coeff = ext(t.coeff_at) - t.konst;
            for (tree, col, w) in &t.terms {
                tie_coeff -= *w * mask[*tree][*col][0];
            }
            evaluation_accumulator.accumulate(denom_inverse * tie_coeff);
            // the committed carry-quotient column is the succinct one
            let tie_q = ext(t.q_at) - eval_carry_quotient_col(&t.mle_point, point);
            evaluation_accumulator.accumulate(denom_inverse * tie_q);
        }
    }
}

impl<B: stwo::prover::backend::Backend> ComponentProver<B> for LinkTie {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        _trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    ) {
        // an honest prover's tie constraints are identically zero as
        // polynomials, so their quotient contribution is the zero polynomial:
        // consume the accumulator coefficients and add nothing
        evaluation_accumulator.skip_coeffs(self.n_constraints());
    }
}

/// Placeholder link parameters for the emit path: the heights are the plan's
/// own, the scalars distinct nonzero pseudo-random values. The blob replays
/// the real scalars at prove time; distinctness keeps `lit` from folding
/// slots away, exactly as `fill_tie_claims` does for the glue ties.
pub fn link_for(plan: &crate::compose::Plan, seed: u64) -> Vec<HeightLink> {
    let mut x = seed | 1;
    let mut felt = || {
        let mut a = [0u32; 4];
        for l in a.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *l = ((x % 0x7fff_fffe) + 1) as u32;
        }
        SecureField::from_m31_array(a.map(M31::from_u32_unchecked))
    };
    link_heights(plan)
        .into_iter()
        .map(|h| {
            let point: Vec<SecureField> = (0..h).map(|_| felt()).collect();
            let claim = felt();
            HeightLink {
                h,
                mle_point: MleEvalPoint::new(&point),
                point,
                claim,
                shift: claim / BaseField::from(1u32 << h),
                weights: vec![],
                konst: felt(),
            }
        })
        .collect()
}
