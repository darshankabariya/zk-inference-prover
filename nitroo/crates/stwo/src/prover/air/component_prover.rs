use dashmap::DashMap;
use itertools::Itertools;

use crate::core::air::{Component, Components};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CircleDomain;
use crate::core::ColumnVec;
use crate::prover::air::accumulation::DomainEvaluationAccumulator;
use crate::prover::backend::{Backend, Col};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureCirclePoly};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::CirclePoint;

/// Type alias for the weights hash map used in barycentric eval_at_point.
pub type WeightsHashMap<B> = DashMap<(u32, CirclePoint<SecureField>), Col<B, SecureField>>;

pub trait ComponentProver<B: Backend>: Component {
    /// Evaluates the constraint quotients of the component on the evaluation domain.
    /// Accumulates quotients in `evaluation_accumulator`.
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, B>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<B>,
    );
}

/// The set of polynomials that make up the trace.
pub struct Trace<'a, B: Backend> {
    /// Polynomials for each column.
    pub polys: TreeVec<ColumnVec<&'a Poly<B>>>,
}

/// A struct for representing a polynomial corresponding to a trace column.
/// A polynomial is defined by it's evaluations on a circle domain of size at least it's degree,
/// and optionally its coefficients in the FFT basis.
///
/// B12 stream HH: the evaluations are optional. Under `lde_only` the commitment
/// scheme drops them once the tree's root has been mixed into the channel and
/// keeps the coefficients alone; every later phase rebuilds what it needs
/// through [`Poly::get_evaluation_on_domain`] and frees it again. The blown-up
/// domain is a field of its own because it is the one part of the evaluations
/// that has to outlive them: most readers of `evals` only ever wanted its
/// log size.
#[derive(Clone)]
pub struct Poly<B: Backend> {
    pub coeffs: Option<CircleCoefficients<B>>,
    pub evals: Option<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    /// The domain the evaluations are, or were, on.
    pub domain: CircleDomain,
}

impl<B: Backend> Poly<B> {
    pub const fn new(
        coeffs: Option<CircleCoefficients<B>>,
        evals: CircleEvaluation<B, BaseField, BitReversedOrder>,
    ) -> Self {
        Self {
            coeffs,
            domain: evals.domain,
            evals: Some(evals),
        }
    }

    /// A polynomial whose evaluations may already be absent.
    pub const fn from_parts(
        coeffs: Option<CircleCoefficients<B>>,
        evals: Option<CircleEvaluation<B, BaseField, BitReversedOrder>>,
        domain: CircleDomain,
    ) -> Self {
        Self {
            coeffs,
            evals,
            domain,
        }
    }

    /// The log size of the blown-up domain, held or dropped.
    pub const fn log_size(&self) -> u32 {
        self.domain.log_size()
    }

    /// The blown-up evaluations. Panics once they have been dropped, which is
    /// only under `lde_only`, where every reader rebuilds them first.
    pub fn evals(&self) -> &CircleEvaluation<B, BaseField, BitReversedOrder> {
        self.evals
            .as_ref()
            .expect("the polynomial's evaluations were dropped (lde_only)")
    }

    /// Frees the blown-up evaluations, keeping the coefficients and the domain.
    pub fn drop_evals(&mut self) {
        assert!(
            self.coeffs.is_some(),
            "lde_only needs the coefficients to rebuild the evaluations"
        );
        self.evals = None;
    }

    pub fn eval_at_point(
        &self,
        point: CirclePoint<SecureField>,
        weights_hash_map: Option<&WeightsHashMap<B>>,
    ) -> SecureField {
        if let Some(coeffs) = &self.coeffs {
            coeffs.eval_at_point(point)
        } else {
            self.evals().barycentric_eval_at_point(
                &weights_hash_map
                    .unwrap()
                    .get(&(self.log_size(), point))
                    .expect("weights should exist for all sampled points"),
            )
        }
    }

    pub fn get_evaluation_on_domain(
        &self,
        domain: CircleDomain,
        twiddles: &TwiddleTree<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        if let Some(coeffs) = &self.coeffs {
            coeffs.evaluate_with_twiddles(domain, twiddles)
        } else {
            panic!("The polynomial's coefficients are not stored");
        }
    }

    /// Rebuilds the blown-up evaluations on the polynomial's own committed
    /// domain. The copy is owned by the caller and freed when it drops; this
    /// is what the phases under `lde_only` call in place of reading `evals`.
    pub fn rebuild_evals(
        &self,
        twiddles: &TwiddleTree<B>,
    ) -> CircleEvaluation<B, BaseField, BitReversedOrder> {
        self.get_evaluation_on_domain(self.domain, twiddles)
    }
}

pub struct ComponentProvers<'a, B: Backend> {
    pub components: Vec<&'a dyn ComponentProver<B>>,
    pub n_preprocessed_columns: usize,
}

impl<B: Backend> ComponentProvers<'_, B> {
    pub fn components(&self) -> Components<'_> {
        Components {
            components: self
                .components
                .iter()
                .map(|c| *c as &dyn Component)
                .collect_vec(),
            n_preprocessed_columns: self.n_preprocessed_columns,
        }
    }
    pub fn compute_composition_polynomial(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
    ) -> SecureCirclePoly<B> {
        let total_constraints: usize = self.components.iter().map(|c| c.n_constraints()).sum();
        let mut accumulator = DomainEvaluationAccumulator::new(
            random_coeff,
            self.components().composition_log_degree_bound(),
            total_constraints,
        );
        for component in self.components.iter() {
            component.evaluate_constraint_quotients_on_domain(trace, &mut accumulator);
        }
        accumulator.finalize()
    }

    /// Computes the composition polynomial using a provided batch function.
    /// If the batch function returns None, falls back to the sequential path.
    pub fn compute_composition_polynomial_with<F>(
        &self,
        random_coeff: SecureField,
        trace: &Trace<'_, B>,
        batch_fn: F,
    ) -> SecureCirclePoly<B>
    where
        F: FnOnce(
            &[&dyn ComponentProver<B>],
            usize,
            SecureField,
            &Trace<'_, B>,
        ) -> Option<SecureCirclePoly<B>>,
    {
        if let Some(result) = batch_fn(
            &self.components,
            self.n_preprocessed_columns,
            random_coeff,
            trace,
        ) {
            return result;
        }
        self.compute_composition_polynomial(random_coeff, trace)
    }
}

