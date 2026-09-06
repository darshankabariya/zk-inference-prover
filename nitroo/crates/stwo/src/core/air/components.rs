use core::iter::zip;

use std_shims::{vec, Vec};

use super::accumulation::PointEvaluationAccumulator;
use super::Component;
use crate::core::circle::CirclePoint;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::verifier::{MODEL_TRACE_IDX, PREPROCESSED_TRACE_IDX};
use crate::core::ColumnVec;

pub struct Components<'a> {
    pub components: Vec<&'a dyn Component>,
    /// tree 0's width: the shape columns, which a verifier rebuilds itself.
    pub n_preprocessed_columns: usize,
    /// Stream PP (E11): tree 1's width -- the model columns.  A component's
    /// preprocessed index space is tree 0's columns followed by tree 1's, so an
    /// index at or above `n_preprocessed_columns` names a model column.
    pub n_model_columns: usize,
}

impl Components<'_> {
    pub fn composition_log_degree_bound(&self) -> u32 {
        self.components
            .iter()
            .map(|component| component.max_constraint_log_degree_bound())
            .max()
            .unwrap()
    }

    pub fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
        include_all_preprocessed_columns: bool,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let mut mask_points = TreeVec::concat_cols(
            self.components
                .iter()
                .map(|component| component.mask_points(point, max_log_degree_bound)),
        );

        // Stream PP (E11): the preprocessed columns live in two committed trees.
        // A component's indices run over tree 0's columns then tree 1's, so the
        // points are scattered into both here.
        let n_shape = self.n_preprocessed_columns;
        let (mut shape_pts, mut model_pts) = if include_all_preprocessed_columns {
            (
                vec![vec![point]; n_shape],
                vec![vec![point]; self.n_model_columns],
            )
        } else {
            let mut shape = vec![vec![]; n_shape];
            let mut model = vec![vec![]; self.n_model_columns];
            for component in &self.components {
                for idx in component.preprocessed_column_indices() {
                    if idx < n_shape {
                        shape[idx] = vec![point];
                    } else {
                        model[idx - n_shape] = vec![point];
                    }
                }
            }
            (shape, model)
        };
        if mask_points.len() <= MODEL_TRACE_IDX {
            mask_points.resize(MODEL_TRACE_IDX + 1, vec![]);
        }
        core::mem::swap(&mut mask_points[PREPROCESSED_TRACE_IDX], &mut shape_pts);
        core::mem::swap(&mut mask_points[MODEL_TRACE_IDX], &mut model_pts);

        mask_points
    }

    pub fn eval_composition_polynomial_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask_values: &TreeVec<Vec<Vec<SecureField>>>,
        random_coeff: SecureField,
        max_log_degree_bound: u32,
    ) -> SecureField {
        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        for (i, component) in self.components.iter().enumerate() {
            let before = evaluation_accumulator.current();
            component.evaluate_constraint_quotients_at_point(
                point,
                mask_values,
                &mut evaluation_accumulator,
                max_log_degree_bound,
            );
        }
        evaluation_accumulator.finalize()
    }

    pub fn column_log_sizes(&self) -> TreeVec<ColumnVec<u32>> {
        // Stream PP (E11): the same scatter as `mask_points` -- a component's
        // preprocessed index space is tree 0's columns then tree 1's, and its
        // own size vector carries the two halves at 0 and 1 in that order.
        let n_shape = self.n_preprocessed_columns;
        let n_all = n_shape + self.n_model_columns;
        let mut pre_log_sizes = vec![0; n_all];
        let mut visited_columns = vec![false; n_all];

        let mut column_log_sizes = TreeVec::concat_cols(self.components.iter().map(|component| {
            let component_trace_log_sizes = component.trace_log_degree_bounds();

            let own: Vec<u32> = [PREPROCESSED_TRACE_IDX, MODEL_TRACE_IDX]
                .into_iter()
                .flat_map(|t| {
                    component_trace_log_sizes
                        .get(t)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                        .iter()
                        .copied()
                })
                .collect();
            for (column_index, log_size) in zip(component.preprocessed_column_indices(), own) {
                let column_log_size = &mut pre_log_sizes[column_index];
                if visited_columns[column_index] {
                    assert!(
                        *column_log_size == log_size,
                        "Preprocessed column size mismatch for column {column_index}"
                    );
                } else {
                    *column_log_size = log_size;
                    visited_columns[column_index] = true;
                }
            }

            component_trace_log_sizes
        }));

        assert!(
            visited_columns.iter().all(|&updated| updated),
            "Column size not set for all reprocessed columns"
        );

        if column_log_sizes.len() <= MODEL_TRACE_IDX {
            column_log_sizes.resize(MODEL_TRACE_IDX + 1, vec![]);
        }
        column_log_sizes[MODEL_TRACE_IDX] = pre_log_sizes.split_off(n_shape);
        column_log_sizes[PREPROCESSED_TRACE_IDX] = pre_log_sizes;

        column_log_sizes
    }
}
