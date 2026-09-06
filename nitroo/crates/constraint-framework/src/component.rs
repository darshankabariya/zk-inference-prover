use core::fmt::{self, Display, Formatter};
use core::iter::zip;
use core::ops::Deref;

use hashbrown::HashMap;
use itertools::Itertools;
use num_traits::Zero;
use std_shims::{vec, String, Vec};
use stwo::core::air::accumulation::PointEvaluationAccumulator;
use stwo::core::air::Component;
use stwo::core::circle::CirclePoint;
use stwo::core::constraints::coset_vanishing;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::pcs::{TreeSubspan, TreeVec};
use stwo::core::poly::circle::{CanonicCoset, MIN_CIRCLE_DOMAIN_LOG_SIZE};
use stwo::core::utils::all_unique;
use stwo::core::ColumnVec;

use super::preprocessed_columns::PreProcessedColumnId;
use super::{EvalAtRow, InfoEvaluator, PointEvaluator, MODEL_TRACE_IDX, PREPROCESSED_TRACE_IDX};

#[derive(Debug, Default)]
enum PreprocessedColumnsAllocationMode {
    #[default]
    Dynamic,
    Static,
}

// TODO(andrew): Docs.
// TODO(andrew): Consider better location for this.
#[derive(Debug, Default)]
pub struct TraceLocationAllocator {
    /// Mapping of tree index to next available column offset.
    next_tree_offsets: TreeVec<usize>,
    /// Mapping of preprocessed columns to their index.
    preprocessed_columns: Vec<PreProcessedColumnId>,
    /// Stream PP (E11): the model tree's columns, in that tree's own commit
    /// order.  A component's preprocessed index space is the shape tree's
    /// columns followed by these, so an index at or above
    /// `preprocessed_columns.len()` names a model column.
    model_columns: Vec<PreProcessedColumnId>,
    /// Controls whether the preprocessed columns are dynamic or static (default=Dynamic).
    preprocessed_columns_allocation_mode: PreprocessedColumnsAllocationMode,
}

impl TraceLocationAllocator {
    pub fn next_for_structure<T>(
        &mut self,
        structure: &TreeVec<ColumnVec<T>>,
    ) -> TreeVec<TreeSubspan> {
        if structure.len() > self.next_tree_offsets.len() {
            self.next_tree_offsets.resize(structure.len(), 0);
        }

        TreeVec::new(
            zip(&mut *self.next_tree_offsets, &**structure)
                .enumerate()
                .map(|(tree_index, (offset, cols))| {
                    let col_start = *offset;
                    let col_end = col_start + cols.len();
                    *offset = col_end;
                    TreeSubspan {
                        tree_index,
                        col_start,
                        col_end,
                    }
                })
                .collect(),
        )
    }

    /// Create a new `TraceLocationAllocator` with fixed preprocessed columns setup.
    pub fn new_with_preprocessed_columns(preprocessed_columns: &[PreProcessedColumnId]) -> Self {
        assert!(
            all_unique(preprocessed_columns),
            "Duplicate preprocessed columns are not allowed!"
        );
        Self {
            next_tree_offsets: Default::default(),
            preprocessed_columns: preprocessed_columns.to_vec(),
            model_columns: Vec::new(),
            preprocessed_columns_allocation_mode: PreprocessedColumnsAllocationMode::Static,
        }
    }

    /// Stream PP (E11): the same, with the preprocessed columns split over the
    /// two committed trees -- `shape` is tree 0's list and `model` is tree 1's,
    /// each in its own tree's commit order.  The two must be disjoint.
    pub fn new_with_split_preprocessed_columns(
        shape: &[PreProcessedColumnId],
        model: &[PreProcessedColumnId],
    ) -> Self {
        let both: Vec<PreProcessedColumnId> =
            shape.iter().chain(model.iter()).cloned().collect();
        assert!(
            all_unique(&both),
            "Duplicate preprocessed columns are not allowed!"
        );
        Self {
            next_tree_offsets: Default::default(),
            preprocessed_columns: shape.to_vec(),
            model_columns: model.to_vec(),
            preprocessed_columns_allocation_mode: PreprocessedColumnsAllocationMode::Static,
        }
    }

    pub const fn preprocessed_columns(&self) -> &Vec<PreProcessedColumnId> {
        &self.preprocessed_columns
    }

    /// Stream PP (E11): tree 1's columns, in commit order.
    pub const fn model_columns(&self) -> &Vec<PreProcessedColumnId> {
        &self.model_columns
    }

    // Validates that `self.preprocessed_columns` is a permutation of
    // `preprocessed_columns`.
    pub fn validate_preprocessed_columns(&self, preprocessed_columns: &[PreProcessedColumnId]) {
        let mut self_columns = self.preprocessed_columns.clone();
        let mut input_columns = preprocessed_columns.to_vec();
        self_columns.sort_by_key(|col| col.id.clone());
        input_columns.sort_by_key(|col| col.id.clone());
        assert_eq!(
            self_columns, input_columns,
            "Preprocessed columns are not a permutation."
        );
    }
}

/// A component defined solely in means of the constraints framework.
///
/// Implementing this trait introduces implementations for [`Component`] and [`ComponentProver`] for
/// the SIMD backend. Note that the constraint framework only supports components with columns of
/// the same size.
///
/// [`ComponentProver`]: stwo::prover::ComponentProver
pub trait FrameworkEval {
    fn log_size(&self) -> u32;

    fn max_constraint_log_degree_bound(&self) -> u32;

    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;

    /// Returns the CUDA eval name used to compute the FNV1a eval_id for GPU dispatch.
    /// Must match the string used in the CUDA `evaluate_constraints.cu` dispatch table.
    /// Override this for components with CUDA kernel support.
    fn cuda_eval_name(&self) -> &'static str {
        unimplemented!(
            "cuda_eval_name() not implemented for this component. \
             Override this method to enable CUDA constraint evaluation."
        )
    }

    /// B7a 2b: the flat parameter block the CUDA kernel should be handed instead
    /// of `&self`'s raw bytes. `None` -- the default -- keeps the fork's original
    /// behaviour, which is correct for a POD eval and meaningless for one holding
    /// `String` or `Vec`.
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        None
    }
}

#[repr(C)]
pub struct FrameworkComponent<C: FrameworkEval> {
    pub eval: C,
    pub(super) trace_locations: TreeVec<TreeSubspan>,
    pub(super) preprocessed_column_indices: Vec<usize>,
    /// Stream PP (E11): where the shape tree's index space ends and the model
    /// tree's begins, inside `preprocessed_column_indices`.
    pub(super) n_shape_columns: usize,
    pub(super) claimed_sum: SecureField,
    info: InfoEvaluator,
    is_disabled: bool,
}

impl<E: FrameworkEval> FrameworkComponent<E> {
    pub fn new(
        location_allocator: &mut TraceLocationAllocator,
        eval: E,
        claimed_sum: SecureField,
    ) -> Self {
        let is_disabled = false;
        Self::new_ex(location_allocator, eval, claimed_sum, is_disabled)
    }

    pub fn disabled(location_allocator: &mut TraceLocationAllocator, eval: E) -> Self {
        let claimed_sum = SecureField::zero();
        let is_disabled = true;
        Self::new_ex(location_allocator, eval, claimed_sum, is_disabled)
    }

    pub fn new_ex(
        location_allocator: &mut TraceLocationAllocator,
        eval: E,
        claimed_sum: SecureField,
        is_disabled: bool,
    ) -> Self {
        let info = eval.evaluate(InfoEvaluator::new(eval.log_size(), vec![], claimed_sum));
        let trace_locations = location_allocator.next_for_structure(&info.mask_offsets);

        // Stream PP (E11): the flat index space is the shape tree's columns
        // followed by the model tree's, so a component's preprocessed view --
        // which is still one vector in `get_preprocessed_column` call order --
        // can be gathered from either tree by `pre_loc`.  A dynamically
        // allocated column joins the shape tree, which is the pre-PP behaviour.
        let n_static_shape = location_allocator.preprocessed_columns.len();
        let preprocessed_column_indices: Vec<usize> = info
            .preprocessed_columns
            .iter()
            .map(|col| {
                let next_column = location_allocator.preprocessed_columns.len();
                if let Some(pos) = location_allocator
                    .preprocessed_columns
                    .iter()
                    .position(|x| x.id == col.id)
                {
                    pos
                } else if let Some(pos) = location_allocator
                    .model_columns
                    .iter()
                    .position(|x| x.id == col.id)
                {
                    n_static_shape + pos
                } else {
                    if matches!(
                        location_allocator.preprocessed_columns_allocation_mode,
                        PreprocessedColumnsAllocationMode::Static
                    ) {
                        panic!("Preprocessed column {col:?} is missing from static allocation");
                    }
                    location_allocator.preprocessed_columns.push(col.clone());
                    next_column
                }
            })
            .collect();
        // With a model tree present the allocation is static, so the shape
        // list cannot have grown; without one every index is the shape tree's.
        let n_shape_columns = if location_allocator.model_columns.is_empty() {
            location_allocator.preprocessed_columns.len()
        } else {
            debug_assert_eq!(location_allocator.preprocessed_columns.len(), n_static_shape);
            n_static_shape
        };
        Self {
            eval,
            trace_locations,
            info,
            preprocessed_column_indices,
            n_shape_columns,
            claimed_sum,
            is_disabled,
        }
    }

    /// Stream PP (E11): a flat preprocessed index as (committed tree, column in
    /// that tree).  Below `n_shape_columns` it is the shape tree's; at or above
    /// it, the model tree's.
    pub fn pre_loc(&self, idx: usize) -> (usize, usize) {
        if idx < self.n_shape_columns {
            (PREPROCESSED_TRACE_IDX, idx)
        } else {
            (MODEL_TRACE_IDX, idx - self.n_shape_columns)
        }
    }

    /// Stream PP (E11): how many of this component's preprocessed columns each
    /// of the two trees carries, in the order the trees are committed.
    pub fn pre_counts(&self) -> (usize, usize) {
        let shape = self
            .preprocessed_column_indices
            .iter()
            .filter(|i| **i < self.n_shape_columns)
            .count();
        (shape, self.preprocessed_column_indices.len() - shape)
    }

    pub fn trace_locations(&self) -> &[TreeSubspan] {
        &self.trace_locations
    }

    pub fn preprocessed_column_indices(&self) -> &[usize] {
        &self.preprocessed_column_indices
    }

    pub const fn claimed_sum(&self) -> SecureField {
        self.claimed_sum
    }

    pub fn logup_counts(&self) -> RelationCounts {
        let size = 1 << self.eval.log_size();
        RelationCounts(
            self.info
                .logup_counts
                .iter()
                .map(|(k, v)| (k.clone(), v * size))
                .collect(),
        )
    }

    pub fn is_disabled(&self) -> bool {
        self.is_disabled
    }
}

pub struct RelationCounts(HashMap<String, usize>);
impl Deref for RelationCounts {
    type Target = HashMap<String, usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: FrameworkEval> Component for FrameworkComponent<E> {
    fn n_constraints(&self) -> usize {
        self.info.n_constraints
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        if self.is_disabled() {
            MIN_CIRCLE_DOMAIN_LOG_SIZE
        } else {
            self.eval.max_constraint_log_degree_bound()
        }
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        let log_size = if self.is_disabled() {
            MIN_CIRCLE_DOMAIN_LOG_SIZE
        } else {
            self.eval.log_size()
        };

        let mut log_degree_bounds = self
            .info
            .mask_offsets
            .as_ref()
            .map(|tree_offsets| vec![log_size; tree_offsets.len()]);

        // Stream PP (E11): the component's preprocessed columns are split over
        // the two committed preprocessed trees, so its own column-size vector
        // has to be too -- `Components::column_log_sizes` scatters them into
        // the global lists per tree, and `compose`'s `sizes_ok` compares the
        // concatenation against the statement's per-tree sizes.
        let (n_shape, n_model) = self.pre_counts();
        if log_degree_bounds.len() <= MODEL_TRACE_IDX {
            log_degree_bounds.resize(MODEL_TRACE_IDX + 1, vec![]);
        }
        log_degree_bounds[PREPROCESSED_TRACE_IDX] = vec![log_size; n_shape];
        log_degree_bounds[MODEL_TRACE_IDX] = vec![log_size; n_model];

        log_degree_bounds
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
        max_log_degree_bound: u32,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        let trace_step = CanonicCoset::new(max_log_degree_bound).step();
        self.info.mask_offsets.as_ref().map_cols(|col_offsets| {
            col_offsets
                .iter()
                .map(|offset| point + trace_step.mul_signed(*offset).into_ef())
                .collect()
        })
    }

    fn preprocessed_column_indices(&self) -> ColumnVec<usize> {
        self.preprocessed_column_indices.clone()
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        max_log_degree_bound: u32,
    ) {
        if self.is_disabled {
            for _ in 0..self.n_constraints() {
                evaluation_accumulator.accumulate(SecureField::zero());
            }
            return;
        }
        // Stream PP (E11): gathered from the shape tree and the model tree, in
        // the component's own `get_preprocessed_column` call order.
        let preprocessed_mask = self
            .preprocessed_column_indices
            .iter()
            .map(|idx| {
                let (tree, col) = self.pre_loc(*idx);
                &mask[tree][col]
            })
            .collect_vec();

        let mut mask_points = mask.sub_tree(&self.trace_locations);
        mask_points[PREPROCESSED_TRACE_IDX] = preprocessed_mask;
        // tree 1 is never named by a component's own mask; it is read only
        // through the gather above.
        if mask_points.len() > MODEL_TRACE_IDX {
            mask_points[MODEL_TRACE_IDX] = vec![];
        }

        self.eval.evaluate(PointEvaluator::new(
            mask_points,
            evaluation_accumulator,
            coset_vanishing(CanonicCoset::new(max_log_degree_bound).coset, point).inverse(),
            self.eval.log_size(),
            self.claimed_sum,
        ));
    }
}

impl<E: FrameworkEval> Deref for FrameworkComponent<E> {
    type Target = E;

    fn deref(&self) -> &E {
        &self.eval
    }
}

impl<E: FrameworkEval> Display for FrameworkComponent<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let log_n_rows = self.log_size();
        let mut n_cols = vec![];
        self.trace_log_degree_bounds()
            .0
            .iter()
            .for_each(|interaction| {
                n_cols.push(interaction.len());
            });
        writeln!(f, "n_rows 2^{log_n_rows}")?;
        writeln!(f, "n_constraints {}", self.n_constraints())?;
        writeln!(
            f,
            "constraint_log_degree_bound {}",
            self.max_constraint_log_degree_bound()
        )?;
        writeln!(
            f,
            "total felts: 2^{} * {}",
            log_n_rows,
            n_cols.iter().sum::<usize>()
        )?;
        for (j, n_cols) in n_cols.into_iter().enumerate() {
            writeln!(f, "\t Interaction {j}: n_cols {n_cols}")?;
        }
        Ok(())
    }
}
