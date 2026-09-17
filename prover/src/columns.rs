//! The pieces a gadget's column schema is built out of (#221).
//!
//! A gadget's AIR reads its columns by position and its trace builder writes
//! them by position, and nothing today names a column on both sides.  The
//! scheme under trial is that each gadget declares its row *once*, as one struct
//! per tree whose fields are the tree's columns, and that both sides go through
//! that declaration: the AIR instantiates it with the evaluator's field types
//! and reads named fields, the builders instantiate it with whole columns and
//! hand the tree back in the order it is committed in.
//! `gadgets/freivalds/columns.rs` is the first one; this module holds the two
//! parts that are not one gadget's.
//!
//! Each struct is generic over one type, which is what a column is *to whoever
//! holds the row*:
//!
//! | holder | preprocessed or base column | interaction column |
//! |---|---|---|
//! | the AIR (`read`) | this row's cell, `E::F` | `Run<E::EF>` -- this row's and the previous row's |
//! | the trace builder | the whole column, an `Eval` | -- |
//! | the interaction builder | -- | the whole column's values |
//! | the id list | the column's `PreProcessedColumnId` | -- |
//!
//! ## Why this is not a macro
//!
//! The issue asks for a `macro_rules!` taking the field list and emitting the
//! struct, the reader, the writer and the names.  The reader half is easy: with
//! every field written `name: ty = <how to read it>,` one non-recursive rule
//! emits the struct and a `read` whose body is a single struct literal, because
//! Rust evaluates a struct expression's fields in the order they are written.
//! The writer half is what stops it.  Flattening a row into its tree's columns
//! has to dispatch on each field's *shape* -- a cell, a vector, a vector of
//! vectors, an option -- and a trait cannot carry that dispatch (`impl<T> Cells
//! for T` and `impl<T> Cells for Vec<T>` overlap, so coherence rejects the pair).
//! Dispatching inside the macro instead means one rule per shape, which means a
//! `tt` muncher, and a muncher's rules are separate expansions: the `eval` a
//! reader rule writes and the `eval` the final rule declares as a parameter are
//! different identifiers under macro hygiene, so every binding has to be
//! threaded through the accumulator by the top rule.  That is around 200 lines
//! of macro to save around 40 lines of straight-line Rust, in the file a
//! reviewer opens first.  The hand-written form below is what a gadget writes;
//! the shape the macro would have, if the tree ever wants one, is
//!
//! ```text
//! column_group! {
//!     /// The Freivalds check's preprocessed tree.
//!     pub struct FreivaldsPre<C>;
//!     cells F;                       // which of the evaluator's two field types
//!     read(ev, sh: &FreivaldsShape); // the bindings the field expressions see
//!     nofirst: C   = ev.get_preprocessed_column(freivalds_sel(&sh.tag, "nofirst"));
//!     w: Vec<Vec<C>> = (0..sh.n_mats).map(|m| ...).collect();
//! }
//! ```
//!
//! and it would emit the struct, `read`, and `n_cols`, but not `into_committed`.
//!
//! ## What the declaration does and does not catch
//!
//! It removes the second copy of the order: a column's position is written down
//! once, and the AIR, the trace builder and the interaction builder all take it
//! from there.  What it does not do is make every mistake a type error.  Two
//! fields of the same type can be exchanged in the declaration and that
//! compiles; because both sides read the same declaration, such an exchange
//! permutes the mask numbering and the committed order together, does not change
//! what is proved, and is caught by the pinned digests and by the generated
//! kernel's fingerprint rather than by any test of the constraints.
//!
//! What the row-level satisfaction test each gadget now carries catches is the
//! other kind: a reader and a writer that disagree about a column's position, a
//! selector true on the wrong rows, an accumulator carried across the wrong
//! boundary, a padded row the constraints do not hold on.  It builds a trace
//! through the writer, evaluates the AIR through the reader on the same witness,
//! and names the constraint and the row.

use stwo_constraint_framework::{EvalAtRow, INTERACTION_TRACE_IDX};

/// One running column, at the two row offsets an accumulator's constraint reads
/// it at: `prev` is the previous row (`-1`) and `cur` is this one (`0`).
///
/// It is one column, not two.  A writer holds the column itself and never a
/// `Run`; only the AIR sees the pair, which is why the reader's row is the only
/// one whose interaction columns are `Run`s.
#[derive(Clone, Debug)]
pub struct Run<T> {
    pub prev: T,
    pub cur: T,
}

impl<T> Run<T> {
    /// The next interaction column, at `[-1, 0]`.  Every gadget's running
    /// columns are read this way, so the offsets are written down once.
    pub fn read<E: EvalAtRow<EF = T>>(eval: &mut E) -> Self {
        let [prev, cur] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        Run { prev, cur }
    }
}

// --------------------------------------------------- the row-level checker ---

/// Every constraint's *value* at one row, where the framework's own evaluator
/// only asserts that it is zero.
///
/// It is `AssertEvaluator` with the assertion taken off: every mask read is
/// delegated to it, so the column indexing and the `[-1, 0]` offsets are the
/// framework's and not a second copy of them, and so is the LogUp finalisation.
/// The LogUp constraints therefore still assert rather than being collected,
/// which is right: they are the relation's constraints and not the row's, and a
/// gadget's controls are written against the row's.
#[cfg(all(test, not(feature = "cuda")))]
pub struct RowValues<'a> {
    inner: stwo_constraint_framework::AssertEvaluator<'a>,
    /// one value per `add_constraint`, in the order the AIR adds them
    pub vals: Vec<stwo::core::fields::qm31::SecureField>,
}

#[cfg(all(test, not(feature = "cuda")))]
impl<'a> RowValues<'a> {
    pub fn new(inner: stwo_constraint_framework::AssertEvaluator<'a>) -> Self {
        RowValues { inner, vals: vec![] }
    }
}

#[cfg(all(test, not(feature = "cuda")))]
impl EvalAtRow for RowValues<'_> {
    type F = stwo::core::fields::m31::BaseField;
    type EF = stwo::core::fields::qm31::SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        self.inner.next_interaction_mask(interaction, offsets)
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: core::ops::Mul<G, Output = Self::EF> + From<G>,
    {
        self.vals.push(Self::EF::from(constraint));
    }

    fn combine_ef(
        values: [Self::F; stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE],
    ) -> Self::EF {
        Self::EF::from_m31_array(values)
    }

    fn write_logup_frac(&mut self, fraction: stwo::core::Fraction<Self::EF, Self::EF>) {
        self.inner.write_logup_frac(fraction);
    }
    fn finalize_logup_batched(&mut self, batching: &Vec<usize>) {
        self.inner.finalize_logup_batched(batching);
    }
    fn finalize_logup(&mut self) {
        self.inner.finalize_logup();
    }
    fn finalize_logup_in_pairs(&mut self) {
        self.inner.finalize_logup_in_pairs();
    }
}

/// Every constraint at every row of the trace domain, in the AIR's own order.
///
/// `trace` is the component's four trees -- preprocessed (in the order the AIR
/// *reads* them, which is how a component's own mask is numbered), the model
/// tree, the base trace and the interaction trace.
#[cfg(all(test, not(feature = "cuda")))]
pub fn constraint_values<E: stwo_constraint_framework::FrameworkEval>(
    air: &E,
    trace: &stwo::core::pcs::TreeVec<Vec<&Vec<stwo::core::fields::m31::BaseField>>>,
    claimed_sum: stwo::core::fields::qm31::SecureField,
) -> Vec<Vec<stwo::core::fields::qm31::SecureField>> {
    let log_size = air.log_size();
    (0..1usize << log_size)
        .map(|row| {
            let inner = stwo_constraint_framework::AssertEvaluator::new(
                trace, row, log_size, claimed_sum);
            air.evaluate(RowValues::new(inner)).vals
        })
        .collect()
}
