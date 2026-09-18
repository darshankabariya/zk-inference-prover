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
//! | the AIR (`read`) | this row's cell, `E::F` | `fold::Run<E::EF>` -- this row's and the previous row's |
//! | the trace builder | the whole column, an `Eval` | -- |
//! | the interaction builder | -- | the whole column's values |
//! | the id list | the column's `PreProcessedColumnId` | -- |
//!
//! ## Cell kinds
//!
//! A field of a row declaration is usually a plain cell: the reader takes the
//! next mask and the writer hands a column over, and nothing else follows from
//! it.  Three things carry more than that, and a gadget gets each by declaring
//! it rather than by writing it out:
//!
//! - [`Bytes`], below: committed cells that are values of a range table.  The
//!   reader takes the masks *and* the row owes the lookups; the writer builds
//!   the columns *and* bumps the table's multiplicities; the offset a signed
//!   value carries to land inside an unsigned table is taken back off by
//!   [`ByteCells::fold_signed`], so a gadget folds its cells and never names it;
//!   and [`ByteCells`] holds the one decision the gadget has no opinion about --
//!   whether the lookups are proved by this component's own LogUp columns or by
//!   the batched backend.
//! - `fold::Run`, in `crate::fold`: one running column at the `[-1, 0]` mask its
//!   two constraint rows come from.  `crate::fold::Fold` is what a gadget
//!   declares over it, and it owns the recurrence on both sides and the column's
//!   place in the tree.
//! - `fold::Groups`, in the same module: not a field but the *table's* own
//!   shape, and the four preprocessed 0/1 columns that stand for the loop the
//!   constraints cannot write out.  A gadget states how its rows are grouped --
//!   how many rows a group takes, how tall the table is, what prefix the
//!   columns carry -- and the primitive writes them, names them, reads them and
//!   resolves a fold's carry rules against them.  A row declaration reads it
//!   first and hands it on; the gadget never names one of the four.
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
//!     /// M: the pinned weights.
//!     pub struct Known<Cell>;
//!     cells F;                       // which of the evaluator's two field types
//!     read(ev, sh: &FreivaldsShape); // the bindings the field expressions see
//!     weights: Vec<Vec<Cell>> = (0..sh.n_mats)
//!         .map(|m| (0..sh.lay.pk).map(|p| ev.get_preprocessed_column(sh.weight_id(m, p)))
//!                                .collect())
//!         .collect();
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

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, Relation, RelationEntry};

use crate::common::Eval;
use crate::device::AnyCol;
use crate::layer::Tables;
use crate::relations::RangeRel;

// ------------------------------------------------------ bytes, and a table ---

/// Who proves a lookup.
///
/// It is a property of the lookup and not of the gadget that makes it: a gadget
/// states that its cells are values of a table, and the plan says which backend
/// carries that statement.  Nothing in a gadget branches on this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lookups {
    /// this component's own LogUp fraction columns, in its interaction tree
    Own,
    /// the batched backend, which proves them outside this component and takes
    /// the site registration instead of the columns
    Batched,
}

/// The declaration of a row's byte cells: committed values that are values of a
/// range table.
///
/// "Byte" is the usual case and not the only width -- the widened attention
/// output's activations are one cell in a 12-bit table.  What the kind carries
/// is everything that follows from the cells being a table's values and nothing
/// that follows from what the gadget does with them: the width, the offset that
/// puts a signed value inside an unsigned table, the relation, and who proves
/// the lookups.
#[derive(Clone)]
pub struct ByteCells {
    /// the table's width in bits, so a cell is in `[0, 2^bits)`
    pub bits: u32,
    /// What a cell carries on top of the signed value it stands for, so that it
    /// lands inside the table: `2^(bits-1)` for a signed value, and whatever a
    /// widened cell needs.  The gadget's constraints subtract it back out where
    /// they fold the cell.
    pub off: u32,
    /// the relation the cells are looked up in
    pub rel: RangeRel,
    /// who proves the lookups
    pub by: Lookups,
}

/// A row's byte cells, as whoever holds the row sees them.
pub struct Bytes<Cell> {
    pub cells: Vec<Cell>,
}

impl<Cell> Bytes<Cell> {
    /// Read `n` byte cells off the row.  What they owe the table is
    /// `ByteCells::look`, which the AIR calls where its constraints end -- the
    /// relation entries have to follow the constraints, or the component's
    /// lowering records them in a different order than it does today.
    pub fn read<E: EvalAtRow<F = Cell>>(eval: &mut E, n: usize) -> Self {
        Bytes { cells: (0..n).map(|_| eval.next_trace_mask()).collect() }
    }

    pub fn into_committed(self) -> Vec<Cell> {
        self.cells
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

impl Bytes<Eval> {
    /// Build `n` byte columns of `2^log_rows` rows, and bump the table's
    /// multiplicities as they are built.
    ///
    /// `value(p, j)` is cell `p` of row `j` with the offset already on it -- a
    /// writer that commits a cell and forgets to count it is the one mistake
    /// this half can make on its own, and it cannot make it here.
    ///
    /// The width is a parameter rather than a `ByteCells` because this side of
    /// the transcript does not have one: the relation the cells are looked up in
    /// is drawn from the transcript *after* they are committed, so the table's
    /// width is all a writer can know and all it needs.
    pub fn write(log_rows: u32, bits: u32, n: usize, tables: &mut Tables,
                 value: impl Fn(usize, usize) -> i64) -> Self {
        let rows = 1usize << log_rows;
        let mut cells = Vec::with_capacity(n);
        for p in 0..n {
            let mut col = vec![M31::zero(); rows];
            for (j, slot) in col.iter_mut().enumerate() {
                let v = value(p, j);
                tables.bump_r(bits, v);
                *slot = crate::encoding::m31_of(v);
            }
            cells.push(crate::common::eval_of(log_rows, col));
        }
        Bytes { cells }
    }
}

impl ByteCells {
    /// Whether the lookups are this component's own columns.
    pub fn own(&self) -> bool {
        self.by == Lookups::Own
    }

    /// The same cells with their lookups proved here, whatever the plan said.
    ///
    /// The batched backend reads the fractions the component *would* have built,
    /// so the site is registered against this and not against the ablated form.
    pub fn proved_here(&self) -> Self {
        ByteCells { by: Lookups::Own, ..self.clone() }
    }

    /// One unit lookup per cell, which is what a site's instance split is
    /// derived from: every byte look has the literal one as its numerator.
    pub fn units(&self, n: usize) -> Vec<bool> {
        vec![true; n]
    }

    /// `sum_p w[p] (cell[p] - off)`: the row's cells folded against `w` as the
    /// *signed* values they stand for, with the whole offset taken off once at
    /// the end.
    ///
    /// A committed byte carries [`ByteCells::off`] on top of the value it
    /// stands for, so that a signed value lands inside an unsigned table.  Only
    /// the cell kind knows that, so only the cell kind should be writing the
    /// subtraction; a gadget folds its cells and says nothing about the offset.
    ///
    /// This spelling weights every cell and subtracts `off * sum(w)`, a
    /// constant, from the result.  [`ByteCells::fold_signed_per_cell`] takes the
    /// offset off each cell instead.  The two are equal as field elements and
    /// are *not* the same expression tree, and the lowering fingerprints the
    /// tree -- which is why both are here rather than one of them.
    pub fn fold_signed<E: EvalAtRow>(&self, bytes: &Bytes<E::F>, w: &[SecureField]) -> E::EF {
        let off_sum: SecureField = w.iter().fold(SecureField::zero(), |a, b| a + *b)
            * SecureField::from_u32_unchecked(self.off, 0, 0, 0);
        crate::fold::dot::<E>(&bytes.cells, w) - off_sum
    }

    /// The same value, with the offset taken off each cell before it is
    /// weighted: `sum_p (cell[p] - off) w[p]`.
    ///
    /// See [`ByteCells::fold_signed`] for why there are two.
    pub fn fold_signed_per_cell<E: EvalAtRow>(&self, bytes: &Bytes<E::F>,
                                              w: &[SecureField]) -> E::EF {
        let off = E::F::from(M31::from_u32_unchecked(self.off));
        let mut f = E::EF::zero();
        for (p, a) in bytes.cells.iter().enumerate() {
            f += (E::EF::from(a.clone()) - E::EF::from(off.clone())) * w[p];
        }
        f
    }

    /// The lookups the cells owe, where the AIR's constraints end.
    ///
    /// Under the batched backend there is nothing here at all: no relation
    /// entries, and so no interaction columns allocated for them either.  The
    /// finalisation is inside, because a component whose only lookups are these
    /// has nothing else to batch them with -- a gadget that looks up more than
    /// one kind of cell wants the two halves apart, and that is the first thing
    /// the fan-out will ask for.
    pub fn look<E: EvalAtRow>(&self, eval: &mut E, bytes: &Bytes<E::F>) {
        if !self.own() {
            return;
        }
        for a in &bytes.cells {
            eval.add_to_relation(RelationEntry::new(&self.rel, E::EF::one(),
                                                    core::slice::from_ref(a)));
        }
        eval.finalize_logup_in_pairs();
    }

    /// The interaction columns the lookups own: this component's LogUp
    /// fractions, batched in pairs, or nothing at all when the batched backend
    /// proves them elsewhere.
    ///
    /// `cells` are the committed byte columns on the host, which only the host
    /// fraction builder reads -- on a component whose trace stayed on the card
    /// the generated builder reads the card's and this list is empty.
    pub fn frac_cols<E: FrameworkEval>(
        &self,
        air: &E,
        tag: &str,
        log_rows: u32,
        pre: crate::devfrac::Cols<'_>,
        base: crate::devfrac::Cols<'_>,
        cells: &[Eval],
        n: usize,
    ) -> (Vec<AnyCol<'static>>, SecureField) {
        use stwo::prover::backend::simd::qm31::PackedQM31;
        if !self.own() {
            return (vec![], SecureField::zero());
        }
        let one = PackedQM31::broadcast(SecureField::one());
        let fr = |i: usize, vr: usize| (one, self.rel.combine(&[cells[i].data[vr]]));
        crate::devfrac::frac_cols(air, tag, log_rows, pre, base, || {
            let (n_pairs, fpair) = crate::plogup::paired(n, fr);
            crate::plogup::build(log_rows, n_pairs, fpair)
        })
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
