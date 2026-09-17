//! The Freivalds check's columns, declared once.
//!
//! Three structs, one per tree.  The AIR instantiates them with the evaluator's
//! field types and reads named fields (`FreivaldsRow::read`); `trace.rs`
//! instantiates the two committed-before-the-challenges trees with whole
//! columns and `claims.rs` the after-the-challenges tree with the running sums'
//! values, and all three hand their tree back through `into_committed`, which is
//! the one place a column's position is written down.  `crate::columns` says
//! what the scheme is and why it is not a macro.
//!
//! **`Cell` is the cell type, not a commitment.**  Every struct here is generic
//! over what one column's cell is *to whoever holds the row*: the evaluator's
//! expression type `E::F` in the AIR, a whole built column (`Eval`) in the trace
//! builder, a slice of extension-field values in the claims builder, a
//! `PreProcessedColumnId` in the id list.  One declaration, four readings.
//!
//! Two orders live here and they are not the same one:
//!
//! - **the read order**, which is the field order of each struct.  It is what
//!   the AIR's masks are numbered by and therefore what `cudagen`'s lowering
//!   records, so moving a field moves a generated kernel's fingerprint.  The
//!   preprocessed tree is read row flags first, then the weights.
//! - **the committed order**, which is `into_committed`.  For the activations
//!   and the running sums it is the read order.  For the preprocessed tree it is
//!   *not*: `compose.rs` and `run.rs` register the weights first and the four row
//!   flags after, and a preprocessed column is resolved by id rather than by
//!   position, so the two orders are free to differ and do.

use stwo_constraint_framework::EvalAtRow;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::columns::Run;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::gadgets::freivalds::{FREIVALDS_SEL, FreivaldsEval, freivalds_sel, freivalds_w};

/// What the row's lengths come from: one Freivalds component's shape.
///
/// It borrows the group's name rather than owning it: the AIR builds one of
/// these on every `evaluate`, which is once per row of the evaluation domain.
#[derive(Clone, Copy, Debug)]
pub struct FreivaldsShape<'a> {
    /// the group's name, which every preprocessed id carries
    pub tag: &'a str,
    pub n_mats: usize,
    /// weights per row per matmul
    pub pk: usize,
    /// activation cells per row
    pub pq: usize,
    /// whether the component publishes a claim over its own activations
    pub x_claim: bool,
}

impl<'a> FreivaldsShape<'a> {
    /// The shape the AIR is over.
    pub fn of(air: &'a FreivaldsEval) -> Self {
        FreivaldsShape { tag: &air.tag, n_mats: air.n_mats, pk: air.pk, pq: air.pq,
                         x_claim: air.qx.is_some() }
    }
    /// The shape a builder is over, off the layout it tiles.
    pub fn new(tag: &'a str, lay: &FreivaldsLayout, n_mats: usize, x_claim: bool) -> Self {
        FreivaldsShape { tag, n_mats, pk: lay.pk, pq: lay.pq, x_claim }
    }

    /// How many columns each tree holds, off the same declaration the structs
    /// are: the weights and the four row flags, the activations, and the running
    /// sums.  The last count is in *running columns*; each is four M31 columns
    /// once encoded, and the LogUp columns that follow them belong to the
    /// relation rather than to the row.
    pub fn n_pre(&self) -> usize {
        self.n_mats * self.pk + FREIVALDS_SEL.len()
    }
    pub fn n_base(&self) -> usize {
        self.pq
    }
    pub fn n_inter(&self) -> usize {
        1 + 2 * self.n_mats + usize::from(self.x_claim)
    }
}

// ----------------------------------------- what the verifier already knows ---

/// M: the pinned weights, and the four row flags that drive the loop.
///
/// Both halves are preprocessed -- the verifier holds them itself and the prover
/// commits to neither.  The flags are the loop control: the AIR is one set of
/// constraints that runs on *every* row, so where an arkworks circuit would
/// write `for k in input_channels { for chunk in .. }` and unroll it, this says the
/// same thing with four 0/1 columns.  See the module comment of
/// `gadgets/freivalds/mod.rs` for the shape they control.
pub struct Known<Cell> {
    /// 0 on an input channel's first row, 1 on its later rows: whether the previous
    /// row's folds carry into this one, or the folds restart here.
    pub carry_in_input_channel: Cell,
    /// 1 on an input channel's last row, where its two folds are complete and their
    /// product is added into the running product once.
    pub input_channel_done: Cell,
    /// 0 on row 0 and 1 everywhere else: whether the running product carries in
    /// from the previous row.  The trace domain is cyclic, so row 0's "previous
    /// row" is the table's *last* row, and without this flag the finished
    /// product would be carried back into the start.
    pub carry_in_table: Cell,
    /// 1 on the table's last row, where the running product is pinned to the
    /// claim the statement carries.
    pub table_done: Cell,
    /// `[matmul][weight in the row]`: M's cells, `pk` of each matmul per row.
    pub weights: Vec<Vec<Cell>>,
}

impl<Cell> Known<Cell> {
    /// The tree's columns in *committed* order: every matmul's weights, then the
    /// four row flags.  `compose.rs` and `run.rs` build the id list in this order
    /// and the device witness generation writes its columns in it.
    pub fn into_committed(self) -> Vec<Cell> {
        let mut v: Vec<Cell> = self.weights.into_iter().flatten().collect();
        v.push(self.carry_in_input_channel);
        v.push(self.input_channel_done);
        v.push(self.carry_in_table);
        v.push(self.table_done);
        v
    }
}

impl Known<PreProcessedColumnId> {
    /// The component's preprocessed ids, one per field.  The id strings are
    /// pinned -- they are the keys a verifier resolves a preprocessed column by
    /// -- so they keep the four original flag names (`nofirst`, `last`,
    /// `notzero`, `end`); the glossary in `mod.rs` maps them to the fields.
    pub fn ids(sh: &FreivaldsShape) -> Self {
        Known {
            carry_in_input_channel: freivalds_sel(sh.tag, FREIVALDS_SEL[0]),
            input_channel_done: freivalds_sel(sh.tag, FREIVALDS_SEL[1]),
            carry_in_table: freivalds_sel(sh.tag, FREIVALDS_SEL[2]),
            table_done: freivalds_sel(sh.tag, FREIVALDS_SEL[3]),
            weights: (0..sh.n_mats)
                .map(|m| (0..sh.pk).map(|p| freivalds_w(sh.tag, m, p)).collect())
                .collect(),
        }
    }
    /// The same ids in committed order -- the list the trace location allocator
    /// takes.
    pub fn committed_ids(sh: &FreivaldsShape) -> Vec<PreProcessedColumnId> {
        Self::ids(sh).into_committed()
    }
}

// ------------------------------------------------ what the prover commits ----

/// A: one token chunk of the activations, committed as bytes.
///
/// Each cell carries the signed activation plus `act_off`, so it is in range for
/// the byte table it is looked up in; the constraints subtract the offset back
/// out where they fold it.
pub struct Activations<Cell> {
    pub x: Vec<Cell>,
}

impl<Cell> Activations<Cell> {
    pub fn into_committed(self) -> Vec<Cell> {
        self.x
    }
}

// ------------------------------------------------------ the running sums ----

/// One matmul's two running sums.
pub struct Matmul<Cell> {
    /// `rv^T W_m` for this input channel: the weights of the input channel's rows
    /// folded against the powers of `rv`, restarted at each input channel.
    pub w_fold: Cell,
    /// `(rv^T W_m)(X s)` so far: one input channel's two folds multiplied and added
    /// in on the input channel's last row, carried across the whole table.
    pub product: Cell,
}

/// The running sums, in the order the tree holds them: the shared `x_fold`,
/// then each matmul's pair, then the published claim over the activations.  The
/// LogUp columns follow these and are the relation's, not the row's.
pub struct RunningSums<Cell> {
    /// `X s` for this input channel: the committed activations of the input channel's
    /// rows folded against the powers of `s`.  It is shared by every matmul of
    /// the group, because a group is the matmuls that read one committed input.
    pub x_fold: Cell,
    pub mats: Vec<Matmul<Cell>>,
    /// The projection of the committed activations that this component
    /// publishes, so whichever component produced A can be bound to it.  `None`
    /// on a segment with no producer to bind to.
    pub x_claim: Option<Cell>,
}

impl<Cell> RunningSums<Cell> {
    pub fn into_committed(self) -> Vec<Cell> {
        let mut v = Vec::with_capacity(1 + 2 * self.mats.len() + 1);
        v.push(self.x_fold);
        for m in self.mats {
            v.push(m.w_fold);
            v.push(m.product);
        }
        v.extend(self.x_claim);
        v
    }
}

// -------------------------------------------------------------- the reader ---

/// One row of the Freivalds check, as the AIR sees it.
pub struct FreivaldsRow<F, EF> {
    pub known: Known<F>,
    pub acts: Activations<F>,
    pub sums: RunningSums<Run<EF>>,
}

impl<F, EF> FreivaldsRow<F, EF> {
    /// Pull the component's columns out of `eval` in declaration order.
    ///
    /// This is the order the masks are numbered in and so the order the
    /// generated kernel is emitted against; it is the same order the check has
    /// read its columns in since it was written, and
    /// `the_honest_trace_satisfies_every_constraint_on_every_row` is what says
    /// so rather than this comment.
    pub fn read<E: EvalAtRow<F = F, EF = EF>>(eval: &mut E, sh: &FreivaldsShape) -> Self {
        let ids = Known::ids(sh);
        FreivaldsRow {
            known: Known {
                carry_in_input_channel: eval.get_preprocessed_column(ids.carry_in_input_channel),
                input_channel_done: eval.get_preprocessed_column(ids.input_channel_done),
                carry_in_table: eval.get_preprocessed_column(ids.carry_in_table),
                table_done: eval.get_preprocessed_column(ids.table_done),
                weights: ids
                    .weights
                    .into_iter()
                    .map(|m| m.into_iter().map(|id| eval.get_preprocessed_column(id)).collect())
                    .collect(),
            },
            acts: Activations { x: (0..sh.pq).map(|_| eval.next_trace_mask()).collect() },
            sums: RunningSums {
                x_fold: Run::read(eval),
                mats: (0..sh.n_mats)
                    .map(|_| Matmul { w_fold: Run::read(eval), product: Run::read(eval) })
                    .collect(),
                x_claim: sh.x_claim.then(|| Run::read(eval)),
            },
        }
    }
}

/// What each of `FreivaldsEval::evaluate`'s constraints states, in the order it
/// adds them, so a row-level test can say which one a control broke.
pub fn constraint_names(sh: &FreivaldsShape) -> Vec<String> {
    let mut names = vec!["x_fold".to_string()];
    for m in 0..sh.n_mats {
        names.push(format!("w_fold[{m}]"));
        names.push(format!("product_step[{m}]"));
        names.push(format!("product_claim[{m}]"));
    }
    if sh.x_claim {
        names.push("x_claim_fold".to_string());
        names.push("x_claim_close".to_string());
    }
    names
}
