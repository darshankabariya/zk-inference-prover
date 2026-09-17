//! One running column, the row grouping it is folded over, and the recurrence
//! that fills it (#221).
//!
//! Most of this crate's gadgets carry columns of the same shape.  A value is
//! folded down a component's rows,
//!
//! ```text
//!     x[j] = carry[j] * x[j-1] + term[j]
//! ```
//!
//! with `term[j]` built out of row `j`'s own cells and `carry[j]` a challenge
//! power the layout picks, and where the value is published the table's last row
//! is pinned to a claim the statement carries.  Every one of them is written out
//! three times: two constraints in the AIR, a serial loop in the builder, and a
//! place in the interaction tree.  This module is those three written once, from
//! one declaration.
//!
//! # The grouping
//!
//! Where a fold restarts, where it takes a finished value, and where it closes
//! are all the same fact about the table: its rows come in *groups*.  A
//! [`Groups`] is that fact -- how many rows one group occupies, how tall the
//! table is, how many rows it is padded with, and the prefix its columns are
//! named under.  A STARK has one set of constraints and they run on every row,
//! so the loop an arkworks circuit would write out has to be expressed as data:
//! four preprocessed 0/1 columns, which this module emits.  They are the fold's
//! and not the gadget's -- a gadget states its grouping and never names a
//! column of it:
//!
//! | column | 1 on | what reads it |
//! |---|---|---|
//! | `{prefix}_nofirst` | a group's rows after its first | [`Carry::PerGroup`], [`Carry::Rates`] |
//! | `{prefix}_last` | a group's last row | [`Term::PerGroup`] |
//! | `{prefix}_notzero` | every row but row 0 | [`Carry::WholeTable`], [`Carry::Rates`] |
//! | `{prefix}_end` | the table's last row | [`Close`] |
//!
//! The group length is not a power of two, and the framework has no closed form
//! for "row `j` is `j mod n` for an `n` that is not a power of two", so the four
//! stay preprocessed columns rather than becoming expressions.  `notzero` exists
//! because the trace domain is *cyclic*: row 0's previous row is the table's
//! last, so a fold that runs down the whole table has to be told not to carry
//! the finished value back into the start.  A padding row is in no group, so
//! `nofirst` and `last` are 0 on it and it passes a whole-table fold through
//! unchanged.
//!
//! [`Groups::write`] builds the four columns on the host, [`Groups::ids`] names
//! them in the order they are committed in, and [`Groups::read`] is the AIR's
//! side -- four masks, read before anything else the row declares, handed back
//! as an opaque [`Rows`].
//!
//! # The fold
//!
//! A [`Fold`] says what one running column is:
//!
//! - [`Carry`] is what the previous row is multiplied by, as a rule about the
//!   grouping rather than as a flag: restart at each group ([`Carry::PerGroup`]),
//!   carry down the whole table ([`Carry::WholeTable`]), run down the whole
//!   table at two rates ([`Carry::Rates`]), or a multiplier the caller built out
//!   of its own columns ([`Carry::By`]);
//! - [`Term`] is this row's own contribution -- on every row ([`Term::Each`]),
//!   usually [`dot`] of the row's cells against fixed powers, or once per group
//!   on the group's last row ([`Term::PerGroup`]);
//! - [`Close`] is the claim the fold is pinned to.  It closes on the table's
//!   last row, which is the grouping's business too.  A column nothing
//!   publishes has none.
//!
//! [`constrain`] emits the one or two constraints, [`carry_down`] runs the
//! recurrence on the host, [`column`] puts the result in the tree and
//! [`M31_COLS`] is what it costs there.  [`place`] is the same group of columns
//! built on the card instead, with the mirror comparison behind it.  The
//! declaration is the same on both sides: [`Carry`] and [`Term`] are generic
//! over what a multiplier and a term *are* to whoever holds the row -- an
//! evaluator's expression in the AIR, a function of the row index on the host.
//!
//! # What this covers, and what it does not
//!
//! Freivalds is the first gadget through it (#221's trial).  Its four columns
//! are `x_fold` and `w_fold` ([`Carry::PerGroup`]), `product`
//! ([`Carry::WholeTable`] over a [`Term::PerGroup`]) and `x_claim`
//! ([`Carry::Rates`]), over one grouping: one group per input channel.  The rest
//! of the crate's running columns, and what each would take -- this is the
//! inventory, not a promise, and no other gadget is converted here:
//!
//! | column | carry | term | as is? |
//! |---|---|---|---|
//! | `dequant` `z` | restart at each token block | the accumulator cells against `rv` | `PerGroup` |
//! | `dequant` `h` | `m1 + m2 s` | `z * last` -- another running column | `By`, and the term is not a [`dot`] |
//! | `dequant` `u`, `v` | two flags, two powers | a bf16 decode, a shared pair | `By` |
//! | `attn_out` `z` | restart at each token of a head | the accumulator cells against `rv` | `PerGroup` |
//! | `attn_out` `h`, `dz` | three flags, three powers | `z * last`, `den * last` | `By` |
//! | `attn_out` `zq`, `hq` | as `z`, `h` | as `z`, `h` | as `z`, `h` |
//! | `requant` `ui`, `uo` | `nofirst k_in + (gt0 - nofirst) k_bd` | the row's cells against `w` | `Rates` exactly, `gt0` being `notzero` |
//! | `requant` `us` | `gt0 + mlast (mu - 1)` | a shared pair times `last` | `By` |
//! | `rope` `ui` | `gt0 * step_in` | the row's cells against `w_in` | `By`: `prev * (notzero step)` is a fifth tree, a whole-table fold *at a step* |
//! | `rope` `uo` | `gt0 s + hstart (s_h - s)` | the row's cells against `w_out` | `By`, and `hstart` is a column of its own |
//! | `embed` `h`, `g` | `s_in k_in + s_bd k_bd` | the row's slots against `w`; one cell gated by `1 - s_in` | `By` today, `Rates` once its padding fits (below) |
//! | `argmax` `u`, `g` | two rates over the head dequant's selectors | as `embed`'s two | `Rates`, over a grouping another component wrote |
//! | `argmax` `ss` | `nofirst` | indicator bits, no powers | not a fold: base field, and it closes on a committed cell rather than on a claim |
//! | `residual_stream` bindings | the whole table, at one power | the witness, not a column | `By`: the tree is `(prev * flag) * step` and not `(prev * step) * flag` |
//!
//! ## Which of their columns this same object would produce
//!
//! The four columns follow from three numbers, so any table tiled as "`lead`
//! inert rows, then one group per something, `rows_per_group` rows each" gets
//! them from one [`Groups`].  What the gadgets' selector lists say, read against
//! that:
//!
//! - **dequant** tiles `crate::layout::MatLayout` -- `lead` inert rows, then
//!   `(token, chunk)` descending -- so its grouping is one group per *token
//!   block*, `chunks` rows each, and this same object writes it.  `DEQ_SEL` is
//!   `nofirst`, `last`, `m1`, `m2`, `end`: three of the four are the grouping's,
//!   and it has no `notzero` because no fold of its runs down the whole table.
//!   `m1`/`m2` are **not** the grouping's -- they are the two challenge powers
//!   `h` steps at, preprocessed only because a constraint cannot pick between
//!   two constants on its own.  Under [`Carry::Rates`] they would be two
//!   `SecureField`s in the declaration and two fewer committed columns, which is
//!   a change to what is committed and so a change of digest: a PR of its own.
//! - **attn_out** tiles `AoLayout` -- `lead` inert rows, then `(head, token,
//!   chunk)` descending -- and its `nofirst`/`last` are per *token*, so the
//!   grouping is one group per `(head, token)` and again this object's.
//!   `AO_SEL` is `nofirst`, `last`, `m1`, `m2`, `m3`, `end`: three of the four,
//!   no `notzero`, and three steps where dequant has two.
//! - **attention**'s key tiles tile a `Grid` whose groups are `(instance, key)`
//!   pairs, so `b6_k_nofirst`, `b6_k_last`, `b6_k_notzero` and `b6_k_end` are
//!   exactly one [`Groups`] -- all four, which is the case Freivalds is.  The
//!   chan grid is another.  But `b6_k_istart` (and the diagonal block's
//!   `b6_x_iend`) is a *second* level above the chunk grouping: where an
//!   instance begins and ends.  A grouping of groupings is the pattern this
//!   object does not have; the honest shape is a second [`Groups`] over the same
//!   rows rather than a nesting field, because the two levels have independent
//!   lengths.  `ROW_SEL` is `notzero`, `end`, `istart` -- two of the four and
//!   that second level again.
//! - **residual_stream** is one group per component: the whole table is the
//!   group, so `nofirst` and `notzero` are the same column and `last` and `end`
//!   are too, and the component commits the two it needs.  This object would
//!   emit four.
//! - **requant** is the one other gadget whose list is all four: `RQ_SEL` is
//!   `nofirst`, `last`, `end`, `gt0`, `mlast`, and `gt0` is `notzero` under
//!   another name.  `mlast` is its own.
//! - **rope** has no chunk grouping at all.  `Q_SEL` is `gt0`, `end`, `hstart`:
//!   two of the four, and `hstart` marks kv-head starts, which is a second level
//!   as attention's `istart` is.
//! - **embed** pads at the *end* and masks the padding with an `act` column
//!   rather than skipping a lead, and it writes `s_in` and `s_bd` -- "inside a
//!   token" and "a token boundary above row 0" -- as two columns where
//!   [`Carry::Rates`] reads `nofirst` and `notzero` and takes the difference.
//!   Same grouping, a different encoding of it and a different place for the
//!   padding, so it is the one that needs [`Groups`] to learn where a table's
//!   inert rows are before it fits.
//! - **argmax** has `b10_am_nofirst`, `b10_am_last`, `b10_am_end` of its own --
//!   three of the four -- and its `u` and `g` read the head dequant's selectors
//!   instead, which the declaration does not care about (a prefix is a prefix)
//!   but which means the [`Groups`] that *writes* a set of columns and the one
//!   that *reads* them are two values, and only the writer may emit them.
//!
//! So the first thing the fan-out will ask [`Groups`] for is a **subset**: only
//! attention's two grids and requant want all four, and a gadget that wants
//! three would otherwise commit a column it has never had, which is a digest
//! move rather than a refactor.  The second is where a table's inert rows are,
//! which is `lead` here and an `act` column in `embed`.  Neither is in this PR:
//! Freivalds wants all four and pads at the front.
//!
//! [`Carry::Rates`] is what most of the carries want, [`Carry::PerGroup`] and
//! [`Carry::WholeTable`] are the plain cases, and [`Carry::By`] stays the escape
//! hatch.  They are separate arms because they are separate expression trees --
//! a fold that wants `prev * flag` must not be handed `(prev * one) * flag`, or
//! the lowering fingerprints a different tree.  Two differences the table does
//! not show, because they are not the declaration's: `embed` and `argmax` read
//! their closed value off the column at the last row their table *uses* rather
//! than at the last row of the domain, which is the caller's line and not
//! [`carry_down`]'s; and only Freivalds, dequant, attn_out and the
//! residual-stream bindings have a device path at all, so [`place`] is dead
//! weight for the rest until they get one.

use num_traits::{One, Zero};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::EvalAtRow;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::common::{Eval, eval_of, m31_u32};
use crate::layout::Grid;

// ------------------------------------------------------------ the grouping ---

/// How a component's rows are grouped: `lead` inert rows, then one group per
/// whatever the gadget counts, `rows_per_group` rows each, in a table of
/// `2^log_rows` rows.
///
/// This is the whole of what the four preprocessed 0/1 columns say, so it is
/// what a gadget states instead of naming them.  Build one off the layout that
/// tiles the table -- [`Groups::of`] for a [`Grid`], or the fields directly for
/// a tiling that is not one -- and it writes the columns, names them and reads
/// them back.
#[derive(Clone, Copy, Debug)]
pub struct Groups<'a> {
    /// the prefix the four ids carry, so that a column is `{id}_{name}`
    pub id: &'a str,
    pub log_rows: u32,
    /// how many rows one group occupies
    pub rows_per_group: usize,
    /// inert rows ahead of the first group, which belong to no group
    pub lead: usize,
}

impl<'a> Groups<'a> {
    /// The names, in the order the columns are committed in.  They are pinned:
    /// they are the keys a verifier resolves a preprocessed column by, and they
    /// are the names the check has carried since it was written.
    pub const NAMES: [&'static str; 4] = ["nofirst", "last", "notzero", "end"];
    /// What the grouping costs a component's preprocessed tree.
    pub const COLS: usize = Self::NAMES.len();

    /// The grouping a [`Grid`] tiles: one group per `Grid` group, `chunks` rows
    /// each, and the table's padding ahead of the first of them.  Which way the
    /// group *index* runs does not reach the flags, so `descending` is not read
    /// here.
    pub fn of(g: &Grid, id: &'a str) -> Self {
        Groups { id, log_rows: g.log_rows, rows_per_group: g.chunks, lead: g.lead }
    }

    /// The four ids, in committed order.
    pub fn ids(&self) -> Vec<PreProcessedColumnId> {
        Self::NAMES.iter().map(|n| crate::relations::id(format!("{}_{n}", self.id))).collect()
    }

    /// The four columns at this row, off the front of the preprocessed mask.
    ///
    /// A row declaration reads this *first*, before any of the gadget's own
    /// columns: the order the masks are numbered in is the order the generated
    /// kernel is emitted against, and it is the order the flags have been read
    /// in since the check was written.
    pub fn read<E: EvalAtRow>(&self, eval: &mut E) -> Rows<E::F> {
        let mut ids = self.ids().into_iter();
        let mut next = |eval: &mut E| eval.get_preprocessed_column(ids.next().unwrap());
        Rows {
            in_group: next(eval),
            group_end: next(eval),
            in_table: next(eval),
            table_end: next(eval),
        }
    }

    /// The four columns themselves, in committed order.
    pub fn write(&self) -> Vec<Eval> {
        let rows = 1usize << self.log_rows;
        let col = |f: &dyn Fn(usize) -> bool| {
            eval_of(self.log_rows, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
        };
        vec![col(&|j| self.in_group(j)), col(&|j| self.group_end(j)),
             col(&|j| self.in_table(j)), col(&|j| self.table_end(j))]
    }

    /// Which of its group's rows row `j` is, counting *down* so that a group's
    /// last row is 0, or `None` on an inert row.  This is `Grid::row`'s chunk
    /// index, without the group index the flags do not need.
    fn row_in_group(&self, j: usize) -> Option<usize> {
        (j >= self.lead && self.rows_per_group > 0)
            .then(|| self.rows_per_group - 1 - (j - self.lead) % self.rows_per_group)
    }
    /// 1 where a fold restarted on an earlier row of this group
    fn in_group(&self, j: usize) -> bool {
        matches!(self.row_in_group(j), Some(c) if c + 1 < self.rows_per_group)
    }
    /// 1 on a group's last row
    fn group_end(&self, j: usize) -> bool {
        matches!(self.row_in_group(j), Some(0))
    }
    /// 1 on every row but row 0 -- the domain is cyclic
    fn in_table(&self, j: usize) -> bool {
        j > 0
    }
    /// 1 on the table's last row
    fn table_end(&self, j: usize) -> bool {
        j + 1 == 1usize << self.log_rows
    }

    /// Row `j`'s multiplier under a carry rule: the host's reading of
    /// [`Carry`], and the one place the two sides could drift apart.
    ///
    /// Row 0 has no previous row, so it is always zero there -- which is what
    /// the columns say too, `nofirst` and `notzero` both being 0 on row 0.
    fn step(&self, carry: &Carry<PerRow>, j: usize) -> SecureField {
        if j == 0 {
            return SecureField::zero();
        }
        match carry {
            Carry::PerGroup { step } if self.in_group(j) => *step,
            Carry::WholeTable if self.in_table(j) => SecureField::one(),
            Carry::Rates { inside, .. } if self.in_group(j) => *inside,
            Carry::Rates { across, .. } if self.in_table(j) => *across,
            Carry::By(m) => m(j),
            _ => SecureField::zero(),
        }
    }
}

/// The grouping's four columns at one row, as the AIR reads them.
///
/// Its fields are private on purpose: what a gadget does with the grouping is
/// say which carry rule a fold takes, and [`constrain`] is what turns a rule
/// into the cells.  A gadget that has to pin a constraint that is not a fold to
/// one of these rows will want an accessor, and it can have one when it comes.
pub struct Rows<F> {
    in_group: F,
    group_end: F,
    in_table: F,
    table_end: F,
}

// ---------------------------------------------------------- the declaration --

/// `sum_p cells[p] * w[p]`: a row's own cells folded against fixed challenge
/// powers, which is what most `term`s are.
pub fn dot<E: EvalAtRow>(cells: &[E::F], w: &[SecureField]) -> E::EF {
    let mut f = E::EF::zero();
    for (p, c) in cells.iter().enumerate() {
        f += c.clone() * w[p];
    }
    f
}

/// A value the host's side of a fold asks for row by row: it is not asked for
/// on a row that does not take it.
pub type PerRow<'a> = &'a dyn Fn(usize) -> SecureField;

/// What the previous row is multiplied by, as a rule about the grouping.
///
/// The four arms are four expression trees and not four ways of writing one:
/// the lowering fingerprints the tree, so a fold that wants `prev * flag` must
/// not be given `(prev * one) * flag`.  `M` is what a multiplier is to whoever
/// holds the row -- the evaluator's extension type in the AIR, a function of the
/// row index on the host.
pub enum Carry<M> {
    /// `x[j-1] * step * nofirst`: the fold runs at `step` inside a group and
    /// restarts on each group's first row.  The two folds down a Freivalds
    /// input channel are this.
    PerGroup { step: SecureField },
    /// `x[j-1] * notzero`: carried down the whole table at no step, and dropped
    /// only on row 0.  A running total over a table is this.
    WholeTable,
    /// `x[j-1] * (nofirst inside + (notzero - nofirst) across)`: a fold that
    /// runs down the whole table but steps at one rate from one row of a group
    /// to the next and at another where a new group begins.  A Horner over a
    /// stream the table walks in a different order than the stream is numbered
    /// in is this.
    Rates { inside: SecureField, across: SecureField },
    /// `x[j-1] * mult`: a multiplier the caller built out of its own columns,
    /// which is what a table stepping at more rates than the grouping knows
    /// about needs.
    By(M),
}

/// What a row adds to the fold.  `T` is what a term is to whoever holds the row,
/// as `M` is for [`Carry`].
pub enum Term<T> {
    /// added on every row
    Each(T),
    /// `term * last`: added once per group, on the group's last row, where the
    /// folds it is built out of are complete
    PerGroup(T),
}

/// The claim a published column is pinned to.
///
/// Which row it closes on is the grouping's: the table's last.  A fold that has
/// to close somewhere else is not one this declaration states yet.
pub struct Close {
    /// the value the statement carries
    pub claim: SecureField,
}

/// One running column: what carries in, what this row adds, and what it closes
/// on.
pub struct Fold<T> {
    pub carry: Carry<T>,
    pub term: Term<T>,
    /// what the fold is pinned to, or `None` on a column nothing publishes
    pub close: Option<Close>,
}

/// The constraints one fold owns, against the `[-1, 0]` mask the row's reader
/// allocated for it and the grouping the row read.
///
/// The recurrence first and the closing constraint after, with the claim drawn
/// between them -- a lowering numbers claims in the order they are asked for, so
/// that order is part of what this states.
///
/// It is a free function and not a method because the arithmetic is the
/// evaluator's: written as `E::F` and `E::EF` the bounds come from `EvalAtRow`,
/// and written as an impl's own two parameters they would all have to be
/// restated.
pub fn constrain<E: EvalAtRow>(eval: &mut E, rows: &Rows<E::F>, run: Run<E::EF>,
                               fold: Fold<E::EF>) {
    let Fold { carry, term, close } = fold;
    let term = match term {
        Term::Each(t) => t,
        Term::PerGroup(t) => t * rows.group_end.clone(),
    };
    match carry {
        Carry::PerGroup { step } => eval.add_constraint(
            run.cur.clone() - (run.prev * step) * rows.in_group.clone() - term),
        Carry::WholeTable => eval.add_constraint(
            run.cur.clone() - run.prev * rows.in_table.clone() - term),
        Carry::Rates { inside, across } => {
            let mult = E::EF::from(rows.in_group.clone()) * inside
                + (E::EF::from(rows.in_table.clone())
                   - E::EF::from(rows.in_group.clone())) * across;
            eval.add_constraint(run.cur.clone() - run.prev * mult - term)
        }
        Carry::By(mult) =>
            eval.add_constraint(run.cur.clone() - run.prev * mult - term),
    }
    if let Some(c) = close {
        let claim = eval.claim(c.claim);
        eval.add_constraint((run.cur - claim) * rows.table_end.clone());
    }
}

/// One running column, at the two row offsets its constraint reads it at:
/// `prev` is the previous row (`-1`) and `cur` is this one (`0`).
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
        let [prev, cur] = eval.next_extension_interaction_mask(
            stwo_constraint_framework::INTERACTION_TRACE_IDX, [-1, 0]);
        Run { prev, cur }
    }
}

// ------------------------------------------------------- the host's side ----

/// `x[j] = carry(j) * x[j-1] + term[j]`, down `rows` rows of the grouping.
///
/// The per-row terms are the caller's: they are independent of each other and
/// every caller builds them in parallel.  What is serial, and what was written
/// out by hand once per column, is this carry.  The declaration is [`Carry`] and
/// [`Term`] again, over functions of the row index instead of over an
/// evaluator's expressions; there is no [`Close`] here, because the value a
/// fold closes on is what this returns and not something it is told.
pub fn carry_down(groups: &Groups, rows: usize, term: Term<PerRow>,
                  carry: Carry<PerRow>) -> Vec<SecureField> {
    let mut out = vec![SecureField::zero(); rows];
    for j in 0..rows {
        let add = match &term {
            Term::Each(t) => t(j),
            Term::PerGroup(t) if groups.group_end(j) => t(j),
            Term::PerGroup(_) => SecureField::zero(),
        };
        out[j] = if j == 0 { SecureField::zero() } else { groups.step(&carry, j) * out[j - 1] }
                 + add;
    }
    out
}

/// The M31 columns one running column becomes once encoded: a QM31 is four.
///
/// This is what a fold costs the interaction tree, and it is what a gadget's
/// column census counts its running columns in.
pub const M31_COLS: usize = stwo::core::fields::qm31::SECURE_EXTENSION_DEGREE;

/// One built running column, on its way into the interaction tree.
pub fn column(log_rows: u32, values: &[SecureField]) -> Vec<Eval> {
    crate::encoding::secure_cols_pub(log_rows, values)
}

// ----------------------------------------------------- the device's side ----

/// Where a component's group of running columns is built.
///
/// `ZK_DEVICE_ACCUMULATORS` picks on the usual three-value pattern: 0 the host
/// builds them, 1 the card does and they are never on the host, 2 both with
/// every cell asserted equal.  A component whose base trace stayed on the host
/// has no device columns to fold from and is the host's whatever the switch
/// says.
#[derive(Clone, Copy, Debug)]
pub struct Sides {
    /// the host builds them -- and in mirror mode the host's are the ones proved
    pub host: bool,
    /// the card builds them
    pub device: bool,
}

/// Which sides run, for a component whose columns are `on_device` or not.
pub fn sides(on_device: bool) -> Sides {
    let m = crate::devacc::mode();
    let device = on_device && m >= 1;
    Sides { host: !device || m == 2, device }
}

/// Build one component's group of running columns wherever [`sides`] says, put
/// them in the tree, and hand back what they closed on.
///
/// Under mirror mode both sides run and `devacc::place` asserts the card's
/// columns and claims against the host's cell for cell before the host's are the
/// ones committed -- so a mirror run's digest is the host path's digest, and a
/// difference is a panic naming the cell rather than a hash that does not.
pub fn place(
    tag: &str,
    sides: Sides,
    sink: &mut crate::device::ColSink,
    device: impl FnOnce() -> crate::devacc::DevAccCols,
    host: impl FnOnce() -> (Vec<Eval>, Vec<SecureField>),
) -> Vec<SecureField> {
    if !sides.device {
        let (cols, claims) = host();
        sink.extend(cols);
        return claims;
    }
    crate::devacc::place(tag, device(), sides.host.then(host), sink)
}

// ------------------------------------------------------------- the check ----

/// What the grouping and the carry are, written out the way a gadget wrote them
/// before this module existed.
///
/// Two small tests and they are the point of the primitive: the four columns
/// have to be the four a gadget built off its own layout, and the carry rules
/// have to be the hand-written recurrences they replaced, including at row 0,
/// where there is no previous row and every one of them starts from its own term
/// alone.
#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;
    use stwo::prover::backend::Column;

    fn f(n: u32) -> SecureField {
        SecureField::from(M31::from_u32_unchecked(n))
    }

    /// A `Grid` of 3 groups of 5 rows in a table of 16: one inert lead row,
    /// then the groups.
    fn grid() -> Grid {
        Grid::new(3, 4, 5, true).expect("a 3 x 5 grid in 16 rows")
    }

    #[test]
    fn the_columns_are_the_four_the_layout_says() {
        let g = grid();
        let groups = Groups::of(&g, "zk_test");
        assert_eq!((groups.rows_per_group, groups.lead), (5, 1));
        assert_eq!(groups.ids().iter().map(|i| i.id.clone()).collect::<Vec<_>>(),
                   ["zk_test_nofirst", "zk_test_last", "zk_test_notzero", "zk_test_end"]);

        // what a gadget wrote by hand: the chunk index counts down inside a
        // group, so chunk `chunks - 1` is a group's first row and chunk 0 its last.
        // The expected column goes through `eval_of` too, because a built column
        // holds its values in the domain's bit-reversed order and not in row order.
        let rows = 1usize << g.log_rows;
        let want = |f: &dyn Fn(usize) -> bool| {
            eval_of(g.log_rows, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
                .values.to_cpu()
        };
        let chunk = |j: usize| g.row(j).map(|(_, c)| c);
        let cols: Vec<Vec<M31>> = groups.write().iter().map(|e| e.values.to_cpu()).collect();
        assert_eq!(cols[0], want(&|j| matches!(chunk(j), Some(c) if c + 1 < g.chunks)));
        assert_eq!(cols[1], want(&|j| matches!(chunk(j), Some(0))));
        assert_eq!(cols[2], want(&|j| j > 0));
        assert_eq!(cols[3], want(&|j| j + 1 == rows));
    }

    #[test]
    fn the_carry_is_the_hand_written_recurrence() {
        let g = grid();
        let groups = Groups::of(&g, "zk_test");
        let rows = 1usize << g.log_rows;
        let term: Vec<SecureField> = (1..=rows as u32).map(f).collect();
        let at = |j: usize| term[j];
        let restart = |j: usize| matches!(g.row(j).map(|(_, c)| c), Some(c) if c + 1 < g.chunks);
        let step = f(7);

        // `x[j] = (nofirst ? x[j-1] * step : 0) + term[j]`
        let mut want = vec![SecureField::zero(); rows];
        for j in 0..rows {
            want[j] = if j > 0 && restart(j) { want[j - 1] * step } else { SecureField::zero() }
                      + term[j];
        }
        assert_eq!(carry_down(&groups, rows, Term::Each(&at), Carry::PerGroup { step }), want);

        // `x[j] = (j > 0 ? x[j-1] : 0) + (last ? term[j] : 0)`: a running total
        // over a table that takes a value once per group
        let last = |j: usize| matches!(g.row(j).map(|(_, c)| c), Some(0));
        let mut want = vec![SecureField::zero(); rows];
        for j in 0..rows {
            want[j] = if j > 0 { want[j - 1] } else { SecureField::zero() }
                      + if last(j) { term[j] } else { SecureField::zero() };
        }
        assert_eq!(carry_down(&groups, rows, Term::PerGroup(&at), Carry::WholeTable), want);

        // and the two-rate carry, which a gadget wrote as a multiplier of its own
        let (inside, across) = (step, f(5));
        let mult = |j: usize| if restart(j) { inside }
                              else if j > 0 { across }
                              else { SecureField::zero() };
        let mut want = vec![SecureField::zero(); rows];
        for j in 0..rows {
            want[j] = if j > 0 { want[j - 1] * mult(j) } else { SecureField::zero() } + term[j];
        }
        assert_eq!(carry_down(&groups, rows, Term::Each(&at), Carry::Rates { inside, across }),
                   want);
        assert_eq!(carry_down(&groups, rows, Term::Each(&at), Carry::By(&mult)), want);
    }
}
