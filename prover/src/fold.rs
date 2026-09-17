//! One running column, and the recurrence that fills it (#221).
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
//! A [`Fold`] says what the column is:
//!
//! - [`Carry`] is what the previous row is multiplied by -- a challenge power
//!   that a 0/1 flag switches off where the fold restarts, a bare flag, or a
//!   multiplier the caller built out of its own flags;
//! - `term` is this row's own contribution, usually [`dot`] of the row's cells
//!   against fixed powers;
//! - [`Close`] is the claim the last row is pinned to, and the flag that says
//!   which row that is.  A column nothing publishes has none.
//!
//! [`constrain`] emits the one or two constraints, [`carry_down`] runs the
//! recurrence on the host, [`column`] puts the result in the tree and
//! [`M31_COLS`] is what it costs there.  [`place`] is the same group of columns
//! built on the card instead, with the mirror comparison behind it.
//!
//! # What this covers, and what it does not
//!
//! Freivalds is the first gadget through it (#221's trial).  Its four columns
//! are `x_fold` and `w_fold` ([`Carry::Restart`]), `product` ([`Carry::Gate`])
//! and `x_claim` ([`Carry::By`]).  The rest of the crate's running columns, and
//! what each would take -- this is the inventory, not a promise, and no other
//! gadget is converted here:
//!
//! | column | carry | term | as is? |
//! |---|---|---|---|
//! | `dequant` `z` | `nofirst * rv_step` | the accumulator cells against `rv` | `Restart` |
//! | `dequant` `h` | `m1 + m2 s` | `z * last` -- another running column | `By`, and the term is not a [`dot`] |
//! | `dequant` `u`, `v` | two flags, two powers | a bf16 decode, a shared pair | `By` |
//! | `attn_out` `z` | `nofirst * rv_step` | the accumulator cells against `rv` | `Restart` |
//! | `attn_out` `h`, `dz` | three flags, three powers | `z * last`, `den * last` | `By` |
//! | `attn_out` `zq`, `hq` | as `z`, `h` | as `z`, `h` | as `z`, `h` |
//! | `requant` `ui`, `uo` | `nofirst k_in + (gt0 - nofirst) k_bd` | the row's cells against `w` | `By` |
//! | `requant` `us` | `gt0 + mlast (mu - 1)` | a shared pair times `last` | `By` |
//! | `rope` `ui` | `gt0 * step_in` | the row's cells against `w_in` | `Restart` |
//! | `rope` `uo` | `gt0 s + hstart (s_h - s)` | the row's cells against `w_out` | `By` |
//! | `embed` `h`, `g` | `s_in k_in + s_bd k_bd`, `s_in + s_bd k_tok` | the row's slots against `w`; one cell gated by `1 - s_in` | `By` |
//! | `argmax` `u`, `g` | `oin k_in + obd k_bd`, `oin + obd k_tok` | as `embed`'s two | `By`, over another component's selectors |
//! | `argmax` `ss` | `nofirst` | indicator bits, no powers | not a fold: base field, and it closes on a committed cell rather than on a claim |
//! | `residual_stream` bindings | one power, no segmentation | the witness, not a column | `Restart` with the flag always 1 |
//!
//! So [`Carry::By`] is what most of them want and the two named shapes are the
//! plain cases.  Two differences the table does not show, because they are not
//! the declaration's: `embed` and `argmax` read their closed value off the
//! column at the last row their table *uses* rather than at the last row of the
//! domain, which is the caller's line and not [`carry_down`]'s; and only
//! Freivalds, dequant, attn_out and the residual-stream bindings have a device
//! path at all, so [`place`] is dead weight for the rest until they get one.
//! `argmax`'s `ss` is the one running column here that is not a fold and should
//! stay hand-written.

use num_traits::{One, Zero};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::EvalAtRow;

use crate::common::Eval;

/// `sum_p cells[p] * w[p]`: a row's own cells folded against fixed challenge
/// powers, which is what most `term`s are.
pub fn dot<E: EvalAtRow>(cells: &[E::F], w: &[SecureField]) -> E::EF {
    let mut f = E::EF::zero();
    for (p, c) in cells.iter().enumerate() {
        f += c.clone() * w[p];
    }
    f
}

/// What the previous row is multiplied by.
///
/// The three arms are three expression trees and not three ways of writing one:
/// the lowering fingerprints the tree, so a fold that wants `prev * flag` must
/// not be given `(prev * one) * flag`.
pub enum Carry<F, EF> {
    /// `x[j-1] * step * flag`: the fold runs at `step` and restarts where the
    /// flag is 0.  The two folds down a Freivalds input channel are this.
    Restart { step: SecureField, flag: F },
    /// `x[j-1] * flag`: carried where the flag is 1 and dropped where it is 0,
    /// at no step.  A running total over a whole table is this, the flag being
    /// "not row 0" -- the trace domain is cyclic and row 0's previous row is the
    /// table's last.
    Gate(F),
    /// `x[j-1] * mult`: a multiplier the caller built out of its own flags,
    /// which is what a table whose rows step at more than one rate needs.
    By(EF),
}

/// The claim a published column's last row is pinned to.
pub struct Close<F> {
    /// the value the statement carries
    pub claim: SecureField,
    /// 1 on the row the fold closes on and 0 elsewhere
    pub at: F,
}

/// One running column: what carries in, what this row adds, and what it closes
/// on.
pub struct Fold<F, EF> {
    pub carry: Carry<F, EF>,
    /// this row's own contribution
    pub term: EF,
    /// what the last row is pinned to, or `None` on a column nothing publishes
    pub close: Option<Close<F>>,
}

/// The constraints one fold owns, against the `[-1, 0]` mask the row's reader
/// allocated for it.
///
/// The recurrence first and the closing constraint after, with the claim drawn
/// between them -- a lowering numbers claims in the order they are asked for, so
/// that order is part of what this states.
///
/// It is a free function and not a method because the arithmetic is the
/// evaluator's: written as `E::F` and `E::EF` the bounds come from `EvalAtRow`,
/// and written as an impl's own two parameters they would all have to be
/// restated.
pub fn constrain<E: EvalAtRow>(eval: &mut E, run: Run<E::EF>, fold: Fold<E::F, E::EF>) {
    let Fold { carry, term, close } = fold;
    match carry {
        Carry::Restart { step, flag } =>
            eval.add_constraint(run.cur.clone() - (run.prev * step) * flag - term),
        Carry::Gate(flag) =>
            eval.add_constraint(run.cur.clone() - run.prev * flag - term),
        Carry::By(mult) =>
            eval.add_constraint(run.cur.clone() - run.prev * mult - term),
    }
    if let Some(c) = close {
        let claim = eval.claim(c.claim);
        eval.add_constraint((run.cur - claim) * c.at);
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

/// What the previous row is multiplied by, on the host: the same three shapes
/// [`Carry`] has in the AIR, over a row index instead of over an evaluator.
pub enum Step<'a> {
    /// `step` where the flag is 1, and nothing where it is 0
    Restart { step: SecureField, flag: &'a dyn Fn(usize) -> bool },
    /// the previous row itself where the flag is 1, and nothing where it is 0
    Gate(&'a dyn Fn(usize) -> bool),
    /// whatever the caller says
    By(&'a dyn Fn(usize) -> SecureField),
}

impl Step<'_> {
    /// Row `j`'s multiplier.  Row 0 has no previous row, so it is always zero
    /// there -- which is what the flags a layout builds say too, and what the
    /// cyclic domain makes the AIR say with `carry_in_table`.
    pub fn at(&self, j: usize) -> SecureField {
        if j == 0 {
            return SecureField::zero();
        }
        match self {
            Step::Restart { step, flag } if flag(j) => *step,
            Step::Gate(flag) if flag(j) => SecureField::one(),
            Step::By(m) => m(j),
            _ => SecureField::zero(),
        }
    }
}

/// `x[j] = step(j) * x[j-1] + term[j]`, down the rows.
///
/// The per-row terms are the caller's: they are independent of each other and
/// every caller builds them in parallel.  What is serial, and what was written
/// out by hand once per column, is this carry.
pub fn carry_down(term: &[SecureField], step: Step) -> Vec<SecureField> {
    let mut out = vec![SecureField::zero(); term.len()];
    for j in 0..term.len() {
        out[j] = if j == 0 { SecureField::zero() } else { step.at(j) * out[j - 1] } + term[j];
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

/// What `carry_down` is, written out the way the four Freivalds columns were
/// written before it existed.
///
/// It is a small test and it is the point of the primitive: the three shapes
/// have to agree with the hand-written recurrences they replaced, including at
/// row 0, where there is no previous row and every one of them starts from its
/// own term alone.
#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;

    fn f(n: u32) -> SecureField {
        SecureField::from(M31::from_u32_unchecked(n))
    }

    #[test]
    fn the_carry_is_the_hand_written_recurrence() {
        let term: Vec<SecureField> = (1..=8u32).map(f).collect();
        let restart = |j: usize| j % 3 != 0;
        let step = f(7);

        // `x[j] = (restart ? x[j-1] * step : 0) + term[j]`
        let mut want = vec![SecureField::zero(); term.len()];
        for j in 0..term.len() {
            want[j] = if j > 0 && restart(j) { want[j - 1] * step } else { SecureField::zero() }
                      + term[j];
        }
        assert_eq!(carry_down(&term, Step::Restart { step, flag: &restart }), want);

        // `x[j] = (j > 0 ? x[j-1] : 0) + term[j]`: a running total over a table
        let in_table = |j: usize| j > 0;
        let mut want = vec![SecureField::zero(); term.len()];
        for j in 0..term.len() {
            want[j] = if j > 0 { want[j - 1] } else { SecureField::zero() } + term[j];
        }
        assert_eq!(carry_down(&term, Step::Gate(&in_table)), want);

        // and a multiplier the caller built, at two rates
        let mult = |j: usize| if restart(j) { step } else { f(5) };
        let mut want = vec![SecureField::zero(); term.len()];
        for j in 0..term.len() {
            want[j] = if j > 0 { want[j - 1] * mult(j) } else { SecureField::zero() } + term[j];
        }
        assert_eq!(carry_down(&term, Step::By(&mult)), want);
    }
}
