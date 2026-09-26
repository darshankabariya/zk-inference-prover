//! Freivalds' check that a matmul's committed product is the product it claims
//! to be, as a STARK component.
//!
//! # The statement
//!
//! One matmul is `C = X W`: activations times weights.  Proving that
//! multiplication cell by cell would cost a constraint per multiply-add.
//! Freivalds' check replaces it with one projection: draw two random vectors
//! after both sides are fixed, and check
//!
//! ```text
//!     s^T C rv  ==  (s^T X)(W rv)
//! ```
//!
//! With `rv` weighting output channels `i` and `s` weighting tokens `t`, and the
//! transposes left implicit, that is
//!
//! ```text
//!     sum_{t,i} s^t rv^i C[t, i]  ==  sum_k (sum_i rv^i W[k, i]) (sum_t s^t X[t, k])
//! ```
//!
//! The right-hand side is two matrix-vector products instead of a matrix-matrix
//! one.  The two vectors are the powers of one challenge each -- `rv^i` over
//! output channels and `s^t` over tokens -- so the check is a bivariate
//! polynomial identity of degree `n_out - 1` in `rv` and `tokens - 1` in `s`,
//! and a wrong `C` passes with probability at most `(n_out + tokens)/|F|` over
//! the draw, `F` being the degree-4 extension the challenges are drawn in.  This
//! component proves the right-hand side.
//!
//! The equality is a field one, so it holds only mod `p`: it certifies the
//! integer matmul just while every accumulated magnitude stays inside one
//! residue window.  That is the dequant gadget's `Spec::sound()`, which requires
//! `max_abs() <= p` so the integer reading is injective mod `p` (B4 §1), and
//! `Spec::max_k()` turns it into the longest int8 dot product that still fits,
//! `K <= 66,572` for a `2^30` window.  Its four parts:
//!
//! - **M, the weights `W`.** Preprocessed: the verifier holds them itself, so
//!   there is nothing to commit and nothing a prover could swap.  They enter as
//!   the `weights` columns.
//! - **A, the activations `X`.** Committed here, as bytes offset into a range
//!   table, one token chunk per row.  Because they are this component's own
//!   commitment, the component also *publishes* a projection of them
//!   (`x_claim`), so that whichever component produced `X` can be bound to the
//!   same values.  A segment with no producer publishes none.
//! - **C, the product.** Not committed here at all.  The dequant gadget commits
//!   it -- it is the accumulator dequant rounds -- and publishes the projection
//!   `s^T C rv` as its claim `claimed_h`, computed through
//!   `gadgets::freivalds::claims::project`.  This component consumes that claim.
//! - **r = (rv, s), the challenges.** Drawn from the transcript *after* every
//!   commitment, so neither side can be chosen against them.  `rv` weights
//!   output channels, `s` weights tokens.
//!
//! The equality itself is `claims::identity_holds`, which `compose::verify_composed`
//! calls: per matmul, this component's `product` claim against the dequant
//! component's `claimed_h`.  Neither side commits `C` twice; the whole check is
//! that two independently accumulated field elements agree.
//!
//! # The table
//!
//! One row group per input channel `k` -- one coordinate of the hidden
//! dimension, the index the matmul sums over -- cut into `chunks` rows, and the
//! groups are padded up to a power of two with inert rows at the top.  A row
//! carries `pk` weights of each matmul and `pq` committed activation cells.
//! ("Chunk" in this directory is always one of those rows' share of an input
//! channel, never a proved segment of the model.)  Down an input channel's rows
//! the component folds
//!
//! ```text
//!     x_fold    = sum_t s^t  X[t, k]          (shared by every matmul)
//!     w_fold_m  = sum_i rv^i W_m[k, i]        (one per matmul)
//! ```
//!
//! and on the input channel's last row adds `w_fold_m * x_fold` into `product_m`,
//! which is `(s^T X)(W_m rv)` once every input channel has been walked.  Matmuls that
//! read one committed input share a group -- q/k/v, and gate/up -- so `X` is
//! committed once for all of them.
//!
//! # The loop, and who owns it
//!
//! A STARK has one set of constraints and they run on *every* row, so the loop
//! an arkworks circuit would write out has to be expressed as data: preprocessed
//! 0/1 columns saying where an input channel begins, where it ends, and where
//! the table begins and ends.  None of that is a fact about Freivalds' check --
//! it is a fact about a table whose rows come in groups -- so this gadget states
//! its *grouping* and `crate::fold::Groups` owns the columns.  One row group per
//! input channel, `chunks` rows each, in a table of `2^log_rows` rows:
//! `columns::FreivaldsShape::groups` is the one place that is said, the
//! primitive writes the four columns, names them and reads them back, and
//! nothing in this directory names one of them.
//!
//! What the constraints below say instead is which *rule* about the grouping
//! each fold carries under: `Carry::PerGroup` restarts at each input channel's
//! first row, `Carry::WholeTable` carries the product down the whole table,
//! `Term::PerGroup` adds the two finished folds in once per input channel on its
//! last row, `Carry::Rates` steps the published activation claim at one rate
//! inside an input channel and another where a new one begins, and `Close` pins
//! a fold to its claim on the table's last row.  The four columns' pinned ids
//! are in the table below; they did not change.
//!
//! A padding row is in no input channel: its folds are zero and it passes the
//! product through unchanged.
//!
//! # The files
//!
//! The split between the two builders is the *transcript phase*: what is
//! committed before the challenges are drawn, and what is computed after.
//!
//! - `mod.rs` -- this comment, `FreivaldsEval` and its constraints.
//! - `witness.rs` -- the integers the check is proved over: `A`, each matmul's
//!   `M`, and where the layer's own list holds them.  The one place in this
//!   directory that names a `Layer`.
//! - `columns.rs` -- the column declaration: one struct per tree, the reader the
//!   AIR uses, the committed order the builders use, and the shape that states
//!   the group's name, its tiling and its two row widths once.
//! - `layout.rs` -- which matmuls share a group, and how the rows are tiled.
//! - `trace.rs` -- *before the challenges*: the committed activation bytes and
//!   the preprocessed weights.  The grouping's own columns are written beside
//!   them by the fold primitive.
//! - `claims.rs` -- *after the challenges*: the values the statement carries
//!   (`FreivaldsClaims`), the equality over them (`identity_holds`), the
//!   projection they are compared at (`project`), and the running sums.
//! - `device.rs` -- the same columns `trace.rs` builds, built on the GPU, and
//!   the switch that says which of the two runs.  Nothing here but the module
//!   list below names it, and `trace.rs` is its only caller: which side built a
//!   column is not something a constraint can see.
//! - `run.rs` -- the standalone proof of this component alone, and what it costs
//!   per weight.
//! - `tests.rs` -- the row-level tests, the controls, and the identity's.
//!
//! # What this gadget does not say
//!
//! Three things a Freivalds check has no opinion about are declared here and
//! implemented elsewhere, so that a reader of this directory meets them once
//! each and can read the primitive on its own (#221):
//!
//! - **The running sums are folds.** `crate::fold::Fold` is one running column:
//!   what carries in from the previous row, what this row adds, and the claim
//!   the last row is pinned to.  The four columns below are four of those, and
//!   the two constraints each one owns, the serial recurrence that fills it, and
//!   its place in the interaction tree all come from that one declaration.
//!   Where the columns are built -- on the host, on the card, or both with every
//!   cell compared -- is the fold's too.
//! - **The activations are bytes.** `crate::columns::Bytes` is the cell kind for
//!   committed values of a range table, and `crate::columns::ByteCells` is the
//!   table: the width, the offset a signed value carries to land inside it, the
//!   relation, and *who proves the lookups*.  Whether they are proved by this
//!   component's own LogUp columns or by the batched backend is a property of
//!   the lookup and not of this check, so the word for the second backend does
//!   not appear in this directory.
//! - **The layout's loop is the grouping.** `crate::fold::Groups` is how a
//!   table's rows are grouped -- how many rows a group takes, how tall the
//!   table is, how many rows it is padded with, and the prefix its columns are
//!   named under.  It emits the four preprocessed 0/1 columns on the host,
//!   names them and reads them in the AIR, and the carry rules a fold declares
//!   are rules about it.  So this gadget states its tiling once and never names
//!   a flag.
//! - **The challenge powers are the fold's.** A Horner over a stream cut into
//!   rows runs at `c^p` across a row's cells and at `c^width` from one row to
//!   the next, and both follow from the challenge and the width.
//!   `crate::fold::Powers::against` derives them, so the AIR names `rv` and `s`
//!   and carries no precomputed vector.
//! - **The framework's own methods follow from two facts.** `crate::air::Air`
//!   is where a component's rows are and what its generated kernel is called;
//!   `crate::framework_boilerplate!` emits `log_size`,
//!   `max_constraint_log_degree_bound`, `cuda_eval_name` and `cuda_eval_blob`
//!   from them.  `crate::air::Ids` is where a preprocessed column is named.
//! - **Proving alone is a harness.** `crate::alone::prove_alone` is the session
//!   every gadget's standalone run repeats -- commit, draw, build, commit, mix,
//!   prove -- so `run.rs` is the witness, the call and the outcome.
//!
//! # The columns
//!
//! `{tag}` is the group's name (`qkv`, `o`, `gateup`, `down`, `head`).  The id
//! strings are pinned: they are the keys a verifier resolves a preprocessed
//! column by, so the Rust fields were renamed and they were not.
//!
//! | field | holds | tree | pinned id |
//! |---|---|---|---|
//! | `known.weights[m][p]` | `W_m[k, i]` | pre | `zk_freivalds{tag}_w{m}_{p:04}` |
//! | the grouping's, `fold::Groups` | 1 after an input channel's first row | pre | `zk_freivalds{tag}_nofirst` |
//! | the grouping's, `fold::Groups` | 1 on an input channel's last row | pre | `zk_freivalds{tag}_last` |
//! | the grouping's, `fold::Groups` | 1 on every row but row 0 | pre | `zk_freivalds{tag}_notzero` |
//! | the grouping's, `fold::Groups` | 1 on the table's last row | pre | `zk_freivalds{tag}_end` |
//! | `acts.x.cells[p]` | `X[t, k] + acts.off` | base | -- |
//! | `sums.x_fold` | `s^T X` so far | inter | -- |
//! | `sums.mats[m].w_fold` | `W_m rv` so far | inter | -- |
//! | `sums.mats[m].product` | `(s^T X)(W_m rv)` so far | inter | -- |
//! | `sums.x_claim` | the published projection of `X` | inter | -- |
//!
//! `pre` is the preprocessed tree (the verifier holds it: `pk` weights a matmul
//! and the grouping's four columns, which the fold primitive emits under the
//! ids above), `base` the committed trace (`pq` activation cells a row), `inter`
//! the interaction tree (the running sums, each one QM31 column).  The four ids
//! are the only place this directory writes those names down.
//!
//! # Glossary
//!
//! The names changed; nothing they name did.
//!
//! | was | is | id string |
//! |---|---|---|
//! | `FreivaldsPre` | `Known` | -- |
//! | `FreivaldsBase` | `Activations` | -- |
//! | `FreivaldsInter` | `RunningSums` | -- |
//! | `FreivaldsMat` | `Matmul` | -- |
//! | `pre.nofirst`, `known.carry_in_input_channel` | the grouping's, read by `Carry::PerGroup` and `Carry::Rates` | `zk_freivalds{tag}_nofirst` |
//! | `pre.last`, `known.input_channel_done` | the grouping's, read by `Term::PerGroup` | `zk_freivalds{tag}_last` |
//! | `pre.notzero`, `known.carry_in_table` | the grouping's, read by `Carry::WholeTable` and `Carry::Rates` | `zk_freivalds{tag}_notzero` |
//! | `pre.end`, `known.table_done` | the grouping's, read by `Close` | `zk_freivalds{tag}_end` |
//! | `columns::RowFlags` | `fold::Groups` | -- |
//! | `pre.w` | `known.weights` | `zk_freivalds{tag}_w{m}_{p:04}` |
//! | `base.act` | `acts.x.cells` | -- |
//! | `inter.v` | `sums.x_fold` | -- |
//! | `inter.mats[m].u` | `sums.mats[m].w_fold` | -- |
//! | `inter.mats[m].acc` | `sums.mats[m].product` | -- |
//! | `inter.qx` | `sums.x_claim` | -- |
//! | `tag`, `g`, `pk`, `pq`, `n_mats` on the AIR | `shape` | -- |
//! | `rv_pow`, `rv_step` | `rv`, a `fold::Powers` | -- |
//! | `s_pow`, `s_step` | `s`, a `fold::Powers` | -- |
//! | `claimed[m]` | `claims.product[m]` | -- |
//! | `qx.claimed` | `claims.x_claim` | -- |
//!
//! The generic parameter on those structs is named `Cell`, not `C`: it is what
//! one column's cell is to whoever holds the row, and it is nothing to do with
//! the product matrix `C` above.

pub mod claims;
pub mod columns;
pub mod device;
pub mod layout;
pub mod run;
pub mod trace;
pub mod witness;

use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval};

use crate::air::Air;
use crate::columns::ByteCells;
use crate::fold::{Carry, Close, Fold, Powers, Term, constrain};
use crate::gadgets::freivalds::claims::FreivaldsClaims;
use crate::gadgets::freivalds::columns::{FreivaldsRow, FreivaldsShape};

// The attention-output component is `gadgets/attn_out` since #197, and
// `compose.rs`, `cli.rs` and `gkr_key_tile.rs` still reach its eval and its
// preprocessed ids through this module.
pub use crate::gadgets::attn_out::air::{AO_SEL, AttnOutEval, EPOW_COLS, ao_chan, ao_sel};

/// The weights and row steps of the claim this component publishes over its own
/// committed activations, so whichever component produced them can be bound to
/// the same values.
///
/// The claim is a Horner run over the activations in the canonical
/// input-channel-major stream order `q = c * tokens + t`, which is the order a
/// producing component publishes its own claim in, so the two are comparable
/// value for value.
///
/// The table runs `(input channel, token chunk)` down the rows with the tokens
/// in the slots, so the row step takes two values -- `m_in` inside one input
/// channel's chunks and `m_bd` where a new input channel begins -- which is the
/// shape `dequant::OutputClaim` already runs at.  That is `fold::Carry::Rates`,
/// a rule about the table's grouping and nothing this component has to spell
/// out: it costs no preprocessed column of its own, because the grouping
/// already says where an input channel begins.
///
/// The weights are not one challenge's powers -- they are the counterparty's
/// convention, which is why this is not a `fold::Powers` -- and the value the
/// fold closes on is not here but in `claims::FreivaldsClaims`, with the rest
/// of what the statement carries.
#[derive(Clone)]
pub struct FreivaldsQx {
    /// one weight per token slot
    pub w: Vec<SecureField>,
    /// the row step inside one input channel's chunks
    pub m_in: SecureField,
    /// the row step where a new input channel begins
    pub m_bd: SecureField,
}

// ------------------------------------------ the check's AIR, one component --

/// One Freivalds group's constraints: the `(s^T X)(W rv)` side of the identity
/// at the top of this file.
#[derive(Clone)]
pub struct FreivaldsEval {
    /// the group's name, its row tiling and its two row widths, stated once
    pub shape: FreivaldsShape,
    /// `rv`, the challenge that weights output channels, at the `pk` weights a
    /// row carries
    pub rv: Powers,
    /// `s`, the challenge that weights tokens, at the `pq` activations a row
    /// carries
    pub s: Powers,
    /// What the statement carries.  The AIR reads the published half for the
    /// rows its folds close on and nothing else; the values themselves are
    /// public input and their own object.
    pub claims: FreivaldsClaims,
    /// The committed activations as what they are: bytes of a range table, 8
    /// bits wide everywhere but the widened attention output's group, whose
    /// activations are one cell in `[1, 2^12)` against a 12-bit table rather
    /// than two limbs.  Everything that follows from a cell being a table's
    /// value -- the offset it carries so a signed activation lands in an
    /// unsigned table, the relation, the lookups and which backend proves them
    /// -- is in `crate::columns::ByteCells` and nowhere in this gadget.
    pub acts: ByteCells,
    /// How the claim over this component's own activations is folded, or `None`
    /// on a segment with no producer to bind to -- which is what keeps such a
    /// segment's proof exactly what it was.
    pub qx: Option<FreivaldsQx>,
}

impl Air for FreivaldsEval {
    const KERNEL: &'static str = "zk_freivalds_eval";
    fn log_rows(&self) -> u32 {
        self.shape.lay.g.log_rows
    }
}

impl FrameworkEval for FreivaldsEval {
    crate::framework_boilerplate!();

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // The columns come out of the one declaration in `columns.rs`, in the
        // order it declares them.  Nothing below names a column by position,
        // and `rows` is the table's grouping, which the fold primitive reads
        // and its carry rules resolve against.
        let FreivaldsRow { rows, known, acts, sums } =
            FreivaldsRow::read(&mut eval, &self.shape);
        let x = acts.x;

        // x_fold: this row's activations folded against s, on top of the
        // previous row's fold shifted by s^pq -- except on an input channel's
        // first row, where the fold restarts.  What is folded is the signed
        // activation, which is the byte cells' business: a committed cell
        // carries the table's offset and this check never says so.
        let x_fold = sums.x_fold.cur.clone();
        constrain(&mut eval, &rows, sums.x_fold, Fold {
            carry: self.s.per_group(),
            term: Term::Each(self.acts.fold_signed::<E>(&x, &self.s.cells)),
            close: None,
        });

        for (m, mat) in sums.mats.into_iter().enumerate() {
            // w_fold_m: this row's weights folded against rv, on the same carry
            let w_fold = mat.w_fold.cur.clone();
            constrain(&mut eval, &rows, mat.w_fold, Fold {
                carry: self.rv.per_group(),
                term: Term::Each(self.rv.dot::<E>(&known.weights[m])),
                close: None,
            });
            // product_m: the two folds multiplied and added in once per input
            // channel, on the input channel's last row and carried otherwise,
            // and closing on the claim the statement carries
            constrain(&mut eval, &rows, mat.product, Fold {
                carry: Carry::WholeTable,
                term: Term::PerGroup(w_fold * x_fold.clone()),
                close: Some(Close { claim: self.claims.product[m] }),
            });
        }
        // The published claim over the committed activations, allocated before
        // the LogUp columns so that the interaction order is claims first and
        // fractions after, as the components it is compared against already are.
        // Its rows step at two rates -- one from one row of an input channel to
        // the next and one where a new input channel begins -- which is a rule
        // about the grouping and so one the fold states.
        if let Some(q) = &self.qx {
            let x_claim = sums.x_claim
                .expect("the row carries the activation claim exactly when the AIR does");
            let claim = self.claims.x_claim
                .expect("the statement carries the activation claim exactly when the AIR does");
            constrain(&mut eval, &rows, x_claim, Fold {
                carry: Carry::Rates { inside: q.m_in, across: q.m_bd },
                term: Term::Each(self.acts.fold_signed_per_cell::<E>(&x, &q.w)),
                close: Some(Close { claim }),
            });
        }
        // and the activations are a range table's values, which is the cell
        // kind's business and not this check's
        self.acts.look(&mut eval, &x);
        eval
    }
}

/// The `r^T W` check: that a matmul's committed input times the pinned weights
/// is its committed accumulator.
///
/// The lifetime is the witness's: what this gadget is proved over is a *view*
/// of the layer the composer holds (`witness::FreivaldsWitness`), because a
/// group's weights are megabytes and copying them per chunk is not what a
/// witness type is for.
pub struct Freivalds<'a>(std::marker::PhantomData<&'a ()>);

/// Which of the layer's matmul groups the check is over is the witness's now;
/// what is left is how wide a row is cut.
pub struct Cfg {
    /// the weights a row carries, which the tiling is chosen closest to
    pub p_target: usize,
}

impl<'a> crate::gadgets::Gadget for Freivalds<'a> {
    type Witness = witness::FreivaldsWitness<'a>;
    type Cfg = Cfg;
    /// What a Freivalds control corrupts is a weight or an activation of the
    /// witness it is handed, not a value beside it.
    type Tamper = ();
    type Outcome = run::Outcome;

    fn run(w: &witness::FreivaldsWitness<'a>, cfg: &Cfg, _bad: Option<()>) -> run::Outcome {
        run::run(w, cfg.p_target)
    }

    fn accepts(o: &run::Outcome) -> bool {
        o.accepts()
    }
}

/// The row-level tests are the SIMD backend's: on the CUDA build
/// `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS` and `ZK_GENERATED_LOGUP` default
/// to the device, which has no `corrupt_qx` path and no host fraction builder,
/// so a component built here would be built by a different generator than the
/// one under test.  The digests still run on both backends.
#[cfg(all(test, not(feature = "cuda")))]
mod tests;
