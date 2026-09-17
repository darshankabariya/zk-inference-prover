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
//!     rv^T C s  ==  (rv^T W)(X s)
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
//! component proves the right-hand side.  Its four parts:
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
//!   `rv^T C s` as its claim `claimed_h`, computed through
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
//!     x_fold    = sum_t s^t  X[k, t]          (shared by every matmul)
//!     w_fold_m  = sum_i rv^i W_m[i, k]        (one per matmul)
//! ```
//!
//! and on the input channel's last row adds `w_fold_m * x_fold` into `product_m`,
//! which is `(rv^T W_m)(X s)` once every input channel has been walked.  Matmuls that
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
//! - `columns.rs` -- the column declaration: one struct per tree, the reader the
//!   AIR uses, the committed order the builders use, and the grouping the table
//!   is tiled by.
//! - `layout.rs` -- which matmuls share a group, and how the rows are tiled.
//! - `trace.rs` -- *before the challenges*: the committed activation bytes and
//!   the preprocessed weights.  The grouping's own columns are written beside
//!   them by the fold primitive.
//! - `claims.rs` -- *after the challenges*: the running sums, the claims this
//!   component publishes, the lookups' columns, and `project`.
//! - `device.rs` -- the same columns `trace.rs` builds, built on the GPU, and
//!   the switch that says which of the two runs.  Nothing here but the module
//!   list below names it, and `trace.rs` is its only caller: which side built a
//!   column is not something a constraint can see.
//! - `run.rs` -- the standalone proof of this component alone, and what it costs
//!   per weight.
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
//!
//! # The columns
//!
//! `{tag}` is the group's name (`qkv`, `o`, `gateup`, `down`, `head`).  The id
//! strings are pinned: they are the keys a verifier resolves a preprocessed
//! column by, so the Rust fields were renamed and they were not.
//!
//! | field | holds | tree | pinned id |
//! |---|---|---|---|
//! | `known.weights[m][p]` | `W_m[i, k]` | pre | `zk_freivalds{tag}_w{m}_{p:04}` |
//! | the grouping's, `fold::Groups` | 1 after an input channel's first row | pre | `zk_freivalds{tag}_nofirst` |
//! | the grouping's, `fold::Groups` | 1 on an input channel's last row | pre | `zk_freivalds{tag}_last` |
//! | the grouping's, `fold::Groups` | 1 on every row but row 0 | pre | `zk_freivalds{tag}_notzero` |
//! | the grouping's, `fold::Groups` | 1 on the table's last row | pre | `zk_freivalds{tag}_end` |
//! | `acts.x.cells[p]` | `X[k, t] + acts.off` | base | -- |
//! | `sums.x_fold` | `X s` so far | inter | -- |
//! | `sums.mats[m].w_fold` | `rv^T W_m` so far | inter | -- |
//! | `sums.mats[m].product` | `(rv^T W_m)(X s)` so far | inter | -- |
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

use num_traits::Zero;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval};

use crate::columns::ByteCells;
use crate::fold::{Carry, Close, Fold, Term, constrain, dot};
use crate::gadgets::freivalds::columns::{FreivaldsRow, FreivaldsShape};
use crate::layout::Grid;
use crate::relations::id;

// The attention-output component is `gadgets/attn_out` since #197, and
// `compose.rs`, `cli.rs` and `gkr_key_tile.rs` still reach its eval and its
// preprocessed ids through this module.
pub use crate::gadgets::attn_out::air::{AO_SEL, AttnOutEval, EPOW_COLS, ao_chan, ao_sel};

/// The prefix every one of this component's preprocessed ids carries.  The row
/// grouping's four columns are `{prefix}_{name}` for the names `crate::fold`
/// pins, and it emits them; the weights are below.
pub fn freivalds_pre(tag: &str) -> String {
    format!("zk_freivalds{tag}")
}
pub fn freivalds_w(tag: &str, mat: usize, p: usize) -> PreProcessedColumnId {
    id(format!("zk_freivalds{tag}_w{mat}_{p:04}"))
}
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
#[derive(Clone)]
pub struct FreivaldsQx {
    /// one weight per token slot
    pub w: Vec<SecureField>,
    /// the row step inside one input channel's chunks
    pub m_in: SecureField,
    /// the row step where a new input channel begins
    pub m_bd: SecureField,
    pub claimed: SecureField,
}

// ------------------------------------------ the check's AIR, one component --

/// One Freivalds group's constraints: the `(rv^T W)(X s)` side of the identity
/// at the top of this file.
#[derive(Clone)]
pub struct FreivaldsEval {
    /// the group's name, which every preprocessed id carries
    pub tag: String,
    /// the row tiling: one row group per input channel, `chunks` rows each
    pub g: Grid,
    /// weights per row per matmul
    pub pk: usize,
    /// activation cells per row -- one token chunk
    pub pq: usize,
    pub n_mats: usize,
    /// `rv^0 .. rv^{pk-1}`, and `rv^pk` as the step from one row to the next
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    /// `s^0 .. s^{pq-1}`, and `s^pq` as the step
    pub s_pow: Vec<SecureField>,
    pub s_step: SecureField,
    /// one claimed `(rv^T W_m)(X s)` per matmul: what the running product is
    /// pinned to on the table's last row
    pub claimed: Vec<SecureField>,
    /// The committed activations as what they are: bytes of a range table, 8
    /// bits wide everywhere but the widened attention output's group, whose
    /// activations are one cell in `[1, 2^12)` against a 12-bit table rather
    /// than two limbs.  Everything that follows from a cell being a table's
    /// value -- the offset it carries so a signed activation lands in an
    /// unsigned table, the relation, the lookups and which backend proves them
    /// -- is in `crate::columns::ByteCells` and nowhere in this gadget.
    pub acts: ByteCells,
    /// The claim this component publishes over its own committed activations,
    /// or `None` on a segment with no producer to bind to -- which is what keeps
    /// such a segment's proof exactly what it was.
    pub qx: Option<FreivaldsQx>,
}

impl FrameworkEval for FreivaldsEval {
    /// The flat parameter block the generated constraint kernel is emitted
    /// against (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// The kernel store's key for this component's generated constraint kernel.
    /// Never rename it: a component whose name has no kernel installed under it
    /// falls back to the CPU accumulation path.
    fn cuda_eval_name(&self) -> &'static str { "zk_freivalds_eval" }

    fn log_size(&self) -> u32 {
        self.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // The columns come out of the one declaration in `columns.rs`, in the
        // order it declares them.  Nothing below names a column by position,
        // and `rows` is the table's grouping, which the fold primitive reads
        // and its carry rules resolve against.
        let sh = FreivaldsShape::of(self);
        let FreivaldsRow { rows, known, acts, sums } = FreivaldsRow::read(&mut eval, &sh);
        let x = acts.x;

        // what the committed activation cells carry on top of the signed value
        let off_sum: SecureField =
            self.s_pow.iter().fold(SecureField::zero(), |a, b| a + *b)
                * SecureField::from_u32_unchecked(self.acts.off, 0, 0, 0);
        // x_fold: this row's activations folded against s, on top of the
        // previous row's fold shifted by s^pq -- except on an input channel's
        // first row, where the fold restarts
        let x_fold = sums.x_fold.cur.clone();
        constrain(&mut eval, &rows, sums.x_fold, Fold {
            carry: Carry::PerGroup { step: self.s_step },
            term: Term::Each(dot::<E>(&x.cells, &self.s_pow) - off_sum),
            close: None,
        });

        for (m, mat) in sums.mats.into_iter().enumerate() {
            // w_fold_m: this row's weights folded against rv, on the same carry
            let w_fold = mat.w_fold.cur.clone();
            constrain(&mut eval, &rows, mat.w_fold, Fold {
                carry: Carry::PerGroup { step: self.rv_step },
                term: Term::Each(dot::<E>(&known.weights[m], &self.rv_pow)),
                close: None,
            });
            // product_m: the two folds multiplied and added in once per input
            // channel, on the input channel's last row and carried otherwise,
            // and closing on the claim the statement carries
            constrain(&mut eval, &rows, mat.product, Fold {
                carry: Carry::WholeTable,
                term: Term::PerGroup(w_fold * x_fold.clone()),
                close: Some(Close { claim: self.claimed[m] }),
            });
        }
        // The published claim over the committed activations, allocated before
        // the LogUp columns so that the interaction order is claims first and
        // fractions after, as the components it is compared against already are.
        // What is folded is `a - act_off`, the signed activation, which is what
        // a producer's requantize publishes.  Its rows step at two rates -- one
        // from one row of an input channel to the next and one where a new input
        // channel begins -- which is a rule about the grouping and so one the
        // fold states.
        if let Some(q) = &self.qx {
            let x_claim = sums.x_claim
                .expect("the row carries the activation claim exactly when the AIR does");
            let off = E::F::from(M31::from_u32_unchecked(self.acts.off));
            let mut xf = E::EF::zero();
            for (pp, a) in x.cells.iter().enumerate() {
                xf += (E::EF::from(a.clone()) - E::EF::from(off.clone())) * q.w[pp];
            }
            constrain(&mut eval, &rows, x_claim, Fold {
                carry: Carry::Rates { inside: q.m_in, across: q.m_bd },
                term: Term::Each(xf),
                close: Some(Close { claim: q.claimed }),
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
pub struct Freivalds;

/// Which of the layer's matmul groups the check is over.
pub struct Cfg {
    pub group: String,
    /// the group's matmuls, as indices into the layer's own list
    pub mats: Vec<usize>,
    pub p_target: usize,
}

impl crate::gadgets::Gadget for Freivalds {
    type Witness = crate::layer::Layer;
    type Cfg = Cfg;
    /// What a Freivalds control corrupts is a weight or an activation of the
    /// layer it is handed, not a value beside it.
    type Tamper = ();
    type Outcome = run::Outcome;

    fn run(l: &crate::layer::Layer, cfg: &Cfg, _bad: Option<()>) -> run::Outcome {
        run::run(l, &cfg.group, &cfg.mats, cfg.p_target)
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
mod tests {
    use stwo::core::channel::Channel;
    use stwo::core::fields::m31::M31;
    use stwo::core::pcs::TreeVec;
    use stwo::prover::backend::Column;

    use num_traits::One;

    use super::*;
    use crate::bf16::Round;
    use crate::columns::{Lookups, constraint_values};
    use crate::common::Eval;
    use crate::gadgets::attention::witness::{Src, Variant};
    use crate::compose::{Claims, FreivaldsPlan, qm, unqm};
    use crate::gadgets::freivalds::claims::{freivalds_interaction, identity_holds,
                                            project_product};
    use crate::gadgets::freivalds::columns::{FreivaldsShape, Known, constraint_names};
    use crate::gadgets::freivalds::layout::FREIVALDS_GROUPS;
    use crate::gadgets::freivalds::trace::freivalds_component;
    use crate::layer::{Layer, Tables};
    use crate::relations::{RangeRel, powers};

    const ACT_BITS: u32 = 8;
    const P_TARGET: usize = 8;

    /// A synthetic layer small enough that a whole Freivalds group builds,
    /// closes its accumulators and is checked row by row in well under a second.
    fn small_layer() -> Layer {
        let src = Src { dir: None, head: 0, q_bits: 12, k_bits: 11, nmax: 255, log_table: 8,
                        seed: 7, layout: String::new(), diagonal_block: false, tile: false };
        Layer::synthetic(8, 8, 8, &[1], Variant::parse("intsplit6"), &src, 16, 16, 0x221)
    }

    /// One built component: the AIR, its three trees, what it published, and
    /// what the controls left outside the constraints.
    struct Built {
        /// the shape is `FreivaldsShape::of(&air)`, which borrows it
        air: FreivaldsEval,
        pre: Vec<Eval>,
        base: Vec<Eval>,
        inter: Vec<Eval>,
        use_sum: SecureField,
        off_table: usize,
        /// the layer's matmul indices this group is over
        mats: Vec<usize>,
        lay: crate::gadgets::freivalds::layout::FreivaldsLayout,
        /// one `(rv^T W_m)(X s)` per matmul, in `mats` order
        claimed: Vec<SecureField>,
        /// the two challenges the claims were built against
        rv: SecureField,
        s: SecureField,
    }

    /// Build one group exactly the way `run.rs` composes it, with the two
    /// controls `trace.rs` takes.
    fn build(group: &str, with_qx: bool, corrupt_qx: Option<usize>,
             oversize_qx: Option<usize>) -> Built {
        let l = small_layer();
        let names = FREIVALDS_GROUPS.iter().find(|(g, _)| *g == group)
            .unwrap_or_else(|| panic!("no Freivalds group {group}")).1;
        let mats: Vec<usize> = l.mats.iter().enumerate()
            .filter(|(_, m)| names.contains(&m.name.as_str())).map(|(i, _)| i).collect();
        let mut tables = Tables::new(vec![0u32; 16], 4, 0, Round::Even, 4, vec![ACT_BITS], 0, 0);
        let r = freivalds_component(&l, group, &mats, P_TARGET, ACT_BITS, &mut tables,
                                    corrupt_qx, oversize_qx);
        let lay = r.lay;
        // fixed challenges: what is under test is the constraints, not the transcript
        let mut ch = stwo::core::channel::Blake2sM31Channel::default();
        ch.mix_u64(0x221);
        let rel_act = RangeRel::draw(&mut ch);
        let f = ch.draw_secure_felts(5);
        let (rv, s_ch) = (f[0], f[1]);
        let (rv_pow, rv_step) = powers(rv, lay.pk);
        let (s_pow, s_step) = powers(s_ch, lay.pq);
        // the activation claim's weights and row steps are the wiring family's;
        // the constraints hold for any of them, so the test draws them
        let qx = |claimed| with_qx.then(|| FreivaldsQx {
            w: powers(f[2], lay.pq).0, m_in: f[3], m_bd: f[4], claimed,
        });
        let air = |claimed: Vec<SecureField>, qx| FreivaldsEval {
            tag: group.to_string(), g: lay.g, pk: lay.pk, pq: lay.pq, n_mats: mats.len(),
            rv_pow: rv_pow.clone(), rv_step, s_pow: s_pow.clone(), s_step, claimed,
            acts: ByteCells { bits: ACT_BITS, off: 1 << (ACT_BITS - 1),
                              rel: rel_act.clone(), by: Lookups::Own },
            qx,
        };
        let open = air(vec![SecureField::zero(); mats.len()], qx(SecureField::zero()));
        let it = freivalds_interaction(&l, &r, rv, s_ch, &open);
        // the claims the component publishes close the last two constraints
        let air = air(it.claimed.clone(), qx(it.qx_claim));
        Built {
            pre: r.pre.clone(),
            base: r.base.clone(),
            inter: it.cols.into_iter().map(|c| c.into_host()).collect(),
            use_sum: it.use_sum,
            off_table: tables.off_table,
            mats: r.mats.clone(),
            lay,
            claimed: it.claimed,
            rv,
            s: s_ch,
            air,
        }
    }

    /// The (row, constraint name) pairs whose constraint does not hold.
    ///
    /// The preprocessed tree is handed over in the order the AIR *reads* it,
    /// which is the row schema's declaration order and is what a component's own
    /// mask is numbered by; the committed order is the schema's too, and this is
    /// the one place the two meet.
    fn broken(b: &Built) -> Vec<(usize, String)> {
        let sh = FreivaldsShape::of(&b.air);
        let committed = Known::committed_ids(&sh);
        assert_eq!(committed.len(), b.pre.len(), "the committed ids are not the tree");
        // A component's own preprocessed mask is numbered in the order it *reads*
        // its columns, and the tree is committed in another one, so the columns
        // are gathered here the way `FrameworkComponent` gathers them: by id.
        // The read order is the AIR's own, recorded rather than restated.
        let read = b.air
            .evaluate(stwo_constraint_framework::InfoEvaluator::new(
                b.air.log_size(), vec![], b.use_sum))
            .preprocessed_columns;
        let pre: Vec<Vec<M31>> = {
            let by_id: std::collections::HashMap<&str, &Eval> =
                committed.iter().map(|i| i.id.as_str()).zip(b.pre.iter()).collect();
            read.iter().map(|id| by_id[id.id.as_str()].values.to_cpu()).collect()
        };
        let base: Vec<Vec<M31>> = b.base.iter().map(|e| e.values.to_cpu()).collect();
        let inter: Vec<Vec<M31>> = b.inter.iter().map(|e| e.values.to_cpu()).collect();
        assert_eq!((pre.len(), base.len()), (sh.n_pre(), sh.n_base()),
                   "the trees are not the ones the row declares");
        let trace: TreeVec<Vec<&Vec<M31>>> = TreeVec::new(vec![
            pre.iter().collect(),
            vec![],
            base.iter().collect(),
            inter.iter().collect(),
        ]);
        let names = constraint_names(&sh);
        let mut bad = vec![];
        for (row, vals) in constraint_values(&b.air, &trace, b.use_sum).into_iter().enumerate() {
            assert_eq!(vals.len(), names.len(),
                       "the AIR adds {} constraints and the declaration names {}",
                       vals.len(), names.len());
            for (v, n) in vals.iter().zip(&names) {
                if !v.is_zero() {
                    bad.push((row, n.clone()));
                }
            }
        }
        bad
    }

    /// A trace built through the declaration's writers satisfies every constraint
    /// of the AIR that reads through it, on every row of the domain.
    ///
    /// This is the half the pinned digests cannot cover: a digest says the proof
    /// did not change, which is also what it says when the circuit was wrong
    /// before and is wrong in the same way now.  What fails here is a reader and
    /// a writer that disagree about a column's position, a selector true on the
    /// wrong rows, a running sum carried across the wrong boundary, or a padded
    /// row the constraints do not hold on -- by name, and at the row.
    #[test]
    fn the_honest_trace_satisfies_every_constraint_on_every_row() {
        for group in ["o", "qkv", "gateup"] {
            for qx in [false, true] {
                let b = build(group, qx, None, None);
                assert_eq!(b.off_table, 0,
                           "{group}: an honest activation fell off the range table");
                let bad = broken(&b);
                assert!(bad.is_empty(), "{group} (qx={qx}): {} constraint(s) do not hold, \
                                         first {:?}", bad.len(), &bad[..bad.len().min(4)]);
            }
        }
    }

    /// And each control breaks the constraint it is written for and no other.
    ///
    /// `corrupt_qx` moves one *committed* activation by one and leaves `x_fold`
    /// on the honest witness, so the one equality that stops holding is
    /// `x_fold`, on the one row that carries that cell -- and `x_claim_fold`
    /// with it on a component that publishes a claim over the activations, which
    /// folds the same committed cells against different weights.
    ///
    /// `oversize_qx` is not a constraint control at all: it puts a committed
    /// activation one past its range table, which is the *table's* business.  The
    /// row sees it only as the same activation disagreeing with `x_fold`, and
    /// what it really broke is the off-table counter.
    #[test]
    fn each_control_breaks_the_constraint_it_targets_and_nothing_else() {
        let idx = 5usize;
        for (control, off_table) in [("corrupt", 0usize), ("oversize", 1usize)] {
            for qx in [false, true] {
                let (c, o) = match control {
                    "corrupt" => (Some(idx), None),
                    _ => (None, Some(idx)),
                };
                let b = build("o", qx, c, o);
                let bad = broken(&b);
                let want: Vec<&str> = if qx { vec!["x_fold", "x_claim_fold"] }
                                      else { vec!["x_fold"] };
                assert_eq!(bad.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>(), want,
                           "{control}_qx (qx={qx}) broke {bad:?}");
                assert!(bad.windows(2).all(|w| w[0].0 == w[1].0),
                        "{control}_qx (qx={qx}) broke more than one row: {bad:?}");
                assert_eq!(b.off_table, off_table,
                           "{control}_qx (qx={qx}): off-table count");
            }
        }
    }

    /// The equality a composed verifier runs, on claims from honest builds.
    ///
    /// One side is what each Freivalds component published for its matmuls; the
    /// other is `project_product` over the layer's exact accumulators, which is
    /// the value the dequant component publishes as `claimed_h`.  They have to
    /// agree matmul for matmul, and moving one published product has to be seen
    /// -- at that matmul and at no other.
    #[test]
    fn the_identity_accepts_honest_claims_and_rejects_a_changed_one() {
        let l = small_layer();
        let mut groups: Vec<FreivaldsPlan> = vec![];
        let mut acc: Vec<Vec<[u32; 4]>> = vec![];
        let mut mat_h = vec![qm(SecureField::zero()); l.mats.len()];
        for group in ["o", "qkv", "gateup"] {
            let b = build(group, false, None, None);
            for mi in &b.mats {
                let m = &l.mats[*mi];
                mat_h[*mi] = qm(project_product(&m.acc, m.tokens, m.n_out, b.rv, b.s));
            }
            acc.push(b.claimed.iter().map(|v| qm(*v)).collect());
            groups.push(FreivaldsPlan { name: group.to_string(), mats: b.mats, lay: b.lay,
                                        act_bits: ACT_BITS });
        }
        let claims = |acc: Vec<Vec<[u32; 4]>>| Claims {
            freivalds_acc: acc, mat_h: mat_h.clone(), ..Default::default()
        };
        let honest = identity_holds(&claims(acc.clone()), &groups);
        assert_eq!(honest.len(), groups.iter().map(|g| g.mats.len()).sum::<usize>());
        assert!(honest.iter().all(|x| *x),
                "an honest run's claims do not satisfy the identity: {honest:?}");

        // and moving any one of the published products breaks that one alone
        for flat in 0..honest.len() {
            let mut bad = acc.clone();
            let (mut gi, mut k) = (0usize, flat);
            while k >= bad[gi].len() {
                k -= bad[gi].len();
                gi += 1;
            }
            bad[gi][k] = qm(unqm(bad[gi][k]) + SecureField::one());
            let got = identity_holds(&claims(bad), &groups);
            let want: Vec<bool> = (0..honest.len()).map(|i| i != flat).collect();
            assert_eq!(got, want, "changing product {flat} was not seen where it was made");
        }
    }
}
