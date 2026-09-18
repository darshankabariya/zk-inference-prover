//! Prove one gadget with no segment around it (#221).
//!
//! Every gadget that proves alone runs the same session, and each one of them
//! has written it out: commit what was fixed before the challenges, commit the
//! empty model tree so that the base trace lands at the index every component's
//! mask names, commit the base trace, draw the relations, build what depends on
//! them, commit that, bind the published claims to the transcript, register the
//! preprocessed ids, build the components and prove.  What differs between two
//! gadgets is the middle: which columns, which claims, which AIR.
//!
//! [`prove_alone`] is that session with the middle left to the caller.  A run
//! file is then the witness, this call and the outcome: [`Trees`] is what the
//! gadget committed before the challenges, [`RangeTable`] is a shared table it
//! looks its cells up in, and the closure is everything after the draw.
//!
//! **The transcript is the order below and nothing else.**  A gadget that
//! draws a felt before its relations, or mixes its claims before it commits its
//! interaction tree, makes a different proof; that is why the order is written
//! here once rather than in each run file, where the next copy of it could
//! drift.  What this function does not decide is what a standalone proof
//! *means*: a gadget on its own proves its own columns, and whatever its claims
//! are compared against in a composed proof is not here.  Each run file says so
//! for itself.

use std::time::Instant;

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval, TraceLocationAllocator};

use crate::common::{Eval, PcsB, Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::device::AnyCol;
use crate::layer::Tables;
use crate::relations::{RangeRel, RangeTableEval, table_col, tbl_range};

/// One shared range table a standalone proof carries: the table's own rows, and
/// the multiplicities the gadget's writers bumped as they committed their cells.
///
/// A composed proof commits one of these for a whole segment and every gadget in
/// it looks values up in the same one.  A gadget on its own carries the tables
/// it uses and no others, and what they cost is reported apart from what the
/// gadget's own trees cost -- a shared table is not a marginal.
pub struct RangeTable {
    /// the table's width in bits: it holds `[0, 2^bits)`, one value a row
    pub bits: u32,
    /// the table's own column
    tbl: Eval,
    /// how often each of its values was looked up
    mult: Eval,
}

impl RangeTable {
    /// The range table at `bits`, off the multiplicities `tables` counted while
    /// the gadget's byte cells were being committed.
    pub fn of(bits: u32, tables: &Tables) -> Self {
        RangeTable {
            bits,
            tbl: eval_of(bits, (0..1u32 << bits).map(m31_u32).collect()),
            mult: eval_of(bits, tables.m_r[tables.ridx(bits)].iter().map(|v| m31_u32(*v)).collect()),
        }
    }
}

/// What a gadget committed before its challenges were drawn.
///
/// The cell counts are handed in rather than taken off the column lists: a
/// component whose columns were built on the card has none of them on the host,
/// and what it cost is still the same number of cells.
pub struct Trees {
    /// the component's own preprocessed columns, in committed order
    pub pre: Vec<Eval>,
    /// the component's own committed base columns
    pub base: Vec<Eval>,
    /// the ids the component's preprocessed tree is registered under, in the
    /// same committed order
    pub ids: Vec<PreProcessedColumnId>,
    /// what those two trees cost the proof
    pub pre_cells: usize,
    pub base_cells: usize,
}

/// What a gadget builds once the challenges are drawn.
///
/// The AIR comes last because it carries the claims, and the claims are what
/// the interaction builder computed.
pub struct After<A> {
    /// the AIR the component is proved against, claims and all
    pub air: A,
    /// the component's own interaction columns: its running sums, and its LogUp
    /// fractions where it proves its own lookups
    pub cols: Vec<AnyCol<'static>>,
    /// what those fractions came to, which is the component's claimed sum
    pub use_sum: SecureField,
    /// what the component publishes, in the order it goes into the transcript
    pub claims: Vec<SecureField>,
}

/// What one gadget's standalone proof committed, published and came to.
pub struct Alone {
    pub proven: Proven,
    /// the component's claimed sum and the tables', which have to cancel
    pub sums: Sums,
    /// the component's own three trees, the shared tables left out
    pub pre_cells: usize,
    pub base_cells: usize,
    pub inter_cells: usize,
    /// the shared tables: their rows, and their multiplicity columns plus their
    /// own LogUp columns
    pub table_cells_pre: usize,
    pub table_cells_fresh: usize,
    /// committed columns per tree, the tables' among them: preprocessed, base,
    /// interaction
    pub n_columns: [usize; 3],
    /// what the component published, which `sess.mix` bound to the transcript
    pub claims: Vec<SecureField>,
    /// how long the after-the-challenges half took
    pub interaction_secs: f64,
}

impl Alone {
    /// Fresh cells the component commits, the shared tables left out: what a
    /// second chunk pays for again.
    pub fn fresh(&self) -> usize {
        self.base_cells + self.inter_cells
    }

    /// The cell counts every run file reports, under the names the cost model
    /// reads them by (`harness/costmodel/cost_model.py`).  A gadget adds its own
    /// ratios to this object; the keys here are the ones that are not per
    /// gadget.
    pub fn cells_json(&self) -> serde_json::Value {
        serde_json::json!({
            "preprocessed": self.pre_cells, "base": self.base_cells,
            "interaction": self.inter_cells, "fresh": self.fresh(),
            "table_preprocessed": self.table_cells_pre,
            "table_fresh": self.table_cells_fresh,
        })
    }

    /// The committed columns per tree, the tables' among them.
    pub fn columns_json(&self) -> serde_json::Value {
        serde_json::json!({
            "preprocessed": self.n_columns[0], "base": self.n_columns[1],
            "interaction": self.n_columns[2],
            "total": self.n_columns.iter().sum::<usize>(),
        })
    }
}

/// Prove one component alone, beside the range tables its cells are looked up
/// in.
///
/// `after` is handed the session and the tables' relations, in the order the
/// tables were given, and draws whatever else it needs from the session --
/// which is what fixes the transcript, so a gadget that draws a challenge draws
/// it there and nowhere else.
pub fn prove_alone<A, F>(trees: Trees, tables: &[RangeTable], after: F) -> Alone
where
    A: FrameworkEval + Sync,
    F: FnOnce(&mut Session, &[RangeRel]) -> After<A>,
{
    let max_log = trees
        .pre
        .iter()
        .chain(trees.base.iter())
        .map(|e| e.domain.log_size())
        .chain(tables.iter().map(|t| t.bits))
        .max()
        .expect("a standalone proof with no column in it");
    let tw = twiddles(max_log);
    let mut sess = Session::new(&tw);

    let mut pre = trees.pre;
    pre.extend(tables.iter().map(|t| t.tbl.clone()));
    let n_pre = pre.len();
    sess.commit(pre);
    // A gadget on its own has no model column, but the tree layout is the same
    // for every proof this crate makes: the empty model tree keeps the base
    // trace at the index every component's mask names.
    sess.commit_model_empty();
    let mut base = trees.base;
    base.extend(tables.iter().map(|t| t.mult.clone()));
    let n_base = base.len();
    sess.commit(base);

    // The relations come first, in the tables' own order, and the gadget draws
    // what it needs after them.
    let rels: Vec<RangeRel> = tables.iter().map(|_| sess.draw_relation(RangeRel::draw)).collect();
    let t1 = Instant::now();
    let built = after(&mut sess, &rels);
    let interaction_secs = t1.elapsed().as_secs_f64();
    let inter_cells: usize = built.cols.iter().map(|c| c.size()).sum();
    let n_inter_cols = built.cols.len();

    // and the tables' own sides, which are `-multiplicity / (z - value)` and
    // are what the component's lookups cancel against
    let mut table_cells_pre = 0;
    let mut table_cells_fresh = 0;
    let mut tsums = vec![];
    let mut tcols = vec![];
    for (t, rel) in tables.iter().zip(&rels) {
        let (cols, sum) = table_col(t.bits, rel, &[&t.tbl], &t.mult);
        table_cells_pre += t.tbl.domain.size();
        table_cells_fresh += t.mult.domain.size() + cols.iter().map(|e| e.domain.size()).sum::<usize>();
        tsums.push(sum);
        tcols.extend(cols);
    }
    let n_tcols = tcols.len();
    let mut icols = built.cols;
    icols.extend(crate::device::host_cols(tcols));
    sess.commit_any(icols);

    // The claims are public input; bind them to the transcript before the
    // components fix them, or a prover picks them adaptively.
    sess.mix(&built.claims);

    let mut ids = trees.ids;
    ids.extend(tables.iter().map(|t| tbl_range(t.bits)));
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let gadget = FrameworkComponent::<A>::new(alloc, built.air, built.use_sum);
    let tabs: Vec<FrameworkComponent<RangeTableEval>> = tables
        .iter()
        .zip(&rels)
        .zip(&tsums)
        .map(|((t, rel), sum)| {
            FrameworkComponent::<RangeTableEval>::new(
                alloc, RangeTableEval { bits: t.bits, rel: rel.clone() }, *sum)
        })
        .collect();
    let mut provers: Vec<&dyn ComponentProver<PcsB>> = vec![&gadget];
    let mut components: Vec<&dyn Component> = vec![&gadget];
    for t in &tabs {
        provers.push(t);
        components.push(t);
    }
    let proven = sess.prove(&provers, &components);

    Alone {
        sums: Sums {
            uses: built.use_sum,
            tables: tsums.iter().fold(SecureField::zero(), |a, b| a + *b),
        },
        proven,
        pre_cells: trees.pre_cells,
        base_cells: trees.base_cells,
        inter_cells,
        table_cells_pre,
        table_cells_fresh,
        n_columns: [n_pre, n_base, n_inter_cols + n_tcols],
        claims: built.claims,
        interaction_secs,
    }
}
