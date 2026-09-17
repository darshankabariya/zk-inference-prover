//! What one LogUp lookup costs in committed cells, proved on its own.
//!
//! Seven subcommands report a whole gadget's marginal cells with its own lookups
//! inside; this one reports the lookup. The use side commits `per_row` lookups a
//! row and the table side is `relations::TableEval`, the shape every shared table in
//! this crate commits, so the two sides are the crate's own and not a model of
//! them. Two kinds and the two batch sizes the crate uses:
//!
//!   range   one committed cell a lookup, against `0 .. 2^bits`
//!   pair    two committed cells a lookup, against the `(q, is q nonzero)` table
//!           the int8 dequant and the attention output read
//!
//!   batch 1 `finalize_logup`, one interaction column a lookup -- what every
//!           table side in this crate finalizes with
//!   batch 2 `finalize_logup_in_pairs`, one column per two lookups -- what every
//!           use side finalizes with
//!
//! The constraint framework also has `finalize_logup_batched`, which would take a
//! batch of four; no component in this crate calls it, so the sweep stops at the
//! two that are in use.
//!
//! An interaction column is one QM31 and a QM31 is four M31 cells a row, whatever
//! the relation's arity, so the marginal is `4 / batch` and the kind decides only
//! what the gadget commits beside it. Neither is assumed: each point proves the
//! same layout at two lookup counts and reports the difference, which leaves the
//! table -- identical in both builds -- out of the figure.

use std::time::Instant;

use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, Relation,
                                RelationEntry, TraceLocationAllocator};

use crate::common::{Eval, Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::gadgets::dequant::QNZ_COLS;
use crate::relations::{PairRel, RangeRel, TableEval, TableKind, id, tbl_range};
use crate::plogup;

/// Which of the crate's tables the lookup is against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// the range check every committed integer takes
    Range,
    /// a general table lookup, key and value both committed
    Pair,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Range => "range",
            Kind::Pair => "pair",
        }
    }
    /// Committed cells one lookup names -- the relation's arity.
    pub fn arity(self) -> usize {
        match self {
            Kind::Range => 1,
            Kind::Pair => 2,
        }
    }
}

/// The relation the two kinds are drawn against, so one builder serves both.
#[derive(Clone)]
pub enum Rel {
    Range(RangeRel),
    Pair(PairRel),
}

impl Rel {
    fn combine(&self, v: &[PackedM31]) -> PackedQM31 {
        match self {
            Rel::Range(r) => r.combine(v),
            Rel::Pair(r) => r.combine(v),
        }
    }
    fn add<E: EvalAtRow>(&self, eval: &mut E, row: &[E::F]) {
        match self {
            Rel::Range(r) => eval.add_to_relation(RelationEntry::new(r, E::EF::one(), row)),
            Rel::Pair(r) => eval.add_to_relation(RelationEntry::new(r, E::EF::one(), row)),
        }
    }
    fn table(&self) -> TableKind {
        match self {
            Rel::Range(r) => TableKind::Range(r.clone()),
            Rel::Pair(r) => TableKind::Pair(r.clone()),
        }
    }
}

/// The use side: `per_row` lookups a row, batched the way the crate batches them.
#[derive(Clone)]
pub struct LookupEval {
    pub log_rows: u32,
    pub per_row: usize,
    /// lookups sharing one interaction column: 1 or 2
    pub batch: usize,
    pub arity: usize,
    pub rel: Rel,
}

impl FrameworkEval for LookupEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_lookup_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        for _ in 0..self.per_row {
            let row: Vec<E::F> = (0..self.arity).map(|_| eval.next_trace_mask()).collect();
            self.rel.add(&mut eval, &row);
        }
        if self.batch == 1 {
            eval.finalize_logup();
        } else {
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub kind: Kind,
    pub bits: u32,
    pub batch: usize,
    pub per_row: usize,
    pub log_rows: u32,
    pub n_lookups: usize,
    /// the use side's own cells
    pub use_base: usize,
    pub use_inter: usize,
    /// the table's: its rows, its multiplicity column and its own LogUp column
    pub table_pre: usize,
    pub table_base: usize,
    pub table_inter: usize,
    pub off_table: usize,
    /// committed lookups whose cells are the table row their key names
    pub ref_ok: usize,
    pub ref_n: usize,
    pub trace_secs: f64,
    pub interaction_secs: f64,
    pub n_columns: [usize; 3],
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0
            && self.ref_ok == self.ref_n
    }
    pub fn json(&self) -> Value {
        json!({
            "kind": self.kind.name(), "arity": self.kind.arity(),
            "logup_batch_size": self.batch,
            "table_bits": self.bits, "table_rows": 1usize << self.bits,
            "log_rows": self.log_rows, "lookups_per_row": self.per_row,
            "n_lookups": self.n_lookups,
            "cells": {"base": self.use_base, "interaction": self.use_inter,
                      "fresh": self.use_base + self.use_inter,
                      "table_preprocessed": self.table_pre,
                      "table_base": self.table_base,
                      "table_interaction": self.table_inter},
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "off_table": self.off_table,
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "trace_secs": self.trace_secs, "interaction_secs": self.interaction_secs,
            "sums": self.sums.json(), "accepts": self.accepts(),
            "point": self.proven.json(),
        })
    }
}

/// One lookup gadget, proved on its own.
pub fn run(kind: Kind, bits: u32, per_row: usize, log_rows: u32, batch: usize, seed: u64)
           -> Outcome {
    assert!(batch == 1 || batch == 2,
            "the crate finalizes its lookups singly or in pairs; batch {batch} is neither");
    assert!(per_row > 0, "a lookup gadget with no lookup measures nothing");
    // B4 section 7: a preprocessed tree shorter than the widest trace column
    // mis-maps the FRI query positions, so the table carries the height here.
    assert!(bits >= log_rows,
            "the table is 2^{bits} rows and the trace 2^{log_rows}: the preprocessed \
             tree has to be at least as tall as the widest committed column");
    let arity = kind.arity();
    let rows = 1usize << log_rows;
    let trows = 1usize << bits;
    let n_lookups = per_row * rows;

    let t0 = Instant::now();
    // the table, in the two shapes the crate has: the key, and for a general
    // lookup the value the key names
    let keys: Vec<u32> = (0..trows as u32).collect();
    let vals: Vec<u32> = keys.iter().map(|q| u32::from(*q != 0)).collect();
    let mut tpre = vec![eval_of(bits, keys.iter().copied().map(m31_u32).collect())];
    if kind == Kind::Pair {
        tpre.push(eval_of(bits, vals.iter().copied().map(m31_u32).collect()));
    }

    // the use side: one key a lookup, drawn from the table's own domain
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut mult = vec![0u32; trows];
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; per_row * arity];
    for j in 0..rows {
        for s in 0..per_row {
            let q = rng.random_range(0..trows as u32);
            cols[s * arity][j] = m31_u32(q);
            if arity == 2 {
                cols[s * arity + 1][j] = m31_u32(vals[q as usize]);
            }
            mult[q as usize] += 1;
        }
    }
    // the reference: every committed lookup read back off the trace it commits,
    // against the table row its key names. A key past the table is off it, which
    // is the incompleteness the sums would then refuse to cancel over.
    let (mut ref_ok, mut off_table) = (0usize, 0usize);
    for j in 0..rows {
        for s in 0..per_row {
            let i = cols[s * arity][j].0 as usize;
            if i >= trows {
                off_table += 1;
                continue;
            }
            if arity == 1 || cols[s * arity + 1][j] == m31_u32(vals[i]) {
                ref_ok += 1;
            }
        }
    }
    let base_use: Vec<Eval> = cols.into_iter().map(|c| eval_of(log_rows, c)).collect();
    let multc = eval_of(bits, mult.iter().map(|m| m31_u32(*m)).collect());
    let trace_secs = t0.elapsed().as_secs_f64();

    let tw = twiddles(bits.max(log_rows));
    let mut sess = Session::new(&tw);
    let n_pre = tpre.len();
    sess.commit(tpre.clone());
    // A gadget on its own has no model column, but the tree layout is the
    // same for every proof this crate makes: the empty model tree keeps the
    // base trace at the index every component's mask names.
    sess.commit_model_empty();
    let mut base = base_use.clone();
    base.push(multc.clone());
    let n_base = base.len();
    sess.commit(base);
    let rel = match kind {
        Kind::Range => Rel::Range(sess.draw_relation(RangeRel::draw)),
        Kind::Pair => Rel::Pair(sess.draw_relation(PairRel::draw)),
    };

    let t1 = Instant::now();
    let one = PackedQM31::broadcast(SecureField::one());
    let look = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        let v: Vec<PackedM31> = (0..arity).map(|k| base_use[i * arity + k].data[vr]).collect();
        (one, rel.combine(&v))
    };
    let (ucols, use_sum) = if batch == 1 {
        plogup::build(log_rows, per_row, look)
    } else {
        let (n_cols, f) = plogup::paired(per_row, look);
        plogup::build(log_rows, n_cols, f)
    };
    let (tcols, table_sum) = plogup::build(bits, 1, |_, vr| {
        let v: Vec<PackedM31> = tpre.iter().map(|c| c.data[vr]).collect();
        (-PackedQM31::from(multc.data[vr]), rel.combine(&v))
    });
    let interaction_secs = t1.elapsed().as_secs_f64();
    let use_inter: usize = ucols.iter().map(|e| e.domain.size()).sum();
    let table_inter: usize = tcols.iter().map(|e| e.domain.size()).sum();
    let n_inter = ucols.len() + tcols.len();
    sess.commit(ucols.into_iter().chain(tcols).collect());

    let ids = match kind {
        Kind::Range => vec![tbl_range(bits)],
        Kind::Pair => QNZ_COLS.iter().map(|c| id(*c)).collect(),
    };
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let use_comp = FrameworkComponent::<LookupEval>::new(
        alloc,
        LookupEval { log_rows, per_row, batch, arity, rel: rel.clone() },
        use_sum,
    );
    let tbl = FrameworkComponent::<TableEval>::new(
        alloc, TableEval { log_rows: bits, cols: ids.clone(), kind: rel.table() }, table_sum);
    let provers: Vec<&dyn ComponentProver<_>> = vec![&use_comp, &tbl];
    let components: Vec<&dyn Component> = vec![&use_comp, &tbl];
    let proven = sess.prove(&provers, &components);

    Outcome {
        proven,
        sums: Sums { uses: use_sum, tables: table_sum },
        kind,
        bits,
        batch,
        per_row,
        log_rows,
        n_lookups,
        use_base: base_use.iter().map(|e| e.domain.size()).sum(),
        use_inter,
        table_pre: tpre.iter().map(|e| e.domain.size()).sum(),
        table_base: multc.domain.size(),
        table_inter,
        off_table,
        ref_ok,
        ref_n: n_lookups,
        trace_secs,
        interaction_secs,
        n_columns: [n_pre, n_base, n_inter],
    }
}

/// One point of the sweep: the same gadget at two lookup counts, and what the
/// lookups the two differ by cost.
///
/// `cells.marginal_cells_per_lookup` is `LOGUP_MARGINAL[batch]`: the interaction
/// cells one lookup adds, with whatever the gadget commits for the lookup left out
/// and the table, which the two builds pay identically, differenced away.
pub fn marginal(small: &Outcome, large: &Outcome) -> Value {
    assert_eq!(small.kind, large.kind, "the two builds are of different kinds");
    assert_eq!(small.batch, large.batch, "the two builds batch differently");
    assert_eq!(small.log_rows, large.log_rows,
               "the two builds have to sit on the same rows, or the table moves with them");
    assert!(large.n_lookups > small.n_lookups,
            "the second build has to carry more lookups than the first");
    let dl = (large.n_lookups - small.n_lookups) as f64;
    let base = (large.use_base - small.use_base) as f64;
    let inter = (large.use_inter - small.use_inter) as f64;
    json!({
        "kind": large.kind.name(), "arity": large.kind.arity(),
        "logup_batch_size": large.batch,
        "table_bits": large.bits, "table_rows": 1usize << large.bits,
        "log_rows": large.log_rows,
        "lookups_per_row": [small.per_row, large.per_row],
        "n_lookups": large.n_lookups,
        "cells": {
            // what one more lookup costs, the gadget's own committed cells included:
            // the figure `harness/costmodel/cost_model.py` reads and takes 2 off
            "per_lookup": (base + inter) / dl,
            // the two halves of it, measured apart
            "committed_per_lookup": base / dl,
            "marginal_cells_per_lookup": inter / dl,
            // the table, which is a fixed cost of a segment and is charged once
            "table": {"preprocessed": large.table_pre, "base": large.table_base,
                      "interaction": large.table_inter,
                      "fresh": large.table_base + large.table_inter},
        },
        "difference": {
            "lookups": large.n_lookups - small.n_lookups,
            "base": large.use_base - small.use_base,
            "interaction": large.use_inter - small.use_inter,
            "fresh": (large.use_base + large.use_inter) - (small.use_base + small.use_inter),
        },
        "prove_secs": large.proven.prove_secs,
        "proof_bytes": large.proven.proof_bytes,
        "accepts": small.accepts() && large.accepts(),
        "off_table": small.off_table + large.off_table,
        "reference_checked": small.ref_n + large.ref_n,
        "reference_mismatch": (small.ref_n - small.ref_ok) + (large.ref_n - large.ref_ok),
        "builds": [small.json(), large.json()],
    })
}
