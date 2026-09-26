//! The lookup relations the gadgets share, and the table side of each one.
//!
//! A relation is what a LogUp lookup is drawn against, so two gadgets that read
//! the same table have to name one relation type.  The range check the attention
//! key tiles take is the range check the dequant, the requantize, the embedding
//! and the argmax take; the normalisation, fraction and mantissa-product tables
//! `crate::bf16` generates are read by more than one component of a composed
//! proof.  Until this module existed those types lived inside
//! `gadgets/attention/air.rs` and `gadgets/dequant/mod.rs`, which made those two
//! gadgets the root of a dependency graph nothing in `docs/statement.md` asks
//! for: every other gadget had to name one of them to say what it looks up.
//!
//! What is here is the relation types, the preprocessed ids of the tables they
//! are stated over, the two table components (`TableEval` for the shared tables,
//! `RangeTableEval` for a plain `0 .. 2^bits` range) and two helpers every gadget
//! needs beside them -- `table_col`, which builds a table's own LogUp column from
//! its multiplicities, and `powers`, the challenge powers a Horner fold runs in.
//!
//! A gadget's *own* relation stays with the gadget: `ExpRel` is the attention
//! block's exp table, and `PowRel`, `SMulRel`, `UnRel` and `ValRel` are the
//! residual stream's.  Nothing else looks into them.

use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkEval, Relation, RelationEntry, relation,
};

use crate::common::Eval;

relation!(NormRel, 6);
relation!(FracRel, 3);
relation!(MulRel, 4);
relation!(PairRel, 2);
relation!(RangeRel, 1);

// B11 stream U, lever 3: the drawn elements, for the GKR link's affine weights.
// `combine(v) = sum alpha_i v_i - z`, so these two are everything the link needs
// to write a lookup's denominator as an affine form over the columns it reads.
impl NormRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl FracRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl MulRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
// B11 stream V, lever 4: the attention-output component's `(qm, nz2)` pair check
// is the first lookup any family's link has needed from this relation.
impl PairRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl RangeRel {
    /// the drawn elements, for the GKR link's affine weights (B11 stream S)
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}

pub fn id(s: impl Into<String>) -> PreProcessedColumnId {
    PreProcessedColumnId { id: s.into() }
}

pub const NORM_COLS: [&str; 6] =
    ["b6_norm_m2", "b6_norm_ph", "b6_norm_st", "b6_norm_m7", "b6_norm_ef", "b6_norm_nz"];
pub const FRAC_COLS: [&str; 3] = ["b6_frac_ph", "b6_frac_r", "b6_frac_st"];
pub const MUL_COLS: [&str; 4] = ["b6_mul_a", "b6_mul_b", "b6_mul_m7", "b6_mul_c"];

pub fn tbl_range(bits: u32) -> PreProcessedColumnId {
    id(format!("b6_range_{bits}"))
}

pub fn powers(base: SecureField, n: usize) -> (Vec<SecureField>, SecureField) {
    let mut v = Vec::with_capacity(n);
    let mut cur = SecureField::one();
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    (v, cur)
}

/// A table's own side: `-multiplicity / (z - combine(row))`.
pub fn table_col<R: Relation<PackedM31, PackedQM31> + Sync>(
    log_rows: u32,
    rel: &R,
    row: &[&Eval],
    mult: &Eval,
) -> (Vec<Eval>, SecureField) {
    crate::plogup::build(log_rows, 1, |_, vec_row| {
        let vals: Vec<_> = row.iter().map(|c| c.data[vec_row]).collect();
        (-PackedQM31::from(mult.data[vec_row]), rel.combine(&vals))
    })
}

// ------------------------------------------------------------------- tables --

/// A preprocessed table with a committed multiplicity column, yielded at `-m`.
/// One shape serves all four tables; the arity follows the column list.
#[derive(Clone)]
pub struct TableEval {
    pub log_rows: u32,
    pub cols: Vec<PreProcessedColumnId>,
    pub kind: TableKind,
}

#[derive(Clone)]
pub enum TableKind {
    Norm(NormRel),
    Frac(FracRel),
    Mul(MulRel),
    Pair(PairRel),
    Range(RangeRel),
}

impl TableKind {
    pub(crate) fn combine(&self, v: &[PackedM31]) -> PackedQM31 {
        match self {
            TableKind::Norm(r) => r.combine(v),
            TableKind::Frac(r) => r.combine(v),
            TableKind::Mul(r) => r.combine(v),
            TableKind::Pair(r) => r.combine(v),
            TableKind::Range(r) => r.combine(v),
        }
    }
}

impl FrameworkEval for TableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_table_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let row: Vec<E::F> =
            self.cols.iter().map(|c| eval.get_preprocessed_column(c.clone())).collect();
        let m = eval.next_trace_mask();
        let neg = -E::EF::from(m);
        match &self.kind {
            TableKind::Norm(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Frac(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Mul(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Pair(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Range(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
        }
        eval.finalize_logup();
        eval
    }
}

/// A `0 .. 2^bits` range table.
#[derive(Clone)]
pub struct RangeTableEval {
    pub bits: u32,
    pub rel: RangeRel,
}

impl FrameworkEval for RangeTableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_range_table_eval" }

    fn log_size(&self) -> u32 {
        self.bits
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.bits + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let v = eval.get_preprocessed_column(tbl_range(self.bits));
        let m = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(&self.rel, -E::EF::from(m),
                                                core::slice::from_ref(&v)));
        eval.finalize_logup();
        eval
    }
}
