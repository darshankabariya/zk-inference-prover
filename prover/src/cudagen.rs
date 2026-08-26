//! B7a milestone 2b step 2: the constraint-kernel generator.
//!
//! Section 14.6's plan, parts 1 and 3 -- **lower** and **check**. This module
//! turns one `FrameworkEval` into a straight-line IR plus an ordered vector of
//! runtime scalars, and then proves the lowering right *without a CUDA compiler
//! in the loop*, by interpreting the IR against random columns and comparing to
//! the fork's own `CpuDomainEvaluator` on the same data. That check covers
//! every part of the lowering that could be silently wrong: the column mapping,
//! the relation binding, the constraint order, and the scalar extraction.
//!
//! Three things make it mechanical (all measured in section 14):
//!
//!  * `ExprEvaluator` handles every construct our eleven evals use, and the
//!    LogUp prefix-sum identities are already in its constraint list, so there
//!    is no separate logup pass and no global `Fraction` buffer;
//!  * the k-th `intermediate` is the k-th `add_to_relation` entry, because our
//!    evals never call `add_intermediate` -- so relation elements bind by
//!    position and the type-name collision (14.5.1) cannot bite;
//!  * relation elements do not need a hand-written accessor per family. A
//!    relation's `combine` is `sum alpha_i v_i - z` and `Relation` is generic in
//!    the value type, so `RelationProbe` below recovers `z` and every `alpha_i`
//!    numerically -- `z = -combine(0)`, `alpha_j = combine(e_j) - combine(0)` --
//!    in `add_to_relation` order, from any eval, with no per-family code.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SECURE_EXTENSION_DEGREE, SecureField};
use stwo::core::pcs::TreeVec;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::prover::backend::CpuBackend;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo_constraint_framework::expr::{BaseExpr, ExprEvaluator, ExtExpr};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    CpuDomainEvaluator, EvalAtRow, FrameworkEval, InfoEvaluator, Relation, RelationEntry,
};

// ------------------------------------------------------ relation recovery ----

/// One `add_to_relation` entry's elements, recovered numerically.
#[derive(Clone, Debug)]
pub struct RelEntry {
    pub name: String,
    pub z: SecureField,
    /// `alpha_0 .. alpha_{n-1}` for the `n` values this entry actually enters
    pub alphas: Vec<SecureField>,
}

/// An `EvalAtRow` over concrete field values whose only job is to see every
/// `add_to_relation` call and interrogate its relation. Mask values are
/// irrelevant to what it extracts, so it hands out a fixed one; nothing our
/// evals do with a mask can fault (section 14.5.3: the only `Inv` in the whole
/// proof is over a compile-time constant).
struct RelationProbe {
    out: Vec<RelEntry>,
}

impl EvalAtRow for RelationProbe {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        _interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        offsets.map(|_| BaseField::one())
    }

    fn get_preprocessed_column(&mut self, _column: PreProcessedColumnId) -> Self::F {
        BaseField::one()
    }

    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: From<G>,
    {
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        SecureField::from_m31_array(values)
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        // `combine` is linear in its values, so two evaluations per element
        // recover the element: `combine(0) = -z` and `combine(e_j) = alpha_j - z`.
        // Probe at the relation's declared width and record every alpha; the
        // lowering only asks for the ones the symbolic expression names, so any
        // extra are simply unused. Reaching the relation needs the accessor
        // `gpu/m2b_fork.py` adds.
        let n = entry.relation().get_size();
        let zeros = vec![BaseField::zero(); n];
        let base = entry.relation().combine(&zeros);
        let z = -base;
        let alphas = (0..n)
            .map(|j| {
                let mut v = zeros.clone();
                v[j] = BaseField::one();
                entry.relation().combine(&v) - base
            })
            .collect();
        self.out.push(RelEntry {
            name: entry.relation().get_name().to_string(),
            z,
            alphas,
        });
    }

    // The prefix-sum identities are not needed here; only the entries are.
    fn write_logup_frac(&mut self, _f: stwo::core::Fraction<Self::EF, Self::EF>) {}
    fn finalize_logup_batched(&mut self, _b: &Vec<usize>) {}
    fn finalize_logup(&mut self) {}
    fn finalize_logup_in_pairs(&mut self) {}
}

pub fn relation_entries<E: FrameworkEval>(eval: &E) -> Vec<RelEntry> {
    eval.evaluate(RelationProbe { out: vec![] }).out
}

// ------------------------------------------------------- the fraction probe ----

/// `ExprEvaluator` in everything except that it also keeps each
/// `add_to_relation` entry's *numerator*.
///
/// Section 18.7's finding, in one sentence: the denominators need no new pass at
/// all. `ExprEvaluator::add_to_relation` is
///
/// ```text
/// let intermediate = self.add_extension_intermediate(combine_formal(relation, values));
/// self.write_logup_frac(Fraction::new(multiplicity, intermediate));
/// ```
///
/// so the k-th extension intermediate *is* the k-th lookup's LogUp denominator,
/// symbolically, in `add_to_relation` order -- and `lower_mut` already lowers
/// every one of them (14.5.1 depends on it). The only thing the constraint pass
/// throws away is the numerator, which goes straight into `write_logup_frac`
/// and is never named. This keeps it.
///
/// Every method delegates, so the inner evaluator's counters, intermediates and
/// constraints come out exactly as they would without the probe -- which is
/// what lets `lower_mut` run through it unconditionally, and what makes 2b's
/// `cudacheck` the regression test for the delegation itself.
pub struct LogupProbe {
    pub inner: ExprEvaluator,
    /// one per `add_to_relation`, in call order
    pub mults: Vec<ExtExpr>,
}

impl LogupProbe {
    pub fn new() -> Self {
        Self { inner: ExprEvaluator::new(), mults: vec![] }
    }
}

impl Default for LogupProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalAtRow for LogupProbe {
    type F = BaseExpr;
    type EF = ExtExpr;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        self.inner.next_interaction_mask(interaction, offsets)
    }

    fn next_extension_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::EF; N] {
        self.inner.next_extension_interaction_mask(interaction, offsets)
    }

    fn next_trace_mask(&mut self) -> Self::F {
        self.inner.next_trace_mask()
    }

    fn get_preprocessed_column(&mut self, column: PreProcessedColumnId) -> Self::F {
        self.inner.get_preprocessed_column(column)
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF> + From<G>,
    {
        self.inner.add_constraint(constraint)
    }

    fn add_intermediate(&mut self, expr: Self::F) -> Self::F {
        self.inner.add_intermediate(expr)
    }

    fn add_extension_intermediate(&mut self, expr: Self::EF) -> Self::EF {
        self.inner.add_extension_intermediate(expr)
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        ExprEvaluator::combine_ef(values)
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        self.mults.push(entry.multiplicity().clone());
        self.inner.add_to_relation(entry);
    }

    fn write_logup_frac(&mut self, f: stwo::core::Fraction<Self::EF, Self::EF>) {
        self.inner.write_logup_frac(f)
    }
    fn finalize_logup_batched(&mut self, b: &Vec<usize>) {
        self.inner.finalize_logup_batched(b)
    }
    fn finalize_logup(&mut self) {
        self.inner.finalize_logup()
    }
    fn finalize_logup_in_pairs(&mut self) {
        self.inner.finalize_logup_in_pairs()
    }
}

// -------------------------------------------------------------------- IR ----

/// One instruction. Every operand is the index of an earlier instruction, so
/// the vector is already in dependency order and the emitter is a single pass.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// `trace[tree][col]` at this row plus `off` on the circle domain
    Load { tree: usize, col: usize, off: isize },
    /// a base-field literal
    ConstM31(u32),
    /// slot `k` of the runtime scalar blob
    Scalar(usize),
    /// the logup shift `claimed_sum / 2^log_size`. The fork already computes it
    /// and passes it to every constraint kernel, so it is an argument rather
    /// than a blob slot -- which is what lets `cuda_eval_blob(&self)` build the
    /// blob without the component's claimed sum.
    CumsumShift,
    AddB(usize, usize),
    SubB(usize, usize),
    MulB(usize, usize),
    NegB(usize),
    AddE(usize, usize),
    SubE(usize, usize),
    MulE(usize, usize),
    NegE(usize),
    /// four base values into one extension value
    Pack(usize, usize, usize, usize),
}

/// The lowered component: what the kernel executes, and what the prover must
/// upload alongside it.
///
/// Serialisable, because the generated kernel set is the *union* over every
/// design point we prove: the constraint shape of a `dequant` region depends on
/// its matmul, so the synthetic layer and the real layer-12 witness do not
/// produce the same shapes. `cudaemit` accumulates into `$GEN/shapes.json` and
/// re-emits every kernel it has ever seen, so one `libstwo_cuda.a` serves all of
/// them and adding a design point is one more emit run.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Ir {
    pub ops: Vec<Op>,
    /// 1 for a base-field value (a `SecureCol` whose upper limbs are literal
    /// zero); 4 otherwise. A product costs `width(a) * width(b)` M31 multiplies.
    pub width: Vec<u8>,
    /// node id per constraint, in the order `add_constraint` was called -- which
    /// is the order `random_coeff_powers` is indexed in
    pub constraints: Vec<usize>,
    /// `(numerator, denominator)` node ids per lookup, in `add_to_relation`
    /// order -- empty unless the IR came from `lower_logup`. Section 18.7's
    /// emitter mode: the same lowering, stopped at the fraction instead of
    /// continued into the constraint.
    #[serde(default)]
    pub fracs: Vec<(usize, usize)>,
    /// the runtime blob, in slot order
    pub scalars: Vec<SecureField>,
    /// references per slot: the structural signature the prover asserts against
    /// the generated kernel, so a value collision or a drift in the B6.5 crate
    /// aborts instead of proving the wrong statement
    pub scalar_refs: Vec<u32>,
    /// columns per tree: `[preprocessed, base, interaction]`
    pub cols: [usize; 3],
    /// the preprocessed ids, in `get_preprocessed_column` call order
    pub pre_ids: Vec<String>,
    pub log_size: u32,
    pub eval_log_size: u32,
    /// only for the interpreter; the kernel receives this as an argument
    pub cumsum_shift: SecureField,
}

fn operands(o: &Op) -> Vec<usize> {
    match *o {
        Op::Load { .. } | Op::ConstM31(_) | Op::Scalar(_) | Op::CumsumShift => vec![],
        Op::NegB(a) | Op::NegE(a) => vec![a],
        Op::AddB(a, b)
        | Op::SubB(a, b)
        | Op::MulB(a, b)
        | Op::AddE(a, b)
        | Op::SubE(a, b)
        | Op::MulE(a, b) => vec![a, b],
        Op::Pack(a, b, c, d) => vec![a, b, c, d],
    }
}

fn remap(o: &Op, p: &[usize]) -> Op {
    match *o {
        Op::Load { tree, col, off } => Op::Load { tree, col, off },
        Op::ConstM31(c) => Op::ConstM31(c),
        Op::Scalar(k) => Op::Scalar(k),
        Op::CumsumShift => Op::CumsumShift,
        Op::AddB(a, b) => Op::AddB(p[a], p[b]),
        Op::SubB(a, b) => Op::SubB(p[a], p[b]),
        Op::MulB(a, b) => Op::MulB(p[a], p[b]),
        Op::NegB(a) => Op::NegB(p[a]),
        Op::AddE(a, b) => Op::AddE(p[a], p[b]),
        Op::SubE(a, b) => Op::SubE(p[a], p[b]),
        Op::MulE(a, b) => Op::MulE(p[a], p[b]),
        Op::NegE(a) => Op::NegE(p[a]),
        Op::Pack(a, b, c, d) => Op::Pack(p[a], p[b], p[c], p[d]),
    }
}

impl Ir {
    /// Reorder into *demand* order: for each constraint in turn, the nodes it
    /// still needs, dependencies first. Dead nodes are dropped.
    ///
    /// This is not cosmetic. Lowering produces the intermediates first and the
    /// constraints after, so all of a component's relation combines -- 192 of
    /// them in `chan` -- are live at once, 3 kB of QM31 per thread, and ptxas
    /// spills. In demand order the two combines a batched LogUp constraint needs
    /// are computed immediately before it and die immediately after. Section 16.3
    /// measures what that is worth.
    pub fn schedule(&mut self) {
        let n = self.ops.len();
        // Depth of each node. Operands always have a lower index than the node
        // that uses them -- the DAG is interned bottom-up -- so one forward pass
        // suffices.
        let mut depth = vec![0u32; n];
        for i in 0..n {
            depth[i] = 1 + operands(&self.ops[i]).iter().map(|c| depth[*c]).max().unwrap_or(0);
        }
        let mut done = vec![false; n];
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let roots: Vec<usize> = if self.fracs.is_empty() {
            self.constraints.clone()
        } else {
            // demand order over the fractions: a lookup's numerator and
            // denominator die together, one lookup at a time
            self.fracs.iter().flat_map(|(n, d)| [*n, *d]).collect()
        };
        for root in roots {
            let mut stack = vec![(root, false)];
            while let Some((id, expanded)) = stack.pop() {
                if done[id] {
                    continue;
                }
                if expanded {
                    done[id] = true;
                    order.push(id);
                } else {
                    stack.push((id, true));
                    // Deepest operand first (Sethi-Ullman's rule): a 128-term
                    // Horner fold has a deep accumulator and a shallow term, and
                    // evaluating the term first would keep it live across the
                    // whole chain. The stack is LIFO, so push shallowest last.
                    let mut kids: Vec<usize> =
                        operands(&self.ops[id]).into_iter().filter(|c| !done[*c]).collect();
                    kids.sort_by_key(|c| depth[*c]);
                    for c in kids {
                        stack.push((c, false));
                    }
                }
            }
        }
        let mut pos = vec![usize::MAX; n];
        for (i, &old) in order.iter().enumerate() {
            pos[old] = i;
        }
        self.ops = order.iter().map(|&o| remap(&self.ops[o], &pos)).collect();
        self.width = order.iter().map(|&o| self.width[o]).collect();
        self.constraints = self.constraints.iter().map(|c| pos[*c]).collect();
        self.fracs = self.fracs.iter().map(|(n, d)| (pos[*n], pos[*d])).collect();
        debug_assert!(self.constraints.iter().all(|c| *c != usize::MAX));
        debug_assert!(self.fracs.iter().all(|(n, d)| *n != usize::MAX && *d != usize::MAX));
    }

    /// M31 multiplies per row, width-aware (section 14.5.4).
    pub fn m31_muls(&self) -> usize {
        self.ops
            .iter()
            .enumerate()
            .map(|(_, o)| match o {
                Op::MulB(..) => 1,
                Op::MulE(a, b) => self.width[*a] as usize * self.width[*b] as usize,
                _ => 0,
            })
            .sum()
    }

    pub fn loads(&self) -> usize {
        self.ops.iter().filter(|o| matches!(o, Op::Load { .. })).count()
    }
}

// ---------------------------------------------------------------- lowering ----

struct Lower {
    ir: Ir,
    intern: HashMap<String, usize>,
    slot: HashMap<[u32; 4], usize>,
    /// interaction -> symbolic `idx` -> real column index in that interaction
    colmap: Vec<HashMap<usize, usize>>,
    /// preprocessed id -> the first slot of tree 0 holding it
    pre_index: HashMap<String, usize>,
    /// "intermediateK" -> node id
    inter: HashMap<String, usize>,
    /// the relation entry whose intermediate is being lowered
    rel: Option<RelEntry>,
    claimed_sum: SecureField,
    zero: usize,
    mutate: Mutate,
}

/// `trace_{i}_column_{idx}_offset_{n|neg_n}` -- `ColumnExpr`'s fields are private
/// in the fork, and `format_expr` on the leaf is the public way to read them.
fn parse_col(e: &BaseExpr) -> (usize, usize, isize) {
    let s = e.format_expr();
    let f: Vec<&str> = s.split('_').collect();
    match f.as_slice() {
        ["trace", i, "column", idx, "offset", "neg", n] => (
            i.parse().unwrap(),
            idx.parse().unwrap(),
            -n.parse::<isize>().unwrap(),
        ),
        ["trace", i, "column", idx, "offset", n] => {
            (i.parse().unwrap(), idx.parse().unwrap(), n.parse().unwrap())
        }
        _ => panic!("unparsed column leaf {s:?}"),
    }
}

fn collect_cols(e: &BaseExpr, out: &mut BTreeSet<(usize, usize)>) {
    match e {
        BaseExpr::Col(_) => {
            let (i, idx, _) = parse_col(e);
            out.insert((i, idx));
        }
        BaseExpr::Const(_) | BaseExpr::Param(_) => {}
        BaseExpr::Add(a, b) | BaseExpr::Sub(a, b) | BaseExpr::Mul(a, b) => {
            collect_cols(a, out);
            collect_cols(b, out);
        }
        BaseExpr::Neg(a) | BaseExpr::Inv(a) => collect_cols(a, out),
    }
}

fn collect_cols_ext(e: &ExtExpr, out: &mut BTreeSet<(usize, usize)>) {
    match e {
        ExtExpr::Const(_) | ExtExpr::Param(_) => {}
        ExtExpr::SecureCol(p) => p.iter().for_each(|b| collect_cols(b, out)),
        ExtExpr::Add(a, b) | ExtExpr::Sub(a, b) | ExtExpr::Mul(a, b) => {
            collect_cols_ext(a, out);
            collect_cols_ext(b, out);
        }
        ExtExpr::Neg(a) => collect_cols_ext(a, out),
    }
}

fn qm31_const(e: &ExtExpr) -> Option<SecureField> {
    match e {
        ExtExpr::Const(c) => Some(*c),
        ExtExpr::SecureCol(p) => {
            let mut out = [M31::zero(); 4];
            for (i, b) in p.iter().enumerate() {
                match &**b {
                    BaseExpr::Const(c) => out[i] = *c,
                    _ => return None,
                }
            }
            Some(SecureField::from_m31_array(out))
        }
        _ => None,
    }
}

fn limbs(v: SecureField) -> [u32; 4] {
    let a = v.to_m31_array();
    [a[0].0, a[1].0, a[2].0, a[3].0]
}

impl Lower {
    fn push(&mut self, key: String, op: Op, w: u8) -> usize {
        if let Some(i) = self.intern.get(&key) {
            // a re-reference: charge it to the scalar's count, so the signature
            // counts uses and not just distinct values
            if let Op::Scalar(k) = self.ir.ops[*i] {
                self.ir.scalar_refs[k] += 1;
            }
            return *i;
        }
        let i = self.ir.ops.len();
        if let Op::Scalar(k) = op {
            self.ir.scalar_refs[k] += 1;
        }
        self.ir.ops.push(op);
        self.ir.width.push(w);
        self.intern.insert(key, i);
        i
    }

    fn scalar(&mut self, v: SecureField) -> usize {
        let l = limbs(v);
        let k = match self.slot.get(&l) {
            Some(k) => *k,
            None => {
                let k = self.ir.scalars.len();
                self.ir.scalars.push(v);
                self.ir.scalar_refs.push(0);
                self.slot.insert(l, k);
                k
            }
        };
        let w = if l[1] == 0 && l[2] == 0 && l[3] == 0 { 1 } else { 4 };
        self.push(format!("S{k}"), Op::Scalar(k), w)
    }

    fn constm31(&mut self, c: M31) -> usize {
        self.push(format!("K{}", c.0), Op::ConstM31(c.0), 1)
    }

    /// The value of a node if it is a compile-time base-field constant.
    fn fold(&self, id: usize) -> Option<M31> {
        match self.ir.ops[id] {
            Op::ConstM31(c) => Some(M31::from_u32_unchecked(c)),
            _ => None,
        }
    }

    /// A node that is literally 0 or 1 -- as an M31 literal or as a scalar slot
    /// whose value the transcript fixes. `combine_formal` starts its fold at
    /// `ExtExpr::zero()` and a relation's `alpha_powers[0]` is 1, so every one of
    /// the 2,026 relation entries otherwise emits an `add(0, .)` and a `mul(1, .)`.
    /// Folding them is the same rewriting `expr::simplify` does, done on the DAG
    /// where it costs one comparison instead of a clone.
    ///
    /// It does mean the IR shape depends on a scalar's *value*, so a
    /// challenge-derived scalar that came out exactly 0 or 1 would fold and
    /// change the fingerprint. That is a 2^-124 event per scalar, and its
    /// consequence is safe by construction: the fingerprint would not be in the
    /// generated table and `blob_for` aborts the prover rather than handing the
    /// component to a kernel that computes something else.
    fn lit(&self, id: usize) -> Option<u32> {
        match self.ir.ops[id] {
            Op::ConstM31(c) => Some(c),
            Op::Scalar(k) => {
                let l = limbs(self.ir.scalars[k]);
                if l[1] == 0 && l[2] == 0 && l[3] == 0 { Some(l[0]) } else { None }
            }
            _ => None,
        }
    }
    fn is0(&self, id: usize) -> bool {
        self.lit(id) == Some(0)
    }
    fn is1(&self, id: usize) -> bool {
        self.lit(id) == Some(1)
    }

    fn base(&mut self, e: &BaseExpr) -> usize {
        match e {
            BaseExpr::Col(_) => {
                let (i, idx, off) = parse_col(e);
                let col = *self.colmap[i]
                    .get(&idx)
                    .unwrap_or_else(|| panic!("column idx {idx} of interaction {i} unmapped"));
                self.push(
                    format!("C{i}_{col}_{off}"),
                    Op::Load { tree: i, col, off },
                    1,
                )
            }
            BaseExpr::Const(c) => self.constm31(*c),
            BaseExpr::Param(p) => {
                if p == "column_size" {
                    // only ever under an `Inv`; folded there
                    return self.constm31(M31::from_u32_unchecked(1 << self.ir.log_size));
                }
                if let Some(c) = self.pre_index.get(p).copied() {
                    return self.push(
                        format!("C0_{c}_0"),
                        Op::Load { tree: 0, col: c, off: 0 },
                        1,
                    );
                }
                panic!("unbound base parameter {p:?}");
            }
            BaseExpr::Add(a, b) => {
                let (x, y) = (self.base(a), self.base(b));
                if self.is0(x) {
                    return y;
                }
                if self.is0(y) {
                    return x;
                }
                self.push(format!("A{x},{y}"), Op::AddB(x, y), 1)
            }
            BaseExpr::Sub(a, b) => {
                let (x, y) = (self.base(a), self.base(b));
                if self.is0(y) {
                    return x;
                }
                self.push(format!("B{x},{y}"), Op::SubB(x, y), 1)
            }
            BaseExpr::Mul(a, b) => {
                let (x, y) = (self.base(a), self.base(b));
                if self.is0(x) || self.is0(y) {
                    return self.zero;
                }
                if self.is1(x) {
                    return y;
                }
                if self.is1(y) {
                    return x;
                }
                self.push(format!("M{x},{y}"), Op::MulB(x, y), 1)
            }
            BaseExpr::Neg(a) => {
                let x = self.base(a);
                self.push(format!("N{x}"), Op::NegB(x), 1)
            }
            BaseExpr::Inv(a) => {
                let x = self.base(a);
                // Section 14.5.3: the only inverse in the composed proof is over
                // `column_size`. Folding it is what keeps a field inversion out
                // of the kernel; a non-constant operand is a hard error, not a
                // silently expensive path.
                let c = self.fold(x).unwrap_or_else(|| {
                    panic!("Inv over a non-constant operand: {}", a.format_expr())
                });
                assert!(!c.is_zero(), "Inv over zero");
                self.constm31(c.inverse())
            }
        }
    }

    fn ext(&mut self, e: &ExtExpr) -> usize {
        if let Some(c) = qm31_const(e) {
            return self.scalar(c);
        }
        match e {
            ExtExpr::Const(_) => unreachable!("handled by qm31_const"),
            ExtExpr::Param(p) => {
                if let Some(id) = self.inter.get(p).copied() {
                    return id;
                }
                assert_ne!(
                    p, "claimed_sum",
                    "claimed_sum outside the cumsum_shift product: the kernel gets \
                     the shift as an argument, not the sum"
                );
                // A relation element. Bound by *position*, never by name: the
                // active entry is the one whose intermediate is being lowered
                // (section 14.5.1).
                let r = self
                    .rel
                    .clone()
                    .unwrap_or_else(|| panic!("relation parameter {p:?} outside a relation entry"));
                let v = if let Some(rest) = p.strip_suffix("_z") {
                    assert_eq!(rest, r.name, "relation name {rest:?} != entry {:?}", r.name);
                    r.z
                } else {
                    let (rest, i) = p.rsplit_once("_alpha").unwrap_or_else(|| {
                        panic!("unrecognised extension parameter {p:?}")
                    });
                    assert_eq!(rest, r.name, "relation name {rest:?} != entry {:?}", r.name);
                    let i: usize = i.parse().unwrap();
                    assert!(i < r.alphas.len(), "alpha{i} beyond {} declared", r.alphas.len());
                    r.alphas[i]
                };
                self.scalar(v)
            }
            ExtExpr::SecureCol(p) => {
                let ids: Vec<usize> = p.iter().map(|b| self.base(b)).collect();
                if ids[1..].iter().all(|i| self.is0(*i)) {
                    return ids[0];
                }
                self.push(
                    format!("V{},{},{},{}", ids[0], ids[1], ids[2], ids[3]),
                    Op::Pack(ids[0], ids[1], ids[2], ids[3]),
                    4,
                )
            }
            ExtExpr::Add(a, b) => {
                let (x, y) = (self.ext(a), self.ext(b));
                if self.is0(x) {
                    return y;
                }
                if self.is0(y) {
                    return x;
                }
                let w = self.ir.width[x].max(self.ir.width[y]);
                self.push(format!("a{x},{y}"), Op::AddE(x, y), w)
            }
            ExtExpr::Sub(a, b) => {
                let (x, y) = (self.ext(a), self.ext(b));
                if self.is0(y) {
                    return x;
                }
                let w = self.ir.width[x].max(self.ir.width[y]);
                self.push(format!("b{x},{y}"), Op::SubE(x, y), w)
            }
            ExtExpr::Mul(a, b) => {
                // `logup_proxy!`'s shift, built by `FormalLogupAtRow::new` as
                // `Param("claimed_sum") * Inv(Param("column_size"))`. Recognised
                // here so it becomes the kernel's own `cumsum_shift` argument.
                if let ExtExpr::Param(p) = &**a {
                    if p == "claimed_sum" {
                        match &**b {
                            ExtExpr::SecureCol(q) => {
                                assert!(
                                    matches!(&*q[0], BaseExpr::Inv(inner)
                                             if matches!(&**inner, BaseExpr::Param(n) if n == "column_size")),
                                    "claimed_sum multiplied by something other than 1/column_size"
                                );
                                for hi in &q[1..] {
                                    assert!(
                                        matches!(&**hi, BaseExpr::Const(c) if c.is_zero()),
                                        "cumsum_shift's upper limbs are not zero"
                                    );
                                }
                            }
                            other => panic!("unexpected cumsum_shift shape: {}", other.format_expr()),
                        }
                        return self.push("Z".to_string(), Op::CumsumShift, 4);
                    }
                }
                let (x, y) = (self.ext(a), self.ext(b));
                if self.is0(x) || self.is0(y) {
                    return self.zero;
                }
                if self.is1(x) {
                    return y;
                }
                if self.is1(y) {
                    return x;
                }
                let w = if self.ir.width[x] == 1 && self.ir.width[y] == 1 { 1 } else { 4 };
                self.push(format!("m{x},{y}"), Op::MulE(x, y), w)
            }
            ExtExpr::Neg(a) => {
                let x = self.ext(a);
                let w = self.ir.width[x];
                self.push(format!("n{x}"), Op::NegE(x), w)
            }
        }
    }
}

/// A deliberate defect, for the negative control. The equivalence check is only
/// worth reporting if it would fail when the lowering is wrong in the ways the
/// design could plausibly be wrong, so each of these is a design decision from
/// section 14.5 inverted:
///
/// * `relbyname` -- bind relation elements by relation *type name* instead of by
///   `add_to_relation` position (14.5.1). Correct for a component that uses one
///   instance per name; wrong for `chan` and `row`.
/// * `colshift` -- rotate the column mapping of the base trace by one (14.2).
/// * `constorder` -- reverse the constraint order, i.e. pair each constraint with
///   the wrong random coefficient.
/// * `fracorder` -- reverse the *lookup* order, which is `constorder`'s analogue
///   for the fraction lowering: `finalize_logup_in_pairs` pairs consecutive
///   entries, so a reversed order pairs the wrong two lookups into a column.
/// * `packwidth` -- claim every packed value is width 4, which is the cost model
///   the width rule replaces (14.5.4). Must *not* change any value: this one is
///   the control's control.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mutate {
    None,
    RelByName,
    ColShift,
    /// B7a milestone 4 (section 3 row 1): `ColShift` for the *preprocessed*
    /// tree. The shared tables commit one base column each -- their
    /// multiplicity -- so a rotation of the base mapping is the identity there
    /// and `colshift` has no teeth; what those families do read is several
    /// preprocessed columns, and this is the defect that bites them.
    PreCol,
    ConstOrder,
    PackWidth,
    FracOrder,
}

impl Mutate {
    pub fn parse(s: &str) -> Self {
        match s {
            "none" => Mutate::None,
            "relbyname" => Mutate::RelByName,
            "colshift" => Mutate::ColShift,
            "precol" => Mutate::PreCol,
            "constorder" => Mutate::ConstOrder,
            "packwidth" => Mutate::PackWidth,
            "fracorder" => Mutate::FracOrder,
            other => panic!("unknown mutation {other:?}"),
        }
    }
}

/// Lower one component. `log_size`/`eval_log_size` are the domains the kernel
/// will run on; `claimed_sum` is the component's own.
pub fn lower<E: FrameworkEval>(
    eval: &E,
    claimed_sum: SecureField,
    log_size: u32,
    eval_log_size: u32,
) -> Ir {
    lower_mut(eval, claimed_sum, log_size, eval_log_size, Mutate::None)
}

/// The same lowering, stopped at the LogUp fractions (18.7's emitter mode).
///
/// The IR's `fracs` are `(numerator, denominator)` node ids in
/// `add_to_relation` order; `constraints` is empty, so `schedule` puts the
/// nodes in demand order over the *lookups*. Nothing about the columns, the
/// relation binding or the scalar blob changes -- this is the same `Lower` over
/// the same expressions, with a different output list.
///
/// `claimed_sum` is not read: a fraction depends on the trace and the relation
/// elements, never on the component's own sum, and `Op::CumsumShift` can only
/// appear inside a constraint. Passing zero is correct and says so.
pub fn lower_logup<E: FrameworkEval>(eval: &E, log_size: u32) -> Ir {
    lower_impl(eval, SecureField::zero(), log_size, log_size, Mutate::None, true)
}

pub fn lower_logup_mut<E: FrameworkEval>(eval: &E, log_size: u32, mutate: Mutate) -> Ir {
    lower_impl(eval, SecureField::zero(), log_size, log_size, mutate, true)
}

pub fn lower_mut<E: FrameworkEval>(
    eval: &E,
    claimed_sum: SecureField,
    log_size: u32,
    eval_log_size: u32,
    mutate: Mutate,
) -> Ir {
    lower_impl(eval, claimed_sum, log_size, eval_log_size, mutate, false)
}

fn lower_impl<E: FrameworkEval>(
    eval: &E,
    claimed_sum: SecureField,
    log_size: u32,
    eval_log_size: u32,
    mutate: Mutate,
    logup: bool,
) -> Ir {
    // The probe delegates every method, so `ev` is what `ExprEvaluator::new()`
    // alone would have produced; `mults` is the one thing it adds.
    let probe = eval.evaluate(LogupProbe::new());
    let mults = probe.mults;
    let ev = probe.inner;
    let info = eval.evaluate(InfoEvaluator::new(log_size, vec![], claimed_sum));
    let rels = relation_entries(eval);

    assert!(
        ev.intermediates.is_empty(),
        "base intermediates are not lowered: the k-th intermediate would stop \
         being the k-th relation entry (section 14.5.1)"
    );
    assert_eq!(
        ev.ext_intermediates.len(),
        rels.len(),
        "relation entries and extension intermediates disagree"
    );
    assert_eq!(
        ev.constraints.len(),
        info.n_constraints,
        "symbolic and billed constraint counts disagree"
    );

    // The column map, verified in section 14.2: `ExprEvaluator` numbers mask
    // columns with one counter across all interactions, so the j-th column of
    // interaction i is the j-th smallest idx seen there.
    let mut seen = BTreeSet::new();
    for x in ev.ext_intermediates.values() {
        collect_cols_ext(x, &mut seen);
    }
    for c in &ev.constraints {
        collect_cols_ext(c, &mut seen);
    }
    let n_int = info.mask_offsets.len().max(3);
    let mut colmap = vec![HashMap::new(); n_int];
    for i in 0..n_int {
        let idxs: Vec<usize> = seen.iter().filter(|(t, _)| *t == i).map(|(_, x)| *x).collect();
        if let Some(cols) = info.mask_offsets.get(i) {
            assert_eq!(
                idxs.len(),
                cols.len(),
                "interaction {i}: {} symbolic columns, {} billed",
                idxs.len(),
                cols.len()
            );
        }
        let n = idxs.len();
        for (j, idx) in idxs.into_iter().enumerate() {
            let j = if mutate == Mutate::ColShift && i == 1 && n > 1 { (j + 1) % n } else { j };
            colmap[i].insert(idx, j);
        }
    }

    let pre_ids: Vec<String> = info.preprocessed_columns.iter().map(|c| c.id.clone()).collect();
    let mut pre_index = HashMap::new();
    let n_pre = pre_ids.len();
    for (i, id) in pre_ids.iter().enumerate() {
        // B7a milestone 4 (section 3 row 1): `PreCol` is `ColShift` for the
        // preprocessed tree, which does not go through `colmap` -- a
        // preprocessed column is resolved by *id*, here. The shared tables
        // commit one base column each (their multiplicity) and make one lookup,
        // so `colshift` and `fracorder` are both the identity on them; this is
        // the defect that has teeth for those two families.
        let i = if mutate == Mutate::PreCol && n_pre > 1 { (i + 1) % n_pre } else { i };
        pre_index.entry(id.clone()).or_insert(i);
    }

    let cols = [
        pre_ids.len(),
        info.mask_offsets.get(1).map(|v| v.len()).unwrap_or(0),
        info.mask_offsets.get(2).map(|v| v.len()).unwrap_or(0),
    ];

    let mut lo = Lower {
        ir: Ir {
            ops: vec![],
            width: vec![],
            constraints: vec![],
            fracs: vec![],
            scalars: vec![],
            scalar_refs: vec![],
            cols,
            pre_ids,
            log_size,
            eval_log_size,
            cumsum_shift: claimed_sum
                * M31::from_u32_unchecked(1 << log_size).inverse(),
        },
        intern: HashMap::new(),
        slot: HashMap::new(),
        colmap,
        pre_index,
        inter: HashMap::new(),
        rel: None,
        claimed_sum,
        zero: 0,
        mutate,
    };
    // node 0 is the literal zero, so `SecureCol`'s width test has a referent
    lo.zero = lo.constm31(M31::zero());

    // The intermediates first, in creation order, each with its relation entry
    // active. `ordered_intermediates` is private in the fork, but the names are
    // `intermediate{n}` from a single counter, so sorting by the suffix recovers
    // the order exactly.
    let mut names: Vec<String> = ev.ext_intermediates.keys().cloned().collect();
    names.sort_by_key(|n| {
        n.strip_prefix("intermediate")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| panic!("unexpected intermediate name {n:?}"))
    });
    // The defect `relbyname` models: one binding per relation type name, taken
    // from the first entry that uses it.
    let mut first_by_name: BTreeMap<String, RelEntry> = BTreeMap::new();
    for r in &rels {
        first_by_name.entry(r.name.clone()).or_insert_with(|| r.clone());
    }

    for (k, n) in names.iter().enumerate() {
        let expr = ev.ext_intermediates[n].clone();
        lo.rel = Some(match mutate {
            Mutate::RelByName => first_by_name[&rels[k].name].clone(),
            _ => rels[k].clone(),
        });
        let id = lo.ext(&expr);
        lo.rel = None;
        lo.inter.insert(n.clone(), id);
    }

    if logup {
        // The k-th intermediate is the k-th lookup's denominator; the numerator
        // is lowered here, with the same entry active, because a multiplicity
        // may name the relation's own elements.
        assert_eq!(mults.len(), names.len(), "numerators and denominators disagree");
        for (k, m) in mults.iter().enumerate() {
            lo.rel = Some(rels[k].clone());
            let num = lo.ext(m);
            lo.rel = None;
            let den = lo.inter[&names[k]];
            lo.ir.fracs.push((num, den));
        }
        if mutate == Mutate::FracOrder {
            lo.ir.fracs.reverse();
        }
    } else {
        for c in &ev.constraints {
            let id = lo.ext(c);
            lo.ir.constraints.push(id);
        }
        if mutate == Mutate::ConstOrder {
            lo.ir.constraints.reverse();
        }
    }
    lo.ir.schedule();
    if mutate == Mutate::PackWidth {
        for (i, o) in lo.ir.ops.iter().enumerate() {
            if matches!(o, Op::Pack(..)) {
                lo.ir.width[i] = 4;
            }
        }
    }
    lo.ir
}

// ------------------------------------------------------------ interpretation ----

/// A node's value. Splitting base from extension is not an optimisation here --
/// it is the type check that catches a lowering that packed the wrong thing.
#[derive(Clone, Copy, Debug)]
enum Val {
    B(M31),
    E(SecureField),
}

impl Val {
    fn b(self) -> M31 {
        match self {
            Val::B(x) => x,
            Val::E(_) => panic!("extension value where a base value was required"),
        }
    }
    fn e(self) -> SecureField {
        match self {
            Val::E(x) => x,
            Val::B(x) => SecureField::from(x),
        }
    }
}

/// The columns one component sees, per tree, already on the evaluation domain.
pub type Cols<'a> = [Vec<&'a [M31]>; 3];

/// Run the IR at one row and return `row_res`: exactly what `CudaEvaluator`
/// accumulates and what the generated kernel will compute.
pub fn interpret(ir: &Ir, cols: &Cols<'_>, row: usize, rcp: &[SecureField]) -> SecureField {
    let mut v: Vec<Val> = Vec::with_capacity(ir.ops.len());
    for op in &ir.ops {
        let x = match *op {
            Op::Load { tree, col, off } => {
                let r = if off == 0 {
                    row
                } else {
                    offset_bit_reversed_circle_domain_index(
                        row,
                        ir.log_size,
                        ir.eval_log_size,
                        off,
                    )
                };
                Val::B(cols[tree][col][r])
            }
            Op::ConstM31(c) => Val::B(M31::from_u32_unchecked(c)),
            Op::Scalar(k) => Val::E(ir.scalars[k]),
            Op::CumsumShift => Val::E(ir.cumsum_shift),
            Op::AddB(a, b) => Val::B(v[a].b() + v[b].b()),
            Op::SubB(a, b) => Val::B(v[a].b() - v[b].b()),
            Op::MulB(a, b) => Val::B(v[a].b() * v[b].b()),
            Op::NegB(a) => Val::B(-v[a].b()),
            Op::AddE(a, b) => Val::E(v[a].e() + v[b].e()),
            Op::SubE(a, b) => Val::E(v[a].e() - v[b].e()),
            Op::MulE(a, b) => Val::E(v[a].e() * v[b].e()),
            Op::NegE(a) => Val::E(-v[a].e()),
            Op::Pack(a, b, c, d) => {
                Val::E(SecureField::from_m31_array([
                    v[a].b(),
                    v[b].b(),
                    v[c].b(),
                    v[d].b(),
                ]))
            }
        };
        v.push(x);
    }
    let mut res = SecureField::zero();
    for (i, id) in ir.constraints.iter().enumerate() {
        res += rcp[i] * v[*id].e();
    }
    res
}

// ------------------------------------------------------------------- check ----

/// Deterministic pseudorandom M31s: the check needs unpredictable column values,
/// not cryptographic ones, and it needs the same ones on both sides.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn m31(&mut self) -> M31 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        M31::from_u32_unchecked((self.0 % ((1 << 31) - 1)) as u32)
    }
    pub fn qm31(&mut self) -> SecureField {
        SecureField::from_m31_array([self.m31(), self.m31(), self.m31(), self.m31()])
    }
}

/// The report of one component's equivalence check.
#[derive(Debug)]
pub struct Check {
    pub name: String,
    pub rows: usize,
    pub mismatches: usize,
    pub first_mismatch: Option<(usize, String, String)>,
    pub n_constraints: usize,
    pub ops: usize,
    pub scalars: usize,
    pub m31_muls: usize,
    pub loads: usize,
    pub cols: [usize; 3],
}

/// Lower `eval` and compare the IR against the fork's own `CpuDomainEvaluator`
/// on the same random columns.
///
/// The check runs at a small `log_size` -- the real one would need gigabytes of
/// random columns for the wide components -- and both sides are told the same
/// one, so nothing is left implicit: `CpuDomainEvaluator` takes `log_size` for
/// its `LogupAtRow`, and the lowering folds `1 / column_size` from the same
/// value. What the check therefore does *not* exercise is that value at the real
/// domain size, which rungs 1-2 cover end to end.
pub fn check_one<E: FrameworkEval>(
    name: &str,
    eval: &E,
    claimed_sum: SecureField,
    log_size: u32,
    max_rows: usize,
    seed: u64,
    mutate: Mutate,
) -> Check {
    let eval_log = log_size + 1;
    let n = 1usize << eval_log;
    let ir = lower_mut(eval, claimed_sum, log_size, eval_log, mutate);

    // Duplicate preprocessed ids must hold identical content: in production the
    // allocator maps them to one global column, so the component sees the same
    // values twice.
    let mut rng = Rng::new(seed);
    let mut by_id: BTreeMap<String, Vec<M31>> = BTreeMap::new();
    for id in &ir.pre_ids {
        by_id
            .entry(id.clone())
            .or_insert_with(|| (0..n).map(|_| rng.m31()).collect());
    }
    let pre: Vec<Vec<M31>> = ir.pre_ids.iter().map(|id| by_id[id].clone()).collect();
    let base: Vec<Vec<M31>> = (0..ir.cols[1])
        .map(|_| (0..n).map(|_| rng.m31()).collect())
        .collect();
    let inter: Vec<Vec<M31>> = (0..ir.cols[2])
        .map(|_| (0..n).map(|_| rng.m31()).collect())
        .collect();
    let rcp: Vec<SecureField> = (0..ir.constraints.len()).map(|_| rng.qm31()).collect();

    // the same data, in the shape each side wants
    let cols: Cols<'_> = [
        pre.iter().map(|c| c.as_slice()).collect(),
        base.iter().map(|c| c.as_slice()).collect(),
        inter.iter().map(|c| c.as_slice()).collect(),
    ];
    let domain = CanonicCoset::new(eval_log).circle_domain();
    let mk = |v: &Vec<M31>| {
        CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(domain, v.clone())
    };
    let owned: Vec<Vec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
        vec![
            pre.iter().map(mk).collect(),
            base.iter().map(mk).collect(),
            inter.iter().map(mk).collect(),
        ];
    let trace: TreeVec<Vec<&CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>>> =
        TreeVec::new(owned.iter().map(|t| t.iter().collect()).collect());

    let step = (n / max_rows).max(1);
    let mut mismatches = 0;
    let mut first = None;
    let mut rows = 0;
    for row in (0..n).step_by(step) {
        rows += 1;
        let want = eval
            .evaluate(CpuDomainEvaluator::new(
                &trace,
                row,
                &rcp,
                log_size,
                eval_log,
                log_size,
                claimed_sum,
            ))
            .row_res;
        let got = interpret(&ir, &cols, row, &rcp);
        if got != want {
            mismatches += 1;
            if first.is_none() {
                first = Some((row, format!("{want:?}"), format!("{got:?}")));
            }
        }
    }

    Check {
        name: name.to_string(),
        rows,
        mismatches,
        first_mismatch: first,
        n_constraints: ir.constraints.len(),
        ops: ir.ops.len(),
        scalars: ir.scalars.len(),
        m31_muls: ir.m31_muls(),
        loads: ir.loads(),
        cols: ir.cols,
    }
}

// ------------------------------------------------------------- the subcommand ---

use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo_constraint_framework::FrameworkComponent;

use crate::compose::BuiltComponents;

fn check_component<E: FrameworkEval>(
    name: String,
    c: &FrameworkComponent<E>,
    log_size: u32,
    max_rows: usize,
    seed: u64,
    mutate: Mutate,
) -> Value {
    let k = check_one(&name, &c.eval, c.claimed_sum(), log_size, max_rows, seed, mutate);
    json!({
        "name": k.name,
        "cuda_eval_name": c.eval.cuda_eval_name(),
        "real_log_size": c.eval.log_size(),
        "checked_log_size": log_size,
        "rows_checked": k.rows,
        "mismatches": k.mismatches,
        "first_mismatch": k.first_mismatch.map(|(r, w, g)| json!({"row": r, "want": w, "got": g})),
        "n_constraints": k.n_constraints,
        "billed_constraints": c.n_constraints(),
        "ir_ops": k.ops,
        "scalars": k.scalars,
        "m31_muls_per_row": k.m31_muls,
        "loads_per_row": k.loads,
        "columns": {"preprocessed": k.cols[0], "base": k.cols[1], "interaction": k.cols[2]},
    })
}

/// `cudacheck`: lower every component of the composed proof and prove the
/// lowering right against `CpuDomainEvaluator`, with no CUDA in the loop.
pub fn cmd(args: &[String]) {
    let (l, vname) = crate::layer_of(args);
    let params = crate::params_of(&l, args, &vname);
    let plan = crate::compose::Plan::new(&params);
    let log_size: u32 = crate::arg_scalar(args, "--check-log-size", 8u32);
    let max_rows: usize = crate::arg_scalar(args, "--check-rows", 512usize);
    let muts: Vec<String> = crate::arg_strs(
        args,
        "--mutate",
        &["none", "relbyname", "colshift", "constorder", "packwidth"],
    );
    eprintln!(
        "cudacheck: {} bands={:?} variant={} at log_size={} ({} rows each)",
        l.source, l.band_sizes, vname, log_size, 1usize << (log_size + 1)
    );

    let mut ch = stwo::core::channel::Blake2sM31Channel::default();
    stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_2B02);
    let rels = crate::compose::Rels::draw(&mut ch, &plan.widths,
                                          plan.glue.as_ref().map(|g| g.uns.as_slice()));
    let f = stwo::core::channel::Channel::draw_secure_felts(&mut ch, 6);
    let claims = crate::exprprobe::claims_for(&plan, 0x0b65_2b02);
    let b = crate::compose::build_components(&plan, &rels, f[0], f[1], f[2], f[3], f[4], f[5], &claims);

    let mut passes = vec![];
    for m in &muts {
        let mutate = Mutate::parse(m);
        let mut rows = vec![];
        let mut seed = 0xC0FFEEu64;
        macro_rules! go {
            ($name:expr, $c:expr) => {{
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                rows.push(check_component($name, $c, log_size, max_rows, seed, mutate));
            }};
        }
        for i in 0..b.chans.len() {
            go!(format!("chan_band{i}"), &b.chans[i]);
            go!(format!("key_band{i}"), &b.keys[i]);
        }
        for i in 0..b.rows.len() {
            go!(format!("row_band{i}"), &b.rows[i]);
        }
        go!("attnout".to_string(), &b.ao);
        for i in 0..b.mats.len() {
            go!(format!("dequant{i}"), &b.mats[i]);
        }
        for i in 0..b.walks.len() {
            go!(format!("walk{i}"), &b.walks[i]);
        }
        go!("exp_table".to_string(), &b.exp_t);
        for i in 0..b.tables.len() {
            go!(format!("table{i}"), &b.tables[i]);
        }

        let bad: Vec<&Value> = rows.iter().filter(|r| r["mismatches"] != json!(0)).collect();
        let caught: Vec<String> = bad
            .iter()
            .map(|r| r["name"].as_str().unwrap().to_string())
            .collect();
        let total_rows: u64 = rows.iter().map(|r| r["rows_checked"].as_u64().unwrap()).sum();
        eprintln!(
            "cudacheck[{m}]: {} of {} components match CpuDomainEvaluator on every row \
             ({} row evaluations){}",
            rows.len() - bad.len(),
            rows.len(),
            total_rows,
            if caught.is_empty() {
                String::new()
            } else {
                format!("; differing: {}", caught.join(" "))
            },
        );
        passes.push(json!({
            "mutation": m,
            "components": rows.len(),
            "matching": rows.len() - bad.len(),
            "differing": caught,
            "row_evaluations": total_rows,
            "ir_instructions": rows.iter().map(|r| r["ir_ops"].as_u64().unwrap()).sum::<u64>(),
            "scalar_slots": rows.iter().map(|r| r["scalars"].as_u64().unwrap()).sum::<u64>(),
            "rows": rows,
        }));
    }
    crate::util::emit(
        "cudacheck",
        json!({
            "source": l.source,
            "checked_log_size": log_size,
            "passes": passes,
        }),
    );
}

// ------------------------------------------------------------------- emit ----

/// FNV-1a over a canonical encoding of the IR's *shape*: opcodes, operands,
/// widths, the constraint list and the scalar-slot count, but never a scalar
/// *value*. Two components with the same fingerprint run the same kernel; a
/// component whose fingerprint is not in the generated table aborts the prover
/// instead of being handed to a kernel that computes something else.
impl Ir {
    pub fn fingerprint(&self) -> u32 {
        let mut h: u32 = 0x811C_9DC5;
        let mut put = |s: &str, h: &mut u32| {
            for b in s.as_bytes() {
                *h ^= u32::from(*b);
                *h = h.wrapping_mul(0x0100_0193);
            }
        };
        for (i, o) in self.ops.iter().enumerate() {
            let t = match *o {
                Op::Load { tree, col, off } => format!("L{tree},{col},{off}"),
                Op::ConstM31(c) => format!("K{c}"),
                Op::Scalar(k) => format!("S{k}"),
                Op::CumsumShift => "Z".to_string(),
                Op::AddB(a, b) => format!("A{a},{b}"),
                Op::SubB(a, b) => format!("B{a},{b}"),
                Op::MulB(a, b) => format!("M{a},{b}"),
                Op::NegB(a) => format!("N{a}"),
                Op::AddE(a, b) => format!("a{a},{b}"),
                Op::SubE(a, b) => format!("b{a},{b}"),
                Op::MulE(a, b) => format!("m{a},{b}"),
                Op::NegE(a) => format!("n{a}"),
                Op::Pack(a, b, c, d) => format!("V{a},{b},{c},{d}"),
            };
            put(&format!("{i}:{t}:{};", self.width[i]), &mut h);
        }
        put(&format!("C{:?};", self.constraints), &mut h);
        // Only when there are fractions: a constraint IR's shape did not change
        // when 18.7's mode was added, so neither may its fingerprint -- the
        // 21 kernels 2b compiled are keyed on it. A fraction IR always has an
        // empty constraint list, so `C[]` already separates the two families of
        // IR and this line only ever distinguishes one fraction IR from another.
        if !self.fracs.is_empty() {
            put(&format!("F{:?};", self.fracs), &mut h);
        }
        put(&format!("N{};", self.scalars.len()), &mut h);
        put(&format!("R{:?};", self.scalar_refs), &mut h);
        h
    }

    fn ty(&self, i: usize) -> &'static str {
        if self.width[i] == 1 { "m31" } else { "qm31" }
    }

    /// The straight-line instruction list, with one hoisted row index per
    /// distinct nonzero mask offset. Shared by the constraint kernel and the
    /// fraction kernel (18.7): the two differ only in what they do with the
    /// nodes, never in how the nodes are computed.
    ///
    /// With a mask (B11 stream K), the list is restricted to the instructions
    /// a segment's constraints reach, so a body too long for ptxas can be
    /// split without renumbering anything -- a node keeps its `v{i}` name in
    /// every segment that recomputes it. `None` emits the whole list,
    /// byte-identically to what the unsplit emitter always produced.
    fn emit_instrs_for(&self, mask: Option<&[bool]>) -> String {
        let keep = |i: usize| mask.is_none_or(|m| m[i]);
        let mut out = String::new();
        // one row index per distinct nonzero mask offset, hoisted
        let mut offs: BTreeSet<isize> = BTreeSet::new();
        for (i, o) in self.ops.iter().enumerate() {
            if !keep(i) {
                continue;
            }
            if let Op::Load { off, .. } = o {
                if *off != 0 {
                    offs.insert(*off);
                }
            }
        }
        for o in &offs {
            out += &format!(
                "    const unsigned r{} = offset_bit_reversed_circle_domain_index(\
                 row, domain_log_size, eval_domain_log_size, {o});\n",
                off_name(*o)
            );
        }
        let rowvar = |off: isize| {
            if off == 0 { "row".to_string() } else { format!("r{}", off_name(off)) }
        };
        for (i, o) in self.ops.iter().enumerate() {
            if !keep(i) {
                continue;
            }
            let t = self.ty(i);
            let e = match *o {
                Op::Load { tree, col, off } => {
                    format!("t{tree}[{col}][{}]", rowvar(off))
                }
                Op::ConstM31(c) => format!("{c}u"),
                Op::Scalar(k) => {
                    if self.width[i] == 1 { format!("sc[{k}].a.a") } else { format!("sc[{k}]") }
                }
                Op::CumsumShift => "cumsum_shift".to_string(),
                Op::AddB(a, b) => format!("add(v{a}, v{b})"),
                Op::SubB(a, b) => format!("sub(v{a}, v{b})"),
                Op::MulB(a, b) => format!("mul(v{a}, v{b})"),
                Op::NegB(a) => format!("neg(v{a})"),
                // fields.cuh has mul/add/sub for (m31, qm31) but not the mirror,
                // so commutative ops are ordered and `sub`/`neg` use b65_ops.cuh.
                Op::AddE(a, b) => match (self.width[a], self.width[b]) {
                    (4, 1) => format!("add(v{b}, v{a})"),
                    _ => format!("add(v{a}, v{b})"),
                },
                Op::SubE(a, b) => match (self.width[a], self.width[b]) {
                    (4, 1) => format!("b65_sub(v{a}, v{b})"),
                    _ => format!("sub(v{a}, v{b})"),
                },
                Op::MulE(a, b) => match (self.width[a], self.width[b]) {
                    (4, 1) => format!("mul(v{b}, v{a})"),
                    _ => format!("mul(v{a}, v{b})"),
                },
                Op::NegE(a) => {
                    if self.width[a] == 1 { format!("neg(v{a})") } else { format!("b65_neg(v{a})") }
                }
                Op::Pack(a, b, c, d) => {
                    if self.width[i] == 1 {
                        format!("v{a}")
                    } else {
                        format!("qm31{{{{v{a}, v{b}}}, {{v{c}, v{d}}}}}")
                    }
                }
            };
            out += &format!("    const {t} v{i} = {e};\n");
        }
        out
    }

    /// One shape's constraint kernel body: the instructions, the accumulation
    /// against `random_coeff_powers`, and the quotient write-back.
    fn emit_body(&self) -> String {
        self.emit_body_for(None, 0..self.constraints.len())
    }

    /// One segment's constraint kernel body (B11 stream K): the instructions
    /// the segment reaches, the accumulation of its slice of the constraint
    /// list -- with the *global* `rcp` indices, so the sum over segments is the
    /// same field sum -- and the same write-back. `(None, full range)` is the
    /// whole body, byte-identical to the unsegmented emitter.
    fn emit_body_for(&self, mask: Option<&[bool]>, ks: std::ops::Range<usize>) -> String {
        let mut out = self.emit_instrs_for(mask);
        out += "    qm31 res = {{0u, 0u}, {0u, 0u}};\n";
        for k in ks {
            let id = self.constraints[k];
            out += &format!("    res = add(res, mul(v{id}, rcp[{k}]));\n");
        }
        out += "    const m31 di = denom_inv[row >> domain_log_size];\n";
        out += "    const qm31 q = mul(di, res);\n";
        out += "    if (accumulate) {\n";
        for (j, f) in ["a.a", "a.b", "b.a", "b.b"].iter().enumerate() {
            out += &format!("        q{j}[row] = add(q{j}[row], q.{f});\n");
        }
        out += "    } else {\n";
        for (j, f) in ["a.a", "a.b", "b.a", "b.b"].iter().enumerate() {
            out += &format!("        q{j}[row] = q.{f};\n");
        }
        out += "    }\n";
        out
    }
}

impl Ir {
    /// One shape's *fraction* kernel body: the same instructions, then
    /// `plogup::paired`'s pairwise batching written out per column.
    ///
    /// The pairing is the constraint framework's `finalize_logup_in_pairs`,
    /// which is what the host builder mirrors, so the three agree by
    /// construction: lookups `2c` and `2c+1` become column `c` with numerator
    /// `n0 d1 + n1 d0` and denominator `d0 d1`, and an odd tail stays alone.
    fn emit_frac_body(&self) -> String {
        self.emit_frac_body_for(None, 0..self.fracs.len().div_ceil(2))
    }

    /// One segment's fraction kernel body (B11 stream K): the per-column
    /// writes are disjoint, so a split by column range needs no accumulate
    /// path -- each segment writes its columns at their *global* indices.
    /// `(None, full range)` is byte-identical to the unsegmented emitter.
    fn emit_frac_body_for(&self, mask: Option<&[bool]>, cols: std::ops::Range<usize>) -> String {
        let mut out = self.emit_instrs_for(mask);
        let ext = |id: usize| {
            if self.width[id] == 1 { format!("b65_ext(v{id})") } else { format!("v{id}") }
        };
        let n = self.fracs.len();
        for c in cols {
            let i = 2 * c;
            let (n0, d0) = self.fracs[i];
            out += "    {\n";
            out += &format!("        const qm31 n0 = {}, d0 = {};\n", ext(n0), ext(d0));
            if i + 1 < n {
                let (n1, d1) = self.fracs[i + 1];
                out += &format!("        const qm31 n1 = {}, d1 = {};\n", ext(n1), ext(d1));
                out += &format!(
                    "        num[{c}u * n_rows + row] = add(mul(n0, d1), mul(n1, d0));\n"
                );
                out += &format!("        den[{c}u * n_rows + row] = mul(d0, d1);\n");
            } else {
                out += &format!("        num[{c}u * n_rows + row] = n0;\n");
                out += &format!("        den[{c}u * n_rows + row] = d0;\n");
            }
            out += "    }\n";
        }
        out
    }
}

fn off_name(off: isize) -> String {
    if off < 0 { format!("m{}", -off) } else { format!("p{off}") }
}

// -------------------------------------------------- the split (B11 stream K) ----
//
// The glue's constraint kernels (~10,400 instructions) and the head's dequant
// (54,847) are past what ptxas will compile in reasonable time. The body is a
// straight-line SSA list followed by a plain weighted sum of the constraints
// against `random_coeff_powers`, and the quotient write-back already has an
// accumulate path -- so the constraint list can be partitioned into segments,
// each segment's kernel carrying only the transitive closure of its own
// constraints, launched in sequence with every segment after the first on the
// add path. M31 addition is exact, so the result is bit-identical by
// construction. The cost is honest: a subexpression shared across segments is
// recomputed in each one.

/// One segment of a split kernel: which units it covers -- constraints for
/// `emit_body_for`, fraction columns for `emit_frac_body_for` -- the emit mask
/// of the instructions they reach, and that mask's population count.
pub struct Seg {
    pub units: std::ops::Range<usize>,
    pub mask: Vec<bool>,
    pub n_instrs: usize,
}

impl Ir {
    /// The transitive operand closure of a set of root nodes, as an emit mask.
    /// Follows exactly what `emit_instrs` reads: in particular a width-1
    /// `Pack` is emitted as its first operand alone, so only that operand is
    /// needed.
    fn emit_mask(&self, roots: &[usize]) -> Vec<bool> {
        let mut mask = vec![false; self.ops.len()];
        let mut stack: Vec<usize> = roots.to_vec();
        while let Some(i) = stack.pop() {
            if mask[i] {
                continue;
            }
            mask[i] = true;
            match self.ops[i] {
                Op::Pack(a, ..) if self.width[i] == 1 => stack.push(a),
                ref o => stack.extend(operands(o)),
            }
        }
        mask
    }

    /// Greedy contiguous partition of `units` (one root list each) into
    /// segments whose emitted closure stays at or under `cap` instructions. A
    /// single unit whose own closure already exceeds the cap gets a segment of
    /// its own -- the cap is a knob, not a promise.
    fn segment_units(&self, units: &[Vec<usize>], cap: usize) -> Vec<Seg> {
        let closures: Vec<Vec<usize>> = units
            .iter()
            .map(|roots| {
                let m = self.emit_mask(roots);
                (0..self.ops.len()).filter(|&i| m[i]).collect()
            })
            .collect();
        let mut segs: Vec<Seg> = vec![];
        let mut start = 0usize;
        let mut mask = vec![false; self.ops.len()];
        let mut count = 0usize;
        for (u, cl) in closures.iter().enumerate() {
            let fresh = cl.iter().filter(|&&i| !mask[i]).count();
            if u > start && count + fresh > cap {
                segs.push(Seg {
                    units: start..u,
                    mask: std::mem::replace(&mut mask, vec![false; self.ops.len()]),
                    n_instrs: count,
                });
                start = u;
                count = 0;
            }
            for &i in cl {
                if !mask[i] {
                    mask[i] = true;
                    count += 1;
                }
            }
        }
        if start < units.len() {
            segs.push(Seg { units: start..units.len(), mask, n_instrs: count });
        }
        segs
    }

    /// The constraint segments of one shape at `cap`, or `None` when the shape
    /// stays whole -- under the cap (so the installed library's kernels do not
    /// churn byte for byte), or unsplittable (one constraint, or one segment).
    pub fn constraint_segments(&self, cap: usize) -> Option<Vec<Seg>> {
        if self.ops.len() <= cap || self.constraints.len() <= 1 {
            return None;
        }
        let units: Vec<Vec<usize>> = self.constraints.iter().map(|&id| vec![id]).collect();
        let segs = self.segment_units(&units, cap);
        if segs.len() <= 1 { None } else { Some(segs) }
    }

    /// The fraction-column segments of one shape at `cap`, or `None` when the
    /// shape stays whole. Column `c` is lookups `2c` and `2c+1`, exactly as
    /// `emit_frac_body` pairs them.
    pub fn frac_segments(&self, cap: usize) -> Option<Vec<Seg>> {
        let n_cols = self.fracs.len().div_ceil(2);
        if self.ops.len() <= cap || n_cols <= 1 {
            return None;
        }
        let units: Vec<Vec<usize>> = (0..n_cols)
            .map(|c| {
                let mut roots = vec![self.fracs[2 * c].0, self.fracs[2 * c].1];
                if 2 * c + 1 < self.fracs.len() {
                    roots.push(self.fracs[2 * c + 1].0);
                    roots.push(self.fracs[2 * c + 1].1);
                }
                roots
            })
            .collect();
        let segs = self.segment_units(&units, cap);
        if segs.len() <= 1 { None } else { Some(segs) }
    }
}

/// One family's generated translation unit: a kernel per shape plus the host
/// launcher the fork's dispatcher calls.
pub struct Emitted {
    pub family: String,
    pub cu: String,
    pub cuh: String,
    /// per shape, in shape-id order
    pub shapes: Vec<(u32, usize)>,
    /// per shape: instruction count per emitted kernel -- one entry when the
    /// shape stayed whole, one per segment when it was split (B11 stream K)
    pub seg_instrs: Vec<Vec<usize>>,
}

/// How the generated kernels are launched. `min_blocks` is the part that
/// matters: the fork's own kernels all use `__launch_bounds__(256, 2)`, which
/// caps ptxas at 65536/512 = 128 registers per thread, and our straight-line
/// bodies are far too long for that -- section 16.3 measures the spill traffic
/// it causes. Independent of the IR, so changing it does not change a
/// fingerprint and the Rust side does not have to be regenerated.
#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub block: u32,
    pub min_blocks: u32,
}

impl Launch {
    fn attr(&self) -> String {
        if self.min_blocks == 0 {
            format!("__launch_bounds__({})", self.block)
        } else {
            format!("__launch_bounds__({}, {})", self.block, self.min_blocks)
        }
    }
}

/// `shapes` must be in shape-id order; each entry is one representative IR.
/// `seg_cap` is the instruction count above which a shape's kernel is split
/// into segments (B11 stream K); a shape at or under it emits byte-identically
/// to what this function always produced.
pub fn emit_family(family: &str, shapes: &[(String, Ir)], launch: Launch, seg_cap: usize) -> Emitted {
    let fname = format!("evaluate_{family}");
    let segs: Vec<Option<Vec<Seg>>> =
        shapes.iter().map(|(_, ir)| ir.constraint_segments(seg_cap)).collect();
    let mut cu = format!(
        "// Generated by experiments/b7/gpu/m2b/cudagen.rs (B7a milestone 2b step 3).\n\
         // Do not edit: regenerate with `b65 cudaemit`.\n\
         //\n\
         // One kernel per distinct constraint shape of the `{family}` family. The\n\
         // shape is chosen by the host launcher from the eval blob, so the fork's\n\
         // dispatch stays keyed on `cuda_eval_name()` alone.\n\
         #include \"fields.cuh\"\n\
         #include \"utils.cuh\"\n\
         #include \"b65_ops.cuh\"\n\
         #include \"{fname}.cuh\"\n\n"
    );
    for (id, (name, ir)) in shapes.iter().enumerate() {
        cu += &format!(
            "// shape {id}: {name} -- {} instructions, {} constraints, {} scalars, \
             fingerprint {:#010x}\n",
            ir.ops.len(),
            ir.constraints.len(),
            ir.scalars.len(),
            ir.fingerprint()
        );
        // one kernel per (shape, segment); an unsegmented shape keeps its
        // historical name so the installed library does not churn
        let kernel = |kname: String, body: String| {
            format!(
                "{}\n\
                 __global__ void {kname}(\n\
                 \x20   m31 *q0, m31 *q1, m31 *q2, m31 *q3,\n\
                 \x20   m31 **t0, m31 **t1, m31 **t2,\n\
                 \x20   const qm31 *__restrict__ sc,\n\
                 \x20   const qm31 *__restrict__ rcp,\n\
                 \x20   const m31 *__restrict__ denom_inv,\n\
                 \x20   unsigned domain_log_size,\n\
                 \x20   unsigned eval_domain_log_size,\n\
                 \x20   qm31 cumsum_shift,\n\
                 \x20   bool accumulate\n\
                 ) {{\n\
                 \x20   const unsigned n = 1u << eval_domain_log_size;\n\
                 \x20   const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;\n\
                 \x20   if (row >= n) return;\n{body}}}\n\n",
                launch.attr()
            )
        };
        match &segs[id] {
            None => cu += &kernel(format!("{fname}_s{id}_kernel"), ir.emit_body()),
            Some(ss) => {
                for (p, s) in ss.iter().enumerate() {
                    cu += &format!(
                        "// shape {id} segment {p}: constraints {}..{}, {} instructions\n",
                        s.units.start, s.units.end, s.n_instrs
                    );
                    cu += &kernel(
                        format!("{fname}_s{id}p{p}_kernel"),
                        ir.emit_body_for(Some(&s.mask), s.units.clone()),
                    );
                }
            }
        }
    }

    // the launcher
    let n_shapes = shapes.len();
    let fps: Vec<String> = shapes.iter().map(|(_, i)| format!("{:#010x}u", i.fingerprint())).collect();
    let ns: Vec<String> = shapes.iter().map(|(_, i)| format!("{}u", i.scalars.len())).collect();
    cu += &format!(
        "static const unsigned {fname}_n_shapes = {n_shapes}u;\n\
         static const unsigned {fname}_fp[{n_shapes}] = {{{}}};\n\
         static const unsigned {fname}_nsc[{n_shapes}] = {{{}}};\n\n",
        fps.join(", "),
        ns.join(", ")
    );
    cu += &format!(
        "void {fname}(\n\
         \x20   m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,\n\
         \x20   m31 **trace0_evaluations, unsigned trace0_evaluations_len,\n\
         \x20   m31 **trace1_evaluations, unsigned trace1_evaluations_len,\n\
         \x20   m31 **trace2_evaluations, unsigned trace2_evaluations_len,\n\
         \x20   qm31 *random_coeff_powers,\n\
         \x20   m31 *denominator_inverses,\n\
         \x20   unsigned int domain_log_size,\n\
         \x20   unsigned int eval_domain_log_size,\n\
         \x20   unsigned int number_of_columns,\n\
         \x20   unsigned int logup_counts,\n\
         \x20   void *eval,\n\
         \x20   qm31 cumsum_shift,\n\
         \x20   bool should_accumulate,\n\
         \x20   bool use_assert_evaluator,\n\
         \x20   cudaStream_t stream\n\
         ) {{\n\
         \x20   cudaGetLastError();\n\
         \x20   const B65Blob *blob = b65_blob(eval, \"{fname}\");\n\
         \x20   const unsigned shape = blob->shape;\n\
         \x20   b65_check(shape < {fname}_n_shapes, \"{fname}: shape id out of range\");\n\
         \x20   b65_check(blob->n_scalars == {fname}_nsc[shape], \"{fname}: scalar count\");\n\
         \x20   b65_check(blob->fingerprint == {fname}_fp[shape], \"{fname}: fingerprint\");\n\
         \x20   const unsigned n = 1u << eval_domain_log_size;\n\
         \x20   m31 **d_t0 = trace0_evaluations_len ? clone_to_device<m31*>(trace0_evaluations, trace0_evaluations_len) : nullptr;\n\
         \x20   m31 **d_t1 = trace1_evaluations_len ? clone_to_device<m31*>(trace1_evaluations, trace1_evaluations_len) : nullptr;\n\
         \x20   m31 **d_t2 = trace2_evaluations_len ? clone_to_device<m31*>(trace2_evaluations, trace2_evaluations_len) : nullptr;\n\
         \x20   qm31 *d_sc = nullptr;\n\
         \x20   if (blob->n_scalars) {{\n\
         \x20       d_sc = cuda_malloc<qm31>(blob->n_scalars);\n\
         \x20       ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(d_sc, b65_scalars(blob),\n\
         \x20           (size_t)blob->n_scalars * sizeof(qm31), cudaMemcpyHostToDevice, stream));\n\
         \x20   }}\n\
         \x20   const unsigned block = n < {block}u ? n : {block}u;\n\
         \x20   const unsigned blocks = (n + block - 1u) / block;\n\
         \x20   switch (shape) {{\n",
        block = launch.block
    );
    for id in 0..n_shapes {
        match &segs[id] {
            None => {
                cu += &format!(
                    "\x20   case {id}u: {fname}_s{id}_kernel<<<blocks, block, 0, stream>>>(\n\
                     \x20       quotients_0, quotients_1, quotients_2, quotients_3, d_t0, d_t1, d_t2,\n\
                     \x20       d_sc, random_coeff_powers, denominator_inverses,\n\
                     \x20       domain_log_size, eval_domain_log_size, cumsum_shift, should_accumulate);\n\
                     \x20       break;\n"
                );
            }
            // the segments run in sequence on the same stream; segment 0 takes
            // the caller's accumulate flag, every later one adds onto it
            Some(ss) => {
                cu += &format!("\x20   case {id}u:\n");
                for p in 0..ss.len() {
                    let acc = if p == 0 { "should_accumulate" } else { "true" };
                    cu += &format!(
                        "\x20       {fname}_s{id}p{p}_kernel<<<blocks, block, 0, stream>>>(\n\
                         \x20           quotients_0, quotients_1, quotients_2, quotients_3, d_t0, d_t1, d_t2,\n\
                         \x20           d_sc, random_coeff_powers, denominator_inverses,\n\
                         \x20           domain_log_size, eval_domain_log_size, cumsum_shift, {acc});\n"
                    );
                }
                cu += "\x20       break;\n";
            }
        }
    }
    cu += &format!(
        "\x20   }}\n\
         \x20   if (b65_launch_sync()) ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));\n\
         \x20   ASSERT_CUDA_SUCCESS(cudaGetLastError());\n\
         \x20   if (d_t0) cuda_free_memory(d_t0);\n\
         \x20   if (d_t1) cuda_free_memory(d_t1);\n\
         \x20   if (d_t2) cuda_free_memory(d_t2);\n\
         \x20   if (d_sc) cuda_free_memory(d_sc);\n\
         }}\n"
    );

    let guard = fname.to_uppercase();
    let cuh = format!(
        "// Generated by experiments/b7/gpu/m2b/cudagen.rs. Do not edit.\n\
         #ifndef {guard}_CUH\n#define {guard}_CUH\n#include \"fields.cuh\"\n\
         void {fname}(\n\
         \x20   m31 *quotients_0, m31 *quotients_1, m31 *quotients_2, m31 *quotients_3,\n\
         \x20   m31 **trace0_evaluations, unsigned trace0_evaluations_len,\n\
         \x20   m31 **trace1_evaluations, unsigned trace1_evaluations_len,\n\
         \x20   m31 **trace2_evaluations, unsigned trace2_evaluations_len,\n\
         \x20   qm31 *random_coeff_powers,\n\
         \x20   m31 *denominator_inverses,\n\
         \x20   unsigned int domain_log_size,\n\
         \x20   unsigned int eval_domain_log_size,\n\
         \x20   unsigned int number_of_columns,\n\
         \x20   unsigned int logup_counts,\n\
         \x20   void *eval,\n\
         \x20   qm31 cumsum_shift,\n\
         \x20   bool should_accumulate,\n\
         \x20   bool use_assert_evaluator,\n\
         \x20   cudaStream_t stream);\n#endif\n"
    );

    Emitted {
        family: family.to_string(),
        cu,
        cuh,
        shapes: shapes.iter().map(|(_, i)| (i.fingerprint(), i.scalars.len())).collect(),
        seg_instrs: shapes
            .iter()
            .zip(&segs)
            .map(|((_, ir), s)| match s {
                None => vec![ir.ops.len()],
                Some(ss) => ss.iter().map(|s| s.n_instrs).collect(),
            })
            .collect(),
    }
}

/// The one hand-written device header: the three operations `fields.cuh` does
/// not provide in the direction the emitter needs, and the blob accessors.
pub const OPS_CUH: &str = r#"// Written by hand (small), included by every generated b65 kernel.
#ifndef B65_OPS_CUH
#define B65_OPS_CUH
#include <cstdio>
#include <cstdlib>
#include "fields.cuh"
#include "utils.cuh"

// fields.cuh offers mul/add/sub for (m31, qm31) but not (qm31, m31), and no
// neg(qm31). These are the mirrors the generator needs.
HOST_DEVICE_FORCEINLINE qm31 b65_neg(qm31 x) {
    return qm31{{neg(x.a.a), neg(x.a.b)}, {neg(x.b.a), neg(x.b.b)}};
}
HOST_DEVICE_FORCEINLINE qm31 b65_sub(qm31 x, m31 y) {
    return qm31{{sub(x.a.a, y), x.a.b}, {x.b.a, x.b.b}};
}
// A width-1 node lifted into the extension, for the fraction kernels: a
// numerator is very often a bare multiplicity column.
HOST_DEVICE_FORCEINLINE qm31 b65_ext(m31 x) {
    return qm31{{x, 0u}, {0u, 0u}};
}

// The parameter block a b65 eval hands the kernel in place of its raw bytes
// (see crates/constraint-framework: FrameworkEval::cuda_eval_blob). Laid out by
// cudagen.rs; `magic` and `fingerprint` make a stale binary a loud failure.
#define B65_BLOB_MAGIC 0xB65C0DE1u
struct B65Blob {
    unsigned magic;
    unsigned shape;
    unsigned n_scalars;
    unsigned fingerprint;
};

inline void b65_check(bool ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "b65 kernel dispatch: %s\n", what);
        fflush(stderr);
        abort();
    }
}

// The fork prepends a u32 eval_id to the blob (component_prover.rs).
inline const B65Blob *b65_blob(void *eval, const char *who) {
    const B65Blob *b = (const B65Blob *)((const unsigned char *)eval + 4);
    if (b->magic != B65_BLOB_MAGIC) {
        fprintf(stderr, "b65 kernel dispatch: %s got a blob with magic %#x\n", who, b->magic);
        fflush(stderr);
        abort();
    }
    return b;
}

inline const void *b65_scalars(const B65Blob *b) {
    return (const void *)((const unsigned char *)b + sizeof(B65Blob));
}

// B7a milestone 3: whether the launcher waits for its own kernel before
// returning. Everything it touches is stream-ordered on stream 0 --
// cudaMallocFromPoolAsync and cudaFreeAsync (cuda_mem_pool.cuh:41,80), the
// pointer-array and scalar copies, and the kernel itself -- so waiting only
// stops the host from queueing the next component while this one runs. 16.5
// measured that as 85 ms of device idle across 32 components. Default off;
// B7_M3_LAUNCHSYNC=1 restores it.
inline bool b65_launch_sync() {
    static const bool on = [] {
        const char *v = getenv("B7_M3_LAUNCHSYNC");
        return v != nullptr && v[0] != '0';
    }();
    return on;
}

#endif // B65_OPS_CUH
"#;

// ------------------------------------------------------------------- blob ----

pub const BLOB_MAGIC: u32 = 0xB65C_0DE1;

impl Ir {
    /// The parameter block the kernel receives: a 16-byte header, then the
    /// scalars. `magic` and `fingerprint` make a stale `libstwo_cuda.a` -- one
    /// built from a different generator run -- an abort at the first component
    /// instead of a wrong proof.
    pub fn blob(&self, shape: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(16 + self.scalars.len() * 16);
        v.extend_from_slice(&BLOB_MAGIC.to_ne_bytes());
        v.extend_from_slice(&shape.to_ne_bytes());
        v.extend_from_slice(&(self.scalars.len() as u32).to_ne_bytes());
        v.extend_from_slice(&self.fingerprint().to_ne_bytes());
        for s in &self.scalars {
            for l in s.to_m31_array() {
                v.extend_from_slice(&l.0.to_ne_bytes());
            }
        }
        v
    }
}

/// `FrameworkEval::cuda_eval_blob`'s body, shared by all eleven evals: lower the
/// eval as the generator did, look the shape up by fingerprint, and hand over
/// the scalars. Aborts if the fingerprint is not in the generated table -- an
/// AIR that changed without the kernels being regenerated must not prove.
// ----------------------------------------------------- B7a milestone 3 ----
// What the prove-time lowering costs, and a cache so it can be paid once, in
// parallel, before the prove loop instead of inside it (gpu/m3_patch.py).

static BLOB_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BLOB_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BLOB_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[allow(clippy::type_complexity)]
static BLOB_CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, Option<Vec<u8>>>>> =
    std::sync::OnceLock::new();

fn blob_cache() -> &'static std::sync::Mutex<HashMap<usize, Option<Vec<u8>>>> {
    BLOB_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Total wall spent lowering, how many components, and how many of the fork's
/// calls the prewarm had already answered.
pub fn blob_report() -> serde_json::Value {
    use std::sync::atomic::Ordering::Relaxed;
    serde_json::json!({
        "lower_secs": BLOB_NS.load(Relaxed) as f64 / 1e9,
        "lowerings": BLOB_CALLS.load(Relaxed),
        "cache_hits": BLOB_HITS.load(Relaxed),
    })
}

/// Lower one component's constraints ahead of the prove loop. Keyed by the
/// eval's address, which is stable for as long as its `FrameworkComponent`
/// lives -- i.e. for the whole proof.
pub fn prewarm<E: FrameworkEval>(eval: &E) {
    let key = eval as *const E as usize;
    let v = blob_timed(eval);
    blob_cache().lock().unwrap().insert(key, v);
}

fn blob_timed<E: FrameworkEval>(eval: &E) -> Option<Vec<u8>> {
    use std::sync::atomic::Ordering::Relaxed;
    let t = std::time::Instant::now();
    let v = blob_lower(eval);
    BLOB_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    BLOB_CALLS.fetch_add(1, Relaxed);
    v
}

pub fn blob_for<E: FrameworkEval>(eval: &E) -> Option<Vec<u8>> {
    let key = eval as *const E as usize;
    if let Some(v) = blob_cache().lock().unwrap().get(&key) {
        BLOB_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return v.clone();
    }
    blob_timed(eval)
}

fn blob_lower<E: FrameworkEval>(eval: &E) -> Option<Vec<u8>> {
    let name = eval.cuda_eval_name();
    if crate::cudakernels::SHAPES.is_empty() {
        // No kernels generated yet: leave the fork on its own path, which is the
        // scalar CPU fallback milestone 2a measured.
        return None;
    }
    // B7b (README 26): by shape, with this proof's scalars replayed.
    let ir = crate::scalars::cached_ir(
        eval,
        eval.log_size(),
        eval.max_constraint_log_degree_bound(),
        false,
        || lower(eval, SecureField::zero(), eval.log_size(), eval.max_constraint_log_degree_bound()),
    );
    let fp = ir.fingerprint();
    let shape = match crate::cudakernels::shape_of(name, fp) {
        Some(s) => s,
        // B10 step 5 (stream B): the glue and the corner-witness shapes lower
        // to kernels ptxas cannot compile in reasonable time (52 min and
        // counting on one), so `B7_KERNEL_FALLBACK=1` lets a component with no
        // generated kernel run on the fork's scalar CPU fallback -- the same
        // bytes, announced per component, never silent.
        None if crate::util::m4_switch("B7_KERNEL_FALLBACK") => {
            eprintln!("kernel fallback: {name} fingerprint {fp:#010x} \
                       ({} instructions) evaluates on the CPU", ir.ops.len());
            return None;
        }
        None => panic!(
            "no generated constraint kernel for {name} fingerprint {fp:#010x} \
             ({} instructions, {} constraints, {} scalars): regenerate with \
             `b65 cudaemit` and rebuild libstwo_cuda.a",
            ir.ops.len(),
            ir.constraints.len(),
            ir.scalars.len()
        ),
    };
    Some(ir.blob(shape))
}

// -------------------------------------------------------------- cudaemit ----

/// `cudaemit`: lower every component, group by fingerprint into kernels, and
/// write the generated CUDA, the Rust shape table and a manifest.
pub fn emit_cmd(args: &[String]) {
    let (l, vname) = crate::layer_of(args);
    let params = crate::params_of(&l, args, &vname);
    let plan = crate::compose::Plan::new(&params);
    let out = crate::arg_str(args, "--out").unwrap_or_else(|| "/root/b7/gen".to_string());
    let launch = Launch {
        block: crate::arg_scalar(args, "--block", 256u32),
        min_blocks: crate::arg_scalar(args, "--min-blocks", 2u32),
    };
    // B11 stream K: instruction count above which a shape's kernel is split.
    let seg_cap: usize = crate::arg_scalar(args, "--seg-cap", 3000usize);
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).unwrap();

    let mut ch = stwo::core::channel::Blake2sM31Channel::default();
    stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_2B03);
    let rels = crate::compose::Rels::draw(&mut ch, &plan.widths,
                                          plan.glue.as_ref().map(|g| g.uns.as_slice()));
    let f = stwo::core::channel::Channel::draw_secure_felts(&mut ch, 6);
    let claims = crate::exprprobe::claims_for(&plan, 0x0b65_2b03);
    let b = crate::compose::build_components(&plan, &rels, f[0], f[1], f[2], f[3], f[4], f[5], &claims);

    // family -> shapes, in first-seen order; a shape is a fingerprint. Seeded
    // from every shape a previous emit run saw, so one library serves every
    // design point (see `Ir`'s note).
    let store = dir.join("shapes.json");
    let mut fams: Vec<(String, Vec<(String, Ir)>)> = if store.exists() {
        let prev: Vec<(String, String, Ir)> =
            serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
        let mut v: Vec<(String, Vec<(String, Ir)>)> = vec![];
        for (family, name, ir) in prev {
            match v.iter_mut().find(|(f, _)| *f == family) {
                Some((_, s)) => s.push((name, ir)),
                None => v.push((family, vec![(name, ir)])),
            }
        }
        eprintln!(
            "cudaemit: carrying {} shapes forward from {}",
            v.iter().map(|(_, s)| s.len()).sum::<usize>(),
            store.display()
        );
        v
    } else {
        vec![]
    };
    let carried: usize = fams.iter().map(|(_, s)| s.len()).sum();
    let mut rows = vec![];
    {
        let mut add = |family: &str, name: String, ir: Ir| {
            let fp = ir.fingerprint();
            let fi = match fams.iter().position(|(f, _)| f == family) {
                Some(i) => i,
                None => {
                    fams.push((family.to_string(), vec![]));
                    fams.len() - 1
                }
            };
            let shapes = &mut fams[fi].1;
            let sid = match shapes.iter().position(|(_, i)| i.fingerprint() == fp) {
                Some(i) => i,
                None => {
                    shapes.push((name.clone(), ir.clone()));
                    shapes.len() - 1
                }
            };
            rows.push(json!({
                "component": name, "family": family, "shape": sid,
                "fingerprint": format!("{fp:#010x}"),
                "instructions": ir.ops.len(), "constraints": ir.constraints.len(),
                "scalars": ir.scalars.len(), "m31_muls_per_row": ir.m31_muls(),
                "loads_per_row": ir.loads(),
                "log_size": ir.log_size, "eval_log_size": ir.eval_log_size,
            }));
        };
        macro_rules! go {
            ($c:expr, $name:expr) => {{
                let ir = lower(
                    &$c.eval,
                    SecureField::zero(),
                    $c.eval.log_size(),
                    $c.eval.max_constraint_log_degree_bound(),
                );
                add($c.eval.cuda_eval_name(), $name, ir);
            }};
        }
        for i in 0..b.chans.len() {
            go!(b.chans[i], format!("chan_band{i}"));
            go!(b.keys[i], format!("key_band{i}"));
        }
        // a corner band has no row component (B10 step 3): `rows` is shorter
        // than `chans`, so it is walked on its own
        for i in 0..b.rows.len() {
            go!(b.rows[i], format!("row_band{i}"));
        }
        go!(b.ao, "attnout".to_string());
        for i in 0..b.mats.len() {
            go!(b.mats[i], format!("dequant{i}"));
        }
        for i in 0..b.glue.len() {
            go!(b.glue[i], format!("glue{i}"));
        }
        for i in 0..b.gtables.len() {
            go!(b.gtables[i], format!("gtable{i}"));
        }
        for i in 0..b.walks.len() {
            go!(b.walks[i], format!("walk{i}"));
        }
        go!(b.exp_t, "exp_table".to_string());
        for i in 0..b.tables.len() {
            go!(b.tables[i], format!("table{i}"));
        }
    }

    let fresh: usize = fams.iter().map(|(_, s)| s.len()).sum::<usize>() - carried;
    let flat: Vec<(String, String, &Ir)> = fams
        .iter()
        .flat_map(|(f, ss)| ss.iter().map(move |(n, i)| (f.clone(), n.clone(), i)))
        .collect();
    std::fs::write(&store, serde_json::to_vec(&flat).unwrap()).unwrap();
    std::fs::write(dir.join("b65_ops.cuh"), OPS_CUH).unwrap();
    let mut manifest = vec![];
    let mut table = String::from(
        "// Generated by `b65 cudaemit` (B7a milestone 2b step 3). Do not edit.\n\
         //\n\
         // Maps a lowered component's IR fingerprint to the generated kernel that\n\
         // implements it. `cuda_eval_blob` refuses to prove a component whose\n\
         // fingerprint is absent, so an AIR change without a kernel regeneration is\n\
         // an abort rather than a wrong proof.\n\n\
         pub struct Shape {\n\
         \x20   pub family: &'static str,\n\
         \x20   pub id: u32,\n\
         \x20   pub fingerprint: u32,\n\
         \x20   pub n_scalars: usize,\n\
         \x20   pub example: &'static str,\n\
         }\n\n\
         pub static SHAPES: &[Shape] = &[\n",
    );
    let mut total_instr = 0usize;
    for (family, shapes) in &fams {
        let e = emit_family(family, shapes, launch, seg_cap);
        std::fs::write(dir.join(format!("evaluate_{family}.cu")), &e.cu).unwrap();
        std::fs::write(dir.join(format!("evaluate_{family}.cuh")), &e.cuh).unwrap();
        for (id, si) in e.seg_instrs.iter().enumerate() {
            if si.len() > 1 {
                eprintln!(
                    "  {family} shape {id}: split into {} segments at cap {seg_cap}: {si:?}",
                    si.len()
                );
            }
        }
        for (id, (name, ir)) in shapes.iter().enumerate() {
            total_instr += ir.ops.len();
            table += &format!(
                "    Shape {{ family: {family:?}, id: {id}, fingerprint: {:#010x}, \
                 n_scalars: {}, example: {name:?} }},\n",
                ir.fingerprint(),
                ir.scalars.len()
            );
        }
        manifest.push(json!({
            "family": family,
            "cu": format!("evaluate_{family}.cu"),
            "shapes": shapes.iter().enumerate().map(|(id, (n, ir))| json!({
                "id": id, "example": n, "fingerprint": format!("{:#010x}", ir.fingerprint()),
                "instructions": ir.ops.len(), "scalars": ir.scalars.len(),
                "segments": e.seg_instrs[id],
            })).collect::<Vec<_>>(),
            "cu_lines": e.cu.lines().count(),
        }));
    }
    table += "];\n\n\
              pub fn shape_of(family: &str, fingerprint: u32) -> Option<u32> {\n\
              \x20   SHAPES\n\
              \x20       .iter()\n\
              \x20       .find(|s| s.family == family && s.fingerprint == fingerprint)\n\
              \x20       .map(|s| s.id)\n\
              }\n";
    std::fs::write(dir.join("cudakernels.rs"), &table).unwrap();

    let man = json!({
        "source": l.source,
        "launch": {"block": launch.block, "min_blocks": launch.min_blocks},
        "seg_cap": seg_cap,
        "families": manifest,
        "components": rows,
        "kernels": fams.iter().map(|(_, s)| s.len()).sum::<usize>(),
        "kernels_new_this_run": fresh,
        "instructions": total_instr,
    });
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&man).unwrap()).unwrap();

    eprintln!(
        "cudaemit: {} families, {} kernels ({fresh} new this run), {} instructions -> {}",
        fams.len(),
        fams.iter().map(|(_, s)| s.len()).sum::<usize>(),
        total_instr,
        dir.display()
    );
    for (family, shapes) in &fams {
        eprintln!(
            "  {family:<22} {} shape(s): {}",
            shapes.len(),
            shapes
                .iter()
                .map(|(n, i)| format!("{n}({} instr)", i.ops.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    crate::util::emit("cudaemit", man);
}

// ------------------------------------------------- the fractions, on the host ----
//
// B7a milestone 4 (18.7). The interaction phase is not a second generator: it
// is the same IR with a different output list, run on the *trace* domain
// instead of the evaluation domain. This is that run, packed the way the
// crate's own builders are, so it can replace them column for column.

use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;

/// A node's packed value. Same split as `Val`, same reason.
#[derive(Clone, Copy)]
pub enum PVal {
    B(PackedM31),
    E(PackedQM31),
}

impl PVal {
    #[inline]
    fn b(self) -> PackedM31 {
        match self {
            PVal::B(x) => x,
            PVal::E(_) => panic!("extension value where a base value was required"),
        }
    }
    #[inline]
    fn e(self) -> PackedQM31 {
        match self {
            PVal::E(x) => x,
            PVal::B(x) => PackedQM31::from(x),
        }
    }
}

/// The relation elements, broadcast once per IR rather than per row.
pub struct FracScalars(Vec<PackedQM31>);

impl FracScalars {
    pub fn new(ir: &Ir) -> Self {
        Self(ir.scalars.iter().map(|s| PackedQM31::broadcast(*s)).collect())
    }
}

/// Every lookup's `(numerator, denominator)` at one packed row of the trace.
///
/// `pre` and `base` are the component's own columns, in committed order --
/// exactly `r.pre` and `r.base`. A fraction reads nothing else: no interaction
/// column, no mask offset, no `cumsum_shift`. All three are asserted, because a
/// family that broke the first would need a shifted read this does not do, and
/// silently reading row `vr` instead would be wrong.
pub fn frac_at(
    ir: &Ir,
    sc: &FracScalars,
    pre: &[&[PackedM31]],
    base: &[&[PackedM31]],
    vr: usize,
    v: &mut Vec<PVal>,
    out: &mut Vec<(PackedQM31, PackedQM31)>,
) {
    v.clear();
    for op in &ir.ops {
        let x = match *op {
            Op::Load { tree, col, off } => {
                assert_eq!(off, 0, "a LogUp value at a nonzero mask offset");
                PVal::B(match tree {
                    0 => pre[col][vr],
                    1 => base[col][vr],
                    _ => panic!("a LogUp value from interaction {tree}"),
                })
            }
            Op::ConstM31(c) => PVal::B(PackedM31::broadcast(M31::from_u32_unchecked(c))),
            Op::Scalar(k) => PVal::E(sc.0[k]),
            Op::CumsumShift => panic!("cumsum_shift inside a fraction"),
            Op::AddB(a, b) => PVal::B(v[a].b() + v[b].b()),
            Op::SubB(a, b) => PVal::B(v[a].b() - v[b].b()),
            Op::MulB(a, b) => PVal::B(v[a].b() * v[b].b()),
            Op::NegB(a) => PVal::B(-v[a].b()),
            Op::AddE(a, b) => PVal::E(v[a].e() + v[b].e()),
            Op::SubE(a, b) => PVal::E(v[a].e() - v[b].e()),
            Op::MulE(a, b) => PVal::E(v[a].e() * v[b].e()),
            Op::NegE(a) => PVal::E(-v[a].e()),
            Op::Pack(a, b, c, d) => PVal::E(PackedQM31::from_packed_m31s([
                v[a].b(),
                v[b].b(),
                v[c].b(),
                v[d].b(),
            ])),
        };
        v.push(x);
    }
    out.clear();
    for (n, d) in &ir.fracs {
        out.push((v[*n].e(), v[*d].e()));
    }
}

/// One component's LogUp columns, built from its fraction IR.
///
/// This is the generic form of `mat_logup`, `walk_interaction`'s LogUp half and
/// every other hand-written fraction builder in the crate: the lookups, their
/// order, their columns and their relation elements all come from the AIR
/// itself, so there is nothing per-family left to write. The output is fed to
/// the same `plogup` phases the hand-written builders use, so equality with
/// them is bit-for-bit or it is a bug.
pub fn logup_cols(
    ir: &Ir,
    log_rows: u32,
    pre: &[&[PackedM31]],
    base: &[&[PackedM31]],
) -> (Vec<crate::common::Eval>, SecureField) {
    let sc = FracScalars::new(ir);
    let n_looks = ir.fracs.len();
    assert!(n_looks > 0, "a component with no lookups has no LogUp columns");
    crate::plogup::build_rows(log_rows, n_looks, |vr, dst| {
        let mut v: Vec<PVal> = Vec::with_capacity(ir.ops.len());
        frac_at(ir, &sc, pre, base, vr, &mut v, dst);
    })
}

// ------------------------------------------------------------- logupcheck ----

/// What one component's fractions would cost, and whether they are reachable
/// this way at all.
fn logup_report<E: FrameworkEval>(name: String, c: &FrameworkComponent<E>) -> Value {
    let log_size = c.eval.log_size();
    let ir = lower_logup(&c.eval, log_size);
    // The three properties a device-side fraction pass depends on. Reported,
    // not asserted, because a family that breaks one is a finding rather than
    // a crash -- 18.7's list of what moves and what does not.
    let mut offsets: BTreeSet<isize> = BTreeSet::new();
    let mut trees: BTreeSet<usize> = BTreeSet::new();
    for o in &ir.ops {
        if let Op::Load { tree, off, .. } = o {
            offsets.insert(*off);
            trees.insert(*tree);
        }
    }
    let const_ir = lower(&c.eval, c.claimed_sum(), log_size, log_size + 1);
    json!({
        "name": name,
        "cuda_eval_name": c.eval.cuda_eval_name(),
        "log_size": log_size,
        "lookups": ir.fracs.len(),
        "logup_columns": 4 * ir.fracs.len().div_ceil(2),
        "ir_ops": ir.ops.len(),
        "scalars": ir.scalars.len(),
        "loads_per_row": ir.loads(),
        "m31_muls_per_row": ir.m31_muls(),
        "mask_offsets": offsets.iter().copied().collect::<Vec<isize>>(),
        "trees_read": trees.iter().copied().collect::<Vec<usize>>(),
        "columns": {"preprocessed": ir.cols[0], "base": ir.cols[1], "interaction": ir.cols[2]},
        // what the constraint IR for the same component costs, for scale: 2b
        // pays this per row of the *evaluation* domain, the fractions once per
        // row of the trace domain
        "constraint_ir_ops": const_ir.ops.len(),
        "constraint_m31_muls_per_row": const_ir.m31_muls(),
        "fingerprint": format!("{:#010x}", ir.fingerprint()),
    })
}

/// `logupcheck`: lower every component's LogUp fractions and report what a
/// device-side interaction pass would have to run.
///
/// Section 18.7 asked whether 2b's generator already emits the denominators.
/// It does -- `ExprEvaluator::add_to_relation` makes each one an extension
/// intermediate and `lower_mut` has lowered them since 2b -- so this is not a
/// new generator but the same one with `fracs` as its output list. What is
/// worth measuring is the *shape*: how many lookups per family, what a row
/// costs, and whether any family reads a lookup value at a nonzero mask offset
/// or out of the interaction trace, either of which a generation kernel would
/// have to handle specially.
pub fn logup_cmd(args: &[String]) {
    let (l, vname) = crate::layer_of(args);
    let params = crate::params_of(&l, args, &vname);
    let plan = crate::compose::Plan::new(&params);

    let mut ch = stwo::core::channel::Blake2sM31Channel::default();
    stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_4B04);
    let rels = crate::compose::Rels::draw(&mut ch, &plan.widths,
                                          plan.glue.as_ref().map(|g| g.uns.as_slice()));
    let f = stwo::core::channel::Channel::draw_secure_felts(&mut ch, 6);
    let claims = crate::exprprobe::claims_for(&plan, 0x0b65_4b04);
    let b = crate::compose::build_components(&plan, &rels, f[0], f[1], f[2], f[3], f[4], f[5], &claims);

    let mut rows = vec![];
    macro_rules! go {
        ($name:expr, $c:expr) => {
            rows.push(logup_report($name, $c))
        };
    }
    for i in 0..b.chans.len() {
        go!(format!("chan_band{i}"), &b.chans[i]);
        go!(format!("key_band{i}"), &b.keys[i]);
    }
    for i in 0..b.rows.len() {
        go!(format!("row_band{i}"), &b.rows[i]);
    }
    go!("attnout".to_string(), &b.ao);
    for i in 0..b.mats.len() {
        go!(format!("dequant{i}"), &b.mats[i]);
    }
    for i in 0..b.walks.len() {
        go!(format!("walk{i}"), &b.walks[i]);
    }
    go!("exp_table".to_string(), &b.exp_t);
    for i in 0..b.tables.len() {
        go!(format!("table{i}"), &b.tables[i]);
    }

    let shifted: Vec<String> = rows
        .iter()
        .filter(|r| r["mask_offsets"].as_array().unwrap().iter().any(|o| o != 0))
        .map(|r| r["name"].as_str().unwrap().to_string())
        .collect();
    let from_interaction: Vec<String> = rows
        .iter()
        .filter(|r| r["trees_read"].as_array().unwrap().iter().any(|t| t == 2))
        .map(|r| r["name"].as_str().unwrap().to_string())
        .collect();
    let lookups: u64 = rows.iter().map(|r| r["lookups"].as_u64().unwrap()).sum();
    let cols: u64 = rows.iter().map(|r| r["logup_columns"].as_u64().unwrap()).sum();
    eprintln!(
        "logupcheck: {} components, {lookups} lookups, {cols} LogUp columns; \
         {} read a shifted row, {} read the interaction trace",
        rows.len(),
        shifted.len(),
        from_interaction.len(),
    );
    crate::util::emit(
        "logupcheck",
        json!({
            "source": l.source,
            "components": rows.len(),
            "lookups": lookups,
            "logup_columns": cols,
            "components_reading_a_shifted_row": shifted,
            "components_reading_the_interaction_trace": from_interaction,
            "rows": rows,
        }),
    );
}

// ------------------------------------------------------------- gen kernels ----
//
// B7a milestone 4 (18.7). The generation counterpart of `emit_family`: the same
// instruction list, stopped at the fractions, over the *trace* domain. The fork
// does not dispatch these -- our own crate calls them -- so the launcher is a
// plain `extern "C"` and there is no blob, no `cuda_eval_name` and no case in
// `evaluate_constraints.cu`.

/// One family's generated fraction translation unit. `seg_cap` splits an
/// over-long shape by column range (B11 stream K), exactly as `emit_family`
/// splits by constraint; at or under the cap the output is byte-identical.
pub fn emit_frac_family(family: &str, shapes: &[(String, Ir)], launch: Launch, seg_cap: usize) -> Emitted {
    // `cuda_eval_name()` already starts with `b65_`; don't say it twice.
    let fname = format!("gen_{family}");
    let segs: Vec<Option<Vec<Seg>>> =
        shapes.iter().map(|(_, ir)| ir.frac_segments(seg_cap)).collect();
    let mut cu = format!(
        "// Generated by experiments/b7/gpu/m2b/cudagen.rs (B7a milestone 4, 18.7).\n\
         // Do not edit: regenerate with `b65 genemit`.\n\
         //\n\
         // One kernel per distinct *fraction* shape of the `{family}` family: the\n\
         // lookups' numerators and denominators at one row of the trace domain,\n\
         // already batched in pairs the way `finalize_logup_in_pairs` batches them.\n\
         // Everything above the write-back is the same straight-line code the\n\
         // constraint kernel runs; only the output list differs.\n\
         #include \"fields.cuh\"\n\
         #include \"utils.cuh\"\n\
         #include \"b65_ops.cuh\"\n\
         #include \"{fname}.cuh\"\n\n"
    );
    for (id, (name, ir)) in shapes.iter().enumerate() {
        assert!(!ir.fracs.is_empty(), "{name}: a fraction kernel with no lookups");
        cu += &format!(
            "// shape {id}: {name} -- {} instructions, {} lookups, {} columns, {} scalars, \
             fingerprint {:#010x}\n",
            ir.ops.len(),
            ir.fracs.len(),
            ir.fracs.len().div_ceil(2),
            ir.scalars.len(),
            ir.fingerprint()
        );
        let kernel = |kname: String, body: String| {
            format!(
                "{}\n\
                 __global__ void {kname}(\n\
                 \x20   m31 **t0, m31 **t1,\n\
                 \x20   const qm31 *__restrict__ sc,\n\
                 \x20   qm31 *__restrict__ num,\n\
                 \x20   qm31 *__restrict__ den,\n\
                 \x20   unsigned n_rows\n\
                 ) {{\n\
                 \x20   const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;\n\
                 \x20   if (row >= n_rows) return;\n{body}}}\n\n",
                launch.attr()
            )
        };
        match &segs[id] {
            None => cu += &kernel(format!("{fname}_s{id}_kernel"), ir.emit_frac_body()),
            Some(ss) => {
                for (p, s) in ss.iter().enumerate() {
                    cu += &format!(
                        "// shape {id} segment {p}: columns {}..{}, {} instructions\n",
                        s.units.start, s.units.end, s.n_instrs
                    );
                    cu += &kernel(
                        format!("{fname}_s{id}p{p}_kernel"),
                        ir.emit_frac_body_for(Some(&s.mask), s.units.clone()),
                    );
                }
            }
        }
    }

    let n_shapes = shapes.len();
    let fps: Vec<String> =
        shapes.iter().map(|(_, i)| format!("{:#010x}u", i.fingerprint())).collect();
    let ns: Vec<String> = shapes.iter().map(|(_, i)| format!("{}u", i.scalars.len())).collect();
    let nc: Vec<String> =
        shapes.iter().map(|(_, i)| format!("{}u", i.fracs.len().div_ceil(2))).collect();
    // B7a milestone 4 (section 3 row 1): a shape's column count is checked only
    // for a tree the kernel actually *loads* from. `Ir::fingerprint` is the
    // signature of the computation and a declared-but-unread column is not part
    // of it -- the three `qkv`/`o`/`gateup` walk regions lower to one shape and
    // declare 412, 140 and 276 preprocessed weight columns, none of which the
    // fractions read. `~0u` is "this shape reads no column of this tree", and
    // the caller-side `devfrac::assert_supplied` is what checks the rest.
    let reads = |ir: &Ir, t: usize| {
        ir.ops.iter().any(|o| matches!(*o, Op::Load { tree, .. } if tree == t))
    };
    let count = |ir: &Ir, t: usize| {
        if reads(ir, t) { format!("{}u", ir.cols[t]) } else { "~0u".to_string() }
    };
    let n0: Vec<String> = shapes.iter().map(|(_, i)| count(i, 0)).collect();
    let n1: Vec<String> = shapes.iter().map(|(_, i)| count(i, 1)).collect();
    cu += &format!(
        "static const unsigned {fname}_n_shapes = {n_shapes}u;\n\
         static const unsigned {fname}_fp[{n_shapes}] = {{{}}};\n\
         static const unsigned {fname}_nsc[{n_shapes}] = {{{}}};\n\
         static const unsigned {fname}_ncols[{n_shapes}] = {{{}}};\n\
         static const unsigned {fname}_npre[{n_shapes}] = {{{}}};\n\
         static const unsigned {fname}_nbase[{n_shapes}] = {{{}}};\n\n",
        fps.join(", "),
        ns.join(", "),
        nc.join(", "),
        n0.join(", "),
        n1.join(", ")
    );
    cu += &format!(
        "extern \"C\" void {fname}(\n\
         \x20   unsigned shape, unsigned fingerprint,\n\
         \x20   m31 **pre_cols, unsigned n_pre,\n\
         \x20   m31 **base_cols, unsigned n_base,\n\
         \x20   const qm31 *sc_host, unsigned n_sc,\n\
         \x20   qm31 *num, qm31 *den,\n\
         \x20   unsigned n_rows, unsigned n_cols\n\
         ) {{\n\
         \x20   cudaGetLastError();\n\
         \x20   b65_check(shape < {fname}_n_shapes, \"{fname}: shape id out of range\");\n\
         \x20   b65_check(fingerprint == {fname}_fp[shape], \"{fname}: fingerprint\");\n\
         \x20   b65_check(n_sc == {fname}_nsc[shape], \"{fname}: scalar count\");\n\
         \x20   b65_check(n_cols == {fname}_ncols[shape], \"{fname}: column count\");\n\
         \x20   b65_check({fname}_npre[shape] == ~0u || n_pre == {fname}_npre[shape],\n\
         \x20       \"{fname}: preprocessed column count\");\n\
         \x20   b65_check({fname}_nbase[shape] == ~0u || n_base == {fname}_nbase[shape],\n\
         \x20       \"{fname}: base column count\");\n\
         \x20   m31 **d_t0 = n_pre ? clone_to_device<m31*>(pre_cols, n_pre) : nullptr;\n\
         \x20   m31 **d_t1 = n_base ? clone_to_device<m31*>(base_cols, n_base) : nullptr;\n\
         \x20   qm31 *d_sc = nullptr;\n\
         \x20   if (n_sc) {{\n\
         \x20       d_sc = cuda_malloc<qm31>(n_sc);\n\
         \x20       ASSERT_CUDA_SUCCESS(cudaMemcpy(d_sc, sc_host,\n\
         \x20           (size_t)n_sc * sizeof(qm31), cudaMemcpyHostToDevice));\n\
         \x20   }}\n\
         \x20   const unsigned block = n_rows < {block}u ? n_rows : {block}u;\n\
         \x20   const unsigned blocks = (n_rows + block - 1u) / block;\n\
         \x20   switch (shape) {{\n",
        block = launch.block
    );
    for id in 0..n_shapes {
        match &segs[id] {
            None => {
                cu += &format!(
                    "\x20   case {id}u: {fname}_s{id}_kernel<<<blocks, block>>>(\n\
                     \x20       d_t0, d_t1, d_sc, num, den, n_rows);\n\
                     \x20       break;\n"
                );
            }
            // the per-column writes are disjoint, so the segments just run in
            // sequence -- no accumulate path involved
            Some(ss) => {
                cu += &format!("\x20   case {id}u:\n");
                for p in 0..ss.len() {
                    cu += &format!(
                        "\x20       {fname}_s{id}p{p}_kernel<<<blocks, block>>>(\n\
                         \x20           d_t0, d_t1, d_sc, num, den, n_rows);\n"
                    );
                }
                cu += "\x20       break;\n";
            }
        }
    }
    cu += &format!(
        "\x20   }}\n\
         \x20   ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());\n\
         \x20   ASSERT_CUDA_SUCCESS(cudaGetLastError());\n\
         \x20   if (d_t0) cuda_free_memory(d_t0);\n\
         \x20   if (d_t1) cuda_free_memory(d_t1);\n\
         \x20   if (d_sc) cuda_free_memory(d_sc);\n\
         }}\n"
    );

    let guard = fname.to_uppercase();
    let cuh = format!(
        "// Generated by experiments/b7/gpu/m2b/cudagen.rs. Do not edit.\n\
         #ifndef {guard}_CUH\n#define {guard}_CUH\n#include \"fields.cuh\"\n\
         extern \"C\" void {fname}(\n\
         \x20   unsigned shape, unsigned fingerprint,\n\
         \x20   m31 **pre_cols, unsigned n_pre,\n\
         \x20   m31 **base_cols, unsigned n_base,\n\
         \x20   const qm31 *sc_host, unsigned n_sc,\n\
         \x20   qm31 *num, qm31 *den,\n\
         \x20   unsigned n_rows, unsigned n_cols);\n#endif\n"
    );

    Emitted {
        family: family.to_string(),
        cu,
        cuh,
        shapes: shapes.iter().map(|(_, i)| (i.fingerprint(), i.scalars.len())).collect(),
        seg_instrs: shapes
            .iter()
            .zip(&segs)
            .map(|((_, ir), s)| match s {
                None => vec![ir.ops.len()],
                Some(ss) => ss.iter().map(|s| s.n_instrs).collect(),
            })
            .collect(),
    }
}

/// `genemit`: lower the requested families' fractions and write the generation
/// kernels, the Rust shape table and a manifest.
///
/// `--families` defaults to the dequant family alone. 18.7's scope: one family
/// end to end beats four ported and none verified, and `logupcheck` already
/// says the other ten are reachable by the same route.
pub fn gen_emit_cmd(args: &[String]) {
    let (l, vname) = crate::layer_of(args);
    let params = crate::params_of(&l, args, &vname);
    let plan = crate::compose::Plan::new(&params);
    let out = crate::arg_str(args, "--out").unwrap_or_else(|| "/root/b7/gen".to_string());
    let want: Vec<String> =
        crate::arg_strs(args, "--families", &["b65_dequant_eval"]);
    let launch = Launch {
        block: crate::arg_scalar(args, "--block", 256u32),
        min_blocks: crate::arg_scalar(args, "--min-blocks", 2u32),
    };
    // B11 stream K: instruction count above which a shape's kernel is split.
    let seg_cap: usize = crate::arg_scalar(args, "--seg-cap", 3000usize);
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).unwrap();

    let mut ch = stwo::core::channel::Blake2sM31Channel::default();
    stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_4B05);
    let rels = crate::compose::Rels::draw(&mut ch, &plan.widths,
                                          plan.glue.as_ref().map(|g| g.uns.as_slice()));
    let f = stwo::core::channel::Channel::draw_secure_felts(&mut ch, 6);
    let claims = crate::exprprobe::claims_for(&plan, 0x0b65_4b05);
    let b = crate::compose::build_components(&plan, &rels, f[0], f[1], f[2], f[3], f[4], f[5], &claims);

    // Same accumulate-across-design-points store as `cudaemit`, in its own file:
    // a dequant region's *fraction* shape follows its matmul just as its
    // constraint shape does.
    let store = dir.join("gen_shapes.json");
    let mut fams: Vec<(String, Vec<(String, Ir)>)> = if store.exists() {
        let prev: Vec<(String, String, Ir)> =
            serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
        let mut v: Vec<(String, Vec<(String, Ir)>)> = vec![];
        for (family, name, ir) in prev {
            match v.iter_mut().find(|(f, _)| *f == family) {
                Some((_, s)) => s.push((name, ir)),
                None => v.push((family, vec![(name, ir)])),
            }
        }
        eprintln!(
            "genemit: carrying {} shapes forward from {}",
            v.iter().map(|(_, s)| s.len()).sum::<usize>(),
            store.display()
        );
        v
    } else {
        vec![]
    };
    let carried: usize = fams.iter().map(|(_, s)| s.len()).sum();

    let mut rows = vec![];
    {
        let mut add = |family: &str, name: String, ir: Ir| {
            if !want.iter().any(|w| w == family) {
                return;
            }
            let fp = ir.fingerprint();
            let fi = match fams.iter().position(|(f, _)| f == family) {
                Some(i) => i,
                None => {
                    fams.push((family.to_string(), vec![]));
                    fams.len() - 1
                }
            };
            let shapes = &mut fams[fi].1;
            let sid = match shapes.iter().position(|(_, i)| i.fingerprint() == fp) {
                Some(i) => i,
                None => {
                    shapes.push((name.clone(), ir.clone()));
                    shapes.len() - 1
                }
            };
            rows.push(json!({
                "component": name, "family": family, "shape": sid,
                "fingerprint": format!("{fp:#010x}"),
                "instructions": ir.ops.len(), "lookups": ir.fracs.len(),
                "logup_columns": 4 * ir.fracs.len().div_ceil(2),
                "scalars": ir.scalars.len(), "m31_muls_per_row": ir.m31_muls(),
                "loads_per_row": ir.loads(), "log_size": ir.log_size,
                "columns": {"preprocessed": ir.cols[0], "base": ir.cols[1]},
            }));
        };
        macro_rules! go {
            ($c:expr, $name:expr) => {{
                let ir = lower_logup(&$c.eval, $c.eval.log_size());
                add($c.eval.cuda_eval_name(), $name, ir);
            }};
        }
        for i in 0..b.chans.len() {
            go!(b.chans[i], format!("chan_band{i}"));
            go!(b.keys[i], format!("key_band{i}"));
        }
        // a corner band has no row component (B10 step 3): `rows` is shorter
        // than `chans`, so it is walked on its own
        for i in 0..b.rows.len() {
            go!(b.rows[i], format!("row_band{i}"));
        }
        go!(b.ao, "attnout".to_string());
        for i in 0..b.mats.len() {
            go!(b.mats[i], format!("dequant{i}"));
        }
        for i in 0..b.glue.len() {
            go!(b.glue[i], format!("glue{i}"));
        }
        for i in 0..b.gtables.len() {
            go!(b.gtables[i], format!("gtable{i}"));
        }
        for i in 0..b.walks.len() {
            go!(b.walks[i], format!("walk{i}"));
        }
        go!(b.exp_t, "exp_table".to_string());
        for i in 0..b.tables.len() {
            go!(b.tables[i], format!("table{i}"));
        }
    }
    let fresh: usize = fams.iter().map(|(_, s)| s.len()).sum::<usize>() - carried;

    let flat: Vec<(String, String, Ir)> = fams
        .iter()
        .flat_map(|(f, s)| s.iter().map(move |(n, i)| (f.clone(), n.clone(), i.clone())))
        .collect();
    std::fs::write(&store, serde_json::to_vec(&flat).unwrap()).unwrap();

    let mut manifest = vec![];
    let mut rs = String::from(
        "// Generated by `b65 genemit` (B7a milestone 4, 18.7). Do not edit.\n\
         //\n\
         // The fraction shapes compiled into `libstwo_cuda.a`, by family and\n\
         // fingerprint. A component whose fraction IR is not in this table has no\n\
         // kernel, and the prover falls back to the host builder rather than\n\
         // calling one that computes something else.\n\
         pub static GEN_SHAPES: &[(&str, u32, usize, usize)] = &[\n",
    );
    for (family, shapes) in &fams {
        let e = emit_frac_family(family, shapes, launch, seg_cap);
        std::fs::write(dir.join(format!("gen_{family}.cu")), &e.cu).unwrap();
        std::fs::write(dir.join(format!("gen_{family}.cuh")), &e.cuh).unwrap();
        for (id, si) in e.seg_instrs.iter().enumerate() {
            if si.len() > 1 {
                eprintln!(
                    "  {family} fraction shape {id}: split into {} segments at cap {seg_cap}: {si:?}",
                    si.len()
                );
            }
        }
        for (sid, (_, ir)) in shapes.iter().enumerate() {
            rs += &format!(
                "    (\"{family}\", {:#010x}, {sid}, {}),\n",
                ir.fingerprint(),
                ir.fracs.len()
            );
        }
        manifest.push(json!({
            "family": family,
            "shapes": shapes.len(),
            "cu_lines": e.cu.lines().count(),
            "lookups": shapes.iter().map(|(_, i)| i.fracs.len()).collect::<Vec<usize>>(),
            "instructions": shapes.iter().map(|(_, i)| i.ops.len()).sum::<usize>(),
            "segments": e.seg_instrs,
        }));
    }
    rs += "];\n\n\
           /// `(shape id, lookups)` for a family's fraction fingerprint.\n\
           pub fn gen_shape_of(name: &str, fp: u32) -> Option<(usize, usize)> {\n\
           \x20   GEN_SHAPES.iter().find(|(n, f, _, _)| *n == name && *f == fp)\n\
           \x20       .map(|(_, _, s, l)| (*s, *l))\n\
           }\n";
    std::fs::write(dir.join("gencudakernels.rs"), &rs).unwrap();
    std::fs::write(dir.join("b65_ops.cuh"), OPS_CUH).unwrap();

    // One dispatcher over the generated families, keyed on the same
    // `cuda_eval_name()` the fork's constraint dispatch uses, so the Rust side
    // declares one `extern "C"` symbol however many families are generated.
    let mut disp = String::from(
        "// Generated by `b65 genemit` (B7a milestone 4, 18.7). Do not edit.\n         #include <cstring>\n#include \"b65_ops.cuh\"\n         #include \"gen_b65_dispatch.cuh\"\n",
    );
    for (family, _) in &fams {
        disp += &format!("#include \"gen_{family}.cuh\"\n");
    }
    disp += "\nextern \"C\" void gen_b65_dispatch(\n                 const char *family, unsigned shape, unsigned fingerprint,\n                 m31 **pre_cols, unsigned n_pre,\n                 m31 **base_cols, unsigned n_base,\n                 const qm31 *sc_host, unsigned n_sc,\n                 qm31 *num, qm31 *den,\n                 unsigned n_rows, unsigned n_cols\n             ) {\n";
    for (family, _) in &fams {
        disp += &format!(
            "    if (!strcmp(family, \"{family}\")) {{\n                     gen_{family}(shape, fingerprint, pre_cols, n_pre, base_cols, n_base,\n                         sc_host, n_sc, num, den, n_rows, n_cols);\n                     return;\n    }}\n"
        );
    }
    disp += "    b65_check(false, \"gen_b65_dispatch: no generation kernel for this family\");\n}\n";
    std::fs::write(dir.join("gen_b65_dispatch.cu"), &disp).unwrap();
    std::fs::write(
        dir.join("gen_b65_dispatch.cuh"),
        "// Generated by `b65 genemit`. Do not edit.\n         #ifndef GEN_B65_DISPATCH_CUH\n#define GEN_B65_DISPATCH_CUH\n         #include \"fields.cuh\"\n         extern \"C\" void gen_b65_dispatch(\n             const char *family, unsigned shape, unsigned fingerprint,\n             m31 **pre_cols, unsigned n_pre,\n             m31 **base_cols, unsigned n_base,\n             const qm31 *sc_host, unsigned n_sc,\n             qm31 *num, qm31 *den,\n             unsigned n_rows, unsigned n_cols);\n#endif\n",
    )
    .unwrap();

    let man = json!({
        "families": manifest,
        "shapes": fams.iter().map(|(_, s)| s.len()).sum::<usize>(),
        "fresh_shapes": fresh,
        "components": rows,
    });
    std::fs::write(dir.join("gen_manifest.json"), serde_json::to_vec_pretty(&man).unwrap())
        .unwrap();
    eprintln!(
        "genemit: {} families, {} fraction shapes ({fresh} new) -> {}",
        fams.len(),
        fams.iter().map(|(_, s)| s.len()).sum::<usize>(),
        dir.display()
    );
    crate::util::emit("genemit", man);
}
