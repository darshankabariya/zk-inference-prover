//! B7a milestone 2b, step 1: run every `FrameworkEval` of the composed proof
//! through the fork's symbolic `ExprEvaluator` and report what comes out.
//!
//! README section 13 named this the cheap first experiment of 2b: `ExprEvaluator`
//! looks complete on inspection, but it had never been run against our eleven
//! evals. This module answers, per component and without writing a kernel:
//!
//!  * does the symbolic run survive at all (`catch_unwind`), and if not, where;
//!  * does it agree with `InfoEvaluator` on constraint count, mask columns and
//!    logup fractions per row -- i.e. is the symbolic trace the same trace the
//!    prover will bill;
//!  * what the generator would have to lower: node counts, depth, the qm31 and
//!    m31 constants, the parameters, the `Inv` nodes, and the per-row operation
//!    count (the analytic cost of one row of constraint evaluation);
//!  * which constants are challenge-derived. Every component is built twice
//!    from two independent challenge draws; a constant that differs between the
//!    two runs is a Fiat-Shamir value and must reach the kernel as a runtime
//!    parameter, not as a literal. A constant that agrees is structural and can
//!    be baked. This is the measurement that decides whether one compiled
//!    kernel can serve every transcript.
//!
//! Nothing here writes CUDA. It is a report.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::channel::Channel;
use stwo::core::fields::m31::M31;
use stwo_constraint_framework::expr::{BaseExpr, ExprEvaluator, ExtExpr};
use stwo_constraint_framework::{FrameworkComponent, FrameworkEval, InfoEvaluator};

use crate::compose::BuiltComponents;

// ------------------------------------------------------------------ walking ----

/// Per-row operation count, in the unit the generated kernel actually pays:
/// one M31 multiply. A QM31 product is schoolbook over the tower, so its cost
/// is `width(a) * width(b)` M31 multiplies, where an expression's *width* is 1
/// if it is a base-field value embedded in the extension (a `SecureCol` whose
/// upper three components are literal zero) and 4 otherwise. That distinction
/// is not cosmetic: our AIRs are full of `column * challenge_power`, which is
/// width 1 x width 4 = 4 multiplies, not 16.
#[derive(Default, Clone)]
pub struct Ops {
    pub m31_mul: usize,
    pub m31_add: usize,
    /// multiplies by width pair: (1,1), (1,4)/(4,1), (4,4)
    pub mul_1x1: usize,
    pub mul_1x4: usize,
    pub mul_4x4: usize,
    /// `Mul` where one side is a literal: same arithmetic, but the operand is
    /// a constant-pool read rather than a live register.
    pub const_mul: usize,
}

impl Ops {
    fn json(&self) -> Value {
        json!({
            "m31_mul": self.m31_mul, "m31_add": self.m31_add,
            "mul_1x1": self.mul_1x1, "mul_1x4": self.mul_1x4, "mul_4x4": self.mul_4x4,
            "const_mul": self.const_mul,
        })
    }
}

/// The hash-consed lowering of one component's constraint trees: every
/// structurally identical subexpression becomes one node, which is exactly one
/// instruction in the generated kernel. `tree_nodes` is the same walk without
/// sharing, so the ratio says how much common-subexpression elimination buys.
#[derive(Default)]
pub struct Stats {
    intern: BTreeMap<String, usize>,
    width: Vec<u8>,
    zero_id: Option<usize>,
    pub tree_nodes: usize,
    pub depth: usize,
    /// distinct (interaction, offset) pairs -> how many distinct columns are
    /// read at that offset: the loads one row of the kernel issues
    pub col_refs: BTreeMap<(usize, isize), usize>,
    /// interaction -> the distinct `idx` values the symbolic run assigned
    pub col_idx: BTreeMap<usize, Vec<usize>>,
    /// (interaction, idx) -> the offsets that column is read at. Compared
    /// element-wise against `InfoEvaluator::mask_offsets` in `probe_one`: that
    /// is what turns "the k-th idx is presumably column k" into a check.
    pub col_offsets: BTreeMap<(usize, usize), std::collections::BTreeSet<isize>>,
    /// m31 constants, in first-appearance order
    pub m31_consts: Vec<u32>,
    /// qm31 constants (a `SecureCol` of four constants, or an `ExtExpr::Const`),
    /// in first-appearance order -- the candidate runtime-parameter list
    pub qm31_consts: Vec<[u32; 4]>,
    pub qm31_const_refs: usize,
    /// parameter name -> reference count
    pub params: BTreeMap<String, usize>,
    /// the operand of every `Inv` node, formatted (empty if there are none)
    pub inv: Vec<String>,
    pub ops: Ops,
}

impl Stats {
    fn nodes(&self) -> usize {
        self.width.len()
    }

    /// Intern a node; returns (id, is_new). Ops are charged once, on first sight.
    fn node(&mut self, key: String, w: u8) -> (usize, bool) {
        if let Some(i) = self.intern.get(&key) {
            return (*i, false);
        }
        let i = self.width.len();
        self.width.push(w);
        self.intern.insert(key, i);
        (i, true)
    }

    fn is_const(&self, id: usize, consts: &BTreeMap<usize, ()>) -> bool {
        consts.contains_key(&id)
    }

    fn base(&mut self, e: &BaseExpr, d: usize, consts: &mut BTreeMap<usize, ()>) -> usize {
        self.tree_nodes += 1;
        self.depth = self.depth.max(d);
        match e {
            BaseExpr::Col(_) => {
                // `ColumnExpr`'s fields are private in the fork, but `format_expr`
                // on the leaf renders them: trace_{i}_column_{idx}_offset_{n|neg_n}.
                let s = e.format_expr();
                let f: Vec<&str> = s.split('_').collect();
                let (i, idx, off) = match f.as_slice() {
                    ["trace", i, "column", idx, "offset", "neg", n] => (
                        i.parse::<usize>().unwrap(),
                        idx.parse::<usize>().unwrap(),
                        -n.parse::<isize>().unwrap(),
                    ),
                    ["trace", i, "column", idx, "offset", n] => (
                        i.parse::<usize>().unwrap(),
                        idx.parse::<usize>().unwrap(),
                        n.parse::<isize>().unwrap(),
                    ),
                    _ => panic!("unparsed column leaf {s:?}"),
                };
                let (id, new) = self.node(format!("C{i}_{idx}_{off}"), 1);
                if new {
                    *self.col_refs.entry((i, off)).or_default() += 1;
                    let v = self.col_idx.entry(i).or_default();
                    if !v.contains(&idx) {
                        v.push(idx);
                    }
                    self.col_offsets.entry((i, idx)).or_default().insert(off);
                }
                id
            }
            BaseExpr::Const(c) => {
                let (id, new) = self.node(format!("K{}", c.0), 1);
                if new {
                    self.m31_consts.push(c.0);
                }
                consts.insert(id, ());
                if c.0 == 0 {
                    self.zero_id = Some(id);
                }
                id
            }
            BaseExpr::Param(p) => {
                *self.params.entry(p.clone()).or_default() += 1;
                self.node(format!("P{p}"), 1).0
            }
            BaseExpr::Add(a, b) | BaseExpr::Sub(a, b) => {
                let t = if matches!(e, BaseExpr::Add(..)) { 'A' } else { 'S' };
                let (x, y) = (self.base(a, d + 1, consts), self.base(b, d + 1, consts));
                let (id, new) = self.node(format!("{t}{x},{y}"), 1);
                if new {
                    self.ops.m31_add += 1;
                }
                id
            }
            BaseExpr::Mul(a, b) => {
                let (x, y) = (self.base(a, d + 1, consts), self.base(b, d + 1, consts));
                let (id, new) = self.node(format!("M{x},{y}"), 1);
                if new {
                    self.ops.m31_mul += 1;
                    self.ops.mul_1x1 += 1;
                    if self.is_const(x, consts) || self.is_const(y, consts) {
                        self.ops.const_mul += 1;
                    }
                }
                id
            }
            BaseExpr::Neg(a) => {
                let x = self.base(a, d + 1, consts);
                let (id, new) = self.node(format!("N{x}"), 1);
                if new {
                    self.ops.m31_add += 1;
                }
                id
            }
            BaseExpr::Inv(a) => {
                self.inv.push(a.format_expr());
                let x = self.base(a, d + 1, consts);
                // No op is charged: every `Inv` our AIRs produce is over a
                // compile-time constant (see the report's `inv` field), so the
                // generator folds it. A non-constant operand would need a device
                // inversion and is reported, not silently priced at zero.
                self.node(format!("I{x}"), 1).0
            }
        }
    }

    fn ext(&mut self, e: &ExtExpr, d: usize, consts: &mut BTreeMap<usize, ()>) -> usize {
        self.tree_nodes += 1;
        self.depth = self.depth.max(d);
        if let Some(c) = qm31_const(e) {
            self.qm31_const_refs += 1;
            let w = if c[1] == 0 && c[2] == 0 && c[3] == 0 { 1 } else { 4 };
            let (id, new) = self.node(format!("Q{},{},{},{}", c[0], c[1], c[2], c[3]), w);
            if new {
                self.qm31_consts.push(c);
            }
            consts.insert(id, ());
            return id;
        }
        match e {
            ExtExpr::Const(_) => unreachable!("handled by qm31_const"),
            ExtExpr::Param(p) => {
                *self.params.entry(p.clone()).or_default() += 1;
                self.node(format!("X{p}"), 4).0
            }
            ExtExpr::SecureCol(parts) => {
                let ids: Vec<usize> = parts
                    .iter()
                    .map(|b| self.base(b, d + 1, consts))
                    .collect();
                // A base-field value lifted into the extension keeps width 1:
                // multiplying by it costs 4 M31 products, not 16.
                let z = self.zero_id;
                let w = if z.is_some() && ids[1..].iter().all(|i| Some(*i) == z) {
                    1
                } else {
                    4
                };
                self.node(format!("V{},{},{},{}", ids[0], ids[1], ids[2], ids[3]), w)
                    .0
            }
            ExtExpr::Add(a, b) | ExtExpr::Sub(a, b) => {
                let t = if matches!(e, ExtExpr::Add(..)) { 'a' } else { 's' };
                let (x, y) = (self.ext(a, d + 1, consts), self.ext(b, d + 1, consts));
                let w = self.width[x].max(self.width[y]);
                let (id, new) = self.node(format!("{t}{x},{y}"), w);
                if new {
                    self.ops.m31_add += w as usize;
                }
                id
            }
            ExtExpr::Mul(a, b) => {
                let (x, y) = (self.ext(a, d + 1, consts), self.ext(b, d + 1, consts));
                let (wx, wy) = (self.width[x], self.width[y]);
                let w = if wx == 1 && wy == 1 { 1 } else { 4 };
                let (id, new) = self.node(format!("m{x},{y}"), w);
                if new {
                    let cost = wx as usize * wy as usize;
                    self.ops.m31_mul += cost;
                    match cost {
                        1 => self.ops.mul_1x1 += 1,
                        4 => self.ops.mul_1x4 += 1,
                        _ => self.ops.mul_4x4 += 1,
                    }
                    if self.is_const(x, consts) || self.is_const(y, consts) {
                        self.ops.const_mul += 1;
                    }
                }
                id
            }
            ExtExpr::Neg(a) => {
                let x = self.ext(a, d + 1, consts);
                let w = self.width[x];
                let (id, new) = self.node(format!("n{x}"), w);
                if new {
                    self.ops.m31_add += w as usize;
                }
                id
            }
        }
    }

    fn json(&self, n_preprocessed: usize) -> Value {
        let loads: usize = self.col_refs.values().sum();
        json!({
            "dag_nodes": self.nodes(),
            "tree_nodes": self.tree_nodes,
            "cse_ratio": self.tree_nodes as f64 / self.nodes().max(1) as f64,
            "depth": self.depth,
            "loads_per_row": loads,
            // every preprocessed column is a load too; it arrives as a Param, so
            // it is not in `col_refs` and has to be added back explicitly.
            "loads_per_row_with_preprocessed": loads + n_preprocessed,
            "bytes_per_row": (loads + n_preprocessed) * 4,
            "col_refs": self.col_refs.iter()
                .map(|((i, o), n)| json!({"interaction": i, "offset": o, "columns": n}))
                .collect::<Vec<_>>(),
            "cols_per_interaction": self.col_idx.iter()
                .map(|(i, v)| (i.to_string(), json!(v.len())))
                .collect::<serde_json::Map<_, _>>(),
            "m31_consts": self.m31_consts.len(),
            "m31_const_values": if self.m31_consts.len() <= 24 {
                json!(self.m31_consts)
            } else {
                json!(self.m31_consts[..24])
            },
            "qm31_consts": self.qm31_consts.len(),
            "qm31_const_refs": self.qm31_const_refs,
            "params": self.params.iter().map(|(k, v)| (k.clone(), json!(v)))
                .collect::<serde_json::Map<_, _>>(),
            "inv": self.inv,
            "ops": self.ops.json(),
        })
    }
}

fn qm31_const(e: &ExtExpr) -> Option<[u32; 4]> {
    match e {
        ExtExpr::Const(c) => {
            let a = c.to_m31_array();
            Some([a[0].0, a[1].0, a[2].0, a[3].0])
        }
        ExtExpr::SecureCol(p) => {
            let mut out = [0u32; 4];
            for (i, b) in p.iter().enumerate() {
                match &**b {
                    BaseExpr::Const(c) => out[i] = c.0,
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

// ------------------------------------------------------------- one component ----

/// What one symbolic run produced; kept so two draws can be diffed.
pub struct Run {
    pub panic: Option<String>,
    pub n_constraints: usize,
    pub n_intermediates: usize,
    pub n_ext_intermediates: usize,
    pub degree_bounds: Vec<usize>,
    pub stats: Stats,
}

fn run_one<E: FrameworkEval>(eval: &E) -> Run {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let res = catch_unwind(AssertUnwindSafe(|| {
        let ev = eval.evaluate(ExprEvaluator::new());
        // `constraint_degree_bounds` walks the same trees; it is part of the
        // surface 2b needs, so it is inside the same catch.
        let deg = ev.constraint_degree_bounds();
        let mut st = Stats::default();
        let mut consts = BTreeMap::new();
        // Seed the literal zero so `SecureCol`'s width test has something to
        // compare against even if no constraint mentions it first.
        st.base(&BaseExpr::Const(M31::from_u32_unchecked(0)), 0, &mut consts);
        // The intermediates are separate trees the generator emits as `let`s,
        // and the constraints reference them by name; lower them first, in the
        // order `add_to_relation` created them, so the DAG is in dependency
        // order and the numbering matches the relation entries.
        let mut names: Vec<String> = ev
            .intermediates
            .keys()
            .chain(ev.ext_intermediates.keys())
            .cloned()
            .collect();
        names.sort_by_key(|n| n.trim_start_matches("intermediate").parse::<usize>().unwrap_or(0));
        for n in &names {
            if let Some(b) = ev.intermediates.get(n) {
                st.base(b, 0, &mut consts);
            }
            if let Some(x) = ev.ext_intermediates.get(n) {
                st.ext(x, 0, &mut consts);
            }
        }
        for c in &ev.constraints {
            st.ext(c, 0, &mut consts);
        }
        (
            ev.constraints.len(),
            ev.intermediates.len(),
            ev.ext_intermediates.len(),
            deg,
            st,
        )
    }));
    std::panic::set_hook(prev);
    match res {
        Ok((n, ni, nx, deg, st)) => Run {
            panic: None,
            n_constraints: n,
            n_intermediates: ni,
            n_ext_intermediates: nx,
            degree_bounds: deg,
            stats: st,
        },
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".to_string());
            Run {
                panic: Some(msg),
                n_constraints: 0,
                n_intermediates: 0,
                n_ext_intermediates: 0,
                degree_bounds: vec![],
                stats: Stats::default(),
            }
        }
    }
}

/// One component, both draws, reported against what `InfoEvaluator` says.
fn probe_one<E: FrameworkEval>(
    family: &str,
    name: String,
    a: &FrameworkComponent<E>,
    b: &FrameworkComponent<E>,
) -> Value {
    let ra = run_one(&a.eval);
    let rb = run_one(&b.eval);

    // The prover's own accounting, for the same eval.
    let info = a
        .eval
        .evaluate(InfoEvaluator::new(a.eval.log_size(), vec![], a.claimed_sum()));
    let mask_cols: Vec<usize> = info.mask_offsets.iter().map(|t| t.len()).collect();
    let logup_per_row: usize = info.logup_counts.iter().map(|(_, v)| *v).sum();
    // Per relation *name*. The generator cannot key the blob on these: several
    // distinct relation instances share one type name (RangeRel is drawn once
    // per width), and `combine_formal` names parameters by type. Entries are
    // bound by `add_to_relation` order instead; this map is the size check.
    let logup_by_name: BTreeMap<String, usize> = info
        .logup_counts
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let pre: Vec<String> = info
        .preprocessed_columns
        .iter()
        .map(|c| c.id.clone())
        .collect();

    // The measurement 2b needs: which constants moved when the challenges did.
    let (mut moved, mut fixed, mut len_mismatch) = (0usize, 0usize, false);
    if ra.panic.is_none() && rb.panic.is_none() {
        let (x, y) = (&ra.stats.qm31_consts, &rb.stats.qm31_consts);
        len_mismatch = x.len() != y.len();
        for (u, v) in x.iter().zip(y.iter()) {
            if u == v {
                fixed += 1
            } else {
                moved += 1
            }
        }
    }

    // Does the symbolic run see the same shape the prover bills, and does the
    // generator's column mapping hold? `ExprEvaluator` numbers mask columns with
    // a single counter across all interactions, so the generator has to recover
    // "the j-th column of interaction i" as the j-th smallest `idx` seen in that
    // interaction. That is only sound if the offsets line up with
    // `InfoEvaluator::mask_offsets`, which is in true call order -- so compare
    // them element by element.
    let mut map_ok = true;
    let mut map_detail = vec![];
    for (i, cols) in info.mask_offsets.iter().enumerate() {
        let mut idxs = ra.stats.col_idx.get(&i).cloned().unwrap_or_default();
        idxs.sort_unstable();
        if idxs.len() != cols.len() {
            map_ok = false;
            map_detail.push(format!(
                "interaction {i}: symbolic {} columns, info {}", idxs.len(), cols.len()));
            continue;
        }
        for (j, want) in cols.iter().enumerate() {
            let mut want: Vec<isize> = want.clone();
            want.sort_unstable();
            let got: Vec<isize> = ra
                .stats
                .col_offsets
                .get(&(i, idxs[j]))
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            if got != want {
                map_ok = false;
                map_detail.push(format!(
                    "interaction {i} column {j} (idx {}): offsets {got:?} != {want:?}",
                    idxs[j]));
            }
        }
    }
    let agrees = ra.panic.is_none() && ra.n_constraints == a.n_constraints() && map_ok;

    json!({
        "family": family,
        "name": name,
        "cuda_eval_name": a.eval.cuda_eval_name(),
        "log_size": a.eval.log_size(),
        "max_constraint_log_degree_bound": a.eval.max_constraint_log_degree_bound(),
        "ok": ra.panic.is_none() && rb.panic.is_none(),
        "panic": ra.panic.clone().or(rb.panic.clone()),
        "agrees_with_info": agrees,
        "mask_map_ok": map_ok,
        "mask_map_detail": map_detail,
        "info": {
            "n_constraints": a.n_constraints(),
            "mask_cols": mask_cols,
            "logup_fracs_per_row": logup_per_row,
            "logup_fracs_by_relation_name": logup_by_name.iter()
                .map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "n_preprocessed": pre.len(),
            "preprocessed": pre,
        },
        "expr": {
            "n_constraints": ra.n_constraints,
            "n_intermediates": ra.n_intermediates,
            "n_ext_intermediates": ra.n_ext_intermediates,
            "max_degree_bound": ra.degree_bounds.iter().max().copied().unwrap_or(0),
            "degree_bounds": ra.degree_bounds,
        },
        "walk": ra.stats.json(pre.len()),
        "constants": {
            "qm31_total": ra.stats.qm31_consts.len(),
            "qm31_runtime_derived": moved,
            "qm31_structural": fixed,
            "second_draw_length_mismatch": len_mismatch,
            "m31_total": ra.stats.m31_consts.len(),
        },
    })
}

// ------------------------------------------------------------------ the report --

/// Probe every component of the composed proof. `a` and `b` must be two
/// `build_components` results over the same plan and two independent challenge
/// draws.
pub fn probe(a: &BuiltComponents, b: &BuiltComponents) -> Value {
    let mut rows = vec![];
    for i in 0..a.chans.len() {
        rows.push(probe_one("chan", format!("chan_band{i}"), &a.chans[i], &b.chans[i]));
        rows.push(probe_one("key", format!("key_band{i}"), &a.keys[i], &b.keys[i]));
    }
    for i in 0..a.rows.len() {
        rows.push(probe_one("row", format!("row_band{i}"), &a.rows[i], &b.rows[i]));
    }
    rows.push(probe_one("attnout", "attnout".into(), &a.ao, &b.ao));
    for i in 0..a.mats.len() {
        rows.push(probe_one("dequant", format!("dequant{i}"), &a.mats[i], &b.mats[i]));
    }
    for i in 0..a.walks.len() {
        rows.push(probe_one("walk", format!("walk{i}"), &a.walks[i], &b.walks[i]));
    }
    rows.push(probe_one("exp_table", "exp_table".into(), &a.exp_t, &b.exp_t));
    for i in 0..a.tables.len() {
        rows.push(probe_one("table", format!("table{i}"), &a.tables[i], &b.tables[i]));
    }

    // The roll-up: what the generator has to produce, and what it will cost.
    let n = rows.len();
    let ok = rows.iter().filter(|r| r["ok"] == json!(true)).count();
    let agree = rows
        .iter()
        .filter(|r| r["agrees_with_info"] == json!(true))
        .count();
    let map_ok = rows
        .iter()
        .filter(|r| r["mask_map_ok"] == json!(true))
        .count();
    // The whole point of the DAG: `m31_mul` is the instruction count of one row
    // of the generated kernel, and 2^max_constraint_log_degree_bound rows of it
    // run per component. Summed, that is the arithmetic 2b moves onto the device.
    let mut by_family: BTreeMap<String, (usize, usize, usize, usize, f64, f64)> = BTreeMap::new();
    let (mut total_mul, mut total_add, mut total_bytes) = (0f64, 0f64, 0f64);
    for r in &rows {
        let f = r["family"].as_str().unwrap().to_string();
        let logn = r["max_constraint_log_degree_bound"].as_u64().unwrap_or(0) as u32;
        let n_rows = (1u64 << logn) as f64;
        let mul = r["walk"]["ops"]["m31_mul"].as_u64().unwrap_or(0) as f64;
        let add = r["walk"]["ops"]["m31_add"].as_u64().unwrap_or(0) as f64;
        let loads = r["walk"]["loads_per_row_with_preprocessed"].as_u64().unwrap_or(0) as f64;
        let e = by_family.entry(f).or_default();
        e.0 += 1; // components
        e.1 += r["walk"]["dag_nodes"].as_u64().unwrap_or(0) as usize;
        e.2 += r["constants"]["qm31_runtime_derived"].as_u64().unwrap_or(0) as usize;
        e.3 = e.3.max(r["walk"]["depth"].as_u64().unwrap_or(0) as usize);
        e.4 += mul * n_rows;
        e.5 += loads * 4.0 * n_rows;
        total_mul += mul * n_rows;
        total_add += add * n_rows;
        total_bytes += loads * 4.0 * n_rows;
    }

    // The distinct kernels a per-instantiation generator would emit: two
    // components share a kernel only if their whole symbolic shape matches.
    let mut shapes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in &rows {
        let key = format!(
            "{}|{}|{}|{}|{:?}|{:?}",
            r["family"], r["expr"]["n_constraints"], r["walk"]["dag_nodes"],
            r["info"]["logup_fracs_per_row"], r["walk"]["cols_per_interaction"],
            r["expr"]["degree_bounds"],
        );
        shapes
            .entry(key)
            .or_default()
            .push(r["name"].as_str().unwrap().to_string());
    }

    json!({
        "components": n,
        "symbolic_ok": ok,
        "agrees_with_info": agree,
        "mask_map_ok": map_ok,
        "distinct_kernels": shapes.len(),
        "kernel_groups": shapes.iter()
            .map(|(_, v)| json!(v)).collect::<Vec<_>>(),
        "per_family": by_family.iter().map(|(k, v)| json!({
            "family": k, "components": v.0, "dag_nodes": v.1,
            "qm31_runtime_derived": v.2, "max_depth": v.3,
            "m31_mul": v.4, "column_bytes": v.5,
        })).collect::<Vec<_>>(),
        "totals": {
            "m31_mul": total_mul,
            "m31_add": total_add,
            "column_bytes": total_bytes,
            "mul_per_byte": total_mul / total_bytes.max(1.0),
        },
        "rows": rows,
    })
}

// ------------------------------------------------------------- the subcommand ---

/// Fabricate a `Claims` large enough for `build_components` to index. Claim
/// *values* do not change the shape of any constraint -- they enter as
/// coefficients -- so the probe does not need an honest witness, only correctly
/// sized vectors. Sizes are taken generously from the plan: `build_components`
/// consumes `use_sums` and `table_sums` sequentially, so over-length is safe and
/// under-length panics loudly.
pub fn claims_for(plan: &crate::compose::Plan, seed: u64) -> crate::compose::Claims {
    let mut x = seed | 1;
    let mut felt = || {
        // xorshift, then reduce each limb into M31 -- distinct, nonzero, and
        // deterministic per seed, which is all the probe needs.
        let mut a = [0u32; 4];
        for l in a.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *l = ((x % 0x7fff_fffe) + 1) as u32;
        }
        a
    };
    let nb = plan.shapes.len();
    let nm = plan.mat_lays.len();
    let nw = plan.walks.len();
    crate::compose::Claims {
        bands: (0..nb).map(|_| std::array::from_fn(|_| felt())).collect(),
        ao_out: felt(),
        ao_den: felt(),
        mat_h: (0..nm).map(|_| felt()).collect(),
        walk_acc: plan
            .walks
            .iter()
            .map(|w| (0..w.mats.len()).map(|_| felt()).collect())
            .collect(),
        // 3 per band + attnout + one per matmul + one per walk, and a margin.
        use_sums: (0..(3 * nb + 1 + nm + nw + 16)).map(|_| felt()).collect(),
        // exp + norm/frac/mul/qnz + one per range width, and a margin (the
        // glue's table sides come out of the same margin).
        table_sums: (0..(5 + plan.widths.len() + 16)).map(|_| felt()).collect(),
        // `Params::ties` reads these via `.get(..).unwrap_or(zero)`, so the
        // placeholder claims can stay empty without changing the IR's shape.
        glue_ties: vec![], deq_out: vec![], head_key: vec![], am_key: vec![],
        // stream BB: the same, and `fill_tie_claims` gives them nonzero
        // placeholders on the emit path so the emitted shape is the proved one
        embed_out: vec![], embed_tok: vec![],
        // stream II: the same again -- the seams read their claims through
        // `.get(..).unwrap_or(zero)`, so empty placeholders keep the IR's shape
        rq_in: vec![], rq_out: vec![], walk_qx: vec![], ao_q: vec![],
        // B11 stream Q: the GKR bundle is bytes, not a claim; nothing the
        // components lower reads it, so the placeholder statement carries none.
        gkr: None,
    }
}

/// `exprprobe`: build the composed proof's components from two independent
/// challenge draws and report what the fork's `ExprEvaluator` makes of them.
/// Takes the same shape flags as `compose`; runs no trace generation and no
/// prover.
pub fn cmd(args: &[String]) {
    let (l, vname) = crate::layer_of(args);
    let params = crate::params_of(&l, args, &vname);
    let plan = crate::compose::Plan::new(&params);
    eprintln!(
        "exprprobe: {} bands={:?} variant={} widths={:?}",
        l.source, l.band_sizes, vname, plan.widths
    );

    let draw = |tag: u8| {
        // Two independent transcripts: same shape, different Fiat-Shamir.
        let mut ch = stwo::core::channel::Blake2sM31Channel::default();
        ch.mix_u64(0xB65_2B00 + u64::from(tag));
        let rels = crate::compose::Rels::draw(&mut ch, &plan.widths,
                                              plan.glue.as_ref().map(|g| g.uns.as_slice()),
                                              plan.emb.map(|l| l.arity()), !plan.rqs.is_empty());
        let f = ch.draw_secure_felts(6);
        (rels, f)
    };
    let (ra, fa) = draw(1);
    let (rb, fb) = draw(2);
    let ca = claims_for(&plan, 0x1234_5678);
    let cb = claims_for(&plan, 0x9abc_def0);

    let a = crate::compose::build_components(&plan, &ra, fa[0], fa[1], fa[2], fa[3], fa[4],
                                             fa[5], &ca);
    let b = crate::compose::build_components(&plan, &rb, fb[0], fb[1], fb[2], fb[3], fb[4],
                                             fb[5], &cb);
    let report = probe(&a, &b);

    eprintln!(
        "exprprobe: {} components, {} symbolic ok, {} agree with InfoEvaluator, \
         {} distinct kernel shapes",
        report["components"], report["symbolic_ok"], report["agrees_with_info"],
        report["distinct_kernels"]
    );
    eprintln!(
        "exprprobe: mask-column mapping verified for {} of {} components; \
         {:.4e} M31 multiplies and {:.4e} column bytes per proof",
        report["mask_map_ok"], report["components"],
        report["totals"]["m31_mul"].as_f64().unwrap_or(0.0),
        report["totals"]["column_bytes"].as_f64().unwrap_or(0.0),
    );
    for r in report["rows"].as_array().unwrap() {
        if r["ok"] != json!(true) || r["agrees_with_info"] != json!(true) {
            eprintln!(
                "  !! {} ok={} agrees={} panic={}",
                r["name"], r["ok"], r["agrees_with_info"], r["panic"]
            );
        }
    }
    crate::util::emit("exprprobe", json!({"source": l.source, "probe": report,
                                    "params": serde_json::to_value(&params).unwrap()}));
}
