//! The requantize and the per-token scale argument — the two largest unmeasured
//! per-token terms left after B6 (B4 §8 named both; the plan prices them here,
//! before B7 reads any gap).
//!
//! The statement, per token row of `n` bf16 values `x` (committed by their
//! producer as sign/exponent/mantissa fields):
//!
//!   scale     s_x = ceil16(max_i |x_i| / 127)        (bf16_ceil, or pow2_ceil)
//!   requant   q_i = round_hafz(x_i / s_x),   |q_i| <= 127
//!
//! # The construction
//!
//! * **The max argument** commits the max's magnitude *pattern* `pm` per token.
//!   bf16 magnitude order is pattern order, so `pm - pat_i` in `[0, 2^15)` is one
//!   range lookup per element, and a committed selector bit with
//!   `sel_i (pm - pat_i) = 0`, `sum_i sel_i = 1` pins `pm` to an attained value.
//!   One `(pm, scale)` lookup per token then pins the committed scale to the
//!   ceil-divide of the max — the table *is* `bf16_ceil(v/127)`.
//! * **The requantize** splits on `nz = [q != 0]`, pinned by B6.5's `(q, nz)`
//!   pair table:
//!   - `nz = 1`: one lookup `(tt, [ms,] m7, q)` into a division table indexed by
//!     the exponent difference `tt` (degree-1 in committed fields) and the
//!     mantissas — quotients in `[1/2, 127.5)` span only 9 exponent steps, so the
//!     table is 2^11 rows for a power-of-two scale and 2^18 for a bf16 one.
//!   - `nz = 0`: `|x| < s_x/2` is a *pattern* comparison against `s_x/2`'s
//!     pattern (degree-1 in the scale fields): one 15-bit range lookup, exact for
//!     zeros and subnormals alike.
//!
//! Both lookups ride at multiplicity `nz` / `1 - nz`, so exactly one applies per
//! element, and the clamp incompleteness sits where B1's does (`s_x = ceil` means
//! no honest clamp).

use num_traits::{One, Zero};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, ORIGINAL_TRACE_IDX, Relation, RelationEntry,
    TraceLocationAllocator,
};

use crate::attn_air::RangeRel;
use crate::bf16;
use crate::common::{Eval, Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::dequant::{FracRel, MulRel, PairRel, TableEval, TableKind, id};
use crate::layer::MatLayout;

pub fn m31_of(v: i64) -> M31 {
    crate::attn_trace::m31_of(v)
}

pub const RQ3_COLS: [&str; 3] = ["b65_rq_t", "b65_rq_m", "b65_rq_q"];
pub const RQ4_COLS: [&str; 4] = ["b65_rq_t", "b65_rq_s", "b65_rq_m", "b65_rq_q"];
pub const SCL2_COLS: [&str; 2] = ["b65_scl_pm", "b65_scl_j"];
pub const SCL3_COLS: [&str; 3] = ["b65_scl_pm", "b65_scl_ms", "b65_scl_es"];
pub const QNZ_COLS: [&str; 2] = ["b65_rqnz_q", "b65_rqnz_nz"];
pub const RANGE15: &str = "b65_range_15";
pub const RQ_SEL: [&str; 3] = ["b65_rq_nofirst", "b65_rq_last", "b65_rq_end"];

/// `tt` offsets: quotients in [1/2, 127.5) need only these exponent windows.
const TT_POW2: i64 = 9; // Delta' = ef - 134 - jx in [-9, -1]  (|x| = (128+m7) 2^(ef-134))
const TT_BF16: i64 = 1; // Delta  = ef - es in [-1, 7]
const TT_RANGE: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SVariant {
    Pow2,
    Bf16,
}

impl SVariant {
    pub fn parse(s: &str) -> Self {
        match s {
            "pow2" => SVariant::Pow2,
            "bf16" => SVariant::Bf16,
            other => panic!("unknown requant variant {other:?}"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            SVariant::Pow2 => "pow2",
            SVariant::Bf16 => "bf16",
        }
    }
}

// -------------------------------------------------------------------- tables ---

/// The division table's rows: `(tt, [ms,] m7) -> q`, only where `q` lands in
/// `[1, 127]` — everything else is the `nz = 0` branch or B1's clamp.
pub fn rq_table(v: SVariant) -> Vec<(u32, u32, u32, u32)> {
    let mut rows = vec![];
    match v {
        SVariant::Pow2 => {
            for tt in 0..TT_RANGE as u32 {
                for m7 in 0..128u32 {
                    let m = 128 + m7 as u64;
                    // q = round_half_up(M * 2^(tt-9)) = (M 2^(tt+1) + 512) / 1024
                    let q = ((m << (tt + 1)) + 512) / 1024;
                    if (1..=127).contains(&q) {
                        rows.push((tt, 0, m7, q as u32));
                    }
                }
            }
        }
        SVariant::Bf16 => {
            for tt in 0..TT_RANGE as u32 {
                for ms in 0..128u32 {
                    for m7 in 0..128u32 {
                        let m = 128 + m7 as u64;
                        let sden = 128 + ms as u64;
                        // q = round_half_up(M 2^(tt-1) / S) = (M 2^tt + S) / (2 S)
                        let q = ((m << tt) + sden) / (2 * sden);
                        if (1..=127).contains(&q) {
                            rows.push((tt, ms, m7, q as u32));
                        }
                    }
                }
            }
        }
    }
    rows
}

/// The scale table: `pm -> ceil-divide-by-127`, one row per nonzero magnitude
/// pattern.  For a power-of-two scale the payload is the exponent `j` (as a
/// field element, since real `j` are negative); for bf16 it is `(ms, es)`.
pub fn scl_table(v: SVariant) -> Vec<(u32, M31, u32)> {
    (1u32..1 << 15)
        .map(|pm| {
            let val = bf16::to_f32(pm as u16) as f64 / 127.0;
            match v {
                SVariant::Pow2 => {
                    let j = val.log2().ceil() as i64;
                    (pm, m31_of(j), 0)
                }
                SVariant::Bf16 => {
                    let (_, es, ms) = bf16::decode(bf16_ceil_f64(val));
                    (pm, m31_u32(ms), es)
                }
            }
        })
        .collect()
}

/// Smallest bf16 >= v, v > 0 — `quant.py`'s `bf16_ceil`, in Rust.
pub fn bf16_ceil_f64(v: f64) -> u16 {
    let b = bf16::from_f64(v);
    if (bf16::to_f32(b) as f64) < v { b + 1 } else { b }
}

// -------------------------------------------------------------------- the AIR --

#[derive(Clone)]
pub struct RequantEval {
    pub log_rows: u32,
    pub per_row: usize,
    pub chunks: usize,
    pub v: SVariant,
    pub with_max: bool,
    pub qnz: PairRel,
    pub rq3: FracRel,
    pub rq4: MulRel,
    pub scl2: PairRel,
    pub scl3: FracRel,
    pub r15: RangeRel,
}

impl FrameworkEval for RequantEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_requant_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> =
            RQ_SEL.iter().map(|n| eval.get_preprocessed_column(id(*n))).collect();
        let (nofirst, last, _end) = (sel[0].clone(), sel[1].clone(), sel[2].clone());
        let c128 = M31::from_u32_unchecked(128);

        // shared per-row columns; pm and the scale must agree across a token's rows
        let (pm, pm_prev) = if self.with_max {
            let [p, c] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
            (c, p)
        } else {
            (E::F::zero(), E::F::zero())
        };
        if self.with_max {
            eval.add_constraint((pm.clone() - pm_prev) * nofirst.clone());
        }
        let n_scale = if self.v == SVariant::Pow2 { 1 } else { 2 };
        let mut scale = vec![];
        for _ in 0..n_scale {
            let [p, c] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
            eval.add_constraint((c.clone() - p) * nofirst.clone());
            scale.push(c);
        }
        // the token's sub-half threshold pattern: s_x/2's magnitude pattern
        let path: E::F = match self.v {
            SVariant::Pow2 => (scale[0].clone() + E::F::from(M31::from_u32_unchecked(126)))
                * c128,
            SVariant::Bf16 => (scale[1].clone() - E::F::one()) * c128 + scale[0].clone(),
        };

        let mut sels: Vec<E::F> = vec![];
        for _ in 0..self.per_row {
            let ef = eval.next_trace_mask();
            let m7 = eval.next_trace_mask();
            let q = eval.next_trace_mask();
            let nz = eval.next_trace_mask();
            let pat = ef.clone() * c128 + m7.clone();
            // (q, nz) pins nz = [q != 0]
            eval.add_to_relation(RelationEntry::new(&self.qnz, E::EF::one(),
                                                    &[q.clone(), nz.clone()]));
            // nz = 1: the division table, keyed by the exponent difference
            match self.v {
                SVariant::Pow2 => {
                    let tt = ef.clone() - scale[0].clone()
                        - E::F::from(M31::from_u32_unchecked((134 - TT_POW2) as u32));
                    eval.add_to_relation(RelationEntry::new(
                        &self.rq3, E::EF::from(nz.clone()), &[tt, m7.clone(), q]));
                }
                SVariant::Bf16 => {
                    let tt = ef.clone() - scale[1].clone()
                        + E::F::from(M31::from_u32_unchecked(TT_BF16 as u32));
                    eval.add_to_relation(RelationEntry::new(
                        &self.rq4, E::EF::from(nz.clone()),
                        &[tt, scale[0].clone(), m7.clone(), q]));
                }
            }
            // nz = 0: |x| < s_x/2, a pattern comparison
            eval.add_to_relation(RelationEntry::new(
                &self.r15, E::EF::one() - E::EF::from(nz.clone()),
                &[path.clone() - E::F::one() - pat.clone()]));
            if self.with_max {
                let s = eval.next_trace_mask();
                eval.add_constraint(s.clone() * s.clone() - s.clone());
                eval.add_constraint(s.clone() * (pm.clone() - pat.clone()));
                eval.add_to_relation(RelationEntry::new(
                    &self.r15, E::EF::one(), &[pm.clone() - pat]));
                sels.push(s);
            }
        }
        if self.with_max {
            // sum of selectors over the token = 1, via a plain running sum
            let [ss_prev, ss] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
            let mut f = E::F::zero();
            for s in sels {
                f = f + s;
            }
            eval.add_constraint(ss.clone() - ss_prev * nofirst - f);
            eval.add_constraint((ss - E::F::one()) * last.clone());
            // the scale is the ceil-divide of the attained max, once per token
            match self.v {
                SVariant::Pow2 => eval.add_to_relation(RelationEntry::new(
                    &self.scl2, E::EF::from(last), &[pm, scale[0].clone()])),
                SVariant::Bf16 => eval.add_to_relation(RelationEntry::new(
                    &self.scl3, E::EF::from(last),
                    &[pm, scale[0].clone(), scale[1].clone()])),
            }
        }
        eval.finalize_logup_in_pairs();
        eval
    }
}

// ------------------------------------------------------------------ witness ----

#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// a different quotient, still in [1, 127]
    WrongQ(usize),
    /// nz flipped on one element
    FlipNz(usize),
    /// the committed scale one step too big on one token
    ScaleUp(usize),
    /// pm below the true max: some pm - pat goes negative
    PmLow(usize),
    /// pm above the true max: no selector can attain it (the prover refuses)
    PmHigh(usize),
}

pub struct RWitness {
    /// (tokens, n) bf16 magnitude patterns (sign stripped; requant is sign-free)
    pub x: Vec<u16>,
    pub tokens: usize,
    pub n: usize,
    /// per-token scale: pow2 exponent, and bf16 (ms, es)
    pub jx: Vec<i64>,
    pub sx: Vec<(u32, u32)>,
    /// the reference: B1's committed quantized activations (magnitudes)
    pub q_ref: Vec<u32>,
    pub source: String,
}

impl RWitness {
    pub fn load(dir: &str, name: &str, n_cap: usize) -> Result<Self, String> {
        let raw = std::fs::read(format!("{dir}/x_{name}.u16"))
            .map_err(|e| format!("x_{name}.u16: {e}"))?;
        let x_all: Vec<u16> =
            raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff).collect();
        let qraw = std::fs::read(format!("{dir}/qx_{name}.i8"))
            .map_err(|e| format!("qx_{name}.i8: {e}"))?;
        let man: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json"))
                .map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let am = &man["accumulators"][name];
        let n = am["n_in"].as_u64().ok_or("manifest: n_in")? as usize;
        let mut tokens = am["tokens"].as_u64().ok_or("manifest: tokens")? as usize;
        if n_cap > 0 {
            // capped runs keep a power-of-two token count so the region tiles
            tokens = tokens.min(n_cap.div_ceil(n).max(1));
            tokens = 1 << tokens.ilog2();
        }
        let jraw: Vec<i64> = std::fs::read(format!("{dir}/jx_{name}.i32"))
            .map_err(|e| e.to_string())?
            .chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
            .collect();
        let sraw: Vec<i64> = std::fs::read(format!("{dir}/sx_{name}.i32"))
            .map_err(|e| e.to_string())?
            .chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
            .collect();
        Ok(Self {
            x: x_all[..tokens * n].to_vec(),
            tokens,
            n,
            jx: jraw[..tokens].to_vec(),
            sx: (0..tokens).map(|t| (sraw[2 * t] as u32, sraw[2 * t + 1] as u32)).collect(),
            q_ref: qraw[..tokens * n].iter().map(|b| (*b as i8).unsigned_abs() as u32).collect(),
            source: format!("B1 {name} inputs [{} tokens x {n}]", tokens),
        })
    }

    pub fn synthetic(tokens: usize, n: usize, v: SVariant, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut x = vec![0u16; tokens * n];
        let mut jx = vec![0i64; tokens];
        let mut sx = vec![(0u32, 0u32); tokens];
        let mut q_ref = vec![0u32; tokens * n];
        for t in 0..tokens {
            // the max is over the *stored* bf16 values, exactly as `qscale` sees it
            let vals: Vec<f64> = (0..n)
                .map(|_| {
                    let g: f64 = (0..3).map(|_| rng.random_range(-1.0f64..1.0)).sum();
                    bf16::to_f32(bf16::from_f64((g * 3.0).abs())) as f64
                })
                .collect();
            let m = vals.iter().fold(0f64, |a, b| a.max(*b)).max(1e-6);
            let (s_val, j, ms_es) = match v {
                SVariant::Pow2 => {
                    let j = (m / 127.0).log2().ceil() as i64;
                    ((j as f64).exp2(), j, (0, 0))
                }
                SVariant::Bf16 => {
                    let b = bf16_ceil_f64(m / 127.0);
                    let (_, es, ms) = bf16::decode(b);
                    (bf16::to_f32(b) as f64, 0, (ms, es))
                }
            };
            jx[t] = j;
            sx[t] = ms_es;
            for i in 0..n {
                let xb = bf16::from_f64(vals[i].abs());
                x[t * n + i] = xb;
                let xv = bf16::to_f32(xb) as f64;
                q_ref[t * n + i] = (xv / s_val + 0.5).floor() as u32;
            }
        }
        Self { x, tokens, n, jx, sx, q_ref, source: format!("synthetic(seed={seed})") }
    }
}

// ------------------------------------------------------------------- the run ---

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub n: usize,
    pub off_table: usize,
    pub ref_ok: usize,
    pub ref_n: usize,
    pub table_cells_pre: usize,
    pub table_cells_fresh: usize,
    pub trace_secs: f64,
    pub interaction_secs: f64,
    pub n_columns: [usize; 3],
    pub with_max: bool,
    pub v: SVariant,
    pub per_row: usize,
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0
    }
    /// Cells the gadget adds per element: the x fields (2) belong to the
    /// producer, the tables amortise.
    pub fn marginal(&self) -> f64 {
        (self.proven.fresh_cells() - self.table_cells_fresh) as f64 / self.n as f64 - 2.0
    }
    pub fn json(&self, src: &str) -> Value {
        let p = &self.proven;
        json!({
            "variant": self.v.name(), "with_max": self.with_max, "source": src,
            "n": self.n, "per_row": self.per_row,
            "cells_per_value_marginal": self.marginal(),
            "cells_per_value_with_tables":
                (p.fresh_cells() + self.table_cells_pre) as f64 / self.n as f64 - 2.0,
            "table_cells_preprocessed": self.table_cells_pre,
            "table_cells_fresh": self.table_cells_fresh,
            "values_per_sec": self.n as f64 / p.prove_secs,
            "off_table": self.off_table,
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "trace_secs": self.trace_secs, "interaction_secs": self.interaction_secs,
            "sums": self.sums.json(), "accepts": self.accepts(),
            "point": p.json(),
        })
    }
}

pub fn run(w: &RWitness, v: SVariant, with_max: bool, per_row_target: usize,
           tamper: Option<Tamper>) -> Outcome {
    let t0 = std::time::Instant::now();
    let lay = MatLayout::choose(w.n, w.tokens, per_row_target);
    let pr = lay.per_row;
    let rows = 1usize << lay.log_rows;
    let n = w.tokens * w.n;

    // ---- tables ---------------------------------------------------------------
    let rq = rq_table(v);
    let log_rq = crate::common::log_rows_for(rq.len(), 1);
    let scl = scl_table(v);
    let log_scl = 15u32;
    let rq_at = |tt: i64, ms: u32, m7: u32| -> Option<usize> {
        if !(0..TT_RANGE as i64).contains(&tt) {
            return None;
        }
        rq.binary_search_by_key(&(tt as u32, ms, m7), |r| (r.0, r.1, r.2)).ok()
    };

    // ---- the witness ------------------------------------------------------------
    let n_scale = if v == SVariant::Pow2 { 1 } else { 2 };
    let per_slot = 4 + usize::from(with_max); // ef, m7, q, nz [, sel]
    // column order = the AIR's mask order: pm, scale, slots, then ss last
    let col_pm = 0usize;
    let col_scale = usize::from(with_max);
    let slot0 = col_scale + n_scale;
    let col_ss = slot0 + pr * per_slot;
    let n_cols = col_ss + usize::from(with_max);
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut m_qnz = vec![0u32; 128];
    let mut m_rq = vec![0u32; 1 << log_rq];
    let mut m_scl = vec![0u32; 1 << log_scl];
    let mut m_r15 = vec![0u32; 1 << 15];
    let mut off_table = 0usize;
    let (mut ref_ok, mut ref_n) = (0usize, 0usize);

    // per-token: the max pattern, the scale, the argmax slot
    let mut pms = vec![0u16; w.tokens];
    let mut argmax = vec![0usize; w.tokens];
    for t in 0..w.tokens {
        let row = &w.x[t * w.n..(t + 1) * w.n];
        let (mut best, mut bi) = (0u16, 0usize);
        for (i, p) in row.iter().enumerate() {
            if *p > best {
                best = *p;
                bi = i;
            }
        }
        pms[t] = best;
        argmax[t] = bi;
        match tamper {
            Some(Tamper::PmLow(tt)) if tt == t => pms[t] -= 1,
            Some(Tamper::PmHigh(tt)) if tt == t => pms[t] += 1,
            _ => {}
        }
    }
    let mut jx = w.jx.clone();
    let mut sx = w.sx.clone();
    if let Some(Tamper::ScaleUp(t)) = tamper {
        match v {
            SVariant::Pow2 => jx[t] += 1,
            SVariant::Bf16 => {
                let b = bf16::pattern(0, sx[t].1, sx[t].0) + 1;
                let (_, es, ms) = bf16::decode(b);
                sx[t] = (ms, es);
            }
        }
    }

    let mut ss_carry = 0u32;
    for j in 0..rows {
        let (t, b) = lay.at(w.tokens, j);
        if b == lay.chunks - 1 {
            ss_carry = 0; // token's first row
        }
        if with_max {
            cols[col_pm][j] = m31_u32(pms[t] as u32);
        }
        match v {
            SVariant::Pow2 => cols[col_scale][j] = m31_of(jx[t]),
            SVariant::Bf16 => {
                cols[col_scale][j] = m31_u32(sx[t].0);
                cols[col_scale + 1][j] = m31_u32(sx[t].1);
            }
        }
        // the sub-half threshold pattern of this token
        let path: i64 = match v {
            SVariant::Pow2 => (jx[t] + 126) * 128,
            SVariant::Bf16 => (sx[t].1 as i64 - 1) * 128 + sx[t].0 as i64,
        };
        for sl in 0..pr {
            let i = b * pr + sl;
            let gi = t * w.n + i;
            let pat = w.x[gi] as i64;
            let (ef, m7) = ((pat >> 7) as u32, (pat & 0x7f) as u32);
            // the branch, decided exactly by the pattern comparison
            let nz = pat >= path;
            let (mut q, mut nzc) = (0u32, u32::from(nz));
            if nz {
                let (tt, ms) = match v {
                    SVariant::Pow2 => (ef as i64 - jx[t] - (134 - TT_POW2), 0u32),
                    SVariant::Bf16 => (ef as i64 - sx[t].1 as i64 + TT_BF16, sx[t].0),
                };
                match rq_at(tt, ms, m7) {
                    Some(ri) => {
                        q = rq[ri].3;
                        m_rq[ri] += 1;
                    }
                    None => off_table += 1,
                }
            } else {
                let d = path - 1 - pat;
                if (0..1 << 15).contains(&d) {
                    m_r15[d as usize] += 1;
                } else {
                    off_table += 1;
                }
            }
            match tamper {
                Some(Tamper::WrongQ(x)) if x == gi => {
                    q = if q > 1 { q - 1 } else { q + 1 };
                }
                Some(Tamper::FlipNz(x)) if x == gi => nzc ^= 1,
                _ => {}
            }
            if (q as usize) < m_qnz.len() {
                m_qnz[q as usize] += 1;
            }
            let c0 = slot0 + sl * per_slot;
            cols[c0][j] = m31_u32(ef);
            cols[c0 + 1][j] = m31_u32(m7);
            cols[c0 + 2][j] = m31_u32(q);
            cols[c0 + 3][j] = m31_u32(nzc);
            if with_max {
                let is_max = i == argmax[t] && !matches!(tamper, Some(Tamper::PmHigh(x)) if x == t);
                cols[c0 + 4][j] = m31_u32(is_max as u32);
                ss_carry += is_max as u32;
                let d = pms[t] as i64 - pat;
                if (0..1 << 15).contains(&d) {
                    m_r15[d as usize] += 1;
                } else {
                    off_table += 1;
                }
            }
            ref_n += 1;
            if q == w.q_ref[gi] {
                ref_ok += 1;
            }
        }
        if with_max {
            cols[col_ss][j] = m31_u32(ss_carry);
            if b == 0 {
                // token's last row: the scale lookup
                let pi = pms[t] as usize;
                if pi >= 1 {
                    let want = match v {
                        SVariant::Pow2 => scl[pi - 1].1 == m31_of(jx[t]),
                        SVariant::Bf16 => scl[pi - 1].1 == m31_u32(sx[t].0)
                            && scl[pi - 1].2 == sx[t].1,
                    };
                    if want {
                        m_scl[pi - 1] += 1;
                    } else {
                        off_table += 1;
                    }
                } else {
                    off_table += 1;
                }
            }
        }
    }
    let trace_secs = t0.elapsed().as_secs_f64();

    // ---- tree 0 ------------------------------------------------------------------
    let mut pre_ids: Vec<PreProcessedColumnId> = vec![];
    let mut pre: Vec<Eval> = vec![];
    let rq_cols: Vec<&str> = if v == SVariant::Pow2 { RQ3_COLS.to_vec() } else { RQ4_COLS.to_vec() };
    for (k, name) in rq_cols.iter().enumerate() {
        pre_ids.push(id(*name));
        pre.push(eval_of(log_rq, (0..1usize << log_rq).map(|i| {
            let r = rq.get(i).copied().unwrap_or(rq[0]);
            let vals4 = [r.0, r.1, r.2, r.3];
            let vals3 = [r.0, r.2, r.3];
            m31_u32(if v == SVariant::Pow2 { vals3[k] } else { vals4[k] })
        }).collect()));
    }
    let scl_cols: Vec<&str> =
        if v == SVariant::Pow2 { SCL2_COLS.to_vec() } else { SCL3_COLS.to_vec() };
    if with_max {
        for (k, name) in scl_cols.iter().enumerate() {
            pre_ids.push(id(*name));
            pre.push(eval_of(log_scl, (0..1usize << log_scl).map(|i| {
                let r = scl.get(i).copied().unwrap_or(scl[0]);
                match k {
                    0 => m31_u32(r.0),
                    1 => r.1,
                    _ => m31_u32(r.2),
                }
            }).collect()));
        }
    }
    for (k, name) in QNZ_COLS.iter().enumerate() {
        pre_ids.push(id(*name));
        pre.push(eval_of(7, (0..128u32)
            .map(|q| m31_u32(if k == 0 { q } else { u32::from(q != 0) })).collect()));
    }
    pre_ids.push(id(RANGE15));
    pre.push(eval_of(15, (0..1u32 << 15).map(m31_u32).collect()));
    for name in RQ_SEL {
        pre_ids.push(id(name));
    }
    let lastf = |j: usize| j % lay.chunks == lay.chunks - 1;
    pre.push(eval_of(lay.log_rows, (0..rows).map(|j| m31_u32((j % lay.chunks != 0) as u32)).collect()));
    pre.push(eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(lastf(j) as u32)).collect()));
    pre.push(eval_of(lay.log_rows, (0..rows).map(|j| m31_u32((j + 1 == rows) as u32)).collect()));
    // RQ_SEL ids were pushed before the selector evals; reorder ids to match
    // (ids: rq table, scl table, qnz, range15, then the three selectors -- the
    // evals were pushed in the same order, so nothing to do)

    // ---- tree 1 -------------------------------------------------------------------
    let mut base: Vec<Eval> = cols.iter().map(|c| eval_of(lay.log_rows, c.clone())).collect();
    let mult0 = base.len();
    base.push(eval_of(log_rq, m_rq.iter().map(|m| m31_u32(*m)).collect()));
    if with_max {
        base.push(eval_of(log_scl, m_scl.iter().map(|m| m31_u32(*m)).collect()));
    }
    base.push(eval_of(7, m_qnz.iter().map(|m| m31_u32(*m)).collect()));
    base.push(eval_of(15, m_r15.iter().map(|m| m31_u32(*m)).collect()));

    // ---- session ---------------------------------------------------------------------
    let max_log = lay.log_rows.max(log_rq).max(15).max(if with_max { log_scl } else { 0 });
    let tw = twiddles(max_log);
    let mut sess = Session::new(&tw);
    sess.commit(pre.clone());
    sess.commit(base.clone());
    let qnz = sess.draw_relation(PairRel::draw);
    let rq3 = sess.draw_relation(FracRel::draw);
    let rq4 = sess.draw_relation(MulRel::draw);
    let scl2 = sess.draw_relation(PairRel::draw);
    let scl3 = sess.draw_relation(FracRel::draw);
    let r15 = sess.draw_relation(RangeRel::draw);

    // ---- interaction: the exact lookup order of the AIR ----------------------------
    let t1 = std::time::Instant::now();
    let n_looks_slot = 3 + usize::from(with_max);
    let n_looks = pr * n_looks_slot + usize::from(with_max);
    let one = PackedQM31::broadcast(SecureField::one());
    let getb = |c: usize, vr: usize| base[c].data[vr];
    let sel_last = pre[pre.len() - 2].clone();
    let c141 = PackedM31::broadcast(m31_u32((134 - TT_POW2) as u32));
    let ctt1 = PackedM31::broadcast(m31_u32(TT_BF16 as u32));
    let c128p = PackedM31::broadcast(m31_u32(128));
    let c126 = PackedM31::broadcast(m31_u32(126));
    let onep = PackedM31::broadcast(M31::one());
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        if i == n_looks - 1 && with_max {
            // the per-token scale lookup, at multiplicity `last`
            let m = PackedQM31::from(sel_last.data[vr]);
            let d = match v {
                SVariant::Pow2 => scl2.combine(&[getb(col_pm, vr), getb(col_scale, vr)]),
                SVariant::Bf16 => scl3.combine(&[getb(col_pm, vr), getb(col_scale, vr),
                                                 getb(col_scale + 1, vr)]),
            };
            return (m, d);
        }
        let (sl, which) = (i / n_looks_slot, i % n_looks_slot);
        let c0 = slot0 + sl * per_slot;
        let (ef, m7, q, nz) = (getb(c0, vr), getb(c0 + 1, vr), getb(c0 + 2, vr),
                               getb(c0 + 3, vr));
        let pat = ef * c128p + m7;
        let path = match v {
            SVariant::Pow2 => (getb(col_scale, vr) + c126) * c128p,
            SVariant::Bf16 => (getb(col_scale + 1, vr) - onep) * c128p + getb(col_scale, vr),
        };
        match which {
            0 => (one, qnz.combine(&[q, nz])),
            1 => {
                let d = match v {
                    SVariant::Pow2 => rq3.combine(&[ef - getb(col_scale, vr) - c141, m7, q]),
                    SVariant::Bf16 => rq4.combine(&[ef - getb(col_scale + 1, vr) + ctt1,
                                                    getb(col_scale, vr), m7, q]),
                };
                (PackedQM31::from(nz), d)
            }
            2 => (one - PackedQM31::from(nz), r15.combine(&[path - onep - pat])),
            _ => (one, r15.combine(&[getb(col_pm, vr) - pat])),
        }
    };
    let (n_pairs, fpair) = crate::plogup::paired(n_looks, fr);
    let (mut inter, use_sum) = crate::plogup::build(lay.log_rows, n_pairs, fpair);

    // table sides, in table order: rq, [scl], qnz, range15
    let mut table_sums = vec![];
    {
        let ncol = rq_cols.len();
        let (c, s) = crate::plogup::build(log_rq, 1, |_, vr| {
            let vals: Vec<PackedM31> = (0..ncol).map(|k| pre[k].data[vr]).collect();
            let d = if v == SVariant::Pow2 { rq3.combine(&vals) } else { rq4.combine(&vals) };
            (-PackedQM31::from(base[mult0].data[vr]), d)
        });
        inter.extend(c);
        table_sums.push(s);
        let mut pi = ncol;
        let mut mi = mult0 + 1;
        if with_max {
            let nsc = scl_cols.len();
            let (c, s) = crate::plogup::build(log_scl, 1, |_, vr| {
                let vals: Vec<PackedM31> = (0..nsc).map(|k| pre[pi + k].data[vr]).collect();
                let d = if v == SVariant::Pow2 { scl2.combine(&vals) } else { scl3.combine(&vals) };
                (-PackedQM31::from(base[mi].data[vr]), d)
            });
            inter.extend(c);
            table_sums.push(s);
            pi += nsc;
            mi += 1;
        }
        let (c, s) = crate::plogup::build(7, 1, |_, vr| {
            (-PackedQM31::from(base[mi].data[vr]),
             qnz.combine(&[pre[pi].data[vr], pre[pi + 1].data[vr]]))
        });
        inter.extend(c);
        table_sums.push(s);
        let (c, s) = crate::plogup::build(15, 1, |_, vr| {
            (-PackedQM31::from(base[mi + 1].data[vr]), r15.combine(&[pre[pi + 2].data[vr]]))
        });
        inter.extend(c);
        table_sums.push(s);
    }
    let interaction_secs = t1.elapsed().as_secs_f64();
    let n_inter = inter.len();
    sess.commit(inter);

    // ---- components ---------------------------------------------------------------
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&pre_ids);
    let use_comp = FrameworkComponent::<RequantEval>::new(
        alloc,
        RequantEval { log_rows: lay.log_rows, per_row: pr, chunks: lay.chunks, v, with_max,
                      qnz: qnz.clone(), rq3: rq3.clone(), rq4: rq4.clone(),
                      scl2: scl2.clone(), scl3: scl3.clone(), r15: r15.clone() },
        use_sum,
    );
    let mut tbl_comps: Vec<FrameworkComponent<TableEval>> = vec![];
    let alloc_tbl = |lr: u32, colids: Vec<PreProcessedColumnId>, kind: TableKind,
                         s: SecureField, tbl_comps: &mut Vec<FrameworkComponent<TableEval>>,
                         alloc: &mut TraceLocationAllocator| {
        tbl_comps.push(FrameworkComponent::<TableEval>::new(
            alloc, TableEval { log_rows: lr, cols: colids, kind }, s));
    };
    let mut ti = 0;
    alloc_tbl(log_rq, rq_cols.iter().map(|c| id(*c)).collect(),
              if v == SVariant::Pow2 { TableKind::Frac(rq3.clone()) }
              else { TableKind::Mul(rq4.clone()) },
              table_sums[ti], &mut tbl_comps, alloc);
    ti += 1;
    if with_max {
        alloc_tbl(log_scl, scl_cols.iter().map(|c| id(*c)).collect(),
                  if v == SVariant::Pow2 { TableKind::Pair(scl2.clone()) }
                  else { TableKind::Frac(scl3.clone()) },
                  table_sums[ti], &mut tbl_comps, alloc);
        ti += 1;
    }
    alloc_tbl(7, QNZ_COLS.iter().map(|c| id(*c)).collect(), TableKind::Pair(qnz.clone()),
              table_sums[ti], &mut tbl_comps, alloc);
    alloc_tbl(15, vec![id(RANGE15)], TableKind::Range(r15.clone()),
              table_sums[ti + 1], &mut tbl_comps, alloc);

    let mut provers: Vec<&dyn ComponentProver<_>> = vec![&use_comp];
    let mut components: Vec<&dyn Component> = vec![&use_comp];
    for c in &tbl_comps {
        provers.push(c);
        components.push(c);
    }
    let n_cols_out = [pre.len(), base.len(), n_inter];
    let proven = sess.prove(&provers, &components);

    let table_cells_pre = (rq_cols.len() << log_rq)
        + if with_max { scl_cols.len() << log_scl } else { 0 }
        + (2 << 7) + (1 << 15);
    let table_cells_fresh = (5usize << log_rq)
        + if with_max { 5 << log_scl } else { 0 }
        + (5 << 7) + (5 << 15);
    Outcome {
        proven,
        sums: Sums { uses: use_sum,
                     tables: table_sums.iter().fold(SecureField::zero(), |a, b| a + *b) },
        n,
        off_table,
        ref_ok,
        ref_n,
        table_cells_pre,
        table_cells_fresh,
        trace_secs,
        interaction_secs,
        n_columns: n_cols_out,
        with_max,
        v,
        per_row: pr,
    }
}

/// Check the two tables against an f64 reference, `check_tables`-style.
pub fn check_tables(v: SVariant, samples: usize) -> Value {
    let rq = rq_table(v);
    let mut bad = vec![];
    let step = (rq.len() / samples.max(1)).max(1);
    let mut tested = 0usize;
    for r in rq.iter().step_by(step) {
        tested += 1;
        let (tt, ms, m7, q) = *r;
        let m = (128 + m7) as f64;
        let quot = match v {
            SVariant::Pow2 => m * ((tt as f64) - TT_POW2 as f64).exp2(),
            SVariant::Bf16 => m / (128 + ms) as f64 * ((tt as f64) - TT_BF16 as f64).exp2(),
        };
        let want = (quot + 0.5).floor() as u32;
        if want != q && bad.len() < 8 {
            bad.push(json!({"tt": tt, "ms": ms, "m7": m7, "table": q, "f64": want}));
        }
    }
    let scl = scl_table(v);
    let mut sbad = vec![];
    for r in scl.iter().step_by((scl.len() / samples.max(1)).max(1)) {
        let val = bf16::to_f32(r.0 as u16) as f64 / 127.0;
        match v {
            SVariant::Pow2 => {
                let j = val.log2().ceil();
                if m31_of(j as i64) != r.1 && sbad.len() < 8 {
                    sbad.push(json!({"pm": r.0}));
                }
            }
            SVariant::Bf16 => {
                let b = bf16_ceil_f64(val);
                let (_, es, ms) = bf16::decode(b);
                if (m31_u32(ms), es) != (r.1, r.2) && sbad.len() < 8 {
                    sbad.push(json!({"pm": r.0}));
                }
            }
        }
    }
    json!({"variant": v.name(), "rq_rows": rq.len(), "rq_tested": tested,
           "rq_mismatch": bad, "scl_rows": scl.len(), "scl_mismatch": sbad})
}
