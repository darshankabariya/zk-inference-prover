//! B10 step 4: the token tie -- "the emitted token is the argmax of the logits".
//!
//! Greedy decoding emits `argmax_v logits[t, v]`, and `torch.argmax` returns the
//! **first** maximal index.  On the real 512-token continuations 15 of 512 rows
//! (C=4K) and 14 of 512 (C=16K) have two or three co-maximal logits
//! (`b10/out/head_ties_*.json`), so the tie-breaking rule decides the token often
//! enough to matter and the statement has to name it.
//!
//! # Two statements, and why the strict one is not the expensive one
//!
//! B6.5's max argument proves "`tok` attains the maximum": commit the maximum's
//! pattern `pm`, range-check `pm - k_i >= 0` on every entry, and a selector with
//! `sel_i (pm - k_i) = 0`, `sum_i sel_i = 1` pins `pm` to an attained value.  On
//! a tied row a prover may point the selector at either co-maximal entry.
//!
//! The strict statement -- "`tok` is the **lowest-indexed** maximum", which is
//! B1's own rule -- looks like it needs an equality indicator pinned in both
//! directions plus a separate "nothing earlier attained it" argument.  It does
//! not.  Replace the selector by a **monotone** indicator
//!
//!   `b_i = [i < tok]`,   non-increasing in `i`,   `sum_i b_i = tok`
//!
//! and range-check `pm - k_i - b_i >= 0` instead of `pm - k_i >= 0`.  Because
//! `b_i` is 0/1 that is the *same* lookup on a shifted argument, so:
//!
//!   * `i < tok`  (b = 1):  `pm - k_i >= 1`, i.e. **strictly** below the maximum;
//!   * `i >= tok` (b = 0):  `pm - k_i >= 0`, i.e. at most the maximum.
//!
//! The selector is then *derived*, `sel_i = b_i - b_{i+1}`, and needs no column
//! of its own; `sel_tok = 1` and all others are 0 by telescoping, so
//! `sel_i (pm - k_i) = 0` still says the maximum is attained -- at `tok`.  The
//! running sum closes against the committed token instead of against 1, which is
//! the tie itself and costs nothing extra.
//!
//! So the strict statement is the lax one with `s` renamed to `b`, one extra
//! degree-2 constraint (monotonicity), and a different closing value.  `--lax`
//! proves the weaker statement so the two can be measured against each other
//! rather than argued about.
//!
//! # What this region does *not* do
//!
//! `k_i` is the committed **order key** of the logit: bf16 magnitude order is
//! pattern order, but logits are signed, so the key is the sign-flipped pattern
//! (`k = s ? 0x7FFF - mag : 0x8000 + mag`), which is a total order matching the
//! float order.  Pinning `k` to the `lm_head` dequant's own output is a
//! positionwise tie of the kind B10 step 2 built for the glue, and it is the
//! composition's job, not this region's.  Standalone, `k` is the statement's
//! input.

use num_traits::{One, Zero};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::m31::M31;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, ORIGINAL_TRACE_IDX, Relation, RelationEntry,
    TraceLocationAllocator,
};

use crate::attn_air::{RangeRel, RangeTableEval, tbl_range};
use crate::common::{Eval, Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::layer::MatLayout;

/// The order key's width: a bf16 sign-flipped pattern is 16 bits.
pub const KEY_BITS: u32 = 16;

pub const AM_SEL: [&str; 3] = ["b10_am_nofirst", "b10_am_last", "b10_am_end"];

pub fn id(s: &str) -> PreProcessedColumnId {
    PreProcessedColumnId { id: s.to_string() }
}

/// The order key of a signed bf16 pattern: negatives descend, positives ascend,
/// and every key is 16 bits.
pub fn order_key(pat: u16) -> u32 {
    let (sign, mag) = (pat >> 15, pat & 0x7FFF);
    if sign == 1 { (0x7FFF - mag) as u32 } else { 0x8000 + mag as u32 }
}

// -------------------------------------------------------------------- the AIR --

#[derive(Clone)]
pub struct ArgmaxEval {
    pub log_rows: u32,
    pub per_row: usize,
    /// prove only "the token attains the maximum" -- B6.5's argument unchanged
    pub lax: bool,
    pub rk: RangeRel,
}

impl FrameworkEval for ArgmaxEval {
    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_argmax_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> = AM_SEL.iter().map(|n| eval.get_preprocessed_column(id(n))).collect();
        let (nofirst, last, end) = (sel[0].clone(), sel[1].clone(), sel[2].clone());

        // shared per-token columns: the maximum's order key, and the token id.
        // Both are constant across the token's rows.
        let [pm_prev, pm] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        eval.add_constraint((pm.clone() - pm_prev) * nofirst.clone());
        let [tk_prev, tk] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        eval.add_constraint((tk.clone() - tk_prev) * nofirst.clone());

        // the keys, and the indicator.  The indicator is read with a `[-1, 0]`
        // mask because the monotonicity check straddles a row boundary: the walk
        // visits the vocabulary ascending, so index `i - 1` for a row's slot 0 is
        // the previous row's last slot.
        let mut ks = vec![];
        let mut bs = vec![];
        let mut bs_prev = vec![];
        for _ in 0..self.per_row {
            ks.push(eval.next_trace_mask());
        }
        for _ in 0..self.per_row {
            let [p, c] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
            bs.push(c);
            bs_prev.push(p);
        }

        for i in 0..self.per_row {
            let b = bs[i].clone();
            eval.add_constraint(b.clone() * b.clone() - b.clone());
            // `pm - k - b` in [0, 2^16): b = 1 makes it strict, which is what
            // "nothing before `tok` attained the maximum" means
            let arg = if self.lax {
                pm.clone() - ks[i].clone()
            } else {
                pm.clone() - ks[i].clone() - b.clone()
            };
            eval.add_to_relation(RelationEntry::new(&self.rk, E::EF::one(), &[arg]));
        }

        // Monotonicity, and the derived selector.  The walk visits the vocabulary
        // index *ascending* (the trace's map is `i = (chunks - 1 - b) * per_row +
        // slot`), so index `i - 1` is the previous slot within the row and, for
        // slot 0, the previous row's last slot -- which is exactly what a
        // `[-1, 0]` mask reaches.  Looking backwards is what puts the selector on
        // `tok` rather than on `tok - 1`:
        //
        //   sel_i = b_{i-1} - b_i = [i-1 < tok] - [i < tok] = [i == tok]
        //
        // with `b_{-1} = 1` by convention, which is the token's first row.
        let mut sels: Vec<E::F> = vec![];
        for i in 0..self.per_row {
            let prev = if i > 0 {
                bs[i - 1].clone()
            } else {
                // b_{-1} = 1 before the vocabulary starts
                bs_prev[self.per_row - 1].clone() * nofirst.clone()
                    + (E::F::one() - nofirst.clone())
            };
            if !self.lax {
                // b_i = 1 implies b_{i-1} = 1
                eval.add_constraint(bs[i].clone() * (E::F::one() - prev.clone()));
            }
            let s = if self.lax { bs[i].clone() } else { prev - bs[i].clone() };
            eval.add_constraint(s.clone() * (pm.clone() - ks[i].clone()));
            sels.push(s);
        }

        // the running sum: `sum_i b_i`, closed at the token's last row.  Lax closes
        // it against 1 (a selector was placed somewhere); strict closes it against
        // the committed token, which *is* the tie.
        let [ss_prev, ss] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        let mut f = E::F::zero();
        for s in if self.lax { &sels } else { &bs } {
            f = f + s.clone();
        }
        eval.add_constraint(ss.clone() - ss_prev * nofirst - f);
        let want = if self.lax { E::F::one() } else { tk };
        eval.add_constraint((ss - want) * last);
        let _ = end;

        eval.finalize_logup_in_pairs();
        eval
    }
}

// ------------------------------------------------------------------ witness ----

#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// point the token at a different entry that also attains the maximum -- the
    /// probe the *lax* statement cannot reject and the strict one must
    Tie(usize),
    /// name a token whose logit is not the maximum at all
    NotMax(usize),
    /// claim a maximum above every entry: attained by none
    PmHigh(usize),
    /// claim a maximum below the true one: some entry exceeds it
    PmLow(usize),
}

pub struct AWitness {
    pub tokens: usize,
    pub vocab: usize,
    /// (tokens, vocab) order keys
    pub key: Vec<u32>,
    /// the emitted token per row, and the maximum's key
    pub tok: Vec<u32>,
    pub pm: Vec<u32>,
    pub source: String,
    /// rows where more than one entry attains the maximum
    pub ties: usize,
}

impl AWitness {
    /// A synthetic block with a controlled tie rate, so the strict statement has
    /// something to bite on.  `tie_frac` of the rows get a second entry at the
    /// maximum, placed *after* the first so the lowest-index rule is what picks.
    pub fn synthetic(tokens: usize, vocab: usize, tie_frac: f64, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut key = vec![0u32; tokens * vocab];
        for k in key.iter_mut() {
            *k = rng.random_range(0..(1u32 << KEY_BITS) - 1);
        }
        let (mut tok, mut pm, mut ties) = (vec![0u32; tokens], vec![0u32; tokens], 0usize);
        for t in 0..tokens {
            let row = &mut key[t * vocab..(t + 1) * vocab];
            let mut best = 0usize;
            for i in 1..vocab {
                if row[i] > row[best] {
                    best = i;
                }
            }
            if rng.random::<f64>() < tie_frac && best + 1 < vocab {
                // a co-maximal entry strictly after the winner: the lax statement
                // would accept either, the strict one only the first
                let j = rng.random_range(best + 1..vocab);
                row[j] = row[best];
                ties += 1;
            }
            // the argmax is the *first* attained entry, as torch's is
            let mx = *row.iter().max().unwrap();
            let first = row.iter().position(|v| *v == mx).unwrap();
            tok[t] = first as u32;
            pm[t] = mx;
        }
        Self { tokens, vocab, key, tok, pm, ties,
               source: format!("synthetic(seed={seed}, ties={tie_frac})") }
    }

    fn tampered(&self, bad: Option<Tamper>) -> (Vec<u32>, Vec<u32>) {
        let (mut tok, mut pm) = (self.tok.clone(), self.pm.clone());
        match bad {
            Some(Tamper::Tie(t)) => {
                // move the token to another entry attaining the same maximum, if the
                // row has one
                let t = t % self.tokens;
                let row = &self.key[t * self.vocab..(t + 1) * self.vocab];
                if let Some(j) = row.iter().enumerate()
                    .position(|(i, v)| *v == pm[t] && i as u32 != tok[t])
                {
                    tok[t] = j as u32;
                }
            }
            Some(Tamper::NotMax(t)) => {
                let t = t % self.tokens;
                tok[t] = (tok[t] + 1) % self.vocab as u32;
            }
            Some(Tamper::PmHigh(t)) => pm[t % self.tokens] += 1,
            Some(Tamper::PmLow(t)) => {
                let t = t % self.tokens;
                pm[t] = pm[t].saturating_sub(1);
            }
            None => {}
        }
        (tok, pm)
    }
}

// ------------------------------------------------------------------- the run ---

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub cells: [usize; 3],
    pub n_cols: [usize; 3],
    pub off_table: usize,
    pub lax: bool,
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0
    }
    pub fn json(&self, w: &AWitness, lay: &MatLayout) -> Value {
        let n = (w.tokens * w.vocab) as f64;
        let fresh = self.cells[1] + self.cells[2];
        json!({
            "tokens": w.tokens, "vocab": w.vocab, "source": w.source,
            "ties_in_witness": w.ties, "lax": self.lax,
            "layout": {"per_row": lay.per_row, "chunks": lay.chunks, "log_rows": lay.log_rows,
                       "lead_rows": lay.lead,
                       "lead_frac": lay.lead as f64 / (1usize << lay.log_rows) as f64},
            "cells": {"preprocessed": self.cells[0], "base": self.cells[1],
                      "interaction": self.cells[2], "fresh": fresh},
            "columns": {"preprocessed": self.n_cols[0], "base": self.n_cols[1],
                        "interaction": self.n_cols[2]},
            // the acceptance number: what one comparison costs
            "cells_per_comparison": fresh as f64 / n,
            "base_per_comparison": self.cells[1] as f64 / n,
            "proof_bytes": self.proven.proof_bytes, "prove_secs": self.proven.prove_secs,
            "verify_secs": self.proven.verify_secs,
            "off_table": self.off_table,
            "checks": {"stark": self.proven.stark_ok, "sums_cancel": self.sums.cancels(),
                       "off_table": self.off_table},
            "accepts": self.accepts(),
            "prove_error": self.proven.prove_error,
        })
    }
}

pub fn run(w: &AWitness, lax: bool, per_row_target: usize, bad: Option<Tamper>)
           -> (Outcome, MatLayout) {
    // the vocabulary is not a power of two at either real model, so this is the
    // padded chooser (B10 §2.41); every other region in the project still uses
    // the exact one and keeps `lead = 0`.
    let lay = MatLayout::choose_padded(w.vocab, w.tokens, per_row_target);
    let pr = lay.per_row;
    let rows = 1usize << lay.log_rows;
    let (tok, pm) = w.tampered(bad);

    // column order = the AIR's mask order: pm, tok, keys, indicators, ss
    let col_pm = 0usize;
    let col_tk = 1usize;
    let k0 = 2usize;
    let b0 = k0 + pr;
    let col_ss = b0 + pr;
    let n_cols = col_ss + 1;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut m_rk = vec![0u32; 1 << KEY_BITS];
    let mut off_table = 0usize;

    let mut ss_carry = 0i64;
    for j in 0..rows {
        let Some((t, b)) = lay.at_opt(w.tokens, j) else {
            // An inert lead row, filled the way `ao_region` fills its own: values
            // that satisfy every constraint and whose lookups land on real table
            // rows.  `pm = k = b = 0` gives `pm - k - b = 0`, which is in range;
            // the indicator is 0 so the monotone and selector constraints hold;
            // and `ss` stays at whatever it was, because `nofirst` is 0 here.
            for sl in 0..pr {
                m_rk[0] += 1;
            }
            continue;
        };
        if b == lay.chunks - 1 {
            ss_carry = 0; // the token's first row
        }
        cols[col_pm][j] = m31_u32(pm[t]);
        cols[col_tk][j] = m31_u32(tok[t]);
        for sl in 0..pr {
            // the walk ascends the vocabulary: rows carry blocks in increasing
            // index order even though `MatLayout` numbers its chunks the other way
            let i = (lay.chunks - 1 - b) * pr + sl;
            let k = w.key[t * w.vocab + i];
            // lax commits the one-hot selector, strict the monotone indicator --
            // the same column, carrying the weaker or the stronger statement
            let ind = if lax {
                u32::from(i as u32 == tok[t])
            } else {
                u32::from((i as u32) < tok[t])
            };
            cols[k0 + sl][j] = m31_u32(k);
            cols[b0 + sl][j] = m31_u32(ind);
            let arg = pm[t] as i64 - k as i64 - if lax { 0 } else { ind as i64 };
            if (0..(1i64 << KEY_BITS)).contains(&arg) {
                m_rk[arg as usize] += 1;
            } else {
                off_table += 1;
            }
            ss_carry += ind as i64;
        }
        cols[col_ss][j] = crate::attn_trace::m31_of(ss_carry);
    }

    // the multiplicity column, then the selectors
    let mult = eval_of(KEY_BITS, m_rk.iter().map(|v| m31_u32(*v)).collect());
    let sel = |f: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
    };
    // `nofirst` and `last` are false on an inert row, so its accumulator neither
    // carries nor closes -- the same shape `ao_region`'s selectors have.
    let first = |j: usize| matches!(lay.at_opt(w.tokens, j), Some((_, b)) if b == lay.chunks - 1);
    let active = |j: usize| lay.at_opt(w.tokens, j).is_some();
    let pre = vec![
        sel(&|j| active(j) && !first(j)),
        sel(&|j| matches!(lay.at_opt(w.tokens, j), Some((_, 0)))),
        sel(&|j| j + 1 == rows),
    ];
    let tbl = eval_of(KEY_BITS, (0..1u32 << KEY_BITS).map(m31_u32).collect());

    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    let widest = lay.log_rows.max(KEY_BITS);
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);
    sess.commit(vec![pre[0].clone(), pre[1].clone(), pre[2].clone(), tbl.clone()]);
    let mut base_all = base.clone();
    base_all.push(mult.clone());
    sess.commit(base_all.clone());
    let rk = sess.draw_relation(RangeRel::draw);

    // the interaction side: one fraction per lookup, paired as the AIR pairs them
    let (inter, use_sum) = {
        let looks: Vec<usize> = (0..pr).collect();
        let one = stwo::prover::backend::simd::qm31::PackedQM31::broadcast(
            stwo::core::fields::qm31::SecureField::one());
        let (n, f) = crate::plogup::paired(looks.len(), |i, vr| {
            let pmv = base_all[col_pm].data[vr];
            let kv = base_all[k0 + i].data[vr];
            let bv = base_all[b0 + i].data[vr];
            let arg = if lax { pmv - kv } else { pmv - kv - bv };
            (one, rk.combine(&[arg]))
        });
        crate::plogup::build(lay.log_rows, n, f)
    };
    let (tcols, tsum) = crate::attn_trace::table_col(KEY_BITS, &rk, &[&tbl], &mult);
    let mut icols = inter;
    icols.extend(tcols);
    sess.commit(icols.clone());

    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(
        &[id(AM_SEL[0]), id(AM_SEL[1]), id(AM_SEL[2]), tbl_range(KEY_BITS)]);
    let am = FrameworkComponent::<ArgmaxEval>::new(
        alloc, ArgmaxEval { log_rows: lay.log_rows, per_row: pr, lax, rk: rk.clone() },
        use_sum);
    let tb = FrameworkComponent::<RangeTableEval>::new(
        alloc, RangeTableEval { bits: KEY_BITS, rel: rk }, tsum);
    let provers: Vec<&dyn ComponentProver<_>> = vec![&am, &tb];
    let components: Vec<&dyn Component> = vec![&am, &tb];
    let proven = sess.prove(&provers, &components);

    let cells = [
        pre.iter().map(|e| e.domain.size()).sum::<usize>() + tbl.domain.size(),
        base_all.iter().map(|e| e.domain.size()).sum::<usize>(),
        icols.iter().map(|e| e.domain.size()).sum::<usize>(),
    ];
    let n_cols_out = [4, base_all.len(), icols.len()];
    (Outcome { proven, sums: Sums { uses: use_sum, tables: tsum }, cells,
               n_cols: n_cols_out, off_table, lax },
     lay)
}
