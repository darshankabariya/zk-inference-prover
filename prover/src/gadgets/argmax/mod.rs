//! B10 step 4: the token binding -- "the emitted token is the argmax of the logits".
//!
//! Greedy decoding emits `argmax_v logits[t, v]`, and `torch.argmax` returns the
//! **first** maximal index.  On the real 512-token continuations 15 of 512 rows
//! (C=4K) and 14 of 512 (C=16K) have two or three co-maximal logits
//! (`b10/out/head_ties_*.json`), so the co-maximum choice rule decides the token often
//! enough to matter and the statement has to name it.
//!
//! # Two statements, and why the strict one is not the expensive one
//!
//! B6.5's max argument proves "`tok` attains the maximum": commit the maximum's
//! pattern `pm`, range-check `pm - k_i >= 0` on every entry, and a selector with
//! `sel_i (pm - k_i) = 0`, `sum_i sel_i = 1` pins `pm` to an attained value.  On
//! a co-maximal row a prover may point the selector at either co-maximal entry.
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
//! the binding itself and costs nothing extra.
//!
//! So the strict statement is the lax one with `s` renamed to `b`, one extra
//! degree-2 constraint (monotonicity), and a different closing value.
//!
//! The strict form is therefore the default and the composition's only sound
//! one.  `--unsound-argmax-lax` proves the weaker statement so the two can be
//! measured against each other rather than argued about, and it is named for
//! what it gives up: a lax proof accepts an emitted token that is *a* maximum
//! where the strict proof accepts only the *first* one, so on the 15 rows in
//! 512 that carry a co-maximal pair the prover picks the token and the
//! statement still holds.  A run that used it lists `argmax-lax` in
//! `provenance.unsound_skips`, and its binding table marks edge 16 unbound.
//!
//! # What this component does *not* do
//!
//! `k_i` is the committed **order key** of the logit: bf16 magnitude order is
//! pattern order, but logits are signed, so the key is the sign-flipped pattern
//! (`k = s ? 0x7FFF - mag : 0x8000 + mag`), which is a total order matching the
//! float order.  Pinning `k` to the `lm_head` dequant's own output is a
//! positionwise binding of the kind B10 step 2 built for the residual stream, and it is the
//! composition's job, not this component's.  Standalone, `k` is the statement's
//! input.

use num_traits::{One, Zero};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
    Relation, RelationEntry, TraceLocationAllocator,
};

use crate::common::{Eval, Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::encoding::{m31_of, secure_cols_pub};
use crate::layout::MatLayout;
use crate::relations::{RangeRel, RangeTableEval, id, table_col, tbl_range};

/// The order key's width: a bf16 sign-flipped pattern is 16 bits.
pub const KEY_BITS: u32 = 16;

pub const AM_SEL: [&str; 3] = ["b10_am_nofirst", "b10_am_last", "b10_am_end"];

/// The order key of a signed bf16 pattern: negatives descend, positives ascend,
/// and every key is 16 bits.
pub fn order_key(pat: u16) -> u32 {
    let (sign, mag) = (pat >> 15, pat & 0x7FFF);
    if sign == 1 { (0x7FFF - mag) as u32 } else { 0x8000 + mag as u32 }
}

// -------------------------------------------------------------------- the AIR --

/// Stream E: the argmax's side of the head binding.  One Horner claim over the
/// component's committed keys, with the *mirror* of the dequant's weights -- this
/// component's index map ascends the vocabulary where the dequant's chunks descend,
/// so its row step is the inverse of the output side's.
///
/// `tag` names whose selectors to read: the head dequant's, because both components
/// walk the same `(token, chunk)` rows and sharing the columns is what makes it
/// impossible for the two sides to disagree about which row is which.
#[derive(Clone)]
pub struct ArgmaxBinding {
    pub w: Vec<SecureField>,
    pub k_in: SecureField,
    pub k_bd: SecureField,
    pub claimed: SecureField,
    /// Stream KK: the second accumulator's step -- one factor of `w` per token,
    /// which is what makes the emitted tokens a *named value* of the head
    /// chunk's statement rather than a committed column nobody outside the
    /// proof can address.  `verify_composed` recomputes this claim from the
    /// statement's published `tok_out` and compares; `boundary-grid` then reads
    /// the vector, position by position, against the ids the next block's
    /// embedding chunk consumed (`notes/witness-slack.md` §5 item 2).
    pub k_tok: SecureField,
    pub claimed_tok: SecureField,
    pub tag: String,
}

#[derive(Clone)]
pub struct ArgmaxEval {
    pub log_rows: u32,
    pub per_row: usize,
    /// prove only "the token attains the maximum" -- B6.5's argument unchanged,
    /// and a weaker statement than the default: see this module's header and
    /// `crate::cli::ARGMAX_LAX`
    pub lax: bool,
    pub rk: RangeRel,
    /// stream E: the binding to the head's dequant output, when this component is
    /// composed rather than proved on its own
    pub binding: Option<ArgmaxBinding>,
}

impl FrameworkEval for ArgmaxEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`); `None` when this
    /// component's fingerprint has no kernel, which is the fork's
    /// scalar CPU route.
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_argmax_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> = AM_SEL.iter().map(|n| eval.get_preprocessed_column(id(*n))).collect();
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
        // the committed token, which *is* the binding.
        let [ss_prev, ss] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [-1, 0]);
        let mut f = E::F::zero();
        for s in if self.lax { &sels } else { &bs } {
            f = f + s.clone();
        }
        eval.add_constraint(ss.clone() - ss_prev * nofirst - f);
        let want = if self.lax { E::F::one() } else { tk.clone() };
        if !self.lax {
            // The all-ones indicator is the one assignment the chain and the
            // closing sum both accept with `tok = vocab` -- no selector fires,
            // so nothing above pins the token *into* the vocabulary.  The final
            // slot of the last row closes the chain at 0 instead: an honest
            // `tok < vocab` has `b = [i < tok] = 0` there already (pad slots
            // included), and the all-ones cheat is the only witness this kills.
            eval.add_constraint(bs[self.per_row - 1].clone() * last.clone());
        }
        eval.add_constraint((ss - want) * last);

        // Stream E: the binding.  `end` was committed from the start and unused; it
        // is exactly the selector a Horner claim closes on, so the binding needs no
        // new column of its own beyond the step selectors it borrows from the
        // head dequant's component.
        if let Some(t) = &self.binding {
            let oin = eval.get_preprocessed_column(
                crate::gadgets::dequant::deq_sel(&t.tag, crate::gadgets::dequant::DEQ_OSEL[0]));
            let obd = eval.get_preprocessed_column(
                crate::gadgets::dequant::deq_sel(&t.tag, crate::gadgets::dequant::DEQ_OSEL[1]));
            let oact = eval.get_preprocessed_column(
                crate::gadgets::dequant::deq_sel(&t.tag, crate::gadgets::dequant::DEQ_OSEL[2]));
            let [u_prev, u] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut f = E::EF::zero();
            for (i, k) in ks.iter().enumerate() {
                f += k.clone() * t.w[i];
            }
            let mult: E::EF =
                E::EF::from(oin.clone()) * t.k_in + E::EF::from(obd.clone()) * t.k_bd;
            eval.add_constraint(u.clone() - u_prev * mult.clone() - f * oact.clone());
            let claim = eval.claim(t.claimed);
            eval.add_constraint((u - claim) * end.clone());

            // stream KK: the same shape over the emitted token, one step per
            // token instead of one per row -- `oin` carries the accumulator
            // unchanged inside a token, `obd` multiplies by `w` at the row that
            // begins one, and the id is added exactly there.  The committed
            // `tk` is the one the component's own closing sum pins to the
            // lowest-indexed maximum, so this claim is a Horner image of the
            // proved argmax and of nothing else.
            let kmult: E::EF = E::EF::from(oin.clone()) + E::EF::from(obd) * t.k_tok;
            let [g_prev, g] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            eval.add_constraint(
                g.clone() - g_prev * kmult
                    - E::EF::from(tk.clone() * (E::F::one() - oin) * oact),
            );
            let claim_tok = eval.claim(t.claimed_tok);
            eval.add_constraint((g - claim_tok) * end);
        } else {
            let _ = end;
        }

        eval.finalize_logup_in_pairs();
        eval
    }
}

// ------------------------------------------------------------------ witness ----

#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// point the token at a different entry that also attains the maximum -- the
    /// probe the *lax* statement cannot reject and the strict one must
    CoMax(usize),
    /// name a token whose logit is not the maximum at all
    NotMax(usize),
    /// claim a maximum above every entry: attained by none
    PmHigh(usize),
    /// claim a maximum below the true one: some entry exceeds it
    PmLow(usize),
    /// set the token to the vocabulary size itself: the all-ones indicator the
    /// strict chain and closing sum both accepted before the final-slot close
    TokVocab(usize),
    /// Stream E: one committed key changed, with `pm` and `tok` recomputed from
    /// the changed keys so the component stays internally consistent.  Every
    /// constraint here still holds -- what this component proves is a statement
    /// *about* its keys -- and only the binding to the head's dequant output can
    /// tell that they are not the logits the layer computed.
    Key(usize),
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
    pub co_max_rows: usize,
}

impl AWitness {
    /// A synthetic block with a controlled co-maximum rate, so the strict statement has
    /// something to bite on.  `co_max_frac` of the rows get a second entry at the
    /// maximum, placed *after* the first so the lowest-index rule is what picks.
    pub fn synthetic(tokens: usize, vocab: usize, co_max_frac: f64, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut key = vec![0u32; tokens * vocab];
        for k in key.iter_mut() {
            *k = rng.random_range(0..(1u32 << KEY_BITS) - 1);
        }
        let (mut tok, mut pm, mut co_max_rows) = (vec![0u32; tokens], vec![0u32; tokens], 0usize);
        for t in 0..tokens {
            let row = &mut key[t * vocab..(t + 1) * vocab];
            let mut best = 0usize;
            for i in 1..vocab {
                if row[i] > row[best] {
                    best = i;
                }
            }
            if rng.random::<f64>() < co_max_frac && best + 1 < vocab {
                // a co-maximal entry strictly after the winner: the lax statement
                // would accept either, the strict one only the first
                let j = rng.random_range(best + 1..vocab);
                row[j] = row[best];
                co_max_rows += 1;
            }
            // the argmax is the *first* attained entry, as torch's is
            let mx = *row.iter().max().unwrap();
            let first = row.iter().position(|v| *v == mx).unwrap();
            tok[t] = first as u32;
            pm[t] = mx;
        }
        Self { tokens, vocab, key, tok, pm, co_max_rows,
               source: format!("synthetic(seed={seed}, co_max={co_max_frac})") }
    }

    /// The real head's own keys and tokens, from an `extract.py --head` dump:
    /// `key_head.u16` is `(tokens, vocab)` of the order key this component compares
    /// and `tok_head.i32` is the token each row emits.  `pm` is not dumped --
    /// it is the row maximum, which is a fact about `key`, so it is derived here
    /// and the component's own constraints are what check that the prover derived
    /// it honestly.
    ///
    /// This is the statement's *input*: what pins `key` to the head's dequant is
    /// a positionwise binding, and that binding is not built yet (see the record).  Until
    /// it is, "the emitted token is the maximum" is proved about a committed key
    /// column that B1 wrote, not about the logits the layer computed.
    /// The same, from columns a `Layer` already holds.
    pub fn from_cols(tokens: usize, vocab: usize, key: Vec<u32>, tok: Vec<u32>)
                     -> Result<Self, String> {
        let (mut pm, mut co_max_rows) = (vec![0u32; tokens], 0usize);
        for t in 0..tokens {
            let row = &key[t * vocab..(t + 1) * vocab];
            let mx = *row.iter().max().expect("a row of the vocabulary");
            pm[t] = mx;
            if row.iter().filter(|v| **v == mx).count() > 1 {
                co_max_rows += 1;
            }
            let first = row.iter().position(|v| *v == mx).expect("the maximum") as u32;
            if tok[t] != first {
                return Err(format!(
                    "row {t}: the dump emits {} but the first maximum is {first}", tok[t]));
            }
        }
        Ok(Self { tokens, vocab, key, tok, pm, co_max_rows, source: "the layer's head columns".into() })
    }

    pub fn from_dump(dir: &str, tokens: usize, vocab: usize) -> Result<Self, String> {
        let path = |n: &str| format!("{dir}/{n}");
        let raw = std::fs::read(path("key_head.u16")).map_err(|e| format!("key_head: {e}"))?;
        let key: Vec<u32> = raw
            .chunks_exact(2)
            .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
            .take(tokens * vocab)
            .collect();
        if key.len() != tokens * vocab {
            return Err(format!("key_head: {} values, want {tokens} x {vocab}", key.len()));
        }
        let raw = std::fs::read(path("tok_head.i32")).map_err(|e| format!("tok_head: {e}"))?;
        let tok: Vec<u32> = raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .take(tokens)
            .collect();
        if tok.len() != tokens {
            return Err(format!("tok_head: {} values, want {tokens}", tok.len()));
        }
        let (mut pm, mut co_max_rows) = (vec![0u32; tokens], 0usize);
        for t in 0..tokens {
            let row = &key[t * vocab..(t + 1) * vocab];
            let mx = *row.iter().max().expect("a row of the vocabulary");
            pm[t] = mx;
            let n_at = row.iter().filter(|v| **v == mx).count();
            if n_at > 1 {
                co_max_rows += 1;
            }
            // B1 decoded greedily with torch's lowest-index rule, so the dump's
            // token has to *be* the first maximum; a dump that disagrees is a
            // spec mismatch, not something for the component to prove around
            let first = row.iter().position(|v| *v == mx).expect("the maximum") as u32;
            if tok[t] != first {
                return Err(format!(
                    "row {t}: the dump emits {} but the first maximum is {first}", tok[t]));
            }
        }
        Ok(Self { tokens, vocab, key, tok, pm, co_max_rows, source: format!("dump({dir})") })
    }

    fn tampered(&self, bad: Option<Tamper>) -> (Vec<u32>, Vec<u32>) {
        let (mut tok, mut pm) = (self.tok.clone(), self.pm.clone());
        match bad {
            Some(Tamper::CoMax(t)) => {
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
            Some(Tamper::TokVocab(t)) => tok[t % self.tokens] = self.vocab as u32,
            // stream E's key tamper is applied in `component`, which owns the keys
            Some(Tamper::Key(_)) | None => {}
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
            "co_max_rows_in_witness": w.co_max_rows, "lax": self.lax,
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

/// The component's committed columns and its table uses, with no transcript of its
/// own -- what a composed proof needs, and what `run` now builds its standalone
/// proof out of so the two cannot drift apart.
///
/// The layout is *given*, not chosen: a binding to the head's dequant is
/// positionwise, so both components have to walk the same `(token, chunk)` rows.
pub struct AComponent {
    pub lay: MatLayout,
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub off_table: usize,
    /// the committed keys, row-major over `(row, slot)` -- the binding's Horner input
    pub keys: Vec<u32>,
    /// stream KK: the committed token per *row*, zero on an inert lead row --
    /// the second binding's Horner input, and the vector the head chunk publishes
    pub toks: Vec<u32>,
    pub col_pm: usize,
    pub k0: usize,
    pub b0: usize,
}

impl AComponent {
    /// Stream KK: the emitted token of every position, in position order.  The
    /// rows walk the tokens descending, so this reads them back off the
    /// committed column rather than assuming an order.
    pub fn token_ids(&self, tokens: usize) -> Vec<u32> {
        let mut out = vec![0u32; tokens];
        for j in 0..1usize << self.lay.log_rows {
            if let Some((t, _)) = self.lay.at_opt(tokens, j) {
                out[t] = self.toks[j];
            }
        }
        out
    }
}

pub fn component(w: &AWitness, lax: bool, lay: MatLayout, m_rk: &mut [u32],
              bad: Option<Tamper>) -> AComponent {
    let pr = lay.per_row;
    let rows = 1usize << lay.log_rows;
    let mut keys = vec![0u32; rows * pr];
    let mut toks = vec![0u32; rows];
    let (mut tok, mut pm) = w.tampered(bad);
    // stream E: a key changed, and the row's own maximum and token move with it,
    // so nothing inside the component notices
    let mut kcol = std::borrow::Cow::Borrowed(&w.key);
    if let Some(Tamper::Key(i)) = bad {
        let q = i % w.key.len();
        let t = q / w.vocab;
        let owned = kcol.to_mut();
        owned[q] = owned[q].wrapping_add(1) & ((1 << KEY_BITS) - 1);
        let row = &owned[t * w.vocab..(t + 1) * w.vocab];
        let mx = *row.iter().max().unwrap();
        pm[t] = mx;
        tok[t] = row.iter().position(|v| *v == mx).unwrap() as u32;
    }
    let wkey: &[u32] = &kcol;

    // column order = the AIR's mask order: pm, tok, keys, indicators, ss
    let col_pm = 0usize;
    let col_tk = 1usize;
    let k0 = 2usize;
    let b0 = k0 + pr;
    let col_ss = b0 + pr;
    let n_cols = col_ss + 1;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut off_table = 0usize;

    let mut ss_carry = 0i64;
    for j in 0..rows {
        let Some((t, b)) = lay.at_opt(w.tokens, j) else {
            // An inert lead row, filled the way `ao_component` fills its own: values
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
        toks[j] = tok[t];
        for sl in 0..pr {
            // the walk ascends the vocabulary: rows carry blocks in increasing
            // index order even though `MatLayout` numbers its chunks the other way
            let i = (lay.chunks - 1 - b) * pr + sl;
            let k = wkey[t * w.vocab + i];
            // lax commits the one-hot selector, strict the monotone indicator --
            // the same column, carrying the weaker or the stronger statement
            let ind = if lax {
                u32::from(i as u32 == tok[t])
            } else {
                u32::from((i as u32) < tok[t])
            };
            cols[k0 + sl][j] = m31_u32(k);
            keys[j * pr + sl] = k;
            cols[b0 + sl][j] = m31_u32(ind);
            let arg = pm[t] as i64 - k as i64 - if lax { 0 } else { ind as i64 };
            if (0..(1i64 << KEY_BITS)).contains(&arg) {
                m_rk[arg as usize] += 1;
            } else {
                off_table += 1;
            }
            ss_carry += ind as i64;
        }
        cols[col_ss][j] = m31_of(ss_carry);
    }

    let sel = |f: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(f(j) as u32)).collect())
    };
    // `nofirst` and `last` are false on an inert row, so its accumulator neither
    // carries nor closes -- the same shape `ao_component`'s selectors have.
    let first = |j: usize| matches!(lay.at_opt(w.tokens, j), Some((_, b)) if b == lay.chunks - 1);
    let active = |j: usize| lay.at_opt(w.tokens, j).is_some();
    let pre = vec![
        sel(&|j| active(j) && !first(j)),
        sel(&|j| matches!(lay.at_opt(w.tokens, j), Some((_, 0)))),
        sel(&|j| j + 1 == rows),
    ];

    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    AComponent { lay, pre, base, off_table, keys, toks, col_pm, k0, b0 }
}

/// The component's LogUp columns and its use sum, in the order the AIR adds the
/// lookups.  Split out of `run` so the composed path builds the same fractions.
pub fn logup(r: &AComponent, lax: bool, rk: &RangeRel) -> (Vec<Eval>, SecureField) {
    let one = stwo::prover::backend::simd::qm31::PackedQM31::broadcast(SecureField::one());
    let (n, f) = crate::plogup::paired(r.lay.per_row, |i, vr| {
        let pmv = r.base[r.col_pm].data[vr];
        let kv = r.base[r.k0 + i].data[vr];
        let bv = r.base[r.b0 + i].data[vr];
        let arg = if lax { pmv - kv } else { pmv - kv - bv };
        (one, rk.combine(&[arg]))
    });
    crate::plogup::build(r.lay.log_rows, n, f)
}

/// Stream E: the binding's accumulator column, and the value it closes on.  The
/// weights are the mirror of the head dequant's output side, and an inert row
/// contributes nothing on either side.
pub fn binding_col(r: &AComponent, tokens: usize, t: &ArgmaxBinding) -> (Vec<Eval>, SecureField, SecureField) {
    let rows = 1usize << r.lay.log_rows;
    let mut u = vec![SecureField::zero(); rows];
    // stream KK: the second accumulator, one step per token over the committed
    // `tk`.  Its multipliers are the same three preprocessed selectors the key
    // binding reads (`oin`, `obd`, `oact`), so the two cannot disagree about which
    // row begins a token.
    let mut g = vec![SecureField::zero(); rows];
    for j in 0..rows {
        let mut f = SecureField::zero();
        if r.lay.at_opt(tokens, j).is_some() {
            for sl in 0..r.lay.per_row {
                f += SecureField::from(m31_u32(r.keys[j * r.lay.per_row + sl])) * t.w[sl];
            }
        }
        let first = matches!(r.lay.at_opt(tokens, j), Some((_, b)) if b + 1 == r.lay.chunks);
        let active = r.lay.at_opt(tokens, j).is_some();
        let oin = j > 0 && active && !first;
        let obd = j > 0 && first;
        let mult = if oin { t.k_in } else if obd { t.k_bd } else { SecureField::zero() };
        let kmult = if oin {
            SecureField::one()
        } else if obd {
            t.k_tok
        } else {
            SecureField::zero()
        };
        let add = if active && !oin {
            SecureField::from(m31_u32(r.toks[j]))
        } else {
            SecureField::zero()
        };
        u[j] = if j == 0 { f } else { u[j - 1] * mult + f };
        g[j] = if j == 0 { add } else { g[j - 1] * kmult + add };
    }
    let mut cols = secure_cols_pub(r.lay.log_rows, &u);
    cols.extend(secure_cols_pub(r.lay.log_rows, &g));
    (cols, u[rows - 1], g[rows - 1])
}

pub fn run(w: &AWitness, lax: bool, per_row_target: usize, bad: Option<Tamper>)
           -> (Outcome, MatLayout) {
    // the vocabulary is not a power of two at either real model, so this is the
    // padded chooser (B10 §2.41); the composed path is handed the head dequant's
    // layout instead, which `--deq-per-row 501` makes exact.
    let lay = MatLayout::choose_padded(w.vocab, w.tokens, per_row_target);
    let pr = lay.per_row;
    let rows = 1usize << lay.log_rows;
    let mut m_rk = vec![0u32; 1 << KEY_BITS];
    let r = component(w, lax, lay, &mut m_rk, bad);
    let (pre, base, off_table) = (r.pre.clone(), r.base.clone(), r.off_table);
    let (col_pm, k0, b0) = (r.col_pm, r.k0, r.b0);
    let mult = eval_of(KEY_BITS, m_rk.iter().map(|v| m31_u32(*v)).collect());
    let tbl = eval_of(KEY_BITS, (0..1u32 << KEY_BITS).map(m31_u32).collect());
    let widest = lay.log_rows.max(KEY_BITS);
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);
    sess.commit(vec![pre[0].clone(), pre[1].clone(), pre[2].clone(), tbl.clone()]);
    // A gadget on its own has no model column, but the tree layout is the
    // same for every proof this crate makes: the empty model tree keeps the
    // base trace at the index every component's mask names.
    sess.commit_model_empty();
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
    let (tcols, tsum) = table_col(KEY_BITS, &rk, &[&tbl], &mult);
    let mut icols = inter;
    icols.extend(tcols);
    sess.commit(icols.clone());

    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(
        &[id(AM_SEL[0]), id(AM_SEL[1]), id(AM_SEL[2]), tbl_range(KEY_BITS)]);
    let am = FrameworkComponent::<ArgmaxEval>::new(
        alloc, ArgmaxEval { log_rows: lay.log_rows, per_row: pr, lax, rk: rk.clone(),
                            binding: None },
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

// ------------------------------------------------------------------- tests ----


/// The argmax over the head's output: which token the logits emit.
pub struct Argmax;

/// Everything about an argmax run that is not the witness.
pub struct Cfg {
    /// the weaker statement, which is a control and never a result
    pub lax: bool,
    pub per_row: usize,
}

impl crate::gadgets::Gadget for Argmax {
    type Witness = AWitness;
    type Cfg = Cfg;
    type Tamper = Tamper;
    type Outcome = Outcome;

    fn run(w: &AWitness, cfg: &Cfg, bad: Option<Tamper>) -> Outcome {
        run(w, cfg.lax, cfg.per_row, bad).0
    }

    fn accepts(o: &Outcome) -> bool {
        o.accepts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stream KK (`notes/witness-slack.md` section 5 item 2): the sum
    /// `verify_composed` recomputes from the statement's published `tok_out` is
    /// the claim the component publishes.  The argmax's rows walk the positions
    /// *descending*, so the exponents are the mirror of the embedding's and
    /// this is where that is checked rather than argued.
    #[test]
    fn the_published_tokens_recompute_the_claim() {
        for (tokens, vocab, target) in [(8usize, 40usize, 8usize),
                                        (16, 97, 16),
                                        (4, 300, 32)] {
            let w = AWitness::synthetic(tokens, vocab, 0.25, 5);
            let lay = MatLayout::choose_padded(vocab, tokens, target);
            let mut m_rk = vec![0u32; 1 << KEY_BITS];
            let r = component(&w, false, lay, &mut m_rk, None);
            let ch = SecureField::from_m31_array(
                [m31_u32(61), m31_u32(67), m31_u32(71), m31_u32(73)]);
            let binding = ArgmaxBinding { w: vec![SecureField::zero(); lay.per_row],
                             k_in: SecureField::zero(), k_bd: SecureField::zero(),
                             claimed: SecureField::zero(), k_tok: ch,
                             claimed_tok: SecureField::zero(), tag: String::new() };
            let (_, _, claimed) = binding_col(&r, tokens, &binding);
            let toks = r.token_ids(tokens);
            assert_eq!(toks, w.tok, "the published tokens are the component's");
            // exactly `verify_composed`'s `horner(.., false)`
            let mut acc = SecureField::zero();
            for v in toks.iter().rev() {
                acc = acc * ch + SecureField::from(m31_u32(*v));
            }
            assert_eq!(acc, claimed, "tokens={tokens} vocab={vocab} target={target}");
        }
    }

    // ------------------------------------------------- one shape per layout ----

    /// `compose::argmax_binding_w`, at the size a test picks: the binding's
    /// weights are challenge powers whose exponents the layout fixes, which is
    /// the half of the lowering's input that is *not* a claim.
    fn binding_of(wch: SecureField, tokens: usize, vocab: usize, per_row: usize) -> ArgmaxBinding {
        use stwo::core::fields::FieldExpOps;
        let pw = |e: i128| if e >= 0 { wch.pow(e as u128) } else { wch.pow((-e) as u128).inverse() };
        let n = (tokens * vocab) as i128;
        let (tk, pr, v) = (tokens as i128, per_row as i128, vocab as i128);
        ArgmaxBinding {
            w: (0..per_row).map(|sl| pw(n - 1 - (v - pr) * tk - sl as i128 * tk)).collect(),
            k_in: pw(pr * tk),
            k_bd: pw(-((v - pr) * tk + 1)),
            claimed: SecureField::zero(),
            k_tok: wch,
            claimed_tok: SecureField::zero(),
            tag: "lm_head".into(),
        }
    }

    /// What the kernel store is keyed on, and the two counts a fold would move.
    fn shape(ev: &ArgmaxEval) -> (u32, usize, usize) {
        let ir = crate::cudagen::lower(ev, SecureField::zero(), ev.log_size(),
                                       ev.max_constraint_log_degree_bound());
        (ir.fingerprint(), ir.ops.len(), ir.scalars.len())
    }

    /// The component this witness composes to, with the claims it publishes.
    fn composed(w: &AWitness, lay: MatLayout, rk: &RangeRel, wch: SecureField) -> ArgmaxEval {
        let mut m_rk = vec![0u32; 1 << KEY_BITS];
        let r = component(w, false, lay, &mut m_rk, None);
        let mut binding = binding_of(wch, w.tokens, w.vocab, lay.per_row);
        let (_, claimed, claimed_tok) = binding_col(&r, w.tokens, &binding);
        binding.claimed = claimed;
        binding.claimed_tok = claimed_tok;
        ArgmaxEval { log_rows: lay.log_rows, per_row: lay.per_row, lax: false,
                     rk: rk.clone(), binding: Some(binding) }
    }

    /// A block whose maximum is column 0 on every row, so every emitted token is
    /// zero and the token claim closes on exactly zero -- the shape of the
    /// model-only stand-in `Layer::from_dir` writes, and the value a fold reads
    /// as nothing at all.  `ties` rows get a second entry at that maximum.
    fn first_column_wins(tokens: usize, vocab: usize, ties: usize) -> AWitness {
        let mut key = vec![1u32; tokens * vocab];
        for t in 0..tokens {
            key[t * vocab] = 2;
            if t < ties {
                key[t * vocab + vocab - 1] = 2;
            }
        }
        AWitness::from_cols(tokens, vocab, key, vec![0u32; tokens])
            .expect("column 0 is the first maximum of every row")
    }

    /// Issue #25: what this component lowers to is a fact about its layout, and
    /// about nothing the witness decides.
    ///
    /// How many rows have two entries at the maximum reaches the lowering
    /// through exactly two values -- the key claim and the token claim this
    /// component publishes -- and through nothing else: every other scalar the
    /// lowering sees is a challenge power whose exponent the layout fixes.  Both
    /// go in through `EvalAtRow::claim`, so each takes a slot of its own by
    /// occurrence, four limbs wide, never folded and never shared by value.
    ///
    /// Read as ordinary constants instead, a block with no co-maximal row and a
    /// block with many would ask the kernel store for two different shapes, and
    /// the one the store was not emitted from would take the announced host
    /// fallback for every head segment of the run.  The degenerate blocks below
    /// are the two that used to move it: a token claim of exactly zero, which a
    /// fold reads as nothing at all, and two claims that coincide, which used to
    /// share one slot and one instruction.
    #[test]
    fn the_co_maximum_count_is_not_a_shape_fact() {
        use stwo::core::channel::{Blake2sM31Channel, Channel};
        for (tokens, vocab, target) in [(8usize, 32usize, 8usize), (16, 97, 16), (4, 300, 32)] {
            let lay = MatLayout::choose_padded(vocab, tokens, target);
            let mut ch = Blake2sM31Channel::default();
            Channel::mix_u64(&mut ch, 0x2500 + tokens as u64);
            let rk = RangeRel::draw(&mut ch);
            let wch = Channel::draw_secure_felts(&mut ch, 1)[0];
            let want = shape(&composed(&AWitness::synthetic(tokens, vocab, 0.0, 11), lay, &rk, wch));

            let mut counts = vec![];
            for frac in [0.0f64, 0.25, 0.5, 1.0] {
                let w = AWitness::synthetic(tokens, vocab, frac, 11);
                counts.push(w.co_max_rows);
                assert_eq!(shape(&composed(&w, lay, &rk, wch)), want,
                           "{} of {tokens} rows co-maximal", w.co_max_rows);
            }
            assert!(counts.iter().any(|c| *c > 0) && counts.contains(&0),
                    "the case needs blocks that differ in their co-maximum count, got {counts:?}");

            // the token claim is exactly zero on both of these, and they differ
            // in how many rows carry a second entry at the maximum
            for ties in [0usize, tokens] {
                let w = first_column_wins(tokens, vocab, ties);
                let ev = composed(&w, lay, &rk, wch);
                assert_eq!(ev.binding.as_ref().unwrap().claimed_tok, SecureField::zero(),
                           "every emitted token is zero, so the token claim is");
                assert_eq!(shape(&ev), want, "a token claim of zero, {ties} rows co-maximal");
            }

            // and the pair a chunk cannot rule out: two claims with one value
            let w = AWitness::synthetic(tokens, vocab, 0.5, 11);
            let mut ev = composed(&w, lay, &rk, wch);
            let t = ev.binding.as_mut().unwrap();
            t.claimed_tok = t.claimed;
            assert_eq!(shape(&ev), want, "two claims that coincide");
        }
    }
}

