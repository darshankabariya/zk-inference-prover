//! Prove this component on its own over one weight block, and report what it
//! costs in committed cells per weight.
//!
//! The component is the composed segment's own: `trace::freivalds_component`
//! builds the columns committed before the challenges and
//! `claims::freivalds_interaction` the running sums and the LogUp columns,
//! exactly as `compose` calls them.  The session around them is
//! `crate::alone::prove_alone`, which every gadget's standalone run shares, so
//! this file is the witness, the call and the outcome.  Nothing else is in the
//! proof but the range table the committed activations are checked against,
//! which is a shared table and is reported apart.
//!
//! **Half of the identity is here and half is not.**  What this proves is the
//! `(s^T X)(W rv)` side: the weights are pinned, the activations are committed
//! and range-checked, and the running product closes on the claim.  The other
//! side, `s^T C rv`, is the dequant component's, and that component is not built
//! here -- a standalone dequant proof would need its own norm, fraction,
//! multiplication and pair tables, which is a second harness this file does not
//! have.  So `ref_ok`/`ref_n` below, and `reference_ok` in the JSON, are a
//! **self-check and not part of the proof**: they recompute `s^T C rv` on the
//! host with `claims::project_product`, straight out of the witness's exact
//! accumulators, and compare.  A composed proof is where the two sides are both
//! proved and compared, by `claims::identity_holds`.
//!
//! What the cost model multiplies by a parameter count is the *marginal*: how many
//! fresh cells one more weight costs.  A group's table runs one row group per
//! input channel, cut into `chunks` row chunks, and a row carries `pk` weights of each
//! matmul and `pq` committed activations.  So the committed activations are
//! `tokens x n_in` whatever `chunks` is, their range check is two more cells each
//! at pairwise batching, and the only thing that grows with the weights is the
//! three QM31 running columns `x_fold`, `w_fold` and `product` -- 12 M31 cells a
//! row for a group of one matmul.  A row carries `pk` weights, so the marginal
//! is `12/pk` and the two fixed parts drop out of a difference.  `run` proves one
//! build; `marginal` takes two builds of the same block at two output widths and
//! reports the difference over the weights they differ by, so nothing here has
//! to be assumed.

use std::time::Instant;

use serde_json::{Value, json};

use crate::alone::{After, Alone, RangeTable, Trees, prove_alone};
use crate::bf16::Round;
use crate::columns::{ByteCells, Lookups};
use crate::fold::Powers;
use crate::gadgets::freivalds::FreivaldsEval;
use crate::gadgets::freivalds::claims::{FreivaldsClaims, freivalds_interaction, project_product};
use crate::gadgets::freivalds::columns::{FreivaldsShape, Known};
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::gadgets::freivalds::trace::freivalds_component;
use crate::gadgets::freivalds::witness::FreivaldsWitness;
use crate::layer::Tables;

/// The activation width every Freivalds group but the widened attention output's
/// commits at, and the width of the range table this proof carries.
const ACT_BITS: u32 = 8;

pub struct Outcome {
    /// what the standalone session proved, committed and published
    pub alone: Alone,
    pub lay: FreivaldsLayout,
    pub group: String,
    pub n_mats: usize,
    pub n_in: usize,
    pub n_out: usize,
    pub tokens: usize,
    pub weights: usize,
    pub off_table: usize,
    /// The self-check, not a proof: how many of the published claims match
    /// `s^T C rv` recomputed on the host from the witness's exact integer
    /// accumulators.  A composed proof compares the same two values with both
    /// sides proved; this run proves only one of them.
    pub ref_ok: usize,
    pub ref_n: usize,
    pub trace_secs: f64,
}

impl Outcome {
    /// The proof verified, the lookups cancel, no activation fell off its range
    /// table -- and the host-side self-check agrees, which is *not* a proof of
    /// the other side of the identity.  See this module's comment.
    pub fn accepts(&self) -> bool {
        self.alone.proven.stark_ok && self.alone.sums.cancels() && self.off_table == 0
            && self.ref_ok == self.ref_n
    }
    /// Fresh cells the component commits, the shared table left out.
    pub fn fresh(&self) -> usize {
        self.alone.fresh()
    }
    pub fn json(&self) -> Value {
        let mut cells = self.alone.cells_json();
        cells["fresh_per_weight"] = json!(self.fresh() as f64 / self.weights as f64);
        cells["fresh_per_token"] = json!(self.fresh() as f64 / self.tokens as f64);
        json!({
            "group": self.group, "matmuls": self.n_mats,
            "n_in": self.n_in, "n_out": self.n_out, "tokens": self.tokens,
            "weights": self.weights,
            "layout_x": self.layout_json(),
            "cells": cells,
            "columns": self.alone.columns_json(),
            "off_table": self.off_table,
            // the host-side self-check; see `reference_note`
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "reference_is_proved": false,
            "reference_note": "self-check, not part of the proof: this run proves \
the (s^T X)(W rv) side only, and recomputes s^T C rv on the host from the witness's \
exact accumulators to compare against it. A composed proof proves both sides and \
compares them in freivalds::claims::identity_holds.",
            "trace_secs": self.trace_secs, "interaction_secs": self.alone.interaction_secs,
            "proof_bytes": self.alone.proven.proof_bytes,
            "sums": self.alone.sums.json(), "accepts": self.accepts(),
            "point": self.alone.proven.json(),
        })
    }
    fn layout_json(&self) -> Value {
        let mut v = self.lay.g.json();
        v["weights_per_row"] = json!(self.lay.pk);
        v["activations_per_row"] = json!(self.lay.pq);
        v
    }
}

/// One weight block's check, proved on its own at `p_target` weights a row.
pub fn run(w: &FreivaldsWitness, p_target: usize) -> Outcome {
    let t0 = Instant::now();
    // The shared table registry.  This proof looks values up in exactly one of its
    // tables -- the activations' 8-bit range -- and the rest are sized to nothing.
    let mut tables = Tables::new(vec![0u32; 16], 4, 0, Round::Even, 4, vec![ACT_BITS], 0, 0);
    let r = freivalds_component(w, p_target, ACT_BITS, &mut tables, None, None);
    let trace_secs = t0.elapsed().as_secs_f64();
    let lay = r.lay;
    let sh = FreivaldsShape::new(w.name, &lay, w.mats.len(), false);

    // the two challenges, kept for the self-check below: they are drawn from
    // the session inside, because that is where the transcript is
    let mut drawn = None;
    let alone = prove_alone(
        Trees {
            // the committed order is the row declaration's, not this file's
            ids: Known::committed_ids(&sh),
            pre: r.pre.clone(),
            base: r.base.clone(),
            pre_cells: r.pre_cells(),
            base_cells: r.base_cells(),
        },
        &[RangeTable::of(ACT_BITS, &tables)],
        |sess, rels| {
            let ch = sess.draw_felts(2);
            drawn = Some((ch[0], ch[1]));
            let (rv, s) = (Powers::against(ch[0], lay.pk), Powers::against(ch[1], lay.pq));
            let air = |claims| FreivaldsEval {
                shape: sh.clone(),
                rv: rv.clone(),
                s: s.clone(),
                claims,
                // the committed activations, as bytes of this proof's own range table
                acts: ByteCells { bits: ACT_BITS, off: 1 << (ACT_BITS - 1),
                                  rel: rels[0].clone(), by: Lookups::Own },
                // the activation claim binds a group to whichever component produced its
                // input; a weight block on its own has no producer and publishes none,
                // which is what a composed segment without the residual stream also does
                qx: None,
            };
            let it = freivalds_interaction(w, &r, &air(FreivaldsClaims::open(w.mats.len(), false)));
            After { air: air(it.claims.clone()), cols: it.cols, use_sum: it.use_sum,
                    claims: it.claims.product }
        });

    // The other side of the identity, recomputed on the host: `s^T C rv` over
    // the exact integer accumulators the witness carries.  In a composed proof
    // this value is the dequant component's published claim and *is* proved; here
    // nothing proves it, so this is a self-check and `accepts()` says so.
    let (rv, s) = drawn.expect("the standalone run draws its two challenges");
    let (mut ref_ok, ref_n) = (0usize, w.mats.len());
    for (m, claimed) in w.mats.iter().zip(&alone.claims) {
        if project_product(m.c, w.tokens, m.n_out, rv, s) == *claimed {
            ref_ok += 1;
        }
    }

    Outcome {
        alone,
        lay,
        group: w.name.to_string(),
        n_mats: w.mats.len(),
        n_in: w.n_in,
        n_out: w.n_out_max(),
        tokens: w.tokens,
        weights: w.n_weights(),
        off_table: tables.off_table,
        ref_ok,
        ref_n,
        trace_secs,
    }
}

/// One point of the sweep: the same weight block at two output widths, and the cells
/// the weights they differ by cost.
///
/// The two builds have to sit on the same row width and commit the same activations,
/// or the difference is not the weights' alone; both are asserted rather than
/// assumed, and a shape that breaks either is a shape this sweep cannot measure.
pub fn marginal(small: &Outcome, large: &Outcome) -> Value {
    assert_eq!(small.lay.pk, large.lay.pk,
               "the two builds landed on {} and {} weights a row: the sweep needs an \
                output width the target divides",
               small.lay.pk, large.lay.pk);
    assert!(large.weights > small.weights,
            "the second build has to carry more weights than the first");
    assert_eq!(small.alone.base_cells, large.alone.base_cells,
               "the two builds commit {} and {} activation cells: the difference is \
                then not the weights' alone, which needs an exact token tiling \
                (`tokens` a multiple of the row-chunk count at both widths)",
               small.alone.base_cells, large.alone.base_cells);
    let dw = (large.weights - small.weights) as f64;
    let fresh = (large.fresh() - small.fresh()) as f64;
    let pre = (large.alone.pre_cells - small.alone.pre_cells) as f64;
    let inter = (large.alone.inter_cells - small.alone.inter_cells) as f64;
    json!({
        "weights_per_row": large.lay.pk,
        "shape": format!("{} n_in={} n_out={},{} tokens={}", large.group, large.n_in,
                         small.n_out, large.n_out, large.tokens),
        "s": large.tokens,
        "group": large.group, "matmuls": large.n_mats,
        "n_in": large.n_in, "tokens": large.tokens,
        "weights": [small.weights, large.weights],
        // the name `harness/costmodel/cost_model.py` reads the design point's
        // weights per row under
        "layout_x": large.layout_json(),
        "cells": {
            // what one more weight costs: the number the cost model multiplies by a
            // parameter count
            "cells_per_weight": fresh / dw,
            "preprocessed_per_weight": pre / dw,
            "interaction_per_weight": inter / dw,
            // the whole component divided by its weights, activations and their
            // range check included -- not a marginal, and not what the model charges
            "fresh_per_weight_with_activations": large.fresh() as f64 / large.weights as f64,
            "fresh_per_token": large.fresh() as f64 / large.tokens as f64,
        },
        // what the two builds differ by, before the division
        "difference": {
            "weights": large.weights - small.weights,
            "rows": large.lay.g.rows - small.lay.g.rows,
            "preprocessed": large.alone.pre_cells - small.alone.pre_cells,
            "base": large.alone.base_cells - small.alone.base_cells,
            "interaction": large.alone.inter_cells - small.alone.inter_cells,
            "fresh": large.fresh() - small.fresh(),
        },
        "proof_bytes": large.alone.proven.proof_bytes,
        "prove_secs": large.alone.proven.prove_secs,
        "accepts": small.accepts() && large.accepts(),
        "off_table": small.off_table + large.off_table,
        "reference_checked": small.ref_n + large.ref_n,
        "reference_mismatch": (small.ref_n - small.ref_ok) + (large.ref_n - large.ref_ok),
        "builds": [small.json(), large.json()],
    })
}
