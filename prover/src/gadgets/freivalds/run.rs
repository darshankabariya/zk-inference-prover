//! Prove the weight Freivalds check on its own over a weight block, and report what
//! it costs in committed cells per weight.
//!
//! The component is the composed segment's own: `trace::freivalds_component` builds the
//! trace and `claims::freivalds_interaction` the accumulators and the LogUp columns,
//! exactly as `compose` calls them.  Nothing else is in the proof but the range
//! table the committed activations are checked against, which is a shared table and
//! is reported apart.
//!
//! What the cost model multiplies by a parameter count is the *marginal*: how many
//! fresh cells one more weight costs.  A group's grid runs one row group per input
//! channel, cut into `chunks` chunks, and a row carries `pk` weights of each matmul
//! and `pq` committed activations.  So the committed activations are `tokens x n_in`
//! whatever `chunks` is, their range check is two more cells each at pairwise
//! batching, and the only thing that grows with the weights is the three QM31
//! accumulator columns `v`, `u` and `acc` -- 12 M31 cells a row for a group of one
//! matmul.  A row carries `pk` weights, so the marginal is `12/pk` and the two fixed
//! parts drop out of a difference.  `run` proves one build; `marginal` takes two
//! builds of the same block at two output widths and reports the difference over the
//! weights they differ by, so nothing here has to be assumed.

use std::time::Instant;

use num_traits::Zero;
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::encoding::m31_of;
use crate::relations::{RangeRel, RangeTableEval, powers, table_col, tbl_range};
use crate::bf16::Round;
use crate::common::{Proven, Session, Sums, eval_of, m31_u32, twiddles};
use crate::compose::Rels;
use crate::gadgets::freivalds::{FREIVALDS_SEL, FreivaldsEval, freivalds_sel, freivalds_w};
use crate::gadgets::freivalds::claims::freivalds_interaction;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::gadgets::freivalds::trace::freivalds_component;
use crate::layer::{Layer, Tables};

/// The activation width every Freivalds group but the widened attention output's
/// commits at, and the width of the range table this proof carries.
const ACT_BITS: u32 = 8;

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub lay: FreivaldsLayout,
    pub group: String,
    pub n_mats: usize,
    pub n_in: usize,
    pub n_out: usize,
    pub tokens: usize,
    pub weights: usize,
    /// the component's own three trees, the shared range table left out
    pub pre_cells: usize,
    pub base_cells: usize,
    pub inter_cells: usize,
    /// the range table: its preprocessed rows, and its multiplicity column plus
    /// its own LogUp column
    pub table_cells_pre: usize,
    pub table_cells_fresh: usize,
    pub off_table: usize,
    /// claims that match the Freivalds identity recomputed from the exact
    /// integer accumulators
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
    /// Fresh cells the component commits, the shared table left out.
    pub fn fresh(&self) -> usize {
        self.base_cells + self.inter_cells
    }
    pub fn json(&self) -> Value {
        json!({
            "group": self.group, "matmuls": self.n_mats,
            "n_in": self.n_in, "n_out": self.n_out, "tokens": self.tokens,
            "weights": self.weights,
            "layout_x": self.layout_json(),
            "cells": {"preprocessed": self.pre_cells, "base": self.base_cells,
                      "interaction": self.inter_cells, "fresh": self.fresh(),
                      "fresh_per_weight": self.fresh() as f64 / self.weights as f64,
                      "fresh_per_token": self.fresh() as f64 / self.tokens as f64,
                      "table_preprocessed": self.table_cells_pre,
                      "table_fresh": self.table_cells_fresh},
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "off_table": self.off_table,
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "trace_secs": self.trace_secs, "interaction_secs": self.interaction_secs,
            "proof_bytes": self.proven.proof_bytes,
            "sums": self.sums.json(), "accepts": self.accepts(),
            "point": self.proven.json(),
        })
    }
    fn layout_json(&self) -> Value {
        let mut v = self.lay.g.json();
        v["weights_per_row"] = json!(self.lay.pk);
        v["activations_per_row"] = json!(self.lay.pq);
        v
    }
}

/// The relation set `freivalds_interaction` takes.  The component reads exactly one
/// of them -- the activations' range check, drawn from this proof's own transcript --
/// so the rest are drawn from a channel of their own: no column here combines
/// against them, and a composed segment draws them for components this proof does
/// not build.
fn rels_of(range: &RangeRel) -> Rels {
    let mut spare = Blake2sM31Channel::default();
    let mut r = Rels::draw(&mut spare, &[ACT_BITS], None, None, false, 0);
    r.ranges = vec![range.clone()];
    r
}

/// One weight block's check, proved on its own at `p_target` weights a row.
pub fn run(layer: &Layer, group: &str, mats: &[usize], p_target: usize) -> Outcome {
    let m0 = &layer.mats[mats[0]];
    let (n_in, tokens) = (m0.n_in, m0.tokens);
    let n_out = mats.iter().map(|i| layer.mats[*i].n_out).max().unwrap();

    let t0 = Instant::now();
    // The shared table registry.  This proof looks values up in exactly one of its
    // tables -- the activations' 8-bit range -- and the rest are sized to nothing.
    let mut tables = Tables::new(vec![0u32; 16], 4, 0, Round::Even, 4, vec![ACT_BITS], 0, 0);
    let r = freivalds_component(layer, group, mats, p_target, ACT_BITS, &mut tables, None, None);
    let lay = r.lay;
    let trace_secs = t0.elapsed().as_secs_f64();

    let ridx = tables.ridx(ACT_BITS);
    let tbl = eval_of(ACT_BITS, (0..1u32 << ACT_BITS).map(m31_u32).collect());
    let mult = eval_of(ACT_BITS, tables.m_r[ridx].iter().map(|v| m31_u32(*v)).collect());

    let tw = twiddles(lay.g.log_rows.max(ACT_BITS));
    let mut sess = Session::new(&tw);
    let mut pre = r.pre.clone();
    pre.push(tbl.clone());
    let n_pre = pre.len();
    sess.commit(pre);
    // A gadget on its own has no model column, but the tree layout is the
    // same for every proof this crate makes: the empty model tree keeps the
    // base trace at the index every component's mask names.
    sess.commit_model_empty();
    let mut base = r.base.clone();
    base.push(mult.clone());
    let n_base = base.len();
    sess.commit(base);
    let rel_act = sess.draw_relation(RangeRel::draw);
    let ch = sess.draw_felts(2);
    let (rv, s_ch) = (ch[0], ch[1]);

    let (rv_pow, rv_step) = powers(rv, lay.pk);
    let (s_pow, s_step) = powers(s_ch, lay.pq);
    let air = |claimed: Vec<SecureField>| FreivaldsEval {
        tag: group.to_string(),
        g: lay.g,
        pk: lay.pk,
        pq: lay.pq,
        n_mats: mats.len(),
        rv_pow: rv_pow.clone(),
        rv_step,
        s_pow: s_pow.clone(),
        s_step,
        claimed,
        rel_act: rel_act.clone(),
        act_off: 1 << (ACT_BITS - 1),
        range_check: true,
        gkr: false,
        // the activation claim binds a group to whichever component produced its
        // input; a weight block on its own has no producer and publishes none,
        // which is what a composed chunk without the residual stream also does
        qx: None,
    };
    let rels = rels_of(&rel_act);

    let t1 = Instant::now();
    let it = freivalds_interaction(layer, &r, rv, s_ch, &rels, true,
                                   &air(vec![SecureField::zero(); mats.len()]), false);
    let interaction_secs = t1.elapsed().as_secs_f64();
    let inter_cells: usize = it.cols.iter().map(|c| c.size()).sum();
    let n_inter_cols = it.cols.len();
    let (tcols, tsum) = table_col(ACT_BITS, &rel_act, &[&tbl], &mult);
    let table_cells_fresh =
        mult.domain.size() + tcols.iter().map(|e| e.domain.size()).sum::<usize>();
    let n_tcols = tcols.len();
    let mut icols = it.cols;
    icols.extend(crate::device::host_cols(tcols));
    sess.commit_any(icols);

    // The claimed accumulators are public input; bind them to the transcript before
    // the components fix them, or a prover picks them adaptively.
    sess.mix(&it.claimed);

    let mut ids = vec![];
    for (mi, _) in mats.iter().enumerate() {
        for pp in 0..lay.pk {
            ids.push(freivalds_w(group, mi, pp));
        }
    }
    ids.extend(FREIVALDS_SEL.iter().map(|n| freivalds_sel(group, n)));
    ids.push(tbl_range(ACT_BITS));
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let fr = FrameworkComponent::<FreivaldsEval>::new(alloc, air(it.claimed.clone()), it.use_sum);
    let tb = FrameworkComponent::<RangeTableEval>::new(
        alloc, RangeTableEval { bits: ACT_BITS, rel: rel_act }, tsum);
    let provers: Vec<&dyn ComponentProver<_>> = vec![&fr, &tb];
    let components: Vec<&dyn Component> = vec![&fr, &tb];
    let proven = sess.prove(&provers, &components);

    // The Freivalds identity itself, recomputed from the exact integer accumulators
    // the witness carries: `acc_m == sum_{t, i} s^t rv^i acc[t, i]`, which is what a
    // composed chunk compares each claim against on the dequant's side.
    let (mut ref_ok, ref_n) = (0usize, mats.len());
    for (mi, claimed) in mats.iter().zip(&it.claimed) {
        let m = &layer.mats[*mi];
        let (rvp, _) = powers(rv, m.n_out);
        let (sp, _) = powers(s_ch, m.tokens);
        let mut h = SecureField::zero();
        for t in 0..m.tokens {
            let mut row = SecureField::zero();
            for i in 0..m.n_out {
                row += rvp[i] * SecureField::from(m31_of(m.acc[t * m.n_out + i]));
            }
            h += sp[t] * row;
        }
        if h == *claimed {
            ref_ok += 1;
        }
    }

    Outcome {
        sums: Sums { uses: it.use_sum, tables: tsum },
        proven,
        lay,
        group: group.to_string(),
        n_mats: mats.len(),
        n_in,
        n_out,
        tokens,
        weights: r.n_weights,
        pre_cells: r.pre_cells(),
        base_cells: r.base_cells(),
        inter_cells,
        table_cells_pre: tbl.domain.size(),
        table_cells_fresh,
        off_table: tables.off_table,
        ref_ok,
        ref_n,
        trace_secs,
        interaction_secs,
        n_columns: [n_pre, n_base, n_inter_cols + n_tcols],
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
    assert_eq!(small.base_cells, large.base_cells,
               "the two builds commit {} and {} activation cells: the difference is \
                then not the weights' alone, which needs an exact token tiling \
                (`tokens` a multiple of the chunk count at both widths)",
               small.base_cells, large.base_cells);
    let dw = (large.weights - small.weights) as f64;
    let fresh = (large.fresh() - small.fresh()) as f64;
    let pre = (large.pre_cells - small.pre_cells) as f64;
    let inter = (large.inter_cells - small.inter_cells) as f64;
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
            "preprocessed": large.pre_cells - small.pre_cells,
            "base": large.base_cells - small.base_cells,
            "interaction": large.inter_cells - small.inter_cells,
            "fresh": large.fresh() - small.fresh(),
        },
        "proof_bytes": large.proven.proof_bytes,
        "prove_secs": large.proven.prove_secs,
        "accepts": small.accepts() && large.accepts(),
        "off_table": small.off_table + large.off_table,
        "reference_checked": small.ref_n + large.ref_n,
        "reference_mismatch": (small.ref_n - small.ref_ok) + (large.ref_n - large.ref_ok),
        "builds": [small.json(), large.json()],
    })
}
