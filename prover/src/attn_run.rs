//! Prove / verify a band of `insts` attention heads sharing one column set, and
//! report what it cost.
//!
//! The three application-level checks are *identical* to B5's even though the proof
//! now covers many instances, because every region carries the same per-instance
//! weight (`attn_air`'s module doc): one claimed value per identity, whatever the
//! band size.
//!
//! Three application-level checks sit on top of stwo's `verify()`, because none
//! of them is a constraint (B2 section 4, B3 section 1):
//!
//!   Q.K^T    (sum_c r^c)(sum_t s^t m_t) - hs  ==  chan     (v1: hs == chan)
//!   A.V      av  ==  out
//!   denom    sn  ==  az
//!
//! plus the LogUp claimed sums, which have to cancel or the exp table and every
//! range check prove nothing at all.

use std::time::Instant;

use itertools::Itertools;
use num_traits::Zero;
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::attn_air::{
    ChanEval, ExpRel, ExpTableEval, KeyEval, Offsets, OutEval, Powers, RangeRel, RangeTableEval,
    RowEval, band_id, id, msk_id, tbl_range, CHAN_SEL, CORNER_SEL, KEY_SEL, OUT_SEL, ROW_SEL,
    TBL_EXP_X, TBL_EXP_Y,
};
use crate::common::{Proven, Session, Sums, twiddles};
use crate::layout::Cells;
use crate::attn_trace::{self, Cfg, Layouts};
use crate::attn_witness::{Attn, Variant};

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub cells: Cells,
    pub trace_secs: f64,
    pub interaction_secs: f64,
    pub layouts: Layouts,
    pub n_columns: [usize; 3],
    /// the Q.K^T identity
    pub qk_ok: bool,
    /// the A.V identity
    pub av_ok: bool,
    /// sum_t s^t Z_t == sum_c sum_t s^t n[t, c]
    pub den_ok: bool,
    /// committed values no table can absorb
    pub off_table: usize,
    pub claims: Vec<(String, SecureField)>,
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.qk_ok && self.av_ok
            && self.den_ok && self.off_table == 0
    }

    pub fn json(&self, a: &Attn, cfg: &Cfg) -> Value {
        let sh = a.shape;
        let p = &self.proven;
        let (obs, committed, blind) = a.av_bounds();
        let grid = sh.scores() as f64;
        json!({
            "shape": sh.label(), "s": sh.s, "c": sh.c, "hd": sh.hd, "insts": sh.insts,
            // the column budget: what packing buys, and what it costs per instance
            "columns_per_instance": self.n_columns.iter().sum::<usize>() as f64
                / sh.insts as f64,
            "proof_bytes_per_instance": self.proven.proof_bytes as f64 / sh.insts as f64,
            "exp_table": {"shared_across_instances": a.table_shared(),
                          "frac_master": a.frac,
                          "frac_per_instance": a.frac_per_inst,
                          "key_scale_per_instance": a.delta,
                          "log_rows": a.log_table},
            "variant": a.variant.name(), "source": a.source,
            "q_bits": a.q_bits, "k_bits": a.k_bits, "nmax": a.nmax,
            "log_exp_table": a.log_table, "frac_exponent": a.frac,
            "layout": {"chan": self.layouts.chan.json(), "grid": self.layouts.key.json(),
                       "out": self.layouts.out.json(),
                       "log_rows_tokens": self.layouts.log_row, "tokens_per_row": self.layouts.rw},
            "cells": self.cells.json(&sh),
            // the acceptance number, and the part of it that is the grid alone
            "alpha": self.cells.fresh() as f64 / grid,
            "alpha_grid": self.cells.grid as f64 / grid,
            "alpha_without_tables":
                (self.cells.fresh() - 5 * (1usize << a.log_table)) as f64 / grid,
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "trace_secs": self.trace_secs, "interaction_secs": self.interaction_secs,
            "prove_secs": p.prove_secs, "verify_secs": p.verify_secs,
            "secs_per_token": p.prove_secs / sh.s as f64,
            "proof_bytes": p.proof_bytes, "peak_rss_bytes": p.peak_rss,
            "fresh_cells_per_sec": self.cells.fresh() as f64 / p.prove_secs,
            "d_bits": a.d_bits(), "off_table": self.off_table,
            "witness_off_table": a.off_table,
            "numerator_zero_frac":
                a.num.iter().filter(|n| **n == 0).count() as f64 / a.num.len() as f64,
            "torch_numerator_mismatch": a.torch_mismatch,
            // which bound the A.V integer reading rests on
            "av_window": {"observed": obs, "committed_sum_n": committed, "blind": blind,
                          "limb_window_2p29": 1i64 << 29,
                          "fits_committed": committed < (1i64 << 29),
                          "fits_blind": blind < (1i64 << 29)},
            // The AIR's own bound on the A.V accumulator: V is range-checked into
            // [-128, 127] and Z into [0, 2^bits), so |A.V| < 128 * 2^bits.  That is
            // the number the dequant's limb decomposition needs, and it is a
            // *proven* bound, not the blind worst case.
            "denominator": {"z_max": a.zz.iter().copied().max().unwrap_or(0),
                            "range_bits": cfg.z_lo_bits + cfg.z_hi_bits,
                            "z_min": cfg.z_min,
                            "z_min_holds": a.zz.iter().all(|z| *z >= cfg.z_min as i64),
                            "av_bound_from_air": 128i64 << (cfg.z_lo_bits + cfg.z_hi_bits),
                            "inside_limb_window":
                                (128i64 << (cfg.z_lo_bits + cfg.z_hi_bits)) <= (1i64 << 29),
                            "blind_worst_case_fits":
                                (a.nmax as i64 * sh.c as i64)
                                    < (1i64 << (cfg.z_lo_bits + cfg.z_hi_bits))},
            "checks": {"stark": p.stark_ok, "qk": self.qk_ok, "av": self.av_ok,
                       "denominator": self.den_ok, "sums_cancel": self.sums.cancels(),
                       "off_table": self.off_table},
            "claims": self.claims.iter().map(|(k, v)| json!({"name": k, "value": format!("{v:?}")}))
                .collect::<Vec<_>>(),
            "accepts": self.accepts(),
            "phases": p.phases, "prove_error": p.prove_error,
        })
    }
}

pub fn run(a: &Attn, cfg: &Cfg) -> Outcome {
    let sh = a.shape;
    let l = Layouts::choose(&sh, cfg);
    let widths = attn_trace::range_widths(a, cfg);

    let t0 = Instant::now();
    let com = attn_trace::commit_phase(a, &l, cfg);
    let trace_secs = t0.elapsed().as_secs_f64();

    // The composition polynomial's degree bound is `max(log_size) + 1` over
    // *every* component, so the twiddle tree has to cover the widest column in
    // the proof -- which is not always a region or the exp table: a 2^16 range
    // table outgrows both whenever the shift is split finely enough.  Getting
    // this wrong surfaces as "Not enough twiddles!" from inside the prover.
    let widest = l.max_log().max(a.log_table).max(widths.iter().copied().max().unwrap_or(0));
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);
    sess.commit(com.pre.clone());
    sess.commit(com.base.clone());
    let rel_exp = sess.draw_relation(ExpRel::draw);
    let rels: Vec<RangeRel> = widths.iter().map(|_| sess.draw_relation(RangeRel::draw)).collect();
    let ch = sess.draw_felts(3);
    let (r, s, rv) = (ch[0], ch[1], ch[2]);
    let pow = Powers::new(r, s, rv, &sh, &l.chan, &l.key, &l.out, l.rw);

    let t1 = Instant::now();
    // The standalone path has no `Plan`, so no components and no evals: it
    // keeps the hand-written builder, which is also what mode 2 compares against.
    let mut inter = attn_trace::interaction_phase(a, &l, cfg, &pow, &rel_exp, &rels, &com,
                                                  &widths, true, true, None, None);
    let pre_x = attn_trace::pre_exp_x(a, &l);
    let mult0 = com.at[4];
    let pre_ranges: Vec<&crate::common::Eval> =
        (0..widths.len()).map(|t| &com.pre[pre_x + 2 + t]).collect();
    let mults: Vec<&crate::common::Eval> =
        (0..1 + widths.len()).map(|t| &com.base[mult0 + t]).collect();
    let (tcols, tsums) = attn_trace::table_sides(a, &widths, &rel_exp, &rels,
                                                 [&com.pre[pre_x], &com.pre[pre_x + 1]],
                                                 &pre_ranges, &mults);
    inter.cols.extend(crate::device::host_cols(tcols));
    inter.table_sums = tsums;
    let interaction_secs = t1.elapsed().as_secs_f64();
    let n_inter = inter.cols.len();
    sess.commit(inter.cols.iter().map(|c| c.host().clone()).collect());

    // The claimed accumulator values are public input; bind them to the transcript
    // before the components fix them, or a prover picks them adaptively.
    let claims_v = vec![inter.claimed_chan, inter.claimed_hs, inter.claimed_av,
                        inter.claimed_sn, inter.claimed_out, inter.claimed_am,
                        inter.claimed_az];
    sess.mix(&claims_v);

    let off = Offsets::new(a.q_bits, a.k_bits);
    let ridx = |b: u32| widths.iter().position(|w| *w == b).unwrap();
    let mut ids = vec![];
    ids.extend(CHAN_SEL.iter().map(|n| band_id(0, n)));
    if a.corner {
        ids.extend(CORNER_SEL.iter().map(|n| band_id(0, n)));
        ids.extend((0..l.key.q).map(|i| msk_id(0, i)));
    } else {
        ids.extend(KEY_SEL.iter().map(|n| band_id(0, n)));
    }
    ids.extend(OUT_SEL.iter().map(|n| band_id(0, n)));
    ids.extend(ROW_SEL.iter().map(|n| band_id(0, n)));
    ids.push(id(TBL_EXP_X));
    ids.push(id(TBL_EXP_Y));
    ids.extend(widths.iter().map(|w| tbl_range(*w)));
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);

    let mut k = 0usize;
    let mut next_use = || {
        let v = inter.use_sums[k];
        k += 1;
        v
    };
    let chan = FrameworkComponent::<ChanEval>::new(
        alloc,
        ChanEval { tag: 0, lay: l.chan, pow: pow.clone(), off, claimed: inter.claimed_chan,
                   rel_k: rels[ridx(a.k_bits)].clone(), rel_q: rels[ridx(a.q_bits)].clone(),
                   range_check: cfg.range_check, gkr: false },
        if cfg.range_check { next_use() } else { SecureField::zero() },
    );
    let key = FrameworkComponent::<KeyEval>::new(
        alloc,
        KeyEval { tag: 0, lay: l.key, pow: pow.clone(), off, variant: a.variant,
                  claimed_hs: inter.claimed_hs, claimed_av: inter.claimed_av,
                  claimed_sn: inter.claimed_sn, rel_exp: rel_exp.clone(),
                  rel_v: rels[ridx(8.min(a.k_bits))].clone(),
                  rel_split: match a.variant {
                      Variant::IntSplit(b) => Some(rels[ridx(b)].clone()),
                      _ => None,
                  },
                  range_check: cfg.range_check, gkr: false, corner: a.corner },
        next_use(),
    );
    let out = FrameworkComponent::<OutEval>::new(
        alloc,
        OutEval { tag: 0, lay: l.out, pow: pow.clone(), claimed: inter.claimed_out },
        SecureField::zero(),
    );
    let rows = FrameworkComponent::<RowEval>::new(
        alloc,
        RowEval { tag: 0, log_rows: l.log_row, rw: l.rw, pow: pow.clone(),
                  with_max: a.variant != Variant::Bf16x,
                  claimed_am: inter.claimed_am, claimed_az: inter.claimed_az,
                  z_lo_bits: cfg.z_lo_bits, z_min: cfg.z_min,
                  rel_lo: rels[ridx(cfg.z_lo_bits)].clone(),
                  rel_hi: rels[ridx(cfg.z_hi_bits)].clone(), gkr: false },
        next_use(),
    );
    let exp_tbl = FrameworkComponent::<ExpTableEval>::new(
        alloc, ExpTableEval { log_rows: a.log_table, rel: rel_exp.clone() }, inter.table_sums[0]);
    let range_tbls: Vec<FrameworkComponent<RangeTableEval>> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| FrameworkComponent::<RangeTableEval>::new(
            alloc, RangeTableEval { bits: *w, rel: rels[i].clone() }, inter.table_sums[i + 1]))
        .collect();

    let mut provers: Vec<&dyn ComponentProver<_>> = vec![&chan, &key, &out, &rows, &exp_tbl];
    let mut components: Vec<&dyn Component> = vec![&chan, &key, &out, &rows, &exp_tbl];
    for t in &range_tbls {
        provers.push(t);
        components.push(t);
    }
    let proven = sess.prove(&provers, &components);

    // ---- the application-level checks --------------------------------------
    let qk_lhs = match a.variant {
        Variant::Bf16x => inter.claimed_hs,
        _ => pow.r_sum * inter.claimed_am - inter.claimed_hs,
    };
    let qk_ok = qk_lhs == inter.claimed_chan;
    let av_ok = inter.claimed_av == inter.claimed_out;
    let den_ok = inter.claimed_sn == inter.claimed_az;
    let sums = Sums {
        uses: inter.use_sums.iter().fold(SecureField::zero(), |x, y| x + *y),
        tables: inter.table_sums.iter().fold(SecureField::zero(), |x, y| x + *y),
    };

    let mut cells = attn_trace::cells(a, &l, cfg);
    let real: usize = com.pre.iter().map(|e| e.domain.size()).sum();
    assert_eq!(cells.preprocessed, real, "preprocessed cell model");
    let real: usize = com.base.iter().map(|e| e.domain.size()).sum();
    assert_eq!(cells.base, real, "base cell model");
    let real: usize = inter.cols.iter().map(|c| c.size()).sum();
    assert_eq!(cells.interaction, real, "interaction cell model");
    cells.grid = cells.grid.min(cells.fresh());

    let names = ["chan", "hs", "av", "sn", "out", "am", "az"];
    Outcome {
        proven,
        sums,
        cells,
        trace_secs,
        interaction_secs,
        layouts: l,
        n_columns: [com.pre.len(), com.base.len(), n_inter],
        qk_ok,
        av_ok,
        den_ok,
        off_table: com.off_table,
        claims: names.iter().zip(claims_v).map(|(n, v)| (n.to_string(), v)).collect_vec(),
    }
}
