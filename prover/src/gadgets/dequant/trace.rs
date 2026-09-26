//! The matmul dequant component: its witness, its committed columns and the
//! reference semantics they are checked against.
//!
//! This is the trace builder a segment composes, as against `dequant::run`'s,
//! which lays out its own columns for the standalone rung (#199 makes the two
//! one). `mat_component` mirrors `run`'s witness loop without the tampers,
//! because a compose probe corrupts the layer witness rather than a column.
//!
//! `deq_ref` and `fold_out` are the same rung's reference semantics, and they
//! live beside the builder as one function each so that a consumer of a matmul's
//! output -- the residual stream's SwiGLU reads `gate` and `up` -- reads exactly
//! the value this component proves rather than a second implementation of it.

use num_traits::Zero;
use rayon::prelude::*;
use stwo::core::fields::m31::M31;

use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::encoding::m31_of;
use crate::gadgets::dequant::{Map, Spec, Variant as DeqVariant};
use crate::layer::Tables;
use crate::layout::MatLayout;

/// One matmul's dequant + Freivalds witness.
pub struct Mat {
    pub name: String,
    pub variant: DeqVariant,
    pub n_in: usize,
    pub n_out: usize,
    pub tokens: usize,
    /// (tokens, n_out) row-major exact integer accumulators
    pub acc: Vec<i64>,
    /// (tokens, n_in) quantized activations
    pub qx: Vec<i32>,
    /// (n_in, n_out) quantized weights
    pub qw: Vec<i32>,
    /// per-token / per-channel power-of-two scale exponents (pow2 variant)
    pub jx: Vec<i32>,
    pub jw: Vec<i32>,
    /// per-token / per-channel bf16 scales as (mantissa field, biased exponent)
    pub sx: Vec<(u32, u32)>,
    pub sw: Vec<(u32, u32)>,
    /// Stream Z, the fold: the consumer's power-of-two scale exponent per
    /// output channel (`s_v` for the `v` matmul, collapsed from the attention
    /// witness's per-query-head copy).  Empty on every other variant.
    pub jv2: Vec<i32>,
}

impl Mat {
    /// The fold's divisor exponents (`dequant.rs`'s `Fold` arm): the token
    /// side `G_t = 134 - es_x` and the channel side `G_c = 134 - ew + jv`.
    pub fn fold_gt(&self, t: usize) -> i64 {
        134 - self.sx[t].1 as i64
    }
    pub fn fold_gc(&self, c: usize) -> i64 {
        134 - self.sw[c].1 as i64 + self.jv2[c] as i64
    }
    /// The largest |acc|, and the `kmax` the shared normalisation table needs.
    pub fn kmax_needed(&self) -> u32 {
        let m = self.acc.iter().map(|a| a.unsigned_abs()).max().unwrap_or(0);
        (64 - m.leading_zeros()).saturating_sub(9)
    }
    /// The blind worst case: `127^2 * n_in`.
    pub fn kmax_blind(&self) -> u32 {
        let m = 127u64 * 127 * self.n_in as u64;
        (64 - m.leading_zeros()).saturating_sub(9)
    }
}

/// A component's committed and preprocessed columns, and what its interaction
/// builder needs to remember.
pub struct MatComponent {
    pub lay: MatLayout,
    pub spec: Spec,
    pub map: Map,
    /// preprocessed: the per-channel scale columns (chan cols), then selectors
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// f64-reference mismatches (must be 0)
    pub ref_bad: usize,
    pub n: usize,
    /// B7a milestone 4 (19.6): the same columns, born on the device. `Some`
    /// only under `ZK_DEVICE_WITNESS_GEN_DEQUANT=1`, where `pre` and `base` are empty and this is
    /// the component.
    pub dev: Option<crate::gadgets::dequant::device::DevDequantWitnessGen>,
}

impl MatComponent {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev {
            Some(d) => d.base.len(),
            None => self.base.len(),
        };
        n << self.lay.log_rows
    }
}

/// Build one matmul's dequant component: witness columns in the compose layout,
/// multiplicities bumped into the shared tables.  Mirrors `dequant::run`'s
/// witness loop without the tampers (compose probes corrupt the layer witness).
/// Stream SS: `ohead` is `Some(rows per kv head)` on the one component whose output
/// side folds under the *key tile* grid -- a folded `v`, whose published output is an
/// integer and whose counterparty is the key tiles' own V (`dequant::OKind::Int`).
/// It changes nothing but which rows the `oin`/`obd` step selectors mark, so no
/// preprocessed column is added and no other component moves.
pub fn mat_component(m: &Mat, spec: &Spec, tables: &mut Tables, output_claim: bool,
                  ohead: Option<usize>,
                  fold_bend: (Option<usize>, Option<usize>)) -> MatComponent {
    assert!(spec.sound(), "unsound spec for {}", m.name);
    let lay = MatLayout::choose_or_padded(m.n_out, m.tokens, spec.per_row);
    let mut spec = spec.clone();
    spec.per_row = lay.per_row;
    let map = Map::new(&spec);
    let rows = 1usize << lay.log_rows;
    let sticky = spec.round.needs_sticky();
    let f = map.f;
    let n = m.tokens * m.n_out;

    // B7a milestone 4 (19.6): which witness generation runs. 1 is the device's alone -- the
    // component has no host columns at all and `pre`/`base` stay empty -- and 2
    // builds both and asserts every cell equal before either is used.
    //
    // Stream Z: the fold has no device witness generation; its component takes the host path
    // whatever the switch says (the diagonal block key tiles' pattern), and
    // `devpre::note_host_witness_gen` is told in `generate`.
    let witness_gen_mode = if spec.variant == DeqVariant::Fold {
        0
    } else {
        crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_DEQUANT")
    };
    let cross_check = witness_gen_mode == 2;
    if witness_gen_mode == 1 {
        // B11 stream M: `output_claim` is the component's, not `with_want`'s. The two
        // parameters sit next to each other and the o-side one was missing
        // entirely, so the head's device witness generation built 506 preprocessed columns
        // where the host builds 510.
        let dev = crate::gadgets::dequant::device::mat_component_dev(m, &spec, &lay, &map, tables, false, output_claim);
        let ref_bad = dev.ref_bad;
        return MatComponent { lay, spec, map, pre: vec![], base: vec![], ref_bad, n,
                           dev: Some(dev) };
    }
    let snap_tables = |t: &Tables| -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let r8 = t.ridx(8);
        (t.m_norm.clone(), t.m_frac.clone(), t.m_mul.clone(), t.m_r[r8].clone(),
         t.off_table)
    };
    let snap_before = cross_check.then(|| snap_tables(tables));

    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; map.n_cols];
    // parallel over row blocks: each block fills its slice of every column and
    // accumulates its own table multiplicities, merged serially below
    let block = 2048usize;
    struct Bumps {
        norm: Vec<(usize, u32)>,
        frac: Vec<(usize, u32)>,
        mul: Vec<(usize, u32)>,
        r8: Vec<(i64, u32)>,
        /// stream Z, the fold: the (q, nz) pair table and the width-keyed
        /// range bumps (13, 16 and the fold's own remainder width)
        qnz: Vec<(i64, u32)>,
        rw: Vec<((u32, i64), u32)>,
        ref_bad: usize,
        /// B7a milestone 4 (19.6): the f64 cross-check's own bf16 answer per
        /// (row, slot), so `ZK_DEVICE_WITNESS_GEN_DEQUANT=2` can compare the one part of the witness generation
        /// that is not integer arithmetic cell for cell instead of comparing
        /// two counts that are both zero on an honest witness.
        want: Vec<u16>,
    }
    let starts: Vec<usize> = (0..rows).step_by(block).collect();
    let tref = &*tables;
    let out: Vec<(Vec<Vec<M31>>, Bumps)> = starts
        .par_iter()
        .map(|start| {
            let len = block.min(rows - start);
            let mut local: Vec<Vec<M31>> = vec![vec![M31::zero(); len]; map.n_cols];
            let mut bn = std::collections::HashMap::new();
            let mut bf2 = std::collections::HashMap::new();
            let mut bm = std::collections::HashMap::new();
            let mut br = std::collections::HashMap::new();
            let mut bq = std::collections::HashMap::new();
            let mut brw = std::collections::HashMap::new();
            let mut ref_bad = 0usize;
            let mut want = vec![0u16; if cross_check { len * lay.per_row } else { 0 }];
            for jj in 0..len {
                let j = start + jj;
                witness_gen_mat_row(m, &spec, &lay, &map, tref, j, jj, &mut local,
                             &mut bn, &mut bf2, &mut bm, &mut br, &mut bq, &mut brw,
                             &mut ref_bad, &mut want, fold_bend);
            }
            let collect = |h: std::collections::HashMap<usize, u32>| {
                h.into_iter().collect::<Vec<_>>()
            };
            (local, Bumps { norm: collect(bn), frac: collect(bf2), mul: collect(bm),
                            r8: br.into_iter().collect(), qnz: bq.into_iter().collect(),
                            rw: brw.into_iter().collect(), ref_bad, want })
        })
        .collect();
    let mut ref_bad = 0usize;
    for (bi, (_, b)) in out.iter().enumerate() {
        let _ = bi;
        for (i, n) in &b.norm {
            tables.m_norm[*i] += n;
        }
        for (i, n) in &b.frac {
            tables.m_frac[*i] += n;
        }
        for (i, n) in &b.mul {
            tables.m_mul[*i] += n;
        }
        for (v, n) in &b.r8 {
            let t = tables.ridx(8);
            if *v >= 0 && (*v as usize) < tables.m_r[t].len() {
                tables.m_r[t][*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("deq_eo", *v, *n as usize);
            }
        }
        // stream Z, the fold: the pair-table and width-keyed range bumps
        for (v, n) in &b.qnz {
            if *v >= 0 && (*v as usize) < tables.m_qnz.len() {
                tables.m_qnz[*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("fold_qnz", *v, *n as usize);
            }
        }
        for ((w, v), n) in &b.rw {
            let t = tables.ridx(*w);
            if *v >= 0 && (*v as usize) < tables.m_r[t].len() {
                tables.m_r[t][*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("fold_rem", *v, *n as usize);
            }
        }
        ref_bad += b.ref_bad;
    }
    cols.par_iter_mut().enumerate().for_each(|(c, col)| {
        for (bi, start) in starts.iter().enumerate() {
            let src = &out[bi].0[c];
            col[*start..start + src.len()].copy_from_slice(src);
        }
    });
    // preprocessed: the per-channel scale columns (full height: the channel a slot
    // holds varies by row), then the yside selectors
    let mut pre = vec![];
    for s in 0..lay.per_row {
        match spec.variant {
            DeqVariant::Pow2 => {
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    m31_of(m.jw[b * lay.per_row + s] as i64)
                }).collect()));
            }
            DeqVariant::Chain => {
                for pick in 0..2 {
                    pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                        let (_, b) = lay.at(m.tokens, j);
                        let (mw, ew) = m.sw[b * lay.per_row + s];
                        m31_u32(if pick == 0 { mw } else { ew })
                    }).collect()));
                }
            }
            // stream Z, the fold: the weight-scale mantissa and the channel
            // half of the divisor, `pj = 2^(G_c - (14 - alpha))`
            DeqVariant::Fold => {
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    m31_u32(m.sw[b * lay.per_row + s].0)
                }).collect()));
                let beta = 14 - spec.fold_alpha as i64;
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    let gc = m.fold_gc(b * lay.per_row + s);
                    assert!(gc >= beta, "{}: fold channel divisor 2^{gc} below {beta}",
                            m.name);
                    m31_of(1i64 << (gc - beta))
                }).collect()));
            }
            _ => unreachable!(),
        }
    }
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    // B10 step 4: written through `at_opt`, so a padded layout's inert lead rows
    // fall out correctly and an exact one is bit-for-bit what it was.  On an
    // inert row `nofirst` and `last` are both false -- the inner fold resets to
    // its own (zero) contribution and the outer Horner takes no term -- while
    // `m1` stays 1 so the outer accumulator *carries*.  Lead rows are at the
    // start and the accumulator begins at zero, so carrying zero through them is
    // exactly right.
    let last = |j: usize| matches!(lay.at_opt(m.tokens, j), Some((_, 0)));
    let nofirst = |j: usize| matches!(lay.at_opt(m.tokens, j),
                                      Some((_, b)) if b + 1 < lay.chunks);
    pre.push(sel(&|j| nofirst(j)));                     // nofirst (within a token)
    pre.push(sel(&|j| last(j)));                        // last (token complete)
    pre.push(sel(&|j| j > 0 && !last(j)));              // m1
    pre.push(sel(&|j| j > 0 && last(j)));               // m2
    pre.push(sel(&|j| j + 1 == rows));                  // end
    if output_claim {
        // stream E: the output side's row step.  `oin` is an ordinary step from
        // one chunk to the next inside a token; `obd` is the jump at a token
        // boundary, which is the row that *begins* a token -- the other end from
        // `last`, and the reason these are their own columns.  Both carry `j > 0`
        // like every other multiplier here: at row 0 the `[-1, 0]` mask wraps to
        // the last row of the circle domain.  An inert lead row is neither, so
        // the accumulator is multiplied by zero there, which is exact because it
        // is still zero at that point.
        //
        // Stream SS: with `ohead` the step is the key tile grid's instead, and the
        // row it changes on is the row *after* one that begins a kv head --
        // `j % mh == 0`, exactly `rope::Q_SEL`'s `nn_q_hstart` on the same
        // layout.  A token boundary is a head start (the channel index wraps to
        // zero), so the same constant serves both and the pair of columns keeps
        // its shape; only which rows are marked changes.
        let first = |j: usize| matches!(lay.at_opt(m.tokens, j),
                                        Some((_, b)) if b + 1 == lay.chunks);
        let bd: Box<dyn Fn(usize) -> bool> = match ohead {
            Some(mh) => {
                assert_eq!(lay.lead, 0,
                           "{}: a key-tile-grid output side needs an exact tiling", m.name);
                assert!(mh > 0, "{}: a key-tile-grid output side needs hd % per_row == 0", m.name);
                Box::new(move |j: usize| j % mh == 0)
            }
            None => Box::new(first),
        };
        pre.push(sel(&|j| j > 0 && lay.at_opt(m.tokens, j).is_some() && !bd(j)));
        pre.push(sel(&|j| j > 0 && bd(j)));
        pre.push(sel(&|j| lay.at_opt(m.tokens, j).is_some()));
    }

    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    if cross_check {
        // `ZK_DEVICE_WITNESS_GEN_DEQUANT=2`: the check with teeth, and it runs before any column
        // has changed hands. Every committed and preprocessed cell, the four
        // multiplicity tables, `off_table`, `ref_bad` and -- because it is the
        // only non-integer arithmetic in the witness generation -- the f64 cross-check's own
        // answer per (row, slot).
        let after = snap_tables(tables);
        let before = snap_before.unwrap();
        let mut want_host = vec![0u16; rows * lay.per_row];
        for (bi, start) in starts.iter().enumerate() {
            let w = &out[bi].1.want;
            want_host[start * lay.per_row..start * lay.per_row + w.len()]
                .copy_from_slice(w);
        }
        let dev = crate::gadgets::dequant::device::mat_component_dev(m, &spec, &lay, &map, tables, true, output_claim);
        check_witness_gen_dev(m, &lay, &base, &pre, ref_bad, &want_host, &dev, tables,
                       &before, &after);
        // the device's own bumps are the check's, not the proof's: put the
        // shared tables back where the host witness generation left them
        let r8 = tables.ridx(8);
        tables.m_norm.copy_from_slice(&after.0);
        tables.m_frac.copy_from_slice(&after.1);
        tables.m_mul.copy_from_slice(&after.2);
        tables.m_r[r8].copy_from_slice(&after.3);
        tables.off_table = after.4;
    }
    MatComponent { lay, spec, map, pre, base, ref_bad, n, dev: None }
}

/// `ZK_DEVICE_WITNESS_GEN_DEQUANT=2`: the device witness generation against the host witness generation, everything at once.
///
/// Not a sample: a witness generation that is wrong on one slot of one row is wrong, and the
/// point of checking here rather than against a proof digest is that a
/// mismatch names the column and the cell.
#[allow(clippy::too_many_arguments)]
fn check_witness_gen_dev(m: &Mat, lay: &MatLayout, base: &[Eval], pre: &[Eval],
                  ref_bad: usize, want_host: &[u16],
                  dev: &crate::gadgets::dequant::device::DevDequantWitnessGen, tables: &Tables,
                  before: &(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize),
                  after: &(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize)) {
    use stwo::prover::backend::Column as _;
    let cmp = |what: &str, host: &[Eval], got: &[stwo::stwo_cuda::base_field_vec::BaseFieldVec]| {
        assert_eq!(host.len(), got.len(), "{}: {what} column count", m.name);
        for (i, (a, b)) in host.iter().zip(got).enumerate() {
            let (a, b) = (a.values.to_cpu(), b.to_vec());
            assert_eq!(a.len(), b.len(), "{}: {what} column {i} length", m.name);
            if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
                panic!("{}: {what} column {i} differs at cell {k}: host {:?}, device {:?}",
                       m.name, a[k], b[k]);
            }
        }
    };
    cmp("base", base, &dev.base);
    cmp("pre", pre, &dev.pre);
    // the multiplicities: the device's delta against the host's, table by table
    let delta = |a: &[u32], b: &[u32]| -> Vec<u32> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    };
    let r8 = tables.ridx(8);
    let dev_now = (tables.m_norm.clone(), tables.m_frac.clone(), tables.m_mul.clone(),
                   tables.m_r[r8].clone(), tables.off_table);
    for (what, h, d) in [
        ("m_norm", delta(&after.0, &before.0), delta(&dev_now.0, &after.0)),
        ("m_frac", delta(&after.1, &before.1), delta(&dev_now.1, &after.1)),
        ("m_mul", delta(&after.2, &before.2), delta(&dev_now.2, &after.2)),
        ("m_r8", delta(&after.3, &before.3), delta(&dev_now.3, &after.3)),
    ] {
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("{}: {what}[{k}] bumped {} times by the host and {} by the device",
                   m.name, h[k], d[k]);
        }
    }
    assert_eq!(dev_now.4 - after.4, after.4 - before.4,
               "{}: off-table bumps differ", m.name);
    if let Some(w) = &dev.want {
        assert_eq!(dev.ref_bad, ref_bad, "{}: ref_bad differs", m.name);
        // The one unknown 19.6 named: `witness_gen_mat_row`'s f64 cross-check, on
        // device fp64. Compared per (row, slot), not as a count.
        assert_eq!(w.len(), want_host.len(), "{}: want length", m.name);
        if let Some(k) = (0..w.len()).find(|k| w[*k] as u16 != want_host[*k]) {
            panic!("{}: the f64 cross-check differs at row {} slot {}: host {:#06x}, \
                    device {:#06x}",
                   m.name, k / lay.per_row, k % lay.per_row, want_host[k], w[k]);
        }
    }
    eprintln!("witness-gen-dev: {} -- {} base and {} pre columns, {} cells, the four \
multiplicity tables and the f64 cross-check all match",
              m.name, base.len(), pre.len(),
              (base.len() + pre.len()) << lay.log_rows);
}

/// B1's own answer for one dequantized accumulator: the `Round::Even` reference
/// `witness_gen_mat_row` cross-checks its committed columns against.  It lives here as
/// one function so that a consumer of a matmul's bf16 output -- the residual stream's
/// SwiGLU reads the `gate` and `up` components' -- reads exactly the value the
/// dequant component proves, rather than a second implementation of the rung.
pub fn deq_ref(m: &Mat, spec: &Spec, t: usize, c: usize) -> u16 {
    deq_ref_a(m.acc[t * m.n_out + c], m, spec, t, c)
}

/// The same, for an accumulator the caller already has -- which is not always
/// `m.acc[..]`: an inert lead row of a padded layout commits a *zero*
/// accumulator against token 0's scale, and its reference has to be the zero's
/// image, not the value that would have been there had the row been real.
pub fn deq_ref_a(a: i64, m: &Mat, spec: &Spec, t: usize, c: usize) -> u16 {
    // Stream JJ (E11): every rounding here is `spec.round`'s, not `from_f64`'s.
    // `from_f64` is round-to-nearest-even and this reference is what the output
    // bindings' claim is built from, so under `--round away` an exact tie made the
    // claim disagree with the component's own committed cells by one ulp and the
    // *honest* proof failed to prove.  It cost ~1% of slots on a synthetic
    // layer; on a witness with no bindings it costs nothing, which is why stream E
    // never saw it.
    let rd = |x: f64| bf16::from_f64_round(x, spec.round);
    match spec.variant {
        DeqVariant::Pow2 => {
            let jjx = m.jx[t] as i64 + m.jw[c] as i64;
            rd(a as f64 * (jjx as f64).exp2())
        }
        DeqVariant::Chain => {
            let y0 = bf16::to_f32(rd(a.unsigned_abs() as f64)) as f64;
            let sxv = bf16::to_f32(bf16::pattern(0, m.sx[t].1, m.sx[t].0)) as f64;
            let swv = bf16::to_f32(bf16::pattern(0, m.sw[c].1, m.sw[c].0)) as f64;
            let t1 = bf16::to_f32(rd(y0 * sxv)) as f64;
            rd(t1 * swv) | (u16::from(a < 0) << 15)
        }
        _ => unreachable!("matmul components are pow2 or chain"),
    }
}

/// Stream SS: a `Variant::Fold` component's **output**, as an integer.
///
/// `deq_ref` cannot answer for the fold: the whole point of `y-delay-v` is that
/// there is no bf16 waypoint between the accumulator and the int8 cell, so this
/// component's output is not a bf16 pattern at all.  What it publishes is
/// `qc (1 - 2 sg)` -- the clamped quotient with its sign -- which is
/// `experiments/b1`'s `quantize(v, s_v)` cell for cell, and which
/// `experiments/b65/extract.py` byte-compares against the recorder's own `vvc`
/// at extraction ("v fold identity holds on N sites").
///
/// The arithmetic is the witness generation's, on integers, so the reference and the committed
/// cells are the same value by construction and not by a second rounding: `witness_gen_mat_row`
/// computes exactly `qm = (mag mxw + 2^(G-1)) >> G` and `qc = min(qm, 127)`.
pub fn fold_out(m: &Mat, t: usize, c: usize) -> i64 {
    fold_out_a(m.acc[t * m.n_out + c], m, t, c)
}

/// The same, for an accumulator the caller already has -- `deq_ref_a`'s role for
/// the fold, and what a probe that bends an accumulator asks in order to see
/// whether the output moved.
pub fn fold_out_a(a: i64, m: &Mat, t: usize, c: usize) -> i64 {
    debug_assert_eq!(m.variant, DeqVariant::Fold, "fold_out on a non-fold matmul");
    let (ms, _) = m.sx[t];
    let (mw, _) = m.sw[c];
    let g = (m.fold_gt(t) + m.fold_gc(c)) as u32;
    let mxw = (128 + ms as i128) * (128 + mw as i128);
    let mag = a.unsigned_abs() as i128;
    let qm = ((mag * mxw + (1i128 << (g - 1))) >> g) as i64;
    let qc = qm.min(127);
    if a < 0 { -qc } else { qc }
}

/// One row of a matmul dequant component: the witness fields of `per_row`
/// accumulators, written into a block-local column slice, with table
/// multiplicities accumulated block-locally (merged by `mat_component`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn witness_gen_mat_row(m: &Mat, spec: &Spec, lay: &MatLayout, map: &Map, tables: &Tables,
                j: usize, jj: usize, local: &mut [Vec<M31>],
                bn: &mut std::collections::HashMap<usize, u32>,
                bf2: &mut std::collections::HashMap<usize, u32>,
                bm: &mut std::collections::HashMap<usize, u32>,
                br: &mut std::collections::HashMap<i64, u32>,
                bq: &mut std::collections::HashMap<i64, u32>,
                brw: &mut std::collections::HashMap<(u32, i64), u32>,
                ref_bad: &mut usize, wref: &mut [u16],
                fold_bend: (Option<usize>, Option<usize>)) {
    let f = map.f;
    let sticky = spec.round.needs_sticky();
    // An inert lead row is filled as a row of *zero* accumulators against token 0's
    // scale: a zero accumulator is an ordinary case this gadget already proves
    // (real dumps carry them), so every constraint holds and every lookup lands
    // on a real table row -- `ao_component`'s pattern, and the reason no lookup here
    // needs an activity multiplier.
    let (t, b, inert) = match lay.at_opt(m.tokens, j) {
        Some((t, b)) => (t, b, false),
        None => (0, 0, true),
    };
    // ---- stream Z: the fold's own witness generation (see `dequant.rs`'s `Fold` arm) ------
    if spec.variant == DeqVariant::Fold {
        let _ = tables;
        let (ms, es) = m.sx[t];
        let alpha = spec.fold_alpha as i64;
        let gt = m.fold_gt(t);
        assert!(gt >= alpha, "{}: fold token divisor 2^{gt} below alpha {alpha}", m.name);
        assert!(es <= 134, "{}: s_x at 2^{} is above 1 -- outside the fold's window",
                m.name, es);
        local[map.shared[0]][jj] = m31_u32(ms);
        let dent_bent = i64::from(!inert && fold_bend.1 == Some(t));
        local[map.shared[1]][jj] = m31_of(1i64 << (gt - alpha + dent_bent));
        for s in 0..lay.per_row {
            let c = b * lay.per_row + s;
            let a = if inert { 0 } else { m.acc[t * m.n_out + c] };
            let (mw, _) = m.sw[c];
            let gc = m.fold_gc(c);
            assert!(gc >= 14 - alpha,
                    "{}: fold channel divisor 2^{gc} below {}", m.name, 14 - alpha);
            let g = (gt + gc) as u32;
            let mxw = (128 + ms as i64) * (128 + mw as i64);
            let sg = u32::from(a < 0);
            let mag = a.unsigned_abs() as i64;
            assert!(mag < 1i64 << (2 * crate::gadgets::dequant::FOLD_LIMB),
                    "{}: fold |acc| {mag} above the two-limb window", m.name);
            let limb = (1i64 << crate::gadgets::dequant::FOLD_LIMB) - 1;
            let (flo, fhi) = (mag & limb, mag >> crate::gadgets::dequant::FOLD_LIMB);
            let lo_prod = flo * mxw;
            let (fa, fb) = (lo_prod >> crate::gadgets::dequant::FOLD_LIMB, lo_prod & limb);
            // `round_hafz` on the magnitude: half away from zero
            let tt = (mag as i128) * (mxw as i128);
            let mut qm = ((tt + (1i128 << (g - 1))) >> g) as i64;
            // stream Z's bend control: one quotient moved by +1, everything else
            // honest -- the high remainder leaves [0, 2 fdp) and must reject
            let bent = !inert && fold_bend.0 == Some(t * m.n_out + c);
            if bent {
                qm += 1;
            }
            let qc = qm.min(127);
            let fdp = 1i64 << (g - 14);
            let fe = fhi * mxw + fa - (2 * qm - 1) * fdp;
            debug_assert!(bent || (0..2 * fdp).contains(&fe),
                          "{}: fold remainder", m.name);
            let set = |off: usize, v: M31, local: &mut [Vec<M31>]| {
                local[map.at(s, off)][jj] = v;
            };
            set(f.flo, m31_of(flo), local);
            set(f.fhi, m31_of(fhi), local);
            set(f.sg, m31_u32(sg), local);
            set(f.fa, m31_of(fa), local);
            set(f.fb, m31_of(fb), local);
            set(f.qm, m31_of(qm), local);
            set(f.qc, m31_of(qc), local);
            set(f.fe, m31_of(fe), local);
            set(f.fdp, m31_of(fdp), local);
            // the bit-for-bit gate's cross-check: the harness's own arithmetic,
            // exact in f64 (the product is 42 bits), against the witness generation's integers
            let sxf = (128.0 + ms as f64) * (es as f64 - 134.0).exp2();
            let (mws, ews) = m.sw[c];
            let swf = (128.0 + mws as f64) * (ews as f64 - 134.0).exp2();
            let svf = (m.jv2[c] as f64).exp2();
            let realq = a as f64 * sxf * swf / svf;
            let want = (realq.abs() + 0.5).floor().min(127.0) as i64
                * if realq < 0.0 { -1 } else { 1 };
            if want != qc * (1 - 2 * sg as i64) {
                *ref_bad += 1;
            }
            *brw.entry((spec.q_bits, qm)).or_default() += 1;
            *brw.entry((7, qc)).or_default() += 1;
            *brw.entry((spec.q_bits, qm - qc)).or_default() += 1;
            *brw.entry((crate::gadgets::dequant::FOLD_LIMB, flo)).or_default() += 1;
            *brw.entry((crate::gadgets::dequant::FOLD_LIMB, fb)).or_default() += 1;
            *brw.entry((16, fa)).or_default() += 1;
            *brw.entry((spec.rem_bits, fe)).or_default() += 1;
            *brw.entry((spec.rem_bits, 2 * fdp - 1 - fe)).or_default() += 1;
        }
        return;
    }
    match spec.variant {
        DeqVariant::Pow2 => local[map.shared[0]][jj] = m31_of(m.jx[t] as i64),
        DeqVariant::Chain => {
            local[map.shared[0]][jj] = m31_u32(m.sx[t].0);
            local[map.shared[1]][jj] = m31_u32(m.sx[t].1);
        }
        _ => unreachable!("matmul components are pow2 or chain"),
    }
    for s in 0..lay.per_row {
        let c = b * lay.per_row + s;
        let a = if inert { 0 } else { m.acc[t * m.n_out + c] };
        let mut set = |off: usize, v: M31, local: &mut [Vec<M31>]| {
            local[map.at(s, off)][jj] = v;
        };
        set(f.acc, m31_of(a), local);
        let sg = u32::from(a < 0);
        let mag = a.unsigned_abs();
        let (m2, ph, mut st, r) = bf16::normalise(mag, spec.kmax)
            .unwrap_or_else(|| panic!("{}: |acc| {mag} above the kmax {} window",
                                      m.name, spec.kmax));
        if !sticky {
            st = 0;
        }
        let ri = tables.norm_at(m2, ph, st).expect("norm row");
        let nr = tables.norm[ri];
        *bn.entry(ri).or_default() += 1;
        let fi = tables.frac_at(ph, r).expect("frac row");
        *bf2.entry(fi).or_default() += 1;
        set(f.sg, m31_u32(sg), local);
        set(f.m2, m31_u32(m2), local);
        set(f.ph, m31_u32(ph), local);
        if sticky {
            set(f.st, m31_u32(st), local);
        }
        set(f.r, m31_of(r as i64), local);
        set(f.m7, m31_u32(nr.m7), local);
        set(f.ef, m31_u32(nr.expfield), local);
        set(f.nz, m31_u32(nr.nz), local);
        match spec.variant {
            DeqVariant::Pow2 => {
                let jjx = m.jx[t] as i64 + m.jw[c] as i64;
                let eo = if nr.nz == 1 { nr.expfield as i64 + jjx } else { 0 };
                set(f.eo, m31_of(eo), local);
                *br.entry(eo).or_default() += 1;
                if spec.round == bf16::Round::Even {
                    let want = deq_ref_a(a, m, spec, t, c);
                    if !wref.is_empty() {
                        wref[jj * lay.per_row + s] = want;
                    }
                    let got = bf16::pattern(sg, eo.max(0) as u32, nr.m7);
                    if want != got {
                        *ref_bad += 1;
                    }
                }
            }
            DeqVariant::Chain => {
                let mut m_in = nr.m7;
                let mut e_in = nr.expfield as i64;
                let scales = [m.sx[t], m.sw[c]];
                for (tt, (ms, es)) in scales.iter().enumerate() {
                    let mr = tables.mul[(m_in * 128 + ms) as usize];
                    let (po, ca) = if nr.nz == 1 { (mr.m7, mr.carry) } else { (0, 0) };
                    let eo = if nr.nz == 1 { e_in + *es as i64 - 127 + ca as i64 } else { 0 };
                    set(f.mul[tt][0], m31_u32(po), local);
                    set(f.mul[tt][1], m31_u32(ca), local);
                    set(f.mul[tt][2], m31_of(eo), local);
                    if nr.nz == 1 {
                        *bm.entry((m_in * 128 + ms) as usize).or_default() += 1;
                    }
                    *br.entry(eo).or_default() += 1;
                    m_in = po;
                    e_in = eo;
                }
                if spec.round == bf16::Round::Even {
                    let want = deq_ref_a(a, m, spec, t, c);
                    if !wref.is_empty() {
                        wref[jj * lay.per_row + s] = want;
                    }
                    let got = bf16::pattern(sg, e_in.max(0) as u32, m_in);
                    if want != got {
                        *ref_bad += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
