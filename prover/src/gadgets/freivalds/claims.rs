//! What the Freivalds component publishes, and its LogUp columns.

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::qm31::SecureField;

use crate::encoding::m31_of;
use crate::gadgets::freivalds::trace::FreivaldsComponent;
use crate::layer::Layer;

pub struct FreivaldsInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed: Vec<SecureField>,
    pub use_sum: SecureField,
    /// Stream II: the group's claim over its own committed activations, in the
    /// canonical channel-major order, or zero when the AIR carries none.
    pub qx_claim: SecureField,
}

pub fn freivalds_interaction(layer: &Layer, r: &FreivaldsComponent, rv: SecureField, s_ch: SecureField,
                             rels: &crate::compose::Rels, range_check: bool,
                             air: &crate::gadgets::freivalds::FreivaldsEval, gkr: bool) -> FreivaldsInteraction {
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let lay = r.lay;
    let rows = 1usize << lay.g.log_rows;
    let m0 = &layer.mats[r.mats[0]];
    let (rv_pow, rv_step) = crate::relations::powers(rv, lay.pk);
    let (s_pow, s_step) = crate::relations::powers(s_ch, lay.pq);

    // B7a milestone 4 (section 3 row 1): the Freivalds checks' v, us and accs, on the same
    // three-value mode every witness generation in this series uses.
    let acc_dev = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let acc_host = acc_dev.is_none() || crate::devacc::mode() == 2;
    let rows = if acc_host { rows } else { 0 };
    // v's inner sums in parallel (the activations), u's per matmul
    let inner: Vec<(SecureField, Vec<SecureField>)> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut vf = SecureField::zero();
            let mut ufs = vec![SecureField::zero(); r.mats.len()];
            if let Some((k, chunk)) = lay.g.row(j) {
                for p in 0..lay.pq {
                    let t = chunk * lay.pq + p;
                    if t < m0.tokens {
                        let x = m0.qx[t * m0.n_in + k];
                        if x != 0 {
                            vf += s_pow[p] * SecureField::from(m31_of(x as i64));
                        }
                    }
                }
                for (mi, uf) in r.mats.iter().zip(ufs.iter_mut()) {
                    let m = &layer.mats[*mi];
                    for p in 0..lay.pk {
                        let i = chunk * lay.pk + p;
                        if i < m.n_out {
                            let w = m.qw[k * m.n_out + i];
                            if w != 0 {
                                *uf += rv_pow[p] * SecureField::from(m31_of(w as i64));
                            }
                        }
                    }
                }
            }
            (vf, ufs)
        })
        .collect();
    let mut v = vec![SecureField::zero(); rows];
    let mut us = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    let mut accs = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    for j in 0..rows {
        let (nofirst, last) = match lay.g.row(j) {
            Some((_, c)) => (c + 1 < lay.g.chunks, c == 0),
            None => (false, false),
        };
        v[j] = if nofirst { v[j - 1] * s_step } else { SecureField::zero() } + inner[j].0;
        for m in 0..r.mats.len() {
            us[m][j] = if nofirst { us[m][j - 1] * rv_step } else { SecureField::zero() }
                + inner[j].1[m];
            accs[m][j] = if j > 0 { accs[m][j - 1] } else { SecureField::zero() };
            if last {
                accs[m][j] += us[m][j] * v[j];
            }
        }
    }
    let mut cols = crate::device::ColSink::new();
    let claimed: Vec<SecureField> = match acc_dev {
        Some(d) => {
            let got = crate::devacc::freivalds(d, &lay, r.mats.len(), rv, s_ch);
            let hh = acc_host.then(|| {
                let mut e = crate::encoding::secure_cols_pub(lay.g.log_rows, &v);
                for m in 0..r.mats.len() {
                    e.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &us[m]));
                    e.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &accs[m]));
                }
                (e, accs.iter().map(|a| a[rows - 1]).collect::<Vec<_>>())
            });
            crate::devacc::place(&format!("freivalds {} 2^{}", r.name, lay.g.log_rows), got,
                                 hh, &mut cols)
        }
        None => {
            cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &v));
            for m in 0..r.mats.len() {
                cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &us[m]));
                cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &accs[m]));
            }
            accs.iter().map(|a| a[rows - 1]).collect()
        }
    };
    // stream II: the activation claim, in the AIR's own mask order -- after the
    // per-matmul accumulators and before the LogUp columns.  It is built from
    // `m0.qx` and not from `r.base`, so it is the same column whether the witness generation
    // left the component on the host or on the card, and it does not depend on
    // `rows` having been zeroed for the device accumulator path above.
    let mut qx_claim = SecureField::zero();
    if let Some(q) = air.qx.as_ref() {
        let nrows = 1usize << lay.g.log_rows;
        let inner_x: Vec<SecureField> = (0..nrows)
            .into_par_iter()
            .map(|j| {
                let mut f = SecureField::zero();
                if let Some((k, chunk)) = lay.g.row(j) {
                    for p in 0..lay.pq {
                        let t = chunk * lay.pq + p;
                        if t < m0.tokens {
                            let x = m0.qx[t * m0.n_in + k];
                            if x != 0 {
                                f += q.w[p] * SecureField::from(m31_of(x as i64));
                            }
                        }
                    }
                }
                f
            })
            .collect();
        let mut xs = vec![SecureField::zero(); nrows];
        for j in 0..nrows {
            let (nofirst, notzero) = match lay.g.row(j) {
                Some((_, c)) => (c + 1 < lay.g.chunks, j > 0),
                None => (false, j > 0),
            };
            let carry = if j == 0 {
                SecureField::zero()
            } else if nofirst {
                xs[j - 1] * q.m_in
            } else if notzero {
                xs[j - 1] * q.m_bd
            } else {
                SecureField::zero()
            };
            xs[j] = carry + inner_x[j];
        }
        qx_claim = xs[nrows - 1];
        cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &xs));
    }
    let mut use_sum = SecureField::zero();
    // B11 stream V, lever 4: under `--gkr-freivalds` this group's range checks were
    // handed to the collector before the GKR proof ran, in `compose`'s own pass,
    // so there are no running-sum columns to build and no claimed sum to carry.
    // The `v`, `u` and `acc` columns above stay: they are Horner accumulators
    // and 38.9% of this family's interaction cells, which the forecast priced.
    if range_check && !gkr {
        let one = PackedQM31::broadcast(SecureField::one());
        let fr = |i: usize, vr: usize|
            (one, rels.range(r.act_bits).combine(&[r.base[i].data[vr]]));
        let (lcols, s) = crate::devfrac::frac_cols(
            air, &format!("freivalds {}", r.name), lay.g.log_rows,
            crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
            crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
            || {
                let (n_pairs, fpair) = crate::plogup::paired(lay.pq, fr);
                crate::plogup::build(lay.g.log_rows, n_pairs, fpair)
            });
        cols.extend_any(lcols);
        use_sum = s;
    }
    FreivaldsInteraction { cols: cols.into_inner(), claimed, use_sum, qx_claim }
}
