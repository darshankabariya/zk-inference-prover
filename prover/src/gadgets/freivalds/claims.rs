//! After the challenges: the running sums, what the component publishes, and
//! its LogUp columns.
//!
//! This is the half of the witness that depends on `rv` and `s`, so none of it
//! can be built before those are drawn from the transcript -- which is after
//! every commitment `trace.rs` made.  It also holds `project`, the one written
//! definition of the projection `rv^T C s` that the two sides of the identity
//! are compared at.

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::qm31::SecureField;

use crate::compose::{Claims, FreivaldsPlan, unqm};
use crate::encoding::m31_of;
use crate::gadgets::freivalds::columns::{FreivaldsShape, Matmul, RunningSums};
use crate::gadgets::freivalds::trace::FreivaldsComponent;
use crate::layer::Layer;

/// `rv^T C s`: the product matrix projected down to one field element, which is
/// the value both sides of the Freivalds identity are compared at.
///
/// The caller hands the rows of `C` already folded against `rv` -- one value per
/// token, `(rv^T C)[t]` -- in the order the tables run them, which is the last
/// token first.  Folding those against `s` by Horner leaves
/// `sum_t s^t (rv^T C)[t]`.
///
/// `C` itself is committed by the dequant gadget and never by this one, so it is
/// the dequant component that publishes this value, as its claim `claimed_h`,
/// and this gadget that consumes it: `identity_holds` compares it against the
/// `(rv^T W)(X s)` the Freivalds component accumulated.
pub fn project(token_folds: impl IntoIterator<Item = SecureField>,
               s: SecureField) -> SecureField {
    token_folds.into_iter().fold(SecureField::zero(), |h, z| h * s + z)
}

/// The same projection, straight off an exact integer product matrix held
/// row-major as `(tokens, n_out)`.
///
/// Nothing in a proof takes this route -- the two sides each reach the value
/// through a committed running column.  It is what a self-check compares them
/// against.
pub fn project_product(c: &[i64], tokens: usize, n_out: usize, rv: SecureField,
                       s: SecureField) -> SecureField {
    let (rv_pow, _) = crate::relations::powers(rv, n_out);
    project((0..tokens).rev().map(|t| {
        let mut row = SecureField::zero();
        for i in 0..n_out {
            row += rv_pow[i] * SecureField::from(m31_of(c[t * n_out + i]));
        }
        row
    }), s)
}

/// The Freivalds identity, once per matmul of every group and in the order the
/// plan runs them: the dequant component's published `rv^T C s` against the
/// Freivalds component's published `(rv^T W)(X s)`.
///
/// This is the whole of what a verifier does with this gadget's claims, and
/// `compose::verify_composed` calls it.  Both sides are values the statement
/// carries; the proofs are what tie each of them to a committed column.
pub fn identity_holds(claims: &Claims, groups: &[FreivaldsPlan]) -> Vec<bool> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(wi, w)| w.mats.iter().enumerate().map(move |(k, mi)| (wi, k, *mi)))
        .map(|(wi, k, mi)| unqm(claims.freivalds_acc[wi][k]) == unqm(claims.mat_h[mi]))
        .collect()
}

pub struct FreivaldsInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    /// one `(rv^T W_m)(X s)` per matmul: the side of the identity this gadget
    /// publishes
    pub claimed: Vec<SecureField>,
    pub use_sum: SecureField,
    /// the group's claim over its own committed activations, in the canonical
    /// input-channel-major order, or zero when the AIR publishes none
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

    // Where the running sums are built: on the host, on the card, or both with
    // every cell compared (`ZK_DEVICE_ACCUMULATORS` = 0, 1, 2).
    let acc_dev = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let acc_host = acc_dev.is_none() || crate::devacc::mode() == 2;
    let rows = if acc_host { rows } else { 0 };
    // each row's own two folds in parallel -- the activations against `s`, and
    // each matmul's weights against `rv`.  The carries down the rows are serial
    // and come after.
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
    let mut x_fold = vec![SecureField::zero(); rows];
    let mut w_folds = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    let mut products = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    for j in 0..rows {
        // the same two row flags the AIR reads, here as the loop they encode
        let (carry_in_input_channel, input_channel_done) = match lay.g.row(j) {
            Some((_, c)) => (c + 1 < lay.g.chunks, c == 0),
            None => (false, false),
        };
        x_fold[j] = if carry_in_input_channel { x_fold[j - 1] * s_step }
                    else { SecureField::zero() } + inner[j].0;
        for m in 0..r.mats.len() {
            w_folds[m][j] = if carry_in_input_channel { w_folds[m][j - 1] * rv_step }
                            else { SecureField::zero() } + inner[j].1[m];
            products[m][j] = if j > 0 { products[m][j - 1] } else { SecureField::zero() };
            if input_channel_done {
                products[m][j] += w_folds[m][j] * x_fold[j];
            }
        }
    }
    let mut cols = crate::device::ColSink::new();
    // The running sums go into the tree through the row declaration, so their
    // order is the order the AIR reads them in and it is written down once.
    // `x_claim` is `None` here because it is built below, out of `m0.qx` rather
    // than out of these, and appended where the declaration has it.
    let sh = FreivaldsShape::new(&r.name, &lay, r.mats.len(), air.qx.is_some());
    let acc_cols = || -> Vec<crate::common::Eval> {
        RunningSums {
            x_fold: x_fold.as_slice(),
            mats: w_folds.iter().zip(products.iter())
                .map(|(w, p)| Matmul { w_fold: w.as_slice(), product: p.as_slice() })
                .collect(),
            x_claim: None,
        }
        .into_committed()
        .into_iter()
        .flat_map(|c| crate::encoding::secure_cols_pub(lay.g.log_rows, c))
        .collect()
    };
    let claimed: Vec<SecureField> = match acc_dev {
        Some(d) => {
            let got = crate::devacc::freivalds(d, &lay, r.mats.len(), rv, s_ch);
            let hh = acc_host.then(|| {
                (acc_cols(), products.iter().map(|a| a[rows - 1]).collect::<Vec<_>>())
            });
            crate::devacc::place(&format!("freivalds {} 2^{}", r.name, lay.g.log_rows), got,
                                 hh, &mut cols)
        }
        None => {
            cols.extend(acc_cols());
            products.iter().map(|a| a[rows - 1]).collect()
        }
    };
    // The published claim over the committed activations, in the AIR's own mask
    // order -- after the per-matmul running sums and before the LogUp columns.
    // It is built from `m0.qx` and not from `r.base`, so it is the same column
    // whether the witness generation left the component on the host or on the
    // card, and it does not depend on `rows` having been zeroed for the device
    // accumulator path above.
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
            let (carry_in_input_channel, carry_in_table) = match lay.g.row(j) {
                Some((_, c)) => (c + 1 < lay.g.chunks, j > 0),
                None => (false, j > 0),
            };
            let carry = if j == 0 {
                SecureField::zero()
            } else if carry_in_input_channel {
                xs[j - 1] * q.m_in
            } else if carry_in_table {
                xs[j - 1] * q.m_bd
            } else {
                SecureField::zero()
            };
            xs[j] = carry + inner_x[j];
        }
        qx_claim = xs[nrows - 1];
        // the declaration's last running column, after the per-matmul pairs
        cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &xs));
    }
    // the row's own running columns are in; the LogUp columns below are the
    // relation's and the row declaration does not name them
    if acc_dev.is_none() {
        assert_eq!(cols.len(), 4 * sh.n_inter(),
                   "freivalds {}: the interaction tree is not the row's", r.name);
    }
    let mut use_sum = SecureField::zero();
    // Under `--gkr-freivalds` this group's range checks were handed to the GKR
    // collector before its proof ran, in `compose`'s own pass, so there are no
    // running-sum columns to build for them and no claimed sum to carry.  The
    // `x_fold`, `w_fold` and `product` columns above stay: they are the
    // identity's own accumulators and not lookups.
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
