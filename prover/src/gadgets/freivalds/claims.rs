//! After the challenges: the running sums, what the component publishes, and
//! its LogUp columns.
//!
//! This is the half of the witness that depends on `rv` and `s`, so none of it
//! can be built before those are drawn from the transcript -- which is after
//! every commitment `trace.rs` made.  It also holds `project`, the one written
//! definition of the projection `rv^T C s` that the two sides of the identity
//! are compared at.
//!
//! The four running columns are four `crate::fold::Fold`s, and this file builds
//! each of them the same way: every row's own term in parallel, and then the
//! primitive's serial carry, under the same `Carry` and `Term` rules the AIR
//! states -- so the two sides of each column are one declaration read twice, and
//! the table's grouping is stated once, in the shape.  Where they are built --
//! on the host, on the card, or both with every cell compared -- is
//! `fold::sides` and `fold::place`, and the lookups' columns are the byte cells'
//! own.

use num_traits::Zero;
use rayon::prelude::*;
use stwo::core::fields::qm31::SecureField;

use crate::compose::{Claims, FreivaldsPlan, unqm};
use crate::encoding::m31_of;
use crate::fold::{Carry, Term, carry_down};
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

pub fn freivalds_interaction(layer: &Layer, r: &FreivaldsComponent, rv: SecureField,
                             s_ch: SecureField,
                             air: &crate::gadgets::freivalds::FreivaldsEval) -> FreivaldsInteraction {
    let lay = r.lay;
    let rows = 1usize << lay.g.log_rows;
    let m0 = &layer.mats[r.mats[0]];
    let (rv_pow, rv_step) = crate::relations::powers(rv, lay.pk);
    let (s_pow, s_step) = crate::relations::powers(s_ch, lay.pq);

    // Where the folds are built: on the host, on the card, or both with every
    // cell compared.
    let sides = crate::fold::sides(r.dev.is_some());
    let rows = if sides.host { rows } else { 0 };
    // each row's own two terms in parallel -- the activations against `s`, and
    // each matmul's weights against `rv`.  The carries down the rows are the
    // fold's and come after.
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
    // The same rules the AIR states, here as the loop they stand for: x_fold
    // and w_fold restart at each input channel, and the product carries across
    // the whole table and takes the two finished folds once per input channel.
    let sh = FreivaldsShape::new(&r.name, &lay, r.mats.len(), air.qx.is_some());
    let groups = sh.groups();
    let x_fold = carry_down(&groups, rows, Term::Each(&|j| inner[j].0),
                            Carry::PerGroup { step: s_step });
    let w_folds: Vec<Vec<SecureField>> = (0..r.mats.len())
        .map(|m| carry_down(&groups, rows, Term::Each(&|j| inner[j].1[m]),
                            Carry::PerGroup { step: rv_step }))
        .collect();
    let products: Vec<Vec<SecureField>> = (0..r.mats.len())
        .map(|m| carry_down(&groups, rows, Term::PerGroup(&|j| w_folds[m][j] * x_fold[j]),
                            Carry::WholeTable))
        .collect();
    let mut cols = crate::device::ColSink::new();
    // The running sums go into the tree through the row declaration, so their
    // order is the order the AIR reads them in and it is written down once.
    // `x_claim` is `None` here because it is built below, out of `m0.qx` rather
    // than out of these, and appended where the declaration has it.
    let claimed: Vec<SecureField> = crate::fold::place(
        &format!("freivalds {} 2^{}", r.name, lay.g.log_rows), sides, &mut cols,
        || crate::devacc::freivalds(
            r.dev.as_ref().expect("the card builds a group's folds off its own columns"),
            &lay, r.mats.len(), rv, s_ch),
        || {
            let built = RunningSums {
                x_fold: x_fold.as_slice(),
                mats: w_folds.iter().zip(products.iter())
                    .map(|(w, p)| Matmul { w_fold: w.as_slice(), product: p.as_slice() })
                    .collect(),
                x_claim: None,
            }
            .into_committed()
            .into_iter()
            .flat_map(|c| crate::fold::column(lay.g.log_rows, c))
            .collect();
            (built, products.iter().map(|a| a[rows - 1]).collect())
        });
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
        // This table's rows step at two rates -- one inside an input channel and
        // one where a new one begins -- which is a rule about the grouping and
        // so one the fold states, exactly as the AIR states it.
        let xs = carry_down(&groups, nrows, Term::Each(&|j| inner_x[j]),
                            Carry::Rates { inside: q.m_in, across: q.m_bd });
        qx_claim = xs[nrows - 1];
        // the declaration's last running column, after the per-matmul pairs
        cols.extend(crate::fold::column(lay.g.log_rows, &xs));
    }
    // the row's own folds are in; the lookups' columns below are the relation's
    // and the row declaration does not name them
    if !sides.device {
        assert_eq!(cols.len(), crate::fold::M31_COLS * sh.n_inter(),
                   "freivalds {}: the interaction tree is not the row's", r.name);
    }
    // And the activations' lookups, which are the byte cells' own: their
    // fraction columns, or nothing at all when the plan has them proved
    // elsewhere and the site was registered instead.  The folds above stay
    // either way -- they are the identity's accumulators and not lookups.
    let (lcols, use_sum) = air.acts.frac_cols(
        air, &format!("freivalds {}", r.name), lay.g.log_rows,
        crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
        crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
        &r.base, lay.pq);
    cols.extend_any(lcols);
    FreivaldsInteraction { cols: cols.into_inner(), claimed, use_sum, qx_claim }
}
