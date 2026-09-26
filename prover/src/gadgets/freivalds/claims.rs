//! After the challenges: what the statement carries, the equality over it, and
//! the running sums that reach it.
//!
//! This is the half of the witness that depends on `rv` and `s`, so none of it
//! can be built before those are drawn from the transcript -- which is after
//! every commitment `trace.rs` made.
//!
//! [`FreivaldsClaims`] is the *public input*: what this component publishes and
//! what it consumes, as values rather than as columns.  It and the AIR are two
//! objects on purpose.  The AIR is the constraint definition -- which columns
//! exist and what holds between them -- and it reads the published half only
//! for the rows its folds close on; a verifier parses these values, and
//! [`identity_holds`] is the whole of what it then does with this gadget.
//! [`project`] is the one written definition of the projection `s^T C rv` the
//! two sides are compared at.
//!
//! The four running columns are four `crate::fold::Fold`s, and
//! [`freivalds_interaction`] builds each of them the same way: every row's own
//! term in parallel, and then the primitive's serial carry, under the same
//! `Carry` and `Term` rules the AIR states -- so the two sides of each column
//! are one declaration read twice, and the table's grouping is stated once, in
//! the shape.  Where they are built -- on the host, on the card, or both with
//! every cell compared -- is `fold::sides` and `fold::place`, and the lookups'
//! columns are the byte cells' own.

use num_traits::Zero;
use rayon::prelude::*;
use stwo::core::fields::qm31::SecureField;

use crate::encoding::m31_of;
use crate::fold::{Carry, Term, carry_down};
use crate::gadgets::freivalds::columns::{Matmul, RunningSums};
use crate::gadgets::freivalds::trace::FreivaldsComponent;
use crate::gadgets::freivalds::witness::FreivaldsWitness;

// ------------------------------------------------ what the statement says ---

/// The values one Freivalds group's statement carries.
///
/// Not the AIR: the AIR says what holds between the columns, and this says what
/// the two ends of them are.  Both halves are public input -- a verifier reads
/// them out of the proof's claims and never recomputes them -- and the proofs
/// are what tie each of them to a committed column.
#[derive(Clone, Debug, Default)]
pub struct FreivaldsClaims {
    /// **Published**, one per matmul in the group's order: `(s^T X)(W_m rv)`,
    /// what the product fold is pinned to on the table's last row.
    pub product: Vec<SecureField>,
    /// **Published**: the projection of this component's own committed
    /// activations, so whichever component produced them can be bound to the
    /// same values.  `None` on a segment with no producer to bind to.
    pub x_claim: Option<SecureField>,
    /// **Consumed**, one per matmul: `s^T C rv`, the projection of the product.
    /// This component never commits `C`; the dequant component does, and
    /// publishes this through [`project`].  Nothing constrains it here -- the
    /// comparison is [`identity_holds`].
    pub consumed: Vec<SecureField>,
}

impl FreivaldsClaims {
    /// The claims a builder starts from: zero everywhere.
    ///
    /// An AIR has to exist before the columns that close on it are built -- the
    /// interaction builder is handed one to read the group's shape off -- and
    /// this is what it is built with until the values are known.
    pub fn open(n_mats: usize, x_claim: bool) -> Self {
        FreivaldsClaims {
            product: vec![SecureField::zero(); n_mats],
            x_claim: x_claim.then(SecureField::zero),
            consumed: vec![],
        }
    }
}

/// The Freivalds identity, once per matmul of every group and in the order the
/// plan runs them: the projection the dequant component published against the
/// one this component accumulated.
///
/// This is the whole of what a verifier does with this gadget's claims, and
/// `compose::verify_composed` calls it.  Neither side commits `C` twice; the
/// check is that two independently accumulated field elements agree.
pub fn identity_holds(groups: &[FreivaldsClaims]) -> Vec<bool> {
    groups
        .iter()
        .flat_map(|g| {
            assert_eq!(g.product.len(), g.consumed.len(),
                       "a Freivalds group published {} products and consumed {} projections",
                       g.product.len(), g.consumed.len());
            g.product.iter().zip(&g.consumed).map(|(a, b)| a == b)
        })
        .collect()
}

/// `s^T C rv`: the product matrix projected down to one field element, which is
/// the value both sides of the Freivalds identity are compared at.
///
/// The caller hands the rows of `C` already folded against `rv` -- one value per
/// token, `(rv^T C)[t]` -- in the order the tables run them, which is the last
/// token first.  Folding those against `s` by Horner leaves
/// `sum_t s^t (rv^T C)[t]`.
///
/// `C` itself is committed by the dequant gadget and never by this one, so it is
/// the dequant component that publishes this value, as its claim `claimed_h`,
/// and this gadget that consumes it.
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

// ------------------------------------------------------- the running sums ---

pub struct FreivaldsInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    /// what the component published: one `(s^T X)(W_m rv)` per matmul, and the
    /// claim over its own activations where the AIR carries one.  `consumed` is
    /// empty -- the other side of the identity is not this builder's to know.
    pub claims: FreivaldsClaims,
    pub use_sum: SecureField,
}

pub fn freivalds_interaction(w: &FreivaldsWitness, r: &FreivaldsComponent,
                             air: &crate::gadgets::freivalds::FreivaldsEval)
                             -> FreivaldsInteraction {
    let lay = r.lay;
    let rows = 1usize << lay.g.log_rows;
    let (rv, s_ch) = (air.rv.c, air.s.c);

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
            let mut ufs = vec![SecureField::zero(); w.mats.len()];
            if let Some((k, chunk)) = lay.g.row(j) {
                for p in 0..lay.pq {
                    let t = chunk * lay.pq + p;
                    if t < w.tokens {
                        let x = w.x[t * w.n_in + k];
                        if x != 0 {
                            vf += air.s.cells[p] * SecureField::from(m31_of(x as i64));
                        }
                    }
                }
                for (m, uf) in w.mats.iter().zip(ufs.iter_mut()) {
                    for p in 0..lay.pk {
                        let i = chunk * lay.pk + p;
                        if i < m.n_out {
                            let wt = m.w[k * m.n_out + i];
                            if wt != 0 {
                                *uf += air.rv.cells[p] * SecureField::from(m31_of(wt as i64));
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
    let sh = &air.shape;
    let groups = sh.groups();
    let x_fold = carry_down(&groups, rows, Term::Each(&|j| inner[j].0), air.s.per_group());
    let w_folds: Vec<Vec<SecureField>> = (0..w.mats.len())
        .map(|m| carry_down(&groups, rows, Term::Each(&|j| inner[j].1[m]), air.rv.per_group()))
        .collect();
    let products: Vec<Vec<SecureField>> = (0..w.mats.len())
        .map(|m| carry_down(&groups, rows, Term::PerGroup(&|j| w_folds[m][j] * x_fold[j]),
                            Carry::WholeTable))
        .collect();
    let mut cols = crate::device::ColSink::new();
    // The running sums go into the tree through the row declaration, so their
    // order is the order the AIR reads them in and it is written down once.
    // `x_claim` is `None` here because it is built below, out of the witness's
    // activations rather than out of these, and appended where the declaration
    // has it.
    let product: Vec<SecureField> = crate::fold::place(
        &format!("freivalds {} 2^{}", r.name, lay.g.log_rows), sides, &mut cols,
        || crate::devacc::freivalds(
            r.dev.as_ref().expect("the card builds a group's folds off its own columns"),
            &lay, w.mats.len(), rv, s_ch),
        || {
            let built = RunningSums {
                x_fold: x_fold.as_slice(),
                mats: w_folds.iter().zip(products.iter())
                    .map(|(wf, p)| Matmul { w_fold: wf.as_slice(), product: p.as_slice() })
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
    // It is built from the witness and not from `r.base`, so it is the same
    // column whether the witness generation left the component on the host or on
    // the card, and it does not depend on `rows` having been zeroed for the
    // device accumulator path above.
    let mut x_claim = None;
    if let Some(q) = air.qx.as_ref() {
        let nrows = 1usize << lay.g.log_rows;
        let inner_x: Vec<SecureField> = (0..nrows)
            .into_par_iter()
            .map(|j| {
                let mut f = SecureField::zero();
                if let Some((k, chunk)) = lay.g.row(j) {
                    for p in 0..lay.pq {
                        let t = chunk * lay.pq + p;
                        if t < w.tokens {
                            let x = w.x[t * w.n_in + k];
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
        x_claim = Some(xs[nrows - 1]);
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
    FreivaldsInteraction {
        cols: cols.into_inner(),
        claims: FreivaldsClaims { product, x_claim, consumed: vec![] },
        use_sum,
    }
}
