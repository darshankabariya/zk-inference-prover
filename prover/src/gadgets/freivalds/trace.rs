//! Before the challenges: the columns that are committed, and the constants the
//! verifier already holds.
//!
//! Nothing here depends on `rv` or `s` -- it cannot, because those are drawn
//! from the transcript only after these columns have been committed.  What is
//! built here is the activation bytes `A` and the preprocessed weights `M`; the
//! grouping's own columns come from `crate::fold`, which owns them on both
//! sides.  The running sums and the claims are `claims.rs`, after the draw.
//!
//! The activations go through `columns::Bytes`, the cell kind for values of a
//! range table, so the multiplicities they bump are counted by the same call
//! that builds them -- a writer that commits a cell and forgets to count it is
//! the one mistake this half can make on its own.  `device.rs` is the same two
//! trees built on the card; which side runs is its `mode`, and this file is the
//! only caller of either.

use num_traits::Zero;
use stwo::core::fields::m31::M31;

use crate::columns::Bytes;
use crate::common::{Eval, eval_of};
use crate::encoding::m31_of;
use crate::gadgets::freivalds::columns::{Activations, FreivaldsShape, Known};
use crate::gadgets::freivalds::device;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::gadgets::freivalds::witness::FreivaldsWitness;
use crate::layer::Tables;

pub struct FreivaldsComponent {
    pub name: String,
    pub lay: FreivaldsLayout,
    /// where the layer's own list holds the group's matmuls
    pub mats: Vec<usize>,
    /// the activation cells' range width: 8 everywhere, 12 for the widened
    /// attention output's group.  The offset is `2^(act_bits - 1)`.
    pub act_bits: u32,
    /// Under `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1` these are the
    /// *mirror* -- the device columns, back on the host, because
    /// `freivalds_interaction` still reads them there.  The commitment takes
    /// `dev`, so they never go back up.
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub n_weights: usize,
    /// the same columns, built on the device instead
    pub dev: Option<device::DevFreivalds>,
}

impl FreivaldsComponent {
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.g.log_rows
    }
    pub fn pre_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.pre.len(), None => self.pre.len() };
        n << self.lay.g.log_rows
    }
}

/// Build one Freivalds group's component: activations committed (offset
/// `2^(act_bits-1)`), weights preprocessed, range multiplicities bumped at
/// `act_bits` -- 8 everywhere but the widened attention output's group.
pub fn freivalds_component(w: &FreivaldsWitness, p_target: usize, act_bits: u32,
                           tables: &mut Tables, corrupt_qx: Option<usize>,
                           oversize_qx: Option<usize>) -> FreivaldsComponent {
    let lay = FreivaldsLayout::choose(w.n_in, w.n_out_max(), w.tokens, p_target);
    let rows = 1usize << lay.g.log_rows;
    let off = 1i32 << (act_bits - 1);
    let qmax = off - 1;

    // Which witness generation runs is `device::mode`; the device path has no
    // `corrupt_qx` in it, so a control asks for the host's.
    let witness_gen_mode = device::mode(act_bits);
    assert!(witness_gen_mode == 0 || corrupt_qx.is_none(),
            "{}: the device Freivalds witness generation has no `corrupt_qx` path", w.name);
    if witness_gen_mode == 1 {
        let dev = device::freivalds_component_dev(w, &lay, tables);
        // no mirror: `freivalds_interaction` reads the device columns directly,
        // and nothing else reads a device component's host columns --
        // `base_cells` counts them off `dev`.
        return FreivaldsComponent { name: w.name.into(), lay, mats: w.at(), act_bits,
                                    pre: vec![], base: vec![], n_weights: w.n_weights(),
                                    dev: Some(dev) };
    }
    let mirror = device::Mirror::before(witness_gen_mode, tables);

    // The two trees are filled through the row declaration and handed back in
    // committed order by it, so nothing here counts columns.
    let sh = FreivaldsShape::new(w.name, &lay, w.mats.len(), false);
    // A: one token chunk a row, each cell the signed activation plus the table's
    // offset.  The cells are `Bytes`, so the same call that builds a column is
    // the one that counts what it committed in the range table.
    let base = Activations {
        x: Bytes::<Eval>::write(lay.g.log_rows, act_bits, lay.pq, tables, |p, j| {
            let v = match lay.g.row(j) {
                Some((k, chunk)) => {
                    let t = chunk * lay.pq + p;
                    if t < w.tokens {
                        let idx = t * w.n_in + k;
                        let mut x = w.x[idx];
                        if corrupt_qx == Some(idx) {
                            x = if x < qmax { x + 1 } else { x - 1 };
                        }
                        // the control that puts one committed activation one
                        // past its range table
                        if oversize_qx == Some(idx) { x = off } // x + off = 2^act_bits
                        x + off
                    } else {
                        off
                    }
                }
                None => off,
            };
            v as i64
        }),
    }
    .into_committed();

    let mut weights = vec![];
    for m in &w.mats {
        let mut row = vec![];
        for p in 0..lay.pk {
            let mut col = vec![M31::zero(); rows];
            for (j, slot) in col.iter_mut().enumerate() {
                if let Some((k, chunk)) = lay.g.row(j) {
                    let i = chunk * lay.pk + p;
                    if i < m.n_out {
                        *slot = m31_of(m.w[k * m.n_out + i] as i64);
                    }
                }
            }
            row.push(eval_of(lay.g.log_rows, col));
        }
        weights.push(row);
    }
    // the weights, and after them the columns the grouping owns, which the
    // fold primitive writes off the same layout the AIR reads them against
    let pre = Known { weights }.into_committed(sh.groups().write());
    assert_eq!(base.len(), sh.n_base(), "{}: the base tree is not the row's", w.name);
    assert_eq!(pre.len(), sh.n_pre(), "{}: the preprocessed tree is not the row's", w.name);

    if let Some(m) = mirror {
        m.check(w, &lay, tables, &base, &pre);
    }
    FreivaldsComponent { name: w.name.into(), lay, mats: w.at(), act_bits, pre, base,
                         n_weights: w.n_weights(), dev: None }
}
