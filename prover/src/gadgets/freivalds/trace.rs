//! The Freivalds component's committed and preprocessed columns.

use num_traits::Zero;
use stwo::core::fields::m31::M31;

use crate::common::{Eval, eval_of, m31_u32};
use crate::encoding::m31_of;
use crate::dev_attention_freivalds_witness_gen::sub_slice;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::layer::{Layer, Tables};

pub struct FreivaldsComponent {
    pub name: String,
    pub lay: FreivaldsLayout,
    pub mats: Vec<usize>,
    /// Stream Z: the activation cells' range width (8, or 12 for the widened
    /// attention output's group).  The offset is `2^(act_bits - 1)`.
    pub act_bits: u32,
    /// Under `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1` these are the *mirror* -- the device columns, back
    /// on the host, because `freivalds_interaction` still reads them there. The
    /// commitment takes `dev`, so they never go back up.
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub n_weights: usize,
    /// B7a milestone 4 (section 3 row 1): the same columns, born on the device.
    pub dev: Option<crate::dev_attention_freivalds_witness_gen::DevFreivalds>,
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
pub fn freivalds_component(layer: &Layer, name: &str, mat_idx: &[usize], p_target: usize,
                        act_bits: u32, tables: &mut Tables,
                        corrupt_qx: Option<usize>, oversize_qx: Option<usize>) -> FreivaldsComponent {
    let m0 = &layer.mats[mat_idx[0]];
    let n_in = m0.n_in;
    let tokens = m0.tokens;
    let n_out_max = mat_idx.iter().map(|i| layer.mats[*i].n_out).max().unwrap();
    let lay = FreivaldsLayout::choose(n_in, n_out_max, tokens, p_target);
    let rows = 1usize << lay.g.log_rows;
    let off = 1i32 << (act_bits - 1);
    let qmax = off - 1;
    for i in mat_idx {
        assert_eq!(layer.mats[*i].n_in, n_in, "Freivalds group mixes input widths");
        // q/k/v (and gate/up) must share one committed input
        assert_eq!(layer.mats[*i].qx, m0.qx, "Freivalds group inputs differ");
    }

    // B7a milestone 4 (section 3 row 1): which witness generation runs. 1 is the device's
    // alone -- `pre` and `base` become the mirror the interaction reads -- and
    // 2 builds both and asserts every cell equal before either is used.
    //
    // Stream Z: the device witness generation packs activations as int8 and bumps the 8-bit
    // table in its kernel, so a wide group takes the host witness generation whatever the
    // switch says -- the diagonal block key tiles' pattern, and `devpre::note_host_witness_gen`
    // is told in `generate`.
    let witness_gen_mode = if act_bits == 8 { crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS") } else { 0 };
    assert!(witness_gen_mode == 0 || corrupt_qx.is_none(),
            "{name}: the device Freivalds witness generation has no `corrupt_qx` path");
    if witness_gen_mode == 1 {
        let dev = crate::dev_attention_freivalds_witness_gen::freivalds_component_dev(layer, mat_idx, &lay, tables);
        // B7a milestone 4 (section 3 row 1): no mirror. `freivalds_interaction` reads
        // the device columns now, and nothing else reads a device component's host
        // columns -- `base_cells` counts them off `dev`.
        return FreivaldsComponent { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits,
                            pre: vec![], base: vec![],
                            n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                            dev: Some(dev) };
    }
    let snap_tables = |t: &Tables| -> (Vec<u32>, usize) {
        (t.m_r[t.ridx(8)].clone(), t.off_table)
    };
    let snap_before = (witness_gen_mode == 2).then(|| snap_tables(tables));

    let mut base = vec![];
    for p in 0..lay.pq {
        let mut col = vec![M31::zero(); rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = match lay.g.row(j) {
                Some((k, chunk)) => {
                    let t = chunk * lay.pq + p;
                    if t < tokens {
                        let idx = t * n_in + k;
                        let mut x = m0.qx[idx];
                        if corrupt_qx == Some(idx) {
                            x = if x < qmax { x + 1 } else { x - 1 };
                        }
                        // stream Z's bend control: one committed activation set
                        // one past its range table -- the o-limb window
                        if oversize_qx == Some(idx) { x = off } // x + off = 2^act_bits
                        x + off
                    } else {
                        off
                    }
                }
                None => off,
            };
            tables.bump_r(act_bits, v as i64);
            *slot = m31_of(v as i64);
        }
        base.push(eval_of(lay.g.log_rows, col));
    }

    let mut pre = vec![];
    for mi in mat_idx {
        let m = &layer.mats[*mi];
        for p in 0..lay.pk {
            let mut col = vec![M31::zero(); rows];
            for (j, slot) in col.iter_mut().enumerate() {
                if let Some((k, chunk)) = lay.g.row(j) {
                    let i = chunk * lay.pk + p;
                    if i < m.n_out {
                        *slot = m31_of(m.qw[k * m.n_out + i] as i64);
                    }
                }
            }
            pre.push(eval_of(lay.g.log_rows, col));
        }
    }
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.g.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    let ch = |j: usize| lay.g.row(j).map(|(_, c)| c);
    pre.push(sel(&|j| matches!(ch(j), Some(c) if c + 1 < lay.g.chunks)));
    pre.push(sel(&|j| matches!(ch(j), Some(0))));
    pre.push(sel(&|j| j > 0));
    pre.push(sel(&|j| j + 1 == rows));

    if witness_gen_mode == 2 {
        // the check with teeth, before any column has changed hands
        let after = snap_tables(tables);
        let before = snap_before.unwrap();
        let dev = crate::dev_attention_freivalds_witness_gen::freivalds_component_dev(layer, mat_idx, &lay, tables);
        crate::dev_attention_freivalds_witness_gen::check_cols(&format!("freivalds {name} base"), &base, &dev.base);
        crate::dev_attention_freivalds_witness_gen::check_cols(&format!("freivalds {name} pre"), &pre, &dev.pre);
        let r8 = tables.ridx(8);
        crate::dev_attention_freivalds_witness_gen::check_delta(&format!("freivalds {name} m_r8"), &before.0, &after.0,
                                     &sub_slice(&tables.m_r[r8], &after.0));
        assert_eq!(tables.off_table - after.1, after.1 - before.1,
                   "freivalds {name}: off-table bumps differ");
        // the device's own bumps are the check's, not the proof's
        tables.m_r[r8].copy_from_slice(&after.0);
        tables.off_table = after.1;
        eprintln!("witness_gen-dev: freivalds {name} -- {} base and {} pre columns, {} cells, \
the 8-bit range multiplicities all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.g.log_rows);
    }
    FreivaldsComponent { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits, pre, base,
                 n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                 dev: None }
}
