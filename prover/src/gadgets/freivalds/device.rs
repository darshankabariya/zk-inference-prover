//! The Freivalds check's witness generation, on the device.
//!
//! The uploads, the packing helpers and the counter mirrors are still
//! `dev_attention_freivalds_witness_gen.rs`'s, shared with the key tiles.

use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::dev_attention_freivalds_witness_gen::{FGrid, FMult, Fam, add_into, pack_i8,
                                                 pack_i8_into, ptrs, timed_up_into, uninit,
                                                 witness_gen_mut, zeroes};
use crate::gadgets::freivalds::columns::FreivaldsShape;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::layer::{Layer, Tables};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FFreivaldsCfg {
    pub g: FGrid,
    pub pq: u32,
    pub pk: u32,
    pub n_in: u32,
    pub tokens: u32,
    pub n_mats: u32,
    pub mutate: u32,
}

unsafe extern "C" {
    fn zk_witness_gen_freivalds(
        qx: *const i8,
        qw: *const *const i8,
        n_out: *const u32,
        base_cols: *const *mut u32,
        n_base: u32,
        pre_cols: *const *mut u32,
        n_pre: u32,
        m_r8: FMult,
        counts: *mut u32,
        cfg: FFreivaldsCfg,
    );
}

/// One Freivalds group's columns, on the device, in committed order.
pub struct DevFreivalds {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    pub log_rows: u32,
    /// B7a milestone 4 (section 3 row 1): the 8-bit range counter, left where
    /// the scatter-add put it.
    pub m_r8: Uint32Vec,
}

/// Build one Freivalds group's component on the device.
///
/// `tables` is read for the 8-bit range table's length and *written* for its
/// multiplicities: the counters come back and are added in, which is the same
/// arithmetic the host's serial bumps do and in an order that cannot matter,
/// because counting is commutative.
pub fn freivalds_component_dev(
    layer: &Layer,
    mat_idx: &[usize],
    lay: &FreivaldsLayout,
    tables: &mut Tables,
) -> DevFreivalds {
    let m0 = &layer.mats[mat_idx[0]];
    let rows = 1usize << lay.g.log_rows;
    let t_up = std::time::Instant::now();

    // The quantised weights are the Freivalds check's bytes (stream F: 231.7 MB a chunk,
    // 218.10 of it `qw`), and their pack is a four-to-a-word narrowing -- the
    // case the pinned pool is for.
    let d_qx = timed_up_into("gen/witness_gen_freivalds_pack", "gen/witness_gen_freivalds_h2d", "freivalds/qx",
                             m0.qx.len().div_ceil(4),
                             |d| pack_i8_into(d, &m0.qx, "freivalds qx"),
                             || pack_i8(&m0.qx, "freivalds qx"));
    let d_qw: Vec<Uint32Vec> = mat_idx
        .iter()
        .map(|i| {
            let qw = &layer.mats[*i].qw;
            timed_up_into("gen/witness_gen_freivalds_pack", "gen/witness_gen_freivalds_h2d", "freivalds/qw",
                          qw.len().div_ceil(4),
                          |d| pack_i8_into(d, qw, "freivalds qw"),
                          || pack_i8(qw, "freivalds qw"))
        })
        .collect();
    let qw_ptrs: Vec<*const i8> = d_qw.iter().map(|d| d.device_ptr as *const i8).collect();
    let n_out: Vec<u32> = mat_idx.iter().map(|i| layer.mats[*i].n_out as u32).collect();
    crate::util::add_phase("gen/witness_gen_freivalds_up", t_up.elapsed().as_secs_f64());

    let t_alloc = std::time::Instant::now();
    // #221: the kernel keeps its own column order -- rewriting it is out of the
    // scheme's scope -- but the trees it fills are sized from the row schema now.
    let sh = FreivaldsShape::new("", lay, mat_idx.len(), false);
    let base = uninit(sh.n_base(), rows);
    let pre = uninit(sh.n_pre(), rows);
    let base_p = ptrs(&base);
    let pre_p = ptrs(&pre);
    let r8 = tables.ridx(8);
    let m_r8 = zeroes(tables.m_r[r8].len());
    let counts = zeroes(3);
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    let cfg = FFreivaldsCfg {
        g: FGrid::of(&lay.g),
        pq: lay.pq as u32,
        pk: lay.pk as u32,
        n_in: m0.n_in as u32,
        tokens: m0.tokens as u32,
        n_mats: mat_idx.len() as u32,
        mutate: witness_gen_mut(Fam::Freivalds),
    };

    // #221: and the shape the kernel is handed has to describe the same trees.
    // A kernel writing a different number of columns than the AIR reads would be
    // a wrong proof rather than a build error, and this is the cheapest thing
    // that ties the two sides together without touching the kernel.
    assert_eq!((base.len(), pre.len()),
               (cfg.pq as usize,
                (cfg.n_mats * cfg.pk) as usize + crate::gadgets::freivalds::FREIVALDS_SEL.len()),
               "the Freivalds witness-generation kernel is handed a different tree than the \
                row schema declares");

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_freivalds(
            d_qx.device_ptr as *const i8,
            qw_ptrs.as_ptr(),
            n_out.as_ptr(),
            base_p.as_ptr(),
            base_p.len() as u32,
            pre_p.as_ptr(),
            pre_p.len() as u32,
            FMult::of(&m_r8),
            counts.device_ptr as *mut u32,
            cfg,
        );
    }
    crate::util::add_phase("gen/witness_gen_freivalds_kernel", t_k.elapsed().as_secs_f64());

    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    if crate::devpre::merge_counters() {
        add_into(&mut tables.m_r[r8], &m_r8.to_vec());
    }
    tables.off_table += got[0] as usize;
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevFreivalds { pre, base, log_rows: lay.g.log_rows, m_r8 }
}
