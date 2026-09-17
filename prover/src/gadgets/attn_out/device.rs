//! The attnout component's witness generation, on the device.
//!
//! The uploads, the packing helpers and the counter mirrors are still
//! `dev_attention_freivalds_witness_gen.rs`'s, shared with the Freivalds check
//! and the key tiles; #197's fourth PR says where they end up.

use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::dev_attention_freivalds_witness_gen::{Fam, FMult, add_into, pack_i32_from_i32,
                                                 pack_i64, ptrs, timed_up, uninit,
                                                 witness_gen_mut, zeroes};
use crate::gadgets::attn_out::layout::AoLayout;
use crate::gadgets::attn_out::witness::AttnOut;
use crate::layer::Tables;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FAoCfg {
    pub nh: u32,
    pub s: u32,
    pub hd: u32,
    pub per_row: u32,
    pub chunks: u32,
    pub log_rows: u32,
    pub lead: u32,
    pub do_ref: u32,
    pub mutate: u32,
    /// stream JJ (E11): the exponent window's two offsets and the accumulator
    /// window's two limb widths
    pub jv_min: i32,
    pub jo_max: i32,
    pub acc_lo_bits: u32,
    pub acc_hi_bits: u32,
}

unsafe extern "C" {
    fn zk_witness_gen_attnout(
        acc: *const u32,
        den: *const u32,
        jo: *const i32,
        jv: *const i32,
        base_cols: *const *mut u32,
        n_base: u32,
        pre_cols: *const *mut u32,
        n_pre: u32,
        m_qnz: FMult,
        m_rem: FMult,
        m_acc_lo: FMult,
        m_acc_hi: FMult,
        q_out: *mut i32,
        want_out: *mut u32,
        counts: *mut u32,
        cfg: FAoCfg,
    );
}

pub struct DevAo {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    pub log_rows: u32,
    /// B7a milestone 4 (section 3 row 1): the `q != 0` and remainder counters.
    pub m_qnz: Uint32Vec,
    pub m_rem: Uint32Vec,
    /// stream JJ (E11): the accumulator window's two limb counters
    pub m_acc_lo: Uint32Vec,
    pub m_acc_hi: Uint32Vec,
    pub ref_bad: usize,
    pub q_out: Vec<i64>,
    /// the f64 reference's own answer per (coset row, slot), for `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=2`
    pub want: Option<Vec<u32>>,
}

#[allow(clippy::too_many_arguments)]
pub fn ao_component_dev(
    ao: &AttnOut,
    lay: &AoLayout,
    q_bits: u32,
    rem_bits: u32,
    acc_bits: (u32, u32),
    jv_min: i32,
    jo_max: i32,
    tables: &mut Tables,
    with_want: bool,
) -> DevAo {
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    let t_up = std::time::Instant::now();
    let d_acc = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/acc",
                         || pack_i64(&ao.acc));
    let d_den = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/den",
                         || pack_i64(&ao.den));
    let d_jo = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/jo",
                        || pack_i32_from_i32(&ao.jo));
    let d_jv = timed_up("gen/witness_gen_ao_pack", "gen/witness_gen_ao_h2d", "ao/jv",
                        || pack_i32_from_i32(&ao.jv));
    crate::util::add_phase("gen/witness_gen_ao_up", t_up.elapsed().as_secs_f64());

    let t_alloc = std::time::Instant::now();
    let base = uninit(2 + 9 * pr, rows);
    let pre = uninit(pr + 6, rows);
    let base_p = ptrs(&base);
    let pre_p = ptrs(&pre);
    let ri = tables.ridx(rem_bits);
    let (rlo, rhi) = (tables.ridx(acc_bits.0), tables.ridx(acc_bits.1));
    let m_qnz = zeroes(tables.m_qnz.len());
    let m_rem = zeroes(tables.m_r[ri].len());
    let m_acc_lo = zeroes(tables.m_r[rlo].len());
    let m_acc_hi = zeroes(tables.m_r[rhi].len());
    let counts = zeroes(3);
    let q_out = zeroes(ao.nh * ao.s * ao.hd);
    let want = with_want.then(|| zeroes(rows * pr));
    crate::util::add_phase("gen/witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    assert_eq!(tables.m_qnz.len(), 1usize << q_bits, "attnout qnz table width");
    let cfg = FAoCfg {
        nh: ao.nh as u32,
        s: ao.s as u32,
        hd: ao.hd as u32,
        per_row: pr as u32,
        chunks: lay.chunks as u32,
        log_rows: lay.log_rows,
        lead: lay.lead as u32,
        // `ao_component` has no switch for its f64 reference: it computes it on
        // every active site, and so does this.
        do_ref: 1,
        mutate: witness_gen_mut(Fam::Ao),
        // stream JJ (E11): the exponent window's offsets and the accumulator
        // window's two limb widths.  `e = jv - jo` is what the kernel derives
        // `num` and `pj` from; the key it commits is `(jv - jv_min) + (jo_max
        // - jo)`, which is `e - e_lo`.
        jv_min,
        jo_max,
        acc_lo_bits: acc_bits.0,
        acc_hi_bits: acc_bits.1,
    };

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_attnout(
            d_acc.device_ptr,
            d_den.device_ptr,
            d_jo.device_ptr as *const i32,
            d_jv.device_ptr as *const i32,
            base_p.as_ptr(),
            base_p.len() as u32,
            pre_p.as_ptr(),
            pre_p.len() as u32,
            FMult::of(&m_qnz),
            FMult::of(&m_rem),
            FMult::of(&m_acc_lo),
            FMult::of(&m_acc_hi),
            q_out.device_ptr as *mut i32,
            want.as_ref().map(|d| d.device_ptr as *mut u32).unwrap_or(std::ptr::null_mut()),
            counts.device_ptr as *mut u32,
            cfg,
        );
    }
    crate::util::add_phase("gen/witness_gen_ao_kernel", t_k.elapsed().as_secs_f64());

    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    assert_eq!(
        got[2], 0,
        "attnout: {} sites with a non-positive denominator or outside the no-wrap window",
        got[2]
    );
    if crate::devpre::merge_counters() {
        add_into(&mut tables.m_qnz, &m_qnz.to_vec());
        add_into(&mut tables.m_r[ri], &m_rem.to_vec());
        add_into(&mut tables.m_r[rlo], &m_acc_lo.to_vec());
        add_into(&mut tables.m_r[rhi], &m_acc_hi.to_vec());
    }
    // The `EPOW` window's multiplicities never leave the host: `e = jv - jo` is
    // a function of `ao.jv`, `ao.jo` and the layout, so the whole column is one
    // cheap loop over rows and no counter has to come back off the card.
    for j in 0..rows {
        match lay.at(ao.nh, ao.s, j) {
            None => {
                for _ in 0..pr {
                    tables.bump_ek(tables.e_lo, "ao_epow_lead");
                }
            }
            Some((h, t, b)) => {
                for s in 0..pr {
                    let e = ao.jv[h * ao.hd + b * pr + s] - ao.jo[t];
                    tables.bump_ek(e, "ao_epow");
                }
            }
        }
    }
    tables.off_table += got[0] as usize;
    let q_out: Vec<i64> = q_out.to_vec().into_iter().map(|v| v as i32 as i64).collect();
    let want = want.map(|d| d.to_vec());
    crate::util::add_phase("gen/witness_gen_counters", t_d.elapsed().as_secs_f64());

    DevAo { pre, base, log_rows: lay.log_rows, m_qnz, m_rem, m_acc_lo, m_acc_hi,
            ref_bad: got[1] as usize, q_out, want }
}
