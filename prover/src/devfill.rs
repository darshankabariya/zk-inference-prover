//! B7a milestone 4 (19.6): the dequant witness fill, on the device.
//!
//! 18.7's row 1 is "the four family fills, 1.98 s and 46% of the wall, the only
//! row with no shortcut". This is the first of the four, and what it moves is
//! not only the arithmetic:
//!
//! | what the host does per component | what happens here |
//! |---|---|
//! | `fill_mat_row` over 2^17 rows x `per_row` slots | one kernel, one thread per committed row |
//! | `eval_of` -- the coset shuffle and the SIMD repack, 37% of trace generation | nothing: the thread owns the output index and reads `m4_perm(i)` |
//! | per-block `HashMap` multiplicities, merged serially | a warp-aggregated scatter-add, one D2H of the counters |
//! | 1.26 GB of columns uploaded at `Session::commit` | nothing: the columns are already there |
//!
//! What *is* uploaded is the witness, not the trace: `acc` as `i32` (which
//! `kmax <= 21` makes lossless -- every magnitude the normalisation table can
//! describe is below `2^30`, asserted below), the per-token and per-channel
//! scales, and the two shared tables, which are cached across components.
//!
//! `ZK_DEVICE_WITNESS_GEN_DEQUANT` says which path runs, on the same three-value pattern
//! `ZK_GENERATED_LOGUP` uses: 0 the host fill, 1 the device fill, 2 both with every
//! cell asserted equal. `ZK_KERNEL_REFERENCE_CHECK=0` turns off the f64 cross-check inside
//! the kernel.

use std::sync::{Arc, Mutex, OnceLock};

use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::dequant::{Map, Spec, Variant as DeqVariant};
use crate::layer::{Mat, MatLayout, Tables};

/// Mirrors `B65FillCfg` in `b65_fill.cuh`, field for field and in order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FillCfg {
    pub variant: u32,
    pub sticky: u32,
    pub kmax: u32,
    pub do_ref: u32,
    pub tokens: u32,
    pub n_out: u32,
    pub per_row: u32,
    pub chunks: u32,
    pub log_rows: u32,
    pub slot0: u32,
    pub per_slot: u32,
    pub chan_w: u32,
    pub f_acc: u32,
    pub f_sg: u32,
    pub f_m2: u32,
    pub f_ph: u32,
    pub f_st: u32,
    pub f_r: u32,
    pub f_m7: u32,
    pub f_ef: u32,
    pub f_nz: u32,
    pub f_eo: u32,
    pub f_mul: [u32; 6],
    pub sh0: u32,
    pub sh1: u32,
    pub n_norm: u32,
    pub n_frac: u32,
    pub n_mul: u32,
    pub n_r8: u32,
    pub mutate: u32,
}

/// `ZK_MUT_WITNESS_GEN_DEQUANT`: one deliberate defect in the device fill, so mode 2's
/// check has negative controls. The names are the kernel's.
fn fill_mut() -> u32 {
    match std::env::var("ZK_MUT_WITNESS_GEN_DEQUANT").ok().as_deref() {
        None | Some("") | Some("none") => 0,
        Some("noperm") => 1,
        Some("chan") => 2,
        Some("nobump") => 3,
        Some("ref32") => 4,
        Some("refaway") => 5,
        Some(other) => panic!("unknown ZK_MUT_WITNESS_GEN_DEQUANT {other:?}"),
    }
}

/// `Fields`' "this variant has no such field", as the kernel spells it.
const NONE: u32 = u32::MAX;

unsafe extern "C" {
    fn b65_fill_dequant(
        acc: *const i32,
        jx: *const i32,
        jw: *const i32,
        sx: *const u32,
        sw: *const u32,
        norm3: *const u32,
        mul2: *const u32,
        base_cols: *const *mut u32,
        n_base: u32,
        pre_cols: *const *mut u32,
        n_pre: u32,
        m_norm: *mut u32,
        m_frac: *mut u32,
        m_mul: *mut u32,
        m_r8: *mut u32,
        counts: *mut u32,
        want_out: *mut u32,
        cfg: FillCfg,
    );
}

/// The two shared tables, resident. They are the same for every component of a
/// run -- `kmax` and the rounding mode fix them -- so they are uploaded once
/// and keyed on exactly what generates them.
pub struct DevTables {
    norm3: Uint32Vec,
    mul2: Uint32Vec,
    n_norm: u32,
    n_mul: u32,
}

type TablesKey = (u32, u32, usize, usize);

fn tables_cache() -> &'static Mutex<Vec<(TablesKey, Arc<DevTables>)>> {
    static CACHE: OnceLock<Mutex<Vec<(TablesKey, Arc<DevTables>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(vec![]))
}

fn dev_tables(t: &Tables) -> Arc<DevTables> {
    let key: TablesKey = (
        t.kmax,
        u32::from(t.round.needs_sticky()),
        t.norm.len(),
        t.mul.len(),
    );
    let mut c = tables_cache().lock().unwrap();
    if let Some((_, v)) = c.iter().find(|(k, _)| *k == key) {
        return v.clone();
    }
    // `(m7, expfield, nz)` and `(m7, carry)`, interleaved: one row is one
    // read, and the whole of both tables is a few hundred kB.
    let mut norm3 = Vec::with_capacity(3 * t.norm.len());
    for r in &t.norm {
        norm3.extend_from_slice(&[r.m7, r.expfield, r.nz]);
    }
    let mut mul2 = Vec::with_capacity(2 * t.mul.len());
    for r in &t.mul {
        mul2.extend_from_slice(&[r.m7, r.carry]);
    }
    let v = Arc::new(DevTables {
        n_norm: t.norm.len() as u32,
        n_mul: t.mul.len() as u32,
        norm3: up_u32(norm3),
        mul2: up_u32(mul2),
    });
    c.push((key, v.clone()));
    v
}

fn up_u32(v: Vec<u32>) -> Uint32Vec {
    let bytes = (v.len() * 4) as u64;
    let t = std::time::Instant::now();
    let d = Uint32Vec::from_vec(v);
    crate::device::note_witness(bytes, t.elapsed().as_secs_f64());
    d
}

/// One component's columns, on the device, in committed order.
pub struct DevFill {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    /// B7a milestone 4 (section 3 row 1): the four multiplicity counters this
    /// fill scatter-added into, left where it left them. `b65_pre_fill` sums
    /// one per component into the column, so with `ZK_DEVICE_PREPROCESSED=1` the download and
    /// the parallel merge below never happen -- 23.1's `m_exp`, four more times.
    pub m_norm: Uint32Vec,
    pub m_frac: Uint32Vec,
    pub m_mul: Uint32Vec,
    pub m_r8: Uint32Vec,
    pub ref_bad: usize,
    /// the f64 cross-check's own result per (coset row, slot), for
    /// `ZK_DEVICE_WITNESS_GEN_DEQUANT=2`; `None` on the prover's path
    pub want: Option<Vec<u32>>,
}

/// Build one matmul dequant component on the device.
///
/// `tables` is read for the shared tables and *written* for the
/// multiplicities: the counters come back and are added in, which is the same
/// arithmetic the host's block-local `HashMap`s do and in an order that cannot
/// matter, because counting is commutative.
///
/// `output_claim` is the plan's `has_output_claim` for this matmul, and it is the whole of
/// stream M's head fix. The kernel's preprocessed list is the per-channel
/// scale columns and five selectors; a component that publishes an output-side
/// claim commits three more (`layer.rs`, the `if output_claim` block), and until this
/// parameter existed the device fill was never told. It sized `n_pre` three
/// columns short -- 506 against the host's 509 on `lm_head`, the sequence's
/// only o-side matmul -- and every later component's preprocessed indices
/// shifted by three. The three are functions of the row index alone, so they
/// are built here on the host and appended in the host's order; nothing about
/// `b65_fill_dequant` moves, which is what keeps the installed kernel
/// fingerprints out of it.
pub fn mat_component_dev(
    m: &Mat,
    spec: &Spec,
    lay: &MatLayout,
    map: &Map,
    tables: &mut Tables,
    with_want: bool,
    output_claim: bool,
) -> DevFill {
    let rows = 1usize << lay.log_rows;
    let f = map.f;
    let sticky = spec.round.needs_sticky();
    let chan_w = if spec.variant == DeqVariant::Chain { 2 } else { 1 };
    // what the kernel writes. The o-side trio is appended after it returns, so
    // this stays exactly the count `b65_fill_dequant` has always been handed.
    let n_pre = chan_w * lay.per_row + 5;
    let do_ref = spec.round == crate::bf16::Round::Even
        && crate::util::m4_switch("ZK_KERNEL_REFERENCE_CHECK");

    // `norm_at` bounds `k` by the *table's* kmax and `normalise` by the
    // spec's; the kernel carries one number, so the two have to be the one
    // number they have always been.
    assert_eq!(spec.kmax, tables.kmax, "{}: spec and table kmax disagree", m.name);
    let t_up = std::time::Instant::now();
    let tbl = dev_tables(tables);

    // The witness. `acc` is `i64` on the host and every magnitude the
    // normalisation table can describe is below `2^30`, so `i32` is lossless
    // -- and checked, because a witness that broke it would otherwise be
    // silently truncated instead of loudly out of the table's window. One
    // parallel pass does the narrowing and the check together: at the design
    // point this is 2.5 M accumulators per component and two serial passes over
    // them cost more than the kernel does.
    use rayon::prelude::*;
    let wide = std::sync::atomic::AtomicBool::new(false);
    let acc32: Vec<u32> = m.acc
        .par_iter()
        .map(|a| {
            if *a < i32::MIN as i64 || *a > i32::MAX as i64 {
                wide.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            *a as i32 as u32
        })
        .collect();
    assert!(
        !wide.load(std::sync::atomic::Ordering::Relaxed),
        "{}: an accumulator does not fit i32; the kmax window cannot hold it either",
        m.name
    );
    crate::util::m4_add("gen/deq_dev_acc", t_up.elapsed().as_secs_f64());
    let t_h2d = std::time::Instant::now();
    let d_acc = up_u32(acc32);
    let (d_jx, d_jw, d_sx, d_sw) = match spec.variant {
        DeqVariant::Pow2 => (
            Some(up_u32(m.jx.iter().map(|v| *v as u32).collect())),
            Some(up_u32(m.jw.iter().map(|v| *v as u32).collect())),
            None,
            None,
        ),
        DeqVariant::Chain => (
            None,
            None,
            Some(up_u32(m.sx.iter().flat_map(|(a, b)| [*a, *b]).collect())),
            Some(up_u32(m.sw.iter().flat_map(|(a, b)| [*a, *b]).collect())),
        ),
        _ => unreachable!("matmul components are pow2 or chain"),
    };

    crate::util::m4_add("gen/deq_dev_h2d", t_h2d.elapsed().as_secs_f64());

    // the columns, in committed order, uninitialised because the fill writes
    // every cell of every one of them
    let t_alloc = std::time::Instant::now();
    let pre: Vec<BaseFieldVec> =
        (0..n_pre).map(|_| BaseFieldVec::new_uninitialized(rows)).collect();
    let base: Vec<BaseFieldVec> =
        (0..map.n_cols).map(|_| BaseFieldVec::new_uninitialized(rows)).collect();
    let pre_ptrs: Vec<*mut u32> = pre.iter().map(|c| c.device_ptr as *mut u32).collect();
    let base_ptrs: Vec<*mut u32> = base.iter().map(|c| c.device_ptr as *mut u32).collect();

    let r8 = tables.ridx(8);
    let m_norm = Uint32Vec::new_zeroes(tables.m_norm.len());
    let m_frac = Uint32Vec::new_zeroes(tables.m_frac.len());
    let m_mul = Uint32Vec::new_zeroes(tables.m_mul.len());
    let m_r8 = Uint32Vec::new_zeroes(tables.m_r[r8].len());
    let counts = Uint32Vec::new_zeroes(3);
    let want = (with_want && do_ref).then(|| Uint32Vec::new_zeroes(rows * lay.per_row));
    crate::util::m4_add("gen/deq_dev_alloc", t_alloc.elapsed().as_secs_f64());
    crate::util::m4_add("gen/deq_dev_up", t_up.elapsed().as_secs_f64());

    let cfg = FillCfg {
        variant: u32::from(spec.variant == DeqVariant::Chain),
        sticky: u32::from(sticky),
        kmax: spec.kmax,
        do_ref: u32::from(do_ref),
        tokens: m.tokens as u32,
        n_out: m.n_out as u32,
        per_row: lay.per_row as u32,
        chunks: lay.chunks as u32,
        log_rows: lay.log_rows,
        slot0: map.slot0 as u32,
        per_slot: map.per_slot as u32,
        chan_w: chan_w as u32,
        f_acc: off(f.acc),
        f_sg: off(f.sg),
        f_m2: off(f.m2),
        f_ph: off(f.ph),
        f_st: off(f.st),
        f_r: off(f.r),
        f_m7: off(f.m7),
        f_ef: off(f.ef),
        f_nz: off(f.nz),
        f_eo: off(f.eo),
        f_mul: [
            off(f.mul[0][0]), off(f.mul[0][1]), off(f.mul[0][2]),
            off(f.mul[1][0]), off(f.mul[1][1]), off(f.mul[1][2]),
        ],
        sh0: map.shared[0] as u32,
        sh1: *map.shared.get(1).unwrap_or(&0) as u32,
        n_norm: tbl.n_norm,
        n_frac: tables.frac.len() as u32,
        n_mul: tbl.n_mul,
        n_r8: tables.m_r[r8].len() as u32,
        mutate: fill_mut(),
    };

    let t_k = std::time::Instant::now();
    let null32 = std::ptr::null::<u32>();
    unsafe {
        b65_fill_dequant(
            d_acc.device_ptr as *const i32,
            d_jx.as_ref().map(|d| d.device_ptr as *const i32).unwrap_or(std::ptr::null()),
            d_jw.as_ref().map(|d| d.device_ptr as *const i32).unwrap_or(std::ptr::null()),
            d_sx.as_ref().map(|d| d.device_ptr).unwrap_or(null32),
            d_sw.as_ref().map(|d| d.device_ptr).unwrap_or(null32),
            tbl.norm3.device_ptr,
            tbl.mul2.device_ptr,
            base_ptrs.as_ptr(),
            base_ptrs.len() as u32,
            pre_ptrs.as_ptr(),
            pre_ptrs.len() as u32,
            m_norm.device_ptr as *mut u32,
            m_frac.device_ptr as *mut u32,
            m_mul.device_ptr as *mut u32,
            m_r8.device_ptr as *mut u32,
            counts.device_ptr as *mut u32,
            want.as_ref().map(|d| d.device_ptr as *mut u32).unwrap_or(std::ptr::null_mut()),
            cfg,
        );
    }
    let kernel_secs = t_k.elapsed().as_secs_f64();
    crate::util::m4_add("gen/deq_dev_kernel", kernel_secs);

    // ---- stream M: the o-side trio, in the host's order --------------------
    // `layer.rs`'s `if output_claim` block, cell for cell. `oin` is an ordinary step
    // from one chunk to the next inside a token, `obd` the jump at the row
    // that begins a token, and the third says the row is live at all. Each is
    // a function of `j` and the layout, so none of them needs the witness and
    // none of them belongs in the kernel.
    let mut pre = pre;
    if output_claim {
        let t_o = std::time::Instant::now();
        let first = |j: usize| {
            matches!(lay.at_opt(m.tokens, j), Some((_, b)) if b + 1 == lay.chunks)
        };
        let sel = |fsel: &dyn Fn(usize) -> bool| {
            let col = crate::common::eval_of(
                lay.log_rows,
                (0..rows).map(|j| crate::common::m31_u32(fsel(j) as u32)).collect(),
            );
            crate::device::upload_col(&col)
        };
        pre.push(sel(&|j| j > 0 && lay.at_opt(m.tokens, j).is_some() && !first(j)));
        pre.push(sel(&|j| j > 0 && first(j)));
        pre.push(sel(&|j| lay.at_opt(m.tokens, j).is_some()));
        crate::util::m4_add("gen/deq_dev_output_claim", t_o.elapsed().as_secs_f64());
    }

    // the counters, back to the host: a few hundred kB, and 4a forbids H2D of
    // trace columns, not D2H of a counter
    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    assert_eq!(
        got[2], 0,
        "{}: {} accumulators outside the kmax {} window or off the shared tables",
        m.name, got[2], spec.kmax
    );
    if crate::devpre::merge_counters() {
        add_into(&mut tables.m_norm, &m_norm.to_vec());
        add_into(&mut tables.m_frac, &m_frac.to_vec());
        add_into(&mut tables.m_mul, &m_mul.to_vec());
        add_into(&mut tables.m_r[r8], &m_r8.to_vec());
    }
    tables.off_table += got[0] as usize;
    let want = want.map(|d| d.to_vec());
    crate::util::m4_add("gen/deq_dev_counters", t_d.elapsed().as_secs_f64());
    eprintln!(
        "fill-dev: {} {} shape 2^{} x {} slots -- {} base and {} pre columns, \
         kernel {:.4} s",
        m.name,
        if spec.variant == DeqVariant::Chain { "chain" } else { "pow2" },
        lay.log_rows,
        lay.per_row,
        base.len(),
        pre.len(),
        kernel_secs
    );

    DevFill { pre, base, m_norm, m_frac, m_mul, m_r8, ref_bad: got[1] as usize, want }
}

fn off(v: usize) -> u32 {
    if v == usize::MAX { NONE } else { v as u32 }
}

fn add_into(dst: &mut [u32], src: &[u32]) {
    assert_eq!(dst.len(), src.len(), "multiplicity counter length");
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}
