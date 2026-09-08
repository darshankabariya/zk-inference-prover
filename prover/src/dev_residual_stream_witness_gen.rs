//! B11 stream M: the residual stream's witness generation, on the device.
//!
//! `gen/residual_stream` was the last named host term with no device path at all: 49.784 s
//! of seq3's 190.94 s wall at C=4K and 49.28 s of 241.40 at C=16K, 97% of it
//! inside four `residual_stream::residual_stream_component` calls per chunk. The context-independence is
//! the tell -- the residual stream's components are sized by the model's hidden dimension and
//! not by the context -- and it is why this term dominates at C=4K.
//!
//! **Interpreted, not emitted.** The route was a choice and the measurement
//! made it. A `genemit` target compiling each component's straight-line `Prog` to
//! CUDA would have been the other one; against it, a residual-stream component's program is
//! tiny and uniform (255 add slots over 256 leaves on `gn1`/`gn2`, 31 over 32
//! on `gn1b`/`gn2b`) while the rows are many, so the per-slot interpretive
//! overhead sits against a normalisation and four table lookups and disappears.
//! And an emit would have to be re-run for every new residual-stream shape, on the family
//! whose *constraint* kernel took 617.7 s of ptxas before stream K split it.
//! `zk_residual_stream_witness_gen.cuh` carries the argument in full.
//!
//! What the kernel needs from here is the program as *column indices*: every
//! `Src` resolves to the columns its `(sm, ef)` pair lives in, so the kernel
//! keeps no per-row state and reads its operands back out of cells it wrote.
//! `Prog::src_cols` is that resolution, and it already existed for the AIR.
//!
//! `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM` says which path runs, on the three-value pattern every witness generation in
//! this series uses: 0 the host, 1 the device, 2 both with every cell, every
//! multiplicity delta, `off_table`, `bad_exp` and both natural-value arrays
//! asserted equal.

use std::sync::{Mutex, OnceLock};

use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::common::{Eval, eval_of, log_rows_for, m31_u32};
use crate::residual_stream::{ResidualStreamTables, ResidualStreamWitness, Kind, Prog, Src, UnKind};

// ---------------------------------------------------------------- the FFI shapes --

#[repr(C)]
#[derive(Clone, Copy)]
struct FGroup {
    n: u32,
    leaf_col: u32,
    width: u32,
    un: u32,
    flat: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FSlot {
    a_sm: u32,
    a_ef: u32,
    b_sm: u32,
    b_ef: u32,
    base: u32,
    o_sm: u32,
    o_ef: u32,
    from_slot: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FMult {
    m: *mut u32,
    n: u32,
}

impl FMult {
    fn of(v: &Uint32Vec) -> Self {
        FMult { m: v.device_ptr as *mut u32, n: v.size as u32 }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FResidualStreamCfg {
    log_rows: u32,
    rows: u32,
    live: u32,
    n_groups: u32,
    n_leaves: u32,
    n_slots: u32,
    n_cols: u32,
    kind: u32,
    sticky: u32,
    kmax: u32,
    out_w: u32,
    mutate: u32,
}

const NONE: u32 = u32::MAX;

unsafe extern "C" {
    fn zk_witness_gen_residual_stream(
        wit: *const u32,
        groups: *const FGroup,
        slots: *const FSlot,
        norm3: *const u32,
        n_norm: u32,
        smul3: *const i32,
        n_smul: u32,
        un2: *const *const i32,
        n_un: u32,
        n_val: u32,
        out_slot: *const u32,
        cols: *const *mut u32,
        out_pat: *mut u32,
        out_nat: *mut i32,
        in_nat: *mut i32,
        m_pow: FMult,
        m_norm: FMult,
        m_val: FMult,
        m_frac: FMult,
        m_smul: FMult,
        m_un: *const FMult,
        counts: *mut u32,
        cfg: FResidualStreamCfg,
    );
}

// ------------------------------------------------------------------ the switches --

/// `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM`: 0 the host, 1 the device, 2 both with every cell asserted.
pub fn mode() -> u32 {
    crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM")
}

/// Whether anything on the host will read the two natural arrays.
///
/// **B11 stream N: the answer is always yes, and that is a finding.** Brief N
/// scoped this download as existing "only to bring the bindings' arrays back", so
/// that moving the binding fold onto the card would delete it. Opening it says
/// otherwise: `in_nat` / `out_nat` have three host readers, and only one of
/// them is the fold.
///
///  * `residual_stream::residual_stream_interaction`'s Horner bindings -- the fold, which stream N moves
///    to the device (`devacc::residual_stream_bind`).
///  * `boundary::cols` (`boundary.rs:130`), which builds the **boundary tree's
///    committed columns** out of the same sequence, packed `sm * PACK + ef`
///    per (row, slot). That is stream H's chunk-to-chunk chaining, it is on in
///    the sequence's pins (`--bounds`), and it is a committed column rather
///    than a check -- so it cannot simply be dropped, and a device path for it
///    is its own lever with its own gate.
///  * `ResidualStreamComponent::binding_claim`, the independent recomputation of every claim
///    from the witness rather than from the trace (`compose.rs:2447`,
///    `residual_stream.rs:2165`). The bar keeps checks, and with the fold on the card
///    this one gains teeth: it now cross-checks the *device's* claim against a
///    host recomputation on every run, not only under mode 2.
///
/// So the download stays and this returns true unconditionally. It is kept as
/// a named predicate rather than deleted because it is where the reasoning
/// lives, and because the day `boundary::cols` moves to the card it is the one
/// line that changes.
pub fn binding_host() -> bool {
    true
}

/// `ZK_MUT_WITNESS_GEN_RESIDUAL_STREAM`: one deliberate defect in the device witness generation, so mode 2's
/// check has negative controls. The names are the kernel's.
fn residual_stream_witness_gen_mut() -> u32 {
    match std::env::var("ZK_MUT_WITNESS_GEN_RESIDUAL_STREAM").ok().as_deref() {
        None | Some("") | Some("none") => 0,
        Some("noperm") => 1,
        Some("anchor") => 2,
        Some("nobump") => 3,
        Some("nosticky") => 4,
        Some("clamp") => 5,
        Some("nomul") => 6,
        Some("offby") => 7,
        Some(other) => panic!("unknown ZK_MUT_WITNESS_GEN_RESIDUAL_STREAM {other:?}"),
    }
}

// -------------------------------------------------------------------- the tables --

/// The two value tables the kernel reads, and the unary images, resident.
///
/// They are a function of the round and `kmax` alone -- the same for every
/// component of a run and, for the unaries, of the plan's kind list -- so they are
/// uploaded once and keyed on exactly what generates them.
pub struct DevGTables {
    norm3: Uint32Vec,
    smul3: Uint32Vec,
    un2: Vec<Uint32Vec>,
    n_norm: u32,
    n_smul: u32,
    n_val: u32,
}

impl DevGTables {
    /// The unary tables' device pointers, per call. They are not cached
    /// alongside the arrays: a raw pointer is neither `Send` nor `Sync`, and a
    /// process-wide cache has to be both. The arrays are what is worth caching;
    /// re-reading seven pointers out of them costs nothing.
    fn un_ptrs(&self) -> Vec<*const i32> {
        self.un2.iter().map(|d| d.device_ptr as *const i32).collect()
    }
}

type GKey = (u32, u32, usize, usize, Vec<UnKind>);

fn gtables_cache() -> &'static Mutex<Vec<(GKey, std::sync::Arc<DevGTables>)>> {
    static CACHE: OnceLock<Mutex<Vec<(GKey, std::sync::Arc<DevGTables>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(vec![]))
}

fn dev_gtables(t: &ResidualStreamTables) -> std::sync::Arc<DevGTables> {
    let key: GKey = (t.kmax, u32::from(t.sticky), t.norm.len(), t.smul.len(), t.uns.clone());
    let mut c = gtables_cache().lock().unwrap();
    if let Some((_, v)) = c.iter().find(|(k, _)| *k == key) {
        return v.clone();
    }
    let mut norm3 = Vec::with_capacity(t.norm.len() * 3);
    for r in &t.norm {
        norm3.push(r.m7);
        norm3.push(r.expfield);
        norm3.push(r.nz);
    }
    // `sgn_of(row[2])` on the host, so the sign is resolved here and the kernel
    // reads a plain `i32`.
    let mut smul3 = Vec::with_capacity(t.smul.len() * 3);
    for r in &t.smul {
        // the sign is resolved here, so the kernel reads a plain `i32` and
        // never has to know the field encoding
        smul3.push(crate::residual_stream::sgn_of(r[2]) as i32 as u32);
        smul3.push(r[3].0);
        smul3.push(r[4].0);
    }
    // One `(osm, oef)` per `val_at` index per unary kind. This is the table the
    // AIR reads, and using it is what keeps `rsqrt` and `silu` off the device:
    // `un_image_parts` is f64 -> f32 -> bf16 and would have to match bit for
    // bit, while the table is the host's own answer.
    let un2: Vec<Uint32Vec> = t
        .un
        .iter()
        .map(|tab| {
            let mut v: Vec<u32> = Vec::with_capacity(tab.len() * 2);
            for r in tab {
                v.push(crate::residual_stream::sgn_of(r[2]) as i32 as u32);
                v.push(r[3].0);
            }
            up_u32(v)
        })
        .collect();
    let n_val = t.val.len() as u32;
    let v = std::sync::Arc::new(DevGTables {
        n_norm: t.norm.len() as u32,
        n_smul: t.smul.len() as u32,
        n_val,
        norm3: up_u32(norm3),
        smul3: up_u32(smul3),
        un2,
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

// --------------------------------------------------------------------- the component --

/// One residual-stream component's columns, on the device, plus the host-side arrays the
/// bindings, the next component and the reference check read back.
pub struct DevResidualStream {
    pub pre: Vec<BaseFieldVec>,
    pub base: Vec<BaseFieldVec>,
    /// the multiplicity counters this witness generation scatter-added into, left where the
    /// kernel left them
    pub m_pow: Uint32Vec,
    pub m_norm: Uint32Vec,
    pub m_val: Uint32Vec,
    pub m_frac: Uint32Vec,
    pub m_smul: Uint32Vec,
    pub m_un: Vec<Uint32Vec>,
    pub off_table: usize,
    pub bad_exp: usize,
    /// `(sm, ef)` per leaf and per output, in row order, regrouped the way
    /// `ResidualStreamComponent` holds them. **Empty when the bindings fold on the device**
    /// (`ZK_DEVICE_RESIDUAL_STREAM_BINDING=1`): nothing on the host reads them then, and bringing them
    /// back is all `gen/residual_stream_witness_gen_d2h` ever was.
    pub in_nat: Vec<Vec<(i64, i64)>>,
    pub out_nat: Vec<Vec<(i64, i64)>>,
    pub outs: Vec<Vec<u16>>,
    /// B11 stream N: the same two arrays where the kernel wrote them, row-flat
    /// and in coset row order, kept alive for `devacc::residual_stream_bind`. The `Uint32Vec`
    /// owns the allocation; the bindings read it as `i32`, which is how
    /// `zk_witness_gen_residual_stream` wrote it.
    pub d_in_nat: Uint32Vec,
    pub d_out_nat: Uint32Vec,
    /// the two arrays' row widths, which are the bindings' strides
    pub n_leaves: usize,
    pub out_w: usize,
}

/// Build one residual-stream component on the device.
///
/// `tables` is read for the value tables and *not* written: the counters come
/// back in `DevResidualStream` and the caller merges them, because mode 2 has to compare
/// the device's delta against the host's before either is used.
pub fn residual_stream_component_dev(w: &ResidualStreamWitness, prog: &Prog, round: crate::bf16::Round,
                                  tables: &ResidualStreamTables) -> DevResidualStream {
    let log_rows = log_rows_for(w.rows, 1);
    let rows = 1usize << log_rows;
    let n_cols = prog.n_cols(round);
    let sticky = round.needs_sticky();
    let n_leaves = prog.n_leaves();

    // ---- the program, as column indices ------------------------------------
    let t_up = std::time::Instant::now();
    let mut flat = 0u32;
    let groups: Vec<FGroup> = prog
        .groups
        .iter()
        .enumerate()
        .map(|(g, gr)| {
            let f = FGroup {
                n: gr.n as u32,
                leaf_col: prog.leaf_at(g, 0) as u32,
                width: (prog.leaf_at(g, 1) - prog.leaf_at(g, 0)) as u32,
                un: if gr.un == UnKind::None { 0 } else { 1 + tables.un_index(gr.un) as u32 },
                flat,
            };
            flat += gr.n as u32;
            f
        })
        .collect();
    assert_eq!(flat as usize, n_leaves, "the row-flat leaf count is the program's");
    let col = |s: &Src| -> (u32, u32, u32) {
        match prog.src_cols(s, round) {
            None => (NONE, NONE, 0),
            Some((a, b)) => (a as u32, b as u32, u32::from(matches!(s, Src::Slot(_)))),
        }
    };
    let slots: Vec<FSlot> = prog
        .slots
        .iter()
        .enumerate()
        .map(|(s, (sa, sb))| {
            let (a_sm, a_ef, a_is) = col(sa);
            let (b_sm, b_ef, b_is) = col(sb);
            let (o_sm, o_ef) = prog.out_cols(s, round);
            FSlot {
                a_sm, a_ef, b_sm, b_ef,
                base: prog.slot_at(s, round) as u32,
                o_sm: o_sm as u32,
                o_ef: o_ef as u32,
                from_slot: a_is | (b_is << 1),
            }
        })
        .collect();
    let out_slot: Vec<u32> =
        prog.outs.iter().flat_map(|o| o.iter().map(|s| *s as u32)).collect();
    let out_w = out_slot.len();

    // ---- the witness, row-flat ---------------------------------------------
    // `w.groups[g]` is `[j * gr.n + i]` per group; the kernel wants one array
    // indexed `[j * n_leaves + flat]`, so the groups interleave here. This is
    // the one host pass this witness generation keeps, and it is a widening copy: 4.2 M
    // values on the sequence's `gn1`.
    use rayon::prelude::*;
    let mut wit = vec![0u32; w.rows * n_leaves];
    wit.par_chunks_mut(n_leaves).enumerate().for_each(|(j, row)| {
        let mut at = 0usize;
        for (g, gr) in prog.groups.iter().enumerate() {
            for i in 0..gr.n {
                row[at] = w.groups[g][j * gr.n + i] as u32;
                at += 1;
            }
        }
    });
    let d_wit = up_u32(wit);
    let d_groups = up_struct(&groups);
    let d_slots = up_struct(&slots);
    let d_out_slot = up_u32(out_slot.clone());
    let tbl = dev_gtables(tables);
    let un_ptrs = tbl.un_ptrs();
    crate::util::add_phase("gen/residual_stream_witness_gen_up", t_up.elapsed().as_secs_f64());

    // ---- the columns and the counters --------------------------------------
    let t_alloc = std::time::Instant::now();
    let base: Vec<BaseFieldVec> =
        (0..n_cols).map(|_| BaseFieldVec::new_uninitialized(rows)).collect();
    let base_ptrs: Vec<*mut u32> = base.iter().map(|c| c.device_ptr as *mut u32).collect();
    let m_pow = Uint32Vec::new_zeroes(tables.m_pow.len());
    let m_norm = Uint32Vec::new_zeroes(tables.m_norm.len());
    let m_val = Uint32Vec::new_zeroes(tables.m_val.len());
    let m_frac = Uint32Vec::new_zeroes(tables.m_frac.len());
    let m_smul = Uint32Vec::new_zeroes(tables.m_smul.len());
    let m_un: Vec<Uint32Vec> =
        tables.m_un.iter().map(|v| Uint32Vec::new_zeroes(v.len())).collect();
    let m_un_f: Vec<FMult> = m_un.iter().map(FMult::of).collect();
    let d_m_un = up_struct(&m_un_f);
    let counts = Uint32Vec::new_zeroes(3);
    let d_out_pat = Uint32Vec::new_zeroes(rows * out_w.max(1));
    let d_out_nat = Uint32Vec::new_zeroes(rows * out_w.max(1) * 2);
    let d_in_nat = Uint32Vec::new_zeroes(rows * n_leaves * 2);
    crate::util::add_phase("gen/residual_stream_witness_gen_alloc", t_alloc.elapsed().as_secs_f64());

    let cfg = FResidualStreamCfg {
        log_rows,
        rows: rows as u32,
        live: w.rows as u32,
        n_groups: prog.groups.len() as u32,
        n_leaves: n_leaves as u32,
        n_slots: prog.slots.len() as u32,
        n_cols: n_cols as u32,
        kind: u32::from(prog.kind == Kind::Mul),
        sticky: u32::from(sticky),
        kmax: tables.kmax,
        out_w: out_w as u32,
        mutate: residual_stream_witness_gen_mut(),
    };

    let t_k = std::time::Instant::now();
    unsafe {
        zk_witness_gen_residual_stream(
            d_wit.device_ptr,
            d_groups.device_ptr as *const FGroup,
            d_slots.device_ptr as *const FSlot,
            tbl.norm3.device_ptr,
            tbl.n_norm,
            tbl.smul3.device_ptr as *const i32,
            tbl.n_smul,
            un_ptrs.as_ptr(),
            tbl.un2.len() as u32,
            tbl.n_val,
            d_out_slot.device_ptr,
            base_ptrs.as_ptr(),
            d_out_pat.device_ptr as *mut u32,
            d_out_nat.device_ptr as *mut i32,
            d_in_nat.device_ptr as *mut i32,
            FMult::of(&m_pow),
            FMult::of(&m_norm),
            FMult::of(&m_val),
            FMult::of(&m_frac),
            FMult::of(&m_smul),
            d_m_un.device_ptr as *const FMult,
            counts.device_ptr as *mut u32,
            cfg,
        );
    }
    let kernel_secs = t_k.elapsed().as_secs_f64();
    crate::util::add_phase("gen/residual_stream_witness_gen_kernel", kernel_secs);

    // ---- back to the host: the counters and the three natural arrays -------
    let t_d = std::time::Instant::now();
    let got = counts.to_vec();
    assert_eq!(
        got[2], 0,
        "residual-stream witness generation: {} leaves whose unary input is off the value table -- the host \
         evaluates the unary there and the device has no table row to read",
        got[2]
    );
    // B11 stream N: the two `(sm, ef)` arrays come back only if something on
    // the host is going to fold them.
    //
    // They exist for one reader -- `residual_stream::residual_stream_interaction`'s Horner bindings --
    // and under `ZK_DEVICE_RESIDUAL_STREAM_BINDING=1` that fold happens on the card, out of the very
    // arrays this would copy. So the download and the regrouping below are
    // skipped and `in_nat` / `out_nat` stay empty; the device arrays are kept
    // alive in `DevResidualStream` instead, which is what `devacc::residual_stream_bind` reads.
    //
    // `out_pat` is a different array and a different reader: `ref_count`'s f64
    // cross-check compares it against `w.ref_out` on the host. It is one
    // `u32` per output slot rather than two per value, it is not what the bindings
    // read, and the check stays -- so it comes back either way.
    let host_nat = binding_host();
    let flat_in = if host_nat { d_in_nat.to_vec() } else { vec![] };
    let flat_on = if host_nat { d_out_nat.to_vec() } else { vec![] };
    let flat_op = d_out_pat.to_vec();
    // regroup: the kernel writes one row-flat array, `ResidualStreamComponent` holds one
    // vector per group and per output
    let mut in_nat: Vec<Vec<(i64, i64)>> = if host_nat {
        prog.groups.iter().map(|g| Vec::with_capacity(rows * g.n)).collect()
    } else {
        vec![]
    };
    if host_nat {
        for j in 0..rows {
            let mut at = 0usize;
            for (g, gr) in prog.groups.iter().enumerate() {
                for _ in 0..gr.n {
                    let k = (j * n_leaves + at) * 2;
                    in_nat[g].push((flat_in[k] as i32 as i64, flat_in[k + 1] as i32 as i64));
                    at += 1;
                }
            }
        }
    }
    let mut out_nat: Vec<Vec<(i64, i64)>> = if host_nat {
        prog.outs.iter().map(|o| Vec::with_capacity(rows * o.len())).collect()
    } else {
        vec![]
    };
    let mut outs: Vec<Vec<u16>> =
        prog.outs.iter().map(|o| vec![0u16; rows * o.len()]).collect();
    for j in 0..rows {
        let mut at = 0usize;
        for (oi, o) in prog.outs.iter().enumerate() {
            for k in 0..o.len() {
                let f = j * out_w + at;
                if host_nat {
                    out_nat[oi]
                        .push((flat_on[f * 2] as i32 as i64, flat_on[f * 2 + 1] as i32 as i64));
                }
                outs[oi][j * o.len() + k] = flat_op[f] as u16;
                at += 1;
            }
        }
    }
    crate::util::add_phase("gen/residual_stream_witness_gen_d2h", t_d.elapsed().as_secs_f64());

    // ---- the two preprocessed selectors, which are row-index functions -----
    let t_p = std::time::Instant::now();
    let mut end = vec![m31_u32(0); rows];
    end[rows - 1] = m31_u32(1);
    let mut nofirst = vec![m31_u32(1); rows];
    nofirst[0] = m31_u32(0);
    let pre: Vec<BaseFieldVec> = [nofirst, end]
        .into_iter()
        .map(|c| crate::device::upload_col(&eval_of(log_rows, c)))
        .collect();
    crate::util::add_phase("gen/residual_stream_witness_gen_pre", t_p.elapsed().as_secs_f64());

    eprintln!(
        "residual_stream_witness_gen-dev: {} 2^{} x {} slots -- {} base and {} pre columns, {} cells, \
         kernel {:.4} s",
        prog.slots.len(),
        log_rows,
        prog.slots.len(),
        base.len(),
        pre.len(),
        (base.len() + pre.len()) << log_rows,
        kernel_secs
    );

    DevResidualStream {
        pre, base, m_pow, m_norm, m_val, m_frac, m_smul, m_un,
        off_table: got[0] as usize,
        bad_exp: got[1] as usize,
        in_nat, out_nat, outs,
        d_in_nat, d_out_nat, n_leaves, out_w,
    }
}

fn up_struct<T: Copy>(v: &[T]) -> Uint32Vec {
    let words = std::mem::size_of_val(v).div_ceil(4).max(1);
    let mut raw = vec![0u32; words];
    unsafe {
        std::ptr::copy_nonoverlapping(
            v.as_ptr() as *const u8,
            raw.as_mut_ptr() as *mut u8,
            std::mem::size_of_val(v),
        );
    }
    up_u32(raw)
}

/// `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM=2`: the device witness generation against the host's, everything at once.
///
/// Not a sample. A witness generation that is wrong on one cell of one row is wrong, and the
/// point of checking here rather than against a digest is that a mismatch names
/// the column and the cell.
#[allow(clippy::too_many_arguments)]
pub fn check_residual_stream_dev(tag: &str, base: &[Eval], pre: &[Eval], dev: &DevResidualStream,
                                 off_host: usize, bad_host: usize,
                                 in_nat: &[Vec<(i64, i64)>], out_nat: &[Vec<(i64, i64)>],
                                 outs: &[Vec<u16>],
                                 before: &ResidualStreamSnap, after: &ResidualStreamSnap) {
    use stwo::prover::backend::Column as _;
    let cmp = |what: &str, host: &[Eval], got: &[BaseFieldVec]| {
        assert_eq!(host.len(), got.len(), "residual stream {tag}: {what} column count");
        for (i, (a, b)) in host.iter().zip(got).enumerate() {
            let (a, b) = (a.values.to_cpu(), b.to_vec());
            assert_eq!(a.len(), b.len(), "residual stream {tag}: {what} column {i} length");
            if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
                panic!("residual stream {tag}: {what} column {i} differs at cell {k}: \
                       host {:?}, device {:?}", a[k], b[k]);
            }
        }
    };
    cmp("base", base, &dev.base);
    cmp("pre", pre, &dev.pre);
    let delta = |a: &[u32], b: &[u32]| -> Vec<u32> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    };
    let one = |what: &str, h: Vec<u32>, d: Vec<u32>| {
        assert_eq!(h.len(), d.len(), "residual stream {tag}: {what} length");
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("residual stream {tag}: {what}[{k}] bumped {} times by the host and {} by \
                   the device", h[k], d[k]);
        }
    };
    one("m_pow", delta(&after.pow, &before.pow), dev.m_pow.to_vec());
    one("m_norm", delta(&after.norm, &before.norm), dev.m_norm.to_vec());
    one("m_val", delta(&after.val, &before.val), dev.m_val.to_vec());
    one("m_frac", delta(&after.frac, &before.frac), dev.m_frac.to_vec());
    one("m_smul", delta(&after.smul, &before.smul), dev.m_smul.to_vec());
    for (u, d) in dev.m_un.iter().enumerate() {
        one(&format!("m_un[{u}]"), delta(&after.un[u], &before.un[u]), d.to_vec());
    }
    assert_eq!(off_host, dev.off_table, "residual stream {tag}: off-table bumps differ");
    assert_eq!(bad_host, dev.bad_exp, "residual stream {tag}: out-of-range exponents differ");
    for (g, (h, d)) in in_nat.iter().zip(&dev.in_nat).enumerate() {
        assert_eq!(h.len(), d.len(), "residual stream {tag}: in_nat[{g}] length");
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("residual stream {tag}: in_nat[{g}][{k}] host {:?} device {:?}", h[k], d[k]);
        }
    }
    for (o, (h, d)) in out_nat.iter().zip(&dev.out_nat).enumerate() {
        assert_eq!(h.len(), d.len(), "residual stream {tag}: out_nat[{o}] length");
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("residual stream {tag}: out_nat[{o}][{k}] host {:?} device {:?}", h[k], d[k]);
        }
    }
    for (o, (h, d)) in outs.iter().zip(&dev.outs).enumerate() {
        assert_eq!(h.len(), d.len(), "residual stream {tag}: outs[{o}] length");
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("residual stream {tag}: outs[{o}][{k}] host {:#06x} device {:#06x}", h[k], d[k]);
        }
    }
    eprintln!(
        "residual_stream_witness_gen-dev: {tag} -- {} base and {} pre columns, {} cells, the five \
         multiplicity tables, off_table, bad_exp and both natural arrays all match",
        base.len(), pre.len(), (base.len() + pre.len()) << dev.pre[0].size.trailing_zeros()
    );
}

/// The five counter families, snapshotted around the host witness generation.
pub struct ResidualStreamSnap {
    pub pow: Vec<u32>,
    pub norm: Vec<u32>,
    pub val: Vec<u32>,
    pub frac: Vec<u32>,
    pub smul: Vec<u32>,
    pub un: Vec<Vec<u32>>,
}

impl ResidualStreamSnap {
    pub fn of(t: &ResidualStreamTables) -> Self {
        ResidualStreamSnap {
            pow: t.m_pow.clone(),
            norm: t.m_norm.clone(),
            val: t.m_val.clone(),
            frac: t.m_frac.clone(),
            smul: t.m_smul.clone(),
            un: t.m_un.clone(),
        }
    }
}

/// Add the device's counters into the host's tables.
///
/// Counting is commutative, so the order the kernel's scatter-add left them in
/// cannot matter -- the same argument `mat_component_dev` makes.
pub fn merge(t: &mut ResidualStreamTables, dev: &DevResidualStream) {
    let add = |h: &mut Vec<u32>, d: Vec<u32>| {
        for (a, b) in h.iter_mut().zip(d) {
            *a += b;
        }
    };
    add(&mut t.m_pow, dev.m_pow.to_vec());
    add(&mut t.m_norm, dev.m_norm.to_vec());
    add(&mut t.m_val, dev.m_val.to_vec());
    add(&mut t.m_frac, dev.m_frac.to_vec());
    add(&mut t.m_smul, dev.m_smul.to_vec());
    for (h, d) in t.m_un.iter_mut().zip(&dev.m_un) {
        add(h, d.to_vec());
    }
    t.off_table += dev.off_table;
}

/// Restore the host's counters to where the host witness generation left them, so mode 2's
/// device bumps are the check's and not the proof's.
pub fn restore(t: &mut ResidualStreamTables, after: &ResidualStreamSnap, off: usize) {
    t.m_pow.copy_from_slice(&after.pow);
    t.m_norm.copy_from_slice(&after.norm);
    t.m_val.copy_from_slice(&after.val);
    t.m_frac.copy_from_slice(&after.frac);
    t.m_smul.copy_from_slice(&after.smul);
    for (h, a) in t.m_un.iter_mut().zip(&after.un) {
        h.copy_from_slice(a);
    }
    t.off_table = off;
}

