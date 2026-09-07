//! B7a milestone 4 (section 3 row 1): the shared tables' own columns.
//!
//! 22.5 left 4a at 0.166 GB in 354 uploads and said where every byte of it was:
//! **0.101 GB is three columns**, `x`, `y` and `m_exp` at 2^23 rows, and none of
//! them is a witness. This file is their fill, in 20.1's shape -- one thread
//! owns the *committed* row, so `eval_of` never runs on the column.
//!
//! | column | what it is | where it comes from here |
//! |---|---|---|
//! | `x` | the row index | the closed form, on the device |
//! | `y` | `Tables::exp` | a 2^16 bf16 map, gathered at `bf16(-d 2^-frac)` |
//! | `m_exp` | the exp table's multiplicities | the counters every key tile's fill already bumped (21.1), summed on the device |
//!
//! **Why `y` is a gather and not an `exp()`.** `exp_int_table` rounds its
//! argument to bf16 *before* the exponential -- B1's own arithmetic, and the
//! reason the table "is the op" (20.1's phrase for the dequant's). So the table
//! has at most 2^16 distinct outputs whatever its row count, and
//! `exp_int_table(rows, nmax, f)[d] == exp_bf16_table(nmax)[from_f32(-d 2^-f)]`
//! exactly. The host builds that 262 kB map -- one 2^16 pass -- and `exp()`
//! never runs on the device, so there is no libm agreement to assume.
//!
//! **Section 5 rule 8, and the one host pass that stays.** Rung 3's sixth probe
//! flips one row of the exp table and rebuilds the witness to match, so that
//! *only* the pinned preprocessed root rejects it. A fill that computed the
//! closed form and stopped would repair it silently. So the host diffs its
//! actual table against the closed form -- 8.4 M rows, one parallel pass, ~4 ms
//! -- and uploads the rows that differ. That pass is in the **measured** path,
//! not a check mode: it is what makes the device column equal to the host's
//! table by construction rather than by assumption, on any witness.
//!
//! `B7_M4_TABDEV` says which path runs, on the same three-value pattern the
//! other fills use: 0 the host builds all 34 shared-table columns, 1 these
//! three are the device's and never the host's, 2 both with every cell
//! asserted equal.

use rayon::prelude::*;
use stwo::prover::backend::Column as _;
use stwo::stwo_cuda::base_field_vec::{BaseFieldVec, Uint32Vec};

use crate::attn_witness::{Attn, Variant};
use crate::common::Eval;
use crate::devfrac::Cols;

// ---------------------------------------------------------------- the FFI shape --

/// Mirrors `B65TabCfg` in `b65_table_fill.cuh`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FTabCfg {
    pub log_rows: u32,
    pub rows: u32,
    pub variant: u32,
    pub n_counters: u32,
    pub two_pow: f64,
    pub mutate: u32,
    pub mut_col: u32,
}

unsafe extern "C" {
    fn b65_fill_tables(
        expmap: *const u32,
        counters: *const *const u32,
        patch_idx: *const u32,
        patch_val: *const u32,
        n_patch: u32,
        x_col: *mut u32,
        y_col: *mut u32,
        m_col: *mut u32,
        cfg: FTabCfg,
    );
}

// ------------------------------------------------------------------ the switch --

/// `B7_M4_TABDEV`: 0 the host, 1 the device, 2 both with every cell asserted.
pub fn mode() -> u32 {
    crate::util::m4b_mode("B7_M4_TABDEV")
}

/// `B7_M4_TABMUT`, `name` or `name:column`. Unaimed, every defect shows at `x`,
/// which is compared first, and the run says nothing about `y` or `m_exp`
/// (section 5 rule 5) -- so the suffix names the column the defect is aimed at.
pub fn tabmut() -> (u32, u32) {
    let raw = std::env::var("B7_M4_TABMUT").unwrap_or_default();
    let (name, col) = match raw.split_once(':') {
        Some((n, c)) => (n.to_string(), c.to_string()),
        None => (raw, String::new()),
    };
    let m = match name.as_str() {
        "" | "none" => 0,
        "noperm" => 1,
        "noscale" => 2,
        "bf16trunc" => 3,
        "nopatch" => 4,
        "onecounter" => 5,
        other => panic!("unknown B7_M4_TABMUT {other:?}"),
    };
    let c = match col.as_str() {
        "" | "all" => 3,
        "x" => 0,
        "y" => 1,
        "m" | "m_exp" => 2,
        other => panic!("unknown B7_M4_TABMUT column {other:?}"),
    };
    (m, c)
}

// ---------------------------------------------------- the flat lists, with holes --

/// A shared-table column the host built. In device mode the three closed-form
/// columns have no host copy at all, and asking for one is a bug that says so
/// rather than a wrong proof.
pub fn htab(v: &[Option<Eval>], i: usize) -> &Eval {
    v[i].as_ref().unwrap_or_else(|| {
        panic!("shared-table column {i} is the device's; run with B7_M4_TABDEV=0 or 2")
    })
}

/// The same, for a slot already in hand.
pub fn hone(e: &Option<Eval>) -> &Eval {
    e.as_ref().expect("this shared-table column is the device's, not the host's")
}

/// A shared table's slice of the two flat lists: `devfrac::tab_cols`, for lists
/// whose first entries may have no host column.
pub fn tab_cols<'a>(
    host: &'a [Option<Eval>],
    dev: &'a [BaseFieldVec],
    lo: usize,
    n: usize,
) -> Cols<'a> {
    if dev.is_empty() {
        Cols::Host((lo..lo + n).map(|i| htab(host, i)).collect())
    } else {
        Cols::Dev(dev[lo..lo + n].iter().collect())
    }
}

/// Assert one device column against the host's, cell for cell.
pub fn check_col(what: &str, host: &Eval, got: &BaseFieldVec) {
    let (a, b) = (host.values.to_cpu(), got.to_vec());
    assert_eq!(a.len(), b.len(), "{what}: column length");
    if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
        panic!("{what}: differs at cell {k}: host {:?}, device {:?}", a[k], b[k]);
    }
    eprintln!("tab-dev: {what} -- {} cells match", a.len());
}

// ------------------------------------------------------------------ the closed form --

/// `2^frac` for the variant's table, as `Attn::build` passes it to
/// `exp_int_table`. The host computes the power and the kernel divides by it,
/// so the division is the same IEEE division on both sides.
fn two_pow(a: &Attn) -> f64 {
    match a.variant {
        Variant::Bf16x => 1.0,
        Variant::IntExp => a.frac.exp2(),
        Variant::IntSplit(k) => (a.frac - k as f64).exp2(),
    }
}

/// The bf16 pattern row `d` of the table looks the map up at.
fn pattern(a: &Attn, tp: f64, d: usize) -> u16 {
    match a.variant {
        Variant::Bf16x => d as u16,
        _ => crate::bf16::from_f32((-(d as f64) / tp) as f32),
    }
}

/// The rows where this witness's table is not the closed form.
///
/// Empty on an honest witness; one entry for `Layer::corrupt_exp_table`. A long
/// list means the closed form is wrong for this design point, and that is a
/// panic and not a silent 67 MB upload.
fn exp_patch(a: &Attn, exp: &[u32], map: &[u32], tp: f64) -> (Vec<u32>, Vec<u32>) {
    const MAX: usize = 4096;
    let (idx, val): (Vec<u32>, Vec<u32>) = exp
        .par_iter()
        .enumerate()
        .filter(|(d, v)| map[pattern(a, tp, *d) as usize] != **v)
        .map(|(d, v)| (d as u32, *v))
        .unzip();
    assert!(
        idx.len() <= MAX,
        "the exp table differs from exp_int_table in {} rows: b65_table_fill expects \
         the closed form plus at most a probe's tamper",
        idx.len()
    );
    (idx, val)
}

// ------------------------------------------------------------------- the fill --

/// The three columns, on the device.
pub struct DevTabCols {
    pub x: BaseFieldVec,
    pub y: BaseFieldVec,
    /// `None` when no key tile fill ran on the device, so there is no counter to sum
    /// and `m_exp` stays a host column and a 4a upload.
    pub m: Option<BaseFieldVec>,
    pub patch: usize,
}

fn up_u32(v: Vec<u32>) -> Uint32Vec {
    let bytes = (v.len() * 4) as u64;
    let t = std::time::Instant::now();
    let d = Uint32Vec::from_vec(v);
    crate::device::note_witness(bytes, t.elapsed().as_secs_f64());
    d
}

/// Build `x`, `y` and (when the key tiles filled on the device) `m_exp`.
///
/// `counters` are the per-key-tile exp counters `commit_base_dev` left on the
/// device. They are summed here, which is what deletes 21.4's "counters: D2H
/// and merge" line for the exp table -- three 33.5 MB downloads and three
/// parallel merges for a column that is then uploaded again.
pub fn table_cols_dev(a: &Attn, log_rows: u32, exp: &[u32],
                      counters: &[&Uint32Vec]) -> DevTabCols {
    let rows = 1usize << log_rows;
    assert_eq!(exp.len(), rows, "the exp table is not 1 << log_table rows");
    let (mutate, mut_col) = tabmut();

    // the map, and the rows this witness's table puts outside it
    let t_m = std::time::Instant::now();
    let tp = two_pow(a);
    let map = crate::bf16::exp_bf16_table(a.nmax);
    let (pidx, pval) = exp_patch(a, exp, &map, tp);
    crate::util::m4_add("gen/tab_map", t_m.elapsed().as_secs_f64());

    let t_u = std::time::Instant::now();
    let d_map = up_u32(map);
    let n_patch = pidx.len();
    let d_pidx = up_u32(if pidx.is_empty() { vec![0u32] } else { pidx });
    let d_pval = up_u32(if pval.is_empty() { vec![0u32] } else { pval });
    crate::util::m4_add("gen/tab_up", t_u.elapsed().as_secs_f64());

    let t_a = std::time::Instant::now();
    let x = BaseFieldVec::new_uninitialized(rows);
    let y = BaseFieldVec::new_uninitialized(rows);
    let m = (!counters.is_empty()).then(|| BaseFieldVec::new_uninitialized(rows));
    crate::util::m4_add("gen/tab_alloc", t_a.elapsed().as_secs_f64());

    let cptr: Vec<*const u32> = counters
        .iter()
        .map(|c| {
            assert_eq!(c.size, rows, "an exp counter is not 1 << log_table wide");
            c.device_ptr
        })
        .collect();
    let cfg = FTabCfg {
        log_rows,
        rows: rows as u32,
        variant: match a.variant {
            Variant::Bf16x => 0,
            Variant::IntExp => 1,
            Variant::IntSplit(_) => 2,
        },
        n_counters: cptr.len() as u32,
        two_pow: tp,
        mutate,
        mut_col,
    };

    let t_k = std::time::Instant::now();
    unsafe {
        b65_fill_tables(
            d_map.device_ptr,
            cptr.as_ptr(),
            d_pidx.device_ptr,
            d_pval.device_ptr,
            n_patch as u32,
            x.device_ptr as *mut u32,
            y.device_ptr as *mut u32,
            m.as_ref().map(|c| c.device_ptr as *mut u32).unwrap_or(std::ptr::null_mut()),
            cfg,
        );
    }
    crate::util::m4_add("gen/tab_kernel", t_k.elapsed().as_secs_f64());
    eprintln!(
        "tab-dev: 2^{log_rows} -- x, y{} from {} exp counters, patch {} rows, kernel {:.4} s",
        if m.is_some() { ", m_exp" } else { "" },
        cptr.len(),
        n_patch,
        t_k.elapsed().as_secs_f64()
    );
    DevTabCols { x, y, m, patch: n_patch }
}
