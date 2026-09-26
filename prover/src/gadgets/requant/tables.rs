//! The requantize's tables: what each one holds, the order its rows are in, and
//! the preprocessed ids it is committed under.
//!
//! They are split out because they are not only this gadget's.  `gadgets/rope`'s
//! quantize component is the power-of-two requantize with the scale moved from a
//! committed per-token column to a preprocessed per-slot one, and in a composed
//! proof the two components share the sign table (`RQV_COLS`) and the 15-bit
//! range: `compose.rs` draws one `rqv` relation and adds rope's multiplicities
//! into this gadget's, so `rqv_at`'s row order has to be one definition and not
//! two.  What rope does *not* share is the division table and the `(q, nz)` pair
//! table, which it sizes by its own side's width (stream NN-3).

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::encoding::m31_of;
use crate::relations::id;

pub const RQ3_COLS: [&str; 3] = ["zk_rq_t", "zk_rq_m", "zk_rq_q"];
pub const RQ4_COLS: [&str; 4] = ["zk_rq_t", "zk_rq_s", "zk_rq_m", "zk_rq_q"];
pub const SCL2_COLS: [&str; 2] = ["zk_scl_pm", "zk_scl_j"];
pub const SCL3_COLS: [&str; 3] = ["zk_scl_pm", "zk_scl_ms", "zk_scl_es"];
pub const QNZ_COLS: [&str; 2] = ["zk_rqnz_q", "zk_rqnz_nz"];
pub const RANGE15: &str = "zk_range_15";

/// Stream II: `(sm, m7, sg)` over every legal signed significand -- 257 rows.
/// The requantize keys its own lookups on the magnitude pair `(ef, m7)`, but a
/// binding to a producing component has to speak the wiring family's `(sm, ef)`
/// language and the Freivalds check's `qX` needs a sign.  One lookup per element pins both
/// derived fields from one committed `sm`, which is cheaper than a 2^15 pair
/// table and keeps the binding's folded value degree 1.
pub const RQV_COLS: [&str; 3] = ["zk_rqv_sm", "zk_rqv_m7", "zk_rqv_sg"];

/// Stream II: the division and scale tables' preprocessed ids, tagged by
/// variant.  `RQ3_COLS` and `RQ4_COLS` share three names, which is fine for a
/// standalone rung that carries one variant and a collision in a composed proof
/// that carries both -- the `qkv` seam's scale is a power of two and the
/// `gateup` and `down` seams' are bf16, on the very first shape this runs on.
pub fn rq_tab_col(v: SVariant, k: usize) -> PreProcessedColumnId {
    id(format!("zk_rqt_{}_{k}", v.name()))
}
pub fn scl_tab_col(v: SVariant, k: usize) -> PreProcessedColumnId {
    id(format!("zk_rqs_{}_{k}", v.name()))
}
/// How many columns each carries, by variant.
pub fn rq_tab_width(v: SVariant) -> usize {
    if v == SVariant::Pow2 { 3 } else { 4 }
}
pub fn scl_tab_width(v: SVariant) -> usize {
    if v == SVariant::Pow2 { 2 } else { 3 }
}
/// The division table's row count as a log, from its closed form.
pub fn rq_log_rows(v: SVariant) -> u32 {
    crate::common::log_rows_for(rq_table(v).len(), 1)
}
/// The two tables' contents, padded by repeating the last real row -- stream
/// II's `residual_stream_table_cols` rule, for the same reason.
pub fn rq_tab_cols(v: SVariant) -> Vec<Eval> {
    let rows = rq_table(v);
    let lr = crate::common::log_rows_for(rows.len(), 1);
    (0..rq_tab_width(v))
        .map(|k| {
            eval_of(lr, (0..1usize << lr)
                .map(|i| {
                    let r = rows[i.min(rows.len() - 1)];
                    let v4 = [r.0, r.1, r.2, r.3];
                    let v3 = [r.0, r.2, r.3];
                    m31_u32(if v == SVariant::Pow2 { v3[k] } else { v4[k] })
                })
                .collect())
        })
        .collect()
}
pub fn scl_tab_cols(v: SVariant) -> Vec<Eval> {
    let rows = scl_table(v);
    (0..scl_tab_width(v))
        .map(|k| {
            eval_of(15, (0..1usize << 15)
                .map(|i| {
                    let r = rows[i.min(rows.len() - 1)];
                    match k {
                        0 => m31_u32(r.0),
                        1 => r.1,
                        _ => m31_u32(r.2),
                    }
                })
                .collect())
        })
        .collect()
}
pub fn rqv_tab_cols() -> Vec<Eval> {
    let rows = rqv_table();
    (0..3)
        .map(|k| {
            eval_of(9, (0..512usize).map(|i| rows[i.min(rows.len() - 1)][k]).collect())
        })
        .collect()
}

pub fn rqv_table() -> Vec<[M31; 3]> {
    let mut rows = vec![[M31::zero(), M31::zero(), M31::zero()]];
    for m in 128i64..256 {
        rows.push([m31_of(m), m31_u32((m - 128) as u32), M31::zero()]);
        rows.push([m31_of(-m), m31_u32((m - 128) as u32), M31::one()]);
    }
    rows
}

/// `rqv_table`'s row order, as an index.
pub fn rqv_at(sm: i64) -> Option<usize> {
    if sm == 0 {
        return Some(0);
    }
    let mag = sm.unsigned_abs();
    if !(128..256).contains(&mag) {
        return None;
    }
    Some(1 + (mag as usize - 128) * 2 + usize::from(sm < 0))
}

/// `tt` offsets: quotients in [1/2, 127.5) need only these exponent windows.
pub const TT_POW2: i64 = 9; // Delta' = ef - 134 - jx in [-9, -1]  (|x| = (128+m7) 2^(ef-134))
pub const TT_BF16: i64 = 1; // Delta  = ef - es in [-1, 7]
pub const TT_RANGE: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SVariant {
    Pow2,
    Bf16,
}

impl SVariant {
    pub fn parse(s: &str) -> Self {
        match s {
            "pow2" => SVariant::Pow2,
            "bf16" => SVariant::Bf16,
            other => panic!("unknown requant variant {other:?}"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            SVariant::Pow2 => "pow2",
            SVariant::Bf16 => "bf16",
        }
    }
}

/// The division table's rows: `(tt, [ms,] m7) -> q`, only where `q` lands in
/// `[1, 127]` — everything else is the `nz = 0` branch or B1's clamp.
pub fn rq_table(v: SVariant) -> Vec<(u32, u32, u32, u32)> {
    let mut rows = vec![];
    match v {
        SVariant::Pow2 => {
            for tt in 0..TT_RANGE as u32 {
                for m7 in 0..128u32 {
                    let m = 128 + m7 as u64;
                    // q = round_half_up(M * 2^(tt-9)) = (M 2^(tt+1) + 512) / 1024
                    let q = ((m << (tt + 1)) + 512) / 1024;
                    if (1..=127).contains(&q) {
                        rows.push((tt, 0, m7, q as u32));
                    }
                }
            }
        }
        SVariant::Bf16 => {
            for tt in 0..TT_RANGE as u32 {
                for ms in 0..128u32 {
                    for m7 in 0..128u32 {
                        let m = 128 + m7 as u64;
                        let sden = 128 + ms as u64;
                        // q = round_half_up(M 2^(tt-1) / S) = (M 2^tt + S) / (2 S)
                        let q = ((m << tt) + sden) / (2 * sden);
                        if (1..=127).contains(&q) {
                            rows.push((tt, ms, m7, q as u32));
                        }
                    }
                }
            }
        }
    }
    rows
}

/// The scale table: `pm -> ceil-divide-by-127`, one row per nonzero magnitude
/// pattern.  For a power-of-two scale the payload is the exponent `j` (as a
/// field element, since real `j` are negative); for bf16 it is `(ms, es)`.
pub fn scl_table(v: SVariant) -> Vec<(u32, M31, u32)> {
    (1u32..1 << 15)
        .map(|pm| {
            let val = bf16::to_f32(pm as u16) as f64 / 127.0;
            match v {
                SVariant::Pow2 => {
                    let j = val.log2().ceil() as i64;
                    (pm, m31_of(j), 0)
                }
                SVariant::Bf16 => {
                    let (_, es, ms) = bf16::decode(bf16_ceil_f64(val));
                    (pm, m31_u32(ms), es)
                }
            }
        })
        .collect()
}

/// Smallest bf16 >= v, v > 0 — `quant.py`'s `bf16_ceil`, in Rust.
pub fn bf16_ceil_f64(v: f64) -> u16 {
    let b = bf16::from_f64(v);
    if (bf16::to_f32(b) as f64) < v { b + 1 } else { b }
}
