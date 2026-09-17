//! How a value becomes the field elements a column carries.
//!
//! Three encodings, and no gadget gets by without at least one of them:
//!
//! * a signed integer as an M31 and back -- `m31_of` and `sgn_of`, which read
//!   the top half of the field as the negatives;
//! * a bf16 pattern as the `(sm, ef)` pair every arithmetic component speaks --
//!   `split`, `join` and `canon`, with `is_odd_pattern` and `to_f64` beside
//!   them;
//! * a QM31 accumulator as its four M31 columns -- `secure_cols`.
//!
//! They were spread over `gadgets/attention/trace.rs` and
//! `gadgets/residual_stream/mod.rs`, with three more gadgets and `layer.rs`
//! holding an `m31_of` that only forwarded to attention's.  None of them is
//! attention's or the residual stream's: they are the language the gadgets state
//! their columns in, which is why they are here and the arithmetic that uses them
//! is not.
//!
//! The pattern language itself -- what a bf16 is, and the tables over it -- is
//! `crate::bf16`, which was already a module of the crate and stays one.

use rayon::prelude::*;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::SecureField;

use crate::bf16::{self, EXP_BIAS};
use crate::common::{Eval, eval_of};

pub fn m31_of(v: i64) -> M31 {
    let p = P as i64;
    M31::from_u32_unchecked((((v % p) + p) % p) as u32)
}

/// A signed significand held as a field element, back to `i64`.
pub fn sgn_of(v: M31) -> i64 {
    if v.0 > P / 2 { v.0 as i64 - P as i64 } else { v.0 as i64 }
}

/// A bf16 pattern as `(sm, ef)`.  Subnormals (`ef = 0, m7 != 0`) and
/// infinities/NaN collapse to zero, which is a *different value*; the witness
/// builders count them and the run reports the count, so one would show up as a
/// reference mismatch rather than pass silently.
pub fn split(pat: u16) -> (i64, u32) {
    let (sg, ef, m7) = bf16::decode(pat);
    if ef == 0 || ef == 0xff {
        (0, 0)
    } else {
        let m = (128 + m7) as i64;
        (if sg == 1 { -m } else { m }, ef)
    }
}

pub fn join(sm: i64, ef: u32) -> u16 {
    if sm == 0 {
        0
    } else {
        bf16::pattern(u32::from(sm < 0), ef, sm.unsigned_abs() as u32 - 128)
    }
}

/// bf16 has two zeros and the `(sm, ef)` representation has one, deliberately:
/// `sm = 0` *is* zero, with no side condition, which is what keeps the add's
/// constraints degree 2.  A negative zero -- which B1 commits whenever a product
/// has one negative operand and one zero one -- is therefore canonicalised to
/// `+0`.  Nothing downstream can see the difference: the requantize sends both to
/// `q = 0`, and every add and multiply here treats them alike.  The runs count
/// how many the real columns actually carry.
pub fn canon(pat: u16) -> u16 {
    if pat == 0x8000 { 0 } else { pat }
}

pub fn is_odd_pattern(pat: u16) -> bool {
    let (_, ef, m7) = bf16::decode(pat);
    (ef == 0 && m7 != 0) || ef == 0xff
}

pub fn to_f64(sm: i64, ef: u32) -> f64 {
    if sm == 0 { 0.0 } else { sm as f64 * (ef as f64 - EXP_BIAS as f64).exp2() }
}

/// Split a QM31 accumulator into its four M31 columns.
///
/// Each of the four is an independent pass with its own bit-reversal, so they go in
/// parallel: 13 accumulators x 4 columns is the second-largest block of our own
/// interaction work after the inner sums.
pub fn secure_cols(log_size: u32, vals: &[SecureField]) -> Vec<Eval> {
    (0..4)
        .into_par_iter()
        .map(|i| eval_of(log_size, vals.iter().map(|v| v.to_m31_array()[i]).collect()))
        .collect()
}

/// The name the component builders outside the attention trace call
/// `secure_cols` by.  One function under two names, as it was.
pub fn secure_cols_pub(log_size: u32, vals: &[SecureField]) -> Vec<Eval> {
    secure_cols(log_size, vals)
}
