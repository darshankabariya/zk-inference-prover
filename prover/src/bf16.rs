//! bf16 bit patterns, and the preprocessed tables the dequant needs.
//!
//! B4 established that a *unary* bf16 op is exactly a 2^16-entry table.  The
//! dequant is not unary — `bf16(acc · s_x · s_w)` takes a ~30-bit integer and one
//! or two scales — so it needs a different table shape, and picking that shape is
//! most of the gadget's design.
//!
//! Three tables live here:
//!
//! * `norm_table` — the rounding.  Rows `(M2, ph, st) -> (m7, expfield, nz)`: the
//!   9-bit extended significand `M2` (8 significand bits plus the round bit), the
//!   weight `ph = 2^k` of that round bit, and the sticky flag `st`, mapped to the
//!   bf16 mantissa field and biased exponent of the correctly-rounded result.  One
//!   arity-6 lookup does the normalisation, the rounding *and* the bf16 encoding,
//!   and — because `M2 < 512` and `ph <= 2^kmax` bound `mag` below 2^30 — it
//!   subsumes B4's limb decomposition instead of adding to it.
//! * `mul_table` — the mantissa product, for scales that are not powers of two.
//!   Rows `(m7_a, m7_b) -> (m7_out, carry)` over all 2^14 mantissa pairs.
//! * the `(ph, r, st)` pair table lives in `dequant.rs`, because its rows are
//!   generated straight from `kmax` and it carries no bf16 semantics.
//!
//! The rounding mode is a knob, not a detail: B1's contract is `Tensor.to(bfloat16)`,
//! i.e. round-to-nearest-even, which needs the sticky bit and therefore one more
//! committed cell per accumulator (and twice the table rows) than
//! round-half-away-from-zero would.  Both are built here so the ladder can price
//! the difference.

use rayon::prelude::*;

/// bf16 bits -> f32 (exact: bf16 is the f32 prefix).
pub fn to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> bf16 bits, round-to-nearest-even, quiet NaN preserved.
pub fn from_f32(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    let bias = 0x7fff + ((u >> 16) & 1);
    ((u + bias) >> 16) as u16
}

/// How the discarded bits decide the last kept one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// round to nearest, ties to even — B1's `.to(bfloat16)`.  Needs the sticky
    /// flag, which the `(ph, r, st)` table supplies for free.
    Even,
    /// round to nearest, ties away from zero.  The round bit alone decides, so the
    /// sticky column disappears from the witness and both tables halve, at a
    /// numeric difference only exact bindings can see.
    Away,
    /// truncate.  The cheapest of the three and the furthest from B1.
    Zero,
}

impl Round {
    pub fn parse(s: &str) -> Self {
        match s {
            "even" | "rne" => Round::Even,
            "away" | "hafz" => Round::Away,
            "zero" | "trunc" => Round::Zero,
            other => panic!("unknown rounding mode {other:?}"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Round::Even => "even",
            Round::Away => "away",
            Round::Zero => "zero",
        }
    }
    /// Does the mode need to know whether the discarded tail is non-zero?
    pub fn needs_sticky(&self) -> bool {
        *self == Round::Even
    }
    /// Given the 8-bit significand, the round bit and the sticky flag, does the
    /// significand step up?
    pub fn carry(&self, m: u32, rb: u32, st: u32) -> u32 {
        match self {
            Round::Even => (rb & (st | (m & 1))) as u32,
            Round::Away => rb,
            Round::Zero => 0,
        }
    }
}

/// bf16's exponent bias, plus the 7 fraction bits: a significand `M` in
/// `[128, 256)` scaled by `2^a` has biased exponent `a + EXP_BIAS`.
pub const EXP_BIAS: u32 = 134; // 127 + 7

/// The largest `k` in `ph = 2^k` the normalisation table carries.
///
/// `mag = M2 * ph + r < 512 * 2^kmax`, so `kmax = 21` bounds every magnitude the
/// table can describe by `2^30 < p` — which is exactly what makes the integer
/// reading unique and lets this table replace B4's limb decomposition.  A larger
/// `kmax` would reintroduce the wrap-around hole B4 §1 found.
pub const KMAX_SOUND: u32 = 21;

/// One row of the normalisation table.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormRow {
    pub m2: u32,
    pub ph: u32,
    pub st: u32,
    pub m7: u32,
    pub expfield: u32,
    /// 0 for the single zero row, 1 otherwise: what stops a scale's exponent from
    /// being added to a zero result.
    pub nz: u32,
}

/// `(M2, ph, st) -> (m7, expfield, nz)` for every normalised significand, every
/// round-bit weight up to `2^kmax`, and both sticky values.
///
/// **Why the decomposition is unique, which is the whole soundness argument.**
/// `mag = M2 * ph + r` with `M2 in [256, 512)`, `0 <= r < ph = 2^k` forces
/// `mag in [2^(8+k), 2^(9+k))`.  Those windows are disjoint and tile `[256, oo)`,
/// so exactly one `k` admits a decomposition and then `M2 = mag >> k`,
/// `r = mag & (ph - 1)` are determined.  For `mag in [1, 256)` there is a second
/// family — `ph = 1` and `M2 = mag < 256` — and its rows are the *only* ones with
/// `M2 < 256`, so a large `mag` cannot borrow them: `ph = 1` and `r < 1` pin
/// `mag = M2 < 256`.  Row 0 is the zero case, reachable only from `mag = 0`.
///
/// `bf16(mag)` really is a function of `(M2, ph, st)`: `mag` has `9 + k` bits, so
/// bf16 keeps `mag >> (k+1) = M2 >> 1`, the round bit is `M2 & 1`, and everything
/// below it is `r`, whose only relevant property is whether it is zero.
pub fn norm_table(kmax: u32, round: Round) -> Vec<NormRow> {
    let mut rows = vec![NormRow { m2: 0, ph: 0, st: 0, m7: 0, expfield: 0, nz: 0 }];
    // the exactly-representable small magnitudes: mag = M2 < 256, ph = 1, r = 0
    for m2 in 1u32..256 {
        let (_, expfield, m7) = decode(from_f32(m2 as f32));
        rows.push(NormRow { m2, ph: 1, st: 0, m7, expfield, nz: 1 });
    }
    for k in 0..=kmax {
        for m2 in 256u32..512 {
            for st in 0..=(round.needs_sticky() as u32) {
                let (m, rb) = (m2 >> 1, m2 & 1);
                let up = round.carry(m, rb, st);
                // m in [128, 256); m + up reaches 256 only from m = 255, rb = 1,
                // which renormalises to significand 128 one binade up.
                let (msig, extra) = if m + up == 256 { (128, 1) } else { (m + up, 0) };
                rows.push(NormRow {
                    m2,
                    ph: 1 << k,
                    st,
                    m7: msig - 128,
                    // the result is (m + up) * 2^(k+1)
                    expfield: k + 1 + extra + EXP_BIAS,
                    nz: 1,
                });
            }
        }
    }
    rows
}

/// The bf16 pattern a `(m7, expfield, sign)` triple encodes.
pub fn pattern(sg: u32, expfield: u32, m7: u32) -> u16 {
    ((sg << 15) | (expfield << 7) | m7) as u16
}

/// f64 -> bf16 bits, round-to-nearest-even, in **one** rounding step.
///
/// `from_f32` is not a usable reference for a 30-bit integer: f32 keeps 24
/// mantissa bits and bf16 keeps 8, so going through f32 double-rounds and really
/// does differ from a direct rounding on some inputs.  f64 holds every integer
/// below 2^53 exactly, so rounding its 52-bit mantissa straight to 7 bits is an
/// independent single-rounding path -- IEEE bit surgery against the table's
/// integer normalisation.  Normal numbers only, which is all the dequant produces.
pub fn from_f64(x: f64) -> u16 {
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let u = x.to_bits();
    let sign = (u >> 63) as u32;
    let mut e = (((u >> 52) & 0x7ff) as i32) - 1023;
    let man = u & ((1u64 << 52) - 1);
    const SH: u32 = 52 - 7;
    let keep = man >> SH;
    let rb = (man >> (SH - 1)) & 1;
    let st = (man & ((1u64 << (SH - 1)) - 1)) != 0;
    let up = rb == 1 && (st || (keep & 1) == 1);
    let mut m7 = keep + u64::from(up);
    if m7 == 128 {
        m7 = 0;
        e += 1;
    }
    pattern(sign, (e + 127) as u32, m7 as u32)
}

/// `from_f64` in a chosen rounding mode -- the reference every residual-stream gadget is
/// checked against, since B10 step 2's adds and multiplies have to round the way
/// the mode the shared `NORM` table was built in rounds.
pub fn from_f64_round(x: f64, round: Round) -> u16 {
    if round == Round::Even {
        return from_f64(x);
    }
    if x == 0.0 {
        return if x.is_sign_negative() { 0x8000 } else { 0 };
    }
    let u = x.to_bits();
    let sign = (u >> 63) as u32;
    let mut e = (((u >> 52) & 0x7ff) as i32) - 1023;
    let man = u & ((1u64 << 52) - 1);
    const SH: u32 = 52 - 7;
    let keep = (man >> SH) as u32;
    let rb = ((man >> (SH - 1)) & 1) as u32;
    let st = u32::from(man & ((1u64 << (SH - 1)) - 1) != 0);
    let mut m7 = keep + round.carry(keep, rb, st);
    if m7 == 128 {
        m7 = 0;
        e += 1;
    }
    pattern(sign, (e + 127) as u32, m7)
}

/// Independent check of `norm_table`: what a single f64 rounding says.
pub fn bf16_of_u64(mag: u64) -> u16 {
    from_f64(mag as f64)
}

/// One row of the mantissa-product table.
#[derive(Clone, Copy, Debug)]
pub struct MulRow {
    pub a: u32,
    pub b: u32,
    pub m7: u32,
    /// 1 when the product's significand needed a binade shift
    pub carry: u32,
}

/// `(m7_a, m7_b) -> (m7_out, carry)`: the correctly-rounded product of two bf16
/// significands, over all 2^14 mantissa pairs.
///
/// `(128 + a)(128 + b)` is a 16-bit integer in `[2^14, 2^16)`, so the product's
/// significand is the top 8 bits and the binade shift is its 16th bit.
pub fn mul_table(round: Round) -> Vec<MulRow> {
    let mut rows = Vec::with_capacity(1 << 14);
    for a in 0u32..128 {
        for b in 0u32..128 {
            let p = (128 + a) * (128 + b); // [2^14, 2^16)
            let carry = (p >> 15) & 1; // 1 if p >= 2^15
            let shift = 7 + carry; // discard this many low bits
            let m = p >> shift; // in [128, 256)
            let rb = (p >> (shift - 1)) & 1;
            let st = (p & ((1 << (shift - 1)) - 1) != 0) as u32;
            let up = round.carry(m, rb, st);
            let (msig, extra) = if m + up == 256 { (128, 1) } else { (m + up, 0) };
            rows.push(MulRow { a, b, m7: msig - 128, carry: carry + extra });
        }
    }
    rows
}

/// Independent check of `mul_table`: what f32 says the bf16 product is.
pub fn bf16_mul_ref(a: u16, b: u16) -> u16 {
    from_f32(to_f32(a) * to_f32(b))
}

/// Decompose a bf16 pattern into `(sign, biased exponent, mantissa field)`.
pub fn decode(bits: u16) -> (u32, u32, u32) {
    let b = bits as u32;
    (b >> 15, (b >> 7) & 0xff, b & 0x7f)
}

/// The normalised decomposition `mag = M2 * ph + r`, `0 <= r < ph` — the witness
/// the normalisation table is indexed by.  Returns `(M2, ph, st, r)`.
///
/// `mag = 0` gives the zero row.  `mag` in `[1, 256)` gives `ph = 1`, `M2 = mag`,
/// `r = 0`: exactly representable, so the rounding is a no-op and small
/// accumulators cost the same as large ones.  `None` means `mag` needs a round-bit
/// weight the table does not carry, which is the incompleteness a `kmax` buys.
pub fn normalise(mag: u64, kmax: u32) -> Option<(u32, u32, u32, u64)> {
    if mag == 0 {
        return Some((0, 0, 0, 0));
    }
    let bits = 64 - mag.leading_zeros();
    // The extended significand keeps 9 bits, so the round bit sits at 2^(bits-9).
    let k = bits.saturating_sub(9);
    if k > kmax {
        return None;
    }
    let ph = 1u64 << k;
    let m2 = (mag >> k) as u32;
    let r = mag & (ph - 1);
    debug_assert_eq!(m2 as u64 * ph + r, mag);
    debug_assert!(m2 < 512 && (bits <= 8 || m2 >= 256));
    Some((m2, ph as u32, (r != 0) as u32, r))
}

/// `bf16(mag)` read straight off the table, for cross-checking against f64.
///
/// `round` decides whether the table carries an `st = 1` row at all, so the lookup
/// key has to be masked the same way the gadget's witness masks it.
pub fn norm_lookup(rows: &[NormRow], mag: u64, kmax: u32, sg: u32, round: Round) -> Option<u16> {
    let (m2, ph, st, _) = normalise(mag, kmax)?;
    let st = if round.needs_sticky() { st } else { 0 };
    let row = rows.iter().find(|r| r.m2 == m2 && r.ph == ph && r.st == st)?;
    Some(pattern(sg, row.expfield, row.m7))
}

// ---------------------------------------------------------------------------
// The integer softmax's exp tables, carried over from B5 unchanged.
//
// B4 established that a unary bf16 op is exactly a 2^16-entry table.  B5 uses that
// twice: `exp_bf16_table` is v1's, indexed by the bf16 pattern of the shifted score
// in real units; `exp_int_table` is the design point's, indexed by the *integer*
// shifted score, so the lookup range-checks the shift at the same time.
// ---------------------------------------------------------------------------

/// The bf16 exp table's row count: a unary bf16 op is a total function on 16-bit
/// patterns (B4 section 1).
pub const LOG_EXP_ROWS: u32 = 16;
pub const TABLE_ROWS: usize = 1 << LOG_EXP_ROWS;

fn round_hafz(v: f64) -> f64 {
    v.abs().floor() * v.signum() + if v.abs().fract() >= 0.5 { v.signum() } else { 0.0 }
}

/// `n = round(nmax * exp(x))`, clamped, for every bf16 pattern `x`.
///
/// B1 computes the numerator exactly this way, so the table *is* the op.
pub fn exp_bf16_table(nmax: u32) -> Vec<u32> {
    (0..TABLE_ROWS)
        .map(|i| {
            let v = round_hafz(nmax as f64 * (to_f32(i as u16) as f64).exp());
            if v.is_nan() { 0 } else { v.clamp(0.0, nmax as f64) as u32 }
        })
        .collect()
}

/// `n = round(nmax * exp(bf16(-d * 2^-frac)))` for the integer shifted score `d`.
///
/// `frac` is the fixed-point exponent: with power-of-two Q/K scales it is a
/// public per-(layer, head) constant, so one table serves every head.
///
/// The argument is rounded to bf16 before `exp` on purpose.  That is B1's
/// numeric contract -- every value flowing into a nonlinearity is a bf16 value --
/// and it makes this table encode *the same function* as the bf16-indexed one:
/// the two variants differ in how the argument is addressed, not in what is
/// computed, which is what lets the torch witness check both.
///
/// Stream P: built in parallel.  Every row is a pure function of its own index,
/// so the rows are the same values in the same order however many threads
/// build them -- and at the sequence's table sizes (2^23..2^27 rows, stream J's
/// tight sizing) this was two thirds of the cost of loading a layer.
pub fn exp_int_table(rows: usize, nmax: u32, frac: f64) -> Vec<u32> {
    (0..rows)
        .into_par_iter()
        .map(|d| {
            let x = to_f32(from_f32((-(d as f64) / frac.exp2()) as f32)) as f64;
            let v = round_hafz(nmax as f64 * x.exp());
            v.clamp(0.0, nmax as f64) as u32
        })
        .collect()
}
