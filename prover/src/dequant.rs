//! The dequant gadget: `bf16(acc · s_x · s_w)`, and the ladder around it.
//!
//! This is the gadget B4 §8 scoped and deliberately refused to improvise, and that
//! B5 handed two more customers.  B3 proves `qY == W qX` *modulo p*; B4 turns a
//! committed accumulator into an integer (5 cells, `acc = lo + 2^16 hi − 2^29`) and
//! stops.  What is left is the multiply-and-round — the only gadget class in the
//! pipeline with no measured template, and the largest unmeasured cost left.
//!
//! # The construction
//!
//! One arity-6 lookup does the normalisation, the rounding *and* the bf16 encoding:
//!
//! ```text
//!   |acc| = M2 · ph + r      M2 ∈ [256,512), 0 ≤ r < ph = 2^k   (or ph = 1, M2 < 256)
//!   NORM(M2, ph, st) → (m7, ef, nz)          the correctly-rounded bf16 of |acc|
//!   FRAC(ph, r, st)                          r < ph, and st = [r ≠ 0], in one lookup
//! ```
//!
//! Three things fall out of that shape, and they are why it is worth writing down
//! rather than assembling the obvious pieces:
//!
//! * **It subsumes B4's limb decomposition instead of adding to it.**  The table's
//!   domain bounds `M2 < 512` and `ph ≤ 2^kmax`, so the reconstruction constraint
//!   *is* an injective integer reading below `2^(kmax+9) ≤ 2^30 < p`.  That is the
//!   statement B4 pays 5 cells for, and `Tamper::Wrap` is the probe: a prover who
//!   decomposes `acc + p` has no valid `(M2, ph, r)` at all.
//! * **`r < ph` is one lookup, not two.**  Range-checking `r` and `ph − 1 − r`
//!   separately costs two lookups; a preprocessed table of every valid `(ph, r)`
//!   pair costs one, and carrying `st` as its third column deletes the sticky
//!   bit's inverse witness as well.  The price is `2^(kmax+1)` preprocessed rows —
//!   B5's exp-table trade, on a second gadget.
//! * **`kmax` is a completeness knob, not a cost knob.**  Below the accumulator
//!   width the gadget cannot decompose at all: the third appearance of B4 §1's
//!   "sound and incomplete" lever.
//!
//! # The ladder
//!
//! `Variant` is the axis the multiplicity ladder cares about — what the *scales*
//! are allowed to be:
//!
//! | variant | statement | why it is on the ladder |
//! |---|---|---|
//! | `limbs` | B4's baseline: the integer, not the rounding | the v1 gap, for reference |
//! | `round` | `y = bf16(acc)` | the normalisation core alone |
//! | `pow2`  | `y = bf16(acc · 2^j)` | power-of-two scales: the multiply is an exponent add |
//! | `chain` | `y = bf16(bf16(bf16(acc)·s_x)·s_w)` | general bf16 scales: two mantissa products |
//! | `int8`  | `q = round(acc · num / den)` | the consumer requantises, so skip bf16 |
//! | `fold`  | `q = round_hafz(acc · s_x s_w / s_v)` | stream Z: the delayed requantize, general bf16 `s_x s_w` over a power-of-two `s_v`, split into limbs because neither the product nor the remainder fits one check |
//!
//! `int8` is B5's review's suggestion made concrete: where a dequant's consumer is
//! a matmul, folding the scale, the softmax division and the requantisation into a
//! *single* rounding straight to int8 deletes the bf16 waypoint, the reciprocal
//! lookup and the requantize at 254 k sites per token.
//!
//! The rounding mode is the other axis.  B1's contract is `.to(bfloat16)`, i.e.
//! round-to-nearest-even, which needs the sticky flag; ties-away needs only the
//! round bit and drops a column from the witness and from both tables.  Both are
//! measured, because the difference lands on the model's second-largest rung.

use num_traits::{One, Zero};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, INTERACTION_TRACE_IDX,
    Relation, RelationEntry, TraceLocationAllocator, relation,
};

use crate::bf16::{self, EXP_BIAS, KMAX_SOUND, MulRow, NormRow, Round};
use crate::common::{Eval, Proven, Session, Sums, eval_of, log_rows_for, m31_u32, twiddles};

relation!(NormRel, 6);
relation!(FracRel, 3);
relation!(MulRel, 4);
relation!(PairRel, 2);

// B11 stream U, lever 3: the drawn elements, for the GKR link's affine weights.
// `combine(v) = sum alpha_i v_i - z`, so these two are everything the link needs
// to write a lookup's denominator as an affine form over the columns it reads --
// the same accessors stream S added to `ExpRel` and `RangeRel`.
impl NormRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl FracRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl MulRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
// B11 stream V, lever 4: the attention-output component's `(qm, nz2)` pair check
// is the first lookup any family's link has needed from this relation.
impl PairRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
// one range-relation type across the whole crate, so a composed proof's range
// tables serve the attention key tiles and the dequant components alike
pub use crate::attn_air::RangeRel;

pub fn id(s: impl Into<String>) -> PreProcessedColumnId {
    PreProcessedColumnId { id: s.into() }
}
pub const NORM_COLS: [&str; 6] =
    ["b6_norm_m2", "b6_norm_ph", "b6_norm_st", "b6_norm_m7", "b6_norm_ef", "b6_norm_nz"];
pub const FRAC_COLS: [&str; 3] = ["b6_frac_ph", "b6_frac_r", "b6_frac_st"];
pub const MUL_COLS: [&str; 4] = ["b6_mul_a", "b6_mul_b", "b6_mul_m7", "b6_mul_c"];
pub fn range_col(bits: u32) -> PreProcessedColumnId {
    id(format!("b6_range_{bits}"))
}
/// Per-channel scale columns.  A weight scale is offline, so it is preprocessed —
/// which is also what keeps it off the accumulator's multiplicity rung.
pub fn chan_col(tag: &str, kind: &str, slot: usize) -> PreProcessedColumnId {
    id(format!("b6_chan{tag}_{kind}{slot:04}"))
}
pub const QNZ_COLS: [&str; 2] = ["b6_qnz_q", "b6_qnz_nz"];
/// Selectors of the composed dequant component's Freivalds claimed side.
pub const DEQ_SEL: [&str; 5] = ["nofirst", "last", "m1", "m2", "end"];
pub fn deq_sel(tag: &str, name: &str) -> PreProcessedColumnId {
    id(format!("zk_deq{tag}_{name}"))
}

// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Limbs,
    Round,
    Pow2,
    Chain,
    Int8,
    /// Stream Z: the delayed requantize, `q = round_hafz(acc s_x s_w / s_v)` in
    /// one rounding, with a general bf16 `s_x s_w` and a power-of-two `s_v`.
    /// The `int8` form above cannot state it: the mantissa product widens the
    /// honest `mag num` to ~2^42 and the divisor to ~2^25, so a single error
    /// check leaves the field.  The statement here splits both -- the magnitude
    /// into two 13-bit limbs and the remainder into a low 13-bit half and a
    /// high half checked against the per-slot divisor -- and every term stays
    /// below 2^30.  See `evaluate`'s `Fold` arm for the identity.
    Fold,
}

/// The fold's limb width: the magnitude splits as `mag = m_lo + 2^13 m_hi` and
/// the remainder's low half is the low 13 bits of `m_lo (128+ms)(128+mw)`.
pub const FOLD_LIMB: u32 = 13;

impl Variant {
    pub fn parse(s: &str) -> Self {
        match s {
            "limbs" => Variant::Limbs,
            "round" => Variant::Round,
            "pow2" => Variant::Pow2,
            "chain" => Variant::Chain,
            "int8" => Variant::Int8,
            "fold" => Variant::Fold,
            other => panic!("unknown dequant variant {other:?}"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Variant::Limbs => "limbs",
            Variant::Round => "round",
            Variant::Pow2 => "pow2",
            Variant::Chain => "chain",
            Variant::Int8 => "int8",
            Variant::Fold => "fold",
        }
    }
    pub fn normalises(&self) -> bool {
        !matches!(self, Variant::Limbs | Variant::Int8 | Variant::Fold)
    }
}

/// Everything about a run that is not the witness.
#[derive(Clone, Debug)]
pub struct Spec {
    pub variant: Variant,
    pub round: Round,
    pub kmax: u32,
    /// accumulators per trace row
    pub per_row: usize,
    /// B4's baseline limb widths, for `Variant::Limbs`
    pub limb_bits: (u32, u32),
    /// `int8`: width of the quotient range check (7 = B1's clamp at 127)
    pub q_bits: u32,
    /// `int8`: width of the rounding-error range check, i.e. of `2 den`.
    /// `fold`: width of the high remainder's check, i.e. of `2 denp14` -- the
    /// per-slot divisor after the 14 low bits are split off.
    pub rem_bits: u32,
    /// `fold`: how the divisor `2^G` splits between the committed per-token
    /// `dent = 2^(G_t - alpha)` and the preprocessed per-channel
    /// `pj = 2^(G_c - (14 - alpha))`.  Data-derived: `alpha <= min_t G_t` and
    /// `14 - alpha <= min_c G_c` must both hold or the witness generation cannot build an
    /// integer column, and together they are what guarantees `G >= 14` on
    /// every slot.  Zero for every other variant.
    pub fold_alpha: u32,
}

impl Spec {
    /// The largest `|acc|` this variant reads as an integer.
    pub fn max_abs(&self) -> u64 {
        match self.variant {
            Variant::Limbs => 1u64 << (self.limb_bits.0 + self.limb_bits.1 - 1),
            // the product is bounded by the error check, not directly:
            // 2 pm = d + (2 qm - 1) den < 2^rem_bits + 2^q_bits * 2^(rem_bits-1)
            Variant::Int8 => 1u64 << (self.q_bits + self.rem_bits - 1),
            // two 13-bit limbs, the low one range-checked, the high one bounded
            // through the high-remainder identity
            Variant::Fold => 1u64 << (2 * FOLD_LIMB),
            _ => 1u64 << (self.kmax + 9),
        }
    }
    /// Is the integer reading injective mod p?  B4 §1: two 2^16 limbs are not.
    ///
    /// The representable magnitudes are `[0, max_abs)`, and two of them are
    /// congruent mod p only if they differ by at least p — so the window has to be
    /// at most p wide, not p/2.  B4's offset variant spends half its window on the
    /// sign; here the sign is an explicit committed bit, so the whole `2^30` is
    /// usable and `K <= 66,572` comes back for free.
    pub fn sound(&self) -> bool {
        match self.variant {
            Variant::Limbs => self.limb_bits.0 + self.limb_bits.1 <= 30,
            // the high-remainder identity's no-wrap wall:
            // (2 qm - 1) fdp + fe < 2^(q_bits + rem_bits) + 2^rem_bits < p
            Variant::Fold => {
                self.max_abs() <= P as u64 && self.q_bits + self.rem_bits <= 30
            }
            _ => self.max_abs() <= P as u64,
        }
    }
    /// The longest int8 dot product whose accumulator still fits.
    pub fn max_k(&self) -> u64 {
        self.max_abs() / (127 * 127)
    }
    pub fn label(&self) -> String {
        format!("{}/{}", self.variant.name(), self.round.name())
    }
    pub fn json(&self) -> Value {
        json!({"variant": self.variant.name(), "round": self.round.name(),
               "kmax": self.kmax, "per_row": self.per_row,
               "max_abs_acc": self.max_abs(), "sound": self.sound(),
               "max_k": self.max_k(), "sticky": self.round.needs_sticky(),
               "limb_bits": [self.limb_bits.0, self.limb_bits.1],
               "q_bits": self.q_bits, "rem_bits": self.rem_bits,
               "den_bound_bits": self.rem_bits - 1})
    }
}

/// B11 stream U, lever 3: the numerator kind of every lookup this component's
/// evaluator emits, in `add_to_relation` order -- which is the order
/// `plogup::paired` batches them in and therefore the order the GKR's input
/// layer holds them in.
///
/// Unit everywhere except the `chain` variant's two `mul` lookups, whose
/// multiplicity is the committed `nz` (a zero result skips the lookup rather
/// than needing a table row). Only `pow2` and `chain` are reachable, exactly as
/// in `layer::mat_logup`'s hand-written builder, which answers `unreachable!()`
/// for the rest.
pub fn look_units(sp: &Spec) -> Vec<bool> {
    let mut u = vec![];
    for _ in 0..sp.per_row {
        // stream Z: the fold has no normalisation core; its six looks -- the
        // (q, nz) pair and five range checks -- are all at unit multiplicity
        if sp.variant == Variant::Fold {
            u.extend([true; 8]);
            continue;
        }
        u.push(true); // norm
        u.push(true); // frac
        match sp.variant {
            Variant::Pow2 => u.push(true),
            Variant::Chain => {
                for _ in 0..2 {
                    u.push(false); // mul, at multiplicity `nz`
                    u.push(true); // the output exponent's range check
                }
            }
            v => panic!("the dequant GKR backend has no looks for variant {}", v.name()),
        }
    }
    u
}


/// The distinct range-table widths a spec needs, sorted (the draw order).
pub fn range_widths(sp: &Spec) -> Vec<u32> {
    let mut w = match sp.variant {
        Variant::Limbs => vec![sp.limb_bits.0, sp.limb_bits.1],
        Variant::Int8 => vec![sp.rem_bits],
        // the limb widths, the high remainder's width, the true quotient's
        // domain and the clamp's 7 bits
        Variant::Fold => vec![FOLD_LIMB, 16, sp.rem_bits, sp.q_bits, 7],
        Variant::Round => vec![],
        // the output exponent field: what an over- or underflowing scale trips.
        // Sound, and incomplete outside bf16's normal range.
        Variant::Pow2 | Variant::Chain => vec![8],
    };
    w.sort_unstable();
    w.dedup();
    w
}

// -------------------------------------------------------------- column map --

/// Where every value of a slot lives, so the AIR, the trace generator and the
/// LogUp generator agree by construction instead of by inspection.
#[derive(Clone, Debug)]
pub struct Map {
    /// per-row columns shared by every slot in the row (the activation scale)
    pub shared: Vec<usize>,
    /// committed columns per slot
    pub per_slot: usize,
    pub slot0: usize,
    /// total committed witness columns (shared + slots)
    pub n_cols: usize,
    /// field offsets inside a slot
    pub f: Fields,
}

/// Slot field offsets.  `usize::MAX` means "this variant has no such field".
#[derive(Clone, Copy, Debug, Default)]
pub struct Fields {
    pub acc: usize,
    pub sg: usize,
    pub m2: usize,
    pub ph: usize,
    pub st: usize,
    pub r: usize,
    pub m7: usize,
    pub ef: usize,
    pub nz: usize,
    pub eo: usize,
    /// `chain`: (product mantissa, carry, exponent) for each of the two scales
    pub mul: [[usize; 3]; 2],
    /// `limbs`: the committed low limb
    pub lo: usize,
    /// `int8`: quotient and the rounding error `d = 2 pm - (2 qm - 1) den`
    pub qm: usize,
    /// `int8`: `[qm != 0]`, pinned by the (q, nz) pair table -- what pins the
    /// sign for sub-half quotients (B6 review, open gap 7)
    pub nz2: usize,
    pub d: usize,
    /// `fold`: the magnitude's limbs (`mag = flo + 2^13 fhi`; there is no acc
    /// column -- the Freivalds side reads `(flo + 2^13 fhi)(1 - 2 sg)`)
    pub flo: usize,
    pub fhi: usize,
    /// `fold`: the low product's split, `flo (128+ms)(128+mw) = fa 2^13 + fb`
    pub fa: usize,
    pub fb: usize,
    /// `fold`: the high remainder `fe = fhi mxw + fa - (2 qm - 1) fdp` and the
    /// committed per-slot divisor `fdp = dent pj = 2^(G - 14)`
    pub fe: usize,
    pub fdp: usize,
    /// `fold`: the clamped output `qc = min(qm, 127)` -- what the consumer and
    /// the harness recorder actually hold; `qm` stays the true rounding
    pub qc: usize,
}

const NONE: usize = usize::MAX;

impl Map {
    pub fn new(sp: &Spec) -> Self {
        let sticky = sp.round.needs_sticky();
        let shared: Vec<usize> = match sp.variant {
            Variant::Pow2 | Variant::Int8 => vec![0],
            Variant::Chain | Variant::Fold => vec![0, 1],
            _ => vec![],
        };
        let mut f = Fields {
            acc: 0,
            sg: NONE, m2: NONE, ph: NONE, st: NONE, r: NONE, m7: NONE, ef: NONE, nz: NONE,
            eo: NONE, mul: [[NONE; 3]; 2], lo: NONE, qm: NONE, nz2: NONE, d: NONE,
            flo: NONE, fhi: NONE, fa: NONE, fb: NONE, fe: NONE, fdp: NONE, qc: NONE,
        };
        let mut k = 1usize;
        let take = |n: usize, k: &mut usize| {
            let v = *k;
            *k += n;
            v
        };
        match sp.variant {
            Variant::Limbs => f.lo = take(1, &mut k),
            Variant::Int8 => {
                f.sg = take(1, &mut k);
                f.qm = take(1, &mut k);
                f.nz2 = take(1, &mut k);
                f.d = take(1, &mut k);
            }
            // no acc column: the magnitude's limbs stand where it would, and
            // the Freivalds side reads `(flo + 2^13 fhi)(1 - 2 sg)`
            Variant::Fold => {
                k = 0;
                f.acc = NONE;
                f.flo = take(1, &mut k);
                f.fhi = take(1, &mut k);
                f.sg = take(1, &mut k);
                f.fa = take(1, &mut k);
                f.fb = take(1, &mut k);
                f.qm = take(1, &mut k);
                f.qc = take(1, &mut k);
                f.fe = take(1, &mut k);
                f.fdp = take(1, &mut k);
            }
            _ => {
                f.sg = take(1, &mut k);
                f.m2 = take(1, &mut k);
                f.ph = take(1, &mut k);
                if sticky {
                    f.st = take(1, &mut k);
                }
                f.r = take(1, &mut k);
                f.m7 = take(1, &mut k);
                f.ef = take(1, &mut k);
                f.nz = take(1, &mut k);
                if sp.variant == Variant::Pow2 {
                    f.eo = take(1, &mut k);
                }
                if sp.variant == Variant::Chain {
                    for t in 0..2 {
                        f.mul[t] = [take(1, &mut k), take(1, &mut k), take(1, &mut k)];
                    }
                }
            }
        }
        let slot0 = shared.len();
        Self { per_slot: k, slot0, n_cols: slot0 + sp.per_row * k, shared, f }
    }
    /// Column index of field `off` in slot `s`.
    ///
    /// A real assert, not a debug one: `NONE` is `usize::MAX` and the release
    /// profile has no overflow checks, so the sentinel would not index out of
    /// range -- it would wrap to `slot0 + s * per_slot - 1`, an in-range column
    /// one below the slot's base, silently.  No live path reaches it (open item
    /// 12 traced them), and the repo's tests never call this, so the debug form
    /// had never had a chance to fire.
    pub fn at(&self, s: usize, off: usize) -> usize {
        assert_ne!(off, NONE, "Map::at: a field this variant does not carry");
        self.slot0 + s * self.per_slot + off
    }
}

// ------------------------------------------------------------------- the AIR --

/// The composed component's Freivalds claimed side: a Horner over the committed
/// accumulators, `h = sum_{t,i} s^t rv^i acc[t,i]`, which the verifier compares
/// against the r^T W Freivalds check's `acc` claim.  Rows are (token, channel-chunk), chunks
/// and tokens descending, exactly tiled -- so `z` closes each token's inner sum
/// and `h` folds tokens by `s`.
#[derive(Clone)]
pub struct YSide {
    /// rv^0..rv^{per_row-1} and rv^per_row
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    pub s: SecureField,
    pub claimed_h: SecureField,
}

/// Stream E: the output side.  One Horner claim over this component's *bf16
/// outputs*, written in the residual stream's `(sm, ef)` language, so that a consumer's own
/// claim over the same values can be compared with it -- which is what stops the
/// dequant's output being a column some other component commits a second time.
///
/// The row weights need two constants where `YSide` needs one.  This component
/// tiles `(token, channel-chunk)` with the chunks descending, so the canonical
/// channel-major exponent `N - 1 - (c tokens + t)` falls by `per_row * tokens`
/// from one chunk to the next inside a token and jumps by
/// `(chunks - 1) per_row tokens - 1` at every token boundary.  `k_in` and `k_bd`
/// are those two steps; the `oin` and `obd` selectors pick between them, keyed on
/// whether the row *begins* a token -- which is why they are their own columns
/// and not the `m1`/`m2` pair, whose `last` is the token's final chunk.
/// What an output side accumulates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OKind {
    /// the residual stream's `(sm, ef)` pair, `sm + lam ef` -- the language the wiring
    /// family's own claims are in, so the residual-stream bindings compare with it
    Pair,
    /// the argmax's order key, `0x8000 + mag - sg(2 mag + 1)` with
    /// `mag = eo 2^7 + m7`.  Degree 2 in the committed fields (one `sg mag`
    /// product), which is the whole reason the head's binding is cheap: the key
    /// never becomes a column, it is read off the dequant's own output.
    Key,
    /// Stream SS: the signed **integer** `qc (1 - 2 sg)`, the only output a
    /// `Variant::Fold` component has.  The fold's single rounding *is* B1's
    /// `quantize(v, s_v)` -- there is no bf16 waypoint to publish -- so what a
    /// consumer reads off this component is the int8 cell the key tile commits, and
    /// the claim is a Horner over those integers rather than over `(sm, ef)`
    /// pairs.  Degree 2, like `Pair` and `Key`: one `qc sg` product.
    Int,
}

/// Stream SS: `k_in`/`k_bd` are a two-valued row step and the `oin`/`obd`
/// selectors say which row takes which -- the AIR does not care what the two
/// mean.  For `Pair` and `Key` they are stream E's: the ordinary chunk step
/// inside a token, and the jump at the row that begins one.  For `Int` they are
/// the *key tile grid*'s: the ordinary step, and the jump on the row after one that
/// begins a kv head (`rope::QTie::step_out` and `step_out_h`), which is what
/// makes the claim comparable with the key tiles' own V fold under grouped-query
/// attention.  A token boundary is a head start, so one pair of constants
/// serves both.
#[derive(Clone)]
pub struct OutputClaim {
    /// the within-row weight per slot
    pub w: Vec<SecureField>,
    pub lam: SecureField,
    pub k_in: SecureField,
    pub k_bd: SecureField,
    pub claimed: SecureField,
    pub kind: OKind,
}

/// Stream II (E11): the x-side scale's claim.  One Horner over the component's
/// shared per-token scale columns on its `last` rows, under the same weights
/// the producing requantize component folds its own committed scale under -- so
/// the equality says "the matmul dequantized by the scale its input was
/// quantized with", which was the last thing left open between a producer and
/// its consumer.
///
/// `lam` folds the bf16 variant's `(ms, es)` into one field element, exactly as
/// the output side folds `(sm, ef)`.  `None` on a component whose Freivalds group has
/// no requantize seam, and on every component of a chunk with no wiring family.
#[derive(Clone)]
pub struct SSide {
    pub mu: SecureField,
    pub lam: SecureField,
    pub claimed: SecureField,
}

/// The preprocessed selectors an output side adds, in this order: the ordinary
/// row step, the token-boundary step, and whether the row is active at all.
///
/// The third is not redundant.  An inert lead row commits zeros, and while the
/// residual stream's encoding of `+0` is the pair `(0, 0)` and so contributes nothing, the
/// *order key* of `+0` is `0x8000` -- a padded row would otherwise fold junk
/// into the claim, and the consumer would have to agree about padding it cannot
/// see.  Gating the row's whole contribution keeps the claim a statement about
/// the real values only.
pub const DEQ_OSEL: [&str; 3] = ["oin", "obd", "oact"];

#[derive(Clone)]
pub struct DequantEval {
    pub log_rows: u32,
    pub sp: Spec,
    /// distinguishes this component's per-matmul preprocessed columns in a composed
    /// proof; empty for the standalone gadget
    pub tag: String,
    pub norm: NormRel,
    pub frac: FracRel,
    pub mul: MulRel,
    pub qnz: PairRel,
    /// one per entry of `range_widths(sp)`
    pub rels: Vec<RangeRel>,
    /// the composed proof's claimed side; `None` for the standalone gadget
    pub yside: Option<YSide>,
    /// stream E: the output-side claim, when this component publishes one
    pub output_claim: Option<OutputClaim>,
    /// stream II (E11): the x-side scale's claim, when a requantize seam feeds
    /// this matmul
    pub sside: Option<SSide>,
    /// B11 stream U, lever 3: this component's lookups are proved by the batched
    /// GKR instead of committed running-sum columns, so the AIR emits neither
    /// the relation entries nor the interaction masks they would allocate. The
    /// trace masks stay exactly where they were -- the committed order is the
    /// mask order by construction here (`Map`), and stream S's lesson is that
    /// the allocator must keep it.
    pub gkr: bool,
}

impl DequantEval {
    fn rel(&self, bits: u32) -> &RangeRel {
        let w = range_widths(&self.sp);
        &self.rels[w.iter().position(|x| *x == bits).expect("range width")]
    }
}

impl FrameworkEval for DequantEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_dequant_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sp = &self.sp;
        let sticky = sp.round.needs_sticky();
        let two = m31_u32(2);
        // The activation scale is per *token*, and a row is one token's worth of
        // channels, so it is one column however wide the row.
        let shared: Vec<E::F> = (0..Map::new(sp).shared.len())
            .map(|_| eval.next_trace_mask())
            .collect();
        let mut accs: Vec<E::F> = Vec::with_capacity(sp.per_row);
        // stream E: `(sg, nz, m7, eo)` per slot, from whichever rung ran, so the
        // output side can read the result as an expression rather than as a
        // column nobody committed.
        //
        // Stream SS: the `Fold` arm has no bf16 rung and so no `(nz, m7, eo)`.
        // It pushes `(sg, 0, qc, 0)` and its output side is `OKind::Int`, which
        // reads the pair `(sg, qc)` and ignores the two zeros -- so every
        // variant fills this vector and the `assert_eq!` below is a statement
        // about the plan and not about which rung ran.
        let mut outs: Vec<(E::F, E::F, E::F, E::F)> = vec![];
        for slot in 0..sp.per_row {
            // ---- stream Z: the delayed requantize, one rounding, in limbs.
            //
            // `q = round_hafz(mag mxw / 2^G)` with `mxw = (128+ms)(128+mw)` the
            // two scale significands and `2^G = 2^(G_t + G_c)` the power-of-two
            // divisor `s_v 2^268 / 2^(es_x + ew)` -- exactly `acc s_x s_w / s_v`.
            // The honest product is ~2^42 and the divisor ~2^25, so neither the
            // `int8` error check nor any single lookup can carry it.  Split
            // `mag = flo + 2^13 fhi` and `flo mxw = fa 2^13 + fb`; then
            // `T = mag mxw = (fhi mxw + fa) 2^13 + fb`, and because the rounding
            // error `d = 2T - (2 qm - 1) 2^G` is even and `G >= 14`,
            //
            //   d/2 = fe 2^13 + fb,   fe = fhi mxw + fa - (2 qm - 1) fdp,
            //
            // with `fdp = 2^(G-14)` committed per slot and pinned to
            // `dent pj`.  `d in [0, 2^(G+1))` -- i.e. the rounding -- is exactly
            // `fe in [0, 2 fdp)`, the ao fold's own two-sided check, and every
            // term is below 2^30: `fhi mxw <= fe + a + (2 qm - 1) fdp` bounds the
            // high limb through the identity, `flo mxw < 2^29` through its two
            // ranged halves.
            if sp.variant == Variant::Fold {
                let ms = shared[0].clone();
                let dent = shared[1].clone();
                let mw = eval.get_preprocessed_column(chan_col(&self.tag, "m", slot));
                let pj = eval.get_preprocessed_column(chan_col(&self.tag, "p", slot));
                let flo = eval.next_trace_mask();
                let fhi = eval.next_trace_mask();
                let sg = eval.next_trace_mask();
                let fa = eval.next_trace_mask();
                let fb = eval.next_trace_mask();
                let qm = eval.next_trace_mask();
                let qc = eval.next_trace_mask();
                let fe = eval.next_trace_mask();
                let fdp = eval.next_trace_mask();
                let limb = E::F::from(m31_u32(1 << FOLD_LIMB));
                let c128 = E::F::from(m31_u32(128));
                // the Freivalds side reads the accumulator as an expression.
                // Unlike the int8 form there is no `sg (1 - nz2)` pin: the fold
                // commits the *magnitude* (in limbs), so the Freivalds identity
                // itself pins `sg` for every nonzero accumulator -- including
                // the sub-half quotients int8's gap-7 fix was for -- and at
                // zero the expression is zero under either sign.
                let mag = flo.clone() + fhi.clone() * limb.clone();
                accs.push(mag.clone() - (mag * sg.clone()) * two);
                eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
                eval.add_constraint(fdp.clone() - dent * pj);
                // the clamp the consumer holds: qc = min(qm, 127).  With
                // qm - qc and qc both ranged, exactly one root survives.
                eval.add_constraint(
                    (qm.clone() - qc.clone()) * (qc.clone() - E::F::from(m31_u32(127))));
                let mxw = (ms + c128.clone()) * (mw + c128);
                eval.add_constraint(
                    flo.clone() * mxw.clone() - fa.clone() * limb - fb.clone());
                eval.add_constraint(
                    fe.clone() - fhi * mxw - fa.clone()
                        + (qm.clone() * two - E::F::one()) * fdp.clone(),
                );
                let fep = fdp * two - E::F::one() - fe.clone();
                let qd = qm.clone() - qc.clone();
                // stream SS: this component's output, for the output side.  `qc` is
                // the clamped quotient and `sg` its sign, and `qc (1 - 2 sg)` is
                // cell for cell what `experiments/b1`'s `quantize(v, s_v)`
                // returns -- the value the key tile commits as `vv.i8`.  The two
                // zeros stand where a bf16 rung's `nz` and `eo` would; `OKind`
                // says which pair is read.
                outs.push((sg.clone(), E::F::zero(), qc.clone(), E::F::zero()));
                if !self.gkr {
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.q_bits), E::EF::one(), core::slice::from_ref(&qm)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(7), E::EF::one(), core::slice::from_ref(&qc)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.q_bits), E::EF::one(), core::slice::from_ref(&qd)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(FOLD_LIMB), E::EF::one(), core::slice::from_ref(&flo)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(FOLD_LIMB), E::EF::one(), core::slice::from_ref(&fb)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(16), E::EF::one(), core::slice::from_ref(&fa)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.rem_bits), E::EF::one(), core::slice::from_ref(&fe)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.rem_bits), E::EF::one(), core::slice::from_ref(&fep)));
                }
                continue;
            }
            let acc = eval.next_trace_mask();
            accs.push(acc.clone());
            match sp.variant {
                // ---- B4's baseline: `acc + off = lo + 2^b hi`, both limbs checked
                Variant::Limbs => {
                    let (b0, b1) = sp.limb_bits;
                    let off = m31_u32(1 << (b0 + b1 - 1));
                    let lo = eval.next_trace_mask();
                    let hi = (acc + E::F::from(off) - lo.clone()) * m31_u32(1 << b0).inverse();
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            self.rel(b0), E::EF::one(), core::slice::from_ref(&lo)));
                        eval.add_to_relation(RelationEntry::new(
                            self.rel(b1), E::EF::one(), core::slice::from_ref(&hi)));
                    }
                }
                // ---- one rounding straight to int8: no bf16 anywhere
                //
                // `q = round_hafz(acc num / den)` stated through its *error* rather
                // than through the product.  Range-checking `mag num` directly would
                // need a 2^29 table; the error does not:
                //
                //   d = 2 mag num - (2 qm - 1) den,      0 <= d < 2 den
                //
                // is exactly `qm = round_half_up(mag num / den)`, and `d` is bounded
                // by `2 den` however large the product is.  Nothing wraps, because
                // `2 mag num = d + (2 qm - 1) den < 2^rem_bits + 2^q_bits 2^(rem_bits-1)`,
                // which is below p for the widths the composition actually uses --
                // and `den < 2^(rem_bits-1)` is the denominator range check B5
                // already pays for at the per-token rung, not a new one.
                Variant::Int8 => {
                    let den = shared[0].clone();
                    let num = eval.get_preprocessed_column(chan_col(&self.tag, "n", slot));
                    let sg = eval.next_trace_mask();
                    let qm = eval.next_trace_mask();
                    let nz2 = eval.next_trace_mask();
                    let d = eval.next_trace_mask();
                    eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
                    // B6 review, open gap 7: nothing pinned the sign for quotients
                    // below 1/2 (and for zero accumulators).  Canonicalise: sg = 0
                    // whenever qm = 0.  The reconstruction still holds for a
                    // negative acc there -- d = den - 2|acc| num, in (0, den] --
                    // and for qm >= 1 the range check on d pins sg on its own.
                    eval.add_constraint(sg.clone() * (E::F::one() - nz2.clone()));
                    let mag = acc.clone() - (acc * sg) * two;
                    eval.add_constraint(
                        d.clone() - (mag * num) * two + (qm.clone() * two - E::F::one()) * den.clone(),
                    );
                    // B6 review, open gap 7: `2 den - d` lets d = 2 den through, so
                    // an exact tie verifies with either of two outputs; `-1` closes
                    // it (d <= 2 den - 1, and the binding's only witness is d = 0).
                    let dp = den * two - E::F::one() - d.clone();
                    // (qm, [qm != 0]) in one pair-table lookup: with q_bits = 7 the
                    // domain *is* B1's clamp at 127, so the statement is "no clamp
                    // was needed" -- sound, and incomplete exactly where B1 clamps.
                    eval.add_to_relation(RelationEntry::new(
                        &self.qnz, E::EF::one(), &[qm, nz2]));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.rem_bits), E::EF::one(), core::slice::from_ref(&d)));
                    eval.add_to_relation(RelationEntry::new(
                        self.rel(sp.rem_bits), E::EF::one(), core::slice::from_ref(&dp)));
                }
                // ---- the normalisation core, plus whatever the scale needs
                _ => {
                    let sg = eval.next_trace_mask();
                    let m2 = eval.next_trace_mask();
                    let ph = eval.next_trace_mask();
                    let st = if sticky { eval.next_trace_mask() } else { E::F::zero() };
                    let r = eval.next_trace_mask();
                    let m7 = eval.next_trace_mask();
                    let ef = eval.next_trace_mask();
                    let nz = eval.next_trace_mask();

                    eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
                    // the sign of zero is zero: without this a prover may commit
                    // -0.0 where B1 has +0.0 -- a different bit pattern, and the
                    // consumer is a table indexed by the pattern.
                    eval.add_constraint(sg.clone() - sg.clone() * nz.clone());
                    // |acc| = M2 ph + r.  This one constraint is the integer
                    // reading: the table's domain bounds the right-hand side below
                    // 2^(kmax+9) <= 2^30 < p, so it cannot decompose `acc + p`.
                    let mag = acc.clone() - (acc * sg.clone()) * two;
                    eval.add_constraint(mag - (m2.clone() * ph.clone() + r.clone()));

                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.norm, E::EF::one(),
                            &[m2, ph.clone(), st.clone(), m7.clone(), ef.clone(),
                              nz.clone()]));
                        eval.add_to_relation(RelationEntry::new(
                            &self.frac, E::EF::one(), &[ph, r, st]));
                    }

                    match sp.variant {
                        // the scale is 2^j: one exponent add, one range check, and
                        // `nz` is what stops it being added to a zero result
                        Variant::Pow2 => {
                            let jw = eval.get_preprocessed_column(chan_col(&self.tag, "j", slot));
                            let eo = eval.next_trace_mask();
                            eval.add_constraint(
                                eo.clone() - ef - nz.clone() * (shared[0].clone() + jw));
                            outs.push((sg.clone(), nz.clone(), m7.clone(), eo.clone()));
                            if !self.gkr {
                                eval.add_to_relation(RelationEntry::new(
                                    self.rel(8), E::EF::one(), core::slice::from_ref(&eo)));
                            }
                            // y = sg 2^15 + eo 2^7 + m7 is degree 1, so the consumer
                            // reads it as an expression and it is never a column.
                        }
                        // general bf16 scales: two mantissa products, each a
                        // 2^14-row lookup plus an exponent add
                        Variant::Chain => {
                            let scales = [
                                (shared[0].clone(), shared[1].clone()),
                                (eval.get_preprocessed_column(chan_col(&self.tag, "m", slot)),
                                 eval.get_preprocessed_column(chan_col(&self.tag, "e", slot))),
                            ];
                            let mut m_in = m7;
                            let mut e_in = ef;
                            for (ms, es) in scales {
                                let po = eval.next_trace_mask();
                                let ca = eval.next_trace_mask();
                                let eo = eval.next_trace_mask();
                                // gated at multiplicity `nz`: a zero result skips
                                // the lookup entirely rather than needing a row
                                if !self.gkr {
                                    eval.add_to_relation(RelationEntry::new(
                                        &self.mul, E::EF::from(nz.clone()),
                                        &[m_in, ms, po.clone(), ca.clone()]));
                                }
                                eval.add_constraint(
                                    eo.clone()
                                        - nz.clone() * (e_in + es + ca - E::F::from(m31_u32(127))),
                                );
                                eval.add_constraint(po.clone() - po.clone() * nz.clone());
                                if !self.gkr {
                                    eval.add_to_relation(RelationEntry::new(
                                        self.rel(8), E::EF::one(), core::slice::from_ref(&eo)));
                                }
                                m_in = po;
                                e_in = eo;
                            }
                            // the chain's output is the *second* multiply's
                            // mantissa and exponent
                            outs.push((sg.clone(), nz.clone(), m_in.clone(), e_in.clone()));
                        }
                        _ => {}
                    }
                }
            }
        }
        // The composed proof's claimed side: one Horner over the committed
        // accumulators, so the matmul identity the r^T W Freivalds check closes reads the
        // *same* cells the dequant rounds -- nothing is committed twice.
        if let Some(y) = &self.yside {
            let sel: Vec<E::F> = DEQ_SEL
                .iter()
                .map(|n| eval.get_preprocessed_column(deq_sel(&self.tag, n)))
                .collect();
            let (nofirst, last, m1, m2, end) = (sel[0].clone(), sel[1].clone(),
                sel[2].clone(), sel[3].clone(), sel[4].clone());
            let [z_prev, z] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let [h_prev, h] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut zf = E::EF::zero();
            for (i, a) in accs.iter().enumerate() {
                zf += a.clone() * y.rv_pow[i];
            }
            eval.add_constraint(z.clone() - (z_prev * y.rv_step) * nofirst - zf);
            let mut mult: E::EF = E::EF::from(m1);
            mult += m2 * y.s;
            eval.add_constraint(h.clone() - h_prev * mult - z * last);
            let claim_h = eval.claim(y.claimed_h);
            eval.add_constraint((h - claim_h) * end);
        }
        // stream E: the output side.  One accumulator, a two-valued row step,
        // and the same `end` the claimed side closes on.
        if let Some(o) = &self.output_claim {
            assert_eq!(outs.len(), sp.per_row, "the output side needs every slot's result");
            let oin = eval.get_preprocessed_column(deq_sel(&self.tag, DEQ_OSEL[0]));
            let obd = eval.get_preprocessed_column(deq_sel(&self.tag, DEQ_OSEL[1]));
            let oact = eval.get_preprocessed_column(deq_sel(&self.tag, DEQ_OSEL[2]));
            let oend = eval.get_preprocessed_column(deq_sel(&self.tag, "end"));
            let [u_prev, u] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut f = E::EF::zero();
            for (i, (sg, nz, m7, eo)) in outs.iter().enumerate() {
                match o.kind {
                    OKind::Pair => {
                        // the residual stream's encoding: a signed significand carrying the
                        // implicit 128, and the biased exponent.  `nz` makes a
                        // zero result the pair `(0, 0)`, which is what `canon`
                        // gives it on the other side, and the component's own
                        // `sg - sg nz = 0` already forbids a signed zero -- so
                        // the two sides cannot disagree about -0.
                        let sm = (nz.clone() * E::F::from(m31_u32(128)) + m7.clone())
                            * (E::F::one() - sg.clone() * two);
                        f += sm * o.w[i] + eo.clone() * (o.lam * o.w[i]);
                    }
                    OKind::Key => {
                        // `mag` is the bf16 pattern's low 15 bits, which is
                        // `eo 2^7 + m7` -- degree 1 -- and the key flips it about
                        // 0x7FFF for a negative, which costs one product.
                        let mag = eo.clone() * E::F::from(m31_u32(128)) + m7.clone();
                        let key = E::F::from(m31_u32(0x8000)) + mag.clone()
                            - sg.clone() * (mag * two + E::F::one());
                        f += key * o.w[i];
                    }
                    // stream SS: the fold's integer.  `m7` holds `qc` here and
                    // `nz`/`eo` are zero (see the `Fold` arm), so this is
                    // `qc (1 - 2 sg)` and nothing else -- the same signed
                    // integer `rope::QuantEval`'s output side folds and the
                    // key tile's V column commits.
                    OKind::Int => {
                        f += (m7.clone() * (E::F::one() - sg.clone() * two)) * o.w[i];
                    }
                }
            }
            let mult: E::EF = E::EF::from(oin) * o.k_in + E::EF::from(obd) * o.k_bd;
            eval.add_constraint(u.clone() - u_prev * mult - f * oact);
            let claim = eval.claim(o.claimed);
            eval.add_constraint((u - claim) * oend);
        }
        // Stream II (E11): the x-side scale, folded on the `last` rows.  `m1`
        // and `m2` are `[j > 0 and not last]` and `[j > 0 and last]`, so the
        // step is `m1 + m2 mu` -- degree 1, as the claimed side's already is --
        // and an inert lead row is neither `last` nor `m2`, so it carries the
        // (still zero) accumulator and folds nothing.
        if let Some(sc) = &self.sside {
            let sel: Vec<E::F> = DEQ_SEL
                .iter()
                .map(|n| eval.get_preprocessed_column(deq_sel(&self.tag, n)))
                .collect();
            let (last, m1, m2, end) =
                (sel[1].clone(), sel[2].clone(), sel[3].clone(), sel[4].clone());
            let [v_prev, v] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let mut f = E::EF::from(shared[0].clone());
            if shared.len() > 1 {
                f += E::EF::from(shared[1].clone()) * sc.lam;
            }
            let mult: E::EF = E::EF::from(m1) + E::EF::from(m2) * sc.mu;
            eval.add_constraint(v.clone() - v_prev * mult - f * E::EF::from(last));
            let claim = eval.claim(sc.claimed);
            eval.add_constraint((v - claim) * end);
        }
        if !self.gkr {
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

// ------------------------------------------------------------------- tables --

/// A preprocessed table with a committed multiplicity column, yielded at `-m`.
/// One shape serves all four tables; the arity follows the column list.
#[derive(Clone)]
pub struct TableEval {
    pub log_rows: u32,
    pub cols: Vec<PreProcessedColumnId>,
    pub kind: TableKind,
}

#[derive(Clone)]
pub enum TableKind {
    Norm(NormRel),
    Frac(FracRel),
    Mul(MulRel),
    Pair(PairRel),
    Range(RangeRel),
}

impl TableKind {
    fn combine(&self, v: &[PackedM31]) -> PackedQM31 {
        match self {
            TableKind::Norm(r) => r.combine(v),
            TableKind::Frac(r) => r.combine(v),
            TableKind::Mul(r) => r.combine(v),
            TableKind::Pair(r) => r.combine(v),
            TableKind::Range(r) => r.combine(v),
        }
    }
}

use stwo::prover::backend::simd::m31::PackedM31;

impl FrameworkEval for TableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_table_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let row: Vec<E::F> =
            self.cols.iter().map(|c| eval.get_preprocessed_column(c.clone())).collect();
        let m = eval.next_trace_mask();
        let neg = -E::EF::from(m);
        match &self.kind {
            TableKind::Norm(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Frac(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Mul(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Pair(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            TableKind::Range(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
        }
        eval.finalize_logup();
        eval
    }
}

// ------------------------------------------------------------------ witness --

/// What a dishonest prover changes before committing.
#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// decompose `acc + p` rather than `acc`: the same field element, a different
    /// integer.  B4's negative control, now against the normalisation's own bound.
    Wrap(usize),
    /// a different but perfectly legal bf16 mantissa: only the lookup can object
    Mantissa(usize),
    /// a non-normalised decomposition: `M2` halved, `ph` doubled
    Denorm(usize),
    /// `r >= ph`, i.e. a decomposition that borrows from the significand
    Borrow(usize),
    /// the sticky flag flipped -- what round-to-nearest-even turns on
    Sticky(usize),
    /// one row of the normalisation table changed, witness rebuilt to match
    Table(usize),
    /// int8: the other side of an exact tie -- commit `qm - 1` and `d + 2 den`,
    /// which B6's `2 den - d` check accepted whenever `d = 0`
    TieShift(usize),
    /// int8: flip the sign on a sub-half quotient -- accepted before the sign pin
    SignFlip(usize),
}

/// The accumulators a run dequantises, and the scales they carry.
pub struct Witness {
    /// accumulators in row-major `(row, slot)` order
    pub acc: Vec<i64>,
    /// per-token power-of-two scale exponent (one per row)
    pub jx: Vec<i32>,
    /// per-channel power-of-two scale exponent (one per slot, preprocessed)
    pub jw: Vec<i32>,
    /// per-token bf16 scale as (mantissa field, biased exponent)
    pub sx: Vec<(u32, u32)>,
    /// per-channel bf16 scale
    pub sw: Vec<(u32, u32)>,
    /// `int8`: per-token denominator, per-channel numerator
    pub den: Vec<i64>,
    pub num: Vec<i64>,
    pub source: String,
}

impl Witness {
    /// Synthetic accumulators with a realistic spread.
    ///
    /// A dynamic max-scale puts the *maximum* activation at 127, so the rms of a
    /// quantized vector is several times smaller — and it is the rms that sets the
    /// accumulator's spread.  B3 measured max |acc| two orders of magnitude below
    /// `127^2 K`; a uniform draw would put every accumulator at the worst case and
    /// make the `kmax` sweep meaningless.
    pub fn synthetic(n: usize, per_row: usize, k: usize, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let rows = n.div_ceil(per_row.max(1)).max(1);
        let acc: Vec<i64> = (0..n)
            .map(|_| {
                let g: f64 = (0..3).map(|_| rng.random_range(-1.0f64..1.0)).sum();
                (g / 3.0 * 127.0 * 127.0 * (k as f64).sqrt() / 8.0).round() as i64
            })
            .collect();
        Self {
            acc,
            jx: (0..rows).map(|_| -(rng.random_range(18..24) as i32)).collect(),
            jw: (0..per_row.max(1)).map(|_| -(rng.random_range(18..24) as i32)).collect(),
            sx: (0..rows).map(|_| (rng.random_range(0..128), rng.random_range(100..120))).collect(),
            sw: (0..per_row.max(1))
                .map(|_| (rng.random_range(0..128), rng.random_range(100..120)))
                .collect(),
            den: vec![1i64 << 17; rows],
            num: vec![1i64; per_row.max(1)],
            source: format!("synthetic(seed={seed}, K={k})"),
        }
        .fit_int8(seed)
    }

    /// Pick the `int8` site's denominator and numerator so the quotient lands
    /// inside int8, as the real site does by construction.
    ///
    /// `q = round(acc num / den)` with `den` a softmax denominator and `num` a
    /// power-of-two scale ratio: `s_o = max|x|/127` is what makes the output fill
    /// the int8 range, so `amax num / den ~ 127`.  A witness that ignores this
    /// measures a clamp rather than a rounding -- and on B1's real attention output
    /// the clamp count is exactly zero (`b6/extract.py`'s `clamped.attnout`).
    fn fit_int8(mut self, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9e37);
        let amax = self.acc.iter().map(|a| a.unsigned_abs()).max().unwrap_or(1).max(1) as i64;
        let rows = self.den.len().max(1);
        self.den = (0..rows).map(|_| (1i64 << 17) + rng.random_range(0..4096)).collect();
        let dmin = *self.den.iter().min().unwrap();
        let mut cap = 1i64;
        while cap * 2 * amax <= 127 * dmin {
            cap *= 2;
        }
        let per_row = self.num.len().max(1);
        self.num = (0..per_row)
            .map(|_| cap >> rng.random_range(0..3u32).min(cap.ilog2()))
            .collect();
        self
    }

    /// B1's real accumulators, as dumped by `b6/extract.py`; tiled to `n`.
    pub fn load(dir: &str, name: &str, n: usize, per_row: usize) -> Result<Self, String> {
        let path = format!("{dir}/acc_{name}.i32");
        let raw = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        let all: Vec<i64> = raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
            .collect();
        if all.is_empty() {
            return Err(format!("{path} is empty"));
        }
        let mut w = Self::synthetic(n, per_row, 1, 7);
        w.acc = (0..n).map(|i| all[i % all.len()]).collect();
        w.source = format!("B1 {name} accumulators [{} values, tiled]", all.len());
        // the int8 site's den/num follow from the accumulators, so they have to be
        // re-derived once the real ones are in place
        Ok(w.fit_int8(7))
    }

    /// Real accumulators from `--witness`, else synthetic.  A `--witness` that
    /// does not load is fatal, not a fallback: the same rule as `layer_of`, for
    /// the same reason (a run must never pass on data it did not name).
    pub fn get(dir: &Option<String>, name: &str, n: usize, per_row: usize, k: usize) -> Self {
        if let Some(d) = dir {
            match Self::load(d, name, n, per_row) {
                Ok(w) => return w,
                Err(e) => {
                    eprintln!("--witness {d}: the {name} accumulators did not load: {e}");
                    eprintln!("refusing to fall back to synthetic ones; drop --witness for those");
                    std::process::exit(2);
                }
            }
        }
        Self::synthetic(n, per_row, k, 7)
    }
}

// ------------------------------------------------------------------ the run --

fn m31_of(v: i64) -> M31 {
    let p = P as i64;
    M31::from_u32_unchecked((((v % p) + p) % p) as u32)
}

/// One lookup, as the AIR adds it: column indices into the generator's `all`, and
/// the column holding its multiplicity (`None` for 1).
enum Look {
    Norm([usize; 6]),
    Frac([usize; 3]),
    Mul([usize; 4], usize),
    Qnz([usize; 2]),
    Range(usize, usize),
}

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub sp: Spec,
    pub n: usize,
    pub log_rows: u32,
    /// values no table could absorb: the incompleteness a bound buys
    pub off_table: usize,
    /// results that match an independent f32 computation
    pub ref_ok: usize,
    pub ref_n: usize,
    pub first_bad: Option<(usize, i64, u32, u32)>,
    pub table_cells_pre: usize,
    pub table_cells_fresh: usize,
    pub table_rows: Vec<(String, u32)>,
    pub trace_secs: f64,
    pub interaction_secs: f64,
    pub n_columns: [usize; 3],
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0
    }
    /// Cells the gadget *adds*: the accumulator column itself is committed by the
    /// matmul, and the tables amortise across a whole proof.
    pub fn marginal(&self) -> f64 {
        (self.proven.fresh_cells() - self.table_cells_fresh) as f64 / self.n as f64 - 1.0
    }
    pub fn json(&self, src: &str) -> Value {
        let p = &self.proven;
        json!({
            "spec": self.sp.json(), "source": src, "n": self.n, "log_rows": self.log_rows,
            "cells_per_acc_marginal": self.marginal(),
            "cells_per_acc_with_tables":
                (p.fresh_cells() + self.table_cells_pre) as f64 / self.n as f64 - 1.0,
            "table_cells_preprocessed": self.table_cells_pre,
            "table_cells_fresh": self.table_cells_fresh,
            "tables": self.table_rows.iter()
                .map(|(k, l)| json!({"table": k, "log_rows": l, "rows": 1u64 << l}))
                .collect::<Vec<_>>(),
            "accs_per_sec": self.n as f64 / p.prove_secs,
            "off_table": self.off_table,
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "first_mismatch": self.first_bad.map(|(i, a, want, got)|
                json!({"index": i, "acc": a, "reference": want, "gadget": got})),
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "trace_secs": self.trace_secs, "interaction_secs": self.interaction_secs,
            "sums": self.sums.json(), "accepts": self.accepts(),
            "point": p.json(),
        })
    }
}

pub fn run(w: &Witness, sp: &Spec, tamper: Option<Tamper>) -> Outcome {
    // `Spec::sound()` was reported, not asserted, so `--kmax 25` would run unsound
    // (B6 review).  Refuse instead.
    assert!(sp.sound(), "unsound spec: {:?} reads an integer window above p", sp.json());
    // Stream Z: the fold's witness needs real per-token and per-channel bf16
    // scales and a per-channel s_v, which the standalone bench witness does not
    // carry.  The fold is exercised through the composed layer path only.
    assert!(sp.variant != Variant::Fold,
            "the standalone gadget bench has no fold witness; compose a layer instead");
    let t0 = std::time::Instant::now();
    let n = w.acc.len();
    let map = Map::new(sp);
    let f = map.f;
    let log_rows = log_rows_for(n, sp.per_row);
    let rows = 1usize << log_rows;
    let sticky = sp.round.needs_sticky();
    let widths = range_widths(sp);
    let ridx = |b: u32| widths.iter().position(|x| *x == b).unwrap();

    // ---- the tables --------------------------------------------------------
    let mut norm = bf16::norm_table(sp.kmax, sp.round);
    if let Some(Tamper::Table(i)) = tamper {
        // B4 §3 again: a corrupted table with a matching witness is internally
        // consistent, and only the preprocessed root gives it away.
        let i = i % norm.len();
        norm[i].m7 = (norm[i].m7 + 1) % 128;
    }
    let log_norm = log_rows_for(norm.len(), 1);
    // every valid (power of two, remainder below it, is-it-nonzero) triple
    let frac: Vec<(u32, u32, u32)> = std::iter::once((0u32, 0u32, 0u32))
        .chain((0..=sp.kmax).flat_map(|k| {
            (0..1u32 << k).map(move |r| (1u32 << k, r, u32::from(sticky && r != 0)))
        }))
        .collect();
    let log_frac = log_rows_for(frac.len(), 1);
    let mul = if sp.variant == Variant::Chain { bf16::mul_table(sp.round) } else { vec![] };
    let log_mul = log_rows_for(mul.len().max(1), 1);
    // fast lookups from value to table row
    let norm_at = |m2: u32, ph: u32, st: u32| -> Option<usize> {
        norm.iter().position(|r| r.m2 == m2 && r.ph == ph && r.st == st)
    };
    let frac_at = |ph: u32, r: u64| -> Option<usize> {
        if ph == 0 {
            return (r == 0).then_some(0);
        }
        (r < ph as u64).then(|| ph as usize + r as usize)
    };
    debug_assert!(frac_at(1, 0) == Some(1) || sp.kmax == 0);

    // ---- the witness, laid out in rows -------------------------------------
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; map.n_cols];
    let mut mult_norm = vec![0u32; 1 << log_norm];
    let mut mult_frac = vec![0u32; 1 << log_frac];
    let mut mult_mul = vec![0u32; 1 << log_mul];
    let mut mult_qnz = vec![0u32; 1usize << sp.q_bits];
    let mut mult_r: Vec<Vec<u32>> = widths.iter().map(|b| vec![0u32; 1usize << b]).collect();
    let mut off_table = 0usize;
    let (mut ref_ok, mut ref_n) = (0usize, 0usize);
    let mut first_bad = None;
    let bump = |m: &mut Vec<u32>, v: i64, off: &mut usize| {
        if v >= 0 && (v as usize) < m.len() {
            m[v as usize] += 1;
        } else {
            *off += 1;
        }
    };

    for row in 0..rows {
        match sp.variant {
            Variant::Pow2 => cols[map.shared[0]][row] = m31_of(w.jx[row % w.jx.len()] as i64),
            Variant::Int8 => cols[map.shared[0]][row] = m31_of(w.den[row % w.den.len()]),
            Variant::Chain => {
                let (m, e) = w.sx[row % w.sx.len()];
                cols[map.shared[0]][row] = m31_u32(m);
                cols[map.shared[1]][row] = m31_u32(e);
            }
            _ => {}
        }
        for s in 0..sp.per_row {
            let i = row * sp.per_row + s;
            let a = if i < n { w.acc[i] } else { 0 };
            let set = |off: usize, v: M31, cols: &mut Vec<Vec<M31>>| {
                cols[map.at(s, off)][row] = v;
            };
            set(f.acc, m31_of(a), &mut cols);
            match sp.variant {
                Variant::Limbs => {
                    let (b0, b1) = sp.limb_bits;
                    let off = 1i64 << (b0 + b1 - 1);
                    let mut v = (a + off) as u64;
                    if matches!(tamper, Some(Tamper::Wrap(t)) if t == i) {
                        v += P as u64;
                    }
                    let lo = (v & ((1u64 << b0) - 1)) as i64;
                    let hi = (v >> b0) as i64;
                    set(f.lo, m31_of(lo), &mut cols);
                    bump(&mut mult_r[ridx(b0)], lo, &mut off_table);
                    bump(&mut mult_r[ridx(b1)], hi, &mut off_table);
                }
                Variant::Int8 => {
                    let den = w.den[row % w.den.len()];
                    let num = w.num[s % w.num.len()];
                    let mag = a.abs();
                    let pm = mag * num;
                    let qm = (2 * pm + den) / (2 * den);
                    // the canonical witness: sg = 0 whenever qm = 0 (the sign pin);
                    // the reconstruction then reads d = den - 2 mag num, in (0, den]
                    let mut sg = i64::from(a < 0 && qm != 0);
                    let mut qm = qm;
                    let mut d = if a < 0 && qm == 0 { den - 2 * pm }
                                else { 2 * pm - (2 * qm - 1) * den };
                    match tamper {
                        // the binding's other witness: d lands at d + 2 den, which the
                        // -1 in the bound now excludes for every d
                        Some(Tamper::TieShift(t)) if t == i && qm >= 1 => {
                            qm -= 1;
                            d += 2 * den;
                        }
                        // sign malleability below one half: sg = 1 with qm = 0
                        // satisfied every pre-fix constraint when acc < 0
                        Some(Tamper::SignFlip(t)) if t == i && qm == 0 && a != 0 => {
                            sg = 1;
                            d = 2 * pm - (2 * qm - 1) * den;
                        }
                        _ => {}
                    }
                    set(f.sg, m31_of(sg), &mut cols);
                    set(f.qm, m31_of(qm), &mut cols);
                    set(f.nz2, m31_of(i64::from(qm != 0)), &mut cols);
                    set(f.d, m31_of(d), &mut cols);
                    bump(&mut mult_qnz, qm, &mut off_table);
                    bump(&mut mult_r[ridx(sp.rem_bits)], d, &mut off_table);
                    bump(&mut mult_r[ridx(sp.rem_bits)], 2 * den - 1 - d, &mut off_table);
                    if i < n {
                        ref_n += 1;
                        // the reference: round half away from zero, in f64
                        let want = ((mag as f64 * num as f64) / den as f64 + 0.5).floor() as i64;
                        if want == qm {
                            ref_ok += 1;
                        } else if first_bad.is_none() {
                            first_bad = Some((i, a, want as u32, qm as u32));
                        }
                    }
                }
                _ => {
                    let sg = u32::from(a < 0);
                    let mut mag = a.unsigned_abs();
                    if matches!(tamper, Some(Tamper::Wrap(t)) if t == i) {
                        mag += P as u64;
                    }
                    let (mut m2, mut ph, mut st, mut r) = match bf16::normalise(mag, sp.kmax) {
                        Some(v) => v,
                        None => {
                            // the magnitude needs a round-bit weight the table does
                            // not carry: report it, do not silently clamp
                            off_table += 1;
                            (0, 0, 0, 0)
                        }
                    };
                    // ties-away and truncation do not read the sticky flag, so
                    // neither table carries an st = 1 row and the witness must not
                    // either -- the AIR passes a constant zero in its place.
                    if !sticky {
                        st = 0;
                    }
                    // Both of these keep `mag = M2 ph + r` satisfied, so the
                    // *lookup* is what has to reject them, not the reconstruction.
                    match tamper {
                        Some(Tamper::Denorm(t)) if t == i && m2 >= 256 => {
                            let add = if m2 % 2 == 1 { ph as u64 } else { 0 };
                            m2 /= 2;
                            ph *= 2;
                            r += add;
                        }
                        Some(Tamper::Borrow(t)) if t == i && m2 > 0 => {
                            m2 -= 1;
                            r += ph as u64;
                        }
                        Some(Tamper::Sticky(t)) if t == i => st ^= 1,
                        _ => {}
                    }
                    let ri = norm_at(m2, ph, st);
                    let mut nr = ri.map(|x| norm[x]).unwrap_or_default();
                    if matches!(tamper, Some(Tamper::Mantissa(t)) if t == i) {
                        nr.m7 = (nr.m7 + 1) % 128;
                    }
                    set(f.sg, m31_u32(sg), &mut cols);
                    set(f.m2, m31_u32(m2), &mut cols);
                    set(f.ph, m31_u32(ph), &mut cols);
                    if sticky {
                        set(f.st, m31_u32(st), &mut cols);
                    }
                    set(f.r, m31_of(r as i64), &mut cols);
                    set(f.m7, m31_u32(nr.m7), &mut cols);
                    set(f.ef, m31_u32(nr.expfield), &mut cols);
                    set(f.nz, m31_u32(nr.nz), &mut cols);
                    match ri {
                        Some(x) => mult_norm[x] += 1,
                        None => off_table += 1,
                    }
                    match frac_at(ph, r) {
                        Some(x) => mult_frac[x] += 1,
                        None => off_table += 1,
                    }
                    match sp.variant {
                        Variant::Pow2 => {
                            let j = w.jx[row % w.jx.len()] as i64 + w.jw[s % w.jw.len()] as i64;
                            let eo =
                                if nr.nz == 1 { nr.expfield as i64 + j } else { 0 };
                            set(f.eo, m31_of(eo), &mut cols);
                            bump(&mut mult_r[ridx(8)], eo, &mut off_table);
                            if i < n && sp.round == Round::Even {
                                ref_n += 1;
                                let want = bf16::from_f64(a as f64 * (j as f64).exp2());
                                let got = bf16::pattern(sg, eo.max(0) as u32, nr.m7);
                                if want == got {
                                    ref_ok += 1;
                                } else if first_bad.is_none() {
                                    first_bad = Some((i, a, want as u32, got as u32));
                                }
                            }
                        }
                        Variant::Chain => {
                            let mut m_in = nr.m7;
                            let mut e_in = nr.expfield as i64;
                            let scales = [w.sx[row % w.sx.len()], w.sw[s % w.sw.len()]];
                            for (t, (ms, es)) in scales.iter().enumerate() {
                                let mr = mul[(m_in * 128 + ms) as usize];
                                let (po, ca) =
                                    if nr.nz == 1 { (mr.m7, mr.carry) } else { (0, 0) };
                                let eo = if nr.nz == 1 {
                                    e_in + *es as i64 - 127 + ca as i64
                                } else {
                                    0
                                };
                                set(f.mul[t][0], m31_u32(po), &mut cols);
                                set(f.mul[t][1], m31_u32(ca), &mut cols);
                                set(f.mul[t][2], m31_of(eo), &mut cols);
                                if nr.nz == 1 {
                                    mult_mul[(m_in * 128 + ms) as usize] += 1;
                                }
                                bump(&mut mult_r[ridx(8)], eo, &mut off_table);
                                m_in = po;
                                e_in = eo;
                            }
                            if i < n && sp.round == Round::Even {
                                ref_n += 1;
                                // the reference: the same chain of three bf16
                                // roundings, each one done in f64
                                let y0 = bf16::to_f32(bf16::from_f64(a.unsigned_abs() as f64))
                                    as f64;
                                let sx = bf16::to_f32(bf16::pattern(0, w.sx[row % w.sx.len()].1,
                                                                    w.sx[row % w.sx.len()].0))
                                    as f64;
                                let sw = bf16::to_f32(bf16::pattern(0, w.sw[s % w.sw.len()].1,
                                                                    w.sw[s % w.sw.len()].0))
                                    as f64;
                                let t1 = bf16::to_f32(bf16::from_f64(y0 * sx)) as f64;
                                // the chain is computed on the magnitude, so the
                                // sign has to be put back on
                                let want = bf16::from_f64(t1 * sw) | (u16::from(a < 0) << 15);
                                let got = bf16::pattern(sg, e_in.max(0) as u32, m_in);
                                if want == got {
                                    ref_ok += 1;
                                } else if first_bad.is_none() {
                                    first_bad = Some((i, a, want as u32, got as u32));
                                }
                            }
                        }
                        Variant::Round if i < n && sp.round == Round::Even => {
                            ref_n += 1;
                            let want = bf16::bf16_of_u64(a.unsigned_abs())
                                | (u16::from(a < 0) << 15);
                            let got = bf16::pattern(sg, nr.expfield, nr.m7);
                            if want == got {
                                ref_ok += 1;
                            } else if first_bad.is_none() {
                                first_bad = Some((i, a, want as u32, got as u32));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let trace_secs = t0.elapsed().as_secs_f64();

    // ---- tree 0: tables and the per-channel scales --------------------------
    let mut pre_ids: Vec<PreProcessedColumnId> = vec![];
    let mut pre: Vec<Eval> = vec![];
    let mut order: Vec<(u32, Vec<PreProcessedColumnId>, TableKindTag)> = vec![];
    if sp.variant.normalises() {
        let get: [fn(&NormRow) -> u32; 6] =
            [|r| r.m2, |r| r.ph, |r| r.st, |r| r.m7, |r| r.expfield, |r| r.nz];
        for (c, g) in NORM_COLS.iter().zip(get) {
            pre_ids.push(id(*c));
            pre.push(eval_of(log_norm, (0..1usize << log_norm)
                .map(|i| m31_u32(norm.get(i).map(g).unwrap_or(0))).collect()));
        }
        order.push((log_norm, NORM_COLS.iter().map(|c| id(*c)).collect(), TableKindTag::Norm));
        for t in 0..3 {
            pre_ids.push(id(FRAC_COLS[t]));
            pre.push(eval_of(log_frac, (0..1usize << log_frac)
                .map(|i| {
                    let x = frac.get(i).copied().unwrap_or((0, 0, 0));
                    m31_u32([x.0, x.1, x.2][t])
                }).collect()));
        }
        order.push((log_frac, FRAC_COLS.iter().map(|c| id(*c)).collect(), TableKindTag::Frac));
    }
    if sp.variant == Variant::Chain {
        for t in 0..4 {
            pre_ids.push(id(MUL_COLS[t]));
            pre.push(eval_of(log_mul, (0..1usize << log_mul)
                .map(|i| {
                    let x = mul.get(i).copied()
                        .unwrap_or(MulRow { a: 0, b: 0, m7: 0, carry: 0 });
                    m31_u32([x.a, x.b, x.m7, x.carry][t])
                }).collect()));
        }
        order.push((log_mul, MUL_COLS.iter().map(|c| id(*c)).collect(), TableKindTag::Mul));
    }
    if sp.variant == Variant::Int8 {
        pre_ids.push(id(QNZ_COLS[0]));
        pre.push(eval_of(sp.q_bits, (0..1u32 << sp.q_bits).map(m31_u32).collect()));
        pre_ids.push(id(QNZ_COLS[1]));
        pre.push(eval_of(sp.q_bits,
                         (0..1u32 << sp.q_bits).map(|q| m31_u32(u32::from(q != 0))).collect()));
        order.push((sp.q_bits, QNZ_COLS.iter().map(|c| id(*c)).collect(), TableKindTag::Qnz));
    }
    for b in &widths {
        pre_ids.push(range_col(*b));
        pre.push(eval_of(*b, (0..1u32 << b).map(m31_u32).collect()));
        order.push((*b, vec![range_col(*b)], TableKindTag::Range(*b)));
    }
    // per-channel scales: offline, hence preprocessed, hence off the acc rung
    let chan0 = pre.len();
    if sp.variant == Variant::Pow2 {
        for s in 0..sp.per_row {
            pre_ids.push(chan_col("", "j", s));
            pre.push(eval_of(log_rows, vec![m31_of(w.jw[s % w.jw.len()] as i64); rows]));
        }
    }
    if sp.variant == Variant::Chain {
        for s in 0..sp.per_row {
            let (m, e) = w.sw[s % w.sw.len()];
            pre_ids.push(chan_col("", "m", s));
            pre.push(eval_of(log_rows, vec![m31_u32(m); rows]));
            pre_ids.push(chan_col("", "e", s));
            pre.push(eval_of(log_rows, vec![m31_u32(e); rows]));
        }
    }
    // the int8 numerator is per (head, channel) and offline, like a weight scale
    if sp.variant == Variant::Int8 {
        for s in 0..sp.per_row {
            pre_ids.push(chan_col("", "n", s));
            pre.push(eval_of(log_rows, vec![m31_of(w.num[s % w.num.len()]); rows]));
        }
    }

    // ---- tree 1: witness, then multiplicities ------------------------------
    let mut base: Vec<Eval> = cols.iter().map(|c| eval_of(log_rows, c.clone())).collect();
    let mult0 = base.len();
    if sp.variant.normalises() {
        base.push(eval_of(log_norm, mult_norm.iter().map(|m| m31_u32(*m)).collect()));
        base.push(eval_of(log_frac, mult_frac.iter().map(|m| m31_u32(*m)).collect()));
    }
    if sp.variant == Variant::Chain {
        base.push(eval_of(log_mul, mult_mul.iter().map(|m| m31_u32(*m)).collect()));
    }
    if sp.variant == Variant::Int8 {
        base.push(eval_of(sp.q_bits, mult_qnz.iter().map(|m| m31_u32(*m)).collect()));
    }
    for (t, b) in widths.iter().enumerate() {
        base.push(eval_of(*b, mult_r[t].iter().map(|m| m31_u32(*m)).collect()));
    }

    // Columns the AIR *derives* rather than commits, rebuilt so the LogUp
    // generator can read them; plus a constant-zero column for the absent
    // sticky field.  They live past the end of `base` and are never committed.
    let derived0 = base.len();
    let mut derived: Vec<Eval> = vec![eval_of(log_rows, vec![M31::zero(); rows])];
    let zero_col = derived0;
    for s in 0..sp.per_row {
        match sp.variant {
            Variant::Limbs => {
                let (b0, b1) = sp.limb_bits;
                let off = 1i64 << (b0 + b1 - 1);
                let inv = m31_u32(1 << b0).inverse();
                let _ = b1;
                derived.push(eval_of(log_rows, (0..rows)
                    .map(|row| (cols[map.at(s, f.acc)][row] + m31_of(off)
                                - cols[map.at(s, f.lo)][row]) * inv)
                    .collect()));
            }
            Variant::Int8 => {
                derived.push(eval_of(log_rows, (0..rows)
                    .map(|row| m31_of(2 * w.den[row % w.den.len()] - 1)
                         - cols[map.at(s, f.d)][row])
                    .collect()));
            }
            _ => derived.push(eval_of(log_rows, vec![M31::zero(); rows])),
        }
    }
    let deriv = |s: usize| derived0 + 1 + s;

    // ---- the lookups, in the order the AIR adds them ------------------------
    let mut looks: Vec<Look> = vec![];
    for s in 0..sp.per_row {
        match sp.variant {
            Variant::Limbs => {
                looks.push(Look::Range(ridx(sp.limb_bits.0), map.at(s, f.lo)));
                looks.push(Look::Range(ridx(sp.limb_bits.1), deriv(s)));
            }
            Variant::Int8 => {
                looks.push(Look::Qnz([map.at(s, f.qm), map.at(s, f.nz2)]));
                looks.push(Look::Range(ridx(sp.rem_bits), map.at(s, f.d)));
                looks.push(Look::Range(ridx(sp.rem_bits), deriv(s)));
            }
            _ => {
                // only the normalising variants carry `st`; asking `Map::at`
                // for it under `limbs` was the sentinel reaching the index
                // arithmetic (open item 12) -- unused there, but computed
                let st = if sticky { map.at(s, f.st) } else { zero_col };
                looks.push(Look::Norm([map.at(s, f.m2), map.at(s, f.ph), st,
                                       map.at(s, f.m7), map.at(s, f.ef), map.at(s, f.nz)]));
                looks.push(Look::Frac([map.at(s, f.ph), map.at(s, f.r), st]));
                match sp.variant {
                    Variant::Pow2 => looks.push(Look::Range(ridx(8), map.at(s, f.eo))),
                    Variant::Chain => {
                        for t in 0..2 {
                            let m_in = if t == 0 { map.at(s, f.m7) } else { map.at(s, f.mul[0][0]) };
                            let ms = if t == 0 {
                                map.shared[0]
                            } else {
                                // the second scale's mantissa is preprocessed
                                base.len() + derived.len() + chan0 + 2 * s
                            };
                            looks.push(Look::Mul(
                                [m_in, ms, map.at(s, f.mul[t][0]), map.at(s, f.mul[t][1])],
                                map.at(s, f.nz),
                            ));
                            looks.push(Look::Range(ridx(8), map.at(s, f.mul[t][2])));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // ---- commit, draw, interact --------------------------------------------
    let max_log = [log_rows, log_norm, log_frac,
                   if sp.variant == Variant::Chain { log_mul } else { 0 }]
        .into_iter()
        .chain(widths.iter().copied())
        .max()
        .unwrap();
    let tw = twiddles(max_log);
    let mut sess = Session::new(&tw);
    sess.commit(pre.clone());
    sess.commit(base.clone());
    let norm_rel = sess.draw_relation(NormRel::draw);
    let frac_rel = sess.draw_relation(FracRel::draw);
    let mul_rel = sess.draw_relation(MulRel::draw);
    let qnz_rel = sess.draw_relation(PairRel::draw);
    let rels: Vec<RangeRel> = widths.iter().map(|_| sess.draw_relation(RangeRel::draw)).collect();

    let t1 = std::time::Instant::now();
    // one flat address space for the generator: committed, then derived, then
    // preprocessed (which the Chain lookups read directly)
    let all: Vec<&Eval> = base.iter().chain(derived.iter()).chain(pre.iter()).collect();
    let n_vec_rows = 1usize << (log_rows - LOG_N_LANES);
    let one = PackedQM31::broadcast(SecureField::one());
    let frac_of = |l: &Look, vr: usize| -> (PackedQM31, PackedQM31) {
        match l {
            Look::Norm(c) => {
                let v: Vec<PackedM31> = c.iter().map(|i| all[*i].data[vr]).collect();
                (one, norm_rel.combine(&v))
            }
            Look::Frac(c) => {
                let v: Vec<PackedM31> = c.iter().map(|i| all[*i].data[vr]).collect();
                (one, frac_rel.combine(&v))
            }
            Look::Mul(c, m) => {
                let v: Vec<PackedM31> = c.iter().map(|i| all[*i].data[vr]).collect();
                (PackedQM31::from(all[*m].data[vr]), mul_rel.combine(&v))
            }
            Look::Qnz(c) => {
                let v: Vec<PackedM31> = c.iter().map(|i| all[*i].data[vr]).collect();
                (one, qnz_rel.combine(&v))
            }
            Look::Range(r, i) => (one, rels[*r].combine(&[all[*i].data[vr]])),
        }
    };
    let _ = n_vec_rows;
    let (n_pairs, fpair) = crate::plogup::paired(looks.len(), |i, vr| frac_of(&looks[i], vr));
    let (use_cols, use_sum) = crate::plogup::build(log_rows, n_pairs, fpair);

    // table sides, in the same order tree 0 built them
    let kinds: Vec<TableKind> = order
        .iter()
        .map(|(_, _, tag)| match tag {
            TableKindTag::Norm => TableKind::Norm(norm_rel.clone()),
            TableKindTag::Frac => TableKind::Frac(frac_rel.clone()),
            TableKindTag::Mul => TableKind::Mul(mul_rel.clone()),
            TableKindTag::Qnz => TableKind::Pair(qnz_rel.clone()),
            TableKindTag::Range(b) => TableKind::Range(rels[ridx(*b)].clone()),
        })
        .collect();
    let mut tcols: Vec<Eval> = vec![];
    let mut per_table: Vec<SecureField> = vec![];
    let mut pi = 0usize;
    for (t, (lr, tcolids, _)) in order.iter().enumerate() {
        let ncol = tcolids.len();
        let (c, s) = crate::plogup::build(*lr, 1, |_, vr| {
            let v: Vec<PackedM31> = (0..ncol).map(|k| pre[pi + k].data[vr]).collect();
            (-PackedQM31::from(base[mult0 + t].data[vr]), kinds[t].combine(&v))
        });
        tcols.extend(c);
        per_table.push(s);
        pi += ncol;
    }
    let table_sum = per_table.iter().fold(SecureField::zero(), |a, b| a + *b);
    let interaction_secs = t1.elapsed().as_secs_f64();
    let n_inter = use_cols.len() + tcols.len();
    sess.commit(use_cols.into_iter().chain(tcols).collect());

    // ---- components --------------------------------------------------------
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&pre_ids);
    let use_comp = FrameworkComponent::<DequantEval>::new(
        alloc,
        DequantEval {
            log_rows,
            sp: sp.clone(),
            tag: String::new(),
            norm: norm_rel.clone(),
            frac: frac_rel.clone(),
            mul: mul_rel.clone(),
            qnz: qnz_rel.clone(),
            rels: rels.clone(),
            yside: None, output_claim: None, sside: None,
            // the standalone dequant probe has no GKR backend to hand its lookups to
            gkr: false,
        },
        use_sum,
    );
    let tbl_comps: Vec<FrameworkComponent<TableEval>> = order
        .iter()
        .zip(&kinds)
        .zip(&per_table)
        .map(|(((lr, tcolids, _), kind), s)| {
            FrameworkComponent::<TableEval>::new(
                alloc,
                TableEval { log_rows: *lr, cols: tcolids.clone(), kind: kind.clone() },
                *s,
            )
        })
        .collect();
    let mut provers: Vec<&dyn ComponentProver<_>> = vec![&use_comp];
    let mut components: Vec<&dyn Component> = vec![&use_comp];
    for c in &tbl_comps {
        provers.push(c);
        components.push(c);
    }
    let n_cols = [pre.len(), base.len(), n_inter];
    let proven = sess.prove(&provers, &components);

    let table_cells_pre: usize = order.iter().map(|(lr, c, _)| c.len() << lr).sum();
    // one multiplicity column plus one QM31 interaction column per table
    let table_cells_fresh: usize = order.iter().map(|(lr, _, _)| 5usize << lr).sum();
    Outcome {
        proven,
        sums: Sums { uses: use_sum, tables: table_sum },
        sp: sp.clone(),
        n,
        log_rows,
        off_table,
        ref_ok,
        ref_n,
        first_bad,
        table_cells_pre,
        table_cells_fresh,
        table_rows: order.iter().map(|(lr, c, tag)| (tag.name(c.len()), *lr)).collect(),
        trace_secs,
        interaction_secs,
        n_columns: n_cols,
    }
}

/// Which table a row of `order` is, without carrying a relation around.
#[derive(Clone, Copy)]
enum TableKindTag {
    Norm,
    Frac,
    Mul,
    Qnz,
    Range(u32),
}

impl TableKindTag {
    fn name(&self, _n: usize) -> String {
        match self {
            TableKindTag::Norm => "norm".into(),
            TableKindTag::Frac => "frac".into(),
            TableKindTag::Mul => "mul".into(),
            TableKindTag::Qnz => "qnz".into(),
            TableKindTag::Range(b) => format!("range{b}"),
        }
    }
}

/// Cross-check the tables against f32, outside any proof.
///
/// B4 §3's discipline: a corrupted table is invisible to everything except its
/// preprocessed root, so the *contents* have to be checked independently.  The
/// tables here are built from integer arithmetic; f32 rounds by a different route,
/// so agreement on every magnitude tested is the check that the table is the op.
pub fn check_tables(kmax: u32, round: Round, samples: usize) -> Value {
    let norm = bf16::norm_table(kmax, round);
    let cap = 1u64 << (kmax + 9);
    let mut mags: Vec<u64> = (0..(1u64 << 14).min(cap)).collect();
    let mut v = 1u64 << 14;
    while v < cap && mags.len() < samples {
        for d in [0u64, 1, 2] {
            mags.push((v + d).min(cap - 1));
        }
        mags.push((v * 3 / 2).min(cap - 1));
        mags.push((v * 2 - 1).min(cap - 1));
        v = (v as f64 * 1.01) as u64 + 1;
    }
    let (mut tested, mut bad) = (0usize, vec![]);
    for m in mags {
        if m >= cap {
            continue;
        }
        tested += 1;
        match bf16::norm_lookup(&norm, m, kmax, 0, round) {
            // f32 is exact on integers below 2^24 and rounds to nearest-even above,
            // which is the same function the table encodes -- for Round::Even.
            Some(g) if round == Round::Even => {
                let want = bf16::bf16_of_u64(m);
                if g != want && bad.len() < 8 {
                    bad.push(json!({"mag": m, "f32": want, "table": g}));
                }
            }
            Some(_) => {}
            None => {
                if bad.len() < 8 {
                    bad.push(json!({"mag": m, "error": "no table row"}));
                }
            }
        }
    }
    let mul = bf16::mul_table(round);
    let (mut mtested, mut mbad) = (0usize, vec![]);
    for a in 0u32..128 {
        for b in 0u32..128 {
            mtested += 1;
            let r = mul[(a * 128 + b) as usize];
            let want = bf16::bf16_mul_ref(bf16::pattern(0, 127, a), bf16::pattern(0, 127, b));
            let (_, we, wm) = bf16::decode(want);
            if round == Round::Even && (wm != r.m7 || we != 127 + r.carry) && mbad.len() < 8 {
                mbad.push(json!({"a": a, "b": b, "f32": [we, wm],
                                 "table": [127 + r.carry, r.m7]}));
            }
        }
    }
    json!({"round": round.name(), "kmax": kmax,
           "norm_rows": norm.len(), "norm_log_rows": log_rows_for(norm.len(), 1),
           "norm_tested": tested, "norm_mismatch": bad,
           "mul_rows": mul.len(), "mul_tested": mtested, "mul_mismatch": mbad,
           "frac_rows": (1u64 << (kmax + 1)) as usize,
           "kmax_sound_limit": KMAX_SOUND, "exp_bias": EXP_BIAS,
           "max_abs_acc": 1u64 << (kmax + 9),
           "max_k_int8": (1u64 << (kmax + 9)) / (127 * 127)})
}

/// The ladder, in the order the README's table reads.
pub const LADDER: [&str; 5] = ["limbs", "round", "pow2", "chain", "int8"];
