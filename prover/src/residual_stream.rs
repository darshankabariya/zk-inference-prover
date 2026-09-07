//! B10 step 2 — the residual stream between the composed layer's components.
//!
//! B6.5's composition bindings every matmul's accumulators to its own committed
//! activations and weights, and nothing at all bindings one component's output to the
//! next component's input (open gap 1).  The wiring is four gadgets, and every one
//! of them is bf16 elementwise arithmetic:
//!
//! ```text
//!   (a) RMSNorm   ms = bf16(sum_i bf16(x_i^2) / d)   r = rsqrt(ms + eps)
//!                 h_i = bf16(bf16(x_i r) w_i)
//!   (b) SwiGLU    y_i = bf16(silu(g_i) u_i)
//!   (c) residual  x' = bf16(x + o),  x'' = bf16(x' + down)
//!   (d) the requantize's placement -- B6.5's gadget between a dequant's output
//!       and a Freivalds check's input; no new cell, only wiring
//! ```
//!
//! Three primitives cover (a)-(c): a bf16 **unary** table (B4: a unary bf16 op
//! *is* a 2^16-entry table), a bf16 **multiply**, and a bf16 **add** — the gadget
//! B4 §9 scoped, costed at "12-15 cells" and deliberately did not build, because
//! at the time its customer was the softmax denominator over C = 16K keys.  The
//! norm's reduction is over `d`, twice a layer, which is three orders of
//! magnitude smaller, so here it is affordable and here it is built.
//!
//! # The value representation
//!
//! A bf16 value lives here as **two** field columns rather than as a pattern:
//!
//! * `sm` — the *signed* significand: `0` for zero, else `±(128 + m7)`, i.e. in
//!   `{0} ∪ ±[128, 256)`;
//! * `ef` — the biased exponent field, `0` for zero, else `[1, 255)`.
//!
//! so the value is exactly `sm · 2^(ef − 134)` and "is zero" is `sm = 0` with no
//! side condition.  That choice is what keeps every constraint **degree 2**,
//! which the security config requires (blowup 2 ⇒ composition degree 2): the
//! signed significand multiplies the alignment power directly, where a
//! `(sign, mantissa, nz)` triple would need three committed factors.
//!
//! # The add
//!
//! ```text
//!   u_a = ef_a − etop + DMAX  →  POW  →  pa      (and the same for b)
//!   (pa − 2^DMAX)(pb − 2^DMAX) = 0                 so etop is the larger exponent
//!   acc = sm_a pa + sm_b pb = mag (1 − 2 sg)       the exact sum, scaled
//!   mag = M2 ph + r,  NORM(M2, ph, st) → (m7, ef, nz),  FRAC(ph, r, st)
//!   sm_c = (128 nz + m7)(1 − 2 sg)   ef_c = nz (ef_n + etop − DMAX − 134)
//! ```
//!
//! Three things make it work, and they are the whole soundness argument:
//!
//! * **The anchor is pinned from below, not from above.**  `POW` carries
//!   `(u, 2^u)` for `u ∈ [0, DMAX]` and `(u, 0)` for `u < 0`, so an operand
//!   *more than* `DMAX` binades below the anchor drops out — and the anchor
//!   constraint forces some operand to sit exactly at it, so the operand that
//!   drops is always the smaller one.  Anchoring at the *minimum* instead would
//!   let a prover drop the larger, which is what `drop-big` probes.
//! * **Dropping is exact for the rounding.**  `u < 0` means
//!   `|lo| < 2^(8−DMAX) ulp(hi)`, and round-to-nearest returns `hi` whenever
//!   `|lo| < ulp(hi)/4`, so `DMAX ≥ 10` makes "drop" and "add" the same bf16.
//!   `DMAX = 12` is the default: it also bounds `|acc| < 2·255·2^12 < 2^21`, so
//!   nothing wraps mod p and the shared `NORM` table needs only `kmax ≥ 13`.
//! * **The normalisation is B6's, unchanged.**  `mag` is a non-negative integer
//!   below `2^(kmax+9)`, which is exactly the domain B6's `NORM`/`FRAC` pair
//!   already covers — so the add rides the tables the dequant components commit
//!   anyway and costs no table row of its own.
//!
//! # The reduction's order is a spec decision
//!
//! B1 accumulates the RMSNorm mean in fp32 and rounds once (`reduce="fp32"`),
//! an order no bf16-add gadget reproduces.  Proving the norm therefore *pins*
//! the order, and B1 already carries the one this AIR implements: `tree_sum`
//! (`reduce="bf16-norm"`), a balanced binary tree padded to a power of two.
//! A component here holds a whole balanced tree of `leaves` values **inside one
//! row**, so the wiring between its adds is fixed column indices and costs
//! nothing; chaining two such components (`leaves`, then `d / leaves`) is the whole
//! reduction, and with both widths powers of two it is bit-identical to
//! `tree_sum`.  Two components are bound by a **Horner claim** over the values they
//! share — B6.5's `h_m == acc_m` trick, at ~0.5 cells per value — so a value is
//! never committed twice.
//!
//! # What bindings two components positionwise
//!
//! A binding's weights are the verifier's own powers of a drawn challenge, chosen so
//! that the value at **canonical stream position** `q` weighs `w^(N-1-q)` in
//! every component that carries it.  The canonical order of a `(tokens x d)`
//! activation stream is channel-major, `q = c*tokens + t`, because that is the
//! order in which `tree_sum`'s own tiling (`Tiling` below) and the residual
//! add's read it with plain geometric weights.  So `resid.h_out == norm.h_in`
//! says "the same values in the same positions", however differently the two
//! components cut the stream into rows, and the layer's statement becomes
//! *committed residual stream in -> committed residual stream out*.  A slot's
//! output also pays a `VAL` lookup — the output range check README 3.7 named:
//! it says the result is a value bf16 can represent.

use num_traits::{One, Zero};
use rayon::prelude::*;
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{M31, P};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::ComponentProver;
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, INTERACTION_TRACE_IDX, Relation, RelationEntry,
    TraceLocationAllocator, relation,
};

use crate::bf16::{self, EXP_BIAS, Round};
use crate::common::{Eval, Proven, Session, Sums, eval_of, log_rows_for, m31_u32, twiddles};
use crate::dequant::{FracRel, NormRel, id};

relation!(PowRel, 2);
relation!(SMulRel, 5);
relation!(UnRel, 4);
relation!(ValRel, 2);

// B11 stream V, lever 3: the drawn elements, for the GKR link's affine weights.
// `combine(v) = sum alpha_i v_i - z`, so these two are everything the link needs
// to write a lookup's denominator as an affine form over the columns it reads --
// the same accessors stream S added to `ExpRel` and stream U to `NormRel`.
impl PowRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl SMulRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl UnRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl ValRel {
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}

pub const POW_COLS: [&str; 2] = ["b10_pow_u", "b10_pow_p"];
pub const SMUL_COLS: [&str; 5] =
    ["b10_smul_a", "b10_smul_b", "b10_smul_c", "b10_smul_cy", "b10_smul_nz"];
pub const VAL_COLS: [&str; 2] = ["b10_val_sm", "b10_val_ef"];
pub const UN_COLS: [&str; 4] = ["b10_un_i_sm", "b10_un_i_ef", "b10_un_o_sm", "b10_un_o_ef"];
/// A component's own selector: `end` marks the last row, for the Horner bindings.
/// Stream II: a component's `i`-th preprocessed value, as its two fields.  The
/// norm weights are the only user today; they are the model's, so they belong
/// beside the tables in tree 0 rather than beside the stream in tree 1.
pub fn residual_stream_pre_col(tag: &str, i: usize, k: usize) -> PreProcessedColumnId {
    PreProcessedColumnId { id: format!("b10_gpre_{tag}_{i}_{}", ["sm", "ef"][k]) }
}

pub fn residual_stream_sel(tag: &str, name: &str) -> PreProcessedColumnId {
    id(format!("b10_g{tag}_{name}"))
}
pub fn un_col(kind: &str, i: usize) -> PreProcessedColumnId {
    id(format!("{}_{kind}", UN_COLS[i]))
}

/// The alignment window.  `>= 10` makes "drop the far operand" and "add it" the
/// same bf16; `12` also keeps `|acc| < 2^21`, so the shared `NORM` table needs
/// only `kmax >= 13` and nothing wraps mod p.
pub const DMAX: u32 = 12;
/// How far below the anchor `POW` still carries a (dropping) row.
pub const UMIN: i64 = 260;

pub fn m31_of(v: i64) -> M31 {
    crate::attn_trace::m31_of(v)
}

// ------------------------------------------------------------------ values ----

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

/// A signed significand held as a field element, back to `i64`.
pub fn sgn_of(v: M31) -> i64 {
    if v.0 > P / 2 { v.0 as i64 - P as i64 } else { v.0 as i64 }
}

// ------------------------------------------------------------------ tables ----

/// `(u, 2^u)` for `u in [0, DMAX]` and `(u, 0)` for `u in [-UMIN, -1]`, the
/// second family written as the field elements `p + u`.
pub fn pow_table() -> Vec<(M31, M31)> {
    let mut rows: Vec<(M31, M31)> = (0..=DMAX).map(|u| (m31_u32(u), m31_u32(1 << u))).collect();
    rows.extend((1..=UMIN).map(|u| (m31_of(-u), M31::zero())));
    rows
}

/// Every legal `(sm, ef)`: the zero value, and `±(128 + m7)` at every finite
/// nonzero exponent field.
pub fn val_table() -> Vec<(M31, M31)> {
    let mut rows = vec![(M31::zero(), M31::zero())];
    for ef in 1u32..0xff {
        for m7 in 0u32..128 {
            rows.push((m31_of(128 + m7 as i64), m31_u32(ef)));
            rows.push((m31_of(-(128 + m7 as i64)), m31_u32(ef)));
        }
    }
    rows
}

/// `(sm_a, sm_b) -> (sm_c, carry, nz)`: the correctly-rounded product of two
/// *signed* significands, over every legal pair.  B6's `mul_table` is the
/// unsigned core; carrying the sign in the key is what keeps the multiply's
/// constraints degree 2 in the `(sm, ef)` representation, and it costs
/// 257^2 rows against 2^14 rather than a committed sign column per operand.
pub fn smul_vals() -> Vec<i64> {
    std::iter::once(0)
        .chain((128..256).map(|m| m as i64))
        .chain((128..256).map(|m| -(m as i64)))
        .collect()
}

pub fn smul_table(round: Round) -> Vec<[M31; 5]> {
    let vals = smul_vals();
    let mut rows = Vec::with_capacity(vals.len() * vals.len());
    for a in &vals {
        for b in &vals {
            let (sm, cy) = if *a == 0 || *b == 0 {
                (0i64, 0u32)
            } else {
                let p = (a.unsigned_abs() * b.unsigned_abs()) as u32; // [2^14, 2^16)
                let carry = (p >> 15) & 1;
                let shift = 7 + carry;
                let m = p >> shift;
                let rb = (p >> (shift - 1)) & 1;
                let st = u32::from(p & ((1 << (shift - 1)) - 1) != 0);
                let up = round.carry(m, rb, st);
                let (msig, extra) = if m + up == 256 { (128, 1) } else { (m + up, 0) };
                let sgn = if (*a < 0) != (*b < 0) { -1 } else { 1 };
                (sgn * msig as i64, carry + extra)
            };
            rows.push([m31_of(*a), m31_of(*b), m31_of(sm), m31_u32(cy),
                       m31_u32(u32::from(sm != 0))]);
        }
    }
    rows
}

/// Which unary bf16 op a component's leaves apply.  `None` commits the leaf value
/// itself and validates it against `VAL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnKind {
    None,
    Square,
    Rsqrt,
    Silu,
    /// Stream II: the norm's own `r`, as one table from the reduction's answer.
    /// The chain B1 runs is `ms = store(root * inv_d)`, `t = store(ms + eps)`,
    /// `r = store(rsqrt(t))` (`b1/model.py`'s `rmsnorm`), and `inv_d` and `eps`
    /// are bf16 constants of the model, so the whole of it is a function of one
    /// bf16 pattern and belongs in a table the preprocessed root stands behind.
    /// A lookup on `(root, r)` is then what makes `r` the reduction's answer
    /// rather than a value the prover chose.
    NormR { inv_d: u16, eps: u16 },
}

impl UnKind {
    pub fn name(&self) -> &'static str {
        match self {
            UnKind::None => "none",
            UnKind::Square => "square",
            UnKind::Rsqrt => "rsqrt",
            UnKind::Silu => "silu",
            UnKind::NormR { .. } => "normr",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "none" => UnKind::None,
            "square" => UnKind::Square,
            "rsqrt" => UnKind::Rsqrt,
            "silu" => UnKind::Silu,
            // `normr` carries the model's two bf16 scalars, so it has no
            // command-line spelling: the plan builds it from the manifest.
            "normr" => panic!("normr is built from the manifest, not named on a command line"),
            other => panic!("unknown unary {other:?}"),
        }
    }
    /// B1's own op: `store(fn(x))`, evaluated in f32 and rounded once, which is
    /// what `model.py`'s `unary` does.
    pub fn apply(&self, x: f64) -> f64 {
        let x = x as f32;
        (match self {
            UnKind::None => x,
            UnKind::Square => x * x,
            UnKind::Rsqrt => {
                if x <= 0.0 { 0.0 } else { x.sqrt().recip() }
            }
            UnKind::Silu => x / (1.0 + (-x).exp()),
            // one `store` per step, exactly as `rmsnorm` writes it
            UnKind::NormR { inv_d, eps } => {
                let ms = bf16::to_f32(bf16::from_f32(x * bf16::to_f32(*inv_d)));
                let t = bf16::to_f32(bf16::from_f32(ms + bf16::to_f32(*eps)));
                if t <= 0.0 { 0.0 } else { t.sqrt().recip() }
            }
        }) as f64
    }
}

/// The unary's image of one bf16 pattern, exactly the row `un_table` tabulates
/// for it: one f32 evaluation and one rounding, which is B1's own
/// `store(fn(x))`.  Exposed so a caller can compare a component's unary against
/// B1's column without rebuilding the table.
pub fn un_image(k: UnKind, pat: u16) -> u16 {
    let (sm, ef) = split(pat);
    un_image_parts(k, sm, ef)
}

/// The same, for a value already split into its `(sm, ef)` pair.
pub fn un_image_parts(k: UnKind, sm: i64, ef: u32) -> u16 {
    bf16::from_f32(k.apply(to_f64(sm, ef)) as f32)
}

/// `(sm_in, ef_in) -> (sm_out, ef_out)` over every legal input value.
pub fn un_table(k: UnKind) -> Vec<[M31; 4]> {
    val_table()
        .into_iter()
        .map(|(sm, ef)| {
            let y = bf16::from_f32(k.apply(to_f64(sgn_of(sm), ef.0)) as f32);
            let (osm, oef) = split(y);
            [sm, ef, m31_of(osm), m31_u32(oef)]
        })
        .collect()
}

// --------------------------------------------------------------- the layout ---

/// Where a slot's operand comes from.  Every one is a fixed column index of the
/// same row, so a whole tree inside a row needs no wiring argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// the constant `+0`.  `x + 0` is exact in bf16, so a tree position the
    /// reduction pads with zero costs neither a column nor a slot: the slot that
    /// would add it is *pruned* and its parent reads the other operand directly.
    Zero,
    /// group `g`'s committed leaf `i`, as committed
    Leaf(usize, usize),
    /// the same leaf's unary image
    Un(usize, usize),
    /// an earlier slot's output in the same row
    Slot(usize),
    /// Stream II: preprocessed value `i` of this row -- a `(sm, ef)` pair the
    /// *statement* carries, not the prover.  The norm weights are the model's,
    /// so they live in tree 0 under `pre_root` and no `VAL` lookup is needed:
    /// the table is trusted by the same anchor every other table is.
    Pre(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Add,
    Mul,
}

/// One committed input group of a component: `n` bf16 values per row, all under the
/// same unary (or none).  A component has as many groups as it has distinct input
/// streams -- the norm-and-residual component has two, the residual stream and what
/// is added to it -- and each group carries its own Horner binding, so two components
/// can be compared positionwise on a stream without committing it twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub n: usize,
    pub un: UnKind,
}

/// One component's program: what every row computes.
#[derive(Clone, Debug)]
pub struct Prog {
    pub groups: Vec<Group>,
    pub kind: Kind,
    /// `(a, b)` per slot, in evaluation order
    pub slots: Vec<(Src, Src)>,
    /// the slot outputs a binding may read, as named groups -- the tree's root, the
    /// residual sums, ...
    pub outs: Vec<Vec<usize>>,
    /// Stream II: how many preprocessed `(sm, ef)` values a row reads, i.e. how
    /// many `Src::Pre` slots the program has.  Zero on every program that
    /// existed before.
    pub n_pre: usize,
}

impl Prog {
    fn src(groups: &[Group], g: usize, i: usize) -> Src {
        if groups[g].un == UnKind::None { Src::Leaf(g, i) } else { Src::Un(g, i) }
    }

    /// `tree_sum`'s own order over `pad` positions of which the first `n` are
    /// group 0's committed leaves and the rest are `+0`: `x[..h] + x[h..]`,
    /// repeatedly.  The zero positions are pruned, so the component proves exactly
    /// `n - 1` adds however far `pad` reaches past `n`.
    pub fn tree(n: usize, pad: usize, un: UnKind) -> Self {
        assert!(pad.is_power_of_two() && pad >= 2, "a tree pads to a power of two");
        assert!(n >= 2 && n <= pad, "a tree of {n} leaves does not fit {pad} positions");
        let groups = vec![Group { n, un }];
        let mut cur: Vec<Src> = (0..pad)
            .map(|i| if i < n { Self::src(&groups, 0, i) } else { Src::Zero })
            .collect();
        let mut slots = vec![];
        while cur.len() > 1 {
            let h = cur.len() / 2;
            let mut next = vec![];
            for i in 0..h {
                // `x + 0` is exact, so the pad never costs a slot
                next.push(match (cur[i], cur[i + h]) {
                    (Src::Zero, y) => y,
                    (x, Src::Zero) => x,
                    (x, y) => {
                        slots.push((x, y));
                        Src::Slot(slots.len() - 1)
                    }
                });
            }
            cur = next;
        }
        let root = match cur[0] {
            Src::Slot(k) => k,
            _ => unreachable!("a tree of at least two leaves ends in a slot"),
        };
        Self { groups, kind: Kind::Add, slots, outs: vec![vec![root]], n_pre: 0 }
    }

    /// `n` independent binary ops on two committed groups of `n` -- the residual
    /// adds, the SwiGLU product, the norm's per-channel multiplies.  The row is
    /// laid out as `[all a, then all b]`, which is what lets a producer's output
    /// binding be compared with the consumer's group-`a` binding positionwise.
    pub fn pairs(n: usize, kind: Kind, un_a: UnKind, un_b: UnKind) -> Self {
        let groups = vec![Group { n, un: un_a }, Group { n, un: un_b }];
        let slots = (0..n)
            .map(|i| (Self::src(&groups, 0, i), Self::src(&groups, 1, i)))
            .collect();
        Self { groups, kind, slots, outs: vec![(0..n).collect()], n_pre: 0 }
    }

    /// The composed layer's wiring component: one row is one `(residue, token)`
    /// block of the residual stream, and it proves both the RMSNorm reduction's
    /// first `log2(pad)` levels over the squares *and* the residual add against
    /// the block's second operand.  The stream is committed **once** and read by
    /// both, so the two gadgets need no binding between them at all.
    pub fn norm_resid(n: usize, pad: usize) -> Self {
        let mut p = Self::tree(n, pad, UnKind::Square);
        p.groups.push(Group { n, un: UnKind::None });
        let base = p.slots.len();
        for i in 0..n {
            p.slots.push((Src::Leaf(0, i), Src::Leaf(1, i)));
        }
        p.outs.push((base..base + n).collect());
        p
    }

    /// The RMSNorm scale, `h = (x * r) * w`, one row per `(residue, token)` block
    /// of the stream -- the same tiling the reduction's first level uses, so the
    /// norm scale's group 0 and the reduction's group 0 are the same values in the
    /// same order and one Horner claim compares them.
    ///
    /// Group 1 is the row's single `r`: "broadcast down a row" is literally one
    /// committed leaf that every slot of the row reads, which costs two cells
    /// per row instead of two per value.  Group 2 is the norm weight's block,
    /// committed per row because a row's channels differ (it is constant down
    /// the tokens, which a preprocessed column could exploit -- priced in the
    /// record, not built).
    ///
    /// The two multiplies chain through `Src::Slot`, so `x * r` never leaves the
    /// row and only `h` is bound onwards.
    /// The RMSNorm scale, `h = (x * r) * w`, one row per `(residue, token)` block
    /// of the stream -- the same tiling the reduction's first level uses, so the
    /// norm scale's group 0 and the reduction's group 0 are the same values in the
    /// same order and one Horner claim compares them.
    ///
    /// Stream II changed both of the other two operands.  Group 1 is no longer
    /// `r` itself but the reduction's own answer, the sum of squares, under
    /// `UnKind::NormR`: the unary lookup *is* `r = rsqrt(ms inv_d + eps)`, and
    /// the `In(1)` binding against `gn1b`/`gn2b`'s `Out(0)` is what says the value
    /// squared is the value the reduction summed.  One committed value per row
    /// still, and because the binding compares every row of a token against the
    /// same reduction row, the `a` rows of one token cannot carry different
    /// `r`s.  The norm weight is `Src::Pre`: preprocessed, under `pre_root`,
    /// because it is the model's and not the prover's.
    ///
    /// The two multiplies chain through `Src::Slot`, so `x * r` never leaves the
    /// row and only `h` is bound onwards.
    pub fn norm_scale(n: usize, un_r: UnKind) -> Self {
        let groups = vec![Group { n, un: UnKind::None },
                          Group { n: 1, un: un_r }];
        let mut slots: Vec<(Src, Src)> = (0..n)
            .map(|i| (Src::Leaf(0, i), Self::src(&groups, 1, 0)))
            .collect();
        for i in 0..n {
            slots.push((Src::Slot(i), Src::Pre(i)));
        }
        Self { groups, kind: Kind::Mul, slots, outs: vec![(n..2 * n).collect()], n_pre: n }
    }

    /// Stream NN: the products of B1's `rotate`, one row per (token, head).
    ///
    /// `rotate(t, cos, sin)` is `store(store(t cos) + store(r sin))` with
    /// `r = cat([-t[h:], t[:h]])`, so per output channel `j` there are three
    /// bf16 roundings in this order: `t[j] cos[j]`, `t[(j + h) mod hd] sin'[j]`,
    /// then their sum.  A `Prog` carries one `Kind`, so the two multiplies live
    /// here and the add is `Prog::rope_add`; one `eqs` entry joins them.
    ///
    /// The sign of `r` is folded into the table (`rope::sgn_sin`), so the
    /// program is a pure permutation of committed leaves against preprocessed
    /// operands and needs no negation node.  `cos[j]` is preprocessed value
    /// `2 j` and `sin'[j]` is `2 j + 1`.
    pub fn rope_mul(hd: usize) -> Self {
        assert!(hd >= 2 && hd % 2 == 0, "a rotated head width is even");
        let h = hd / 2;
        let groups = vec![Group { n: hd, un: UnKind::None }];
        let mut slots: Vec<(Src, Src)> = Vec::with_capacity(2 * hd);
        for j in 0..hd {
            slots.push((Src::Leaf(0, j), Src::Pre(2 * j)));
            slots.push((Src::Leaf(0, (j + h) % hd), Src::Pre(2 * j + 1)));
        }
        Self { groups, kind: Kind::Mul, slots, outs: vec![(0..2 * hd).collect()], n_pre: 2 * hd }
    }

    /// Stream NN: the third rounding.  Its leaves are `rope_mul`'s `2 hd`
    /// outputs in the same row order, so the join is one `ResidualStreamEq` between
    /// `rope_mul.Out(0)` and this component's `In(0)`, and its own `Out(0)` is the
    /// rotated value the quantize consumes.
    pub fn rope_add(hd: usize) -> Self {
        assert!(hd >= 2 && hd % 2 == 0, "a rotated head width is even");
        let groups = vec![Group { n: 2 * hd, un: UnKind::None }];
        let slots: Vec<(Src, Src)> =
            (0..hd).map(|j| (Src::Leaf(0, 2 * j), Src::Leaf(0, 2 * j + 1))).collect();
        Self { groups, kind: Kind::Add, slots, outs: vec![(0..hd).collect()], n_pre: 0 }
    }

    pub fn leaf_width(&self, g: usize) -> usize {
        if self.groups[g].un == UnKind::None { 2 } else { 4 }
    }
    pub fn n_leaves(&self) -> usize {
        self.groups.iter().map(|g| g.n).sum()
    }
    pub fn group_base(&self, g: usize) -> usize {
        (0..g).map(|k| self.groups[k].n * self.leaf_width(k)).sum()
    }
    pub fn leaf_at(&self, g: usize, i: usize) -> usize {
        self.group_base(g) + i * self.leaf_width(g)
    }
    pub fn leaves_width(&self) -> usize {
        self.group_base(self.groups.len())
    }
    pub fn slot_width(&self, round: Round) -> usize {
        match self.kind {
            // pa, pb, etop, mag, sg, m2, ph, r, [st], m7n, efn, nzn, c_sm, c_ef
            Kind::Add => 13 + usize::from(round.needs_sticky()),
            // c_sm, cy, nzc, c_ef
            Kind::Mul => 4,
        }
    }
    pub fn slot_at(&self, s: usize, round: Round) -> usize {
        self.leaves_width() + s * self.slot_width(round)
    }
    pub fn n_cols(&self, round: Round) -> usize {
        self.leaves_width() + self.slots.len() * self.slot_width(round)
    }
    /// Column offsets of the `(sm, ef)` a slot output holds.
    pub fn out_cols(&self, s: usize, round: Round) -> (usize, usize) {
        let b = self.slot_at(s, round);
        match self.kind {
            Kind::Add => {
                let k = 11 + usize::from(round.needs_sticky());
                (b + k, b + k + 1)
            }
            Kind::Mul => (b, b + 3),
        }
    }
    /// Stream II: where the component's `i`-th preprocessed value sits inside
    /// `ResidualStreamComponent::pre`.  The two selectors come first, as `residual_stream_component` and
    /// `Plan::pre_ids` both write them.
    pub fn pre_at(&self, i: usize, k: usize) -> usize {
        2 + 2 * i + k
    }

    /// Where a source's `(sm, ef)` columns are; `None` for the constant zero.
    pub fn src_cols(&self, s: &Src, round: Round) -> Option<(usize, usize)> {
        match s {
            Src::Zero => None,
            Src::Leaf(g, i) => {
                let b = self.leaf_at(*g, *i);
                Some((b, b + 1))
            }
            Src::Un(g, i) => {
                let b = self.leaf_at(*g, *i);
                Some((b + 2, b + 3))
            }
            Src::Slot(k) => Some(self.out_cols(*k, round)),
            Src::Pre(_) => None,
        }
    }
    /// The unary kinds this program's groups draw, deduplicated, in group order.
    pub fn un_kinds(&self) -> Vec<UnKind> {
        let mut v = vec![];
        for g in &self.groups {
            if g.un != UnKind::None && !v.contains(&g.un) {
                v.push(g.un);
            }
        }
        v
    }
}

// ------------------------------------------------------------------ the AIR ---

/// Which of a component's value sequences a binding accumulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// group `g`'s committed leaves, as committed (never the unary image)
    In(usize),
    /// the slots named by `outs[i]`
    Out(usize),
}

/// One Horner claim over one of a component's value sequences:
/// `H = sum_p w_p (sm_p + lam ef_p)`, accumulated a row at a time.  The weights
/// are the verifier's own powers of a drawn challenge, chosen so that a value at
/// canonical stream position `q` weighs `w^(N-1-q)` in **every** component that
/// carries it -- which is what makes `resid.h_out == norm.h_in` mean "the same
/// values, in the same positions", however differently the two components tile the
/// stream into rows.
#[derive(Clone, Debug)]
pub struct TieSpec {
    pub kind: BindingKind,
    /// one weight per member of the sequence within a row
    pub w: Vec<SecureField>,
    /// the row-to-row step
    pub step: SecureField,
    pub claimed: SecureField,
}

#[derive(Clone)]
pub struct Binding {
    pub lam: SecureField,
    pub specs: Vec<TieSpec>,
}

#[derive(Clone)]
pub struct ResidualStreamEval {
    pub log_rows: u32,
    pub tag: String,
    pub prog: Prog,
    pub round: Round,
    pub norm: NormRel,
    pub frac: FracRel,
    pub pow: PowRel,
    pub smul: SMulRel,
    /// one per entry of `uns`
    pub un: Vec<UnRel>,
    /// the kinds `un` is indexed by -- the plan's union, see `ResidualStreamRels::uns`
    pub uns: Vec<UnKind>,
    pub val: ValRel,
    pub binding: Option<Binding>,
    /// B11 stream V, lever 3: this component's lookups are proved by the batched
    /// GKR instead of committed running-sum columns, so the AIR emits neither
    /// the relation entries nor the interaction masks they would allocate. The
    /// trace masks stay exactly where they were -- stream S's lesson is that
    /// the allocator must keep the committed order, and here the order is
    /// `Prog`'s, which the collector reads the same way.
    pub gkr: bool,
    /// Stream H: when this component carries one side of the layer's boundary, the
    /// tree that side is committed in and which of the component's value sequences
    /// it holds.  `None` on every component of a chunk proved with the boundary
    /// flag off, which is why that chunk's proof is byte-for-byte the old one.
    pub bound: Option<crate::boundary::Boundary>,
}


impl FrameworkEval for ResidualStreamEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`); `None` when this
    /// component's fingerprint has no kernel, which is the fork's
    /// scalar CPU route.
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_residual_stream_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sticky = self.round.needs_sticky();
        let two = m31_u32(2);
        let pmax = m31_u32(1 << DMAX);
        let anchor = m31_u32(DMAX + EXP_BIAS);
        let c128 = m31_u32(128);
        let c127 = m31_u32(127);
        let dmax = m31_u32(DMAX);

        // ---- tree 0, in the component's own column order (stream OO) -------------
        // The order these are *read* in is the order the lowered IR indexes
        // tree 0 in, and `devfrac::frac_cols` is handed the component's `pre`
        // vector by position -- so the read order has to be `residual_stream_component`'s
        // build order and `Plan::pre_ids`' registration order, which are both
        // the two selectors and then the `Src::Pre` pairs. Stream II read the
        // pairs first, which made no difference while `n_pre` was zero
        // everywhere and made the generated fraction builder read the `nofirst`
        // selector as a norm-weight mantissa on every chunk with a norm scale.
        // The selectors are read whether or not this component carries a binding: the
        // component commits them either way, so a conditional read would leave the
        // supplied column list two longer than the IR's.
        let nofirst = eval.get_preprocessed_column(residual_stream_sel(&self.tag, "nofirst"));
        let end = eval.get_preprocessed_column(residual_stream_sel(&self.tag, "end"));
        // the statement's own values (stream II): preprocessed, so nothing pins
        // them but the root every table hangs on.
        let pre_vals: Vec<(E::F, E::F)> = (0..self.prog.n_pre)
            .map(|i| (eval.get_preprocessed_column(residual_stream_pre_col(&self.tag, i, 0)),
                      eval.get_preprocessed_column(residual_stream_pre_col(&self.tag, i, 1))))
            .collect();

        // ---- leaves: the committed inputs, and the unary that consumes them ---
        let kinds = &self.uns;
        let mut leaf_in: Vec<Vec<(E::F, E::F)>> = vec![];
        let mut leaf_un: Vec<Vec<(E::F, E::F)>> = vec![];
        for g in &self.prog.groups {
            let (mut ins, mut uns) = (vec![], vec![]);
            for _ in 0..g.n {
                let i_sm = eval.next_trace_mask();
                let i_ef = eval.next_trace_mask();
                ins.push((i_sm.clone(), i_ef.clone()));
                if g.un == UnKind::None {
                    // nothing else pins a committed pair to a legal bf16 value
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.val, E::EF::one(), &[i_sm, i_ef]));
                    }
                    uns.push((E::F::zero(), E::F::zero()));
                } else {
                    let o_sm = eval.next_trace_mask();
                    let o_ef = eval.next_trace_mask();
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.un[kinds.iter().position(|x| *x == g.un).unwrap()],
                            E::EF::one(),
                            &[i_sm, i_ef, o_sm.clone(), o_ef.clone()]));
                    }
                    uns.push((o_sm, o_ef));
                }
            }
            leaf_in.push(ins);
            leaf_un.push(uns);
        }

        // ---- slots ------------------------------------------------------------
        let mut outs: Vec<(E::F, E::F)> = vec![];
        let zero_pair = (E::F::zero(), E::F::zero());
        for (sa, sb) in &self.prog.slots {
            let get = |s: &Src, outs: &Vec<(E::F, E::F)>| -> (E::F, E::F) {
                match s {
                    Src::Zero => zero_pair.clone(),
                    Src::Leaf(g, i) => leaf_in[*g][*i].clone(),
                    Src::Un(g, i) => leaf_un[*g][*i].clone(),
                    Src::Slot(k) => outs[*k].clone(),
                    Src::Pre(i) => pre_vals[*i].clone(),
                }
            };
            let (a_sm, a_ef) = get(sa, &outs);
            let (b_sm, b_ef) = get(sb, &outs);
            match self.prog.kind {
                Kind::Add => {
                    let pa = eval.next_trace_mask();
                    let pb = eval.next_trace_mask();
                    let etop = eval.next_trace_mask();
                    let mag = eval.next_trace_mask();
                    let sg = eval.next_trace_mask();
                    let m2 = eval.next_trace_mask();
                    let ph = eval.next_trace_mask();
                    let r = eval.next_trace_mask();
                    let st = if sticky { eval.next_trace_mask() } else { E::F::zero() };
                    let m7n = eval.next_trace_mask();
                    let efn = eval.next_trace_mask();
                    let nzn = eval.next_trace_mask();
                    let c_sm = eval.next_trace_mask();
                    let c_ef = eval.next_trace_mask();
                    // the two alignments.  `u < 0` rows carry pow = 0, so an
                    // operand far below the anchor drops -- and the anchor
                    // constraint keeps the dropped one the smaller.
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.pow, E::EF::one(),
                            &[a_ef.clone() - etop.clone() + E::F::from(dmax), pa.clone()]));
                        eval.add_to_relation(RelationEntry::new(
                            &self.pow, E::EF::one(),
                            &[b_ef - etop.clone() + E::F::from(dmax), pb.clone()]));
                    }
                    eval.add_constraint(
                        (pa.clone() - E::F::from(pmax)) * (pb.clone() - E::F::from(pmax)));
                    // acc = mag (1 - 2 sg): the exact sum, scaled by 2^(DMAX-etop)
                    eval.add_constraint(sg.clone() * sg.clone() - sg.clone());
                    eval.add_constraint(a_sm * pa + b_sm * pb - mag.clone()
                                        + (mag.clone() * sg.clone()) * two);
                    eval.add_constraint(mag - m2.clone() * ph.clone() - r.clone());
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.norm, E::EF::one(),
                            &[m2, ph.clone(), st.clone(), m7n.clone(), efn.clone(),
                              nzn.clone()]));
                        eval.add_to_relation(RelationEntry::new(
                            &self.frac, E::EF::one(), &[ph, r, st]));
                    }
                    // the zero result has one sign, as B6.5's fold does
                    eval.add_constraint(sg.clone() * (E::F::one() - nzn.clone()));
                    eval.add_constraint(
                        c_sm.clone() - (nzn.clone() * c128 + m7n) * (E::F::one() - sg * two));
                    eval.add_constraint(
                        c_ef.clone() - nzn * (efn + etop - E::F::from(anchor)));
                    // B10 step 2, README 3.7: the output's own range check.  A
                    // `VAL` row exists for every legal `(sm, ef)` and for no
                    // other pair, so this is exactly "the result is a bf16 value"
                    // -- the check B6's dequant pays for its exponent field and
                    // the add did not, and the one thing between a sound gadget
                    // and a gadget that cannot prove what bf16 cannot represent.
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.val, E::EF::one(), &[c_sm.clone(), c_ef.clone()]));
                    }
                    outs.push((c_sm, c_ef));
                }
                Kind::Mul => {
                    let c_sm = eval.next_trace_mask();
                    let cy = eval.next_trace_mask();
                    let nzc = eval.next_trace_mask();
                    let c_ef = eval.next_trace_mask();
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.smul, E::EF::one(),
                            &[a_sm, b_sm, c_sm.clone(), cy.clone(), nzc.clone()]));
                    }
                    eval.add_constraint(
                        c_ef.clone() - nzc * (a_ef + b_ef + cy - E::F::from(c127)));
                    if !self.gkr {
                        eval.add_to_relation(RelationEntry::new(
                            &self.val, E::EF::one(), &[c_sm.clone(), c_ef.clone()]));
                    }
                    outs.push((c_sm, c_ef));
                }
            }
        }

        // ---- the boundary commitment (stream H) -------------------------------
        // One side of the layer's boundary is committed in its own tree, and the
        // whole of what binds it to the proof is this equality: the column in
        // that tree holds the packed value the component itself carries, row for
        // row.  There is no randomness in it, which is the reason the tree may
        // be committed after the interaction tree -- the constraint is checked
        // at the OODS point, drawn after every commitment, so a prover that
        // knows the earlier challenges has gained nothing.  See
        // notes/bounds-design.md.
        if let Some(b) = &self.bound {
            let members: Vec<(E::F, E::F)> = match b.kind {
                BindingKind::In(g) => leaf_in[g].clone(),
                BindingKind::Out(o) => {
                    self.prog.outs[o].iter().map(|s| outs[*s].clone()).collect()
                }
            };
            assert_eq!(members.len(), b.n, "the boundary's width disagrees with its plan");
            let pack = m31_u32(crate::boundary::PACK as u32);
            for (sm, ef) in members.iter() {
                let [c] = eval.next_interaction_mask(b.tree, [0]);
                eval.add_constraint(c - (sm.clone() * E::F::from(pack) + ef.clone()));
            }
        }

        // ---- the bindings ---------------------------------------------------------
        if let Some(t) = &self.binding {
            for spec in &t.specs {
                let [h_prev, h] =
                    eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
                let members: Vec<(E::F, E::F)> = match spec.kind {
                    BindingKind::In(g) => leaf_in[g].clone(),
                    BindingKind::Out(o) => {
                        self.prog.outs[o].iter().map(|s| outs[*s].clone()).collect()
                    }
                };
                let mut f = E::EF::zero();
                for (k, (sm, ef)) in members.iter().enumerate() {
                    f += (E::EF::from(sm.clone()) + E::EF::from(ef.clone()) * t.lam) * spec.w[k];
                }
                eval.add_constraint(h.clone() - (h_prev * nofirst.clone()) * spec.step - f);
                let claim = eval.claim(spec.claimed);
                eval.add_constraint((h - claim) * end.clone());
            }
        }
        if !self.gkr {
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

// ------------------------------------------------------------------ witness ---

/// The values a component consumes: one `rows x n` block of bf16 patterns per group.
pub struct ResidualStreamWitness {
    pub rows: usize,
    pub groups: Vec<Vec<u16>>,
    /// Stream II: `rows * prog.n_pre` bf16 patterns the *statement* carries --
    /// preprocessed, so they are not committed and no lookup pins them.
    pub pre: Vec<u16>,
    /// what one f64-rounded evaluation says each `outs` group holds, per row
    pub ref_out: Vec<Vec<u16>>,
    pub source: String,
    pub odd: usize,
}

impl ResidualStreamWitness {
    pub fn new(rows: usize, groups: Vec<Vec<u16>>, prog: &Prog, round: Round, source: String)
               -> Self {
        Self::with_pre(rows, groups, vec![], prog, round, source)
    }

    /// Stream II: the same, with the row's preprocessed values.
    pub fn with_pre(rows: usize, groups: Vec<Vec<u16>>, pre: Vec<u16>, prog: &Prog, round: Round,
                    source: String) -> Self {
        for (g, vals) in groups.iter().enumerate() {
            assert_eq!(vals.len(), rows * prog.groups[g].n, "group {g} has the wrong shape");
        }
        assert_eq!(pre.len(), rows * prog.n_pre, "the preprocessed block has the wrong shape");
        let odd = groups.iter().flatten().filter(|v| is_odd_pattern(**v)).count();
        let ref_out = reference(&groups, &pre, rows, prog, round);
        Self { rows, groups, pre, ref_out, source, odd }
    }

    /// Random bf16 values over a wide exponent spread, so the far branch, the
    /// exact zero and the cancellation edge are all exercised.
    pub fn synthetic(rows: usize, prog: &Prog, round: Round, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut groups = vec![];
        for g in &prog.groups {
            // a product's exponent is the *sum* of its operands', so the
            // multiply's synthetic spread has to be half the add's or the result
            // leaves bf16's normal range -- the gadget's incompleteness edge, not
            // a soundness one, and one the model's real values never reach
            let spread: i32 = match (prog.kind, g.un) {
                (Kind::Mul, UnKind::Silu) => 4,
                (Kind::Mul, _) => 10,
                (_, UnKind::Square) => 10,
                _ => 20,
            };
            let mut vals = vec![0u16; rows * g.n];
            for v in vals.iter_mut() {
                let e: i32 = rng.random_range(-spread..spread);
                let m: f64 = rng.random_range(-2.0f64..2.0);
                *v = match rng.random_range(0..32u32) {
                    // one value in 32 is an exact zero, two in 32 sit far away so
                    // the POW table's dropping rows are exercised
                    0 => 0,
                    1..=2 if prog.kind != Kind::Mul =>
                        bf16::from_f64(m * ((e as f64) * 4.0).exp2()),
                    _ => bf16::from_f64(m * (e as f64).exp2()),
                };
            }
            groups.push(vals);
        }
        // the preprocessed half of a synthetic component: the same spread, since
        // the norm weight is an ordinary bf16 value of the model
        let mut pre = vec![0u16; rows * prog.n_pre];
        for v in pre.iter_mut() {
            let e: i32 = rng.random_range(-4..4);
            let m: f64 = rng.random_range(-2.0f64..2.0);
            *v = bf16::from_f64(m * (e as f64).exp2());
        }
        Self::with_pre(rows, groups, pre, prog, round, format!("synthetic(seed={seed})"))
    }
}

/// What the program computes, in f64 with one bf16 rounding per op -- the
/// independent side of every `ref_mismatch` count below.
pub fn reference(groups: &[Vec<u16>], pre: &[u16], rows: usize, prog: &Prog, round: Round)
                 -> Vec<Vec<u16>> {
    let per_row: Vec<Vec<u16>> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let lin: Vec<Vec<u16>> = prog
                .groups
                .iter()
                .enumerate()
                .map(|(g, gr)| {
                    (0..gr.n).map(|i| canon(groups[g][j * gr.n + i])).collect()
                })
                .collect();
            let lun: Vec<Vec<u16>> = prog
                .groups
                .iter()
                .enumerate()
                .map(|(g, gr)| {
                    (0..gr.n)
                        .map(|i| match gr.un {
                            UnKind::None => 0u16,
                            k => {
                                let (sm, ef) = split(lin[g][i]);
                                bf16::from_f32(k.apply(to_f64(sm, ef)) as f32)
                            }
                        })
                        .collect()
                })
                .collect();
            let mut out: Vec<u16> = vec![];
            for (sa, sb) in &prog.slots {
                let get = |s: &Src, out: &Vec<u16>| -> u16 {
                    match s {
                        Src::Zero => 0,
                        Src::Leaf(g, i) => lin[*g][*i],
                        Src::Un(g, i) => lun[*g][*i],
                        Src::Slot(k) => out[*k],
                        Src::Pre(i) => canon(pre[j * prog.n_pre + *i]),
                    }
                };
                let (a, b) = (get(sa, &out), get(sb, &out));
                let (asm, aef) = split(a);
                let (bsm, bef) = split(b);
                let v = match prog.kind {
                    Kind::Add => to_f64(asm, aef) + to_f64(bsm, bef),
                    Kind::Mul => to_f64(asm, aef) * to_f64(bsm, bef),
                };
                out.push(canon(bf16::from_f64_round(v, round)));
            }
            prog.outs.iter().flat_map(|o| o.iter().map(|s| out[*s])).collect::<Vec<u16>>()
        })
        .collect();
    // re-cut the flat per-row answers into one vector per `outs` group
    let mut cut = vec![];
    let mut base = 0usize;
    for o in &prog.outs {
        let w = o.len();
        cut.push((0..rows).flat_map(|j| per_row[j][base..base + w].to_vec()).collect::<Vec<u16>>());
        base += w;
    }
    cut
}

// ------------------------------------------------------------------- tables ---

/// Every table this family reads, with the multiplicity columns the components fill.
/// `NORM` and `FRAC` are B6's, shared with the dequant components in a composed
/// proof -- there the layer's own multiplicity vectors are moved in here and
/// moved back, so the two families' uses land in one column.  The other four are
/// the residual stream's own.
pub struct ResidualStreamTables {
    pub pow: Vec<(M31, M31)>,
    pub smul: Vec<[M31; 5]>,
    pub uns: Vec<UnKind>,
    pub un: Vec<Vec<[M31; 4]>>,
    pub val: Vec<(M31, M31)>,
    pub norm: Vec<bf16::NormRow>,
    pub frac: Vec<(u32, u32, u32)>,
    pub m_pow: Vec<u32>,
    pub m_smul: Vec<u32>,
    pub m_un: Vec<Vec<u32>>,
    pub m_val: Vec<u32>,
    pub m_norm: Vec<u32>,
    pub m_frac: Vec<u32>,
    pub log_pow: u32,
    pub log_smul: u32,
    pub log_un: u32,
    pub log_val: u32,
    pub log_norm: u32,
    pub log_frac: u32,
    pub kmax: u32,
    pub sticky: bool,
    pub off_table: usize,
}

/// Stream PP (E11): the residual-stream tables' four `round`-only heights, for the
/// preprocessed split.  See `ResidualStreamTables::log_sizes`.
pub struct ResidualStreamLogSizes {
    pub pow: u32,
    pub smul: u32,
    pub un: u32,
    pub val: u32,
}

impl ResidualStreamTables {
    pub fn new(kmax: u32, round: Round, uns: &[UnKind]) -> Self {
        assert!(kmax >= DMAX + 1, "kmax {kmax} is below the add's DMAX + 1 = {}", DMAX + 1);
        let pow = pow_table();
        let smul = smul_table(round);
        let un: Vec<Vec<[M31; 4]>> = uns.iter().map(|k| un_table(*k)).collect();
        let val = val_table();
        let norm = bf16::norm_table(kmax, round);
        let sticky = round.needs_sticky();
        let mut frac = vec![(0u32, 0u32, 0u32)];
        for k in 0..=kmax {
            for r in 0..1u32 << k {
                frac.push((1 << k, r, u32::from(sticky && r != 0)));
            }
        }
        let log_pow = log_rows_for(pow.len(), 1);
        let log_smul = log_rows_for(smul.len(), 1);
        let log_un = log_rows_for(val.len(), 1);
        let log_val = log_rows_for(val.len(), 1);
        let log_norm = log_rows_for(norm.len(), 1);
        let log_frac = log_rows_for(frac.len(), 1);
        Self {
            m_pow: vec![0; 1 << log_pow],
            m_smul: vec![0; 1 << log_smul],
            m_un: un.iter().map(|_| vec![0; 1 << log_un]).collect(),
            m_val: vec![0; 1 << log_val],
            m_norm: vec![0; 1 << log_norm],
            m_frac: vec![0; 1 << log_frac],
            pow, smul, uns: uns.to_vec(), un, val, norm, frac,
            log_pow, log_smul, log_un, log_val, log_norm, log_frac,
            kmax, sticky, off_table: 0,
        }
    }

    /// Stream PP (E11): the four residual-stream table heights the preprocessed split
    /// needs, without building the tables.
    ///
    /// `Plan::pre_split` classifies a column by its id and its `log_size`, and
    /// the four residual-stream tables it has to size -- `pow`, `smul`, `val` and the
    /// unary ones -- depend on `round` alone.  Only `log_norm` and `log_frac`
    /// depend on `kmax`, and the split names neither.  Asking `ResidualStreamTables::new`
    /// for them built a `frac` table of `2^(kmax+1)` rows on every call to
    /// `pre_split_report`, and -- the reason this exists -- it asserts
    /// `kmax >= DMAX + 1`, which a plan that carries a residual-stream gadget it never
    /// proves need not satisfy (`embed`'s fixture, `kmax` 12).
    ///
    /// The heights are `new`'s own, `log_un` included: it is `val`'s length
    /// there and it is `val`'s length here.
    pub fn log_sizes(round: Round) -> ResidualStreamLogSizes {
        let log_val = log_rows_for(val_table().len(), 1);
        ResidualStreamLogSizes {
            pow: log_rows_for(pow_table().len(), 1),
            smul: log_rows_for(smul_table(round).len(), 1),
            un: log_val,
            val: log_val,
        }
    }

    pub fn un_index(&self, k: UnKind) -> usize {
        self.uns.iter().position(|x| *x == k).expect("unary table not registered")
    }
    pub fn pow_at(&self, u: i64) -> Option<usize> {
        if (0..=DMAX as i64).contains(&u) {
            Some(u as usize)
        } else if (-UMIN..0).contains(&u) {
            Some(DMAX as usize + 1 + (-u - 1) as usize)
        } else {
            None
        }
    }
    pub fn val_at(&self, sm: i64, ef: u32) -> Option<usize> {
        val_at(sm, ef)
    }
    pub fn smul_at(&self, a: i64, b: i64) -> Option<usize> {
        let idx = |v: i64| -> Option<usize> {
            if v == 0 {
                Some(0)
            } else if (128..256).contains(&v.unsigned_abs()) {
                Some(1 + usize::from(v < 0) * 128 + (v.unsigned_abs() - 128) as usize)
            } else {
                None
            }
        };
        Some(idx(a)? * 257 + idx(b)?)
    }
    /// `norm_table`'s own row order: the zero row, the 255 exactly-representable
    /// small magnitudes at `ph = 1`, then `(k, m2, st)` ascending.
    pub fn norm_at(&self, m2: u32, ph: u32, st: u32) -> Option<usize> {
        if m2 == 0 && ph == 0 {
            return Some(0);
        }
        if ph == 1 && (1..256).contains(&m2) {
            return Some(m2 as usize);
        }
        if !ph.is_power_of_two() || !(256..512).contains(&m2) {
            return None;
        }
        let k = ph.trailing_zeros();
        if k > self.kmax {
            return None;
        }
        let per = 1 + usize::from(self.sticky);
        Some(256 + (k as usize * 256 + (m2 - 256) as usize) * per
             + if self.sticky { st as usize } else { 0 })
    }
    pub fn frac_at(&self, ph: u32, r: u32) -> Option<usize> {
        if ph == 0 {
            return if r == 0 { Some(0) } else { None };
        }
        if !ph.is_power_of_two() || ph.trailing_zeros() > self.kmax || r >= ph {
            return None;
        }
        Some(ph as usize + r as usize)
    }
}

/// `val_table`'s row order, as an index: the zero row, then `(ef, m7, sign)`.
pub fn val_at(sm: i64, ef: u32) -> Option<usize> {
    if sm == 0 && ef == 0 {
        return Some(0);
    }
    if ef == 0 || ef >= 0xff || !(128..256).contains(&sm.unsigned_abs()) {
        return None;
    }
    let m7 = sm.unsigned_abs() - 128;
    Some(1 + ((ef - 1) as usize * 128 + m7 as usize) * 2 + usize::from(sm < 0))
}

// -------------------------------------------------------------- the component ----

/// What a probe corrupts.  Every one is a value the *prover* controls, and each
/// aims at a different half of the argument.
#[derive(Clone, Copy, Debug)]
pub enum Tamper {
    /// the committed result, one significand step up (the output constraint)
    Result(usize),
    /// the normalisation's committed output, consistently -- the algebra still
    /// holds and only the `NORM` row is wrong (the lookup)
    NormOut(usize),
    /// the anchor one step above the larger exponent (the anchor constraint)
    Anchor(usize),
    /// the anchor at the *smaller* exponent, so the larger operand would drop --
    /// the step the whole "drop is safe" argument rests on
    DropBig(usize),
    /// a leaf committed outside the legal `(sm, ef)` set
    NotBf16(usize),
    /// the result's sign, flipped
    Sign(usize),
    /// one committed leaf of group 0 replaced -- the residual stream a consumer
    /// reads is not the one its producer wrote (the Horner binding)
    Stream(usize),
    /// Stream II: one row's committed sum of squares, bent by one pattern step.
    /// It is still a legal `(sm, ef)`, so the `UnKind::NormR` lookup finds a row
    /// and the component's arithmetic stays consistent -- `r` is simply the
    /// `rsqrt` of a number this token's reduction did not produce.  Only the
    /// binding `nt.In(1) == G gnb.Out(0)` can catch it.
    Ms(usize),
    /// Stream II: one row's committed sum of squares replaced by the *next
    /// row's*.  Rows of one norm-scale component step through tokens at a fixed
    /// residue, so this gives two rows of one token two different `r`s, out of
    /// values that are each some token's honest reduction.  Again only the binding
    /// can catch it -- and this is the one the record's "one `r` per token"
    /// sentence rests on.
    MsRow(usize),
}

impl Tamper {
    pub fn name(&self) -> &'static str {
        match self {
            Tamper::Result(_) => "result one step up",
            Tamper::NormOut(_) => "normalisation output, algebra kept",
            Tamper::Anchor(_) => "anchor above the larger exponent",
            Tamper::DropBig(_) => "anchor at the smaller exponent (drop the larger)",
            Tamper::NotBf16(_) => "a leaf that is not a bf16 value",
            Tamper::Sign(_) => "result sign flipped",
            Tamper::Stream(_) => "one value of the residual stream, swapped",
            Tamper::Ms(_) => "one row's sum of squares, one step up",
            Tamper::MsRow(_) => "one row's sum of squares taken from the next row",
        }
    }
}

pub struct ResidualStreamComponent {
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub log_rows: u32,
    pub rows: usize,
    pub n_ops: usize,
    pub n_leaves: usize,
    pub ref_ok: usize,
    pub ref_n: usize,
    /// what each `outs` group holds, as bf16 patterns and in natural row order --
    /// the next component's leaves
    pub outs: Vec<Vec<u16>>,
    pub prog: Prog,
    pub round: Round,
    pub tag: String,
    pub bad_exp: usize,
    /// how many committed values the tamper actually changed.  Zero under a
    /// tamper means the probe measured nothing, and the caller must say so
    /// rather than report the honest proof's verdict as the tamper's.
    pub fired: usize,
    /// natural row order, for the Horner bindings: per group, then per `outs` group
    pub in_nat: Vec<Vec<(i64, i64)>>,
    pub out_nat: Vec<Vec<(i64, i64)>>,
    /// B11 stream M: the component's columns when the witness generation left them on the card,
    /// under `ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM=1`. `pre` and `base` are then empty and this is where
    /// the columns are. The three arrays above are *not* on the card: the bindings
    /// and the boundary sides read them by row, they are a fifth of the
    /// columns' bytes, and bringing them back is what lets the whole binding half
    /// stay exactly where it was.
    pub dev: Option<crate::dev_residual_stream_witness_gen::DevResidualStream>,
}

impl ResidualStreamComponent {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev {
            Some(d) => d.base.len(),
            None => self.base.len(),
        };
        n << self.log_rows
    }

    /// The values one binding accumulates, in natural row order.
    pub fn binding_vals(&self, kind: BindingKind) -> &[(i64, i64)] {
        match kind {
            BindingKind::In(g) => &self.in_nat[g],
            BindingKind::Out(o) => &self.out_nat[o],
        }
    }
    /// The width of one binding's sequence within a row.
    pub fn binding_width(&self, kind: BindingKind) -> usize {
        match kind {
            BindingKind::In(g) => self.prog.groups[g].n,
            BindingKind::Out(o) => self.prog.outs[o].len(),
        }
    }
    /// The claim itself, recomputed from the witness rather than from the trace.
    pub fn binding_claim(&self, spec: &TieSpec, lam: SecureField) -> SecureField {
        let vals = self.binding_vals(spec.kind);
        let w = self.binding_width(spec.kind);
        let rows = 1usize << self.log_rows;
        let mut h = SecureField::zero();
        for j in 0..rows {
            let mut f = SecureField::zero();
            for k in 0..w {
                let (sm, ef) = vals[j * w + k];
                f += (SecureField::from(m31_of(sm)) + SecureField::from(m31_of(ef)) * lam)
                    * spec.w[k];
            }
            h = h * spec.step + f;
        }
        h
    }
}

struct RowOut {
    cells: Vec<M31>,
    outs: Vec<u16>,
    /// the row's committed leaves and its named outputs, in natural row order and
    /// as the columns actually hold them -- the Horner bindings read these, because a
    /// committed `Eval` is bit-reversed
    in_nat: Vec<(i64, i64)>,
    out_nat: Vec<(i64, i64)>,
    look: Vec<(u8, usize)>,
    off: usize,
    bad_exp: usize,
    /// how many times the tamper changed a committed value in this row.  A
    /// tamper that did not fire proves the honest witness, and "accepted" is
    /// then the wrong word for what happened -- `drop-big` at layer 0 was this:
    /// its target slot's two operands shared an exponent, so there was no
    /// larger one to drop, and the probe read as a soundness gap for a day.
    fired: usize,
}

/// The tag a row's lookup carries so the parallel witness generation's multiplicities can be
/// merged into the right table afterwards.  `T_UN + i` is the `i`-th unary
/// table, so the base has to sit above every fixed tag: with one unary the old
/// base of 5 worked by luck, and the second one collided with `T_SMUL`, which
/// credited every silu lookup to the multiply table and left the sums
/// uncancelled with every constraint still satisfied.
const T_POW: u8 = 0;
const T_NORM: u8 = 1;
const T_VAL: u8 = 3;
const T_FRAC: u8 = 4;
const T_SMUL: u8 = 6;
const T_UN: u8 = 8;

/// Build one component's committed columns and bump the shared tables.
/// The reference count the host loop keeps inline: how many of this component's
/// outputs agree with `ResidualStreamWitness::ref_out`, the one f64-rounded evaluation.
///
/// B11 stream M: pulled out so the device path reports the same two numbers
/// from the same comparison. It reads `outs` and nothing else, so it does not
/// care which side built them.
fn ref_count(w: &ResidualStreamWitness, prog: &Prog, outs: &[Vec<u16>]) -> (usize, usize) {
    let (mut ok, mut n) = (0usize, 0usize);
    for (oi, o) in prog.outs.iter().enumerate() {
        for j in 0..w.rows {
            for k in 0..o.len() {
                n += 1;
                if outs[oi][j * o.len() + k] == w.ref_out[oi][j * o.len() + k] {
                    ok += 1;
                }
            }
        }
    }
    (ok, n)
}

pub fn residual_stream_component(w: &ResidualStreamWitness, prog: &Prog, round: Round, tag: &str,
                              tables: &mut ResidualStreamTables, tamper: Option<Tamper>)
                              -> ResidualStreamComponent {
    let log_rows = log_rows_for(w.rows, 1);
    let rows = 1usize << log_rows;
    // B11 stream M: which witness generation runs. 1 is the device's alone -- `pre` and
    // `base` stay empty and the columns live in `dev` -- and 2 builds both and
    // asserts every cell, every multiplicity delta, `off_table`, `bad_exp` and
    // both natural arrays before either is used.
    //
    // The tampers stay host-side, and a run with one set declines the device
    // path rather than running a kernel that has never heard of them. That is
    // announced: a probe that silently proved the untampered witness would read
    // as a gadget that accepts.
    let m4g = crate::dev_residual_stream_witness_gen::mode();
    let m4g = if m4g != 0 && tamper.is_some() {
        eprintln!("residual_stream_witness_gen-dev: {tag}: a residual-stream tamper is set, \
                  so this component generates its witness on the host");
        0
    } else if m4g != 0 && prog.n_pre > 0 {
        // stream II: `FSlot` has no encoding for a preprocessed operand -- a
        // `Src::Pre` lowers to `(NONE, NONE)`, which the kernel reads as the
        // constant zero -- so a component with one takes the host witness generation whatever the
        // switch says, as a wide Freivalds group already does.  Announced, because a
        // device witness generation that silently multiplied by zero would look like a bad
        // witness.
        eprintln!("residual_stream_witness_gen-dev: {tag}: the component reads preprocessed values, \
                  so it generates its witness on the host");
        0
    } else {
        m4g
    };
    if m4g == 1 {
        let dev = crate::dev_residual_stream_witness_gen::residual_stream_component_dev(w, prog, round, tables);
        let (ref_ok, ref_n) = ref_count(w, prog, &dev.outs);
        let bad_exp = dev.bad_exp;
        let (in_nat, out_nat, outs) =
            (dev.in_nat.clone(), dev.out_nat.clone(), dev.outs.clone());
        crate::dev_residual_stream_witness_gen::merge(tables, &dev);
        return ResidualStreamComponent {
            pre: vec![], base: vec![], log_rows, rows: w.rows,
            n_ops: w.rows * prog.slots.len(),
            n_leaves: w.rows * prog.n_leaves(),
            ref_ok, ref_n, outs, prog: prog.clone(), round,
            tag: tag.to_string(), bad_exp, fired: 0, in_nat, out_nat,
            dev: Some(dev),
        };
    }
    let m4g_snap = (m4g == 2).then(|| crate::dev_residual_stream_witness_gen::ResidualStreamSnap::of(tables));
    let m4g_off = tables.off_table;
    let n_cols = prog.n_cols(round);
    let sticky = round.needs_sticky();
    let n_slots = prog.slots.len().max(1);
    let n_ops = w.rows * prog.slots.len();
    let n_leaf_tot = w.rows * prog.n_leaves();
    let un_idx: Vec<usize> =
        prog.groups.iter().map(|g| {
            if g.un == UnKind::None { usize::MAX } else { tables.un_index(g.un) }
        }).collect();
    let out_w: usize = prog.outs.iter().map(|o| o.len()).sum();

    let witness_gen: Vec<RowOut> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut cells = vec![M31::zero(); n_cols];
            let mut look: Vec<(u8, usize)> = vec![];
            let (mut off, mut bad_exp, mut fired) = (0usize, 0usize, 0usize);
            let live = j < w.rows;
            // ---- leaves ----------------------------------------------------
            let mut lin: Vec<Vec<(i64, u32)>> = vec![];
            let mut lun: Vec<Vec<(i64, u32)>> = vec![];
            // the *global* leaf index, so a probe aimed at value `t` lands on it
            // whatever the component's row width (a per-row counter silently missed
            // every target past the first row -- and a probe that does not fire
            // reads as a gadget that accepts)
            let mut flat = j * prog.n_leaves();
            let mut in_nat: Vec<(i64, i64)> = vec![];
            for (g, gr) in prog.groups.iter().enumerate() {
                let (mut ins, mut uns) = (vec![], vec![]);
                for i in 0..gr.n {
                    let mut pat = if live { w.groups[g][j * gr.n + i] } else { 0 };
                    if let Some(Tamper::Stream(t)) = tamper {
                        if live && g == 0 && t % n_leaf_tot.max(1) == flat {
                            pat ^= 1;
                            fired += 1;
                        }
                    }
                    // stream II: the two probes the `In(1)` binding exists for.
                    // Both leave a legal bf16 pattern, so the `NormR` lookup
                    // still finds its row and the component stays self-consistent.
                    if let Some(Tamper::Ms(t)) = tamper {
                        if live && g == 1 && t % w.rows.max(1) == j {
                            pat ^= 1;
                            fired += 1;
                        }
                    }
                    if let Some(Tamper::MsRow(t)) = tamper {
                        if live && g == 1 && t % w.rows.max(1) == j {
                            let nxt = (j + 1) % w.rows;
                            let other = w.groups[g][nxt * gr.n + i];
                            if other != pat {
                                pat = other;
                                fired += 1;
                            }
                        }
                    }
                    let (mut sm, ef) = split(pat);
                    if let Some(Tamper::NotBf16(t)) = tamper {
                        if live && t % n_leaf_tot.max(1) == flat {
                            sm = 5;
                            fired += 1;
                        }
                    }
                    flat += 1;
                    let b = prog.leaf_at(g, i);
                    cells[b] = m31_of(sm);
                    cells[b + 1] = m31_u32(ef);
                    in_nat.push((sm, ef as i64));
                    if gr.un == UnKind::None {
                        match val_at(sm, ef) {
                            Some(k) => look.push((T_VAL, k)),
                            None => off += 1,
                        }
                        uns.push((0i64, 0u32));
                    } else {
                        match val_at(sm, ef) {
                            Some(k) => look.push((T_UN + un_idx[g] as u8, k)),
                            None => off += 1,
                        }
                        let y = un_image_parts(gr.un, sm, ef);
                        let (osm, oef) = split(y);
                        cells[b + 2] = m31_of(osm);
                        cells[b + 3] = m31_u32(oef);
                        uns.push((osm, oef));
                    }
                    ins.push((sm, ef));
                }
                lin.push(ins);
                lun.push(uns);
            }
            // ---- slots ------------------------------------------------------
            let mut outs: Vec<(i64, u32)> = vec![];
            // `drop-big` needs a slot whose operands differ in exponent, and
            // whether the aimed slot has one is the data's choice, not the
            // probe's (17.5% of a real layer's residual adds do not).  So the
            // aimed slot *arms* the tamper and the first eligible slot at or
            // after it, in this row, fires it.
            let mut armed = false;
            for (si, (sa, sb)) in prog.slots.iter().enumerate() {
                let hit = |t: usize| live && t % (w.rows.max(1) * n_slots) == j * n_slots + si;
                let get = |s: &Src, outs: &Vec<(i64, u32)>| -> (i64, u32) {
                    match s {
                        Src::Zero => (0, 0),
                        Src::Leaf(g, i) => lin[*g][*i],
                        Src::Un(g, i) => lun[*g][*i],
                        Src::Slot(k) => outs[*k],
                        // stream II: the statement's own value, read from the
                        // same block the preprocessed column is built from
                        Src::Pre(i) => {
                            let (sm, ef) = split(if live { w.pre[j * prog.n_pre + *i] } else { 0 });
                            (sm, ef)
                        }
                    }
                };
                let (a_sm, a_ef) = get(sa, &outs);
                let (b_sm, b_ef) = get(sb, &outs);
                let b = prog.slot_at(si, round);
                match prog.kind {
                    Kind::Add => {
                        let mut etop = a_ef.max(b_ef) as i64;
                        if let Some(Tamper::Anchor(i)) = tamper {
                            if hit(i) {
                                etop += 1;
                                fired += 1;
                            }
                        }
                        if let Some(Tamper::DropBig(i)) = tamper {
                            if hit(i) {
                                armed = true;
                            }
                            if armed && a_ef != b_ef {
                                etop = a_ef.min(b_ef) as i64;
                                armed = false;
                                fired += 1;
                            }
                        }
                        let ua = a_ef as i64 - etop + DMAX as i64;
                        let ub = b_ef as i64 - etop + DMAX as i64;
                        let mut pw = |u: i64| -> i64 {
                            match tables.pow_at(u) {
                                Some(i) => {
                                    look.push((T_POW, i));
                                    if u >= 0 { 1i64 << u } else { 0 }
                                }
                                None => {
                                    off += 1;
                                    0
                                }
                            }
                        };
                        let (fa, fb) = (pw(ua), pw(ub));
                        cells[b] = m31_of(fa);
                        cells[b + 1] = m31_of(fb);
                        let acc = a_sm * fa + b_sm * fb;
                        let mut sg = u32::from(acc < 0);
                        let mag = acc.unsigned_abs();
                        if let Some(Tamper::Sign(i)) = tamper {
                            if hit(i) {
                                sg ^= 1;
                                // a zero sum has no sign to flip: `sm_c` is 0
                                // either way and nothing constrains `sg`
                                if mag != 0 {
                                    fired += 1;
                                }
                            }
                        }
                        let (m2, ph, st_raw, r) = match bf16::normalise(mag, tables.kmax) {
                            Some(v) => v,
                            None => {
                                off += 1;
                                (0, 0, 0, 0)
                            }
                        };
                        let st = if sticky { st_raw } else { 0 };
                        let (mut m7n, efn, nzn) = match tables.norm_at(m2, ph, st) {
                            Some(i) => {
                                let row = tables.norm[i];
                                (row.m7, row.expfield, row.nz)
                            }
                            None => {
                                off += 1;
                                (0, 0, 0)
                            }
                        };
                        if let Some(Tamper::NormOut(i)) = tamper {
                            if hit(i) {
                                m7n = (m7n + 1) % 128;
                                fired += 1;
                            }
                        }
                        // the bumped row is not in the table, so nothing is
                        // bumped for it and the sums stop cancelling
                        match tables.norm_at(m2, ph, st).filter(|_| {
                            !matches!(tamper, Some(Tamper::NormOut(i)) if hit(i))
                        }) {
                            Some(i) => look.push((T_NORM, i)),
                            None => off += 1,
                        }
                        match tables.frac_at(ph, r as u32) {
                            Some(i) => look.push((T_FRAC, i)),
                            None => off += 1,
                        }
                        let mut c_sm =
                            (128 * nzn as i64 + m7n as i64) * if sg == 1 { -1 } else { 1 };
                        let c_ef = if nzn == 1 {
                            efn as i64 + etop - DMAX as i64 - EXP_BIAS as i64
                        } else {
                            0
                        };
                        if nzn == 1 && !(1..0xff).contains(&c_ef) {
                            bad_exp += 1;
                        }
                        if let Some(Tamper::Result(i)) = tamper {
                            if hit(i) {
                                c_sm += 1;
                                fired += 1;
                            }
                        }
                        cells[b + 2] = m31_of(etop);
                        cells[b + 3] = m31_of(mag as i64);
                        cells[b + 4] = m31_u32(sg);
                        cells[b + 5] = m31_u32(m2);
                        cells[b + 6] = m31_u32(ph);
                        cells[b + 7] = m31_of(r as i64);
                        let mut k = 8;
                        if sticky {
                            cells[b + k] = m31_u32(st);
                            k += 1;
                        }
                        cells[b + k] = m31_u32(m7n);
                        cells[b + k + 1] = m31_u32(efn);
                        cells[b + k + 2] = m31_u32(nzn);
                        cells[b + k + 3] = m31_of(c_sm);
                        cells[b + k + 4] = m31_of(c_ef);
                        match val_at(c_sm, c_ef.clamp(0, 0xff) as u32).filter(|_| {
                            (0..0xff).contains(&c_ef)
                        }) {
                            Some(i) => look.push((T_VAL, i)),
                            None => off += 1,
                        }
                        outs.push((c_sm, c_ef.clamp(0, 0xfe) as u32));
                    }
                    Kind::Mul => {
                        let (c_sm0, cy, nzc) = match tables.smul_at(a_sm, b_sm) {
                            Some(i) => {
                                look.push((T_SMUL, i));
                                let row = tables.smul[i];
                                (sgn_of(row[2]), row[3].0, row[4].0)
                            }
                            None => {
                                off += 1;
                                (0, 0, 0)
                            }
                        };
                        let c_ef = if nzc == 1 {
                            a_ef as i64 + b_ef as i64 + cy as i64 - 127
                        } else {
                            0
                        };
                        if nzc == 1 && !(1..0xff).contains(&c_ef) {
                            bad_exp += 1;
                        }
                        let mut c_sm = c_sm0;
                        if let Some(Tamper::Result(i)) = tamper {
                            if hit(i) {
                                c_sm += 1;
                                fired += 1;
                            }
                        }
                        cells[b] = m31_of(c_sm);
                        cells[b + 1] = m31_u32(cy);
                        cells[b + 2] = m31_u32(nzc);
                        cells[b + 3] = m31_of(c_ef);
                        match val_at(c_sm, c_ef.clamp(0, 0xff) as u32).filter(|_| {
                            (0..0xff).contains(&c_ef)
                        }) {
                            Some(i) => look.push((T_VAL, i)),
                            None => off += 1,
                        }
                        outs.push((c_sm, c_ef.clamp(0, 0xfe) as u32));
                    }
                }
            }
            let mut out_pat = Vec::with_capacity(out_w);
            let mut out_nat = Vec::with_capacity(out_w);
            for o in &prog.outs {
                for s in o {
                    let (sm, ef) = outs[*s];
                    out_pat.push(join(sm, ef));
                    let (c0, c1) = prog.out_cols(*s, round);
                    out_nat.push((sgn_of(cells[c0]), sgn_of(cells[c1])));
                }
            }
            RowOut { cells, outs: out_pat, in_nat, out_nat, look, off, bad_exp, fired }
        })
        .collect();

    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let (mut ref_ok, mut ref_n, mut bad_exp) = (0usize, 0usize, 0usize);
    let fired: usize = witness_gen.iter().map(|f| f.fired).sum();
    if tamper.is_some() && fired == 0 {
        eprintln!("{tag}: the {} tamper did not fire -- this component is the honest one",
                  tamper.unwrap().name());
    }
    let mut outs: Vec<Vec<u16>> = prog.outs.iter().map(|o| vec![0u16; rows * o.len()]).collect();
    for (j, f) in witness_gen.iter().enumerate() {
        for (c, v) in f.cells.iter().enumerate() {
            cols[c][j] = *v;
        }
        let mut base = 0usize;
        for (oi, o) in prog.outs.iter().enumerate() {
            for k in 0..o.len() {
                outs[oi][j * o.len() + k] = f.outs[base + k];
                if j < w.rows {
                    ref_n += 1;
                    if f.outs[base + k] == w.ref_out[oi][j * o.len() + k] {
                        ref_ok += 1;
                    } else if std::env::var("ZK_RESIDUAL_STREAM_DEBUG").is_ok() && ref_n - ref_ok < 8 {
                        eprintln!("  {tag} ref mismatch row {j} out {oi}.{k}: got {:#06x} \
                                   want {:#06x}", f.outs[base + k], w.ref_out[oi][j * o.len() + k]);
                    }
                }
            }
            base += o.len();
        }
        for (which, i) in &f.look {
            match *which {
                T_POW => tables.m_pow[*i] += 1,
                T_NORM => tables.m_norm[*i] += 1,
                T_VAL => tables.m_val[*i] += 1,
                T_FRAC => tables.m_frac[*i] += 1,
                T_SMUL => tables.m_smul[*i] += 1,
                u if u >= T_UN => tables.m_un[(u - T_UN) as usize][*i] += 1,
                u => unreachable!("unknown residual-stream lookup tag {u}"),
            }
        }
        tables.off_table += f.off;
        bad_exp += f.bad_exp;
    }
    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(log_rows, c)).collect();
    let mut end = vec![M31::zero(); rows];
    end[rows - 1] = M31::one();
    let mut nofirst = vec![M31::one(); rows];
    nofirst[0] = M31::zero();
    let mut pre = vec![eval_of(log_rows, nofirst), eval_of(log_rows, end)];
    // stream II: the component's own preprocessed values, `(sm, ef)` per slot, in
    // the order `Plan::pre_ids` names them.  A tamper on one of these is a
    // tamper on tree 0 and shows up as `root_ok` false, never as a rejected
    // constraint -- which is the point of moving the norm weights here.
    for i in 0..prog.n_pre {
        for k in 0..2 {
            let col: Vec<M31> = (0..rows)
                .map(|j| {
                    let (sm, ef) = split(if j < w.rows { w.pre[j * prog.n_pre + i] } else { 0 });
                    if k == 0 { m31_of(sm) } else { m31_u32(ef) }
                })
                .collect();
            pre.push(eval_of(log_rows, col));
        }
    }
    // the bindings read the values in natural row order, per group
    let mut in_nat: Vec<Vec<(i64, i64)>> =
        prog.groups.iter().map(|g| Vec::with_capacity(rows * g.n)).collect();
    let mut out_nat: Vec<Vec<(i64, i64)>> =
        prog.outs.iter().map(|o| Vec::with_capacity(rows * o.len())).collect();
    for f in &witness_gen {
        let mut b = 0usize;
        for (g, gr) in prog.groups.iter().enumerate() {
            in_nat[g].extend_from_slice(&f.in_nat[b..b + gr.n]);
            b += gr.n;
        }
        let mut b = 0usize;
        for (oi, o) in prog.outs.iter().enumerate() {
            out_nat[oi].extend_from_slice(&f.out_nat[b..b + o.len()]);
            b += o.len();
        }
    }
    // B11 stream M, mode 2: both witness generations built, everything asserted equal before
    // any column is used. The device's own bumps are the check's and not the
    // proof's, so the host's tables go back where the host witness generation left them.
    if m4g == 2 {
        let after = crate::dev_residual_stream_witness_gen::ResidualStreamSnap::of(tables);
        let after_off = tables.off_table;
        let dev = crate::dev_residual_stream_witness_gen::residual_stream_component_dev(w, prog, round, tables);
        crate::dev_residual_stream_witness_gen::check_residual_stream_dev(tag, &base, &pre, &dev,
                                                                   after_off - m4g_off, bad_exp,
                                                                   &in_nat, &out_nat, &outs,
                                                                   &m4g_snap.unwrap(), &after);
        crate::dev_residual_stream_witness_gen::restore(tables, &after, after_off);
    }
    ResidualStreamComponent {
        pre, base, log_rows, rows: w.rows, n_ops, n_leaves: n_leaf_tot,
        ref_ok, ref_n, outs, prog: prog.clone(), round,
        tag: tag.to_string(), bad_exp, fired, in_nat, out_nat, dev: None,
    }
}

// ------------------------------------------------------------- interaction ----

/// Every relation the residual stream draws, in one canonical order.
#[derive(Clone)]
pub struct ResidualStreamRels {
    pub norm: NormRel,
    pub frac: FracRel,
    pub pow: PowRel,
    pub smul: SMulRel,
    pub un: Vec<UnRel>,
    /// the kinds `un` is indexed by.  This is the *plan's* list, the union over
    /// every component, not any one program's: two components drawing different
    /// unaries share one relation vector, and indexing it by a single component's
    /// `un_kinds()` would point the second component at the first one's table.
    pub uns: Vec<UnKind>,
    pub val: ValRel,
}

pub struct ResidualStreamInteraction {
    /// B11 stream M: columns of either kind. The bindings are host accumulators and
    /// always will be; the LogUp half goes wherever `devfrac::frac_cols` sends
    /// it, which since this stream is the generated kernel.
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub use_sum: SecureField,
    pub claimed: Vec<SecureField>,
}

/// One lookup, as the columns (and the one expression) it reads.
enum L {
    /// `(sm, ef)` at a base column -- a leaf, or a slot's output
    Val(usize, usize),
    /// `(sm, ef, sm', ef')` of a leaf and its unary image
    Un(usize, usize),
    /// `ef - etop + DMAX` against the committed power
    Pow(usize, usize, usize),
    /// the normalisation, at a slot's base column
    Norm(usize),
    /// `(ph, r, st)`, at a slot's base column
    Frac(usize),
    /// `(sm_a, sm_b, sm_c, cy, nz)`: two operand columns and a slot base
    SMul(usize, usize, usize),
}

/// Stream II: an operand's two column indices for the interaction, which reads
/// out of two different vectors.  `PRE_TAG` says "preprocessed, index minus the
/// tag"; `usize::MAX` is still the constant `+0`.
pub const PRE_TAG: usize = 1 << 40;

fn src_cols_any(prog: &Prog, s: &Src, round: Round) -> (usize, usize) {
    match s {
        Src::Pre(i) => (PRE_TAG + prog.pre_at(*i, 0), PRE_TAG + prog.pre_at(*i, 1)),
        _ => prog.src_cols(s, round).unwrap_or((usize::MAX, usize::MAX)),
    }
}

/// Where each binding's sequence sits in the witness generation's two row-flat natural arrays.
///
/// B11 stream N. The host reads a binding through `binding_vals`, which hands back the
/// one regrouped vector for that group or output. The device has no regrouped
/// vectors -- `zk_witness_gen_residual_stream` writes one array per kind, `n_leaves` or `out_w`
/// `(sm, ef)` pairs to a row -- so a binding is named by where its sequence starts
/// in a row and how wide it is. Both offsets are the same running sums the witness generation
/// itself uses: `FGroup::flat` for the leaves, and `out_slot`'s concatenation
/// order for the outputs.
///
/// The weights of every binding are packed into one array, so the kernel takes one
/// upload rather than one per lane.
fn binding_specs(r: &ResidualStreamComponent, t: &Binding)
                 -> (Vec<crate::devacc::FAccResidualStreamBinding>, Vec<SecureField>) {
    let prog = &r.prog;
    let n_leaves = prog.n_leaves();
    let out_w: usize = prog.outs.iter().map(|o| o.len()).sum();
    let mut weights: Vec<SecureField> = vec![];
    let specs = t
        .specs
        .iter()
        .map(|spec| {
            let (src, off, width, stride) = match spec.kind {
                BindingKind::In(g) => {
                    let off: usize = prog.groups[..g].iter().map(|x| x.n).sum();
                    (0u32, off, prog.groups[g].n, n_leaves)
                }
                BindingKind::Out(o) => {
                    let off: usize = prog.outs[..o].iter().map(|x| x.len()).sum();
                    (1u32, off, prog.outs[o].len(), out_w)
                }
            };
            assert_eq!(width, spec.w.len(), "a binding's weight count is its width");
            let w_off = weights.len();
            weights.extend_from_slice(&spec.w);
            crate::devacc::FAccResidualStreamBinding {
                src,
                off: off as u32,
                width: width as u32,
                stride: stride as u32,
                w_off: w_off as u32,
                pad: 0,
                step: spec.step.to_m31_array().map(|x| x.0),
            }
        })
        .collect();
    (specs, weights)
}

/// The component's Horner bindings and its LogUp columns, in the AIR's own order.
///
/// B11 stream M: `air` is this component's own eval, and it is here because the
/// LogUp half now goes through `devfrac::frac_cols` like every other family's.
/// The residual stream was the last component with no device route at all -- not for want
/// of a kernel, which stream K's split emitter has compiled and installed since
/// `zk_residual_stream_eval` fit under the cap, but because nothing ever called the
/// router. The bindings stay on the host: they are a serial Horner fold over the
/// component's own values, which is `devacc`'s shape and not this stream's.
pub fn residual_stream_interaction(r: &ResidualStreamComponent, rels: &ResidualStreamRels,
                                   binding: Option<&Binding>, air: &ResidualStreamEval,
                                   gkr: bool) -> ResidualStreamInteraction {
    use stwo::prover::backend::simd::m31::PackedM31;
    let prog = &r.prog;
    let round = r.round;
    let rows = 1usize << r.log_rows;
    let sticky = round.needs_sticky();
    let mut cols = crate::device::ColSink::new();
    let mut claimed: Vec<SecureField> = vec![];
    if let Some(t) = binding {
        // B11 stream N: which side folds the bindings. The device route needs the
        // witness generation's own natural arrays, so it exists exactly when this component
        // built on the card; mode 2 builds both and asserts every cell and
        // every claim.
        let m4t = r.dev.as_ref().filter(|_| crate::devacc::binding_mode() >= 1);
        let host = m4t.is_none() || crate::devacc::binding_mode() == 2;
        let mut host_cols: Vec<crate::common::Eval> = vec![];
        let mut host_claims: Vec<SecureField> = vec![];
        if host {
            for spec in &t.specs {
                let vals = r.binding_vals(spec.kind);
                let wdt = r.binding_width(spec.kind);
                let mut h = vec![SecureField::zero(); rows];
                for j in 0..rows {
                    let mut f = SecureField::zero();
                    for k in 0..wdt {
                        let (sm, ef) = vals[j * wdt + k];
                        f += (SecureField::from(m31_of(sm))
                              + SecureField::from(m31_of(ef)) * t.lam) * spec.w[k];
                    }
                    h[j] = if j == 0 { f } else { h[j - 1] * spec.step + f };
                }
                host_claims.push(h[rows - 1]);
                host_cols.extend(crate::attn_trace::secure_cols_pub(r.log_rows, &h));
            }
        }
        match m4t {
            Some(d) => {
                let (specs, weights) = binding_specs(r, t);
                let got = crate::devacc::residual_stream_bind(
                    d.d_in_nat.device_ptr as *const i32,
                    d.d_out_nat.device_ptr as *const i32,
                    r.log_rows,
                    &specs,
                    &weights,
                    t.lam,
                );
                claimed = crate::devacc::place(
                    &format!("residual stream {} bindings", r.tag),
                    got,
                    host.then_some((host_cols, host_claims)),
                    &mut cols,
                );
            }
            None => {
                claimed = host_claims;
                cols.extend(host_cols);
            }
        }
    }
    let n_binding = cols.len();

    // the lookups, in exactly the order `evaluate` adds them
    let mut looks: Vec<L> = vec![];
    let mut un_of: Vec<usize> = vec![];
    let kinds = &rels.uns;
    for (g, gr) in prog.groups.iter().enumerate() {
        for i in 0..gr.n {
            let b = prog.leaf_at(g, i);
            if gr.un == UnKind::None {
                looks.push(L::Val(b, b + 1));
                un_of.push(usize::MAX);
            } else {
                looks.push(L::Un(b, b + 1));
                un_of.push(kinds.iter().position(|k| *k == gr.un).unwrap());
            }
        }
    }
    for (si, (sa, sb)) in prog.slots.iter().enumerate() {
        let b = prog.slot_at(si, round);
        let (o0, o1) = prog.out_cols(si, round);
        match prog.kind {
            Kind::Add => {
                let ea = src_cols_any(prog, sa, round).1;
                let eb = src_cols_any(prog, sb, round).1;
                looks.push(L::Pow(ea, b + 2, b));
                un_of.push(usize::MAX);
                looks.push(L::Pow(eb, b + 2, b + 1));
                un_of.push(usize::MAX);
                looks.push(L::Norm(b));
                un_of.push(usize::MAX);
                looks.push(L::Frac(b));
                un_of.push(usize::MAX);
                looks.push(L::Val(o0, o1));
                un_of.push(usize::MAX);
            }
            Kind::Mul => {
                let ca = src_cols_any(prog, sa, round).0;
                let cb = src_cols_any(prog, sb, round).0;
                looks.push(L::SMul(ca, cb, b));
                un_of.push(usize::MAX);
                looks.push(L::Val(o0, o1));
                un_of.push(usize::MAX);
            }
        }
    }
    let one = PackedQM31::broadcast(SecureField::one());
    let zero = PackedM31::broadcast(M31::zero());
    let dmax = PackedM31::broadcast(m31_u32(DMAX));
    let k = 8 + usize::from(sticky);
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        // `usize::MAX` is the constant `+0` operand: sm = ef = 0, no column.
        // stream II: `PRE_TAG` is a preprocessed operand, which lives in the
        // other vector -- `L::Un`'s `c(a + 2)` never sees one, because a
        // preprocessed value carries no unary.
        let c = |x: usize| {
            if x == usize::MAX {
                zero
            } else if x >= PRE_TAG {
                r.pre[x - PRE_TAG].data[vr]
            } else {
                r.base[x].data[vr]
            }
        };
        match &looks[i] {
            L::Val(a, b) => (one, rels.val.combine(&[c(*a), c(*b)])),
            L::Un(a, b) => (one, rels.un[un_of[i]].combine(&[c(*a), c(*b), c(a + 2), c(b + 2)])),
            L::Pow(ef, etop, p) => (one, rels.pow.combine(&[c(*ef) - c(*etop) + dmax, c(*p)])),
            L::Norm(b) => {
                let st = if sticky { c(b + 8) } else { zero };
                (one, rels.norm.combine(&[c(b + 5), c(b + 6), st,
                                          c(b + k), c(b + k + 1), c(b + k + 2)]))
            }
            L::Frac(b) => {
                let st = if sticky { c(b + 8) } else { zero };
                (one, rels.frac.combine(&[c(b + 6), c(b + 7), st]))
            }
            L::SMul(a, b2, out) => {
                (one, rels.smul.combine(&[c(*a), c(*b2), c(*out), c(out + 1), c(out + 2)]))
            }
        }
    };
    // B11 stream M: the router, with the hand-written builder as the closure it
    // compares against at modes 2 and 4 and falls back to at mode 0. The
    // component's columns are the host's today -- `gen/residual_stream` has no device path --
    // so this is `Cols::Host` and the opt-in `ZK_GENERATED_LOGUP` decides. When
    // `gen/residual_stream` puts them on the card the same call takes the kernel
    // unconditionally, which is how the two levers compose.
    // B11 stream V, lever 3: under `--gkr-residual-stream` the component's fractions were
    // handed to the collector before the GKR proof ran, in `compose`'s own pass
    // -- see the comment there for why the two are in different places -- so
    // there are no running-sum columns to build and no claimed sum to carry.
    // The binding columns above are untouched: they are Horner accumulators, they
    // are 0.4% of this family's interaction cells, and nothing about them is a
    // lookup.
    if gkr {
        let _ = fr;
        return ResidualStreamInteraction {
            cols: cols.into_inner(),
            use_sum: SecureField::zero(),
            claimed,
        };
    }
    let (lcols, use_sum) = crate::devfrac::frac_cols(
        air, &format!("residual stream {}", r.tag), r.log_rows,
        crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
        crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
        || {
            let (n_pairs, fpair) = crate::plogup::paired(looks.len(), fr);
            crate::plogup::build(r.log_rows, n_pairs, fpair)
        });
    cols.extend_any(lcols);
    debug_assert!(cols.len() > n_binding, "residual stream {}: no LogUp columns", r.tag);
    ResidualStreamInteraction { cols: cols.into_inner(), use_sum, claimed }
}

// ---------------------------------------------------------------- the tables --

/// The residual stream's own tables, in the shape `dequant::TableEval` has for B6's: a
/// preprocessed row, a committed multiplicity, yielded at `-m`.
#[derive(Clone)]
pub enum GKind {
    Pow(PowRel),
    SMul(SMulRel),
    Un(UnRel),
    Val(ValRel),
    Norm(NormRel),
    Frac(FracRel),
}

impl GKind {
    fn combine(&self, v: &[stwo::prover::backend::simd::m31::PackedM31]) -> PackedQM31 {
        match self {
            GKind::Pow(r) => r.combine(v),
            GKind::SMul(r) => r.combine(v),
            GKind::Un(r) => r.combine(v),
            GKind::Val(r) => r.combine(v),
            GKind::Norm(r) => r.combine(v),
            GKind::Frac(r) => r.combine(v),
        }
    }
}

#[derive(Clone)]
pub struct GTableEval {
    pub log_rows: u32,
    pub cols: Vec<PreProcessedColumnId>,
    pub kind: GKind,
}

impl FrameworkEval for GTableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`); `None` when this
    /// component's fingerprint has no kernel, which is the fork's
    /// scalar CPU route.
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "zk_residual_stream_table_eval" }

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
            GKind::Pow(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            GKind::SMul(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            GKind::Un(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            GKind::Val(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            GKind::Norm(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
            GKind::Frac(r) => eval.add_to_relation(RelationEntry::new(r, neg, &row)),
        }
        eval.finalize_logup();
        eval
    }
}

pub fn table_side(log_rows: u32, kind: &GKind, row: &[&Eval], mult: &Eval)
                  -> (Vec<Eval>, SecureField) {
    crate::plogup::build(log_rows, 1, |_, vr| {
        let vals: Vec<_> = row.iter().map(|c| c.data[vr]).collect();
        (-PackedQM31::from(mult.data[vr]), kind.combine(&vals))
    })
}

/// The residual stream's own preprocessed columns, in the order the prover commits them:
/// `POW`, `SMUL`, `VAL`, then one unary table per kind in `uns`.
pub fn residual_stream_table_ids(uns: &[UnKind]) -> Vec<PreProcessedColumnId> {
    let mut ids: Vec<PreProcessedColumnId> = POW_COLS.iter().map(|c| id(*c)).collect();
    ids.extend(SMUL_COLS.iter().map(|c| id(*c)));
    ids.extend(VAL_COLS.iter().map(|c| id(*c)));
    for k in uns {
        ids.extend((0..4).map(|i| un_col(k.name(), i)));
    }
    ids
}

/// The same columns' contents, padded to their own row counts.
///
/// Stream II, `notes/witness-slack.md` section 7 item 5: the padding **repeats
/// the last real row** rather than writing zeros.  `pow_table` has 273 rows and
/// pads to 512, and a zero-padded `POW` carries the tuple `(0, 0)` beside the
/// real `u = 0` row `(0, 1)` -- two legal rows for one key, so a prover may drop
/// an operand the anchor meant to add.  `GTableEval` commits the multiplicity
/// as a bare mask with no row selector, so nothing else forbids it.  Repeating
/// the last row makes every padding row a duplicate of a real one, which is the
/// state section 3 already records for `VAL` and `SMUL`, and it costs no column.
/// The unary tables need it for a second reason: `UnKind::NormR`'s image of
/// `(0, 0)` is `rsqrt(eps)`, not zero, so a zero-padded `UN` would have offered
/// `r = 0` for a zero sum of squares.
pub fn residual_stream_table_cols(t: &ResidualStreamTables) -> Vec<Eval> {
    let pad = |n: u32| 1usize << n;
    // the last real row, repeated -- `i.min(len - 1)`
    let at = |len: usize, i: usize| i.min(len.saturating_sub(1));
    let mut out = vec![];
    for k in 0..2 {
        out.push(eval_of(t.log_pow, (0..pad(t.log_pow))
            .map(|i| t.pow.get(at(t.pow.len(), i)).map(|r| if k == 0 { r.0 } else { r.1 })
                 .unwrap_or(M31::zero())).collect()));
    }
    for k in 0..5 {
        out.push(eval_of(t.log_smul, (0..pad(t.log_smul))
            .map(|i| t.smul.get(at(t.smul.len(), i)).map(|r| r[k])
                 .unwrap_or(M31::zero())).collect()));
    }
    for k in 0..2 {
        out.push(eval_of(t.log_val, (0..pad(t.log_val))
            .map(|i| t.val.get(at(t.val.len(), i)).map(|r| if k == 0 { r.0 } else { r.1 })
                 .unwrap_or(M31::zero())).collect()));
    }
    for u in &t.un {
        for k in 0..4 {
            out.push(eval_of(t.log_un, (0..pad(t.log_un))
                .map(|i| u.get(at(u.len(), i)).map(|r| r[k]).unwrap_or(M31::zero())).collect()));
        }
    }
    out
}

/// Their multiplicity columns, in the same order.
pub fn residual_stream_mult_cols(t: &ResidualStreamTables) -> Vec<Eval> {
    let mut out = vec![
        eval_of(t.log_pow, t.m_pow.iter().map(|m| m31_u32(*m)).collect()),
        eval_of(t.log_smul, t.m_smul.iter().map(|m| m31_u32(*m)).collect()),
        eval_of(t.log_val, t.m_val.iter().map(|m| m31_u32(*m)).collect()),
    ];
    for m in &t.m_un {
        out.push(eval_of(t.log_un, m.iter().map(|v| m31_u32(*v)).collect()));
    }
    out
}

/// The residual-stream tables' row counts, from their closed forms -- the verifier needs
/// them and has no reason to materialise 200k rows to find out.
pub fn residual_stream_table_logs() -> (u32, u32, u32) {
    let pow = DMAX as usize + 1 + UMIN as usize;
    let val = 1 + 254 * 128 * 2;
    let smul = 257 * 257;
    (log_rows_for(pow, 1), log_rows_for(smul, 1), log_rows_for(val, 1))
}

/// `(log_rows, columns, kind)` for the residual stream's own table components, in order.
pub fn residual_stream_table_kinds(uns: &[UnKind], rels: &ResidualStreamRels)
                        -> Vec<(u32, Vec<PreProcessedColumnId>, GKind)> {
    let (log_pow, log_smul, log_val) = residual_stream_table_logs();
    let mut v = vec![
        (log_pow, POW_COLS.iter().map(|c| id(*c)).collect::<Vec<_>>(),
         GKind::Pow(rels.pow.clone())),
        (log_smul, SMUL_COLS.iter().map(|c| id(*c)).collect(), GKind::SMul(rels.smul.clone())),
        (log_val, VAL_COLS.iter().map(|c| id(*c)).collect(), GKind::Val(rels.val.clone())),
    ];
    for (i, k) in uns.iter().enumerate() {
        v.push((log_val, (0..4).map(|j| un_col(k.name(), j)).collect(),
                GKind::Un(rels.un[i].clone())));
    }
    v
}

// ----------------------------------------------------------------- the run ----

pub struct Outcome {
    pub proven: Proven,
    pub sums: Sums,
    pub prog: Prog,
    pub round: Round,
    pub n_ops: usize,
    pub n_leaves: usize,
    pub off_table: usize,
    pub bad_exp: usize,
    /// how many committed values the tamper changed (0 with no tamper set)
    pub fired: usize,
    pub ref_ok: usize,
    pub ref_n: usize,
    pub odd: usize,
    pub table_cells_pre: usize,
    pub table_cells_fresh: usize,
    pub component_cells: usize,
    /// the residual-stream component's own interaction cells: the Horner bindings and the
    /// LogUp columns its lookups drag along (a pair of lookups is 4 columns)
    pub inter_cells: usize,
    pub n_columns: [usize; 3],
    pub binding_ok: bool,
    pub trace_secs: f64,
}

impl Outcome {
    pub fn accepts(&self) -> bool {
        self.proven.stark_ok && self.sums.cancels() && self.off_table == 0 && self.binding_ok
            && self.bad_exp == 0
    }
    /// Cells the gadget adds per op, witness *and* LogUp columns, the way B6 and
    /// B6.5 count.  The tables amortise over a whole layer, so they are reported
    /// apart; so are the leaves, which in a chained component are the previous
    /// component's outputs and are committed there (`leaf_width` columns plus the
    /// 2 cells the validity or unary lookup costs at pairwise batching).
    pub fn cells_per_op_marginal(&self) -> f64 {
        let leaf: usize = self
            .prog
            .groups
            .iter()
            .enumerate()
            .map(|(g, gr)| gr.n * (self.prog.leaf_width(g) + 2))
            .sum();
        let rows = self.n_ops / self.prog.slots.len().max(1);
        (self.component_cells + self.inter_cells) as f64 / self.n_ops as f64
            - (leaf * rows) as f64 / self.n_ops as f64
    }
    pub fn cells_per_op_total(&self) -> f64 {
        (self.component_cells + self.inter_cells) as f64 / self.n_ops as f64
    }
    pub fn json(&self, src: &str, gadget: &str) -> Value {
        json!({
            "gadget": gadget,
            "kind": match self.prog.kind { Kind::Add => "add", Kind::Mul => "mul" },
            "unary": self.prog.groups.iter().map(|g| g.un.name()).collect::<Vec<_>>(),
            "round": self.round.name(),
            "groups": self.prog.groups.iter().map(|g| g.n).collect::<Vec<_>>(),
            "slots": self.prog.slots.len(),
            "source": src,
            "ops": self.n_ops,
            "cells_per_op_marginal": self.cells_per_op_marginal(),
            "cells_per_op_total": self.cells_per_op_total(),
            "component_cells": self.component_cells,
            "interaction_cells": self.inter_cells,
            "table_cells_preprocessed": self.table_cells_pre,
            "table_cells_fresh": self.table_cells_fresh,
            "dmax": DMAX,
            "off_table": self.off_table,
            "exponent_out_of_range": self.bad_exp,
            "tamper_fired": self.fired,
            "odd_input_patterns": self.odd,
            "reference_checked": self.ref_n, "reference_ok": self.ref_ok,
            "reference_mismatch": self.ref_n - self.ref_ok,
            "binding_ok": self.binding_ok,
            "columns": {"preprocessed": self.n_columns[0], "base": self.n_columns[1],
                        "interaction": self.n_columns[2],
                        "total": self.n_columns.iter().sum::<usize>()},
            "trace_secs": self.trace_secs,
            "sums": self.sums.json(), "accepts": self.accepts(),
            "point": self.proven.json(),
        })
    }
}

/// The binding specs a standalone component carries: one per input group and one per
/// `outs` group, with the plain descending weights -- the composed proof picks
/// its own so that two differently tiled components line up (see `compose::residual_stream`).
pub fn plain_bindings(prog: &Prog, log_rows: u32, w: SecureField) -> Vec<TieSpec> {
    let _ = log_rows;
    let mut specs = vec![];
    let mut push = |kind: BindingKind, width: usize| {
        // the value at flat position `q` weighs `w^(N-1-q)`: within a row the
        // weights descend, and the row step is `w^width`
        let ws: Vec<SecureField> =
            (0..width).map(|k| w.pow((width - 1 - k) as u128)).collect();
        specs.push(TieSpec { kind, w: ws, step: w.pow(width as u128),
                             claimed: SecureField::zero() });
    };
    for (g, gr) in prog.groups.iter().enumerate() {
        push(BindingKind::In(g), gr.n);
    }
    for (oi, o) in prog.outs.iter().enumerate() {
        push(BindingKind::Out(oi), o.len());
    }
    specs
}

/// One component, proved on its own: the measurement and the probes both run here.
pub fn run(w: &ResidualStreamWitness, prog: &Prog, round: Round, kmax: u32,
           tamper: Option<Tamper>) -> Outcome {
    let t0 = std::time::Instant::now();
    let uns = prog.un_kinds();
    let mut tables = ResidualStreamTables::new(kmax, round, &uns);
    let component = residual_stream_component(w, prog, round, "r", &mut tables, tamper);

    // ---- tree 0: the component's selector, then every table ----------------------
    let mut ids = vec![residual_stream_sel("r", "nofirst"), residual_stream_sel("r", "end")];
    // stream II: and the component's own preprocessed values, in `residual_stream_component`'s
    // order -- the standalone rung has to name them too or tree 0 has columns
    // the allocator has never heard of
    for i in 0..prog.n_pre {
        ids.push(residual_stream_pre_col("r", i, 0));
        ids.push(residual_stream_pre_col("r", i, 1));
    }
    let mut pre: Vec<Eval> = component.pre.clone();
    ids.extend(residual_stream_table_ids(&uns));
    pre.extend(residual_stream_table_cols(&tables));
    let pad = |n: u32| 1usize << n;
    {
        let get: [fn(&bf16::NormRow) -> u32; 6] =
            [|r| r.m2, |r| r.ph, |r| r.st, |r| r.m7, |r| r.expfield, |r| r.nz];
        for (t, g) in get.into_iter().enumerate() {
            ids.push(id(crate::dequant::NORM_COLS[t]));
            pre.push(eval_of(tables.log_norm, (0..pad(tables.log_norm))
                .map(|i| m31_u32(tables.norm.get(i).map(g).unwrap_or(0))).collect()));
        }
        for t in 0..3 {
            ids.push(id(crate::dequant::FRAC_COLS[t]));
            pre.push(eval_of(tables.log_frac, (0..pad(tables.log_frac))
                .map(|i| {
                    let x = tables.frac.get(i).copied().unwrap_or((0, 0, 0));
                    m31_u32([x.0, x.1, x.2][t])
                }).collect()));
        }
    }
    let table_cells_pre: usize = pre[component.pre.len()..].iter().map(|e| e.domain.size()).sum();

    // ---- tree 1: the component, then the multiplicity columns --------------------
    let mut base: Vec<Eval> = component.base.clone();
    let component_cells: usize = base.iter().map(|e| e.domain.size()).sum();
    let mult_start = base.len();
    base.extend(residual_stream_mult_cols(&tables));
    base.push(eval_of(tables.log_norm, tables.m_norm.iter().map(|m| m31_u32(*m)).collect()));
    base.push(eval_of(tables.log_frac, tables.m_frac.iter().map(|m| m31_u32(*m)).collect()));
    let table_cells_fresh: usize = base[mult_start..].iter().map(|e| e.domain.size()).sum();
    let trace_secs = t0.elapsed().as_secs_f64();

    // ---- one transcript --------------------------------------------------------
    let widest = pre.iter().chain(base.iter()).map(|e| e.domain.log_size()).max().unwrap();
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);
    sess.commit(pre.clone());
    sess.commit(base.clone());
    let rels = ResidualStreamRels {
        norm: sess.draw_relation(NormRel::draw),
        frac: sess.draw_relation(FracRel::draw),
        pow: sess.draw_relation(PowRel::draw),
        smul: sess.draw_relation(SMulRel::draw),
        un: uns.iter().map(|_| sess.draw_relation(UnRel::draw)).collect(),
        uns: uns.clone(),
        val: sess.draw_relation(ValRel::draw),
    };
    let ch = sess.draw_felts(2);
    let (w_ch, lam) = (ch[0], ch[1]);
    let notie = std::env::var("ZK_RESIDUAL_STREAM_NO_BINDING").is_ok();
    let specs = plain_bindings(prog, component.log_rows, w_ch);
    let tie0 = Binding { lam, specs: specs.clone() };
    // B11 stream M: the eval the LogUp half lowers, built before the claims
    // exist. `compose` does the same thing for the same reason (`claims_for`'s
    // placeholders): a claimed sum enters constraints and never a lookup, so
    // the fraction IR this produces is the one `genemit` compiled.
    let air0 = ResidualStreamEval {
        log_rows: component.log_rows, tag: "r".into(), prog: prog.clone(), round,
        norm: rels.norm.clone(), frac: rels.frac.clone(), pow: rels.pow.clone(),
        smul: rels.smul.clone(), un: rels.un.clone(), uns: rels.uns.clone(),
        val: rels.val.clone(),
        binding: if notie { None } else { Some(tie0.clone()) }, bound: None,
        // the standalone gadget has no GKR backend; the flag lives in `compose`
        gkr: false,
    };
    let it = residual_stream_interaction(&component, &rels, if notie { None } else { Some(&tie0) },
                                         &air0, false);
    let inter_cells = it.cols.len() << component.log_rows;
    let binding = Binding {
        lam,
        specs: specs
            .iter()
            .enumerate()
            .map(|(i, s)| TieSpec { claimed: it.claimed[i], ..s.clone() })
            .collect(),
    };

    // an independent recomputation of every claim, from the witness rather than
    // from the trace: a binding is only worth something if it is checkable
    let binding_ok = notie
        || binding.specs.iter().all(|s| component.binding_claim(s, lam) == s.claimed);

    let mut sums = Sums { uses: it.use_sum, tables: SecureField::zero() };
    // B11 stream M: this rung is the SimdBackend probe path and commits through
    // a `Session`, so the router's columns come back to the host here. They are
    // already the host's -- the rung has no device witness generation for `frac_cols` to
    // notice -- and `into_host` is loud if that ever stops being true.
    let mut icols: Vec<Eval> = it.cols.into_iter().map(|c| c.into_host()).collect();

    // the tables' own sides
    let mut kinds = residual_stream_table_kinds(&tables.uns, &rels);
    kinds.push((tables.log_norm, crate::dequant::NORM_COLS.iter().map(|c| id(*c)).collect(),
                GKind::Norm(rels.norm.clone())));
    kinds.push((tables.log_frac, crate::dequant::FRAC_COLS.iter().map(|c| id(*c)).collect(),
                GKind::Frac(rels.frac.clone())));
    let mults: Vec<&Eval> = base[mult_start..].iter().collect();
    let mut tsums: Vec<SecureField> = vec![];
    let mut tkinds: Vec<(u32, Vec<PreProcessedColumnId>, GKind)> = vec![];
    for (ki, (lr, cs, kind)) in kinds.into_iter().enumerate() {
        let refs: Vec<&Eval> = cs.iter().map(|c| &pre[pre_index(&ids, c)]).collect();
        let (c, sum) = table_side(lr, &kind, &refs, mults[ki]);
        icols.extend(c);
        sums.tables += sum;
        tsums.push(sum);
        tkinds.push((lr, cs, kind));
    }
    let n_inter = icols.len();
    sess.commit(icols);

    // ---- components -------------------------------------------------------------
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let comp = FrameworkComponent::<ResidualStreamEval>::new(
        alloc,
        ResidualStreamEval { log_rows: component.log_rows, tag: "r".into(), prog: prog.clone(), round,
                   norm: rels.norm.clone(), frac: rels.frac.clone(), pow: rels.pow.clone(),
                   smul: rels.smul.clone(), un: rels.un.clone(), uns: rels.uns.clone(),
                   val: rels.val.clone(),
                   binding: if notie { None } else { Some(binding) }, bound: None, gkr: false },
        it.use_sum,
    );
    let tcomps: Vec<FrameworkComponent<GTableEval>> = tkinds
        .into_iter()
        .zip(&tsums)
        .map(|((lr, cs, kind), sum)| {
            FrameworkComponent::<GTableEval>::new(
                alloc, GTableEval { log_rows: lr, cols: cs, kind }, *sum)
        })
        .collect();

    let mut provers: Vec<&dyn ComponentProver<crate::common::PcsB>> =
        vec![&comp];
    let mut comps: Vec<&dyn Component> = vec![&comp];
    for t in &tcomps {
        provers.push(t);
        comps.push(t);
    }
    let n_columns = [pre.len(), base.len(), n_inter];
    let proven = sess.prove(&provers, &comps);
    Outcome {
        proven, sums, prog: prog.clone(), round,
        n_ops: component.n_ops, n_leaves: component.n_leaves,
        off_table: tables.off_table, bad_exp: component.bad_exp, fired: component.fired,
        ref_ok: component.ref_ok, ref_n: component.ref_n, odd: w.odd,
        table_cells_pre, table_cells_fresh, component_cells, inter_cells,
        n_columns, binding_ok, trace_secs,
    }
}

pub fn pre_index(ids: &[PreProcessedColumnId], want: &PreProcessedColumnId) -> usize {
    ids.iter().position(|i| i == want).expect("preprocessed column not registered")
}

/// The tables alone, checked against an independent evaluation: `smul` against
/// f32's own product, `un` against f32's own op, exactly as B6 checks `NORM`
/// against a single f64 rounding.
pub fn check_tables(round: Round, samples: usize) -> Value {
    let smul = smul_table(round);
    let vals = smul_vals();
    let mut smul_bad = vec![];
    for (i, row) in smul.iter().enumerate() {
        if i % (smul.len() / samples.max(1)).max(1) != 0 {
            continue;
        }
        let (a, b) = (vals[i / vals.len()], vals[i % vals.len()]);
        if a == 0 || b == 0 {
            continue;
        }
        // one exponent apiece, so the product's significand is the table's
        let want = bf16::bf16_mul_ref(join(a, 134), join(b, 134));
        let got = join(sgn_of(row[2]), (134 + 134 + row[3].0) as u32 - 127);
        if want != got {
            smul_bad.push(json!({"a": a, "b": b, "want": want, "got": got}));
        }
    }
    let (mut un_bad, mut sub_out) = (vec![], 0usize);
    for k in [UnKind::Square, UnKind::Rsqrt, UnKind::Silu] {
        let t = un_table(k);
        for (i, row) in t.iter().enumerate() {
            if i % (t.len() / samples.max(1)).max(1) != 0 {
                continue;
            }
            let x = to_f64(sgn_of(row[0]), row[1].0);
            // bf16 subnormals are one value in this representation (`sm = 0`),
            // so the comparison is between *values*, not patterns; the count of
            // inputs whose true image is subnormal is reported instead
            let want = bf16::from_f32(k.apply(x) as f32);
            let got = join(sgn_of(row[2]), row[3].0);
            if is_odd_pattern(want) {
                sub_out += 1;
            } else if canon(want) != canon(got) {
                un_bad.push(json!({"op": k.name(), "x": x, "want": want, "got": got}));
            }
        }
    }
    json!({"smul_rows": smul.len(), "smul_mismatch": smul_bad,
           "val_rows": val_table().len(), "pow_rows": pow_table().len(),
           "un_mismatch": un_bad, "un_subnormal_image": sub_out,
           "note": "a subnormal image is zero in the (sm, ef) representation, so it is counted rather than called a mismatch"})
}

// ------------------------------------------------------- the norm's layout ----

/// How the composed proof tiles one `(tokens x d)` bf16 stream into rows.
///
/// `tree_sum` is `x[..h] + x[h..]` on the array padded to `pad = pad2(d)`, so its
/// first `log2(b)` levels group the channels **by residue mod `a`**, not into
/// contiguous blocks: after those levels position `r` holds the stride tree of
/// `x[r], x[r+a], ..., x[r+(b-1)a]`.  So a row here is one `(residue, token)`
/// pair, its `nreal = d/a` committed leaves are the channels `r + j a`, and a
/// second component over the `a` roots of one token is the rest of the reduction --
/// the composite is *bit-identical* to `tree_sum`.
///
/// The canonical position of channel `c` of token `t` is `c * tokens + t`
/// (channel-major), which is exactly what makes the Horner binding's weights plain
/// powers in both tilings: in this one the row index is `r * tokens + t` and the
/// leaf `j` sits `j a tokens` positions later.
#[derive(Clone, Copy, Debug)]
pub struct Tiling {
    pub tokens: usize,
    pub d: usize,
    /// rows per token at level 1, and the level-2 tree's width
    pub a: usize,
    /// the level-1 tree's width, `pad2(d) / a`
    pub b: usize,
    /// committed leaves per level-1 row, `d / a`
    pub nreal: usize,
}

impl Tiling {
    pub fn new(tokens: usize, d: usize, a: usize) -> Self {
        let pad = d.next_power_of_two();
        assert!(a.is_power_of_two() && a >= 1, "the tiling's a must be a power of two");
        assert!(tokens.is_power_of_two(), "the tiling needs a power-of-two token count");
        assert_eq!(d % a, 0, "the tiling's a = {a} must divide d = {d}");
        let b = pad / a;
        let nreal = d / a;
        assert!(nreal >= 2 && nreal <= b, "d/a = {nreal} does not fit a tree of {b}");
        Self { tokens, d, a, b, nreal }
    }
    pub fn rows1(&self) -> usize {
        self.tokens * self.a
    }
    /// The natural-order value block a level-1 component's group reads: row
    /// `r * tokens + t`, leaf `j` is channel `r + j a` of token `t`.
    pub fn lay(&self, src: &[u16]) -> Vec<u16> {
        let mut v = vec![0u16; self.rows1() * self.nreal];
        for r in 0..self.a {
            for t in 0..self.tokens {
                let m = r * self.tokens + t;
                for j in 0..self.nreal {
                    v[m * self.nreal + j] = canon(src[t * self.d + r + j * self.a]);
                }
            }
        }
        v
    }
    /// The inverse: a level-1 group's block back to `(token, channel)` order.
    pub fn unlay(&self, v: &[u16]) -> Vec<u16> {
        let mut out = vec![0u16; self.tokens * self.d];
        for r in 0..self.a {
            for t in 0..self.tokens {
                let m = r * self.tokens + t;
                for j in 0..self.nreal {
                    out[t * self.d + r + j * self.a] = v[m * self.nreal + j];
                }
            }
        }
        out
    }
    /// The level-1 roots, one per row, as the level-2 component's leaves: row `t`,
    /// leaf `r`.
    pub fn roots_to_l2(&self, roots: &[u16]) -> Vec<u16> {
        let mut v = vec![0u16; self.tokens * self.a];
        for r in 0..self.a {
            for t in 0..self.tokens {
                v[t * self.a + r] = roots[r * self.tokens + t];
            }
        }
        v
    }
}

/// Read a `(tokens x d)` bf16 column off a witness directory.
pub fn read_u16(dir: &str, name: &str) -> Result<Vec<u16>, String> {
    let raw = std::fs::read(format!("{dir}/{name}.u16")).map_err(|e| format!("{name}.u16: {e}"))?;
    Ok(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
}

/// What the reduction's order costs numerically: B1's fp32 `sum(x^2)/d + eps`
/// against the balanced bf16 tree this AIR proves, on the same real column.
/// `n1_in` is `store(ms + eps)`, so the comparison is one bf16 add away from the
/// mean itself and is the number the spec change has to be judged on.  With
/// `roots` supplied it is the *composed* two-level reduction that is compared,
/// which is the check that the tiling above really is `tree_sum`'s order.
pub fn norm_drift(dir: &str, d: usize, tokens_cap: usize, round: Round, which: &str,
                  roots: Option<&[u16]>) -> Result<Value, String> {
    let x = read_u16(dir, if which == "n2" { "xmid" } else { "xin" })?;
    let want = read_u16(dir, if which == "n2" { "n2_in" } else { "n1_in" })?;
    let man: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json"))
            .map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let eps = man["residual_stream"]["eps"].as_u64().ok_or("manifest: residual_stream.eps")? as u16;
    let inv_d = man["residual_stream"]["inv_d"].as_u64().ok_or("manifest: residual_stream.inv_d")? as u16;
    let tokens = (x.len() / d).min(tokens_cap.max(1)).max(1);
    let pad = d.next_power_of_two();
    let (mut same, mut worst, mut worst_ulp) = (0usize, 0f64, 0i64);
    let (mut component_same, mut component_n) = (0usize, 0usize);
    for t in 0..tokens {
        // the tree, exactly as `tree_sum` builds it: pad with +0, halve, round
        let mut lv: Vec<f64> = (0..pad)
            .map(|c| {
                if c >= d {
                    0.0
                } else {
                    let (sm, ef) = split(x[t * d + c]);
                    let sq = bf16::from_f32(UnKind::Square.apply(to_f64(sm, ef)) as f32);
                    let (a, b) = split(sq);
                    to_f64(a, b)
                }
            })
            .collect();
        while lv.len() > 1 {
            let h = lv.len() / 2;
            lv = (0..h)
                .map(|i| {
                    let s = bf16::from_f64_round(lv[i] + lv[i + h], round);
                    let (a, b) = split(s);
                    to_f64(a, b)
                })
                .collect();
        }
        let root = bf16::from_f64_round(lv[0], round);
        if let Some(rt) = roots {
            component_n += 1;
            if canon(rt[t]) == canon(root) {
                component_same += 1;
            }
        }
        let (im, ie) = split(inv_d);
        let ms = bf16::from_f64_round(lv[0] * to_f64(im, ie), round);
        let (mm, me) = split(ms);
        let (em, ee) = split(eps);
        let got = bf16::from_f64_round(to_f64(mm, me) + to_f64(em, ee), round);
        let w = want[t];
        if got == canon(w) {
            same += 1;
        } else {
            let (gs, ge) = split(got);
            let (ws, we) = split(w);
            let (g, wv) = (to_f64(gs, ge), to_f64(ws, we));
            let rel = if wv != 0.0 { ((g - wv) / wv).abs() } else { 0.0 };
            worst = worst.max(rel);
            worst_ulp = worst_ulp.max((got as i64 - w as i64).abs());
        }
    }
    Ok(json!({"norm": which, "tokens": tokens, "d": d, "identical": same,
              "differ": tokens - same,
              "worst_relative": worst, "worst_ulps": worst_ulp,
              "two_level_checked": component_n, "two_level_matches_tree_sum": component_same,
              "note": "B1's fp32 reduction against the balanced bf16 tree the AIR proves; \
                       two_level_* compares the composed components' root with tree_sum's own"}))
}
