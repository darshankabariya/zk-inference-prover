//! B6.5 — the composed layer proof: one FS transcript over the attention bands,
//! the seven matmuls' dequant regions, the attention-output fold and the r^T W
//! walks, with the shared tables committed once — and the trust boundary closed.
//!
//! The **statement** is two serialized documents:
//!
//! * `Fixed` — produced at setup, trusted by the verifier: the shape and gadget
//!   parameters that determine every column size and selector, the security
//!   config, the tree column sizes, and the **pinned preprocessed root** (every
//!   table, selector, weight and scale column lives in tree 0, and B4 §3 / B5 /
//!   B6 all showed a corrupted table is invisible to everything else).
//! * `Claims` — produced by the prover: the Freivalds accumulator claims and the
//!   LogUp sums, exactly what `Session::mix` binds to the transcript.
//!
//! The **verifier** (`verify_composed`) takes those two byte strings and the
//! serialized proof, and nothing else: it replays the transcript, recomputes
//! `Powers` and `mu` from its own drawn challenges, rebuilds every component from
//! the parsed statement, runs stwo's `verify()`, checks the claimed sums cancel,
//! and checks the application identities:
//!
//!   per band b:   r_sum * am_b - hs_b == chan_b          (Q.K^T)
//!                 sn_b == az_b                            (denominator, in-band)
//!   attnout:      sum_b rv^(hd*off_b) * av_b == h_ao      (A.V, all bands)
//!                 sum_b r^(C*off_b)  * az_b == dz_ao      (den, all bands)
//!   per matmul m: h_m == acc_m                            (Y = W X, mod p)
//!
//! There is no prover-side struct comparison anywhere on the accept path.

use num_traits::{One, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, TreeVec};
use stwo::core::verifier::verify;
use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::ComponentProver;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::attn_air::{
    ChanEval, ExpRel, ExpTableEval, KeyEval, Offsets, Powers, RangeRel, RowEval, band_id, id,
    msk_id, powers, tbl_range, CHAN_SEL, CORNER_SEL, KEY_SEL, ROW_SEL, TBL_EXP_X, TBL_EXP_Y,
};
use crate::attn_trace::{self, Cfg, Committed, Layouts, table_col};
use crate::attn_witness::Variant as AttnVariant;
use crate::bf16::Round;
use crate::common::{Eval, Proof, RawStats, Session, eval_of, m31_u32, twiddles};
use crate::util::pcs_config;
use crate::dequant::{
    DEQ_SEL, DequantEval, FRAC_COLS, FracRel, MUL_COLS, MulRel, NORM_COLS, NormRel, PairRel,
    QNZ_COLS, Spec, TableEval, TableKind, Variant as DeqVariant, YSide, chan_col, deq_sel,
    range_widths,
};
use crate::layer::{
    AoLayout, Layer, MatLayout, Tables, WALK_GROUPS, WalkLayout, ao_interaction, ao_region,
    mat_interaction, mat_region, walk_interaction, walk_region,
};
use crate::glue::{self, GKind, GTableEval, GlueEval, GlueRels, GlueTables, Kind, Prog, Tie,
                  TieKind, TieSpec, Tiling, UnKind};
use crate::layout::Shape;
use crate::walk_air::{AO_SEL, AttnOutEval, WALK_SEL, WalkEval, ao_chan, ao_sel, walk_sel, walk_w};

// ---------------------------------------------------------------- relations ----

/// Every relation of the composed proof, drawn in one canonical order.
pub struct Rels {
    pub exp: ExpRel,
    pub norm: NormRel,
    pub frac: FracRel,
    pub mul: MulRel,
    pub qnz: PairRel,
    pub widths: Vec<u32>,
    pub ranges: Vec<RangeRel>,
    /// B10 step 2: `POW`, `SMUL`, `VAL` and one unary table per kind.  `NORM` and
    /// `FRAC` are *not* redrawn -- the glue reads the very relations the dequant
    /// regions do, which is what makes the two families share one multiplicity
    /// column per shared table.  Absent when the layer carries no glue, so a
    /// proof without it draws exactly what B6.5 drew.
    pub glue: Option<GlueRels>,
    /// Stream BB: the embedding relation, drawn **last** so that a proof
    /// without the gadget draws exactly what it always drew.  Its arity is the
    /// channel block's width plus two, which is why it is not a `relation!`.
    pub emb: Option<crate::embed::EmbRel>,
}

impl Rels {
    /// `uns` is the plan's unary-kind list: the relations are drawn one per
    /// entry, in that order, and both sides index them by it.  `emb` is the
    /// embedding relation's arity, or `None` when the gadget is off.
    pub fn draw(ch: &mut Blake2sM31Channel, widths: &[u32],
                uns: Option<&[crate::glue::UnKind]>, emb: Option<usize>) -> Self {
        let exp = ExpRel::draw(ch);
        let norm = NormRel::draw(ch);
        let frac = FracRel::draw(ch);
        let mul = MulRel::draw(ch);
        let qnz = PairRel::draw(ch);
        let ranges: Vec<RangeRel> = widths.iter().map(|_| RangeRel::draw(ch)).collect();
        let glue = uns.map(|ks| GlueRels {
            norm: norm.clone(),
            frac: frac.clone(),
            pow: crate::glue::PowRel::draw(ch),
            smul: crate::glue::SMulRel::draw(ch),
            val: crate::glue::ValRel::draw(ch),
            un: ks.iter().map(|_| crate::glue::UnRel::draw(ch)).collect(),
            uns: ks.to_vec(),
        });
        let emb = emb.map(|n| crate::embed::EmbRel::draw(ch, n));
        Self { exp, norm, frac, mul, qnz, widths: widths.to_vec(), ranges, glue, emb }
    }
    pub fn range(&self, bits: u32) -> &RangeRel {
        &self.ranges[self.widths.iter().position(|w| *w == bits).unwrap()]
    }
}

// ---------------------------------------------------------------- statement ----

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MatParams {
    pub name: String,
    pub variant: String,
    pub n_in: usize,
    pub n_out: usize,
    pub tokens: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Params {
    pub s: usize,
    pub c: usize,
    pub hd: usize,
    pub nh: usize,
    pub bands: Vec<usize>,
    pub variant: String,
    pub log_table: u32,
    pub q_bits: u32,
    pub k_bits: u32,
    pub nmax: u32,
    pub chan_target: usize,
    pub grid_target: usize,
    pub rw: usize,
    pub z_lo_bits: u32,
    pub z_hi_bits: u32,
    pub z_min: u32,
    pub mats: Vec<MatParams>,
    pub deq_target: usize,
    pub kmax: u32,
    pub round: String,
    pub ao_target: usize,
    pub ao_q_bits: u32,
    pub ao_rem_bits: u32,
    /// Stream Z: the fold's two data-derived pins, present only on a statement
    /// with a `fold` matmul (a witness whose manifest declares one).  Zero --
    /// and skipped in the serialization, so every pre-change statement is
    /// byte-identical -- everywhere else.  `fold_rem_bits` is the width of the
    /// high remainder's range check (of `2 fdp`); `fold_alpha` is where the
    /// divisor splits between the committed per-token `dent` and the
    /// preprocessed per-channel `pj` (`dequant::Spec::fold_alpha`).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub fold_rem_bits: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub fold_alpha: u32,
    pub walk_p: usize,
    /// B10 step 2.  Absent on a witness `extract.py --glue` did not write.
    #[serde(default)]
    pub glue: Option<GlueParams>,
    /// B10 step 3: the causal corner rides as a second set of bands over the same
    /// heads, `C = S`, masked.  Absent on a witness extracted without it.
    #[serde(default)]
    pub corner: bool,
    /// B11 stream Q: prove the attention bands' lookups with LogUp-GKR instead
    /// of committing their running-sum columns. A flag and not an env mode,
    /// because it changes the proof: the GKR arm has its own digests by design.
    /// Skipped when false, so a statement written with the flag off is byte
    /// identical to one from a crate that never had it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gkr_bands: bool,
    /// B11 stream U, lever 3: the same for the *dequant* family, composing with
    /// `gkr_bands` rather than replacing it -- the two families' instances join
    /// one `prove_batch` call and their heights one link. A flag and not an env
    /// mode for the same reason: it changes the proof, and its oracle is its
    /// own. Skipped when false, so every statement written without it is byte
    /// identical to one from a crate that never had it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gkr_dequant: bool,
    /// B11 stream V, lever 3: the glue family's lookups proved by the same
    /// batched GKR, under `--gkr-glue`. Composes with `--gkr-bands` and, like
    /// dequant's, is a loud configuration error without it.
    pub gkr_glue: bool,
    /// B11 stream V, lever 4: the walks' 8-bit activation range checks, and the
    /// attention-output region's three lookups per slot, under `--gkr-walks`
    /// and `--gkr-attnout`. Both compose with `--gkr-bands`.
    pub gkr_walks: bool,
    pub gkr_attnout: bool,
    /// Stream H: the layer's residual-stream input and output are each committed
    /// in a tree of their own, so that two chunks' boundaries can be compared
    /// from bytes.  Needs the wiring family, since the boundary *is* its two
    /// stream ties; absent, and a chunk composes exactly as it did before.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bounds: bool,
    /// Stream AA (E11): the chunk's keys and values are committed in boundary
    /// trees of their own, one **out** side and one **in** side per head band
    /// group, so that "the keys chunk `(L, b+1)` attended to are the ones chunk
    /// `(L, b)` proved" is checkable from serialized bytes by root equality --
    /// the token axis's version of what `bounds` does for the layer axis.
    ///
    /// It implies `corner`, and it *tiles the declared prefix into sub-bands of
    /// exactly `S` keys* (layout design A), because a V column's height is a
    /// function of the chunk's own `C` and a producer's corner V column and a
    /// consumer's untiled prefix V column can therefore never be equal.  So the
    /// flag changes the band decomposition, not only the tree count.  Skipped
    /// when false, so every statement written without it is byte identical to
    /// one from a crate that never had the feature.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kv_bind: bool,
    /// Stream E: publish the head's order-key claim.  It sits here rather than in
    /// `GlueParams` because a head dump carries no wiring columns at all -- the
    /// two features share the tie challenge, not the witness.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub head_key: bool,
    /// Stream E: compose the argmax region into this proof, tied to the head's
    /// key claim.  Implies `head_key` -- that claim is its counterparty.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub head_argmax: bool,
    /// the weaker statement ("the token attains the maximum"), which is what the
    /// tie probe separates from the strict one
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub head_argmax_lax: bool,
    /// Stream BB (E11): the embedding gadget's vocabulary, and zero when the
    /// gadget is off -- which is what the plan reads as "no embedding region".
    /// The gadget proves that the layer's committed residual-stream input is
    /// row `tok[t]` of an embedding matrix pinned in tree 0, so it needs the
    /// wiring family (its tie's counterparty is `glue_ties[stream_in]`) and it
    /// belongs on the layer-0 chunk of a block.  Skipped when zero, so every
    /// statement written without it is byte identical to one from a crate that
    /// never had the feature.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub embed_vocab: usize,
    /// Channels per row of both embedding regions.  It is the lookup tuple's
    /// arity minus two, so it is the one number in the gadget the generated
    /// kernel's per-row width depends on; `EmbLayout::choose` rounds it down to
    /// `d` over a power of two.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub embed_per_row: usize,
}

/// The wiring family's two shape numbers.  `a` is how many rows one token's
/// residual stream occupies at the reduction's first level: `tree_sum` groups the
/// channels by residue mod `a`, so a row's committed leaves are the channels
/// `r, r+a, ...` and the second level is one row per token over the `a` roots.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct GlueParams {
    pub d: usize,
    pub a: usize,
    /// The FFN width, for the SwiGLU's own tiling.  Zero on a dump whose
    /// manifest predates it, which is what `swiglu` then refuses to run on.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub dff: usize,
    /// Stream E's two additive regions, off by default.  They are opt-in rather
    /// than opt-out so that the pre-change proof is reproduced by the *default*
    /// invocation -- the identity control is then the ordinary run, not a run
    /// with a flag in it.  Both are skipped when false so the serialized
    /// statement is byte-identical to the pre-change crate's.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tails: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub swiglu: bool,
    /// Stream E: make `o_out` and `d_out` the `o` and `down` dequant regions'
    /// own published outputs, tied to the copies the wiring family commits.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub deq_ties: bool,
}

fn is_zero(v: &usize) -> bool {
    *v == 0
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// The trusted half of the statement, produced at setup.
#[derive(Serialize, Deserialize, Clone)]
pub struct Fixed {
    pub params: Params,
    pub log_blowup: u32,
    pub n_queries: usize,
    pub pow_bits: u32,
    /// column log sizes per tree, in commit order
    pub sizes: Vec<Vec<u32>>,
    /// the pinned root of tree 0: tables, selectors, weights and scales
    pub pre_root: String,
}

/// The prover-supplied half: every claimed value the identities compare.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Claims {
    /// per band: chan, hs, av, sn, am, az
    pub bands: Vec<[[u32; 4]; 6]>,
    pub ao_out: [u32; 4],
    pub ao_den: [u32; 4],
    /// per matmul: the dequant region's Freivalds side
    pub mat_h: Vec<[u32; 4]>,
    /// per walk group, per matmul: the walk's side
    pub walk_acc: Vec<Vec<[u32; 4]>>,
    /// per component with lookups, in component order
    pub use_sums: Vec<[u32; 4]>,
    /// per table component, in table order
    pub table_sums: Vec<[u32; 4]>,
    /// B10 step 2: per glue region, per tie, the Horner claim over the values it
    /// carries.  The verifier's equalities between them are what say the residual
    /// stream is threaded; `glue.stream_in` and `glue.stream_out` name the two
    /// that are the layer's boundary.
    #[serde(default)]
    pub glue_ties: Vec<Vec<[u32; 4]>>,
    /// stream E: one per dequant output tie, in `GluePlan::out_ties` order --
    /// the value the `o` and `down` regions publish for their own bf16 outputs
    #[serde(default)]
    pub deq_out: Vec<[u32; 4]>,
    /// stream E: the head's order-key claim, when it is published
    #[serde(default)]
    pub head_key: Vec<[u32; 4]>,
    /// stream E: the argmax region's claim over the keys it compared -- equal to
    /// `head_key` is what says it compared the head's own output
    #[serde(default)]
    pub am_key: Vec<[u32; 4]>,
    /// stream BB: the embedding region's Horner claim over the residual stream
    /// it proved.  Equal to `glue_ties[stream_in]` is what says the row it
    /// proved is the row the layer consumed.
    #[serde(default)]
    pub embed_out: Vec<[u32; 4]>,
    /// stream BB: the same region's claim over the committed token ids, so the
    /// ids are a named value of the statement and a transcript anchor has
    /// something to bind to.  Nothing inside this proof compares it.
    #[serde(default)]
    pub embed_tok: Vec<[u32; 4]>,
    /// B11 stream Q: the bands' batch GKR proof, as bytes, on the GKR arm only.
    /// It rides here because the statement already travels everywhere the proof
    /// does and `Claims::flat` -- what the transcript mixes -- reads only the
    /// felt fields, so carrying it changes no channel state. Absent with the
    /// flag off, which is what keeps a flag-off statement byte identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gkr: Option<Vec<u8>>,
}

pub fn qm(v: SecureField) -> [u32; 4] {
    let a = v.to_m31_array();
    [a[0].0, a[1].0, a[2].0, a[3].0]
}
pub fn unqm(a: [u32; 4]) -> SecureField {
    SecureField::from_m31_array([
        M31::from_u32_unchecked(a[0]),
        M31::from_u32_unchecked(a[1]),
        M31::from_u32_unchecked(a[2]),
        M31::from_u32_unchecked(a[3]),
    ])
}

impl Claims {
    /// The canonical flattening mixed into the transcript and replayed by the
    /// verifier; any tampering diverges the channel and stwo's verify fails.
    pub fn flat(&self) -> Vec<SecureField> {
        let mut v = vec![];
        for b in &self.bands {
            v.extend(b.iter().map(|x| unqm(*x)));
        }
        v.push(unqm(self.ao_out));
        v.push(unqm(self.ao_den));
        v.extend(self.mat_h.iter().map(|x| unqm(*x)));
        for w in &self.walk_acc {
            v.extend(w.iter().map(|x| unqm(*x)));
        }
        v.extend(self.use_sums.iter().map(|x| unqm(*x)));
        v.extend(self.table_sums.iter().map(|x| unqm(*x)));
        for g in &self.glue_ties {
            v.extend(g.iter().map(|x| unqm(*x)));
        }
        // stream E, appended last: empty unless the dequant output ties are on,
        // so the transcript is unchanged without them
        v.extend(self.deq_out.iter().map(|x| unqm(*x)));
        v.extend(self.head_key.iter().map(|x| unqm(*x)));
        v.extend(self.am_key.iter().map(|x| unqm(*x)));
        // stream BB, appended last: empty unless the embedding gadget is on, so
        // the transcript is unchanged without it
        v.extend(self.embed_out.iter().map(|x| unqm(*x)));
        v.extend(self.embed_tok.iter().map(|x| unqm(*x)));
        v
    }
}

/// `w^e` for a signed exponent: an output side's row step points backwards
/// through the stream, so half of these are inverses.
fn wpow(w: SecureField, e: i128) -> SecureField {
    if e >= 0 { w.pow(e as u128) } else { w.pow((-e) as u128).inverse() }
}

/// Stream E: one matmul's output-side weights, from the wiring family's own tie
/// challenge -- the consumer builds its claim from the same `w`, which is what
/// makes the two comparable.  The value at channel `c` of token `t` carries
/// `w^(N - 1 - (c tokens + t))`; that splits into a per-slot weight and a row
/// step of two constants, derived in `dequant::OSide`.
fn oside_w(w: SecureField, lam: SecureField, tokens: usize, n_out: usize,
           per_row: usize, kind: crate::dequant::OKind) -> crate::layer::OSideW {
    let n = (tokens * n_out) as i128;
    let (tok, pr, no) = (tokens as i128, per_row as i128, n_out as i128);
    crate::layer::OSideW {
        w: (0..per_row).map(|s| wpow(w, n - 1 - s as i128 * tok)).collect(),
        lam,
        k_in: wpow(w, -(pr * tok)),
        k_bd: wpow(w, (no - pr) * tok - 1),
        kind,
    }
}

// -------------------------------------------------------------------- plan ----

pub struct WalkPlan {
    pub name: String,
    pub mats: Vec<usize>,
    pub lay: WalkLayout,
    /// Stream Z: the group's committed-activation range width -- 8 everywhere,
    /// `ao_q_bits + 1` on `o`'s group when the attention output is widened.
    pub act_bits: u32,
}

// ------------------------------------------------------------ the glue plan ----

/// One region of the wiring family, as both sides derive it from `Params`.
pub struct GlueRegionPlan {
    pub tag: String,
    pub prog: Prog,
    pub log_rows: u32,
    /// per tie: what it accumulates, the within-row weight *exponents* and the
    /// row-step exponent -- powers of the challenge the transcript draws, so the
    /// verifier builds the same weights the prover did
    pub ties: Vec<(TieKind, Vec<u128>, u128)>,
}

/// B6.5 open gap 1, as four regions and three equalities.
///
/// ```text
///   gn1   xin, o_out  ->  norm1's level-1 roots, and xmid
///   gn1b  those roots ->  norm1's mean-of-squares
///   gn2   xmid, d_out ->  norm2's level-1 roots, and xout
///   gn2b  those roots ->  norm2's mean-of-squares
/// ```
///
/// `gn1` proves the reduction's first `log2(b)` levels **and** the residual add
/// on the same row, so the residual stream is committed once and read by both;
/// the three equalities are `gn1.roots == gn1b.in`, `gn1.xmid == gn2.in` and
/// `gn2.roots == gn2b.in`.  What is left open is named in the record: the two
/// norms' tails (`* r`, `* w`), the SwiGLU, and the ties from `o`'s and `down`'s
/// dequant regions into `o_out` and `d_out`.
pub struct GluePlan {
    pub t: Tiling,
    /// the SwiGLU's own tiling, over `dff` instead of `d`
    pub tff: Option<Tiling>,
    pub regions: Vec<GlueRegionPlan>,
    /// `((region, tie), (region, tie))` the verifier compares
    pub eqs: Vec<((usize, usize), (usize, usize))>,
    pub uns: Vec<UnKind>,
    /// the layer's committed residual-stream input and output claims
    pub stream_in: (usize, usize),
    pub stream_out: (usize, usize),
    /// stream E: the region index of each norm's tail, and of the SwiGLU, so the
    /// accounting can name their cells apart from the reduction's
    pub tails: Vec<usize>,
    pub swiglu: Option<usize>,
    /// stream E: `((region, tie), name)` per dequant output tie -- the wiring
    /// side's claim over `o_out` and over `d_out`, and the matmul whose region
    /// has to publish the same value
    pub out_ties: Vec<((usize, usize), &'static str)>,
}

impl GluePlan {
    pub fn new(p: &Params, gp: &GlueParams) -> Self {
        let t = Tiling::new(p.s, gp.d, gp.a);
        let (a, tok, nreal) = (t.a, t.tokens, t.nreal);
        // the canonical stream position of channel `c` of token `t` is
        // `c * tokens + t`, so a level-1 row's leaf `j` sits `j * a * tokens`
        // positions past the row's own index and the row step is one
        let stream: Vec<u128> =
            (0..nreal).map(|j| (a * tok * (nreal - 1 - j)) as u128).collect();
        let l2_in: Vec<u128> = (0..a).map(|r| (tok * (a - 1 - r)) as u128).collect();
        let one = vec![0u128];
        let l1 = Prog::norm_resid(nreal, t.b);
        let l2 = Prog::tree(a, a, UnKind::None);
        let lr1 = crate::common::log_rows_for(t.rows1(), 1);
        let lr2 = crate::common::log_rows_for(tok, 1);
        let mut regions = vec![];
        for (n1, n2) in [("gn1", "gn1b"), ("gn2", "gn2b")] {
            regions.push(GlueRegionPlan {
                tag: n1.into(),
                prog: l1.clone(),
                log_rows: lr1,
                ties: vec![
                    (TieKind::In(0), stream.clone(), 1),
                    (TieKind::Out(0), one.clone(), 1),
                    (TieKind::Out(1), stream.clone(), 1),
                ],
            });
            regions.push(GlueRegionPlan {
                tag: n2.into(),
                prog: l2.clone(),
                log_rows: lr2,
                ties: vec![(TieKind::In(0), l2_in.clone(), 1),
                           (TieKind::Out(0), one.clone(), 1)],
            });
        }
        let mut eqs = vec![((0, 1), (1, 0)), ((0, 2), (2, 0)), ((2, 1), (3, 0))];
        let mut uns = l1.un_kinds();
        // ---- stream E: the dequant output ties -------------------------------
        //
        // `gn1`'s group 1 *is* `o_out` and `gn2`'s is `d_out`, and neither has
        // ever carried a Horner claim -- the wiring family committed those
        // columns and only a `VAL` lookup pinned them.  One claim each, on the
        // same channel-major weights the stream uses, and the `o` and `down`
        // dequant regions publish the same value from their own outputs.
        let mut out_ties = vec![];
        if gp.deq_ties {
            for (ri, name) in [(0usize, "o"), (2usize, "down")] {
                regions[ri].ties.push((TieKind::In(1), stream.clone(), 1));
                out_ties.push(((ri, regions[ri].ties.len() - 1), name));
            }
        }
        // ---- stream E: the two norm tails ------------------------------------
        //
        // A tail's group 0 is the same block of the residual stream the
        // reduction's own group 0 is, in the same order and under the same
        // weights, so the tie `nt.In(0) == gn.In(0)` says the tail scaled the
        // very values the reduction squared -- which is the half of "the norm
        // computed its own output" a committed column cannot fake.  `Out(0)` is
        // published for the requantize region that will read it.
        let mut tails = vec![];
        if gp.tails {
            let tail = Prog::norm_tail(nreal);
            for (gn, tag) in [(0usize, "nt1"), (2usize, "nt2")] {
                let i = regions.len();
                regions.push(GlueRegionPlan {
                    tag: tag.into(),
                    prog: tail.clone(),
                    log_rows: lr1,
                    ties: vec![(TieKind::In(0), stream.clone(), 1),
                               (TieKind::Out(0), stream.clone(), 1)],
                });
                eqs.push(((i, 0), (gn, 0)));
                tails.push(i);
            }
        }
        // ---- stream E: the SwiGLU product -----------------------------------
        //
        // Over `dff`, so it carries its own tiling.  Its three claims have no
        // counterparty inside the glue yet: group 0 is the gate matmul's
        // dequantized output, group 1 is `up`'s, and the output is what `down`
        // requantizes.  All three land when the dequant ties and the requantize
        // placement do; until then the claims are published and compared against
        // B1's own columns, and the record says so.
        let mut tff = None;
        let mut swiglu = None;
        if gp.swiglu {
            assert!(gp.dff > 0, "the SwiGLU needs the manifest's dff");
            let t2 = Tiling::new(p.s, gp.dff, gp.a);
            let prog = Prog::pairs(t2.nreal, Kind::Mul, UnKind::Silu, UnKind::None);
            let ff: Vec<u128> = (0..t2.nreal)
                .map(|j| (t2.a * t2.tokens * (t2.nreal - 1 - j)) as u128)
                .collect();
            for k in prog.un_kinds() {
                if !uns.contains(&k) {
                    uns.push(k);
                }
            }
            swiglu = Some(regions.len());
            regions.push(GlueRegionPlan {
                tag: "gsw".into(),
                prog,
                log_rows: crate::common::log_rows_for(t2.rows1(), 1),
                ties: vec![(TieKind::In(0), ff.clone(), 1),
                           (TieKind::In(1), ff.clone(), 1),
                           (TieKind::Out(0), ff, 1)],
            });
            tff = Some(t2);
        }
        Self {
            t,
            tff,
            regions,
            eqs,
            uns,
            stream_in: (0, 0),
            stream_out: (2, 2),
            tails,
            swiglu,
            out_ties,
        }
    }

    /// The tie specs, with the claims the statement carries filled in.
    pub fn ties(&self, w: SecureField, lam: SecureField, claims: &Claims) -> Vec<Tie> {
        self.regions
            .iter()
            .enumerate()
            .map(|(i, r)| Tie {
                lam,
                specs: r
                    .ties
                    .iter()
                    .enumerate()
                    .map(|(k, (kind, we, se))| TieSpec {
                        kind: *kind,
                        w: we.iter().map(|e| w.pow(*e)).collect(),
                        step: w.pow(*se),
                        claimed: claims
                            .glue_ties
                            .get(i)
                            .and_then(|g| g.get(k))
                            .map(|c| unqm(*c))
                            .unwrap_or(SecureField::zero()),
                    })
                    .collect(),
            })
            .collect()
    }

    pub fn n_tables(&self) -> usize {
        3 + self.uns.len()
    }
}

/// Everything both sides derive from `Params` alone: layouts, specs, widths,
/// the preprocessed id list.  One constructor, used by prover and verifier, so
/// they cannot disagree.
pub struct Plan {
    pub p: Params,
    pub attn_variant: AttnVariant,
    pub round: Round,
    pub shapes: Vec<Shape>,
    pub lays: Vec<Layouts>,
    pub cfg: Cfg,
    pub head_offsets: Vec<usize>,
    /// per band: is it the causal corner?  The prefix bands come first, then one
    /// corner band per prefix band with the same `insts` and the same head offset.
    pub corner: Vec<bool>,
    pub specs: Vec<Spec>,
    pub mat_lays: Vec<MatLayout>,
    pub ao_lay: AoLayout,
    pub walks: Vec<WalkPlan>,
    pub widths: Vec<u32>,
    pub log_norm: u32,
    pub log_frac: u32,
    pub log_mul: u32,
    pub glue: Option<GluePlan>,
    /// Stream H: the two boundary sides, or empty.  A statement asks for the
    /// boundary and has the wiring family, or it has no boundary at all — the
    /// plan settles that once so the prover and the verifier cannot disagree
    /// about how many trees the transcript has.
    pub bounds: Vec<crate::bounds::Side>,
    /// Stream AA: the K/V boundary sides, or empty.  They follow `bounds` in the
    /// tree order, so turning either feature on does not move the other's tree
    /// indices.
    pub kv: Vec<crate::bounds::KvSide>,
    /// Stream AA, per band: which head band group it belongs to.  Without
    /// `kv_bind` this is the band's own index modulo the prefix count, which is
    /// exactly the prefix/corner pairing `verify_composed` used to spell as
    /// `b - n_pre`; with it, a group is one prefix *sub-band per key block* plus
    /// the chunk's own corner.
    pub group: Vec<usize>,
    /// Stream AA, per band: does it carry the group's per-token region (the row
    /// maxima and the denominator)?  There is one such region per head band
    /// group -- one row maximum and one denominator per *token*, not per key
    /// block -- so with the prefix tiled it is the group's first sub-band's, and
    /// every other band of the group rides on it exactly as a corner band
    /// always has.
    pub rows_band: Vec<bool>,
    /// Stream BB: the embedding gadget's two row counts and its block width, or
    /// `None`.  Both sides derive it from `Params` alone, so the use region, the
    /// table region, the preprocessed id list and the verifier cannot disagree
    /// about the gadget's shape.
    pub emb: Option<crate::embed::EmbLayout>,
}

impl Plan {
    pub fn new(p: &Params) -> Self {
        let attn_variant = AttnVariant::parse(&p.variant);
        let round = Round::parse(&p.round);
        let cfg = Cfg {
            chan_target: p.chan_target,
            grid_target: p.grid_target,
            rw: p.rw,
            range_check: true,
            z_lo_bits: p.z_lo_bits,
            z_hi_bits: p.z_hi_bits,
            z_min: p.z_min,
        };
        // Stream AA (E11), layout design A: with `--kv-bind` the declared
        // prefix is tiled into `c / s` sub-bands of exactly `S` keys each -- the
        // shape the corner band already has -- instead of one rectangle of `c`
        // keys.  The tiling is not an optimisation, it is the precondition for
        // pointwise K/V binding: `LayoutKey` puts one row group per
        // `(instance, key)`, so a V column's *height* is a function of the
        // chunk's own `c`, and a producer's corner V column and a consumer's
        // untiled prefix V column are different heights and can never be equal.
        // Tiled at `S`, a consumer's sub-band `b` is the same `Shape` -- and so
        // the same `Grid`, `LayoutChan` and `LayoutKey` -- as the producer's
        // corner, and the two column lists coincide by construction rather than
        // by argument.  See `briefs/aa-kvbind.md`.
        assert!(!p.kv_bind || p.corner,
                "`--kv-bind` tiles the prefix beside the chunk's own block; it needs `--corner`");
        assert!(!p.kv_bind || p.c % p.s == 0,
                "`--kv-bind` tiles the prefix in blocks of S: C={} is not a multiple of S={}",
                p.c, p.s);
        // with the flag off this is 1, so the loop below builds exactly the one
        // prefix band per head band it always built
        let nsub = if p.kv_bind { p.c / p.s } else { 1 };
        let sub_c = if p.kv_bind { p.s } else { p.c };
        let mut head_offsets = vec![];
        let mut off = 0;
        let mut shapes = vec![];
        let mut group = vec![];
        let mut rows_band = vec![];
        for (g, insts) in p.bands.iter().enumerate() {
            for k in 0..nsub {
                head_offsets.push(off);
                shapes.push(Shape { s: p.s, c: sub_c, hd: p.hd, insts: *insts });
                group.push(g);
                // one per-token region per head band group, on its first band
                rows_band.push(k == 0);
            }
            off += insts;
        }
        assert_eq!(off, p.nh, "bands must cover every head");
        let mut corner = vec![false; shapes.len()];
        if p.corner {
            let mut off = 0;
            for (g, insts) in p.bands.iter().enumerate() {
                head_offsets.push(off);
                off += insts;
                shapes.push(Shape { s: p.s, c: p.s, hd: p.hd, insts: *insts });
                corner.push(true);
                group.push(g);
                // block 0 of a sequence has no declared prefix at all, so its
                // solo corner is the group's only band and carries the region
                rows_band.push(nsub == 0);
            }
        }
        // B11 stream U, lever 3: the two flags compose, they do not replace one
        // another -- the dequant family's instances join the *same*
        // `prove_batch` call and its heights the same link. Asking for the
        // second family without the first is a configuration error, and a loud
        // one: the arm's shape assumes the bands' collector exists.
        assert!(!p.gkr_dequant || p.gkr_bands,
                "--gkr-dequant composes with --gkr-bands; it does not replace it");
        assert!(!p.gkr_glue || p.gkr_bands,
                "`--gkr-glue` extends the bands' GKR arm; it needs `--gkr-bands`");
        assert!(!p.gkr_walks || p.gkr_bands,
                "`--gkr-walks` extends the bands' GKR arm; it needs `--gkr-bands`");
        assert!(!p.gkr_attnout || p.gkr_bands,
                "`--gkr-attnout` extends the bands' GKR arm; it needs `--gkr-bands`");
        // Stream AA: the bands' GKR arm still gates the per-token region on
        // `plan.corner` in two places of its own (`gkrband.rs`, `gkrlink.rs`),
        // which is the pairing `--kv-bind` generalises.  Composing the two is a
        // follow-on and not a hazard to leave silent, so it is a loud
        // configuration error until that arm learns `rows_band`.
        assert!(!(p.kv_bind && p.gkr_bands),
                "`--kv-bind` and `--gkr-bands` do not compose yet: the GKR arm still pairs \
                 bands by `corner`");
        let lays: Vec<Layouts> = shapes.iter().map(|sh| Layouts::choose(sh, &cfg)).collect();

        let mut widths = vec![p.k_bits, p.q_bits, 8, p.z_lo_bits, p.z_hi_bits, p.ao_rem_bits];
        if let AttnVariant::IntSplit(k) = attn_variant {
            widths.push(k);
        }
        // Stream Z: a fold matmul range-checks its magnitude and low-remainder
        // limbs at 13 bits, its low-product quotient at 16, and its high
        // remainder at the pinned fold width; a widened attention output ranges
        // `o`'s committed activations at `ao_q_bits + 1`.
        if p.mats.iter().any(|m| m.variant == "fold") {
            assert!(p.fold_rem_bits > 0, "a fold matmul needs --fold-rem-bits");
            widths.extend([crate::dequant::FOLD_LIMB, 16, p.fold_rem_bits]);
        }
        if p.ao_q_bits > 7 {
            widths.push(p.ao_q_bits + 1);
        }
        widths.sort_unstable();
        widths.dedup();

        let mut specs = vec![];
        let mut mat_lays = vec![];
        for m in &p.mats {
            let lay = MatLayout::choose_or_padded(m.n_out, m.tokens, p.deq_target);
            let fold = m.variant == "fold";
            specs.push(Spec {
                variant: DeqVariant::parse(&m.variant),
                round,
                kmax: p.kmax,
                per_row: lay.per_row,
                limb_bits: (16, 14),
                q_bits: p.ao_q_bits,
                rem_bits: if fold { p.fold_rem_bits } else { p.ao_rem_bits },
                fold_alpha: if fold { p.fold_alpha } else { 0 },
            });
            mat_lays.push(lay);
        }
        let ao_lay = AoLayout::choose(p.nh, p.s, p.hd, p.ao_target);
        let mut walks = vec![];
        for (name, members) in WALK_GROUPS {
            let idx: Vec<usize> = members
                .iter()
                .filter_map(|n| p.mats.iter().position(|m| m.name == *n))
                .collect();
            if idx.is_empty() {
                continue;
            }
            let n_in = p.mats[idx[0]].n_in;
            let n_out_max = idx.iter().map(|i| p.mats[*i].n_out).max().unwrap();
            walks.push(WalkPlan {
                name: name.to_string(),
                mats: idx.clone(),
                lay: WalkLayout::choose(n_in, n_out_max, p.mats[idx[0]].tokens, p.walk_p),
                // stream Z: the widened attention output is `o`'s committed input
                act_bits: if name == "o" && p.ao_q_bits > 7 { p.ao_q_bits + 1 } else { 8 },
            });
        }
        // table row counts, exactly as `Tables::new` builds them
        let norm_rows = crate::bf16::norm_table(p.kmax, round).len();
        let log_norm = crate::common::log_rows_for(norm_rows, 1);
        let frac_rows = 1usize + (0..=p.kmax).map(|k| 1usize << k).sum::<usize>();
        let log_frac = crate::common::log_rows_for(frac_rows, 1);
        let log_mul = 14;
        let glue = p.glue.as_ref().map(|g| GluePlan::new(p, g));
        // Stream H: the boundary is the wiring family's two stream ties, so it
        // exists exactly when the statement asks for it and the family is there.
        let bounds = match (&glue, p.bounds) {
            (Some(g), true) => crate::bounds::sides(g),
            _ => vec![],
        };
        // Stream AA: the K/V sides follow stream H's, so the two features'
        // tree indices are independent of each other.  `bounds::kv_layout`
        // rederives exactly this arithmetic from `Fixed.params` alone, which is
        // how the pair check finds the roots without building a plan.
        let kv = if p.kv_bind {
            let kb: Vec<crate::bounds::KvBand> = (0..shapes.len())
                .map(|b| crate::bounds::KvBand {
                    group: group[b],
                    corner: corner[b],
                    pk: lays[b].chan.pk,
                    pv: lays[b].key.pv,
                    chan_rows: lays[b].chan.g.rows,
                    key_rows: lays[b].key.g.rows,
                })
                .collect();
            crate::bounds::kv_sides(crate::bounds::FIRST_TREE + bounds.len(), &kb)
        } else {
            vec![]
        };
        // Stream BB (E11): the embedding gadget.  It ties to the wiring
        // family's `stream_in` claim, so a statement that asks for it without
        // the wiring family is a loud configuration error rather than a proof
        // with nothing to compare against.
        let emb = (p.embed_vocab > 0).then(|| {
            let g = glue.as_ref()
                .expect("`--embed` ties to the wiring family's stream input; it needs the glue");
            assert!(p.embed_per_row > 0, "`--embed` needs a channel-block target");
            crate::embed::EmbLayout::choose(g.t.d, p.s, p.embed_vocab, p.embed_per_row)
        });
        Self { p: p.clone(), attn_variant, round, shapes, lays, cfg, head_offsets, corner, specs,
               mat_lays, ao_lay, walks, widths, log_norm, log_frac, log_mul, glue, bounds,
               kv, group, rows_band, emb }
    }

    /// How many trees the transcript commits: the three that were always there,
    /// plus one per boundary side.
    pub fn n_trees(&self) -> usize {
        3 + self.bounds.len() + self.kv.len()
    }

    /// Tree 0's column ids, in exactly the order the prover commits them.
    /// One band's challenge powers.  The corner bands run their key Horner in
    /// `r_corner`, an independent challenge, and borrow the prefix's `r^C` and
    /// `mu` for the claims that are summed with -- or compared against -- the
    /// prefix's (attn_air's `as_corner`).  Prover and verifier both come here.
    pub fn powers(&self, b: usize, r: SecureField, s: SecureField, rv: SecureField,
                  r_corner: SecureField) -> Powers {
        let l = &self.lays[b];
        let sh = &self.shapes[b];
        if self.corner[b] {
            let l0 = &self.lays[0];
            let pre0 = Powers::new(r, s, rv, &self.shapes[0], &l0.chan, &l0.key, &l0.out, l0.rw);
            Powers::new(r_corner, s, rv, sh, &l.chan, &l.key, &l.out, l.rw)
                .as_corner(pre0.r_c, sh.c)
        } else {
            Powers::new(r, s, rv, sh, &l.chan, &l.key, &l.out, l.rw)
        }
    }

    /// The exponent of the global per-head instance weight `r^{·}`.
    ///
    /// Every region that packs heads has to carry the same one: the bands'
    /// `Powers::r_c` and `sn_c`, the per-token region's `mu`, the attention
    /// output region's `mu_den`, and `verify_composed`'s own `r_c^{head}` fold.
    /// Untiled it is the context width `C`, because a prefix band's key Horner
    /// walks exactly `C` keys per instance and so produces `r^C` for free.
    ///
    /// Stream AA: with `--kv-bind` every band is `c = S`, so the same Horner
    /// produces `r^S` for free, and the global weight moves with it.  That is
    /// what keeps design A free of any new AIR mechanism -- a prefix sub-band
    /// needs no `iend`/`kappa` conversion, its natural instance stride *is* the
    /// global one, and it keeps `KEY_SEL` and CC's frozen per-slot content.
    pub fn head_weight_c(&self) -> usize {
        if self.p.kv_bind { self.p.s } else { self.p.c }
    }

    /// How many secure felts the transcript draws after the relations, and where
    /// the corner's key challenge sits.  It goes *last* so a proof without the
    /// corner draws exactly what it drew before.
    /// Stream E: the tie challenge pair is drawn for the wiring family *or* for
    /// a head-key claim on its own, since a head dump has no wiring columns.
    pub fn has_ties(&self) -> bool {
        self.glue.is_some() || self.p.head_key
    }

    /// Stream E: the index of the matmul the argmax region is tied to, when it
    /// is composed.  Its layout is that matmul's, so the tie is positionwise.
    pub fn argmax_mat(&self) -> Option<usize> {
        self.p.head_argmax
            .then(|| self.p.mats.iter().position(|m| m.name == "lm_head"))
            .flatten()
    }
    pub fn n_challenges(&self) -> usize {
        3 + if self.has_ties() { 2 } else { 0 } + if self.p.corner { 1 } else { 0 }
    }
    pub fn corner_challenge_at(&self) -> usize {
        3 + if self.has_ties() { 2 } else { 0 }
    }

    /// Stream E: does this matmul's dequant region publish an output-side claim?
    /// Both sides ask the plan, so the preprocessed list, the region, the AIR and
    /// the verifier cannot disagree about it.
    pub fn has_oside(&self, name: &str) -> bool {
        self.glue.as_ref().is_some_and(|g| g.out_ties.iter().any(|(_, n)| *n == name))
            || (self.p.head_key && name == "lm_head")
    }

    /// What that output side accumulates: the head publishes the argmax's order
    /// key, everything else the glue's `(sm, ef)` pair.
    pub fn oside_kind(&self, name: &str) -> crate::dequant::OKind {
        if name == "lm_head" {
            crate::dequant::OKind::Key
        } else {
            crate::dequant::OKind::Pair
        }
    }

    pub fn pre_ids(&self) -> Vec<PreProcessedColumnId> {
        let mut ids = vec![];
        for b in 0..self.shapes.len() {
            ids.extend(CHAN_SEL.iter().map(|n| band_id(b, n)));
            if self.corner[b] {
                // the corner's own selector set, then one causal-mask column per
                // token slot -- the order `commit_pre` writes them in
                ids.extend(CORNER_SEL.iter().map(|n| band_id(b, n)));
                ids.extend((0..self.lays[b].key.q).map(|i| msk_id(b, i)));
            } else {
                ids.extend(KEY_SEL.iter().map(|n| band_id(b, n)));
            }
            // Stream CC (E11): the inert-lead-row selector closes the grid's
            // block, and exists only when the grid leaves such rows.
            ids.extend(crate::attn_air::act_id(b, self.corner[b],
                                               self.lays[b].key.g.lead));
            // Stream AA: the predicate is "does this band carry the per-token
            // region", not "is it a prefix band".  With `--kv-bind` off the two
            // are the same predicate on the same bands, so this id sequence is
            // the one tree 0 always had.
            if self.rows_band[b] {
                ids.extend(ROW_SEL.iter().map(|n| band_id(b, n)));
            }
        }
        for s in 0..self.ao_lay.per_row {
            ids.push(ao_chan("n", s));
            ids.push(ao_chan("p", s));
        }
        ids.extend(AO_SEL.iter().map(|n| ao_sel(n)));
        for (m, sp) in self.p.mats.iter().zip(&self.specs) {
            for s in 0..sp.per_row {
                match sp.variant {
                    DeqVariant::Pow2 => ids.push(chan_col(&m.name, "j", s)),
                    DeqVariant::Chain => {
                        ids.push(chan_col(&m.name, "m", s));
                        ids.push(chan_col(&m.name, "e", s));
                    }
                    // stream Z: the weight-scale mantissa and the channel half
                    // of the divisor, `mat_region`'s order
                    DeqVariant::Fold => {
                        ids.push(chan_col(&m.name, "m", s));
                        ids.push(chan_col(&m.name, "p", s));
                    }
                    _ => unreachable!(),
                }
            }
            ids.extend(DEQ_SEL.iter().map(|n| deq_sel(&m.name, n)));
            if self.has_oside(&m.name) {
                ids.extend(crate::dequant::DEQ_OSEL.iter().map(|n| deq_sel(&m.name, n)));
            }
        }
        for w in &self.walks {
            for (mi, _) in w.mats.iter().enumerate() {
                for pp in 0..w.lay.pk {
                    ids.push(walk_w(&w.name, mi, pp));
                }
            }
            ids.extend(WALK_SEL.iter().map(|n| walk_sel(&w.name, n)));
        }
        if let Some(g) = &self.glue {
            for r in &g.regions {
                ids.push(crate::glue::glue_sel(&r.tag, "nofirst"));
                ids.push(crate::glue::glue_sel(&r.tag, "end"));
            }
        }
        ids.push(id(TBL_EXP_X));
        ids.push(id(TBL_EXP_Y));
        ids.extend(NORM_COLS.iter().map(|c| id(*c)));
        ids.extend(FRAC_COLS.iter().map(|c| id(*c)));
        ids.extend(MUL_COLS.iter().map(|c| id(*c)));
        ids.extend(QNZ_COLS.iter().map(|c| id(*c)));
        ids.extend(self.widths.iter().map(|w| tbl_range(*w)));
        if let Some(g) = &self.glue {
            ids.extend(crate::glue::glue_table_ids(&g.uns));
        }
        // stream E: the argmax region's own three selectors, last, so that no
        // family before it moves an index
        if self.argmax_mat().is_some() {
            ids.extend(crate::argmax::AM_SEL.iter().map(|n| crate::argmax::id(n)));
        }
        // stream BB: the embedding gadget's own columns -- the use region's four
        // selectors and its block index, then the table's index pair and its
        // `pe` value columns.  After the argmax and before the GKR arm's, so no
        // family that exists today moves an index.
        if let Some(l) = &self.emb {
            ids.extend(crate::embed::EMB_SEL.iter().map(|n| crate::embed::id(n)));
            ids.push(crate::embed::cb_id());
            ids.push(crate::embed::tab_v_id());
            ids.push(crate::embed::tab_cb_id());
            ids.extend((0..l.pe).map(crate::embed::tab_e_id));
        }
        // B11 stream S, lever 3: the link heights' boundary selectors, after
        // everything -- they exist only on the GKR arm, so the flag-off tree
        // is untouched
        if self.p.gkr_bands {
            for h in crate::gkrlink::link_heights(self) {
                ids.push(crate::gkrlink::first_id(h));
                ids.push(crate::gkrlink::second_id(h));
            }
        }
        ids
    }

    /// The number of components that carry lookups, in `use_sums` order.
    pub fn n_use(&self) -> usize {
        // chan + grid + per-token per prefix band; a corner band has no per-token
        // region (the pair shares the prefix band's), so it contributes two.
        // B11 stream S, lever 1: on the GKR arm the bands' sums are proved by
        // the GKR and never enter `use_sums`.
        (if self.p.gkr_bands {
            0
        } else {
            // stream AA: three sums for a band that carries the per-token
            // region, two for one that rides on its group's -- with `--kv-bind`
            // off, `rows_band` is `!corner` and this is the sum it always was
            self.rows_band.iter().map(|r| if *r { 3 } else { 2 }).sum::<usize>()
        })
            + 1
            // B11 stream U, lever 3: the same for the dequant family
            + if self.p.gkr_dequant { 0 } else { self.p.mats.len() }
            + if self.p.gkr_walks { 0 } else { self.walks.len() }
            // B11 stream V, levers 3 and 4: and the same for glue and attnout
            + if self.p.gkr_glue { 0 }
              else { self.glue.as_ref().map(|g| g.regions.len()).unwrap_or(0) }
            - usize::from(self.p.gkr_attnout)
            + usize::from(self.argmax_mat().is_some())
            // stream BB: the embedding gadget's use region, last of all
            + usize::from(self.emb.is_some())
    }
    pub fn n_table_sums(&self) -> usize {
        5 + self.widths.len() + self.glue.as_ref().map(|g| g.n_tables()).unwrap_or(0)
            // stream BB: the embedding table's own side, appended last
            + usize::from(self.emb.is_some())
    }
}

// -------------------------------------------------- components (both sides) ----

pub struct BuiltComponents {
    pub chans: Vec<FrameworkComponent<ChanEval>>,
    pub keys: Vec<FrameworkComponent<KeyEval>>,
    pub rows: Vec<FrameworkComponent<RowEval>>,
    /// B11 stream S, lever 3: one univariate IOP per link height, and the tie
    /// component that pins its columns to the base columns at OODS. Empty and
    /// `None` off the GKR arm (and on the emit and probe paths, which have no
    /// artifact to derive them from).
    pub links: Vec<FrameworkComponent<crate::gkrlink::LinkEval>>,
    pub link_tie: Option<crate::gkrlink::LinkTie>,
    pub ao: FrameworkComponent<AttnOutEval>,
    pub mats: Vec<FrameworkComponent<DequantEval>>,
    pub walks: Vec<FrameworkComponent<WalkEval>>,
    pub glue: Vec<FrameworkComponent<GlueEval>>,
    pub exp_t: FrameworkComponent<ExpTableEval>,
    pub tables: Vec<FrameworkComponent<TableEval>>,
    pub gtables: Vec<FrameworkComponent<GTableEval>>,
    /// stream E: the argmax region, last of all, so adding it moves nothing
    pub argmax: Option<FrameworkComponent<crate::argmax::ArgmaxEval>>,
    /// stream BB: the embedding gadget's two components, after the argmax and
    /// after everything else, in this order -- the use side then the table side
    pub embed: Option<FrameworkComponent<crate::embed::EmbedEval>>,
    pub emb_tab: Option<FrameworkComponent<crate::embed::EmbTableEval>>,
    pub pows: Vec<Powers>,
    /// per band, in `chans`/`keys` order: does it have a per-token region?  A
    /// corner band does not (B10 step 3), so `rows` is shorter than `chans` and
    /// the component order has to be walked, not indexed.
    pub has_rows: Vec<bool>,
}

impl BuiltComponents {
    /// B7a milestone 3: lower every component's constraints to its kernel
    /// parameter blob once, in parallel, before proving. The fork calls
    /// `cuda_eval_blob()` from its sequential component loop, which put the
    /// whole lowering on one thread with the device idle (16.5's 85 ms).
    /// One list of closures over the *evals*, then one `par_iter`.
    ///
    /// Not `par_iter` over the components: `FrameworkComponent` holds an
    /// `InfoEvaluator`, whose `ArithmeticCounter` is an `Rc<RefCell<..>>`
    /// (`constraint-framework/src/info.rs:166`), so a component is `!Sync` and
    /// cannot cross a thread boundary. The evals themselves are plain data.
    pub fn prewarm_blobs(&self) {
        use rayon::prelude::*;
        let mut tasks: Vec<Box<dyn Fn() + Send + Sync + '_>> = Vec::new();
        for c in &self.chans {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.keys {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.rows {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.mats {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.walks {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.tables {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        let ao = &self.ao.eval;
        tasks.push(Box::new(move || crate::cudagen::prewarm(ao)));
        let exp_t = &self.exp_t.eval;
        tasks.push(Box::new(move || crate::cudagen::prewarm(exp_t)));
        // B11 stream X: the list has to be *every* dispatched component, and
        // until here it was eight of thirteen -- the glue, the glue's tables,
        // the links and the argmax were dispatched but never prewarmed.
        //
        // That is a correctness matter and not a speed one. `cudagen`'s blob
        // cache is keyed by the evaluator's address, on the stated assumption
        // that an address is "stable for as long as its `FrameworkComponent`
        // lives -- i.e. for the whole proof". The pipelined loop proves 32
        // chunks in one process, so a chunk's components are dropped and the
        // next chunk's are allocated over the same addresses. A component that
        // *is* prewarmed overwrites the stale entry at its address before
        // anything dispatches it, so it is safe by construction; a component
        // that is dispatched but never prewarmed can only ever *read* an entry
        // some other family wrote in an earlier chunk. Measured on the H100 at
        // C=4096 over the 32-chunk pipelined loop, that happened twice in nine
        // runs, and both times in exactly one of the five families missing
        // here: `evaluate_b65_glue_eval: scalar count` and
        // `evaluate_b65_g_table_eval: shape id out of range`. Both are the
        // generated dispatch's own checks, so the failure is fail-stop and
        // cannot reach a digest -- which is why every completed run has always
        // reproduced stream O's references. Completing the list restores the
        // invariant the cache's comment already claims.
        for c in &self.glue {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.gtables {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.links {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        if let Some(c) = &self.argmax {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        // B11 stream BB: and the embedding gadget's two, in the commit that
        // adds them. The list has to be every dispatched component: a
        // component that is dispatched but never prewarmed can only read the
        // blob some other family left at its address, and the pipelined loop
        // reuses addresses across chunks. Stream X measured that failing twice
        // in nine runs on the five families that were missing then.
        if let Some(c) = &self.embed {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        if let Some(c) = &self.emb_tab {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        tasks.par_iter().for_each(|t| t());
    }

    pub fn provers(&self) -> Vec<&dyn ComponentProver<crate::common::PcsB>> {
        let mut v: Vec<&dyn ComponentProver<_>> = vec![];
        let mut ri = 0usize;
        for b in 0..self.chans.len() {
            v.push(&self.chans[b]);
            v.push(&self.keys[b]);
            if self.has_rows[b] {
                v.push(&self.rows[ri]);
                ri += 1;
            }
        }
        for l in &self.links {
            v.push(l);
        }
        if let Some(t) = &self.link_tie {
            v.push(t);
        }
        v.push(&self.ao);
        for m in &self.mats {
            v.push(m);
        }
        for w in &self.walks {
            v.push(w);
        }
        for g in &self.glue {
            v.push(g);
        }
        v.push(&self.exp_t);
        for t in &self.tables {
            v.push(t);
        }
        for t in &self.gtables {
            v.push(t);
        }
        if let Some(a) = &self.argmax {
            v.push(a);
        }
        if let Some(e) = &self.embed {
            v.push(e);
        }
        if let Some(e) = &self.emb_tab {
            v.push(e);
        }
        v
    }
    pub fn components(&self) -> Vec<&dyn Component> {
        let mut v: Vec<&dyn Component> = vec![];
        let mut ri = 0usize;
        for b in 0..self.chans.len() {
            v.push(&self.chans[b]);
            v.push(&self.keys[b]);
            if self.has_rows[b] {
                v.push(&self.rows[ri]);
                ri += 1;
            }
        }
        for l in &self.links {
            v.push(l);
        }
        if let Some(t) = &self.link_tie {
            v.push(t);
        }
        v.push(&self.ao);
        for m in &self.mats {
            v.push(m);
        }
        for w in &self.walks {
            v.push(w);
        }
        for g in &self.glue {
            v.push(g);
        }
        v.push(&self.exp_t);
        for t in &self.tables {
            v.push(t);
        }
        for t in &self.gtables {
            v.push(t);
        }
        if let Some(a) = &self.argmax {
            v.push(a);
        }
        if let Some(e) = &self.embed {
            v.push(e);
        }
        if let Some(e) = &self.emb_tab {
            v.push(e);
        }
        v
    }
}

/// B11 stream F step 3: what one `.extend(...)` added to the interaction set.
///
/// Called after each region appends its columns, with the running (columns,
/// cells) pair the previous call left behind. The answer c-rider 5.1b wants —
/// what share of the interaction cells a scoped GKR over `bands` alone would
/// cover — is this table, and it also gives `family_cells` the `tables` field
/// it was missing.
fn inter_mark(
    what: &str,
    all: &[crate::device::AnyCol<'static>],
    seen: &mut (usize, usize),
    out: &mut Vec<Value>,
) {
    let cols = all.len();
    let cells: usize = all.iter().map(|c| c.size()).sum();
    out.push(json!({
        "region": what,
        "columns": cols - seen.0,
        "cells": cells - seen.1,
    }));
    *seen = (cols, cells);
}

/// Build every component from the plan, the drawn relations and challenges, and
/// the parsed claims — the verifier's route, also used by the prover so the two
/// sides cannot diverge.
#[allow(clippy::too_many_arguments)]
/// `build_components` without link parameters: the AIR probe and any caller
/// that runs before a GKR artifact exists. On the GKR arm the prove and
/// verify paths call `build_components_linked` with the derived heights.
#[allow(clippy::too_many_arguments)]
pub fn build_components(plan: &Plan, rels: &Rels, r: SecureField, s: SecureField,
                        rv: SecureField, r_corner: SecureField, w_glue: SecureField,
                        lam_glue: SecureField, claims: &Claims) -> BuiltComponents {
    build_components_linked(plan, rels, r, s, rv, r_corner, w_glue, lam_glue, claims, None)
}

#[allow(clippy::too_many_arguments)]
pub fn build_components_linked(plan: &Plan, rels: &Rels, r: SecureField, s: SecureField,
                        rv: SecureField, r_corner: SecureField, w_glue: SecureField,
                        lam_glue: SecureField, claims: &Claims,
                        link: Option<&[crate::gkrlink::HeightLink]>) -> BuiltComponents {
    let p = &plan.p;
    let ids = plan.pre_ids();
    let alloc = &mut TraceLocationAllocator::new_with_preprocessed_columns(&ids);
    let off = Offsets::new(p.q_bits, p.k_bits);
    let mut use_k = 0usize;
    let mut next_use = |claims: &Claims| {
        let v = unqm(claims.use_sums[use_k]);
        use_k += 1;
        v
    };

    let (mut chans, mut keys, mut rows, mut pows) = (vec![], vec![], vec![], vec![]);
    for (b, _sh) in plan.shapes.iter().enumerate() {
        let l = &plan.lays[b];
        let pow = plan.powers(b, r, s, rv, r_corner);
        let bc = claims.bands[b];
        // B11 stream S, lever 1: on the GKR arm a band component carries no
        // LogUp, so there is no claimed sum to hand it and none to consume.
        chans.push(FrameworkComponent::<ChanEval>::new(
            alloc,
            ChanEval { tag: b, lay: l.chan, pow: pow.clone(), off, claimed: unqm(bc[0]),
                       rel_k: rels.range(p.k_bits).clone(), rel_q: rels.range(p.q_bits).clone(),
                       range_check: true, gkr: p.gkr_bands,
                       // stream AA: empty unless `--kv-bind`, so the IR, the
                       // fingerprint and the proof are unmoved with it off
                       bound_k: crate::bounds::kv_bounds_of(&plan.kv, b, l.chan.pk) },
            if p.gkr_bands { SecureField::zero() } else { next_use(claims) },
        ));
        keys.push(FrameworkComponent::<KeyEval>::new(
            alloc,
            KeyEval { tag: b, lay: l.key, pow: pow.clone(), off, variant: plan.attn_variant,
                      claimed_hs: unqm(bc[1]), claimed_av: unqm(bc[2]), claimed_sn: unqm(bc[3]),
                      rel_exp: rels.exp.clone(), rel_v: rels.range(8).clone(),
                      rel_split: match plan.attn_variant {
                          AttnVariant::IntSplit(k) => Some(rels.range(k).clone()),
                          _ => None,
                      },
                      range_check: true, gkr: p.gkr_bands, corner: plan.corner[b],
                      bound_v: crate::bounds::kv_bounds_of(&plan.kv, b, l.key.pv) },
            if p.gkr_bands { SecureField::zero() } else { next_use(claims) },
        ));
        // stream AA: "carries the per-token region", which with `--kv-bind` off
        // is `!corner[b]` on the same bands
        if plan.rows_band[b] {
            rows.push(FrameworkComponent::<RowEval>::new(
                alloc,
                RowEval { tag: b, log_rows: l.log_row, rw: l.rw, pow: pow.clone(),
                          with_max: plan.attn_variant != AttnVariant::Bf16x,
                          claimed_am: unqm(bc[4]), claimed_az: unqm(bc[5]),
                          z_lo_bits: p.z_lo_bits, z_min: p.z_min,
                          rel_lo: rels.range(p.z_lo_bits).clone(),
                          rel_hi: rels.range(p.z_hi_bits).clone(), gkr: p.gkr_bands },
                if p.gkr_bands { SecureField::zero() } else { next_use(claims) },
            ));
        }
        pows.push(pow);
    }

    // B11 stream S, lever 3: the link components allocate right after the
    // bands, matching where the prover commits their columns.
    //
    // B11 stream U, lever 3: the *tie* is built lower down, after the dequant
    // components exist, because a dequant region's folded weights name columns
    // in that region's own span. The allocation order -- what the statement
    // depends on -- is unchanged; only where the tie reads the spans back moves.
    let mut links = vec![];
    if let Some(hl) = link {
        for l in hl {
            links.push(FrameworkComponent::<crate::gkrlink::LinkEval>::new(
                alloc,
                crate::gkrlink::LinkEval {
                    h: l.h, mle_point: l.mle_point.clone(), shift: l.shift,
                },
                SecureField::zero(),
            ));
        }
    }

    let (rv_pow_ao, rv_step_ao) = powers(rv, plan.ao_lay.per_row);
    let (_, rv_hd) = powers(rv, p.hd);
    // stream AA: the global per-head instance weight, `r^C` untiled and `r^S`
    // under `--kv-bind` -- see `Plan::head_weight_c`
    let (_, r_c) = powers(r, plan.head_weight_c());
    let (_, s_span) = powers(s, p.s - 1);
    let ao = FrameworkComponent::<AttnOutEval>::new(
        alloc,
        AttnOutEval {
            log_rows: plan.ao_lay.log_rows,
            per_row: plan.ao_lay.per_row,
            rv_pow: rv_pow_ao,
            rv_step: rv_step_ao,
            s,
            mu_out: rv_hd * s_span.inverse(),
            mu_den: r_c * s_span.inverse(),
            claimed_out: unqm(claims.ao_out),
            claimed_den: unqm(claims.ao_den),
            qnz: rels.qnz.clone(),
            rel_rem: rels.range(p.ao_rem_bits).clone(),
            gkr: p.gkr_attnout,
        },
        if p.gkr_attnout { SecureField::zero() } else { next_use(claims) },
    );

    let mut mats = vec![];
    for (mi, (m, sp)) in p.mats.iter().zip(&plan.specs).enumerate() {
        let sub: Vec<RangeRel> =
            range_widths(sp).iter().map(|w| rels.range(*w).clone()).collect();
        let (rv_pow, rv_step) = powers(rv, sp.per_row);
        mats.push(FrameworkComponent::<DequantEval>::new(
            alloc,
            DequantEval {
                log_rows: plan.mat_lays[mi].log_rows,
                sp: sp.clone(),
                tag: m.name.clone(),
                norm: rels.norm.clone(),
                frac: rels.frac.clone(),
                mul: rels.mul.clone(),
                qnz: rels.qnz.clone(),
                rels: sub,
                yside: Some(YSide { rv_pow, rv_step, s, claimed_h: unqm(claims.mat_h[mi]) }),
                // stream E: the output side, on the matmuls the plan names.  The
                // claims arrive in the same order the plan lists the ties, which
                // is mat order -- `o` before `down` -- and the assert below is
                // what says so rather than trusting it.
                oside: plan.has_oside(&m.name).then(|| {
                    let kind = plan.oside_kind(&m.name);
                    let ow = oside_w(w_glue, lam_glue, m.tokens, m.n_out,
                                     plan.mat_lays[mi].per_row, kind);
                    let claimed = if m.name == "lm_head" {
                        claims.head_key.first().copied()
                    } else {
                        // the claims arrive in mat order and the plan lists the
                        // ties in the same order, `o` before `down`
                        plan.glue.as_ref().unwrap().out_ties.iter()
                            .position(|(_, n)| *n == m.name)
                            .and_then(|k| claims.deq_out.get(k).copied())
                    };
                    crate::dequant::OSide {
                        w: ow.w,
                        lam: ow.lam,
                        k_in: ow.k_in,
                        k_bd: ow.k_bd,
                        claimed: claimed.map(unqm).unwrap_or(SecureField::zero()),
                        kind,
                    }
                }),
                // B11 stream U, lever 3: under the flag this region's lookups
                // are the batched GKR's, so the AIR emits none of them and the
                // interaction masks they would allocate are not there to claim.
                gkr: p.gkr_dequant,
            },
            // the use sum belongs to whoever owns the lookups: with the
            // collector owning them there is no claimed sum in `use_sums` for
            // this region at all, exactly as stream S did for the bands
            if p.gkr_dequant { SecureField::zero() } else { next_use(claims) },
        ));
    }


    let mut walks = vec![];
    for (wi, w) in plan.walks.iter().enumerate() {
        let (rv_pow, rv_step) = powers(rv, w.lay.pk);
        let (s_pow, s_step) = powers(s, w.lay.pq);
        walks.push(FrameworkComponent::<WalkEval>::new(
            alloc,
            WalkEval {
                tag: w.name.clone(),
                g: w.lay.g,
                pk: w.lay.pk,
                pq: w.lay.pq,
                n_mats: w.mats.len(),
                rv_pow,
                rv_step,
                s_pow,
                s_step,
                claimed: claims.walk_acc[wi].iter().map(|x| unqm(*x)).collect(),
                rel_act: rels.range(w.act_bits).clone(),
                act_off: 1 << (w.act_bits - 1),
                range_check: true,
                gkr: p.gkr_walks,
            },
            if p.gkr_walks { SecureField::zero() } else { next_use(claims) },
        ));
    }

    // B10 step 2: the wiring family, between the walks and the shared tables --
    // the order the prover commits its columns in, so the two cannot diverge
    let mut glue = vec![];
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let ties = gp.ties(w_glue, lam_glue, claims);
        for (i, r) in gp.regions.iter().enumerate() {
            glue.push(FrameworkComponent::<GlueEval>::new(
                alloc,
                GlueEval {
                    log_rows: r.log_rows,
                    tag: r.tag.clone(),
                    prog: r.prog.clone(),
                    round: plan.round,
                    norm: gr.norm.clone(),
                    frac: gr.frac.clone(),
                    pow: gr.pow.clone(),
                    smul: gr.smul.clone(),
                    un: gr.un.clone(),
                    uns: gr.uns.clone(),
                    val: gr.val.clone(),
                    tie: Some(ties[i].clone()),
                    // Stream H: a region carries a boundary side when the plan
                    // named it one.  The sides are registered in `stream_in`,
                    // `stream_out` order and the components in region order, so
                    // the tree a side lands in is the same for prover and
                    // verifier without either being told.
                    bound: plan.bounds.iter().find(|s| s.region == i).map(|s| s.bound),
                    // B11 stream V, lever 3: under the flag this region's
                    // lookups are the batched GKR's, so the AIR emits none of
                    // them and the interaction masks they would allocate are
                    // not there to claim.
                    gkr: p.gkr_glue,
                },
                // the use sum belongs to whoever owns the lookups: with the
                // collector owning them there is no claimed sum in `use_sums`
                // for this region at all, exactly as stream U did for dequant
                if p.gkr_glue { SecureField::zero() } else { next_use(claims) },
            ));
        }
    }

    // B11 stream S, lever 3, moved down by streams U and V: the tie, from the spans
    // the allocator handed out above.
    let mut link_tie = None;
    if let Some(hl) = link {
        let span_of = |locs: &[stwo::core::pcs::TreeSubspan], tree: usize| {
            locs.iter().find(|sp| sp.tree_index == tree)
                .map(|sp| sp.col_start)
                .expect("a component with no span in the tree the tie reads")
        };
        // band -> index into `rows` (the bands that carry a per-token region)
        let mut row_of = vec![usize::MAX; plan.rows_band.len()];
        let mut ri = 0usize;
        for (b, has) in plan.rows_band.iter().enumerate() {
            if *has {
                row_of[b] = ri;
                ri += 1;
            }
        }
        let mut pre_idx = vec![];
        let mut heights = vec![];
        for (l, comp) in hl.iter().zip(&links) {
            let t2 = span_of(comp.trace_locations(), 2);
            let mut terms = vec![];
            for (reg, c, w) in &l.weights {
                use crate::gkrlink::{RCol, Reg};
                let band = match reg {
                    Reg::Band(b) => *b,
                    // B11 stream V, lever 3: a glue region is one component, so
                    // its own span is the whole resolution -- the mask index
                    // *is* the column index within it, which is `Prog`'s order.
                    Reg::Glue(g) => {
                        let RCol::Glue(i) = c else {
                            panic!("a non-glue column named on a glue region")
                        };
                        terms.push((1, span_of(glue[*g].trace_locations(), 1) + i, *w));
                        continue;
                    }
                    Reg::Walk(k) => {
                        let RCol::Walk(i) = c else {
                            panic!("a non-walk column named on a walk region")
                        };
                        terms.push((1, span_of(walks[*k].trace_locations(), 1) + i, *w));
                        continue;
                    }
                    Reg::Ao => {
                        let RCol::Ao(i) = c else {
                            panic!("a non-attnout column named on the attnout region")
                        };
                        terms.push((1, span_of(ao.trace_locations(), 1) + i, *w));
                        continue;
                    }
                    // B11 stream U, lever 3: a dequant region is one component,
                    // so its own span is the whole resolution -- the mask index
                    // *is* the column index within it (`dequant::Map`).
                    Reg::Deq(m) => {
                        let (tree, g) = match c {
                            RCol::Deq(i) =>
                                (1, span_of(mats[*m].trace_locations(), 1) + i),
                            RCol::DeqPre(sl) => {
                                let want =
                                    crate::dequant::chan_col(&p.mats[*m].name, "m", *sl).id;
                                let g = ids.iter().position(|x| x.id == want)
                                    .expect("a dequant weight-scale column missing \
                                             from the preprocessed tree");
                                pre_idx.push(g);
                                (0, g)
                            }
                            _ => panic!("a band column named on a dequant region"),
                        };
                        terms.push((tree, g, *w));
                        continue;
                    }
                };
                let (tree, g) = match c {
                    RCol::Chan(i) => (1, span_of(chans[band].trace_locations(), 1) + i),
                    RCol::Key(i) => (1, span_of(keys[band].trace_locations(), 1) + i),
                    RCol::Row(i) =>
                        (1, span_of(rows[row_of[band]].trace_locations(), 1) + i),
                    RCol::Msk(i) => {
                        let want = crate::attn_air::msk_id(band, *i).id;
                        let g = ids.iter().position(|x| x.id == want)
                            .expect("a corner mask the preprocessed tree does not carry");
                        pre_idx.push(g);
                        (0, g)
                    }
                    _ => panic!("a dequant column named on a band"),
                };
                terms.push((tree, g, *w));
            }
            heights.push(crate::gkrlink::TieHeight {
                h: l.h, mle_point: l.mle_point.clone(), konst: l.konst, terms,
                q_at: t2, coeff_at: t2 + 4,
            });
        }
        link_tie = Some(crate::gkrlink::LinkTie { heights, pre_idx });
    }

    let exp_t = FrameworkComponent::<ExpTableEval>::new(
        alloc,
        ExpTableEval { log_rows: p.log_table, rel: rels.exp.clone() },
        unqm(claims.table_sums[0]),
    );
    let mut tables = vec![];
    let kinds: Vec<(u32, Vec<PreProcessedColumnId>, TableKind)> = vec![
        (plan.log_norm, NORM_COLS.iter().map(|c| id(*c)).collect(),
         TableKind::Norm(rels.norm.clone())),
        (plan.log_frac, FRAC_COLS.iter().map(|c| id(*c)).collect(),
         TableKind::Frac(rels.frac.clone())),
        (plan.log_mul, MUL_COLS.iter().map(|c| id(*c)).collect(),
         TableKind::Mul(rels.mul.clone())),
        (p.ao_q_bits, QNZ_COLS.iter().map(|c| id(*c)).collect(),
         TableKind::Pair(rels.qnz.clone())),
    ];
    for (t, (lr, cols, kind)) in kinds.into_iter().enumerate() {
        tables.push(FrameworkComponent::<TableEval>::new(
            alloc,
            TableEval { log_rows: lr, cols, kind },
            unqm(claims.table_sums[1 + t]),
        ));
    }
    for (t, w) in plan.widths.iter().enumerate() {
        tables.push(FrameworkComponent::<TableEval>::new(
            alloc,
            TableEval { log_rows: *w, cols: vec![tbl_range(*w)],
                        kind: TableKind::Range(rels.range(*w).clone()) },
            unqm(claims.table_sums[5 + t]),
        ));
    }

    let mut gtables = vec![];
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let base = 5 + plan.widths.len();
        for (t, (lr, cols, kind)) in
            crate::glue::glue_table_kinds(&gp.uns, gr).into_iter().enumerate()
        {
            gtables.push(FrameworkComponent::<GTableEval>::new(
                alloc,
                GTableEval { log_rows: lr, cols, kind },
                unqm(claims.table_sums[base + t]),
            ));
        }
    }

    // stream E: the argmax, tied to the head's key claim.  It shares the layer's
    // 16-bit range table and the head dequant's step selectors, so it adds one
    // component and no table.
    let argmax = plan.argmax_mat().map(|mi| {
        let lay = plan.mat_lays[mi];
        let m = &plan.p.mats[mi];
        let tie = crate::argmax::ATie {
            claimed: claims.am_key.first().copied().map(unqm).unwrap_or(SecureField::zero()),
            tag: m.name.clone(),
            ..argmax_tie_w(w_glue, m.tokens, m.n_out, lay.per_row)
        };
        FrameworkComponent::<crate::argmax::ArgmaxEval>::new(
            alloc,
            crate::argmax::ArgmaxEval {
                log_rows: lay.log_rows,
                per_row: lay.per_row,
                lax: plan.p.head_argmax_lax,
                rk: rels.range(crate::argmax::KEY_BITS).clone(),
                tie: Some(tie),
            },
            next_use(claims),
        )
    });

    // stream BB: the embedding gadget, after the argmax and after everything
    // else.  The use region takes the last `use_sums` cursor position and the
    // table region the last `table_sums` slot, which is what keeps both lists
    // append-only.  Its tie reads the wiring family's own `w` and `lam`, so it
    // draws no challenge of its own.
    let (embed, emb_tab) = match (&plan.emb, &rels.emb) {
        (Some(l), Some(er)) => {
            let g = plan.glue.as_ref().expect("the embedding gadget needs the wiring family");
            let mut tie = crate::embed::tie_w(w_glue, lam_glue, g.t.d, p.s, l);
            tie.claimed =
                claims.embed_out.first().copied().map(unqm).unwrap_or(SecureField::zero());
            tie.claimed_tok =
                claims.embed_tok.first().copied().map(unqm).unwrap_or(SecureField::zero());
            let em = FrameworkComponent::<crate::embed::EmbedEval>::new(
                alloc,
                crate::embed::EmbedEval {
                    log_rows: l.log_use,
                    per_row: l.pe,
                    rel: er.clone(),
                    rel_ef: rels.range(crate::embed::EF_BITS).clone(),
                    tie,
                },
                next_use(claims),
            );
            let et = FrameworkComponent::<crate::embed::EmbTableEval>::new(
                alloc,
                crate::embed::EmbTableEval { log_rows: l.log_tab, per_row: l.pe,
                                             rel: er.clone() },
                unqm(claims.table_sums[plan.n_table_sums() - 1]),
            );
            (Some(em), Some(et))
        }
        _ => (None, None),
    };

    BuiltComponents { chans, keys, rows, links, link_tie, ao, mats, walks, glue, exp_t, tables,
                      gtables, argmax, embed, emb_tab, pows,
                      has_rows: plan.rows_band.clone() }
}

/// Stream E: the argmax side's tie weights -- the mirror of the head dequant's
/// output side.  Its index map ascends the vocabulary (`i = (chunks-1-b) pr + sl`)
/// where the dequant's descends, so the ordinary row step is the *inverse* of the
/// output side's and the boundary step differs in the token term's sign.
fn argmax_tie_w(w: SecureField, tokens: usize, vocab: usize, per_row: usize)
                -> crate::argmax::ATie {
    let n = (tokens * vocab) as i128;
    let (tok, pr, v) = (tokens as i128, per_row as i128, vocab as i128);
    crate::argmax::ATie {
        w: (0..per_row)
            .map(|sl| wpow(w, n - 1 - (v - pr) * tok - sl as i128 * tok))
            .collect(),
        k_in: wpow(w, pr * tok),
        k_bd: wpow(w, -((v - pr) * tok + 1)),
        claimed: SecureField::zero(),
        tag: String::new(),
    }
}

// ------------------------------------------------------------------ proving ----

pub struct Composed {
    pub proof: Option<Proof>,
    pub prove_error: Option<String>,
    pub stats: RawStats,
    pub fixed: Fixed,
    pub claims: Claims,
    pub trace_secs: f64,
    pub interaction_secs: f64,
    pub ref_bad: usize,
    pub off_table: usize,
    /// fresh cells by family: bands, attnout, dequant, walks, glue, tables
    pub family_cells: Value,
    pub n_columns: [usize; 3],
    /// B10 step 2: the glue regions' own reference and B1 comparisons, and the
    /// tie claims each of them recomputed from the witness rather than the trace
    pub glue_ties_ok: bool,
    /// Stream H: what the boundary trees cost this chunk, or null when the flag
    /// was off and there are none
    pub bounds: Value,
    /// Stream AA: the same for the K/V boundary trees -- per side, its tree,
    /// its head band group, whether it is the out or the in side, the bands
    /// that wrote it, its columns and its cells.  Null without `--kv-bind`.
    pub kv_bounds: Value,
}

/// Tampers that need a hook inside the prover (everything else corrupts the
/// `Layer` before calling in).
#[derive(Default, Clone, Copy)]
pub struct ProverTamper {
    /// flip one row of the shared normalisation table, witness rebuilt to match
    pub norm_row: Option<usize>,
    /// B10 step 2: corrupt one glue region's own copy of the stream it shares
    /// with another region -- `(region index, value index)`.  Everything inside
    /// the region stays consistent, so what has to reject is the tie.
    pub glue_stream: Option<(usize, usize)>,
    /// a tamper inside one glue region's arithmetic, by region index
    pub glue: Option<(usize, crate::glue::Tamper)>,
    /// Stream H: corrupt one cell of *one* boundary tree, leaving the region it
    /// is tied to alone -- `(which side, which cell)`, side 0 being the input
    /// and side 1 the output.  Nothing else in the chunk changes, so what has to
    /// reject is that side's equality constraint: the probe that says the root
    /// is bound to the proof and not merely appended to it.
    ///
    /// Stream AA: the side index runs over `plan.bounds` and then over
    /// `plan.kv`, so side `plan.bounds.len() + j` is the `j`th K/V side --
    /// which for a chunk with a prefix is group `j / 2`'s out side when `j` is
    /// even and its in side when `j` is odd.
    pub bounds: Option<(usize, usize)>,
    /// Stream E: corrupt the wiring family's own copy of a dequant output --
    /// `(region index, value index)` into `o_out` for `gn1` or `d_out` for
    /// `gn2`.  The region recomputes its add from the corrupted value, so
    /// everything inside the glue stays consistent and the only thing that can
    /// catch it is the dequant output tie.
    pub glue_add: Option<(usize, usize)>,
    /// Stream E: a tamper inside the composed argmax region
    pub argmax: Option<crate::argmax::Tamper>,
    /// Stream BB: a tamper inside the composed embedding region.  `TokRow` is
    /// the aimed one -- the id moved to another valid entry with the committed
    /// row recomputed to match, so every constraint inside the region holds and
    /// only the tie to the layer's `bin` rejects.
    pub embed: Option<crate::embed::Tamper>,
    /// Stream Z, the four wired-site bends.  Each bends exactly one committed
    /// cell after the honest value is derived, so the constraint that pins the
    /// cell is the one that has to reject.  `fold_qm` bends a fold slot's
    /// quotient by +1 (the high remainder leaves its range); `fold_dent`
    /// doubles one row's committed `dent` (the `denp14 = dent pj` pin fails);
    /// `ao_clamp` pushes one attnout quotient past the pair table's domain;
    /// `walk_over` sets one committed activation of the widened walk to
    /// `2^act_bits`, one past its range table.
    pub fold_qm: Option<usize>,
    pub fold_dent: Option<usize>,
    pub ao_clamp: Option<usize>,
    pub walk_over: Option<usize>,
}

pub fn prove_layer(layer: &Layer, params: &Params, tamper: &ProverTamper) -> Composed {
    // B7a milestone 4 (19.6): 4a counts bytes host to device across the whole
    // wall, and the fill's uploads happen before `Session::new` exists.
    crate::device::reset();
    let g = generate(layer, params, tamper, None);
    prove_gen(layer, params, g)
}

/// B7b (README 26): everything `generate` built, owned, so a chunk's
/// generation can be handed from the thread that made it to the one that
/// proves it.
pub struct Gen {
    pub plan: Plan,
    pub tables: Tables,
    pub band_committed: Vec<Committed>,
    pub band_dev: Vec<Option<crate::devwfill::DevBand>>,
    pub band_pre: Vec<Option<crate::devpre::DevPreBand>>,
    pub aor: crate::layer::AoRegion,
    pub mat_regions: Vec<crate::layer::MatRegion>,
    pub walk_regions: Vec<crate::layer::WalkRegion>,
    // The glue family, which postdates the split: its regions, the owned
    // columns of its shared tables, and its against-B1 summary.
    pub glue_regions: Vec<crate::glue::GlueRegion>,
    pub glue_tcols: Vec<Eval>,
    pub glue_mcols: Vec<Eval>,
    pub glue_b1: Value,
    /// stream E: the argmax region, when it is composed
    pub am_region: Option<crate::argmax::ARegion>,
    /// stream BB: the embedding gadget's two regions, when it is composed
    pub emb_region: Option<crate::embed::ERegion>,
    pub n_deq_base: usize,
    pub pre_tables: Vec<Option<Eval>>,
    pub base_mults: Vec<Option<Eval>>,
    pub pre_logs: Vec<u32>,
    pub mult_logs: Vec<u32>,
    pub tab_pre: Vec<stwo::stwo_cuda::base_field_vec::BaseFieldVec>,
    pub tab_mult: Vec<stwo::stwo_cuda::base_field_vec::BaseFieldVec>,
    pub m4e_tabdev: bool,
    pub trace_secs: f64,
    /// what names this chunk's preprocessed tree (`chunks::key_of`); `None`
    /// is the single-proof path, which never caches
    pub pre_key: Option<u64>,
    /// Stream H: the boundary tamper, carried across the generate/prove split
    /// because the boundary trees are built where they are committed.
    pub bounds_tamper: Option<(usize, usize)>,
}

/// B7b (README 26): the first half of `prove_layer` -- every fill, every
/// table, every column born where it will be committed -- up to the
/// transcript.
pub fn generate(layer: &Layer, params: &Params, tamper: &ProverTamper, pre_key: Option<u64>)
                -> Gen {
    let m4_t = std::time::Instant::now();
    crate::util::m4_touch_device();
    crate::util::m4_add("wall/cuda_init", m4_t.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    let plan = Plan::new(params);
    let p = &plan.p;
    // stream AA: with the prefix tiled, `band_sizes` is no longer `p.bands`
    // followed by the corners -- it is one entry per *band*, which is what the
    // plan's shapes are.  Comparing against those is the same check untiled and
    // the right one tiled.
    assert_eq!(layer.bands.len(), plan.shapes.len(), "layer bands disagree with the plan");
    assert!(layer.band_sizes.iter().zip(&plan.shapes).all(|(n, sh)| *n == sh.insts),
            "layer bands disagree with params");
    assert_eq!(layer.bands[0].log_table, p.log_table, "exp table rows disagree");

    let mut tables = Tables::new(layer.bands[0].table.clone(), p.log_table, p.kmax,
                                 plan.round, p.ao_q_bits, plan.widths.clone());
    if let Some(i) = tamper.norm_row {
        let i = i % tables.norm.len();
        tables.norm[i].m7 = (tables.norm[i].m7 + 1) % 128;
    }

    // ---- region traces -------------------------------------------------------
    // B7a milestone 4 (18.5): the regions keep their columns and the two flat
    // lists are built from references once generation is done. `pre_tables`
    // and `base_mults` are the only columns with no region of their own -- and
    // they are exactly the ones the interaction's table sides read.
    // B7a milestone 4 (section 3 row 1): a slot with no host column is one the
    // device builds -- `x`, `y` and `m_exp`. The log sizes travel beside the
    // lists because that is the one thing the commitment needs of a column and
    // a missing `Eval` cannot say it.
    let mut pre_tables: Vec<Option<Eval>> = vec![];
    let mut base_mults: Vec<Option<Eval>> = vec![];
    let mut pre_logs: Vec<u32> = vec![];
    let mut mult_logs: Vec<u32> = vec![];
    let mut band_committed: Vec<Committed> = vec![];
    // B7a milestone 4 (section 3 row 1): the band columns, when they are on the
    // device. Parallel to `band_committed` because `Committed` belongs to the
    // standalone path too and has no business knowing about a backend.
    let mut band_dev: Vec<Option<crate::devwfill::DevBand>> = vec![];
    let m4_t = std::time::Instant::now();
    // B11 stream M: before the first fill, because the fills read the answer.
    //
    // `devpre::fills_dev` decides whether the ten multiplicity columns are
    // built on the card from device counters alone, and it is asked from inside
    // the fills, where the layer is not in scope. It has to be told about every
    // fill that bumps one of those counters and is *not* on the card, for a
    // reason no switch states. There are two:
    //
    //  * a **corner band**, which fills on the host whatever
    //    `B7_M4_WFILL_BAND` says because the device fill has no notion of its
    //    layout (`devwfill::band_fill_mode`, the one place that decision
    //    lives);
    //  * the **argmax region**, which has no device fill at all and whose range
    //    lookups go on the layer's 16-bit table -- the block below takes
    //    `tables.m_r[ri]` and bumps it on the host.
    //
    // Both were silent before this stream because `B7_M4_PRE` had never been
    // on: with the columns host-built the split does not matter. With it on,
    // the kernel sums the device counters and the host half is simply dropped,
    // and the proof is rejected by `sums_cancel` with every other check green.
    crate::devpre::note_host_fill(
        layer.bands.iter().any(|a| crate::devwfill::band_fill_mode(a) != 1)
            // stream AA: a band that does not carry the per-token region fills
            // on the host too (`commit_base_any`), and `band_fill_mode` has no
            // way to know that -- it is told the `Attn`, not the plan
            || plan.rows_band.iter().any(|r| !*r)
            || plan.argmax_mat().is_some()
            // stream BB: the embedding gadget has no device fill either, and
            // its `ef` range checks bump the layer's own 8-bit table on the host
            || plan.emb.is_some()
            // stream Z: the fold region and the widened attention output's walk
            // both fill on the host, whatever the switches say
            || plan.specs.iter().any(|s| s.variant == DeqVariant::Fold)
            || plan.walks.iter().any(|w| w.act_bits != 8),
    );
    let mut mult_exp = std::mem::take(&mut tables.m_exp);
    let mut mult_r = std::mem::take(&mut tables.m_r);
    for (b, a) in layer.bands.iter().enumerate() {
        let l = &plan.lays[b];
        // B7a milestone 4 (section 3 row 1): 48 columns of ACTIVE / NOFIRST /
        // LAST / ISTART, which `b65_acc.cu` has computed since 24.1 -- at the
        // wrong time. Mode 1 never builds them here; 2 builds them and asserts
        // every cell against the kernel's below.  A corner band always builds
        // and keeps its `pre`: the interaction phase reads the causal-mask
        // columns back (the exp lookup's multiplicity is `1 - msk`), and its
        // fill is host-side, so devpre never owns its selectors.
        let pre = if crate::devpre::mode() == 1 && !a.corner {
            vec![]
        } else {
            attn_trace::commit_pre(a, l, false, plan.rows_band[b])
        };
        let (base, at, dev) =
            attn_trace::commit_base_any(a, l, &plan.cfg, &plan.widths, &mut mult_exp,
                                        &mut mult_r, &mut tables.off_table, false,
                                        plan.rows_band[b]);
        band_dev.push(dev);
        band_committed.push(Committed { pre, base, at, off_table: 0 });
    }
    tables.m_exp = mult_exp;
    tables.m_r = mult_r;
    crate::util::m4_add("gen/bands", m4_t.elapsed().as_secs_f64());

    // B7a milestone 4 (section 3 row 1): the flags, and the 48 selector columns
    // built out of them. The flags are wanted whenever the accumulators will
    // run -- which is whenever the band's fill was on the device -- so this
    // happens one line after that fill and not inside the switch.
    let m4_t = std::time::Instant::now();
    let pre_mode = crate::devpre::mode();
    // The shared tables' 31 columns share one flat list with `x`, `y` and
    // `m_exp`, so there is no configuration where this row's half of it is on
    // the device and 23's is not.
    assert!(pre_mode == 0 || crate::devtab::mode() >= 1,
            "B7_M4_PRE needs B7_M4_TABDEV: the shared tables are one list");
    let band_pre: Vec<Option<crate::devpre::DevPreBand>> = layer
        .bands
        .iter()
        .enumerate()
        .map(|(b, a)| {
            // B11 stream N, lever 2 (b): a corner band gets its flags too, once
            // its fill is on the card -- the accumulators segment on them and
            // `interaction_phase` asserts they are there. Its *columns* are a
            // separate question: `commit_pre` still builds a corner's selectors
            // on the host below, because the interaction phase reads the
            // causal-mask columns back, so `want_cols` stays false for it.
            let corner_dev = a.corner && band_dev[b].is_some();
            ((!a.corner && (pre_mode >= 1 || band_dev[b].is_some())) || corner_dev)
                .then(|| crate::devpre::band_dev(a, &plan.lays[b], pre_mode >= 1 && !a.corner,
                                                 plan.rows_band[b]))
        })
        .collect();
    if pre_mode == 2 {
        for (b, d) in band_pre.iter().enumerate() {
            let Some(d) = d.as_ref() else { continue };
            let h: Vec<&Eval> = band_committed[b].pre.iter().collect();
            crate::devpre::check_run(&format!("band{b} selectors"), &h, &d.cols);
        }
    }
    crate::util::m4_add("gen/pre_bands", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    let aor = ao_region(&layer.ao, p.ao_target, p.ao_q_bits, p.ao_rem_bits, &mut tables,
                        None, tamper.ao_clamp);

    crate::util::m4_add("gen/attnout", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    let mut mat_regions = vec![];
    for (m, sp) in layer.mats.iter().zip(&plan.specs) {
        let r = mat_region(m, sp, &mut tables, plan.has_oside(&m.name),
                           (tamper.fold_qm, tamper.fold_dent));
        mat_regions.push(r);
    }
    let n_deq_base: usize = mat_regions
        .iter()
        .map(|r| r.base_cells())
        .sum();

    crate::util::m4_add("gen/dequant", m4_t.elapsed().as_secs_f64());

    // Stream E: the argmax region, on the head dequant's own layout so the tie
    // is positionwise.  Its range lookups go on the layer's 16-bit table, so its
    // multiplicities are bumped into that table's vector the way the wiring
    // family's go into `NORM` and `FRAC`.
    let m4_t = std::time::Instant::now();
    let am_region = plan.argmax_mat().map(|mi| {
        let m = &plan.p.mats[mi];
        let keys = layer.head_keys.as_ref().expect("--head-argmax needs key_head.u16");
        let tok = layer.head_tok.as_ref().expect("--head-argmax needs tok_head.i32");
        let w = crate::argmax::AWitness::from_cols(
            m.tokens, m.n_out, keys.iter().map(|k| u32::from(*k)).collect(), tok.clone())
            .expect("the head dump's argmax columns");
        let ri = tables.widths.iter().position(|w| *w == crate::argmax::KEY_BITS)
            .expect("the argmax needs the layer's 16-bit range table (--z-lo-bits 16)");
        let mut m_rk = std::mem::take(&mut tables.m_r[ri]);
        let r = crate::argmax::region(&w, plan.p.head_argmax_lax, plan.mat_lays[mi],
                                      &mut m_rk, tamper.argmax);
        tables.m_r[ri] = m_rk;
        tables.off_table += r.off_table;
        r
    });
    crate::util::m4_add("gen/argmax", m4_t.elapsed().as_secs_f64());

    // Stream BB: the embedding gadget.  Its use side commits the layer's own
    // residual-stream input -- the wiring family's `gn1` group 0, value for
    // value -- so that the tie it publishes and the one the wiring family
    // publishes are two claims over the same sequence.  Its `ef` range lookups
    // go on the layer's 8-bit table, exactly as the argmax's go on the 16-bit
    // one.
    let m4_t = std::time::Instant::now();
    let emb_region = plan.emb.map(|l| {
        let gc = layer.glue.as_ref()
            .expect("`--embed` ties to the wiring family; extract with `--glue`");
        let tab = layer.embed_tab.as_ref()
            .expect("`--embed` needs `embed.u16` in the dump");
        let tok = layer.tok_in.as_ref().expect("`--embed` needs `tok_in.i32` in the dump");
        let w = crate::embed::EWitness::from_cols(
            p.s, gc.d, p.embed_vocab, tab.clone(), tok.clone(), gc.xin.clone())
            .expect("the dump's embedding columns");
        let ri = tables.widths.iter().position(|w| *w == crate::embed::EF_BITS)
            .expect("the embedding gadget needs the layer's 8-bit range table");
        let mut m_ef = std::mem::take(&mut tables.m_r[ri]);
        let r = crate::embed::region(&w, l, &mut m_ef, tamper.embed);
        tables.m_r[ri] = m_ef;
        tables.off_table += r.off_table;
        r
    });
    crate::util::m4_add("gen/embed", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    let mut walk_regions = vec![];
    for w in &plan.walks {
        let r = walk_region(layer, &w.name, &w.mats, p.walk_p, w.act_bits, &mut tables,
                            None, tamper.walk_over.filter(|_| w.act_bits != 8));
        walk_regions.push(r);
    }
    crate::util::m4_add("gen/walks", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    // ---- B10 step 2: the wiring family -------------------------------------
    // The glue's `NORM`/`FRAC` uses are *moved into* the layer's own multiplicity
    // vectors and moved back, so the two families' lookups land in one column
    // apiece -- which is the real integration test: the shared tables' sums only
    // cancel if both families agree on every row they read.
    let mut glue_regions: Vec<crate::glue::GlueRegion> = vec![];
    let mut gtables: Option<GlueTables> = None;
    let mut glue_b1: Value = json!(null);
    if let Some(gp) = &plan.glue {
        let gc = layer.glue.as_ref().expect("params ask for the glue, the witness has none");
        assert_eq!(gc.d, gp.t.d, "the witness's d disagrees with the glue params");
        let mut gt = GlueTables::new(p.kmax, plan.round, &gp.uns);
        gt.norm = tables.norm.clone();
        gt.m_norm = std::mem::take(&mut tables.m_norm);
        gt.m_frac = std::mem::take(&mut tables.m_frac);
        let t = gp.t;
        let rows1 = t.rows1();
        let mut mismatch = vec![];
        // what the two reduction regions read as their stream, kept so a tail
        // region reads the identical block -- the tie compares those, and B1's
        // own column is the separate check
        let mut tail_in: Vec<Vec<u16>> = vec![];
        // stream 0: xin + o_out into xmid, and the squares' first levels;
        // stream 1: xmid + d_out into xout, and norm2's
        let mut stream = t.lay(&gc.xin);
        for (k, (add, want, wname)) in
            [(&gc.o_out, &gc.xmid, "xmid"), (&gc.d_out, &gc.xout, "xout")].iter().enumerate()
        {
            let i1 = 2 * k;
            let (prog1, tag1) = (gp.regions[i1].prog.clone(), gp.regions[i1].tag.clone());
            let (prog2, tag2) =
                (gp.regions[i1 + 1].prog.clone(), gp.regions[i1 + 1].tag.clone());
            if let Some((ri, vi)) = tamper.glue_stream {
                if ri == i1 {
                    let n = stream.len();
                    stream[vi % n] ^= 1;
                }
            }
            let mut addv = t.lay(add);
            if let Some((ri, vi)) = tamper.glue_add {
                if ri == i1 {
                    let n = addv.len();
                    addv[vi % n] ^= 1;
                }
            }
            tail_in.push(stream.clone());
            let w1 = crate::glue::GlueWitness::new(
                rows1, vec![stream.clone(), addv], &prog1, plan.round,
                format!("B1 residual stream [{} x {}]", t.tokens, t.d));
            let tam1 = tamper.glue.filter(|(ri, _)| *ri == i1).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r1 = crate::glue::glue_region(&w1, &prog1, plan.round, &tag1, &mut gt, tam1);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            // what the region computed is what the next one reads: that is what
            // the tie says, and it is checked against B1's own column here
            let got = t.unlay(&r1.outs[1]);
            let bad = (0..t.tokens * t.d)
                .filter(|i| crate::glue::canon(got[*i]) != crate::glue::canon(want[*i]))
                .count();
            mismatch.push(json!({"column": wname, "values": t.tokens * t.d, "mismatch": bad}));
            let l2 = t.roots_to_l2(&r1.outs[0]);
            let w2 = crate::glue::GlueWitness::new(
                t.tokens, vec![l2], &prog2, plan.round,
                format!("the level-1 roots [{} x {}]", t.tokens, t.a));
            let tam2 = tamper.glue.filter(|(ri, _)| *ri == i1 + 1).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r2 = crate::glue::glue_region(&w2, &prog2, plan.round, &tag2, &mut gt, tam2);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            stream = t.lay(&got);
            glue_regions.push(r1);
            glue_regions.push(r2);
        }
        // ---- stream E: h = (x * r) * w, once per norm -------------------------
        for (k, i) in gp.tails.iter().copied().enumerate() {
            let (rcol, wcol, want, wname) = if k == 0 {
                (&gc.r1, &gc.w1, &gc.h1, "x_q")
            } else {
                (&gc.r2, &gc.w2, &gc.h2, "x_gate")
            };
            let prog = gp.regions[i].prog.clone();
            let tag = gp.regions[i].tag.clone();
            // `r` is per token and a row is one `(residue, token)` pair, so the
            // row's single leaf is its own token's `r`; `w` is per channel and
            // constant down the tokens, so the row reads channels `res + j a`.
            let rblk: Vec<u16> =
                (0..rows1).map(|m| crate::glue::canon(rcol[m % t.tokens])).collect();
            // B11 stream L, anatomy: the same shape of serial gather as the
            // SwiGLU's below -- `rows1 * nreal` canonicalisations, one row
            // independent of the next.
            let t_wb = std::time::Instant::now();
            let mut wblk = vec![0u16; rows1 * t.nreal];
            wblk.par_chunks_mut(t.nreal).enumerate().for_each(|(m, row)| {
                let res = m / t.tokens;
                for j in 0..t.nreal {
                    row[j] = crate::glue::canon(wcol[res + j * t.a]);
                }
            });
            crate::util::m4_add("gen/glue_wblk", t_wb.elapsed().as_secs_f64());
            let w = crate::glue::GlueWitness::new(
                rows1, vec![tail_in[k].clone(), rblk, wblk], &prog, plan.round,
                format!("B1 norm{} tail [{} x {}]", k + 1, t.tokens, t.d));
            let tam = tamper.glue.filter(|(ri, _)| *ri == i).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r = crate::glue::glue_region(&w, &prog, plan.round, &tag, &mut gt, tam);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            // the tail's output is the q/k/v (resp. gate/up) matmuls' own bf16
            // input column, so B1 has already written down what it must be
            let t_b1 = std::time::Instant::now();
            if let Some(wv) = want {
                let got = t.unlay(&r.outs[0]);
                let bad = (0..t.tokens * t.d)
                    .into_par_iter()
                    .filter(|i| crate::glue::canon(got[*i]) != crate::glue::canon(wv[*i]))
                    .count();
                mismatch.push(json!({"column": wname, "values": t.tokens * t.d,
                                     "mismatch": bad}));
            } else {
                mismatch.push(json!({"column": wname, "values": 0,
                                     "mismatch": serde_json::Value::Null,
                                     "note": "the dump does not carry this column"}));
            }
            crate::util::m4_add("gen/glue_b1", t_b1.elapsed().as_secs_f64());
            glue_regions.push(r);
        }
        // ---- stream E: silu(gate) * up --------------------------------------
        if let (Some(i), Some(tf)) = (gp.swiglu, gp.tff) {
            let find = |n: &str| -> usize {
                layer.mats.iter().position(|m| m.name == n)
                    .unwrap_or_else(|| panic!("the SwiGLU needs the {n} matmul"))
            };
            let (gi, ui) = (find("gate"), find("up"));
            assert_eq!(layer.mats[gi].n_out, tf.d,
                       "the gate matmul's width is not the SwiGLU's dff");
            // The two matmuls' dequantized outputs, read through the dequant
            // region's own reference so this region consumes exactly what that
            // region proves.  What is *not* here is the tie pinning them to it:
            // that is the open dequant tie, and until it lands these leaves are
            // pinned only by their VAL lookups.
            // B11 stream L, anatomy: this is `tokens * d` calls to `deq_ref`,
            // twice -- 4.2 M of them on a C=4K chunk's `gn1` -- and it was a
            // serial nested loop inside the one term of trace generation that
            // has no device path at all. `deq_ref` is a pure function of the
            // matmul, its spec and the (token, channel) pair, so the rows are
            // independent and rayon is the whole change.
            let t_dq = std::time::Instant::now();
            let mut gate_out = vec![0u16; tf.tokens * tf.d];
            let mut up_out = vec![0u16; tf.tokens * tf.d];
            gate_out
                .par_chunks_mut(tf.d)
                .zip(up_out.par_chunks_mut(tf.d))
                .enumerate()
                .for_each(|(t, (g, u))| {
                    for c in 0..tf.d {
                        g[c] = crate::layer::deq_ref(&layer.mats[gi], &plan.specs[gi], t, c);
                        u[c] = crate::layer::deq_ref(&layer.mats[ui], &plan.specs[ui], t, c);
                    }
                });
            crate::util::m4_add("gen/glue_deqref", t_dq.elapsed().as_secs_f64());
            let prog = gp.regions[i].prog.clone();
            let tag = gp.regions[i].tag.clone();
            let w = crate::glue::GlueWitness::new(
                tf.rows1(), vec![tf.lay(&gate_out), tf.lay(&up_out)], &prog, plan.round,
                format!("the gate and up dequant outputs [{} x {}]", tf.tokens, tf.d));
            let tam = tamper.glue.filter(|(ri, _)| *ri == i).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r = crate::glue::glue_region(&w, &prog, plan.round, &tag, &mut gt, tam);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            let n = tf.tokens * tf.d;
            let cmp = |got: &[u16], want: &[u16]| -> usize {
                (0..n)
                    .into_par_iter()
                    .filter(|q| crate::glue::canon(got[*q]) != crate::glue::canon(want[*q]))
                    .count()
            };
            // three columns B1 wrote for this gadget: the gate's dequantized
            // output (the silu's input), the silu image the region's table
            // supplies, and the product `down` reads
            // B11 stream L, anatomy: everything from here to the end of the
            // block is a *check* against B1's dumped columns, not a committed
            // column -- three serial passes over `tokens * d` values. Timed
            // apart so the record can say what the glue's seconds actually buy.
            let t_b1 = std::time::Instant::now();
            if let Some(v) = &gc.silu_in {
                mismatch.push(json!({"column": "silu_in", "values": n,
                                     "mismatch": cmp(&gate_out, v)}));
            }
            if let Some(v) = &gc.silu_out {
                let img: Vec<u16> = gate_out.iter()
                    .map(|p| crate::glue::un_image(UnKind::Silu, *p)).collect();
                mismatch.push(json!({"column": "silu_out", "values": n,
                                     "mismatch": cmp(&img, v)}));
            }
            if let Some(v) = &gc.x_down {
                mismatch.push(json!({"column": "x_down", "values": n,
                                     "mismatch": cmp(&tf.unlay(&r.outs[0]), v)}));
            }
            crate::util::m4_add("gen/glue_b1", t_b1.elapsed().as_secs_f64());
            glue_regions.push(r);
        }
        tables.m_norm = std::mem::take(&mut gt.m_norm);
        tables.m_frac = std::mem::take(&mut gt.m_frac);
        tables.off_table += gt.off_table;
        gt.off_table = 0;
        glue_b1 = json!({"against_b1": mismatch,
                         "reference_mismatch": glue_regions.iter()
                             .map(|r| r.ref_n - r.ref_ok).sum::<usize>(),
                         "exponent_out_of_range":
                             glue_regions.iter().map(|r| r.bad_exp).sum::<usize>()});
        gtables = Some(gt);
    }
    crate::util::m4_add("gen/glue", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    // ---- shared tables (tree 0 tail) and their multiplicities (tree 1 tail) --
    // The glue's shared tables, built once and owned here: the tree assembly
    // below references them last (the order the by-value fill used), and the
    // table-sides block reads them again.
    let glue_tcols: Vec<Eval> =
        gtables.as_ref().map(crate::glue::glue_table_cols).unwrap_or_default();
    let glue_mcols: Vec<Eval> =
        gtables.as_ref().map(crate::glue::glue_mult_cols).unwrap_or_default();
    let tbl_rows = 1usize << p.log_table;
    // B7a milestone 4 (section 3 row 1): the 31 columns 23.1 called "data, not
    // closed forms". Measured column by column that is wrong twice over: the
    // ten multiplicities are counters the fills already scatter-added into, and
    // the normalisation, fraction and multiply rows are closed forms of the row
    // index -- a leading-zero count, a shift and one rounding step apiece. Mode
    // 1 never builds them here; 2 builds them and asserts every cell.
    let host_rest = pre_mode != 1;
    let host_mult = crate::devpre::merge_counters();
    // the log sizes of the 21 and the 10, which the two flat lists need whether
    // or not a host column exists to carry one
    let rest_logs: Vec<u32> = [
        vec![plan.log_norm; 6], vec![plan.log_frac; 3], vec![plan.log_mul; 4],
        vec![p.ao_q_bits; 2], plan.widths.clone(),
    ].concat();
    let mult_logs_rest: Vec<u32> = [
        vec![plan.log_norm, plan.log_frac, plan.log_mul, p.ao_q_bits],
        plan.widths.clone(),
    ].concat();
    let mut pre_rest: Vec<Eval> = vec![];
    let mut mult_rest: Vec<Eval> = vec![];
    if host_rest {
        let get: [fn(&crate::bf16::NormRow) -> u32; 6] =
            [|r| r.m2, |r| r.ph, |r| r.st, |r| r.m7, |r| r.expfield, |r| r.nz];
        for g in get {
            pre_rest.push(eval_of(plan.log_norm, (0..1usize << plan.log_norm)
                .map(|i| m31_u32(tables.norm.get(i).map(g).unwrap_or(0))).collect()));
        }
        for t in 0..3 {
            pre_rest.push(eval_of(plan.log_frac, (0..1usize << plan.log_frac)
                .map(|i| {
                    let x = tables.frac.get(i).copied().unwrap_or((0, 0, 0));
                    m31_u32([x.0, x.1, x.2][t])
                }).collect()));
        }
        for t in 0..4 {
            pre_rest.push(eval_of(plan.log_mul, (0..1usize << plan.log_mul)
                .map(|i| {
                    let x = tables.mul.get(i).copied()
                        .unwrap_or(crate::bf16::MulRow { a: 0, b: 0, m7: 0, carry: 0 });
                    m31_u32([x.a, x.b, x.m7, x.carry][t])
                }).collect()));
        }
        pre_rest.push(eval_of(p.ao_q_bits, (0..1u32 << p.ao_q_bits).map(m31_u32).collect()));
        pre_rest.push(eval_of(p.ao_q_bits,
                             (0..1u32 << p.ao_q_bits)
                                 .map(|q| m31_u32(u32::from(q != 0))).collect()));
        for w in &plan.widths {
            pre_rest.push(eval_of(*w, (0..1u32 << w).map(m31_u32).collect()));
        }
    }
    if host_mult {
        mult_rest.push(eval_of(plan.log_norm,
                               tables.m_norm.iter().map(|m| m31_u32(*m)).collect()));
        mult_rest.push(eval_of(plan.log_frac,
                               tables.m_frac.iter().map(|m| m31_u32(*m)).collect()));
        mult_rest.push(eval_of(plan.log_mul,
                               tables.m_mul.iter().map(|m| m31_u32(*m)).collect()));
        mult_rest.push(eval_of(p.ao_q_bits,
                               tables.m_qnz.iter().map(|m| m31_u32(*m)).collect()));
        for (t, w) in plan.widths.iter().enumerate() {
            mult_rest.push(eval_of(*w, tables.m_r[t].iter().map(|m| m31_u32(*m)).collect()));
        }
    }

    // B7a milestone 4 (section 3 row 1): `x`, `y` and `m_exp`, 2^23 rows each
    // and 0.101 GB of the 0.166 GB 4a had left. `B7_M4_TABDEV=1` means the host
    // never builds them; 2 builds them here as well and asserts every cell
    // against the kernel's, which is how they are checked.
    let tab_mode = crate::devtab::mode();
    let host_big = tab_mode != 1;
    pre_logs.push(p.log_table);
    pre_tables.push(host_big.then(|| {
        eval_of(p.log_table, (0..tbl_rows).map(|i| m31_u32(i as u32)).collect())
    }));
    pre_logs.push(p.log_table);
    pre_tables.push(host_big.then(|| {
        eval_of(p.log_table, tables.exp.iter().map(|v| m31_u32(*v)).collect())
    }));
    if host_rest {
        for e in pre_rest {
            pre_logs.push(e.domain.log_size());
            pre_tables.push(Some(e));
        }
    } else {
        for l in &rest_logs {
            pre_logs.push(*l);
            pre_tables.push(None);
        }
    }
    // `m_exp` needs a counter to sum, and there is one only where a band's fill
    // ran on the device (21.1). With the bands on the host it stays a host
    // column and a 4a upload -- the honest accounting, not a special case.
    //
    // B11 stream L: and only where *every* band's are. A corner band fills on
    // the host, so on a corner-bearing witness the bumps are split and the
    // device half alone is not the column -- see `devwfill::exp_mult_on_host`,
    // which is the same predicate `commit_base_dev` was given, asked here of the
    // whole layer. With a corner present the device bands merged their counters
    // into `tables.m_exp`, so the host column is complete and the counter list
    // is dropped rather than summed into a column nothing commits.
    // B11 stream N, lever 2: this asked "is there a corner band?" because a
    // corner band's fill was always on the host. Now it asks the question it
    // always meant -- "is any band's fill on the host?" -- which is
    // `devwfill::exp_mult_on_host`'s twin read over the whole layer, and the
    // two have to move together or `sums_cancel` rejects with everything else
    // green.
    let host_corner = layer.bands.iter().any(|b| crate::devwfill::band_fill_mode(b) != 1);
    let dev_counters: Vec<&stwo::stwo_cuda::base_field_vec::Uint32Vec> = if host_corner {
        vec![]
    } else {
        band_dev.iter().filter_map(|b| b.as_ref().map(|d| &d.m_exp)).collect()
    };
    let host_m = host_big || dev_counters.is_empty();
    mult_logs.push(p.log_table);
    base_mults.push(host_m.then(|| {
        eval_of(p.log_table, tables.m_exp.iter().map(|m| m31_u32(*m)).collect())
    }));
    if host_mult {
        for e in mult_rest {
            mult_logs.push(e.domain.log_size());
            base_mults.push(Some(e));
        }
    } else {
        for l in &mult_logs_rest {
            mult_logs.push(*l);
            base_mults.push(None);
        }
    }
    crate::util::m4_add("gen/tables", m4_t.elapsed().as_secs_f64());

    // B7a milestone 4 (section 3 row 1): the shared tables' columns, uploaded
    // once. Both trees take them from here and so do the table families'
    // fraction kernels, so 4a counts each of these columns exactly once -- the
    // same column `take_all` used to upload, moved earlier.
    let m4_t = std::time::Instant::now();
    let m4e_tabdev = tab_mode >= 1;
    // B7a milestone 4 (section 3 row 1): the three closed-form columns, built
    // where they are used. The other 31 are uploaded exactly as 22.4 uploaded
    // all 34, so 4a still counts every shared-table column exactly once.
    let dev3 = m4e_tabdev.then(|| {
        crate::devtab::table_cols_dev(&layer.bands[0], p.log_table, &tables.exp,
                                      &dev_counters)
    });
    if tab_mode == 2 {
        let d = dev3.as_ref().unwrap();
        crate::devtab::check_col("tables x", crate::devtab::htab(&pre_tables, 0), &d.x);
        crate::devtab::check_col("tables y", crate::devtab::htab(&pre_tables, 1), &d.y);
        if let Some(m) = d.m.as_ref() {
            crate::devtab::check_col("tables m_exp",
                                     crate::devtab::htab(&base_mults, 0), m);
        }
    }
    // B7a milestone 4 (section 3 row 1): one counter list per shared table, in
    // the order the two flat lists lay them out. Every one of these arrays is
    // where a fill's scatter-add left it (20.1, 21.1), and the kernel sums them
    // there rather than downloading, merging and uploading -- which is what
    // deletes `gen/deq_dev_counters` and `gen/wfill_counters`.
    let mult_dev = crate::devpre::mult_dev();
    let mut pre_counters: Vec<Vec<&stwo::stwo_cuda::base_field_vec::Uint32Vec>> =
        vec![vec![]; 4 + plan.widths.len()];
    if mult_dev {
        let ri = |w: u32| 4 + plan.widths.iter().position(|x| *x == w).unwrap();
        let r8 = ri(8);
        for r in &mat_regions {
            let d = r.dev.as_ref().expect("B7_M4_FILL=1 but a dequant region is host-built");
            pre_counters[0].push(&d.m_norm);
            pre_counters[1].push(&d.m_frac);
            pre_counters[2].push(&d.m_mul);
            pre_counters[r8].push(&d.m_r8);
        }
        for r in &walk_regions {
            let d = r.dev.as_ref().expect("B7_M4_WFILL=1 but a walk group is host-built");
            pre_counters[r8].push(&d.m_r8);
        }
        let d = aor.dev.as_ref().expect("B7_M4_WFILL=1 but the attnout is host-built");
        pre_counters[3].push(&d.m_qnz);
        pre_counters[ri(p.ao_rem_bits)].push(&d.m_rem);
        for (b, a) in layer.bands.iter().enumerate() {
            // A corner band is host-built (the static fill kernels do not know
            // its layout), so its bumps sit in the host multiplicity arrays and
            // reach the table through the patch rows -- no device counter to sum.
            let Some(d) = band_dev[b].as_ref() else {
                assert!(a.corner || !plan.rows_band[b],
                        "B7_M4_WFILL=1 but a band that carries the per-token region is \
                         host-built");
                continue;
            };
            if plan.cfg.range_check {
                pre_counters[ri(a.k_bits)].push(&d.m_k);
                pre_counters[ri(a.q_bits)].push(&d.m_q);
                pre_counters[r8].push(&d.m_v);
            }
            if let crate::attn_witness::Variant::IntSplit(k) = a.variant {
                pre_counters[ri(k)].push(d.m_aux.as_ref().unwrap());
            }
            pre_counters[ri(plan.cfg.z_lo_bits)].push(&d.m_lo);
            pre_counters[ri(plan.cfg.z_hi_bits)].push(&d.m_hi);
        }
    }
    let dev_rest = (pre_mode >= 1).then(|| {
        crate::devpre::tables_dev(&tables, plan.log_norm, plan.log_frac, plan.log_mul,
                                  p.ao_q_bits, &plan.widths, &pre_counters, mult_dev)
    });
    if pre_mode == 2 {
        let d = dev_rest.as_ref().unwrap();
        let hp: Vec<&Eval> =
            (2..pre_tables.len()).map(|i| crate::devtab::htab(&pre_tables, i)).collect();
        crate::devpre::check_run("shared tables", &hp, &d.pre);
        if !d.mult.is_empty() {
            let hm: Vec<&Eval> = (1..base_mults.len())
                .map(|i| crate::devtab::htab(&base_mults, i)).collect();
            crate::devpre::check_run("shared multiplicities", &hm, &d.mult);
        }
    }
    let (tab_pre, tab_mult): (Vec<stwo::stwo_cuda::base_field_vec::BaseFieldVec>,
                              Vec<stwo::stwo_cuda::base_field_vec::BaseFieldVec>) =
        if m4e_tabdev {
            let d = dev3.unwrap();
            let mut pre = vec![d.x, d.y];
            let m0 = match d.m {
                Some(m) => m,
                None => crate::device::upload_col(crate::devtab::hone(&base_mults[0])),
            };
            let mut mult = vec![m0];
            match dev_rest {
                Some(r) => {
                    pre.extend(r.pre);
                    if r.mult.is_empty() {
                        mult.extend(base_mults[1..].iter()
                            .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
                    } else {
                        mult.extend(r.mult);
                    }
                }
                None => {
                    pre.extend(pre_tables[2..].iter()
                        .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
                    mult.extend(base_mults[1..].iter()
                        .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
                }
            }
            (pre, mult)
        } else {
            (vec![], vec![])
        };
    crate::util::m4_add("gen/tables_up", m4_t.elapsed().as_secs_f64());
    let trace_secs = t0.elapsed().as_secs_f64();
    Gen { plan, tables, band_committed, band_dev, band_pre, aor, mat_regions, walk_regions,
          glue_regions, glue_tcols, glue_mcols, glue_b1, am_region, emb_region,
          n_deq_base, pre_tables, base_mults, pre_logs, mult_logs, tab_pre, tab_mult,
          m4e_tabdev, trace_secs, pre_key, bounds_tamper: tamper.bounds }
}

/// B7b (README 26): the second half -- the session, the three commits, the
/// interaction, the components and the proof.
pub fn prove_gen(layer: &Layer, params: &Params, g: Gen) -> Composed {
    let Gen { plan, tables, band_committed, band_dev, band_pre, aor, mat_regions, walk_regions,
              glue_regions, glue_tcols, glue_mcols, glue_b1, am_region, emb_region,
              n_deq_base, pre_tables, base_mults, pre_logs, mult_logs, tab_pre, tab_mult,
              m4e_tabdev, trace_secs, pre_key, bounds_tamper } = g;
    let p = &plan.p;
    let t0 = std::time::Instant::now();

    // The two trees, as references, in the order the owned vectors used to be
    // filled: every band, then attnout, then the matmuls, then the walks, then
    // the shared tables and their multiplicities. `Op::Tree`'s column order is
    // part of the transcript, so this order is load-bearing and `proof_sha256`
    // is what checks it.
    let m4_t = std::time::Instant::now();
    use crate::device::AnyCol;
    let mut pre_all: Vec<AnyCol> = vec![];
    let mut base_all: Vec<AnyCol> = vec![];
    for ((c, d), pd) in band_committed.iter().zip(&band_dev).zip(&band_pre) {
        match pd.as_ref().filter(|p| !p.cols.is_empty()) {
            Some(p) => pre_all.extend(
                p.cols.iter().zip(&p.logs).map(|(c, l)| AnyCol::Dev(c, *l))),
            None => pre_all.extend(c.pre.iter().map(AnyCol::Host)),
        }
        match d {
            // B7a milestone 4 (section 3 row 1): a band's committed columns are
            // not all the same height -- three grids and a token block -- so the
            // log size travels with the column.
            Some(d) => base_all.extend(
                d.base.iter().zip(&d.logs).map(|(c, l)| AnyCol::Dev(c, *l))),
            None => base_all.extend(c.base.iter().map(AnyCol::Host)),
        }
    }
    match &aor.dev {
        Some(d) => {
            let l = aor.lay.log_rows;
            pre_all.extend(d.pre.iter().map(|c| AnyCol::Dev(c, l)));
            base_all.extend(d.base.iter().map(|c| AnyCol::Dev(c, l)));
        }
        None => {
            pre_all.extend(aor.pre.iter().map(AnyCol::Host));
            base_all.extend(aor.base.iter().map(AnyCol::Host));
        }
    }
    for r in &mat_regions {
        // B7a milestone 4 (19.6): under `B7_M4_FILL=1` this region's columns
        // are already on the device, in committed order, and this is where
        // they join the tree without an upload.
        match &r.dev {
            Some(d) => {
                let l = r.lay.log_rows;
                pre_all.extend(d.pre.iter().map(|c| AnyCol::Dev(c, l)));
                base_all.extend(d.base.iter().map(|c| AnyCol::Dev(c, l)));
            }
            None => {
                pre_all.extend(r.pre.iter().map(AnyCol::Host));
                base_all.extend(r.base.iter().map(AnyCol::Host));
            }
        }
    }
    for r in &walk_regions {
        match &r.dev {
            Some(d) => {
                let l = r.lay.g.log_rows;
                pre_all.extend(d.pre.iter().map(|c| AnyCol::Dev(c, l)));
                base_all.extend(d.base.iter().map(|c| AnyCol::Dev(c, l)));
            }
            None => {
                pre_all.extend(r.pre.iter().map(AnyCol::Host));
                base_all.extend(r.base.iter().map(AnyCol::Host));
            }
        }
    }
    // The glue regions' columns follow the walks, as they did when the lists
    // were built by value. B11 stream M: and they are on the card under
    // `B7_M4_GFILL=1`, where the fill left them.
    for r in &glue_regions {
        match &r.dev {
            Some(d) => {
                pre_all.extend(d.pre.iter().map(|c| AnyCol::Dev(c, r.log_rows)));
                base_all.extend(d.base.iter().map(|c| AnyCol::Dev(c, r.log_rows)));
            }
            None => {
                pre_all.extend(r.pre.iter().map(AnyCol::Host));
                base_all.extend(r.base.iter().map(AnyCol::Host));
            }
        }
    }
    if m4e_tabdev {
        pre_all.extend(tab_pre.iter().zip(&pre_logs).map(|(d, l)| AnyCol::Dev(d, *l)));
        base_all.extend(tab_mult.iter().zip(&mult_logs).map(|(d, l)| AnyCol::Dev(d, *l)));
    } else {
        pre_all.extend(pre_tables.iter().map(|e| AnyCol::Host(crate::devtab::hone(e))));
        base_all.extend(base_mults.iter().map(|e| AnyCol::Host(crate::devtab::hone(e))));
    }
    // And the glue's shared tables close both trees, as before.
    pre_all.extend(glue_tcols.iter().map(AnyCol::Host));
    base_all.extend(glue_mcols.iter().map(AnyCol::Host));
    // stream E: the argmax region closes both trees, matching `pre_ids` and the
    // component order -- appending it moves no family before it
    if let Some(r) = am_region.as_ref() {
        pre_all.extend(r.pre.iter().map(AnyCol::Host));
        base_all.extend(r.base.iter().map(AnyCol::Host));
    }
    // stream BB: the embedding gadget closes both trees after the argmax, in
    // the order `pre_ids` writes its ids and the order `components()` places
    // its two components -- the use side, then the table side
    if let Some(r) = emb_region.as_ref() {
        pre_all.extend(r.pre.iter().map(AnyCol::Host));
        pre_all.extend(r.tab_pre.iter().map(AnyCol::Host));
        base_all.extend(r.base.iter().map(AnyCol::Host));
        base_all.push(AnyCol::Host(&r.tab_mult));
    }
    // B11 stream S, lever 3: the link heights' boundary selectors close the
    // preprocessed tree, matching `pre_ids` -- GKR arm only
    if p.gkr_bands {
        for h in crate::gkrlink::link_heights(&plan) {
            pre_all.extend(crate::gkrlink::boundary_cols(h)
                .into_iter().map(crate::device::AnyCol::OwnedHost));
        }
    }
    crate::util::m4_add("gen/refs", m4_t.elapsed().as_secs_f64());
    let trace_secs = trace_secs + t0.elapsed().as_secs_f64();

    // ---- one transcript -------------------------------------------------------
    let widest = pre_all.iter().chain(base_all.iter())
        .map(|c| c.log_size()).max().unwrap();
    let m4_t = std::time::Instant::now();
    crate::util::m4_touch_device();
    crate::util::m4_add("wall/cuda_init", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    let tw = twiddles(widest);
    crate::util::m4_add("wall/twiddles", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    let mut sess = Session::new(&tw);
    crate::util::m4_add("wall/session_new", m4_t.elapsed().as_secs_f64());
    // B7a milestone 4 (18.5): the trees take the columns by reference.
    // `B7_M4_NOCLONE=0` commits from a clone of both lists instead, which is
    // the cost this change removes.
    let m4_pre_cells: usize = pre_all.iter().map(|c| c.size()).sum();
    let m4_refs = crate::util::m4_switch("B7_M4_NOCLONE");
    let m4_t = std::time::Instant::now();
    // B7b (README 26): tree 0 from the cache when this layer and shape have
    // committed it before (`B7_M5_PRECACHE=1`); otherwise committed here and,
    // with the cache on, kept -- or, in mode 2, checked against what is kept.
    let pre_sizes: Vec<u32> = pre_all.iter().map(|c| c.log_size()).collect();
    match crate::precache::take(pre_key, &pre_sizes) {
        Some(tree) => {
            sess.commit_cached(tree, pre_sizes);
            crate::util::m4_add("wall/commit_pre_cached", m4_t.elapsed().as_secs_f64());
        }
        None => {
            if m4_refs {
                sess.commit_any(pre_all);
            } else {
                sess.commit(pre_all.iter().map(|c| c.host().clone()).collect());
            }
            crate::precache::fill(pre_key, &sess);
        }
    }
    crate::util::m4_add("wall/commit_pre", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    if m4_refs {
        sess.commit_any(base_all);
    } else {
        sess.commit(base_all.iter().map(|c| c.host().clone()).collect());
    }
    crate::util::m4_add("wall/commit_base", m4_t.elapsed().as_secs_f64());
    let widths = plan.widths.clone();
    let rels = {
        let exp = sess.draw_relation(ExpRel::draw);
        let norm = sess.draw_relation(NormRel::draw);
        let frac = sess.draw_relation(FracRel::draw);
        let mul = sess.draw_relation(MulRel::draw);
        let qnz = sess.draw_relation(PairRel::draw);
        let ranges: Vec<RangeRel> =
            widths.iter().map(|_| sess.draw_relation(RangeRel::draw)).collect();
        let glue = plan.glue.as_ref().map(|g| GlueRels {
            norm: norm.clone(),
            frac: frac.clone(),
            pow: sess.draw_relation(crate::glue::PowRel::draw),
            smul: sess.draw_relation(crate::glue::SMulRel::draw),
            val: sess.draw_relation(crate::glue::ValRel::draw),
            un: g.uns.iter().map(|_| sess.draw_relation(crate::glue::UnRel::draw)).collect(),
            uns: g.uns.clone(),
        });
        // stream BB: last, so that the flag-off draw order is the one every
        // recorded digest was taken under
        let emb = plan.emb.map(|l| {
            sess.draw_relation(|ch| crate::embed::EmbRel::draw(ch, l.arity()))
        });
        Rels { exp, norm, frac, mul, qnz, widths: widths.clone(), ranges, glue, emb }
    };
    let n_ch = plan.n_challenges();
    let ch = sess.draw_felts(n_ch);
    let (r, s, rv) = (ch[0], ch[1], ch[2]);
    let (w_glue, lam_glue) = if plan.has_ties() {
        (ch[3], ch[4])
    } else {
        (SecureField::zero(), SecureField::zero())
    };
    let r_corner = if p.corner { ch[plan.corner_challenge_at()] } else { SecureField::zero() };

    // B7a milestone 4 (section 3 row 1): the evals every family's interaction
    // lowers, built by `build_components` -- the verifier's own route, and the
    // one `genemit` lowers -- with `claims_for`'s placeholders in place of the
    // claimed sums, which the accumulators below have not produced yet. 19.1 is
    // what makes that correct: a claimed sum enters constraints and never a
    // lookup. `Ir::fingerprint` records the number of scalars and not their
    // values, so the shape a component lowers to here is the shape `genemit`
    // compiled from the same design point.
    let m4e_t = std::time::Instant::now();
    let m4e_air = build_components(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue,
                                   &crate::exprprobe::claims_for(&plan, 0x0b65_4b05));
    crate::util::m4_add("int/air", m4e_t.elapsed().as_secs_f64());

    // B11 stream V, lever 2: whether each family's looks can be linked at all,
    // decided from the same fraction IR the generated kernel runs.
    //
    // The model in `band_split.py --families` counted columns; the arm needs the
    // *looks*, and memo section 4's collapse is a property of the fraction. So
    // this lowers every family's evaluator -- the un-ablated one, the shape the
    // feed asks for -- and reports, per region, whether every denominator is
    // affine in committed base masks and preprocessed columns at this row, and
    // which numerators are the literal one. It is the family's own `look_units`
    // list, derived rather than written down, and a family that fails here gets
    // a named stop in the record and no arm.
    //
    // `B11_GKR_AFFINE=1` only. It lowers a dozen evaluators and proves nothing,
    // so it never runs under a wall number.
    // B11 stream X: the table components' names and heights in the order
    // `m4e_air.tables` holds them -- norm, frac, mul, qnz, then one per range
    // width. The glue's tables are not here and cannot be: they have no
    // generated kernel and so no lowered evaluator to decide, which is a gap
    // the record names rather than one the checker hides.
    let table_heights: Vec<(String, u32)> = [
        ("table/norm".to_string(), plan.log_norm),
        ("table/frac".to_string(), plan.log_frac),
        ("table/mul".to_string(), plan.log_mul),
        ("table/qnz".to_string(), p.ao_q_bits),
    ]
    .into_iter()
    .chain(plan.widths.iter().map(|w| (format!("table/range{w}"), *w)))
    .collect();
    let affinity: Vec<crate::gkrlink::Affinity> = if crate::util::m4_switch_off("B11_GKR_AFFINE") {
        vec![]
    } else {
        let m4a_t = std::time::Instant::now();
        let mut out = vec![];
        for (m, reg) in layer.mats.iter().zip(&mat_regions) {
            let ev = crate::layer::deq_logup_eval(m, reg, rv, s, &rels);
            let ir = crate::devfrac::ir_of(&ev, reg.lay.log_rows);
            out.push(crate::gkrlink::affinity_of(
                "dequant", &format!("dequant/{}", m.name), &ir));
        }
        for (i, reg) in glue_regions.iter().enumerate() {
            let ir = crate::devfrac::ir_of(&m4e_air.glue[i].eval, reg.log_rows);
            out.push(crate::gkrlink::affinity_of(
                "glue", &format!("glue/{}", reg.tag), &ir));
        }
        for (wi, reg) in walk_regions.iter().enumerate() {
            let ir = crate::devfrac::ir_of(&m4e_air.walks[wi].eval, reg.lay.g.log_rows);
            out.push(crate::gkrlink::affinity_of(
                "walks", &format!("walk/{}", reg.name), &ir));
        }
        let ir = crate::devfrac::ir_of(&m4e_air.ao.eval, aor.lay.log_rows);
        out.push(crate::gkrlink::affinity_of("attnout", "attnout", &ir));
        // B11 stream X, lever 2 step 2: the shared tables' provide sides, which
        // are 47% of what stream V left in the interaction tree. Their fraction
        // is `-multiplicity / (z - combine(row))` -- one look a row, a
        // non-unit numerator the way dequant's `mul` lookups have one -- and
        // the question the checker answers is whether the denominator is affine
        // in committed base masks and preprocessed columns at this row and no
        // other. A look the fold cannot carry is a named stop and a result.
        let ir = crate::devfrac::ir_of(&m4e_air.exp_t.eval, p.log_table);
        out.push(crate::gkrlink::affinity_of("table", "table/exp", &ir));
        for (tr, ev) in table_heights.iter().zip(&m4e_air.tables) {
            let ir = crate::devfrac::ir_of(&ev.eval, tr.1);
            out.push(crate::gkrlink::affinity_of("table", &tr.0, &ir));
        }
        crate::util::m4_add("int/affine", m4a_t.elapsed().as_secs_f64());
        out
    };

    // ---- interaction ----------------------------------------------------------
    let t1 = std::time::Instant::now();
    let mut inter_all: Vec<crate::device::AnyCol<'static>> = vec![];
    // B11 stream F step 3 (c-rider 5.1b): `inter_all` is built region by region
    // in discrete `.extend(...)` calls, and the question "could a scoped GKR
    // cover the bands alone" is a per-region cell count. `mark` records what
    // each region added; purely additive, ~30 O(n) sums against a 3.5k-column
    // vector.
    let mut inter_split: Vec<serde_json::Value> = vec![];
    let mut inter_seen = (0usize, 0usize);
    let mut claims = Claims::default();
    // B11 stream Q: on the GKR arm the same lookup lists the committed columns
    // are built from are handed to the collector at each of the three band
    // sites. Collecting is additive -- it reads the committed columns and
    // writes nothing the committed path can see -- so the flag-off pass is the
    // pass it always was.
    let mut gkr: Option<Box<crate::gkrband::Collect>> = p.gkr_bands.then(|| {
        let mut g = crate::gkrband::Collect::with_direct(false);
        // B11 stream S, lever 1: on the composed arm the collector owns the
        // sums -- the committed running-sum columns leave with it, and the
        // GKR's per-site sums stand in for them in the cancellation.
        g.replace = true;
        Box::new(g)
    });
    let m4_t = std::time::Instant::now();
    let mut row_i = 0usize;
    for (b, a) in layer.bands.iter().enumerate() {
        let l = &plan.lays[b];
        let pow = plan.powers(b, r, s, rv, r_corner);
        // `build_components` gives a corner band no row component, and a corner
        // band's fractions never take the device route -- so it gets no BandAir,
        // and the row index counts prefix bands only.
        let air = plan.rows_band[b].then(|| {
            let ri = row_i;
            row_i += 1;
            crate::devfrac::BandAir {
                tag: b,
                chan: &m4e_air.chans[b].eval,
                key: &m4e_air.keys[b].eval,
                row: &m4e_air.rows[ri].eval,
                dev: band_dev[b].as_ref(),
                pre: band_pre[b].as_ref(),
            }
        });
        // B11 stream L, anatomy: the LogUp build apart from `inter_mark`'s
        // accounting, which is the only other thing in this loop. Whatever
        // `int/bands` minus `int/bands_build` comes to is the accounting.
        let t_ib = std::time::Instant::now();
        if let Some(g) = gkr.as_mut() {
            g.reg = crate::gkrlink::Reg::Band(b);
        }
        let it = attn_trace::interaction_phase(a, l, &plan.cfg, &pow, &rels.exp, &rels.ranges,
                                               &band_committed[b], &plan.widths, false,
                                               plan.rows_band[b], air.as_ref(),
                                               gkr.as_deref_mut());
        crate::util::m4_add("int/bands_build", t_ib.elapsed().as_secs_f64());
        inter_all.extend(it.cols);
        inter_mark(&format!("band{b}{}", if a.corner { " (corner)" } else { "" }),
                   &inter_all, &mut inter_seen, &mut inter_split);
        claims.bands.push([qm(it.claimed_chan), qm(it.claimed_hs), qm(it.claimed_av),
                           qm(it.claimed_sn), qm(it.claimed_am), qm(it.claimed_az)]);
        claims.use_sums.extend(it.use_sums.iter().map(|v| qm(*v)));
    }
    crate::util::m4_add("int/bands", m4_t.elapsed().as_secs_f64());
    // B11 stream U, lever 3: the dequant family's sites, collected here rather
    // than where its interaction columns are built.
    //
    // The GKR proof lands at a fixed transcript position -- after tree 1 and the
    // relation challenges, before the interaction tree's own commit -- and the
    // dequant regions' interaction columns come *after* that position in the
    // tree's column order. Their fractions do not: a region's fractions are a
    // function of its committed base and preprocessed columns alone, and those
    // have existed since tree 1. So the collection runs here, in the plan's
    // matmul order, which is `sites_of`'s order and the order the interaction
    // loop below reaches them in; `mat_interaction` then simply stops building
    // the LogUp columns. Nothing in the tree moves except the columns that
    // leave, which is exactly the shape stream S gave the bands -- and it is
    // why the link's own columns stay where S put them, between the bands and
    // `attnout`, in both flag configurations.
    if p.gkr_dequant {
        let g = gkr.as_mut().expect("--gkr-dequant needs --gkr-bands");
        let m4_t = std::time::Instant::now();
        for (mi, (m, reg)) in layer.mats.iter().zip(&mat_regions).enumerate() {
            let eval = crate::layer::deq_logup_eval(m, reg, rv, s, &rels);
            crate::gkrband::deq_site(g, mi, &eval, reg);
        }
        crate::util::m4_add("int/deq_gkr", m4_t.elapsed().as_secs_f64());
    }
    // B11 stream V, lever 3: the glue family's sites, collected here for the
    // same reason -- the GKR proof lands before the interaction tree's own
    // commit and the glue regions' interaction columns come after that position
    // in the tree's column order, while their *fractions* are a function of
    // committed base columns that have existed since tree 1. The order is the
    // plan's region order, which is `sites_of`'s and the order the interaction
    // loop below reaches them in.
    //
    // The evaluator is the component's own with the ablation undone. Stream U
    // rebuilt dequant's from the plan; cloning the built one and clearing `gkr`
    // is the same lowering by construction rather than by inspection, which is
    // the property `B7_M4_LOGUPIR=2` exists to check for the committed half.
    if p.gkr_glue {
        let g = gkr.as_mut().expect("--gkr-glue needs --gkr-bands");
        let m4_t = std::time::Instant::now();
        for (i, reg) in glue_regions.iter().enumerate() {
            let mut ev = m4e_air.glue[i].eval.clone();
            ev.gkr = false;
            crate::gkrband::glue_site(g, i, &ev, reg);
        }
        crate::util::m4_add("int/glue_gkr", m4_t.elapsed().as_secs_f64());
    }
    // B11 stream V, lever 4: the walks, then attnout -- memo section 8's order,
    // which is `sites_of`'s and the order the interaction loop reaches them in.
    // Both evaluators are the components' own with the ablation undone, as the
    // glue family's is.
    if p.gkr_walks {
        let g = gkr.as_mut().expect("--gkr-walks needs --gkr-bands");
        let m4_t = std::time::Instant::now();
        for (wi, reg) in walk_regions.iter().enumerate() {
            let mut ev = m4e_air.walks[wi].eval.clone();
            ev.gkr = false;
            crate::gkrband::walk_site(g, wi, &ev, reg);
        }
        crate::util::m4_add("int/walk_gkr", m4_t.elapsed().as_secs_f64());
    }
    if p.gkr_attnout {
        let g = gkr.as_mut().expect("--gkr-attnout needs --gkr-bands");
        let m4_t = std::time::Instant::now();
        let mut ev = m4e_air.ao.eval.clone();
        ev.gkr = false;
        crate::gkrband::ao_site(g, &ev, &aor);
        crate::util::m4_add("int/ao_gkr", m4_t.elapsed().as_secs_f64());
    }
    // B11 stream Q: the bands' fractional sums, proved by one batched GKR at the
    // transcript position the verifier will replay -- after tree 1 and the
    // relation challenges, before the interaction tree's own commit. The sums it
    // yields are the same quantity `use_sums` carries, so what changes is the
    // argument and not the statement.
    let gkr_proved = gkr.map(|g| {
        // The verifier derives its instance list from the plan alone. If that
        // derivation ever drifts from what the interaction pass actually
        // collected, it has to fail here -- loudly, on the proving side --
        // rather than as an unexplained rejection later.
        let want = crate::gkrband::sites_of(&plan);
        crate::gkrband::same_sites(&g.sites, &want)
            .expect("`gkrband::sites_of` disagrees with the bands' own lookup sites");
        let t = std::time::Instant::now();
        let mut pr = crate::gkrband::prove_collected(&mut sess.channel, *g);
        crate::util::m4_add("int/gkr", t.elapsed().as_secs_f64());
        pr.cost.total_secs = t.elapsed().as_secs_f64();
        pr
    });
    if let Some(pr) = gkr_proved.as_ref() {
        let mut b = serde_json::to_vec(&pr.proof).expect("the GKR proof serialises");
        // B11 stream Q, the arm's own control: `B11_GKRMUT=<index>` flips one
        // digit of the serialised GKR bundle. Everything else about the chunk
        // stays honest, so the only thing that can catch it is the sumcheck
        // itself -- which is the point. Off, the bundle is untouched.
        if let Ok(spec) = std::env::var("B11_GKRMUT") {
            let at: usize = spec.parse().unwrap_or(0);
            let digits: Vec<usize> = (0..b.len())
                .filter(|i| b[*i].is_ascii_digit() && b[*i] != b'0')
                .collect();
            assert!(!digits.is_empty(), "no digit to corrupt in the GKR bundle");
            let i = digits[at % digits.len()];
            eprintln!("gkrmut: byte {i} of {} ({} -> {})", b.len(),
                      b[i] as char, (b[i] - 1) as char);
            b[i] -= 1;
        }
        claims.gkr = Some(b);
    }
    // B11 stream S, lever 3: one univariate IOP per height ties the GKR's
    // input-layer claims to the committed base columns. The fold challenge is
    // drawn right after the sumcheck -- the verifier draws at the same spot --
    // and the columns join the interaction tree right after the bands, where
    // the link components allocate their spans.
    let mut link_hl: Option<Vec<crate::gkrlink::HeightLink>> = None;
    if let Some(pr) = gkr_proved.as_ref() {
        let m4_t = std::time::Instant::now();
        let alpha = sess.draw_felts(1)[0];
        let mut hl = crate::gkrlink::link_params(&plan, &rels, &pr.insts, &pr.artifact, alpha)
            .expect("the prover's own GKR artifact fails the link derivation");
        let msk0 = crate::attn_air::CHAN_SEL.len() + crate::attn_air::CORNER_SEL.len();
        // B11 stream T: a band whose fill left its columns on the device has no
        // host mirror to fold, so the fold reaches them where they are. The
        // causal-mask columns are preprocessed and always host-built, and a
        // corner band is always host-filled, so one height can mix the two.
        let src_of = |reg: crate::gkrlink::Reg, c: &crate::gkrlink::RCol| {
            use crate::gkrlink::{RCol, Reg, Src};
            // B11 stream U, lever 3: the same resolution for the second family.
            // A dequant region's own `base` and `pre` are the columns its masks
            // name, in committed order, and under a device fill they live where
            // the fill left them -- so this is `band_committed`'s two lines with
            // the region's own vectors in place of the band's spans.
            // B11 stream V, lever 3: the same resolution for the glue family.
            // A glue region's own `base` and `pre` are the columns its masks
            // name, in committed order, and under a device fill they live where
            // the fill left them.
            if let Reg::Glue(g) = reg {
                let r = &glue_regions[g];
                return match (c, &r.dev) {
                    (RCol::Glue(i), Some(d)) => Src::Dev(d.base[*i].device_ptr),
                    (RCol::Glue(i), None) => Src::Host(&r.base[*i]),
                    _ => panic!("a non-glue column named on a glue region"),
                };
            }
            if let Reg::Walk(w) = reg {
                let r = &walk_regions[w];
                return match (c, &r.dev) {
                    (RCol::Walk(i), Some(d)) => Src::Dev(d.base[*i].device_ptr),
                    (RCol::Walk(i), None) => Src::Host(&r.base[*i]),
                    _ => panic!("a non-walk column named on a walk region"),
                };
            }
            if let Reg::Ao = reg {
                return match (c, &aor.dev) {
                    (RCol::Ao(i), Some(d)) => Src::Dev(d.base[*i].device_ptr),
                    (RCol::Ao(i), None) => Src::Host(&aor.base[*i]),
                    _ => panic!("a non-attnout column named on the attnout region"),
                };
            }
            if let Reg::Deq(m) = reg {
                let r = &mat_regions[m];
                let chan_w = if r.spec.variant == DeqVariant::Chain { 2 } else { 1 };
                return match (c, &r.dev) {
                    (RCol::Deq(i), Some(d)) => Src::Dev(d.base[*i].device_ptr),
                    (RCol::Deq(i), None) => Src::Host(&r.base[*i]),
                    (RCol::DeqPre(sl), Some(d)) => Src::Dev(d.pre[chan_w * sl].device_ptr),
                    (RCol::DeqPre(sl), None) => Src::Host(&r.pre[chan_w * sl]),
                    _ => panic!("a band column named on a dequant region"),
                };
            }
            let Reg::Band(band) = reg else { unreachable!() };
            let com = &band_committed[band];
            let dev = band_dev[band].as_ref();
            let base = |i: usize| match dev {
                Some(d) => Src::Dev(d.base[i].device_ptr),
                None => Src::Host(&com.base[i]),
            };
            match c {
                RCol::Chan(i) => base(com.at[0] + i),
                RCol::Key(i) => base(com.at[1] + i),
                RCol::Row(i) => base(com.at[3] + i),
                RCol::Msk(i) => Src::Host(&com.pre[msk0 + i]),
                _ => panic!("a dequant column named on a band"),
            }
        };
        // B11 stream S, the link's own control: `B11_LINKMUT=<k>` bends one
        // folded weight and re-derives the claim from the bent column, so the
        // prover's every self-check passes and the proof is *self-consistent*
        // -- the only thing that can reject it is the verifier deriving the
        // honest fold from the GKR artifact. Off, nothing is touched.
        if let Ok(spec) = std::env::var("B11_LINKMUT") {
            // B11 stream U, lever 3: `<k>` bends height `k`'s first folded
            // weight, as stream S wrote it; `<k>:deq` bends the first weight on
            // that height that belongs to a *dequant* region instead. The
            // weights are keyed by `(region, column)` and `Reg::Band` sorts
            // before `Reg::Deq`, so at a height both families share the plain
            // form always lands on a band -- and a control that cannot be aimed
            // at the new family would not be a control for it.
            let is_deq = |w: &(crate::gkrlink::Reg, crate::gkrlink::RCol, SecureField)| {
                matches!(w.0, crate::gkrlink::Reg::Deq(_))
            };
            // B11 stream V, lever 3: and the same self-aim for the glue family.
            // `Reg::Band` sorts before `Reg::Deq` before `Reg::Glue`, so at a
            // height that several families share the plain form always lands on
            // a band and a control that cannot be aimed at the new family would
            // not be a control for it.
            let is_glue = |w: &(crate::gkrlink::Reg, crate::gkrlink::RCol, SecureField)| {
                matches!(w.0, crate::gkrlink::Reg::Glue(_))
            };
            let is_walk = |w: &(crate::gkrlink::Reg, crate::gkrlink::RCol, SecureField)| {
                matches!(w.0, crate::gkrlink::Reg::Walk(_))
            };
            let is_ao = |w: &(crate::gkrlink::Reg, crate::gkrlink::RCol, SecureField)| {
                matches!(w.0, crate::gkrlink::Reg::Ao)
            };
            let (at, w) = if spec == "walks" {
                let at = hl.iter().position(|h| h.weights.iter().any(is_walk))
                    .expect("linkmut: the fold carries no walk column to bend");
                (at, hl[at].weights.iter().position(is_walk).unwrap())
            } else if spec == "attnout" {
                let at = hl.iter().position(|h| h.weights.iter().any(is_ao))
                    .expect("linkmut: the fold carries no attnout column to bend");
                (at, hl[at].weights.iter().position(is_ao).unwrap())
            } else if spec == "glue" {
                let at = hl.iter().position(|h| h.weights.iter().any(is_glue))
                    .expect("linkmut: the fold carries no glue column to bend");
                (at, hl[at].weights.iter().position(is_glue).unwrap())
            } else if spec == "deq" {
                // self-aiming: the first dequant weight anywhere in the fold. A
                // height's weights are keyed by `(region, column)` and
                // `Reg::Band` sorts first, so this finds the family without the
                // caller having to know which heights it landed on.
                let at = hl.iter().position(|h| h.weights.iter().any(is_deq))
                    .expect("linkmut: the fold carries no dequant column to bend");
                (at, hl[at].weights.iter().position(is_deq).unwrap())
            } else {
                let (num, fam) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
                let at: usize = num.parse::<usize>().unwrap_or(0) % hl.len();
                let w = if fam == "deq" {
                    hl[at].weights.iter().position(is_deq).unwrap_or_else(|| panic!(
                        "linkmut: height 2^{} carries no dequant column to bend",
                        hl[at].h))
                } else if fam == "glue" {
                    hl[at].weights.iter().position(is_glue).unwrap_or_else(|| panic!(
                        "linkmut: height 2^{} carries no glue column to bend",
                        hl[at].h))
                } else {
                    0
                };
                (at, w)
            };
            eprintln!("linkmut: height 2^{}, folded weight {w} bent ({:?} {:?})",
                      hl[at].h, hl[at].weights[w].0, hl[at].weights[w].1);
            hl[at].weights[w].2 += SecureField::one();
            hl[at].claim = crate::gkrlink::fold_total(&hl[at], &src_of);
            hl[at].shift = hl[at].claim
                / SecureField::from(M31::from_u32_unchecked(1u32 << hl[at].h));
        }
        // B11 stream U, lever 1: the fold and the five column builds are one
        // call now, because on the cuda arm they are one chain that stays on
        // the card -- the coefficient column is no longer downloaded for the
        // host loops to read. `B11_LINKDEV=0` is the host control.
        for l in &hl {
            inter_all.extend(crate::gkrlink::height_cols(l, &src_of));
        }
        inter_mark("gkrlink", &inter_all, &mut inter_seen, &mut inter_split);
        crate::util::m4_add("int/link", m4_t.elapsed().as_secs_f64());
        link_hl = Some(hl);
    }
    let m4_t = std::time::Instant::now();
    let aoi = ao_interaction(&layer.ao, &aor, r, s, rv, plan.head_weight_c(), &rels,
                            &m4e_air.ao.eval,
                            p.gkr_attnout);
    inter_all.extend(aoi.cols);
    inter_mark("attnout", &inter_all, &mut inter_seen, &mut inter_split);
    claims.ao_out = qm(aoi.claimed_out);
    claims.ao_den = qm(aoi.claimed_den);
    if !p.gkr_attnout {
        claims.use_sums.push(qm(aoi.use_sum));
    }
    crate::util::m4_add("int/attnout", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    for (m, reg) in layer.mats.iter().zip(&mat_regions) {
        // stream E: the output side rides on the wiring family's tie challenge,
        // so it exists exactly when that family does and the plan says this
        // matmul publishes one
        let ow = plan.has_oside(&m.name).then(|| {
            oside_w(w_glue, lam_glue, m.tokens, m.n_out, reg.lay.per_row,
                    plan.oside_kind(&m.name))
        });
        let it = mat_interaction(m, reg, rv, s, &rels, ow.as_ref(), p.gkr_dequant);
        inter_all.extend(it.cols);
        inter_mark(&format!("dequant/{}", m.name), &inter_all, &mut inter_seen,
                   &mut inter_split);
        claims.mat_h.push(qm(it.claimed_h));
        // B11 stream U, lever 3: with the collector owning this region's
        // lookups there is no claimed sum to carry -- the GKR's per-site sum
        // stands exactly where it stood, in `sums_cancel`.
        if !p.gkr_dequant {
            claims.use_sums.push(qm(it.use_sum));
        }
        if let Some(c) = it.claimed_out {
            if m.name == "lm_head" {
                claims.head_key.push(qm(c));
            } else {
                claims.deq_out.push(qm(c));
            }
        }
    }
    // Stream E: the head's key claim against B1's own `key_head` column.  The
    // proof derives its keys from the head dequant's committed output; this
    // rebuilds the same Horner sum from the dump's column and compares the two
    // field elements, which is 512 x vocab values checked at once.
    let mut head_key_b1 = json!(null);
    if plan.p.head_key {
        if let (Some(mi), Some(keys)) =
            (layer.mats.iter().position(|m| m.name == "lm_head"), layer.head_keys.as_ref())
        {
            let (m, lay) = (&layer.mats[mi], plan.mat_lays[mi]);
            let ow = oside_w(w_glue, lam_glue, m.tokens, m.n_out, lay.per_row,
                             crate::dequant::OKind::Key);
            let rows = 1usize << lay.log_rows;
            let mut u = SecureField::zero();
            for j in 0..rows {
                let mut f = SecureField::zero();
                if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                    for sl in 0..lay.per_row {
                        let c = b * lay.per_row + sl;
                        f += SecureField::from(m31_u32(u32::from(keys[t * m.n_out + c]))) * ow.w[sl];
                    }
                }
                let mult = if j == 0 {
                    SecureField::zero()
                } else {
                    match lay.at_opt(m.tokens, j) {
                        Some((_, b)) if b + 1 == lay.chunks => ow.k_bd,
                        Some(_) => ow.k_in,
                        None => SecureField::zero(),
                    }
                };
                u = u * mult + f;
            }
            let got = claims.head_key.first().copied().map(unqm);
            head_key_b1 = json!({
                "values": m.tokens * m.n_out,
                "matches_b1": got == Some(u),
            });
        }
    }
    crate::util::m4_add("int/dequant", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    for (wi, reg) in walk_regions.iter().enumerate() {
        let t_iw = std::time::Instant::now();
        let it = walk_interaction(layer, reg, rv, s, &rels, true,
                                  &m4e_air.walks[wi].eval, p.gkr_walks);
        crate::util::m4_add("int/walks_build", t_iw.elapsed().as_secs_f64());
        inter_all.extend(it.cols);
        inter_mark(&format!("walk/{}", reg.name), &inter_all, &mut inter_seen,
                   &mut inter_split);
        claims.walk_acc.push(it.claimed.iter().map(|v| qm(*v)).collect());
        if !p.gkr_walks {
            claims.use_sums.push(qm(it.use_sum));
        }
    }
    crate::util::m4_add("int/walks", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    // the glue regions' ties and lookups, in component order
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let blank = Claims::default();
        let specs = gp.ties(w_glue, lam_glue, &blank);
        for (i, reg) in glue_regions.iter().enumerate() {
            let t_ig = std::time::Instant::now();
            let it = crate::glue::glue_interaction(reg, gr, Some(&specs[i]),
                                                   &m4e_air.glue[i].eval, p.gkr_glue);
            crate::util::m4_add("int/glue_build", t_ig.elapsed().as_secs_f64());
            inter_all.extend(it.cols);
            inter_mark(&format!("glue/{}", reg.tag), &inter_all, &mut inter_seen,
                       &mut inter_split);
            claims.glue_ties.push(it.claimed.iter().map(|v| qm(*v)).collect());
            // B11 stream V, lever 3: with the collector owning this region's
            // lookups there is no claimed sum to carry -- the GKR's per-site
            // sum stands exactly where it stood, in `sums_cancel`.
            if !p.gkr_glue {
                claims.use_sums.push(qm(it.use_sum));
            }
        }
    }
    crate::util::m4_add("int/glue", m4_t.elapsed().as_secs_f64());

    // B11 stream X, lever 2: one entry per table region, in the order the
    // interaction pass emits them. `interaction_split` already records what
    // each region *cost*; this records what it *is* -- the height, which
    // decides both what an arm could remove and whether the link already pays
    // for that height, and how many committed key columns the relation
    // combines, which is the read set a link would have to name. It is the
    // block `band_split.py`'s family model needs and the only one the tables
    // did not have, and it is metadata: nothing reads it back.
    let mut table_regions: Vec<Value> = vec![];
    let m4_t = std::time::Instant::now();
    // the tables' own sides, in table order
    {
        // The two table families go the same route as the other six: their
        // eval, lowered, run where their columns are. `pre_tables` and
        // `base_mults` are the owned host columns and `tab_pre` / `tab_mult`
        // their device copies when `B7_M4_TABDEV` uploaded them; the closure is
        // the hand-written host builder the device result is checked against.
        use crate::devtab::{htab, tab_cols as tc};
        let (c, sum) = crate::devfrac::frac_cols(
            &m4e_air.exp_t.eval, "exp_table", p.log_table,
            tc(&pre_tables, &tab_pre, 0, 2), tc(&base_mults, &tab_mult, 0, 1),
            || table_col(p.log_table, &rels.exp,
                         &[htab(&pre_tables, 0), htab(&pre_tables, 1)],
                         htab(&base_mults, 0)));
        inter_all.extend(c);
        inter_mark("table/exp", &inter_all, &mut inter_seen, &mut inter_split);
        table_regions.push(json!({"region": "table/exp", "log_rows": p.log_table,
                                  "key_cols": 2, "looks": 1}));
        claims.table_sums.push(qm(sum));
        let (c, sum) = crate::devfrac::frac_cols(
            &m4e_air.tables[0].eval, "table norm", plan.log_norm,
            tc(&pre_tables, &tab_pre, 2, 6), tc(&base_mults, &tab_mult, 1, 1),
            || {
                let refs: Vec<&Eval> = (0..6).map(|i| htab(&pre_tables, 2 + i)).collect();
                table_col(plan.log_norm, &rels.norm, &refs, htab(&base_mults, 1))
            });
        inter_all.extend(c);
        inter_mark("table/norm", &inter_all, &mut inter_seen, &mut inter_split);
        table_regions.push(json!({"region": "table/norm", "log_rows": plan.log_norm,
                                  "key_cols": 6, "looks": 1}));
        claims.table_sums.push(qm(sum));
        let (c, sum) = crate::devfrac::frac_cols(
            &m4e_air.tables[1].eval, "table frac", plan.log_frac,
            tc(&pre_tables, &tab_pre, 8, 3), tc(&base_mults, &tab_mult, 2, 1),
            || {
                let refs: Vec<&Eval> = (0..3).map(|i| htab(&pre_tables, 8 + i)).collect();
                table_col(plan.log_frac, &rels.frac, &refs, htab(&base_mults, 2))
            });
        inter_all.extend(c);
        inter_mark("table/frac", &inter_all, &mut inter_seen, &mut inter_split);
        table_regions.push(json!({"region": "table/frac", "log_rows": plan.log_frac,
                                  "key_cols": 3, "looks": 1}));
        claims.table_sums.push(qm(sum));
        let (c, sum) = crate::devfrac::frac_cols(
            &m4e_air.tables[2].eval, "table mul", plan.log_mul,
            tc(&pre_tables, &tab_pre, 11, 4), tc(&base_mults, &tab_mult, 3, 1),
            || {
                let refs: Vec<&Eval> = (0..4).map(|i| htab(&pre_tables, 11 + i)).collect();
                table_col(plan.log_mul, &rels.mul, &refs, htab(&base_mults, 3))
            });
        inter_all.extend(c);
        inter_mark("table/mul", &inter_all, &mut inter_seen, &mut inter_split);
        table_regions.push(json!({"region": "table/mul", "log_rows": plan.log_mul,
                                  "key_cols": 4, "looks": 1}));
        claims.table_sums.push(qm(sum));
        let (c, sum) = crate::devfrac::frac_cols(
            &m4e_air.tables[3].eval, "table qnz", p.ao_q_bits,
            tc(&pre_tables, &tab_pre, 15, 2), tc(&base_mults, &tab_mult, 4, 1),
            || {
                let refs: Vec<&Eval> = (0..2).map(|i| htab(&pre_tables, 15 + i)).collect();
                table_col(p.ao_q_bits, &rels.qnz, &refs, htab(&base_mults, 4))
            });
        inter_all.extend(c);
        inter_mark("table/qnz", &inter_all, &mut inter_seen, &mut inter_split);
        table_regions.push(json!({"region": "table/qnz", "log_rows": p.ao_q_bits,
                                  "key_cols": 2, "looks": 1}));
        claims.table_sums.push(qm(sum));
        for (t, w) in plan.widths.iter().enumerate() {
            let (c, sum) = crate::devfrac::frac_cols(
                &m4e_air.tables[4 + t].eval, &format!("table range{w}"), *w,
                tc(&pre_tables, &tab_pre, 17 + t, 1),
                tc(&base_mults, &tab_mult, 5 + t, 1),
                || table_col(*w, &rels.ranges[t], &[htab(&pre_tables, 17 + t)],
                             htab(&base_mults, 5 + t)));
            inter_all.extend(c);
            inter_mark(&format!("table/range{w}"), &inter_all, &mut inter_seen,
                       &mut inter_split);
            table_regions.push(json!({"region": format!("table/range{w}"),
                                      "log_rows": *w, "key_cols": 1, "looks": 1}));
            claims.table_sums.push(qm(sum));
        }
        // The glue's tables have no device path (and no generated kernels), so
        // their sides keep the hand-written builder, reading the owned glue
        // vectors directly.
        if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
            let gcols = crate::glue::glue_table_ids(&gp.uns);
            for (t, (lr, cols, kind)) in
                crate::glue::glue_table_kinds(&gp.uns, gr).into_iter().enumerate()
            {
                let refs: Vec<&Eval> = cols
                    .iter()
                    .map(|c| {
                        let k = gcols.iter().position(|x| x == c).expect("glue table column");
                        &glue_tcols[k]
                    })
                    .collect();
                let (c, sum) = crate::glue::table_side(lr, &kind, &refs, &glue_mcols[t]);
                inter_all.extend(crate::device::host_cols(c));
                inter_mark(&format!("table/glue{t}"), &inter_all, &mut inter_seen,
                           &mut inter_split);
                table_regions.push(json!({"region": format!("table/glue{t}"),
                                          "log_rows": lr, "key_cols": refs.len(),
                                          "looks": 1}));
                claims.table_sums.push(qm(sum));
            }
        }
    }
    crate::util::m4_add("int/tables", m4_t.elapsed().as_secs_f64());

    // Stream E: the argmax's interaction columns come *after* the tables' own
    // sides, because the framework hands out interaction spans in component
    // order and the argmax component is last of all.  Appended earlier, this
    // component is handed the exp table's columns instead and the prover dies
    // evaluating a 2^23 polynomial on a 2^16 domain -- which is exactly how it
    // failed the first time.

    // stream E: the argmax's tie column then its lookups, the order the AIR
    // reads them, and its use sum last of all -- the component is last too
    let m4_t = std::time::Instant::now();
    if let (Some(r), Some(mi)) = (am_region.as_ref(), plan.argmax_mat()) {
        let m = &plan.p.mats[mi];
        let tie = crate::argmax::ATie {
            claimed: SecureField::zero(),
            tag: m.name.clone(),
            ..argmax_tie_w(w_glue, m.tokens, m.n_out, plan.mat_lays[mi].per_row)
        };
        let (tcols, claimed) = crate::argmax::tie_col(r, m.tokens, &tie);
        inter_all.extend(crate::device::host_cols(tcols));
        claims.am_key.push(qm(claimed));
        let (lcols, use_sum) =
            crate::argmax::logup(r, plan.p.head_argmax_lax,
                                 rels.range(crate::argmax::KEY_BITS));
        inter_all.extend(crate::device::host_cols(lcols));
        claims.use_sums.push(qm(use_sum));
        inter_mark("argmax", &inter_all, &mut inter_seen, &mut inter_split);
    }
    crate::util::m4_add("int/argmax", m4_t.elapsed().as_secs_f64());

    // stream BB: the embedding gadget's interaction columns, last of all,
    // because its two components are last of all.  Within the use region the
    // AIR reads the two Horner accumulators before it finalizes its LogUp, so
    // the tie columns come before the fraction columns here too; the table
    // region's single column follows, matching the component order.
    let m4_t = std::time::Instant::now();
    if let (Some(r), Some(er)) = (emb_region.as_ref(), rels.emb.as_ref()) {
        let mut tie = crate::embed::tie_w(w_glue, lam_glue, r.d, r.tokens, &r.lay);
        let (tcols, hc, gc) = crate::embed::tie_cols(r, &tie);
        tie.claimed = hc;
        tie.claimed_tok = gc;
        inter_all.extend(crate::device::host_cols(tcols));
        claims.embed_out.push(qm(hc));
        claims.embed_tok.push(qm(gc));
        let (lcols, use_sum) =
            crate::embed::logup(r, er, rels.range(crate::embed::EF_BITS));
        inter_all.extend(crate::device::host_cols(lcols));
        claims.use_sums.push(qm(use_sum));
        let (tbcols, tab_sum) = crate::embed::table_side(r, er);
        inter_all.extend(crate::device::host_cols(tbcols));
        claims.table_sums.push(qm(tab_sum));
        inter_mark("embed", &inter_all, &mut inter_seen, &mut inter_split);
    }
    crate::util::m4_add("int/embed", m4_t.elapsed().as_secs_f64());
    // an independent recomputation of every tie claim, from the region's witness
    // rather than from its trace: a tie is only worth something if it is checkable
    let glue_ties_ok = match (&plan.glue, &rels.glue) {
        (Some(gp), Some(_)) => {
            let specs = gp.ties(w_glue, lam_glue, &claims);
            glue_regions.iter().zip(&specs).all(|(reg, tie)| {
                tie.specs.iter().all(|sp| reg.tie_claim(sp, lam_glue) == sp.claimed)
            })
        }
        _ => true,
    };
    let interaction_secs = t1.elapsed().as_secs_f64();
    let n_inter = inter_all.len();
    // B11 stream F step 3: the per-region deltas rolled up by family, so the
    // "could a scoped GKR cover the bands alone" share is one division.
    let inter_by_family = {
        let mut by: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for e in &inter_split {
            let r = e["region"].as_str().unwrap_or("");
            let head = r.split('/').next().unwrap_or(r);
            let fam = if head.starts_with("band") { "bands" } else { head };
            let v = by.entry(fam.to_string()).or_default();
            v.0 += e["columns"].as_u64().unwrap_or(0) as usize;
            v.1 += e["cells"].as_u64().unwrap_or(0) as usize;
        }
        let m: serde_json::Map<String, Value> = by
            .into_iter()
            .map(|(k, (c, n))| (k, json!({"columns": c, "cells": n})))
            .collect();
        Value::Object(m)
    };
    let clamp_edge = aor.q_out.iter().filter(|q| q.unsigned_abs() == 127).count();
    let family_cells = json!({
        "bands": band_committed.iter().zip(&band_dev).map(|(c, d)| match d {
            Some(d) => d.logs.iter().map(|l| 1usize << l).sum::<usize>(),
            None => c.base.iter().map(|e| e.domain.size()).sum::<usize>(),
        }).sum::<usize>(),
        "attnout": aor.base_cells(),
        "dequant": n_deq_base,
        "walks": walk_regions.iter().map(|w| w.base_cells()).sum::<usize>(),
        "glue": glue_regions.iter().map(|g| g.base_cells()).sum::<usize>(),
        "glue_regions": glue_regions.iter().map(|g| json!({
            "tag": g.tag, "rows": g.rows, "log_rows": g.log_rows,
            "leaves": g.n_leaves, "ops": g.n_ops,
            "columns": match &g.dev { Some(d) => d.base.len(), None => g.base.len() },
            "cells": g.base_cells(),
            "reference_mismatch": g.ref_n - g.ref_ok,
            "exponent_out_of_range": g.bad_exp,
        })).collect::<Vec<_>>(),
        "glue_vs_b1": glue_b1,
        "head_key_vs_b1": head_key_b1,
        // stream BB: what the embedding gadget costs this chunk, or null with
        // the flag off.  Reported per position and per block so E1 folds it into
        // the frontier table without re-deriving the layout.
        "embed": emb_region.as_ref().map(|r| {
            let (ur, tr) = (1usize << r.lay.log_use, 1usize << r.lay.log_tab);
            let base = r.base.len() * ur + tr;
            json!({
                "layout": r.lay.json(r.tokens, r.vocab),
                "columns": {"use_pre": r.pre.len(), "use_base": r.base.len(),
                            "table_pre": r.tab_pre.len(), "table_base": 1},
                "cells": {"preprocessed": r.pre_cells(), "base": base},
                "base_per_position": base as f64 / r.tokens as f64,
                "looks_per_use_row": r.n_looks(),
                "off_table": r.off_table,
            })
        }),
        "interaction": inter_all.iter().map(|c| c.size()).sum::<usize>(),
        // B11 stream F step 3: the interaction set region by region, in the
        // order `inter_all` is built, plus the rollup by family. `tables` is the
        // field `family_cells` was missing, and `bands` here is the interaction
        // share a scoped GKR over the bands alone would cover.
        "interaction_split": inter_split,
        "interaction_by_family": inter_by_family,
        // B11 stream X, lever 2: the tables' own sides as a family block, the
        // shape `mats` and `walk_weights` have, and the heights the link pays
        // for at this configuration. The second is what turns the first into a
        // decision: a family joins a height the link already carries for
        // nothing, and pays 16 committed columns over 2^h for a height it
        // brings with it.
        "tables": table_regions,
        "link_heights": crate::gkrlink::link_heights(&plan),
        // B11 stream V, lever 2: the affinity verdict per region, or an empty
        // list when `B11_GKR_AFFINE` did not ask for it.
        "affinity": affinity,
        // B11 stream Q: what the GKR arm proved and what it cost, or null with
        // the flag off. Reported beside `interaction_by_family` because the two
        // are the same question from the two sides.
        "gkr": gkr_proved.as_ref().map(|p| p.cost.json()),
        "preprocessed": m4_pre_cells,
        "walk_weights": walk_regions.iter().map(|w| json!({
            "group": w.name, "weights": w.n_weights, "lead": w.lay.g.lead,
            "pk": w.lay.pk, "pq": w.lay.pq, "log_rows": w.lay.g.log_rows,
            "pre_cells": w.pre_cells(),
        })).collect::<Vec<_>>(),
        "mats": layer.mats.iter().zip(&mat_regions).map(|(m, r)| json!({
            "name": m.name, "variant": m.variant.name(), "n": r.n,
            "per_row": r.lay.per_row, "log_rows": r.lay.log_rows,
            "kmax_needed": m.kmax_needed(), "kmax_blind": m.kmax_blind(),
            "cells": r.base_cells(),
            "lead": (1usize << r.lay.log_rows) - m.tokens * r.lay.chunks,
        })).collect::<Vec<_>>(),
        "band_leads": plan.lays.iter().map(|l| json!({
            "chan": l.chan.g.lead, "key": l.key.g.lead, "out": l.out.g.lead,
        })).collect::<Vec<_>>(),
        "attnout_stats": {"n": aor.n, "per_row": aor.lay.per_row,
                          "log_rows": aor.lay.log_rows, "lead_rows": aor.lay.lead,
                          "q_bits": aor.q_bits, "rem_bits": aor.rem_bits,
                          "clamp_edge_count": clamp_edge},
    });
    let m4_t = std::time::Instant::now();
    sess.commit_any(inter_all);
    crate::util::m4_add("wall/commit_inter", m4_t.elapsed().as_secs_f64());
    // Stream H: the boundary sides, one tree each, committed after the
    // interaction tree and before the claims are mixed.  Nothing is drawn
    // between them and nothing reads them but the equality constraints, which
    // is what the note argues makes this position sound.
    let m4_t = std::time::Instant::now();
    let bounds_report = if plan.bounds.is_empty() {
        json!(null)
    } else {
        for (i, side) in plan.bounds.iter().enumerate() {
            let tam = bounds_tamper.filter(|(s, _)| *s == i).map(|(_, c)| c);
            let cols = crate::bounds::cols(&glue_regions[side.region], side, tam);
            sess.commit(cols);
        }
        crate::bounds::report(&plan.bounds)
    };
    // Stream AA: the K/V sides, in the same position and for the same reason,
    // right after stream H's.  A side's columns are the *very columns tree 1
    // committed*, cloned band by band in component-registration order --
    // `[band's K][band's V]` -- so byte-identity with the region is by
    // construction and not by a second copy of `attn_trace`'s index map.
    let kv_report = if plan.kv.is_empty() {
        json!(null)
    } else {
        for (i, side) in plan.kv.iter().enumerate() {
            let mut cols: Vec<Eval> = vec![];
            for b in &side.bands {
                let com = &band_committed[*b];
                assert!(!com.base.is_empty(),
                        "--kv-bind needs the band fill on the host: the K/V boundary tree is a \
                         copy of the committed columns, and with B7_M4_WFILL_BAND=1 band {b} \
                         has none on this side of the bus");
                let (l, at) = (&plan.lays[*b], &com.at);
                let n_aux = if plan.attn_variant == AttnVariant::IntExp { 0 } else { l.key.q };
                let v0 = at[1] + 2 * l.key.q + n_aux;
                cols.extend(com.base[at[0]..at[0] + l.chan.pk].iter().cloned());
                cols.extend(com.base[v0..v0 + l.key.pv].iter().cloned());
            }
            assert_eq!(cols.len(), side.columns, "a K/V side's width disagrees with its plan");
            // the tamper indexes the sides as one list, stream H's first
            let tam = bounds_tamper
                .filter(|(s, _)| *s == plan.bounds.len() + i)
                .map(|(_, c)| c);
            sess.commit(crate::bounds::kv_cols(cols, tam));
        }
        crate::bounds::kv_report(&plan.kv)
    };
    crate::util::m4_add("wall/commit_bounds", m4_t.elapsed().as_secs_f64());
    sess.mix(&claims.flat());

    // ---- components, prove ----------------------------------------------------
    let m4_t = std::time::Instant::now();
    let built = build_components_linked(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue,
                                        &claims, link_hl.as_deref());
    crate::util::m4_add("wall/build_components", m4_t.elapsed().as_secs_f64());
    let sizes: Vec<Vec<u32>> = sess
        .ops
        .iter()
        .filter_map(|op| match op {
            crate::common::Op::Tree(sz) => Some(sz.clone()),
            _ => None,
        })
        .collect();
    let n_columns = [sizes[0].len(), sizes[1].len(), n_inter];
    // B7a milestone 3: the lowering, off the prove loop's critical path.
    #[cfg(feature = "cuda")]
    if crate::util::m3_switch("B7_M3_PREWARM") {
        built.prewarm_blobs();
    }
    let provers = built.provers();
    let m4_t = std::time::Instant::now();
    let (res, stats) = sess.prove_raw(&provers);
    crate::util::m4_add("wall/prove_raw", m4_t.elapsed().as_secs_f64());
    let (proof, prove_error) = match res {
        Ok(pf) => (Some(pf), None),
        Err(e) => (None, Some(e)),
    };
    let fixed = Fixed {
        params: params.clone(),
        log_blowup: crate::util::LOG_BLOWUP,
        n_queries: crate::util::N_QUERIES,
        pow_bits: crate::util::POW_BITS,
        sizes,
        pre_root: proof
            .as_ref()
            .map(|pf| format!("{:?}", pf.commitments[0]))
            .unwrap_or_default(),
    };
    Composed {
        proof,
        prove_error,
        stats,
        fixed,
        claims,
        trace_secs,
        interaction_secs,
        ref_bad: mat_regions.iter().map(|r| r.ref_bad).sum::<usize>() + aor.ref_bad,
        off_table: {
            if tables.off_table > 0 {
                use std::sync::atomic::Ordering::Relaxed;
                eprintln!("off_table {} by table: {:?}", tables.off_table, tables.off_kinds);
                eprintln!("exp table: {} rows, max key {}, off {} in [{}, {}]",
                          crate::attn_trace::EXP_ROWS.load(Relaxed),
                          crate::attn_trace::EXP_KEY_MAX.load(Relaxed),
                          crate::attn_trace::EXP_OFF_N.load(Relaxed),
                          crate::attn_trace::EXP_OFF_MIN.load(Relaxed),
                          crate::attn_trace::EXP_OFF_MAX.load(Relaxed));
            }
            tables.off_table
        },
        family_cells,
        n_columns,
        glue_ties_ok,
        bounds: bounds_report,
        kv_bounds: kv_report,
    }
}

// ---------------------------------------------------------------- verifying ----

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub stark_ok: bool,
    pub root_ok: bool,
    pub sums_cancel: bool,
    pub qk_ok: Vec<bool>,
    pub den_ok: Vec<bool>,
    pub av_ok: bool,
    pub den_link_ok: bool,
    pub walk_ok: Vec<bool>,
    pub sizes_ok: bool,
    /// B10 step 2, one per equality of the glue plan: `gn1.roots == gn1b.in`,
    /// `gn1.xmid == gn2.in`, `gn2.roots == gn2b.in` -- the residual stream is
    /// threaded, from serialized bytes alone
    pub glue_ok: Vec<bool>,
    /// stream E, one per dequant output tie: the region's own output claim equals
    /// the wiring family's claim over the same values
    pub deq_out_ok: Vec<bool>,
    /// stream E: the argmax's key claim equals the head dequant's -- "the token
    /// this row emits is the maximum of the logits *this layer computed*"
    pub head_tie_ok: bool,
    /// stream BB: the embedding region's claim equals the wiring family's
    /// `stream_in` claim -- "the residual stream this layer consumed is row
    /// `tok[t]` of the pinned embedding matrix".  True when the gadget is off.
    pub embed_ok: bool,
    /// the layer's committed residual-stream input and output claims
    pub stream_in: Option<[u32; 4]>,
    pub stream_out: Option<[u32; 4]>,
    /// Stream H: the two boundary trees' roots, off the parsed proof.  These are
    /// what `bounds::chain_check` compares across two proofs; a verifier reading
    /// one proof alone gets them here so it never has to re-derive anything.
    pub bound_in: Option<String>,
    pub bound_out: Option<String>,
    /// Stream AA: every boundary tree's root, stream H's two first and then the
    /// K/V sides in `plan.kv`'s order.  A verifier reading one proof alone gets
    /// them here so it never has to re-derive anything; `bounds::chain_check`
    /// reads the same roots straight out of `proof.commitments`.
    pub bound_roots: Vec<String>,
    pub error: Option<String>,
    pub verify_secs: f64,
}

impl Verdict {
    pub fn accepts(&self) -> bool {
        self.stark_ok && self.root_ok && self.sums_cancel && self.av_ok && self.den_link_ok
            && self.sizes_ok && self.qk_ok.iter().all(|x| *x) && self.den_ok.iter().all(|x| *x)

            && self.walk_ok.iter().all(|x| *x) && self.glue_ok.iter().all(|x| *x)
            && self.deq_out_ok.iter().all(|x| *x) && self.head_tie_ok && self.embed_ok
    }
    fn rejected(msg: &str) -> Self {
        Self { stark_ok: false, root_ok: false, sums_cancel: false, qk_ok: vec![],
               den_ok: vec![], av_ok: false, den_link_ok: false, walk_ok: vec![],
               sizes_ok: false, glue_ok: vec![false], deq_out_ok: vec![false],
               head_tie_ok: false, embed_ok: false,
               stream_in: None, stream_out: None,
               bound_in: None, bound_out: None, bound_roots: vec![],
               error: Some(msg.to_string()), verify_secs: 0.0 }
    }
}

/// The only accept path: three byte strings in, a verdict out.  Replays the
/// transcript, pins the preprocessed root, recomputes every power and boundary
/// multiplier from its own challenges, and runs stwo's verify -- no prover
/// struct crosses this boundary.
pub fn verify_composed(fixed_bytes: &[u8], claims_bytes: &[u8], proof_bytes: &[u8]) -> Verdict {
    let t0 = std::time::Instant::now();
    let Ok(fixed) = serde_json::from_slice::<Fixed>(fixed_bytes) else {
        return Verdict::rejected("unparseable fixed statement");
    };
    let Ok(claims) = serde_json::from_slice::<Claims>(claims_bytes) else {
        return Verdict::rejected("unparseable claims");
    };
    let Ok(proof) = serde_json::from_slice::<Proof>(proof_bytes) else {
        return Verdict::rejected("unparseable proof");
    };
    let plan = Plan::new(&fixed.params);
    let p = &plan.p;
    // the claims' shape must match the plan before anything indexes into them
    let n_glue = plan.glue.as_ref().map(|g| g.regions.len()).unwrap_or(0);
    if claims.bands.len() != plan.shapes.len() || claims.mat_h.len() != p.mats.len()
        || claims.walk_acc.len() != plan.walks.len()
        || claims.walk_acc.iter().zip(&plan.walks).any(|(c, w)| c.len() != w.mats.len())
        || claims.use_sums.len() != plan.n_use()
        || claims.table_sums.len() != plan.n_table_sums()
        || claims.glue_ties.len() != n_glue
        || plan.glue.as_ref().is_some_and(|g| {
            g.regions.iter().zip(&claims.glue_ties).any(|(r, c)| r.ties.len() != c.len())
        })
        || fixed.sizes.len() != plan.n_trees()
        // stream BB: one claim each, or none at all
        || claims.embed_out.len() != usize::from(plan.emb.is_some())
        || claims.embed_tok.len() != usize::from(plan.emb.is_some())
    {
        return Verdict::rejected("claims shape mismatch");
    }
    let config = pcs_config();
    if fixed.log_blowup != crate::util::LOG_BLOWUP || fixed.n_queries != crate::util::N_QUERIES
        || fixed.pow_bits != crate::util::POW_BITS
    {
        return Verdict::rejected("security parameters drifted");
    }

    // ---- replay the transcript ------------------------------------------------
    let mut ch = Blake2sM31Channel::default();
    config.mix_into(&mut ch);
    let mut vcs = CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
    let root_ok = format!("{:?}", proof.commitments[0]) == fixed.pre_root;
    vcs.commit(proof.commitments[0], &fixed.sizes[0], &mut ch);
    vcs.commit(proof.commitments[1], &fixed.sizes[1], &mut ch);
    let rels = Rels::draw(&mut ch, &plan.widths, plan.glue.as_ref().map(|g| g.uns.as_slice()),
                          plan.emb.map(|l| l.arity()));
    let n_ch = plan.n_challenges();
    let f = ch.draw_secure_felts(n_ch);
    let (r, s, rv) = (f[0], f[1], f[2]);
    let (w_glue, lam_glue) = if plan.has_ties() {
        (f[3], f[4])
    } else {
        (SecureField::zero(), SecureField::zero())
    };
    let r_corner = if plan.p.corner { f[plan.corner_challenge_at()] } else { SecureField::zero() };
    // B11 stream Q: on the GKR arm the bands' fractional sums are proved here,
    // between the relation challenges the fractions depend on and the
    // interaction tree's own commit -- the position the prover occupies. The
    // sumcheck moves the channel, so a verifier that skipped it would diverge
    // at OODS; that is the shape of the failure if the two ever drift.
    let mut gkr_sums: Vec<SecureField> = vec![];
    let mut link_hl: Option<Vec<crate::gkrlink::HeightLink>> = None;
    if plan.p.gkr_bands {
        let Some(bytes) = claims.gkr.as_ref() else {
            return Verdict::rejected("the statement asks for the GKR arm and carries no GKR proof");
        };
        let sites = crate::gkrband::sites_of(&plan);
        let (sums, artifact, insts) = match crate::gkrband::verify(&mut ch, &sites, bytes) {
            Ok(v) => v,
            Err(e) => return Verdict::rejected(&format!("the bands' GKR proof: {e}")),
        };
        gkr_sums = sums;
        // B11 stream S, lever 1: the bands' sums never enter `use_sums` on
        // this arm -- the committed running-sum columns are gone. The GKR's
        // per-site sums join the cancellation directly below, standing exactly
        // where the claimed sums stood, and the sumcheck the channel just
        // replayed is what certifies them.
        if gkr_sums.len() != sites.len() {
            return Verdict::rejected("the GKR proved a different number of sites than the plan has");
        }
        // B11 stream S, lever 3: the fold challenge, at the same transcript
        // position the prover drew it, then the per-height parameters the tie
        // constraints assert. A malformed artifact is a rejection here.
        let alpha = ch.draw_secure_felts(1)[0];
        match crate::gkrlink::link_params(&plan, &rels, &insts, &artifact, alpha) {
            Ok(hl) => link_hl = Some(hl),
            Err(e) => return Verdict::rejected(&format!("the bands' GKR link: {e}")),
        }
    } else if claims.gkr.is_some() {
        return Verdict::rejected("the statement carries a GKR proof the flag did not ask for");
    }
    vcs.commit(proof.commitments[2], &fixed.sizes[2], &mut ch);
    // Stream H: one tree per boundary side, replayed in the prover's order.  The
    // roots are read out below; nothing is drawn from them here, because nothing
    // in the AIR reads a boundary column except its equality constraint.
    if proof.commitments.len() < plan.n_trees() {
        return Verdict::rejected("the proof carries fewer roots than the statement claims");
    }
    for t in 3..plan.n_trees() {
        vcs.commit(proof.commitments[t], &fixed.sizes[t], &mut ch);
    }
    // Stream AA: every boundary tree's root, in the prover's order -- stream H's
    // two, then the K/V sides.  `bound_in`/`bound_out` stay as the first pair so
    // that H's and I's probe JSON does not move; `Verdict` is *output* and not
    // `Fixed`, so neither field touches the digest.
    let bound_roots: Vec<String> = (crate::bounds::FIRST_TREE..plan.n_trees())
        .map(|t| format!("{:?}", proof.commitments[t]))
        .collect();
    let (bound_in, bound_out) = match plan.bounds.len() {
        2 => (Some(format!("{:?}", proof.commitments[3])),
              Some(format!("{:?}", proof.commitments[4]))),
        _ => (None, None),
    };
    ch.mix_felts(&claims.flat());

    // ---- components from the parsed statement ----------------------------------
    let built = build_components_linked(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue,
                                        &claims, link_hl.as_deref());
    let components = built.components();
    // the AIR's own column sizes must agree with the committed ones
    let derived = TreeVec::concat_cols(components.iter().map(|c| c.trace_log_degree_bounds()));
    let sizes_ok = derived.len() == plan.n_trees()
        && (1..plan.n_trees()).all(|t| derived[t] == fixed.sizes[t]);

    let vres = verify(&components, &mut ch, &mut vcs, proof);
    let stark_ok = vres.is_ok();
    let error = vres.err().map(|e| format!("{e:?}"));

    // ---- claimed sums and identities -------------------------------------------
    // B11 stream S, lever 1: on the GKR arm the bands' fractional sums arrive
    // from the GKR's output claims instead of `use_sums`; the cancellation is
    // the same global identity over the same fractions either way.
    let total = claims
        .use_sums
        .iter()
        .chain(claims.table_sums.iter())
        .fold(SecureField::zero(), |a, b| a + unqm(*b))
        + gkr_sums.iter().fold(SecureField::zero(), |a, b| a + *b);
    let sums_cancel = total == SecureField::zero();

    let mut qk_ok = vec![];
    let mut av_sum = SecureField::zero();
    let mut az_sum = SecureField::zero();
    // B10 step 3: the corner bands come after the prefix bands, one for one.  A
    // corner band has no per-token region of its own -- there is one row maximum
    // and one denominator per *token*, and both live on the prefix band -- so its
    // Q.K^T identity is read against the prefix band's `am`, and its numerator sum
    // is added to the prefix band's before the denominator identity is checked.
    // Stream AA: the corner's `b - n_pre` pairing generalises to "which head
    // band group is this band in", because with `--kv-bind` a group is one
    // prefix sub-band *per key block* plus the chunk's own corner, not a pair.
    // `plan.group` and `plan.rows_band` are that generalisation and with the
    // flag off they name exactly the bands the two lines below named.
    let row_of: Vec<usize> = (0..plan.p.bands.len())
        .map(|g| {
            (0..plan.shapes.len())
                .find(|b| plan.group[*b] == g && plan.rows_band[*b])
                .expect("every head band group carries one per-token region")
        })
        .collect();
    let (_, rv_hd) = powers(rv, p.hd);
    let (_, r_c) = powers(r, plan.head_weight_c());
    for (b, bc) in claims.bands.iter().enumerate() {
        let pow = &built.pows[b];
        let (chan, hs, av) = (unqm(bc[0]), unqm(bc[1]), unqm(bc[2]));
        // a band without a per-token region reads its group's row maximum, by
        // construction: it has no `am` of its own to disagree with
        let am = unqm(claims.bands[row_of[plan.group[b]]][4]);
        qk_ok.push(pow.r_sum * am - hs == chan);
        // fold this band's A.V claim by its head offset; the denominator claim
        // is the group's row band's alone, and every other band's `sn` joins it
        // below
        let woff_out = rv_hd.pow(plan.head_offsets[b] as u128);
        av_sum += woff_out * av;
        if plan.rows_band[b] {
            az_sum += r_c.pow(plan.head_offsets[b] as u128) * unqm(bc[5]);
        }
    }
    let den_ok: Vec<bool> = (0..plan.p.bands.len())
        .map(|g| {
            let sn = (0..plan.shapes.len())
                .filter(|b| plan.group[*b] == g)
                .fold(SecureField::zero(), |a, b| a + unqm(claims.bands[b][3]));
            sn == unqm(claims.bands[row_of[g]][5])
        })
        .collect();
    let av_ok = av_sum == unqm(claims.ao_out);
    let den_link_ok = az_sum == unqm(claims.ao_den);
    let walk_ok: Vec<bool> = plan
        .walks
        .iter()
        .enumerate()
        .flat_map(|(wi, w)| {
            w.mats.iter().enumerate().map(move |(k, mi)| (wi, k, *mi))
        })
        .map(|(wi, k, mi)| unqm(claims.walk_acc[wi][k]) == unqm(claims.mat_h[mi]))
        .collect();

    // B10 step 2: the wiring family's equalities, from the parsed claims alone.
    // `gn1.roots == gn1b.in` and `gn2.roots == gn2b.in` say each norm's two
    // levels are one reduction over the stream the layer committed; `gn1.xmid ==
    // gn2.in` says the residual stream is threaded from the first add to the
    // second.  With them the statement runs from `stream_in` to `stream_out`.
    let (glue_ok, stream_in, stream_out) = match &plan.glue {
        Some(g) => (
            g.eqs
                .iter()
                .map(|((ra, ta), (rb, tb))| claims.glue_ties[*ra][*ta] == claims.glue_ties[*rb][*tb])
                .collect(),
            Some(claims.glue_ties[g.stream_in.0][g.stream_in.1]),
            Some(claims.glue_ties[g.stream_out.0][g.stream_out.1]),
        ),
        None => (vec![], None, None),
    };
    // stream E: and one per dequant output tie -- the `o` and `down` regions'
    // own published outputs against the copies the wiring family committed.
    // This is what makes `o_out` and `d_out` the dequant's results rather than
    // two columns that merely happen to agree.
    let deq_out_ok: Vec<bool> = match &plan.glue {
        Some(g) => g
            .out_ties
            .iter()
            .enumerate()
            .map(|(k, ((ra, ta), _))| {
                claims.deq_out.get(k).copied() == Some(claims.glue_ties[*ra][*ta])
            })
            .collect(),
        None => vec![],
    };

    // stream E: the argmax compared the head's own output.  Both claims are
    // Horner sums over the same values with mirrored weights, so equality is the
    // whole statement -- and it is checked from the parsed claims alone.
    let head_tie_ok = if plan.p.head_argmax {
        !claims.am_key.is_empty() && claims.am_key.first() == claims.head_key.first()
    } else {
        true
    };

    // stream BB: the embedding gadget's tie.  Both claims are Horner sums over
    // the same sequence of stream values under the same weights, so equality is
    // the whole statement -- and it is checked from the parsed claims alone.
    let embed_ok = match (&plan.emb, &plan.glue) {
        (Some(_), Some(g)) => {
            claims.embed_out.first() == Some(&claims.glue_ties[g.stream_in.0][g.stream_in.1])
        }
        (Some(_), None) => false,
        (None, _) => true,
    };

    Verdict { stark_ok, root_ok, sums_cancel, qk_ok, den_ok, av_ok, den_link_ok, walk_ok, deq_out_ok,
              head_tie_ok, embed_ok,
              sizes_ok, glue_ok, stream_in, stream_out, bound_in, bound_out, bound_roots, error,
              verify_secs: t0.elapsed().as_secs_f64() }
}
