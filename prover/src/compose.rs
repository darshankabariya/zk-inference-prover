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
use crate::glue::{self, GKind, GTableEval, GlueEval, GlueRels, GlueTables, Kind, Prog, Binding,
                  BindingKind, TieSpec, Tiling, UnKind};
use crate::layout::Shape;
use crate::walk_air::{AO_SEL, AttnOutEval, EPOW_COLS, WALK_SEL, WalkEval, ao_chan, ao_sel,
                      walk_sel, walk_w};

// ---------------------------------------------------------------- relations ----

/// Every relation of the composed proof, drawn in one canonical order.
pub struct Rels {
    pub exp: ExpRel,
    pub norm: NormRel,
    pub frac: FracRel,
    pub mul: MulRel,
    pub qnz: PairRel,
    /// Stream JJ (E11): the `(e - e_min, 2^max(e,0), 2^max(-e,0))` window table
    pub epow: FracRel,
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
    /// Stream II: the requantize seams' own relations, drawn after everything
    /// else so a proof without them draws exactly what it always drew.  All
    /// five are drawn whenever any seam exists, whichever variants are in use --
    /// a relation nobody combines against costs one channel draw and keeps the
    /// order a function of "are there seams" alone.
    pub rq: Option<RqRels>,
    /// Stream NN-3: the quantize family's own tables, drawn after the seams'
    /// and only when the chunk carries a quantize region -- so a proof without
    /// the rotation draws exactly what it always drew.
    pub qq: Option<QuantRels>,
}

/// Stream NN-3: one division relation per side width, in `Plan::q_tabs` order,
/// and the family's own `(q, nz)` pair relation.  The regions keep reading the
/// layer's 15-bit range and `requant.rs`'s sign table, which are width-free.
#[derive(Clone)]
pub struct QuantRels {
    pub div: Vec<FracRel>,
    pub qnz: PairRel,
}

/// Stream II: the five relations a requantize seam reads, beyond the `qnz` and
/// 15-bit range relations it shares with the rest of the layer.
#[derive(Clone)]
pub struct RqRels {
    pub rqv: FracRel,
    pub rq3: FracRel,
    pub rq4: MulRel,
    pub scl2: PairRel,
    pub scl3: FracRel,
}

impl Rels {
    /// `uns` is the plan's unary-kind list: the relations are drawn one per
    /// entry, in that order, and both sides index them by it.  `emb` is the
    /// embedding relation's arity, or `None` when the gadget is off; `rq` says
    /// whether the chunk carries requantize seams.
    pub fn draw(ch: &mut Blake2sM31Channel, widths: &[u32],
                uns: Option<&[crate::glue::UnKind]>, emb: Option<usize>, rq: bool,
                qtabs: usize) -> Self {
        let exp = ExpRel::draw(ch);
        let norm = NormRel::draw(ch);
        let frac = FracRel::draw(ch);
        let mul = MulRel::draw(ch);
        let qnz = PairRel::draw(ch);
        let ranges: Vec<RangeRel> = widths.iter().map(|_| RangeRel::draw(ch)).collect();
        // stream JJ: drawn after the ranges so the earlier relations keep their
        // place in the transcript and only this proof's own tail moves
        let epow = FracRel::draw(ch);
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
        let rq = rq.then(|| RqRels {
            rqv: FracRel::draw(ch),
            rq3: FracRel::draw(ch),
            rq4: MulRel::draw(ch),
            scl2: PairRel::draw(ch),
            scl3: FracRel::draw(ch),
        });
        // stream NN-3: last of all, so every relation before it keeps its place
        // in the transcript
        let qq = (qtabs > 0).then(|| QuantRels {
            div: (0..qtabs).map(|_| FracRel::draw(ch)).collect(),
            qnz: PairRel::draw(ch),
        });
        Self { exp, norm, frac, mul, qnz, epow, widths: widths.to_vec(), ranges, glue, emb, rq,
               qq }
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

/// `Default` is here for `bounds`'s pair-check tests, which need a statement
/// that is well-formed for the chain check and for nothing else; no prover path
/// builds a `Params` this way.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
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
    /// Stream JJ (E11), `witness-slack.md` section 1f.  The attention output's
    /// rescaling exponent is `e = jv[h,ch] - jo[t]`.  `jv` is calibrated
    /// offline and is a model constant, so `ao_jv_min`/`ao_jv_max` are a
    /// statement of the model; `jo` is per-token prover data, so
    /// `ao_jo_min`/`ao_jo_max` are the window the statement admits the prover's
    /// choice in.  Together they fix the `EPOW` table's domain
    /// `[ao_jv_min - ao_jo_max, ao_jv_max - ao_jo_min]`, which is what bounds
    /// the committed `num` and `pj` inside the AIR.
    /// `ao_acc_lo_bits`/`ao_acc_hi_bits` are the two limbs of the range check
    /// on `acc + 2^(lo + hi - 1)`, the other half of the no-wrap window.
    /// `Plan::new` asserts the whole window fits under the field.
    #[serde(default)]
    pub ao_jv_min: i32,
    #[serde(default)]
    pub ao_jv_max: i32,
    #[serde(default)]
    pub ao_jo_min: i32,
    #[serde(default)]
    pub ao_jo_max: i32,
    #[serde(default)]
    pub ao_acc_lo_bits: u32,
    #[serde(default)]
    pub ao_acc_hi_bits: u32,
    /// Stream TT (E11), log row 73.  The **magnitude form** of the same window.
    /// With it on, the slot commits `mag = |acc|` beside `acc` and the two
    /// limbs range-check *`mag`*, not `acc + 2^(lo+hi-1)`; the sign is carried
    /// by the `sg` column the region already has, so the limbs cover
    /// `AttnOut::acc_bits()` rather than `acc_bits() + 1` and the same
    /// `1 + lo + hi + max(e_hi, 0) <= 31` window admits one more bit of
    /// exponent.  It costs one committed column per slot and no lookup.
    ///
    /// It is **off unless the offset form does not fit** (`main.rs`'s
    /// `params_of`), so every chunk that composed before this field existed
    /// keeps the offset form, its columns, its proof and its digest; the field
    /// is skipped in the serialization then, so the statement bytes do not move
    /// either.  `y-delay-vw12`'s layer-0 chunks are what need it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ao_acc_mag: bool,
    /// Stream TT (E11), log row 73, the sibling bound's half.  With it on the
    /// region looks up `2 denp` in the `ao_rem_bits` table rather than `denp`,
    /// which states `denp <= 2^(rem-1) - 1` -- the width an honest `denp`
    /// actually has, since `AttnOut::rem_bits` is the bit length of `max
    /// 2 denp`.  It costs no table, no column and no lookup, and it moves
    /// `Plan::new`'s quotient assert from `ao_q_bits + ao_rem_bits <= 30` to
    /// `<= 31`.  Off, and skipped in the serialization, unless the plain form
    /// does not fit.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ao_denp2: bool,
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
    /// mode for the same reason: it changes the proof, and its reference digest is its
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
    /// stream bindings; absent, and a chunk composes exactly as it did before.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub layer_boundary: bool,
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
    pub kv_boundary: bool,
    /// Stream E: publish the head's order-key claim.  It sits here rather than in
    /// `GlueParams` because a head dump carries no wiring columns at all -- the
    /// two features share the binding challenge, not the witness.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub logits_claim: bool,
    /// Stream E: compose the argmax region into this proof, bound to the head's
    /// key claim.  Implies `logits_claim` -- that claim is its counterparty.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prove_argmax: bool,
    /// the weaker statement ("the token attains the maximum"), which is what the
    /// binding probe separates from the strict one
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prove_argmax_lax: bool,
    /// Stream BB (E11): the embedding gadget's vocabulary, and zero when the
    /// gadget is off -- which is what the plan reads as "no embedding region".
    /// The gadget proves that the layer's committed residual-stream input is
    /// row `tok[t]` of an embedding matrix pinned in tree 0, so it needs the
    /// wiring family (its binding's counterparty is `stream_bindings[stream_in]`) and it
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
    /// Stream NN (E11), `witness-slack.md` section 1e: RoPE and the q/k/v
    /// quantize, the two steps `b1/model.py` takes between the projections and
    /// `attention_int`.  Absent -- and skipped in the serialization, so a
    /// statement written without it is byte identical to one from a crate that
    /// never had the feature -- on every chunk that does not prove them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope: Option<RopeParams>,
}

/// Stream NN: what the verifier needs to rebuild the rotation's own table.
///
/// `cos` and `sin` are preprocessed operands of the RoPE regions, and
/// `rope(pos)` reads the token's **absolute** position -- `pos0 + t` -- so tree
/// 0 is a function of the block index as well as of (model, layer, shape,
/// params).  `pos0` is that dependence, written down: `verify_composed` checks
/// it against `Fixed::block`, which `boundary::chain_of` already compares along
/// the token axis, so the block a chunk's angles were built for is the block the
/// chain puts it at.  Stream LL's tree-0 derivation must take the same
/// coordinate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct RopeParams {
    /// key/value heads; `nh / nkv` is the GQA replication factor
    pub nkv: usize,
    /// absolute position of this block's token 0
    pub pos0: usize,
    /// `rope_theta`, as an IEEE-754 bit pattern -- exact, and reproducible from
    /// a serialized statement without a float parser's rounding in the way
    pub theta: u64,
    /// which branch of `inv_freq_of` this model takes: `default`, `llama3` or
    /// `proportional`
    pub rope_type: String,
    /// that branch's own constants, as bit patterns and in its own order:
    /// `llama3` reads `[factor, low_freq_factor, high_freq_factor,
    /// original_max_position_embeddings]`, `proportional` reads
    /// `[partial_rotary_factor, factor]`, `default` reads none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rope_scale: Vec<u64>,
}

impl RopeParams {
    /// The angle table's frequencies, from the statement and nothing else.
    /// Stream LL: this is what makes tree 0's `cos` and `sin` columns
    /// `needs: Params, varies: Block` -- a verifier never reads them off a dump.
    pub fn inv(&self, hd: usize) -> Vec<f64> {
        let sc: Vec<f64> = self.rope_scale.iter().map(|b| f64::from_bits(*b)).collect();
        crate::rope::inv_freq_of(hd, f64::from_bits(self.theta), &self.rope_type, &sc)
    }
}

/// The wiring family's two shape numbers.  `a` is how many rows one token's
/// residual stream occupies at the reduction's first level: `tree_sum` groups the
/// channels by residue mod `a`, so a row's committed leaves are the channels
/// `r, r+a, ...` and the second level is one row per token over the `a` roots.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct GlueParams {
    pub d: usize,
    pub a: usize,
    /// The FFN width, for the SwiGLU's own tiling.  Stream II made the region
    /// unconditional, so a dump whose manifest predates `dff` no longer composes
    /// with the wiring family at all -- and says so rather than proving less.
    pub dff: usize,
    /// Stream II: the model's two RMSNorm constants, as bf16 patterns.  They are
    /// in the *statement* because the norm scale's `r` is now a table lookup
    /// whose table bakes them in (`UnKind::NormR`), and a table is only as
    /// pinned as the parameters the verifier builds it from.
    pub eps: u16,
    pub inv_d: u16,
    /// Stream E: make `o_out` and `d_out` the `o` and `down` dequant regions'
    /// own published outputs, bound to the copies the wiring family commits.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bind_matmul_outputs: bool,
    /// Stream II (E11), stream KK's finding: this chunk is the `lm_head` chunk,
    /// so its wiring family is the *final* RMSNorm and nothing else -- one
    /// reduction pair and one norm scale, whose input is the last layer's committed
    /// residual and whose output the head requantizes.  Skipped in the
    /// serialization, so a layer chunk's statement is byte identical to one
    /// from a crate that never had the mode.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub head_norm: bool,
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
    /// the pinned root of tree 0, the shape tree: tables, selectors, masks,
    /// cos/sin and `dinv` -- everything a verifier rebuilds from this statement
    /// and the model manifest's per-layer calibration, with no weights.
    pub pre_root: String,
    /// Stream PP (E11): the pinned root of tree 1, the model tree -- the
    /// weights and their scales.  One constant per (model, layer); the verifier
    /// compares it against the published manifest and derives nothing.
    pub model_root: String,
    /// Stream KK: which layer of the model this chunk proves, and which token
    /// block.  Declared -- like `pre_root`, and pinned to the model only when
    /// stream LL derives the expected tree-0 root per (layer, shape) -- but
    /// *compared*: `boundary::chain_of` demands `layer_b == layer_a + 1` on a
    /// layer edge and `block_b == block_a + 1` on a token edge, and
    /// `boundary-grid` takes the grid coordinate from here rather than from the
    /// tag.  `None` on a statement written before the field existed, which is
    /// why the comparisons are conditional on both sides declaring one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<usize>,
    /// Stream LL (E11), `notes/witness-slack.md` section 5 item 1's last
    /// sentence: the content hash of the model data tree 0 reads
    /// (`Layer::model_hash`).  Declared, like `pre_root` and like `layer` --
    /// but *compared*: `boundary-grid` refuses a grid whose chunks name two
    /// different models, and refuses two chunks at the same layer whose model
    /// hashes differ, so "two blocks of one layer used different weights" is
    /// no longer unseen.  What bindings the hash to the *proof* is the derived
    /// tree-0 root (`--expect-pre-roots`), which `zk-prover model-roots` computes
    /// from the same model.  `None` on a statement written before the field
    /// existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Stream KK (`notes/witness-slack.md` §1b): the token ids the embedding
    /// region committed, one per position of this block, in position order.
    /// `verify_composed` recomputes the Horner claim over them under the
    /// chunk's own replayed `w` and compares it against `claims.embed_tok`, so
    /// this vector is the proof's ids and not a second, free-standing
    /// declaration.  Empty on a chunk with no embedding region.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tok_in: Vec<u32>,
    /// Stream KK (§5 item 2): the tokens the `lm_head` chunk's argmax region
    /// emitted, one per position, pinned the same way against
    /// `claims.head_tok`.  Empty on a chunk that proves no argmax.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tok_out: Vec<u32>,
}

/// The prover-supplied half: every claimed value the identities compare.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Claims {
    /// per band: chan, hs, av, sn, am, az
    pub bands: Vec<[[u32; 4]; 6]>,
    pub ao_out: [u32; 4],
    pub ao_den: [u32; 4],
    /// Stream JJ (E11): the attention-output region's claim over its *own
    /// output* -- the signed requantized value `qm (1 - 2 sg)`, folded under
    /// exactly `ao_out`'s weights.  `AttnOutEval` is the requantization of the
    /// attention output, so this is the value the `o` walk's committed
    /// activations equal row for row; the equality is stream II's to add.  A
    /// `Vec` so a statement written before this stream still parses.
    #[serde(default)]
    pub ao_q: Vec<[u32; 4]>,
    /// per matmul: the dequant region's Freivalds side
    pub mat_h: Vec<[u32; 4]>,
    /// per walk group, per matmul: the walk's side
    pub walk_acc: Vec<Vec<[u32; 4]>>,
    /// per component with lookups, in component order
    pub use_sums: Vec<[u32; 4]>,
    /// per table component, in table order
    pub table_sums: Vec<[u32; 4]>,
    /// B10 step 2: per glue region, per binding, the Horner claim over the values it
    /// carries.  The verifier's equalities between them are what say the residual
    /// stream is threaded; `glue.stream_in` and `glue.stream_out` name the two
    /// that are the layer's boundary.
    #[serde(default)]
    pub stream_bindings: Vec<Vec<[u32; 4]>>,
    /// stream E: one per dequant output binding, in `GluePlan::output_bindings` order --
    /// the value the `o` and `down` regions publish for their own bf16 outputs
    #[serde(default)]
    pub deq_out: Vec<[u32; 4]>,
    /// Stream NN: per quantize region, the bf16 side it consumed and the
    /// integer side it produced.  `nn_in` is compared against the rotation's
    /// `Out(0)` (for q and k) or against `deq_out` (for v); `nn_out` against the
    /// sum of the corner bands' own claims over their committed columns.
    #[serde(default)]
    pub nn_in: Vec<[u32; 4]>,
    #[serde(default)]
    pub nn_out: Vec<[u32; 4]>,
    /// Stream NN: per band of `Plan::nn_bands`, that band's K, Q and V claims.
    #[serde(default)]
    pub band_qkv: Vec<[[u32; 4]; 3]>,
    /// Stream NN (NN-2): per band of `Plan::nn_qbands`, that **prefix** band's
    /// own Q claim.  Its counterparty is its group's corner, not a quantize
    /// region: `nn_q_ok` says the two folds agree, and the corner's is already
    /// bound to the rotated query, so the prefix's is too.
    #[serde(default)]
    pub band_q: Vec<[u32; 4]>,
    /// stream E: the head's order-key claim, when it is published
    #[serde(default)]
    pub logits_claim: Vec<[u32; 4]>,
    /// stream E: the argmax region's claim over the keys it compared -- equal to
    /// `logits_claim` is what says it compared the head's own output
    #[serde(default)]
    pub am_key: Vec<[u32; 4]>,
    /// stream KK: the argmax region's claim over the tokens it emitted, under
    /// the same `w` its key claim uses.  The counterparty is not another claim
    /// inside this proof but the statement's own `tok_out`, which the verifier
    /// recomputes it from.
    #[serde(default)]
    pub head_tok: Vec<[u32; 4]>,
    /// stream BB: the embedding region's Horner claim over the residual stream
    /// it proved.  Equal to `stream_bindings[stream_in]` is what says the row it
    /// proved is the row the layer consumed.
    #[serde(default)]
    pub embed_out: Vec<[u32; 4]>,
    /// stream BB: the same region's claim over the committed token ids, so the
    /// ids are a named value of the statement and a transcript anchor has
    /// something to bind to.  Nothing inside this proof compares it.
    #[serde(default)]
    pub embed_tok: Vec<[u32; 4]>,
    /// stream II: per requantize seam, its claim over the bf16 values it
    /// consumed -- equal to the producing region's `Out(0)` is what says the
    /// matmul's input is the producer's output.
    #[serde(default)]
    pub rq_in: Vec<[u32; 4]>,
    /// stream II: the same seam's claim over the signed integers it derived --
    /// equal to `walk_qx[the group]` is what says the walk multiplied *those*
    /// activations.
    #[serde(default)]
    pub rq_out: Vec<[u32; 4]>,
    /// stream II: per walk group, its own claim over its committed activations
    /// in the canonical channel-major order.
    #[serde(default)]
    pub walk_qx: Vec<[u32; 4]>,
    /// Stream II (E11): the requantize↔dequant scale binding.  `rq_sc` is one claim
    /// per seam over the scale that seam pinned to the attained maximum;
    /// `deq_sc` is one per *matmul* over the x-side scale its dequant used, and
    /// is empty on a chunk with no seams.  Indexed by matmul so the verifier
    /// reads a consumer's entry straight off the walk group's mat list.
    #[serde(default)]
    pub rq_sc: Vec<[u32; 4]>,
    #[serde(default)]
    pub deq_sc: Vec<[u32; 4]>,
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
/// Stream QQ (log row 61): at emit time a claim the placeholder set does not
/// supply is a defect, not a zero.
///
/// Every region reads its binding claims through `.get(..)`, and the missing case
/// used to be `SecureField::zero()`. That zero is not neutral: `Lower::ext`
/// folds `x - 0` and `Ir::schedule` drops the dead slot with it, so the emitter
/// installs a shape two instructions and one slot shorter than the one every
/// real chunk asks for, and the component takes the announced scalar fallback
/// for the life of the store. It happened once on `b65_key_eval` (open item 21,
/// the other direction: a claim the *witness* drove to zero) and again on
/// `deq_sc` and `walk_qx`, which stream II added without extending
/// `fill_binding_claims` -- ten of the wired C=4K chunk's thirteen fallback lines.
///
/// So on the emit path the missing case panics and names the vector. On the
/// prove path it stays a zero: a statement written before a claim existed still
/// parses, and that is what `#[serde(default)]` on those fields is for.
static EMIT_STRICT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_emit_strict(on: bool) {
    EMIT_STRICT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn claim_at(v: Option<[u32; 4]>, what: &str) -> SecureField {
    match v {
        Some(a) => unqm(a),
        None => {
            assert!(
                !EMIT_STRICT.load(std::sync::atomic::Ordering::Relaxed),
                "cudaemit: no placeholder for claim {what} -- every claim vector a plan \
                 can carry must be filled in `cudagen::fill_binding_claims`, or the emitted \
                 shape folds the missing claim away and no chunk ever finds it"
            );
            SecureField::zero()
        }
    }
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
        for g in &self.stream_bindings {
            v.extend(g.iter().map(|x| unqm(*x)));
        }
        // stream E, appended last: empty unless the dequant output bindings are on,
        // so the transcript is unchanged without them
        v.extend(self.deq_out.iter().map(|x| unqm(*x)));
        v.extend(self.logits_claim.iter().map(|x| unqm(*x)));
        v.extend(self.am_key.iter().map(|x| unqm(*x)));
        // stream BB, appended last: empty unless the embedding gadget is on, so
        // the transcript is unchanged without it
        v.extend(self.embed_out.iter().map(|x| unqm(*x)));
        v.extend(self.embed_tok.iter().map(|x| unqm(*x)));
        // stream KK, appended after both, for the same reason: a chunk without
        // the argmax mixes exactly what it mixed before
        v.extend(self.head_tok.iter().map(|x| unqm(*x)));
        // stream JJ, appended after those for the same reason
        v.extend(self.ao_q.iter().map(|x| unqm(*x)));
        // stream II, appended last: empty unless the chunk carries requantize
        // seams, so the transcript is unchanged without them
        v.extend(self.rq_in.iter().map(|x| unqm(*x)));
        v.extend(self.rq_out.iter().map(|x| unqm(*x)));
        v.extend(self.walk_qx.iter().map(|x| unqm(*x)));
        v.extend(self.rq_sc.iter().map(|x| unqm(*x)));
        v.extend(self.deq_sc.iter().map(|x| unqm(*x)));
        // stream NN, appended last for the same reason
        v.extend(self.nn_in.iter().map(|x| unqm(*x)));
        v.extend(self.nn_out.iter().map(|x| unqm(*x)));
        v.extend(self.band_qkv.iter().flatten().map(|x| unqm(*x)));
        // stream NN (NN-2), after the corner's three so no claim before it moves
        v.extend(self.band_q.iter().map(|x| unqm(*x)));
        v
    }
}

/// Stream II: one requantize seam's Horner weights.  The region tiles by
/// `MatLayout` -- row `j` is `(token, chunk)` with both descending, slot `sl`
/// holding channel `chunk * per_row + sl` -- which is the dequant's own map, so
/// these are `output_claim_w`'s constants: within a row the weight falls by `tokens`
/// per slot, and the row step takes `k_in` inside a token and `k_bd` at its
/// boundary.  Multiplying the accumulated steps out from row `(t, b)` to the
/// last row gives `w^(N - 1 - c*tokens - t)` per value, which is the canonical
/// stream weight the producing region's `Out(0)` binding already carries.
fn rq_binding_w(w: SecureField, lam: SecureField, rp: &RqPlan,
            claimed_in: SecureField, claimed_out: SecureField,
            claimed_sc: SecureField) -> crate::requant::RqTie {
    let n = (rp.tokens * rp.n) as i128;
    let (tok, pr, nn) = (rp.tokens as i128, rp.lay.per_row as i128, rp.n as i128);
    crate::requant::RqTie {
        w: (0..rp.lay.per_row).map(|sl| wpow(w, n - 1 - sl as i128 * tok)).collect(),
        k_in: wpow(w, -(pr * tok)),
        k_bd: wpow(w, (nn - pr) * tok - 1),
        lam,
        claimed_in,
        claimed_out,
        // stream II (E11): the scale binding's row step.  Both sides fold one value
        // per token on their own `last` rows with the tokens descending, so
        // token `t` carries `w^t` either side and no other constant is needed.
        mu: w,
        claimed_sc,
    }
}

/// Stream NN: one quantize region's two Horner weight sets.
fn q_binding_w(w: SecureField, lam: SecureField, qp: &QuantPlan, nh: usize,
           claimed_in: SecureField, claimed_out: SecureField) -> crate::rope::QTie {
    let m_in = (qp.tokens * qp.n) as i128;
    let n_out = (qp.tokens * nh * qp.hd) as i128;
    let (pr, hd, rep) = (qp.lay.per_row as i128, qp.hd as i128, qp.rep as i128);
    // the `rep` query heads one kv head is replicated into, as one weight
    let r_sum = (0..rep).fold(SecureField::zero(), |a, k| a + wpow(w, -(k * hd)));
    crate::rope::QTie {
        w_in: (0..qp.lay.per_row).map(|sl| wpow(w, m_in - 1 - sl as i128)).collect(),
        step_in: wpow(w, -pr),
        w_out: (0..qp.lay.per_row)
            .map(|sl| r_sum * wpow(w, n_out - 1 - sl as i128)).collect(),
        step_out: wpow(w, -pr),
        step_out_h: wpow(w, -pr - (rep - 1) * hd),
        lam,
        claimed_in,
        claimed_out,
    }
}

/// Stream NN: the corner bands' own committed Q, K and V, gathered onto the
/// `(token, query head, channel)` grid the quantize regions are read against.
/// A reference for the record, not an input to any constraint.
fn nn_band_cols(plan: &Plan, layer: &Layer) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (nh, hd, s) = (plan.p.nh, plan.p.hd, plan.p.s);
    let mut q = vec![0i64; s * nh * hd];
    let mut k = vec![0i64; s * nh * hd];
    let mut v = vec![0i64; s * nh * hd];
    for b in &plan.nn_bands {
        let a = &layer.bands[*b];
        let off = plan.head_offsets[*b];
        let sh = a.shape;
        for inst in 0..sh.insts {
            for t in 0..sh.s {
                for ch in 0..hd {
                    let dst = t * nh * hd + (off + inst) * hd + ch;
                    q[dst] = a.qq[(inst * sh.s + t) * hd + ch] as i64;
                    k[dst] = a.kk[(inst * sh.c + t) * hd + ch] as i64;
                    v[dst] = a.vv[(inst * sh.c + t) * hd + ch] as i64;
                }
            }
        }
    }
    (q, k, v)
}

/// Stream NN: band `b`'s three claims over its own committed K, Q and V, or
/// `None` when this band publishes none.
///
/// Every weight is `w^(N - 1 - (t nh hd + hq hd + ch))` -- the token-major
/// position on the **query-head** grid, which is the grid the bands hold and
/// which the quantize region's `w_out` reaches through its replication sum.
fn nn_band_bindings(plan: &Plan, b: usize, w: SecureField, claims: &Claims)
                -> Option<(crate::attn_air::BandTie, crate::attn_air::BandTie,
                           crate::attn_air::BandTie)> {
    let k = plan.nn_bands.iter().position(|x| *x == b)?;
    let p = &plan.p;
    let l = &plan.lays[b];
    let sh = &plan.shapes[b];
    let off = Offsets::new(p.q_bits, p.k_bits);
    let (hd, nh) = (p.hd as i128, p.nh as i128);
    let n = (p.s as i128) * nh * hd;
    let base = n - 1 - plan.head_offsets[b] as i128 * hd;
    // Stream QQ (log row 66): the same backstop the other claim vectors read
    // through -- an emit that reaches here without a placeholder folds this
    // band's three claims away and installs a `b65_chan_eval`/`b65_key_eval`
    // shape no rotated chunk asks for.
    let cl = claims.band_qkv.get(k).copied();
    let cl = |j: usize| claim_at(cl.map(|c| c[j]), "band_qkv");
    let (pk, pq, pv) = (l.chan.pk as i128, l.chan.pq as i128, l.key.pv as i128);
    let (cc, kc) = (l.chan.g.chunks as i128, l.key.g.chunks as i128);
    // the channel region's group index is `inst hd + ch`, which is affine in the
    // position, so one boundary constant serves
    let kt = crate::attn_air::BandTie {
        w: (0..l.chan.pk).map(|q| wpow(w, base - q as i128 * nh * hd)).collect(),
        s_in: wpow(w, -(pk * nh * hd)),
        s_bd: wpow(w, (cc - 1) * pk * nh * hd - 1),
        s_ib: None,
        off: off.k,
        claimed: cl(0),
    };
    let qt = crate::attn_air::BandTie {
        w: (0..l.chan.pq).map(|q| wpow(w, base - q as i128 * nh * hd)).collect(),
        s_in: wpow(w, -(pq * nh * hd)),
        s_bd: wpow(w, (cc - 1) * pq * nh * hd - 1),
        s_ib: None,
        off: off.q,
        claimed: cl(1),
    };
    // the key region's group index is `inst C + key`, which is not: crossing an
    // instance moves the head by one and the key by a whole context
    let vt = crate::attn_air::BandTie {
        w: (0..l.key.pv).map(|q| wpow(w, base - q as i128)).collect(),
        s_in: wpow(w, -pv),
        s_bd: wpow(w, (kc - 1) * pv - nh * hd),
        s_ib: Some(wpow(w, (sh.c as i128 - 1) * nh * hd + (kc - 1) * pv - hd)),
        off: off.v,
        claimed: cl(2),
    };
    Some((kt, qt, vt))
}

/// Stream NN (NN-2): a **prefix** band's Q claim, under the same canonical
/// weights the corner's takes.
///
/// The two bands of a group hold the same `S` queries over the same heads, so
/// the weight of the cell at (query head `hq`, channel `ch`, token `t`) is the
/// same `w^(N - 1 - (t nh hd + hq hd + ch))` on both -- only the *layout* that
/// scatters those cells over rows differs, and that is exactly what `pq` and
/// `chunks` carry.  Equality of the two folds is therefore "the same queries",
/// with no positional assumption beyond the one the corner's own binding already
/// makes.
fn nn_band_qtie(plan: &Plan, b: usize, w: SecureField, claims: &Claims)
                -> Option<crate::attn_air::BandTie> {
    let k = plan.nn_qbands.iter().position(|x| *x == b)?;
    let p = &plan.p;
    let l = &plan.lays[b];
    let off = Offsets::new(p.q_bits, p.k_bits);
    let (hd, nh) = (p.hd as i128, p.nh as i128);
    let n = (p.s as i128) * nh * hd;
    let base = n - 1 - plan.head_offsets[b] as i128 * hd;
    let pq = l.chan.pq as i128;
    let cc = l.chan.g.chunks as i128;
    Some(crate::attn_air::BandTie {
        w: (0..l.chan.pq).map(|q| wpow(w, base - q as i128 * nh * hd)).collect(),
        s_in: wpow(w, -(pq * nh * hd)),
        s_bd: wpow(w, (cc - 1) * pq * nh * hd - 1),
        s_ib: None,
        off: off.q,
        claimed: claim_at(claims.band_q.get(k).copied(), "band_q"),
    })
}

/// `w^e` for a signed exponent: an output side's row step points backwards
/// through the stream, so half of these are inverses.
fn wpow(w: SecureField, e: i128) -> SecureField {
    if e >= 0 { w.pow(e as u128) } else { w.pow((-e) as u128).inverse() }
}

/// Stream E: one matmul's output-side weights, from the wiring family's own binding
/// challenge -- the consumer builds its claim from the same `w`, which is what
/// makes the two comparable.  The value at channel `c` of token `t` carries
/// `w^(N - 1 - (c tokens + t))`; that splits into a per-slot weight and a row
/// step of two constants, derived in `dequant::OutputClaim`.
fn output_claim_w(w: SecureField, lam: SecureField, tokens: usize, n_out: usize,
           per_row: usize, kind: crate::dequant::OKind) -> crate::layer::OutputClaimW {
    let n = (tokens * n_out) as i128;
    let (tok, pr, no) = (tokens as i128, per_row as i128, n_out as i128);
    crate::layer::OutputClaimW {
        w: (0..per_row).map(|s| wpow(w, n - 1 - s as i128 * tok)).collect(),
        lam,
        k_in: wpow(w, -(pr * tok)),
        k_bd: wpow(w, (no - pr) * tok - 1),
        kind,
        head: None,
    }
}

/// Stream SS: a folded `v`'s output side, on the **band grid**.
///
/// The fold publishes an integer, and the only thing in the statement that holds
/// that integer is the corner bands' own V column -- so the weights are
/// `rope::QTie::w_out`'s, not a producer-grid Horner's: the canonical position is
/// `p = t nh hd + h hd + ch` over the *query* heads, and grouped-query attention
/// is inside the weight (`r_sum` folds the `rep` copies one kv channel has).  A
/// single equality still forces each copy separately, for the reason NN gives:
/// the exponents are distinct over (query head, channel, token), so the
/// difference is a polynomial in `w` whose coefficients were fixed before `w` was
/// drawn.
///
/// The row step is two constants keyed on the row *after* one that begins a kv
/// head, which is where the expanded channel index jumps by `(rep - 1) hd`; a
/// token boundary is such a row too, so no third constant is wanted.  It needs
/// `per_row` to divide `hd` -- asserted by the caller, and true of every layout
/// `MatLayout::choose` picks beside `rope::q_layout`'s.
fn output_claim_w_band(w: SecureField, lam: SecureField, tokens: usize, nh: usize, hd: usize,
                rep: usize, per_row: usize) -> crate::layer::OutputClaimW {
    let n = (tokens * nh * hd) as i128;
    let (pr, hdi, repi) = (per_row as i128, hd as i128, rep as i128);
    let r_sum = (0..repi).fold(SecureField::zero(), |a, k| a + wpow(w, -(k * hdi)));
    crate::layer::OutputClaimW {
        w: (0..per_row).map(|s| r_sum * wpow(w, n - 1 - s as i128)).collect(),
        lam,
        k_in: wpow(w, -pr),
        k_bd: wpow(w, -pr - (repi - 1) * hdi),
        kind: crate::dequant::OKind::Int,
        head: Some(hd / per_row),
    }
}

/// Stream NN: the same output side on the **token-major** canonical position
/// `p = t n_out + c`, which is `MatLayout`'s own row order -- so the weight
/// falls by `per_row` from one row to the next whether or not the row crosses a
/// token, and `k_in` and `k_bd` are the same constant.
///
/// Only `q`, `k` and `v` take it, and only because those three claims had no
/// counterparty before this stream: the rotation's rows are `hd` consecutive
/// channels of one head, which is a run of `hd` consecutive *token-major*
/// positions and a scattered set of channel-major ones.  `o`, `gate`, `up`,
/// `down` and `lm_head` keep `output_claim_w`.
fn output_claim_w_tok(w: SecureField, lam: SecureField, tokens: usize, n_out: usize,
               per_row: usize, kind: crate::dequant::OKind) -> crate::layer::OutputClaimW {
    let n = (tokens * n_out) as i128;
    let step = wpow(w, -(per_row as i128));
    crate::layer::OutputClaimW {
        w: (0..per_row).map(|s| wpow(w, n - 1 - s as i128)).collect(),
        lam,
        k_in: step,
        k_bd: step,
        kind,
        head: None,
    }
}

// -------------------------------------------------------------------- plan ----

/// Stream NN: one quantize gadget -- the bf16 stream a rotation (or the `v`
/// dequant) produced, and the integers a band committed.
pub struct QuantPlan {
    /// `"q"`, `"k"` or `"v"`; also the region's tag
    pub name: &'static str,
    /// index into `Params::mats`
    pub mat: usize,
    /// heads on the *producer* side: `nh` for q, `nkv` for k and v
    pub heads: usize,
    pub hd: usize,
    /// `nh / heads` -- how many query heads the band replicates one kv head into
    pub rep: usize,
    /// `heads * hd`
    pub n: usize,
    pub tokens: usize,
    pub lay: MatLayout,
    /// Stream NN-3: this side's integer width, `Params::q_bits` for `q`,
    /// `Params::k_bits` for `k` and `V_BITS` for `v`.  It sizes the region's own
    /// division table, and `rope::qmax_of` reads B1's clamp off it.
    pub bits: u32,
    /// `(glue region, binding)` publishing the bf16 side -- a rotation's `Out(0)`.
    /// `None` for `v`, whose bf16 side is the `v` dequant region's own published
    /// output and is compared against `claims.deq_out` instead.
    pub producer: Option<(usize, usize)>,
}

/// Stream II: one requantize seam -- a producing region's bf16 output, the
/// integers a walk group multiplies, and the region that proves one is the
/// requantize of the other.
///
/// There are three: `norm1`'s norm scale feeds the q/k/v group, `norm2`'s feeds
/// gate/up, and the SwiGLU's product feeds `down`.  The fourth matmul input,
/// `o`'s, has no region here because `AttnOutEval` **is** its requantize -- it
/// commits `qm` and `sg` and pins them by the exact-rounding identity -- so
/// what `o` needs is a claim on that region and not a second gadget.
pub struct RqPlan {
    /// the walk group's name, which is also the region's tag
    pub name: String,
    /// index into `Plan::walks`
    pub walk: usize,
    /// `(glue region, binding)` publishing the bf16 side
    pub producer: (usize, usize),
    pub n: usize,
    pub tokens: usize,
    pub lay: crate::layer::MatLayout,
    pub v: crate::requant::SVariant,
}

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
    /// per binding: what it accumulates, the within-row weight *exponents* and the
    /// row-step exponent -- powers of the challenge the transcript draws, so the
    /// verifier builds the same weights the prover did
    pub bindings: Vec<(BindingKind, Vec<u128>, u128)>,
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
/// norms' norm scales (`* r`, `* w`), the SwiGLU, and the bindings from `o`'s and `down`'s
/// dequant regions into `o_out` and `d_out`.
/// One equality the verifier checks between two regions' Horner claims.
///
/// Stream II added `scale`, because one of the new bindings compares sequences of
/// different *lengths*: the norm scale carries one committed sum of squares per
/// `(residue, token)` row and the reduction publishes one per token, so the
/// norm scale's claim is the reduction's multiplied by `sum_res w^(...)`.  The factor
/// is a function of the plan and the drawn challenge alone, so it is still one
/// equality between two claims and not a second mechanism; empty means one.
#[derive(Clone, Debug)]
pub struct GlueEq {
    pub a: (usize, usize),
    pub b: (usize, usize),
    /// exponents of the binding challenge whose powers sum to the factor `b` carries
    pub scale: Vec<i128>,
}

impl GlueEq {
    fn plain(a: (usize, usize), b: (usize, usize)) -> Self {
        Self { a, b, scale: vec![] }
    }
    /// The factor, from the challenge the transcript drew.
    pub fn factor(&self, w: SecureField) -> SecureField {
        if self.scale.is_empty() {
            return SecureField::one();
        }
        self.scale.iter().fold(SecureField::zero(), |acc, e| acc + wpow(w, *e))
    }
}

pub struct GluePlan {
    pub t: Tiling,
    /// the SwiGLU's own tiling, over `dff` instead of `d`
    pub tff: Option<Tiling>,
    pub regions: Vec<GlueRegionPlan>,
    /// what the verifier compares, `a == factor * b`
    pub eqs: Vec<GlueEq>,
    pub uns: Vec<UnKind>,
    /// the layer's committed residual-stream input and output claims
    pub stream_in: (usize, usize),
    pub stream_out: (usize, usize),
    /// stream E: the region index of each norm's scale, and of the SwiGLU, so the
    /// accounting can name their cells apart from the reduction's
    pub norm_scales: Vec<usize>,
    pub swiglu: Option<usize>,
    /// stream E: `((region, binding), name)` per dequant output binding -- the wiring
    /// side's claim over `o_out` and over `d_out`, and the matmul whose region
    /// has to publish the same value
    pub output_bindings: Vec<OutputBinding>,
    /// stream II: the head chunk's family (`gnf`, `gnfb`, `ntf`) instead of the
    /// layer's seven regions.  The witness build and the seam list both branch
    /// on it, and nothing else does.
    pub head_norm: bool,
    /// Stream NN: the rotation, as `(Mul region, Add region)` per rotated side,
    /// in `["q", "k"]` order.  Empty on a chunk without `Params::rope`.
    pub ropes: Vec<RopeRegions>,
}

/// Stream NN: one side's rotation, as two glue regions and the shapes both
/// halves are built from.
#[derive(Clone, Debug)]
pub struct RopeRegions {
    pub name: &'static str,
    /// index into `Params::mats` / `Plan::specs`
    pub mat: usize,
    /// heads on this side -- `nh` for q, `nkv` for k
    pub heads: usize,
    pub hd: usize,
    pub tokens: usize,
    /// region indices inside `GluePlan::regions`
    pub mul: usize,
    pub add: usize,
}

/// Stream E, generalised by stream JJ: one matmul's dequant output side.
///
/// `at` is the wiring-family binding the output is compared against -- region index
/// and binding index -- and it is `None` for the three whose consumer is not in the
/// statement at all.  `q` and `k` feed the attention through RoPE, which no
/// region proves, and `v` through a per-channel power-of-two requantize which no
/// region proves either; those three publish their claim and the verifier has
/// nothing to read it against yet (`briefs/jj-layerwire-out.md`'s record).
pub struct OutputBinding {
    pub name: &'static str,
    pub at: Option<(usize, usize)>,
    /// Stream NN: the counterparty is a quantize region's *input* claim rather
    /// than a wiring-family binding -- `v`, whose dequant output is rounded straight
    /// into the band's V column with no rotation between.  An index into
    /// `Plan::quants`.
    pub quant: Option<usize>,
    /// Stream SS: the counterparty is the corner bands' own V fold -- a folded
    /// `v`, whose dequant output already *is* the integer the band commits, so
    /// there is no quantize region between the two and the claim is compared
    /// against `sum_b band_qkv[b][2]` directly.  Set by `Plan::new`, which is
    /// where the bands and the layouts are known.
    pub band: bool,
}

impl GluePlan {
    /// Stream II (E11): the `lm_head` chunk's wiring family -- the final
    /// RMSNorm, and nothing else.
    ///
    /// The head consumes the int8 activation of `rmsnorm(x, final_norm)`, and
    /// before this the chunk committed that activation as a free column: no
    /// dump carried the final norm's weight, so the norm was proved by nobody
    /// (stream KK).  The family here is the layer's minus the parts a head
    /// chunk has not got.  There is no residual add, so level 1 is
    /// `Prog::tree(nreal, b, Square)` and not `norm_resid`; there is no second
    /// norm and no SwiGLU.  What is left is exactly the three statements the
    /// layer's own norms make, with the same table and the same weights:
    /// `gnf.Out(0) == gnfb.In(0)` (one reduction over two levels),
    /// `ntf.In(0) == gnf.In(0)` (the norm scale scaled the values the reduction
    /// squared) and `ntf.In(1) == G gnfb.Out(0)` (`r` is that reduction's own
    /// `rsqrt` answer, one per token across all `a` of its rows).
    ///
    /// `stream_in` is the residual claim, and **stream UU took the hook**: with
    /// `--bounds` the head chunk commits tree 3 over exactly that claim
    /// (`claims.stream_bindings[0][0]`), which says the residual this chunk normalized
    /// is the one the last layer's chunk published, and `boundary::chain_of`
    /// compares the two roots on the ordinary layer axis.  `stream_out` is the
    /// norm's output, which no chunk consumes, so `boundary::sides` gives the head
    /// the input side alone.
    fn head(p: &Params, gp: &GlueParams) -> Self {
        let t = Tiling::new(p.s, gp.d, gp.a);
        let (a, tok, nreal) = (t.a, t.tokens, t.nreal);
        let stream: Vec<u128> =
            (0..nreal).map(|j| (a * tok * (nreal - 1 - j)) as u128).collect();
        let l2_in: Vec<u128> = (0..a).map(|r| (tok * (a - 1 - r)) as u128).collect();
        let one = vec![0u128];
        let lr1 = crate::common::log_rows_for(t.rows1(), 1);
        let lr2 = crate::common::log_rows_for(tok, 1);
        let (rows1, rows2) = (1usize << lr1, 1usize << lr2);
        assert_eq!(rows1, a * tok, "the head norm's binding needs level 1 to tile exactly");
        assert_eq!(rows2, tok, "the head norm's binding needs level 2 to tile exactly");
        let un_r = UnKind::NormR { inv_d: gp.inv_d, eps: gp.eps };
        let l1 = Prog::tree(nreal, t.b, UnKind::Square);
        let l2 = Prog::tree(a, a, UnKind::None);
        let norm_scale = Prog::norm_scale(nreal, un_r);
        let mut uns = l1.un_kinds();
        if !uns.contains(&un_r) {
            uns.push(un_r);
        }
        let g_scale: Vec<i128> = (0..a)
            .map(|res| (rows1 as i128) - (rows2 as i128) - (res * tok) as i128)
            .collect();
        let regions = vec![
            GlueRegionPlan { tag: "gnf".into(), prog: l1, log_rows: lr1,
                             bindings: vec![(BindingKind::In(0), stream.clone(), 1),
                                        (BindingKind::Out(0), one.clone(), 1)] },
            GlueRegionPlan { tag: "gnfb".into(), prog: l2, log_rows: lr2,
                             bindings: vec![(BindingKind::In(0), l2_in, 1),
                                        (BindingKind::Out(0), one.clone(), 1)] },
            GlueRegionPlan { tag: "ntf".into(), prog: norm_scale, log_rows: lr1,
                             bindings: vec![(BindingKind::In(0), stream.clone(), 1),
                                        (BindingKind::In(1), one, 1),
                                        (BindingKind::Out(0), stream, 1)] },
        ];
        let eqs = vec![GlueEq::plain((0, 1), (1, 0)),
                       GlueEq::plain((2, 0), (0, 0)),
                       GlueEq { a: (2, 1), b: (1, 1), scale: g_scale }];
        Self { t, tff: None, regions, eqs, uns, stream_in: (0, 0), stream_out: (2, 2),
               norm_scales: vec![2], swiglu: None, output_bindings: vec![], head_norm: true,
               ropes: vec![] }
    }

    pub fn new(p: &Params, gp: &GlueParams) -> Self {
        if gp.head_norm {
            return Self::head(p, gp);
        }
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
                bindings: vec![
                    (BindingKind::In(0), stream.clone(), 1),
                    (BindingKind::Out(0), one.clone(), 1),
                    (BindingKind::Out(1), stream.clone(), 1),
                ],
            });
            regions.push(GlueRegionPlan {
                tag: n2.into(),
                prog: l2.clone(),
                log_rows: lr2,
                bindings: vec![(BindingKind::In(0), l2_in.clone(), 1),
                           (BindingKind::Out(0), one.clone(), 1)],
            });
        }
        let mut eqs = vec![GlueEq::plain((0, 1), (1, 0)), GlueEq::plain((0, 2), (2, 0)),
                           GlueEq::plain((2, 1), (3, 0))];
        let mut uns = l1.un_kinds();
        // ---- stream E: the dequant output bindings -------------------------------
        //
        // `gn1`'s group 1 *is* `o_out` and `gn2`'s is `d_out`, and neither has
        // ever carried a Horner claim -- the wiring family committed those
        // columns and only a `VAL` lookup pinned them.  One claim each, on the
        // same channel-major weights the stream uses, and the `o` and `down`
        // dequant regions publish the same value from their own outputs.
        //
        // Stream JJ (E11): unconditional, and all seven.  `o` and `down` keep
        // E's `In(1)` binding on the two reduction regions; `gate` and `up` are
        // read against the SwiGLU region's own `In(0)`/`In(1)`, which need no
        // new binding because the region already publishes them; `q`, `k` and `v`
        // publish a claim with no counterparty (see `OutputBinding`).  The list ends
        // up in `MAT_NAMES` order because `claims.deq_out` is pushed in matmul
        // order and the verifier reads the two side by side.
        let mut output_claim_at: Vec<(&'static str, Option<(usize, usize)>)> = vec![];
        for (ri, name) in [(0usize, "o"), (2usize, "down")] {
            regions[ri].bindings.push((BindingKind::In(1), stream.clone(), 1));
            output_claim_at.push((name, Some((ri, regions[ri].bindings.len() - 1))));
        }
        // ---- stream E: the two norm scales ------------------------------------
        //
        // A norm scale's group 0 is the same block of the residual stream the
        // reduction's own group 0 is, in the same order and under the same
        // weights, so the binding `nt.In(0) == gn.In(0)` says the norm scale scaled the
        // very values the reduction squared -- which is the half of "the norm
        // computed its own output" a committed column cannot fake.  `Out(0)` is
        // published for the requantize region that will read it.
        //
        // Stream II made the region unconditional and closed both of the gaps
        // the first build left.  `r` is no longer a committed leaf: group 1
        // commits the reduction's *answer* and `UnKind::NormR`'s lookup is
        // `r = rsqrt(ms inv_d + eps)`, so the binding `nt.In(1) == G gnb.Out(0)`
        // makes it the reduction's own answer -- and because every one of a
        // token's `a` rows is compared against that one reduction row, they
        // cannot carry different `r`s either.  `G` is the geometric sum over
        // the `a` rows and is the verifier's own arithmetic (`GlueEq::scale`).
        // The norm weight moved to `Src::Pre`, i.e. into tree 0 under
        // `pre_root`, because it is the model's and not the prover's.
        let mut norm_scales = vec![];
        {
            let rows1 = 1usize << lr1;
            let rows2 = 1usize << lr2;
            assert_eq!(rows1, a * tok,
                       "the norm scale's binding needs the level-1 region to tile exactly");
            assert_eq!(rows2, tok,
                       "the norm scale's binding needs the level-2 region to tile exactly");
            let un_r = UnKind::NormR { inv_d: gp.inv_d, eps: gp.eps };
            if !uns.contains(&un_r) {
                uns.push(un_r);
            }
            let norm_scale = Prog::norm_scale(nreal, un_r);
            let one_pos = vec![0u128];
            // sum_{res} w^(rows1 - rows2 - res * tokens)
            let g_scale: Vec<i128> =
                (0..a).map(|res| (rows1 as i128) - (rows2 as i128) - (res * tok) as i128).collect();
            for (gn, gnb, tag) in [(0usize, 1usize, "nt1"), (2usize, 3usize, "nt2")] {
                let i = regions.len();
                regions.push(GlueRegionPlan {
                    tag: tag.into(),
                    prog: norm_scale.clone(),
                    log_rows: lr1,
                    bindings: vec![(BindingKind::In(0), stream.clone(), 1),
                               (BindingKind::In(1), one_pos.clone(), 1),
                               (BindingKind::Out(0), stream.clone(), 1)],
                });
                eqs.push(GlueEq::plain((i, 0), (gn, 0)));
                eqs.push(GlueEq { a: (i, 1), b: (gnb, 1), scale: g_scale.clone() });
                norm_scales.push(i);
            }
        }
        // ---- stream E: the SwiGLU product -----------------------------------
        //
        // Over `dff`, so it carries its own tiling.  Its three claims have no
        // counterparty inside the glue yet: group 0 is the gate matmul's
        // dequantized output, group 1 is `up`'s, and the output is what `down`
        // requantizes.  All three land when the dequant bindings and the requantize
        // placement do; until then the claims are published and compared against
        // B1's own columns, and the record says so.
        let mut tff = None;
        let mut swiglu = None;
        {
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
            let i = regions.len();
            regions.push(GlueRegionPlan {
                tag: "gsw".into(),
                prog,
                log_rows: crate::common::log_rows_for(t2.rows1(), 1),
                bindings: vec![(BindingKind::In(0), ff.clone(), 1),
                           (BindingKind::In(1), ff.clone(), 1),
                           (BindingKind::Out(0), ff, 1)],
            });
            // stream JJ: the gate/up seam.  `Prog::pairs` puts the unary on
            // group 0, so `In(0)` is the silu's operand -- `gate` -- and `In(1)`
            // is `up`.
            output_claim_at.push(("gate", Some((i, 0))));
            output_claim_at.push(("up", Some((i, 1))));
            tff = Some(t2);
        }
        // ---- stream NN: the rotation ------------------------------------------
        //
        // Two regions per rotated side, because a `Prog` carries one `Kind` and
        // `rotate` is two multiplies and an add: `rpqm`/`rpqa` for q,
        // `rpkm`/`rpka` for k.  A row is one (token, head), its `hd` committed
        // leaves are that head's dequant output and `cos`/`sgnsin` are
        // preprocessed (`Src::Pre`), which is what makes tree 0 a function of
        // the block.
        //
        // The weights are the **token-major** canonical position `t n_out + c`
        // and not the channel-major one every earlier binding uses.  That is the
        // whole reason a single row step serves here: a (token, head) row's
        // leaves are `hd` consecutive positions, so a region's rows walk the
        // stream in order and the Horner is the flat one.  `output_claim_w_tok` puts
        // the q/k/v dequant regions' own output claims on the same convention,
        // which is free -- those three claims had no counterparty before this
        // stream and nothing else reads them.
        let mut ropes = vec![];
        if let Some(rp) = &p.rope {
            let hd = p.hd;
            for (name, heads) in [("q", p.nh), ("k", rp.nkv)] {
                let mi = p.mats.iter().position(|m| m.name == name)
                    .unwrap_or_else(|| panic!("--rope needs the {name} matmul"));
                let m = &p.mats[mi];
                assert_eq!(m.n_out, heads * hd,
                           "the {name} matmul is {} wide, not {heads} x {hd}", m.n_out);
                let rows = m.tokens * heads;
                let lr = crate::common::log_rows_for(rows, 1);
                assert_eq!(1usize << lr, rows,
                           "the rotation's rows must tile exactly: {rows} is not a power of two");
                let n_all = m.tokens * m.n_out;
                // `w[k] = w^(width - 1 - k)`, `step = w^width`: the flat Horner
                // over the stream, rows ascending in (token, head)
                let win: Vec<u128> = (0..hd).map(|c| (hd - 1 - c) as u128).collect();
                let wmid: Vec<u128> = (0..2 * hd).map(|c| (2 * hd - 1 - c) as u128).collect();
                let (mul, add) = (regions.len(), regions.len() + 1);
                regions.push(GlueRegionPlan {
                    tag: format!("rp{name}m"),
                    prog: Prog::rope_mul(hd),
                    log_rows: lr,
                    bindings: vec![(BindingKind::In(0), win.clone(), hd as u128),
                               (BindingKind::Out(0), wmid.clone(), (2 * hd) as u128)],
                });
                regions.push(GlueRegionPlan {
                    tag: format!("rp{name}a"),
                    prog: Prog::rope_add(hd),
                    log_rows: lr,
                    bindings: vec![(BindingKind::In(0), wmid, (2 * hd) as u128),
                               (BindingKind::Out(0), win, hd as u128)],
                });
                // the two halves are one rotation: the products the multiplies
                // rounded are the values the add rounds
                eqs.push(GlueEq::plain((mul, 1), (add, 0)));
                // and the multiplies' committed leaves are the matmul's own
                // dequant output, which is what shrinks `deq_out_open`
                output_claim_at.push((name, Some((mul, 0))));
                let _ = n_all;
                ropes.push(RopeRegions { name, mat: mi, heads, hd, tokens: m.tokens,
                                         mul, add });
            }
            // v is not rotated: its dequant output goes straight into the
            // quantize gadget, whose input claim `verify_composed` compares
            // against `deq_out` directly (`OutputBinding::quant`).
            output_claim_at.push(("v", None));
        } else {
            // stream JJ: q, k and v publish without a counterparty, then the list
            // is put in matmul order -- the order `claims.deq_out` arrives in
            for n in ["q", "k", "v"] {
                output_claim_at.push((n, None));
            }
        }
        let output_bindings: Vec<OutputBinding> = crate::layer::MAT_NAMES
            .iter()
            .filter_map(|n| output_claim_at.iter().find(|(m, _)| m == n))
            .map(|(name, at)| OutputBinding {
                name,
                at: *at,
                // filled in by `Plan::new`, which is where the quantize regions
                // are laid out; the wiring plan does not know their order
                quant: None,
                band: false,
            })
            .collect();
        Self {
            t,
            tff,
            regions,
            eqs,
            uns,
            stream_in: (0, 0),
            stream_out: (2, 2),
            norm_scales,
            swiglu,
            output_bindings,
            head_norm: false,
            ropes,
        }
    }

    /// The binding specs, with the claims the statement carries filled in.
    pub fn bindings(&self, w: SecureField, lam: SecureField, claims: &Claims) -> Vec<Binding> {
        self.regions
            .iter()
            .enumerate()
            .map(|(i, r)| Binding {
                lam,
                specs: r
                    .bindings
                    .iter()
                    .enumerate()
                    .map(|(k, (kind, we, se))| TieSpec {
                        kind: *kind,
                        w: we.iter().map(|e| w.pow(*e)).collect(),
                        step: w.pow(*se),
                        claimed: claim_at(
                            claims.stream_bindings.get(i).and_then(|g| g.get(k)).copied(),
                            "stream_bindings",
                        ),
                    })
                    .collect(),
            })
            .collect()
    }

    pub fn n_tables(&self) -> usize {
        3 + self.uns.len()
    }
}

/// Stream LL (E11): stop the prover after tree 0 is committed and report its
/// root.  Set by `zk-prover expect-root` and by nothing else; the ordinary `compose`
/// path never reads a set flag, so no recorded digest can move because of it.
pub static PRE_ONLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Stream PP (E11): stop one tree earlier -- after the **shape** tree and
/// before the model tree.  This is the verifier's own derivation: the shape
/// tree is a function of `Fixed.params` and of the manifest's per-layer
/// calibration, so a run that stops here has read no weight, and the desk gate
/// says so by deriving the same root at two seeds.  Set by
/// `zk-prover expect-root --shape-only` and by nothing else.
pub static SHAPE_ONLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Stream LL (E11): what a preprocessed column needs in order to be rebuilt.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub enum PreNeeds {
    /// `Fixed.params` alone determines every cell.
    Params,
    /// Stream PP (E11): a per-layer calibration constant -- `frac_per_head` and
    /// `master_table_bits` -- that is not in the weights but is published, per
    /// layer, by the model manifest `zk-prover model-roots` writes.  Public input, so
    /// the verifier rebuilds the column; it is not a weight, so it does not
    /// belong to the model tree.
    Manifest,
    /// the model dump: a weight or a weight scale.
    Model,
}

/// Stream PP (E11): which of the two committed preprocessed trees a column
/// takes.  `Params` and `Manifest` are the verifier's to rebuild and go to the
/// shape tree; `Model` is the publisher's and goes to the model tree, whose
/// root is the per-layer model commitment.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub enum PreTree {
    Shape,
    Model,
}

impl PreNeeds {
    pub fn tree(self) -> PreTree {
        match self {
            PreNeeds::Params | PreNeeds::Manifest => PreTree::Shape,
            PreNeeds::Model => PreTree::Model,
        }
    }
}

/// Stream LL (E11): does the column move down the block axis of a grid?
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub enum PreVaries {
    /// identical on every token block of the layer.
    Layer,
    /// its height or its contents are a function of the chunk's own `c`.
    Block,
}

/// Stream LL (E11): one entry of `Plan::pre_split`.
pub struct PreCol {
    pub id: PreProcessedColumnId,
    pub family: String,
    pub log_size: u32,
    pub needs: PreNeeds,
    pub varies: PreVaries,
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
    /// Stream II: the requantize seams, empty on a chunk with no wiring family.
    pub rqs: Vec<RqPlan>,
    /// the scale variants those seams use, deduplicated and in a fixed order --
    /// one division table and one scale table per variant
    pub rq_vars: Vec<crate::requant::SVariant>,
    /// Stream NN: the three quantize regions -- `q`, `k`, `v` -- in that order.
    /// Empty on a chunk without `Params::rope`.
    pub quants: Vec<QuantPlan>,
    /// Stream NN-3: the widths the quantize regions' own division tables are
    /// built at -- the distinct `QuantPlan::bits`, ascending, one table each.
    /// Empty when the chunk carries no quantize region.
    pub q_tabs: Vec<u32>,
    /// and the width of their shared `(q, nz)` pair table: the widest side's,
    /// because the division table is what fixes the admissible quotient and this
    /// one only pins `nz = [q != 0]`.  Zero when there are no regions.
    pub q_nz_bits: u32,
    /// Stream NN: the bands that publish a K/Q/V claim over their own committed
    /// columns.  Exactly the **corner** bands: their keys are this block's own
    /// tokens, which are the ones this chunk's `k` and `v` matmuls produced,
    /// while a prefix band's keys are the cache and are bound across blocks by
    /// root equality instead.  The corner also carries the queries, which every
    /// band of a group shares.
    pub nn_bands: Vec<usize>,
    /// Stream NN (NN-2): the bands that publish a Q claim **alone** -- every
    /// prefix band of the same chunk, in band order.  Their counterparty is
    /// their group's corner (`nn_q_ok`), which closes the asymmetry NN-1 left:
    /// before this, a prover could attend to the cache with queries the `q`
    /// matmul did not produce while attending to this block's own keys with
    /// queries it did.  Empty on a chunk without `Params::rope`.
    pub nn_qbands: Vec<usize>,
    pub widths: Vec<u32>,
    pub log_norm: u32,
    pub log_frac: u32,
    pub log_mul: u32,
    pub glue: Option<GluePlan>,
    /// Stream H: the two boundary sides, or empty.  A statement asks for the
    /// boundary and has the wiring family, or it has no boundary at all — the
    /// plan settles that once so the prover and the verifier cannot disagree
    /// about how many trees the transcript has.
    pub layer_sides: Vec<crate::boundary::Side>,
    /// Stream AA: the K/V boundary sides, or empty.  They follow `bounds` in the
    /// tree order, so turning either feature on does not move the other's tree
    /// indices.
    pub kv: Vec<crate::boundary::KvSide>,
    /// Stream AA, per band: which head band group it belongs to.  Without
    /// `kv_boundary` this is the band's own index modulo the prefix count, which is
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
        // Stream AA (E11), layout design A: with `--kv-boundary` the declared
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
        assert!(!p.kv_boundary || p.corner,
                "`--kv-boundary` tiles the prefix beside the chunk's own block; it needs `--corner`");
        assert!(!p.kv_boundary || p.c % p.s == 0,
                "`--kv-boundary` tiles the prefix in blocks of S: C={} is not a multiple of S={}",
                p.c, p.s);
        // with the flag off this is 1, so the loop below builds exactly the one
        // prefix band per head band it always built
        // Open item 1 / CC deliverable 4: block 0 of a sequence has no declared
        // prefix, so `c == 0` and there is no prefix band to build.  Without
        // this the `else` arm built one anyway and `Grid::new` divided by
        // `groups = c * insts = 0`.  With `nsub == 0` the loop below builds no
        // prefix band and `rows_band.push(nsub == 0)` hands the group's one
        // per-token region to the solo corner, which is where it belongs.
        let nsub = if p.c == 0 { 0 } else if p.kv_boundary { p.c / p.s } else { 1 };
        assert!(p.c > 0 || p.corner,
                "a c = 0 chunk is block 0: it is the solo corner and needs `--corner`");
        let sub_c = if p.kv_boundary { p.s } else { p.c };
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
        // Stream AA left `--kv-boundary` and `--gkr-bands` asserted apart, because
        // the bands' GKR arm gated the per-token region on `plan.corner` in two
        // places of its own while the composed path had already moved to
        // `plan.rows_band`.  What pairing by `corner` assumed is "every band
        // that is not the corner carries the group's per-token region", and
        // design A breaks exactly that: a group is now `c/s` prefix sub-bands
        // plus a corner, and the region lives on the *first* sub-band only.
        // Both derivations now read `plan.rows_band[b]`, which is `!corner[b]`
        // whenever the flag is off, so the two flags compose and the arm's
        // flag-off digests do not move.  See `briefs/aa-kvbind.md`, item 4.
        let lays: Vec<Layouts> = shapes.iter().map(|sh| Layouts::choose(sh, &cfg)).collect();

        // Stream JJ (E11): the no-wrap window, stated rather than asserted on
        // the host.  `2 mag num < p` needs both factors bounded: `num` by the
        // `EPOW` table's own domain, `mag = |acc|` by the two limbs.  The
        // second line is the other term of the same identity, which the design
        // has always needed and never said out loud.
        assert!(p.ao_jv_min <= p.ao_jv_max, "the jv window is empty");
        assert!(p.ao_jo_min <= p.ao_jo_max, "the jo window is empty");
        let e_hi = p.ao_jv_max - p.ao_jo_min;
        let e_lo = p.ao_jv_min - p.ao_jo_max;
        assert!(e_hi <= 30 && -e_lo <= 30,
                "the attention output's exponent window does not fit in the field:                  e in [{e_lo}, {e_hi}]");
        assert!(p.ao_acc_lo_bits >= crate::common::MIN_LOG_ROWS
                    && p.ao_acc_hi_bits >= crate::common::MIN_LOG_ROWS,
                "an accumulator limb is narrower than a range table's shortest column");
        // Stream TT (E11), log row 73: the same line reads both forms of the
        // magnitude bound, because both end at the same product.
        //   offset form (`ao_acc_mag` false): the limbs range-check
        //     `acc + 2^(lo+hi-1)`, so `mag = |acc| <= 2^(lo+hi-1)` and
        //     `2 mag num <= 2^(lo+hi+e_hi)`, which is under `p` exactly when
        //     `1 + lo + hi + e_hi <= 31`;
        //   magnitude form (`ao_acc_mag` true): they range-check the committed
        //     `mag` itself, so `mag <= 2^(lo+hi) - 1` and
        //     `2 mag num <= 2^(1+lo+hi+e_hi) - 2^(1+e_hi)`, under `p` at the
        //     same `1 + lo + hi + e_hi <= 31`.
        // The one bit between them is the sign, which the offset form spends on
        // the window and the magnitude form reads off `sg`.  `params_of` sizes
        // the limbs at `acc_bits() + 1` for the first and `acc_bits()` for the
        // second, and takes the second only when the first does not fit here.
        assert!(1 + p.ao_acc_lo_bits + p.ao_acc_hi_bits + e_hi.max(0) as u32 <= 31,
                "the attention output's no-wrap window does not fit:                  1 + {} + {} + {} > 31{}",
                p.ao_acc_lo_bits, p.ao_acc_hi_bits, e_hi.max(0),
                if p.ao_acc_mag { " (magnitude form)" } else { "" });
        // Stream OO (E11): the same term, at the bound the identity actually
        // needs.  The error identity is
        // `d = 2 mag num - (2 qm - 1) denp`, and what makes it an *integer*
        // identity rather than a field one is that both of its non-negative
        // sides stay under `p = 2^31 - 1`.  The right-hand side is the other
        // assert's (`2 mag num < 2^30`).  The left-hand side is
        // `d + (2 qm - 1) denp`, and the range checks bound both factors:
        // `qm` is looked up in the `qnz` table, whose domain is `2^ao_q_bits`
        // rows, so `qm <= 2^q_bits - 1`; `denp` is looked up in the
        // `ao_rem_bits` range table, so `denp <= 2^rem_bits - 1`, and `d` is in
        // the same table.  So
        //     d + (2 qm - 1) denp <= (2^(q+1) - 2)(2^rem - 1)
        //                          = 2^(q+rem+1) - 2^(q+1) - 2^(rem+1) + 2,
        // which is below `2^31 - 1` exactly when `q_bits + rem_bits <= 30`.
        // Stream JJ wrote the bound with an extra `1 +`, which is one bit
        // tighter and refuses `ao_rem_bits = 23` at `ao_q_bits = 7` -- the
        // C=4096 `b10-final` dump's own worst case (`shapes.json`).  Nothing
        // the assert protects is loosened: the two ends of the identity are the
        // same two products, bounded by the same two range tables.
        //
        // Stream TT (E11), log row 73: and the same one bit is loose here, for
        // the same reason.  `ao_rem_bits` is the bit length of `max 2 denp`
        // (`AttnOut::rem_bits`), so an honest `denp` needs `rem - 1` bits and
        // it is only the *lookup* -- `denp` in the `ao_rem_bits` table -- that
        // states the wider `denp <= 2^rem - 1`.  Looking up `2 denp` in the
        // same table instead states `denp <= 2^(rem-1) - 1`, at no new table
        // and no new column, and then
        //     d + (2 qm - 1) denp <= (2^rem - 1) + (2^(q+1) - 3)(2^(rem-1) - 1)
        //                          = 2^(q+rem) - 2^(q+1) - 2^(rem-1) + 2
        // for `q + rem = 31`, which is below `2^31 - 1`.  So the `denp2` form
        // admits `q_bits + rem_bits <= 31`.  `y-delay-vw12`'s layer 0 reads
        // `11 + 20 = 31` and is the reason it exists; `params_of` takes the
        // form only when the plain one does not fit, so no statement that
        // composed before this moves.
        let qcap = if p.ao_denp2 { 31 } else { 30 };
        assert!(p.ao_q_bits + p.ao_rem_bits <= qcap,
                "the attention output's quotient term does not fit: {} + {} > {qcap}{}",
                p.ao_q_bits, p.ao_rem_bits,
                if p.ao_denp2 { " (denp2 form)" } else { "" });
        // The inert lead row commits `denp = 2^max(-e_lo, 0)`, forced by the
        // `EPOW` lookup at key 0, so the `denp2` form needs twice that to still
        // be a row of the range table.
        assert!(!p.ao_denp2 || 1 + (-e_lo).max(0) as u32 <= p.ao_rem_bits,
                "the denp2 form needs 2^(1 + {}) inside the {}-bit remainder table",
                (-e_lo).max(0), p.ao_rem_bits);
        let mut widths = vec![p.k_bits, p.q_bits, 8, p.z_lo_bits, p.z_hi_bits, p.ao_rem_bits,
                              p.ao_acc_lo_bits, p.ao_acc_hi_bits];
        if let AttnVariant::IntSplit(k) = attn_variant {
            widths.push(k);
        }
        // Stream Z: a fold matmul range-checks its magnitude and low-remainder
        // limbs at 13 bits, its low-product quotient at 16, and its high
        // remainder at the pinned fold width; a widened attention output ranges
        // `o`'s committed activations at `ao_q_bits + 1`.
        //
        // Stream SS: and its clamped output `qc` at 7, and its quotient and
        // clamp slack at `ao_q_bits` (`dequant.rs`'s `Fold` arm asks for
        // `self.rel(7)` and `self.rel(sp.q_bits)`).  Neither was ever listed:
        // every fold shape run so far carried `--z-hi-bits 7` and a `k_bits`
        // equal to `ao_q_bits`, which put both in the list by accident, and a
        // shape that does not panicked in `Tables::ridx`.  The list is
        // deduplicated and sorted, so this moves no shape that already had them
        // -- the E11 set (`--z-hi-bits 7`, `k_bits = ao_q_bits = 11`) included.
        if p.mats.iter().any(|m| m.variant == "fold") {
            assert!(p.fold_rem_bits > 0, "a fold matmul needs --fold-rem-bits");
            widths.extend([crate::dequant::FOLD_LIMB, 16, p.fold_rem_bits, 7, p.ao_q_bits]);
        }
        if p.ao_q_bits > 7 {
            widths.push(p.ao_q_bits + 1);
        }
        // Stream II: the requantize seams' sub-half comparison and their max
        // argument are 15-bit range lookups, and a chunk that has the wiring
        // family always has the seams.  Only then -- a chunk without it keeps
        // the exact width list it had, and with it the digest moves anyway.
        if p.glue.is_some() {
            widths.push(15);
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
        // Stream II: the three requantize seams.  Each names the region whose
        // `Out(0)` it consumes, so the producer index comes from the plan and is
        // never counted by hand; the layout is the dequant's own `MatLayout`
        // map, which is what makes `output_claim_w`'s weights apply unchanged.
        let mut rqs = vec![];
        let mut rq_vars: Vec<crate::requant::SVariant> = vec![];
        if let Some(g) = glue.as_ref() {
            // stream II: on the head chunk the only producer is the final
            // norm's scale and the only consumer is the head's own walk, so the
            // seam list is one entry and the layer's three are absent -- there
            // is no layer here to have produced them.
            let seams: Vec<(&str, (usize, usize), usize)> = if g.head_norm {
                vec![("head", (g.norm_scales[0], 2), g.t.d)]
            } else {
                vec![
                    ("qkv", (g.norm_scales[0], 2), g.t.d),
                    ("gateup", (g.norm_scales[1], 2), g.t.d),
                    ("down", (g.swiglu.expect("the SwiGLU region is unconditional"), 2),
                     g.tff.expect("the SwiGLU's tiling").d),
                ]
            };
            for (name, producer, n) in seams {
                let Some(wi) = walks.iter().position(|w| w.name == name) else { continue };
                let m0 = &p.mats[walks[wi].mats[0]];
                assert_eq!(m0.n_in, n, "the {name} walk's input width is not its producer's");
                let v = if m0.variant == "pow2" {
                    crate::requant::SVariant::Pow2
                } else {
                    crate::requant::SVariant::Bf16
                };
                if !rq_vars.contains(&v) {
                    rq_vars.push(v);
                }
                rqs.push(RqPlan {
                    name: name.to_string(),
                    walk: wi,
                    producer,
                    n,
                    tokens: m0.tokens,
                    lay: MatLayout::choose(n, m0.tokens, p.deq_target),
                    v,
                });
            }
            rq_vars.sort_by_key(|v| *v == crate::requant::SVariant::Bf16);
        }
        // ---- stream NN: the three quantize regions ---------------------------
        //
        // `b1/model.py` quantizes q, k and v by scales that are per (head,
        // channel) and constant down the tokens -- the transpose of the
        // per-token scale `requant.rs` pins -- so the exponent is a preprocessed
        // per-slot column and the region is `rope::QuantEval`.  The tables are
        // `requant.rs`'s own, so the power-of-two division table has to be
        // registered whether or not a *seam* asked for it.
        let mut quants = vec![];
        if let (Some(g), Some(rp)) = (glue.as_ref(), p.rope.as_ref()) {
            assert!(p.corner,
                    "the q/k/v bindings read the corner band's own columns; --rope needs --corner");
            assert!(p.nh % rp.nkv == 0, "nh={} is not a multiple of nkv={}", p.nh, rp.nkv);
            assert_eq!(rp.inv(p.hd).len(), p.hd / 2, "inv_freq must be hd/2 long");
            for (i, (name, heads, bits)) in
                [("q", p.nh, p.q_bits), ("k", rp.nkv, p.k_bits),
                 ("v", rp.nkv, crate::rope::V_BITS)]
                    .into_iter().enumerate()
            {
                let mi = p.mats.iter().position(|m| m.name == name)
                    .unwrap_or_else(|| panic!("--rope needs the {name} matmul"));
                // Stream SS: a folded `v` has no quantize region, because the
                // fold *is* the quantize -- `y-delay-v` states
                // `round_hafz(acc s_x s_w / s_v)` in one rounding, so there is
                // no bf16 stream for `rope::QuantEval` to round and nothing for
                // it to prove.  The dequant region publishes the integer
                // instead, and the bands' own V is what reads it
                // (`Plan::output_claim_band`).
                if name == "v" && specs[mi].variant == DeqVariant::Fold {
                    continue;
                }
                let m = &p.mats[mi];
                assert_eq!(m.n_out, heads * p.hd,
                           "the {name} matmul is {} wide, not {heads} x {}", m.n_out, p.hd);
                quants.push(QuantPlan {
                    name,
                    mat: mi,
                    heads,
                    hd: p.hd,
                    rep: p.nh / heads,
                    n: m.n_out,
                    tokens: m.tokens,
                    lay: crate::rope::q_layout(m.n_out, p.hd, m.tokens, p.deq_target),
                    bits,
                    // q and k read a rotation's output; v reads its own dequant
                    producer: (i < 2).then(|| (g.ropes[i].add, 1)),
                });
            }
        }
        // Stream H: the boundary is the wiring family's two stream bindings, so it
        // exists exactly when the statement asks for it and the family is there.
        let layer_sides = match (&glue, p.layer_boundary) {
            (Some(g), true) => crate::boundary::sides(g),
            _ => vec![],
        };
        // Stream AA: the K/V sides follow stream H's, so the two features'
        // tree indices are independent of each other.  `boundary::kv_layout`
        // rederives exactly this arithmetic from `Fixed.params` alone, which is
        // how the pair check finds the roots without building a plan.
        let kv = if p.kv_boundary {
            let kb: Vec<crate::boundary::KvBand> = (0..shapes.len())
                .map(|b| crate::boundary::KvBand {
                    group: group[b],
                    corner: corner[b],
                    pk: lays[b].chan.pk,
                    pv: lays[b].key.pv,
                    chan_rows: lays[b].chan.g.rows,
                    key_rows: lays[b].key.g.rows,
                })
                .collect();
            crate::boundary::kv_sides(crate::boundary::FIRST_TREE + layer_sides.len(), &kb)
        } else {
            vec![]
        };
        // Stream BB (E11): the embedding gadget.  It bindings to the wiring
        // family's `stream_in` claim, so a statement that asks for it without
        // the wiring family is a loud configuration error rather than a proof
        // with nothing to compare against.
        let emb = (p.embed_vocab > 0).then(|| {
            let g = glue.as_ref()
                .expect("`--embed` bindings to the wiring family's stream input; it needs the glue");
            assert!(p.embed_per_row > 0, "`--embed` needs a channel-block target");
            crate::embed::EmbLayout::choose(g.t.d, p.s, p.embed_vocab, p.embed_per_row)
        });
        // stream NN: the corner bands publish the K/Q/V claims
        let nn_bands: Vec<usize> = if quants.is_empty() {
            vec![]
        } else {
            (0..shapes.len()).filter(|b| corner[*b]).collect()
        };
        // stream NN (NN-2): every prefix band publishes its own Q, against its
        // group's corner.  A group with no corner publishes nothing -- there is
        // nothing to compare against -- and `--corner` is already required by
        // the quantize regions, so that case does not arise on a rope chunk.
        let nn_qbands: Vec<usize> = if quants.is_empty() {
            vec![]
        } else {
            (0..shapes.len())
                .filter(|b| !corner[*b] && corner.iter().enumerate()
                        .any(|(c, is)| *is && group[c] == group[*b]))
                .collect()
        };
        let mut glue = glue;
        if !quants.is_empty() {
            let g = glue.as_mut().expect("the quantize regions need the wiring family");
            for (qi, q) in quants.iter().enumerate() {
                if q.producer.is_none() {
                    if let Some(t) = g.output_bindings.iter_mut().find(|t| t.name == q.name) {
                        t.quant = Some(qi);
                    }
                }
            }
        }
        // Stream SS: and the folded `v`'s counterparty, which is the bands' own
        // V.  Three conditions, each of which the band weights need: the bands
        // are in the statement (`nn_bands`, i.e. the rotation is on), the
        // rotation says how many query heads a kv head is replicated into, and
        // the region's rows sit inside one kv head so that the step is the two
        // constants `output_claim_w_band` derives.  Without them the fold still
        // publishes its integer and the verdict names it open.
        if let (Some(g), true) = (glue.as_mut(), p.rope.is_some()) {
            if let Some(mi) = p.mats.iter().position(|m| m.name == "v") {
                let pr = mat_lays[mi].per_row;
                let fits = mat_lays[mi].lead == 0 && pr <= p.hd && p.hd % pr == 0;
                if specs[mi].variant == DeqVariant::Fold && !nn_bands.is_empty() && fits {
                    if let Some(t) = g.output_bindings.iter_mut().find(|t| t.name == "v") {
                        t.band = true;
                    }
                }
            }
        }
        // stream NN-3: the tables the quantize regions own, deduplicated by
        // width.  One per distinct side width, so `q` at 12 bits and `k` at 11
        // get their own and `v`'s int8 side keeps the eight-bit window; the
        // shared `(q, nz)` table is the widest of them.
        let q_tabs = {
            let mut w: Vec<u32> = quants.iter().map(|q| q.bits).collect();
            w.sort_unstable();
            w.dedup();
            w
        };
        let q_nz_bits = q_tabs.last().copied().unwrap_or(0);
        Self { p: p.clone(), attn_variant, round, shapes, lays, cfg, head_offsets, corner, specs,
               mat_lays, ao_lay, walks, rqs, rq_vars, quants, q_tabs, q_nz_bits,
               nn_bands, nn_qbands,
               widths, log_norm,
               log_frac, log_mul,
               glue, layer_sides,
               kv, group, rows_band, emb }
    }

    /// How many trees the transcript commits: stream PP's two preprocessed
    /// trees (shape, then model), the base trace and the interaction trace,
    /// plus one per boundary side.
    pub fn n_trees(&self) -> usize {
        4 + self.layer_sides.len() + self.kv.len()
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
            let cp = Powers::new(r_corner, s, rv, sh, &l.chan, &l.key, &l.out, l.rw);
            // A corner that carries the region is block 0's solo corner and
            // needs its own `mu`; a paired one rides on the prefix band's.
            if self.rows_band[b] {
                cp.as_solo_corner(pre0.r_c, sh.c)
            } else {
                cp.as_corner(pre0.r_c, sh.c)
            }
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
    /// Stream AA: with `--kv-boundary` every band is `c = S`, so the same Horner
    /// produces `r^S` for free, and the global weight moves with it.  That is
    /// what keeps design A free of any new AIR mechanism -- a prefix sub-band
    /// needs no `iend`/`kappa` conversion, its natural instance stride *is* the
    /// global one, and it keeps `KEY_SEL` and CC's frozen per-slot content.
    ///
    /// Block 0 (open item 1) has `c = 0` and no prefix band: its only band is
    /// the solo corner, whose key Horner walks `S` keys, so the weight is `r^S`
    /// with the flag off too.  `r^0 = 1` here left the attention output region
    /// and the verifier's fold on a different weight from the band's own
    /// `Powers::r_c`, and `den_link_ok` was the one verdict field that failed.
    pub fn head_weight_c(&self) -> usize {
        if self.p.kv_boundary || self.p.c == 0 { self.p.s } else { self.p.c }
    }

    /// How many secure felts the transcript draws after the relations, and where
    /// the corner's key challenge sits.  It goes *last* so a proof without the
    /// corner draws exactly what it drew before.
    /// Stream E: the binding challenge pair is drawn for the wiring family *or* for
    /// a head-key claim on its own, since a head dump has no wiring columns.
    pub fn has_bindings(&self) -> bool {
        self.glue.is_some() || self.p.logits_claim
    }

    /// Stream E: the index of the matmul the argmax region is bound to, when it
    /// is composed.  Its layout is that matmul's, so the binding is positionwise.
    pub fn argmax_mat(&self) -> Option<usize> {
        self.p.prove_argmax
            .then(|| self.p.mats.iter().position(|m| m.name == "lm_head"))
            .flatten()
    }
    pub fn n_challenges(&self) -> usize {
        3 + if self.has_bindings() { 2 } else { 0 } + if self.p.corner { 1 } else { 0 }
    }
    pub fn corner_challenge_at(&self) -> usize {
        3 + if self.has_bindings() { 2 } else { 0 }
    }

    /// Stream E: does this matmul's dequant region publish an output-side claim?
    /// Both sides ask the plan, so the preprocessed list, the region, the AIR and
    /// the verifier cannot disagree about it.
    pub fn has_output_claim(&self, name: &str) -> bool {
        self.glue.as_ref().is_some_and(|g| g.output_bindings.iter().any(|t| t.name == name))
            || (self.p.logits_claim && name == "lm_head")
    }

    /// What that output side accumulates: the head publishes the argmax's order
    /// key, everything else the glue's `(sm, ef)` pair.
    /// Stream NN: which convention this matmul's output side folds under.
    /// `q`, `k` and `v` take the token-major one exactly when the rotation is
    /// in the statement -- without it their claims have no counterparty and
    /// keep the weights every recorded digest was taken with.
    pub fn output_claim_tok(&self, name: &str) -> bool {
        self.glue.as_ref().is_some_and(|g| !g.ropes.is_empty())
            && matches!(name, "q" | "k" | "v")
    }
    pub fn output_claim_of(&self, w: SecureField, lam: SecureField, name: &str, tokens: usize,
                    n_out: usize, per_row: usize) -> crate::layer::OutputClaimW {
        if self.output_claim_band(name) {
            let (nh, hd) = (self.p.nh, self.p.hd);
            let nkv = self.p.rope.as_ref().map(|r| r.nkv).unwrap_or(nh);
            return output_claim_w_band(w, lam, tokens, nh, hd, nh / nkv, per_row);
        }
        let f = if self.output_claim_tok(name) { output_claim_w_tok } else { output_claim_w };
        f(w, lam, tokens, n_out, per_row, self.output_claim_kind(name))
    }

    /// Stream SS: is this matmul's dequant the delayed requantize?  The fold is
    /// the one variant whose output is an integer rather than a bf16 pattern,
    /// and every side that reads an output has to ask.
    pub fn is_fold(&self, name: &str) -> bool {
        self.p.mats.iter().position(|m| m.name == name)
            .is_some_and(|mi| self.specs[mi].variant == DeqVariant::Fold)
    }

    /// Stream SS: does this matmul's output side fold under the band grid --
    /// i.e. is it a folded `v` whose bands are in the statement and whose layout
    /// the band weights fit?  When the answer is no, a folded `v` still publishes
    /// its integer, on the producer grid, with nothing to read it against
    /// (`deq_out_open`).
    pub fn output_claim_band(&self, name: &str) -> bool {
        self.glue.as_ref()
            .is_some_and(|g| g.output_bindings.iter().any(|t| t.name == name && t.band))
    }

    pub fn output_claim_kind(&self, name: &str) -> crate::dequant::OKind {
        if name == "lm_head" {
            crate::dequant::OKind::Key
        } else if self.is_fold(name) {
            // stream SS: the fold has no bf16 output to publish
            crate::dequant::OKind::Int
        } else {
            crate::dequant::OKind::Pair
        }
    }

    /// Stream LL (E11), `notes/witness-slack.md` section 5 item 1: the same list
    /// `pre_ids` produces, with what each column needs in order to be rebuilt and
    /// whether it is the same column on every token block of a layer.
    ///
    /// Two axes, because the two questions the item asks are not the same one:
    ///
    /// * `needs` -- `Params` means `Fixed.params` determines every cell, so a
    ///   verifier holding the statement can rebuild the column. `Model` means it
    ///   cannot: the column holds a weight, a weight scale, or a per-layer
    ///   calibration constant that lives only in the dump's manifest
    ///   (`b6_k_dinv` / `b6_x_dinv`, `b6_exp_y`).
    /// * `varies` -- `Layer` means the column is identical on every block of the
    ///   layer, so one published constant per layer covers a whole grid; `Block`
    ///   means its height or its contents move with the chunk's own `c`.
    ///
    /// `pre_split_sizes` is asserted against the committed tree in `compose`, so
    /// this list and tree 0 cannot drift apart.
    pub fn pre_split(&self) -> Vec<PreCol> {
        let mut v: Vec<PreCol> = vec![];
        let mut put = |id: PreProcessedColumnId, family: &str, log_size: u32,
                       needs: PreNeeds, varies: PreVaries| {
            v.push(PreCol { id, family: family.to_string(), log_size, needs, varies });
        };
        for b in 0..self.shapes.len() {
            let (lc, lk) = (&self.lays[b].chan, &self.lays[b].key);
            for n in CHAN_SEL {
                put(band_id(b, n), "band/sel", lc.g.log_rows, PreNeeds::Params,
                    PreVaries::Block);
            }
            if self.corner[b] {
                for n in CORNER_SEL {
                    // the eighth is `dinv`, the per-instance master-table scale:
                    // a model constant off the dump's `frac_per_head`, and the
                    // one column of a band that is not a function of the shape
                    let (fam, needs) = if n == CORNER_SEL[7] {
                        // stream PP: `dinv` is `2^-(F - frac_i)`, a function of
                        // the manifest's `frac_per_head` and of no weight -- so
                        // it is public and stays in the shape tree.  It is the
                        // only `Block`-varying column that was ever `Model`,
                        // and moving it is what makes the model tree a
                        // per-layer constant.
                        ("band/dinv", PreNeeds::Manifest)
                    } else {
                        ("band/sel", PreNeeds::Params)
                    };
                    put(band_id(b, n), fam, lk.g.log_rows, needs, PreVaries::Block);
                }
                for i in 0..self.lays[b].key.q {
                    put(msk_id(b, i), "band/mask", lk.g.log_rows, PreNeeds::Params,
                        PreVaries::Block);
                }
            } else {
                for n in KEY_SEL {
                    let (fam, needs) = if n == KEY_SEL[7] {
                        ("band/dinv", PreNeeds::Manifest)
                    } else {
                        ("band/sel", PreNeeds::Params)
                    };
                    put(band_id(b, n), fam, lk.g.log_rows, needs, PreVaries::Block);
                }
            }
            if let Some(a) = crate::attn_air::act_id(b, self.corner[b], lk.g.lead) {
                put(a, "band/sel", lk.g.log_rows, PreNeeds::Params, PreVaries::Block);
            }
            if self.rows_band[b] {
                for n in ROW_SEL {
                    put(band_id(b, n), "band/sel", self.lays[b].log_row,
                        PreNeeds::Params, PreVaries::Block);
                }
            }
        }
        for s in 0..self.ao_lay.per_row {
            put(ao_chan("jv", s), "attnout/jv", self.ao_lay.log_rows, PreNeeds::Model,
                PreVaries::Layer);
        }
        for n in AO_SEL {
            put(ao_sel(n), "attnout/sel", self.ao_lay.log_rows, PreNeeds::Params,
                PreVaries::Layer);
        }
        for ((m, sp), lay) in self.p.mats.iter().zip(&self.specs).zip(&self.mat_lays) {
            for s in 0..sp.per_row {
                let names: &[&str] = match sp.variant {
                    DeqVariant::Pow2 => &["j"],
                    DeqVariant::Chain => &["m", "e"],
                    DeqVariant::Fold => &["m", "p"],
                    _ => unreachable!(),
                };
                for nm in names {
                    put(chan_col(&m.name, nm, s), "dequant/scale", lay.log_rows,
                        PreNeeds::Model, PreVaries::Layer);
                }
            }
            for n in DEQ_SEL {
                put(deq_sel(&m.name, n), "dequant/sel", lay.log_rows, PreNeeds::Params,
                    PreVaries::Layer);
            }
            if self.has_output_claim(&m.name) {
                for n in crate::dequant::DEQ_OSEL {
                    put(deq_sel(&m.name, n), "dequant/sel", lay.log_rows,
                        PreNeeds::Params, PreVaries::Layer);
                }
            }
        }
        for w in &self.walks {
            for (mi, _) in w.mats.iter().enumerate() {
                for pp in 0..w.lay.pk {
                    put(walk_w(&w.name, mi, pp), "walk/weight", w.lay.g.log_rows,
                        PreNeeds::Model, PreVaries::Layer);
                }
            }
            for n in WALK_SEL {
                put(walk_sel(&w.name, n), "walk/sel", w.lay.g.log_rows, PreNeeds::Params,
                    PreVaries::Layer);
            }
        }
        if let Some(g) = &self.glue {
            // Stream NN: a rotation's Mul region carries the angle table, which
            // is a pure function of `Fixed.params` -- `theta`, the rope branch
            // and that branch's own constants -- and of the block, because
            // `RopeParams::pos0` is this chunk's absolute position offset.  So
            // it is `Params`/`Block`, never `Model`.  Every other region's
            // `Src::Pre` pair is a norm weight off the dump.
            let rope_mul: std::collections::BTreeSet<usize> =
                g.ropes.iter().map(|r| r.mul).collect();
            for (ri, r) in g.regions.iter().enumerate() {
                put(crate::glue::glue_sel(&r.tag, "nofirst"), "glue/sel", r.log_rows,
                    PreNeeds::Params, PreVaries::Layer);
                put(crate::glue::glue_sel(&r.tag, "end"), "glue/sel", r.log_rows,
                    PreNeeds::Params, PreVaries::Layer);
                let (fam, needs, varies) = if rope_mul.contains(&ri) {
                    ("rope/angle", PreNeeds::Params, PreVaries::Block)
                } else {
                    ("glue/normw", PreNeeds::Model, PreVaries::Layer)
                };
                for i in 0..r.prog.n_pre {
                    for k in 0..2 {
                        put(crate::glue::glue_pre_col(&r.tag, i, k), fam,
                            r.log_rows, needs, varies);
                    }
                }
            }
        }
        // the master exp table: the index column is `0..2^log_table`, the value
        // column is this layer's own table and its exponent is the dump's
        put(id(TBL_EXP_X), "table/exp", self.p.log_table, PreNeeds::Params,
            PreVaries::Layer);
        // stream PP: the value column is `table_of(1 << log_table, nmax, frac)`
        // -- a closed form of the manifest's `frac_per_head` and the statement's
        // table width, and of no weight.  It is the only column whose height is
        // `1 << log_table`, so moving it to the shape tree is what frees the
        // model root of the table width and retires design (a)'s per-layer
        // `--table-bits` map (log row 75).
        put(id(TBL_EXP_Y), "table/exp", self.p.log_table, PreNeeds::Manifest,
            PreVaries::Layer);
        for c in NORM_COLS {
            put(id(c), "table/norm", self.log_norm, PreNeeds::Params, PreVaries::Layer);
        }
        for c in FRAC_COLS {
            put(id(c), "table/frac", self.log_frac, PreNeeds::Params, PreVaries::Layer);
        }
        for c in MUL_COLS {
            put(id(c), "table/mul", self.log_mul, PreNeeds::Params, PreVaries::Layer);
        }
        for c in QNZ_COLS {
            put(id(c), "table/qnz", self.p.ao_q_bits, PreNeeds::Params, PreVaries::Layer);
        }
        for w in &self.widths {
            put(tbl_range(*w), "table/range", *w, PreNeeds::Params, PreVaries::Layer);
        }
        for c in EPOW_COLS {
            put(id(c), "table/epow", self.log_epow(), PreNeeds::Params, PreVaries::Layer);
        }
        if let Some(g) = &self.glue {
            // Stream PP (E11): the four heights, and not the tables.  Building
            // the tables here cost a `2^(kmax+1)`-row `frac` on every call and
            // asserted `kmax >= DMAX + 1`, which a plan that carries a glue
            // gadget it never proves need not satisfy.
            let t = crate::glue::GlueTables::log_sizes(self.round);
            let mut logs: Vec<u32> = vec![t.pow; crate::glue::POW_COLS.len()];
            logs.extend(vec![t.smul; crate::glue::SMUL_COLS.len()]);
            logs.extend(vec![t.val; crate::glue::VAL_COLS.len()]);
            logs.extend(vec![t.un; 4 * g.uns.len()]);
            for (gid, ls) in crate::glue::glue_table_ids(&g.uns).into_iter().zip(logs) {
                put(gid, "table/glue", ls, PreNeeds::Params, PreVaries::Layer);
            }
        }
        for r in &self.rqs {
            for n in crate::requant::RQ_SEL {
                put(crate::requant::rq_sel(&r.name, n), "requant/sel", r.lay.log_rows,
                    PreNeeds::Params, PreVaries::Layer);
            }
        }
        // Stream NN: the q/k/v quantize regions, in `Plan::pre_ids`' order --
        // three selectors off the shape, then one scale exponent per slot.  The
        // exponents are B1's own `s_q`/`s_k`/`s_v`, model constants of exactly
        // the kind the attention output's `jv` already was.
        for q in &self.quants {
            for n in crate::rope::Q_SEL {
                put(crate::rope::q_sel(q.name, n), "quant/sel", q.lay.log_rows,
                    PreNeeds::Params, PreVaries::Layer);
            }
            for sl in 0..q.lay.per_row {
                put(crate::rope::q_scale_col(q.name, sl), "quant/scale", q.lay.log_rows,
                    PreNeeds::Model, PreVaries::Layer);
            }
        }
        for sv in &self.rq_vars {
            for k in 0..crate::requant::rq_tab_width(*sv) {
                put(crate::requant::rq_tab_col(*sv, k), "table/requant",
                    crate::requant::rq_log_rows(*sv), PreNeeds::Params, PreVaries::Layer);
            }
            for k in 0..crate::requant::scl_tab_width(*sv) {
                put(crate::requant::scl_tab_col(*sv, k), "table/requant", 15,
                    PreNeeds::Params, PreVaries::Layer);
            }
        }
        if self.needs_rq_rels() {
            for c in crate::requant::RQV_COLS {
                put(id(c), "table/requant", 9, PreNeeds::Params, PreVaries::Layer);
            }
        }
        // stream NN-3: the quantize family's own tables.  Both are a closed
        // form of the side's width -- `Params::q_bits` and `Params::k_bits` --
        // so they need the statement's parameters and no model data at all.
        for w in &self.q_tabs {
            for k in 0..crate::rope::QDIV_COLS {
                put(crate::rope::qdiv_col(*w, k), "table/quant",
                    crate::rope::qdiv_log_rows(*w), PreNeeds::Params, PreVaries::Layer);
            }
        }
        if !self.quants.is_empty() {
            for k in 0..2 {
                put(crate::rope::qnz_col(self.q_nz_bits, k), "table/quant",
                    crate::rope::qnz_log_rows(self.q_nz_bits), PreNeeds::Params,
                    PreVaries::Layer);
            }
        }
        if let Some(mi) = self.argmax_mat() {
            for n in crate::argmax::AM_SEL {
                put(crate::argmax::id(n), "argmax/sel", self.mat_lays[mi].log_rows,
                    PreNeeds::Params, PreVaries::Layer);
            }
        }
        if let Some(l) = &self.emb {
            for n in crate::embed::EMB_SEL {
                put(crate::embed::id(n), "embed/sel", l.log_use, PreNeeds::Params,
                    PreVaries::Layer);
            }
            put(crate::embed::cb_id(), "embed/sel", l.log_use, PreNeeds::Params,
                PreVaries::Layer);
            put(crate::embed::tab_v_id(), "embed/table", l.log_tab, PreNeeds::Params,
                PreVaries::Layer);
            put(crate::embed::tab_cb_id(), "embed/table", l.log_tab, PreNeeds::Params,
                PreVaries::Layer);
            for k in 0..l.pe {
                put(crate::embed::tab_e_id(k), "embed/table", l.log_tab, PreNeeds::Model,
                    PreVaries::Layer);
            }
        }
        if self.p.gkr_bands {
            for h in crate::gkrlink::link_heights(self) {
                put(crate::gkrlink::first_id(h), "gkrlink/sel", h, PreNeeds::Params,
                    PreVaries::Block);
                put(crate::gkrlink::second_id(h), "gkrlink/sel", h, PreNeeds::Params,
                    PreVaries::Block);
            }
        }
        v
    }

    /// The log sizes `pre_split` says the two preprocessed trees have, in
    /// commit order: the shape tree first, then the model tree.
    pub fn pre_split_sizes(&self) -> Vec<u32> {
        let cols = self.pre_split();
        cols.iter()
            .filter(|c| c.needs.tree() == PreTree::Shape)
            .chain(cols.iter().filter(|c| c.needs.tree() == PreTree::Model))
            .map(|c| c.log_size)
            .collect()
    }

    /// Stream PP (E11): the shape tree's log sizes, in commit order.
    pub fn shape_sizes(&self) -> Vec<u32> {
        self.pre_split()
            .iter()
            .filter(|c| c.needs.tree() == PreTree::Shape)
            .map(|c| c.log_size)
            .collect()
    }

    /// Stream PP (E11): the model tree's log sizes, in commit order.
    pub fn model_sizes(&self) -> Vec<u32> {
        self.pre_split()
            .iter()
            .filter(|c| c.needs.tree() == PreTree::Model)
            .map(|c| c.log_size)
            .collect()
    }

    /// Stream PP (E11): `true` for every column of `pre_ids` that the model
    /// tree commits, in `pre_ids`' own order -- the mask the commit path uses
    /// to split one flat column list into the two trees.
    pub fn pre_is_model(&self) -> Vec<bool> {
        self.pre_split()
            .iter()
            .map(|c| c.needs.tree() == PreTree::Model)
            .collect()
    }

    /// Stream PP (E11): the shape tree's ids and the model tree's, each in its
    /// own tree's commit order.  The model tree's root is the per-layer model
    /// commitment, so every column in it must be identical on every block of
    /// the layer -- asserted here rather than argued.
    pub fn pre_ids_split(&self) -> (Vec<PreProcessedColumnId>, Vec<PreProcessedColumnId>) {
        let cols = self.pre_split();
        let ids = self.pre_ids();
        assert_eq!(cols.len(), ids.len(),
                   "pre_split and pre_ids disagree on how many preprocessed columns there are");
        let mut shape = vec![];
        let mut model = vec![];
        for (c, id) in cols.iter().zip(ids) {
            assert_eq!(c.id, id, "pre_split and pre_ids disagree on the column order");
            match c.needs.tree() {
                PreTree::Shape => shape.push(id),
                PreTree::Model => {
                    assert!(c.varies == PreVaries::Layer,
                            "the model tree carries {:?}, which varies with the block -- its \
                             root would not be a per-layer constant", c.id);
                    model.push(id);
                }
            }
        }
        (shape, model)
    }

    /// The rollup stream LL's record quotes: per family, and per the two axes.
    pub fn pre_split_report(&self) -> Value {
        let cols = self.pre_split();
        let mut fam: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        let (mut model_c, mut model_n) = (0usize, 0usize);
        let (mut man_c, mut man_n) = (0usize, 0usize);
        let (mut block_c, mut block_n) = (0usize, 0usize);
        for c in &cols {
            let cells = 1usize << c.log_size;
            let e = fam.entry(c.family.clone()).or_insert((0, 0));
            e.0 += 1;
            e.1 += cells;
            if c.needs == PreNeeds::Manifest {
                man_c += 1;
                man_n += cells;
            }
            if c.needs == PreNeeds::Model {
                model_c += 1;
                model_n += cells;
            }
            if c.varies == PreVaries::Block {
                block_c += 1;
                block_n += cells;
            }
        }
        let total: usize = cols.iter().map(|c| 1usize << c.log_size).sum();
        let sh = |n: usize| n as f64 / total as f64;
        json!({
            "columns": cols.len(),
            "cells": total,
            "families": fam.iter().map(|(k, (c, n))| json!({
                "family": k, "columns": c, "cells": n, "share": sh(*n),
            })).collect::<Vec<_>>(),
            // what a verifier cannot rebuild from `Fixed.params` alone
            "needs_model": {"columns": model_c, "cells": model_n, "share": sh(model_n)},
            "needs_params": {"columns": cols.len() - model_c, "cells": total - model_n,
                             "share": sh(total - model_n)},
            // stream PP: the per-layer calibration the model manifest publishes
            // -- public input, so the verifier rebuilds these itself
            "needs_manifest": {"columns": man_c, "cells": man_n, "share": sh(man_n)},
            // stream PP: the two committed preprocessed trees
            "model_tree": {"columns": model_c, "cells": model_n, "share": sh(model_n)},
            "shape_tree": {"columns": cols.len() - model_c, "cells": total - model_n,
                           "share": sh(total - model_n)},
            // what moves down the block axis: what a per-layer published root
            // could not cover, and so what the verifier must rebuild per chunk
            "varies_block": {"columns": block_c, "cells": block_n, "share": sh(block_n)},
            "varies_layer": {"columns": cols.len() - block_c, "cells": total - block_n,
                             "share": sh(total - block_n)},
        })
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
            // region", not "is it a prefix band".  With `--kv-boundary` off the two
            // are the same predicate on the same bands, so this id sequence is
            // the one tree 0 always had.
            if self.rows_band[b] {
                ids.extend(ROW_SEL.iter().map(|n| band_id(b, n)));
            }
        }
        for s in 0..self.ao_lay.per_row {
            ids.push(ao_chan("jv", s));
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
            if self.has_output_claim(&m.name) {
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
                // stream II: then the region's own preprocessed values, in the
                // order `glue_region` appends their columns
                for i in 0..r.prog.n_pre {
                    ids.push(crate::glue::glue_pre_col(&r.tag, i, 0));
                    ids.push(crate::glue::glue_pre_col(&r.tag, i, 1));
                }
            }
        }
        ids.push(id(TBL_EXP_X));
        ids.push(id(TBL_EXP_Y));
        ids.extend(NORM_COLS.iter().map(|c| id(*c)));
        ids.extend(FRAC_COLS.iter().map(|c| id(*c)));
        ids.extend(MUL_COLS.iter().map(|c| id(*c)));
        ids.extend(QNZ_COLS.iter().map(|c| id(*c)));
        ids.extend(self.widths.iter().map(|w| tbl_range(*w)));
        // Stream JJ (E11): the exponent window, last of the shared tables so
        // that no table before it moves an index.
        ids.extend(EPOW_COLS.iter().map(|c| id(*c)));
        if let Some(g) = &self.glue {
            ids.extend(crate::glue::glue_table_ids(&g.uns));
        }
        // stream II: the requantize seams register before the two ends -- four selectors per
        // seam, then one division and one scale table per variant, then the
        // 257-row sign table.  Strictly last, so no family that exists today
        // moves an index.
        for r in &self.rqs {
            ids.extend(crate::requant::RQ_SEL.iter()
                       .map(|n| crate::requant::rq_sel(&r.name, n)));
        }
        // stream NN: and the quantize regions -- three selectors, then one
        // preprocessed scale exponent per slot, in `rope::q_region`'s own order
        for q in &self.quants {
            ids.extend(crate::rope::Q_SEL.iter().map(|n| crate::rope::q_sel(q.name, n)));
            ids.extend((0..q.lay.per_row).map(|sl| crate::rope::q_scale_col(q.name, sl)));
        }
        for v in &self.rq_vars {
            ids.extend((0..crate::requant::rq_tab_width(*v))
                       .map(|k| crate::requant::rq_tab_col(*v, k)));
            ids.extend((0..crate::requant::scl_tab_width(*v))
                       .map(|k| crate::requant::scl_tab_col(*v, k)));
        }
        if self.needs_rq_rels() {
            ids.extend(crate::requant::RQV_COLS.iter().map(|c| id(*c)));
        }
        // stream NN-3: the quantize family's own tables, after the seams' and
        // the sign table -- one division table per side width, then the shared
        // `(q, nz)` pair table
        for w in &self.q_tabs {
            ids.extend((0..crate::rope::QDIV_COLS).map(|k| crate::rope::qdiv_col(*w, k)));
        }
        if !self.quants.is_empty() {
            ids.extend((0..2).map(|k| crate::rope::qnz_col(self.q_nz_bits, k)));
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
            // region, two for one that rides on its group's -- with `--kv-boundary`
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
            // stream II: one per requantize seam, before the two ends
            + self.rqs.len()
            // stream NN: and one per quantize region
            + self.quants.len()
            + usize::from(self.argmax_mat().is_some())
            // stream BB: the embedding gadget's use region, last of all
            + usize::from(self.emb.is_some())
    }
    /// Stream JJ (E11): the `EPOW` window's domain and height.  Derived from
    /// `Params` alone -- both sides compute them, nothing commits them.
    pub fn e_window(&self) -> (i32, i32) {
        (self.p.ao_jv_min - self.p.ao_jo_max, self.p.ao_jv_max - self.p.ao_jo_min)
    }
    pub fn log_epow(&self) -> u32 {
        let (lo, hi) = self.e_window();
        crate::common::log_rows_for((hi - lo + 1) as usize, 1)
    }
    /// Stream II: where the requantize seams' tables start, and where the
    /// embedding's table sum sits.  Written out rather than read off the end of
    /// the list, because the seams register *before* the two ends -- stream
    /// BB's `registration_is_append_only` says the embedding's ids and its two
    /// components are the last thing any plan adds, and a family appended past
    /// it would make that false.  The 6 is stream JJ's `EPOW` counted in.
    /// Stream II's five relations are drawn for the requantize seams *and* for
    /// stream NN's quantize regions, which read the sign table and (before
    /// NN-3) the division table.  One predicate, so the prover, the emitter and
    /// the probes cannot disagree about whether they exist.
    pub fn needs_rq_rels(&self) -> bool {
        !self.rqs.is_empty() || !self.quants.is_empty()
    }
    /// Stream NN-3: which of `q_tabs` a side of this width reads.
    pub fn q_tab_at(&self, bits: u32) -> usize {
        self.q_tabs.iter().position(|w| *w == bits)
            .unwrap_or_else(|| panic!("no quantize division table at {bits} bits"))
    }
    pub fn rq_table_at(&self) -> usize {
        6 + self.widths.len() + self.glue.as_ref().map(|g| g.n_tables()).unwrap_or(0)
    }
    pub fn n_rq_tables(&self) -> usize {
        2 * self.rq_vars.len()
            + usize::from(self.needs_rq_rels())
            // stream NN-3: one division table per side width, then the quantize
            // family's own `(q, nz)` table -- after the seams' tables and the
            // sign table, so nothing before them moves an index
            + self.q_tabs.len()
            + usize::from(!self.quants.is_empty())
    }
    pub fn emb_table_at(&self) -> usize {
        self.rq_table_at() + self.n_rq_tables()
    }
    pub fn n_table_sums(&self) -> usize {
        6 + self.widths.len() + self.glue.as_ref().map(|g| g.n_tables()).unwrap_or(0)
            // stream II: one division and one scale table per variant, then the
            // sign table, before the two ends
            + self.n_rq_tables()
            // stream BB: the embedding table's own side, appended last
            + usize::from(self.emb.is_some())
    }
}

// -------------------------------------------------- components (both sides) ----

pub struct BuiltComponents {
    pub chans: Vec<FrameworkComponent<ChanEval>>,
    pub keys: Vec<FrameworkComponent<KeyEval>>,
    pub rows: Vec<FrameworkComponent<RowEval>>,
    /// B11 stream S, lever 3: one univariate IOP per link height, and the binding
    /// component that pins its columns to the base columns at OODS. Empty and
    /// `None` off the GKR arm (and on the emit and probe paths, which have no
    /// artifact to derive them from).
    pub links: Vec<FrameworkComponent<crate::gkrlink::LinkEval>>,
    pub link_binding: Option<crate::gkrlink::LinkTie>,
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
    /// stream II: the requantize seams and their tables, after everything else
    pub rqs: Vec<FrameworkComponent<crate::requant::RequantEval>>,
    /// stream NN: the q/k/v quantize regions, in `Plan::quants` order
    pub quants: Vec<FrameworkComponent<crate::rope::QuantEval>>,
    pub rq_tabs: Vec<FrameworkComponent<TableEval>>,
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
        // stream II: and the requantize seams'.  Every dispatched component has
        // to be here: the blob cache is keyed by the evaluator's address and the
        // pipelined loop reuses addresses across chunks, so a component that is
        // dispatched but never prewarmed can only read what another family left.
        for c in &self.rqs {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.quants {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        for c in &self.rq_tabs {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        if let Some(c) = &self.emb_tab {
            let e = &c.eval;
            tasks.push(Box::new(move || crate::cudagen::prewarm(e)));
        }
        tasks.par_iter().for_each(|t| t());
    }

    /// Stream EE (open item 21): the fingerprint every dispatched component
    /// will look its kernel up by, in `cudaemit`'s own component order and
    /// under `cudaemit`'s own names, so the two reports compare line for line.
    ///
    /// It is the same lowering the dispatch does (`cudagen::lowered_ir`) over
    /// the same evals -- this chunk's real claims and challenges included --
    /// so it answers "what will this chunk ask the store for", not "what
    /// should it ask for".
    pub fn fp_rows(&self) -> Vec<Value> {
        let mut v = vec![];
        for i in 0..self.chans.len() {
            v.push(crate::cudagen::fp_row(&self.chans[i].eval, &format!("chan_band{i}")));
            v.push(crate::cudagen::fp_row(&self.keys[i].eval, &format!("key_band{i}")));
        }
        for i in 0..self.rows.len() {
            v.push(crate::cudagen::fp_row(&self.rows[i].eval, &format!("row_band{i}")));
        }
        for i in 0..self.links.len() {
            v.push(crate::cudagen::fp_row(&self.links[i].eval, &format!("link{i}")));
        }
        v.push(crate::cudagen::fp_row(&self.ao.eval, "attnout"));
        for i in 0..self.mats.len() {
            v.push(crate::cudagen::fp_row(&self.mats[i].eval, &format!("dequant{i}")));
        }
        for i in 0..self.glue.len() {
            v.push(crate::cudagen::fp_row(&self.glue[i].eval, &format!("glue{i}")));
        }
        for i in 0..self.gtables.len() {
            v.push(crate::cudagen::fp_row(&self.gtables[i].eval, &format!("gtable{i}")));
        }
        for i in 0..self.walks.len() {
            v.push(crate::cudagen::fp_row(&self.walks[i].eval, &format!("walk{i}")));
        }
        v.push(crate::cudagen::fp_row(&self.exp_t.eval, "exp_table"));
        for i in 0..self.tables.len() {
            v.push(crate::cudagen::fp_row(&self.tables[i].eval, &format!("table{i}")));
        }
        for i in 0..self.rqs.len() {
            v.push(crate::cudagen::fp_row(&self.rqs[i].eval, &format!("requant{i}")));
        }
        for i in 0..self.quants.len() {
            v.push(crate::cudagen::fp_row(&self.quants[i].eval, &format!("quant{i}")));
        }
        for i in 0..self.rq_tabs.len() {
            v.push(crate::cudagen::fp_row(&self.rq_tabs[i].eval, &format!("rqtable{i}")));
        }
        if let Some(c) = &self.argmax {
            v.push(crate::cudagen::fp_row(&c.eval, "argmax"));
        }
        if let Some(c) = &self.embed {
            v.push(crate::cudagen::fp_row(&c.eval, "embed"));
        }
        if let Some(c) = &self.emb_tab {
            v.push(crate::cudagen::fp_row(&c.eval, "emb_table"));
        }
        v
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
        if let Some(t) = &self.link_binding {
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
        // stream II: the requantize seams and their tables, before the two ends,
        // as `Plan::pre_ids` has them
        for r in &self.rqs {
            v.push(r);
        }
        // stream NN: the quantize regions sit between the seams and their shared
        // tables, which is where `Plan::pre_ids` puts their selectors too
        for r in &self.quants {
            v.push(r);
        }
        for t in &self.rq_tabs {
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
        if let Some(t) = &self.link_binding {
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
        // stream II: the requantize seams and their tables, before the two ends,
        // as `Plan::pre_ids` has them
        for r in &self.rqs {
            v.push(r);
        }
        // stream NN: the quantize regions sit between the seams and their shared
        // tables, which is where `Plan::pre_ids` puts their selectors too
        for r in &self.quants {
            v.push(r);
        }
        for t in &self.rq_tabs {
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
    // Stream PP (E11): tree 0's ids and tree 1's, each in its own tree's commit
    // order.  The AIR still asks for a preprocessed column by id and never
    // names a tree; the allocator is what decides which of the two carries it.
    let (shape_ids, model_ids) = plan.pre_ids_split();
    let alloc =
        &mut TraceLocationAllocator::new_with_split_preprocessed_columns(&shape_ids, &model_ids);
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
        // stream NN: the corner bands' own K, Q and V claims
        let nn = nn_band_bindings(plan, b, w_glue, claims);
        // stream NN (NN-2): a prefix band's Q claim, against its group's corner
        let nn_q = nn_band_qtie(plan, b, w_glue, claims);
        // B11 stream S, lever 1: on the GKR arm a band component carries no
        // LogUp, so there is no claimed sum to hand it and none to consume.
        chans.push(FrameworkComponent::<ChanEval>::new(
            alloc,
            ChanEval { tag: b, lay: l.chan, pow: pow.clone(), off, claimed: unqm(bc[0]),
                       rel_k: rels.range(p.k_bits).clone(), rel_q: rels.range(p.q_bits).clone(),
                       range_check: true, gkr: p.gkr_bands,
                       // stream AA: empty unless `--kv-boundary`, so the IR, the
                       // fingerprint and the proof are unmoved with it off
                       boundary_k: crate::boundary::kv_boundaries_of(&plan.kv, b, l.chan.pk),
                       qk_bind: nn.as_ref().map(|(k, q, _)| (k.clone(), q.clone())),
                       q_bind: nn_q.clone() },
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
                      boundary_v: crate::boundary::kv_boundaries_of(&plan.kv, b, l.key.pv),
                      v_bind: nn.as_ref().map(|(_, _, v)| v.clone()) },
            if p.gkr_bands { SecureField::zero() } else { next_use(claims) },
        ));
        // stream AA: "carries the per-token region", which with `--kv-boundary` off
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
    // B11 stream U, lever 3: the *binding* is built lower down, after the dequant
    // components exist, because a dequant region's folded weights name columns
    // in that region's own span. The allocation order -- what the statement
    // depends on -- is unchanged; only where the binding reads the spans back moves.
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
    // under `--kv-boundary` -- see `Plan::head_weight_c`
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
            claimed_q: claim_at(claims.ao_q.first().copied(), "ao_q"),
            qnz: rels.qnz.clone(),
            rel_rem: rels.range(p.ao_rem_bits).clone(),
            epow: rels.epow.clone(),
            rel_acc_lo: rels.range(p.ao_acc_lo_bits).clone(),
            rel_acc_hi: rels.range(p.ao_acc_hi_bits).clone(),
            acc_lo_bits: p.ao_acc_lo_bits,
            acc_hi_bits: p.ao_acc_hi_bits,
            acc_mag: p.ao_acc_mag,
            denp2: p.ao_denp2,
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
                // claims arrive in the same order the plan lists the bindings, which
                // is mat order -- `o` before `down` -- and the assert below is
                // what says so rather than trusting it.
                output_claim: plan.has_output_claim(&m.name).then(|| {
                    let kind = plan.output_claim_kind(&m.name);
                    let ow = plan.output_claim_of(w_glue, lam_glue, &m.name, m.tokens, m.n_out,
                                           plan.mat_lays[mi].per_row);
                    let claimed = if m.name == "lm_head" {
                        claims.logits_claim.first().copied()
                    } else {
                        // the claims arrive in mat order and the plan lists the
                        // bindings in the same order, `o` before `down`
                        plan.glue.as_ref().unwrap().output_bindings.iter()
                            .position(|t| t.name == m.name)
                            .and_then(|k| claims.deq_out.get(k).copied())
                    };
                    crate::dequant::OutputClaim {
                        w: ow.w,
                        lam: ow.lam,
                        k_in: ow.k_in,
                        k_bd: ow.k_bd,
                        claimed: claim_at(claimed, "deq_out/logits_claim"),
                        kind,
                    }
                }),
                // Stream II (E11): the x-side scale's claim, on every matmul a
                // requantize seam feeds.  `mu` is the wiring family's own `w`
                // and `lam` its `lam`, the same two the seam's own fold uses,
                // so the two claims are comparable with no scale factor.
                //
                // Stream SS: never on a `Variant::Fold` region -- its second
                // shared column is the divisor's token half, not the scale's
                // exponent, so there is no claim to state (see `mat_interaction`'s
                // caller and `rq_scale_open`).  The predicate is the same one
                // there: the two have to agree or the AIR reads an interaction
                // column the fill never wrote.
                sside: (plan.rqs.iter()
                        .any(|rp| plan.walks[rp.walk].mats.contains(&mi))
                        && plan.specs[mi].variant != DeqVariant::Fold)
                    .then(|| crate::dequant::SSide {
                        mu: w_glue,
                        lam: lam_glue,
                        claimed: claim_at(claims.deq_sc.get(mi).copied(), "deq_sc"),
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
                // stream II: the group's own claim over its committed
                // activations, in the canonical channel-major order.  Present
                // exactly when the chunk carries the wiring family, so a chunk
                // without it proves what it always proved.
                qx: plan.glue.as_ref().map(|_| {
                    let m0 = &p.mats[w.mats[0]];
                    let nn = (m0.tokens * m0.n_in) as i128;
                    let claimed = claim_at(claims.walk_qx.get(wi).copied(), "walk_qx");
                    // Stream II (E11), at the JJ merge: the `o` group folds
                    // under the *attnout* region's weights, not the wiring
                    // family's.  Every other group's counterparty is a
                    // requantize region on the dequant's `MatLayout` map, whose
                    // canonical weight is `w_glue^(N-1-k tokens-t)`; `o` has no
                    // requantize region (`AttnOutEval` is its requantize) and
                    // its only counterparty is `claims.ao_q`, which weights
                    // element `(channel k, token t)` by `rv^k s^t`.  Both
                    // conventions factorize as (channel power)(token power), so
                    // this is a change of the three constants and nothing else:
                    // slot `p` of chunk `b` carries token `t = b pq + p`, so the
                    // slot weight is `s^p`, the step inside a channel is
                    // `s^pq`, and the step at a channel boundary is `rv` with
                    // the `chunks - 1` in-channel steps of that channel undone.
                    if w.name == "o" {
                        let (spow, sstep) = powers(s, w.lay.pq);
                        return crate::walk_air::WalkQx {
                            w: spow,
                            m_in: sstep,
                            m_bd: rv * sstep.inverse().pow(w.lay.g.chunks as u128 - 1),
                            claimed,
                        };
                    }
                    crate::walk_air::WalkQx {
                        w: (0..w.lay.pq).map(|pp| wpow(w_glue, nn - 1 - pp as i128)).collect(),
                        m_in: wpow(w_glue, -(w.lay.pq as i128)),
                        m_bd: wpow(w_glue, (w.lay.pq as i128) * (w.lay.g.chunks as i128 - 1)
                                   - m0.tokens as i128),
                        claimed,
                    }
                }),
            },
            if p.gkr_walks { SecureField::zero() } else { next_use(claims) },
        ));
    }

    // B10 step 2: the wiring family, between the walks and the shared tables --
    // the order the prover commits its columns in, so the two cannot diverge
    let mut glue = vec![];
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let bindings = gp.bindings(w_glue, lam_glue, claims);
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
                    binding: Some(bindings[i].clone()),
                    // Stream H: a region carries a boundary side when the plan
                    // named it one.  The sides are registered in `stream_in`,
                    // `stream_out` order and the components in region order, so
                    // the tree a side lands in is the same for prover and
                    // verifier without either being told.
                    bound: plan.layer_sides.iter().find(|s| s.region == i).map(|s| s.bound),
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

    // B11 stream S, lever 3, moved down by streams U and V: the binding, from the spans
    // the allocator handed out above.
    let mut link_binding = None;
    if let Some(hl) = link {
        let span_of = |locs: &[stwo::core::pcs::TreeSubspan], tree: usize| {
            locs.iter().find(|sp| sp.tree_index == tree)
                .map(|sp| sp.col_start)
                .expect("a component with no span in the tree the binding reads")
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
        // Stream PP (E11): the binding reads the *global* mask, so its `(tree,
        // column)` pairs are physical -- the base trace is tree 2 and the
        // interaction tree 3.  Its preprocessed reads are indices into the
        // concatenated preprocessed space (the shape tree's columns then the
        // model tree's), which is the space `preprocessed_column_indices`
        // speaks and `Components::mask_points` scatters; `pre_loc` turns one
        // back into a physical pair.  Some of them are model columns -- the
        // attention output's `jv`, a dequant weight scale, a norm weight.
        let pre_ids_all: Vec<crate::compose::PreProcessedColumnId> =
            shape_ids.iter().chain(model_ids.iter()).cloned().collect();
        let n_shape = shape_ids.len();
        let pre_loc = |gi: usize| {
            if gi < n_shape { (0usize, gi) } else { (1usize, gi - n_shape) }
        };
        let mut pre_idx = vec![];
        let mut heights = vec![];
        for (l, comp) in hl.iter().zip(&links) {
            let t2 = span_of(comp.trace_locations(), crate::common::INTER_TREE);
            let mut terms = vec![];
            for (reg, c, w) in &l.weights {
                use crate::gkrlink::{RCol, Reg};
                let band = match reg {
                    Reg::Band(b) => *b,
                    // B11 stream V, lever 3: a glue region is one component, so
                    // its own span is the whole resolution -- the mask index
                    // *is* the column index within it, which is `Prog`'s order.
                    Reg::Glue(g) => {
                        let (tree, gi) = match c {
                            RCol::Glue(i) =>
                                (crate::common::BASE_TREE, span_of(glue[*g].trace_locations(), crate::common::BASE_TREE) + i),
                            // stream OO: the norm weight is the statement's, so
                            // it is a tree-0 column of the same region and the
                            // link names it by its preprocessed id
                            RCol::GluePre(i, k) => {
                                let tag = &plan.glue.as_ref()
                                    .expect("a glue column with no glue plan")
                                    .regions[*g].tag;
                                let want = crate::glue::glue_pre_col(tag, *i, *k).id;
                                let gi = pre_ids_all.iter().position(|x| x.id == want)
                                    .expect("a glue preprocessed column missing \
                                             from the preprocessed trees");
                                pre_idx.push(gi);
                                pre_loc(gi)
                            }
                            _ => panic!("a non-glue column named on a glue region"),
                        };
                        terms.push((tree, gi, *w));
                        continue;
                    }
                    Reg::Walk(k) => {
                        let RCol::Walk(i) = c else {
                            panic!("a non-walk column named on a walk region")
                        };
                        terms.push((crate::common::BASE_TREE,
                                    span_of(walks[*k].trace_locations(), crate::common::BASE_TREE) + i, *w));
                        continue;
                    }
                    Reg::Ao => {
                        let (tree, g) = match c {
                            RCol::Ao(i) => (crate::common::BASE_TREE,
                                             span_of(ao.trace_locations(), crate::common::BASE_TREE) + i),
                            // stream JJ: the slot's `jvp`, preprocessed
                            RCol::AoPre(sl) => {
                                let want = ao_chan("jv", *sl).id;
                                let g = pre_ids_all.iter().position(|x| x.id == want)
                                    .expect("an attnout jv column missing from the \
                                             preprocessed trees");
                                pre_idx.push(g);
                                pre_loc(g)
                            }
                            _ => panic!("a non-attnout column named on the \
                                         attnout region"),
                        };
                        terms.push((tree, g, *w));
                        continue;
                    }
                    // B11 stream U, lever 3: a dequant region is one component,
                    // so its own span is the whole resolution -- the mask index
                    // *is* the column index within it (`dequant::Map`).
                    Reg::Deq(m) => {
                        let (tree, g) = match c {
                            RCol::Deq(i) =>
                                (crate::common::BASE_TREE,
                                 span_of(mats[*m].trace_locations(), crate::common::BASE_TREE) + i),
                            RCol::DeqPre(sl) => {
                                let want =
                                    crate::dequant::chan_col(&p.mats[*m].name, "m", *sl).id;
                                let g = pre_ids_all.iter().position(|x| x.id == want)
                                    .expect("a dequant weight-scale column missing \
                                             from the preprocessed trees");
                                pre_idx.push(g);
                                pre_loc(g)
                            }
                            _ => panic!("a band column named on a dequant region"),
                        };
                        terms.push((tree, g, *w));
                        continue;
                    }
                };
                let (tree, g) = match c {
                    RCol::Chan(i) => (crate::common::BASE_TREE,
                                      span_of(chans[band].trace_locations(), crate::common::BASE_TREE) + i),
                    RCol::Key(i) => (crate::common::BASE_TREE,
                                     span_of(keys[band].trace_locations(), crate::common::BASE_TREE) + i),
                    RCol::Row(i) =>
                        (crate::common::BASE_TREE,
                         span_of(rows[row_of[band]].trace_locations(), crate::common::BASE_TREE) + i),
                    RCol::Msk(i) => {
                        let want = crate::attn_air::msk_id(band, *i).id;
                        let g = pre_ids_all.iter().position(|x| x.id == want)
                            .expect("a corner mask the preprocessed trees do not carry");
                        pre_idx.push(g);
                        pre_loc(g)
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
        link_binding = Some(crate::gkrlink::LinkTie { heights, pre_idx });
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
    // Stream JJ (E11), `witness-slack.md` section 1f: the exponent window.  It
    // is last of the core tables so that every table before it keeps its index,
    // and it is the only one with no device build -- `1 << log_epow` is at most
    // a few dozen rows, so there is nothing for a kernel to win.
    tables.push(FrameworkComponent::<TableEval>::new(
        alloc,
        TableEval { log_rows: plan.log_epow(),
                    cols: EPOW_COLS.iter().map(|c| id(*c)).collect(),
                    kind: TableKind::Frac(rels.epow.clone()) },
        unqm(claims.table_sums[5 + plan.widths.len()]),
    ));

    let mut gtables = vec![];
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let base = 6 + plan.widths.len();
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

    // stream II: the requantize seams, before the two ends.  Their weights are
    // the wiring family's own `w` and `lam` on the dequant's `MatLayout` map, so
    // a seam's input claim and its producer's `Out(0)` are two Horner sums over
    // the same sequence under the same weights -- which is what equality means.
    let mut rq_comps = vec![];
    let mut quant_comps = vec![];
    let mut rq_tabs = vec![];
    if let Some(rr) = &rels.rq {
        for (k, rp) in plan.rqs.iter().enumerate() {
            let binding = rq_binding_w(w_glue, lam_glue, rp,
                               claim_at(claims.rq_in.get(k).copied(), "rq_in"),
                               claim_at(claims.rq_out.get(k).copied(), "rq_out"),
                               claim_at(claims.rq_sc.get(k).copied(), "rq_sc"));
            rq_comps.push(FrameworkComponent::<crate::requant::RequantEval>::new(
                alloc,
                crate::requant::RequantEval {
                    tag: rp.name.clone(),
                    log_rows: rp.lay.log_rows,
                    per_row: rp.lay.per_row,
                    chunks: rp.lay.chunks,
                    v: rp.v,
                    with_max: true,
                    qnz: rels.qnz.clone(),
                    rq3: rr.rq3.clone(),
                    rq4: rr.rq4.clone(),
                    scl2: rr.scl2.clone(),
                    scl3: rr.scl3.clone(),
                    r15: rels.range(15).clone(),
                    rqv: Some(rr.rqv.clone()),
                    binding: Some(binding),
                },
                next_use(claims),
            ));
        }
        // stream NN: the quantize regions, between the seams and their shared
        // tables -- the same order `Plan::pre_ids` and the component lists use
        for (k, qp) in plan.quants.iter().enumerate() {
            let binding = q_binding_w(w_glue, lam_glue, qp, p.nh,
                              claim_at(claims.nn_in.get(k).copied(), "nn_in"),
                              claim_at(claims.nn_out.get(k).copied(), "nn_out"));
            // stream NN-3: this side's own division table, and the family's own
            // `(q, nz)` table rather than the attention output's
            let qr = rels.qq.as_ref().expect("the quantize regions draw their own tables");
            quant_comps.push(FrameworkComponent::<crate::rope::QuantEval>::new(
                alloc,
                crate::rope::QuantEval {
                    tag: qp.name.to_string(),
                    log_rows: qp.lay.log_rows,
                    per_row: qp.lay.per_row,
                    chunks: qp.lay.chunks,
                    bits: qp.bits,
                    qnz: qr.qnz.clone(),
                    rq3: qr.div[plan.q_tab_at(qp.bits)].clone(),
                    rqv: rr.rqv.clone(),
                    r15: rels.range(15).clone(),
                    binding: Some(binding),
                },
                next_use(claims),
            ));
        }
        let base = plan.rq_table_at();
        let mut t = 0usize;
        for v in &plan.rq_vars {
            rq_tabs.push(FrameworkComponent::<TableEval>::new(
                alloc,
                TableEval {
                    log_rows: crate::requant::rq_log_rows(*v),
                    cols: (0..crate::requant::rq_tab_width(*v))
                        .map(|k| crate::requant::rq_tab_col(*v, k)).collect(),
                    kind: if *v == crate::requant::SVariant::Pow2 {
                        TableKind::Frac(rr.rq3.clone())
                    } else {
                        TableKind::Mul(rr.rq4.clone())
                    },
                },
                unqm(claims.table_sums[base + t]),
            ));
            t += 1;
            rq_tabs.push(FrameworkComponent::<TableEval>::new(
                alloc,
                TableEval {
                    log_rows: 15,
                    cols: (0..crate::requant::scl_tab_width(*v))
                        .map(|k| crate::requant::scl_tab_col(*v, k)).collect(),
                    kind: if *v == crate::requant::SVariant::Pow2 {
                        TableKind::Pair(rr.scl2.clone())
                    } else {
                        TableKind::Frac(rr.scl3.clone())
                    },
                },
                unqm(claims.table_sums[base + t]),
            ));
            t += 1;
        }
        rq_tabs.push(FrameworkComponent::<TableEval>::new(
            alloc,
            TableEval { log_rows: 9,
                        cols: crate::requant::RQV_COLS.iter().map(|c| id(*c)).collect(),
                        kind: TableKind::Frac(rr.rqv.clone()) },
            unqm(claims.table_sums[base + t]),
        ));
        t += 1;
        // stream NN-3: the quantize family's own tables, in `Plan::pre_ids`'
        // order -- one division table per side width, then the `(q, nz)` pair
        if let Some(qr) = &rels.qq {
            for (i, w) in plan.q_tabs.iter().enumerate() {
                rq_tabs.push(FrameworkComponent::<TableEval>::new(
                    alloc,
                    TableEval { log_rows: crate::rope::qdiv_log_rows(*w),
                                cols: (0..crate::rope::QDIV_COLS)
                                    .map(|k| crate::rope::qdiv_col(*w, k)).collect(),
                                kind: TableKind::Frac(qr.div[i].clone()) },
                    unqm(claims.table_sums[base + t]),
                ));
                t += 1;
            }
            rq_tabs.push(FrameworkComponent::<TableEval>::new(
                alloc,
                TableEval { log_rows: crate::rope::qnz_log_rows(plan.q_nz_bits),
                            cols: (0..2).map(|k| crate::rope::qnz_col(plan.q_nz_bits, k))
                                .collect(),
                            kind: TableKind::Pair(qr.qnz.clone()) },
                unqm(claims.table_sums[base + t]),
            ));
        }
    }

    // stream E: the argmax, bound to the head's key claim.  It shares the layer's
    // 16-bit range table and the head dequant's step selectors, so it adds one
    // component and no table.
    let argmax = plan.argmax_mat().map(|mi| {
        let lay = plan.mat_lays[mi];
        let m = &plan.p.mats[mi];
        let binding = crate::argmax::ArgmaxBinding {
            claimed: claim_at(claims.am_key.first().copied(), "am_key"),
            claimed_tok: claim_at(claims.head_tok.first().copied(), "head_tok"),
            tag: m.name.clone(),
            ..argmax_binding_w(w_glue, m.tokens, m.n_out, lay.per_row)
        };
        FrameworkComponent::<crate::argmax::ArgmaxEval>::new(
            alloc,
            crate::argmax::ArgmaxEval {
                log_rows: lay.log_rows,
                per_row: lay.per_row,
                lax: plan.p.prove_argmax_lax,
                rk: rels.range(crate::argmax::KEY_BITS).clone(),
                binding: Some(binding),
            },
            next_use(claims),
        )
    });

    // stream BB: the embedding gadget, after the argmax and after everything
    // else.  The use region takes the last `use_sums` cursor position and the
    // table region the last `table_sums` slot, which is what keeps both lists
    // append-only.  Its binding reads the wiring family's own `w` and `lam`, so it
    // draws no challenge of its own.
    let (embed, emb_tab) = match (&plan.emb, &rels.emb) {
        (Some(l), Some(er)) => {
            let g = plan.glue.as_ref().expect("the embedding gadget needs the wiring family");
            let mut binding = crate::embed::binding_w(w_glue, lam_glue, g.t.d, p.s, l);
            binding.claimed = claim_at(claims.embed_out.first().copied(), "embed_out");
            binding.claimed_tok = claim_at(claims.embed_tok.first().copied(), "embed_tok");
            let em = FrameworkComponent::<crate::embed::EmbedEval>::new(
                alloc,
                crate::embed::EmbedEval {
                    log_rows: l.log_use,
                    per_row: l.pe,
                    rel: er.clone(),
                    rel_ef: rels.range(crate::embed::EF_BITS).clone(),
                    binding,
                },
                next_use(claims),
            );
            let et = FrameworkComponent::<crate::embed::EmbTableEval>::new(
                alloc,
                crate::embed::EmbTableEval { log_rows: l.log_tab, per_row: l.pe,
                                             rel: er.clone() },
                unqm(claims.table_sums[plan.emb_table_at()]),
            );
            (Some(em), Some(et))
        }
        _ => (None, None),
    };


    BuiltComponents { chans, keys, rows, links, link_binding, ao, mats, walks, glue, exp_t, tables,
                      gtables, argmax, embed, emb_tab, rqs: rq_comps, quants: quant_comps,
                      rq_tabs, pows,
                      has_rows: plan.rows_band.clone() }
}

/// Stream E: the argmax side's binding weights -- the mirror of the head dequant's
/// output side.  Its index map ascends the vocabulary (`i = (chunks-1-b) pr + sl`)
/// where the dequant's descends, so the ordinary row step is the *inverse* of the
/// output side's and the boundary step differs in the token term's sign.
fn argmax_binding_w(w: SecureField, tokens: usize, vocab: usize, per_row: usize)
                -> crate::argmax::ArgmaxBinding {
    let n = (tokens * vocab) as i128;
    let (tok, pr, v) = (tokens as i128, per_row as i128, vocab as i128);
    crate::argmax::ArgmaxBinding {
        w: (0..per_row)
            .map(|sl| wpow(w, n - 1 - (v - pr) * tok - sl as i128 * tok))
            .collect(),
        k_in: wpow(w, pr * tok),
        k_bd: wpow(w, -((v - pr) * tok + 1)),
        claimed: SecureField::zero(),
        // stream KK: the token accumulator's step is `w` itself, one per token
        k_tok: w,
        claimed_tok: SecureField::zero(),
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
    /// binding claims each of them recomputed from the witness rather than the trace
    pub stream_bindings_ok: bool,
    /// Stream H: what the boundary trees cost this chunk, or null when the flag
    /// was off and there are none
    pub layer_boundary: Value,
    /// Stream AA: the same for the K/V boundary trees -- per side, its tree,
    /// its head band group, whether it is the out or the in side, the bands
    /// that wrote it, its columns and its cells.  Null without `--kv-boundary`.
    pub kv_boundary: Value,
    /// how many committed values a glue tamper changed; `None` when none was
    /// set.  A probe whose tamper fired nowhere proved the honest witness, and
    /// its verdict says nothing about the tamper -- the caller must not read
    /// `Some(0)` as "accepted".
    pub glue_tamper_fired: Option<usize>,
}

/// Tampers that need a hook inside the prover (everything else corrupts the
/// `Layer` before calling in).
#[derive(Default, Clone, Copy)]
pub struct ProverTamper {
    /// flip one row of the shared normalisation table, witness rebuilt to match
    pub norm_row: Option<usize>,
    /// B10 step 2: corrupt one glue region's own copy of the stream it shares
    /// with another region -- `(region index, value index)`.  Everything inside
    /// the region stays consistent, so what has to reject is the binding.
    pub glue_stream: Option<(usize, usize)>,
    /// a tamper inside one glue region's arithmetic, by region index
    pub glue: Option<(usize, crate::glue::Tamper)>,
    /// Stream H: corrupt one cell of *one* boundary tree, leaving the region it
    /// is bound to alone -- `(which side, which cell)`, side 0 being the input
    /// and side 1 the output.  Nothing else in the chunk changes, so what has to
    /// reject is that side's equality constraint: the probe that says the root
    /// is bound to the proof and not merely appended to it.
    ///
    /// Stream AA: the side index runs over `plan.layer_sides` and then over
    /// `plan.kv`, so side `plan.layer_sides.len() + j` is the `j`th K/V side --
    /// which for a chunk with a prefix is group `j / 2`'s out side when `j` is
    /// even and its in side when `j` is odd.
    pub layer_boundary: Option<(usize, usize)>,
    /// Stream E: corrupt the wiring family's own copy of a dequant output --
    /// `(region index, value index)` into `o_out` for `gn1` or `d_out` for
    /// `gn2`.  The region recomputes its add from the corrupted value, so
    /// everything inside the glue stays consistent and the only thing that can
    /// catch it is the dequant output binding.
    pub glue_add: Option<(usize, usize)>,
    /// Stream II: a probe inside one requantize seam -- `(seam index, what)`.
    /// Each leaves the region internally consistent and aims at one of its two
    /// bindings, or at the scale lookup.
    pub rq: Option<(usize, crate::requant::RqTamper)>,
    /// Stream II, control 3: bend one cell of a norm scale's *preprocessed*
    /// weight block.  The region recomputes `h` from the bent weight, so every
    /// constraint holds and the tree-0 root is what moves -- `root_ok` against
    /// the statement's `pre_root` is the rejection, and it is a different kind
    /// of rejection from every other control here.
    pub glue_w: Option<usize>,
    /// Stream NN: bend one cell of a rotation's *preprocessed* angle block.
    /// The products are recomputed from the bent `cos`, so every constraint in
    /// both halves still holds and the tree-0 root is what moves -- the same
    /// shape stream II's `glue_w` has, under its own field so the two controls
    /// cannot fire together and the probe stays a one-field rejection.
    pub rope_w: Option<usize>,
    /// Stream NN: a probe inside one quantize region -- `(region index, what)`.
    /// Each leaves the region internally consistent and aims at one of its two
    /// bindings, at the division table, or at `pre_root`.
    pub quant: Option<(usize, crate::rope::QTamper)>,
    /// Stream NN: bend one cell of a **band's** own committed K, Q or V and
    /// rebuild the band around it, so the attention it proves is honest and the
    /// only thing that can see it is the band's new claim against the quantize
    /// region's output.  `(which of K/Q/V, which cell)`.
    pub band_qkv: Option<(usize, usize)>,
    /// Stream E: a tamper inside the composed argmax region
    pub argmax: Option<crate::argmax::Tamper>,
    /// Stream BB: a tamper inside the composed embedding region.  `TokRow` is
    /// the aimed one -- the id moved to another valid entry with the committed
    /// row recomputed to match, so every constraint inside the region holds and
    /// only the binding to the layer's `bin` rejects.
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
    /// Stream JJ (witness-slack section 7 item 1): round one *non-last* row of
    /// the attention output against `den - 1`, self-consistently -- the whole
    /// row's `denp`, `qm`, `sg` and `d` are rederived from it, so every
    /// constraint and every lookup of the pre-change AIR holds and the pre-change
    /// statement accepts a rounding by a denominator nothing pinned.  The index
    /// is a flat `(h, t, ch)` site, as `ao_clamp`'s is.
    pub ao_den: Option<usize>,
    /// Stream JJ (E11): move one *row*'s `jo` by one and leave the token's
    /// other rows alone.  Every `EPOW` lookup still lands on a real table row
    /// and the row's own arithmetic is self-consistent, so only the `nofirst`
    /// pin on `joff` -- "one output exponent per token" -- can reject it.
    pub ao_jo: Option<usize>,
    /// One slot's `num` doubled: a `(key, num, pj)` triple the `EPOW` table
    /// does not carry, so only the window lookup can reject it.  The index is
    /// a flat `(h, t, ch)` site.
    pub ao_num: Option<usize>,
    /// One slot's low accumulator limb moved by one, so `alo + 2^lo ahi` is no
    /// longer `acc + acc_off`: the other half of the no-wrap window, and only
    /// its own two range lookups can reject it.
    pub ao_acc: Option<usize>,
    /// One *token*'s `jo` moved by one, on every row of it, with the whole
    /// region rederived.  This one must be **accepted**: `jo` is the prover's
    /// choice inside the window and nothing in this statement reads the
    /// attention output's scale.  It is the honest limit of what section 1f
    /// buys, and the probe that says so.  The index is a token.
    pub ao_jo_tok: Option<usize>,
    /// Stream TT (E11), log row 73: one slot's committed magnitude replaced by
    /// `-|acc|` -- the other root of the pin `(amag - acc)(amag + acc) = 0`,
    /// so every constraint of the region still holds and every other lookup
    /// lands.  As a field element it is about `2^31`, which is exactly the
    /// forgery `2 mag num < p` exists to stop, and the only reader of it is
    /// the high accumulator limb's range check.  The magnitude form only; the
    /// index is a flat `(h, t, ch)` site, as `ao_num`'s is.
    pub ao_mag: Option<usize>,
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
    /// stream II: the requantize seams, in `Plan::rqs` order
    pub rq_regions: Vec<crate::requant::RqRegion>,
    /// stream NN: the quantize regions, in `Plan::quants` order
    pub q_regions: Vec<crate::rope::QRegion>,
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
    pub layer_boundary_tamper: Option<(usize, usize)>,
    /// how many committed values a `ProverTamper::glue` changed, `None` when
    /// none was set.  `Some(0)` is a probe that measured nothing.
    pub glue_tamper_fired: Option<usize>,
}

/// B7b (README 26): the first half of `prove_layer` -- every fill, every
/// table, every column born where it will be committed -- up to the
/// transcript.
pub fn generate(layer: &Layer, params: &Params, tamper: &ProverTamper, pre_key: Option<u64>)
                -> Gen {
    let m4_t = std::time::Instant::now();
    crate::util::m4_touch_device();
    crate::util::m4_add("wall/cuda_init", m4_t.elapsed().as_secs_f64());
    // B12 stream GG: the context, before any column.
    crate::device::mem_mark("init");
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
                                 plan.round, p.ao_q_bits, plan.widths.clone(),
                                 plan.e_window().0, plan.e_window().1);
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
            || plan.walks.iter().any(|w| w.act_bits != 8)
            // stream JJ (E11): an armed attnout tamper does too -- `ao_region`
            // sets its own mode to 0 because the device fill has no `corrupt`
            // path -- so its counters are the host's and the shared
            // multiplicity columns have to be as well.  Without this the
            // defaults cuda regime stops at "B7_M4_WFILL=1 but the attnout is
            // host-built" and no attnout control can be run on the card.
            || tamper.ao_clamp.is_some() || tamper.ao_den.is_some()
            || tamper.ao_jo.is_some() || tamper.ao_num.is_some()
            || tamper.ao_acc.is_some() || tamper.ao_jo_tok.is_some()
            || tamper.ao_mag.is_some()
            // stream TT (E11), log row 73: and the magnitude form, whose tenth
            // column the fork's `b65_fill_attnout` does not write -- `ao_region`
            // sets its own mode to 0 for it on both arms
            || p.ao_acc_mag || p.ao_denp2,
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
    let aor = ao_region(&layer.ao, p.ao_target, p.ao_q_bits, p.ao_rem_bits,
                        (p.ao_acc_lo_bits, p.ao_acc_hi_bits), p.ao_acc_mag, p.ao_denp2,
                        p.ao_jv_min, p.ao_jo_max,
                        &mut tables, None, tamper.ao_clamp, tamper.ao_den, tamper.ao_jo,
                        tamper.ao_num, tamper.ao_acc, tamper.ao_jo_tok, tamper.ao_mag);

    crate::util::m4_add("gen/attnout", m4_t.elapsed().as_secs_f64());

    let m4_t = std::time::Instant::now();
    let mut mat_regions = vec![];
    for (mi, (m, sp)) in layer.mats.iter().zip(&plan.specs).enumerate() {
        let r = mat_region(m, sp, &mut tables, plan.has_output_claim(&m.name),
                           plan.output_claim_band(&m.name)
                               .then(|| plan.p.hd / plan.mat_lays[mi].per_row),
                           (tamper.fold_qm, tamper.fold_dent));
        mat_regions.push(r);
    }
    let n_deq_base: usize = mat_regions
        .iter()
        .map(|r| r.base_cells())
        .sum();

    crate::util::m4_add("gen/dequant", m4_t.elapsed().as_secs_f64());

    // Stream E: the argmax region, on the head dequant's own layout so the binding
    // is positionwise.  Its range lookups go on the layer's 16-bit table, so its
    // multiplicities are bumped into that table's vector the way the wiring
    // family's go into `NORM` and `FRAC`.
    let m4_t = std::time::Instant::now();
    let am_region = plan.argmax_mat().map(|mi| {
        let m = &plan.p.mats[mi];
        let keys = layer.head_keys.as_ref().expect("the argmax needs key_head.u16");
        let tok = layer.head_tok.as_ref().expect("the argmax needs tok_head.i32");
        let w = crate::argmax::AWitness::from_cols(
            m.tokens, m.n_out, keys.iter().map(|k| u32::from(*k)).collect(), tok.clone())
            .expect("the head dump's argmax columns");
        let ri = tables.widths.iter().position(|w| *w == crate::argmax::KEY_BITS)
            .expect("the argmax needs the layer's 16-bit range table (--z-lo-bits 16)");
        let mut m_rk = std::mem::take(&mut tables.m_r[ri]);
        let r = crate::argmax::region(&w, plan.p.prove_argmax_lax, plan.mat_lays[mi],
                                      &mut m_rk, tamper.argmax);
        tables.m_r[ri] = m_rk;
        tables.off_table += r.off_table;
        r
    });
    crate::util::m4_add("gen/argmax", m4_t.elapsed().as_secs_f64());

    // Stream BB: the embedding gadget.  Its use side commits the layer's own
    // residual-stream input -- the wiring family's `gn1` group 0, value for
    // value -- so that the binding it publishes and the one the wiring family
    // publishes are two claims over the same sequence.  Its `ef` range lookups
    // go on the layer's 8-bit table, exactly as the argmax's go on the 16-bit
    // one.
    let m4_t = std::time::Instant::now();
    let emb_region = plan.emb.map(|l| {
        let gc = layer.glue.as_ref()
            .expect("`--embed` bindings to the wiring family; extract with `--glue`");
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
        // Stream II, at the merge with JJ: a synthetic layer's family is built
        // in `Layer::synth_glue`, which also rewires the matmul inputs to be
        // the family's own outputs -- JJ's marker-plus-substitution route is
        // gone, and `synth_glue` sets `o_out`/`d_out` to the `o` and `down`
        // dequant references so JJ's output bindings still bite on the desk.
        assert_eq!(gc.d, gp.t.d, "the witness's d disagrees with the glue params");
        let mut gt = GlueTables::new(p.kmax, plan.round, &gp.uns);
        gt.norm = tables.norm.clone();
        gt.m_norm = std::mem::take(&mut tables.m_norm);
        gt.m_frac = std::mem::take(&mut tables.m_frac);
        let t = gp.t;
        let rows1 = t.rows1();
        let mut mismatch = vec![];
        // what the two reduction regions read as their stream, kept so a norm scale
        // region reads the identical block -- the binding compares those, and B1's
        // own column is the separate check
        let mut norm_scale_in: Vec<Vec<u16>> = vec![];
        // stream II: and each norm's own sum of squares, one per token, which is
        // what the norm scale's group 1 commits and `UnKind::NormR` turns into `r`
        let mut ms_out: Vec<Vec<u16>> = vec![];
        // stream 0: xin + o_out into xmid, and the squares' first levels;
        // stream 1: xmid + d_out into xout, and norm2's
        //
        // Stream II, the head chunk: one reduction and no add.  `xin` is the
        // last layer's committed residual, level 1 is the squares' tree alone,
        // and there is no second stream to thread -- so the loop runs once,
        // over an empty second operand, and the region's output group 1 (the
        // add) does not exist to be compared.
        let empty: Vec<u16> = vec![];
        let sources: Vec<(&Vec<u16>, &Vec<u16>, &str)> = if gp.head_norm {
            vec![(&empty, &empty, "xin_head")]
        } else {
            vec![(&gc.o_out, &gc.xmid, "xmid"), (&gc.d_out, &gc.xout, "xout")]
        };
        let mut stream = t.lay(&gc.xin);
        for (k, (add, want, wname)) in sources.iter().enumerate() {
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
            let mut addv = if gp.head_norm { vec![] } else { t.lay(add) };
            if let Some((ri, vi)) = tamper.glue_add {
                if ri == i1 {
                    let n = addv.len();
                    addv[vi % n] ^= 1;
                }
            }
            norm_scale_in.push(stream.clone());
            let leaves = if gp.head_norm {
                vec![stream.clone()]
            } else {
                vec![stream.clone(), addv]
            };
            let w1 = crate::glue::GlueWitness::new(
                rows1, leaves, &prog1, plan.round,
                format!("B1 residual stream [{} x {}]", t.tokens, t.d));
            let tam1 = tamper.glue.filter(|(ri, _)| *ri == i1).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r1 = crate::glue::glue_region(&w1, &prog1, plan.round, &tag1, &mut gt, tam1);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            // what the region computed is what the next one reads: that is what
            // the binding says, and it is checked against B1's own column here.
            // The head chunk's level 1 has no add, so there is no second output
            // group and nothing to compare -- its own reference is the norm scale's.
            let got = if gp.head_norm { vec![] } else { t.unlay(&r1.outs[1]) };
            // streams JJ and II: a synthetic layer carries no B1 column to be
            // right about, and an absent reference is reported as absent
            // (`null`) rather than as zero mismatches.
            if !gp.head_norm && want.len() >= t.tokens * t.d {
                let bad = (0..t.tokens * t.d)
                    .filter(|i| crate::glue::canon(got[*i]) != crate::glue::canon(want[*i]))
                    .count();
                mismatch.push(json!({"column": wname, "values": t.tokens * t.d,
                                     "mismatch": bad}));
            } else {
                mismatch.push(json!({"column": wname, "values": 0,
                                     "mismatch": serde_json::Value::Null,
                                     "note": "the dump does not carry this column"}));
            }
            let l2 = t.roots_to_l2(&r1.outs[0]);
            let w2 = crate::glue::GlueWitness::new(
                t.tokens, vec![l2], &prog2, plan.round,
                format!("the level-1 roots [{} x {}]", t.tokens, t.a));
            // stream II: the norm scale reads this norm's own sum of squares, one per
            // token -- `r` is its `UnKind::NormR` image and nothing else
            let tam2 = tamper.glue.filter(|(ri, _)| *ri == i1 + 1).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r2 = crate::glue::glue_region(&w2, &prog2, plan.round, &tag2, &mut gt, tam2);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            ms_out.push(r2.outs[0][..t.tokens].to_vec());
            if !gp.head_norm {
                stream = t.lay(&got);
            }
            glue_regions.push(r1);
            glue_regions.push(r2);
        }
        // ---- stream E: h = (x * r) * w, once per norm -------------------------
        for (k, i) in gp.norm_scales.iter().copied().enumerate() {
            let (rcol, wcol, want, wname, rname) = if gp.head_norm {
                (&gc.r1, &gc.w1, &gc.h1, "h_head", "r_head")
            } else if k == 0 {
                (&gc.r1, &gc.w1, &gc.h1, "x_q", "n1_out")
            } else {
                (&gc.r2, &gc.w2, &gc.h2, "x_gate", "n2_out")
            };
            let un_r = crate::glue::UnKind::NormR { inv_d: gc.inv_d, eps: gc.eps };
            let prog = gp.regions[i].prog.clone();
            let tag = gp.regions[i].tag.clone();
            // stream II: a row is one `(residue, token)` pair, so its single
            // group-1 leaf is its own token's sum of squares -- the reduction's
            // published answer, not a committed `r`.  `w` is per channel and
            // constant down the tokens, so the row reads channels `res + j a`;
            // it is preprocessed now, so it is `pre` and not a group.
            let msblk: Vec<u16> =
                (0..rows1).map(|m| crate::glue::canon(ms_out[k][m % t.tokens])).collect();
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
            // stream II's control 3: one norm-weight cell bent on the prover
            // side.  It is preprocessed, so the constraint set stays satisfied
            // and the *root* moves -- `root_ok` against the statement's
            // `pre_root` is what rejects it, and the record says so.
            if let Some(x) = tamper.glue_w.filter(|_| k == 0) {
                let at = x % wblk.len();
                wblk[at] ^= 1;
                eprintln!("compose: norm weight cell {at} of {tag} bent (a rejection is the pass)");
            }
            crate::util::m4_add("gen/glue_wblk", t_wb.elapsed().as_secs_f64());
            let w = crate::glue::GlueWitness::with_pre(
                rows1, vec![norm_scale_in[k].clone(), msblk], wblk, &prog, plan.round,
                format!("B1 norm{} scale [{} x {}]", k + 1, t.tokens, t.d));
            let tam = tamper.glue.filter(|(ri, _)| *ri == i).map(|(_, x)| x);
            let t_gr = std::time::Instant::now();
            let r = crate::glue::glue_region(&w, &prog, plan.round, &tag, &mut gt, tam);
            crate::util::m4_add("gen/glue_region", t_gr.elapsed().as_secs_f64());
            // the norm scale's output is the q/k/v (resp. gate/up) matmuls' own bf16
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
            // stream II: and the derived `r` itself, against B1's own
            // `n1_out`/`n2_out`.  The norm scale no longer commits `r`, so this is the
            // one number that says the `NormR` table reproduces the model: a
            // dump whose norm reduction is B1's fp32 accumulation rather than
            // its balanced tree (`reduce = "fp32"`) will differ here, and the
            // `x_q`/`x_gate` count above will differ with it.
            if rcol.len() >= t.tokens {
                let rbad = (0..t.tokens)
                    .filter(|tt| {
                        let got = crate::glue::un_image(un_r,
                                                        crate::glue::canon(ms_out[k][*tt]));
                        crate::glue::canon(got) != crate::glue::canon(rcol[*tt])
                    })
                    .count();
                mismatch.push(json!({"column": rname, "values": t.tokens, "mismatch": rbad}));
            } else {
                mismatch.push(json!({"column": rname, "values": 0,
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
            // region proves.  What is *not* here is the binding pinning them to it:
            // that is the open dequant binding, and until it lands these leaves are
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
            // Stream JJ (E11): the SwiGLU seam's control.  One of the region's
            // two committed input copies is bent and the product is recomputed
            // from it, so every constraint inside the region still holds and
            // only the binding against the gate (or up) dequant region's published
            // output can reject.  `vi` past `n` aims at `up`.
            if let Some((ri, vi)) = tamper.glue_add {
                if ri == i {
                    let n2 = gate_out.len();
                    if vi < n2 { gate_out[vi] ^= 1 } else { up_out[(vi - n2) % n2] ^= 1 }
                }
            }
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
        // ---- stream NN: the rotation ----------------------------------------
        //
        // Two regions per side, in the plan's own order.  A row is one
        // (token, head) and its `hd` leaves are that head's slice of the
        // matmul's dequant output, so the row walks `hd` consecutive
        // token-major positions and the binding is a flat Horner.  `cos` and
        // `sgnsin` are preprocessed and depend on the token's **absolute**
        // position, `pos0 + t`.
        if let Some(rp) = &p.rope {
            // the angle table, from the STATEMENT.  A dump that carries B1's own
            // `inv_freq.f64` is checked against it below and never consumed:
            // what tree 0 holds is a function of `Params`, which is stream LL's
            // precondition for the cheap half of the weights-tree split.
            let hd = p.hd;
            let inv = rp.inv(hd);
            if let Some(refv) = layer.rope.as_ref().and_then(|r| r.inv_ref.as_ref()) {
                // Two comparisons, because the f64 vector is not what the proof
                // commits and equality of it is the wrong bar.
                //
                // `theta ** x` is a libm `pow` in this crate and torch's own in
                // `b1/model.py`, and the two round differently: on
                // Llama-3.1-8B's `llama3` branch they disagree in 3 of 64 pairs
                // by exactly one ulp of f64.  That is a rounding difference in
                // a transcendental, not a different rotation, and asserting
                // f64 equality made every real chunk refuse to prove.
                //
                // What tree 0 holds is `rope_at`'s bf16 `cos` and `sgnsin` at
                // this chunk's own positions.  So the f64 gap is *reported*
                // (`rope_inv_vs_b1`, which is what NN session 1 section 6 said
                // it was), and the bf16 table the region actually commits is
                // *asserted* -- every cell, at every position this chunk uses.
                let bad = inv.iter().zip(refv).filter(|(a, b)| a != b).count();
                let t_iv = std::time::Instant::now();
                let cells: usize = (0..p.s)
                    .into_par_iter()
                    .map(|t| {
                        let pos = rp.pos0 + t;
                        let a = crate::rope::pre_row(&inv, hd, pos);
                        let b = crate::rope::pre_row(refv, hd, pos);
                        a.iter().zip(&b).filter(|(x, y)| x != y).count()
                    })
                    .sum();
                crate::util::m4_add("gen/rope_invref", t_iv.elapsed().as_secs_f64());
                assert_eq!(cells, 0,
                           "the angle table the statement builds differs from the dump's \
inv_freq in {cells} of {} committed bf16 cells: the rotation this chunk would prove is not \
B1's (the f64 vectors differ in {bad} of {} places)",
                           2 * hd * p.s, inv.len());
                if bad != 0 {
                    eprintln!("compose: rope inv_freq differs from the dump's in {bad} of {} \
f64 places and in 0 of {} committed bf16 cells -- a `pow` rounding difference, not a \
different rotation", inv.len(), 2 * hd * p.s);
                }
            }
            for rr in &gp.ropes {
                let mi = layer.mats.iter().position(|m| m.name == rr.name)
                    .unwrap_or_else(|| panic!("--rope needs the {} matmul", rr.name));
                let n_out = rr.heads * hd;
                let rows = rr.tokens * rr.heads;
                // the dequant region's own reference, which is exactly what its
                // `Out(0)` claim folds -- so the binding below is between two folds
                // of one sequence and not between two computations of it
                let t_dq = std::time::Instant::now();
                let mut deq = vec![0u16; rr.tokens * n_out];
                deq.par_chunks_mut(n_out).enumerate().for_each(|(t, row)| {
                    for (c, slot) in row.iter_mut().enumerate() {
                        *slot = crate::layer::deq_ref(&layer.mats[mi], &plan.specs[mi], t, c);
                    }
                });
                crate::util::m4_add("gen/rope_deqref", t_dq.elapsed().as_secs_f64());
                // the control on the producer side: one dequant output copy
                // bent, the whole rotation recomputed around it, so every
                // constraint inside both regions still holds and only the binding
                // against the matmul's own published output can reject
                if let Some((ri, vi)) = tamper.glue_add {
                    if ri == rr.mul {
                        let n = deq.len();
                        deq[vi % n] ^= 1;
                    }
                }
                let mut pre = vec![0u16; rows * 2 * hd];
                pre.par_chunks_mut(2 * hd).enumerate().for_each(|(j, row)| {
                    row.copy_from_slice(&crate::rope::pre_row(&inv, hd, rp.pos0 + j / rr.heads));
                });
                // stream NN's control 3: one preprocessed angle bent.  The
                // region stays satisfiable with the products recomputed around
                // it, so `root_ok` against the statement's `pre_root` is the
                // only thing that can reject -- the same shape stream II's
                // norm-weight control has.
                if let Some(x) = tamper.rope_w.filter(|_| rr.name == "q") {
                    let at = x % pre.len();
                    let was = pre[at];
                    pre[at] = crate::rope::bend_pat(was);
                    assert_ne!(crate::glue::split(pre[at]), crate::glue::split(was),
                               "the rope angle probe did not fire at cell {at}");
                    eprintln!("compose: rope angle cell {at} of rp{}m bent {was:#06x} -> \
{:#06x} (a rejection is the pass)", rr.name, pre[at]);
                }
                let mprog = gp.regions[rr.mul].prog.clone();
                let mtag = gp.regions[rr.mul].tag.clone();
                let wm = crate::glue::GlueWitness::with_pre(
                    rows, vec![deq], pre, &mprog, plan.round,
                    format!("the {} dequant output [{} x {}]", rr.name, rr.tokens, n_out));
                let tam = tamper.glue.filter(|(ri, _)| *ri == rr.mul).map(|(_, x)| x);
                let rm = crate::glue::glue_region(&wm, &mprog, plan.round, &mtag, &mut gt, tam);
                let aprog = gp.regions[rr.add].prog.clone();
                let atag = gp.regions[rr.add].tag.clone();
                let wa = crate::glue::GlueWitness::new(
                    rows, vec![rm.outs[0].clone()], &aprog, plan.round,
                    format!("the {} rotation's products [{} x {}]", rr.name, rows, 2 * hd));
                let tam = tamper.glue.filter(|(ri, _)| *ri == rr.add).map(|(_, x)| x);
                let ra = crate::glue::glue_region(&wa, &aprog, plan.round, &atag, &mut gt, tam);
                glue_regions.push(rm);
                glue_regions.push(ra);
            }
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

    // ---- stream II: the requantize seams ------------------------------------
    //
    // Each reads its producing region's *computed* output -- not a column of
    // the dump -- so what the seam requantizes is what the wiring family just
    // proved, and the binding is what says the walk multiplied the result.  The
    // per-token scale is the dump's own `jx`/`sx`, i.e. the same field element
    // the matmul's dequant region commits, which is what leaves the remaining
    // scale seam a one-constraint job.
    let m4_t = std::time::Instant::now();
    let mut rq_regions: Vec<crate::requant::RqRegion> = vec![];
    for (k, rp) in plan.rqs.iter().enumerate() {
        let g = plan.glue.as_ref().expect("a requantize seam needs the wiring family");
        let reg = &glue_regions[rp.producer.0];
        let x = if rp.producer.0 == g.swiglu.unwrap_or(usize::MAX) {
            g.tff.expect("the SwiGLU's tiling").unlay(&reg.outs[0])
        } else {
            g.t.unlay(&reg.outs[0])
        };
        let m0 = &layer.mats[plan.walks[rp.walk].mats[0]];
        let r = crate::requant::rq_region(
            &rp.name, &x, &m0.qx, &m0.jx.iter().map(|v| *v as i64).collect::<Vec<_>>(),
            &m0.sx, rp.tokens, rp.n, rp.v, rp.lay, &mut tables,
            tamper.rq.filter(|(i, _)| *i == k).map(|(_, t)| t));
        rq_regions.push(r);
    }
    crate::util::m4_add("gen/requant", m4_t.elapsed().as_secs_f64());

    // ---- stream NN: the three quantize regions -------------------------------
    //
    // q and k read the rotation's own output, v its matmul's dequant reference:
    // in both cases the region consumes what another region *proved*, not a
    // column of the dump, so the input binding compares two folds of one sequence.
    // `want` is the bands' own committed integers, gathered onto the query-head
    // grid -- a reference count and nothing the AIR reads.
    let m4_t = std::time::Instant::now();
    let mut q_regions: Vec<crate::rope::QRegion> = vec![];
    let nn_cols = (!plan.quants.is_empty()).then(|| nn_band_cols(&plan, layer));
    for (k, qp) in plan.quants.iter().enumerate() {
        let mi = layer.mats.iter().position(|m| m.name == qp.name)
            .unwrap_or_else(|| panic!("--rope needs the {} matmul", qp.name));
        let x: Vec<u16> = match qp.producer {
            Some((ri, _)) => glue_regions[ri].outs[0].clone(),
            None => {
                let mut v = vec![0u16; qp.tokens * qp.n];
                let n_out = qp.n;
                v.par_chunks_mut(n_out).enumerate().for_each(|(t, row)| {
                    for (c, slot) in row.iter_mut().enumerate() {
                        *slot = crate::layer::deq_ref(&layer.mats[mi], &plan.specs[mi], t, c);
                    }
                });
                v
            }
        };
        // the exponents, from the query-head grid onto the producer's own: a
        // GQA group's `rep` query heads share one scale, and `Layer::qkv_j`
        // says so out loud rather than picking a head quietly
        let jj = layer.qkv_j(qp.name, qp.heads, qp.hd, qp.rep);
        let want: Vec<i64> = nn_cols.as_ref().map(|(q, kk, vv)| {
            let src = match qp.name { "q" => q, "k" => kk, _ => vv };
            (0..qp.tokens)
                .flat_map(|t| (0..qp.n).map(move |c| (t, c)))
                .map(|(t, c)| {
                    let (kv, ch) = (c / qp.hd, c % qp.hd);
                    src[t * plan.p.nh * qp.hd + kv * qp.rep * qp.hd + ch]
                })
                .collect()
        }).unwrap_or_default();
        let r = crate::rope::q_region(
            qp.name, &x, &jj, Some(&want), qp.tokens, qp.n, qp.hd, qp.bits, plan.q_nz_bits,
            qp.lay, &mut tables,
            tamper.quant.filter(|(i, _)| *i == k).map(|(_, t)| t));
        q_regions.push(r);
    }
    crate::util::m4_add("gen/quant", m4_t.elapsed().as_secs_f64());

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
    // Stream JJ (E11): the exponent window table.  Host-built on both arms --
    // it is `1 << log_epow` rows, a few dozen at most -- so it sits after the
    // 21 columns `tables_dev` builds and the device parity check skips it.
    let log_epow = plan.log_epow();
    {
        let rows = tables.epow_rows();
        for k in 0..3 {
            pre_logs.push(log_epow);
            pre_tables.push(Some(eval_of(log_epow, rows.iter()
                .map(|r| m31_u32(match k { 0 => r.0, 1 => r.1, _ => r.2 })).collect())));
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
    // and its multiplicity, bumped by `ao_region` (and, on the device arm, by
    // the host loop `ao_region_dev` runs beside the kernel -- `e = jv - jo` is
    // a function of `ao.jv`, `ao.jo` and the layout alone, so no counter has to
    // come back off the card for it)
    mult_logs.push(log_epow);
    base_mults.push(Some(eval_of(log_epow,
                                 tables.m_epow.iter().map(|m| m31_u32(*m)).collect())));
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
        // stream JJ (E11): the accumulator window's two limb tables.  Every
        // shared table a device fill bumps must hand its counter over here, or
        // `tables_dev` has nothing to sum for it and stops -- which is what the
        // `ao_acc_hi_bits` table did on the H200's reference chunk ("shared
        // table 4 has no device counter to sum"), the defaults cuda regime
        // being the only place that asks.  `EPOW` is deliberately not here: its
        // column is a host loop on both arms, so it has no device counter.
        pre_counters[ri(p.ao_acc_lo_bits)].push(&d.m_acc_lo);
        pre_counters[ri(p.ao_acc_hi_bits)].push(&d.m_acc_hi);
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
        let hp: Vec<&Eval> = (2..pre_tables.len() - 3)
            .map(|i| crate::devtab::htab(&pre_tables, i)).collect();
        crate::devpre::check_run("shared tables", &hp, &d.pre);
        if !d.mult.is_empty() {
            let hm: Vec<&Eval> = (1..base_mults.len() - 1)
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
                    // stream JJ: the exponent window has no device build, so
                    // its three columns and its multiplicity are uploaded
                    let n = pre_tables.len();
                    pre.extend(pre_tables[n - 3..].iter()
                        .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
                    if r.mult.is_empty() {
                        mult.extend(base_mults[1..].iter()
                            .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
                    } else {
                        mult.extend(r.mult);
                        let m = base_mults.len();
                        mult.extend(base_mults[m - 1..].iter()
                            .map(|e| crate::device::upload_col(crate::devtab::hone(e))));
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
    let glue_tamper_fired =
        tamper.glue.map(|_| glue_regions.iter().map(|r| r.fired).sum::<usize>());
    Gen { plan, tables, band_committed, band_dev, band_pre, aor, mat_regions, walk_regions,
          glue_regions, glue_tcols, glue_mcols, glue_b1, am_region, emb_region, rq_regions,
          q_regions, n_deq_base, pre_tables, base_mults, pre_logs, mult_logs, tab_pre, tab_mult,
          m4e_tabdev, trace_secs, pre_key, layer_boundary_tamper: tamper.layer_boundary, glue_tamper_fired }
}

/// B7b (README 26): the second half -- the session, the three commits, the
/// interaction, the components and the proof.
pub fn prove_gen(layer: &Layer, params: &Params, g: Gen) -> Composed {
    let Gen { plan, tables, band_committed, band_dev, band_pre, aor, mat_regions, walk_regions,
              glue_regions, glue_tcols, glue_mcols, glue_b1, am_region, emb_region, rq_regions,
              q_regions, n_deq_base, pre_tables, base_mults, pre_logs, mult_logs, tab_pre, tab_mult,
              m4e_tabdev, trace_secs, pre_key, layer_boundary_tamper, glue_tamper_fired } = g;
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
    // stream II: the requantize seams join both trees before the two ends, in `Plan::pre_ids`'
    // order -- every seam's four selectors, then the division and scale tables
    // per variant, then the sign table.  The tables' own multiplicity columns
    // follow in tree 1, in the same order, matching the component list.
    let rq_tab_pre: Vec<Eval> = plan.rq_vars.iter()
        .flat_map(|v| {
            let mut c = crate::requant::rq_tab_cols(*v);
            c.extend(crate::requant::scl_tab_cols(*v));
            c
        })
        .chain(if rq_regions.is_empty() && q_regions.is_empty() { vec![] }
               else { crate::requant::rqv_tab_cols() })
        // stream NN-3: and the quantize family's own two, last
        .chain(plan.q_tabs.iter().flat_map(|w| crate::rope::qdiv_tab_cols(*w)))
        .chain(if plan.quants.is_empty() { vec![] }
               else { crate::rope::qnz_tab_cols(plan.q_nz_bits) })
        .collect();
    // one multiplicity vector per table, merged across the seams that share it
    let rq_tab_mult: Vec<Eval> = {
        let mut out = vec![];
        for v in &plan.rq_vars {
            let lr = crate::requant::rq_log_rows(*v);
            let mut m = vec![0u32; 1usize << lr];
            let mut ms = vec![0u32; 1usize << 15];
            for r in rq_regions.iter().filter(|r| r.v == *v) {
                for (a, b) in m.iter_mut().zip(&r.m_rq) {
                    *a += *b;
                }
                for (a, b) in ms.iter_mut().zip(&r.m_scl) {
                    *a += *b;
                }
            }
            out.push(crate::common::eval_of(lr, m.iter().map(|x| m31_u32(*x)).collect()));
            out.push(crate::common::eval_of(15, ms.iter().map(|x| m31_u32(*x)).collect()));
        }
        if !rq_regions.is_empty() || !q_regions.is_empty() {
            let mut m = vec![0u32; 512];
            for r in &rq_regions {
                for (a, b) in m.iter_mut().zip(&r.m_rqv) {
                    *a += *b;
                }
            }
            for r in &q_regions {
                for (a, b) in m.iter_mut().zip(&r.m_rqv) {
                    *a += *b;
                }
            }
            out.push(crate::common::eval_of(9, m.iter().map(|x| m31_u32(*x)).collect()));
        }
        // stream NN-3: the quantize family's own two, in `Plan::pre_ids`' order.
        // A division table is shared by every side of its width; the `(q, nz)`
        // table by all three regions.
        for w in &plan.q_tabs {
            let lr = crate::rope::qdiv_log_rows(*w);
            let mut m = vec![0u32; 1usize << lr];
            for r in q_regions.iter().filter(|r| r.bits == *w) {
                for (a, b) in m.iter_mut().zip(&r.m_rq) {
                    *a += *b;
                }
            }
            out.push(crate::common::eval_of(lr, m.iter().map(|x| m31_u32(*x)).collect()));
        }
        if !plan.quants.is_empty() {
            let lr = crate::rope::qnz_log_rows(plan.q_nz_bits);
            let mut m = vec![0u32; 1usize << lr];
            for r in &q_regions {
                for (a, b) in m.iter_mut().zip(&r.m_qnz) {
                    *a += *b;
                }
            }
            out.push(crate::common::eval_of(lr, m.iter().map(|x| m31_u32(*x)).collect()));
        }
        out
    };
    for r in &rq_regions {
        pre_all.extend(r.pre.iter().map(AnyCol::Host));
        base_all.extend(r.base.iter().map(AnyCol::Host));
    }
    // stream NN: the quantize regions, between the seams and the shared tables
    for r in &q_regions {
        pre_all.extend(r.pre.iter().map(AnyCol::Host));
        base_all.extend(r.base.iter().map(AnyCol::Host));
    }
    pre_all.extend(rq_tab_pre.iter().map(AnyCol::Host));
    base_all.extend(rq_tab_mult.iter().map(AnyCol::Host));
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
    // B12 stream GG: after every fill and upload, before the transcript --
    // what the device-generated columns and their staging hold.
    crate::device::mem_mark("after_fill");
    let m4_t = std::time::Instant::now();
    let tw = twiddles(widest);
    crate::util::m4_add("wall/twiddles", m4_t.elapsed().as_secs_f64());
    crate::device::mem_mark("after_twiddles");
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
    // Stream PP (E11): the one flat preprocessed list splits into the two
    // committed trees -- the shape tree, which the verifier rebuilds from the
    // statement, then the model tree, whose root is the published per-layer
    // model commitment.  The split is `pre_split`'s classification applied to
    // `pre_all` in `pre_ids`' own order, so there is no second column list to
    // drift.  Both are committed before the base trace, because their cells
    // enter the LogUp numerators.
    let is_model = plan.pre_is_model();
    assert_eq!(is_model.len(), pre_all.len(),
               "pre_split disagrees with the committed preprocessed trees");
    let (mut shape_all, mut model_all) = (vec![], vec![]);
    for (c, m) in pre_all.into_iter().zip(&is_model) {
        if *m { model_all.push(c) } else { shape_all.push(c) }
    }
    let shape_sizes: Vec<u32> = shape_all.iter().map(|c| c.log_size()).collect();
    let model_sizes: Vec<u32> = model_all.iter().map(|c| c.log_size()).collect();
    // Stream LL (E11): the classification is the same list, in the same order,
    // at the same heights -- asserted here so `pre_split` and the two trees
    // cannot drift apart, on every shape any gate runs.
    assert_eq!(shape_sizes, plan.shape_sizes(),
               "pre_split disagrees with the committed shape tree");
    assert_eq!(model_sizes, plan.model_sizes(),
               "pre_split disagrees with the committed model tree");
    let commit_one = |sess: &mut Session, cols: Vec<crate::device::AnyCol>| {
        if m4_refs {
            sess.commit_any(cols);
        } else {
            sess.commit(cols.iter().map(|c| c.host().clone()).collect());
        }
    };
    let cached = crate::precache::take(pre_key, &shape_sizes, &model_sizes);
    let shape_t = std::time::Instant::now();
    match &cached {
        Some((shape_tree, _)) => sess.commit_cached(shape_tree.clone(), shape_sizes.clone()),
        None => commit_one(&mut sess, shape_all),
    }
    let pp_shape_secs = shape_t.elapsed().as_secs_f64();
    crate::util::m4_add("wall/commit_shape", pp_shape_secs);
    // Stream PP (E11): the verifier's own work stops here.  Everything after
    // this line reads the model; the shape root does not.
    if SHAPE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        crate::util::emit("shape_root", json!({
            "pre_root": sess.tree_root(0),
            "layer": layer.layer,
            "block": layer.block,
            "c": p.c,
            "s": p.s,
            "log_table": p.log_table,
            "columns": shape_sizes.len(),
            "cells": shape_sizes.iter().map(|s| 1usize << s).sum::<usize>(),
            "split": plan.pre_split_report(),
            "trace_secs": trace_secs,
            "commit_shape_secs": pp_shape_secs,
            "peak_rss_bytes": crate::util::peak_rss_bytes(),
        }));
        std::process::exit(0);
    }
    match cached {
        Some((_, model_tree)) => {
            sess.commit_cached(model_tree, model_sizes.clone());
            crate::util::m4_add("wall/commit_pre_cached", m4_t.elapsed().as_secs_f64());
        }
        None => {
            commit_one(&mut sess, model_all);
            crate::precache::fill(pre_key, &sess);
        }
    }
    let ll_pre_secs = m4_t.elapsed().as_secs_f64();
    crate::util::m4_add("wall/commit_pre", ll_pre_secs);
    // Stream LL (E11), `notes/witness-slack.md` section 5 item 1: `zk-prover
    // expect-root` wants tree 0's root and nothing after it.  The stop is here
    // rather than in a second builder so that the derived root is the prover's
    // own by construction -- there is no second implementation of tree 0 to
    // drift.  `experiments/b13/ll/gate.sh` step 3 is the control that says the
    // two agree: the same shape through `compose` and through `expect-root`.
    if PRE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        let root = sess.tree_root(0);
        // Stream PP (E11): the two preprocessed roots the split produced.  The
        // shape root is what the verifier derives; the model root is what it
        // compares against the published manifest.
        let model_root = sess.tree_root(1);
        crate::util::emit("expect_root", json!({
            "pre_root": root,
            "model_root": model_root,
            "model": layer.model_hash(),
            "layer": layer.layer,
            "block": layer.block,
            "source": layer.source,
            "c": p.c,
            "s": p.s,
            // Stream LL box half: `master_table_bits` is the one shape field that
            // is neither the statement's nor the model's -- it comes off each
            // extraction's own manifest -- so the row carries it and
            // `model-roots` pins it per layer across blocks.
            "log_table": p.log_table,
            // whether this row was derived from the model half of the dump alone
            // (`--weights-only`), which is the property that makes it a
            // *verifier's* derivation and not a second prover's
            "weights_only": layer.source.starts_with("B6.5 model-only"),
            "split": plan.pre_split_report(),
            "trace_secs": trace_secs,
            "commit_pre_secs": ll_pre_secs,
            "peak_rss_bytes": crate::util::peak_rss_bytes(),
        }));
        std::process::exit(0);
    }
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
        // stream JJ, in `Rels::draw`'s order
        let epow = sess.draw_relation(FracRel::draw);
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
        // stream II: last, after the embedding's, so the flag-off draw order is
        // the one every recorded digest was taken under
        let rq = plan.needs_rq_rels().then(|| RqRels {
            rqv: sess.draw_relation(FracRel::draw),
            rq3: sess.draw_relation(FracRel::draw),
            rq4: sess.draw_relation(MulRel::draw),
            scl2: sess.draw_relation(PairRel::draw),
            scl3: sess.draw_relation(FracRel::draw),
        });
        // stream NN-3: and the quantize family's own tables, last of all
        let qq = (!plan.q_tabs.is_empty()).then(|| QuantRels {
            div: plan.q_tabs.iter().map(|_| sess.draw_relation(FracRel::draw)).collect(),
            qnz: sess.draw_relation(PairRel::draw),
        });
        Rels { exp, norm, frac, mul, qnz, epow, widths: widths.clone(), ranges, glue, emb, rq,
               qq }
    };
    let n_ch = plan.n_challenges();
    let ch = sess.draw_felts(n_ch);
    let (r, s, rv) = (ch[0], ch[1], ch[2]);
    let (w_glue, lam_glue) = if plan.has_bindings() {
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
    .chain(std::iter::once(("table/epow".to_string(), plan.log_epow())))
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
        //
        // Open item 1 (E11, H200 2026-09-02): block 0's solo corner carries the
        // region and so has a row component, but it is still a corner -- its
        // `KeyEval` reads the causal mask (tree 0, `msk_id`) for the exp
        // lookups' multiplicities, which the device fraction route never
        // supplies (`devfrac::band_cols` hands the bands `Cols::Unread`, 19.2's
        // measurement over prefix bands).  It takes the host route like every
        // other corner; the row index still advances past its row component.
        let air = plan.rows_band[b]
            .then(|| {
                let ri = row_i;
                row_i += 1;
                ri
            })
            .filter(|_| !a.corner)
            .map(|ri| crate::devfrac::BandAir {
                tag: b,
                chan: &m4e_air.chans[b].eval,
                key: &m4e_air.keys[b].eval,
                row: &m4e_air.rows[ri].eval,
                dev: band_dev[b].as_ref(),
                pre: band_pre[b].as_ref(),
            });
        // B11 stream L, anatomy: the LogUp build apart from `inter_mark`'s
        // accounting, which is the only other thing in this loop. Whatever
        // `int/bands` minus `int/bands_build` comes to is the accounting.
        let t_ib = std::time::Instant::now();
        if let Some(g) = gkr.as_mut() {
            g.reg = crate::gkrlink::Reg::Band(b);
        }
        // stream NN: the corner bands' own K, Q and V claims.  The weights are
        // the verifier's own; the claims are what this pass computes, so the
        // builder is handed a blank statement and its `claimed` fields are
        // unused here.
        let nn_t = nn_band_bindings(&plan, b, w_glue, &Claims::default());
        // stream NN (NN-2): and a prefix band's Q claim alone
        let nn_qt = nn_band_qtie(&plan, b, w_glue, &Claims::default());
        let nn_bind = match (nn_t.as_ref(), nn_qt.as_ref()) {
            (Some((k, q, v)), _) =>
                Some(attn_trace::NnBind { k: Some(k), q, v: Some(v) }),
            (None, Some(q)) => Some(attn_trace::NnBind { k: None, q, v: None }),
            (None, None) => None,
        };
        let it = attn_trace::interaction_phase(a, l, &plan.cfg, &pow, &rels.exp, &rels.ranges,
                                               &band_committed[b], &plan.widths, false,
                                               plan.rows_band[b], air.as_ref(),
                                               gkr.as_deref_mut(), nn_bind.as_ref());
        crate::util::m4_add("int/bands_build", t_ib.elapsed().as_secs_f64());
        inter_all.extend(it.cols);
        inter_mark(&format!("band{b}{}", if a.corner { " (corner)" } else { "" }),
                   &inter_all, &mut inter_seen, &mut inter_split);
        claims.bands.push([qm(it.claimed_chan), qm(it.claimed_hs), qm(it.claimed_av),
                           qm(it.claimed_sn), qm(it.claimed_am), qm(it.claimed_az)]);
        if it.claimed_nn.len() == 3 {
            claims.band_qkv.push([qm(it.claimed_nn[0]), qm(it.claimed_nn[1]),
                                  qm(it.claimed_nn[2])]);
        } else if it.claimed_nn.len() == 1 {
            // stream NN (NN-2): a prefix band publishes Q alone
            claims.band_q.push(qm(it.claimed_nn[0]));
        }
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
    // B11 stream S, lever 3: one univariate IOP per height bindings the GKR's
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
                    // stream OO: `Prog::pre_at` is where the pair sits in the
                    // region's own preprocessed list, the same index the fill
                    // reads through `glue::src_cols_any`'s `PRE_TAG`
                    (RCol::GluePre(i, k), Some(d)) =>
                        Src::Dev(d.pre[r.prog.pre_at(*i, *k)].device_ptr),
                    (RCol::GluePre(i, k), None) =>
                        Src::Host(&r.pre[r.prog.pre_at(*i, *k)]),
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
                    (RCol::AoPre(sl), Some(d)) => Src::Dev(d.pre[*sl].device_ptr),
                    (RCol::AoPre(sl), None) => Src::Host(&aor.pre[*sl]),
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
    claims.ao_q = vec![qm(aoi.claimed_q)];
    if !p.gkr_attnout {
        claims.use_sums.push(qm(aoi.use_sum));
    }
    crate::util::m4_add("int/attnout", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    claims.deq_sc = vec![[0u32; 4]; if plan.rqs.is_empty() { 0 } else { layer.mats.len() }];
    for (mi, (m, reg)) in layer.mats.iter().zip(&mat_regions).enumerate() {
        // stream E: the output side rides on the wiring family's binding challenge,
        // so it exists exactly when that family does and the plan says this
        // matmul publishes one
        let ow = plan.has_output_claim(&m.name).then(|| {
            plan.output_claim_of(w_glue, lam_glue, &m.name, m.tokens, m.n_out, reg.lay.per_row)
        });
        // stream II (E11): the x-side scale's fold, on the matmuls a seam feeds.
        //
        // Stream SS: not on a folded one.  This claim folds the region's *shared*
        // columns as `(ms, es)`, the language the seam's own scale claim is in,
        // and a fold region's second shared column is not `es` -- it is
        // `dent = 2^(G_t - alpha)`, the token half of the divisor.  Folding it
        // would compare two different quantities and the honest proof would not
        // prove.  The binding is named open instead (`rq_scale_open`); closing it
        // wants `es` committed beside `dent` with a power-of-two lookup pinning
        // the one to the other, which is a column and a table this stream did
        // not add.
        let sc = (plan.rqs.iter().any(|rp| plan.walks[rp.walk].mats.contains(&mi))
                  && plan.specs[mi].variant != DeqVariant::Fold)
            .then_some((w_glue, lam_glue));
        let it = mat_interaction(m, reg, rv, s, &rels, ow.as_ref(), p.gkr_dequant, sc);
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
                claims.logits_claim.push(qm(c));
            } else {
                claims.deq_out.push(qm(c));
            }
        }
        if let Some(c) = it.claimed_sc {
            claims.deq_sc[mi] = qm(c);
        }
    }
    // Stream E: the head's key claim against B1's own `key_head` column.  The
    // proof derives its keys from the head dequant's committed output; this
    // rebuilds the same Horner sum from the dump's column and compares the two
    // field elements, which is 512 x vocab values checked at once.
    let mut head_key_b1 = json!(null);
    if plan.p.logits_claim {
        if let (Some(mi), Some(keys)) =
            (layer.mats.iter().position(|m| m.name == "lm_head"), layer.head_keys.as_ref())
        {
            let (m, lay) = (&layer.mats[mi], plan.mat_lays[mi]);
            let ow = output_claim_w(w_glue, lam_glue, m.tokens, m.n_out, lay.per_row,
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
            let got = claims.logits_claim.first().copied().map(unqm);
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
        // stream II: and the group's claim over its own committed activations,
        // pushed for every group whether or not a seam consumes it -- `o`'s
        // counterparty is stream JJ's to publish, and this is the half of that
        // seam II owns.
        if plan.glue.is_some() {
            claims.walk_qx.push(qm(it.qx_claim));
        }
        if !p.gkr_walks {
            claims.use_sums.push(qm(it.use_sum));
        }
    }
    crate::util::m4_add("int/walks", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    // the glue regions' bindings and lookups, in component order
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let blank = Claims::default();
        let specs = gp.bindings(w_glue, lam_glue, &blank);
        for (i, reg) in glue_regions.iter().enumerate() {
            let t_ig = std::time::Instant::now();
            let it = crate::glue::glue_interaction(reg, gr, Some(&specs[i]),
                                                   &m4e_air.glue[i].eval, p.gkr_glue);
            crate::util::m4_add("int/glue_build", t_ig.elapsed().as_secs_f64());
            inter_all.extend(it.cols);
            inter_mark(&format!("glue/{}", reg.tag), &inter_all, &mut inter_seen,
                       &mut inter_split);
            claims.stream_bindings.push(it.claimed.iter().map(|v| qm(*v)).collect());
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
        // stream JJ: the exponent window, last of the core tables
        {
            let nw = plan.widths.len();
            let lj = plan.log_epow();
            let (c, sum) = crate::devfrac::frac_cols(
                &m4e_air.tables[4 + nw].eval, "table epow", lj,
                tc(&pre_tables, &tab_pre, 17 + nw, 3),
                tc(&base_mults, &tab_mult, 5 + nw, 1),
                || {
                    let refs: Vec<&Eval> =
                        (0..3).map(|i| htab(&pre_tables, 17 + nw + i)).collect();
                    table_col(lj, &rels.epow, &refs, htab(&base_mults, 5 + nw))
                });
            inter_all.extend(c);
            inter_mark("table/epow", &inter_all, &mut inter_seen, &mut inter_split);
            table_regions.push(json!({"region": "table/epow", "log_rows": lj,
                                      "key_cols": 3, "looks": 1}));
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

    // stream II: the requantize seams' interaction columns, before the two ends'.  Within
    // a seam the AIR reads its two Horner accumulators before it finalizes its
    // LogUp, so the binding columns come before the fraction columns here too; the
    // seams' shared tables follow, in `Plan::pre_ids`' order.
    let m4_t = std::time::Instant::now();
    if let Some(rr) = rels.rq.as_ref() {
        for (k, reg) in rq_regions.iter().enumerate() {
            let binding = rq_binding_w(w_glue, lam_glue, &plan.rqs[k],
                               SecureField::zero(), SecureField::zero(),
                               SecureField::zero());
            let (cols, use_sum, cin, cout, csc) = crate::requant::rq_interaction(
                reg, &binding, &rels.qnz, &rr.rq3, &rr.rq4, &rr.scl2, &rr.scl3,
                rels.range(15), &rr.rqv);
            inter_all.extend(crate::device::host_cols(cols));
            claims.rq_in.push(qm(cin));
            claims.rq_out.push(qm(cout));
            claims.rq_sc.push(qm(csc));
            claims.use_sums.push(qm(use_sum));
            inter_mark(&format!("requant/{}", reg.tag), &inter_all, &mut inter_seen,
                       &mut inter_split);
        }
        // stream NN: and the quantize regions, in the same place the components
        // and the preprocessed ids put them
        for (k, reg) in q_regions.iter().enumerate() {
            let qp = &plan.quants[k];
            let binding = q_binding_w(w_glue, lam_glue, qp, p.nh,
                              SecureField::zero(), SecureField::zero());
            let qr = rels.qq.as_ref().expect("the quantize regions draw their own tables");
            let (cols, use_sum, cin, cout) = crate::rope::q_interaction(
                reg, &binding, qp.hd, &qr.qnz, &qr.div[plan.q_tab_at(qp.bits)],
                rels.range(15), &rr.rqv);
            inter_all.extend(crate::device::host_cols(cols));
            claims.nn_in.push(qm(cin));
            claims.nn_out.push(qm(cout));
            claims.use_sums.push(qm(use_sum));
            inter_mark(&format!("quant/{}", reg.tag), &inter_all, &mut inter_seen,
                       &mut inter_split);
        }
        // the table sides, one per preprocessed table, in the same order the
        // ids and the components are in: per variant a division table and a
        // scale table, then the sign table.
        let mut pi = 0usize; // cursor into `rq_tab_pre`
        let mut mi = 0usize; // cursor into `rq_tab_mult`
        use stwo::prover::backend::simd::m31::PackedM31;
        use stwo::prover::backend::simd::qm31::PackedQM31;
        use stwo_constraint_framework::Relation;
        // stream NN-3: the combiner is passed in rather than selected by an
        // index, because the quantize family's tables are relations of their
        // own and the seams' five no longer cover the list
        let side = |lr: u32, ncol: usize, pi: usize, mi: usize,
                    comb: &(dyn Fn(&[PackedM31]) -> PackedQM31 + Sync)| {
            crate::plogup::build(lr, 1, |_, vr| {
                let vals: Vec<PackedM31> = (0..ncol).map(|k| rq_tab_pre[pi + k].data[vr]).collect();
                (-PackedQM31::from(rq_tab_mult[mi].data[vr]), comb(&vals))
            })
        };
        for v in &plan.rq_vars {
            let is_p2 = *v == crate::requant::SVariant::Pow2;
            let lr = crate::requant::rq_log_rows(*v);
            let w = crate::requant::rq_tab_width(*v);
            let (c, sum) = if is_p2 {
                side(lr, w, pi, mi, &|x| rr.rq3.combine(x))
            } else {
                side(lr, w, pi, mi, &|x| rr.rq4.combine(x))
            };
            inter_all.extend(crate::device::host_cols(c));
            claims.table_sums.push(qm(sum));
            pi += w;
            mi += 1;
            let ws = crate::requant::scl_tab_width(*v);
            let (c, sum) = if is_p2 {
                side(15, ws, pi, mi, &|x| rr.scl2.combine(x))
            } else {
                side(15, ws, pi, mi, &|x| rr.scl3.combine(x))
            };
            inter_all.extend(crate::device::host_cols(c));
            claims.table_sums.push(qm(sum));
            pi += ws;
            mi += 1;
        }
        let (c, sum) = side(9, 3, pi, mi, &|x| rr.rqv.combine(x));
        inter_all.extend(crate::device::host_cols(c));
        claims.table_sums.push(qm(sum));
        pi += 3;
        mi += 1;
        // stream NN-3: and the quantize family's own, in `Plan::pre_ids`' order
        if let Some(qr) = rels.qq.as_ref() {
            for (i, w) in plan.q_tabs.iter().enumerate() {
                let lr = crate::rope::qdiv_log_rows(*w);
                let (c, sum) =
                    side(lr, crate::rope::QDIV_COLS, pi, mi, &|x| qr.div[i].combine(x));
                inter_all.extend(crate::device::host_cols(c));
                claims.table_sums.push(qm(sum));
                pi += crate::rope::QDIV_COLS;
                mi += 1;
            }
            let (c, sum) = side(crate::rope::qnz_log_rows(plan.q_nz_bits), 2, pi, mi,
                                &|x| qr.qnz.combine(x));
            inter_all.extend(crate::device::host_cols(c));
            claims.table_sums.push(qm(sum));
        }
        inter_mark("requant/tables", &inter_all, &mut inter_seen, &mut inter_split);
    }
    crate::util::m4_add("int/requant", m4_t.elapsed().as_secs_f64());
    // Stream E: the argmax's interaction columns come *after* the tables' own
    // sides, because the framework hands out interaction spans in component
    // order and the argmax component is last of all.  Appended earlier, this
    // component is handed the exp table's columns instead and the prover dies
    // evaluating a 2^23 polynomial on a 2^16 domain -- which is exactly how it
    // failed the first time.

    // stream E: the argmax's binding column then its lookups, the order the AIR
    // reads them, and its use sum last of all -- the component is last too
    let m4_t = std::time::Instant::now();
    if let (Some(r), Some(mi)) = (am_region.as_ref(), plan.argmax_mat()) {
        let m = &plan.p.mats[mi];
        let binding = crate::argmax::ArgmaxBinding {
            claimed: SecureField::zero(),
            claimed_tok: SecureField::zero(),
            tag: m.name.clone(),
            ..argmax_binding_w(w_glue, m.tokens, m.n_out, plan.mat_lays[mi].per_row)
        };
        let (tcols, claimed, claimed_tok) = crate::argmax::binding_col(r, m.tokens, &binding);
        inter_all.extend(crate::device::host_cols(tcols));
        claims.am_key.push(qm(claimed));
        claims.head_tok.push(qm(claimed_tok));
        let (lcols, use_sum) =
            crate::argmax::logup(r, plan.p.prove_argmax_lax,
                                 rels.range(crate::argmax::KEY_BITS));
        inter_all.extend(crate::device::host_cols(lcols));
        claims.use_sums.push(qm(use_sum));
        inter_mark("argmax", &inter_all, &mut inter_seen, &mut inter_split);
    }
    crate::util::m4_add("int/argmax", m4_t.elapsed().as_secs_f64());

    // stream BB: the embedding gadget's interaction columns, last of all,
    // because its two components are last of all.  Within the use region the
    // AIR reads the two Horner accumulators before it finalizes its LogUp, so
    // the binding columns come before the fraction columns here too; the table
    // region's single column follows, matching the component order.
    let m4_t = std::time::Instant::now();
    if let (Some(r), Some(er)) = (emb_region.as_ref(), rels.emb.as_ref()) {
        let mut binding = crate::embed::binding_w(w_glue, lam_glue, r.d, r.tokens, &r.lay);
        let (tcols, hc, gc) = crate::embed::binding_cols(r, &binding);
        binding.claimed = hc;
        binding.claimed_tok = gc;
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

    // an independent recomputation of every binding claim, from the region's witness
    // rather than from its trace: a binding is only worth something if it is checkable
    let stream_bindings_ok = match (&plan.glue, &rels.glue) {
        (Some(gp), Some(_)) => {
            let specs = gp.bindings(w_glue, lam_glue, &claims);
            glue_regions.iter().zip(&specs).all(|(reg, binding)| {
                binding.specs.iter().all(|sp| reg.binding_claim(sp, lam_glue) == sp.claimed)
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
        // stream NN: the angle frequencies the STATEMENT derives against B1's
        // own fp64 vector, when the dump carries it.  A reference and never an
        // input -- what tree 0 holds is `RopeParams::inv`'s answer, so this
        // number says whether this repo's `inv_freq_of` is torch's.
        "rope_inv_vs_b1": match (&p.rope, layer.rope.as_ref().and_then(|r| r.inv_ref.as_ref())) {
            (Some(rp), Some(refv)) => {
                let inv = rp.inv(p.hd);
                // `mismatch` is the f64 vector's; `cell_mismatch` is the bf16
                // table the region commits, which is the one that is asserted.
                let cells: usize = (0..p.s)
                    .into_par_iter()
                    .map(|t| {
                        let a = crate::rope::pre_row(&inv, p.hd, rp.pos0 + t);
                        let b = crate::rope::pre_row(refv, p.hd, rp.pos0 + t);
                        a.iter().zip(&b).filter(|(x, y)| x != y).count()
                    })
                    .sum();
                json!({"values": inv.len(), "rope_type": rp.rope_type,
                       "mismatch": inv.iter().zip(refv).filter(|(a, b)| a != b).count(),
                       "cells": 2 * p.hd * p.s, "cell_mismatch": cells})
            }
            _ => json!(null),
        },
        // Stream NN-3: `s_v` is used at two sites and the statement now says so.
        // The `v` quantize divides by the per-(head, channel) exponent it
        // commits to tree 0 and the attention output rescales by `jv - jo` off
        // its own tree-0 column; in B1 both are one `s_v`.  The two columns tile
        // the grid differently -- the attention output's rows are
        // (head, token, chunk) and the region's are (token, channel block) --
        // so the equality is per (head, channel) and not per row, and a GQA
        // group's `rep` query heads all read the kv head's one exponent.  It
        // costs **no proof cells**: both sides are preprocessed, so a prover who
        // rescales by one scale and quantizes by another commits a tree 0 that
        // is not the statement's and `root_ok` rejects it.  `mismatch` is the
        // count of (head, channel) sites where the two disagree.
        "sv_binding": match q_regions.iter().zip(&plan.quants).find(|(_, qp)| qp.name == "v") {
            Some((r, qp)) => {
                let (nh, hd) = (layer.ao.nh, layer.ao.hd);
                let mut bad = 0usize;
                for h in 0..nh {
                    for ch in 0..hd {
                        let kv = h / qp.rep;
                        if i64::from(layer.ao.jv[h * hd + ch]) != r.je[kv * qp.hd + ch] {
                            bad += 1;
                        }
                    }
                }
                // and the dump's own two loads of `jv.i32`, which is what a
                // real chunk's binding rests on
                let refbad = layer.rope.as_ref()
                    .map(|rc| (0..nh * hd).filter(|i| rc.jv[*i] != layer.ao.jv[*i]).count());
                json!({"sites": nh * hd, "mismatch": bad, "ok": bad == 0,
                       "rope_jv_mismatch": refbad})
            }
            None => json!(null),
        },
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
        // stream NN: what each quantize region costs, and the number that
        // matters -- how many of the integers it derived from the rotation's
        // output equal the value the *band* committed.  A nonzero `clamped` is
        // an incompleteness and not a disagreement: B1 clamps at this side's own
        // `qmax` and the region's division table has no row past it, so a
        // clamped site has no honest witness here at all.  Stream NN-3: the
        // table is the side's own, so `clamped` counts B1's clamp and nothing
        // else -- before it, at `qk_bits (12, 11)`, it counted "the region is
        // narrower than the model" as well.
        "quant": q_regions.iter().zip(&plan.quants).map(|(r, qp)| json!({
            "region": r.tag,
            "bits": r.bits, "qmax": crate::rope::qmax_of(r.bits),
            "div_rows": crate::rope::qdiv_table(r.bits).len(),
            "heads": qp.heads, "rep": qp.rep, "n": qp.n, "tokens": qp.tokens,
            "layout": {"per_row": r.lay.per_row, "chunks": r.lay.chunks,
                       "log_rows": r.lay.log_rows, "lead": r.lay.lead},
            "columns": {"base": r.base.len(), "preprocessed": r.pre.len()},
            "cells": {"base": r.base_cells(), "preprocessed": r.pre_cells()},
            "cells_per_value": r.base_cells() as f64 / (r.tokens * r.n) as f64,
            "checked": r.ref_n, "matches_band": r.ref_ok,
            "mismatch": r.ref_n - r.ref_ok,
            "clamped": r.clamped, "off_table": r.off_table,
        })).collect::<Vec<_>>(),
        // stream II: what each requantize seam costs and, the number that
        // matters, how many of the integers it *derived* from the producer's
        // output equal the activation the walk actually committed.  On a real
        // dump that is a B1 reference check: a non-zero count is the two sides
        // disagreeing about the forward pass, which the binding then rejects.
        "requant": rq_regions.iter().zip(&plan.rqs).map(|(r, rp)| json!({
            "seam": r.tag,
            "producer": {"region": plan.glue.as_ref().map(|g| g.regions[rp.producer.0].tag.clone()),
                         "binding": rp.producer.1},
            "variant": r.v.name(),
            "n": r.n, "tokens": r.tokens,
            "layout": {"per_row": r.lay.per_row, "chunks": r.lay.chunks,
                       "log_rows": r.lay.log_rows, "lead": r.lay.lead},
            "columns": {"preprocessed": r.pre.len(), "base": r.base.len()},
            "cells": {"preprocessed": r.pre_cells(), "base": r.base_cells()},
            "cells_per_value": r.base_cells() as f64 / (r.tokens * r.n) as f64,
            "off_table": r.off_table,
            "checked": r.ref_n, "matches_walk_qx": r.ref_ok,
            "mismatch": r.ref_n - r.ref_ok,
        })).collect::<Vec<_>>(),
        "requant_tables": {
            "preprocessed": rq_tab_pre.iter().map(|e| e.domain.size()).sum::<usize>(),
            "multiplicity": rq_tab_mult.iter().map(|e| e.domain.size()).sum::<usize>(),
        },
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
    let layer_boundary_report = if plan.layer_sides.is_empty() {
        json!(null)
    } else {
        for (i, side) in plan.layer_sides.iter().enumerate() {
            let tam = layer_boundary_tamper.filter(|(s, _)| *s == i).map(|(_, c)| c);
            let cols = crate::boundary::cols(&glue_regions[side.region], side, tam);
            sess.commit(cols);
        }
        crate::boundary::report(&plan.layer_sides)
    };
    // Stream AA: the K/V sides, in the same position and for the same reason,
    // right after stream H's.  A side's columns are the *very columns tree 1
    // committed*, cloned band by band in component-registration order --
    // `[band's K][band's V]` -- so byte-for-byte with the region is by
    // construction and not by a second copy of `attn_trace`'s index map.
    //
    // Open item 18: the columns are wherever the band's fill left them.  Under
    // the CUDA default `B7_M4_WFILL_BAND=1` a band's committed columns are on
    // the card (`band_dev[b].base`, `commit_base_dev`) and `com.base` is empty;
    // tree 1 committed them from there as `AnyCol::Dev` references (the
    // `base_all` loop above), and a boundary side takes the same references
    // to the same buffers -- `device::take_all` clones a borrowed device column
    // on the device, so the side sees the bytes tree 1 saw and nothing crosses
    // the bus.  A host-filled band (the reference arm, `B7_M4_WFILL_BAND=0`,
    // every corner band) hands its host columns over exactly as before.  The
    // one reader that needs the cells on the host is the tamper probe, which
    // bumps one cell of one column: that side comes down (`AnyCol::download`)
    // and commits through the host path, so `boundary::kv_cols` is unchanged.
    let kv_report = if plan.kv.is_empty() {
        json!(null)
    } else {
        for (i, side) in plan.kv.iter().enumerate() {
            let mut cols: Vec<AnyCol> = vec![];
            for b in &side.bands {
                let (com, l) = (&band_committed[*b], &plan.lays[*b]);
                let at = &com.at;
                let n_aux = if plan.attn_variant == AttnVariant::IntExp { 0 } else { l.key.q };
                let v0 = at[1] + 2 * l.key.q + n_aux;
                let kv = (at[0]..at[0] + l.chan.pk).chain(v0..v0 + l.key.pv);
                match band_dev[*b].as_ref() {
                    Some(d) => cols.extend(kv.map(|j| AnyCol::Dev(&d.base[j], d.logs[j]))),
                    None => {
                        assert!(!com.base.is_empty(),
                                "--kv-boundary: band {b} has its committed columns on neither side \
                                 of the bus, so the K/V boundary tree has nothing to copy");
                        cols.extend(kv.map(|j| AnyCol::Host(&com.base[j])));
                    }
                }
            }
            assert_eq!(cols.len(), side.columns, "a K/V side's width disagrees with its plan");
            // the tamper indexes the sides as one list, stream H's first
            let tam = layer_boundary_tamper
                .filter(|(s, _)| *s == plan.layer_sides.len() + i)
                .map(|(_, c)| c);
            match tam {
                None => sess.commit_any(cols),
                Some(_) => {
                    let host: Vec<Eval> = cols.into_iter().map(AnyCol::download).collect();
                    sess.commit(crate::boundary::kv_cols(host, tam));
                }
            }
        }
        crate::boundary::kv_report(&plan.kv)
    };
    crate::util::m4_add("wall/commit_boundary", m4_t.elapsed().as_secs_f64());
    // B12 stream GG: the boundary trees, one mark for all of them.
    sess.mark_boundary();
    sess.mix(&claims.flat());

    // ---- components, prove ----------------------------------------------------
    let m4_t = std::time::Instant::now();
    let built = build_components_linked(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue,
                                        &claims, link_hl.as_deref());
    crate::util::m4_add("wall/build_components", m4_t.elapsed().as_secs_f64());
    // Stream EE (open item 21): `--fp-out FILE` writes the fingerprint set this
    // chunk will look up, for the pre-run comparison against `cudaemit
    // --fp-out` on the same plan line. `--fp-only` stops here: the components
    // exist and their fingerprints are the whole answer, so the report is
    // written and the process ends without proving.
    if let Some((path, only)) = crate::cudagen::fp_out() {
        let rows = built.fp_rows();
        crate::cudagen::fp_write(path, "compose", &layer.source, &rows);
        if *only {
            eprintln!("--fp-only: fingerprints written, proving skipped");
            std::process::exit(0);
        }
    }
    let sizes: Vec<Vec<u32>> = sess
        .ops
        .iter()
        .filter_map(|op| match op {
            crate::common::Op::Tree(sz) => Some(sz.clone()),
            _ => None,
        })
        .collect();
    // stream PP: the shape tree and the model tree are both preprocessed, so
    // the reported preprocessed width is their sum; the base tree is at 2.
    let n_columns = [sizes[0].len() + sizes[1].len(), sizes[2].len(), n_inter];
    // B7a milestone 3: the lowering, off the prove loop's critical path.
    #[cfg(feature = "cuda")]
    if crate::util::m3_switch("B7_M3_PREWARM") {
        built.prewarm_blobs();
    }
    let provers = built.provers();
    let m4_t = std::time::Instant::now();
    let (res, stats) = sess.prove_raw(&provers);
    crate::util::m4_add("wall/prove_raw", m4_t.elapsed().as_secs_f64());
    // B12 stream GG: proof in hand, the session consumed -- what `prove`
    // freed on return, and what the generation still holds.
    crate::device::mem_mark("after_prove");
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
        // stream PP: the model commitment this chunk proves against
        model_root: proof
            .as_ref()
            .map(|pf| format!("{:?}", pf.commitments[1]))
            .unwrap_or_default(),
        // stream KK: the chunk's own coordinate, off the dump's manifest (or
        // the `--layer` / `--block` overrides a plan line carries for a chunk
        // whose manifest cannot know where the grid puts it, the `lm_head`
        // chunk being the one that cannot).
        layer: layer.layer,
        block: layer.block,
        // stream LL: the content hash of this chunk's model half, so a grid can
        // compare "which model" as well as "which layer".
        model: Some(layer.model_hash()),
        // stream KK: the ids the embedding region committed, one per position,
        // and the tokens the argmax region emitted -- published so that a
        // reader of two chunks can compare them position by position.  Each is
        // pinned inside its own proof by the Horner claim beside it.
        tok_in: emb_region.as_ref().map(|r| r.token_ids()).unwrap_or_default(),
        tok_out: am_region.as_ref()
            .map(|r| r.token_ids(p.mats[plan.argmax_mat().expect("the head matmul")].tokens))
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
        stream_bindings_ok,
        layer_boundary: layer_boundary_report,
        kv_boundary: kv_report,
        glue_tamper_fired,
    }
}

// ---------------------------------------------------------------- verifying ----

#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub stark_ok: bool,
    pub root_ok: bool,
    /// Stream PP (E11): the proof's second root is the one the statement
    /// declares as its model commitment.
    pub model_root_ok: bool,
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
    /// stream E, one per dequant output binding: the region's own output claim equals
    /// the wiring family's claim over the same values
    pub deq_out_ok: Vec<bool>,
    /// Stream JJ: the dequant outputs that publish a claim with no counterparty
    /// in this statement -- `q`, `k` and `v`, whose consumers reach the
    /// attention through RoPE and a per-channel requantize that no region
    /// proves.  Reported so a reader cannot mistake seven published claims for
    /// seven checked bindings; it is not part of `accepts`.
    pub deq_out_open: Vec<String>,
    /// stream E: the argmax's key claim equals the head dequant's -- "the token
    /// this row emits is the maximum of the logits *this layer computed*"
    pub head_binding_ok: bool,
    /// stream BB: the embedding region's claim equals the wiring family's
    /// `stream_in` claim -- "the residual stream this layer consumed is row
    /// `tok[t]` of the pinned embedding matrix".  True when the gadget is off.
    pub embed_ok: bool,
    /// stream KK (`notes/witness-slack.md` §1b): the ids the statement publishes
    /// are the ids the proof committed -- the Horner sum of `fixed.tok_in`
    /// under this chunk's own replayed `w` equals `claims.embed_tok`.  A
    /// probabilistic binding over a post-commitment challenge (§0), not a
    /// constraint.  True when the gadget is off *and* the statement publishes
    /// no ids; an embedding chunk that publishes none is rejected.
    pub embed_tok_ok: bool,
    /// stream KK (§5 item 2): the same for the head chunk's emitted tokens --
    /// `fixed.tok_out` against `claims.head_tok`.  True when the argmax is
    /// off and nothing is published.
    pub head_tok_ok: bool,
    /// stream II: per requantize seam, "the matmul's input is the producer's
    /// output", and "the walk multiplied the integers this seam derived"
    pub rq_in_ok: Vec<bool>,
    pub rq_out_ok: Vec<bool>,
    /// Stream II (E11): per (seam, consuming matmul) pair, in plan order -- "the
    /// scale this matmul dequantized by is the scale the requantize pinned to
    /// its input's attained maximum".  Empty on a chunk with no seams.
    pub rq_scale_ok: Vec<bool>,
    /// Stream SS: the `(seam, matmul)` pairs whose consumer is a `Variant::Fold`
    /// region.  The fold's second shared column is the divisor's token half and
    /// not the scale's exponent, so the claim cannot be stated in the seam's
    /// language; these are named rather than counted.
    pub rq_scale_open: Vec<String>,
    /// Stream II (E11): true when this chunk publishes both the attnout
    /// region's `ao_q` and the `o` walk's `qX` but its `o` matmul does not read
    /// `nh * hd` channels -- so the two folds are over different sequences and
    /// `ao_q_ok` says nothing.  A desk shape, never a real layer; reported so a
    /// reader cannot mistake a vacuous `ao_q_ok` for a checked seam.
    pub ao_q_open: bool,
    /// stream II: the `o` seam, once stream JJ publishes `ao_q`.  True while
    /// that claim is absent -- an equality nobody can state yet is not a
    /// failure, and the record says so rather than the verdict pretending.
    pub ao_q_ok: bool,
    /// Stream NN, per quantize region in `q`, `k`, `v` order: "the value this
    /// region rounded is the value the rotation produced" (or, for `v`, the
    /// value the matmul's dequant published).
    pub nn_in_ok: Vec<bool>,
    /// and per region: "the integers this region produced are the columns the
    /// bands committed" -- the sum of the corner bands' own claims, which is
    /// what closes `witness-slack.md` section 1e inside one chunk.
    pub nn_out_ok: Vec<bool>,
    /// Stream NN: the statement's `pos0` is the block coordinate `Fixed::block`
    /// declares.  The rotation's `cos`/`sin` are preprocessed and read the
    /// token's *absolute* position, so tree 0 is a function of the block; this
    /// is what stops a chunk from proving one block's angles while the chain
    /// puts it at another.  True when the statement declares no block.
    pub nn_pos_ok: bool,
    /// Stream NN (NN-2), per prefix band of `Plan::nn_qbands`: "this band's own
    /// Q column is its group's corner's".  The corner's Q is bound to the
    /// rotated query (`nn_out_ok`), so this carries the binding to the band that
    /// attends to the cache -- which NN-1 left untied, and asymmetrically so.
    pub nn_q_ok: Vec<bool>,
    /// Stream NN: true when the chunk carries the wiring family but no rotation
    /// -- a shape whose q/k/v widths do not line up with its bands.  Reported so
    /// a reader cannot mistake an absent family for a checked one.
    pub nn_open: bool,
    /// the layer's committed residual-stream input and output claims
    pub stream_in: Option<[u32; 4]>,
    pub stream_out: Option<[u32; 4]>,
    /// Stream AA: every boundary tree's root, stream H's two first and then the
    /// K/V sides in `plan.kv`'s order.  A verifier reading one proof alone gets
    /// them here so it never has to re-derive anything; `boundary::chain_check`
    /// reads the same roots straight out of `proof.commitments`.
    pub boundary_roots: Vec<String>,
    /// Stream EE: the same roots, one entry per boundary side and each labelled
    /// with what the side is -- its tree, its axis, whether it is the in or the
    /// out side, and for a K/V side its head band group.  This replaces stream
    /// H's `boundary_in`/`boundary_out`, which named trees 3 and 4 by position and so
    /// only made sense while a statement carried exactly one boundary pair.  A
    /// statement now carries as many sides as its plan lists, and a reader asks
    /// for a side by its label (`Verdict::side_root`) rather than by a
    /// hard-coded index.
    pub boundary_sides: Vec<crate::boundary::SideRoot>,
    pub error: Option<String>,
    pub verify_secs: f64,
}

impl Verdict {
    /// The root of the boundary side with the given label, if the statement
    /// committed one: the residual pair is `(Axis::Layer, "in")` and
    /// `(Axis::Layer, "out")`; a K/V side is `(Axis::Token, "out")` or
    /// `(Axis::Token, "in")` with its group.
    pub fn side_root(&self, axis: crate::boundary::Axis, role: &str, group: Option<usize>)
        -> Option<&str> {
        self.boundary_sides
            .iter()
            .find(|s| s.axis == axis && s.role == role && s.group == group)
            .map(|s| s.root.as_str())
    }
    pub fn accepts(&self) -> bool {
        self.stark_ok && self.root_ok && self.model_root_ok
            && self.sums_cancel && self.av_ok && self.den_link_ok
            && self.sizes_ok && self.qk_ok.iter().all(|x| *x) && self.den_ok.iter().all(|x| *x)

            && self.walk_ok.iter().all(|x| *x) && self.glue_ok.iter().all(|x| *x)
            && self.deq_out_ok.iter().all(|x| *x) && self.head_binding_ok && self.embed_ok
            && self.embed_tok_ok && self.head_tok_ok
            && self.rq_in_ok.iter().all(|x| *x) && self.rq_out_ok.iter().all(|x| *x)
            && self.rq_scale_ok.iter().all(|x| *x) && self.ao_q_ok
            && self.nn_in_ok.iter().all(|x| *x) && self.nn_out_ok.iter().all(|x| *x)
            && self.nn_pos_ok && self.nn_q_ok.iter().all(|x| *x)
    }
    fn rejected(msg: &str) -> Self {
        Self { stark_ok: false, root_ok: false, model_root_ok: false,
               sums_cancel: false, qk_ok: vec![],
               den_ok: vec![], av_ok: false, den_link_ok: false, walk_ok: vec![],
               sizes_ok: false, glue_ok: vec![false], deq_out_ok: vec![false],
               deq_out_open: vec![], head_binding_ok: false, embed_ok: false,
               embed_tok_ok: false, head_tok_ok: false,
               rq_in_ok: vec![false], rq_out_ok: vec![false], rq_scale_ok: vec![false],
               rq_scale_open: vec![],
               ao_q_open: false, ao_q_ok: false,
               nn_in_ok: vec![false], nn_out_ok: vec![false], nn_pos_ok: false,
               nn_q_ok: vec![false], nn_open: false,
               stream_in: None, stream_out: None,
               boundary_roots: vec![], boundary_sides: vec![],
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
        || claims.stream_bindings.len() != n_glue
        || plan.glue.as_ref().is_some_and(|g| {
            g.regions.iter().zip(&claims.stream_bindings).any(|(r, c)| r.bindings.len() != c.len())
        })
        || fixed.sizes.len() != plan.n_trees()
        // stream BB: one claim each, or none at all
        || claims.embed_out.len() != usize::from(plan.emb.is_some())
        || claims.embed_tok.len() != usize::from(plan.emb.is_some())
        // stream KK: the head's token claim, exactly when the argmax is composed
        || claims.head_tok.len() != usize::from(plan.argmax_mat().is_some())
        // stream II: one claim per seam per side, and one per walk group when
        // the chunk carries the wiring family
        || claims.rq_in.len() != plan.rqs.len()
        || claims.rq_out.len() != plan.rqs.len()
        || claims.walk_qx.len()
            != if plan.glue.is_some() { plan.walks.len() } else { 0 }
        // stream II: the scale binding -- one per seam, and one per matmul when the
        // chunk carries any seam at all
        || claims.rq_sc.len() != plan.rqs.len()
        || claims.deq_sc.len() != if plan.rqs.is_empty() { 0 } else { p.mats.len() }
        // stream NN: one claim per quantize region per side, and three per band
        // that publishes its own columns
        || claims.nn_in.len() != plan.quants.len()
        || claims.nn_out.len() != plan.quants.len()
        || claims.band_qkv.len() != plan.nn_bands.len()
        // stream NN (NN-2): and one per prefix band
        || claims.band_q.len() != plan.nn_qbands.len()
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
    // stream PP: the model tree's root is the second declared one -- a
    // *declaration* here, like `pre_root`; what pins it to a model is
    // `boundary-grid --expect-model-roots` against the published manifest.
    let model_root_ok = format!("{:?}", proof.commitments[1]) == fixed.model_root;
    vcs.commit(proof.commitments[0], &fixed.sizes[0], &mut ch);
    vcs.commit(proof.commitments[1], &fixed.sizes[1], &mut ch);
    vcs.commit(proof.commitments[2], &fixed.sizes[2], &mut ch);
    let rels = Rels::draw(&mut ch, &plan.widths, plan.glue.as_ref().map(|g| g.uns.as_slice()),
                          plan.emb.map(|l| l.arity()),
                          plan.needs_rq_rels(), plan.q_tabs.len());
    let n_ch = plan.n_challenges();
    let f = ch.draw_secure_felts(n_ch);
    let (r, s, rv) = (f[0], f[1], f[2]);
    let (w_glue, lam_glue) = if plan.has_bindings() {
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
        // position the prover drew it, then the per-height parameters the binding
        // constraints assert. A malformed artifact is a rejection here.
        let alpha = ch.draw_secure_felts(1)[0];
        match crate::gkrlink::link_params(&plan, &rels, &insts, &artifact, alpha) {
            Ok(hl) => link_hl = Some(hl),
            Err(e) => return Verdict::rejected(&format!("the bands' GKR link: {e}")),
        }
    } else if claims.gkr.is_some() {
        return Verdict::rejected("the statement carries a GKR proof the flag did not ask for");
    }
    vcs.commit(proof.commitments[3], &fixed.sizes[3], &mut ch);
    // Stream H: one tree per boundary side, replayed in the prover's order.  The
    // roots are read out below; nothing is drawn from them here, because nothing
    // in the AIR reads a boundary column except its equality constraint.
    if proof.commitments.len() < plan.n_trees() {
        return Verdict::rejected("the proof carries fewer roots than the statement claims");
    }
    for t in crate::boundary::FIRST_TREE..plan.n_trees() {
        vcs.commit(proof.commitments[t], &fixed.sizes[t], &mut ch);
    }
    // Stream AA: every boundary tree's root, in the prover's order -- stream H's
    // two, then the K/V sides.  Stream EE: the same roots labelled side by side
    // off the plan's own side lists, so nothing here assumes how many sides a
    // statement carries or which tree any of them took.  `Verdict` is *output*
    // and not `Fixed`, so neither field touches the digest.
    let root = |t: usize| format!("{:?}", proof.commitments[t]);
    let boundary_roots: Vec<String> = (crate::boundary::FIRST_TREE..plan.n_trees()).map(root).collect();
    let boundary_sides = crate::boundary::side_roots(&plan.layer_sides, &plan.kv, root);
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
    // band group is this band in", because with `--kv-boundary` a group is one
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
                .map(|e| {
                    // stream II: `a == factor * b`, the factor being the plan's
                    // own geometric sum over the norm scale's `a` rows per token.
                    // With no factor this is the equality it always was.
                    unqm(claims.stream_bindings[e.a.0][e.a.1])
                        == e.factor(w_glue) * unqm(claims.stream_bindings[e.b.0][e.b.1])
                })
                .collect(),
            Some(claims.stream_bindings[g.stream_in.0][g.stream_in.1]),
            Some(claims.stream_bindings[g.stream_out.0][g.stream_out.1]),
        ),
        None => (vec![], None, None),
    };
    // stream E: and one per dequant output binding -- the `o` and `down` regions'
    // own published outputs against the copies the wiring family committed.
    // This is what makes `o_out` and `d_out` the dequant's results rather than
    // two columns that merely happen to agree.
    //
    // Stream JJ (E11): one entry per binding that HAS a counterparty, in the plan's
    // order.  `deq_out_open` names the outputs that publish a claim the
    // statement cannot yet read -- they are not counted as passing.
    let (deq_out_ok, deq_out_open): (Vec<bool>, Vec<String>) = match &plan.glue {
        Some(g) => {
            let mut ok = vec![];
            let mut open = vec![];
            for (k, t) in g.output_bindings.iter().enumerate() {
                match (t.at, t.quant) {
                    (Some((ra, ta)), _) => ok.push(
                        claims.deq_out.get(k).copied() == Some(claims.stream_bindings[ra][ta])),
                    // stream NN: `v` has no rotation between its dequant and
                    // its band column, so its counterparty is the quantize
                    // region's input claim and `nn_in_ok` is where it is read
                    (None, Some(qi)) => ok.push(
                        claims.deq_out.get(k).copied() == claims.nn_in.get(qi).copied()),
                    // stream SS: a *folded* `v` has no quantize region either --
                    // the fold is the quantize -- so the published output is the
                    // int8 cell itself and the counterparty is the corner bands'
                    // own V fold, the sum `nn_out_ok` compares the other two
                    // against.  Under the same weights, so the equality says the
                    // bands attended over the very integers this region rounded.
                    (None, None) if t.band => {
                        let sum = claims.band_qkv.iter()
                            .fold(SecureField::zero(), |acc, c| acc + unqm(c[2]));
                        ok.push(claims.deq_out.get(k).copied().map(unqm) == Some(sum));
                    }
                    (None, None) => open.push(t.name.to_string()),
                }
            }
            (ok, open)
        }
        None => (vec![], vec![]),
    };

    // stream E: the argmax compared the head's own output.  Both claims are
    // Horner sums over the same values with mirrored weights, so equality is the
    // whole statement -- and it is checked from the parsed claims alone.
    let head_binding_ok = if plan.p.prove_argmax {
        !claims.am_key.is_empty() && claims.am_key.first() == claims.logits_claim.first()
    } else {
        true
    };

    // stream II: the requantize seams.  Two equalities each, both between
    // claims the statement already carries: the seam's input claim against the
    // producing region's `Out(0)`, which says the matmul's input is the bf16
    // value the wiring family proved; and the seam's output claim against the
    // walk group's own claim over its committed activations, which says the
    // walk multiplied *those* integers.  With the requantize's own constraints
    // in between -- the division table, the attained maximum and the scale
    // lookup -- the pair is "the activation is the requantize of the producer's
    // output under the pinned per-token scale".
    let rq_in_ok: Vec<bool> = match &plan.glue {
        Some(_) => plan
            .rqs
            .iter()
            .enumerate()
            .map(|(k, rp)| {
                claims.rq_in.get(k).copied()
                    == Some(claims.stream_bindings[rp.producer.0][rp.producer.1])
            })
            .collect(),
        None => vec![],
    };
    let rq_out_ok: Vec<bool> = plan
        .rqs
        .iter()
        .enumerate()
        .map(|(k, rp)| claims.rq_out.get(k).copied() == claims.walk_qx.get(rp.walk).copied())
        .collect();
    // Stream II (E11): the scale binding.  The per-token activation scale is
    // committed twice and, before this, never compared: once by the requantize
    // region, where `with_max` pins it to the ceil-divide of the attained
    // maximum of the seam's own input, and once as the x-side scale of every
    // consuming matmul's dequant.  A prover could requantize under one scale
    // and dequantize under another; these equalities say it did not.
    //
    // One entry per (seam, consumer) pair -- three for `qkv` (q, k and v share
    // the seam's input, each has its own dequant region), two for `gateup`, one
    // for `down`, one for the head.  `o` is not here and cannot be: its
    // producer's scale is the attnout region's preprocessed `num`/`pj`, so that
    // comparison is preprocessed against committed and not a claim pair.
    let mut rq_scale_ok: Vec<bool> = vec![];
    let mut rq_scale_open: Vec<String> = vec![];
    for (k, rp) in plan.rqs.iter().enumerate() {
        for mi in &plan.walks[rp.walk].mats {
            // stream SS: a folded matmul publishes no x-side scale -- its shared
            // columns are `(ms, dent)` and not `(ms, es)` -- so it is named,
            // not counted as a pass by silence (stream JJ's `deq_out_open`
            // convention, for the same reason).
            if plan.specs[*mi].variant == DeqVariant::Fold {
                rq_scale_open.push(format!("{}/{}", rp.name, plan.p.mats[*mi].name));
                continue;
            }
            rq_scale_ok.push(claims.rq_sc.get(k).copied() == claims.deq_sc.get(*mi).copied());
        }
    }
    // Stream II, the `o` seam.  `AttnOutEval` is the attention output's own
    // requantize, so what closes it is one claim on that region -- stream JJ's
    // `claims.ao_q`, a Horner fold over the signed `qm (1 - 2 sg)` -- against
    // the `o` walk's activation claim.  Both now fold the same values in the
    // same order under the same weights: the walk's `WalkQx` for the `o` group
    // carries `rv^k s^t` (see `build_components`), and its fold is over
    // `a - act_off`, which is the signed value the attnout region publishes.
    //
    // Live on every chunk that publishes both claims.  A chunk with no wiring
    // family publishes no `walk_qx` at all and there is nothing to compare; a
    // chunk with **no attention** -- the `lm_head` chunk, whose only matmul is
    // the head -- has no `o` walk, so the attnout region is absent, `ao_q` is
    // empty and the seam is vacuous.  Both report `true`, and neither is a
    // skipped check: in both cases the statement contains no `o` matmul whose
    // input could be wrong.
    //
    // The third case is a shape, not a statement: a *synthetic* layer whose `o`
    // walk is not `nh * hd` channels wide folds a different sequence from the
    // attnout region's, and the equality is not statable there.  The witness
    // builder wires `o` from `AttnOut::q_ref` exactly when the widths line up
    // (`Layer::synth_glue`), so on any shape a gate proves this is either live
    // or structurally absent -- it is never a mismatched comparison read as a
    // pass.
    //
    // The width test is the statement's own: the attention output is `nh * hd`
    // channels and the `o` walk reads `n_in` of them, and both are `Params`.  A
    // shape where they disagree is not a layer of any transformer -- its `o`
    // matmul does not consume the attention output at all -- so the two folds
    // run over different sequences and the equality is not statable.  That case
    // is *reported*, in `ao_q_open`, and never counted as a pass by silence;
    // it is stream JJ's `deq_out_open` convention, for the same reason.
    let ao_q_at = plan.walks.iter().position(|w| w.name == "o").filter(|_| {
        plan.glue.is_some() && !claims.ao_q.is_empty()
    });
    let ao_q_wide = p.mats.iter().find(|m| m.name == "o")
        .is_some_and(|m| m.n_in == p.nh * p.hd);
    let ao_q_open = ao_q_at.is_some() && !ao_q_wide;
    let ao_q_ok = match ao_q_at {
        Some(wi) if ao_q_wide => {
            claims.ao_q.first().copied() == claims.walk_qx.get(wi).copied()
        }
        _ => true,
    };

    // stream BB: the embedding gadget's binding.  Both claims are Horner sums over
    // the same sequence of stream values under the same weights, so equality is
    // the whole statement -- and it is checked from the parsed claims alone.
    let embed_ok = match (&plan.emb, &plan.glue) {
        (Some(_), Some(g)) => {
            claims.embed_out.first() == Some(&claims.stream_bindings[g.stream_in.0][g.stream_in.1])
        }
        (Some(_), None) => false,
        (None, _) => true,
    };

    // stream KK: the published token vectors against the claims that pin them.
    // Both are Horner images under `w_glue`, which is drawn after tree 1 -- so
    // the ids were fixed before the challenge existed and a second vector with
    // the same image is a collision, not a choice (`notes/witness-slack.md`
    // §0's probabilistic binding).  The conventions are each region's own: the
    // embedding walks its positions ascending, the argmax's rows walk them
    // descending, so the two exponents are mirrored and this is where that is
    // written down.
    let horner = |ids: &[u32], asc: bool| -> SecureField {
        let mut acc = SecureField::zero();
        if asc {
            for v in ids {
                acc = acc * w_glue + SecureField::from(m31_u32(*v));
            }
        } else {
            for v in ids.iter().rev() {
                acc = acc * w_glue + SecureField::from(m31_u32(*v));
            }
        }
        acc
    };
    let embed_tok_ok = match &plan.emb {
        Some(_) => {
            fixed.tok_in.len() == p.s
                && claims.embed_tok.first().map(|c| unqm(*c))
                    == Some(horner(&fixed.tok_in, true))
        }
        None => fixed.tok_in.is_empty(),
    };
    let head_tok_ok = match plan.argmax_mat() {
        Some(mi) => {
            fixed.tok_out.len() == p.mats[mi].tokens
                && claims.head_tok.first().map(|c| unqm(*c))
                    == Some(horner(&fixed.tok_out, false))
        }
        None => fixed.tok_out.is_empty(),
    };

    // ---- stream NN: the rotation's two seams ---------------------------------
    //
    // The input side: what the quantize rounded is what the rotation's `Out(0)`
    // published (q and k), or what the matmul's own dequant published (v).  Two
    // Horner sums over the same sequence under the same weights, from the parsed
    // claims alone.
    //
    // The output side: the integers the quantize produced are the columns the
    // *bands* committed -- and the counterparty is a **sum** over the corner
    // bands, because a chunk's heads are split across bands and each covers a
    // disjoint slice of them.  Grouped-query attention is inside the weights,
    // not inside this sum: a producer value carries the sum of the `rep`
    // weights its copies have on the band side (`rope::QTie::w_out`), so a
    // single equality forces every replicated copy separately.
    let (nn_in_ok, nn_out_ok): (Vec<bool>, Vec<bool>) = {
        let (mut a, mut b) = (vec![], vec![]);
        for (k, qp) in plan.quants.iter().enumerate() {
            let want_in = match qp.producer {
                Some((ri, ti)) => Some(claims.stream_bindings[ri][ti]),
                None => {
                    // `v`: the counterparty is the dequant output binding the plan
                    // marked with this region's index
                    plan.glue.as_ref().and_then(|g| {
                        g.output_bindings.iter().position(|t| t.quant == Some(k))
                            .and_then(|j| claims.deq_out.get(j).copied())
                    })
                }
            };
            a.push(claims.nn_in.get(k).copied() == want_in && want_in.is_some());
            let idx = match qp.name { "q" => 1usize, "k" => 0, _ => 2 };
            let sum = claims.band_qkv.iter()
                .fold(SecureField::zero(), |acc, c| acc + unqm(c[idx]));
            b.push(claims.nn_out.get(k).copied().map(unqm) == Some(sum));
        }
        (a, b)
    };
    // and the block coordinate the angles were built for.  `pos0` is in the
    // statement, `block` is what `boundary::chain_of` compares along the token
    // axis, and this is the one line that joins them -- without it a chunk could
    // prove block 3's rotation and chain in at block 5.
    let nn_pos_ok = match (&p.rope, fixed.block) {
        (Some(r), Some(b)) => r.pos0 == b * p.s,
        _ => true,
    };
    let nn_open = plan.glue.as_ref().is_some_and(|g| !g.head_norm) && plan.quants.is_empty();
    // Stream NN (NN-2): the prefix band's own Q against its group's corner.
    // Both folds run over the same (query head, channel, token) set under the
    // same canonical weights -- only the layout that scatters those cells over
    // rows differs -- so equality is "the same queries", and the corner's side
    // is already the rotated query by `nn_out_ok`.  The difference of the two
    // folds is a polynomial in `w_glue` whose coefficients were fixed before
    // `w_glue` was drawn, which is the argument every binding in this family rests
    // on.
    let nn_q_ok: Vec<bool> = plan.nn_qbands.iter().enumerate().map(|(k, b)| {
        let corner = plan.nn_bands.iter().position(|c| plan.group[*c] == plan.group[*b]);
        match corner {
            Some(ci) => claims.band_q.get(k).copied().map(unqm)
                == claims.band_qkv.get(ci).map(|c| unqm(c[1])),
            None => false,
        }
    }).collect();

    Verdict { stark_ok, root_ok, model_root_ok,
              sums_cancel, qk_ok, den_ok, av_ok, den_link_ok, walk_ok, deq_out_ok,
              deq_out_open, head_binding_ok, embed_ok, embed_tok_ok, head_tok_ok,
              rq_in_ok, rq_out_ok, rq_scale_ok, rq_scale_open, ao_q_open, ao_q_ok,
              nn_in_ok, nn_out_ok, nn_pos_ok, nn_q_ok, nn_open,
              sizes_ok, glue_ok, stream_in, stream_out, boundary_roots, boundary_sides, error,
              verify_secs: t0.elapsed().as_secs_f64() }
}

// ----------------------------------------------------------------- binding ----

/// Whether one edge of the forward pass is bound by this statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeState {
    /// the verifier compares the producer's claim with the consumer's, or the
    /// two sides are columns of one AIR component
    Bound,
    /// the consumer reads a value nothing bindings to its producer: the prover
    /// asserts it and the verifier checks nothing
    Unbound,
    /// neither side is in this statement (a head chunk has no attention, a
    /// layer chunk emits no token)
    Absent,
}

/// One row of the binding table.
#[derive(Debug, Clone, Serialize)]
pub struct Edge {
    pub n: u8,
    pub edge: &'static str,
    pub state: EdgeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// The edges the binding table reports: the fourteen of one layer segment, in
/// the order of the move brief's section 11, then the two ends of the
/// sequence.
pub const EDGES: [&str; 16] = [
    "residual stream into the first norm's reduction",
    "first norm's output into the q, k, v matmul inputs",
    "q, k, v outputs into the attention block",
    "attention scores and values into the attention output",
    "attention output into the o matmul input",
    "o output into the first residual add",
    "first residual add into the second norm's reduction",
    "second norm's output into the gate and up matmul inputs",
    "gate and up outputs into the SwiGLU product",
    "SwiGLU output into the down matmul input",
    "down output into the second residual add",
    "each matmul's input times the pinned weights into its accumulator",
    "residual stream out of this segment into the next layer's segment",
    "keys and values out of this token block into the next block",
    "embedding row into the first segment's residual stream",
    "logits into the emitted token",
];

impl Plan {
    /// The binding table of this statement: every edge of `EDGES` marked bound,
    /// unbound or absent, decided from the plan fields `verify_composed` reads
    /// for the same edge -- the residual-stream equalities (`glue.eqs`), the
    /// requantize seams (`rqs`), the dequant output bindings (`glue.output_bindings`), the
    /// quantize regions and their key tiles (`quants`, `nn_bands`), the `o`
    /// walk's width, the boundary trees (`bounds`, `kv`), the embedding layout
    /// (`emb`) and the argmax (`argmax_mat`).  `layer` is the segment's declared
    /// layer coordinate, which decides whether the embedding edge applies.
    ///
    /// The table is a report on the statement and enters nothing: not the
    /// transcript, not a claim, not the proof.
    pub fn binding(&self, layer: Option<usize>) -> Vec<Edge> {
        use EdgeState::*;
        let p = &self.p;
        let g = self.glue.as_ref();
        let head = g.is_some_and(|g| g.head_norm);
        let layer_family = g.is_some() && !head;
        let attention = !head && !self.shapes.is_empty();
        let has_mat = |n: &str| p.mats.iter().any(|m| m.name == n);
        let seam = |n: &str| self.rqs.iter().any(|r| r.name == n);
        let bound = |n: &str| {
            g.is_some_and(|g| g.output_bindings.iter().any(|t| t.name == n && t.at.is_some()))
        };
        let no_family = || {
            Some(String::from(
                "no residual-stream family in this statement: the norms, the residual adds \
                 and the requantize seams are not proved, so every value between two \
                 matmuls is the prover's word",
            ))
        };
        let head_only = || Some(String::from("the head chunk has no attention and no MLP"));
        let mut rows: Vec<Edge> = Vec::with_capacity(EDGES.len());
        let mut put = |n: usize, state: EdgeState, why: Option<String>| {
            rows.push(Edge { n: n as u8, edge: EDGES[n - 1], state, why });
        };

        // 1 and 7: the two norm reductions read the committed residual stream
        // (`glue_ok`: `nt1.in == gn1.roots`, `gn1.xmid == gn2.in`, ...)
        if layer_family {
            put(1, Bound, None);
            put(7, Bound, None);
        } else if head {
            put(1, Bound, Some("the head chunk's final norm reads the residual its \
                                 statement commits".into()));
            put(7, Absent, head_only());
        } else {
            put(1, Unbound, no_family());
            put(7, Unbound, no_family());
        }

        // 2, 8, 10: a requantize seam per consuming matmul group (`rq_in_ok`,
        // `rq_out_ok`, `rq_scale_ok`)
        for (n, name) in [(2usize, "qkv"), (8, "gateup"), (10, "down")] {
            if layer_family {
                if seam(name) {
                    put(n, Bound, None);
                } else {
                    put(n, Unbound, Some(format!(
                        "no requantize seam for {name}: the matmul's int8 input is not \
                         bound to its producer's output")));
                }
            } else if head {
                if n == 2 {
                    if seam("head") {
                        put(n, Bound, Some("the final norm's output into the lm_head \
                                            input".into()));
                    } else {
                        put(n, Unbound, Some("no requantize seam for the head".into()));
                    }
                } else {
                    put(n, Absent, head_only());
                }
            } else {
                put(n, Unbound, no_family());
            }
        }

        // 3: q, k, v leave their dequant through the rotation and the quantize
        // into the key tiles' own columns (`deq_out_ok`/`deq_out_open`,
        // `nn_in_ok`, `nn_out_ok`, `nn_q_ok`)
        if !attention {
            put(3, Absent, if head { head_only() } else { Some("no attention".into()) });
        } else if let Some(g) = g {
            let open: Vec<&str> = g.output_bindings.iter()
                .filter(|t| matches!(t.name, "q" | "k" | "v")
                            && t.at.is_none() && t.quant.is_none() && !t.band)
                .map(|t| t.name)
                .collect();
            if p.rope.is_none() {
                put(3, Unbound, Some("the statement carries no rotation parameters, so q, \
                                      k and v publish output claims with no counterparty \
                                      (deq_out_open)".into()));
            } else if !open.is_empty() {
                put(3, Unbound, Some(format!(
                    "dequant outputs with no counterparty: {}", open.join(", "))));
            } else if self.quants.is_empty() || self.nn_bands.is_empty() {
                put(3, Unbound, Some("no quantize region bindings the rotated q, k, v to the \
                                      key tiles' columns".into()));
            } else {
                put(3, Bound, None);
            }
        } else {
            put(3, Unbound, no_family());
        }

        // 4: the attention component's own identities (`qk_ok`, `den_ok`,
        // `av_ok`, `den_link_ok`)
        if attention {
            put(4, Bound, None);
        } else {
            put(4, Absent, if head { head_only() } else { Some("no attention".into()) });
        }

        // 5: the attention output's requantize against the `o` walk's
        // activations (`ao_q_ok`, `ao_q_open`)
        match self.walks.iter().position(|w| w.name == "o") {
            None => put(5, Absent, if head { head_only() } else { Some("no o matmul".into()) }),
            Some(_) if g.is_none() => put(5, Unbound, no_family()),
            Some(_) if head => put(5, Absent, head_only()),
            Some(_) => {
                let n_in = p.mats.iter().find(|m| m.name == "o").map(|m| m.n_in).unwrap_or(0);
                if n_in == p.nh * p.hd {
                    put(5, Bound, None);
                } else {
                    put(5, Unbound, Some(format!(
                        "the o matmul reads {n_in} channels and the attention output has \
                         {} (nh x hd), so the two folds run over different sequences \
                         (ao_q_open)", p.nh * p.hd)));
                }
            }
        }

        // 6, 9, 11: a dequant output bound to the region that consumes it
        // (`deq_out_ok`)
        for (n, names) in [(6usize, &["o"][..]), (9, &["gate", "up"][..]), (11, &["down"][..])] {
            if !names.iter().all(|m| has_mat(m)) {
                put(n, Absent, if head { head_only() } else {
                    Some(format!("no {} matmul", names.join("/")))
                });
            } else if head {
                put(n, Absent, head_only());
            } else if g.is_none() {
                put(n, Unbound, no_family());
            } else if names.iter().all(|m| bound(m))
                && (n != 9 || g.is_some_and(|g| g.swiglu.is_some()))
            {
                put(n, Bound, None);
            } else {
                put(n, Unbound, Some(format!(
                    "the {} dequant publishes no output claim bound to its consumer",
                    names.join(" and "))));
            }
        }

        // 12: every matmul is a Freivalds check over the pinned weights
        // (`walk_ok`)
        if self.walks.is_empty() {
            put(12, Absent, Some("no matmul".into()));
        } else {
            put(12, Bound, None);
        }

        // 13: the residual-stream boundary trees (`bounds`; the link itself is
        // the grid's root equality between two statements)
        if g.is_none() {
            put(13, Unbound, Some("no residual-stream family, so no boundary tree can be \
                                   committed".into()));
        } else if self.layer_sides.is_empty() {
            put(13, Unbound, Some("the layer boundary trees are off \
                                   (--unsound-skip-layer-boundary): the residual stream \
                                   leaving this segment is compared with nothing".into()));
        } else if head {
            put(13, Bound, Some("the head commits its residual-in side only; nothing \
                                 leaves it along the layer axis".into()));
        } else {
            put(13, Bound, None);
        }

        // 14: the K/V boundary trees (`kv`)
        if !attention {
            put(14, Absent, if head { head_only() } else { Some("no attention".into()) });
        } else if !self.kv.is_empty() {
            put(14, Bound, None);
        } else if !p.corner {
            put(14, Unbound, Some("the chunk has no diagonal block (not extracted with \
                                   --corner), so its own keys and values are not in the \
                                   statement and no K/V boundary can be committed".into()));
        } else {
            put(14, Unbound, Some("the K/V boundary trees are off \
                                   (--unsound-skip-kv-boundary): the keys and values this \
                                   block writes are compared with nothing in the next \
                                   block".into()));
        }

        // 15: the embedding gadget (`embed_ok`, `embed_tok_ok`), a layer-0 edge
        if self.emb.is_some() {
            put(15, Bound, None);
        } else if layer == Some(0) {
            put(15, Unbound, Some("a layer-0 segment without the embedding gadget \
                                   (--unsound-skip-embedding): the residual stream it \
                                   consumes is not bound to the embedding rows of its \
                                   ids".into()));
        } else {
            put(15, Absent, Some(match layer {
                Some(l) => format!("layer {l} is not the first"),
                None => "the segment declares no layer coordinate".into(),
            }));
        }

        // 16: the argmax over the head's logits (`head_binding_ok`, `head_tok_ok`)
        if self.argmax_mat().is_some() {
            put(16, Bound, None);
        } else if has_mat("lm_head") {
            put(16, Unbound, Some("the head chunk publishes no argmax \
                                   (--unsound-skip-argmax): the emitted token is not bound \
                                   to the logits".into()));
        } else {
            put(16, Absent, Some("no lm_head matmul".into()));
        }

        rows.sort_by_key(|e| e.n);
        rows
    }

    /// `binding` as the JSON the `provenance` block and every chunk report
    /// carry: the rows, the edge numbers left unbound, and `all_bound`.
    pub fn binding_json(&self, layer: Option<usize>) -> Value {
        let rows = self.binding(layer);
        let of = |st: EdgeState| -> Vec<u8> {
            rows.iter().filter(|e| e.state == st).map(|e| e.n).collect()
        };
        let unbound = of(EdgeState::Unbound);
        json!({
            "edges": rows,
            "all_bound": unbound.is_empty(),
            "unbound": unbound,
            "absent": of(EdgeState::Absent),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The binding table, on the synthetic shapes the digest gate pins.
    fn plan_of(args: Vec<String>) -> Plan {
        let (l, vname) = crate::layer_of(&args);
        Plan::new(&crate::params_of(&l, &args, &vname))
    }

    fn states(plan: &Plan, layer: Option<usize>) -> Vec<(u8, EdgeState)> {
        plan.binding(layer).iter().map(|e| (e.n, e.state)).collect()
    }

    const SYN: &[&str] = &["compose", "--round", "away", "--variants", "intsplit6",
                           "--table-bits", "16", "--s", "32", "--ctx", "128", "--hd", "32"];

    fn argv(extra: &[&str]) -> Vec<String> {
        SYN.iter().chain(extra.iter()).map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_pinned_wired_shape_reports_its_two_desk_holes() {
        use EdgeState::*;
        // `glue_corner` of prover/README.md, with both boundaries on: 64-wide
        // matmuls under three heads of 32, so the rotation cannot tile (edge 3)
        // and the `o` matmul does not read the attention output's width (edge 5)
        let plan = plan_of(argv(&["--bands", "2,1", "--nout", "64", "--glue-a", "32",
                                   "--kmax", "16", "--synth-glue", "--corner"]));
        let want = vec![
            (1, Bound), (2, Bound), (3, Unbound), (4, Bound), (5, Unbound), (6, Bound),
            (7, Bound), (8, Bound), (9, Bound), (10, Bound), (11, Bound), (12, Bound),
            (13, Bound), (14, Bound), (15, Absent), (16, Absent),
        ];
        assert_eq!(states(&plan, None), want);
        let j = plan.binding_json(None);
        assert_eq!(j["all_bound"], false);
        assert_eq!(j["unbound"], json!([3, 5]));
        assert_eq!(j["absent"], json!([15, 16]));
        // and the same shape at layer 0 owes the embedding edge
        assert_eq!(states(&plan, Some(0))[14], (15, Unbound));
    }

    #[test]
    fn the_aligned_wired_shape_binds_every_layer_edge() {
        use EdgeState::*;
        // four heads of 32 over 128-wide matmuls: q/k/v are `nh x hd` wide, the
        // rotation tiles, and the `o` matmul reads the attention output
        let plan = plan_of(argv(&["--bands", "2,2", "--nin", "128", "--nout", "128",
                                   "--glue-a", "32", "--kmax", "16", "--synth-glue",
                                   "--corner"]));
        let got = states(&plan, Some(3));
        for (n, st) in &got[..14] {
            assert_eq!(*st, Bound, "edge {n}: {:?}", plan.binding(Some(3))[*n as usize - 1]);
        }
        assert_eq!(got[14], (15, Absent));
        assert_eq!(got[15], (16, Absent));
    }

    #[test]
    fn the_skips_are_reported_as_unbound_not_absent() {
        use EdgeState::*;
        let plan = plan_of(argv(&["--bands", "2,1", "--nout", "64", "--glue-a", "32",
                                   "--kmax", "16", "--synth-glue", "--corner",
                                   "--unsound-skip-layer-boundary",
                                   "--unsound-skip-kv-boundary"]));
        let got = states(&plan, None);
        assert_eq!(got[12], (13, Unbound));
        assert_eq!(got[13], (14, Unbound));
        // no residual-stream family at all: every seam and add is the prover's
        // word, while the K/V boundary (on by default on a corner chunk) and the
        // attention and Freivalds components stand
        let bare = plan_of(argv(&["--bands", "2,1", "--corner",
                                   "--unsound-skip-residual-stream"]));
        let got = states(&bare, None);
        for n in [1u8, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13] {
            assert_eq!(got[n as usize - 1], (n, Unbound), "edge {n}");
        }
        assert_eq!(got[3], (4, Bound));
        assert_eq!(got[11], (12, Bound));
        assert_eq!(got[13], (14, Bound));
    }

    #[test]
    #[should_panic(expected = "--unsound-skip-kv-boundary")]
    fn a_chunk_without_its_own_block_is_refused_unless_the_kv_boundary_is_skipped() {
        plan_of(argv(&["--bands", "2,1", "--unsound-skip-residual-stream"]));
    }

    #[test]
    #[should_panic(expected = "--unsound-skip-residual-stream")]
    fn a_chunk_without_the_family_is_refused_unless_the_family_is_skipped() {
        plan_of(argv(&["--bands", "2,1", "--corner"]));
    }

    #[test]
    #[should_panic(expected = "not a skip")]
    fn an_unknown_skip_is_an_error() {
        plan_of(argv(&["--bands", "2,1", "--corner", "--unsound-skip-everything"]));
    }

    #[test]
    fn the_head_chunk_reports_its_norm_its_in_side_and_the_skipped_argmax() {
        use EdgeState::*;
        // a synthetic head has no `key_head.u16`, so the argmax has to be
        // skipped to prove it at all, and the table says what that costs
        let plan = plan_of(argv(&["--bands", "2,1", "--nout", "64", "--glue-a", "32",
                                   "--kmax", "16", "--synth-head", "--synth-vocab", "256",
                                   "--corner", "--unsound-skip-argmax"]));
        let got = states(&plan, Some(32));
        assert_eq!(got[0], (1, Bound));
        assert_eq!(got[1], (2, Bound));
        for n in [3u8, 4, 5, 6, 7, 8, 9, 10, 11, 14] {
            assert_eq!(got[n as usize - 1], (n, Absent), "edge {n}");
        }
        assert_eq!(got[11], (12, Bound));
        assert_eq!(got[12], (13, Bound));
        assert_eq!(got[14], (15, Absent));
        assert_eq!(got[15], (16, Unbound));
    }

    /// Stream QQ, at the NN merge (log row 66): the guard itself, gated.
    ///
    /// Everything else in this stream says "the vectors are filled"; nothing
    /// said what happens when the next stream adds one and forgets. That is the
    /// case the guard exists for, and it fired on stream NN's four before this
    /// test was written -- `cudaemit` on NN's rotation shapes panicked naming
    /// `band_q`. Here it is as a control, so a later refactor that turns the
    /// panic back into a zero fails at `cargo test` and not on a card.
    #[test]
    fn the_emit_guard_names_the_claim_vector_it_is_missing() {
        // the prove path is unchanged: a statement written before a claim
        // existed still parses, which is what `#[serde(default)]` is for
        assert_eq!(claim_at(None, "band_q"), SecureField::zero());
        assert_eq!(claim_at(Some([1, 2, 3, 4]), "band_q"), unqm([1, 2, 3, 4]));

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        set_emit_strict(true);
        // a claim that IS supplied is returned on the emit path too
        let ok = std::panic::catch_unwind(|| claim_at(Some([1, 2, 3, 4]), "band_q"));
        let missing = std::panic::catch_unwind(|| claim_at(None, "band_q"));
        set_emit_strict(false);
        std::panic::set_hook(prev);

        assert_eq!(ok.expect("a supplied claim must not panic"), unqm([1, 2, 3, 4]));
        let e = missing.expect_err("a missing claim must panic on the emit path");
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(msg.contains("no placeholder for claim band_q"),
                "the panic must name the vector, got: {msg}");
        assert!(msg.contains("fill_binding_claims"),
                "the panic must name where to fix it, got: {msg}");

        // and the flag is off again, so nothing else in the suite sees it
        assert_eq!(claim_at(None, "band_q"), SecureField::zero());
    }
}
