//! The layer witness B6.5 composes into one proof, and the trace builders for
//! the components B6 did not have and that no gadget directory holds yet: the
//! attention-output component, and the r^T W Freivalds components.  The
//! per-matmul dequant components (with the Freivalds claimed side riding on the
//! same committed accumulators) are `gadgets::dequant` since #197.
//!
//! Layouts, once, because every builder and every selector has to agree:
//!
//! * **attnout component**: rows = (head, token, channel-chunk), heads and tokens and
//!   chunks descending, with `lead` inert rows *first* (nh*S*chunks is not a power
//!   of two); inert rows commit `den = 1` and zeros, which satisfies every
//!   constraint and looks up real table rows.
//! * **Freivalds component** (one per input): `layout::Grid` over input channels, `pk`
//!   preprocessed weights per row per matmul and `pq` committed activation bytes.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use num_traits::{One, Zero};

use crate::gadgets::attention::witness::{Attn, LayerRaw as AttnRaw, Src, Variant};
use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::dev_attention_freivalds_witness_gen::sub_slice;
use crate::gadgets::dequant::{Spec, Variant as DeqVariant};
use crate::gadgets::attention::layout::{Grid, Shape};
use crate::encoding::m31_of;
use crate::witread::{self, Site};

// The dequant gadget's own names, under the spellings the rest of the crate
// still says. `compose.rs`, `cli.rs`, `cudagen.rs`, `devacc.rs` and
// `gkr_key_tile.rs` name every one of these as `crate::layer::X`, and #197 owns
// `layer.rs`, the `dev_*` files and `gadgets/` and nothing else in `prover/src`,
// so the code moves and the paths it is reached by stay until a lane that owns
// those callers updates them.
pub use crate::gadgets::dequant::claims::{MatInteraction, OutputClaimW, deq_logup_eval,
                                          mat_interaction};
pub use crate::gadgets::dequant::layout::{MAT_NAMES, MAT_VARIANTS, MAT_VARIANTS_CHAIN,
                                          mat_variants};
pub use crate::gadgets::dequant::trace::{Mat, MatComponent, deq_ref, deq_ref_a, fold_out,
                                         fold_out_a, mat_component};
pub use crate::layout::{MIN_MAT_LOG_ROWS, MatLayout};

// The attnout gadget's names, likewise: `compose.rs`, `cli.rs`, `devacc.rs`,
// `gkrlink.rs`, `gkr_key_tile.rs` and `witread.rs` say `crate::layer::X` for
// every one of them.
pub use crate::gadgets::attn_out::claims::{AoInteraction, ao_interaction};
pub use crate::gadgets::attn_out::layout::AoLayout;
pub use crate::gadgets::attn_out::trace::{AoComponent, ao_component};
pub use crate::gadgets::attn_out::witness::AttnOut;

/// Stream LL (E11), `notes/inconsistency-log.md` row 58: the precondition tree 0
/// silently depends on, pinned.
///
/// Two preprocessed families come off the dump's *manifest* and not off the
/// statement -- the key tiles' `dinv` (`b6_k_dinv` / `b6_x_dinv`, the per-instance
/// master-table scale `2^-(F - frac_i)`) and the master exp table's value column
/// (`b6_exp_y`) -- and both are functions of `frac_per_head`. With
/// `frac_shared_per_head` **false** the forward pass did not publish one
/// exponent per head, `Attn::build` derives `frac_per_inst` from the key tile's own
/// shifts (`attn_witness.rs`, `d_max_of`), and tree 0 is then a function of the
/// *witness*: two chunks of one layer would legitimately carry two different
/// tree-0 roots, and a derived root would be checking nothing.
///
/// Every 8B dump on the box sets the flag, so this refuses nothing that has ever
/// been proved; what it removes is the possibility of proving a chunk whose
/// tree 0 the verifier cannot derive. `Some(why)` is the refusal, named.
pub fn frac_shared_refusal(man: &serde_json::Value) -> Option<String> {
    if man["frac_shared_per_head"].as_bool().unwrap_or(false) {
        return None;
    }
    Some(format!(
        "manifest: frac_shared_per_head is {} -- the master exponent is then \
         derived from this chunk's own scores, so the key_tiles' `dinv` and the exp \
         table (tree 0) would be a function of the witness and no verifier could \
         derive this chunk's `pre_root` from the model. Re-extract the layer with \
         a per-head Q scale (`extract.py` writes `frac_per_head`).",
        if man["frac_shared_per_head"].is_null() { "absent".into() }
        else { man["frac_shared_per_head"].to_string() }))
}

/// Issue #53: the dump's own rounding against `--round`.
///
/// A dump is quantized under one rounding mode; the prover builds its dequant
/// tables under the one the line names. Where the two disagree the witness's
/// values miss the rows the tables hold, the LogUp sums do not cancel, and the
/// honest proof is rejected after the whole prove -- a dump/command mismatch
/// that costs a proof to find and reads as a prover defect. `extract.py` writes
/// the mode it used into the manifest, so the two are compared here instead.
///
/// A manifest that does not say (a dump older than the field) is not refused:
/// there is nothing to compare `--round` with. `Some(why)` is the refusal,
/// naming both modes.
pub fn round_refusal(man: &serde_json::Value, round: bf16::Round) -> Option<String> {
    let dump = man["dequant_round"].as_str()?;
    // both spellings `Round::parse` takes for each mode
    let same = matches!((dump, round),
                        ("even" | "rne", bf16::Round::Even)
                        | ("away" | "hafz", bf16::Round::Away)
                        | ("zero" | "trunc", bf16::Round::Zero));
    if same {
        return None;
    }
    Some(format!(
        "manifest: dequant_round is {dump:?} and the line says --round {} -- this dump's \
         values were rounded one way and the dequant tables would be built the other, \
         so every value that misses a row falls off the table and the honest proof is \
         rejected. Prove this dump with --round {dump}.",
        round.name()))
}

// ------------------------------------------------------------------ the witness --

pub struct Layer {
    pub key_tiles: Vec<Attn>,
    pub key_tile_sizes: Vec<usize>,
    pub head_offsets: Vec<usize>,
    /// Stream AA (E11): how many key tiles one head key tile group's unchecked prefix is
    /// tiled into.  1 everywhere before this stream and without `--kv-boundary`;
    /// `C / S` with it, which is what makes a consumer's sub-key-tile the same
    /// `Shape` as the producer's diagonal block; 0 on block 0 of a sequence, which has
    /// no unchecked prefix and so no prefix key tile.  `params_of` reads it back to
    /// recover the chunk's real context and its head key tile list.
    pub nsub: usize,
    pub mats: Vec<Mat>,
    pub ao: AttnOut,
    pub source: String,
    /// B10 step 2: the wiring columns, present only on a witness extracted with
    /// `extract.py --residual-stream`.  Every one is a `(tokens x d)` bf16 *pattern* column
    /// of B1's own residual stream; `xmid` and `xout` are what the residual stream's two
    /// adds have to reproduce, and the two norm weights and the two scalars are
    /// what the reduction's norm scale consumes.
    pub residual_stream: Option<ResidualStreamCols>,
    /// stream E: B1's own `key_head.u16`, the order keys the head's rows compare.
    /// `None` unless the dump was written with `--head`.  It is a reference, not
    /// an input: the proof derives its keys from the head's dequant output, and
    /// this is what says the two agree.
    pub head_keys: Option<Vec<u16>>,
    /// stream E: `tok_head.i32`, the token each row emits.  The argmax component's
    /// own input when it is composed; `None` unless the dump has a head.
    pub head_tok: Option<Vec<u32>>,
    /// Stream Z: the attention-output fold's target width, from the manifest;
    /// 8 on every dump that predates the knob.  `ao_q_bits` is this minus one.
    pub attn_out_bits: u32,
    /// Stream BB (E11): the model's embedding matrix, `(vocab, d)` bf16
    /// patterns, from `embed.u16`.  `None` unless the dump carries it, and
    /// `--embed` then refuses to run -- there is no synthetic stand-in on the
    /// composed path, because a stand-in that proves the wrong table reads as a
    /// gadget that works.
    pub embed_tab: Option<Vec<u16>>,
    /// Stream BB: `tok_in.i32`, the vocabulary index of each of the block's
    /// positions.  These are the ids the gadget commits.
    pub tok_in: Option<Vec<u32>>,
    /// Stream KK: where this dump sits in the model and in the sequence, off the
    /// manifest's own `layer` and `block` (`extract.py` has written both since
    /// stream DD).  They go into the statement, so the grid's coordinates are
    /// the statement's and not the plan tag's.  `--layer` / `--block` override
    /// them for a chunk whose manifest cannot know where the grid puts it --
    /// the `lm_head` chunk, which sits after the last layer of its block.
    pub layer: Option<usize>,
    pub block: Option<usize>,
    /// Stream NN (E11): what the rotation and the q/k/v quantize need -- the
    /// model's `inv_freq`, this block's absolute position offset, and the three
    /// calibrated power-of-two scales.  `None` on a chunk that does not prove
    /// them; `params_of` turns the first three fields into the statement's
    /// `RopeParams` and the exponents go to tree 0, where `jv` already lived.
    pub rope: Option<RopeCols>,
}

/// Stream NN: the witness side of the rotation and the quantize.
///
/// `inv_freq` is `hd/2` fp64 values; the three exponent vectors are `nh * hd`
/// each, on the query-head grid B1 applies them on.
#[derive(Clone, Debug)]
pub struct RopeCols {
    /// key/value heads; `nh / nkv` is the GQA replication factor
    pub nkv: usize,
    /// absolute position of this block's token 0 -- `block * S`
    pub pos0: usize,
    /// `rope_theta`, the branch of `inv_freq_of` this model takes, and that
    /// branch's own constants -- what the *statement* carries, so the angle
    /// table is a function of `Params` and of nothing read off the dump
    pub theta: f64,
    pub rope_type: String,
    pub rope_scale: Vec<f64>,
    /// B1's own `inv_freq`, when the dump wrote it: a reference the run
    /// compares the derived vector against and never consumes
    pub inv_ref: Option<Vec<f64>>,
    /// `s_q`, `s_k`, `s_v` as power-of-two exponents on the **query-head** grid
    /// (`nh * hd` each).  B1 applies all three after `repeat_interleave`, so
    /// they are per query head even where the values behind them are shared;
    /// `Layer::qkv_j` is where "and they agree inside a GQA group" is checked.
    pub jq: Vec<i32>,
    pub jk: Vec<i32>,
    pub jv: Vec<i32>,
}

impl RopeCols {
    /// The four files `extract.py` writes for this stream, beside `jv.i32`.
    /// `Ok(None)` -- not an error -- when the dump carries none of them: a dump
    /// written before this stream proves no rotation.
    pub fn load(dir: &str, man: &serde_json::Value, nh: usize, hd: usize, s: usize)
                -> Result<Option<Self>, String> {
        let rp = &man["rope"];
        let Some(theta) = rp["theta"].as_f64() else {
            return Ok(None);
        };
        let kind = rp["rope_type"].as_str().unwrap_or("default").to_string();
        let rope_scale: Vec<f64> = rp["scale"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        // Stream PP (E11): `inv_freq.f64` is a *reference*, not an input -- the
        // angle table tree 0 commits is built from `RopeParams::inv`, and this
        // vector is only compared against it (`compose.rs`'s "the angle table
        // the statement builds differs from the dump's inv_freq").  A verifier
        // deriving the shape root holds no dump and so has nothing to compare,
        // so under `--shape-only` there is no reference and the comparison does
        // not run.  It is skipped rather than zeroed: a zeroed `inv_freq` is not
        // a missing reference, it is a *wrong* one, and the assert fires on it
        // (measured -- the box record's section 14b).
        let shape_only =
            crate::compose::SHAPE_ONLY.load(std::sync::atomic::Ordering::Relaxed);
        let inv_ref = if shape_only { None } else {
            witread::read_file(&format!("{dir}/inv_freq.f64")).ok().map(|raw| {
                raw.chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .take(hd / 2)
                    .collect::<Vec<f64>>()
            }).filter(|v| v.len() == hd / 2)
        };
        let j = |n: &str| -> Result<Vec<i32>, String> {
            let v = model_i32(dir, man, &format!("{n}.i32"))?;
            if v.len() < nh * hd {
                return Err(format!("{n}.i32: {} values is under {}", v.len(), nh * hd));
            }
            Ok(v[..nh * hd].iter().map(|x| *x as i32).collect())
        };
        let nkv = man["nkv"].as_u64().map(|v| v as usize).unwrap_or(nh);
        let block = man["block"].as_u64().map(|v| v as usize).unwrap_or(0);
        Ok(Some(Self {
            nkv,
            pos0: block * s,
            theta,
            rope_type: kind,
            rope_scale,
            inv_ref,
            jq: j("jq")?,
            jk: j("jk")?,
            jv: j("jv")?,
        }))
    }
}

impl Layer {
    /// Stream LL (E11), `notes/witness-slack.md` section 5 item 1, deliverable 3:
    /// the content hash of exactly the model data tree 0 reads -- the quantized
    /// weights and the weight scales of every matmul, the attention output's
    /// offline `jv` exponents, the two norm weights and the embedding table.
    ///
    /// It is *not* a hash of the dump directory: a dump also holds the
    /// activations, which are this chunk's and not the model's, and two blocks
    /// of one layer are supposed to differ in them.  Hashing the model half is
    /// what makes "these two chunks used the same weights" a comparison the
    /// grid can make, and it is **shape-independent** on purpose -- every block
    /// of a layer hashes to the same value whatever its `c` or its table size,
    /// which is the property the comparison needs.
    ///
    /// Two model-derived things it deliberately leaves out, because both are a
    /// function of the *chunk's* shape as well as the layer's: the master exp
    /// table (`1 << log_table` rows) and the key tiles' per-instance `dinv` scales.
    /// Those are covered by the derived tree-0 root, not here.  SHA-256,
    /// little-endian, in the order `Plan::pre_ids` walks the families.
    pub fn model_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"b65-model-v1");
        h.update((self.ao.jv.len() as u64).to_le_bytes());
        for v in &self.ao.jv {
            h.update(v.to_le_bytes());
        }
        for m in &self.mats {
            h.update(m.name.as_bytes());
            h.update((m.qw.len() as u64).to_le_bytes());
            let bytes: Vec<u8> = m.qw.iter().map(|x| *x as u8).collect();
            h.update(&bytes);
            for v in &m.jw {
                h.update(v.to_le_bytes());
            }
            for (a, b) in &m.sw {
                h.update(a.to_le_bytes());
                h.update(b.to_le_bytes());
            }
        }
        if let Some(g) = &self.residual_stream {
            for w in [&g.w1, &g.w2] {
                h.update((w.len() as u64).to_le_bytes());
                for x in w {
                    h.update(x.to_le_bytes());
                }
            }
        }
        if let Some(e) = &self.embed_tab {
            h.update((e.len() as u64).to_le_bytes());
            for x in e {
                h.update(x.to_le_bytes());
            }
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

pub struct ResidualStreamCols {
    pub d: usize,
    pub dff: usize,
    pub tokens: usize,
    pub xin: Vec<u16>,
    pub o_out: Vec<u16>,
    pub xmid: Vec<u16>,
    pub d_out: Vec<u16>,
    pub xout: Vec<u16>,
    /// `n1_out` and `n2_out`: each norm's `r = rsqrt(mean of squares + eps)`,
    /// one value per token.  The norm scales multiply the stream by it, so a row of
    /// the norm scale component reads it once and every channel of that row shares it.
    pub r1: Vec<u16>,
    pub r2: Vec<u16>,
    pub w1: Vec<u16>,
    pub w2: Vec<u16>,
    /// B1's own norm outputs, which are what the norm scales have to reproduce:
    /// `x_q` is norm1's (q, k and v all read the same column, and the dump's
    /// digests say so) and `x_gate` is norm2's.  `None` on a dump that did not
    /// write them, in which case the norm scale component runs without its reference.
    pub h1: Option<Vec<u16>>,
    pub h2: Option<Vec<u16>>,
    /// The SwiGLU's B1 columns, over `dff`: the gate matmul's dequantized
    /// output, its silu image, and the product that `down` reads.
    pub silu_in: Option<Vec<u16>>,
    pub silu_out: Option<Vec<u16>>,
    pub x_down: Option<Vec<u16>>,
    pub eps: u16,
    pub inv_d: u16,
    pub reduce: String,
    /// Stream II (E11): these columns are the *head* chunk's -- `xin` is the
    /// last layer's residual, `w1` the model's final-norm weight, `r1` B1's own
    /// `rsqrt` answer and `h1` the head's bf16 input.  `o_out`, `xmid`,
    /// `d_out`, `xout` and the second norm's are empty, because a head chunk
    /// has no residual add and no second norm.
    pub head: bool,
}

// Stream P: this widening and the two below it are parallel.  Each output
// element is a function of its own one, two or four bytes and an indexed
// parallel collect keeps the order, so these are the same arrays the serial
// versions built -- and on a sequence chunk they were 0.8 s of the 17 s the
// layer took to build.
fn read_u16(path: &str) -> Result<Vec<u16>, String> {
    let raw = witread::read_file(path)?;
    let _s = witread::scope(Site::DecodeU16);
    Ok(raw.par_chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
}

impl ResidualStreamCols {
    /// The seven columns `extract.py --residual-stream` writes, plus `eps` and `1/d` from
    /// the manifest.  Absent `residual_stream` in the manifest the layer simply has none and
    /// the composition runs as it did before.
    pub fn load(dir: &str, man: &serde_json::Value, tokens: usize) -> Result<Option<Self>, String> {
        let g = &man["residual_stream"];
        let Some(d) = g["d"].as_u64().map(|v| v as usize) else {
            return Ok(None);
        };
        let rd = |n: &str| -> Result<Vec<u16>, String> {
            let v = read_u16(&format!("{dir}/{n}.u16"))?;
            if v.len() < tokens * d {
                return Err(format!("{n}: {} values is under {tokens} x {d}", v.len()));
            }
            let _s = witread::scope(Site::DecodeTrunc);
            Ok(v[..tokens * d].to_vec())
        };
        let dff = g["dff"].as_u64().unwrap_or(0) as usize;
        // A column the norm scales and the SwiGLU only *check* against is optional:
        // an older dump that does not carry it costs the reference, not the
        // component, and the record says which of the two happened.
        let opt = |n: &str, w: usize| -> Option<Vec<u16>> {
            read_u16(&format!("{dir}/{n}.u16"))
                .ok()
                .filter(|v| v.len() >= tokens * w)
                .map(|v| {
                    let _s = witread::scope(Site::DecodeTrunc);
                    v[..tokens * w].to_vec()
                })
        };
        // `r` is one value per token, not per channel
        let rd1 = |n: &str| -> Result<Vec<u16>, String> {
            let v = read_u16(&format!("{dir}/{n}.u16"))?;
            if v.len() < tokens {
                return Err(format!("{n}: {} values is under {tokens}", v.len()));
            }
            let _s = witread::scope(Site::DecodeTrunc);
            Ok(v[..tokens].to_vec())
        };
        Ok(Some(Self {
            d,
            dff,
            tokens,
            xin: rd("xin")?,
            o_out: rd("o_out")?,
            xmid: rd("xmid")?,
            d_out: rd("d_out")?,
            xout: rd("xout")?,
            r1: rd1("n1_out")?,
            r2: rd1("n2_out")?,
            w1: read_u16(&format!("{dir}/w1.u16"))?,
            w2: read_u16(&format!("{dir}/w2.u16"))?,
            h1: opt("x_q", d),
            h2: opt("x_gate", d),
            silu_in: opt("silu_in", dff),
            silu_out: opt("silu_out", dff),
            x_down: opt("x_down", dff),
            eps: g["eps"].as_u64().ok_or("manifest: residual_stream.eps")? as u16,
            inv_d: g["inv_d"].as_u64().ok_or("manifest: residual_stream.inv_d")? as u16,
            reduce: g["reduce"].as_str().unwrap_or("").to_string(),
            head: false,
        }))
    }

    /// Stream II (E11): the head chunk's own five columns, which
    /// `extract.py --head` writes -- the last layer's residual, the model's
    /// final-norm weight, B1's `rsqrt` in and out and the head's bf16 input.
    /// `None` on a dump written before this, in which case the head chunk
    /// composes as it did and the final norm stays proved by nobody.
    pub fn load_head(dir: &str, man: &serde_json::Value, tokens: usize)
                     -> Result<Option<Self>, String> {
        let h = &man["head"];
        if !h["final_norm"].as_bool().unwrap_or(false) {
            return Ok(None);
        }
        let d = h["d_in"].as_u64().ok_or("manifest: head.d_in")? as usize;
        let cut = |v: Vec<u16>, n: usize, name: &str| -> Result<Vec<u16>, String> {
            if v.len() < n {
                return Err(format!("{name}: {} values is under {n}", v.len()));
            }
            let _s = witread::scope(Site::DecodeTrunc);
            Ok(v[..n].to_vec())
        };
        Ok(Some(Self {
            d,
            dff: 0,
            tokens,
            xin: cut(read_u16(&format!("{dir}/xin_head.u16"))?, tokens * d, "xin_head")?,
            o_out: vec![],
            xmid: vec![],
            d_out: vec![],
            xout: vec![],
            r1: cut(read_u16(&format!("{dir}/r_head.u16"))?, tokens, "r_head")?,
            r2: vec![],
            w1: cut(read_u16(&format!("{dir}/w_head.u16"))?, d, "w_head")?,
            w2: vec![],
            h1: read_u16(&format!("{dir}/h_head.u16"))
                .ok()
                .filter(|v| v.len() >= tokens * d)
                .map(|v| v[..tokens * d].to_vec()),
            h2: None,
            silu_in: None,
            silu_out: None,
            x_down: None,
            eps: h["eps"].as_u64().ok_or("manifest: head.eps")? as u16,
            inv_d: h["inv_d"].as_u64().ok_or("manifest: head.inv_d")? as u16,
            reduce: h["reduce"].as_str().unwrap_or("").to_string(),
            head: true,
        }))
    }

    /// Stream LL (E11): the wiring family's **model half** -- the two norm
    /// weights and the manifest's `eps` / `inv_d`, which are the only things
    /// this family puts in tree 0 (stream II's `Src::Pre` norm weights and the
    /// `NormR` table's two constants).  Every other column here is the chunk's
    /// residual stream, so it is allocated at its shape and left zero.
    pub fn load_weights(dir: &str, man: &serde_json::Value, tokens: usize)
                        -> Result<Option<Self>, String> {
        let g = &man["residual_stream"];
        let Some(d) = g["d"].as_u64().map(|v| v as usize) else {
            return Ok(None);
        };
        let dff = g["dff"].as_u64().unwrap_or(0) as usize;
        Ok(Some(Self {
            d,
            dff,
            tokens,
            xin: vec![0u16; tokens * d],
            o_out: vec![0u16; tokens * d],
            xmid: vec![0u16; tokens * d],
            d_out: vec![0u16; tokens * d],
            xout: vec![0u16; tokens * d],
            r1: vec![0u16; tokens],
            r2: vec![0u16; tokens],
            w1: model_u16(dir, man, "w1.u16")?,
            w2: model_u16(dir, man, "w2.u16")?,
            h1: None,
            h2: None,
            silu_in: None,
            silu_out: None,
            x_down: None,
            eps: g["eps"].as_u64().ok_or("manifest: residual_stream.eps")? as u16,
            inv_d: g["inv_d"].as_u64().ok_or("manifest: residual_stream.inv_d")? as u16,
            reduce: g["reduce"].as_str().unwrap_or("").to_string(),
            head: false,
        }))
    }

    /// The head chunk's final norm, model half: `w_head.u16` and the two
    /// constants.  `ResidualStreamCols::load_weights`' counterpart.
    pub fn load_head_weights(dir: &str, man: &serde_json::Value, tokens: usize)
                             -> Result<Option<Self>, String> {
        let h = &man["head"];
        if !h["final_norm"].as_bool().unwrap_or(false) {
            return Ok(None);
        }
        let d = h["d_in"].as_u64().ok_or("manifest: head.d_in")? as usize;
        let w1 = model_u16(dir, man, "w_head.u16")?;
        if w1.len() < d {
            return Err(format!("w_head: {} values is under {d}", w1.len()));
        }
        Ok(Some(Self {
            d,
            dff: 0,
            tokens,
            xin: vec![0u16; tokens * d],
            o_out: vec![],
            xmid: vec![],
            d_out: vec![],
            xout: vec![],
            r1: vec![0u16; tokens],
            r2: vec![],
            w1: w1[..d].to_vec(),
            w2: vec![],
            h1: None,
            h2: None,
            silu_in: None,
            silu_out: None,
            x_down: None,
            eps: h["eps"].as_u64().ok_or("manifest: head.eps")? as u16,
            inv_d: h["inv_d"].as_u64().ok_or("manifest: head.inv_d")? as u16,
            reduce: h["reduce"].as_str().unwrap_or("").to_string(),
            head: true,
        }))
    }
}

impl ResidualStreamCols {
    /// Stream II: the wiring family on a synthetic layer, so the desk can run
    /// the whole composed statement -- bindings, norm scales, SwiGLU and all -- at a
    /// shape that proves in seconds and without a dump.  Only the columns the
    /// components *consume* are built: `xmid`, `xout` and the five reference
    /// columns are what the components compute, and a synthetic layer has no B1 to
    /// compare against, so they are absent and the run reports them as absent.
    ///
    /// The exponent spread is deliberately narrow.  A residual stream of wildly
    /// scaled values gives a sum of squares whose `rsqrt` leaves bf16's normal
    /// range, and `h = (x r) w` then lands off the `VAL` table -- the gadget's
    /// incompleteness edge, not a soundness one, and not what this shape is for.
    pub fn synthetic(tokens: usize, d: usize, dff: usize, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut col = |n: usize, spread: i32| -> Vec<u16> {
            (0..n)
                .map(|_| {
                    let e: i32 = rng.random_range(-spread..spread);
                    let m: f64 = rng.random_range(-2.0f64..2.0);
                    crate::encoding::canon(bf16::from_f64(m * (e as f64).exp2()))
                })
                .collect()
        };
        Self {
            d,
            dff,
            tokens,
            xin: col(tokens * d, 2),
            o_out: col(tokens * d, 2),
            xmid: vec![],
            d_out: col(tokens * d, 2),
            xout: vec![],
            // the derived `r` has no B1 counterpart here; the run says so
            r1: vec![],
            r2: vec![],
            w1: col(d, 2),
            w2: col(d, 2),
            h1: None,
            h2: None,
            silu_in: None,
            silu_out: None,
            x_down: None,
            eps: bf16::from_f64(1e-5),
            inv_d: bf16::from_f64(1.0 / d as f64),
            reduce: "bf16-norm".into(),
            head: false,
        }
    }

    /// Stream II: the head chunk's family on a synthetic layer.  One residual
    /// column and one norm weight; everything else a head chunk has not got.
    pub fn synthetic_head(tokens: usize, d: usize, seed: u64) -> Self {
        let mut c = Self::synthetic(tokens, d, d, seed);
        c.dff = 0;
        c.o_out = vec![];
        c.d_out = vec![];
        c.w2 = vec![];
        c.head = true;
        c
    }
}


/// Stream PP (E11), box half: the model half of a dump, when the derivation is
/// the **shape** tree's and the files need not exist.
///
/// `expect-root --shape-only` commits tree 0 alone, and tree 0 holds no model
/// value. That is not an argument here, it is a measurement: on the 8B chunk
/// `B8L16` the shape root is bit-identical against a dump whose every model
/// file has been replaced by the same number of zero bytes, while the model
/// root moves (the box record's section 14b). So under `--shape-only` these
/// four readers **allocate instead of reading**, and a directory holding only
/// `manifest.json` is a whole dump as far as the shape root is concerned.
///
/// The lengths are the manifest's own: `files[<name>].values` is the element
/// count the extractor wrote, so the allocation is exact rather than inferred
/// from a shape the loader would have to re-derive. A file the manifest does
/// not describe falls through to the real read, which is what keeps a dump
/// written before the `files` map existed working.
///
/// This is `Layer::from_params` (the brief's open item 1) in the form the box
/// half could reach: the files are not *opened*, though the manifest still is.
fn model_values(man: &serde_json::Value, name: &str) -> Option<usize> {
    if !crate::compose::SHAPE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    man["files"][name]["values"].as_u64().map(|v| v as usize)
}
fn model_i8(dir: &str, man: &serde_json::Value, name: &str) -> Result<Vec<i32>, String> {
    match model_values(man, name) {
        Some(n) => Ok(vec![0i32; n]),
        None => read_i8(&format!("{dir}/{name}")),
    }
}
fn model_i32(dir: &str, man: &serde_json::Value, name: &str) -> Result<Vec<i64>, String> {
    match model_values(man, name) {
        Some(n) => Ok(vec![0i64; n]),
        None => read_i32(&format!("{dir}/{name}")),
    }
}
fn model_i32_narrow(dir: &str, man: &serde_json::Value, name: &str)
                    -> Result<Vec<i32>, String> {
    match model_values(man, name) {
        Some(n) => Ok(vec![0i32; n]),
        None => read_i32_narrow(&format!("{dir}/{name}")),
    }
}
fn model_u16(dir: &str, man: &serde_json::Value, name: &str) -> Result<Vec<u16>, String> {
    match model_values(man, name) {
        Some(n) => Ok(vec![0u16; n]),
        None => read_u16(&format!("{dir}/{name}")),
    }
}

fn read_i32(path: &str) -> Result<Vec<i64>, String> {
    let raw = witread::read_file(path)?;
    let _s = witread::scope(Site::DecodeI32);
    Ok(raw.par_chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        .collect())
}
fn read_i8(path: &str) -> Result<Vec<i32>, String> {
    let raw = witread::read_file(path)?;
    let _s = witread::scope(Site::DecodeI8);
    Ok(raw.par_iter().map(|b| *b as i8 as i32).collect())
}
/// Stream Z: the widened attention output commits `qx_o` at int16 storage.
fn read_i16(path: &str) -> Result<Vec<i32>, String> {
    let raw = witread::read_file(path)?;
    let _s = witread::scope(Site::DecodeI8);
    Ok(raw.par_chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
        .collect())
}
/// The same file as `i32`.  `read_i32` widens every value to `i64` and the
/// columns that want `i32` then narrow it back, so this is a second pass over
/// an array that was just built, which is what `decode/recast` counts.
fn read_i32_narrow(path: &str) -> Result<Vec<i32>, String> {
    let v = read_i32(path)?;
    let _s = witread::scope(Site::DecodeRecast);
    Ok(v.par_iter().map(|x| *x as i32).collect())
}

/// The Freivalds groups: matmuls sharing one committed input.
/// B10 step 4: `lm_head` is its own group.  It shares no committed input with the
/// layer's matmuls -- its input is the *final* norm's output, not any layer's --
/// so it Freivalds checks alone, over `d` input channels like every other group.
pub const FREIVALDS_GROUPS: [(&str, &[&str]); 5] =
    [("qkv", &["q", "k", "v"]), ("o", &["o"]), ("gateup", &["gate", "up"]),
     ("down", &["down"]), ("head", &["lm_head"])];

impl Layer {
    pub fn load(dir: &str, cfg: &Src, key_tile_sizes: &[usize], variant: Variant,
                round: bf16::Round) -> Result<Self, String> {
        let man: serde_json::Value = witread::read_manifest(dir)?;
        // Issue #53: before the dump is read at all, let alone committed -- the
        // rounding the line names has to be the one this dump was written under.
        if let Some(why) = round_refusal(&man, round) {
            return Err(why);
        }
        let g = |k: &str| man[k].as_u64().ok_or(format!("manifest: missing {k}"));
        let (nh, s, c, hd) = (g("heads")? as usize, g("s")? as usize,
                              g("c")? as usize, g("hd")? as usize);
        assert_eq!(key_tile_sizes.iter().sum::<usize>(), nh, "key tiles must cover every head");
        // Stream LL (E11), log row 58: refused here rather than caught later, so
        // that no chunk whose tree 0 depends on its own scores can be composed at
        // all.  `Layer::load_weights` refuses on the same predicate.
        if let Some(why) = frac_shared_refusal(&man) {
            return Err(why);
        }
        // B9 step 1a: the master exponent is layer-wide.  `Attn::build` takes the
        // largest `frac` over the key tile's *own* heads, so which table a key tile builds
        // depends on how the heads were split -- and the three assertions below
        // then hold only where every key tile happens to contain a head at the
        // layer-wide maximum (4 of 24 Qwen-0.5B layers; 0 of 32 Llama-3.1-8B ones,
        // B8.md section 2 item 3).  The forward pass already writes every head's
        // exponent into the manifest, so the layer-wide maximum is read, not
        // derived, and handed to every key tile: each then scales its shifts up to it
        // and reads one table of `d_bits + spread` rows.  The rows are what the
        // sharing costs -- 2^`master_table_bits`, up to 8x a spread-0 layer's --
        // and `off_table` reports any key the chosen table cannot hold.
        let master: Option<f64> = man["frac_shared_per_head"].as_bool().unwrap_or(false)
            .then(|| man["frac_per_head"].as_array().map(|a| {
                a.iter().take(nh).filter_map(|v| v.as_f64()).fold(f64::MIN, f64::max)
            }))
            .flatten()
            .filter(|f| *f > f64::MIN);
        // B10 step 1: on a real dump `--table-bits 0` means *this layer's* table.
        // `Attn::build`'s own default caps at 2^22 -- right for a single-key-tile
        // benchmark, wrong for a model whose layers want 2^24..2^26 -- and every
        // driver so far has worked around it by reading `master_table_bits` out of
        // the manifest and passing it back in.  A chunk loop cannot: one
        // `--table-bits` cannot be three sizes.  So the manifest's own value is
        // the default here, and an explicit `--table-bits N` still overrides it.
        let cfg = &if cfg.log_table == 0 {
            match man["master_table_bits"].as_u64() {
                Some(m) if m > 0 => Src { log_table: m as u32, ..cfg.clone() },
                _ => cfg.clone(),
            }
        } else {
            cfg.clone()
        };
        // B10 step 3: a dump extracted with `--diagonal-block` carries the diagonal block
        // beside the prefix rectangle, and it rides as one more key tile per prefix
        // key tile -- same `insts`, same head offset, `C = S`.
        let has_diagonal_block = man["diagonal_block"].as_bool().unwrap_or(false);
        // Stream W: the layer's attention columns, read and widened once for
        // every key tile that will gather out of them.  Each of the six key tiles used
        // to open these files for itself -- `qq.i16` and `mm.i32` six times, a
        // half's `kk`, `vv` and `num` three times each -- so a chunk read
        // 1,502 MB where 740 MB exists and the read was the largest single
        // site stream P left (`briefs/p-witread.md` (6) item 1).  Nothing
        // about the bytes or the layout changed; the same gathers now run over
        // one image instead of over three copies of it.
        // Open item 1: block 0 of a sequence has no unchecked prefix, so `c == 0`,
        // `kk.i16` is empty and there is no prefix half to read and no prefix
        // key tile to build -- the diagonal block is the group's only key tile and carries its
        // per-token component (`compose.rs`'s `rows_key_tile`, `Powers::as_solo_diagonal_block`).
        assert!(c > 0 || has_diagonal_block,
                "a c = 0 dump is block 0: it is the diagonal block and needs a `--diagonal-block` dump");
        let raw = {
            let _s = witread::scope(Site::KeyTiles);
            AttnRaw::read(dir, &man, c > 0, has_diagonal_block)?
        };
        let mut key_tiles = vec![];
        let mut head_offsets = vec![];
        let mut sizes = vec![];
        let mut off = 0usize;
        // Stream AA (E11), layout design A: with `--kv-boundary` the unchecked prefix
        // is loaded as `c / s` sub-key-tiles of exactly `S` keys each rather than as
        // one rectangle, because that is what makes a consumer's sub-key-tile `b`
        // the same `Shape` -- and so the same `LayoutKey`, and so a column of
        // the same height -- as the producer's diagonal block.  `Attn::load_from` reads
        // the same bytes with a key offset; nothing about the dump changes.
        assert!(!cfg.tile || cfg.diagonal_block, "`--kv-boundary` needs a `--diagonal-block` dump");
        assert!(!cfg.tile || c % s == 0,
                "`--kv-boundary` tiles the prefix in blocks of S: C={c} is not a multiple of S={s}");
        let nsub = if c == 0 { 0 } else if cfg.tile { c / s } else { 1 };
        let sub_c = if cfg.tile { s } else { c };
        {
            let _s = witread::scope(Site::KeyTiles);
            for insts in key_tile_sizes {
                for k in 0..nsub {
                    let sh = Shape { s, c: sub_c, hd, insts: *insts };
                    let src = Src { head: off, ..cfg.clone() };
                    key_tiles.push(Attn::load_from(&raw, &man, off, sh, variant, src.nmax,
                                               src.log_table, master, false, k * sub_c)?);
                    head_offsets.push(off);
                    sizes.push(*insts);
                }
                off += insts;
            }
        }
        if has_diagonal_block {
            let mut off = 0usize;
            {
                let _s = witread::scope(Site::KeyTiles);
                for insts in key_tile_sizes {
                    let sh = Shape { s, c: s, hd, insts: *insts };
                    let src = Src { head: off, ..cfg.clone() };
                    key_tiles.push(Attn::load_from(&raw, &man, off, sh, variant, src.nmax,
                                               src.log_table, master, true, 0)?);
                    head_offsets.push(off);
                    sizes.push(*insts);
                    off += insts;
                }
            }
            // the per-token component belongs to the pair, and the denominator it
            // commits is the total: `Z >= nmax` is true of the sum and of neither
            // half, because the key whose shift is zero may be in either
            //
            // Stream AA: with the prefix tiled, "the pair" is a head key tile
            // group -- `nsub` sub-key-tiles and the diagonal block -- and the total is over
            // all of them.  The component lives on the group's *first* sub-key-tile,
            // which is `plan.rows_key_tile`'s choice on the other side of the wall.
            let n_pre = key_tile_sizes.len() * nsub;
            // the key tile that carries group `g`'s component: its first prefix
            // sub-key-tile, or -- block 0, no prefix at all -- its diagonal block.
            // `g * nsub` alone is key tile 0 for every group when `nsub == 0`, and
            // that read as "the extractor and the AIR disagree about the mask
            // on 16351 of 16384 rows" (open item 1) when every row agreed.
            let first = |g: usize| if nsub > 0 { g * nsub } else { n_pre + g };
            {
                let _s = witread::scope(Site::ZzTotal);
                for g in 0..key_tile_sizes.len() {
                    let mut tot = vec![0i64; key_tiles[first(g)].zz.len()];
                    for b in (g * nsub..(g + 1) * nsub).chain(std::iter::once(n_pre + g)) {
                        assert_eq!(key_tiles[b].zz.len(), tot.len(),
                                   "key tile {b} of group {g} has a different row count");
                        for (a, x) in tot.iter_mut().zip(&key_tiles[b].zz) {
                            *a += x;
                        }
                    }
                    key_tiles[first(g)].zz_total = Some(tot);
                }
            }
            // Rule 21, on the join: the prover recomputes both halves' numerators
            // from `qq`, `kk`, `vv` and the dump's row maximum, and the extractor
            // computed the same thing in torch.  `zz.i32` is the extractor's total,
            // so this is the two implementations agreeing about the causal
            // computation -- and it is the only check that would catch a mask off
            // by one, which is self-consistent on either side alone.
            let _s = witread::scope(Site::CheckRule21);
            let zz32 = read_i32(&format!("{dir}/zz.i32"))?;
            let mut bad = 0usize;
            for (g, insts) in key_tile_sizes.iter().enumerate() {
                let tot = key_tiles[first(g)].zz_total.as_ref().unwrap();
                for i in 0..*insts {
                    let h = head_offsets[first(g)] + i;
                    for t in 0..s {
                        if tot[i * s + t] != zz32[h * s + t] {
                            bad += 1;
                        }
                    }
                }
            }
            assert_eq!(bad, 0,
                       "the prover's total denominator disagrees with the dump's `zz.i32` \
                        on {bad} of {} (head, token) rows -- the extractor and the AIR do \
                        not agree about the causal mask", nh * s);
        }
        // The image has served every key tile it will serve, and it is a third of a
        // gigabyte.  Freeing it here rather than at the end of the function
        // keeps the layer's high-water mark where it was before this stream.
        drop(raw);
        let key_tile_sizes = &sizes[..];
        // one master table serves every key tile: same rows, same contents.  With a
        // layer-wide `master` this holds by construction; without one (a dump whose
        // Q scale is not per head) it is the old precondition, and the message
        // names the fix.
        {
            let _s = witread::scope(Site::CheckTableEq);
            for b in &key_tiles[1..] {
                assert_eq!(b.log_table, key_tiles[0].log_table, "key tiles disagree on table rows");
                assert_eq!(b.table, key_tiles[0].table,
                           "key_tiles disagree on the exp table (no layer-wide master exponent: \
                            the dump has frac_shared_per_head = false)");
                assert_eq!(b.frac, key_tiles[0].frac, "key tiles disagree on the master exponent");
            }
        }

        // B10 step 0: the dump names its own dequant layout, so a `b10-design`
        // witness cannot be composed under B6.5's mixed one by accident (the
        // scales it carries for q/k/v/o are general bf16 and `jx`/`jw` are the
        // -1000 sentinel, which the Pow2 gadget would silently commit).  Older
        // dumps have no such key and are B6.5's by construction; `--layout`
        // overrides both, for the synthetic shapes the emit chain uses.
        let vars = if !cfg.layout.is_empty() {
            mat_variants(&cfg.layout)
        } else if let serde_json::Value::Object(o) = &man["deq_variants"] {
            let mut v = MAT_VARIANTS;
            for (i, name) in MAT_NAMES.iter().enumerate() {
                let d = o.get(*name).and_then(|x| x.as_str())
                    .ok_or(format!("manifest: deq_variants is missing {name}"))?;
                v[i] = (name, match d {
                    "pow2" => DeqVariant::Pow2,
                    "chain" => DeqVariant::Chain,
                    // stream Z: the delayed requantize -- the manifest declares
                    // it, so a fold witness cannot compose under the chain
                    "fold" => DeqVariant::Fold,
                    other => return Err(format!("manifest: {name} has dequant {other}")),
                });
            }
            v
        } else {
            MAT_VARIANTS
        };
        let mut mats = vec![];
        let mats_scope = witread::scope(Site::Mats);
        // Stream Z: a widened attention output stores `o`'s quantized input at
        // int16.  The manifest says so; a dump without the key is int8.
        let attn_out_bits = man["attn_out_bits"].as_u64().unwrap_or(8) as u32;
        for (name, var) in vars {
            let acc = read_i32(&format!("{dir}/acc_{name}.i32"))?;
            let qx = if name == "o" && attn_out_bits > 8 {
                read_i16(&format!("{dir}/qx_{name}.i16"))?
            } else {
                read_i8(&format!("{dir}/qx_{name}.i8"))?
            };
            let qw = read_i8(&format!("{dir}/qw_{name}.i8"))?;
            let am = &man["accumulators"][name];
            let (n_in, n_out, tokens) = (am["n_in"].as_u64().unwrap() as usize,
                                         am["n_out"].as_u64().unwrap() as usize,
                                         am["tokens"].as_u64().unwrap() as usize);
            assert_eq!(acc.len(), tokens * n_out, "{name}: acc shape");
            assert_eq!(qx.len(), tokens * n_in, "{name}: qx shape");
            assert_eq!(qw.len(), n_in * n_out, "{name}: qw shape");
            let jx = read_i32_narrow(&format!("{dir}/jx_{name}.i32"))?;
            let jw = read_i32_narrow(&format!("{dir}/jw_{name}.i32"))?;
            let pair = |v: Vec<i64>| -> Vec<(u32, u32)> {
                let _s = witread::scope(Site::DecodeRecast);
                v.chunks_exact(2).map(|p| (p[0] as u32, p[1] as u32)).collect()
            };
            let sx = pair(read_i32(&format!("{dir}/sx_{name}.i32"))?);
            let sw = pair(read_i32(&format!("{dir}/sw_{name}.i32"))?);
            mats.push(Mat { name: name.into(), variant: var, n_in, n_out, tokens,
                            acc, qx, qw, jx, jw, sx, sw, jv2: vec![] });
        }
        // B10 step 4: a dump written with `--head` carries `lm_head` as an eighth
        // matmul, in the same file layout as the seven, and declares its rung in
        // `deq_variants` like they do.  Loaded here so the head can be composed
        // through the machinery that already proves a matmul, rather than through
        // a second one.
        if man["accumulators"]["lm_head"].is_object() {
            let name = "lm_head";
            let var = match man["deq_variants"][name].as_str() {
                Some("pow2") => DeqVariant::Pow2,
                Some("chain") => DeqVariant::Chain,
                other => return Err(format!("manifest: lm_head deq_variant {other:?}")),
            };
            let acc = read_i32(&format!("{dir}/acc_{name}.i32"))?;
            let qx = read_i8(&format!("{dir}/qx_{name}.i8"))?;
            let qw = read_i8(&format!("{dir}/qw_{name}.i8"))?;
            let am = &man["accumulators"][name];
            let (n_in, n_out, tokens) = (am["n_in"].as_u64().unwrap() as usize,
                                         am["n_out"].as_u64().unwrap() as usize,
                                         am["tokens"].as_u64().unwrap() as usize);
            assert_eq!(acc.len(), tokens * n_out, "lm_head: acc shape");
            assert_eq!(qx.len(), tokens * n_in, "lm_head: qx shape");
            assert_eq!(qw.len(), n_in * n_out, "lm_head: qw shape");
            let jx = read_i32_narrow(&format!("{dir}/jx_{name}.i32"))?;
            let jw = read_i32_narrow(&format!("{dir}/jw_{name}.i32"))?;
            let pair = |v: Vec<i64>| -> Vec<(u32, u32)> {
                let _s = witread::scope(Site::DecodeRecast);
                v.chunks_exact(2).map(|p| (p[0] as u32, p[1] as u32)).collect()
            };
            let sx = pair(read_i32(&format!("{dir}/sx_{name}.i32"))?);
            let sw = pair(read_i32(&format!("{dir}/sw_{name}.i32"))?);
            mats.push(Mat { name: name.into(), variant: var, n_in, n_out, tokens,
                            acc, qx, qw, jx, jw, sx, sw, jv2: vec![] });
        }

        drop(mats_scope);

        // the attnout component proves what the key tiles prove -- which with the diagonal block
        // is two key tiles per head, so a head's totals accumulate
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        {
            let _s = witread::scope(Site::AttnOut);
            for (bi, b) in key_tiles.iter().enumerate() {
                for i in 0..key_tile_sizes[bi] {
                    let h = head_offsets[bi] + i;
                    for t in 0..s {
                        den[h * s + t] += b.zz[i * s + t];
                        for j in 0..hd {
                            acc[(h * s + t) * hd + j] += b.out[(i * s + t) * hd + j];
                        }
                    }
                }
            }
        }
        let jo: Vec<i32> = read_i32_narrow(&format!("{dir}/jo.i32"))?;
        let jv: Vec<i32> = read_i32_narrow(&format!("{dir}/jv.i32"))?;
        assert_eq!(jo.len(), s, "jo shape");
        assert_eq!(jv.len(), nh * hd, "jv shape");
        // Stream Z: a fold matmul's consumer scale.  `jv.i32` is per query
        // head; `v`'s output channels are per KV head, so the GQA groups must
        // agree before one copy is kept.
        for m in &mut mats {
            if m.variant != DeqVariant::Fold {
                continue;
            }
            assert_eq!(m.name, "v", "the fold's only wired site is v");
            let rep = nh * hd / m.n_out;
            assert!(rep >= 1 && m.n_out % hd == 0, "fold: v shape vs heads");
            m.jv2 = (0..m.n_out)
                .map(|c| {
                    let (kvh, j) = (c / hd, c % hd);
                    let v0 = jv[(kvh * rep) * hd + j];
                    for g in 1..rep {
                        assert_eq!(jv[(kvh * rep + g) * hd + j], v0,
                                   "fold: jv disagrees inside a GQA group");
                    }
                    v0
                })
                .collect();
        }
        // Stream II (E11): the wiring family is the layer's when the dump
        // carries `residual_stream`, and the *head* chunk's final norm when it does not
        // but carries `head.final_norm`.  No flag: which family a chunk can
        // prove is a property of the columns the extractor wrote, and a dump
        // written before either simply has none, as before.
        let residual_stream = match ResidualStreamCols::load(dir, &man, s)? {
            Some(g) => Some(g),
            None => ResidualStreamCols::load_head(dir, &man, s)?,
        };
        // stream E: B1's key column, when the dump carries a head.  Read as
        // patterns; the comparison happens on the claim, not value by value.
        let head_keys = man["head"]["vocab"].as_u64().and_then(|v| {
            let want = s * v as usize;
            read_u16(&format!("{dir}/key_head.u16")).ok().filter(|k| k.len() >= want)
                .map(|k| k[..want].to_vec())
        });
        let head_tok = head_keys.as_ref().and(
            read_i32(&format!("{dir}/tok_head.i32")).ok()
                .filter(|v| v.len() >= s)
                .map(|v| v[..s].iter().map(|x| *x as u32).collect()));
        // Stream BB (E11): the embedding matrix and the block's input ids, when
        // the dump carries them.  Both are optional here and both are required
        // by `--embed`; the flag panics on a dump without them rather than
        // standing anything in.
        let embed_tab = read_u16(&format!("{dir}/embed.u16")).ok();
        let tok_in = read_i32(&format!("{dir}/tok_in.i32")).ok()
            .filter(|v| v.len() >= s)
            .map(|v| v[..s].iter().map(|x| *x as u32).collect());
        Ok(Self {
            key_tiles,
            key_tile_sizes: key_tile_sizes.to_vec(),
            head_offsets,
            nsub,
            mats,
            ao: AttnOut { nh, s, hd, acc, den, jo, jv },
            source: format!("B6.5 layer witness [{dir}]"),
            residual_stream,
            head_keys,
            head_tok,
            attn_out_bits,
            embed_tab,
            tok_in,
            layer: man["layer"].as_u64().map(|v| v as usize),
            block: man["block"].as_u64().map(|v| v as usize),
            // stream NN: `extract.py` writes `inv_freq.f64`, `jq.i32` and
            // `jk.i32` beside the `jv.i32` it already wrote; a dump without
            // them proves no rotation and says so rather than guessing one.
            rope: RopeCols::load(dir, &man, nh, hd, s).unwrap_or(None),
        })
    }

    /// Stream LL (E11), `notes/witness-slack.md` section 5 item 1: the dump's
    /// **model half**, and nothing else.
    ///
    /// `Layer::load` reads the whole dump -- the key tiles' `qq`/`kk`/`vv`/`mm`/
    /// `num`/`zz`, every matmul's `acc` and `qx`, the wiring stream's seven
    /// activation columns -- and recomputes `Q.K^T` over the whole rectangle on
    /// the way. Tree 0 reads none of it: every preprocessed column is a function
    /// of `Fixed.params` and of the model, which is the weights (`qw_*.i8`), the
    /// weight scales (`jw_*.i32`, `sw_*.i32`), the attention output's offline
    /// `jv.i32`, the two norm weights, the embedding table, this stream's rope
    /// exponents, and the manifest's own `frac_per_head` / `master_table_bits`.
    /// So a verifier deriving a chunk's expected tree-0 root reads exactly those
    /// files here, and every activation array is allocated at its shape and left
    /// at a well-formed constant.
    ///
    /// Two consequences the caller has to honour, both checked rather than
    /// assumed:
    ///
    /// * the `Params` cannot be derived from this layer -- `ao_jo_min/max` and
    ///   `ao_acc_*_bits` come off the activations -- so `zk-prover expect-root
    ///   --weights-only` requires `--params FILE`, the statement's own
    ///   `Fixed.params`, and that is the right shape for the check anyway: the
    ///   verifier derives the root from *the statement's* own shape and the
    ///   model, not from the prover's re-derivation of either;
    /// * the table size must be pinned (the manifest's `master_table_bits` or
    ///   `--table-bits N`), because `--tight-table` sizes it off the witness's
    ///   own shifts and there is no witness here.
    pub fn load_weights(dir: &str, cfg: &Src, key_tile_sizes: &[usize], variant: Variant)
                        -> Result<Self, String> {
        let man: serde_json::Value = witread::read_manifest(dir)?;
        let g = |k: &str| man[k].as_u64().ok_or(format!("manifest: missing {k}"));
        let (nh, s, c, hd) = (g("heads")? as usize, g("s")? as usize,
                              g("c")? as usize, g("hd")? as usize);
        assert_eq!(key_tile_sizes.iter().sum::<usize>(), nh, "key tiles must cover every head");
        if let Some(why) = frac_shared_refusal(&man) {
            return Err(why);
        }
        let master: Option<f64> = man["frac_per_head"].as_array().map(|a| {
            a.iter().take(nh).filter_map(|v| v.as_f64()).fold(f64::MIN, f64::max)
        }).filter(|f| *f > f64::MIN);
        let cfg = &if cfg.log_table == 0 {
            match man["master_table_bits"].as_u64() {
                Some(m) if m > 0 => Src { log_table: m as u32, ..cfg.clone() },
                _ => cfg.clone(),
            }
        } else {
            cfg.clone()
        };
        let has_diagonal_block = man["diagonal_block"].as_bool().unwrap_or(false);
        assert!(c > 0 || has_diagonal_block,
                "a c = 0 dump is block 0: it is the diagonal block and needs a `--diagonal-block` dump");
        assert!(!cfg.tile || cfg.diagonal_block, "`--kv-boundary` needs a `--diagonal-block` dump");
        assert!(!cfg.tile || c % s == 0,
                "`--kv-boundary` tiles the prefix in blocks of S: C={c} is not a multiple of S={s}");
        let q_bits = man["qk_bits"][0].as_u64().unwrap_or(8) as u32;
        let k_bits = man["qk_bits"][1].as_u64().unwrap_or(8) as u32;
        // the per-head exponents this key tile's instances take, in `Attn::load_from`'s
        // own order, so `dinv` is the column the full load builds
        let fracs_at = |head: usize, insts: usize| -> Vec<f64> {
            (0..insts)
                .map(|i| man["frac_per_head"][(head + i) % nh].as_f64().unwrap_or(0.0))
                .collect()
        };
        let mut key_tiles = vec![];
        let mut head_offsets = vec![];
        let mut sizes = vec![];
        let nsub = if c == 0 { 0 } else if cfg.tile { c / s } else { 1 };
        let sub_c = if cfg.tile { s } else { c };
        let mut off = 0usize;
        for insts in key_tile_sizes {
            for k in 0..nsub {
                let sh = Shape { s, c: sub_c, hd, insts: *insts };
                key_tiles.push(Attn::weights_only(
                    sh, variant, q_bits, k_bits, cfg.nmax, cfg.log_table,
                    fracs_at(off, *insts), master, false,
                    format!("model-only [{nh}x{s}x{c}, heads {off}..+{insts}, keys {}..]",
                            k * sub_c)));
                head_offsets.push(off);
                sizes.push(*insts);
            }
            off += insts;
        }
        if has_diagonal_block {
            let mut off = 0usize;
            for insts in key_tile_sizes {
                let sh = Shape { s, c: s, hd, insts: *insts };
                key_tiles.push(Attn::weights_only(
                    sh, variant, q_bits, k_bits, cfg.nmax, cfg.log_table,
                    fracs_at(off, *insts), master, true,
                    format!("model-only [{nh}x{s}x{s}, heads {off}..+{insts}, diagonal block]")));
                head_offsets.push(off);
                sizes.push(*insts);
                off += insts;
            }
            // `Layer::load`'s own accumulation, over the stand-in denominators:
            // the group's component commits the total, so the total has to be the
            // sum of the parts here too or the AO fold divides by the wrong one
            let n_pre = key_tile_sizes.len() * nsub;
            let first = |g: usize| if nsub > 0 { g * nsub } else { n_pre + g };
            for gi in 0..key_tile_sizes.len() {
                let mut tot = vec![0i64; key_tiles[first(gi)].zz.len()];
                for b in (gi * nsub..(gi + 1) * nsub).chain(std::iter::once(n_pre + gi)) {
                    for (a, x) in tot.iter_mut().zip(&key_tiles[b].zz) {
                        *a += x;
                    }
                }
                key_tiles[first(gi)].zz_total = Some(tot);
            }
        }
        let key_tile_sizes = &sizes[..];
        let vars = if !cfg.layout.is_empty() {
            mat_variants(&cfg.layout)
        } else if let serde_json::Value::Object(o) = &man["deq_variants"] {
            let mut v = MAT_VARIANTS;
            for (i, name) in MAT_NAMES.iter().enumerate() {
                let d = o.get(*name).and_then(|x| x.as_str())
                    .ok_or(format!("manifest: deq_variants is missing {name}"))?;
                v[i] = (name, match d {
                    "pow2" => DeqVariant::Pow2,
                    "chain" => DeqVariant::Chain,
                    "fold" => DeqVariant::Fold,
                    other => return Err(format!("manifest: {name} has dequant {other}")),
                });
            }
            v
        } else {
            MAT_VARIANTS
        };
        let attn_out_bits = man["attn_out_bits"].as_u64().unwrap_or(8) as u32;
        let mut mats = vec![];
        let mut one_mat = |name: &str, var: DeqVariant| -> Result<(), String> {
            let am = &man["accumulators"][name];
            let (n_in, n_out, tokens) = (am["n_in"].as_u64().unwrap() as usize,
                                         am["n_out"].as_u64().unwrap() as usize,
                                         am["tokens"].as_u64().unwrap() as usize);
            let qw = model_i8(dir, &man, &format!("qw_{name}.i8"))?;
            let jw = model_i32_narrow(dir, &man, &format!("jw_{name}.i32"))?;
            let sw: Vec<(u32, u32)> = model_i32(dir, &man, &format!("sw_{name}.i32"))?
                .chunks_exact(2).map(|p| (p[0] as u32, p[1] as u32)).collect();
            assert_eq!(qw.len(), n_in * n_out, "{name}: qw shape");
            mats.push(Mat {
                name: name.into(), variant: var, n_in, n_out, tokens,
                acc: vec![0i64; tokens * n_out],
                qx: vec![0i32; tokens * n_in],
                qw,
                jx: vec![0i32; tokens],
                jw,
                // a per-token scale of 1.0, not of zero: `fold_gt` reads the
                // biased exponent and a subnormal there is not a shape the
                // dequant gadget is built for
                sx: vec![(0u32, 127u32); tokens],
                sw,
                jv2: vec![],
            });
            Ok(())
        };
        for (name, var) in vars {
            one_mat(name, var)?;
        }
        if man["accumulators"]["lm_head"].is_object() {
            let var = match man["deq_variants"]["lm_head"].as_str() {
                Some("pow2") => DeqVariant::Pow2,
                Some("chain") => DeqVariant::Chain,
                other => return Err(format!("manifest: lm_head deq_variant {other:?}")),
            };
            one_mat("lm_head", var)?;
        }
        let jv: Vec<i32> = model_i32_narrow(dir, &man, "jv.i32")?;
        if jv.len() != nh * hd {
            return Err(format!("jv.i32: {} values, need {}", jv.len(), nh * hd));
        }
        for m in &mut mats {
            if m.variant != DeqVariant::Fold {
                continue;
            }
            assert_eq!(m.name, "v", "the fold's only wired site is v");
            let rep = nh * hd / m.n_out;
            assert!(rep >= 1 && m.n_out % hd == 0, "fold: v shape vs heads");
            m.jv2 = (0..m.n_out).map(|ch| jv[((ch / hd) * rep) * hd + ch % hd]).collect();
        }
        // the AO component's denominators are the key tiles' stand-ins, summed exactly
        // as `Layer::load` sums the real ones
        let (acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for (bi, b) in key_tiles.iter().enumerate() {
            for i in 0..key_tile_sizes[bi] {
                let h = head_offsets[bi] + i;
                for t in 0..s {
                    den[h * s + t] += b.zz[i * s + t];
                }
            }
        }
        let mut residual_stream = match ResidualStreamCols::load_weights(dir, &man, s)? {
            Some(g) => Some(g),
            None => ResidualStreamCols::load_head_weights(dir, &man, s)?,
        };
        let vocab = man["head"]["vocab"].as_u64().map(|v| v as usize);
        // `.ok()`, because a layer that is not layer 0 has no embedding table
        // and its absence is the signal.  Under `--shape-only` the manifest's
        // own `files` map is that signal instead: a dump that declares the file
        // has the gadget whether or not the file is on disk.
        let embed_tab = model_values(&man, "embed.u16")
            .map(|n| vec![0u16; n])
            .or_else(|| read_u16(&format!("{dir}/embed.u16")).ok());
        // Stream BB's gadget checks the wiring family's committed input against
        // the embedding row the block's ids name (`EWitness::from_cols`), so on
        // a layer-0 chunk the two stand-ins have to agree with each other.  The
        // ids stand in as zero, so the residual stream stands in as row 0 --
        // which is model data, read from `embed.u16` and from nothing else.
        if let (Some(g), Some(e)) = (residual_stream.as_mut(), embed_tab.as_ref()) {
            if e.len() >= g.d && g.xin.len() == s * g.d {
                for t in 0..s {
                    g.xin[t * g.d..(t + 1) * g.d].copy_from_slice(&e[..g.d]);
                }
            }
        }
        Ok(Self {
            key_tiles,
            key_tile_sizes: key_tile_sizes.to_vec(),
            head_offsets,
            nsub,
            mats,
            ao: AttnOut { nh, s, hd, acc, den, jo: vec![0i32; s], jv },
            source: format!("B6.5 model-only witness [{dir}]"),
            residual_stream,
            // stream E's order key is a *committed* column and carries an
            // argmax claim, so the stand-in has to have a unique maximum per
            // row or the head chunk's argmax component has no answer to give.
            // Column 0 wins on every row, and `head_tok` says so.
            head_keys: vocab.map(|v| {
                let mut k = vec![1u16; s * v];
                for t in 0..s {
                    k[t * v] = 2;
                }
                k
            }),
            head_tok: vocab.map(|_| vec![0u32; s]),
            attn_out_bits,
            tok_in: embed_tab.as_ref().map(|_| vec![0u32; s]),
            embed_tab,
            layer: man["layer"].as_u64().map(|v| v as usize),
            block: man["block"].as_u64().map(|v| v as usize),
            rope: RopeCols::load(dir, &man, nh, hd, s).unwrap_or(None),
        })
    }

    /// Issue #112: the segment, from the **statement and the calibration**, with
    /// no dump on disk.
    ///
    /// `Layer::load_weights` reads no activation and, under `--shape-only`, no
    /// model file either -- but it still opens `manifest.json`, and it still
    /// builds the exponent table and every stand-in array off what it reads
    /// there. This builds the same segment from `Fixed.params` and the
    /// per-layer calibration instead, so `zk-prover expect-root --shape-only`
    /// and `zk-prover model-roots` derive a shape root with no dump in reach.
    /// A verifier holds both of those (issue #95): the statement is what it
    /// verifies against, and the calibration is what the model manifest
    /// publishes per layer.
    ///
    /// **What it is a function of is issue #18's answer, used as a
    /// constructor.** The shape tree is a function of the statement, which
    /// `Plan::new` turns into every layout and every table's width, and of the
    /// per-key-tile calibration -- the master exponent, the per-head exponents
    /// cycled into that key tile's instances, and the scale each of those gives
    /// (`shaperoot::Calibration`). Those are exactly the two things this takes.
    /// Everything else a segment carries -- its activations, its weights, its
    /// `jv`, its norm weights, its embedding table -- reaches the *model* tree
    /// and no column of the shape tree, so each is allocated at the shape the
    /// statement gives and left at the same well-formed constant
    /// `load_weights` leaves it at under `--shape-only`.
    ///
    /// Three consequences, all checked rather than assumed:
    ///
    /// * the key tiles are `Plan::new`'s own `shapes`, `head_offsets` and
    ///   `diagonal_block`, so the decomposition cannot disagree with the plan
    ///   the derivation then builds;
    /// * a calibration that does not describe this statement's key tiles is
    ///   refused by name -- the wrong number of key tiles, the wrong number of
    ///   instances, or a `delta` that is not the one the master exponent gives;
    /// * `jo` is committed data and no shape column reads it, so it stands in
    ///   at the statement's own `ao_jo_min` -- the window's floor, which is
    ///   where `expect-root --weights-only` already puts it.
    ///
    /// This derives a root; it does not check one. Whether a verifier compares
    /// the root it derives with the statement's is issue #95's question.
    pub fn from_params(p: &crate::compose::Params,
                       calib: &crate::shaperoot::Calibration) -> Self {
        let plan = crate::compose::Plan::new(p);
        let variant = Variant::parse(&p.variant);
        let (nh, s, hd) = (p.nh, p.s, p.hd);
        assert_eq!(calib.0.len(), plan.shapes.len(),
                   "the calibration names {} key tiles and this statement has {}",
                   calib.0.len(), plan.shapes.len());
        let mut key_tiles: Vec<Attn> = vec![];
        for (b, sh) in plan.shapes.iter().enumerate() {
            let k = &calib.0[b];
            let fracs: Vec<f64> = k.frac_per_inst.iter().map(|f| f64::from_bits(*f)).collect();
            assert_eq!(fracs.len(), sh.insts,
                       "key tile {b}: the calibration names {} instances and the statement {}",
                       fracs.len(), sh.insts);
            let frac = f64::from_bits(k.frac);
            let a = Attn::weights_only(
                *sh, variant, p.q_bits, p.k_bits, p.nmax, p.log_table,
                fracs, Some(frac), plan.diagonal_block[b],
                format!("from params [key tile {b}, heads {}..+{}, {} keys{}]",
                        plan.head_offsets[b], sh.insts, sh.c,
                        if plan.diagonal_block[b] { ", diagonal block" } else { "" }));
            // the scale the master exponent gives is `Attn::build`'s own
            // expression, so a calibration whose `delta` is not that is a
            // calibration of some other key tile and not of this one
            assert_eq!(a.delta, k.delta,
                       "key tile {b}: the calibration's delta is not the one its master \
                        exponent {frac} gives");
            key_tiles.push(a);
        }
        let key_tile_sizes: Vec<usize> = plan.shapes.iter().map(|sh| sh.insts).collect();
        // `Layer::load`'s own accumulation over the stand-in denominators: the
        // group's per-token component commits the total, so the total has to be
        // the sum of the parts here too or the attention-output fold divides by
        // the wrong one.
        if p.diagonal_block {
            for b in 0..key_tiles.len() {
                if !plan.rows_key_tile[b] {
                    continue;
                }
                let mut tot = vec![0i64; key_tiles[b].zz.len()];
                for other in 0..key_tiles.len() {
                    if plan.group[other] == plan.group[b] {
                        for (a, x) in tot.iter_mut().zip(&key_tiles[other].zz) {
                            *a += x;
                        }
                    }
                }
                key_tiles[b].zz_total = Some(tot);
            }
        }
        let mats: Vec<Mat> = p.mats.iter().map(|m| {
            let variant = DeqVariant::parse(&m.variant);
            Mat {
                name: m.name.clone(), variant, n_in: m.n_in, n_out: m.n_out, tokens: m.tokens,
                acc: vec![0i64; m.tokens * m.n_out],
                qx: vec![0i32; m.tokens * m.n_in],
                qw: vec![0i32; m.n_in * m.n_out],
                jx: vec![0i32; m.tokens],
                jw: vec![0i32; m.n_out],
                // a per-token scale of 1.0, not of zero, exactly as the
                // model-only load leaves it: `fold_gt` reads the biased
                // exponent and a subnormal there is not a shape the dequant
                // gadget is built for
                sx: vec![(0u32, 127u32); m.tokens],
                sw: vec![(0u32, 0u32); m.n_out],
                jv2: if variant == DeqVariant::Fold { vec![0i32; m.n_out] } else { vec![] },
            }
        }).collect();
        let (acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for (bi, b) in key_tiles.iter().enumerate() {
            for i in 0..key_tile_sizes[bi] {
                let h = plan.head_offsets[bi] + i;
                for t in 0..s {
                    den[h * s + t] += b.zz[i * s + t];
                }
            }
        }
        let residual_stream = p.residual_stream.as_ref().map(|g| {
            let head = g.head_norm;
            let col = |n: usize| if head { vec![] } else { vec![0u16; n] };
            ResidualStreamCols {
                d: g.d,
                dff: if head { 0 } else { g.dff },
                tokens: s,
                xin: vec![0u16; s * g.d],
                o_out: col(s * g.d),
                xmid: col(s * g.d),
                d_out: col(s * g.d),
                xout: col(s * g.d),
                r1: vec![0u16; s],
                r2: col(s),
                w1: vec![0u16; g.d],
                w2: col(g.d),
                h1: None,
                h2: None,
                silu_in: None,
                silu_out: None,
                x_down: None,
                eps: g.eps,
                inv_d: g.inv_d,
                // a report of the extraction's own reduction, which no
                // component reads and no statement carries
                reduce: String::new(),
                head,
            }
        });
        let embed_tab = (p.embed_vocab > 0).then(|| {
            let d = p.residual_stream.as_ref().map(|g| g.d)
                .expect("the embedding gadget needs the residual-stream family");
            vec![0u16; p.embed_vocab * d]
        });
        // stream E's order key is a committed column carrying an argmax claim,
        // so the stand-in has to have a unique maximum per row or the head
        // segment's argmax component has no answer to give. Column 0 wins on
        // every row, and `head_tok` says so.
        let vocab = (p.logits_claim || p.prove_argmax)
            .then(|| p.mats.iter().find(|m| m.name == "lm_head").map(|m| m.n_out))
            .flatten();
        Self {
            key_tiles,
            key_tile_sizes,
            head_offsets: plan.head_offsets.clone(),
            nsub: if p.c == 0 { 0 } else if p.kv_boundary { p.c / p.s } else { 1 },
            mats,
            ao: AttnOut {
                nh, s, hd, acc, den,
                // the window's floor: it makes `e = jv - jo` non-negative, so
                // the fold's `pj` is 1 and `denp` stays small. No shape column
                // reads it.
                jo: vec![p.ao_jo_min; s],
                jv: vec![0i32; nh * hd],
            },
            source: "B6.5 from params".to_string(),
            residual_stream,
            head_keys: vocab.map(|v| {
                let mut k = vec![1u16; s * v];
                for t in 0..s {
                    k[t * v] = 2;
                }
                k
            }),
            head_tok: vocab.map(|_| vec![0u32; s]),
            attn_out_bits: p.ao_q_bits + 1,
            tok_in: embed_tab.as_ref().map(|_| vec![0u32; s]),
            embed_tab,
            layer: None,
            block: None,
            rope: p.rope.as_ref().map(|r| RopeCols {
                nkv: r.nkv,
                pos0: r.pos0,
                theta: f64::from_bits(r.theta),
                rope_type: r.rope_type.clone(),
                rope_scale: r.rope_scale.iter().map(|b| f64::from_bits(*b)).collect(),
                // the dump's own `inv_freq` is a reference the derivation
                // compares against, and there is no dump here: the angle table
                // is built from the statement, which is what makes it a shape
                // column at all
                inv_ref: None,
                jq: vec![0i32; nh * hd],
                jk: vec![0i32; nh * hd],
                jv: vec![0i32; nh * hd],
            }),
        }
    }

    /// Issue #112: the calibration of a segment whose per-head exponents are
    /// `frac_per_head` -- the per-layer calibration `zk-prover model-roots`
    /// publishes, and the form `extract.py` writes into a dump's manifest.
    ///
    /// It is `Layer::load_weights`' own two rules and no others: a key tile's
    /// instances take the exponents of the heads it covers, cycled, and the
    /// master exponent is the largest over the whole layer -- one master for
    /// every key tile, which is what makes one exponent table serve them all.
    pub fn calibration_of(p: &crate::compose::Params, frac_per_head: &[f64])
                          -> crate::shaperoot::Calibration {
        assert!(frac_per_head.len() >= p.nh,
                "the calibration names {} heads and this statement has {}",
                frac_per_head.len(), p.nh);
        let master = frac_per_head[..p.nh].iter().copied().fold(f64::MIN, f64::max);
        let plan = crate::compose::Plan::new(p);
        crate::shaperoot::Calibration(
            plan.shapes.iter().enumerate().map(|(b, sh)| {
                let per_inst: Vec<f64> = (0..sh.insts)
                    .map(|i| frac_per_head[(plan.head_offsets[b] + i) % p.nh])
                    .collect();
                crate::shaperoot::KeyTileCalibration {
                    frac: master.to_bits(),
                    frac_per_inst: per_inst.iter().map(|f| f.to_bits()).collect(),
                    // `Attn::weights_only`'s own expression, which is
                    // `Attn::build`'s; `from_params` asserts the key tile it
                    // builds against it, so the two cannot drift apart
                    delta: per_inst.iter()
                        .map(|f| 1i64 << ((master - f).round().max(0.0) as u32).min(30))
                        .collect(),
                }
            }).collect())
    }

    /// Stream NN: derive the key tiles' Q, K and V from this layer's own `q`, `k`
    /// and `v` matmuls, and rebuild the key tiles from them.
    ///
    /// Returns the `RopeCols` a chunk proving the rotation needs, or `None` when
    /// the shapes do not admit the statement at all -- in which case nothing is
    /// rewired and the run reports the family as absent.
    ///
    /// `bend` moves one *key tile* value after the derivation and rebuilds the
    /// key tiles around it: the attention it proves is then honest, the matmuls are
    /// untouched, and the only thing in the chunk that can see it is the key tile's
    /// own claim against the quantize component's output.  `(which of q/k/v, cell)`.
    ///
    /// `pinned` is `rebuild_key_tiles`'s: the master exp table keeps the width
    /// the statement carries, rather than being sized again from the rebuilt
    /// data.
    pub fn rope_from_mats(&mut self, round: bf16::Round, bend: Option<(usize, usize)>,
                          pinned: bool) -> Option<RopeCols> {
        use crate::gadgets::residual_stream::{Prog, UnKind, reference};
        let (nh, hd, tokens) = (self.ao.nh, self.ao.hd, self.ao.s);
        // stream NN-3: a tiled prefix (`--kv-boundary`) is rebuilt like any other
        // now, so `nsub` is no longer a reason to leave the family off
        // stream NN-3: a shape that leaves the family off says which test it
        // failed.  Silence here cost a session an afternoon of guessing.
        let off = |why: &str| -> Option<RopeCols> {
            eprintln!("layer: no rotation -- {why}");
            None
        };
        if hd % 2 != 0 {
            return off("hd is odd");
        }
        if !self.key_tiles.iter().any(|b| b.diagonal_block) {
            return off("no diagonal block key tile");
        }
        let width = |n: &str| self.mats.iter().find(|m| m.name == n).map(|m| m.n_out);
        let (nq, nk, nv) = match (width("q"), width("k"), width("v")) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return off("this layer has no q/k/v matmul"),
        };
        if nq != nh * hd || nk != nv || nk % hd != 0 || nh % (nk / hd) != 0 {
            return off(&format!("widths do not line up: nq={nq} nk={nk} nv={nv} nh={nh} hd={hd}"));
        }
        let nkv = nk / hd;
        let rep = nh / nkv;
        // NN-2 used to ask a second question here: whether the components could
        // *tile* the shape and not only fit its widths.  A rotation component's
        // rows are `tokens * heads` and a quantize component's are `q_layout`'s,
        // and both had to be an exact power of two, which refused every head
        // count with an odd factor at every block length (#73).  Since the
        // decision of 2026-09-08 both round their row count up and carry the
        // slack as inert lead rows, so any shape whose widths line up tiles and
        // the widths above are the whole test.
        let b0 = &self.key_tiles[0];
        let (q_bits, k_bits) = (b0.q_bits, b0.k_bits);
        let (qmax_q, qmax_k) = ((1i64 << (q_bits - 1)) - 1, (1i64 << (k_bits - 1)) - 1);
        // the model's own angles.  A synthetic layer has no model, so it takes
        // Llama's default `inv_freq` at theta = 10000 and this block's own
        // offset; a real dump carries both and never reaches here.
        // a synthetic layer has no model, so it takes Llama's default branch at
        // theta = 10000 and this block's own offset; a real dump carries both
        let (theta, kind) = (10000.0f64, "default".to_string());
        let inv_freq = crate::gadgets::rope::inv_freq_of(hd, theta, &kind, &[]);
        let pos0 = self.block.unwrap_or(0) * tokens;
        let sp = |m: &Mat| crate::gadgets::dequant::Spec {
            variant: m.variant, round, kmax: 0, per_row: 1, limb_bits: (16, 14),
            q_bits: 0, rem_bits: 0, fold_alpha: 0,
        };
        let deq = |me: &Self, name: &str, n: usize| -> Vec<u16> {
            let mi = me.mats.iter().position(|m| m.name == name).unwrap();
            let spec = sp(&me.mats[mi]);
            let mut v = vec![0u16; tokens * n];
            v.par_chunks_mut(n).enumerate().for_each(|(t, row)| {
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = deq_ref(&me.mats[mi], &spec, t, c);
                }
            });
            v
        };
        // Issue #172: a q/k/v projection bias, as a witness-construction switch.
        // B1 adds the model's bias to the *dequantized* matmul output, before the
        // rotation and before the quantize, so the key tiles hold integers the
        // dequant component -- which states `acc s_x s_w` and nothing else --
        // cannot derive, and `nn_out` is the one equality that compares them.
        // Qwen2.5 gives q, k and v a bias; Llama gives none, which is why every
        // witness this repo had proved carried none.  At 0 -- the default, and
        // every recorded digest -- nothing here moves.
        let bias_frac = std::env::var("ZK_SYNTHETIC_QKV_BIAS").ok()
            .and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        let val = |p: u16| -> f64 {
            let (sm, ef) = crate::encoding::split(crate::encoding::canon(p));
            crate::encoding::to_f64(sm, ef)
        };
        let add_bias = |x: Vec<u16>, n: usize| -> Vec<u16> {
            if bias_frac == 0.0 {
                return x;
            }
            // one constant per output channel, at `bias_frac` of that channel's own
            // largest magnitude, signs alternating -- the size a real bias has
            // against the values beside it, which is what decides how many
            // derived integers it moves
            let b: Vec<f64> = (0..n)
                .map(|c| {
                    let m = (0..tokens).map(|t| val(x[t * n + c]).abs()).fold(0.0f64, f64::max);
                    if c % 2 == 0 { bias_frac * m } else { -bias_frac * m }
                })
                .collect();
            let mut out = x;
            out.par_chunks_mut(n).for_each(|row| {
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = crate::encoding::canon(bf16::from_f64(val(*slot) + b[c]));
                }
            });
            out
        };
        if bias_frac != 0.0 {
            eprintln!("layer: ZK_SYNTHETIC_QKV_BIAS={bias_frac} -- the key tiles carry a q/k/v \
projection bias the statement has no site for (a rejection is the pass)");
        }

        // the rotation, through the components' own programs: one row per
        // (token, head), `hd` leaves, `cos`/`sgnsin` preprocessed
        let rotate = |x: &[u16], heads: usize| -> Vec<u16> {
            let rows = tokens * heads;
            let mut pre = vec![0u16; rows * 2 * hd];
            pre.par_chunks_mut(2 * hd).enumerate().for_each(|(j, row)| {
                row.copy_from_slice(&crate::gadgets::rope::pre_row(&inv_freq, hd, pos0 + j / heads));
            });
            let prod = reference(&[x.to_vec()], &pre, rows, &Prog::rope_mul(hd), round);
            let out = reference(&[prod[0].clone()], &[], rows, &Prog::rope_add(hd), round);
            out[0].clone()
        };
        let qrot = rotate(&add_bias(deq(self, "q", nq), nq), nh);
        let krot = rotate(&add_bias(deq(self, "k", nk), nk), nkv);
        // Stream SS: a folded `v` has no bf16 stream to quantize -- the fold's
        // one rounding *is* the quantize -- so the component below is not built for
        // it and `vv` comes off `fold_out` directly.  `deq` would panic here
        // (`deq_ref` answers for `pow2` and `chain` only), which is why a
        // `--layout zfold` shape could not carry the wiring family at all.
        let vmi = self.mats.iter().position(|m| m.name == "v").unwrap();
        let v_fold = self.mats[vmi].variant == DeqVariant::Fold;
        let vdeq = if v_fold { vec![] } else { add_bias(deq(self, "v", nv), nv) };
        // the scales.  B1 calibrates them offline; a synthetic layer picks the
        // smallest power of two that leaves every value inside the domain, so
        // the desk's witness is complete (`quantize`'s clamp has no honest
        // witness in this component -- the table has no row past 127).
        let scale = |x: &[u16], n: usize, qmax: i64| -> Vec<i32> {
            (0..n)
                .map(|c| {
                    let col: Vec<u16> = (0..tokens).map(|t| x[t * n + c]).collect();
                    crate::gadgets::rope::pow2_scale(&col, qmax) as i32
                })
                .collect()
        };
        // Stream NN-3: `ZK_QUANTIZE_SCALE_SHIFT=n` calibrates every scale `n` exponents
        // *below* the smallest one that fits, so a synthetic layer clamps where
        // B1 does and the desk can prove a chunk that carries the clamp.  A
        // witness-construction switch like `--synth-residual-stream`: at 0 -- the default,
        // and every recorded digest -- nothing here moves.
        let shift = std::env::var("ZK_QUANTIZE_SCALE_SHIFT").ok()
            .and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
        if shift != 0 {
            eprintln!("layer: ZK_QUANTIZE_SCALE_SHIFT={shift} -- the q/k/v scales are {shift} \
exponent(s) small, so B1's clamp fires");
        }
        let down = |v: Vec<i32>| -> Vec<i32> { v.into_iter().map(|x| x - shift as i32).collect() };
        let jq = down(scale(&qrot, nq, qmax_q));
        let jk_kv = down(scale(&krot, nk, qmax_k));
        // stream SS: the fold's `s_v` is the witness's own `jv2` and is not
        // calibrated off a bf16 column, so `ZK_QUANTIZE_SCALE_SHIFT` does not move it --
        // the clamp a folded site takes is the fold's own `qc = min(qm, 127)`.
        let jv_kv = if v_fold {
            self.mats[vmi].jv2.clone()
        } else {
            down(scale(&vdeq, nv, crate::gadgets::rope::qmax_of(crate::gadgets::rope::V_BITS)))
        };
        // and on the query-head grid, which is where B1 applies them and where
        // the statement carries them
        let spread = |src: &[i32]| -> Vec<i32> {
            (0..nh * hd).map(|i| src[(i / hd / rep) * hd + i % hd]).collect()
        };
        let (jk, jv) = (spread(&jk_kv), spread(&jv_kv));
        // Stream NN-3: `s_v` is one scale in B1 and is used at two sites -- the
        // `v` quantize divides by it and the attention output rescales by it
        // (`AttnOut::e` is `jv - jo`).  A real dump reads both off one
        // `jv.i32`; a synthetic layer drew the attention output's at random, so
        // the two sites disagreed and no desk shape could state the binding.  The
        // attention output is rebuilt below, so this is the value it takes.
        self.ao.jv = jv.clone();
        // the key tiles' own columns, on the query-head grid: the kv head's values
        // replicated across its `rep` query heads, exactly as
        // `attention`'s `repeat_interleave` does
        let quant = |x: &[u16], n: usize, jj: &[i32], qmax: i64| -> Vec<i32> {
            let mut out = vec![0i32; tokens * nh * hd];
            out.par_chunks_mut(nh * hd).enumerate().for_each(|(t, row)| {
                for (i, slot) in row.iter_mut().enumerate() {
                    let (h, ch) = (i / hd, i % hd);
                    let c = (h / (nh / (n / hd))) * hd + ch;
                    *slot = crate::gadgets::rope::quant_ref(x[t * n + c], jj[c] as i64, qmax) as i32;
                }
            });
            out
        };
        let mut qq = quant(&qrot, nq, &jq, qmax_q);
        let mut kk = quant(&krot, nk, &jk_kv, qmax_k);
        let mut vv = if v_fold {
            let m = &self.mats[vmi];
            let mut out = vec![0i32; tokens * nh * hd];
            out.par_chunks_mut(nh * hd).enumerate().for_each(|(t, row)| {
                for (i, slot) in row.iter_mut().enumerate() {
                    let (h, ch) = (i / hd, i % hd);
                    *slot = fold_out(m, t, (h / rep) * hd + ch) as i32;
                }
            });
            out
        } else {
            quant(&vdeq, nv, &jv_kv, 127)
        };
        // stream NN (NN-2): `which == 3` is the *prefix* key tile's own copy of Q --
        // the grid above is untouched, so the diagonal block keeps the honest queries
        // and only the prefix's copy moves.  `rebuild_key_tiles` applies it inside
        // both of its passes, so the group's joint row maximum, its master
        // exponent and the prefix's whole attention follow the bent query and
        // the chunk is an honest attention over it.
        let mut pre_q_bend = None;
        if let Some((which, at)) = bend {
            if which == 3 {
                pre_q_bend = Some(at);
            } else {
                let col: &mut Vec<i32> =
                    match which % 3 { 0 => &mut qq, 1 => &mut kk, _ => &mut vv };
                let n = col.len();
                let i = at % n;
                let lim = match which % 3 { 0 => qmax_q, 1 => qmax_k, _ => 127 } as i32;
                col[i] += if col[i] < lim { 1 } else { -1 };
                eprintln!("layer: key tile {} cell {i} bent (a rejection is the pass)",
                          ["Q", "K", "V"][which % 3]);
            }
        }
        self.rebuild_key_tiles(&qq, &kk, &vv, pre_q_bend, pinned);
        Some(RopeCols { nkv, pos0, theta, rope_type: kind, rope_scale: vec![],
                        inv_ref: Some(inv_freq), jq, jk, jv })
    }

    /// Stream NN: rebuild every key tile from Q, K and V on the query-head grid.
    ///
    /// The diagonal block's keys *are* this block's tokens, so its K and V are what the
    /// layer's own `k` and `v` matmuls produced; a prefix key tile's keys are the
    /// cache and keep the ones they had, because this chunk did not produce
    /// them (they are bound across blocks by root equality instead -- stream
    /// AA/EE).  Every key tile of a group takes the new queries, and the group's two
    /// halves share one row maximum and one master exponent, exactly as
    /// `Attn::synthetic_pair` builds them.
    ///
    /// Stream NN-3: a tiled prefix (`--kv-boundary`, `nsub > 1`) is rebuilt too.
    /// Its sub-key-tiles are `nsub` windows of one pair's prefix, so each keeps its
    /// own slice of the cache and takes the new queries, the group's master
    /// exponent is the joint maximum over every sub-key-tile **and** the diagonal block, and
    /// the pair's total denominator -- which the group's per-token component
    /// commits on its first sub-key-tile -- is the sum of all of their `zz`.  Before
    /// this the function asserted `nsub <= 1` and the rotation simply turned
    /// itself off under `--kv-boundary`.
    ///
    /// `pre_q_bend` is stream NN-2's control: one cell of the **first group's
    /// prefix key tile's own** copy of Q, moved before that key tile's attention is
    /// built, so the diagonal block keeps the honest queries and the chunk is still an
    /// honest attention over the cache -- and the only thing that can see it is
    /// `nn_q_ok`, the equality between the two halves' Q folds.
    ///
    /// `pinned` says the statement this segment is proved against is already
    /// written, which is what every control's rebuild is: the master exp table
    /// keeps the width that statement carries, and a key past that width stays
    /// off the table where the lookup refuses it (issue #134).  An honest
    /// construction runs before there is a statement and sizes the table from
    /// its own data.
    pub fn rebuild_key_tiles(&mut self, qq: &[i32], kk: &[i32], vv: &[i32],
                         pre_q_bend: Option<usize>, pinned: bool) {
        let (nh, hd, s) = (self.ao.nh, self.ao.hd, self.ao.s);
        let nsub = self.nsub.max(1);
        let n_pre = self.key_tiles.iter().filter(|b| !b.diagonal_block).count();
        let groups = self.key_tiles.len() - n_pre;
        assert!(groups > 0, "the q/k/v rewiring needs the diagonal block");
        assert!(n_pre == 0 || n_pre == groups * nsub,
                "{nsub} prefix sub-key_tiles per group, or none: {n_pre} prefix key_tiles, \
{groups} groups");
        // the key tiles of one group, prefix sub-key-tiles first and the diagonal block last --
        // the order `Layer::synthetic` lays them out in
        let gidx = |g: usize| -> Vec<usize> {
            let pre = if n_pre == 0 { 0 } else { nsub };
            (0..pre).map(|k| g * nsub + k).chain(std::iter::once(n_pre + g)).collect()
        };
        let take = |src: &[i32], off: usize, insts: usize, rows: usize| -> Vec<i32> {
            (0..insts * rows * hd)
                .map(|i| {
                    let (inst, rest) = (i / (rows * hd), i % (rows * hd));
                    let (t, ch) = (rest / hd, rest % hd);
                    src[t * nh * hd + (off + inst) * hd + ch]
                })
                .collect()
        };
        // stream NN (NN-2): the prefix half of group 0 takes its Q through here,
        // so both passes see the same bent cell and the key tile's own attention is
        // rebuilt around it.
        if let Some(at) = pre_q_bend {
            if n_pre == 0 {
                eprintln!("layer: prefix-key_tile Q bend asked for on a chunk with no prefix \
key_tile -- nothing bent, and this control measures nothing");
            } else {
                eprintln!("layer: prefix key_tile (group 0) Q cell {at} bent, the diagonal block's \
unmoved (a rejection is the pass)");
            }
        }
        let q_bits0 = self.key_tiles[0].q_bits;
        let take_q = |src: &[i32], off: usize, insts: usize, rows: usize,
                      bend_here: bool| -> Vec<i32> {
            let mut q = take(src, off, insts, rows);
            if bend_here {
                if let Some(at) = pre_q_bend {
                    let n = q.len();
                    let i = at % n;
                    let lim = ((1i64 << (q_bits0 - 1)) - 1) as i32;
                    q[i] += if q[i] < lim { 1 } else { -1 };
                }
            }
            q
        };
        // pass 1: learn the master exponent and the table height the new data
        // needs, over both halves of every group -- the two-pass shape
        // `Layer::synthetic` uses, for the same reason
        let mut probe: Vec<Vec<Attn>> = vec![];
        for g in 0..groups {
            let mut row = vec![];
            for (n, b) in gidx(g).into_iter().enumerate() {
                let a = &self.key_tiles[b];
                let sh = a.shape;
                let off = self.head_offsets[b];
                let q = take_q(qq, off, sh.insts, sh.s, !a.diagonal_block && g == 0 && n == 0);
                let (k, v) = if a.diagonal_block {
                    (take(kk, off, sh.insts, sh.c), take(vv, off, sh.insts, sh.c))
                } else {
                    (a.kk.clone(), a.vv.clone())
                };
                row.push(Attn::from_qkv(sh, a.variant, q, k, v, a.q_bits, a.k_bits, a.nmax,
                                        a.log_table, None, None, a.source.clone(), a.diagonal_block,
                                        None));
            }
            probe.push(row);
        }
        let frac = probe.iter().flatten().map(|b| b.frac).fold(f64::MIN, f64::max).ceil();
        let d_bits = probe.iter().flatten().map(|b| b.d_bits()).max().unwrap() + 1;
        // Issue #72: the rotation rewrites q, k and v, so the master exp table is
        // sized again from the new data -- and what bounds that size is
        // `--table-bits N` where the line gave one.  The key tiles carry it,
        // `Layer::synthetic` having built them under it; without one the bound is
        // the 2^22 rows `Attn::build` caps its own default at, past which the table
        // dominates a single-key-tile benchmark.  At the 8B model's attention width
        // the data needs 24 bits of key, and clamping to 2^22 here left them off
        // the table, where only a rejected proof reported them.
        //
        // Issue #134: under `pinned` the table is not this rebuild's to size.  A
        // control rebuilds a segment whose statement is already written -- and that
        // statement says how many rows the master table has -- so the width stays
        // where the statement put it, whatever the rebuilt data would rather have.
        // Widening it would prove another statement; narrowing it would prove a
        // smaller one.
        let stated = self.key_tiles.iter().map(|b| b.log_table).max().unwrap_or(0);
        let ceiling = if pinned { stated } else { stated.max(22) };
        let mut log_table = if pinned { stated } else { d_bits.clamp(4, ceiling) };
        // pass 2: the joint row maximum per group -- over every sub-key-tile and the
        // diagonal block -- then all of them under it.  Issue #72: `d_bits` is the
        // shift before that joint maximum widens it, so it can fall short of the
        // keys the rebuilt key tiles look up; where it does, they are counted and
        // pass 2 is taken again at the size they need.
        loop {
            for g in 0..groups {
                let idx = gidx(g);
                let mm: Vec<i64> = probe[g].iter().skip(1)
                    .fold(probe[g][0].mm.clone(), |acc, b| {
                        acc.iter().zip(&b.mm).map(|(x, y)| *x.max(y)).collect()
                    });
                for (n, b) in idx.iter().copied().enumerate() {
                    let a = &self.key_tiles[b];
                    let sh = a.shape;
                    let off = self.head_offsets[b];
                    let q = take_q(qq, off, sh.insts, sh.s, !a.diagonal_block && g == 0 && n == 0);
                    let (k, v) = if a.diagonal_block {
                        (take(kk, off, sh.insts, sh.c), take(vv, off, sh.insts, sh.c))
                    } else {
                        (a.kk.clone(), a.vv.clone())
                    };
                    let (variant, qb, kb, nmax, src) =
                        (a.variant, a.q_bits, a.k_bits, a.nmax, a.source.clone());
                    let diagonal_block = a.diagonal_block;
                    self.key_tiles[b] =
                        Attn::from_qkv(sh, variant, q, k, v, qb, kb, nmax, log_table,
                                       Some(vec![frac; sh.insts]), Some(frac), src,
                                       diagonal_block, Some(mm.clone()));
                }
                // the group's denominator: the pair's total, which every sub-key-tile's
                // `zz` is a partial sum of.  It lives on the key tile that carries the
                // per-token component -- the first sub-key-tile when there is a prefix, and
                // the diagonal block itself when there is none.
                let total: Vec<i64> = idx.iter().skip(1)
                    .fold(self.key_tiles[idx[0]].zz.clone(), |acc, b| {
                        acc.iter().zip(&self.key_tiles[*b].zz).map(|(x, y)| x + y).collect()
                    });
                self.key_tiles[idx[0]].zz_total = Some(total);
            }
            let need = self.key_bits();
            if need <= log_table {
                break;
            }
            if need > ceiling {
                // Issue #72: an honest construction sizes its own table, so a
                // rebuild that outgrows the width the line gave is the line's
                // mistake, and it is said here rather than left to a rejected
                // proof to report.
                assert!(pinned,
                        "the rotated key tiles look up an exp table key of {need} bits and \
--table-bits gives at most {ceiling}: pass --table-bits {need}");
                // Issue #134: a control's table is the statement's, so a key past
                // it is a refusal and not a mistake.  The keys that do not fit stay
                // off the table, where no lookup argument can absorb them, and the
                // proof is rejected -- which is what a control that moves the data
                // is for.  The count is the witness's own, recounted by the rebuild
                // above.
                let off: usize = self.key_tiles.iter().map(|b| b.off_table).sum();
                eprintln!("layer: this witness is refused -- the rebuilt key tiles look up an \
exp table key of {need} bits, the statement's table takes at most {log_table}, and {off} keys are \
off it");
                break;
            }
            log_table = need;
        }
        // the attention output follows the key tiles: its accumulator and
        // denominator *are* their sums, and the per-token output scale is the
        // smallest power of two that leaves the quotient inside int8
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for bi in 0..self.key_tiles.len() {
            let b = &self.key_tiles[bi];
            for i in 0..self.key_tile_sizes[bi] {
                let h = self.head_offsets[bi] + i;
                for t in 0..s {
                    den[h * s + t] += b.zz[i * s + t];
                    for j in 0..hd {
                        acc[(h * s + t) * hd + j] += b.out[(i * s + t) * hd + j];
                    }
                }
            }
        }
        for d in den.iter_mut() {
            *d = (*d).max(1);
        }
        let jv = self.ao.jv.clone();
        let mut jo = vec![0i32; s];
        for (t, slot) in jo.iter_mut().enumerate() {
            let mut m = 0f64;
            for h in 0..nh {
                for j in 0..hd {
                    let q = acc[(h * s + t) * hd + j].unsigned_abs() as f64
                        * (jv[h * hd + j] as f64).exp2() / den[h * s + t] as f64;
                    m = m.max(q);
                }
            }
            *slot = (m / 127.0).log2().ceil().max(-30.0) as i32;
        }
        self.ao.acc = acc;
        self.ao.den = den;
        self.ao.jo = jo;
    }

    /// Stream NN: one side's quantize exponents, from the query-head grid onto
    /// the producer's own.
    ///
    /// B1 quantizes q, k and v *after* `repeat_interleave`, so `s_k` and `s_v`
    /// are per query head even though the values behind them are one kv head's.
    /// A gadget over the kv grid is only the same statement when the `rep`
    /// query heads of a group carry the same exponent -- `extract.py` asserts
    /// exactly that for `s_v` and this asserts it for all three, rather than
    /// picking the group's first head quietly.
    pub fn qkv_j(&self, name: &str, heads: usize, hd: usize, rep: usize) -> Vec<i64> {
        let r = self.rope.as_ref().expect("the quantize components need the rope columns");
        let src = match name {
            "q" => &r.jq,
            "k" => &r.jk,
            // Stream NN-3: `s_v` is one scale used at two sites, and this is
            // where the two read **one array**.  The attention output rescales
            // by `jv - jo` off `AttnOut::jv`; the `v` quantize divides by the
            // same exponent, so it takes it from there rather than from the
            // rotation's own copy of `jv.i32`.  A dump loads both from that one
            // file (`Layer::load` and `RopeCols::load`), and `family_cells.sv_binding`
            // reports the two against each other on every run.
            "v" => &self.ao.jv,
            other => panic!("no quantize exponents for {other}"),
        };
        let mut out = vec![0i64; heads * hd];
        for kv in 0..heads {
            for ch in 0..hd {
                let j = src[(kv * rep) * hd + ch];
                for x in 1..rep {
                    assert_eq!(src[(kv * rep + x) * hd + ch], j,
                               "{name}: the scale disagrees inside GQA group {kv} at channel \
                                {ch}; a gadget over the kv grid cannot state this model");
                }
                out[kv * hd + ch] = j as i64;
            }
        }
        out
    }

    /// Stream Z: the fold's two data-derived pins, `None` when no matmul is a
    /// fold.  `alpha` splits the divisor `2^G = 2^(G_t + G_c)` between the
    /// committed per-token `dent = 2^(G_t - alpha)` and the preprocessed
    /// per-channel `pj = 2^(G_c - (14 - alpha))`; both halves must be integers
    /// on every token and channel, which is exactly `alpha <= min_t G_t` and
    /// `14 - alpha <= min_c G_c` -- and those two together put every slot's
    /// `G` at 14 or more, which the limb identity needs.  `rem_bits` is the
    /// width of the high remainder's range check, `bits(2^(max G - 13))`.
    pub fn fold_pins(&self) -> Option<(u32, u32)> {
        let mut alpha_hi = i64::MAX;   // min_t G_t
        let mut beta_hi = i64::MAX;    // min_c G_c
        let mut g_max = i64::MIN;
        let mut any = false;
        for m in &self.mats {
            if m.variant != DeqVariant::Fold {
                continue;
            }
            any = true;
            let gt: Vec<i64> = (0..m.tokens).map(|t| m.fold_gt(t)).collect();
            let gc: Vec<i64> = (0..m.n_out).map(|c| m.fold_gc(c)).collect();
            alpha_hi = alpha_hi.min(gt.iter().copied().min().unwrap());
            beta_hi = beta_hi.min(gc.iter().copied().min().unwrap());
            g_max = g_max.max(gt.iter().copied().max().unwrap()
                              + gc.iter().copied().max().unwrap());
        }
        if !any {
            return None;
        }
        assert!(alpha_hi + beta_hi >= 14,
                "fold: min G_t + min G_c = {} < 14 -- the divisor cannot split",
                alpha_hi + beta_hi);
        let alpha = (14 - beta_hi).max(0).min(alpha_hi);
        let rem = (g_max - 13).max(1) as u32;
        Some((alpha as u32, rem))
    }

    /// A small, self-consistent synthetic layer, for the container selftest.
    pub fn synthetic(s: usize, c: usize, hd: usize, key_tile_sizes: &[usize], variant: Variant,
                     cfg: &Src, n_in: usize, n_out: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        // Stream AA (E11): a tiled synthetic layer is the *same* pair, with its
        // prefix cut into `C / S` windows by `Attn::key_slice`, so the windows'
        // denominators really do sum to the pair's total.  What a synthetic
        // layer still cannot do is two *adjacent chunks* over one shared key
        // stream, which is what a chain probe needs; that generator is the
        // follow-on this stream's record names, and until it exists a chain is
        // proved on a real dump.
        assert!(!cfg.tile || cfg.diagonal_block, "`--kv-boundary` needs `--diagonal-block`");
        assert!(!cfg.tile || c % s == 0,
                "`--kv-boundary` tiles the prefix in blocks of S: C={c} is not a multiple of S={s}");
        let s_nsub = if cfg.tile { c / s } else { 1 };
        let nh: usize = key_tile_sizes.iter().sum();
        // pass 1: learn the data-derived exponents; pass 2: rebuild every key tile
        // against one shared master exponent and one shared table size
        let mut probe = vec![];
        let mut head_offsets = vec![];
        let mut off = 0usize;
        // B10 step 3: with the diagonal block, a key tile is a *pair* -- the prefix
        // rectangle and the diagonal block over the same heads, sharing the queries and one
        // row maximum.  Both halves' shifts go into the same master exponent, so
        // both are probed.
        let mut probe_diagonal_block = vec![];
        for insts in key_tile_sizes {
            let sh = Shape { s, c, hd, insts: *insts };
            if cfg.diagonal_block {
                let (p, cc) = Attn::synthetic_pair(sh, variant, cfg.q_bits, cfg.k_bits,
                                                   cfg.nmax, cfg.log_table,
                                                   seed ^ (off as u64 + 1), None);
                probe.push(p);
                probe_diagonal_block.push(cc);
            } else {
                probe.push(Attn::synthetic(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax,
                                           cfg.log_table, seed ^ (off as u64 + 1)));
            }
            // stream AA: one entry per *key tile*, and a tiled prefix is `s_nsub`
            // key tiles over the same heads
            for _ in 0..s_nsub {
                head_offsets.push(off);
            }
            off += insts;
        }
        if cfg.diagonal_block {
            let mut off = 0usize;
            for insts in key_tile_sizes {
                head_offsets.push(off);
                off += insts;
            }
        }
        probe.extend(probe_diagonal_block);
        let frac = probe.iter().map(|b| b.frac).fold(f64::MIN, f64::max).ceil();
        let d_bits = probe.iter().map(|b| b.d_bits()).max().unwrap() + 1;
        let log_table = if cfg.log_table > 0 { cfg.log_table } else { d_bits.clamp(4, 22) };
        let mut key_tiles = vec![];
        let mut diagonal_block_key_tiles = vec![];
        let mut sizes = vec![];
        let mut off = 0usize;
        for insts in key_tile_sizes.iter() {
            let sh = Shape { s, c, hd, insts: *insts };
            if cfg.diagonal_block {
                // the same seeds the probe used, so the master exponent it derived
                // is an upper bound on the data these key tiles actually carry
                let (p, cc) = Attn::synthetic_pair(sh, variant, cfg.q_bits, cfg.k_bits,
                                                   cfg.nmax, log_table,
                                                   seed ^ (off as u64 + 1), Some(frac));
                // Stream AA (E11): the tiled prefix, as `nsub` windows of the
                // pair's own prefix rather than as `nsub` unrelated key tiles -- so
                // the sub-key-tiles' `zz` really do sum to the pair's total and the
                // synthetic layer is a layer and not a shape rehearsal.
                if cfg.tile {
                    let total = p.zz_total.clone();
                    for k in 0..s_nsub {
                        key_tiles.push(p.key_slice(k * s, s));
                    }
                    // the group's per-token component lives on its first sub-key-tile,
                    // and the denominator it commits is the pair's total -- the
                    // slices' own `zz` are the partial sums that make it up
                    let first = key_tiles.len() - s_nsub;
                    key_tiles[first].zz_total = total;
                } else {
                    key_tiles.push(p);
                }
                diagonal_block_key_tiles.push(cc);
            } else {
                key_tiles.push(Attn::synthetic_shared(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax,
                                                  log_table, seed ^ (off as u64 + 1),
                                                  frac, false));
            }
            for _ in 0..s_nsub {
                sizes.push(*insts);
            }
            off += insts;
        }
        if cfg.diagonal_block {
            sizes.extend(key_tile_sizes.iter().copied());
            key_tiles.extend(diagonal_block_key_tiles);
        }
        let key_tile_sizes_all = sizes;
        let mut mats = vec![];
        // matmuls in one Freivalds group share one committed input, as the real layer's do
        let mut group_qx: std::collections::HashMap<&str, Vec<i32>> = Default::default();
        // Issue #192: `ZK_SYNTHETIC_V_SCALE_BINADES=n` is the width, in binades, of
        // the window the `v` matmul's per-channel weight-scale exponent is drawn from
        // where `v` is on the chain gadget.  The default is 3 -- the narrow draw the
        // fold arm below already takes, for the reason it gives there -- so a driver
        // that states nothing gets the narrow one, and a driver that wants the old
        // 20-binade window states it and the run records what it was built at.
        let v_binades: u32 = std::env::var("ZK_SYNTHETIC_V_SCALE_BINADES")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        assert!(v_binades > 0,
                "ZK_SYNTHETIC_V_SCALE_BINADES is a window width and must be at least 1");
        for (name, var) in mat_variants(&cfg.layout) {
            let (ni, no) = if name == "down" { (n_out, n_in) } else { (n_in, n_out) };
            // B7b (README 26): `ZK_SYNTHETIC_DIMS=q:3584x3584,k:3584x512,...` sets a
            // matmul's (n_in, n_out) by name; unnamed matmuls keep the defaults.
            let (ni, no) = std::env::var("ZK_SYNTHETIC_DIMS")
                .ok()
                .and_then(|spec| {
                    spec.split(',').find_map(|e| {
                        let (n, d) = e.split_once(':')?;
                        if n != name { return None; }
                        let (a, b) = d.split_once('x')?;
                        Some((a.parse().ok()?, b.parse().ok()?))
                    })
                })
                .unwrap_or((ni, no));
            let gname = FREIVALDS_GROUPS.iter().find(|(_, ms)| ms.contains(&name)).unwrap().0;
            let qx: Vec<i32> = group_qx
                .entry(gname)
                .or_insert_with(|| (0..s * ni).map(|_| rng.random_range(-127..=127)).collect())
                .clone();
            let qw: Vec<i32> = (0..ni * no).map(|_| rng.random_range(-127..=127)).collect();
            let mut acc = vec![0i64; s * no];
            // B7b (README 26): one token per task, the output channel innermost
            // -- the same sums, in the order the memory is laid out.
            {
                use rayon::prelude::*;
                let (qx, qw) = (&qx, &qw);
                acc.par_chunks_mut(no).enumerate().for_each(|(t, row)| {
                    for k in 0..ni {
                        let x = qx[t * ni + k] as i64;
                        let w = &qw[k * no..(k + 1) * no];
                        for i in 0..no {
                            row[i] += x * w[i] as i64;
                        }
                    }
                });
            }
            mats.push(Mat {
                name: name.into(), variant: var, n_in: ni, n_out: no, tokens: s,
                acc, qx, qw,
                jx: (0..s).map(|_| -(rng.random_range(14..20) as i32)).collect(),
                jw: (0..no).map(|_| -(rng.random_range(14..20) as i32)).collect(),
                sx: (0..s).map(|_| (rng.random_range(0..128), rng.random_range(100..120))).collect(),
                sw: (0..no).map(|_| (rng.random_range(0..128), rng.random_range(100..120))).collect(),
                jv2: vec![],
            });
            // stream Z: a fold matmul's scales must keep every quotient inside
            // its domain and the remainder width inside the field wall at any
            // attention-output width the shape composes with -- the divisor is
            // sized off the accumulators this layer actually drew.  `G_t` sits
            // in [16, 18] and the channel side makes up the rest.
            if var == DeqVariant::Fold {
                let m = mats.last_mut().unwrap();
                let amax = m.acc.iter().map(|a| a.unsigned_abs()).max().unwrap().max(1);
                let worst = (amax as i64) << 16;
                let g_min = (64 - worst.leading_zeros() as i64) - 7 + 1;
                m.sx = (0..s)
                    .map(|_| (rng.random_range(0..128), 118 - rng.random_range(0..3)))
                    .collect();
                // Stream SS: `s_v`'s exponent is the fold's own `jv2`, and the
                // attention output's `e` is `jv - jo` -- so the weight scale's
                // spread here comes straight back as the width of `ao_rem`, and
                // the 20-exponent draw above put it at 33 bits, past the field
                // wall `ao_q_bits + ao_rem_bits <= 30`.  A synthetic weight
                // scale is arbitrary; a real dump's spread is B1's and is what
                // `--ao-rem-bits` pins.  Nothing outside `--layout zfold` sees
                // this line.
                m.sw = (0..no)
                    .map(|_| (rng.random_range(0..128), 110 + rng.random_range(0..3)))
                    .collect();
                let gc = (g_min - 16).max(1);
                m.jv2 = m.sw.iter()
                    .map(|(_, ew)| (gc + rng.random_range(0..2) - 134 + *ew as i64) as i32)
                    .collect();
            }
            // Issue #192: the same narrowing, for the same reason, on the gadget
            // `--layout b10` puts `v` on.  The chain's per-channel scale is `sw[c]`,
            // `jv` is `pow2_scale` over this matmul's dequantize output and so carries
            // that spread, and the attention output's `e` is `jv - jo` -- so a
            // 20-binade draw comes straight back as the width of `ao_rem`: at the 70B
            // width it read 35 bits against the field wall `ao_q_bits + ao_rem_bits
            // <= 30`, and no flag may state a narrower attention output than the
            // witness needs.  A real checkpoint's per-channel V scales span two to
            // five binades; a synthetic weight scale is arbitrary, and what matters is
            // that it is stated.  Nothing with `v` off the chain reaches this arm, and
            // no recorded synthetic run puts it there.
            if name == "v" && var == DeqVariant::Chain {
                let m = mats.last_mut().unwrap();
                m.sw = (0..no)
                    .map(|_| (rng.random_range(0..128), 110 + rng.random_range(0..v_binades)))
                    .collect();
                eprintln!("synthetic layer: v is on the chain, so its weight scale is drawn \
from a {v_binades}-binade window (ZK_SYNTHETIC_V_SCALE_BINADES) -- that spread is the width of \
the attention output's remainder");
            }
        }
        // attnout from the key tiles, with scales that keep the quotient inside int8
        // B10 step 3: `+=`, not `=`.  With the diagonal block riding as a second set of
        // key tiles over the same heads, a head's denominator and A.V output are the
        // sum of its two halves -- which is what the attention output *is*.
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for (bi, b) in key_tiles.iter().enumerate() {
            for i in 0..key_tile_sizes_all[bi] {
                let h = head_offsets[bi] + i;
                for t in 0..s {
                    den[h * s + t] += b.zz[i * s + t];
                    for j in 0..hd {
                        acc[(h * s + t) * hd + j] += b.out[(i * s + t) * hd + j];
                    }
                }
            }
        }
        for d in den.iter_mut() {
            *d = (*d).max(1);
        }
        let jv: Vec<i32> = (0..nh * hd).map(|_| -(rng.random_range(6..9) as i32)).collect();
        let mut jo = vec![0i32; s];
        for t in 0..s {
            // the per-token output scale: smallest power of two with no clamp
            let mut m = 0f64;
            for h in 0..nh {
                for j in 0..hd {
                    let q = acc[(h * s + t) * hd + j].unsigned_abs() as f64
                        * (jv[h * hd + j] as f64).exp2() / den[h * s + t] as f64;
                    m = m.max(q);
                }
            }
            jo[t] = (m / 127.0).log2().ceil().max(-30.0) as i32;
        }
        Self { key_tiles, key_tile_sizes: key_tile_sizes_all, head_offsets, nsub: s_nsub, mats,
               ao: AttnOut { nh, s, hd, acc, den, jo, jv },
               source: format!("synthetic layer (seed={seed})"), residual_stream: None,
               head_keys: None, head_tok: None, attn_out_bits: 8,
               embed_tab: None, tok_in: None, layer: None, block: None, rope: None }
    }
}

impl Layer {
    /// Stream II: give a synthetic layer the wiring family's own columns, sized
    /// off the matmuls it already has -- `d` is the q/k/v group's input width
    /// and `dff` the gate matmul's output width, exactly as a real dump's are.
    ///
    /// And then make the *matmuls* consistent with them.  A synthetic layer's
    /// activations were seven unrelated random columns, which is fine while
    /// nothing bindings a matmul's input to anything; with the requantize seams in,
    /// an honest proof needs `qX` to actually be the requantize of the
    /// producer's output under the token's own scale, or the seam's own scale
    /// lookup goes off-table before any binding is reached.  So this runs the
    /// wiring family's own reference forward -- `residual_stream::reference` on the very
    /// programs the components prove, so the two cannot disagree -- and rebuilds
    /// each group's input, scale and accumulator from it.
    ///
    /// `o`'s input is deliberately left alone: `AttnOutEval` is that matmul's
    /// requantize and its claim is stream JJ's, so there is nothing here for a
    /// consistent witness to satisfy yet.
    pub fn synth_residual_stream(&mut self, seed: u64, a: usize, round: bf16::Round) {
        self.synth_family(seed, a, round, 0, None, None, None);
    }

    /// Stream NN, the key tile side's control.  The same layer with **one** key tile
    /// cell moved and the key tiles rebuilt around it: the attention is an honest
    /// attention over the bent K (or Q, or V), the matmuls are untouched, and
    /// the only thing that can see it is that key tile's claim against the
    /// quantize component's output.  `which` picks Q, K or V.
    pub fn synth_residual_stream_key_tile_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
                                           which: usize, idx: usize) {
        self.synth_family(seed, a, round, 0, None, Some((which, idx)), None);
    }

    /// Stream NN (NN-2), the prefix key tile's control.  The same layer with **one**
    /// cell of the first group's *prefix* key tile's own copy of Q moved, and that
    /// key tile's whole attention rebuilt around it -- the diagonal block's queries are
    /// untouched, so the chunk proves an honest attention over the cache with
    /// queries the `q` matmul did not produce, and the only thing that can see
    /// it is `nn_q_ok`.
    pub fn synth_residual_stream_prefix_q_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
                                               idx: usize) {
        self.synth_family(seed, a, round, 0, None, Some((3, idx)), None);
    }

    /// Stream II (E11), the `o` seam's control.  The same layer, with **one**
    /// `o` activation moved to another legal value and the whole layer rebuilt
    /// around it: the `o` matmul's accumulator follows, and so do `o_out`,
    /// `xmid`, norm2, the gate/up inputs, the SwiGLU and `down`.  So every
    /// constraint and every other equality in the chunk still holds -- this is
    /// an honest proof of a layer whose attention output is not the one the
    /// attnout component requantized, and only `ao_q_ok` can see it.
    ///
    /// Returns `None` when this shape's `o` Freivalds check is not `nh * hd` wide, which
    /// is the shape on which the seam is not statable at all.
    pub fn synth_residual_stream_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
                                      idx: usize) -> Option<usize> {
        let oi = self.mats.iter().position(|m| m.name == "o")?;
        if self.mats[oi].n_in != self.ao.nh * self.ao.hd {
            return None;
        }
        let n = self.mats[oi].qx.len();
        let at = idx % n;
        // which family this layer carries is the witness's to say, and a rebuild
        // must not change it: `--synth-head` already pushed the `lm_head` matmul
        // and set `head`, so pass a nonzero vocabulary to keep the head branch.
        // `synth_family` reads only the sign of it.
        let vocab = usize::from(self.residual_stream.as_ref().is_some_and(|g| g.head));
        self.synth_family(seed, a, round, vocab, Some(at), None, None);
        Some(at)
    }

    /// Stream II (E11): the *head* chunk's family on a synthetic layer -- an
    /// `lm_head` matmul of `vocab` columns whose committed input is the final
    /// norm's output, so the desk can run the head's norm scale and its
    /// requantize seam without a dump.  The layer's own seven matmuls keep
    /// their untied inputs, exactly as a head chunk's do.
    pub fn synth_head(&mut self, seed: u64, a: usize, round: bf16::Round, vocab: usize) {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        assert!(vocab > 0, "--synth-head needs a vocabulary");
        let tokens = self.ao.s;
        let d = self.mats.iter().find(|m| m.name == "q")
            .expect("--synth-head needs a q matmul").n_in;
        let variant = self.mats[0].variant;
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x1ead_1ead);
        let qw: Vec<i32> = (0..d * vocab).map(|_| rng.random_range(-127..=127)).collect();
        self.mats.push(Mat {
            name: "lm_head".into(), variant, n_in: d, n_out: vocab, tokens,
            acc: vec![0i64; tokens * vocab],
            qx: vec![0i32; tokens * d],
            qw,
            jx: (0..tokens).map(|_| -(rng.random_range(14..20) as i32)).collect(),
            jw: (0..vocab).map(|_| -(rng.random_range(14..20) as i32)).collect(),
            sx: (0..tokens).map(|_| (rng.random_range(0..128), rng.random_range(100..120)))
                .collect(),
            sw: (0..vocab).map(|_| (rng.random_range(0..128), rng.random_range(100..120)))
                .collect(),
            jv2: vec![],
        });
        self.synth_family(seed, a, round, vocab, None, None, None);
    }

    /// Issue #17: the model tree's own control -- a whole-layer swap.
    ///
    /// The model tree carries the matmul weights, their weight scales, the norm
    /// weights, `jv` and the embedding table.  This takes the half of it that
    /// hangs off the segment's **second** norm -- the `gate`, `up` and `down`
    /// weights and weight scales, and that norm's own weight -- from a second
    /// draw and rebuilds the whole layer around them, exactly as
    /// `synth_residual_stream_bent` rebuilds it around one activation.  What
    /// comes out is an honest proof of this segment under another layer's
    /// weights: every constraint holds, every equality holds, both boundaries
    /// hold, and the pinned model root is the only thing that can tell.  On a
    /// head chunk the same half is the final norm's weight and `lm_head`.
    ///
    /// The attention half is left alone on purpose.  `q`, `k`, `v`, `o` and the
    /// first norm's weight decide the key tiles and the attention output, and a
    /// statement pins that output's exponent windows from the honest data
    /// (`ao_jv_min`, `ao_jo_max`, `ao_rem_bits` and the rest) -- so a second
    /// draw there would leave those windows and be refused by the `EPOW`
    /// lookup before the root it is aimed at could see it, which is a control
    /// that no longer isolates the model tree.  A fold matmul keeps its own for
    /// the same kind of reason: stream Z draws its weight scale inside a window
    /// derived from the accumulators this layer drew.
    pub fn synth_model_swap(&mut self, seed: u64, a: usize, round: bf16::Round, swap: u64) {
        let head = self.residual_stream.as_ref().is_some_and(|g| g.head);
        let names: &[&str] = if head { &["lm_head"] } else { &["gate", "up", "down"] };
        let mut rng = SmallRng::seed_from_u64(swap);
        for m in self.mats.iter_mut() {
            if !names.contains(&m.name.as_str()) || m.variant == DeqVariant::Fold {
                continue;
            }
            for w in m.qw.iter_mut() {
                *w = rng.random_range(-127..=127);
            }
            for j in m.jw.iter_mut() {
                *j = -(rng.random_range(14..20) as i32);
            }
            for sc in m.sw.iter_mut() {
                *sc = (rng.random_range(0..128), rng.random_range(100..120));
            }
        }
        // which family this layer carries is the witness's to say, and a
        // rebuild must not change it -- `synth_family` reads only the sign
        let vocab = usize::from(head);
        self.synth_family(seed, a, round, vocab, None, None, Some(swap));
    }

    fn synth_family(&mut self, seed: u64, a: usize, round: bf16::Round, head_vocab: usize,
                    o_bend: Option<usize>, rope_bend: Option<(usize, usize)>,
                    model_swap: Option<u64>) {
        use crate::encoding::canon;
        use crate::gadgets::residual_stream::{Kind, Prog, Tiling, UnKind, reference};
        let head = head_vocab > 0;
        // Issue #134: which of the two rebuilds this is.  An honest construction
        // -- `synth_residual_stream`, `synth_head` -- runs before the statement is
        // derived and sizes the master exp table from its own data.  A rebuild that
        // carries a bend or a swap is a *control*, made to be proved against the
        // honest statement, so its table is that statement's and a key past it is a
        // refusal rather than a wider table.
        let pinned = o_bend.is_some() || rope_bend.is_some() || model_swap.is_some();
        let tokens = self.ao.s;
        let d = self.mats.iter().find(|m| m.name == "q")
            .expect("--synth-residual-stream needs a q matmul").n_in;
        let dff = self.mats.iter().find(|m| m.name == "gate")
            .expect("--synth-residual-stream needs a gate matmul").n_out;
        let gc = if head {
            ResidualStreamCols::synthetic_head(tokens, d, seed ^ 0x9e37_79b9)
        } else {
            ResidualStreamCols::synthetic(tokens, d, dff, seed ^ 0x9e37_79b9)
        };
        let (eps, inv_d) = (gc.eps, gc.inv_d);
        let un_r = UnKind::NormR { inv_d, eps };
        let t1 = Tiling::new(tokens, d, a);
        let tff = Tiling::new(tokens, dff, a);

        // `h = (x r) w` for one norm, through the same three programs the
        // components carry: the squares' two levels, then the norm scale.
        let norm_scale = |stream: &[u16], wcol: &[u16], t: &Tiling| -> Vec<u16> {
            let laid = t.lay(stream);
            let roots = reference(&[laid.clone()], &[], t.rows1(),
                                  &Prog::tree(t.nreal, t.b, UnKind::Square), round);
            let ms = reference(&[t.roots_to_l2(&roots[0])], &[], t.tokens,
                               &Prog::tree(t.a, t.a, UnKind::None), round);
            let msblk: Vec<u16> =
                (0..t.rows1()).map(|m| canon(ms[0][m % t.tokens])).collect();
            let mut wblk = vec![0u16; t.rows1() * t.nreal];
            for m in 0..t.rows1() {
                let res = m / t.tokens;
                for j in 0..t.nreal {
                    wblk[m * t.nreal + j] = canon(wcol[res + j * t.a]);
                }
            }
            let out = reference(&[laid, msblk], &wblk, t.rows1(),
                                &Prog::norm_scale(t.nreal, un_r), round);
            t.unlay(&out[0])
        };

        // one Freivalds group's input, scale and accumulators, from a bf16 column
        let mut wire = |me: &mut Self, names: &[&str], x: &[u16], n: usize| {
            let v = match me.mats.iter().find(|m| names.contains(&m.name.as_str())) {
                Some(m) if m.variant == DeqVariant::Pow2 => crate::gadgets::requant::SVariant::Pow2,
                _ => crate::gadgets::requant::SVariant::Bf16,
            };
            let (qx, jx, sx) = crate::gadgets::requant::requant_ref(x, tokens, n, v);
            for name in names {
                let Some(mi) = me.mats.iter().position(|m| m.name == *name) else { continue };
                let m = &mut me.mats[mi];
                assert_eq!(m.n_in, n, "{name}: the Freivalds check's input width is not its producer's");
                m.qx = qx.clone();
                m.jx = jx.iter().map(|j| *j as i32).collect();
                m.sx = sx.clone();
                let (ni, no) = (m.n_in, m.n_out);
                let (qxr, qw) = (m.qx.clone(), m.qw.clone());
                let mut acc = vec![0i64; tokens * no];
                acc.par_chunks_mut(no).enumerate().for_each(|(t, row)| {
                    for k in 0..ni {
                        let xv = qxr[t * ni + k] as i64;
                        if xv == 0 {
                            continue;
                        }
                        for i in 0..no {
                            row[i] += xv * qw[k * no + i] as i64;
                        }
                    }
                });
                me.mats[mi].acc = acc;
            }
        };

        let mut gc = gc;
        // Issue #17, the model tree's control: the norm weight that feeds the
        // swapped matmuls, re-drawn.  It is drawn here with the residual
        // stream, so this is where it is taken -- from the same distribution
        // the honest draw uses, and everything below is rebuilt from it.
        // The *other* norm's weight is left alone for the reason
        // `synth_model_swap` gives: it feeds the attention, whose output
        // windows this shape's statement pins from the honest data.
        if let Some(sw) = model_swap {
            let mut rng = SmallRng::seed_from_u64(sw ^ 0x0017_0017);
            let mut draw = || {
                let e: i32 = rng.random_range(-2..2);
                let m: f64 = rng.random_range(-2.0f64..2.0);
                canon(bf16::from_f64(m * (e as f64).exp2()))
            };
            let w = if head { &mut gc.w1 } else { &mut gc.w2 };
            for x in w.iter_mut() {
                *x = draw();
            }
        }
        // the dequant reference of one matmul's output, under the spec the
        // plan builds for it -- `deq_ref` reads only `variant` and `round`
        let sp = |m: &Mat| crate::gadgets::dequant::Spec {
            variant: m.variant, round, kmax: 0, per_row: 1, limb_bits: (16, 14),
            q_bits: 0, rem_bits: 0, fold_alpha: 0,
        };
        let out_col = |me: &Self, name: &str, w: usize| -> Vec<u16> {
            let mi = me.mats.iter().position(|m| m.name == name)
                .unwrap_or_else(|| panic!("--synth-residual-stream needs the {name} matmul"));
            let sp = sp(&me.mats[mi]);
            (0..tokens)
                .flat_map(|t| (0..w).map(move |c| (t, c)))
                .map(|(t, c)| deq_ref(&me.mats[mi], &sp, t, c))
                .collect()
        };


        let h1 = norm_scale(&gc.xin, &gc.w1, &t1);
        // NN-2: the head chunk's own early return is BELOW, after the `o` seam
        // is wired, which is where it has always been.  Stream NN moved it up
        // here so that the q/k/v wiring and the rotation could run before the
        // `o` requantize -- and took the head chunk's `o` seam with it, so a
        // synthetic head chunk stopped rewiring `o`'s committed input and
        // rejected by `ao_q_ok` and nothing else (stream II's gate step 6,
        // `6075044491ce8db0`).  Only the *non-head* half moves up.
        if !head {
            wire(self, &["q", "k", "v"], &h1, d);
        }
        // ---- stream NN: the key tiles are the q/k/v matmuls' own outputs --------
        //
        // A synthetic layer's key tiles were three unrelated draws, so no shape it
        // could prove carried a positionwise q/k/v binding
        // (`briefs/jj-layerwire-out.md` section 11.4 item 2).  Here the layer's
        // own `q`/`k`/`v` dequant outputs are rotated and quantized -- through
        // `residual_stream::reference` on the very programs the components prove, so the two
        // cannot disagree -- and the key tiles are rebuilt from the result.
        //
        // Wired exactly when the widths line up, the way `o` is: `q` must be
        // `nh hd` channels wide, `k` and `v` equally wide and a whole number of
        // heads, and `nh` a multiple of that number.  A shape where they do not
        // leaves the family off and the run reports it, rather than proving
        // something narrower quietly.
        if !head {
            self.rope = self.rope_from_mats(round, rope_bend, pinned);
        }
        // the `o` matmul's committed input is the attention output requantized,
        // and the attention output has just moved -- so this runs after the
        // rebuild and not before it
        // Stream II (E11), at the JJ merge: the `o` matmul's committed input is
        // the attention output requantized, and `AttnOutEval` is that
        // requantize -- so on a synthetic layer, where the `o` activations are
        // otherwise drawn at random, the seam can only be honest if the witness
        // says so.  Wired exactly when the widths line up: the attention output
        // is `nh * hd` channels wide and the `o` Freivalds check reads `n_in` of them.  A
        // shape where they disagree leaves `o` untied and `ao_q_ok` reports it
        // as unstatable rather than passing quietly.
        let o_wired = {
            let oi = self.mats.iter().position(|m| m.name == "o")
                .expect("--synth-residual-stream needs the o matmul");
            let ok = self.mats[oi].n_in == self.ao.nh * self.ao.hd
                && self.mats[oi].tokens == self.ao.s;
            if ok {
                let q = self.ao.q_ref();
                let (nh, hd, ss) = (self.ao.nh, self.ao.hd, self.ao.s);
                let ni = self.mats[oi].n_in;
                let mut qx = vec![0i32; ss * ni];
                for h in 0..nh {
                    for t in 0..ss {
                        for c in 0..hd {
                            qx[t * ni + h * hd + c] = q[(h * ss + t) * hd + c] as i32;
                        }
                    }
                }
                if let Some(at) = o_bend {
                    // any other legal activation: both neighbours of an
                    // interior value are inside the Freivalds check's range window
                    let w = at % qx.len();
                    qx[w] += if qx[w] < 126 { 1 } else { -1 };
                }
                self.mats[oi].qx = qx;
                let m = &self.mats[oi];
                let (no, qxr, qw) = (m.n_out, m.qx.clone(), m.qw.clone());
                let mut acc = vec![0i64; ss * no];
                acc.par_chunks_mut(no).enumerate().for_each(|(t, row)| {
                    for k in 0..ni {
                        let xv = qxr[t * ni + k] as i64;
                        if xv == 0 {
                            continue;
                        }
                        for i in 0..no {
                            row[i] += xv * qw[k * no + i] as i64;
                        }
                    }
                });
                self.mats[oi].acc = acc;
            }
            ok
        };
        let _ = o_wired;
        // the head chunk: the final norm's output is the head's committed
        // input, and there is no add, no second norm and no SwiGLU after it.
        // It returns here, after the `o` seam, exactly as it did before stream
        // NN (NN-2).
        if head {
            wire(self, &["lm_head"], &h1, d);
            self.residual_stream = Some(gc);
            return;
        }
        // Stream JJ's dequant output bindings: `o_out` is the `o` component's own
        // dequant reference, so an honest run bindings and a bent copy does not.
        // `o`'s input is not rewired (its requantize is `AttnOutEval`'s), so
        // its accumulator is final by here.
        gc.o_out = out_col(self, "o", d);
        let resid = reference(&[t1.lay(&gc.xin), t1.lay(&gc.o_out)], &[], t1.rows1(),
                              &Prog::norm_resid(t1.nreal, t1.b), round);
        let xmid = t1.unlay(&resid[1]);
        let h2 = norm_scale(&xmid, &gc.w2, &t1);
        wire(self, &["gate", "up"], &h2, d);

        // the SwiGLU reads the gate and up components' *dequantized* outputs, so
        // its input is a function of the accumulators `wire` just rebuilt
        let gi = self.mats.iter().position(|m| m.name == "gate").unwrap();
        let ui = self.mats.iter().position(|m| m.name == "up").unwrap();
        let (gs, us) = (sp(&self.mats[gi]), sp(&self.mats[ui]));
        let out_of = |m: &Mat, spec: &Spec| -> Vec<u16> {
            let mut v = vec![0u16; tokens * dff];
            v.par_chunks_mut(dff).enumerate().for_each(|(t, row)| {
                for (c, x) in row.iter_mut().enumerate() {
                    *x = deq_ref(m, spec, t, c);
                }
            });
            v
        };
        let up_out = out_of(&self.mats[ui], &us);
        // Issue #136: the gate matmul's weight scale, sized off the data this
        // layer drew.
        //
        // `silu(x)` is about `x exp(x)` well below zero, so a gate value near
        // -85 has an image a few binades above bf16's smallest normal, and its
        // product with a small `up` is *below* that -- a value the multiply has
        // no exponent for, which the component counts as
        // `exponent_out_of_range` and no `VAL` row answers.  It is the
        // incompleteness edge `ResidualStreamWitness::synthetic` avoids by
        // halving a multiply's spread, and the model's real values never reach
        // it: there the two scales invert the quantization and the dequantized
        // activation is order one whatever the matmul's inner dimension.  Here
        // the accumulator grows with that dimension and the weight scale is an
        // arbitrary draw, so at the 8B width the gate reached -88 and 31 of a
        // 64-token block's products fell out of the range.
        //
        // So the draw is shifted down by the fewest binades that leave every
        // product inside it, the way the fold's divisor and the master
        // exponential table are sized off their own data.  A shape whose
        // products are already inside takes no shift at all and does not move.
        let mut gate_out = out_of(&self.mats[gi], &gs);
        let outside = |g: &[u16]| -> usize {
            g.par_iter().zip(up_out.par_iter())
                .filter(|(a, b)| {
                    crate::gadgets::residual_stream::mul_exponent(
                        crate::gadgets::residual_stream::un_image(UnKind::Silu, **a), **b, round).is_none()
                })
                .count()
        };
        let mut shift = 0u32;
        while outside(&gate_out) > 0 {
            shift += 1;
            assert!(shift <= 24, "gate: no weight scale of this layer keeps the SwiGLU inside \
bf16's normal range");
            for (_, es) in self.mats[gi].sw.iter_mut() {
                assert!(*es > 1, "gate: the weight scale ran out of exponent below the SwiGLU");
                *es -= 1;
            }
            gate_out = out_of(&self.mats[gi], &gs);
        }
        if shift > 0 {
            eprintln!("synthetic layer: the gate weight scale is {shift} binades down, \
so the SwiGLU stays inside bf16's normal range");
        }
        let sw = reference(&[tff.lay(&gate_out), tff.lay(&up_out)], &[], tff.rows1(),
                           &Prog::pairs(tff.nreal, Kind::Mul, UnKind::Silu, UnKind::None),
                           round);
        let x_down = tff.unlay(&sw[0]);
        wire(self, &["down"], &x_down, dff);
        // and the second output binding, after `down`'s accumulator is final
        gc.d_out = out_col(self, "down", d);

        self.residual_stream = Some(gc);
    }
}

// -------------------------------------------------------------- component layouts --

/// One Freivalds component's layout: `Grid` over input channels.
#[derive(Clone, Copy, Debug)]
pub struct FreivaldsLayout {
    pub g: Grid,
    pub pk: usize,
    pub pq: usize,
}

impl FreivaldsLayout {
    pub fn choose(n_in: usize, n_out_max: usize, tokens: usize, p_target: usize) -> Self {
        // pick the chunk count whose weights-per-row is closest to the target,
        // preferring fewer padded rows on bindings
        let mut best: Option<(isize, usize, Self)> = None;
        for chunks in 1..=n_out_max {
            let pk = n_out_max.div_ceil(chunks);
            let pq = tokens.div_ceil(chunks);
            let rows = n_in * chunks;
            let log_rows = (usize::BITS - (rows - 1).leading_zeros()).max(4);
            let Some(g) = Grid::new(n_in, log_rows, chunks, true) else { continue };
            if g.chunks != chunks {
                continue;
            }
            let score = (pk as isize - p_target as isize).abs();
            let pad = (1usize << log_rows) - rows;
            let better = match &best {
                None => true,
                Some((s, p, _)) => score < *s || (score == *s && pad < *p),
            };
            if better {
                best = Some((score, pad, Self { g, pk, pq }));
            }
        }
        best.expect("no Freivalds layout").2
    }
}

// -------------------------------------------------------- shared table registry --

/// Every shared table's contents and multiplicities, accumulated across components.
pub struct Tables {
    pub log_exp: u32,
    pub exp: Vec<u32>,
    pub kmax: u32,
    pub round: bf16::Round,
    pub norm: Vec<bf16::NormRow>,
    pub log_norm: u32,
    pub frac: Vec<(u32, u32, u32)>,
    pub log_frac: u32,
    pub mul: Vec<bf16::MulRow>,
    pub log_mul: u32,
    pub q_bits: u32,
    pub widths: Vec<u32>,
    pub m_exp: Vec<u32>,
    pub m_norm: Vec<u32>,
    pub m_frac: Vec<u32>,
    pub m_mul: Vec<u32>,
    pub m_qnz: Vec<u32>,
    /// Stream JJ (E11): the `EPOW` window table's multiplicities, indexed by
    /// `e - e_lo` with `e = jv - jo`.  `log_epow` is its height; rows past the
    /// window repeat the last real one, so padding adds no admissible triple.
    pub m_epow: Vec<u32>,
    pub e_lo: i32,
    pub e_hi: i32,
    pub log_epow: u32,
    pub m_r: Vec<Vec<u32>>,
    pub off_table: usize,
    /// B9: which table lost a row, how many, and the range of the values that
    /// did -- `off_table` alone cannot tell the two failure modes apart
    pub off_kinds: std::collections::BTreeMap<String, (usize, i64, i64)>,
}

impl Tables {
    #[allow(clippy::too_many_arguments)]
    pub fn new(exp: Vec<u32>, log_exp: u32, kmax: u32, round: bf16::Round, q_bits: u32,
               widths: Vec<u32>, e_lo: i32, e_hi: i32) -> Self {
        let norm = bf16::norm_table(kmax, round);
        let log_norm = crate::common::log_rows_for(norm.len(), 1);
        let sticky = round.needs_sticky();
        let frac: Vec<(u32, u32, u32)> = std::iter::once((0u32, 0u32, 0u32))
            .chain((0..=kmax).flat_map(|k| {
                (0..1u32 << k).map(move |r| (1u32 << k, r, u32::from(sticky && r != 0)))
            }))
            .collect();
        let log_frac = crate::common::log_rows_for(frac.len(), 1);
        let mul = bf16::mul_table(round);
        let log_mul = crate::common::log_rows_for(mul.len(), 1);
        Self {
            m_exp: vec![0; 1 << log_exp],
            m_norm: vec![0; 1 << log_norm],
            m_frac: vec![0; 1 << log_frac],
            m_mul: vec![0; 1 << log_mul],
            m_qnz: vec![0; 1 << q_bits],
            m_epow: vec![0; 1usize << crate::common::log_rows_for(
                (e_hi - e_lo + 1) as usize, 1)],
            e_lo,
            e_hi,
            log_epow: crate::common::log_rows_for((e_hi - e_lo + 1) as usize, 1),
            m_r: widths.iter().map(|w| vec![0; 1usize << w]).collect(),
            log_exp, exp, kmax, round, norm, log_norm, frac, log_frac, mul, log_mul,
            q_bits, widths,
            off_table: 0,
            off_kinds: Default::default(),
        }
    }
    pub fn ridx(&self, bits: u32) -> usize {
        self.widths.iter().position(|w| *w == bits).unwrap()
    }
    pub fn bump_r(&mut self, bits: u32, v: i64) {
        let t = self.ridx(bits);
        bump(&mut self.m_r[t], v, &mut self.off_table);
    }
    pub fn note_off(&mut self, kind: &str, v: i64, n: usize) {
        let e = self.off_kinds.entry(kind.to_string()).or_insert((0, i64::MAX, i64::MIN));
        e.0 += n;
        e.1 = e.1.min(v);
        e.2 = e.2.max(v);
    }
    /// `bump_r` with attribution
    pub fn bump_rk(&mut self, bits: u32, v: i64, kind: &str) {
        let t = self.ridx(bits);
        let before = self.off_table;
        bump(&mut self.m_r[t], v, &mut self.off_table);
        if self.off_table != before {
            self.note_off(kind, v, 1);
        }
    }
    /// Stream JJ (E11): the `EPOW` window table, with attribution.  An `e`
    /// outside the window lands in `off_table`, which is a loud stop.
    pub fn bump_ek(&mut self, e: i32, kind: &str) {
        let before = self.off_table;
        let idx = i64::from(e) - i64::from(self.e_lo);
        bump(&mut self.m_epow, idx, &mut self.off_table);
        if self.off_table != before {
            self.note_off(kind, i64::from(e), 1);
        }
    }
    /// The `EPOW` table's three preprocessed columns, row for row: the key
    /// `e - e_lo`, then `num = 2^max(e,0)` and `pj = 2^max(-e,0)`.  Rows past
    /// the window repeat the last real one, so no padded row is a triple a
    /// lookup could name that the window does not already admit.
    pub fn epow_rows(&self) -> Vec<(u32, u32, u32)> {
        let n = 1usize << self.log_epow;
        let w = (self.e_hi - self.e_lo + 1) as usize;
        (0..n)
            .map(|i| {
                let k = i.min(w - 1);
                let e = self.e_lo + k as i32;
                (k as u32, 1u32 << e.max(0), 1u32 << (-e).max(0))
            })
            .collect()
    }
    /// the (q, nz) pair table, with attribution
    pub fn bump_qnzk(&mut self, v: i64, kind: &str) {
        let before = self.off_table;
        bump(&mut self.m_qnz, v, &mut self.off_table);
        if self.off_table != before {
            self.note_off(kind, v, 1);
        }
    }
    /// O(1) index into the normalisation table (the rows are generated in a
    /// fixed order; a linear scan here was most of B6.5's first trace-gen time).
    pub fn norm_at(&self, m2: u32, ph: u32, st: u32) -> Option<usize> {
        let idx = if m2 == 0 && ph == 0 && st == 0 {
            0
        } else if (1..256).contains(&m2) && ph == 1 && st == 0 {
            m2 as usize
        } else if (256..512).contains(&m2) && ph.is_power_of_two() {
            let k = ph.ilog2() as usize;
            if k > self.kmax as usize || (st == 1 && !self.round.needs_sticky()) || st > 1 {
                return None;
            }
            let per = 1 + self.round.needs_sticky() as usize;
            256 + (k * 256 + (m2 as usize - 256)) * per + st as usize
        } else {
            return None;
        };
        debug_assert!({
            let r = self.norm[idx];
            r.m2 == m2 && r.ph == ph && r.st == st
        });
        Some(idx)
    }
    pub fn frac_at(&self, ph: u32, r: u64) -> Option<usize> {
        if ph == 0 {
            return (r == 0).then_some(0);
        }
        (r < ph as u64).then(|| ph as usize + r as usize)
    }
}

impl Layer {
    /// Stream J: size the layer's master exp table by the largest key the key tiles
    /// actually look up, instead of by the manifest's `d_bits + spread`.
    ///
    /// The two differ because the formula maximises the raw shift and the spread
    /// separately: the key is `d * 2^(F - frac_i)`, so `bits(max_i d_i) + (F -
    /// min_i frac_i)` is only tight when the largest raw shift comes from the
    /// lowest-exponent head.  It never does -- `d_i` is itself proportional to
    /// `2^frac_i`, so the raw maximum sits at a head *near* `F` -- and every layer
    /// of b10-final measures 1-3 bits below the formula
    /// (`experiments/b10/out/oversize/`).
    ///
    /// This changes no numerator and no committed column: the table is a prefix of
    /// the same closed form, so the shrink is a truncation.  What it changes is the
    /// statement's table size, exactly as `--table-bits` does -- and unlike
    /// `--table-bits` it can differ per layer inside one chunk loop.
    ///
    /// Returns `(before, after)` in table bits, and leaves `off_table` on each key tile
    /// recounted against the new size (zero whenever `bits >= key_bits`).
    pub fn set_master_table(&mut self, bits: u32) -> (u32, u32) {
        let _s = witread::scope(Site::Tighten);
        let before = self.key_tiles.iter().map(|b| b.log_table).max().unwrap_or(0);
        let after = bits.clamp(1, before);
        for b in self.key_tiles.iter_mut() {
            b.retable(after);
        }
        (before, after)
    }

    /// The largest exp-table key any key tile of this layer looks up, in bits.
    pub fn key_bits(&self) -> u32 {
        let _s = witread::scope(Site::Tighten);
        self.key_tiles.iter().map(|b| b.key_bits()).max().unwrap_or(1)
    }

    /// B10 step 3's probe on the composed proof: one token attends to a key the
    /// causal mask removes.
    ///
    /// The mask is preprocessed, so the only shape the attack can take is to
    /// *claim* a denominator and an output that count a masked key: the diagonal block
    /// key tile's numerator gets a legal table output, and `Z`, `A.V` and the attnout
    /// witness are rebuilt from it.  The AIR reads neither -- its numerator fold
    /// and its exp lookup drop the slot -- so the diagonal block's `sn` no longer joins
    /// the prefix's to make the denominator the per-token component claims.
    pub fn unmask_diagonal_block(&mut self, i: usize) {
        let n_pre = self.key_tiles.iter().filter(|b| !b.diagonal_block).count();
        let ci = self.key_tiles.iter().position(|b| b.diagonal_block).expect("no diagonal block key tile");
        let nmax = self.key_tiles[ci].nmax as i64;
        let sh = self.key_tiles[ci].shape;
        let masked: Vec<usize> = (0..sh.insts * sh.s * sh.c)
            .filter(|g| !self.key_tiles[ci].visible(*g))
            .collect();
        assert!(!masked.is_empty(), "the diagonal block key tile has no masked entries");
        let g = masked[i % masked.len()];
        self.key_tiles[ci].num[g] = nmax;
        // rebuild that key tile's denominator and A.V output from the changed numerator
        let (ni, ns, nc, nhd) = (sh.insts, sh.s, sh.c, sh.hd);
        let (inst, t, c) = (g / (ns * nc), (g / nc) % ns, g % nc);
        self.key_tiles[ci].zz[inst * ns + t] += nmax;
        for j in 0..nhd {
            let v = self.key_tiles[ci].vv[(inst * nc + c) * nhd + j] as i64;
            self.key_tiles[ci].out[(inst * ns + t) * nhd + j] += nmax * v;
        }
        let _ = ni;
        // and the total the paired prefix key tile's per-token component commits
        let pi = ci - n_pre;
        let add = self.key_tiles[ci].zz[inst * ns + t];
        let own = self.key_tiles[pi].zz[inst * ns + t];
        if let Some(tot) = self.key_tiles[pi].zz_total.as_mut() {
            tot[inst * ns + t] = own + add;
        }
        self.rebuild_ao();
    }

    /// B4 section 3's probe on the composed proof: flip one row of the shared exp
    /// table and rebuild every key tile's numerators, denominators and outputs to
    /// match.  Internally consistent everywhere; only the preprocessed root
    /// differs.
    pub fn corrupt_exp_table(&mut self, i: usize) {
        for b in self.key_tiles.iter_mut() {
            let n = b.table.len();
            let i = i % n;
            b.table[i] = if b.table[i] == 0 { 1 } else { b.table[i] - 1 };
            let sh = b.shape;
            for g in 0..sh.insts * sh.s * sh.c {
                let idx = b.table_key(g);
                b.num[g] = if b.visible(g) && idx >= 0 && (idx as usize) < n {
                    b.table[idx as usize] as i64
                } else {
                    0
                };
            }
            for ti in 0..sh.insts * sh.s {
                b.zz[ti] = b.num[ti * sh.c..(ti + 1) * sh.c].iter().sum();
            }
            let mut out = vec![0i64; sh.insts * sh.s * sh.hd];
            for inst in 0..sh.insts {
                for t in 0..sh.s {
                    for c in 0..sh.c {
                        let nn = b.num[(inst * sh.s + t) * sh.c + c];
                        if nn != 0 {
                            for j in 0..sh.hd {
                                out[(inst * sh.s + t) * sh.hd + j] +=
                                    nn * b.vv[(inst * sh.c + c) * sh.hd + j] as i64;
                            }
                        }
                    }
                }
            }
            b.out = out;
        }
        self.rebuild_ao();
    }
}

pub fn bump(m: &mut [u32], v: i64, off: &mut usize) {
    if v >= 0 && (v as usize) < m.len() {
        m[v as usize] += 1;
    } else {
        *off += 1;
    }
}

// ------------------------------------------------------------- Freivalds components --

pub struct FreivaldsComponent {
    pub name: String,
    pub lay: FreivaldsLayout,
    pub mats: Vec<usize>,
    /// Stream Z: the activation cells' range width (8, or 12 for the widened
    /// attention output's group).  The offset is `2^(act_bits - 1)`.
    pub act_bits: u32,
    /// Under `ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS=1` these are the *mirror* -- the device columns, back
    /// on the host, because `freivalds_interaction` still reads them there. The
    /// commitment takes `dev`, so they never go back up.
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub n_weights: usize,
    /// B7a milestone 4 (section 3 row 1): the same columns, born on the device.
    pub dev: Option<crate::dev_attention_freivalds_witness_gen::DevFreivalds>,
}

impl FreivaldsComponent {
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.g.log_rows
    }
    pub fn pre_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.pre.len(), None => self.pre.len() };
        n << self.lay.g.log_rows
    }
}

/// Build one Freivalds group's component: activations committed (offset
/// `2^(act_bits-1)`), weights preprocessed, range multiplicities bumped at
/// `act_bits` -- 8 everywhere but the widened attention output's group.
pub fn freivalds_component(layer: &Layer, name: &str, mat_idx: &[usize], p_target: usize,
                        act_bits: u32, tables: &mut Tables,
                        corrupt_qx: Option<usize>, oversize_qx: Option<usize>) -> FreivaldsComponent {
    let m0 = &layer.mats[mat_idx[0]];
    let n_in = m0.n_in;
    let tokens = m0.tokens;
    let n_out_max = mat_idx.iter().map(|i| layer.mats[*i].n_out).max().unwrap();
    let lay = FreivaldsLayout::choose(n_in, n_out_max, tokens, p_target);
    let rows = 1usize << lay.g.log_rows;
    let off = 1i32 << (act_bits - 1);
    let qmax = off - 1;
    for i in mat_idx {
        assert_eq!(layer.mats[*i].n_in, n_in, "Freivalds group mixes input widths");
        // q/k/v (and gate/up) must share one committed input
        assert_eq!(layer.mats[*i].qx, m0.qx, "Freivalds group inputs differ");
    }

    // B7a milestone 4 (section 3 row 1): which witness generation runs. 1 is the device's
    // alone -- `pre` and `base` become the mirror the interaction reads -- and
    // 2 builds both and asserts every cell equal before either is used.
    //
    // Stream Z: the device witness generation packs activations as int8 and bumps the 8-bit
    // table in its kernel, so a wide group takes the host witness generation whatever the
    // switch says -- the diagonal block key tiles' pattern, and `devpre::note_host_witness_gen`
    // is told in `generate`.
    let witness_gen_mode = if act_bits == 8 { crate::util::path_mode("ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS") } else { 0 };
    assert!(witness_gen_mode == 0 || corrupt_qx.is_none(),
            "{name}: the device Freivalds witness generation has no `corrupt_qx` path");
    if witness_gen_mode == 1 {
        let dev = crate::dev_attention_freivalds_witness_gen::freivalds_component_dev(layer, mat_idx, &lay, tables);
        // B7a milestone 4 (section 3 row 1): no mirror. `freivalds_interaction` reads
        // the device columns now, and nothing else reads a device component's host
        // columns -- `base_cells` counts them off `dev`.
        return FreivaldsComponent { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits,
                            pre: vec![], base: vec![],
                            n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                            dev: Some(dev) };
    }
    let snap_tables = |t: &Tables| -> (Vec<u32>, usize) {
        (t.m_r[t.ridx(8)].clone(), t.off_table)
    };
    let snap_before = (witness_gen_mode == 2).then(|| snap_tables(tables));

    let mut base = vec![];
    for p in 0..lay.pq {
        let mut col = vec![M31::zero(); rows];
        for (j, slot) in col.iter_mut().enumerate() {
            let v = match lay.g.row(j) {
                Some((k, chunk)) => {
                    let t = chunk * lay.pq + p;
                    if t < tokens {
                        let idx = t * n_in + k;
                        let mut x = m0.qx[idx];
                        if corrupt_qx == Some(idx) {
                            x = if x < qmax { x + 1 } else { x - 1 };
                        }
                        // stream Z's bend control: one committed activation set
                        // one past its range table -- the o-limb window
                        if oversize_qx == Some(idx) { x = off } // x + off = 2^act_bits
                        x + off
                    } else {
                        off
                    }
                }
                None => off,
            };
            tables.bump_r(act_bits, v as i64);
            *slot = m31_of(v as i64);
        }
        base.push(eval_of(lay.g.log_rows, col));
    }

    let mut pre = vec![];
    for mi in mat_idx {
        let m = &layer.mats[*mi];
        for p in 0..lay.pk {
            let mut col = vec![M31::zero(); rows];
            for (j, slot) in col.iter_mut().enumerate() {
                if let Some((k, chunk)) = lay.g.row(j) {
                    let i = chunk * lay.pk + p;
                    if i < m.n_out {
                        *slot = m31_of(m.qw[k * m.n_out + i] as i64);
                    }
                }
            }
            pre.push(eval_of(lay.g.log_rows, col));
        }
    }
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.g.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    let ch = |j: usize| lay.g.row(j).map(|(_, c)| c);
    pre.push(sel(&|j| matches!(ch(j), Some(c) if c + 1 < lay.g.chunks)));
    pre.push(sel(&|j| matches!(ch(j), Some(0))));
    pre.push(sel(&|j| j > 0));
    pre.push(sel(&|j| j + 1 == rows));

    if witness_gen_mode == 2 {
        // the check with teeth, before any column has changed hands
        let after = snap_tables(tables);
        let before = snap_before.unwrap();
        let dev = crate::dev_attention_freivalds_witness_gen::freivalds_component_dev(layer, mat_idx, &lay, tables);
        crate::dev_attention_freivalds_witness_gen::check_cols(&format!("freivalds {name} base"), &base, &dev.base);
        crate::dev_attention_freivalds_witness_gen::check_cols(&format!("freivalds {name} pre"), &pre, &dev.pre);
        let r8 = tables.ridx(8);
        crate::dev_attention_freivalds_witness_gen::check_delta(&format!("freivalds {name} m_r8"), &before.0, &after.0,
                                     &sub_slice(&tables.m_r[r8], &after.0));
        assert_eq!(tables.off_table - after.1, after.1 - before.1,
                   "freivalds {name}: off-table bumps differ");
        // the device's own bumps are the check's, not the proof's
        tables.m_r[r8].copy_from_slice(&after.0);
        tables.off_table = after.1;
        eprintln!("witness_gen-dev: freivalds {name} -- {} base and {} pre columns, {} cells, \
the 8-bit range multiplicities all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.g.log_rows);
    }
    FreivaldsComponent { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits, pre, base,
                 n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                 dev: None }
}

pub struct FreivaldsInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed: Vec<SecureField>,
    pub use_sum: SecureField,
    /// Stream II: the group's claim over its own committed activations, in the
    /// canonical channel-major order, or zero when the AIR carries none.
    pub qx_claim: SecureField,
}

pub fn freivalds_interaction(layer: &Layer, r: &FreivaldsComponent, rv: SecureField, s_ch: SecureField,
                             rels: &crate::compose::Rels, range_check: bool,
                             air: &crate::gadgets::freivalds::FreivaldsEval, gkr: bool) -> FreivaldsInteraction {
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let lay = r.lay;
    let rows = 1usize << lay.g.log_rows;
    let m0 = &layer.mats[r.mats[0]];
    let (rv_pow, rv_step) = crate::relations::powers(rv, lay.pk);
    let (s_pow, s_step) = crate::relations::powers(s_ch, lay.pq);

    // B7a milestone 4 (section 3 row 1): the Freivalds checks' v, us and accs, on the same
    // three-value mode every witness generation in this series uses.
    let acc_dev = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let acc_host = acc_dev.is_none() || crate::devacc::mode() == 2;
    let rows = if acc_host { rows } else { 0 };
    // v's inner sums in parallel (the activations), u's per matmul
    let inner: Vec<(SecureField, Vec<SecureField>)> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut vf = SecureField::zero();
            let mut ufs = vec![SecureField::zero(); r.mats.len()];
            if let Some((k, chunk)) = lay.g.row(j) {
                for p in 0..lay.pq {
                    let t = chunk * lay.pq + p;
                    if t < m0.tokens {
                        let x = m0.qx[t * m0.n_in + k];
                        if x != 0 {
                            vf += s_pow[p] * SecureField::from(m31_of(x as i64));
                        }
                    }
                }
                for (mi, uf) in r.mats.iter().zip(ufs.iter_mut()) {
                    let m = &layer.mats[*mi];
                    for p in 0..lay.pk {
                        let i = chunk * lay.pk + p;
                        if i < m.n_out {
                            let w = m.qw[k * m.n_out + i];
                            if w != 0 {
                                *uf += rv_pow[p] * SecureField::from(m31_of(w as i64));
                            }
                        }
                    }
                }
            }
            (vf, ufs)
        })
        .collect();
    let mut v = vec![SecureField::zero(); rows];
    let mut us = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    let mut accs = vec![vec![SecureField::zero(); rows]; r.mats.len()];
    for j in 0..rows {
        let (nofirst, last) = match lay.g.row(j) {
            Some((_, c)) => (c + 1 < lay.g.chunks, c == 0),
            None => (false, false),
        };
        v[j] = if nofirst { v[j - 1] * s_step } else { SecureField::zero() } + inner[j].0;
        for m in 0..r.mats.len() {
            us[m][j] = if nofirst { us[m][j - 1] * rv_step } else { SecureField::zero() }
                + inner[j].1[m];
            accs[m][j] = if j > 0 { accs[m][j - 1] } else { SecureField::zero() };
            if last {
                accs[m][j] += us[m][j] * v[j];
            }
        }
    }
    let mut cols = crate::device::ColSink::new();
    let claimed: Vec<SecureField> = match acc_dev {
        Some(d) => {
            let got = crate::devacc::freivalds(d, &lay, r.mats.len(), rv, s_ch);
            let hh = acc_host.then(|| {
                let mut e = crate::encoding::secure_cols_pub(lay.g.log_rows, &v);
                for m in 0..r.mats.len() {
                    e.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &us[m]));
                    e.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &accs[m]));
                }
                (e, accs.iter().map(|a| a[rows - 1]).collect::<Vec<_>>())
            });
            crate::devacc::place(&format!("freivalds {} 2^{}", r.name, lay.g.log_rows), got,
                                 hh, &mut cols)
        }
        None => {
            cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &v));
            for m in 0..r.mats.len() {
                cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &us[m]));
                cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &accs[m]));
            }
            accs.iter().map(|a| a[rows - 1]).collect()
        }
    };
    // stream II: the activation claim, in the AIR's own mask order -- after the
    // per-matmul accumulators and before the LogUp columns.  It is built from
    // `m0.qx` and not from `r.base`, so it is the same column whether the witness generation
    // left the component on the host or on the card, and it does not depend on
    // `rows` having been zeroed for the device accumulator path above.
    let mut qx_claim = SecureField::zero();
    if let Some(q) = air.qx.as_ref() {
        let nrows = 1usize << lay.g.log_rows;
        let inner_x: Vec<SecureField> = (0..nrows)
            .into_par_iter()
            .map(|j| {
                let mut f = SecureField::zero();
                if let Some((k, chunk)) = lay.g.row(j) {
                    for p in 0..lay.pq {
                        let t = chunk * lay.pq + p;
                        if t < m0.tokens {
                            let x = m0.qx[t * m0.n_in + k];
                            if x != 0 {
                                f += q.w[p] * SecureField::from(m31_of(x as i64));
                            }
                        }
                    }
                }
                f
            })
            .collect();
        let mut xs = vec![SecureField::zero(); nrows];
        for j in 0..nrows {
            let (nofirst, notzero) = match lay.g.row(j) {
                Some((_, c)) => (c + 1 < lay.g.chunks, j > 0),
                None => (false, j > 0),
            };
            let carry = if j == 0 {
                SecureField::zero()
            } else if nofirst {
                xs[j - 1] * q.m_in
            } else if notzero {
                xs[j - 1] * q.m_bd
            } else {
                SecureField::zero()
            };
            xs[j] = carry + inner_x[j];
        }
        qx_claim = xs[nrows - 1];
        cols.extend(crate::encoding::secure_cols_pub(lay.g.log_rows, &xs));
    }
    let mut use_sum = SecureField::zero();
    // B11 stream V, lever 4: under `--gkr-freivalds` this group's range checks were
    // handed to the collector before the GKR proof ran, in `compose`'s own pass,
    // so there are no running-sum columns to build and no claimed sum to carry.
    // The `v`, `u` and `acc` columns above stay: they are Horner accumulators
    // and 38.9% of this family's interaction cells, which the forecast priced.
    if range_check && !gkr {
        let one = PackedQM31::broadcast(SecureField::one());
        let fr = |i: usize, vr: usize|
            (one, rels.range(r.act_bits).combine(&[r.base[i].data[vr]]));
        let (lcols, s) = crate::devfrac::frac_cols(
            air, &format!("freivalds {}", r.name), lay.g.log_rows,
            crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
            crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
            || {
                let (n_pairs, fpair) = crate::plogup::paired(lay.pq, fr);
                crate::plogup::build(lay.g.log_rows, n_pairs, fpair)
            });
        cols.extend_any(lcols);
        use_sum = s;
    }
    FreivaldsInteraction { cols: cols.into_inner(), claimed, use_sum, qx_claim }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #72: `--table-bits` sizes a synthetic layer's master exp table.
    ///
    /// The rotation rebuilds every key tile and sizes the table again from the
    /// rebuilt data.  That second sizing was clamped to 2^22 rows whatever the
    /// line said, so a shape whose keys need more -- this one needs 23 bits at a
    /// 128-token block, and the 8B model's attention width needs 24 -- left them
    /// off the table, where only a rejected proof reported them.
    #[test]
    fn table_bits_sizes_the_rotated_synthetic_table() {
        let argv = |extra: &[&str]| -> Vec<String> {
            ["compose", "--round", "even", "--numbits", "9", "--z-hi-bits", "7",
             "--variants", "intexp", "--hd", "32", "--head-groups", "2,2",
             "--nin", "128", "--nout", "128", "--residual-stream-a", "32", "--kmax", "16",
             "--synth-residual-stream", "--diagonal-block", "--s", "128", "--ctx", "128"]
                .iter().chain(extra.iter()).map(|s| s.to_string()).collect()
        };
        let (l, _) = crate::cli::layer_of(&argv(&["--table-bits", "23"]));
        let bits = l.key_tiles[0].log_table;
        assert!(bits > 22, "the exp table is still clamped to 2^{bits} rows");
        assert!(l.key_tiles.iter().all(|b| b.log_table == bits), "key tiles disagree on table rows");
        assert!(l.key_bits() <= bits,
                "keys of {} bits against a table of 2^{bits} rows", l.key_bits());
        assert_eq!(l.key_tiles.iter().map(|b| b.off_table).sum::<usize>(), 0);
    }

    /// Issue #136: the SwiGLU of an honest synthetic layer at the 8B model's
    /// attention width stays inside bf16's normal range.
    ///
    /// The gate matmul's dequantized output reached -88 there, `silu` of which
    /// is a few binades above the smallest normal, and its product with a small
    /// `up` was below it: 31 of a 64-token block's products with no exponent the
    /// multiply can write, 29 of them disagreeing with the reference, and an
    /// honest proof rejected with `ConstraintsNotSatisfied`.  The counts here are
    /// the run's own.
    #[test]
    fn the_synthetic_swiglu_stays_in_bf16_range_at_the_8b_width() {
        use crate::gadgets::residual_stream::{UnKind, mul_exponent, un_image};
        let argv: Vec<String> =
            ["compose", "--round", "even", "--numbits", "9", "--z-hi-bits", "7",
             "--variants", "intexp", "--table-bits", "24", "--hd", "128",
             "--head-groups", "16,8,8", "--nin", "4096", "--nout", "4096",
             "--residual-stream-a", "32", "--kmax", "19", "--ao-rem-bits", "22",
             "--synth-residual-stream", "--diagonal-block", "--s", "64", "--ctx", "64"]
            .iter().map(|x| x.to_string()).collect();
        let (l, _) = crate::cli::layer_of(&argv);
        let round = bf16::Round::Even;
        let tokens = l.ao.s;
        let dff = l.mats.iter().find(|m| m.name == "gate").unwrap().n_out;
        let spec = |m: &Mat| Spec {
            variant: m.variant, round, kmax: 0, per_row: 1, limb_bits: (16, 14),
            q_bits: 0, rem_bits: 0, fold_alpha: 0,
        };
        let out = |name: &str| -> Vec<u16> {
            let m = l.mats.iter().find(|m| m.name == name).unwrap();
            let sp = spec(m);
            (0..tokens * dff).map(|i| deq_ref(m, &sp, i / dff, i % dff)).collect()
        };
        let (gate_out, up_out) = (out("gate"), out("up"));
        let outside = gate_out.iter().zip(up_out.iter())
            .filter(|(a, b)| mul_exponent(un_image(UnKind::Silu, **a), **b, round).is_none())
            .count();
        assert_eq!(outside, 0,
                   "{outside} of {} SwiGLU products leave bf16's normal range",
                   tokens * dff);
    }

    /// Issue #53: a dump whose rounding disagrees with `--round` is refused at
    /// load, before the first commitment.
    ///
    /// `b10-final` rounds a half to even and its manifest says so. Proved with
    /// `--round away` -- the mode the old reference row named -- the dequant
    /// tables are built for the other rounding, 797 of the dump's values fall
    /// off them, and the honest proof was made in full and then rejected.
    #[test]
    fn a_dump_rounded_the_other_way_is_refused_at_load() {
        let man = serde_json::json!({"dequant_round": "even"});
        let why = round_refusal(&man, bf16::Round::Away).expect("the mismatch must be refused");
        assert!(why.contains("dequant_round is \"even\""), "{why}");
        assert!(why.contains("--round away"), "{why}");
        assert!(why.contains("--round even"), "{why}");
        // the dump's own mode passes, under either spelling of it
        assert_eq!(round_refusal(&man, bf16::Round::Even), None);
        assert_eq!(round_refusal(&serde_json::json!({"dequant_round": "rne"}),
                                 bf16::Round::Even), None);
        // and a dump that does not say is not refused: there is nothing to compare
        assert_eq!(round_refusal(&serde_json::json!({"s": 512}), bf16::Round::Away), None);

        // the loader refuses before it reads anything else: this directory holds a
        // manifest and no dump at all, and the error is the rounding and not a
        // missing field or a missing file
        let dir = std::env::temp_dir().join(format!("zk-prover-round-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory to hold the manifest");
        std::fs::write(dir.join("manifest.json"), man.to_string()).expect("write the manifest");
        let cfg = Src { dir: None, head: 0, q_bits: 12, k_bits: 11, nmax: 511, log_table: 0,
                        seed: 7, layout: String::new(), diagonal_block: false, tile: false };
        let err = Layer::load(dir.to_str().unwrap(), &cfg, &[2, 1], Variant::IntExp,
                              bf16::Round::Away)
            .err().expect("the loader must refuse this dump");
        assert!(err.contains("dequant_round"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
