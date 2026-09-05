//! The layer witness B6.5 composes into one proof, and the trace builders for
//! the regions B6 did not have: the per-matmul dequant regions (with the
//! Freivalds claimed side riding on the same committed accumulators), the
//! attention-output region, and the r^T W walk regions.
//!
//! Layouts, once, because every builder and every selector has to agree:
//!
//! * **matmul dequant region** (one per matmul): rows = (token, channel-chunk),
//!   `per_row` channels per row with `per_row | n_out`, so the region tiles
//!   *exactly* (no padded slots, no gating).  Tokens and chunks descend, slots
//!   ascend: row `j` holds token `T-1 - j/chunks`, chunk `b = chunks-1 - j%chunks`,
//!   channel `c = b*per_row + slot`.
//! * **attnout region**: rows = (head, token, channel-chunk), heads and tokens and
//!   chunks descending, with `lead` inert rows *first* (nh*S*chunks is not a power
//!   of two); inert rows commit `den = 1` and zeros, which satisfies every
//!   constraint and looks up real table rows.
//! * **walk region** (one per input): `layout::Grid` over input channels, `pk`
//!   preprocessed weights per row per matmul and `pq` committed activation bytes.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use num_traits::{One, Zero};

use crate::attn_witness::{Attn, LayerRaw as AttnRaw, Src, Variant};
use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::dequant::{Map, Spec, Variant as DeqVariant};
use crate::layout::{Grid, Shape};
use crate::witread::{self, Site};

pub fn m31_of(v: i64) -> M31 {
    crate::attn_trace::m31_of(v)
}

// ------------------------------------------------------------------ the witness --

/// One matmul's dequant + walk witness.
pub struct Mat {
    pub name: String,
    pub variant: DeqVariant,
    pub n_in: usize,
    pub n_out: usize,
    pub tokens: usize,
    /// (tokens, n_out) row-major exact integer accumulators
    pub acc: Vec<i64>,
    /// (tokens, n_in) quantized activations
    pub qx: Vec<i32>,
    /// (n_in, n_out) quantized weights
    pub qw: Vec<i32>,
    /// per-token / per-channel power-of-two scale exponents (pow2 variant)
    pub jx: Vec<i32>,
    pub jw: Vec<i32>,
    /// per-token / per-channel bf16 scales as (mantissa field, biased exponent)
    pub sx: Vec<(u32, u32)>,
    pub sw: Vec<(u32, u32)>,
    /// Stream Z, the fold: the consumer's power-of-two scale exponent per
    /// output channel (`s_v` for the `v` matmul, collapsed from the attention
    /// witness's per-query-head copy).  Empty on every other variant.
    pub jv2: Vec<i32>,
}

impl Mat {
    /// The fold's divisor exponents (`dequant.rs`'s `Fold` arm): the token
    /// side `G_t = 134 - es_x` and the channel side `G_c = 134 - ew + jv`.
    pub fn fold_gt(&self, t: usize) -> i64 {
        134 - self.sx[t].1 as i64
    }
    pub fn fold_gc(&self, c: usize) -> i64 {
        134 - self.sw[c].1 as i64 + self.jv2[c] as i64
    }
    /// The largest |acc|, and the `kmax` the shared normalisation table needs.
    pub fn kmax_needed(&self) -> u32 {
        let m = self.acc.iter().map(|a| a.unsigned_abs()).max().unwrap_or(0);
        (64 - m.leading_zeros()).saturating_sub(9)
    }
    /// The blind worst case: `127^2 * n_in`.
    pub fn kmax_blind(&self) -> u32 {
        let m = 127u64 * 127 * self.n_in as u64;
        (64 - m.leading_zeros()).saturating_sub(9)
    }
}

/// The attention-output fold's witness: the A.V outputs of every head, their
/// denominators, and the power-of-two scale exponents.
pub struct AttnOut {
    pub nh: usize,
    pub s: usize,
    pub hd: usize,
    /// (nh, S, hd): the A.V outputs (the rectangle recomputation, as the bands)
    pub acc: Vec<i64>,
    /// (nh, S): the committed denominators Z
    pub den: Vec<i64>,
    /// per-token output scale exponent (pow2 s_o)
    pub jo: Vec<i32>,
    /// per-(head, channel) V scale exponent (pow2 s_v)
    pub jv: Vec<i32>,
}

impl AttnOut {
    pub fn e(&self, h: usize, t: usize, j: usize) -> i32 {
        self.jv[h * self.hd + j] - self.jo[t]
    }
    /// Stream JJ (E11): the per-token output exponent's window, from the data.
    /// `jo` is prover data; this is the window the statement admits it in.
    pub fn jo_range(&self) -> (i32, i32) {
        let lo = self.jo.iter().copied().min().unwrap_or(0);
        let hi = self.jo.iter().copied().max().unwrap_or(0);
        (lo, hi)
    }
    /// Stream II (E11), at the JJ merge: the attention output's requantized
    /// integers, `sign(acc) * round(|acc| 2^(jv - jo) / den)` per
    /// `(head, token, channel)` in that order -- exactly what `ao_region`
    /// commits as `q_out` and what `AttnOutEval`'s `claimed_q` folds.
    ///
    /// It lives here, off `AttnOut` alone, so a witness builder can produce the
    /// `o` matmul's committed input without running the region: on a real dump
    /// B1 wrote that input and the region recomputes it, but a *synthetic*
    /// layer draws its `o` activations at random and the `o` seam could not be
    /// honest on any desk shape.  `ao_region` asserts its own `q_out` against
    /// this on every untampered run, so the two cannot drift apart.
    pub fn q_ref(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.nh * self.s * self.hd];
        for h in 0..self.nh {
            for t in 0..self.s {
                let den = self.den[h * self.s + t];
                for c in 0..self.hd {
                    let idx = (h * self.s + t) * self.hd + c;
                    let e = self.jv[h * self.hd + c] - self.jo[t];
                    let (num, pj) = (1i64 << e.max(0), 1i64 << (-e).max(0));
                    let (denp, pm) = (den * pj, self.acc[idx].abs() * num);
                    let qm = (2 * pm + denp) / (2 * denp);
                    out[idx] = if self.acc[idx] < 0 { -qm } else { qm };
                }
            }
        }
        out
    }

    /// The per-(head, channel) V exponent's window.  `jv` is calibrated offline
    /// and is a model constant, so this is a statement of the model.
    pub fn jv_range(&self) -> (i32, i32) {
        let lo = self.jv.iter().copied().min().unwrap_or(0);
        let hi = self.jv.iter().copied().max().unwrap_or(0);
        (lo, hi)
    }
    /// The bit length of `1 + max |acc|`: the width the two-limb range check on
    /// the accumulator has to cover, and with it the `2 mag num < p` half of
    /// the no-wrap window.
    pub fn acc_bits(&self) -> u32 {
        let m = self.acc.iter().map(|a| a.abs()).max().unwrap_or(0);
        64 - ((m as u64) + 1).leading_zeros()
    }

    /// The rounding-error range check's width: `d < 2 den 2^max(-e,0)` over every
    /// active site, from the data rather than a blind bound.
    pub fn rem_bits(&self) -> u32 {
        let mut m = 2i64;
        for h in 0..self.nh {
            for t in 0..self.s {
                for j in 0..self.hd {
                    let denp = self.den[h * self.s + t] << (-self.e(h, t, j)).max(0);
                    m = m.max(2 * denp);
                }
            }
        }
        64 - (m as u64).leading_zeros()
    }
}

pub struct Layer {
    pub bands: Vec<Attn>,
    pub band_sizes: Vec<usize>,
    pub head_offsets: Vec<usize>,
    /// Stream AA (E11): how many bands one head band group's declared prefix is
    /// tiled into.  1 everywhere before this stream and without `--kv-bind`;
    /// `C / S` with it, which is what makes a consumer's sub-band the same
    /// `Shape` as the producer's corner; 0 on block 0 of a sequence, which has
    /// no declared prefix and so no prefix band.  `params_of` reads it back to
    /// recover the chunk's real context and its head band list.
    pub nsub: usize,
    pub mats: Vec<Mat>,
    pub ao: AttnOut,
    pub source: String,
    /// B10 step 2: the wiring columns, present only on a witness extracted with
    /// `extract.py --glue`.  Every one is a `(tokens x d)` bf16 *pattern* column
    /// of B1's own residual stream; `xmid` and `xout` are what the glue's two
    /// adds have to reproduce, and the two norm weights and the two scalars are
    /// what the reduction's tail consumes.
    pub glue: Option<GlueCols>,
    /// stream E: B1's own `key_head.u16`, the order keys the head's rows compare.
    /// `None` unless the dump was written with `--head`.  It is a reference, not
    /// an input: the proof derives its keys from the head's dequant output, and
    /// this is what says the two agree.
    pub head_keys: Option<Vec<u16>>,
    /// stream E: `tok_head.i32`, the token each row emits.  The argmax region's
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
        let inv_ref = witread::read_file(&format!("{dir}/inv_freq.f64")).ok().map(|raw| {
            raw.chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .take(hd / 2)
                .collect::<Vec<f64>>()
        }).filter(|v| v.len() == hd / 2);
        let j = |n: &str| -> Result<Vec<i32>, String> {
            let v = read_i32(&format!("{dir}/{n}.i32"))?;
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
    /// table (`1 << log_table` rows) and the bands' per-instance `dinv` scales.
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
        if let Some(g) = &self.glue {
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

pub struct GlueCols {
    pub d: usize,
    pub dff: usize,
    pub tokens: usize,
    pub xin: Vec<u16>,
    pub o_out: Vec<u16>,
    pub xmid: Vec<u16>,
    pub d_out: Vec<u16>,
    pub xout: Vec<u16>,
    /// `n1_out` and `n2_out`: each norm's `r = rsqrt(mean of squares + eps)`,
    /// one value per token.  The tails multiply the stream by it, so a row of
    /// the tail region reads it once and every channel of that row shares it.
    pub r1: Vec<u16>,
    pub r2: Vec<u16>,
    pub w1: Vec<u16>,
    pub w2: Vec<u16>,
    /// B1's own norm outputs, which are what the tails have to reproduce:
    /// `x_q` is norm1's (q, k and v all read the same column, and the dump's
    /// digests say so) and `x_gate` is norm2's.  `None` on a dump that did not
    /// write them, in which case the tail region runs without its reference.
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

impl GlueCols {
    /// The seven columns `extract.py --glue` writes, plus `eps` and `1/d` from
    /// the manifest.  Absent `glue` in the manifest the layer simply has none and
    /// the composition runs as it did before.
    pub fn load(dir: &str, man: &serde_json::Value, tokens: usize) -> Result<Option<Self>, String> {
        let g = &man["glue"];
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
        // A column the tails and the SwiGLU only *check* against is optional:
        // an older dump that does not carry it costs the reference, not the
        // region, and the record says which of the two happened.
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
            eps: g["eps"].as_u64().ok_or("manifest: glue.eps")? as u16,
            inv_d: g["inv_d"].as_u64().ok_or("manifest: glue.inv_d")? as u16,
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
}

impl GlueCols {
    /// Stream II: the wiring family on a synthetic layer, so the desk can run
    /// the whole composed statement -- ties, norm tails, SwiGLU and all -- at a
    /// shape that proves in seconds and without a dump.  Only the columns the
    /// regions *consume* are built: `xmid`, `xout` and the five reference
    /// columns are what the regions compute, and a synthetic layer has no B1 to
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
                    crate::glue::canon(bf16::from_f64(m * (e as f64).exp2()))
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

/// The seven layer matmuls, in the order every region and every selector reads
/// them.  Which dequant gadget each takes is *not* fixed here: it is a property
/// of the spec the witness was extracted under, and the two differ.
pub const MAT_NAMES: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];

/// B6.5's mixed dequant layout, which the B6 review's per-matmul attribution
/// licensed: the power-of-two dequant where it is free (q/k/v/o, -0.43%) and the
/// bf16 chain where it is not.
pub const MAT_VARIANTS: [(&str, DeqVariant); 7] = [
    ("q", DeqVariant::Pow2), ("k", DeqVariant::Pow2), ("v", DeqVariant::Pow2),
    ("o", DeqVariant::Pow2),
    ("gate", DeqVariant::Chain), ("up", DeqVariant::Chain), ("down", DeqVariant::Chain),
];

/// B10's layout: the bf16 chain on all seven.  This is what B8 licensed
/// (`b65-design` costs +11.69% perplexity at 8B against `b6-design-p2o`'s
/// +8.81%) and what the denominator charges, so it is the layout of the spec
/// that is both scored and proved.
pub const MAT_VARIANTS_CHAIN: [(&str, DeqVariant); 7] = [
    ("q", DeqVariant::Chain), ("k", DeqVariant::Chain), ("v", DeqVariant::Chain),
    ("o", DeqVariant::Chain),
    ("gate", DeqVariant::Chain), ("up", DeqVariant::Chain), ("down", DeqVariant::Chain),
];

/// B10 step 0: the layout by name.  `b65` is B6.5's mixed one, `b10` the chain
/// on all seven; a real witness carries its own (`deq_variants` in the manifest,
/// written by `extract.py` from the spec) and this is only the fallback and the
/// synthetic layer's choice.
pub fn mat_variants(layout: &str) -> [(&str, DeqVariant); 7] {
    match layout {
        "" | "b65" => MAT_VARIANTS,
        "b10" | "chain" => MAT_VARIANTS_CHAIN,
        // stream Z: the wired layout on a synthetic layer -- the chain
        // everywhere, the delayed requantize on v.  A smoke shape for the fold's
        // AIR, fill and LogUp before a real witness is anywhere near it.
        "zfold" => {
            let mut v = MAT_VARIANTS_CHAIN;
            v[2] = ("v", DeqVariant::Fold);
            v
        }
        other => panic!("unknown dequant layout {other} (b65 | b10 | zfold)"),
    }
}

/// The walk groups: matmuls sharing one committed input.
/// B10 step 4: `lm_head` is its own group.  It shares no committed input with the
/// layer's matmuls -- its input is the *final* norm's output, not any layer's --
/// so it walks alone, over `d` input channels like every other group.
pub const WALK_GROUPS: [(&str, &[&str]); 5] =
    [("qkv", &["q", "k", "v"]), ("o", &["o"]), ("gateup", &["gate", "up"]),
     ("down", &["down"]), ("head", &["lm_head"])];

impl Layer {
    pub fn load(dir: &str, cfg: &Src, band_sizes: &[usize], variant: Variant)
                -> Result<Self, String> {
        let man: serde_json::Value = witread::read_json(&format!("{dir}/manifest.json"))?;
        let g = |k: &str| man[k].as_u64().ok_or(format!("manifest: missing {k}"));
        let (nh, s, c, hd) = (g("heads")? as usize, g("s")? as usize,
                              g("c")? as usize, g("hd")? as usize);
        assert_eq!(band_sizes.iter().sum::<usize>(), nh, "bands must cover every head");
        // B9 step 1a: the master exponent is layer-wide.  `Attn::build` takes the
        // largest `frac` over the band's *own* heads, so which table a band builds
        // depends on how the heads were split -- and the three assertions below
        // then hold only where every band happens to contain a head at the
        // layer-wide maximum (4 of 24 Qwen-0.5B layers; 0 of 32 Llama-3.1-8B ones,
        // B8.md section 2 item 3).  The forward pass already writes every head's
        // exponent into the manifest, so the layer-wide maximum is read, not
        // derived, and handed to every band: each then scales its shifts up to it
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
        // `Attn::build`'s own default caps at 2^22 -- right for a single-band
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
        // B10 step 3: a dump extracted with `--corner` carries the causal corner
        // beside the prefix rectangle, and it rides as one more band per prefix
        // band -- same `insts`, same head offset, `C = S`.
        let has_corner = man["corner"].as_bool().unwrap_or(false);
        // Stream W: the layer's attention columns, read and widened once for
        // every band that will gather out of them.  Each of the six bands used
        // to open these files for itself -- `qq.i16` and `mm.i32` six times, a
        // half's `kk`, `vv` and `num` three times each -- so a chunk read
        // 1,502 MB where 740 MB exists and the read was the largest single
        // site stream P left (`briefs/p-witread.md` (6) item 1).  Nothing
        // about the bytes or the layout changed; the same gathers now run over
        // one image instead of over three copies of it.
        // Open item 1: block 0 of a sequence has no declared prefix, so `c == 0`,
        // `kk.i16` is empty and there is no prefix half to read and no prefix
        // band to build -- the corner is the group's only band and carries its
        // per-token region (`compose.rs`'s `rows_band`, `Powers::as_solo_corner`).
        assert!(c > 0 || has_corner,
                "a c = 0 dump is block 0: it is the solo corner and needs a `--corner` dump");
        let raw = {
            let _s = witread::scope(Site::Bands);
            AttnRaw::read(dir, &man, c > 0, has_corner)?
        };
        let mut bands = vec![];
        let mut head_offsets = vec![];
        let mut sizes = vec![];
        let mut off = 0usize;
        // Stream AA (E11), layout design A: with `--kv-bind` the declared prefix
        // is loaded as `c / s` sub-bands of exactly `S` keys each rather than as
        // one rectangle, because that is what makes a consumer's sub-band `b`
        // the same `Shape` -- and so the same `LayoutKey`, and so a column of
        // the same height -- as the producer's corner.  `Attn::load_from` reads
        // the same bytes with a key offset; nothing about the dump changes.
        assert!(!cfg.tile || cfg.corner, "`--kv-bind` needs a `--corner` dump");
        assert!(!cfg.tile || c % s == 0,
                "`--kv-bind` tiles the prefix in blocks of S: C={c} is not a multiple of S={s}");
        let nsub = if c == 0 { 0 } else if cfg.tile { c / s } else { 1 };
        let sub_c = if cfg.tile { s } else { c };
        {
            let _s = witread::scope(Site::Bands);
            for insts in band_sizes {
                for k in 0..nsub {
                    let sh = Shape { s, c: sub_c, hd, insts: *insts };
                    let src = Src { head: off, ..cfg.clone() };
                    bands.push(Attn::load_from(&raw, &man, off, sh, variant, src.nmax,
                                               src.log_table, master, false, k * sub_c)?);
                    head_offsets.push(off);
                    sizes.push(*insts);
                }
                off += insts;
            }
        }
        if has_corner {
            let mut off = 0usize;
            {
                let _s = witread::scope(Site::Bands);
                for insts in band_sizes {
                    let sh = Shape { s, c: s, hd, insts: *insts };
                    let src = Src { head: off, ..cfg.clone() };
                    bands.push(Attn::load_from(&raw, &man, off, sh, variant, src.nmax,
                                               src.log_table, master, true, 0)?);
                    head_offsets.push(off);
                    sizes.push(*insts);
                    off += insts;
                }
            }
            // the per-token region belongs to the pair, and the denominator it
            // commits is the total: `Z >= nmax` is true of the sum and of neither
            // half, because the key whose shift is zero may be in either
            //
            // Stream AA: with the prefix tiled, "the pair" is a head band
            // group -- `nsub` sub-bands and the corner -- and the total is over
            // all of them.  The region lives on the group's *first* sub-band,
            // which is `plan.rows_band`'s choice on the other side of the wall.
            let n_pre = band_sizes.len() * nsub;
            // the band that carries group `g`'s region: its first prefix
            // sub-band, or -- block 0, no prefix at all -- its solo corner.
            // `g * nsub` alone is band 0 for every group when `nsub == 0`, and
            // that read as "the extractor and the AIR disagree about the mask
            // on 16351 of 16384 rows" (open item 1) when every row agreed.
            let first = |g: usize| if nsub > 0 { g * nsub } else { n_pre + g };
            {
                let _s = witread::scope(Site::ZzTotal);
                for g in 0..band_sizes.len() {
                    let mut tot = vec![0i64; bands[first(g)].zz.len()];
                    for b in (g * nsub..(g + 1) * nsub).chain(std::iter::once(n_pre + g)) {
                        assert_eq!(bands[b].zz.len(), tot.len(),
                                   "band {b} of group {g} has a different row count");
                        for (a, x) in tot.iter_mut().zip(&bands[b].zz) {
                            *a += x;
                        }
                    }
                    bands[first(g)].zz_total = Some(tot);
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
            for (g, insts) in band_sizes.iter().enumerate() {
                let tot = bands[first(g)].zz_total.as_ref().unwrap();
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
        // The image has served every band it will serve, and it is a third of a
        // gigabyte.  Freeing it here rather than at the end of the function
        // keeps the layer's high-water mark where it was before this stream.
        drop(raw);
        let band_sizes = &sizes[..];
        // one master table serves every band: same rows, same contents.  With a
        // layer-wide `master` this holds by construction; without one (a dump whose
        // Q scale is not per head) it is the old precondition, and the message
        // names the fix.
        {
            let _s = witread::scope(Site::CheckTableEq);
            for b in &bands[1..] {
                assert_eq!(b.log_table, bands[0].log_table, "bands disagree on table rows");
                assert_eq!(b.table, bands[0].table,
                           "bands disagree on the exp table (no layer-wide master exponent: \
                            the dump has frac_shared_per_head = false)");
                assert_eq!(b.frac, bands[0].frac, "bands disagree on the master exponent");
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

        // the attnout region proves what the bands prove -- which with the corner
        // is two bands per head, so a head's totals accumulate
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        {
            let _s = witread::scope(Site::AttnOut);
            for (bi, b) in bands.iter().enumerate() {
                for i in 0..band_sizes[bi] {
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
        // carries `glue`, and the *head* chunk's final norm when it does not
        // but carries `head.final_norm`.  No flag: which family a chunk can
        // prove is a property of the columns the extractor wrote, and a dump
        // written before either simply has none, as before.
        let glue = match GlueCols::load(dir, &man, s)? {
            Some(g) => Some(g),
            None => GlueCols::load_head(dir, &man, s)?,
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
            bands,
            band_sizes: band_sizes.to_vec(),
            head_offsets,
            nsub,
            mats,
            ao: AttnOut { nh, s, hd, acc, den, jo, jv },
            source: format!("B6.5 layer witness [{dir}]"),
            glue,
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

    /// Stream NN: derive the bands' Q, K and V from this layer's own `q`, `k`
    /// and `v` matmuls, and rebuild the bands from them.
    ///
    /// Returns the `RopeCols` a chunk proving the rotation needs, or `None` when
    /// the shapes do not admit the statement at all -- in which case nothing is
    /// rewired and the run reports the family as absent.
    ///
    /// `bend` moves one *band* value after the derivation and rebuilds the
    /// bands around it: the attention it proves is then honest, the matmuls are
    /// untouched, and the only thing in the chunk that can see it is the band's
    /// own claim against the quantize region's output.  `(which of q/k/v, cell)`.
    pub fn rope_from_mats(&mut self, round: bf16::Round, bend: Option<(usize, usize)>)
                          -> Option<RopeCols> {
        use crate::glue::{Prog, UnKind, reference};
        let (nh, hd, tokens) = (self.ao.nh, self.ao.hd, self.ao.s);
        // stream NN-3: a tiled prefix (`--kv-bind`) is rebuilt like any other
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
        if !self.bands.iter().any(|b| b.corner) {
            return off("no corner band");
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
        // NN-2: and the regions have to *tile* the shape, not only fit its
        // widths.  A rotation region's rows are `tokens * heads` and a quantize
        // region's are `q_layout`'s, and both must be an exact power of two --
        // `GluePlan::new` and `q_layout` assert it.  Stream II's synthetic gate
        // shape is `nh = nkv = 3` at `hd = 32`, so `nq = nk = nv = 96` passes
        // the width test above and `tokens * nh = 96` fails the tiling one:
        // before this the family turned on where nobody meant it to and the run
        // panicked.  A shape the regions cannot tile leaves the family off and
        // reports `nn_open`, which is what the width test already does.
        if !(tokens * nh).is_power_of_two() || !(tokens * nkv).is_power_of_two() {
            return off(&format!("the rotation's rows do not tile: tokens={tokens} nh={nh} nkv={nkv}"));
        }
        if !crate::rope::q_tiles(nq, hd, tokens) || !crate::rope::q_tiles(nk, hd, tokens) {
            return off("a quantize region cannot tile this shape");
        }
        let b0 = &self.bands[0];
        let (q_bits, k_bits) = (b0.q_bits, b0.k_bits);
        let (qmax_q, qmax_k) = ((1i64 << (q_bits - 1)) - 1, (1i64 << (k_bits - 1)) - 1);
        // the model's own angles.  A synthetic layer has no model, so it takes
        // Llama's default `inv_freq` at theta = 10000 and this block's own
        // offset; a real dump carries both and never reaches here.
        // a synthetic layer has no model, so it takes Llama's default branch at
        // theta = 10000 and this block's own offset; a real dump carries both
        let (theta, kind) = (10000.0f64, "default".to_string());
        let inv_freq = crate::rope::inv_freq_of(hd, theta, &kind, &[]);
        let pos0 = self.block.unwrap_or(0) * tokens;
        let sp = |m: &Mat| crate::dequant::Spec {
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
        // the rotation, through the regions' own programs: one row per
        // (token, head), `hd` leaves, `cos`/`sgnsin` preprocessed
        let rotate = |x: &[u16], heads: usize| -> Vec<u16> {
            let rows = tokens * heads;
            let mut pre = vec![0u16; rows * 2 * hd];
            pre.par_chunks_mut(2 * hd).enumerate().for_each(|(j, row)| {
                row.copy_from_slice(&crate::rope::pre_row(&inv_freq, hd, pos0 + j / heads));
            });
            let prod = reference(&[x.to_vec()], &pre, rows, &Prog::rope_mul(hd), round);
            let out = reference(&[prod[0].clone()], &[], rows, &Prog::rope_add(hd), round);
            out[0].clone()
        };
        let qrot = rotate(&deq(self, "q", nq), nh);
        let krot = rotate(&deq(self, "k", nk), nkv);
        // Stream SS: a folded `v` has no bf16 stream to quantize -- the fold's
        // one rounding *is* the quantize -- so the region below is not built for
        // it and `vv` comes off `fold_out` directly.  `deq` would panic here
        // (`deq_ref` answers for `pow2` and `chain` only), which is why a
        // `--layout zfold` shape could not carry the wiring family at all.
        let vmi = self.mats.iter().position(|m| m.name == "v").unwrap();
        let v_fold = self.mats[vmi].variant == DeqVariant::Fold;
        let vdeq = if v_fold { vec![] } else { deq(self, "v", nv) };
        // the scales.  B1 calibrates them offline; a synthetic layer picks the
        // smallest power of two that leaves every value inside the domain, so
        // the desk's witness is complete (`quantize`'s clamp has no honest
        // witness in this region -- the table has no row past 127).
        let scale = |x: &[u16], n: usize, qmax: i64| -> Vec<i32> {
            (0..n)
                .map(|c| {
                    let col: Vec<u16> = (0..tokens).map(|t| x[t * n + c]).collect();
                    crate::rope::pow2_scale(&col, qmax) as i32
                })
                .collect()
        };
        // Stream NN-3: `B7_NN_CLAMP=n` calibrates every scale `n` exponents
        // *below* the smallest one that fits, so a synthetic layer clamps where
        // B1 does and the desk can prove a chunk that carries the clamp.  A
        // witness-construction switch like `--synth-glue`: at 0 -- the default,
        // and every recorded digest -- nothing here moves.
        let shift = std::env::var("B7_NN_CLAMP").ok()
            .and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
        if shift != 0 {
            eprintln!("layer: B7_NN_CLAMP={shift} -- the q/k/v scales are {shift} \
exponent(s) small, so B1's clamp fires");
        }
        let down = |v: Vec<i32>| -> Vec<i32> { v.into_iter().map(|x| x - shift as i32).collect() };
        let jq = down(scale(&qrot, nq, qmax_q));
        let jk_kv = down(scale(&krot, nk, qmax_k));
        // stream SS: the fold's `s_v` is the witness's own `jv2` and is not
        // calibrated off a bf16 column, so `B7_NN_CLAMP` does not move it --
        // the clamp a folded site takes is the fold's own `qc = min(qm, 127)`.
        let jv_kv = if v_fold {
            self.mats[vmi].jv2.clone()
        } else {
            down(scale(&vdeq, nv, crate::rope::qmax_of(crate::rope::V_BITS)))
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
        // the two sites disagreed and no desk shape could state the tie.  The
        // attention output is rebuilt below, so this is the value it takes.
        self.ao.jv = jv.clone();
        // the bands' own columns, on the query-head grid: the kv head's values
        // replicated across its `rep` query heads, exactly as
        // `attention`'s `repeat_interleave` does
        let quant = |x: &[u16], n: usize, jj: &[i32], qmax: i64| -> Vec<i32> {
            let mut out = vec![0i32; tokens * nh * hd];
            out.par_chunks_mut(nh * hd).enumerate().for_each(|(t, row)| {
                for (i, slot) in row.iter_mut().enumerate() {
                    let (h, ch) = (i / hd, i % hd);
                    let c = (h / (nh / (n / hd))) * hd + ch;
                    *slot = crate::rope::quant_ref(x[t * n + c], jj[c] as i64, qmax) as i32;
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
        // stream NN (NN-2): `which == 3` is the *prefix* band's own copy of Q --
        // the grid above is untouched, so the corner keeps the honest queries
        // and only the prefix's copy moves.  `rebuild_bands` applies it inside
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
                eprintln!("layer: band {} cell {i} bent (a rejection is the pass)",
                          ["Q", "K", "V"][which % 3]);
            }
        }
        self.rebuild_bands(&qq, &kk, &vv, pre_q_bend);
        Some(RopeCols { nkv, pos0, theta, rope_type: kind, rope_scale: vec![],
                        inv_ref: Some(inv_freq), jq, jk, jv })
    }

    /// Stream NN: rebuild every band from Q, K and V on the query-head grid.
    ///
    /// The corner's keys *are* this block's tokens, so its K and V are what the
    /// layer's own `k` and `v` matmuls produced; a prefix band's keys are the
    /// cache and keep the ones they had, because this chunk did not produce
    /// them (they are bound across blocks by root equality instead -- stream
    /// AA/EE).  Every band of a group takes the new queries, and the group's two
    /// halves share one row maximum and one master exponent, exactly as
    /// `Attn::synthetic_pair` builds them.
    ///
    /// Stream NN-3: a tiled prefix (`--kv-bind`, `nsub > 1`) is rebuilt too.
    /// Its sub-bands are `nsub` windows of one pair's prefix, so each keeps its
    /// own slice of the cache and takes the new queries, the group's master
    /// exponent is the joint maximum over every sub-band **and** the corner, and
    /// the pair's total denominator -- which the group's per-token region
    /// commits on its first sub-band -- is the sum of all of their `zz`.  Before
    /// this the function asserted `nsub <= 1` and the rotation simply turned
    /// itself off under `--kv-bind`.
    ///
    /// `pre_q_bend` is stream NN-2's control: one cell of the **first group's
    /// prefix band's own** copy of Q, moved before that band's attention is
    /// built, so the corner keeps the honest queries and the chunk is still an
    /// honest attention over the cache -- and the only thing that can see it is
    /// `nn_q_ok`, the equality between the two halves' Q folds.
    pub fn rebuild_bands(&mut self, qq: &[i32], kk: &[i32], vv: &[i32],
                         pre_q_bend: Option<usize>) {
        let (nh, hd, s) = (self.ao.nh, self.ao.hd, self.ao.s);
        let nsub = self.nsub.max(1);
        let n_pre = self.bands.iter().filter(|b| !b.corner).count();
        let groups = self.bands.len() - n_pre;
        assert!(groups > 0, "the q/k/v rewiring needs the causal corner");
        assert!(n_pre == 0 || n_pre == groups * nsub,
                "{nsub} prefix sub-bands per group, or none: {n_pre} prefix bands, \
{groups} groups");
        // the bands of one group, prefix sub-bands first and the corner last --
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
        // so both passes see the same bent cell and the band's own attention is
        // rebuilt around it.
        if let Some(at) = pre_q_bend {
            if n_pre == 0 {
                eprintln!("layer: prefix-band Q bend asked for on a chunk with no prefix \
band -- nothing bent, and this control measures nothing");
            } else {
                eprintln!("layer: prefix band (group 0) Q cell {at} bent, the corner's \
unmoved (a rejection is the pass)");
            }
        }
        let q_bits0 = self.bands[0].q_bits;
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
                let a = &self.bands[b];
                let sh = a.shape;
                let off = self.head_offsets[b];
                let q = take_q(qq, off, sh.insts, sh.s, !a.corner && g == 0 && n == 0);
                let (k, v) = if a.corner {
                    (take(kk, off, sh.insts, sh.c), take(vv, off, sh.insts, sh.c))
                } else {
                    (a.kk.clone(), a.vv.clone())
                };
                row.push(Attn::from_qkv(sh, a.variant, q, k, v, a.q_bits, a.k_bits, a.nmax,
                                        a.log_table, None, None, a.source.clone(), a.corner,
                                        None));
            }
            probe.push(row);
        }
        let frac = probe.iter().flatten().map(|b| b.frac).fold(f64::MIN, f64::max).ceil();
        let d_bits = probe.iter().flatten().map(|b| b.d_bits()).max().unwrap() + 1;
        let log_table = d_bits.clamp(4, 22);
        // pass 2: the joint row maximum per group -- over every sub-band and the
        // corner -- then all of them under it
        for g in 0..groups {
            let idx = gidx(g);
            let mm: Vec<i64> = probe[g].iter().skip(1)
                .fold(probe[g][0].mm.clone(), |acc, b| {
                    acc.iter().zip(&b.mm).map(|(x, y)| *x.max(y)).collect()
                });
            for (n, b) in idx.iter().copied().enumerate() {
                let a = &self.bands[b];
                let sh = a.shape;
                let off = self.head_offsets[b];
                let q = take_q(qq, off, sh.insts, sh.s, !a.corner && g == 0 && n == 0);
                let (k, v) = if a.corner {
                    (take(kk, off, sh.insts, sh.c), take(vv, off, sh.insts, sh.c))
                } else {
                    (a.kk.clone(), a.vv.clone())
                };
                let (variant, qb, kb, nmax, src) =
                    (a.variant, a.q_bits, a.k_bits, a.nmax, a.source.clone());
                let corner = a.corner;
                self.bands[b] = Attn::from_qkv(sh, variant, q, k, v, qb, kb, nmax, log_table,
                                               Some(vec![frac; sh.insts]), Some(frac), src,
                                               corner, Some(mm.clone()));
            }
            // the group's denominator: the pair's total, which every sub-band's
            // `zz` is a partial sum of.  It lives on the band that carries the
            // per-token region -- the first sub-band when there is a prefix, and
            // the corner itself when there is none.
            let total: Vec<i64> = idx.iter().skip(1)
                .fold(self.bands[idx[0]].zz.clone(), |acc, b| {
                    acc.iter().zip(&self.bands[*b].zz).map(|(x, y)| x + y).collect()
                });
            self.bands[idx[0]].zz_total = Some(total);
        }
        // the attention output follows the bands: its accumulator and
        // denominator *are* their sums, and the per-token output scale is the
        // smallest power of two that leaves the quotient inside int8
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for bi in 0..self.bands.len() {
            let b = &self.bands[bi];
            for i in 0..self.band_sizes[bi] {
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
        let r = self.rope.as_ref().expect("the quantize regions need the rope columns");
        let src = match name {
            "q" => &r.jq,
            "k" => &r.jk,
            // Stream NN-3: `s_v` is one scale used at two sites, and this is
            // where the two read **one array**.  The attention output rescales
            // by `jv - jo` off `AttnOut::jv`; the `v` quantize divides by the
            // same exponent, so it takes it from there rather than from the
            // rotation's own copy of `jv.i32`.  A dump loads both from that one
            // file (`Layer::load` and `RopeCols::load`), and `family_cells.sv_tie`
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
    pub fn synthetic(s: usize, c: usize, hd: usize, band_sizes: &[usize], variant: Variant,
                     cfg: &Src, n_in: usize, n_out: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        // Stream AA (E11): a tiled synthetic layer is the *same* pair, with its
        // prefix cut into `C / S` windows by `Attn::key_slice`, so the windows'
        // denominators really do sum to the pair's total.  What a synthetic
        // layer still cannot do is two *adjacent chunks* over one shared key
        // stream, which is what a chain probe needs; that generator is the
        // follow-on this stream's record names, and until it exists a chain is
        // proved on a real dump.
        assert!(!cfg.tile || cfg.corner, "`--kv-bind` needs `--corner`");
        assert!(!cfg.tile || c % s == 0,
                "`--kv-bind` tiles the prefix in blocks of S: C={c} is not a multiple of S={s}");
        let s_nsub = if cfg.tile { c / s } else { 1 };
        let nh: usize = band_sizes.iter().sum();
        // pass 1: learn the data-derived exponents; pass 2: rebuild every band
        // against one shared master exponent and one shared table size
        let mut probe = vec![];
        let mut head_offsets = vec![];
        let mut off = 0usize;
        // B10 step 3: with the causal corner, a band is a *pair* -- the prefix
        // rectangle and the corner over the same heads, sharing the queries and one
        // row maximum.  Both halves' shifts go into the same master exponent, so
        // both are probed.
        let mut probe_corner = vec![];
        for insts in band_sizes {
            let sh = Shape { s, c, hd, insts: *insts };
            if cfg.corner {
                let (p, cc) = Attn::synthetic_pair(sh, variant, cfg.q_bits, cfg.k_bits,
                                                   cfg.nmax, cfg.log_table,
                                                   seed ^ (off as u64 + 1), None);
                probe.push(p);
                probe_corner.push(cc);
            } else {
                probe.push(Attn::synthetic(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax,
                                           cfg.log_table, seed ^ (off as u64 + 1)));
            }
            // stream AA: one entry per *band*, and a tiled prefix is `s_nsub`
            // bands over the same heads
            for _ in 0..s_nsub {
                head_offsets.push(off);
            }
            off += insts;
        }
        if cfg.corner {
            let mut off = 0usize;
            for insts in band_sizes {
                head_offsets.push(off);
                off += insts;
            }
        }
        probe.extend(probe_corner);
        let frac = probe.iter().map(|b| b.frac).fold(f64::MIN, f64::max).ceil();
        let d_bits = probe.iter().map(|b| b.d_bits()).max().unwrap() + 1;
        let log_table = if cfg.log_table > 0 { cfg.log_table } else { d_bits.clamp(4, 22) };
        let mut bands = vec![];
        let mut corner_bands = vec![];
        let mut sizes = vec![];
        let mut off = 0usize;
        for insts in band_sizes.iter() {
            let sh = Shape { s, c, hd, insts: *insts };
            if cfg.corner {
                // the same seeds the probe used, so the master exponent it derived
                // is an upper bound on the data these bands actually carry
                let (p, cc) = Attn::synthetic_pair(sh, variant, cfg.q_bits, cfg.k_bits,
                                                   cfg.nmax, log_table,
                                                   seed ^ (off as u64 + 1), Some(frac));
                // Stream AA (E11): the tiled prefix, as `nsub` windows of the
                // pair's own prefix rather than as `nsub` unrelated bands -- so
                // the sub-bands' `zz` really do sum to the pair's total and the
                // synthetic layer is a layer and not a shape rehearsal.
                if cfg.tile {
                    let total = p.zz_total.clone();
                    for k in 0..s_nsub {
                        bands.push(p.key_slice(k * s, s));
                    }
                    // the group's per-token region lives on its first sub-band,
                    // and the denominator it commits is the pair's total -- the
                    // slices' own `zz` are the partial sums that make it up
                    let first = bands.len() - s_nsub;
                    bands[first].zz_total = total;
                } else {
                    bands.push(p);
                }
                corner_bands.push(cc);
            } else {
                bands.push(Attn::synthetic_shared(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax,
                                                  log_table, seed ^ (off as u64 + 1),
                                                  frac, false));
            }
            for _ in 0..s_nsub {
                sizes.push(*insts);
            }
            off += insts;
        }
        if cfg.corner {
            sizes.extend(band_sizes.iter().copied());
            bands.extend(corner_bands);
        }
        let band_sizes_all = sizes;
        let mut mats = vec![];
        // matmuls in one walk group share one committed input, as the real layer's do
        let mut group_qx: std::collections::HashMap<&str, Vec<i32>> = Default::default();
        for (name, var) in mat_variants(&cfg.layout) {
            let (ni, no) = if name == "down" { (n_out, n_in) } else { (n_in, n_out) };
            // B7b (README 26): `B7_SYNTH_DIMS=q:3584x3584,k:3584x512,...` sets a
            // matmul's (n_in, n_out) by name; unnamed matmuls keep the defaults.
            let (ni, no) = std::env::var("B7_SYNTH_DIMS")
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
            let gname = WALK_GROUPS.iter().find(|(_, ms)| ms.contains(&name)).unwrap().0;
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
        }
        // attnout from the bands, with scales that keep the quotient inside int8
        // B10 step 3: `+=`, not `=`.  With the corner riding as a second set of
        // bands over the same heads, a head's denominator and A.V output are the
        // sum of its two halves -- which is what the attention output *is*.
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
        for (bi, b) in bands.iter().enumerate() {
            for i in 0..band_sizes_all[bi] {
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
        Self { bands, band_sizes: band_sizes_all, head_offsets, nsub: s_nsub, mats,
               ao: AttnOut { nh, s, hd, acc, den, jo, jv },
               source: format!("synthetic layer (seed={seed})"), glue: None,
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
    /// nothing ties a matmul's input to anything; with the requantize seams in,
    /// an honest proof needs `qX` to actually be the requantize of the
    /// producer's output under the token's own scale, or the seam's own scale
    /// lookup goes off-table before any tie is reached.  So this runs the
    /// wiring family's own reference forward -- `glue::reference` on the very
    /// programs the regions prove, so the two cannot disagree -- and rebuilds
    /// each group's input, scale and accumulator from it.
    ///
    /// `o`'s input is deliberately left alone: `AttnOutEval` is that matmul's
    /// requantize and its claim is stream JJ's, so there is nothing here for a
    /// consistent witness to satisfy yet.
    pub fn synth_glue(&mut self, seed: u64, a: usize, round: bf16::Round) {
        self.synth_family(seed, a, round, 0, None, None);
    }

    /// Stream NN, the band side's control.  The same layer with **one** band
    /// cell moved and the bands rebuilt around it: the attention is an honest
    /// attention over the bent K (or Q, or V), the matmuls are untouched, and
    /// the only thing that can see it is that band's claim against the
    /// quantize region's output.  `which` picks Q, K or V.
    pub fn synth_glue_band_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
                                which: usize, idx: usize) {
        self.synth_family(seed, a, round, 0, None, Some((which, idx)));
    }

    /// Stream NN (NN-2), the prefix band's control.  The same layer with **one**
    /// cell of the first group's *prefix* band's own copy of Q moved, and that
    /// band's whole attention rebuilt around it -- the corner's queries are
    /// untouched, so the chunk proves an honest attention over the cache with
    /// queries the `q` matmul did not produce, and the only thing that can see
    /// it is `nn_q_ok`.
    pub fn synth_glue_prefix_q_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
                                    idx: usize) {
        self.synth_family(seed, a, round, 0, None, Some((3, idx)));
    }

    /// Stream II (E11), the `o` seam's control.  The same layer, with **one**
    /// `o` activation moved to another legal value and the whole layer rebuilt
    /// around it: the `o` matmul's accumulator follows, and so do `o_out`,
    /// `xmid`, norm2, the gate/up inputs, the SwiGLU and `down`.  So every
    /// constraint and every other equality in the chunk still holds -- this is
    /// an honest proof of a layer whose attention output is not the one the
    /// attnout region requantized, and only `ao_q_ok` can see it.
    ///
    /// Returns `None` when this shape's `o` walk is not `nh * hd` wide, which
    /// is the shape on which the seam is not statable at all.
    pub fn synth_glue_bent(&mut self, seed: u64, a: usize, round: bf16::Round,
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
        let vocab = usize::from(self.glue.as_ref().is_some_and(|g| g.head));
        self.synth_family(seed, a, round, vocab, Some(at), None);
        Some(at)
    }

    /// Stream II (E11): the *head* chunk's family on a synthetic layer -- an
    /// `lm_head` matmul of `vocab` columns whose committed input is the final
    /// norm's output, so the desk can run the head's norm tail and its
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
        self.synth_family(seed, a, round, vocab, None, None);
    }

    fn synth_family(&mut self, seed: u64, a: usize, round: bf16::Round, head_vocab: usize,
                    o_bend: Option<usize>, rope_bend: Option<(usize, usize)>) {
        use crate::glue::{Kind, Prog, Tiling, UnKind, canon, reference};
        let head = head_vocab > 0;
        let tokens = self.ao.s;
        let d = self.mats.iter().find(|m| m.name == "q")
            .expect("--synth-glue needs a q matmul").n_in;
        let dff = self.mats.iter().find(|m| m.name == "gate")
            .expect("--synth-glue needs a gate matmul").n_out;
        let gc = if head {
            GlueCols::synthetic_head(tokens, d, seed ^ 0x9e37_79b9)
        } else {
            GlueCols::synthetic(tokens, d, dff, seed ^ 0x9e37_79b9)
        };
        let (eps, inv_d) = (gc.eps, gc.inv_d);
        let un_r = UnKind::NormR { inv_d, eps };
        let t1 = Tiling::new(tokens, d, a);
        let tff = Tiling::new(tokens, dff, a);

        // `h = (x r) w` for one norm, through the same three programs the
        // regions carry: the squares' two levels, then the tail.
        let tail = |stream: &[u16], wcol: &[u16], t: &Tiling| -> Vec<u16> {
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
                                &Prog::norm_tail(t.nreal, un_r), round);
            t.unlay(&out[0])
        };

        // one walk group's input, scale and accumulators, from a bf16 column
        let mut wire = |me: &mut Self, names: &[&str], x: &[u16], n: usize| {
            let v = match me.mats.iter().find(|m| names.contains(&m.name.as_str())) {
                Some(m) if m.variant == DeqVariant::Pow2 => crate::requant::SVariant::Pow2,
                _ => crate::requant::SVariant::Bf16,
            };
            let (qx, jx, sx) = crate::requant::requant_ref(x, tokens, n, v);
            for name in names {
                let Some(mi) = me.mats.iter().position(|m| m.name == *name) else { continue };
                let m = &mut me.mats[mi];
                assert_eq!(m.n_in, n, "{name}: the walk's input width is not its producer's");
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
        // the dequant reference of one matmul's output, under the spec the
        // plan builds for it -- `deq_ref` reads only `variant` and `round`
        let sp = |m: &Mat| crate::dequant::Spec {
            variant: m.variant, round, kmax: 0, per_row: 1, limb_bits: (16, 14),
            q_bits: 0, rem_bits: 0, fold_alpha: 0,
        };
        let out_col = |me: &Self, name: &str, w: usize| -> Vec<u16> {
            let mi = me.mats.iter().position(|m| m.name == name)
                .unwrap_or_else(|| panic!("--synth-glue needs the {name} matmul"));
            let sp = sp(&me.mats[mi]);
            (0..tokens)
                .flat_map(|t| (0..w).map(move |c| (t, c)))
                .map(|(t, c)| deq_ref(&me.mats[mi], &sp, t, c))
                .collect()
        };


        let h1 = tail(&gc.xin, &gc.w1, &t1);
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
        // ---- stream NN: the bands are the q/k/v matmuls' own outputs --------
        //
        // A synthetic layer's bands were three unrelated draws, so no shape it
        // could prove carried a positionwise q/k/v tie
        // (`briefs/jj-layerwire-out.md` section 11.4 item 2).  Here the layer's
        // own `q`/`k`/`v` dequant outputs are rotated and quantized -- through
        // `glue::reference` on the very programs the regions prove, so the two
        // cannot disagree -- and the bands are rebuilt from the result.
        //
        // Wired exactly when the widths line up, the way `o` is: `q` must be
        // `nh hd` channels wide, `k` and `v` equally wide and a whole number of
        // heads, and `nh` a multiple of that number.  A shape where they do not
        // leaves the family off and the run reports it, rather than proving
        // something narrower quietly.
        if !head {
            self.rope = self.rope_from_mats(round, rope_bend);
        }
        // the `o` matmul's committed input is the attention output requantized,
        // and the attention output has just moved -- so this runs after the
        // rebuild and not before it
        // Stream II (E11), at the JJ merge: the `o` matmul's committed input is
        // the attention output requantized, and `AttnOutEval` is that
        // requantize -- so on a synthetic layer, where the `o` activations are
        // otherwise drawn at random, the seam can only be honest if the witness
        // says so.  Wired exactly when the widths line up: the attention output
        // is `nh * hd` channels wide and the `o` walk reads `n_in` of them.  A
        // shape where they disagree leaves `o` untied and `ao_q_ok` reports it
        // as unstatable rather than passing quietly.
        let o_wired = {
            let oi = self.mats.iter().position(|m| m.name == "o")
                .expect("--synth-glue needs the o matmul");
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
                    // interior value are inside the walk's range window
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
            self.glue = Some(gc);
            return;
        }
        // Stream JJ's dequant output ties: `o_out` is the `o` region's own
        // dequant reference, so an honest run ties and a bent copy does not.
        // `o`'s input is not rewired (its requantize is `AttnOutEval`'s), so
        // its accumulator is final by here.
        gc.o_out = out_col(self, "o", d);
        let resid = reference(&[t1.lay(&gc.xin), t1.lay(&gc.o_out)], &[], t1.rows1(),
                              &Prog::norm_resid(t1.nreal, t1.b), round);
        let xmid = t1.unlay(&resid[1]);
        let h2 = tail(&xmid, &gc.w2, &t1);
        wire(self, &["gate", "up"], &h2, d);

        // the SwiGLU reads the gate and up regions' *dequantized* outputs, so
        // its input is a function of the accumulators `wire` just rebuilt
        let gi = self.mats.iter().position(|m| m.name == "gate").unwrap();
        let ui = self.mats.iter().position(|m| m.name == "up").unwrap();
        let (gs, us) = (sp(&self.mats[gi]), sp(&self.mats[ui]));
        let mut gate_out = vec![0u16; tokens * dff];
        let mut up_out = vec![0u16; tokens * dff];
        for t in 0..tokens {
            for c in 0..dff {
                gate_out[t * dff + c] = deq_ref(&self.mats[gi], &gs, t, c);
                up_out[t * dff + c] = deq_ref(&self.mats[ui], &us, t, c);
            }
        }
        let sw = reference(&[tff.lay(&gate_out), tff.lay(&up_out)], &[], tff.rows1(),
                           &Prog::pairs(tff.nreal, Kind::Mul, UnKind::Silu, UnKind::None),
                           round);
        let x_down = tff.unlay(&sw[0]);
        wire(self, &["down"], &x_down, dff);
        // and the second output tie, after `down`'s accumulator is final
        gc.d_out = out_col(self, "down", d);

        self.glue = Some(gc);
    }
}

// -------------------------------------------------------------- region layouts --

/// SIMD lane floor, as `layout.rs`'s `MIN_LOG_ROWS`.
pub const MIN_MAT_LOG_ROWS: u32 = 4;

/// The matmul dequant region's tiling: `per_row | n_out`, with `tokens * chunks`
/// a power of two exactly (`choose`) or padded up to one (`choose_padded`).
#[derive(Clone, Copy, Debug)]
pub struct MatLayout {
    pub per_row: usize,
    pub chunks: usize,
    pub log_rows: u32,
    /// B10 step 4: inert lead rows, as `AoLayout` and `Grid` have always had.
    /// Zero for every region that existed before the `lm_head` chunk, because a
    /// layer matmul's output width is a power of two and `tokens * chunks` lands
    /// on one exactly.  A *vocabulary* does not: 8B's 128,256 is `2^8 x 501` and
    /// Qwen's 151,936 is `2^7 x 1187`.
    pub lead: usize,
}

impl MatLayout {
    /// The padded chooser: a width that divides `n_out` and is near `target`,
    /// with the row count rounded *up* to a power of two and the slack carried as
    /// inert lead rows.
    ///
    /// The policy is not just "pad": how much padding costs depends entirely on
    /// the odd part of `tokens * n_out`.  At 8B that odd part is 501 against a
    /// 512 boundary, so any width pads **2.1%**; at Qwen it is 1187 against 2048,
    /// so any power-of-two width pads **42%** and the sane choice is the exact
    /// width 1187 instead.  So a candidate is rejected when its lead fraction
    /// exceeds `MAX_LEAD_FRAC`, and the search widens until an exact tiling
    /// appears -- which one always does at `pr = odd_part(tokens * n_out)`.
    pub fn choose_padded(n_out: usize, tokens: usize, target: usize) -> Self {
        const MAX_LEAD_FRAC: f64 = 0.25;
        let mk = |pr: usize| -> Option<Self> {
            if pr == 0 || n_out % pr != 0 {
                return None;
            }
            let chunks = n_out / pr;
            let active = tokens * chunks;
            let log_rows = (usize::BITS - (active - 1).leading_zeros()).max(MIN_MAT_LOG_ROWS);
            let rows = 1usize << log_rows;
            Some(Self { per_row: pr, chunks, log_rows, lead: rows - active })
        };
        let mut best: Option<Self> = None;
        // widen the search until something acceptable turns up; `n_out` itself is
        // always a legal width (chunks = 1, rows = tokens), so this terminates
        for pr in 1..=n_out {
            let Some(l) = mk(pr) else { continue };
            if (l.lead as f64) / ((1usize << l.log_rows) as f64) > MAX_LEAD_FRAC {
                continue;
            }
            let score = (pr as isize - target as isize).abs();
            let better = best.as_ref().map(|b| {
                let bs = (b.per_row as isize - target as isize).abs();
                score < bs
            }).unwrap_or(true);
            if better {
                best = Some(l);
            }
            // once past the target, a wider row is only ever a worse score unless
            // it is the exact tiling the odd part forces, so keep scanning but do
            // not let the loop run to `n_out` for no reason
            if pr > target && best.as_ref().is_some_and(|b| b.lead == 0) {
                break;
            }
        }
        best.expect("no layout, padded or otherwise")
    }

    /// `row -> (token, chunk)`, or `None` for an inert lead row.
    pub fn at_opt(&self, tokens: usize, j: usize) -> Option<(usize, usize)> {
        if j < self.lead {
            return None;
        }
        let ja = j - self.lead;
        Some((tokens - 1 - ja / self.chunks, self.chunks - 1 - ja % self.chunks))
    }

    /// Exact when one exists, padded otherwise.  Every shape the project proved
    /// before the `lm_head` chunk takes the exact branch, so this is a no-op for
    /// them; a vocabulary takes the other one.
    pub fn choose_or_padded(n_out: usize, tokens: usize, target: usize) -> Self {
        Self::try_choose(n_out, tokens, target)
            .unwrap_or_else(|| Self::choose_padded(n_out, tokens, target))
    }

    pub fn choose(n_out: usize, tokens: usize, target: usize) -> Self {
        Self::try_choose(n_out, tokens, target).expect("no exact tiling exists")
    }

    fn try_choose(n_out: usize, tokens: usize, target: usize) -> Option<Self> {
        // the largest divisor of n_out at most `target` whose row count is a
        // power of two (tokens is one in every real shape here)
        let mut best = None;
        for pr in 1..=n_out.min(4 * target.max(1)) {
            if n_out % pr != 0 {
                continue;
            }
            let chunks = n_out / pr;
            let rows = tokens * chunks;
            if !rows.is_power_of_two() || rows < 16 {
                continue;
            }
            let score = (pr as isize - target as isize).abs();
            if best.map(|(s, _)| score < s).unwrap_or(true) {
                best = Some((score, Self { per_row: pr, chunks, log_rows: rows.ilog2(),
                                           lead: 0 }));
            }
        }
        best.map(|(_, l)| l)
    }
    /// row -> (token, chunk); tokens and chunks descend
    pub fn at(&self, tokens: usize, j: usize) -> (usize, usize) {
        (tokens - 1 - j / self.chunks, self.chunks - 1 - j % self.chunks)
    }
}

/// The attnout region: rows = (head, token, chunk) descending, inert lead first.
#[derive(Clone, Copy, Debug)]
pub struct AoLayout {
    pub per_row: usize,
    pub chunks: usize,
    pub log_rows: u32,
    pub lead: usize,
}

impl AoLayout {
    pub fn choose(nh: usize, s: usize, hd: usize, target: usize) -> Self {
        let mut pr = target.min(hd).max(1);
        while hd % pr != 0 {
            pr -= 1;
        }
        let chunks = hd / pr;
        let active = nh * s * chunks;
        let log_rows = (usize::BITS - (active - 1).leading_zeros()).max(4);
        Self { per_row: pr, chunks, log_rows, lead: (1usize << log_rows) - active }
    }
    /// row -> (head, token, chunk), or None for an inert lead row
    pub fn at(&self, nh: usize, s: usize, j: usize) -> Option<(usize, usize, usize)> {
        if j < self.lead {
            return None;
        }
        let ja = j - self.lead;
        let per_head = s * self.chunks;
        let h = nh - 1 - ja / per_head;
        let rest = ja % per_head;
        Some((h, s - 1 - rest / self.chunks, self.chunks - 1 - rest % self.chunks))
    }
}

/// One walk region's layout: `Grid` over input channels.
#[derive(Clone, Copy, Debug)]
pub struct WalkLayout {
    pub g: Grid,
    pub pk: usize,
    pub pq: usize,
}

impl WalkLayout {
    pub fn choose(n_in: usize, n_out_max: usize, tokens: usize, p_target: usize) -> Self {
        // pick the chunk count whose weights-per-row is closest to the target,
        // preferring fewer padded rows on ties
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
        best.expect("no walk layout").2
    }
}

// -------------------------------------------------------- shared table registry --

/// Every shared table's contents and multiplicities, accumulated across regions.
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
    /// Stream J: size the layer's master exp table by the largest key the bands
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
    /// Returns `(before, after)` in table bits, and leaves `off_table` on each band
    /// recounted against the new size (zero whenever `bits >= key_bits`).
    pub fn set_master_table(&mut self, bits: u32) -> (u32, u32) {
        let _s = witread::scope(Site::Tighten);
        let before = self.bands.iter().map(|b| b.log_table).max().unwrap_or(0);
        let after = bits.clamp(1, before);
        for b in self.bands.iter_mut() {
            b.retable(after);
        }
        (before, after)
    }

    /// The largest exp-table key any band of this layer looks up, in bits.
    pub fn key_bits(&self) -> u32 {
        let _s = witread::scope(Site::Tighten);
        self.bands.iter().map(|b| b.key_bits()).max().unwrap_or(1)
    }

    /// Rebuild the attnout witness from the (possibly corrupted) bands, keeping
    /// the scales: the probes mutate a band and need `ao` to follow it.
    pub fn rebuild_ao(&mut self) {
        let (nh, s, hd) = (self.ao.nh, self.ao.s, self.ao.hd);
        // zero first, then accumulate: a head's totals are the sum over every band
        // that covers it, which with B10 step 3's corner is two of them
        self.ao.den.iter_mut().for_each(|v| *v = 0);
        self.ao.acc.iter_mut().for_each(|v| *v = 0);
        for (bi, b) in self.bands.iter().enumerate() {
            for i in 0..self.band_sizes[bi] {
                let h = self.head_offsets[bi] + i;
                for t in 0..s {
                    self.ao.den[h * s + t] += b.zz[i * s + t];
                    for j in 0..hd {
                        self.ao.acc[(h * s + t) * hd + j] += b.out[(i * s + t) * hd + j];
                    }
                }
            }
        }
        let _ = nh;
    }

    /// B10 step 3's probe on the composed proof: one token attends to a key the
    /// causal mask removes.
    ///
    /// The mask is preprocessed, so the only shape the attack can take is to
    /// *claim* a denominator and an output that count a masked key: the corner
    /// band's numerator gets a legal table output, and `Z`, `A.V` and the attnout
    /// witness are rebuilt from it.  The AIR reads neither -- its numerator fold
    /// and its exp lookup drop the slot -- so the corner's `sn` no longer joins
    /// the prefix's to make the denominator the per-token region claims.
    pub fn unmask_corner(&mut self, i: usize) {
        let n_pre = self.bands.iter().filter(|b| !b.corner).count();
        let ci = self.bands.iter().position(|b| b.corner).expect("no corner band");
        let nmax = self.bands[ci].nmax as i64;
        let sh = self.bands[ci].shape;
        let masked: Vec<usize> = (0..sh.insts * sh.s * sh.c)
            .filter(|g| !self.bands[ci].visible(*g))
            .collect();
        assert!(!masked.is_empty(), "the corner band has no masked entries");
        let g = masked[i % masked.len()];
        self.bands[ci].num[g] = nmax;
        // rebuild that band's denominator and A.V output from the changed numerator
        let (ni, ns, nc, nhd) = (sh.insts, sh.s, sh.c, sh.hd);
        let (inst, t, c) = (g / (ns * nc), (g / nc) % ns, g % nc);
        self.bands[ci].zz[inst * ns + t] += nmax;
        for j in 0..nhd {
            let v = self.bands[ci].vv[(inst * nc + c) * nhd + j] as i64;
            self.bands[ci].out[(inst * ns + t) * nhd + j] += nmax * v;
        }
        let _ = ni;
        // and the total the paired prefix band's per-token region commits
        let pi = ci - n_pre;
        let add = self.bands[ci].zz[inst * ns + t];
        let own = self.bands[pi].zz[inst * ns + t];
        if let Some(tot) = self.bands[pi].zz_total.as_mut() {
            tot[inst * ns + t] = own + add;
        }
        self.rebuild_ao();
    }

    /// B4 section 3's probe on the composed proof: flip one row of the shared exp
    /// table and rebuild every band's numerators, denominators and outputs to
    /// match.  Internally consistent everywhere; only the preprocessed root
    /// differs.
    pub fn corrupt_exp_table(&mut self, i: usize) {
        for b in self.bands.iter_mut() {
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

// -------------------------------------------------- matmul dequant region traces --

/// A region's committed and preprocessed columns, and what its interaction
/// builder needs to remember.
pub struct MatRegion {
    pub lay: MatLayout,
    pub spec: Spec,
    pub map: Map,
    /// preprocessed: the per-channel scale columns (chan cols), then selectors
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// f64-reference mismatches (must be 0)
    pub ref_bad: usize,
    pub n: usize,
    /// B7a milestone 4 (19.6): the same columns, born on the device. `Some`
    /// only under `B7_M4_FILL=1`, where `pre` and `base` are empty and this is
    /// the region.
    pub dev: Option<crate::devfill::DevFill>,
}

impl MatRegion {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev {
            Some(d) => d.base.len(),
            None => self.base.len(),
        };
        n << self.lay.log_rows
    }
}

/// Build one matmul's dequant region: witness columns in the compose layout,
/// multiplicities bumped into the shared tables.  Mirrors `dequant::run`'s
/// witness loop without the tampers (compose probes corrupt the layer witness).
/// Stream SS: `ohead` is `Some(rows per kv head)` on the one region whose output
/// side folds under the *band* grid -- a folded `v`, whose published output is an
/// integer and whose counterparty is the bands' own V (`dequant::OKind::Int`).
/// It changes nothing but which rows the `oin`/`obd` step selectors mark, so no
/// preprocessed column is added and no other region moves.
pub fn mat_region(m: &Mat, spec: &Spec, tables: &mut Tables, oside: bool,
                  ohead: Option<usize>,
                  fold_bend: (Option<usize>, Option<usize>)) -> MatRegion {
    assert!(spec.sound(), "unsound spec for {}", m.name);
    let lay = MatLayout::choose_or_padded(m.n_out, m.tokens, spec.per_row);
    let mut spec = spec.clone();
    spec.per_row = lay.per_row;
    let map = Map::new(&spec);
    let rows = 1usize << lay.log_rows;
    let sticky = spec.round.needs_sticky();
    let f = map.f;
    let n = m.tokens * m.n_out;

    // B7a milestone 4 (19.6): which fill runs. 1 is the device's alone -- the
    // region has no host columns at all and `pre`/`base` stay empty -- and 2
    // builds both and asserts every cell equal before either is used.
    //
    // Stream Z: the fold has no device fill; its region takes the host path
    // whatever the switch says (the corner bands' pattern), and
    // `devpre::note_host_fill` is told in `generate`.
    let m4c = if spec.variant == DeqVariant::Fold {
        0
    } else {
        crate::util::m4b_mode("B7_M4_FILL")
    };
    let m4c_want = m4c == 2;
    if m4c == 1 {
        // B11 stream M: `oside` is the region's, not `with_want`'s. The two
        // parameters sit next to each other and the o-side one was missing
        // entirely, so the head's device fill built 506 preprocessed columns
        // where the host builds 510.
        let dev = crate::devfill::mat_region_dev(m, &spec, &lay, &map, tables, false, oside);
        let ref_bad = dev.ref_bad;
        return MatRegion { lay, spec, map, pre: vec![], base: vec![], ref_bad, n,
                           dev: Some(dev) };
    }
    let m4c_snap = |t: &Tables| -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let r8 = t.ridx(8);
        (t.m_norm.clone(), t.m_frac.clone(), t.m_mul.clone(), t.m_r[r8].clone(),
         t.off_table)
    };
    let m4c_before = m4c_want.then(|| m4c_snap(tables));

    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; map.n_cols];
    // parallel over row blocks: each block fills its slice of every column and
    // accumulates its own table multiplicities, merged serially below
    let block = 2048usize;
    struct Bumps {
        norm: Vec<(usize, u32)>,
        frac: Vec<(usize, u32)>,
        mul: Vec<(usize, u32)>,
        r8: Vec<(i64, u32)>,
        /// stream Z, the fold: the (q, nz) pair table and the width-keyed
        /// range bumps (13, 16 and the fold's own remainder width)
        qnz: Vec<(i64, u32)>,
        rw: Vec<((u32, i64), u32)>,
        ref_bad: usize,
        /// B7a milestone 4 (19.6): the f64 cross-check's own bf16 answer per
        /// (row, slot), so `B7_M4_FILL=2` can compare the one part of the fill
        /// that is not integer arithmetic cell for cell instead of comparing
        /// two counts that are both zero on an honest witness.
        want: Vec<u16>,
    }
    let starts: Vec<usize> = (0..rows).step_by(block).collect();
    let tref = &*tables;
    let out: Vec<(Vec<Vec<M31>>, Bumps)> = starts
        .par_iter()
        .map(|start| {
            let len = block.min(rows - start);
            let mut local: Vec<Vec<M31>> = vec![vec![M31::zero(); len]; map.n_cols];
            let mut bn = std::collections::HashMap::new();
            let mut bf2 = std::collections::HashMap::new();
            let mut bm = std::collections::HashMap::new();
            let mut br = std::collections::HashMap::new();
            let mut bq = std::collections::HashMap::new();
            let mut brw = std::collections::HashMap::new();
            let mut ref_bad = 0usize;
            let mut want = vec![0u16; if m4c_want { len * lay.per_row } else { 0 }];
            for jj in 0..len {
                let j = start + jj;
                fill_mat_row(m, &spec, &lay, &map, tref, j, jj, &mut local,
                             &mut bn, &mut bf2, &mut bm, &mut br, &mut bq, &mut brw,
                             &mut ref_bad, &mut want, fold_bend);
            }
            let collect = |h: std::collections::HashMap<usize, u32>| {
                h.into_iter().collect::<Vec<_>>()
            };
            (local, Bumps { norm: collect(bn), frac: collect(bf2), mul: collect(bm),
                            r8: br.into_iter().collect(), qnz: bq.into_iter().collect(),
                            rw: brw.into_iter().collect(), ref_bad, want })
        })
        .collect();
    let mut ref_bad = 0usize;
    for (bi, (_, b)) in out.iter().enumerate() {
        let _ = bi;
        for (i, n) in &b.norm {
            tables.m_norm[*i] += n;
        }
        for (i, n) in &b.frac {
            tables.m_frac[*i] += n;
        }
        for (i, n) in &b.mul {
            tables.m_mul[*i] += n;
        }
        for (v, n) in &b.r8 {
            let t = tables.ridx(8);
            if *v >= 0 && (*v as usize) < tables.m_r[t].len() {
                tables.m_r[t][*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("deq_eo", *v, *n as usize);
            }
        }
        // stream Z, the fold: the pair-table and width-keyed range bumps
        for (v, n) in &b.qnz {
            if *v >= 0 && (*v as usize) < tables.m_qnz.len() {
                tables.m_qnz[*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("fold_qnz", *v, *n as usize);
            }
        }
        for ((w, v), n) in &b.rw {
            let t = tables.ridx(*w);
            if *v >= 0 && (*v as usize) < tables.m_r[t].len() {
                tables.m_r[t][*v as usize] += n;
            } else {
                tables.off_table += *n as usize;
                tables.note_off("fold_rem", *v, *n as usize);
            }
        }
        ref_bad += b.ref_bad;
    }
    cols.par_iter_mut().enumerate().for_each(|(c, col)| {
        for (bi, start) in starts.iter().enumerate() {
            let src = &out[bi].0[c];
            col[*start..start + src.len()].copy_from_slice(src);
        }
    });
    // preprocessed: the per-channel scale columns (full height: the channel a slot
    // holds varies by row), then the yside selectors
    let mut pre = vec![];
    for s in 0..lay.per_row {
        match spec.variant {
            DeqVariant::Pow2 => {
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    m31_of(m.jw[b * lay.per_row + s] as i64)
                }).collect()));
            }
            DeqVariant::Chain => {
                for pick in 0..2 {
                    pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                        let (_, b) = lay.at(m.tokens, j);
                        let (mw, ew) = m.sw[b * lay.per_row + s];
                        m31_u32(if pick == 0 { mw } else { ew })
                    }).collect()));
                }
            }
            // stream Z, the fold: the weight-scale mantissa and the channel
            // half of the divisor, `pj = 2^(G_c - (14 - alpha))`
            DeqVariant::Fold => {
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    m31_u32(m.sw[b * lay.per_row + s].0)
                }).collect()));
                let beta = 14 - spec.fold_alpha as i64;
                pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                    let (_, b) = lay.at(m.tokens, j);
                    let gc = m.fold_gc(b * lay.per_row + s);
                    assert!(gc >= beta, "{}: fold channel divisor 2^{gc} below {beta}",
                            m.name);
                    m31_of(1i64 << (gc - beta))
                }).collect()));
            }
            _ => unreachable!(),
        }
    }
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    // B10 step 4: written through `at_opt`, so a padded layout's inert lead rows
    // fall out correctly and an exact one is bit-for-bit what it was.  On an
    // inert row `nofirst` and `last` are both false -- the inner fold resets to
    // its own (zero) contribution and the outer Horner takes no term -- while
    // `m1` stays 1 so the outer accumulator *carries*.  Lead rows are at the
    // start and the accumulator begins at zero, so carrying zero through them is
    // exactly right.
    let last = |j: usize| matches!(lay.at_opt(m.tokens, j), Some((_, 0)));
    let nofirst = |j: usize| matches!(lay.at_opt(m.tokens, j),
                                      Some((_, b)) if b + 1 < lay.chunks);
    pre.push(sel(&|j| nofirst(j)));                     // nofirst (within a token)
    pre.push(sel(&|j| last(j)));                        // last (token complete)
    pre.push(sel(&|j| j > 0 && !last(j)));              // m1
    pre.push(sel(&|j| j > 0 && last(j)));               // m2
    pre.push(sel(&|j| j + 1 == rows));                  // end
    if oside {
        // stream E: the output side's row step.  `oin` is an ordinary step from
        // one chunk to the next inside a token; `obd` is the jump at a token
        // boundary, which is the row that *begins* a token -- the other end from
        // `last`, and the reason these are their own columns.  Both carry `j > 0`
        // like every other multiplier here: at row 0 the `[-1, 0]` mask wraps to
        // the last row of the circle domain.  An inert lead row is neither, so
        // the accumulator is multiplied by zero there, which is exact because it
        // is still zero at that point.
        //
        // Stream SS: with `ohead` the step is the band grid's instead, and the
        // row it changes on is the row *after* one that begins a kv head --
        // `j % mh == 0`, exactly `rope::Q_SEL`'s `nn_q_hstart` on the same
        // layout.  A token boundary is a head start (the channel index wraps to
        // zero), so the same constant serves both and the pair of columns keeps
        // its shape; only which rows are marked changes.
        let first = |j: usize| matches!(lay.at_opt(m.tokens, j),
                                        Some((_, b)) if b + 1 == lay.chunks);
        let bd: Box<dyn Fn(usize) -> bool> = match ohead {
            Some(mh) => {
                assert_eq!(lay.lead, 0,
                           "{}: a band-grid output side needs an exact tiling", m.name);
                assert!(mh > 0, "{}: a band-grid output side needs hd % per_row == 0", m.name);
                Box::new(move |j: usize| j % mh == 0)
            }
            None => Box::new(first),
        };
        pre.push(sel(&|j| j > 0 && lay.at_opt(m.tokens, j).is_some() && !bd(j)));
        pre.push(sel(&|j| j > 0 && bd(j)));
        pre.push(sel(&|j| lay.at_opt(m.tokens, j).is_some()));
    }

    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    if m4c_want {
        // `B7_M4_FILL=2`: the check with teeth, and it runs before any column
        // has changed hands. Every committed and preprocessed cell, the four
        // multiplicity tables, `off_table`, `ref_bad` and -- because it is the
        // only non-integer arithmetic in the fill -- the f64 cross-check's own
        // answer per (row, slot).
        let after = m4c_snap(tables);
        let before = m4c_before.unwrap();
        let mut want_host = vec![0u16; rows * lay.per_row];
        for (bi, start) in starts.iter().enumerate() {
            let w = &out[bi].1.want;
            want_host[start * lay.per_row..start * lay.per_row + w.len()]
                .copy_from_slice(w);
        }
        let dev = crate::devfill::mat_region_dev(m, &spec, &lay, &map, tables, true, oside);
        check_fill_dev(m, &lay, &base, &pre, ref_bad, &want_host, &dev, tables,
                       &before, &after);
        // the device's own bumps are the check's, not the proof's: put the
        // shared tables back where the host fill left them
        let r8 = tables.ridx(8);
        tables.m_norm.copy_from_slice(&after.0);
        tables.m_frac.copy_from_slice(&after.1);
        tables.m_mul.copy_from_slice(&after.2);
        tables.m_r[r8].copy_from_slice(&after.3);
        tables.off_table = after.4;
    }
    MatRegion { lay, spec, map, pre, base, ref_bad, n, dev: None }
}

/// `B7_M4_FILL=2`: the device fill against the host fill, everything at once.
///
/// Not a sample: a fill that is wrong on one slot of one row is wrong, and the
/// point of checking here rather than against a proof digest is that a
/// mismatch names the column and the cell.
#[allow(clippy::too_many_arguments)]
fn check_fill_dev(m: &Mat, lay: &MatLayout, base: &[Eval], pre: &[Eval],
                  ref_bad: usize, want_host: &[u16],
                  dev: &crate::devfill::DevFill, tables: &Tables,
                  before: &(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize),
                  after: &(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize)) {
    use stwo::prover::backend::Column as _;
    let cmp = |what: &str, host: &[Eval], got: &[stwo::stwo_cuda::base_field_vec::BaseFieldVec]| {
        assert_eq!(host.len(), got.len(), "{}: {what} column count", m.name);
        for (i, (a, b)) in host.iter().zip(got).enumerate() {
            let (a, b) = (a.values.to_cpu(), b.to_vec());
            assert_eq!(a.len(), b.len(), "{}: {what} column {i} length", m.name);
            if let Some(k) = (0..a.len()).find(|k| a[*k] != b[*k]) {
                panic!("{}: {what} column {i} differs at cell {k}: host {:?}, device {:?}",
                       m.name, a[k], b[k]);
            }
        }
    };
    cmp("base", base, &dev.base);
    cmp("pre", pre, &dev.pre);
    // the multiplicities: the device's delta against the host's, table by table
    let delta = |a: &[u32], b: &[u32]| -> Vec<u32> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    };
    let r8 = tables.ridx(8);
    let dev_now = (tables.m_norm.clone(), tables.m_frac.clone(), tables.m_mul.clone(),
                   tables.m_r[r8].clone(), tables.off_table);
    for (what, h, d) in [
        ("m_norm", delta(&after.0, &before.0), delta(&dev_now.0, &after.0)),
        ("m_frac", delta(&after.1, &before.1), delta(&dev_now.1, &after.1)),
        ("m_mul", delta(&after.2, &before.2), delta(&dev_now.2, &after.2)),
        ("m_r8", delta(&after.3, &before.3), delta(&dev_now.3, &after.3)),
    ] {
        if let Some(k) = (0..h.len()).find(|k| h[*k] != d[*k]) {
            panic!("{}: {what}[{k}] bumped {} times by the host and {} by the device",
                   m.name, h[k], d[k]);
        }
    }
    assert_eq!(dev_now.4 - after.4, after.4 - before.4,
               "{}: off-table bumps differ", m.name);
    if let Some(w) = &dev.want {
        assert_eq!(dev.ref_bad, ref_bad, "{}: ref_bad differs", m.name);
        // The one unknown 19.6 named: `fill_mat_row`'s f64 cross-check, on
        // device fp64. Compared per (row, slot), not as a count.
        assert_eq!(w.len(), want_host.len(), "{}: want length", m.name);
        if let Some(k) = (0..w.len()).find(|k| w[*k] as u16 != want_host[*k]) {
            panic!("{}: the f64 cross-check differs at row {} slot {}: host {:#06x}, \
                    device {:#06x}",
                   m.name, k / lay.per_row, k % lay.per_row, want_host[k], w[k]);
        }
    }
    eprintln!("fill-dev: {} -- {} base and {} pre columns, {} cells, the four \
multiplicity tables and the f64 cross-check all match",
              m.name, base.len(), pre.len(),
              (base.len() + pre.len()) << lay.log_rows);
}

/// B1's own answer for one dequantized accumulator: the `Round::Even` reference
/// `fill_mat_row` cross-checks its committed columns against.  It lives here as
/// one function so that a consumer of a matmul's bf16 output -- the glue's
/// SwiGLU reads the `gate` and `up` regions' -- reads exactly the value the
/// dequant region proves, rather than a second implementation of the rung.
pub fn deq_ref(m: &Mat, spec: &Spec, t: usize, c: usize) -> u16 {
    deq_ref_a(m.acc[t * m.n_out + c], m, spec, t, c)
}

/// The same, for an accumulator the caller already has -- which is not always
/// `m.acc[..]`: an inert lead row of a padded layout commits a *zero*
/// accumulator against token 0's scale, and its reference has to be the zero's
/// image, not the value that would have been there had the row been real.
pub fn deq_ref_a(a: i64, m: &Mat, spec: &Spec, t: usize, c: usize) -> u16 {
    // Stream JJ (E11): every rounding here is `spec.round`'s, not `from_f64`'s.
    // `from_f64` is round-to-nearest-even and this reference is what the output
    // ties' claim is built from, so under `--round away` an exact tie made the
    // claim disagree with the region's own committed cells by one ulp and the
    // *honest* proof failed to prove.  It cost ~1% of slots on a synthetic
    // layer; on a witness with no ties it costs nothing, which is why stream E
    // never saw it.
    let rd = |x: f64| bf16::from_f64_round(x, spec.round);
    match spec.variant {
        DeqVariant::Pow2 => {
            let jjx = m.jx[t] as i64 + m.jw[c] as i64;
            rd(a as f64 * (jjx as f64).exp2())
        }
        DeqVariant::Chain => {
            let y0 = bf16::to_f32(rd(a.unsigned_abs() as f64)) as f64;
            let sxv = bf16::to_f32(bf16::pattern(0, m.sx[t].1, m.sx[t].0)) as f64;
            let swv = bf16::to_f32(bf16::pattern(0, m.sw[c].1, m.sw[c].0)) as f64;
            let t1 = bf16::to_f32(rd(y0 * sxv)) as f64;
            rd(t1 * swv) | (u16::from(a < 0) << 15)
        }
        _ => unreachable!("matmul regions are pow2 or chain"),
    }
}

/// Stream SS: a `Variant::Fold` region's **output**, as an integer.
///
/// `deq_ref` cannot answer for the fold: the whole point of `y-delay-v` is that
/// there is no bf16 waypoint between the accumulator and the int8 cell, so this
/// region's output is not a bf16 pattern at all.  What it publishes is
/// `qc (1 - 2 sg)` -- the clamped quotient with its sign -- which is
/// `experiments/b1`'s `quantize(v, s_v)` cell for cell, and which
/// `experiments/b65/extract.py` byte-compares against the recorder's own `vvc`
/// at extraction ("v fold identity holds on N sites").
///
/// The arithmetic is the fill's, on integers, so the reference and the committed
/// cells are the same value by construction and not by a second rounding: `fill_mat_row`
/// computes exactly `qm = (mag mxw + 2^(G-1)) >> G` and `qc = min(qm, 127)`.
pub fn fold_out(m: &Mat, t: usize, c: usize) -> i64 {
    fold_out_a(m.acc[t * m.n_out + c], m, t, c)
}

/// The same, for an accumulator the caller already has -- `deq_ref_a`'s role for
/// the fold, and what a probe that bends an accumulator asks in order to see
/// whether the output moved.
pub fn fold_out_a(a: i64, m: &Mat, t: usize, c: usize) -> i64 {
    debug_assert_eq!(m.variant, DeqVariant::Fold, "fold_out on a non-fold matmul");
    let (ms, _) = m.sx[t];
    let (mw, _) = m.sw[c];
    let g = (m.fold_gt(t) + m.fold_gc(c)) as u32;
    let mxw = (128 + ms as i128) * (128 + mw as i128);
    let mag = a.unsigned_abs() as i128;
    let qm = ((mag * mxw + (1i128 << (g - 1))) >> g) as i64;
    let qc = qm.min(127);
    if a < 0 { -qc } else { qc }
}

/// One row of a matmul dequant region: the witness fields of `per_row`
/// accumulators, written into a block-local column slice, with table
/// multiplicities accumulated block-locally (merged by `mat_region`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn fill_mat_row(m: &Mat, spec: &Spec, lay: &MatLayout, map: &Map, tables: &Tables,
                j: usize, jj: usize, local: &mut [Vec<M31>],
                bn: &mut std::collections::HashMap<usize, u32>,
                bf2: &mut std::collections::HashMap<usize, u32>,
                bm: &mut std::collections::HashMap<usize, u32>,
                br: &mut std::collections::HashMap<i64, u32>,
                bq: &mut std::collections::HashMap<i64, u32>,
                brw: &mut std::collections::HashMap<(u32, i64), u32>,
                ref_bad: &mut usize, wref: &mut [u16],
                fold_bend: (Option<usize>, Option<usize>)) {
    let f = map.f;
    let sticky = spec.round.needs_sticky();
    // An inert lead row is filled as a row of *zero* accumulators against token 0's
    // scale: a zero accumulator is an ordinary case this gadget already proves
    // (real dumps carry them), so every constraint holds and every lookup lands
    // on a real table row -- `ao_region`'s pattern, and the reason no lookup here
    // needs an activity multiplier.
    let (t, b, inert) = match lay.at_opt(m.tokens, j) {
        Some((t, b)) => (t, b, false),
        None => (0, 0, true),
    };
    // ---- stream Z: the fold's own fill (see `dequant.rs`'s `Fold` arm) ------
    if spec.variant == DeqVariant::Fold {
        let _ = tables;
        let (ms, es) = m.sx[t];
        let alpha = spec.fold_alpha as i64;
        let gt = m.fold_gt(t);
        assert!(gt >= alpha, "{}: fold token divisor 2^{gt} below alpha {alpha}", m.name);
        assert!(es <= 134, "{}: s_x at 2^{} is above 1 -- outside the fold's window",
                m.name, es);
        local[map.shared[0]][jj] = m31_u32(ms);
        let dent_bent = i64::from(!inert && fold_bend.1 == Some(t));
        local[map.shared[1]][jj] = m31_of(1i64 << (gt - alpha + dent_bent));
        for s in 0..lay.per_row {
            let c = b * lay.per_row + s;
            let a = if inert { 0 } else { m.acc[t * m.n_out + c] };
            let (mw, _) = m.sw[c];
            let gc = m.fold_gc(c);
            assert!(gc >= 14 - alpha,
                    "{}: fold channel divisor 2^{gc} below {}", m.name, 14 - alpha);
            let g = (gt + gc) as u32;
            let mxw = (128 + ms as i64) * (128 + mw as i64);
            let sg = u32::from(a < 0);
            let mag = a.unsigned_abs() as i64;
            assert!(mag < 1i64 << (2 * crate::dequant::FOLD_LIMB),
                    "{}: fold |acc| {mag} above the two-limb window", m.name);
            let limb = (1i64 << crate::dequant::FOLD_LIMB) - 1;
            let (flo, fhi) = (mag & limb, mag >> crate::dequant::FOLD_LIMB);
            let lo_prod = flo * mxw;
            let (fa, fb) = (lo_prod >> crate::dequant::FOLD_LIMB, lo_prod & limb);
            // `round_hafz` on the magnitude: half away from zero
            let tt = (mag as i128) * (mxw as i128);
            let mut qm = ((tt + (1i128 << (g - 1))) >> g) as i64;
            // stream Z's bend control: one quotient moved by +1, everything else
            // honest -- the high remainder leaves [0, 2 fdp) and must reject
            let bent = !inert && fold_bend.0 == Some(t * m.n_out + c);
            if bent {
                qm += 1;
            }
            let qc = qm.min(127);
            let fdp = 1i64 << (g - 14);
            let fe = fhi * mxw + fa - (2 * qm - 1) * fdp;
            debug_assert!(bent || (0..2 * fdp).contains(&fe),
                          "{}: fold remainder", m.name);
            let set = |off: usize, v: M31, local: &mut [Vec<M31>]| {
                local[map.at(s, off)][jj] = v;
            };
            set(f.flo, m31_of(flo), local);
            set(f.fhi, m31_of(fhi), local);
            set(f.sg, m31_u32(sg), local);
            set(f.fa, m31_of(fa), local);
            set(f.fb, m31_of(fb), local);
            set(f.qm, m31_of(qm), local);
            set(f.qc, m31_of(qc), local);
            set(f.fe, m31_of(fe), local);
            set(f.fdp, m31_of(fdp), local);
            // the bit-for-bit gate's cross-check: the harness's own arithmetic,
            // exact in f64 (the product is 42 bits), against the fill's integers
            let sxf = (128.0 + ms as f64) * (es as f64 - 134.0).exp2();
            let (mws, ews) = m.sw[c];
            let swf = (128.0 + mws as f64) * (ews as f64 - 134.0).exp2();
            let svf = (m.jv2[c] as f64).exp2();
            let realq = a as f64 * sxf * swf / svf;
            let want = (realq.abs() + 0.5).floor().min(127.0) as i64
                * if realq < 0.0 { -1 } else { 1 };
            if want != qc * (1 - 2 * sg as i64) {
                *ref_bad += 1;
            }
            *brw.entry((spec.q_bits, qm)).or_default() += 1;
            *brw.entry((7, qc)).or_default() += 1;
            *brw.entry((spec.q_bits, qm - qc)).or_default() += 1;
            *brw.entry((crate::dequant::FOLD_LIMB, flo)).or_default() += 1;
            *brw.entry((crate::dequant::FOLD_LIMB, fb)).or_default() += 1;
            *brw.entry((16, fa)).or_default() += 1;
            *brw.entry((spec.rem_bits, fe)).or_default() += 1;
            *brw.entry((spec.rem_bits, 2 * fdp - 1 - fe)).or_default() += 1;
        }
        return;
    }
    match spec.variant {
        DeqVariant::Pow2 => local[map.shared[0]][jj] = m31_of(m.jx[t] as i64),
        DeqVariant::Chain => {
            local[map.shared[0]][jj] = m31_u32(m.sx[t].0);
            local[map.shared[1]][jj] = m31_u32(m.sx[t].1);
        }
        _ => unreachable!("matmul regions are pow2 or chain"),
    }
    for s in 0..lay.per_row {
        let c = b * lay.per_row + s;
        let a = if inert { 0 } else { m.acc[t * m.n_out + c] };
        let mut set = |off: usize, v: M31, local: &mut [Vec<M31>]| {
            local[map.at(s, off)][jj] = v;
        };
        set(f.acc, m31_of(a), local);
        let sg = u32::from(a < 0);
        let mag = a.unsigned_abs();
        let (m2, ph, mut st, r) = bf16::normalise(mag, spec.kmax)
            .unwrap_or_else(|| panic!("{}: |acc| {mag} above the kmax {} window",
                                      m.name, spec.kmax));
        if !sticky {
            st = 0;
        }
        let ri = tables.norm_at(m2, ph, st).expect("norm row");
        let nr = tables.norm[ri];
        *bn.entry(ri).or_default() += 1;
        let fi = tables.frac_at(ph, r).expect("frac row");
        *bf2.entry(fi).or_default() += 1;
        set(f.sg, m31_u32(sg), local);
        set(f.m2, m31_u32(m2), local);
        set(f.ph, m31_u32(ph), local);
        if sticky {
            set(f.st, m31_u32(st), local);
        }
        set(f.r, m31_of(r as i64), local);
        set(f.m7, m31_u32(nr.m7), local);
        set(f.ef, m31_u32(nr.expfield), local);
        set(f.nz, m31_u32(nr.nz), local);
        match spec.variant {
            DeqVariant::Pow2 => {
                let jjx = m.jx[t] as i64 + m.jw[c] as i64;
                let eo = if nr.nz == 1 { nr.expfield as i64 + jjx } else { 0 };
                set(f.eo, m31_of(eo), local);
                *br.entry(eo).or_default() += 1;
                if spec.round == bf16::Round::Even {
                    let want = deq_ref_a(a, m, spec, t, c);
                    if !wref.is_empty() {
                        wref[jj * lay.per_row + s] = want;
                    }
                    let got = bf16::pattern(sg, eo.max(0) as u32, nr.m7);
                    if want != got {
                        *ref_bad += 1;
                    }
                }
            }
            DeqVariant::Chain => {
                let mut m_in = nr.m7;
                let mut e_in = nr.expfield as i64;
                let scales = [m.sx[t], m.sw[c]];
                for (tt, (ms, es)) in scales.iter().enumerate() {
                    let mr = tables.mul[(m_in * 128 + ms) as usize];
                    let (po, ca) = if nr.nz == 1 { (mr.m7, mr.carry) } else { (0, 0) };
                    let eo = if nr.nz == 1 { e_in + *es as i64 - 127 + ca as i64 } else { 0 };
                    set(f.mul[tt][0], m31_u32(po), local);
                    set(f.mul[tt][1], m31_u32(ca), local);
                    set(f.mul[tt][2], m31_of(eo), local);
                    if nr.nz == 1 {
                        *bm.entry((m_in * 128 + ms) as usize).or_default() += 1;
                    }
                    *br.entry(eo).or_default() += 1;
                    m_in = po;
                    e_in = eo;
                }
                if spec.round == bf16::Round::Even {
                    let want = deq_ref_a(a, m, spec, t, c);
                    if !wref.is_empty() {
                        wref[jj * lay.per_row + s] = want;
                    }
                    let got = bf16::pattern(sg, e_in.max(0) as u32, m_in);
                    if want != got {
                        *ref_bad += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

/// The matmul region's Freivalds claimed side and LogUp columns.
pub struct MatInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed_h: SecureField,
    pub use_sum: SecureField,
    /// stream E: the output side's closing value, when this region carries one
    pub claimed_out: Option<SecureField>,
    /// stream II (E11): the x-side scale's, when a requantize seam feeds this
    /// matmul
    pub claimed_sc: Option<SecureField>,
}

/// The weights an output-side Horner claim runs on, as the composition derives
/// them from the tie challenge.  Same numbers on both sides of the tie: the
/// consumer's own claim is built from `w_glue` too.
pub struct OSideW {
    pub w: Vec<SecureField>,
    pub lam: SecureField,
    pub k_in: SecureField,
    pub k_bd: SecureField,
    pub kind: crate::dequant::OKind,
    /// Stream SS: rows per kv head, on the one output side that folds under the
    /// band grid (a folded `v`).  `None` everywhere else, and then `k_bd` is
    /// taken at a token boundary exactly as stream E wrote it.  Whatever this
    /// says, `mat_region` marks the same rows -- the two have to agree or the
    /// interaction column and its AIR read different steps.
    pub head: Option<usize>,
}

/// B11 stream U, lever 3: `gkr` is `--gkr-dequant`, and it is the one thing in
/// this file the second GKR family changes.
///
/// The region's Horner accumulators (`z`, `h`, and stream E's output side) are
/// untouched -- they are what stays in the tree. What the flag removes is the
/// LogUp half: instead of `mat_logup`'s committed running-sum columns, the
/// region's fractions become GKR input layers (collected in `compose`, before
/// the proof's transcript position) and its claimed sum leaves `use_sums` for
/// `sums_cancel`. This is the same shape stream S gave the bands in
/// `attn_trace::interaction_phase`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn mat_interaction(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
                       rels: &crate::compose::Rels, out: Option<&OSideW>, gkr: bool,
                       sc: Option<(SecureField, SecureField)>)
                       -> MatInteraction {
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, lay.per_row);
    // B7a milestone 4 (section 3 row 1): `z` and `h` where the region's own
    // accumulator columns are. `B7_M4_ACC=2` runs both and asserts every cell.
    let m4g = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let m4g_host = m4g.is_none() || crate::devacc::mode() == 2;
    // B11 stream M: the region's real height, kept before `rows` is shadowed to
    // zero. `devacc` builds `z` and `h`; it does *not* build stream E's output
    // side, which is a third accumulator over the *witness* rather than over
    // this region's columns and so has nothing to do with where the fill went.
    // Suppressing it with the other two left `u[rows - 1]` indexing an empty
    // vector -- reachable only on an o-side matmul with `B7_M4_ACC` on, which
    // is `lm_head` and only `lm_head`, and `ACC` had never been on.
    let n_rows = rows;
    let rows = if m4g_host { rows } else { 0 };
    // the Horner accumulators, inner sums in parallel, carries serial
    let m4_t = std::time::Instant::now();
    let inner: Vec<SecureField> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut zf = SecureField::zero();
            if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                for s in 0..lay.per_row {
                    let a = m.acc[t * m.n_out + b * lay.per_row + s];
                    if a != 0 {
                        zf += rv_pow[s] * SecureField::from(m31_of(a));
                    }
                }
            }
            zf
        })
        .collect();
    crate::util::m4_add("int/deq_inner", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    let (mut z, mut h) = (vec![SecureField::zero(); rows], vec![SecureField::zero(); rows]);
    // B7a milestone 4 (18.4): the two accumulators, each with its own
    // dependency structure honoured rather than one serial pass over both.
    // The lead prefix keeps this exact: `inner` is zero on every lead row, so
    // the scan's extra `s_ch` steps there multiply zero — but only when the
    // lead is chunk-aligned, so the guard falls back to the serial loop if not.
    if crate::util::m4_switch("B7_M4_SCAN") && lay.chunks > 0 && rows % lay.chunks == 0
        && lay.lead % lay.chunks == 0 {
        z.par_chunks_mut(lay.chunks).enumerate().for_each(|(c, zc)| {
            let base = c * lay.chunks;
            let mut run = SecureField::zero();
            for (k, zk) in zc.iter_mut().enumerate() {
                run = if k > 0 { run * rv_step } else { SecureField::zero() } + inner[base + k];
                *zk = run;
            }
        });
        let n_last = rows / lay.chunks;  // B7a m4g: `rows` is 0 on the device route
        let mut hh = vec![SecureField::zero(); n_last];
        let mut run = SecureField::zero();
        for (m, hm) in hh.iter_mut().enumerate() {
            run = run * s_ch + z[(m + 1) * lay.chunks - 1];
            *hm = run;
        }
        h.par_iter_mut().enumerate().for_each(|(j, hj)| {
            let m = (j + 1) / lay.chunks;
            *hj = if m == 0 { SecureField::zero() } else { hh[m - 1] };
        });
    } else {
        for j in 0..rows {
            let nofirst = matches!(lay.at_opt(m.tokens, j), Some((_, b)) if b + 1 < lay.chunks);
            let last = matches!(lay.at_opt(m.tokens, j), Some((_, 0)));
            z[j] = if nofirst { z[j - 1] * rv_step } else { SecureField::zero() } + inner[j];
            h[j] = if j == 0 {
                SecureField::zero()
            } else if last {
                h[j - 1] * s_ch
            } else {
                h[j - 1]
            };
            if last {
                h[j] += z[j];
            }
        }
    }
    crate::util::m4_add("int/deq_carry", m4_t.elapsed().as_secs_f64());
    let m4_t = std::time::Instant::now();
    let mut sink = crate::device::ColSink::new();
    let claimed_h = match m4g {
        Some(d) => {
            let got = crate::devacc::deq(d, &lay, &r.map, r.map.f.acc, rv, s_ch);
            let hh = m4g_host.then(|| {
                let mut e = crate::attn_trace::secure_cols_pub(lay.log_rows, &z);
                e.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &h));
                (e, vec![h[rows - 1]])
            });
            crate::devacc::place(&format!("deq {} 2^{}", m.name, lay.log_rows), got, hh,
                                 &mut sink)[0]
        }
        None => {
            sink.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &z));
            sink.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &h));
            h[rows - 1]
        }
    };
    let mut cols = sink.into_inner();
    crate::util::m4_add("int/deq_cols", m4_t.elapsed().as_secs_f64());

    // stream E: the output side, between the claimed side's columns and the
    // LogUp columns -- the order the AIR reads them in.
    let claimed_out = out.map(|o| {
        // `n_rows`, not `rows`: this accumulator is the witness's, and it runs
        // whatever `B7_M4_ACC` did to the other two.
        let mut u = vec![SecureField::zero(); n_rows];
        for j in 0..n_rows {
            let mut f = SecureField::zero();
            if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                for s in 0..lay.per_row {
                    let c = b * lay.per_row + s;
                    // stream SS: the fold has no bf16 image, so its reference is
                    // the integer itself and `deq_ref` is not asked for one.
                    if o.kind == crate::dequant::OKind::Int {
                        f += SecureField::from(m31_of(fold_out(m, t, c))) * o.w[s];
                        continue;
                    }
                    let pat = crate::glue::canon(deq_ref(m, &r.spec, t, c));
                    f += match o.kind {
                        crate::dequant::OKind::Pair => {
                            let (sm, ef) = crate::glue::split(pat);
                            (SecureField::from(m31_of(sm))
                             + SecureField::from(m31_u32(ef)) * o.lam) * o.w[s]
                        }
                        // the same key the argmax region compares, off the same
                        // host function the extractor's column was built with
                        // taken above, off `fold_out`, before `deq_ref` is asked
                        crate::dequant::OKind::Int => unreachable!(),
                        crate::dequant::OKind::Key => SecureField::from(
                            m31_u32(crate::argmax::order_key(pat))) * o.w[s],
                    };
                }
            }
            // the multiplier is the step *into* this row: an ordinary chunk step
            // inside a token, the boundary jump on a row that begins one --
            // and, with `o.head`, the band grid's jump on the row after one that
            // begins a kv head.  The same predicate `mat_region` marked.
            let mult = if j == 0 {
                SecureField::zero()
            } else if let Some(mh) = o.head {
                if j % mh == 0 { o.k_bd } else { o.k_in }
            } else {
                match lay.at_opt(m.tokens, j) {
                    Some((_, b)) if b + 1 == lay.chunks => o.k_bd,
                    Some(_) => o.k_in,
                    None => SecureField::zero(),
                }
            };
            u[j] = if j == 0 { f } else { u[j - 1] * mult + f };
        }
        let claimed = u[n_rows - 1];
        cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &u)
            .into_iter().map(crate::device::AnyCol::OwnedHost));
        claimed
    });

    // Stream II (E11): the x-side scale, between the output side's column and
    // the LogUp columns -- the order the AIR reads them in.  One value per
    // token on the `last` rows, the same field elements this region's shared
    // columns commit, so the producing seam's own fold is comparable to it.
    let claimed_sc = sc.map(|(mu, lam)| {
        let mut v = vec![SecureField::zero(); n_rows];
        for j in 0..n_rows {
            let (f, last) = match lay.at_opt(m.tokens, j) {
                Some((t, 0)) => {
                    let x = match r.spec.variant {
                        DeqVariant::Pow2 => SecureField::from(m31_of(m.jx[t] as i64)),
                        _ => SecureField::from(m31_u32(m.sx[t].0))
                             + SecureField::from(m31_u32(m.sx[t].1)) * lam,
                    };
                    (x, true)
                }
                _ => (SecureField::zero(), false),
            };
            let mult = if j == 0 { SecureField::zero() }
                       else if last { mu } else { SecureField::one() };
            v[j] = if j == 0 { f } else { v[j - 1] * mult + f };
        }
        let claimed = v[n_rows - 1];
        cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &v)
            .into_iter().map(crate::device::AnyCol::OwnedHost));
        claimed
    });

    // the LogUp columns, in the exact order the AIR adds lookups -- or, under
    // the flag, the same fractions handed to the GKR collector instead
    let m4_t = std::time::Instant::now();
    let use_sum = if gkr {
        // the fractions were handed to the collector before the GKR proof ran,
        // in `compose`'s own pass -- see the comment there for why the two are
        // in different places
        SecureField::zero()
    } else {
        let (lcols, use_sum) = mat_logup(m, r, rv, s_ch, rels);
        crate::util::m4_add("int/deq_logup", m4_t.elapsed().as_secs_f64());
        cols.extend(lcols);
        use_sum
    };
    MatInteraction { cols, claimed_h, use_sum, claimed_out, claimed_sc }
}

/// The interaction-mask columns come first (z, h), then the LogUp columns; this
/// builds the latter with the parallel builder.
/// The same fractions, from the AIR itself (B7a milestone 4, 18.7).
///
/// `DequantEval` is the component the verifier checks; lowering it with
/// `cudagen::lower_logup` gives every lookup's numerator and denominator as a
/// straight-line IR over this region's own columns, in `add_to_relation`
/// order -- which is the order `finalize_logup_in_pairs` batches them in and
/// therefore the order `paired` needs. Nothing below is dequant-specific:
/// swap the eval and it builds another family's LogUp columns.
///
/// The eval is rebuilt here rather than borrowed from `build_components`,
/// which does not exist yet at generation time. Only `yside.claimed_h` is
/// unavailable, and it enters constraints alone -- never a lookup -- so a
/// placeholder is correct. `B7_M4_LOGUPIR=2` is what says the two agree.
/// The evaluator whose fraction IR *is* this region's lookups.
///
/// B11 stream U, lever 3: the GKR arm needs the same evaluator, so it is built
/// here once and both callers take it. `gkr` is false in both -- the fractions
/// the arm proves are the ones the committed arm would have built, so the
/// lowering is of the *un-ablated* evaluator and asks for the flag-off shape,
/// which is stream T's precedent for the bands and is why the kernel store
/// already carries this fingerprint.
pub fn deq_logup_eval(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
                      rels: &crate::compose::Rels) -> crate::dequant::DequantEval {
    let sub: Vec<crate::attn_air::RangeRel> = crate::dequant::range_widths(&r.spec)
        .iter()
        .map(|w| rels.range(*w).clone())
        .collect();
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, r.lay.per_row);
    crate::dequant::DequantEval {
        log_rows: r.lay.log_rows,
        sp: r.spec.clone(),
        tag: m.name.clone(),
        norm: rels.norm.clone(),
        frac: rels.frac.clone(),
        mul: rels.mul.clone(),
        qnz: rels.qnz.clone(),
        rels: sub,
        yside: Some(crate::dequant::YSide {
            rv_pow,
            rv_step,
            s: s_ch,
            claimed_h: SecureField::zero(),
        }),
        // the output side carries no lookups, so it cannot change this eval's
        // LogUp IR -- leaving it off here keeps the lowering identical whether
        // the tie is on or not, which is what `B7_M4_LOGUPIR=2` compares
        oside: None,
        sside: None,
        gkr: false,
    }
}

fn mat_logup_ir(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
                rels: &crate::compose::Rels, on_device: bool)
                -> (Vec<crate::device::AnyCol<'static>>, SecureField) {
    use stwo::prover::backend::simd::m31::PackedM31;
    let eval = deq_logup_eval(m, r, rv, s_ch, rels);
    // `B7_M4_LOGUPMUT` picks one of 2b's deliberate defects, so mode 2 has a
    // negative control. B7a milestone 4 (section 3 row 1): one place reads it
    // for every family, and it takes a `:family` suffix -- unaimed, a defect
    // panics at whichever family is lowered first and says nothing about the
    // rest (section 5 rule 5).
    let ir = crate::devfrac::ir_of(&eval, r.lay.log_rows);
    if on_device {
        // The same IR, run by the generated kernel instead of the interpreter.
        // The shape id comes from the compiled table by fingerprint, so a
        // stale library is a loud failure and never a wrong proof.
        let family = <crate::dequant::DequantEval as
            stwo_constraint_framework::FrameworkEval>::cuda_eval_name(&eval);
        let fp = ir.fingerprint();
        let (shape, looks) = crate::gencudakernels::gen_shape_of(family, fp).unwrap_or_else(|| {
            panic!("no generation kernel for {family} fingerprint {fp:#010x}                     ({} lookups): regenerate with `b65 genemit` and rebuild libstwo_cuda.a",
                   ir.fracs.len())
        });
        assert_eq!(looks, ir.fracs.len(), "{family}: lookup count disagrees with the kernel");
        // B11 stream M: the region's preprocessed slice is wider than this
        // eval's on an o-side matmul.
        //
        // `mat_region` commits three extra selectors for a region that
        // publishes an output-side claim, and they are the *argmax*
        // component's -- it "shares the head dequant's step selectors"
        // (`compose.rs`, where it is built). `DequantEval` never reads them, so
        // its fraction IR declares the 506 columns it does read while the
        // region carries 509, and handing the kernel all of them fails
        // `logup_cols_dev_ptrs`' count check. The three sit *after* the ones
        // the IR loads, so the prefix is exactly what to supply -- and the
        // assert says so rather than trusting it.
        let np = ir.cols[0];
        let npre = |n: usize| {
            assert!(n >= np,
                    "{family}: the region has {n} preprocessed columns, fewer than \
                     the {np} its fraction IR loads");
            np
        };
        if let Some(d) = &r.dev {
            // B7a milestone 4 (19.6): the fill left this region's columns on
            // the device, so the fractions read them where they are and the
            // LogUp columns never come back. 19.4 measured both transfers and
            // said they disappear together; this is the together.
            let (cols, sum) = crate::devlogup::logup_cols_dev(
                &ir, family, shape, r.lay.log_rows, &d.pre[..npre(d.pre.len())], &d.base);
            return (cols.into_iter()
                        .map(|c| crate::device::AnyCol::OwnedDev(c, r.lay.log_rows))
                        .collect(),
                    sum);
        }
        let (cols, sum) = crate::devlogup::logup_cols_dev_to_host(
            &ir, family, shape, r.lay.log_rows, &r.pre[..npre(r.pre.len())], &r.base);
        return (crate::device::host_cols(cols), sum);
    }
    // the same prefix, for the interpreter: `logup_cols` indexes this list by
    // the IR's own column numbers, so a wider one is silently ignored rather
    // than loudly wrong -- but only the prefix is ever meant
    let pre: Vec<&[PackedM31]> =
        r.pre.iter().take(ir.cols[0]).map(|e| &e.data[..]).collect();
    let base: Vec<&[PackedM31]> = r.base.iter().map(|e| &e.data[..]).collect();
    let (cols, sum) = crate::cudagen::logup_cols(&ir, r.lay.log_rows, &pre, &base);
    (crate::device::host_cols(cols), sum)
}

fn mat_logup(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
             rels: &crate::compose::Rels)
             -> (Vec<crate::device::AnyCol<'static>>, SecureField) {
    // B7a milestone 4 (19.6): with the fill on the device there is no host
    // column to build the fractions from, so the device path is the only one
    // -- and it is also the one that leaves 4a's bytes at zero.
    if r.dev.is_some() {
        return mat_logup_ir(m, r, rv, s_ch, rels, true);
    }
    // Stream Z: the fold's fill is on the host and so is its LogUp -- the
    // hand-written builder below carries its looks, and no generated fraction
    // kernel is asked for.
    let m4b = if r.spec.variant == DeqVariant::Fold {
        0
    } else {
        crate::util::m4b_mode("B7_M4_LOGUPIR")
    };
    let from_ir = (m4b > 0).then(|| mat_logup_ir(m, r, rv, s_ch, rels, m4b >= 3));
    // 2 checks the interpreter against the hand-written builder, 4 the kernel.
    if m4b == 1 || m4b == 3 {
        return from_ir.unwrap();
    }
    use stwo::prover::backend::simd::m31::PackedM31;
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let map = &r.map;
    let f = map.f;
    let sp = &r.spec;
    let sticky = sp.round.needs_sticky();
    let one = PackedQM31::broadcast(SecureField::one());
    // pre layout: per-slot chan cols first (1 or 2 per slot), then selectors
    let chan_w = if matches!(sp.variant, DeqVariant::Chain | DeqVariant::Fold) { 2 } else { 1 };
    enum L {
        Norm([Src2; 6]),
        Frac([Src2; 3]),
        Mul([Src2; 4], Src2),
        Range(u32, Src2),
        /// stream Z, the fold: the derived complement `2 fdp - 1 - fe` and
        /// the clamp gap `qm - qc`, each into its range table
        Rem2(u32, Src2, Src2),
        Diff(u32, Src2, Src2),
    }
    #[derive(Clone, Copy)]
    enum Src2 {
        B(usize),
        P(usize),
        Zero,
    }
    let mut looks: Vec<L> = vec![];
    for s in 0..sp.per_row {
        // stream Z, the fold: no normalisation core -- its six looks stand alone,
        // in the AIR's `add_to_relation` order
        if sp.variant == DeqVariant::Fold {
            looks.push(L::Range(sp.q_bits, Src2::B(map.at(s, f.qm))));
            looks.push(L::Range(7, Src2::B(map.at(s, f.qc))));
            looks.push(L::Diff(sp.q_bits, Src2::B(map.at(s, f.qm)),
                               Src2::B(map.at(s, f.qc))));
            looks.push(L::Range(crate::dequant::FOLD_LIMB, Src2::B(map.at(s, f.flo))));
            looks.push(L::Range(crate::dequant::FOLD_LIMB, Src2::B(map.at(s, f.fb))));
            looks.push(L::Range(16, Src2::B(map.at(s, f.fa))));
            looks.push(L::Range(sp.rem_bits, Src2::B(map.at(s, f.fe))));
            looks.push(L::Rem2(sp.rem_bits, Src2::B(map.at(s, f.fdp)),
                               Src2::B(map.at(s, f.fe))));
            continue;
        }
        let st = if sticky { Src2::B(map.at(s, f.st)) } else { Src2::Zero };
        looks.push(L::Norm([Src2::B(map.at(s, f.m2)), Src2::B(map.at(s, f.ph)), st,
                            Src2::B(map.at(s, f.m7)), Src2::B(map.at(s, f.ef)),
                            Src2::B(map.at(s, f.nz))]));
        looks.push(L::Frac([Src2::B(map.at(s, f.ph)), Src2::B(map.at(s, f.r)), st]));
        match sp.variant {
            DeqVariant::Pow2 => looks.push(L::Range(8, Src2::B(map.at(s, f.eo)))),
            DeqVariant::Chain => {
                for t in 0..2 {
                    let m_in = if t == 0 { Src2::B(map.at(s, f.m7)) }
                               else { Src2::B(map.at(s, f.mul[0][0])) };
                    let ms = if t == 0 { Src2::B(map.shared[0]) } else { Src2::P(chan_w * s) };
                    looks.push(L::Mul([m_in, ms, Src2::B(map.at(s, f.mul[t][0])),
                                       Src2::B(map.at(s, f.mul[t][1]))],
                                      Src2::B(map.at(s, f.nz))));
                    looks.push(L::Range(8, Src2::B(map.at(s, f.mul[t][2]))));
                }
            }
            _ => unreachable!(),
        }
    }
    let zero = eval_of(r.lay.log_rows, vec![M31::zero(); 1 << r.lay.log_rows]);
    let get = |s: Src2, vr: usize| match s {
        Src2::B(i) => r.base[i].data[vr],
        Src2::P(i) => r.pre[i].data[vr],
        Src2::Zero => zero.data[vr],
    };
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        match &looks[i] {
            L::Norm(c) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (one, rels.norm.combine(&v))
            }
            L::Frac(c) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (one, rels.frac.combine(&v))
            }
            L::Mul(c, m) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (PackedQM31::from(get(*m, vr)), rels.mul.combine(&v))
            }
            L::Range(b, s) => (one, rels.range(*b).combine(&[get(*s, vr)])),
            L::Diff(b, x, y) => {
                let v = get(*x, vr) - get(*y, vr);
                (one, rels.range(*b).combine(&[v]))
            }
            L::Rem2(b, fdp, fe) => {
                let two = PackedM31::broadcast(M31::from_u32_unchecked(2));
                let onep = PackedM31::broadcast(M31::from_u32_unchecked(1));
                let v = get(*fdp, vr) * two - onep - get(*fe, vr);
                (one, rels.range(*b).combine(&[v]))
            }
        }
    };
    let (n_pairs, fpair) = crate::plogup::paired(looks.len(), fr);
    let host = crate::plogup::build(r.lay.log_rows, n_pairs, fpair);
    if let Some((cols, sum)) = from_ir {
        // `B7_M4_LOGUPIR=2`: the check. Every column, every cell, and the
        // claimed sum -- not a sample, because a lowering that is wrong on one
        // lookup of one row is wrong.
        use stwo::prover::backend::Column as _;
        assert_eq!(cols.len(), host.0.len(),
                   "{}: IR built {} LogUp columns, the hand-written builder {}",
                   m.name, cols.len(), host.0.len());
        for (i, (a, b)) in cols.iter().zip(&host.0).enumerate() {
            assert!(a.host().values.to_cpu() == b.values.to_cpu(),
                    "{}: LogUp column {i} differs between the IR and the hand-written builder",
                    m.name);
        }
        assert_eq!(sum, host.1, "{}: claimed sums differ", m.name);
        eprintln!("logup-ir: {} -- {} columns and the claimed sum match", m.name, cols.len());
    }
    (crate::device::host_cols(host.0), host.1)
}

// ------------------------------------------------------------- attnout region --

pub struct AoRegion {
    pub lay: AoLayout,
    /// as `WalkRegion`'s: the mirror under `B7_M4_WFILL=1`
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    /// B7a milestone 4 (section 3 row 1): the same columns, born on the device.
    pub dev: Option<crate::devwfill::DevAo>,
    pub ref_bad: usize,
    /// the committed int8 outputs, for cross-checks
    pub q_out: Vec<i64>,
    pub n: usize,
    pub rem_bits: u32,
    pub q_bits: u32,
    /// stream JJ: the two limb widths of the range check on the accumulator,
    /// and the offset that makes it non-negative
    pub acc_lo_bits: u32,
    pub acc_hi_bits: u32,
}

impl AoRegion {
    /// `2^(acc_lo_bits + acc_hi_bits - 1)`: what a committed `acc` is shifted by
    /// so its two limbs are a range check on `|acc|`.
    pub fn acc_off(&self) -> i64 {
        1i64 << (self.acc_lo_bits + self.acc_hi_bits - 1)
    }
}

impl AoRegion {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.log_rows
    }
}

/// Column order: the row's `den` and `joff`, then per slot acc, sg, qm, nz2,
/// denp, d, num, pj, alo.
///
/// Stream JJ (E11), `witness-slack.md` section 1f: `joff = jo_max - jo` is the
/// per-token output exponent, committed because `jo` is prover data and pinned
/// across the token's rows the way `den` is.  `num = 2^max(e,0)` and
/// `pj = 2^max(-e,0)` with `e = jv - jo` are committed too and pinned by one
/// lookup per slot into the preprocessed `EPOW` window table at the key
/// `jvp + joff`, where `jvp = jv - jv_min` is preprocessed and model-only.
/// `alo` is the low limb of the offset accumulator, the other half of the
/// no-wrap window.
#[allow(clippy::too_many_arguments)]
pub fn ao_region(ao: &AttnOut, per_row_target: usize, q_bits: u32, rem_bits: u32,
                 acc_bits: (u32, u32), jv_min: i32, jo_max: i32,
                 tables: &mut Tables, corrupt: Option<usize>,
                 corrupt_q: Option<usize>, corrupt_den: Option<usize>,
                 corrupt_jo: Option<usize>, corrupt_num: Option<usize>,
                 corrupt_acc: Option<usize>, corrupt_jo_tok: Option<usize>) -> AoRegion {
    let lay = AoLayout::choose(ao.nh, ao.s, ao.hd, per_row_target);
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    // B7a milestone 4 (section 3 row 1): the same three-value mode.
    let m4d = if corrupt_q.is_some() || corrupt_den.is_some() || corrupt_jo.is_some()
                 || corrupt_num.is_some() || corrupt_acc.is_some()
                 || corrupt_jo_tok.is_some() {
        0
    } else {
        crate::util::m4b_mode("B7_M4_WFILL")
    };
    assert!(m4d == 0 || corrupt.is_none(),
            "attnout: the device fill has no `corrupt` path");
    // Stream JJ (witness-slack section 7 item 1, and the row-`den` hole beside
    // it): the row whose denominator is bent.  `den` was read once per row and
    // folded into `dz` only on the token's *last* row, so on any other row it
    // -- and with it `denp` -- was free, and the slot's rounding was a rounding
    // by a denominator of the prover's choosing.  This control rounds one
    // non-last row honestly against `den - 1`: every constraint and every lookup
    // of the pre-change AIR holds, so it accepts there, and the `den` pin is
    // what rejects it now.
    // stream JJ: the row whose `jo` is moved by one, for the control that the
    // `nofirst` pin on `pjo`/`njo` exists to catch.  Same resolution as
    // `bend_den`'s, and it must also land off a token's first row -- the pin
    // compares a row with its predecessor.
    let row_of_site = |idx: usize, what: &str| -> usize {
        let (ht, ch) = (idx / ao.hd, idx % ao.hd);
        let (h, t) = (ht / ao.s, ht % ao.s);
        let j = (0..rows)
            .find(|j| matches!(lay.at(ao.nh, ao.s, *j), Some((hh, tt, b))
                               if hh == h && tt == t && b * pr <= ch && ch < (b + 1) * pr))
            .unwrap_or_else(|| panic!("{what}: no row carries channel {ch}"));
        let b = lay.at(ao.nh, ao.s, j).unwrap().2;
        assert!(b + 1 < lay.chunks,
                "{what} needs a row that is not its token's first (b = chunks - 1);                  index {idx} lands on b = {b}");
        j
    };
    let bend_jo: Option<usize> = corrupt_jo.map(|idx| row_of_site(idx, "--corrupt-ao-jo"));
    // Which way the `jo` control moves.  It has to stay inside the statement's
    // own windows or it would be caught by the `EPOW` lookup instead of by the
    // pin it exists to test -- `joff = jo_max - jo` must stay non-negative and
    // every slot's `e = jv - jo` must stay in `[e_lo, e_hi]`.  Down when the
    // top of the exponent window has room, up otherwise; a window with no room
    // either way is a statement too tight to test on, and says so.
    let bend_jo_step: i32 = match bend_jo {
        None => 0,
        Some(j) => {
            let (h, t, b) = lay.at(ao.nh, ao.s, j).expect("--corrupt-ao-jo: an inert row");
            let es: Vec<i32> = (0..pr)
                .map(|s| ao.jv[h * ao.hd + b * pr + s] - ao.jo[t])
                .collect();
            // Up first: a larger `jo` is a smaller `num`, and a smaller `num`
            // cannot push `qm` past the `ao_q_bits` pair table.  Moving `jo`
            // *down* can, which is itself worth knowing -- see the record.
            let down = es.iter().all(|e| e + 1 <= tables.e_hi);
            let up = ao.jo[t] < jo_max && es.iter().all(|e| e - 1 >= tables.e_lo);
            assert!(down || up,
                    "--corrupt-ao-jo: the jo window (top {jo_max}) and the exponent \
                     window [{}, {}] leave no room to move one row's jo; widen one with \
                     --ao-jo-min / --ao-jo-max",
                    tables.e_lo, tables.e_hi);
            if up { 1 } else { -1 }
        }
    };
    // The same choice for the whole-token control, over that token's every row.
    let bend_tok_step: i32 = match corrupt_jo_tok {
        None => 0,
        Some(t) => {
            let t = t % ao.s;
            let es: Vec<i32> = (0..ao.nh)
                .flat_map(|h| (0..ao.hd).map(move |c| (h, c)))
                .map(|(h, c)| ao.jv[h * ao.hd + c] - ao.jo[t])
                .collect();
            let down = es.iter().all(|e| e + 1 <= tables.e_hi);
            let up = ao.jo[t] < jo_max && es.iter().all(|e| e - 1 >= tables.e_lo);
            assert!(down || up,
                    "--corrupt-ao-jo-tok: the windows leave no room to move token {t}'s \
                     jo; widen one with --ao-jo-min / --ao-jo-max");
            if up { 1 } else { -1 }
        }
    };
    let (acc_lo_bits, acc_hi_bits) = acc_bits;
    let acc_off = 1i64 << (acc_lo_bits + acc_hi_bits - 1);
    let bend_den: Option<usize> = corrupt_den.map(|idx| {
        let (ht, ch) = (idx / ao.hd, idx % ao.hd);
        let (h, t) = (ht / ao.s, ht % ao.s);
        let j = (0..rows)
            .find(|j| matches!(lay.at(ao.nh, ao.s, *j), Some((hh, tt, b))
                               if hh == h && tt == t && b * pr <= ch && ch < (b + 1) * pr))
            .expect("--corrupt-ao-den: no row carries that channel");
        let b = lay.at(ao.nh, ao.s, j).unwrap().2;
        assert!(b != 0,
                "--corrupt-ao-den needs a non-last row of a token (b = 0 is the row \
                 whose `den` the `dz` fold publishes); index {idx} lands on b = 0");
        assert!(ao.den[h * ao.s + t] > 1, "--corrupt-ao-den: this token's Z is 1");
        j
    });
    if m4d == 1 {
        let dev = crate::devwfill::ao_region_dev(ao, &lay, q_bits, rem_bits, acc_bits,
                                                 jv_min, jo_max, tables, false);
        let (ref_bad, q_out) = (dev.ref_bad, dev.q_out.clone());
        return AoRegion { lay, pre: vec![], base: vec![], dev: Some(dev), ref_bad, q_out,
                          n: ao.nh * ao.s * ao.hd, rem_bits, q_bits,
                          acc_lo_bits, acc_hi_bits };
    }
    // `B7_M4_WFILL=2` builds this region twice -- host here, device below --
    // and the device call bumps the same counters a second time, so every
    // counter this fill touches has to be snapshotted before, compared as a
    // delta, and restored after.  Stream JJ (E11) added three: the two
    // accumulator limb tables and `m_epow` (whose bumps are a host loop on
    // *both* sides, `devwfill::ao_region_dev`).  A counter left out of this
    // tuple is counted twice and the honest proof is then rejected with
    // `sums_cancel: false` -- which is what it did on the H200's reference
    // chunk, and what no desk shape shows, because the desk never runs mode 2.
    let m4d_snap = |t: &Tables| -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        (t.m_qnz.clone(), t.m_r[t.ridx(rem_bits)].clone(),
         t.m_r[t.ridx(acc_lo_bits)].clone(), t.m_r[t.ridx(acc_hi_bits)].clone(),
         t.m_epow.clone(), t.off_table)
    };
    let m4d_before = (m4d == 2).then(|| m4d_snap(tables));
    let mut want_host = vec![0u32; if m4d == 2 { rows * pr } else { 0 }];
    let n_cols = 2 + 9 * pr;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut q_out = vec![0i64; ao.nh * ao.s * ao.hd];
    let mut ref_bad = 0usize;
    for j in 0..rows {
        match lay.at(ao.nh, ao.s, j) {
            None => {
                // inert lead row: den = 1, everything else 0 -> d = denp = 1,
                // dp = 0, (qm, nz2) = (0, 0); all constraints hold, all lookups
                // land on real rows
                // `joff = jo_max - jo` with `jo = jo_max` and `jvp = 0`, so the
                // `EPOW` key is 0 and the pair is `(2^max(e_lo,0), 2^max(-e_lo,0))`
                cols[0][j] = M31::one();
                let (num0, pj0) = (1i64 << tables.e_lo.max(0),
                                   1i64 << (-tables.e_lo).max(0));
                for s in 0..pr {
                    let c0 = 2 + 9 * s;
                    // `acc` stays the raw accumulator (the `z` fold reads it);
                    // what the two limbs range-check is `acc + acc_off`
                    cols[c0 + 4][j] = m31_of(pj0);                 // denp = den pj
                    cols[c0 + 5][j] = m31_of(pj0);                 // d = denp
                    cols[c0 + 6][j] = m31_of(num0);
                    cols[c0 + 7][j] = m31_of(pj0);
                    cols[c0 + 8][j] = m31_of(acc_off & ((1i64 << acc_lo_bits) - 1));
                    tables.bump_ek(tables.e_lo, "ao_epow_lead");
                    tables.bump_qnzk(0, "ao_qnz_lead");
                    tables.bump_rk(rem_bits, pj0, "ao_rem_lead");           // d
                    tables.bump_rk(rem_bits, pj0 - 1, "ao_rem_lead");       // dp
                    tables.bump_rk(rem_bits, pj0, "ao_rem_lead");           // denp
                    tables.bump_rk(acc_lo_bits, acc_off & ((1i64 << acc_lo_bits) - 1),
                                   "ao_acc_lead");
                    tables.bump_rk(acc_hi_bits, acc_off >> acc_lo_bits, "ao_acc_lead");
                }
            }
            Some((h, t, b)) => {
                // stream JJ's control bends this row's denominator only
                let den = ao.den[h * ao.s + t] - i64::from(bend_den == Some(j));
                assert!(den > 0, "attnout denominator must be positive");
                // stream JJ: `jo` is committed now, as `joff = jo_max - jo`.
                // The control moves this row's `jo` by one and leaves the
                // token's other rows alone, which only the `nofirst` pin sees.
                let jo = ao.jo[t]
                    + if bend_jo == Some(j) { bend_jo_step } else { 0 }
                    + if corrupt_jo_tok.map(|x| x % ao.s) == Some(t) { bend_tok_step }
                      else { 0 };
                let joff = i64::from(jo_max - jo);
                assert!(joff >= 0, "attnout: jo above the window's top");
                cols[0][j] = m31_of(den);
                cols[1][j] = m31_of(joff);
                for s in 0..pr {
                    let ch = b * pr + s;
                    let idx = (h * ao.s + t) * ao.hd + ch;
                    let mut a = ao.acc[idx];
                    if corrupt == Some(idx) {
                        a += 1;
                    }
                    let jv = ao.jv[h * ao.hd + ch];
                    let e = jv - jo;
                    // the window pair, and the control that bends it: a `num`
                    // one doubling out is a triple the `EPOW` table does not
                    // carry, so only the lookup can reject it
                    let num = (1i64 << e.max(0)) << u32::from(corrupt_num == Some(idx));
                    let pj = 1i64 << (-e).max(0);
                    let denp = den * pj;
                    let mag = a.abs();
                    let pm = mag * num;
                    // the no-wrap window: every term of the error identity must be
                    // an integer below p, or the field reading is ambiguous
                    assert!(2 * pm < (1i64 << 31) - 1 && 2 * denp < (1i64 << 31) - 1,
                            "attnout fold outgrows the field: pm={pm} denp={denp}");
                    let mut qm = (2 * pm + denp) / (2 * denp);
                    // stream Z's bend control: one quotient pushed past the pair
                    // table's domain (the clamp), everything else derived from it
                    if corrupt_q == Some(idx) {
                        qm = 1i64 << q_bits;
                    }
                    let sg = i64::from(a < 0 && qm != 0);
                    let d = if a < 0 && qm == 0 { denp - 2 * pm }
                            else { 2 * pm - (2 * qm - 1) * denp };
                    let c0 = 2 + 9 * s;
                    // the offset accumulator's two limbs.  The control moves the
                    // committed low limb by one *inside its own table*, so both
                    // range lookups still land on real rows and the only thing
                    // that can see it is that `(acc + acc_off - alo) / 2^lo` is
                    // no longer the integer the high table was bumped at.
                    let ao_v = a + acc_off;
                    assert!(ao_v >= 0 && ao_v < 1i64 << (acc_lo_bits + acc_hi_bits),
                            "attnout: |acc| outside its declared window: {a}");
                    let alo = if corrupt_acc == Some(idx) {
                        ((ao_v & ((1i64 << acc_lo_bits) - 1)) + 1) & ((1i64 << acc_lo_bits) - 1)
                    } else {
                        ao_v & ((1i64 << acc_lo_bits) - 1)
                    };
                    cols[c0][j] = m31_of(a);
                    cols[c0 + 1][j] = m31_of(sg);
                    cols[c0 + 2][j] = m31_of(qm);
                    cols[c0 + 3][j] = m31_of(i64::from(qm != 0));
                    cols[c0 + 4][j] = m31_of(denp);
                    cols[c0 + 5][j] = m31_of(d);
                    cols[c0 + 6][j] = m31_of(num);
                    cols[c0 + 7][j] = m31_of(pj);
                    cols[c0 + 8][j] = m31_of(alo);
                    tables.bump_ek(e, "ao_epow");
                    tables.bump_qnzk(qm, "ao_qnz");
                    tables.bump_rk(rem_bits, d, "ao_rem_d");
                    tables.bump_rk(rem_bits, 2 * denp - 1 - d, "ao_rem_dp");
                    // section 7 item 1: `denp`'s own bound, without which the
                    // pair above leaves the `2 denp = d + dp + 1` halving branch
                    tables.bump_rk(rem_bits, denp, "ao_rem_denp");
                    // and the other half of the no-wrap window: `2 mag num < p`
                    // is `2 mag num` with `num <= 2^e_hi` from the `EPOW` table
                    // and `mag` bounded here, in two limbs
                    tables.bump_rk(acc_lo_bits, alo, "ao_acc_lo");
                    tables.bump_rk(acc_hi_bits, (ao_v - alo) >> acc_lo_bits, "ao_acc_hi");
                    q_out[idx] = if a < 0 { -qm } else { qm };
                    // the reference: one f64 rounding, half away from zero.  It
                    // reads `num / pj` rather than `2^(jv - jo)` -- the same
                    // number on an honest row, and on a bent one the reference
                    // follows the bend instead of counting it twice.
                    let want = ((mag as f64 * (num as f64 / pj as f64)) / den as f64 + 0.5)
                        .floor() as i64;
                    if !want_host.is_empty() {
                        want_host[j * pr + s] = want as u32;
                    }
                    if want != qm {
                        ref_bad += 1;
                    }
                }
            }
        }
    }
    // preprocessed: one per-slot column, then the selectors.  Stream JJ (E11):
    // `jvp = jv - jv_min`, a per-(head, channel) model constant offset to be
    // non-negative.  It replaces the old `(num, pj)` pair, which was a function
    // of `e = jv - jo` and so of this chunk's tokens; tree 0 no longer is.
    let mut pre = vec![];
    for s in 0..pr {
        pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
            match lay.at(ao.nh, ao.s, j) {
                None => M31::zero(),
                Some((h, _t, b)) => m31_of(i64::from(ao.jv[h * ao.hd + b * pr + s] - jv_min)),
            }
        }).collect()));
    }
    let active = |j: usize| lay.at(ao.nh, ao.s, j).is_some();
    let last = |j: usize| matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
    let head_first = |j: usize| matches!(lay.at(ao.nh, ao.s, j),
                                          Some((_, t, 0)) if t == ao.s - 1);
    let sel = |fsel: &dyn Fn(usize) -> bool| {
        eval_of(lay.log_rows, (0..rows).map(|j| m31_u32(fsel(j) as u32)).collect())
    };
    pre.push(sel(&|j| matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks)));
    pre.push(sel(&|j| last(j)));
    pre.push(sel(&|j| j > 0 && !last(j)));                       // m1 (incl. inert)
    pre.push(sel(&|j| j > 0 && last(j) && !head_first(j)));      // m2: token step s
    pre.push(sel(&|j| j > 0 && last(j) && head_first(j)));       // m3: head step mu
    pre.push(sel(&|j| j + 1 == rows));                           // end
    let _ = active;
    let base: Vec<Eval> = cols.into_iter().map(|c| eval_of(lay.log_rows, c)).collect();
    if m4d == 2 {
        let after = m4d_snap(tables);
        let before = m4d_before.unwrap();
        let dev = crate::devwfill::ao_region_dev(ao, &lay, q_bits, rem_bits, acc_bits,
                                                 jv_min, jo_max, tables, true);
        crate::devwfill::check_cols("attnout base", &base, &dev.base);
        crate::devwfill::check_cols("attnout pre", &pre, &dev.pre);
        let ri = tables.ridx(rem_bits);
        let (rlo, rhi) = (tables.ridx(acc_lo_bits), tables.ridx(acc_hi_bits));
        crate::devwfill::check_delta("attnout m_qnz", &before.0, &after.0,
                                     &sub_slice(&tables.m_qnz, &after.0));
        crate::devwfill::check_delta("attnout m_rem", &before.1, &after.1,
                                     &sub_slice(&tables.m_r[ri], &after.1));
        // stream JJ (E11): the accumulator window's two limbs, and the `EPOW`
        // window, whose column is a host loop on both sides
        crate::devwfill::check_delta("attnout m_acc_lo", &before.2, &after.2,
                                     &sub_slice(&tables.m_r[rlo], &after.2));
        crate::devwfill::check_delta("attnout m_acc_hi", &before.3, &after.3,
                                     &sub_slice(&tables.m_r[rhi], &after.3));
        crate::devwfill::check_delta("attnout m_epow", &before.4, &after.4,
                                     &sub_slice(&tables.m_epow, &after.4));
        assert_eq!(tables.off_table - after.5, after.5 - before.5,
                   "attnout: off-table bumps differ");
        assert_eq!(dev.ref_bad, ref_bad, "attnout: ref_bad differs");
        assert_eq!(dev.q_out, q_out, "attnout: the committed int8 outputs differ");
        // the one part of this family's fill that is not integer arithmetic,
        // compared per (row, slot) rather than as a count -- which on an honest
        // witness is zero on both sides and would agree vacuously
        let w = dev.want.as_ref().expect("mode 2 asks for the reference");
        assert_eq!(w.len(), want_host.len(), "attnout: want length");
        if let Some(k) = (0..w.len()).find(|k| w[*k] != want_host[*k]) {
            panic!("attnout: the f64 reference differs at row {} slot {}: host {}, device {}",
                   k / pr, k % pr, want_host[k], w[k]);
        }
        // restore in table order, so a width shared between two of these (say
        // `rem_bits == acc_lo_bits`) ends on the one snapshot that covers it
        tables.m_qnz.copy_from_slice(&after.0);
        tables.m_r[ri].copy_from_slice(&after.1);
        tables.m_r[rlo].copy_from_slice(&after.2);
        tables.m_r[rhi].copy_from_slice(&after.3);
        tables.m_epow.copy_from_slice(&after.4);
        tables.off_table = after.5;
        eprintln!("wfill-dev: attnout -- {} base and {} pre columns, {} cells, \
all five multiplicity tables and the f64 reference all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.log_rows);
    }
    // Stream II (E11): the witness-side reference and the region's own
    // committed output are the same function of `AttnOut`, and a synthetic
    // layer's `o` input is built from the former -- so an untampered run
    // asserts they agree rather than trusting that they do.  Every tamper index
    // here is `None` on an honest run, which is exactly when the assert holds.
    if bend_den.is_none() && bend_jo.is_none() && corrupt.is_none() && corrupt_q.is_none()
        && corrupt_num.is_none() && corrupt_acc.is_none() && corrupt_jo_tok.is_none()
    {
        debug_assert_eq!(q_out, ao.q_ref(), "attnout: q_out and AttnOut::q_ref disagree");
    }
    AoRegion { lay, pre, base, dev: None, ref_bad, q_out, n: ao.nh * ao.s * ao.hd,
               rem_bits, q_bits, acc_lo_bits, acc_hi_bits }
}

pub struct AoInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed_out: SecureField,
    pub claimed_den: SecureField,
    /// Stream JJ (E11): the region's own requantized output, folded under
    /// `claimed_out`'s weights -- what the `o` walk's `qX` equals.
    pub claimed_q: SecureField,
    pub use_sum: SecureField,
}

#[allow(clippy::too_many_arguments)]
pub fn ao_interaction(ao: &AttnOut, r: &AoRegion, rr: SecureField, s_ch: SecureField,
                      rv: SecureField, c_ctx: usize, rels: &crate::compose::Rels,
                      air: &crate::walk_air::AttnOutEval, gkr: bool) -> AoInteraction {
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, pr);
    let (_, rv_hd) = crate::attn_air::powers(rv, ao.hd);
    let (_, r_c) = crate::attn_air::powers(rr, c_ctx);
    let (_, s_span) = crate::attn_air::powers(s_ch, ao.s - 1);
    let mu_out = rv_hd * s_span.inverse();
    let mu_den = r_c * s_span.inverse();

    // B7a milestone 4 (section 3 row 1): the attnout's z, h and dz, on the same
    // three-value mode every fill in this series uses.
    let m4g = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let m4g_host = m4g.is_none() || crate::devacc::mode() == 2;
    // stream JJ's fourth accumulator is the host's whatever the device fill
    // did with the other three (stream M's lesson in `mat_interaction`: keep
    // the real height before `rows` is shadowed to zero)
    let n_rows = rows;
    let rows = if m4g_host { rows } else { 0 };
    let inner: Vec<SecureField> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut zf = SecureField::zero();
            if let Some((h, t, b)) = lay.at(ao.nh, ao.s, j) {
                for s in 0..pr {
                    let a = ao.acc[(h * ao.s + t) * ao.hd + b * pr + s];
                    if a != 0 {
                        zf += rv_pow[s] * SecureField::from(m31_of(a));
                    }
                }
            }
            zf
        })
        .collect();
    let (mut z, mut h, mut dz) = (vec![SecureField::zero(); rows],
                                  vec![SecureField::zero(); rows],
                                  vec![SecureField::zero(); rows]);
    for j in 0..rows {
        let nofirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks);
        let last = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
        let hfirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, t, 0)) if t == ao.s - 1);
        z[j] = if nofirst { z[j - 1] * rv_step } else { SecureField::zero() } + inner[j];
        let (mo, md) = if j == 0 {
            (SecureField::zero(), SecureField::zero())
        } else if last && hfirst {
            (mu_out, mu_den)
        } else if last {
            (s_ch, s_ch)
        } else {
            (SecureField::one(), SecureField::one())
        };
        h[j] = if j > 0 { h[j - 1] * mo } else { SecureField::zero() };
        dz[j] = if j > 0 { dz[j - 1] * md } else { SecureField::zero() };
        if last {
            h[j] += z[j];
            if let Some((hh, t, _)) = lay.at(ao.nh, ao.s, j) {
                dz[j] += SecureField::from(m31_of(ao.den[hh * ao.s + t]));
            }
        }
    }
    let mut cols = crate::device::ColSink::new();
    let (claimed_out, claimed_den) = match m4g {
        Some(d) => {
            let got = crate::devacc::ao(d, &lay, ao.nh, ao.s, rv, s_ch, mu_out, mu_den);
            let hh = m4g_host.then(|| {
                let mut e = crate::attn_trace::secure_cols_pub(lay.log_rows, &z);
                e.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &h));
                e.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &dz));
                (e, vec![h[rows - 1], dz[rows - 1]])
            });
            let c = crate::devacc::place(&format!("attnout 2^{}", lay.log_rows), got, hh,
                                         &mut cols);
            (c[0], c[1])
        }
        None => {
            cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &z));
            cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &h));
            cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &dz));
            (h[rows - 1], dz[rows - 1])
        }
    };

    // ---- stream JJ (E11): the region's own output, published --------------
    //
    // `z`/`h`'s shape exactly, over `q = qm (1 - 2 sg)` instead of over `acc`.
    // `AttnOutEval` *is* the requantization of the attention output, so this
    // claim is what the `o` walk's committed activations have to equal, row for
    // row -- `q_out` below is the same value this folds.  The `eqs` entry that
    // compares them is stream II's, on the consumer's side.
    let (mut zq, mut hq) = (vec![SecureField::zero(); n_rows],
                            vec![SecureField::zero(); n_rows]);
    for j in 0..n_rows {
        let nofirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, b)) if b + 1 < lay.chunks);
        let last = matches!(lay.at(ao.nh, ao.s, j), Some((_, _, 0)));
        let hfirst = matches!(lay.at(ao.nh, ao.s, j), Some((_, t, 0)) if t == ao.s - 1);
        let mut qf = SecureField::zero();
        if let Some((h, t, b)) = lay.at(ao.nh, ao.s, j) {
            for s in 0..pr {
                let q = r.q_out[(h * ao.s + t) * ao.hd + b * pr + s];
                if q != 0 {
                    qf += rv_pow[s] * SecureField::from(m31_of(q));
                }
            }
        }
        zq[j] = if nofirst { zq[j - 1] * rv_step } else { SecureField::zero() } + qf;
        let mq = if j == 0 {
            SecureField::zero()
        } else if last && hfirst {
            mu_out
        } else if last {
            s_ch
        } else {
            SecureField::one()
        };
        hq[j] = if j > 0 { hq[j - 1] * mq } else { SecureField::zero() };
        if last {
            hq[j] += zq[j];
        }
    }
    let claimed_q = hq[n_rows - 1];
    cols.extend_any(crate::attn_trace::secure_cols_pub(lay.log_rows, &zq)
                        .into_iter().map(crate::device::AnyCol::OwnedHost).collect::<Vec<_>>());
    cols.extend_any(crate::attn_trace::secure_cols_pub(lay.log_rows, &hq)
                        .into_iter().map(crate::device::AnyCol::OwnedHost).collect::<Vec<_>>());

    // lookups in AIR order, per slot: the `EPOW` window, (qm, nz2), d, dp,
    // denp, the offset accumulator's low limb, then its high limb
    let one = PackedQM31::broadcast(SecureField::one());
    let n_looks = 7 * pr;
    let inv_lo = stwo::prover::backend::simd::m31::PackedM31::broadcast(
        M31::from_u32_unchecked(1 << r.acc_lo_bits).inverse());
    let acc_off = stwo::prover::backend::simd::m31::PackedM31::broadcast(
        M31::from_u32_unchecked(r.acc_off() as u32));
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        let (s, which) = (i / 7, i % 7);
        let c0 = 2 + 9 * s;
        match which {
            // the window: `(jvp + joff, num, pj)`, `jvp` preprocessed and
            // model-only, `joff` the row's committed output exponent
            0 => (one, rels.epow.combine(&[r.pre[s].data[vr] + r.base[1].data[vr],
                                           r.base[c0 + 6].data[vr],
                                           r.base[c0 + 7].data[vr]])),
            1 => (one, rels.qnz.combine(&[r.base[c0 + 2].data[vr], r.base[c0 + 3].data[vr]])),
            2 => (one, rels.range(r.rem_bits).combine(&[r.base[c0 + 5].data[vr]])),
            3 => {
                let two = stwo::prover::backend::simd::m31::PackedM31::broadcast(
                    M31::from_u32_unchecked(2));
                let onem = stwo::prover::backend::simd::m31::PackedM31::broadcast(M31::one());
                let dp = r.base[c0 + 4].data[vr] * two - onem - r.base[c0 + 5].data[vr];
                (one, rels.range(r.rem_bits).combine(&[dp]))
            }
            // section 7 item 1: `denp` itself
            4 => (one, rels.range(r.rem_bits).combine(&[r.base[c0 + 4].data[vr]])),
            // the no-wrap window's two limbs on `acc + acc_off`
            5 => (one, rels.range(r.acc_lo_bits).combine(&[r.base[c0 + 8].data[vr]])),
            _ => {
                let hi = (r.base[c0].data[vr] + acc_off - r.base[c0 + 8].data[vr]) * inv_lo;
                (one, rels.range(r.acc_hi_bits).combine(&[hi]))
            }
        }
    };
    // B11 stream V, lever 4: as the walks, and for the same reason. The `z`,
    // `h` and `dz` accumulators stay.
    if gkr {
        let _ = fr;
        return AoInteraction {
            cols: cols.into_inner(),
            claimed_out,
            claimed_den,
            claimed_q,
            use_sum: SecureField::zero(),
        };
    }
    let (lcols, use_sum) = crate::devfrac::frac_cols(
        air, "attnout", lay.log_rows,
        crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
        crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
        || {
            let (n_pairs, fpair) = crate::plogup::paired(n_looks, fr);
            crate::plogup::build(lay.log_rows, n_pairs, fpair)
        });
    cols.extend_any(lcols);
    AoInteraction { cols: cols.into_inner(), claimed_out, claimed_den, claimed_q, use_sum }
}

// ------------------------------------------------------------------ walk regions --

pub struct WalkRegion {
    pub name: String,
    pub lay: WalkLayout,
    pub mats: Vec<usize>,
    /// Stream Z: the activation cells' range width (8, or 12 for the widened
    /// attention output's group).  The offset is `2^(act_bits - 1)`.
    pub act_bits: u32,
    /// Under `B7_M4_WFILL=1` these are the *mirror* -- the device columns, back
    /// on the host, because `walk_interaction` still reads them there. The
    /// commitment takes `dev`, so they never go back up.
    pub pre: Vec<Eval>,
    pub base: Vec<Eval>,
    pub n_weights: usize,
    /// B7a milestone 4 (section 3 row 1): the same columns, born on the device.
    pub dev: Option<crate::devwfill::DevWalk>,
}

impl WalkRegion {
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.g.log_rows
    }
    pub fn pre_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.pre.len(), None => self.pre.len() };
        n << self.lay.g.log_rows
    }
}

/// Build one walk group's region: activations committed (offset
/// `2^(act_bits-1)`), weights preprocessed, range multiplicities bumped at
/// `act_bits` -- 8 everywhere but the widened attention output's group.
pub fn walk_region(layer: &Layer, name: &str, mat_idx: &[usize], p_target: usize,
                   act_bits: u32, tables: &mut Tables,
                   corrupt_qx: Option<usize>, oversize_qx: Option<usize>) -> WalkRegion {
    let m0 = &layer.mats[mat_idx[0]];
    let n_in = m0.n_in;
    let tokens = m0.tokens;
    let n_out_max = mat_idx.iter().map(|i| layer.mats[*i].n_out).max().unwrap();
    let lay = WalkLayout::choose(n_in, n_out_max, tokens, p_target);
    let rows = 1usize << lay.g.log_rows;
    let off = 1i32 << (act_bits - 1);
    let qmax = off - 1;
    for i in mat_idx {
        assert_eq!(layer.mats[*i].n_in, n_in, "walk group mixes input widths");
        // q/k/v (and gate/up) must share one committed input
        assert_eq!(layer.mats[*i].qx, m0.qx, "walk group inputs differ");
    }

    // B7a milestone 4 (section 3 row 1): which fill runs. 1 is the device's
    // alone -- `pre` and `base` become the mirror the interaction reads -- and
    // 2 builds both and asserts every cell equal before either is used.
    //
    // Stream Z: the device fill packs activations as int8 and bumps the 8-bit
    // table in its kernel, so a wide group takes the host fill whatever the
    // switch says -- the corner bands' pattern, and `devpre::note_host_fill`
    // is told in `generate`.
    let m4d = if act_bits == 8 { crate::util::m4b_mode("B7_M4_WFILL") } else { 0 };
    assert!(m4d == 0 || corrupt_qx.is_none(),
            "{name}: the device walk fill has no `corrupt_qx` path");
    if m4d == 1 {
        let dev = crate::devwfill::walk_region_dev(layer, mat_idx, &lay, tables);
        // B7a milestone 4 (section 3 row 1): no mirror. `walk_interaction` reads
        // the device columns now, and nothing else reads a device region's host
        // columns -- `base_cells` counts them off `dev`.
        return WalkRegion { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits,
                            pre: vec![], base: vec![],
                            n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                            dev: Some(dev) };
    }
    let m4d_snap = |t: &Tables| -> (Vec<u32>, usize) {
        (t.m_r[t.ridx(8)].clone(), t.off_table)
    };
    let m4d_before = (m4d == 2).then(|| m4d_snap(tables));

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

    if m4d == 2 {
        // the check with teeth, before any column has changed hands
        let after = m4d_snap(tables);
        let before = m4d_before.unwrap();
        let dev = crate::devwfill::walk_region_dev(layer, mat_idx, &lay, tables);
        crate::devwfill::check_cols(&format!("walk {name} base"), &base, &dev.base);
        crate::devwfill::check_cols(&format!("walk {name} pre"), &pre, &dev.pre);
        let r8 = tables.ridx(8);
        crate::devwfill::check_delta(&format!("walk {name} m_r8"), &before.0, &after.0,
                                     &sub_slice(&tables.m_r[r8], &after.0));
        assert_eq!(tables.off_table - after.1, after.1 - before.1,
                   "walk {name}: off-table bumps differ");
        // the device's own bumps are the check's, not the proof's
        tables.m_r[r8].copy_from_slice(&after.0);
        tables.off_table = after.1;
        eprintln!("wfill-dev: walk {name} -- {} base and {} pre columns, {} cells, \
the 8-bit range multiplicities all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.g.log_rows);
    }
    WalkRegion { name: name.into(), lay, mats: mat_idx.to_vec(), act_bits, pre, base,
                 n_weights: mat_idx.iter().map(|i| layer.mats[*i].qw.len()).sum(),
                 dev: None }
}

/// `a - b`, entry by entry: what the *device* added on top of what the host
/// left, so the two deltas can be compared without a second snapshot type.
fn sub_slice(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

pub struct WalkInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed: Vec<SecureField>,
    pub use_sum: SecureField,
    /// Stream II: the group's claim over its own committed activations, in the
    /// canonical channel-major order, or zero when the AIR carries none.
    pub qx_claim: SecureField,
}

pub fn walk_interaction(layer: &Layer, r: &WalkRegion, rv: SecureField, s_ch: SecureField,
                        rels: &crate::compose::Rels, range_check: bool,
                        air: &crate::walk_air::WalkEval, gkr: bool) -> WalkInteraction {
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let lay = r.lay;
    let rows = 1usize << lay.g.log_rows;
    let m0 = &layer.mats[r.mats[0]];
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, lay.pk);
    let (s_pow, s_step) = crate::attn_air::powers(s_ch, lay.pq);

    // B7a milestone 4 (section 3 row 1): the walks' v, us and accs, on the same
    // three-value mode every fill in this series uses.
    let m4g = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let m4g_host = m4g.is_none() || crate::devacc::mode() == 2;
    let rows = if m4g_host { rows } else { 0 };
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
    let claimed: Vec<SecureField> = match m4g {
        Some(d) => {
            let got = crate::devacc::walk(d, &lay, r.mats.len(), rv, s_ch);
            let hh = m4g_host.then(|| {
                let mut e = crate::attn_trace::secure_cols_pub(lay.g.log_rows, &v);
                for m in 0..r.mats.len() {
                    e.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &us[m]));
                    e.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &accs[m]));
                }
                (e, accs.iter().map(|a| a[rows - 1]).collect::<Vec<_>>())
            });
            crate::devacc::place(&format!("walk {} 2^{}", r.name, lay.g.log_rows), got,
                                 hh, &mut cols)
        }
        None => {
            cols.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &v));
            for m in 0..r.mats.len() {
                cols.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &us[m]));
                cols.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &accs[m]));
            }
            accs.iter().map(|a| a[rows - 1]).collect()
        }
    };
    // stream II: the activation claim, in the AIR's own mask order -- after the
    // per-matmul accumulators and before the LogUp columns.  It is built from
    // `m0.qx` and not from `r.base`, so it is the same column whether the fill
    // left the region on the host or on the card, and it does not depend on
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
        cols.extend(crate::attn_trace::secure_cols_pub(lay.g.log_rows, &xs));
    }
    let mut use_sum = SecureField::zero();
    // B11 stream V, lever 4: under `--gkr-walks` this group's range checks were
    // handed to the collector before the GKR proof ran, in `compose`'s own pass,
    // so there are no running-sum columns to build and no claimed sum to carry.
    // The `v`, `u` and `acc` columns above stay: they are Horner accumulators
    // and 38.9% of this family's interaction cells, which the forecast priced.
    if range_check && !gkr {
        let one = PackedQM31::broadcast(SecureField::one());
        let fr = |i: usize, vr: usize|
            (one, rels.range(r.act_bits).combine(&[r.base[i].data[vr]]));
        let (lcols, s) = crate::devfrac::frac_cols(
            air, &format!("walk {}", r.name), lay.g.log_rows,
            crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
            crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
            || {
                let (n_pairs, fpair) = crate::plogup::paired(lay.pq, fr);
                crate::plogup::build(lay.g.log_rows, n_pairs, fpair)
            });
        cols.extend_any(lcols);
        use_sum = s;
    }
    WalkInteraction { cols: cols.into_inner(), claimed, use_sum, qx_claim }
}
