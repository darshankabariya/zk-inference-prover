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

use crate::attn_witness::{Attn, Src, Variant};
use crate::bf16;
use crate::common::{Eval, eval_of, m31_u32};
use crate::dequant::{Map, Spec, Variant as DeqVariant};
use crate::layout::{Grid, Shape};

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
}

impl Mat {
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
}

fn read_u16(path: &str) -> Result<Vec<u16>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
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
                .map(|v| v[..tokens * w].to_vec())
        };
        // `r` is one value per token, not per channel
        let rd1 = |n: &str| -> Result<Vec<u16>, String> {
            let v = read_u16(&format!("{dir}/{n}.u16"))?;
            if v.len() < tokens {
                return Err(format!("{n}: {} values is under {tokens}", v.len()));
            }
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
        }))
    }
}

fn read_i32(path: &str) -> Result<Vec<i64>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(raw.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64).collect())
}
fn read_i8(path: &str) -> Result<Vec<i32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(raw.iter().map(|b| *b as i8 as i32).collect())
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
        other => panic!("unknown dequant layout {other} (b65 | b10)"),
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
        let man: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json"))
                .map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
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
        let mut bands = vec![];
        let mut head_offsets = vec![];
        let mut sizes = vec![];
        let mut off = 0usize;
        for insts in band_sizes {
            let sh = Shape { s, c, hd, insts: *insts };
            let src = Src { head: off, ..cfg.clone() };
            bands.push(Attn::load(dir, off, sh, variant, src.nmax, src.log_table, master,
                                  false)?);
            head_offsets.push(off);
            sizes.push(*insts);
            off += insts;
        }
        if has_corner {
            let mut off = 0usize;
            for insts in band_sizes {
                let sh = Shape { s, c: s, hd, insts: *insts };
                let src = Src { head: off, ..cfg.clone() };
                bands.push(Attn::load(dir, off, sh, variant, src.nmax, src.log_table, master,
                                      true)?);
                head_offsets.push(off);
                sizes.push(*insts);
                off += insts;
            }
            // the per-token region belongs to the pair, and the denominator it
            // commits is the total: `Z >= nmax` is true of the sum and of neither
            // half, because the key whose shift is zero may be in either
            let n_pre = band_sizes.len();
            for b in 0..n_pre {
                let tot: Vec<i64> = bands[b].zz.iter().zip(&bands[b + n_pre].zz)
                    .map(|(x, y)| x + y).collect();
                bands[b].zz_total = Some(tot);
            }
            // Rule 21, on the join: the prover recomputes both halves' numerators
            // from `qq`, `kk`, `vv` and the dump's row maximum, and the extractor
            // computed the same thing in torch.  `zz.i32` is the extractor's total,
            // so this is the two implementations agreeing about the causal
            // computation -- and it is the only check that would catch a mask off
            // by one, which is self-consistent on either side alone.
            let zz32 = read_i32(&format!("{dir}/zz.i32"))?;
            let mut bad = 0usize;
            for (b, insts) in band_sizes.iter().enumerate() {
                let tot = bands[b].zz_total.as_ref().unwrap();
                for i in 0..*insts {
                    let h = head_offsets[b] + i;
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
        let band_sizes = &sizes[..];
        // one master table serves every band: same rows, same contents.  With a
        // layer-wide `master` this holds by construction; without one (a dump whose
        // Q scale is not per head) it is the old precondition, and the message
        // names the fix.
        for b in &bands[1..] {
            assert_eq!(b.log_table, bands[0].log_table, "bands disagree on table rows");
            assert_eq!(b.table, bands[0].table,
                       "bands disagree on the exp table (no layer-wide master exponent: \
                        the dump has frac_shared_per_head = false)");
            assert_eq!(b.frac, bands[0].frac, "bands disagree on the master exponent");
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
                    other => return Err(format!("manifest: {name} has dequant {other}")),
                });
            }
            v
        } else {
            MAT_VARIANTS
        };
        let mut mats = vec![];
        for (name, var) in vars {
            let acc = read_i32(&format!("{dir}/acc_{name}.i32"))?;
            let qx = read_i8(&format!("{dir}/qx_{name}.i8"))?;
            let qw = read_i8(&format!("{dir}/qw_{name}.i8"))?;
            let am = &man["accumulators"][name];
            let (n_in, n_out, tokens) = (am["n_in"].as_u64().unwrap() as usize,
                                         am["n_out"].as_u64().unwrap() as usize,
                                         am["tokens"].as_u64().unwrap() as usize);
            assert_eq!(acc.len(), tokens * n_out, "{name}: acc shape");
            assert_eq!(qx.len(), tokens * n_in, "{name}: qx shape");
            assert_eq!(qw.len(), n_in * n_out, "{name}: qw shape");
            let jx = read_i32(&format!("{dir}/jx_{name}.i32"))?.iter().map(|v| *v as i32).collect();
            let jw = read_i32(&format!("{dir}/jw_{name}.i32"))?.iter().map(|v| *v as i32).collect();
            let pair = |v: Vec<i64>| -> Vec<(u32, u32)> {
                v.chunks_exact(2).map(|p| (p[0] as u32, p[1] as u32)).collect()
            };
            let sx = pair(read_i32(&format!("{dir}/sx_{name}.i32"))?);
            let sw = pair(read_i32(&format!("{dir}/sw_{name}.i32"))?);
            mats.push(Mat { name: name.into(), variant: var, n_in, n_out, tokens,
                            acc, qx, qw, jx, jw, sx, sw });
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
            let jx = read_i32(&format!("{dir}/jx_{name}.i32"))?.iter().map(|v| *v as i32)
                .collect();
            let jw = read_i32(&format!("{dir}/jw_{name}.i32"))?.iter().map(|v| *v as i32)
                .collect();
            let pair = |v: Vec<i64>| -> Vec<(u32, u32)> {
                v.chunks_exact(2).map(|p| (p[0] as u32, p[1] as u32)).collect()
            };
            let sx = pair(read_i32(&format!("{dir}/sx_{name}.i32"))?);
            let sw = pair(read_i32(&format!("{dir}/sw_{name}.i32"))?);
            mats.push(Mat { name: name.into(), variant: var, n_in, n_out, tokens,
                            acc, qx, qw, jx, jw, sx, sw });
        }

        // the attnout region proves what the bands prove -- which with the corner
        // is two bands per head, so a head's totals accumulate
        let (mut acc, mut den) = (vec![0i64; nh * s * hd], vec![0i64; nh * s]);
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
        let jo: Vec<i32> = read_i32(&format!("{dir}/jo.i32"))?.iter().map(|v| *v as i32).collect();
        let jv: Vec<i32> = read_i32(&format!("{dir}/jv.i32"))?.iter().map(|v| *v as i32).collect();
        assert_eq!(jo.len(), s, "jo shape");
        assert_eq!(jv.len(), nh * hd, "jv shape");
        let glue = GlueCols::load(dir, &man, s)?;
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
        Ok(Self {
            bands,
            band_sizes: band_sizes.to_vec(),
            head_offsets,
            mats,
            ao: AttnOut { nh, s, hd, acc, den, jo, jv },
            source: format!("B6.5 layer witness [{dir}]"),
            glue,
            head_keys,
            head_tok,
        })
    }

    /// A small, self-consistent synthetic layer, for the container selftest.
    pub fn synthetic(s: usize, c: usize, hd: usize, band_sizes: &[usize], variant: Variant,
                     cfg: &Src, n_in: usize, n_out: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
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
            head_offsets.push(off);
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
                bands.push(p);
                corner_bands.push(cc);
            } else {
                bands.push(Attn::synthetic_shared(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax,
                                                  log_table, seed ^ (off as u64 + 1),
                                                  frac, false));
            }
            sizes.push(*insts);
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
            });
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
        Self { bands, band_sizes: band_sizes_all, head_offsets, mats,
               ao: AttnOut { nh, s, hd, acc, den, jo, jv },
               source: format!("synthetic layer (seed={seed})"), glue: None,
               head_keys: None, head_tok: None }
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
    pub m_r: Vec<Vec<u32>>,
    pub off_table: usize,
    /// B9: which table lost a row, how many, and the range of the values that
    /// did -- `off_table` alone cannot tell the two failure modes apart
    pub off_kinds: std::collections::BTreeMap<String, (usize, i64, i64)>,
}

impl Tables {
    pub fn new(exp: Vec<u32>, log_exp: u32, kmax: u32, round: bf16::Round, q_bits: u32,
               widths: Vec<u32>) -> Self {
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
pub fn mat_region(m: &Mat, spec: &Spec, tables: &mut Tables, oside: bool) -> MatRegion {
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
    let m4c = crate::util::m4b_mode("B7_M4_FILL");
    let m4c_want = m4c == 2;
    if m4c == 1 {
        let dev = crate::devfill::mat_region_dev(m, &spec, &lay, &map, tables, false);
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
            let mut ref_bad = 0usize;
            let mut want = vec![0u16; if m4c_want { len * lay.per_row } else { 0 }];
            for jj in 0..len {
                let j = start + jj;
                fill_mat_row(m, &spec, &lay, &map, tref, j, jj, &mut local,
                             &mut bn, &mut bf2, &mut bm, &mut br, &mut ref_bad,
                             &mut want);
            }
            let collect = |h: std::collections::HashMap<usize, u32>| {
                h.into_iter().collect::<Vec<_>>()
            };
            (local, Bumps { norm: collect(bn), frac: collect(bf2), mul: collect(bm),
                            r8: br.into_iter().collect(), ref_bad, want })
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
        let first = |j: usize| matches!(lay.at_opt(m.tokens, j),
                                        Some((_, b)) if b + 1 == lay.chunks);
        pre.push(sel(&|j| j > 0 && lay.at_opt(m.tokens, j).is_some() && !first(j)));
        pre.push(sel(&|j| j > 0 && first(j)));
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
        let dev = crate::devfill::mat_region_dev(m, &spec, &lay, &map, tables, true);
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
    match spec.variant {
        DeqVariant::Pow2 => {
            let jjx = m.jx[t] as i64 + m.jw[c] as i64;
            bf16::from_f64(a as f64 * (jjx as f64).exp2())
        }
        DeqVariant::Chain => {
            let y0 = bf16::to_f32(bf16::from_f64(a.unsigned_abs() as f64)) as f64;
            let sxv = bf16::to_f32(bf16::pattern(0, m.sx[t].1, m.sx[t].0)) as f64;
            let swv = bf16::to_f32(bf16::pattern(0, m.sw[c].1, m.sw[c].0)) as f64;
            let t1 = bf16::to_f32(bf16::from_f64(y0 * sxv)) as f64;
            bf16::from_f64(t1 * swv) | (u16::from(a < 0) << 15)
        }
        _ => unreachable!("matmul regions are pow2 or chain"),
    }
}

/// One row of a matmul dequant region: the witness fields of `per_row`
/// accumulators, written into a block-local column slice, with table
/// multiplicities accumulated block-locally (merged by `mat_region`).
#[allow(clippy::too_many_arguments)]
fn fill_mat_row(m: &Mat, spec: &Spec, lay: &MatLayout, map: &Map, tables: &Tables,
                j: usize, jj: usize, local: &mut [Vec<M31>],
                bn: &mut std::collections::HashMap<usize, u32>,
                bf2: &mut std::collections::HashMap<usize, u32>,
                bm: &mut std::collections::HashMap<usize, u32>,
                br: &mut std::collections::HashMap<i64, u32>,
                ref_bad: &mut usize, wref: &mut [u16]) {
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
}

pub fn mat_interaction(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
                       rels: &crate::compose::Rels, out: Option<&OSideW>) -> MatInteraction {
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, lay.per_row);
    // B7a milestone 4 (section 3 row 1): `z` and `h` where the region's own
    // accumulator columns are. `B7_M4_ACC=2` runs both and asserts every cell.
    let m4g = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let m4g_host = m4g.is_none() || crate::devacc::mode() == 2;
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
        let mut u = vec![SecureField::zero(); rows];
        for j in 0..rows {
            let mut f = SecureField::zero();
            if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                for s in 0..lay.per_row {
                    let pat = crate::glue::canon(deq_ref(m, &r.spec, t, b * lay.per_row + s));
                    f += match o.kind {
                        crate::dequant::OKind::Pair => {
                            let (sm, ef) = crate::glue::split(pat);
                            (SecureField::from(m31_of(sm))
                             + SecureField::from(m31_u32(ef)) * o.lam) * o.w[s]
                        }
                        // the same key the argmax region compares, off the same
                        // host function the extractor's column was built with
                        crate::dequant::OKind::Key => SecureField::from(
                            m31_u32(crate::argmax::order_key(pat))) * o.w[s],
                    };
                }
            }
            // the multiplier is the step *into* this row: an ordinary chunk step
            // inside a token, the boundary jump on a row that begins one
            let mult = if j == 0 {
                SecureField::zero()
            } else {
                match lay.at_opt(m.tokens, j) {
                    Some((_, b)) if b + 1 == lay.chunks => o.k_bd,
                    Some(_) => o.k_in,
                    None => SecureField::zero(),
                }
            };
            u[j] = if j == 0 { f } else { u[j - 1] * mult + f };
        }
        let claimed = u[rows - 1];
        cols.extend(crate::attn_trace::secure_cols_pub(lay.log_rows, &u)
            .into_iter().map(crate::device::AnyCol::OwnedHost));
        claimed
    });

    // the LogUp columns, in the exact order the AIR adds lookups
    let m4_t = std::time::Instant::now();
    let (lcols, use_sum) = mat_logup(m, r, rv, s_ch, rels);
    crate::util::m4_add("int/deq_logup", m4_t.elapsed().as_secs_f64());
    cols.extend(lcols);
    MatInteraction { cols, claimed_h, use_sum, claimed_out }
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
fn mat_logup_ir(m: &Mat, r: &MatRegion, rv: SecureField, s_ch: SecureField,
                rels: &crate::compose::Rels, on_device: bool)
                -> (Vec<crate::device::AnyCol<'static>>, SecureField) {
    use stwo::prover::backend::simd::m31::PackedM31;
    let sub: Vec<crate::attn_air::RangeRel> = crate::dequant::range_widths(&r.spec)
        .iter()
        .map(|w| rels.range(*w).clone())
        .collect();
    let (rv_pow, rv_step) = crate::attn_air::powers(rv, r.lay.per_row);
    let eval = crate::dequant::DequantEval {
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
    };
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
        if let Some(d) = &r.dev {
            // B7a milestone 4 (19.6): the fill left this region's columns on
            // the device, so the fractions read them where they are and the
            // LogUp columns never come back. 19.4 measured both transfers and
            // said they disappear together; this is the together.
            let (cols, sum) = crate::devlogup::logup_cols_dev(
                &ir, family, shape, r.lay.log_rows, &d.pre, &d.base);
            return (cols.into_iter()
                        .map(|c| crate::device::AnyCol::OwnedDev(c, r.lay.log_rows))
                        .collect(),
                    sum);
        }
        let (cols, sum) = crate::devlogup::logup_cols_dev_to_host(
            &ir, family, shape, r.lay.log_rows, &r.pre, &r.base);
        return (crate::device::host_cols(cols), sum);
    }
    let pre: Vec<&[PackedM31]> = r.pre.iter().map(|e| &e.data[..]).collect();
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
    let m4b = crate::util::m4b_mode("B7_M4_LOGUPIR");
    let from_ir = (m4b > 0).then(|| mat_logup_ir(m, r, rv, s_ch, rels, m4b >= 3));
    // 2 checks the interpreter against the hand-written builder, 4 the kernel.
    if m4b == 1 || m4b == 3 {
        return from_ir.unwrap();
    }
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let map = &r.map;
    let f = map.f;
    let sp = &r.spec;
    let sticky = sp.round.needs_sticky();
    let one = PackedQM31::broadcast(SecureField::one());
    // pre layout: per-slot chan cols first (1 or 2 per slot), then selectors
    let chan_w = if sp.variant == DeqVariant::Chain { 2 } else { 1 };
    enum L {
        Norm([Src2; 6]),
        Frac([Src2; 3]),
        Mul([Src2; 4], Src2),
        Range(u32, Src2),
    }
    #[derive(Clone, Copy)]
    enum Src2 {
        B(usize),
        P(usize),
        Zero,
    }
    let mut looks: Vec<L> = vec![];
    for s in 0..sp.per_row {
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
}

impl AoRegion {
    /// Committed cells, wherever the columns are.
    pub fn base_cells(&self) -> usize {
        let n = match &self.dev { Some(d) => d.base.len(), None => self.base.len() };
        n << self.lay.log_rows
    }
}

/// Column order per slot: acc, sg, qm, nz2, denp, d (after the shared den col).
pub fn ao_region(ao: &AttnOut, per_row_target: usize, q_bits: u32, rem_bits: u32,
                 tables: &mut Tables, corrupt: Option<usize>) -> AoRegion {
    let lay = AoLayout::choose(ao.nh, ao.s, ao.hd, per_row_target);
    let rows = 1usize << lay.log_rows;
    let pr = lay.per_row;
    // B7a milestone 4 (section 3 row 1): the same three-value mode.
    let m4d = crate::util::m4b_mode("B7_M4_WFILL");
    assert!(m4d == 0 || corrupt.is_none(),
            "attnout: the device fill has no `corrupt` path");
    if m4d == 1 {
        let dev = crate::devwfill::ao_region_dev(ao, &lay, q_bits, rem_bits, tables, false);
        let (ref_bad, q_out) = (dev.ref_bad, dev.q_out.clone());
        return AoRegion { lay, pre: vec![], base: vec![], dev: Some(dev), ref_bad, q_out,
                          n: ao.nh * ao.s * ao.hd, rem_bits, q_bits };
    }
    let m4d_snap = |t: &Tables| -> (Vec<u32>, Vec<u32>, usize) {
        (t.m_qnz.clone(), t.m_r[t.ridx(rem_bits)].clone(), t.off_table)
    };
    let m4d_before = (m4d == 2).then(|| m4d_snap(tables));
    let mut want_host = vec![0u32; if m4d == 2 { rows * pr } else { 0 }];
    let n_cols = 1 + 6 * pr;
    let mut cols: Vec<Vec<M31>> = vec![vec![M31::zero(); rows]; n_cols];
    let mut q_out = vec![0i64; ao.nh * ao.s * ao.hd];
    let mut ref_bad = 0usize;
    for j in 0..rows {
        match lay.at(ao.nh, ao.s, j) {
            None => {
                // inert lead row: den = 1, everything else 0 -> d = denp = 1,
                // dp = 0, (qm, nz2) = (0, 0); all constraints hold, all lookups
                // land on real rows
                cols[0][j] = M31::one();
                for s in 0..pr {
                    cols[1 + 6 * s + 4][j] = M31::one();     // denp = den * pj = 1
                    cols[1 + 6 * s + 5][j] = M31::one();     // d = denp
                    tables.bump_qnzk(0, "ao_qnz_lead");
                    tables.bump_rk(rem_bits, 1, "ao_rem_lead");    // d
                    tables.bump_rk(rem_bits, 0, "ao_rem_lead");    // dp = 2*1 - 1 - 1
                }
            }
            Some((h, t, b)) => {
                let den = ao.den[h * ao.s + t];
                assert!(den > 0, "attnout denominator must be positive");
                cols[0][j] = m31_of(den);
                for s in 0..pr {
                    let ch = b * pr + s;
                    let idx = (h * ao.s + t) * ao.hd + ch;
                    let mut a = ao.acc[idx];
                    if corrupt == Some(idx) {
                        a += 1;
                    }
                    let e = ao.e(h, t, ch);
                    let num = 1i64 << e.max(0);
                    let pj = 1i64 << (-e).max(0);
                    let denp = den * pj;
                    let mag = a.abs();
                    let pm = mag * num;
                    // the no-wrap window: every term of the error identity must be
                    // an integer below p, or the field reading is ambiguous
                    assert!(2 * pm < (1i64 << 31) - 1 && 2 * denp < (1i64 << 31) - 1,
                            "attnout fold outgrows the field: pm={pm} denp={denp}");
                    let qm = (2 * pm + denp) / (2 * denp);
                    let sg = i64::from(a < 0 && qm != 0);
                    let d = if a < 0 && qm == 0 { denp - 2 * pm }
                            else { 2 * pm - (2 * qm - 1) * denp };
                    let c0 = 1 + 6 * s;
                    cols[c0][j] = m31_of(a);
                    cols[c0 + 1][j] = m31_of(sg);
                    cols[c0 + 2][j] = m31_of(qm);
                    cols[c0 + 3][j] = m31_of(i64::from(qm != 0));
                    cols[c0 + 4][j] = m31_of(denp);
                    cols[c0 + 5][j] = m31_of(d);
                    tables.bump_qnzk(qm, "ao_qnz");
                    tables.bump_rk(rem_bits, d, "ao_rem_d");
                    tables.bump_rk(rem_bits, 2 * denp - 1 - d, "ao_rem_dp");
                    q_out[idx] = if a < 0 { -qm } else { qm };
                    // the reference: one f64 rounding, half away from zero
                    let want = ((mag as f64 * (e as f64).exp2()) / den as f64 + 0.5).floor()
                        as i64;
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
    // preprocessed: per-slot (num, pj), then the selectors
    let mut pre = vec![];
    for s in 0..pr {
        for pick in 0..2 {
            pre.push(eval_of(lay.log_rows, (0..rows).map(|j| {
                match lay.at(ao.nh, ao.s, j) {
                    None => M31::one(),
                    Some((h, t, b)) => {
                        let e = ao.e(h, t, b * pr + s);
                        let v = if pick == 0 { 1i64 << e.max(0) } else { 1i64 << (-e).max(0) };
                        m31_of(v)
                    }
                }
            }).collect()));
        }
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
        let dev = crate::devwfill::ao_region_dev(ao, &lay, q_bits, rem_bits, tables, true);
        crate::devwfill::check_cols("attnout base", &base, &dev.base);
        crate::devwfill::check_cols("attnout pre", &pre, &dev.pre);
        let ri = tables.ridx(rem_bits);
        crate::devwfill::check_delta("attnout m_qnz", &before.0, &after.0,
                                     &sub_slice(&tables.m_qnz, &after.0));
        crate::devwfill::check_delta("attnout m_rem", &before.1, &after.1,
                                     &sub_slice(&tables.m_r[ri], &after.1));
        assert_eq!(tables.off_table - after.2, after.2 - before.2,
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
        tables.m_qnz.copy_from_slice(&after.0);
        tables.m_r[ri].copy_from_slice(&after.1);
        tables.off_table = after.2;
        eprintln!("wfill-dev: attnout -- {} base and {} pre columns, {} cells, \
both multiplicity tables and the f64 reference all match",
                  base.len(), pre.len(), (base.len() + pre.len()) << lay.log_rows);
    }
    AoRegion { lay, pre, base, dev: None, ref_bad, q_out, n: ao.nh * ao.s * ao.hd,
               rem_bits, q_bits }
}

pub struct AoInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    pub claimed_out: SecureField,
    pub claimed_den: SecureField,
    pub use_sum: SecureField,
}

#[allow(clippy::too_many_arguments)]
pub fn ao_interaction(ao: &AttnOut, r: &AoRegion, rr: SecureField, s_ch: SecureField,
                      rv: SecureField, c_ctx: usize, rels: &crate::compose::Rels,
                      air: &crate::walk_air::AttnOutEval) -> AoInteraction {
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

    // lookups per slot, in AIR order: (qm, nz2), d, dp
    let one = PackedQM31::broadcast(SecureField::one());
    let n_looks = 3 * pr;
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        let (s, which) = (i / 3, i % 3);
        let c0 = 1 + 6 * s;
        match which {
            0 => (one, rels.qnz.combine(&[r.base[c0 + 2].data[vr], r.base[c0 + 3].data[vr]])),
            1 => (one, rels.range(r.rem_bits).combine(&[r.base[c0 + 5].data[vr]])),
            _ => {
                let two = stwo::prover::backend::simd::m31::PackedM31::broadcast(
                    M31::from_u32_unchecked(2));
                let onem = stwo::prover::backend::simd::m31::PackedM31::broadcast(M31::one());
                let dp = r.base[c0 + 4].data[vr] * two - onem - r.base[c0 + 5].data[vr];
                (one, rels.range(r.rem_bits).combine(&[dp]))
            }
        }
    };
    let (lcols, use_sum) = crate::devfrac::frac_cols(
        air, "attnout", lay.log_rows,
        crate::devfrac::cols_of(&r.pre, r.dev.as_ref().map(|d| &d.pre[..])),
        crate::devfrac::cols_of(&r.base, r.dev.as_ref().map(|d| &d.base[..])),
        || {
            let (n_pairs, fpair) = crate::plogup::paired(n_looks, fr);
            crate::plogup::build(lay.log_rows, n_pairs, fpair)
        });
    cols.extend_any(lcols);
    AoInteraction { cols: cols.into_inner(), claimed_out, claimed_den, use_sum }
}

// ------------------------------------------------------------------ walk regions --

pub struct WalkRegion {
    pub name: String,
    pub lay: WalkLayout,
    pub mats: Vec<usize>,
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

/// Build one walk group's region: activations committed (offset +128), weights
/// preprocessed, byte-range multiplicities bumped.
pub fn walk_region(layer: &Layer, name: &str, mat_idx: &[usize], p_target: usize,
                   tables: &mut Tables, corrupt_qx: Option<usize>) -> WalkRegion {
    let m0 = &layer.mats[mat_idx[0]];
    let n_in = m0.n_in;
    let tokens = m0.tokens;
    let n_out_max = mat_idx.iter().map(|i| layer.mats[*i].n_out).max().unwrap();
    let lay = WalkLayout::choose(n_in, n_out_max, tokens, p_target);
    let rows = 1usize << lay.g.log_rows;
    for i in mat_idx {
        assert_eq!(layer.mats[*i].n_in, n_in, "walk group mixes input widths");
        // q/k/v (and gate/up) must share one committed input
        assert_eq!(layer.mats[*i].qx, m0.qx, "walk group inputs differ");
    }

    // B7a milestone 4 (section 3 row 1): which fill runs. 1 is the device's
    // alone -- `pre` and `base` become the mirror the interaction reads -- and
    // 2 builds both and asserts every cell equal before either is used.
    let m4d = crate::util::m4b_mode("B7_M4_WFILL");
    assert!(m4d == 0 || corrupt_qx.is_none(),
            "{name}: the device walk fill has no `corrupt_qx` path");
    if m4d == 1 {
        let dev = crate::devwfill::walk_region_dev(layer, mat_idx, &lay, tables);
        // B7a milestone 4 (section 3 row 1): no mirror. `walk_interaction` reads
        // the device columns now, and nothing else reads a device region's host
        // columns -- `base_cells` counts them off `dev`.
        return WalkRegion { name: name.into(), lay, mats: mat_idx.to_vec(),
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
                            x = if x < 127 { x + 1 } else { x - 1 };
                        }
                        x + 128
                    } else {
                        128
                    }
                }
                None => 128,
            };
            tables.bump_r(8, v as i64);
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
    WalkRegion { name: name.into(), lay, mats: mat_idx.to_vec(), pre, base,
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
}

pub fn walk_interaction(layer: &Layer, r: &WalkRegion, rv: SecureField, s_ch: SecureField,
                        rels: &crate::compose::Rels, range_check: bool,
                        air: &crate::walk_air::WalkEval) -> WalkInteraction {
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
    let mut use_sum = SecureField::zero();
    if range_check {
        let one = PackedQM31::broadcast(SecureField::one());
        let fr = |i: usize, vr: usize| (one, rels.range(8).combine(&[r.base[i].data[vr]]));
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
    WalkInteraction { cols: cols.into_inner(), claimed, use_sum }
}
