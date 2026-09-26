//! The columns a dump carries for the wiring family, and the synthetic ones.
//!
//! This is the witness side of `residual_stream`: the seven columns
//! `extract.py --residual-stream` writes, the two the head chunk writes instead,
//! and the synthetic generators the desk proves the composed statement with.
//! The byte readers stay in `layer.rs`, which is where the rest of a dump is
//! read.

use crate::bf16;
use crate::layer::{model_u16, read_u16};
use crate::witread::{self, Site};

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
