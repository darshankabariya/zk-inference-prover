//! The data the packed attention region proves: `insts` heads' quantized Q, K, V
//! and everything the integer softmax derives from them.
//!
//! Every array is `(instance, ...)` row-major, and every derived value is
//! recomputed here by exact integer arithmetic, so the honest witness is by
//! construction what the AIR must prove and a corruption can be injected after
//! the fact.
//!
//! **The exp table across heads.**  The numerator is
//! `n = round(nmax exp(-d 2^-frac))` with `frac` a per-(layer, head) constant, so a
//! packed proof either carries one table per head -- which is what "the table
//! amortises across heads" quietly assumed -- or one *master* table at the largest
//! `frac = F` and looks it up at the scaled key `d 2^(F - frac_i)`.  The scaled key
//! is what this witness commits; the AIR multiplies it back by a preprocessed
//! `2^-(F - frac_i)` inside the score accumulator, so the sharing costs one
//! preprocessed column and no fresh cell.  What it *does* cost is table rows: the
//! master table is sized by the head with the smallest `frac`, so the spread of
//! `frac_per_head` is the number that decides whether sharing is affordable.
//!
//! Real data comes from `b5/extract.py` (B1's forward pass under an
//! integer-softmax spec, one layer, the last prefill chunk); synthetic data
//! stands in for the contexts the 0.5B model was not run at.  Either way the
//! derived values -- scores, the shift, the numerators, the denominator and the
//! A.V outputs -- are recomputed here by exact integer arithmetic, so the honest
//! witness is by construction the thing the AIR has to prove and a corruption can
//! be injected after the fact.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::bf16;
use crate::layout::Shape;
use crate::witread::{self, Site};

/// A raw column, read and widened.  `site` says which decode the widening is,
/// so that the two widths this is called at are visible apart.
fn read<T: Copy + Send>(path: &str, width: usize, site: Site,
                        conv: impl Fn(&[u8]) -> T + Send + Sync) -> Result<Vec<T>, String> {
    let raw = witread::read_file(path)?;
    let _s = witread::scope(site);
    Ok(raw.par_chunks_exact(width).map(conv).collect())
}

/// One half of a layer's attention witness: the keys, the values and torch's
/// own numerators, over either the cache's context or the causal corner.
pub struct Half {
    pub kk16: Vec<i16>,
    pub vv8: Vec<u8>,
    pub num32: Vec<i32>,
}

impl Half {
    /// The three files of one half, read and widened once.  `tag` is the
    /// suffix the corner's copies carry (`kkc.i16` against `kk.i16`).
    fn read(dir: &str, k: &str, v: &str, n: &str) -> Result<Self, String> {
        Ok(Self {
            kk16: read(&format!("{dir}/{k}"), 2, Site::DecodeI16,
                       |b| i16::from_le_bytes([b[0], b[1]]))?,
            vv8: witread::read_file(&format!("{dir}/{v}"))?,
            num32: read(&format!("{dir}/{n}"), 4, Site::DecodeI32,
                        |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))?,
        })
    }
}

/// Stream W: one layer's attention columns, read and widened **once**.
///
/// A layer's six bands -- three over the cache's context, three over the causal
/// corner -- each called `Attn::load`, and each of them opened `qq.i16`, its
/// own half's `kk`, `vv` and `num`, and `mm.i32` for itself before gathering
/// its slice out of them.  At the sequence's sizes that made a chunk read
/// **1,502 MB where 740 MB exists**, with all 268 MB of `num.i32` read three
/// times over, and the read was the largest single site left after stream P
/// (19.0 s of an 81.9 s pipelined build, `briefs/p-witread.md` (6) item 1).
///
/// This is not a format change and it does not touch `extract.py`: the same
/// files hold the same bytes in the same layout, and the same gathers run over
/// the same images.  What changed is how many times each image is built.
/// `Layer::load` reads one of these and hands it to every band; the bands
/// gather out of it exactly as they gathered out of their private copies, so
/// every array `Attn::build` receives is element-for-element the array it
/// received before.
pub struct LayerRaw {
    /// The queries.  Both halves of a band read the same `qq.i16` -- the corner
    /// is the same tokens attending to a different set of keys -- so this was
    /// read six times a layer.
    pub qq16: Vec<i16>,
    /// The joint row maximum, on a `--corner` dump.  Read by both halves of
    /// every band, for the same reason.
    pub mm32: Option<Vec<i32>>,
    pub pre: Option<Half>,
    pub cor: Option<Half>,
}

impl LayerRaw {
    /// Read what the bands this layer will build actually want.  A dump with no
    /// corner has no second half and no `mm.i32`; a single band asked for on
    /// its own (`Attn::get`'s probe path) reads its own half and no other, so
    /// that path reads exactly the bytes it read before this existed.
    pub fn read(dir: &str, man: &serde_json::Value, prefix: bool, corner: bool)
                -> Result<Self, String> {
        let has_corner = man["corner"].as_bool().unwrap_or(false);
        Ok(Self {
            qq16: read(&format!("{dir}/qq.i16"), 2, Site::DecodeI16,
                       |b| i16::from_le_bytes([b[0], b[1]]))?,
            mm32: has_corner
                .then(|| read(&format!("{dir}/mm.i32"), 4, Site::DecodeI32,
                              |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .transpose()?,
            pre: prefix.then(|| Half::read(dir, "kk.i16", "vv.i8", "num.i32")).transpose()?,
            cor: corner.then(|| Half::read(dir, "kkc.i16", "vvc.i8", "numc.i32"))
                .transpose()?,
        })
    }

    fn half(&self, corner: bool) -> Result<&Half, String> {
        let h = if corner { &self.cor } else { &self.pre };
        h.as_ref().ok_or_else(|| {
            format!("the layer image holds no {} half",
                    if corner { "corner" } else { "prefix" })
        })
    }
}

/// The exp table the last band asked for, kept so the next band does not derive
/// it again.
///
/// Stream P: a layer's bands all read *one* master table -- same rows, same
/// exponent, same contents, which `Layer::load` asserts right after building
/// them -- and each of them was deriving all 2^23..2^27 rows of it for itself.
/// At the sequence's sizes that was two thirds of the cost of loading a layer
/// (`briefs/p-witread.md`, its Record).
///
/// The key is the whole of what the table is a function of, so this changes
/// which work happens and nothing about which values come out: a band that
/// resolved a different exponent or a different row count misses and derives
/// its own, and `Layer::load`'s cross-band assert still compares two separate
/// arrays -- one derived, one copied -- and still fires when the two disagree.
/// One entry is the right size: a layer's six bands ask for the same table in a
/// row, and the chunk after them wants a different one.
static TABLE_MEMO: std::sync::Mutex<Option<((usize, u32, u64), Vec<u32>)>> =
    std::sync::Mutex::new(None);

fn table_of(rows: usize, nmax: u32, frac: f64) -> Vec<u32> {
    let key = (rows, nmax, frac.to_bits());
    let mut memo = TABLE_MEMO.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((k, t)) = memo.as_ref() {
        if *k == key {
            return t.clone();
        }
    }
    let t = bf16::exp_int_table(rows, nmax, frac);
    *memo = Some((key, t.clone()));
    t
}

/// Which table the numerators come from, i.e. which column the AIR commits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// commit the score and a bf16 exp-table input; the max shift and the
    /// dynamic-scale product that produced the input are outside the AIR (v1)
    Bf16x,
    /// commit the integer shift `d` and look it up directly: one 2^16 table that
    /// range-checks `d` and yields the numerator in the same lookup
    IntExp,
    /// as IntExp, but `d` is wider than the table: commit a `k`-bit remainder and
    /// index the table with the high part, `d = u * 2^k + rem`
    IntSplit(u32),
}

impl Variant {
    pub fn parse(s: &str) -> Self {
        match s {
            "bf16x" => Variant::Bf16x,
            "intexp" => Variant::IntExp,
            other => match other.strip_prefix("intsplit") {
                Some(k) => Variant::IntSplit(k.parse().expect("intsplit<k>")),
                None => panic!("unknown variant {other:?}"),
            },
        }
    }
    pub fn name(&self) -> String {
        match self {
            Variant::Bf16x => "bf16x".into(),
            Variant::IntExp => "intexp".into(),
            Variant::IntSplit(k) => format!("intsplit{k}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Corrupt {
    /// one quantized key changed: the Q.K^T identity must reject
    Key(usize, i32),
    /// one quantized value changed: the A.V identity must reject
    Value(usize, i32),
    /// one committed numerator replaced by a different, still legal, table output
    Numerator(usize),
    /// one row maximum lowered, so some `d` goes negative and leaves the table
    LowMax(usize),
    /// one row maximum raised: `d` stays non-negative, so only the *numerators*
    /// change -- the probe for what the upper-bound shift really costs
    HighMax(usize, i64),
    /// one committed denominator changed, leaving the sum it claims
    Denominator(usize, i64),
    /// one table row flipped, with the witness rebuilt to match
    Table(usize),
    /// B10 step 3: one *future* key given a legal numerator, so the token attends
    /// to a key the causal mask removes.  The denominator and the A.V output are
    /// rebuilt from it; the AIR reads neither, so `sn` and `av` stop matching the
    /// `az` and `out` this witness claims.  The corner's own probe.
    Unmask(usize),
}

/// Where a witness comes from, and the knobs that are not part of the shape.
#[derive(Clone, Debug)]
pub struct Src {
    pub dir: Option<String>,
    /// first head of the band; the band covers `insts` consecutive heads, cycling
    pub head: usize,
    pub q_bits: u32,
    pub k_bits: u32,
    pub nmax: u32,
    /// rows of the integer variants' exp table
    pub log_table: u32,
    pub seed: u64,
    /// B10 step 0: which dequant gadget each of the seven layer matmuls takes
    /// (`b65` = B6.5's mixed layout, `b10` = the bf16 chain on all seven).  Empty
    /// means "whatever the witness says", which is what a real dump carries.
    pub layout: String,
    /// B10 step 3: build the band as the causal corner (`C == S`, key `c` is
    /// decoded token `c`) rather than as the prefix rectangle.
    pub corner: bool,
    /// Stream AA (E11), from `--kv-bind`: load the declared prefix as `C / S`
    /// bands of `S` keys each instead of one rectangle of `C`.  Implies
    /// `corner`.  See `Attn::load_from`'s `key_off`.
    pub tile: bool,
}

pub struct Attn {
    pub shape: Shape,
    pub variant: Variant,
    /// (S, hd) quantized queries
    pub qq: Vec<i32>,
    /// (C, hd) quantized keys -- the committed cache
    pub kk: Vec<i32>,
    /// (C, hd) quantized values
    pub vv: Vec<i32>,
    /// (S, C) exact integer scores
    pub sc: Vec<i64>,
    /// (S,) committed row maxima (an upper bound on the row's scores)
    pub mm: Vec<i64>,
    /// (S, C) the committed shift, `mm[t] - sc[t][c] >= 0`
    pub dd: Vec<i64>,
    /// (S, C) bf16 patterns of the exp-table input (v1 only)
    pub xx: Vec<u16>,
    /// (S, C) fixed-point numerators, read off the table
    pub num: Vec<i64>,
    /// (S,) denominators, `sum_c num`
    pub zz: Vec<i64>,
    /// (S, hd) A.V outputs
    pub out: Vec<i64>,
    /// the table the numerators come from
    pub table: Vec<u32>,
    pub log_table: u32,
    pub nmax: u32,
    /// fixed-point exponent of the master table (the largest over the band)
    pub frac: f64,
    /// per-instance `frac`, as the forward pass produced it
    pub frac_per_inst: Vec<f64>,
    /// per-instance table-key scale `2^(F - frac_i)`: 1 when every head agrees
    pub delta: Vec<i64>,
    pub q_bits: u32,
    pub k_bits: u32,
    pub source: String,
    /// grid entries whose committed index is outside the table
    pub off_table: usize,
    /// numerators that disagree with the dumped (torch) ones, if there are any
    pub torch_mismatch: Option<usize>,
    /// B10 step 3: this band is the *causal corner* -- key `c` is the decoded
    /// stream's token `c`, so token `t` attends to it iff `c <= t`.  The masked
    /// half still commits its score (Freivalds is over the whole rectangle) and
    /// nothing else: no numerator, no denominator term, no table row.
    pub corner: bool,
    /// B10 step 3: the denominator the *per-token region* commits, when it is not
    /// this band's own `zz`.  With the corner there is one per-token region for
    /// the pair and it carries the total `Z_prefix + Z_corner` -- which is the
    /// only quantity the `Z >= nmax` lower bound holds for, since the key whose
    /// shift is zero may be in either half.
    pub zz_total: Option<Vec<i64>>,
}

pub const LOG_EXP_TABLE: u32 = 16;

impl Attn {
    pub fn off_k(&self) -> i32 {
        1 << (self.k_bits - 1)
    }
    pub fn off_q(&self) -> i32 {
        1 << (self.q_bits - 1)
    }
    /// Instance a flat `(instance, token, key)` index belongs to.
    pub fn inst_of(&self, g: usize) -> usize {
        g / (self.shape.s * self.shape.c)
    }
    /// The table-key scale of the instance a grid entry belongs to.
    pub fn delta_of(&self, g: usize) -> i64 {
        self.delta[self.inst_of(g)]
    }
    /// The value the grid region's first committed column holds.
    ///
    /// For the integer variants this is the shift *scaled to the master table's
    /// exponent*, so one table serves every head in the band; the AIR divides it
    /// back out by a preprocessed constant inside the score accumulator.
    pub fn col_idx(&self, g: usize) -> i64 {
        match self.variant {
            Variant::Bf16x => self.sc[g],
            Variant::IntExp => self.dd[g] * self.delta_of(g),
            Variant::IntSplit(k) => (self.dd[g] * self.delta_of(g)) >> k,
        }
    }
    /// The value the grid region's auxiliary column holds, if it has one.
    ///
    /// The split has to be taken on the *scaled* shift, not the raw one: the AIR
    /// reconstructs `d = (idx 2^k + aux) 2^-(F - frac_i)`, so `idx` and `aux` must be
    /// the high and low parts of the same `d 2^(F - frac_i)`.  With one instance
    /// the scale is 1 and the distinction is invisible, which is exactly why it has
    /// to be got right before the band is packed.
    pub fn aux_of(&self, g: usize) -> i64 {
        match self.variant {
            Variant::Bf16x => self.xx[g] as i64,
            Variant::IntExp => 0,
            Variant::IntSplit(k) => (self.dd[g] * self.delta_of(g)) & ((1i64 << k) - 1),
        }
    }
    /// The value looked up in the exp table.
    pub fn table_key(&self, g: usize) -> i64 {
        match self.variant {
            Variant::Bf16x => self.xx[g] as i64,
            _ => self.col_idx(g),
        }
    }
    /// The value the Q.K^T score accumulator sees: the score itself in v1, the
    /// shift in the integer variants (which the identity then folds through `m`).
    pub fn score_col(&self, g: usize) -> i64 {
        match self.variant {
            Variant::Bf16x => self.sc[g],
            _ => self.dd[g],
        }
    }
    /// Does every instance in the band share one exp-table exponent?
    pub fn table_shared(&self) -> bool {
        self.delta.iter().all(|d| *d == 1)
    }
    /// Did the token of grid entry `g` attend to its key?  Always, unless this band
    /// is the causal corner, where the key axis *is* the decoded token axis.
    pub fn visible(&self, g: usize) -> bool {
        !self.corner || {
            let (s, c) = (self.shape.s, self.shape.c);
            g % c <= (g / c) % s
        }
    }
    /// The denominator column the per-token region commits: the pair's total where
    /// there is one, this band's own otherwise.
    pub fn zz_row(&self) -> &[i64] {
        self.zz_total.as_deref().unwrap_or(&self.zz)
    }
    /// Entries the mask removes, over the whole band.
    pub fn masked(&self) -> usize {
        if self.corner { self.shape.insts * self.shape.s * (self.shape.s - 1) / 2 } else { 0 }
    }
}

impl Attn {
    #[allow(clippy::too_many_arguments)]
    pub fn synthetic(sh: Shape, variant: Variant, q_bits: u32, k_bits: u32, nmax: u32,
                     log_table: u32, seed: u64) -> Self {
        let (qmax_q, qmax_k) = ((1i32 << (q_bits - 1)) - 1, (1i32 << (k_bits - 1)) - 1);
        // Heavy-tailed on purpose.  A dynamic max-scale puts the *maximum* at
        // qmax, so the rms of a real quantized vector is several times smaller,
        // and it is the rms that sets the dot product's spread -- B1 measures
        // max |acc| two orders of magnitude below 127^2 K for exactly this
        // reason.  A uniform draw would put every synthetic score at the worst
        // case and make the exp table's row count meaningless.
        let draw = |rng: &mut SmallRng, m: i32| -> i32 {
            let g: f64 = (0..3).map(|_| rng.random_range(-1.0f64..1.0)).sum();
            ((g * m as f64 / 6.0).round() as i32).clamp(-m, m)
        };
        let mut rng = SmallRng::seed_from_u64(seed);
        let qq = (0..sh.insts * sh.s * sh.hd).map(|_| draw(&mut rng, qmax_q)).collect();
        let kk = (0..sh.insts * sh.c * sh.hd).map(|_| draw(&mut rng, qmax_k)).collect();
        let vv = (0..sh.insts * sh.c * sh.hd).map(|_| draw(&mut rng, 127)).collect();
        Self::build(sh, variant, qq, kk, vv, None, q_bits, k_bits, nmax, log_table, None,
                    None, format!("synthetic(seed={seed})"), None, false, None)
    }

    /// Load `b5/extract.py`'s dump as a band of `insts` heads, cycling if the dump
    /// has fewer, and tiling along S or C if the shape runs past it.
    /// `master`: the layer-wide exponent every band's table is pinned to (B9
    /// step 1a).  `None` keeps the band's own maximum, which is what B6.5 did and
    /// what makes a layer's bands disagree.
    #[allow(clippy::too_many_arguments)]
    pub fn load(dir: &str, head: usize, sh: Shape, variant: Variant, nmax: u32,
                log_table: u32, master: Option<f64>, corner: bool) -> Result<Self, String> {
        let man: serde_json::Value = witread::read_json(&format!("{dir}/manifest.json"))?;
        let raw = LayerRaw::read(dir, &man, !corner, corner)?;
        Self::load_from(&raw, &man, head, sh, variant, nmax, log_table, master, corner, 0)
    }

    /// The same band, out of a layer image its caller already holds.
    ///
    /// Stream W: this is where the signature change lands.  `Layer::load` reads
    /// one `LayerRaw` and one `manifest.json` for the whole layer and calls
    /// this six times; `load` above keeps the one-band entry point for the
    /// probe path, by reading an image of exactly one band's worth.  Neither
    /// path changed a value: the gathers below are the gathers that were here,
    /// over arrays with the same contents.
    ///
    /// Stream AA (E11): `key_off` is where this band's keys start inside the
    /// dump's context.  Zero everywhere before this stream and on every corner
    /// band; with `--kv-bind` the declared prefix is loaded as `C / S` bands of
    /// `S` keys each and `key_off` is `k * S`.  It is a window into the same
    /// bytes -- no format change, no re-extraction, and the derived values are
    /// recomputed from the window exactly as they always were from the whole.
    #[allow(clippy::too_many_arguments)]
    pub fn load_from(raw: &LayerRaw, man: &serde_json::Value, head: usize, sh: Shape,
                     variant: Variant, nmax: u32, log_table: u32, master: Option<f64>,
                     corner: bool, key_off: usize) -> Result<Self, String> {
        let g = |k: &str| man[k].as_u64().ok_or(format!("manifest: missing {k}"));
        let (nh, s0, c0, hd0) = (g("heads")? as usize, g("s")? as usize,
                                 g("c")? as usize, g("hd")? as usize);
        if sh.hd != hd0 {
            return Err(format!("dump has hd={hd0}, need {}", sh.hd));
        }
        let q_bits = man["qk_bits"][0].as_u64().unwrap_or(8) as u32;
        let k_bits = man["qk_bits"][1].as_u64().unwrap_or(8) as u32;
        // the exp-table exponent torch used, per head: a public constant only when
        // the Q scale is shared per head, and the spread across heads is what
        // decides whether one master table can serve the whole band
        let shared = man["frac_shared_per_head"].as_bool().unwrap_or(false);
        let fracs: Option<Vec<f64>> = shared.then(|| {
            (0..sh.insts)
                .map(|i| man["frac_per_head"][(head + i) % nh].as_f64().unwrap_or(0.0))
                .collect()
        });
        let hh = |i: usize| (head + i) % nh;
        // B10 step 3: a dump extracted with `--corner` carries the decoded stream's
        // own keys and values beside the cache's, and one row maximum for the pair.
        // The corner band reads `kkc`/`vvc` and `numc`; both bands read `mm`,
        // because with a corner the maximum is no longer the max over either
        // half's keys and recomputing it per band would give two different ones.
        let has_corner = man["corner"].as_bool().unwrap_or(false);
        if corner && !has_corner {
            return Err("the dump was not extracted with --corner".to_string());
        }
        let c0 = if corner { s0 } else { c0 };
        if corner && sh.c != s0 {
            return Err(format!("the corner is square: S={s0}, asked for C={}", sh.c));
        }
        let qq16 = &raw.qq16;
        let Half { kk16, vv8, num32 } = raw.half(corner)?;
        let mm_in: Option<Vec<i64>> = if has_corner {
            let mm32 = raw.mm32.as_ref().ok_or("the layer image holds no `mm.i32`")?;
            let _s = witread::scope(Site::DecodeGather);
            Some((0..sh.insts * sh.s)
                .map(|i| mm32[hh(i / sh.s) * s0 + i % sh.s] as i64)
                .collect())
        } else {
            None
        };
        let gather = witread::scope(Site::DecodeGather);
        // Stream P: the three gathers in parallel. Each is a pure function of
        // the index it writes, and an indexed parallel collect keeps the order,
        // so the arrays are the ones the serial version built.
        let qq = (0..sh.insts * sh.s * sh.hd)
            .into_par_iter()
            .map(|i| {
                let (inst, rest) = (i / (sh.s * sh.hd), i % (sh.s * sh.hd));
                qq16[hh(inst) * s0 * hd0 + (rest / hd0 % s0) * hd0 + rest % hd0] as i32
            })
            .collect();
        // stream AA: `+ key_off` before the `% c0`, so a tiled band reads its
        // own window of the dump's context and an untiled one (`key_off == 0`)
        // reads exactly the bytes it read before
        if key_off + sh.c > c0 && key_off > 0 {
            return Err(format!("key window {key_off}..{} runs past the dump's C={c0}",
                               key_off + sh.c));
        }
        let kk = (0..sh.insts * sh.c * sh.hd)
            .into_par_iter()
            .map(|i| {
                let (inst, rest) = (i / (sh.c * sh.hd), i % (sh.c * sh.hd));
                let c = (rest / hd0 % sh.c + key_off) % c0;
                kk16[hh(inst) * c0 * hd0 + c * hd0 + rest % hd0] as i32
            })
            .collect();
        let vv = (0..sh.insts * sh.c * sh.hd)
            .into_par_iter()
            .map(|i| {
                let (inst, rest) = (i / (sh.c * sh.hd), i % (sh.c * sh.hd));
                let c = (rest / hd0 % sh.c + key_off) % c0;
                vv8[hh(inst) * c0 * hd0 + c * hd0 + rest % hd0] as i8 as i32
            })
            .collect();
        drop(gather);
        // torch's own numerators, for the bit-for-bit comparison, only where the
        // shape fits inside the dump (no tiling) and the band is real heads
        let tiled = sh.s > s0 || sh.c > c0;
        let torch_num = (!tiled).then(|| {
            let _s = witread::scope(Site::DecodeGather);
            (0..sh.insts * sh.s * sh.c)
                .into_par_iter()
                .map(|i| {
                    let (inst, rest) = (i / (sh.s * sh.c), i % (sh.s * sh.c));
                    num32[hh(inst) * s0 * c0 + (rest / sh.c) * c0 + rest % sh.c + key_off] as i64
                })
                .collect::<Vec<i64>>()
        });
        Ok(Self::build(sh, variant, qq, kk, vv, torch_num, q_bits, k_bits, nmax, log_table,
                       fracs, master,
                       format!("B1 layer witness [{nh}x{s0}x{c0}, heads {head}..+{}{}{}]",
                               sh.insts, if tiled { ", tiled" } else { "" },
                               if corner { ", causal corner" } else { "" }),
                       None, corner, mm_in))
    }

    /// A synthetic band whose exp table is pinned to a given master exponent and
    /// row count, so several bands can share one table (the composed selftest).
    #[allow(clippy::too_many_arguments)]
    pub fn synthetic_shared(sh: Shape, variant: Variant, q_bits: u32, k_bits: u32, nmax: u32,
                            log_table: u32, seed: u64, frac: f64, corner: bool) -> Self {
        let h = Self::synthetic(sh, variant, q_bits, k_bits, nmax, log_table, seed);
        let src = if corner { format!("{} [causal corner]", h.source) } else { h.source };
        Self::build(sh, variant, h.qq, h.kk, h.vv, None, q_bits, k_bits, nmax, log_table,
                    Some(vec![frac; sh.insts]), None, src, None, corner, None)
    }

    /// A prefix band and its causal corner, built as one object.
    ///
    /// Two things are not optional and neither is visible from inside a single
    /// band.  They share the **queries** -- the corner is the same tokens
    /// attending to a different set of keys, so `qQ` is the same matrix -- and
    /// they share one **row maximum**, taken over the keys of both halves.  A
    /// per-band maximum would put the two halves' numerators on two different
    /// scales, and the sum the attention output is built from would not be a
    /// softmax.  `frac` pins the master exponent when the caller has one;
    /// `None` lets each derive its own, which is what the probe pass wants.
    pub fn synthetic_pair(sh: Shape, variant: Variant, q_bits: u32, k_bits: u32, nmax: u32,
                          log_table: u32, seed: u64, frac: Option<f64>) -> (Self, Self) {
        let csh = Shape { c: sh.s, ..sh };
        let p0 = Self::synthetic(sh, variant, q_bits, k_bits, nmax, log_table, seed);
        let c0 = Self::synthetic(csh, variant, q_bits, k_bits, nmax, log_table, seed ^ 0x5c1);
        // the corner's own keys and values, against the prefix band's queries
        let c0 = Self::build(csh, variant, p0.qq.clone(), c0.kk, c0.vv, None, q_bits, k_bits,
                             nmax, log_table, None, None, c0.source, None, true, None);
        let mm: Vec<i64> = p0.mm.iter().zip(&c0.mm).map(|(a, b)| *a.max(b)).collect();
        let fin = frac.map(|f| vec![f; sh.insts]);
        let p = Self::build(sh, variant, p0.qq.clone(), p0.kk, p0.vv, None, q_bits, k_bits,
                            nmax, log_table, fin.clone(), None, p0.source, None, false,
                            Some(mm.clone()));
        let c = Self::build(csh, variant, p0.qq, c0.kk, c0.vv, None, q_bits, k_bits, nmax,
                            log_table, fin, None, c0.source, None, true, Some(mm));
        let total: Vec<i64> = p.zz.iter().zip(&c.zz).map(|(a, b)| a + b).collect();
        let mut p = p;
        p.zz_total = Some(total);
        (p, c)
    }

    /// The same Q, K and V read as the causal corner: the row maximum, the table's
    /// exponent, the numerators, the denominator and the A.V outputs are all
    /// re-derived over the visible half.  `frac` is re-derived too rather than
    /// carried over -- the corner's largest shift is not the rectangle's.
    pub fn into_corner(self) -> Self {
        let src = format!("{} [causal corner]", self.source);
        Self::build(self.shape, self.variant, self.qq, self.kk, self.vv, None, self.q_bits,
                    self.k_bits, self.nmax, self.log_table, None, None, src, None, true, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn corrupted(cfg: &Src, sh: Shape, variant: Variant, bad: Corrupt) -> Self {
        let h = Self::get(cfg, sh, variant);
        Self::build(sh, variant, h.qq, h.kk, h.vv, None, h.q_bits, h.k_bits, h.nmax,
                    h.log_table, Some(h.frac_per_inst), None, h.source, Some(bad), h.corner,
                    Some(h.mm.clone()))
    }

    pub fn get(cfg: &Src, sh: Shape, variant: Variant) -> Self {
        // A real dump already carries the corner (and, crucially, the *joint* row
        // maximum both halves shift by), so `load` builds it directly.  Only the
        // synthetic path has a rectangle to re-read as one.
        if let Some(d) = &cfg.dir {
            match Self::load(d, cfg.head, sh, variant, cfg.nmax, cfg.log_table, None,
                             cfg.corner) {
                Ok(a) => return a,
                Err(e) => eprintln!("  (no real attention witness: {e})"),
            }
        }
        let a = Self::synthetic(sh, variant, cfg.q_bits, cfg.k_bits, cfg.nmax, cfg.log_table,
                                cfg.seed);
        if cfg.corner { a.into_corner() } else { a }
    }

    /// Stream AA (E11): this band restricted to the key window `[c0, c0 + len)`.
    ///
    /// The window's derived values -- the scores, the shift, the numerators, the
    /// denominator, the A.V output -- are *recomputed from the window* by the
    /// same `build` the whole band came out of, against the parent's row
    /// maximum.  That is the same construction the causal corner already uses
    /// (`mm_in`), and it is what makes a partial band's `zz` the partial sum the
    /// group's total is assembled from.
    ///
    /// It exists so a synthetic layer can be tiled without a second generator.
    /// A real dump does not come through here: `Attn::load_from` reads its own
    /// window straight out of the layer image, which is one gather rather than a
    /// gather and a slice.
    pub fn key_slice(&self, c0: usize, len: usize) -> Self {
        let sh = self.shape;
        assert!(!self.corner, "the causal corner is not a prefix window");
        assert!(c0 + len <= sh.c, "key window {c0}..{} runs past C={}", c0 + len, sh.c);
        let out = Shape { c: len, ..sh };
        let gather = |src: &[i32]| -> Vec<i32> {
            (0..sh.insts * len * sh.hd)
                .map(|i| {
                    let (inst, rest) = (i / (len * sh.hd), i % (len * sh.hd));
                    src[(inst * sh.c + c0 + rest / sh.hd) * sh.hd + rest % sh.hd]
                })
                .collect()
        };
        Self::build(out, self.variant, self.qq.clone(), gather(&self.kk), gather(&self.vv),
                    None, self.q_bits, self.k_bits, self.nmax, self.log_table,
                    Some(self.frac_per_inst.clone()), Some(self.frac),
                    format!("{} [keys {c0}..{}]", self.source, c0 + len),
                    None, false, Some(self.mm.clone()))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(sh: Shape, variant: Variant, mut qq: Vec<i32>, mut kk: Vec<i32>,
             mut vv: Vec<i32>, torch_num: Option<Vec<i64>>, q_bits: u32, k_bits: u32,
             nmax: u32, log_table: u32, frac_in: Option<Vec<f64>>,
             master: Option<f64>, source: String,
             bad: Option<Corrupt>, corner: bool, mm_in: Option<Vec<i64>>) -> Self {
        // the grid's token axis has to tile exactly (layout.rs), so this holds on
        // *every* path, not only the pack subcommand's (B6 review carry-forward)
        assert!(sh.insts.is_power_of_two(), "insts must be a power of two, got {}", sh.insts);
        let (ni, ns, nc, nhd) = (sh.insts, sh.s, sh.c, sh.hd);
        assert!(!corner || nc == ns, "the causal corner is square: S={ns} C={nc}");
        // token `t` of a grid entry attends to key `c` unless this is the corner and
        // `c` is in its future.  Everything derived from a numerator -- the row
        // maximum the shift is taken against, the exponent the table is sized by,
        // the denominator, the A.V output -- is over the visible half alone.
        let vis = |i: usize| !corner || i % nc <= (i / nc) % ns;
        // exact integer scores, one (instance, token) row at a time
        //
        // Stream W: the allocation is timed apart from the loop that fills it.
        // These bare `vec![]`s sat between two of stream P's scopes and so were
        // charged to neither -- at the sequence's sizes this one alone is a
        // quarter of a gigabyte -- and naming them is what says whether the
        // 15 s the split could not attribute is allocation or is not.
        let mut sc = {
            let _s = witread::scope(Site::Alloc);
            vec![0i64; ni * ns * nc]
        };
        let scores = witread::scope(Site::Scores);
        // Stream W measured this loop three ways and moved it none of them, so
        // it is the loop stream P left. The record is `briefs/w-genbuild.md`,
        // lever 3: eight independent accumulators instead of one (it is not
        // waiting on the accumulator chain), blocking the token axis so a key
        // row serves eight tokens (it is not streaming K), and faulting the
        // output in beforehand (0.33 s of 12.1 s). What is left is the 9.6 G
        // integer multiply-accumulates themselves, and the device route that
        // would remove them is barred by *when* the answer is wanted: every
        // consumer of the shift -- the table sizing, the numerators, the
        // denominator, the A.V outputs, `check/torch_num` and `--tight-table`
        // -- runs inside `Layer::load`, before `compose::generate` first
        // touches the card, and this loader is also the SimdBackend arm's.
        sc.par_chunks_mut(nc).enumerate().for_each(|(ti, row)| {
            let (inst, t) = (ti / ns, ti % ns);
            let qrow = &qq[(inst * ns + t) * nhd..(inst * ns + t + 1) * nhd];
            for (c, dst) in row.iter_mut().enumerate() {
                let krow = &kk[(inst * nc + c) * nhd..(inst * nc + c + 1) * nhd];
                *dst = qrow.iter().zip(krow).map(|(a, b)| *a as i64 * *b as i64).sum();
            }
        });
        drop(scores);
        // The row maximum is a property of the *token*, not of the band: with the
        // causal corner the token's keys are split over two bands and one maximum
        // has to serve both, or their numerators are on two different scales and
        // `Z_prefix + Z_corner` is not a softmax denominator.  `mm_in` is how the
        // caller hands the joint one down; alone, a band takes its own.
        let maxima = witread::scope(Site::Maxima);
        let mut mm: Vec<i64> = match mm_in {
            Some(m) => {
                assert_eq!(m.len(), ni * ns, "external row maxima have the wrong shape");
                m
            }
            None => (0..ni * ns)
                .into_par_iter()
                .map(|ti| {
                    let vk = if corner { ti % ns + 1 } else { nc };
                    *sc[ti * nc..ti * nc + vk].iter().max().unwrap()
                })
                .collect(),
        };
        // The shift is committed, not derived, so a dishonest one is a witness the
        // AIR has to reject (or, for HighMax, silently accept -- the probe that
        // measures what softmax's shift invariance actually buys a prover).
        match bad {
            Some(Corrupt::LowMax(t)) => mm[t % (ni * ns)] -= 1,
            Some(Corrupt::HighMax(t, d)) => mm[t % (ni * ns)] += d,
            _ => {}
        }
        let dd: Vec<i64> =
            (0..ni * ns * nc).into_par_iter().map(|i| mm[i / nc] - sc[i]).collect();

        // the fixed-point exponent, per instance: put the largest shift the
        // instance can produce at the point where the numerator rounds to zero
        let d_max_of = |inst: usize| -> f64 {
            let lo = inst * ns * nc;
            dd[lo..lo + ns * nc].iter().enumerate()
                .filter(|(i, _)| vis(*i)).map(|(_, d)| *d).max().unwrap_or(1) as f64
        };
        let frac_per_inst: Vec<f64> = match &frac_in {
            Some(f) => (0..ni).map(|i| f[i % f.len()]).collect(),
            None => (0..ni)
                .into_par_iter()
                .map(|i| (d_max_of(i).max(1.0) / (2.0 * nmax as f64).ln()).log2())
                .collect(),
        };
        // one master table at the largest exponent; every other instance's shift is
        // scaled up to it by a public power of two, so the table is shared.
        //
        // B9 step 1a: that exponent has to be a *layer*-wide constant, not a
        // band-wide one.  Taking the largest `frac` over the band's own heads makes
        // the table a function of how the heads were split into bands, so
        // `Layer::load`'s "every band's table is identical" holds only where every
        // band happens to contain a head at the layer-wide maximum -- 4 of 24
        // Qwen-0.5B layers, and by B8's `frac_per_head`, 0 of 32 Llama-3.1-8B ones.
        // With `master` supplied every band scales its shifts up to the same
        // exponent and reads one table of `d_bits + spread` rows (B6 section 6).
        let natural = frac_per_inst.iter().cloned().fold(f64::MIN, f64::max);
        let frac = match master {
            Some(m) => {
                assert!(m >= natural - 1e-9,
                        "master exponent {m} is below this band's own maximum {natural}");
                m
            }
            None => natural,
        };
        let delta: Vec<i64> = frac_per_inst
            .iter()
            .map(|f| 1i64 << ((frac - f).round().max(0.0) as u32).min(30))
            .collect();
        let frac_integer_separated =
            frac_per_inst.iter().all(|f| ((frac - f) - (frac - f).round()).abs() < 1e-9);

        // v1's table is the bf16 one B4 measured, so its size is fixed at 2^16;
        // the integer variants' table is a knob, and how many rows they need is
        // exactly the question `d_bits` answers.  `log_table = 0` means "as many
        // rows as this witness needs", i.e. the complete table for these inputs.
        let scaled_max = (0..ni * ns * nc)
            .into_par_iter()
            .filter(|i| vis(*i))
            .map(|i| dd[i] * delta[i / (ns * nc)])
            .max()
            .unwrap_or(1)
            .max(1);
        let d_bits = 64 - scaled_max.leading_zeros();
        drop(maxima);
        let log_table = if log_table > 0 {
            log_table
        } else {
            // capped: past 2^22 rows the table dominates a single-band benchmark,
            // and an index that no longer fits is exactly the incompleteness the
            // `off_table` count is there to report
            match variant {
                Variant::IntSplit(k) => d_bits.saturating_sub(k).clamp(4, 22),
                _ => d_bits.clamp(4, 22),
            }
        };
        let table_scope = witread::scope(Site::Table);
        let (log_table, mut table) = match variant {
            Variant::Bf16x => (LOG_EXP_TABLE, bf16::exp_bf16_table(nmax)),
            Variant::IntExp => (log_table, table_of(1 << log_table, nmax, frac)),
            Variant::IntSplit(k) => (log_table,
                                     table_of(1 << log_table, nmax, frac - k as f64)),
        };

        drop(table_scope);
        if let Some(Corrupt::Table(i)) = bad {
            // a corrupted *table* with the witness rebuilt to match: internally
            // consistent, and only the preprocessed root gives it away (B4 section 3)
            let i = i % table.len();
            table[i] = if table[i] == 0 { 1 } else { table[i] - 1 };
        }

        // the committed index, and the numerator the table gives for it
        let (mut xx, mut num) = {
            let _s = witread::scope(Site::Alloc);
            (vec![0u16; ni * ns * nc], vec![0i64; ni * ns * nc])
        };
        let numerators = witread::scope(Site::Numerators);
        // Stream P: one (instance, token) row per task. Every entry is written
        // from its own index and no task reads what another writes, so the two
        // columns are the ones the serial loop built; what changes is that
        // `off_table` is added up over the rows instead of counted in place.
        let off_table: usize = xx
            .par_chunks_mut(nc)
            .zip(num.par_chunks_mut(nc))
            .enumerate()
            .map(|(ti, (xrow, nrow))| {
                let inst = ti / ns;
                let t = ti % ns;
                let mut off = 0usize;
                for c in 0..nc {
                    // a masked entry has no numerator and is not a table row: its
                    // shift is taken against a maximum over keys it is not one
                    // of, so it can be negative, and `off_table` must not count it
                    if corner && c > t {
                        continue;
                    }
                    let i = ti * nc + c;
                    let idx: usize = match variant {
                        Variant::Bf16x => {
                            let x = bf16::from_f32(
                                -(dd[i] as f64 / frac_per_inst[inst].exp2()) as f32);
                            xrow[c] = x;
                            x as usize
                        }
                        Variant::IntExp => (dd[i] * delta[inst]) as usize,
                        Variant::IntSplit(k) => ((dd[i] * delta[inst]) >> k) as usize,
                    };
                    debug_assert!(dd[i] >= 0 || bad.is_some());
                    if idx < table.len() {
                        nrow[c] = table[idx] as i64;
                    } else {
                        off += 1;
                    }
                }
                off
            })
            .sum();
        drop(numerators);
        if let Some(Corrupt::Unmask(i)) = bad {
            // the mask is preprocessed, so a prover cannot move it; what it *can*
            // do is claim a denominator and an output that count a masked key.
            let masked: Vec<usize> = (0..ni * ns * nc).filter(|g| !vis(*g)).collect();
            assert!(!masked.is_empty(), "Unmask on a band with no masked entries");
            num[masked[i % masked.len()]] = nmax as i64;
        }
        if let Some(Corrupt::Numerator(i)) = bad {
            // a different but perfectly legal numerator: still a table *output*,
            // just not the one this index maps to.  Everything downstream is
            // rebuilt from it, so only the lookup can object.
            let i = i % num.len();
            num[i] = if num[i] == 0 { nmax as i64 } else { 0 };
        }
        let den_scope = witread::scope(Site::Denominator);
        let mut zz: Vec<i64> =
            (0..ni * ns).map(|ti| num[ti * nc..(ti + 1) * nc].iter().sum()).collect();
        drop(den_scope);
        let mut out = vec![0i64; ni * ns * nhd];
        let av = witread::scope(Site::Av);
        out.par_chunks_mut(nhd).enumerate().for_each(|(ti, row)| {
            let (inst, t) = (ti / ns, ti % ns);
            let nrow = &num[(inst * ns + t) * nc..(inst * ns + t + 1) * nc];
            for (c, n) in nrow.iter().enumerate() {
                if *n != 0 {
                    let vrow = &vv[(inst * nc + c) * nhd..(inst * nc + c + 1) * nhd];
                    for (dst, v) in row.iter_mut().zip(vrow) {
                        *dst += n * *v as i64;
                    }
                }
            }
        });
        drop(av);
        // the last two corruptions leave the committed columns inconsistent with
        // each other, which is what the two Freivalds identities are there to catch
        match bad {
            Some(Corrupt::Key(i, d)) => {
                let n = kk.len();
                kk[i % n] += d;
            }
            Some(Corrupt::Value(i, d)) => {
                // A V column that no numerator touches is genuinely outside the
                // statement -- A.V cannot depend on it -- and with most numerators
                // zero at this shape a blindly chosen index usually lands on one.
                // Walk to the first value whose key participates.
                let start = i % vv.len();
                let idx = (0..vv.len())
                    .map(|st| (start + st) % vv.len())
                    .find(|cand| {
                        let inst = cand / (nc * nhd);
                        let c = (cand % (nc * nhd)) / nhd;
                        (0..ns).any(|t| num[(inst * ns + t) * nc + c] != 0)
                    })
                    .unwrap_or(start);
                vv[idx] += d;
            }
            Some(Corrupt::Denominator(t, d)) => zz[t % (ni * ns)] += d,
            _ => {}
        }
        let _ = &mut qq;                        // queries are left alone on purpose
        let torch_mismatch = torch_num.map(|t| {
            let _s = witread::scope(Site::CheckTorchNum);
            (0..num.len()).into_par_iter().filter(|i| t[*i] != num[*i]).count()
        });
        // One master table serves the band only if the per-head exponents differ by
        // integers; computed-and-discarded in B6, asserted here (B6 review).  Only
        // the real-witness path carries calibrated fracs -- a synthetic witness
        // derives them from random data, where the sharing is a rounded
        // approximation by construction (b6/README section 6's `intsplit` caveat).
        assert!(frac_in.is_none() || bad.is_some() || frac_integer_separated,
                "frac_per_inst not integer-separated: {frac_per_inst:?}");
        Self { shape: sh, variant, qq, kk, vv, sc, mm, dd, xx, num, zz, out, table,
               log_table, nmax, frac, frac_per_inst, delta, q_bits, k_bits, source,
               off_table, torch_mismatch, corner, zz_total: None }
    }

    /// The largest |A.V accumulator| the *committed* denominators allow, against
    /// the blind worst case a verifier would have to assume without them.
    pub fn av_bounds(&self) -> (i64, i64, i64) {
        let observed = self.out.iter().map(|v| v.abs()).max().unwrap_or(0);
        let committed = 127 * self.zz_row().iter().copied().max().unwrap_or(0);
        let blind = 127 * self.nmax as i64 * self.shape.c as i64;
        (observed, committed, blind)
    }

    pub fn d_bits(&self) -> u32 {
        // over the *visible* half: a masked shift is taken against a maximum over
        // keys it is not one of, so it says nothing about how many rows the table
        // needs (and can be larger than any real one)
        let d = self.dd.iter().enumerate().filter(|(g, _)| self.visible(*g))
            .map(|(_, d)| *d).max().unwrap_or(0);
        64 - d.max(1).leading_zeros()
    }

    /// Stream J: the largest key this band actually looks up, and the rows that
    /// hold it.  `d_bits` is the *unscaled* shift; the table is indexed by the
    /// shift scaled to the master exponent, so the two differ by up to the band's
    /// spread -- and the driver's `d_bits + spread` sizing takes the maximum of
    /// each separately, which over-counts whenever the largest raw shift does not
    /// come from the lowest-exponent head (measured: every layer of b10-final,
    /// `experiments/b10/out/oversize/`).
    /// Stream W: in parallel, for the same reason stream P's maxima are.
    ///
    /// This and the `off_table` count in `retable` below are the two passes
    /// `--tight-table` makes over every shift a layer holds -- 76 M entries a
    /// chunk at the sequence's sizes, twice -- and they ran serially, in
    /// `main.rs`'s `tighten_of`, after `Layer::load` had returned. They were
    /// therefore inside `build_secs` and inside no scope, which is most of the
    /// 15 s stream P's split could not attribute (`construct/tighten`, 9.79 s
    /// of a 68.9 s pipelined build; allocation, the term the brief expected,
    /// measured 0.46 s).
    ///
    /// An integer maximum is associative and commutative, so which order the
    /// reduction takes cannot change the value -- the same argument that
    /// licensed stream P's change 4 -- and the scope stays on the thread that
    /// opens it, in `Layer::key_bits`, with none opened in the workers.
    pub fn key_max(&self) -> i64 {
        (0..self.dd.len()).into_par_iter().filter(|g| self.visible(*g))
            .map(|g| self.table_key(g)).max().unwrap_or(0)
    }
    pub fn key_bits(&self) -> u32 {
        64 - self.key_max().max(1).leading_zeros()
    }

    /// Re-size this band's exp table to `bits` rows.  The table is a prefix of the
    /// same closed form, so shrinking it is a truncation and every numerator the
    /// witness already holds is unchanged -- what changes is which keys are still
    /// in range, which is what `off_table` counts.
    pub fn retable(&mut self, bits: u32) {
        assert!(bits <= self.log_table, "retable only shrinks: {bits} > {}", self.log_table);
        self.table.truncate(1usize << bits);
        self.log_table = bits;
        // Stream W: the second of the two passes, in parallel on the same
        // grounds -- a count is a sum, and the order it is added in cannot
        // change it. The predicate is per entry and reads nothing another
        // entry writes.
        self.off_table = (0..self.dd.len()).into_par_iter().filter(|g| self.visible(*g))
            .filter(|g| {
                let k = self.table_key(*g);
                k < 0 || k as usize >= self.table.len()
            })
            .count();
    }

    /// Relative change in the attention output `out / Z` against another witness:
    /// what a prover who shifted by more than the maximum actually changes.
    pub fn out_drift(&self, other: &Attn) -> f64 {
        let (mut num, mut den) = (0f64, 0f64);
        for t in 0..self.shape.s {
            for j in 0..self.shape.hd {
                let a = self.out[t * self.shape.hd + j] as f64 / self.zz[t].max(1) as f64;
                let b = other.out[t * self.shape.hd + j] as f64 / other.zz[t].max(1) as f64;
                num += (a - b).abs();
                den += a.abs();
            }
        }
        if den == 0.0 { 0.0 } else { num / den }
    }
}
