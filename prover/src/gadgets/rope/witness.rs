//! The witness side of the rotation and the quantize, and the two rebuilds that
//! produce it.
//!
//! `RopeCols` is what a dump carries for this gadget; `rope_from_mats` is what
//! the prover derives when it rotates the layer itself, and `rebuild_key_tiles`
//! is the same derivation's second half, the key tiles rebuilt from the rotated
//! q/k/v. Both are methods on `Layer` because they read and write most of it --
//! the q/k/v matmuls, the key tiles, the attnout witness -- and they stay
//! methods here so that no call site and no body changes.

use rayon::prelude::*;

use crate::bf16;
use crate::gadgets::attention::witness::Attn;
use crate::gadgets::dequant::Variant as DeqVariant;
use crate::gadgets::dequant::trace::{Mat, deq_ref, fold_out};
use crate::layer::{Layer, model_i32};
use crate::witread;

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
}
