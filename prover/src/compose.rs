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

use num_traits::Zero;
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
use crate::glue::{self, GKind, GTableEval, GlueEval, GlueRels, GlueTables, Prog, Tie,
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
}

impl Rels {
    pub fn draw(ch: &mut Blake2sM31Channel, widths: &[u32], n_un: Option<usize>) -> Self {
        let exp = ExpRel::draw(ch);
        let norm = NormRel::draw(ch);
        let frac = FracRel::draw(ch);
        let mul = MulRel::draw(ch);
        let qnz = PairRel::draw(ch);
        let ranges: Vec<RangeRel> = widths.iter().map(|_| RangeRel::draw(ch)).collect();
        let glue = n_un.map(|n| GlueRels {
            norm: norm.clone(),
            frac: frac.clone(),
            pow: crate::glue::PowRel::draw(ch),
            smul: crate::glue::SMulRel::draw(ch),
            val: crate::glue::ValRel::draw(ch),
            un: (0..n).map(|_| crate::glue::UnRel::draw(ch)).collect(),
        });
        Self { exp, norm, frac, mul, qnz, widths: widths.to_vec(), ranges, glue }
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
    pub walk_p: usize,
    /// B10 step 2.  Absent on a witness `extract.py --glue` did not write.
    #[serde(default)]
    pub glue: Option<GlueParams>,
    /// B10 step 3: the causal corner rides as a second set of bands over the same
    /// heads, `C = S`, masked.  Absent on a witness extracted without it.
    #[serde(default)]
    pub corner: bool,
}

/// The wiring family's two shape numbers.  `a` is how many rows one token's
/// residual stream occupies at the reduction's first level: `tree_sum` groups the
/// channels by residue mod `a`, so a row's committed leaves are the channels
/// `r, r+a, ...` and the second level is one row per token over the `a` roots.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct GlueParams {
    pub d: usize,
    pub a: usize,
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
        v
    }
}

// -------------------------------------------------------------------- plan ----

pub struct WalkPlan {
    pub name: String,
    pub mats: Vec<usize>,
    pub lay: WalkLayout,
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
    pub regions: Vec<GlueRegionPlan>,
    /// `((region, tie), (region, tie))` the verifier compares
    pub eqs: Vec<((usize, usize), (usize, usize))>,
    pub uns: Vec<UnKind>,
    /// the layer's committed residual-stream input and output claims
    pub stream_in: (usize, usize),
    pub stream_out: (usize, usize),
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
        Self {
            t,
            regions,
            eqs: vec![((0, 1), (1, 0)), ((0, 2), (2, 0)), ((2, 1), (3, 0))],
            uns: l1.un_kinds(),
            stream_in: (0, 0),
            stream_out: (2, 2),
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
        let mut head_offsets = vec![];
        let mut off = 0;
        let mut shapes = vec![];
        for insts in &p.bands {
            head_offsets.push(off);
            off += insts;
            shapes.push(Shape { s: p.s, c: p.c, hd: p.hd, insts: *insts });
        }
        assert_eq!(off, p.nh, "bands must cover every head");
        let mut corner = vec![false; shapes.len()];
        if p.corner {
            let mut off = 0;
            for insts in &p.bands {
                head_offsets.push(off);
                off += insts;
                shapes.push(Shape { s: p.s, c: p.s, hd: p.hd, insts: *insts });
                corner.push(true);
            }
        }
        let lays: Vec<Layouts> = shapes.iter().map(|sh| Layouts::choose(sh, &cfg)).collect();

        let mut widths = vec![p.k_bits, p.q_bits, 8, p.z_lo_bits, p.z_hi_bits, p.ao_rem_bits];
        if let AttnVariant::IntSplit(k) = attn_variant {
            widths.push(k);
        }
        widths.sort_unstable();
        widths.dedup();

        let mut specs = vec![];
        let mut mat_lays = vec![];
        for m in &p.mats {
            let lay = MatLayout::choose_or_padded(m.n_out, m.tokens, p.deq_target);
            specs.push(Spec {
                variant: DeqVariant::parse(&m.variant),
                round,
                kmax: p.kmax,
                per_row: lay.per_row,
                limb_bits: (16, 14),
                q_bits: p.ao_q_bits,
                rem_bits: p.ao_rem_bits,
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
            });
        }
        // table row counts, exactly as `Tables::new` builds them
        let norm_rows = crate::bf16::norm_table(p.kmax, round).len();
        let log_norm = crate::common::log_rows_for(norm_rows, 1);
        let frac_rows = 1usize + (0..=p.kmax).map(|k| 1usize << k).sum::<usize>();
        let log_frac = crate::common::log_rows_for(frac_rows, 1);
        let log_mul = 14;
        let glue = p.glue.as_ref().map(|g| GluePlan::new(p, g));
        Self { p: p.clone(), attn_variant, round, shapes, lays, cfg, head_offsets, corner, specs,
               mat_lays, ao_lay, walks, widths, log_norm, log_frac, log_mul, glue }
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
                .as_corner(pre0.r_c, pre0.mu, sh.c)
        } else {
            Powers::new(r, s, rv, sh, &l.chan, &l.key, &l.out, l.rw)
        }
    }

    /// How many secure felts the transcript draws after the relations, and where
    /// the corner's key challenge sits.  It goes *last* so a proof without the
    /// corner draws exactly what it drew before.
    pub fn n_challenges(&self) -> usize {
        3 + if self.glue.is_some() { 2 } else { 0 } + if self.p.corner { 1 } else { 0 }
    }
    pub fn corner_challenge_at(&self) -> usize {
        3 + if self.glue.is_some() { 2 } else { 0 }
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
            if !self.corner[b] {
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
                    _ => unreachable!(),
                }
            }
            ids.extend(DEQ_SEL.iter().map(|n| deq_sel(&m.name, n)));
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
        ids
    }

    /// The number of components that carry lookups, in `use_sums` order.
    pub fn n_use(&self) -> usize {
        // chan + grid + per-token per prefix band; a corner band has no per-token
        // region (the pair shares the prefix band's), so it contributes two
        self.corner.iter().map(|c| if *c { 2 } else { 3 }).sum::<usize>()
            + 1 + self.p.mats.len() + self.walks.len()
            + self.glue.as_ref().map(|g| g.regions.len()).unwrap_or(0)
    }
    pub fn n_table_sums(&self) -> usize {
        5 + self.widths.len() + self.glue.as_ref().map(|g| g.n_tables()).unwrap_or(0)
    }
}

// -------------------------------------------------- components (both sides) ----

pub struct BuiltComponents {
    pub chans: Vec<FrameworkComponent<ChanEval>>,
    pub keys: Vec<FrameworkComponent<KeyEval>>,
    pub rows: Vec<FrameworkComponent<RowEval>>,
    pub ao: FrameworkComponent<AttnOutEval>,
    pub mats: Vec<FrameworkComponent<DequantEval>>,
    pub walks: Vec<FrameworkComponent<WalkEval>>,
    pub glue: Vec<FrameworkComponent<GlueEval>>,
    pub exp_t: FrameworkComponent<ExpTableEval>,
    pub tables: Vec<FrameworkComponent<TableEval>>,
    pub gtables: Vec<FrameworkComponent<GTableEval>>,
    pub pows: Vec<Powers>,
    /// per band, in `chans`/`keys` order: does it have a per-token region?  A
    /// corner band does not (B10 step 3), so `rows` is shorter than `chans` and
    /// the component order has to be walked, not indexed.
    pub has_rows: Vec<bool>,
}

impl BuiltComponents {
    pub fn provers(&self) -> Vec<&dyn ComponentProver<crate::common::Backend>> {
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
        v
    }
}

/// Build every component from the plan, the drawn relations and challenges, and
/// the parsed claims — the verifier's route, also used by the prover so the two
/// sides cannot diverge.
#[allow(clippy::too_many_arguments)]
pub fn build_components(plan: &Plan, rels: &Rels, r: SecureField, s: SecureField,
                        rv: SecureField, r_corner: SecureField, w_glue: SecureField,
                        lam_glue: SecureField, claims: &Claims) -> BuiltComponents {
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
        chans.push(FrameworkComponent::<ChanEval>::new(
            alloc,
            ChanEval { tag: b, lay: l.chan, pow: pow.clone(), off, claimed: unqm(bc[0]),
                       rel_k: rels.range(p.k_bits).clone(), rel_q: rels.range(p.q_bits).clone(),
                       range_check: true },
            next_use(claims),
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
                      range_check: true, corner: plan.corner[b] },
            next_use(claims),
        ));
        if !plan.corner[b] {
            rows.push(FrameworkComponent::<RowEval>::new(
                alloc,
                RowEval { tag: b, log_rows: l.log_row, rw: l.rw, pow: pow.clone(),
                          with_max: plan.attn_variant != AttnVariant::Bf16x,
                          claimed_am: unqm(bc[4]), claimed_az: unqm(bc[5]),
                          z_lo_bits: p.z_lo_bits, z_min: p.z_min,
                          rel_lo: rels.range(p.z_lo_bits).clone(),
                          rel_hi: rels.range(p.z_hi_bits).clone() },
                next_use(claims),
            ));
        }
        pows.push(pow);
    }

    let (rv_pow_ao, rv_step_ao) = powers(rv, plan.ao_lay.per_row);
    let (_, rv_hd) = powers(rv, p.hd);
    let (_, r_c) = powers(r, p.c);
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
        },
        next_use(claims),
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
            },
            next_use(claims),
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
                rel_act: rels.range(8).clone(),
                range_check: true,
            },
            next_use(claims),
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
                    val: gr.val.clone(),
                    tie: Some(ties[i].clone()),
                },
                next_use(claims),
            ));
        }
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

    BuiltComponents { chans, keys, rows, ao, mats, walks, glue, exp_t, tables, gtables, pows,
                      has_rows: plan.corner.iter().map(|c| !*c).collect() }
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
}

pub fn prove_layer(layer: &Layer, params: &Params, tamper: &ProverTamper) -> Composed {
    let t0 = std::time::Instant::now();
    let plan = Plan::new(params);
    let p = &plan.p;
    assert_eq!(layer.band_sizes[..p.bands.len()], p.bands[..],
               "layer bands disagree with params");
    assert_eq!(layer.bands.len(), plan.shapes.len(), "layer bands disagree with the plan");
    assert_eq!(layer.bands[0].log_table, p.log_table, "exp table rows disagree");

    let mut tables = Tables::new(layer.bands[0].table.clone(), p.log_table, p.kmax,
                                 plan.round, p.ao_q_bits, plan.widths.clone());
    if let Some(i) = tamper.norm_row {
        let i = i % tables.norm.len();
        tables.norm[i].m7 = (tables.norm[i].m7 + 1) % 128;
    }

    // ---- region traces -------------------------------------------------------
    let mut pre_all: Vec<Eval> = vec![];
    let mut base_all: Vec<Eval> = vec![];
    let mut band_committed: Vec<Committed> = vec![];
    let mut mult_exp = std::mem::take(&mut tables.m_exp);
    let mut mult_r = std::mem::take(&mut tables.m_r);
    for (b, a) in layer.bands.iter().enumerate() {
        let l = &plan.lays[b];
        let pre = attn_trace::commit_pre(a, l, false, !a.corner);
        let (base, at) = attn_trace::commit_base(a, l, &plan.cfg, &plan.widths, &mut mult_exp,
                                                 &mut mult_r, &mut tables.off_table, false,
                                                 !a.corner);
        pre_all.extend(pre.iter().cloned());
        base_all.extend(base.iter().cloned());
        // A corner band's interaction phase reads its own causal-mask columns back
        // out of `pre` (the exp lookup's multiplicity is `1 - msk`), so that band
        // keeps them.  A prefix band does not, and its selectors are the wide ones
        // -- `C` keys against the corner's `S` -- so they are dropped as before.
        let keep = if a.corner { pre } else { vec![] };
        band_committed.push(Committed { pre: keep, base, at, off_table: 0 });
    }
    tables.m_exp = mult_exp;
    tables.m_r = mult_r;

    let aor = ao_region(&layer.ao, p.ao_target, p.ao_q_bits, p.ao_rem_bits, &mut tables, None);
    pre_all.extend(aor.pre.iter().cloned());
    base_all.extend(aor.base.iter().cloned());

    let mut mat_regions = vec![];
    for (m, sp) in layer.mats.iter().zip(&plan.specs) {
        let r = mat_region(m, sp, &mut tables);
        pre_all.extend(r.pre.iter().cloned());
        base_all.extend(r.base.iter().cloned());
        mat_regions.push(r);
    }
    let n_deq_base: usize = mat_regions
        .iter()
        .map(|r| r.base.iter().map(|e| e.domain.size()).sum::<usize>())
        .sum();

    let mut walk_regions = vec![];
    for w in &plan.walks {
        let r = walk_region(layer, &w.name, &w.mats, p.walk_p, &mut tables, None);
        pre_all.extend(r.pre.iter().cloned());
        base_all.extend(r.base.iter().cloned());
        walk_regions.push(r);
    }

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
            let w1 = crate::glue::GlueWitness::new(
                rows1, vec![stream.clone(), t.lay(add)], &prog1, plan.round,
                format!("B1 residual stream [{} x {}]", t.tokens, t.d));
            let tam1 = tamper.glue.filter(|(ri, _)| *ri == i1).map(|(_, x)| x);
            let r1 = crate::glue::glue_region(&w1, &prog1, plan.round, &tag1, &mut gt, tam1);
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
            let r2 = crate::glue::glue_region(&w2, &prog2, plan.round, &tag2, &mut gt, tam2);
            stream = t.lay(&got);
            glue_regions.push(r1);
            glue_regions.push(r2);
        }
        for r in &glue_regions {
            pre_all.extend(r.pre.iter().cloned());
            base_all.extend(r.base.iter().cloned());
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

    // ---- shared tables (tree 0 tail) and their multiplicities (tree 1 tail) --
    let pre_tbl0 = pre_all.len();
    let tbl_rows = 1usize << p.log_table;
    pre_all.push(eval_of(p.log_table, (0..tbl_rows).map(|i| m31_u32(i as u32)).collect()));
    pre_all.push(eval_of(p.log_table, tables.exp.iter().map(|v| m31_u32(*v)).collect()));
    {
        let get: [fn(&crate::bf16::NormRow) -> u32; 6] =
            [|r| r.m2, |r| r.ph, |r| r.st, |r| r.m7, |r| r.expfield, |r| r.nz];
        for g in get {
            pre_all.push(eval_of(plan.log_norm, (0..1usize << plan.log_norm)
                .map(|i| m31_u32(tables.norm.get(i).map(g).unwrap_or(0))).collect()));
        }
        for t in 0..3 {
            pre_all.push(eval_of(plan.log_frac, (0..1usize << plan.log_frac)
                .map(|i| {
                    let x = tables.frac.get(i).copied().unwrap_or((0, 0, 0));
                    m31_u32([x.0, x.1, x.2][t])
                }).collect()));
        }
        for t in 0..4 {
            pre_all.push(eval_of(plan.log_mul, (0..1usize << plan.log_mul)
                .map(|i| {
                    let x = tables.mul.get(i).copied()
                        .unwrap_or(crate::bf16::MulRow { a: 0, b: 0, m7: 0, carry: 0 });
                    m31_u32([x.a, x.b, x.m7, x.carry][t])
                }).collect()));
        }
        pre_all.push(eval_of(p.ao_q_bits, (0..1u32 << p.ao_q_bits).map(m31_u32).collect()));
        pre_all.push(eval_of(p.ao_q_bits,
                             (0..1u32 << p.ao_q_bits)
                                 .map(|q| m31_u32(u32::from(q != 0))).collect()));
        for w in &plan.widths {
            pre_all.push(eval_of(*w, (0..1u32 << w).map(m31_u32).collect()));
        }
    }
    let mult_start = base_all.len();
    base_all.push(eval_of(p.log_table, tables.m_exp.iter().map(|m| m31_u32(*m)).collect()));
    base_all.push(eval_of(plan.log_norm, tables.m_norm.iter().map(|m| m31_u32(*m)).collect()));
    base_all.push(eval_of(plan.log_frac, tables.m_frac.iter().map(|m| m31_u32(*m)).collect()));
    base_all.push(eval_of(plan.log_mul, tables.m_mul.iter().map(|m| m31_u32(*m)).collect()));
    base_all.push(eval_of(p.ao_q_bits, tables.m_qnz.iter().map(|m| m31_u32(*m)).collect()));
    for (t, w) in plan.widths.iter().enumerate() {
        base_all.push(eval_of(*w, tables.m_r[t].iter().map(|m| m31_u32(*m)).collect()));
    }
    let glue_mult_start = base_all.len();
    if let Some(gt) = &gtables {
        pre_all.extend(crate::glue::glue_table_cols(gt));
        base_all.extend(crate::glue::glue_mult_cols(gt));
    }
    let trace_secs = t0.elapsed().as_secs_f64();

    // ---- one transcript -------------------------------------------------------
    let widest = pre_all.iter().chain(base_all.iter())
        .map(|e| e.domain.log_size()).max().unwrap();
    let tw = twiddles(widest);
    let mut sess = Session::new(&tw);
    sess.commit(pre_all.clone());
    sess.commit(base_all.clone());
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
        });
        Rels { exp, norm, frac, mul, qnz, widths: widths.clone(), ranges, glue }
    };
    let n_ch = plan.n_challenges();
    let ch = sess.draw_felts(n_ch);
    let (r, s, rv) = (ch[0], ch[1], ch[2]);
    let (w_glue, lam_glue) = if plan.glue.is_some() {
        (ch[3], ch[4])
    } else {
        (SecureField::zero(), SecureField::zero())
    };
    let r_corner = if p.corner { ch[plan.corner_challenge_at()] } else { SecureField::zero() };

    // ---- interaction ----------------------------------------------------------
    let t1 = std::time::Instant::now();
    let mut inter_all: Vec<Eval> = vec![];
    let mut claims = Claims::default();
    for (b, a) in layer.bands.iter().enumerate() {
        let l = &plan.lays[b];
        let pow = plan.powers(b, r, s, rv, r_corner);
        let it = attn_trace::interaction_phase(a, l, &plan.cfg, &pow, &rels.exp, &rels.ranges,
                                               &band_committed[b], &plan.widths, false,
                                               !a.corner);
        inter_all.extend(it.cols);
        claims.bands.push([qm(it.claimed_chan), qm(it.claimed_hs), qm(it.claimed_av),
                           qm(it.claimed_sn), qm(it.claimed_am), qm(it.claimed_az)]);
        claims.use_sums.extend(it.use_sums.iter().map(|v| qm(*v)));
    }
    let aoi = ao_interaction(&layer.ao, &aor, r, s, rv, p.c, &rels);
    inter_all.extend(aoi.cols);
    claims.ao_out = qm(aoi.claimed_out);
    claims.ao_den = qm(aoi.claimed_den);
    claims.use_sums.push(qm(aoi.use_sum));
    for (m, reg) in layer.mats.iter().zip(&mat_regions) {
        let it = mat_interaction(m, reg, rv, s, &rels);
        inter_all.extend(it.cols);
        claims.mat_h.push(qm(it.claimed_h));
        claims.use_sums.push(qm(it.use_sum));
    }
    for reg in &walk_regions {
        let it = walk_interaction(layer, reg, rv, s, &rels, true);
        inter_all.extend(it.cols);
        claims.walk_acc.push(it.claimed.iter().map(|v| qm(*v)).collect());
        claims.use_sums.push(qm(it.use_sum));
    }
    // the glue regions' ties and lookups, in component order
    if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
        let blank = Claims::default();
        let specs = gp.ties(w_glue, lam_glue, &blank);
        for (i, reg) in glue_regions.iter().enumerate() {
            let it = crate::glue::glue_interaction(reg, gr, Some(&specs[i]));
            inter_all.extend(it.cols);
            claims.glue_ties.push(it.claimed.iter().map(|v| qm(*v)).collect());
            claims.use_sums.push(qm(it.use_sum));
        }
    }
    // the tables' own sides, in table order
    {
        let (c, sum) = table_col(p.log_table, &rels.exp,
                                 &[&pre_all[pre_tbl0], &pre_all[pre_tbl0 + 1]],
                                 &base_all[mult_start]);
        inter_all.extend(c);
        claims.table_sums.push(qm(sum));
        let norm_refs: Vec<&Eval> = (0..6).map(|i| &pre_all[pre_tbl0 + 2 + i]).collect();
        let (c, sum) = table_col(plan.log_norm, &rels.norm, &norm_refs, &base_all[mult_start + 1]);
        inter_all.extend(c);
        claims.table_sums.push(qm(sum));
        let frac_refs: Vec<&Eval> = (0..3).map(|i| &pre_all[pre_tbl0 + 8 + i]).collect();
        let (c, sum) = table_col(plan.log_frac, &rels.frac, &frac_refs, &base_all[mult_start + 2]);
        inter_all.extend(c);
        claims.table_sums.push(qm(sum));
        let mul_refs: Vec<&Eval> = (0..4).map(|i| &pre_all[pre_tbl0 + 11 + i]).collect();
        let (c, sum) = table_col(plan.log_mul, &rels.mul, &mul_refs, &base_all[mult_start + 3]);
        inter_all.extend(c);
        claims.table_sums.push(qm(sum));
        let qnz_refs: Vec<&Eval> = (0..2).map(|i| &pre_all[pre_tbl0 + 15 + i]).collect();
        let (c, sum) = table_col(p.ao_q_bits, &rels.qnz, &qnz_refs, &base_all[mult_start + 4]);
        inter_all.extend(c);
        claims.table_sums.push(qm(sum));
        for (t, w) in plan.widths.iter().enumerate() {
            let (c, sum) = table_col(*w, &rels.ranges[t], &[&pre_all[pre_tbl0 + 17 + t]],
                                     &base_all[mult_start + 5 + t]);
            inter_all.extend(c);
            claims.table_sums.push(qm(sum));
        }
        if let (Some(gp), Some(gr)) = (&plan.glue, &rels.glue) {
            let gcols = crate::glue::glue_table_ids(&gp.uns);
            let base = pre_all.len() - gcols.len();
            for (t, (lr, cols, kind)) in
                crate::glue::glue_table_kinds(&gp.uns, gr).into_iter().enumerate()
            {
                let refs: Vec<&Eval> = cols
                    .iter()
                    .map(|c| {
                        let k = gcols.iter().position(|x| x == c).expect("glue table column");
                        &pre_all[base + k]
                    })
                    .collect();
                let (c, sum) = crate::glue::table_side(lr, &kind, &refs,
                                                       &base_all[glue_mult_start + t]);
                inter_all.extend(c);
                claims.table_sums.push(qm(sum));
            }
        }
    }
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
    let clamp_edge = aor.q_out.iter().filter(|q| q.unsigned_abs() == 127).count();
    let family_cells = json!({
        "bands": band_committed.iter().map(|c| c.base.iter().map(|e| e.domain.size()).sum::<usize>()).sum::<usize>(),
        "attnout": aor.base.iter().map(|e| e.domain.size()).sum::<usize>(),
        "dequant": n_deq_base,
        "walks": walk_regions.iter().map(|w| w.base.iter().map(|e| e.domain.size()).sum::<usize>()).sum::<usize>(),
        "glue": glue_regions.iter().map(|g| g.base.iter().map(|e| e.domain.size()).sum::<usize>()).sum::<usize>(),
        "glue_regions": glue_regions.iter().map(|g| json!({
            "tag": g.tag, "rows": g.rows, "log_rows": g.log_rows,
            "leaves": g.n_leaves, "ops": g.n_ops,
            "columns": g.base.len(),
            "cells": g.base.iter().map(|e| e.domain.size()).sum::<usize>(),
            "reference_mismatch": g.ref_n - g.ref_ok,
            "exponent_out_of_range": g.bad_exp,
        })).collect::<Vec<_>>(),
        "glue_vs_b1": glue_b1,
        "interaction": inter_all.iter().map(|e| e.domain.size()).sum::<usize>(),
        "preprocessed": pre_all.iter().map(|e| e.domain.size()).sum::<usize>(),
        "walk_weights": walk_regions.iter().map(|w| json!({
            "group": w.name, "weights": w.n_weights,
            "pk": w.lay.pk, "pq": w.lay.pq, "log_rows": w.lay.g.log_rows,
            "pre_cells": w.pre.iter().map(|e| e.domain.size()).sum::<usize>(),
        })).collect::<Vec<_>>(),
        "mats": layer.mats.iter().zip(&mat_regions).map(|(m, r)| json!({
            "name": m.name, "variant": m.variant.name(), "n": r.n,
            "per_row": r.lay.per_row, "log_rows": r.lay.log_rows,
            "kmax_needed": m.kmax_needed(), "kmax_blind": m.kmax_blind(),
            "cells": r.base.iter().map(|e| e.domain.size()).sum::<usize>(),
        })).collect::<Vec<_>>(),
        "attnout_stats": {"n": aor.n, "per_row": aor.lay.per_row,
                          "log_rows": aor.lay.log_rows, "lead_rows": aor.lay.lead,
                          "q_bits": aor.q_bits, "rem_bits": aor.rem_bits,
                          "clamp_edge_count": clamp_edge},
    });
    sess.commit(inter_all);
    sess.mix(&claims.flat());

    // ---- components, prove ----------------------------------------------------
    let built = build_components(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue, &claims);
    let sizes: Vec<Vec<u32>> = sess
        .ops
        .iter()
        .filter_map(|op| match op {
            crate::common::Op::Tree(sz) => Some(sz.clone()),
            _ => None,
        })
        .collect();
    let n_columns = [sizes[0].len(), sizes[1].len(), n_inter];
    let provers = built.provers();
    let (res, stats) = sess.prove_raw(&provers);
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
        off_table: tables.off_table,
        family_cells,
        n_columns,
        glue_ties_ok,
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
    /// the layer's committed residual-stream input and output claims
    pub stream_in: Option<[u32; 4]>,
    pub stream_out: Option<[u32; 4]>,
    pub error: Option<String>,
    pub verify_secs: f64,
}

impl Verdict {
    pub fn accepts(&self) -> bool {
        self.stark_ok && self.root_ok && self.sums_cancel && self.av_ok && self.den_link_ok
            && self.sizes_ok && self.qk_ok.iter().all(|x| *x) && self.den_ok.iter().all(|x| *x)

            && self.walk_ok.iter().all(|x| *x) && self.glue_ok.iter().all(|x| *x)
    }
    fn rejected(msg: &str) -> Self {
        Self { stark_ok: false, root_ok: false, sums_cancel: false, qk_ok: vec![],
               den_ok: vec![], av_ok: false, den_link_ok: false, walk_ok: vec![],
               sizes_ok: false, glue_ok: vec![false], stream_in: None, stream_out: None,
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
        || fixed.sizes.len() != 3
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
    let rels = Rels::draw(&mut ch, &plan.widths, plan.glue.as_ref().map(|g| g.uns.len()));
    let n_ch = plan.n_challenges();
    let f = ch.draw_secure_felts(n_ch);
    let (r, s, rv) = (f[0], f[1], f[2]);
    let (w_glue, lam_glue) = if plan.glue.is_some() {
        (f[3], f[4])
    } else {
        (SecureField::zero(), SecureField::zero())
    };
    let r_corner = if plan.p.corner { f[plan.corner_challenge_at()] } else { SecureField::zero() };
    vcs.commit(proof.commitments[2], &fixed.sizes[2], &mut ch);
    ch.mix_felts(&claims.flat());

    // ---- components from the parsed statement ----------------------------------
    let built = build_components(&plan, &rels, r, s, rv, r_corner, w_glue, lam_glue, &claims);
    let components = built.components();
    // the AIR's own column sizes must agree with the committed ones
    let derived = TreeVec::concat_cols(components.iter().map(|c| c.trace_log_degree_bounds()));
    let sizes_ok = derived[1] == fixed.sizes[1] && derived[2] == fixed.sizes[2];

    let vres = verify(&components, &mut ch, &mut vcs, proof);
    let stark_ok = vres.is_ok();
    let error = vres.err().map(|e| format!("{e:?}"));

    // ---- claimed sums and identities -------------------------------------------
    let total = claims
        .use_sums
        .iter()
        .chain(claims.table_sums.iter())
        .fold(SecureField::zero(), |a, b| a + unqm(*b));
    let sums_cancel = total == SecureField::zero();

    let mut qk_ok = vec![];
    let mut av_sum = SecureField::zero();
    let mut az_sum = SecureField::zero();
    // B10 step 3: the corner bands come after the prefix bands, one for one.  A
    // corner band has no per-token region of its own -- there is one row maximum
    // and one denominator per *token*, and both live on the prefix band -- so its
    // Q.K^T identity is read against the prefix band's `am`, and its numerator sum
    // is added to the prefix band's before the denominator identity is checked.
    let n_pre = plan.p.bands.len();
    let (_, rv_hd) = powers(rv, p.hd);
    let (_, r_c) = powers(r, p.c);
    for (b, bc) in claims.bands.iter().enumerate() {
        let pow = &built.pows[b];
        let (chan, hs, av) = (unqm(bc[0]), unqm(bc[1]), unqm(bc[2]));
        // a corner band's row maximum is the prefix band's, by construction: it has
        // no `am` of its own to disagree with
        let am = unqm(claims.bands[if plan.corner[b] { b - n_pre } else { b }][4]);
        qk_ok.push(pow.r_sum * am - hs == chan);
        // fold this band's A.V claim by its head offset; the denominator claim is
        // the prefix band's alone, and the corner's `sn` joins it below
        let woff_out = rv_hd.pow(plan.head_offsets[b] as u128);
        av_sum += woff_out * av;
        if !plan.corner[b] {
            az_sum += r_c.pow(plan.head_offsets[b] as u128) * unqm(bc[5]);
        }
    }
    let den_ok: Vec<bool> = (0..n_pre)
        .map(|b| {
            let mut sn = unqm(claims.bands[b][3]);
            if plan.p.corner {
                sn += unqm(claims.bands[b + n_pre][3]);
            }
            sn == unqm(claims.bands[b][5])
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

    Verdict { stark_ok, root_ok, sums_cancel, qk_ok, den_ok, av_ok, den_link_ok, walk_ok,
              sizes_ok, glue_ok, stream_in, stream_out, error,
              verify_secs: t0.elapsed().as_secs_f64() }
}
