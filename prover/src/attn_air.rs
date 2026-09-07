//! The attention AIR: two Freivalds instances, one exp table, one shift -- packed
//! over `insts` head instances sharing one column set.
//!
//! B5's arithmetic is unchanged.  What B6 adds is the instance axis, and the whole
//! of it is three preprocessed selectors and three public constants:
//!
//! * the two accumulators that are already *Horners* fold instances for free.
//!   `hs = sum_{gk} r^{gk} zs_{gk}` over the packed key axis `gk = i C + c` is
//!   exactly `sum_i r^{C i} (sum_c r^c zs)`, and `h` likewise gives `rv^{hd i}`.
//! * the three that are plain *sums* -- `acc` in the channel region, `av` and `sn`
//!   in the grid -- have to pick up the same per-instance weight, or the two sides
//!   of an identity are no longer comparable.  Each gets one preprocessed
//!   instance-start selector and multiplies by `r^C` (or `rv^hd`) there.
//! * the per-token region's Horner runs in `s`, so its natural instance weight is
//!   `s^S`; the boundary multiplier `mu = r^C s^{-(S-rw)}` converts it to `r^C`.
//!   `mu` is a public constant the verifier computes from `r` and `s`.
//!
//! The three application-level checks come out *identical* to B5's -- `r_sum am - hs
//! == chan`, `av == out`, `sn == az` -- because every term carries the same weight.
//! So one claimed value per identity covers every instance in the key tile, and
//! per-instance soundness is Schwartz-Zippel on a polynomial of degree `insts C`
//! rather than `C`: 336 instances cost 8.4 bits of the 124 QM31 has.
//!
//! Claims, for one head, with `r` over keys, `s` over query tokens and `rv` over
//! output channels (three independent QM31 challenges drawn after the witness
//! commits):
//!
//!   Q.K^T   sum_{t,c} r^c s^t score[t,c]  ==  sum_j (sum_c r^c qK[c,j])(sum_t s^t qQ[t,j])
//!   A.V     sum_{t,j} rv^j s^t out[t,j]   ==  sum_c (sum_j rv^j qV[c,j])(sum_t s^t n[t,c])
//!   denom   sum_t s^t Z_t                 ==  sum_c (sum_t s^t n[t,c])
//!
//! The integer variants never commit `score`.  They commit the shift
//! `d[t,c] = m_t - score[t,c]` and one maximum `m_t` per token, and the Q.K^T
//! claim is read as
//!
//!   sum_{t,c} r^c s^t score[t,c] = (sum_c r^c)(sum_t s^t m_t) - sum_{t,c} r^c s^t d[t,c]
//!
//! so the max shift costs one accumulator over `S` values instead of a column
//! over the `S x C` grid, and `d`'s own lookup into the exp table doubles as the
//! range check that makes `m_t >= max_c score[t,c]`.  What is *not* proven is
//! `m_t <= max`: a prover may shift by more.  Softmax is exactly invariant under
//! that shift, so the probs it changes are only the fixed-point rounding -- the
//! `HighMax` probe measures how much -- and the committed `Z_t`, range-checked
//! here, is what stops the numerators from being shifted into oblivion.
//!
//! Every constraint reads at most a `[-1, 0]` mask and is degree <= 3, which is
//! what blowup 2 supports (B2).

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkEval, INTERACTION_TRACE_IDX, RelationEntry, relation,
};

use crate::layout::{LayoutChan, LayoutKey, LayoutOut, Shape};
use crate::attn_witness::Variant;

relation!(ExpRel, 2);
relation!(RangeRel, 1);

impl ExpRel {
    /// the drawn elements, for the GKR link's affine weights (B11 stream S)
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}
impl RangeRel {
    /// the drawn elements, for the GKR link's affine weights (B11 stream S)
    pub fn parts(&self) -> (SecureField, &[SecureField]) {
        (self.0.z, &self.0.alpha_powers)
    }
}

pub fn id(s: impl Into<String>) -> PreProcessedColumnId {
    PreProcessedColumnId { id: s.into() }
}
/// A key-tile-local preprocessed column: several key tiles share one proof in B6.5, so
/// every selector id carries its key tile's tag.
pub fn key_tile_id(tag: usize, s: &str) -> PreProcessedColumnId {
    id(format!("{s}_key_tile{tag}"))
}
pub const CHAN_SEL: [&str; 5] =
    ["b6_c_nofirst", "b6_c_last", "b6_c_notzero", "b6_c_end", "b6_c_istart"];
/// The grid region's preprocessed columns.  The last is not a selector: it is
/// `2^-(F - frac_i)`, the constant that turns the committed *scaled* shift back into
/// the shift itself, so one master exp table can serve every head in the key tile.
pub const KEY_SEL: [&str; 8] =
    ["b6_k_nofirst", "b6_k_last", "b6_k_notzero", "b6_k_end", "b6_k_m1", "b6_k_m2",
     "b6_k_istart", "b6_k_dinv"];
/// The *diagonal block* grid's preprocessed columns (B10 step 3).  The same eight the
/// prefix grid reads -- under their own ids, because the two regions are two
/// components with two tilings -- plus `iend`, the row that completes an
/// instance's key block.  The diagonal block's key Horner runs in an independent
/// challenge `r'`, so its natural per-instance weight is `r'^S`; the identity it
/// is read against carries `am`, whose weight is the *prefix's* `r^C`.  `iend` is
/// where the one converts to the other, at one preprocessed column and no fresh
/// cell.
pub const DIAGONAL_BLOCK_SEL: [&str; 9] =
    ["b6_x_nofirst", "b6_x_last", "b6_x_notzero", "b6_x_end", "b6_x_m1", "b6_x_m2",
     "b6_x_istart", "b6_x_dinv", "b6_x_iend"];
/// Stream CC (E11): the grid's **inert lead row** selector -- 1 where
/// `Grid::row(j)` is `Some`, 0 on the rows `layout.rs:101` pads with.
///
/// It exists only when the grid actually has such rows (`g.lead > 0`).  Every
/// shape proved before E11 tiled exactly, and on those the AIR reads the literal
/// `1` in its place, so tree 0, the constraint set and the digest are what they
/// were.  `lead` is a function of the shape, which is public, so this is not a
/// prover choice.
///
/// Why the grid needs it and no other region does: `commit_base` pads the
/// numerator column with `a.table[0]` rather than zero, because the exp lookup
/// reads `(idx, num)` on every row and index 0 has to stay a real table row for
/// the multiplicities to balance.  Every other padded column cancels on its own
/// -- `idx` pads with 0, `vals` pad with the offset the fold subtracts back --
/// so the numerator fold is the one place the AIR has to be told the row is not
/// there.
pub const KEY_ACT: &str = "b6_k_act";
pub const DIAGONAL_BLOCK_ACT: &str = "b6_x_act";
/// The `act` column's id for a key tile, `None` when the grid tiles exactly and the
/// column is not committed at all.  One function, so `commit_pre`, `pre_ids`,
/// `attn_run` and the AIR cannot disagree about whether it is there.
pub fn act_id(tag: usize, diagonal_block: bool, lead: usize) -> Option<PreProcessedColumnId> {
    (lead > 0).then(|| key_tile_id(tag, if diagonal_block { DIAGONAL_BLOCK_ACT } else { KEY_ACT }))
}
/// The diagonal block's causal mask, one preprocessed column per token slot of a row:
/// 1 where the key is in the decoded stream's future of that token.  A slot's
/// token is a function of the row *and* the column, so the mask cannot be one
/// column; what makes it affordable is that it lives on the diagonal block's rows alone
/// (`S` keys, not `C`), which is 4.2 M preprocessed cells a key tile at 8B.
pub fn msk_id(tag: usize, i: usize) -> PreProcessedColumnId {
    id(format!("b6_x_msk{i}_key_tile{tag}"))
}
pub const OUT_SEL: [&str; 5] =
    ["b6_o_nofirst", "b6_o_chanend", "b6_o_m1", "b6_o_m2", "b6_o_end"];
pub const ROW_SEL: [&str; 3] = ["b6_r_notzero", "b6_r_end", "b6_r_istart"];
pub const TBL_EXP_X: &str = "b6_exp_x";
pub const TBL_EXP_Y: &str = "b6_exp_y";
pub fn tbl_range(bits: u32) -> PreProcessedColumnId {
    id(format!("b6_range_{bits}"))
}

/// Challenge powers, precomputed per region width.
#[derive(Clone, Debug)]
pub struct Powers {
    pub r: SecureField,
    pub rv: SecureField,
    /// r^0..r^{pk-1} and r^pk, for the key chunk of the channel region
    pub r_pow: Vec<SecureField>,
    pub r_step: SecureField,
    /// s^0..s^{pq-1} and s^pq, for the query chunk of the channel region
    pub sq_pow: Vec<SecureField>,
    pub sq_step: SecureField,
    /// s^0..s^{q-1} and s^q, for the grid region
    pub sg_pow: Vec<SecureField>,
    pub sg_step: SecureField,
    /// rv^0..rv^{pv-1} and rv^pv, for the value chunk of the grid region
    pub rv_pow: Vec<SecureField>,
    pub rv_step: SecureField,
    /// s^0..s^{qo-1} and s^qo, for the A.V output region
    pub so_pow: Vec<SecureField>,
    pub so_step: SecureField,
    /// s^0..s^{rw-1} and s^rw, for the per-token region
    pub sr_pow: Vec<SecureField>,
    pub sr_step: SecureField,
    /// sum_{c<C} r^c, the coefficient the shift is folded through.  Per *instance*:
    /// the packed key axis contributes `r^{C i}` on both sides, so this stays local.
    pub r_sum: SecureField,
    /// r^C -- the per-instance weight of the Q.K^T and denominator identities
    pub r_c: SecureField,
    /// rv^hd -- the per-instance weight of the A.V identity
    pub rv_hd: SecureField,
    /// the per-token region's instance-start multiplier, `r^C s^{-(S-rw)}`: its
    /// Horner runs in `s`, so this is what converts its natural weight `s^S` into
    /// the `r^C` every other region carries
    pub mu: SecureField,
    /// the per-instance weight of the *denominator* sum `sn`.  `sn` is a plain sum
    /// over keys with an explicit multiplier, so the constant is a free choice --
    /// and it has to be the same one on both grids, because `sn_prefix + sn_diagonal_block`
    /// is what `az` is read against.  `r^C` of the prefix; equal to `r_c` there.
    pub sn_c: SecureField,
    /// the diagonal block's instance-boundary multiplier `r^C r'^{-(S-1)}`, which turns the
    /// key Horner's natural `r'^S` per instance into the prefix's `r^C`.  Unused by
    /// the prefix grid, where the two already agree.
    pub kappa: SecureField,
}

pub fn powers(base: SecureField, n: usize) -> (Vec<SecureField>, SecureField) {
    let mut v = Vec::with_capacity(n);
    let mut cur = SecureField::one();
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    (v, cur)
}

impl Powers {
    pub fn new(r: SecureField, s: SecureField, rv: SecureField, sh: &Shape,
               lc: &LayoutChan, lk: &LayoutKey, lo: &LayoutOut, rw: usize) -> Self {
        let (r_pow, r_step) = powers(r, lc.pk);
        let (sq_pow, sq_step) = powers(s, lc.pq);
        let (sg_pow, sg_step) = powers(s, lk.q);
        let (rv_pow, rv_step) = powers(rv, lk.pv);
        let (so_pow, so_step) = powers(s, lo.q);
        let (sr_pow, sr_step) = powers(s, rw);
        let (all_r, r_c) = powers(r, sh.c);
        let r_sum = all_r.iter().fold(SecureField::zero(), |a, b| a + *b);
        let (_, rv_hd) = powers(rv, sh.hd);
        // s^{S-rw}, inverted: the per-token region multiplies by s^rw on every row
        // after the first of an instance, so `mu s^{S-rw} = r^C`.
        let (_, s_span) = powers(s, sh.s - rw);
        let mu = r_c * s_span.inverse();
        Self { r, rv, r_pow, r_step, sq_pow, sq_step, sg_pow, sg_step, rv_pow, rv_step,
               so_pow, so_step, sr_pow, sr_step, r_sum, r_c, rv_hd, mu,
               sn_c: r_c, kappa: r }
    }

    /// The diagonal block grid's powers: built on the diagonal block's own shape (`c = S`) and its
    /// own key challenge `r'`, then told the two constants it shares with the
    /// prefix -- the denominator's per-instance weight and the per-instance
    /// weight of the channel region.
    ///
    /// `mu` is deliberately *not* among them: it is read only by `RowEval`'s
    /// step and the per-token trace loop, and a diagonal block key tile has neither -- the
    /// pair's one per-token region is the prefix key tile's (B10 step 3), and the
    /// row maximum both halves shift by is *its* `am`, checked against the
    /// diagonal block's Q.K^T by `verify_composed`'s claim equality.  (An earlier
    /// version overrode `self.mu` here with a docstring claiming that override
    /// carried the shared-maximum identity; the override was dead code and the
    /// identity lives in the claim equality -- B10 review, stream B.)
    pub fn as_diagonal_block(mut self, prefix_r_c: SecureField, s_keys: usize) -> Self {
        let (_, rp_span) = powers(self.r, s_keys - 1);
        self.sn_c = prefix_r_c;
        self.kappa = prefix_r_c * rp_span.inverse();
        // and the channel region's own per-instance weight: `chan` is the side the
        // diagonal block's Q.K^T identity is read against, and the other two terms of that
        // identity -- `am` (the prefix's per-token region) and `hs` (via `kappa`)
        // -- both carry `r^C`.  Its natural `r'^S` would be a third weight.
        self.r_c = prefix_r_c;
        self
    }

    /// The **solo** diagonal block's powers: block 0 of a sequence has no unchecked
    /// prefix, so its diagonal block is its head key tile group's only key tile and it carries
    /// the per-token region itself (CC deliverable 4, option A).
    ///
    /// That is `as_diagonal_block` plus one thing.  `as_diagonal_block` deliberately drops `mu`
    /// because a *paired* diagonal block has no per-token region and rides on the
    /// prefix key tile's; a solo one has one and needs its own.  `Powers::new`
    /// already built `mu = r_c s^{-(S-rw)}` from **this** grid's `r_c`
    /// (`r_diagonal_block^S`), so the `s` factor is recoverable as `mu / r_c` without
    /// re-deriving `s` or `rw`, and rebasing it onto the borrowed global `r^C`
    /// gives the region the same instance weight every other region carries.
    pub fn as_solo_diagonal_block(self, prefix_r_c: SecureField, s_keys: usize) -> Self {
        let s_span_inv = self.mu * self.r_c.inverse();
        let mut p = self.as_diagonal_block(prefix_r_c, s_keys);
        p.mu = prefix_r_c * s_span_inv;
        p
    }
}

/// Offsets that make a signed committed value a non-negative table index.
#[derive(Clone, Copy, Debug)]
pub struct Offsets {
    pub q: u32,
    pub k: u32,
    pub v: u32,
}

impl Offsets {
    pub fn new(q_bits: u32, k_bits: u32) -> Self {
        Self { q: 1 << (q_bits - 1), k: 1 << (k_bits - 1), v: 128 }
    }
}

fn sum_pow(p: &[SecureField]) -> SecureField {
    p.iter().fold(SecureField::zero(), |a, b| a + *b)
}

/// Stream NN: one Horner claim over a key tile's own committed Q, K or V column.
///
/// The key tile's three existing folds are all *bilinear* -- `u v` for `Q.K^T`,
/// `wv zn` for `A.V` -- so none of them publishes Q, K or V alone, and the
/// value the layer's matmuls produced had nothing in the proof to be compared
/// against (`notes/witness-slack.md` section 1e).  This is that missing sum: one
/// extension accumulator per column set, weighting the cell at (query head
/// `hq`, channel `ch`, token `t`) by `w^(N - 1 - (t nh hd + hq hd + ch))` --
/// the **token-major** canonical position the quantize region folds under, so
/// the two claims are directly comparable.
///
/// The offset every key tile commits (`off_k`, `off_q`, 128) is subtracted here for
/// the same reason `kf` subtracts it: an inert lead row and an out-of-range slot
/// both commit exactly the offset, so they fold to zero and no `act` selector is
/// needed.
#[derive(Clone)]
pub struct KeyTileTie {
    /// per slot within a row
    pub w: Vec<SecureField>,
    /// the row step inside one row group
    pub s_in: SecureField,
    /// the row step at a group boundary
    pub s_bd: SecureField,
    /// the row step at an *instance* boundary; `None` where the group index is
    /// affine in the position (the channel region) and one boundary constant
    /// serves
    pub s_ib: Option<SecureField>,
    /// the offset the key tile's cells carry
    pub off: u32,
    pub claimed: SecureField,
}

impl KeyTileTie {
    /// The claim, from the values rather than from the trace: `vals[j][k]` is
    /// the key tile's own committed cell, offset included.
    pub fn claim_of(&self, rows: usize, per_row: usize, vals: &[i64],
                    step: &dyn Fn(usize) -> SecureField) -> SecureField {
        let off = SecureField::from(M31::from_u32_unchecked(self.off));
        let mut h = SecureField::zero();
        for j in 0..rows {
            let mut f = SecureField::zero();
            for k in 0..per_row {
                let v = SecureField::from(crate::attn_trace::m31_of(vals[j * per_row + k])) - off;
                f += v * self.w[k];
            }
            h = h * step(j) + f;
        }
        h
    }
}

// ------------------------------------------------------------ Q.K^T region ---

/// One row group per channel: the two inner sums of `Q.K^T`, multiplied per
/// channel and accumulated.
#[derive(Clone)]
pub struct ChanEval {
    pub tag: usize,
    pub lay: LayoutChan,
    pub pow: Powers,
    pub off: Offsets,
    pub claimed: SecureField,
    pub rel_k: RangeRel,
    pub rel_q: RangeRel,
    pub range_check: bool,
    /// B11 stream S, lever 1: the region's lookups are proved by the batched
    /// GKR instead of committed running-sum columns, so the AIR emits neither
    /// the relation entries nor the interaction masks they would allocate.
    pub gkr: bool,
    /// Stream AA: the K/V boundary trees this key tile's committed **keys** are
    /// copied into -- 0, 1 or 2 of them (an unchecked-prefix sub-key-tile writes both
    /// its group's out and in sides; the chunk's own block writes the out side
    /// alone).  Empty on a chunk proved with `--kv-boundary` off, which is why that
    /// chunk's constraint IR, its kernel fingerprint and its proof are
    /// byte-for-byte the old ones.
    pub boundary_k: Vec<crate::boundary::KvBoundary>,
    /// Stream NN: this key tile's own K and Q claims, in that order.  `Some` only on
    /// a **diagonal block** key tile of a chunk that carries the RoPE regions -- the diagonal block's
    /// keys are this block's own tokens, which are the ones the layer's `k`
    /// matmul produced; a prefix key tile's keys are the cache and are bound across
    /// blocks by root equality instead (stream AA/EE).  `None` everywhere else,
    /// which is why every other chunk's constraint IR and kernel fingerprint are
    /// byte-for-byte the old ones.
    pub qk_bind: Option<(KeyTileTie, KeyTileTie)>,
    /// Stream NN (NN-2): this key tile's own Q claim alone, on a **prefix** key tile.
    ///
    /// A key tile group is a prefix key tile and a diagonal block over the same heads and the
    /// same `S` queries, and each commits its own `Q` column.  NN-1 bound the
    /// diagonal block's and left the prefix's untied, so a prover could attend to the
    /// cache with queries the `q` matmul did not produce while attending to
    /// this block's own keys with queries it did.  This is the other half: the
    /// same fold under the same canonical weights, and one equality per group
    /// (`nn_q_ok`) saying the two are the same queries.
    ///
    /// At most one of `qk_bind` and `q_bind` is `Some` on any key tile, so the
    /// interaction masks below are read in one unambiguous order.
    pub q_bind: Option<KeyTileTie>,
}

impl FrameworkEval for ChanEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_chan_eval" }

    fn log_size(&self) -> u32 {
        self.lay.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.lay.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> =
            CHAN_SEL.iter().map(|n| eval.get_preprocessed_column(key_tile_id(self.tag, n))).collect();
        let (nofirst, last, notzero, end, istart) =
            (sel[0].clone(), sel[1].clone(), sel[2].clone(), sel[3].clone(), sel[4].clone());
        let ks: Vec<E::F> = (0..self.lay.pk).map(|_| eval.next_trace_mask()).collect();
        // ---- the K half of the K/V boundary (stream AA) ----------------------
        // The exact analogue of the residual-stream boundary at `residual_stream.rs`'s `leaf_in`, and
        // simpler: a key is a single integer, so there is no `PACK` and the
        // constraint is a bare equality, one cell per value.  Being an equality
        // and not a lookup is what lets these trees be committed *after* the
        // interaction tree -- it is checked at the OODS point, drawn after every
        // commitment, so a prover who knows the earlier challenges has gained
        // nothing, and `INTERACTION_TRACE_IDX = 2` stays where the vendored
        // framework and every generated kernel expect it.  See
        // `notes/bounds-design.md` and `boundary.rs`'s stream AA section.
        for b in &self.boundary_k {
            assert_eq!(b.n, self.lay.pk, "a K boundary side's width is not the key tile's `pk`");
            for kp in ks.iter() {
                let [c] = eval.next_interaction_mask(b.tree, [0]);
                eval.add_constraint(c - kp.clone());
            }
        }
        let qs: Vec<E::F> = (0..self.lay.pq).map(|_| eval.next_trace_mask()).collect();

        let [u_prev, u] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [v_prev, v] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [acc_prev, acc] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);

        let mut kf = E::EF::zero();
        for (p, kp) in ks.iter().enumerate() {
            kf += kp.clone() * self.pow.r_pow[p];
        }
        kf = kf - sum_pow(&self.pow.r_pow) * SecureField::from_u32_unchecked(self.off.k, 0, 0, 0);
        eval.add_constraint(u.clone() - (u_prev * self.pow.r_step) * nofirst.clone() - kf);

        let mut qf = E::EF::zero();
        for (p, qp) in qs.iter().enumerate() {
            qf += qp.clone() * self.pow.sq_pow[p];
        }
        qf = qf - sum_pow(&self.pow.sq_pow) * SecureField::from_u32_unchecked(self.off.q, 0, 0, 0);
        eval.add_constraint(v.clone() - (v_prev * self.pow.sq_step) * nofirst.clone() - qf);

        // a plain sum over channels within an instance, times r^C at each instance
        // start, so the packed claim carries the same weight the grid's r-Horner
        // gives its side of the identity
        let mut mult: E::EF = E::EF::from(notzero.clone() - istart.clone());
        mult += istart * self.pow.r_c;
        eval.add_constraint(acc.clone() - acc_prev * mult - (u * v) * last);
        let claim = eval.claim(self.claimed);
        eval.add_constraint((acc - claim) * end.clone());

        // ---- stream NN: the K and Q claims -----------------------------------
        // `u` and `v` above are inner sums of one row group and are consumed by
        // the bilinear `acc`; these two run over the whole region under the
        // quantize region's own weights, so an equality against it says the
        // key tile's columns are the values the layer's matmuls produced.  Both
        // steps are built from selectors the region already reads.
        if let Some((kt, qt)) = &self.qk_bind {
            for (t, vals) in [(kt, &ks), (qt, &qs)] {
                let [h_prev, h] =
                    eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
                let step = E::EF::from(nofirst.clone()) * t.s_in
                    + (E::EF::from(notzero.clone()) - E::EF::from(nofirst.clone())) * t.s_bd;
                let off = E::F::from(M31::from_u32_unchecked(t.off));
                let mut f = E::EF::zero();
                for (k, c) in vals.iter().enumerate() {
                    f += E::EF::from(c.clone() - off.clone()) * t.w[k];
                }
                eval.add_constraint(h.clone() - h_prev * step - f);
                let claim = eval.claim(t.claimed);
                eval.add_constraint((h - claim) * end.clone());
            }
        }

        // ---- stream NN (NN-2): the prefix key tile's Q claim ----------------------
        // The same fold as the diagonal block's `qt` above, on this key tile's own layout.
        // The diagonal block is not on this arm and the prefix is not on that one, so
        // no key tile reads both and the mask order is the trace's.
        if let Some(t) = &self.q_bind {
            let [h_prev, h] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let step = E::EF::from(nofirst.clone()) * t.s_in
                + (E::EF::from(notzero.clone()) - E::EF::from(nofirst.clone())) * t.s_bd;
            let off = E::F::from(M31::from_u32_unchecked(t.off));
            let mut f = E::EF::zero();
            for (k, c) in qs.iter().enumerate() {
                f += E::EF::from(c.clone() - off.clone()) * t.w[k];
            }
            eval.add_constraint(h.clone() - h_prev * step - f);
            let claim = eval.claim(t.claimed);
            eval.add_constraint((h - claim) * end.clone());
        }

        if self.range_check && !self.gkr {
            for kp in &ks {
                eval.add_to_relation(RelationEntry::new(&self.rel_k, E::EF::one(),
                                                        core::slice::from_ref(kp)));
            }
            for qp in &qs {
                eval.add_to_relation(RelationEntry::new(&self.rel_q, E::EF::one(),
                                                        core::slice::from_ref(qp)));
            }
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

// -------------------------------------------------------------- grid region ---

/// One row group per key: the whole `(t, c)` grid, and A.V's inner sums.
#[derive(Clone)]
pub struct KeyEval {
    pub tag: usize,
    pub lay: LayoutKey,
    pub pow: Powers,
    pub off: Offsets,
    pub variant: Variant,
    /// end-of-trace values of the three accumulators this region claims
    pub claimed_hs: SecureField,
    pub claimed_av: SecureField,
    pub claimed_sn: SecureField,
    pub rel_exp: ExpRel,
    pub rel_v: RangeRel,
    /// the split variant's remainder table
    pub rel_split: Option<RangeRel>,
    pub range_check: bool,
    /// B11 stream S, lever 1: the region's lookups are proved by the batched
    /// GKR instead of committed running-sum columns (see `ChanEval::gkr`).
    pub gkr: bool,
    /// B10 step 3: this is the *diagonal block*, the S x S grid of the decoded
    /// tokens against each other.  It reads its own selectors, one causal-mask
    /// column per token slot, and the instance-boundary multiplier `kappa`.  Its
    /// masked slots stay in the score fold -- Freivalds is over the whole
    /// rectangle and a masked entry is an honest score -- and leave the numerator
    /// fold and the exp lookup, which is what "the token did not attend to it"
    /// means in the AIR.
    pub diagonal_block: bool,
    /// Stream AA: the K/V boundary trees this key tile's committed **values** are
    /// copied into -- the V half of `ChanEval::boundary_k`, same sides, same
    /// order, so that one tree holds `[key_tile's K][key_tile's V]` per key tile in
    /// component-registration order.
    pub boundary_v: Vec<crate::boundary::KvBoundary>,
    /// Stream NN: this key tile's own V claim, the counterpart of `ChanEval::qk_bind`.
    /// `Some` only on a diagonal block key tile of a chunk that carries the quantize regions.
    pub v_bind: Option<KeyTileTie>,
}

impl FrameworkEval for KeyEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_key_eval" }

    fn log_size(&self) -> u32 {
        self.lay.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.lay.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let names: &[&str] = if self.diagonal_block { &DIAGONAL_BLOCK_SEL } else { &KEY_SEL };
        let sel: Vec<E::F> =
            names.iter().map(|n| eval.get_preprocessed_column(key_tile_id(self.tag, n))).collect();
        let (nofirst, last, notzero, end, m1, m2, istart) = (sel[0].clone(), sel[1].clone(),
            sel[2].clone(), sel[3].clone(), sel[4].clone(), sel[5].clone(), sel[6].clone());
        // stream NN: the V claim's instance-boundary step reads the same column
        let istart_v = sel[6].clone();
        // Stream CC (E11): 1 on a row the grid actually holds, 0 on the inert
        // lead rows `Grid` pads with.  Absent -- and read as the literal `1` --
        // on a grid that tiles exactly, which is every shape proved before E11,
        // so the lowered constraint there is character for character the one it
        // was.
        let act: E::F = match act_id(self.tag, self.diagonal_block, self.lay.g.lead) {
            Some(c) => eval.get_preprocessed_column(c),
            None => E::F::one(),
        };
        // the causal mask, one column per token slot; empty on the prefix grid
        let msk: Vec<E::F> = (0..if self.diagonal_block { self.lay.q } else { 0 })
            .map(|i| eval.get_preprocessed_column(msk_id(self.tag, i)))
            .collect();
        let vis = |i: usize| -> E::F {
            match msk.get(i) {
                Some(m) => E::F::one() - m.clone(),
                None => E::F::one(),
            }
        };
        // The numerator fold's visibility: `vis` **and** the row is real.  Written
        // as `act - msk` rather than `act * (1 - msk)` on purpose -- the two agree
        // because `msk` is built from `Grid::row(j)` and so is already zero
        // wherever `act` is, and the subtraction keeps the factor degree 1, so the
        // fold's degree is the one `max_constraint_log_degree_bound` promises.
        let nvis = |i: usize| -> E::F {
            match msk.get(i) {
                Some(m) => act.clone() - m.clone(),
                None => act.clone(),
            }
        };
        // 2^-(F - frac_i): the committed index is the shift scaled to the master
        // table's exponent, and this divides it back out (one preprocessed column
        // for the whole key tile, zero fresh cells -- the cost of sharing is table rows)
        let dinv = sel[7].clone();
        let q = self.lay.q;
        let idx: Vec<E::F> = (0..q).map(|_| eval.next_trace_mask()).collect();
        let num: Vec<E::F> = (0..q).map(|_| eval.next_trace_mask()).collect();
        let aux: Vec<E::F> = (0..self.n_aux()).map(|_| eval.next_trace_mask()).collect();
        let vals: Vec<E::F> = (0..self.lay.pv).map(|_| eval.next_trace_mask()).collect();
        // ---- the V half of the K/V boundary (stream AA) ----------------------
        // `ChanEval`'s block, on the other geometry: the key region has one row
        // group per `(instance, key)` and holds head channels on the column
        // axis, so a V column is `lay.pv` wide where a K column is `lay.pk`.
        // Nothing else differs -- same trees, same order, same bare equality.
        for b in &self.boundary_v {
            assert_eq!(b.n, self.lay.pv, "a V boundary side's width is not the key tile's `pv`");
            for vp in vals.iter() {
                let [c] = eval.next_interaction_mask(b.tree, [0]);
                eval.add_constraint(c - vp.clone());
            }
        }

        let [zs_prev, zs] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [hs_prev, hs] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [zn_prev, zn] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [wv_prev, wv] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [av_prev, av] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [sn_prev, sn] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);

        // the value the score accumulator sees: the committed column, plus the
        // remainder the split variant carries alongside it
        let score: Vec<E::F> = (0..q)
            .map(|i| match self.variant {
                Variant::IntSplit(k) => (idx[i].clone() * M31::from_u32_unchecked(1 << k)
                    + aux[i].clone()) * dinv.clone(),
                _ => idx[i].clone() * dinv.clone(),
            })
            .collect();

        let mut sf = E::EF::zero();
        for (i, si) in score.iter().enumerate() {
            sf += si.clone() * self.pow.sg_pow[i];
        }
        eval.add_constraint(zs.clone() - (zs_prev * self.pow.sg_step) * nofirst.clone() - sf);

        // the numerator fold drops the masked slots: they are not keys this token
        // attended to, so they are in neither `Z` nor `A.V`.  The score fold above
        // keeps them -- Freivalds is over the whole rectangle.
        //
        // Stream CC (E11): it also drops the inert lead rows.  `commit_base` pads
        // the numerator column with `a.table[0]`, a real table row so the exp
        // lookup's multiplicities stay exact, and both interaction builders --
        // `interaction_phase` here and `b65a_grid_kernel`, which gates `nf` on
        // `act` -- contribute no term on such a row.  Without `act` this fold
        // demanded `zn = table[0] * sum sg_pow[i]` where the builders wrote 0, and
        // that single disagreement is why an honest witness at a padded key grid
        // returned `ConstraintsNotSatisfied`.  The `idx` fold above needs no guard:
        // its pad is zero.
        let mut nf = E::EF::zero();
        for (i, ni) in num.iter().enumerate() {
            nf += (ni.clone() * nvis(i)) * self.pow.sg_pow[i];
        }
        eval.add_constraint(zn.clone() - (zn_prev * self.pow.sg_step) * nofirst.clone() - nf);

        let mut vf = E::EF::zero();
        for (p, vp) in vals.iter().enumerate() {
            vf += vp.clone() * self.pow.rv_pow[p];
        }
        vf = vf - sum_pow(&self.pow.rv_pow) * SecureField::from_u32_unchecked(self.off.v, 0, 0, 0);
        eval.add_constraint(wv.clone() - (wv_prev * self.pow.rv_step) * nofirst.clone() - vf);

        // sum over keys of wv * zn, closed on each key's last chunk row, times
        // rv^hd at each instance start (the A.V identity's per-instance weight,
        // which the output region's rv-Horner produces on its own side)
        let mut av_mult: E::EF = E::EF::from(notzero.clone() - istart.clone());
        av_mult += istart.clone() * self.pow.rv_hd;
        eval.add_constraint(av.clone() - av_prev * av_mult
            - (wv * zn.clone()) * last.clone());
        // sum over keys of zn: the denominator statement, weighted by r^C to match
        // the per-token region's `az`
        let mut sn_mult: E::EF = E::EF::from(notzero.clone() - istart.clone());
        sn_mult += istart * self.pow.sn_c;
        eval.add_constraint(sn.clone() - sn_prev * sn_mult - zn.clone() * last);
        // r^c on each completed key (keys are laid out in descending order).  On the
        // diagonal block, the row that closes an instance's key block takes `kappa` instead,
        // which is `r^C r'^{-(S-1)}`: the block's own weights stay geometric in `r'`
        // and the instance stride becomes the prefix's.
        let mut mult: E::EF = E::EF::from(m1);
        mult += m2.clone() * self.pow.r;
        if self.diagonal_block {
            mult += sel[8].clone() * (self.pow.kappa - self.pow.r);
        }
        eval.add_constraint(hs.clone() - hs_prev * mult - zs * self.key_end(&sel));

        let claim_hs = eval.claim(self.claimed_hs);
        eval.add_constraint((hs - claim_hs) * end.clone());
        let claim_av = eval.claim(self.claimed_av);
        eval.add_constraint((av - claim_av) * end.clone());
        let claim_sn = eval.claim(self.claimed_sn);
        eval.add_constraint((sn - claim_sn) * end.clone());

        // ---- stream NN: the V claim ------------------------------------------
        // `wv` above is one key row's inner sum and is consumed by the bilinear
        // `av`; this runs over the whole region under the quantize region's own
        // weights.  The key region groups by (instance, key), and the position
        // is not affine in that group index -- an instance boundary jumps a whole
        // context -- so this fold takes a third row step, on the region's own
        // `istart` selector.
        if let Some(t) = &self.v_bind {
            let [h_prev, h] =
                eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
            let s_ib = t.s_ib.expect("the V binding needs an instance-boundary step");
            let step = E::EF::from(nofirst.clone()) * t.s_in
                + (E::EF::from(notzero.clone()) - E::EF::from(nofirst.clone())
                   - E::EF::from(istart_v.clone())) * t.s_bd
                + E::EF::from(istart_v.clone()) * s_ib;
            let off = E::F::from(M31::from_u32_unchecked(t.off));
            let mut f = E::EF::zero();
            for (k, c) in vals.iter().enumerate() {
                f += E::EF::from(c.clone() - off.clone()) * t.w[k];
            }
            eval.add_constraint(h.clone() - h_prev * step - f);
            let claim = eval.claim(t.claimed);
            eval.add_constraint((h - claim) * end);
        }

        if !self.gkr {
            // the exp lookups: index and numerator together, arity 2
            for i in 0..q {
                let key = match self.variant {
                    Variant::Bf16x => aux[i].clone(),
                    _ => idx[i].clone(),
                };
                // a masked slot is looked up at multiplicity zero: it never entered the
                // numerator fold, and its shift may be negative (the row maximum is over
                // the keys the token *did* attend to), so it is not a table row at all.
                eval.add_to_relation(RelationEntry::new(&self.rel_exp, E::EF::from(vis(i)),
                                                        &[key, num[i].clone()]));
            }
            // the remainder the table's granularity leaves behind, when there is one
            if let Some(rel) = &self.rel_split {
                for ai in aux.iter() {
                    eval.add_to_relation(RelationEntry::new(rel, E::EF::one(),
                                                            core::slice::from_ref(ai)));
                }
            }
            if self.range_check {
                for vp in &vals {
                    eval.add_to_relation(RelationEntry::new(&self.rel_v, E::EF::one(),
                                                            core::slice::from_ref(vp)));
                }
            }
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

impl KeyEval {
    pub fn n_aux(&self) -> usize {
        match self.variant {
            Variant::IntExp => 0,
            _ => self.lay.q,
        }
    }
    fn key_end<F: Clone>(&self, sel: &[F]) -> F {
        sel[1].clone() // `last` is also the chunk-end selector for the key Horner
    }
}

// --------------------------------------------------------------- A.V outputs ---

#[derive(Clone)]
pub struct OutEval {
    pub tag: usize,
    pub lay: LayoutOut,
    pub pow: Powers,
    pub claimed: SecureField,
}

impl FrameworkEval for OutEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_out_eval" }

    fn log_size(&self) -> u32 {
        self.lay.g.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.lay.g.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> =
            OUT_SEL.iter().map(|n| eval.get_preprocessed_column(key_tile_id(self.tag, n))).collect();
        let (nofirst, chanend, m1, m2, end) = (sel[0].clone(), sel[1].clone(),
            sel[2].clone(), sel[3].clone(), sel[4].clone());
        let ys: Vec<E::F> = (0..self.lay.q).map(|_| eval.next_trace_mask()).collect();
        let [z_prev, z] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [h_prev, h] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);

        let mut yf = E::EF::zero();
        for (i, yi) in ys.iter().enumerate() {
            yf += yi.clone() * self.pow.so_pow[i];
        }
        eval.add_constraint(z.clone() - (z_prev * self.pow.so_step) * nofirst - yf);
        let mut mult: E::EF = E::EF::from(m1);
        mult += m2 * self.pow.rv;
        eval.add_constraint(h.clone() - h_prev * mult - z * chanend);
        let claim = eval.claim(self.claimed);
        eval.add_constraint((h - claim) * end);
        eval
    }
}

// ------------------------------------------------------- per-token region ---

/// One row per `rw` query tokens: the committed maxima and denominators, and the
/// two-limb range check that makes the A.V window explicit.
#[derive(Clone)]
pub struct RowEval {
    pub tag: usize,
    pub log_rows: u32,
    pub rw: usize,
    pub pow: Powers,
    pub with_max: bool,
    pub claimed_am: SecureField,
    pub claimed_az: SecureField,
    /// low limb width of the denominator range check
    pub z_lo_bits: u32,
    /// lower bound on the denominator.  An honest shift is the row maximum, so
    /// some key has `d = 0` and contributes the full-scale numerator: `Z >= nmax`
    /// holds by construction, and requiring it caps how far a prover can shift
    /// *past* the maximum -- the one direction the range check on `d` leaves open.
    pub z_min: u32,
    pub rel_lo: RangeRel,
    pub rel_hi: RangeRel,
    /// B11 stream S, lever 1: the region's lookups are proved by the batched
    /// GKR instead of committed running-sum columns (see `ChanEval::gkr`).
    pub gkr: bool,
}

impl FrameworkEval for RowEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_row_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let sel: Vec<E::F> =
            ROW_SEL.iter().map(|n| eval.get_preprocessed_column(key_tile_id(self.tag, n))).collect();
        let (notzero, end, istart) = (sel[0].clone(), sel[1].clone(), sel[2].clone());
        // s^rw inside an instance, mu at an instance start
        let step = |istart: E::F, notzero: E::F, pow: &Powers| -> E::EF {
            let mut m: E::EF = E::EF::from(notzero - istart.clone()) * pow.sr_step;
            m += istart * pow.mu;
            m
        };
        let ms: Vec<E::F> = (0..if self.with_max { self.rw } else { 0 })
            .map(|_| eval.next_trace_mask()).collect();
        let zs: Vec<E::F> = (0..self.rw).map(|_| eval.next_trace_mask()).collect();
        let los: Vec<E::F> = (0..self.rw).map(|_| eval.next_trace_mask()).collect();
        let los2: Vec<E::F> = (0..self.rw).map(|_| eval.next_trace_mask()).collect();
        let [am_prev, am] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);
        let [az_prev, az] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [-1, 0]);

        let mut mf = E::EF::zero();
        for (i, mi) in ms.iter().enumerate() {
            mf += mi.clone() * self.pow.sr_pow[i];
        }
        eval.add_constraint(
            am.clone() - am_prev * step(istart.clone(), notzero.clone(), &self.pow) - mf);
        let mut zf = E::EF::zero();
        for (i, zi) in zs.iter().enumerate() {
            zf += zi.clone() * self.pow.sr_pow[i];
        }
        eval.add_constraint(
            az.clone() - az_prev * step(istart, notzero, &self.pow) - zf);
        let claim_am = eval.claim(self.claimed_am);
        eval.add_constraint((am - claim_am) * end.clone());
        let claim_az = eval.claim(self.claimed_az);
        eval.add_constraint((az - claim_az) * end);

        // Z = lo + 2^b hi and Z - z_min = lo2 + 2^b hi2, all four limbs
        // range-checked.  The upper bound is what the A.V integer reading relies
        // on, stated in the AIR instead of assumed from a blind worst case; the
        // lower bound is what stops a prover from shifting the whole row into
        // underflow, where every numerator is zero and the softmax is 0/0.
        if !self.gkr {
            let inv = M31::from_u32_unchecked(1 << self.z_lo_bits).inverse();
            let zmin = M31::from_u32_unchecked(self.z_min);
            for i in 0..self.rw {
                let hi = (zs[i].clone() - los[i].clone()) * inv;
                let hi2 = (zs[i].clone() - E::F::from(zmin) - los2[i].clone()) * inv;
                eval.add_to_relation(RelationEntry::new(&self.rel_lo, E::EF::one(),
                                                        core::slice::from_ref(&los[i])));
                eval.add_to_relation(RelationEntry::new(&self.rel_hi, E::EF::one(),
                                                        core::slice::from_ref(&hi)));
                eval.add_to_relation(RelationEntry::new(&self.rel_lo, E::EF::one(),
                                                        core::slice::from_ref(&los2[i])));
                eval.add_to_relation(RelationEntry::new(&self.rel_hi, E::EF::one(),
                                                        core::slice::from_ref(&hi2)));
            }
            eval.finalize_logup_in_pairs();
        }
        eval
    }
}

// ------------------------------------------------------------------ tables ---

/// The exp table: `(index, numerator)` over all its rows, yielded at the
/// multiplicity the prover committed.
#[derive(Clone)]
pub struct ExpTableEval {
    pub log_rows: u32,
    pub rel: ExpRel,
}

impl FrameworkEval for ExpTableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_exp_table_eval" }

    fn log_size(&self) -> u32 {
        self.log_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let row = [eval.get_preprocessed_column(id(TBL_EXP_X)),
                   eval.get_preprocessed_column(id(TBL_EXP_Y))];
        let m = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(&self.rel, -E::EF::from(m), &row));
        eval.finalize_logup();
        eval
    }
}

/// A `0 .. 2^bits` range table.
#[derive(Clone)]
pub struct RangeTableEval {
    pub bits: u32,
    pub rel: RangeRel,
}

impl FrameworkEval for RangeTableEval {
    /// milestone 2b: the flat parameter block for the generated
    /// constraint kernel (`gpu/m2b/cudagen.rs`).
    fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
        crate::cudagen::blob_for(self)
    }

    /// milestone 2a: no kernel is registered under this name, so the
    /// fork routes this component to `accumulate_pointwise_cpu`.
    fn cuda_eval_name(&self) -> &'static str { "b65_range_table_eval" }

    fn log_size(&self) -> u32 {
        self.bits
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.bits + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let v = eval.get_preprocessed_column(tbl_range(self.bits));
        let m = eval.next_trace_mask();
        eval.add_to_relation(RelationEntry::new(&self.rel, -E::EF::from(m),
                                                core::slice::from_ref(&v)));
        eval.finalize_logup();
        eval
    }
}
