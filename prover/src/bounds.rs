//! Stream H — the boundary commitment: what makes two chunk proofs chain.
//!
//! The design and its soundness argument live in `notes/bounds-design.md`; this
//! module is the implementation, and the comments here only say what the code
//! does and which part of the note it is.
//!
//! The layer statement already runs from a committed residual-stream input to a
//! committed residual-stream output — the glue plan's `stream_in` and
//! `stream_out` Horner claims. Those claims cannot chain, because the challenge
//! they are taken under is drawn inside the proof that carries them, so two
//! chunks weigh their boundaries differently. Worse, the repair that suggests
//! itself — draw the challenge from public data so both chunks share it — is
//! unsound, because a challenge known before the values are committed makes
//! "my image equals yours" one linear condition the prover can simply solve.
//!
//! So the exposed value is a **commitment root** and not a challenge image.
//! Each chunk commits two more Merkle trees, one per boundary side, each holding
//! that side's values and nothing else:
//!
//! ```text
//!   tree 3   bin    the residual-stream input   (region `gn1`, group 0's leaves)
//!   tree 4   bout   the residual-stream output  (region `gn2`, the add's outputs)
//! ```
//!
//! The AIR ties each boundary column to the region column the rest of the layer
//! statement already binds, by a **plain equality constraint** — no lookup, no
//! challenge, no accumulator. That is what lets the two trees be committed
//! *after* the interaction tree: an equality constraint is checked at the OODS
//! point, which is drawn after every commitment, so the prover gains nothing
//! from knowing the earlier challenges. Committing after the interaction tree in
//! turn is what keeps `INTERACTION_TRACE_IDX = 2` where the vendored framework
//! and every generated kernel expect it.
//!
//! The roots need no new statement field: stwo serializes every tree root in
//! `StarkProof::commitments`, so `chain_check` below reads them out of the proof
//! bytes. The pair check is one root comparison, after five integer comparisons
//! that pin the layout — O(1), and no witness anywhere near it.
//!
//! Stream AA added a second boundary on a second axis (the K/V cache along the
//! token-block axis), and with it the question of *which* boundary a given pair
//! of chunks is supposed to share. See `Axis`: the two edge families are
//! disjoint, so the pair check derives the axis from the two statements and
//! asks the one link that axis carries.

use serde::Serialize;
use serde_json::{Value, json};
use stwo::core::fields::m31::M31;
// stream AA: `Column::at`/`set`, for the one-cell tamper on a K/V side's
// already-built columns
use stwo::prover::backend::Column;

use crate::common::{Eval, eval_of};
use crate::compose::{Fixed, GluePlan};
use crate::glue::{GlueRegion, TieKind};

/// The multiplier that folds a value's `(sm, ef)` pair into one cell.
///
/// A bf16 value is carried in the region as a signed significand and an
/// exponent. Packing them as `PACK * sm + ef` keeps the tie constraint degree 1
/// and is injective on the representation: `sm` spans only `{0} ∪ ±[128, 255]`
/// and `ef` only `[0, 255]`, so two distinct pairs differ by at least `PACK` in
/// the first term while the second can differ by at most 255, and the whole
/// range stays far inside the field. One cell per value is therefore the floor
/// for a degree-1 packing, and it is what the tree commits.
pub const PACK: i64 = 512;

/// The first tree index the boundary sides take. Trees 0–2 are the
/// preprocessed, base and interaction trees, in that order, and the note says
/// why the boundary trees follow rather than precede them.
pub const FIRST_TREE: usize = 3;

/// Whether the feature is on for this run. Off by default, deliberately: with
/// the flag off not one committed cell moves, which is what lets the recorded
/// digest stand as the bit-for-bit line.
pub fn enabled() -> bool {
    std::env::var("B10_BOUNDS").map(|v| v != "0").unwrap_or(false)
}

/// One side of a chunk's boundary: which tree it is committed in, which glue
/// region carries it, which of that region's value sequences it is, and the
/// shape that follows.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    pub tree: usize,
    pub kind: TieKind,
    /// values per row, which is also the tree's column count
    pub n: usize,
}

/// A boundary side as the prover needs it: the `Bound` the AIR reads, plus the
/// region it is taken from and the row count.
#[derive(Clone, Copy, Debug)]
pub struct Side {
    pub bound: Bound,
    pub region: usize,
    pub log_rows: u32,
}

/// The two sides, read off the glue plan. `stream_in` and `stream_out` already
/// name the two ties that are the layer's boundary, so this is only turning that
/// naming into a tree assignment — the input side first, so tree 3 is always the
/// input and tree 4 always the output, in every chunk of every shape.
pub fn sides(gp: &GluePlan) -> Vec<Side> {
    [gp.stream_in, gp.stream_out]
        .iter()
        .enumerate()
        .map(|(i, (region, tie))| {
            let r = &gp.regions[*region];
            let kind = r.ties[*tie].0;
            let n = match kind {
                TieKind::In(g) => r.prog.groups[g].n,
                TieKind::Out(o) => r.prog.outs[o].len(),
            };
            Side {
                bound: Bound { tree: FIRST_TREE + i, kind, n },
                region: *region,
                log_rows: r.log_rows,
            }
        })
        .collect()
}

/// One side's committed columns: `n` columns of `2^log_rows` rows, column `j`
/// holding the packed value at slot `j` of each row.
///
/// The values come from `GlueRegion::tie_vals`, which is the same sequence in
/// the same natural row order that the Horner tie accumulates — so the boundary
/// tree and the tie are two views of one thing, and the layout question reduces
/// to the one the glue family already answered. `gn1`'s input tie and `gn2`'s
/// output tie carry the same weight vector, which is exactly the statement that
/// they place canonical stream position `q` at the same `(row, slot)`; that is
/// why one chunk's output tree and the next chunk's input tree are comparable
/// at all.
///
/// `tamper` flips one cell, by flat index over `(row, slot)`. It exists for the
/// probe that has to show the tie bites: a boundary column that disagrees with
/// the region must take its own chunk's proof down.
pub fn cols(region: &GlueRegion, side: &Side, tamper: Option<usize>) -> Vec<Eval> {
    let vals = region.tie_vals(side.bound.kind);
    let w = region.tie_width(side.bound.kind);
    assert_eq!(w, side.bound.n, "the boundary's width disagrees with its plan");
    let rows = 1usize << side.log_rows;
    assert_eq!(vals.len(), rows * w, "the tie's values do not fill the region");
    (0..w)
        .map(|j| {
            let mut v: Vec<M31> = (0..rows)
                .map(|m| {
                    let (sm, ef) = vals[m * w + j];
                    crate::glue::m31_of(sm * PACK + ef)
                })
                .collect();
            if let Some(t) = tamper {
                let (tj, tm) = (t % w, (t / w) % rows);
                if tj == j {
                    v[tm] = v[tm] + M31::from_u32_unchecked(1);
                }
            }
            eval_of(side.log_rows, v)
        })
        .collect()
}

/// What the boundary cost this chunk, for the record: cells and columns per
/// side, and the totals.
pub fn report(sides: &[Side]) -> Value {
    let per: Vec<Value> = sides
        .iter()
        .map(|s| {
            json!({
                "tree": s.bound.tree,
                "columns": s.bound.n,
                "log_rows": s.log_rows,
                "cells": s.bound.n * (1usize << s.log_rows),
            })
        })
        .collect();
    json!({
        "sides": per,
        "columns": sides.iter().map(|s| s.bound.n).sum::<usize>(),
        "cells": sides.iter().map(|s| s.bound.n * (1usize << s.log_rows)).sum::<usize>(),
    })
}

// ------------------------------------------------------------------ chaining ---

/// Which edge of the chunk grid an ordered pair of chunks sits on.
///
/// A sequence proof is a two-dimensional grid of chunks indexed by layer `L`
/// and token block `b`, and it has **two** families of edges, on orthogonal
/// axes:
///
/// ```text
///                 block b        block b+1
///   layer L      (L, b)   --kv-->  (L, b+1)
///                   |                 |
///                residual          residual
///                   |                 |
///   layer L+1    (L+1, b) --kv-->  (L+1, b+1)
/// ```
///
/// * a **residual** edge runs `(L, b) -> (L+1, b)`: the same tokens, the next
///   layer.  Stream H binds it, by `roots_equal` on trees 3 and 4.
/// * a **K/V** edge runs `(L, b) -> (L, b+1)`: the same layer, the next token
///   block.  Stream AA binds it, by `kv_roots_equal` over the per-group out/in
///   trees.
///
/// The two edge sets are **disjoint**.  A residual edge has `c_a == c_b` (both
/// chunks declare the same prefix width, because they are the same tokens); a
/// K/V edge has `c_a + S == c_b`.  Both at once would need `S == 0`.
///
/// Stream AA's merged `chains()` *conjoined* the two links, and that predicate
/// holds on neither axis:
///
/// * on a **residual** edge it is false structurally, from the statements
///   alone.  `kv_shape_ok` carries the adjacency `c_a + S == c_b`, which
///   `c_a == c_b` cannot satisfy, so with `--kv-bind` on no assignment of roots
///   makes a layer pair chain.
/// * on a **K/V** edge it is false on the data.  `roots_equal` compares block
///   `b`'s residual *output* root against block `b+1`'s residual *input* root
///   -- two different token sets' residual streams, related by nothing, equal
///   only by collision.
///
/// So the H200 gate's step 5 FAIL (`kv_roots_equal` true, all three groups
/// equal, `roots_equal` false --
/// `experiments/b12/e11/out/h200/aa5_chain.json`) is a true report of a defect
/// in the predicate, not only of a gate asking the wrong question.
///
/// The axis is **derived from the two statements**, never chosen by the caller:
/// a chunk would have to declare a different `c` to move itself onto the other
/// axis, and its own `sizes` are a function of `c`.  Anything that is not one
/// of the two edges is `None` and does not chain -- the default is reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    /// `(L, b) -> (L+1, b)`: same token block, consecutive layer
    Layer,
    /// `(L, b) -> (L, b+1)`: same layer, consecutive token block
    Token,
    /// the pair is not an edge of the grid
    None,
}

impl Axis {
    pub fn name(&self) -> &'static str {
        match self {
            Axis::Layer => "layer",
            Axis::Token => "token",
            Axis::None => "none",
        }
    }
    pub fn parse(s: &str) -> Option<Axis> {
        match s {
            "layer" => Some(Axis::Layer),
            "token" => Some(Axis::Token),
            "none" => Some(Axis::None),
            _ => None,
        }
    }
}

/// The axis an ordered pair sits on, from the two statements' declared token
/// count and prefix width and nothing else.
///
/// `s_a != s_b` is not an edge at all: the two chunks decode different numbers
/// of tokens, so neither the residual boundary's row count nor the K/V tiling
/// can line up.
fn axis_of(a: &crate::compose::Params, b: &crate::compose::Params) -> Axis {
    if a.s != b.s || a.s == 0 {
        return Axis::None;
    }
    if a.c == b.c {
        Axis::Layer
    } else if a.c + a.s == b.c {
        Axis::Token
    } else {
        Axis::None
    }
}

/// What the pair check found. `chains` is the whole answer; the rest is so a
/// failure says which half of it failed.
#[derive(Debug, Clone, Serialize)]
pub struct ChainVerdict {
    /// which edge of the grid the pair sits on, derived from the statements
    pub axis: Axis,
    /// both statements carry the residual boundary, i.e. `--bounds` was on for
    /// both
    pub present: bool,
    /// the two statements agree on everything that fixes the boundary layout
    pub shape_ok: bool,
    /// the first chunk's output root equals the second's input root
    pub roots_equal: bool,
    pub out_root: Option<String>,
    pub in_root: Option<String>,
    /// Stream AA: both statements carry the K/V boundary.  `false` on a pair
    /// proved without `--kv-bind`, where the chain is the layer boundary's
    /// alone and says nothing about which prefix either chunk attended to.
    pub kv_present: bool,
    /// the two statements agree on everything that fixes the K/V layout, the
    /// adjacency `c_a + S == c_b` included
    pub kv_shape_ok: bool,
    /// every head band group's out root equals its successor's in root
    pub kv_roots_equal: bool,
    /// one entry per head band group, so a failure says which group moved
    pub kv: Vec<KvLink>,
    /// which conjunct of the pair's own axis failed, when it did.  `None` when
    /// the pair chains.
    pub reason: Option<String>,
    pub error: Option<String>,
}

/// One head band group's K/V link, for the record and for a failure to be
/// attributable.
#[derive(Debug, Clone, Serialize)]
pub struct KvLink {
    pub group: usize,
    pub out_tree: usize,
    pub in_tree: usize,
    pub out_root: String,
    pub in_root: String,
    pub equal: bool,
}

impl ChainVerdict {
    /// The pair's own axis decides which link is asked for, because the two
    /// links bind different things and no pair carries both (see `Axis`).
    ///
    /// * a **residual** edge is stream H's predicate exactly as it was, so a
    ///   pre-AA pair -- and every layer chain stream I and stream O recorded --
    ///   is answered by the same three booleans it always was;
    /// * a **K/V** edge asks for the K/V link and only the K/V link.  It
    ///   *requires* `kv_present`: a pair proved without `--kv-bind` is not a
    ///   K/V edge, so turning the flag off makes a token-block pair unchainable
    ///   rather than vacuously chained;
    /// * anything that is not an edge does not chain.
    ///
    /// This is narrower than the merged predicate on the residual axis (the K/V
    /// conjunct there was vacuously true anyway, `kv_present` being false
    /// whenever the flag was off) and it is satisfiable on the K/V axis, where
    /// the merged predicate was not.
    pub fn chains(&self) -> bool {
        self.reason.is_none() && self.error.is_none()
    }
    /// The failing conjunct of the pair's own axis, or `None` if it chains.
    fn why(axis: Axis, v: &Self) -> Option<String> {
        let need = |ok: bool, what: &str| (!ok).then(|| what.to_string());
        match axis {
            Axis::Layer => need(v.present, "the residual boundary is absent from one of the \
                                            statements (`--bounds` was off)")
                .or_else(|| need(v.shape_ok, "the two statements disagree about the residual \
                                              boundary's layout"))
                .or_else(|| need(v.roots_equal, "the first chunk's residual out root is not the \
                                                 second's in root")),
            Axis::Token => need(v.kv_present, "the K/V boundary is absent from one of the \
                                               statements (`--kv-bind` was off)")
                .or_else(|| need(v.kv_shape_ok, "the two statements disagree about the K/V \
                                                 boundary's layout"))
                .or_else(|| need(v.kv_roots_equal, "a head band group's out root is not its \
                                                    successor's in root")),
            Axis::None => Some(
                "the two chunks share no edge of the (layer, token block) grid: they are \
                 neither the same block at consecutive layers nor the same layer at \
                 consecutive blocks"
                    .to_string(),
            ),
        }
    }
    fn rejected(msg: &str) -> Self {
        Self {
            axis: Axis::None,
            present: false,
            shape_ok: false,
            roots_equal: false,
            out_root: None,
            in_root: None,
            kv_present: false,
            kv_shape_ok: false,
            kv_roots_equal: false,
            kv: vec![],
            reason: Some(msg.to_string()),
            error: Some(msg.to_string()),
        }
    }
}

/// Where a statement's K/V sides sit, from its parameters alone.
///
/// Arithmetic on `Fixed.params`, deliberately: the pair check never builds a
/// plan, never opens a commitment and never touches a witness, so it cannot be
/// made to panic by a malformed statement the way `Plan::new` can.  The layout
/// it derives is the one `Plan::new` builds -- stream H's two glue sides first,
/// then, per head band group, the out side and (unless the chunk is block 0,
/// which has no declared prefix) the in side.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    pub first: usize,
    pub groups: usize,
    pub with_in: bool,
}

impl KvLayout {
    pub fn out_tree(&self, g: usize) -> usize {
        self.first + if self.with_in { 2 * g } else { g }
    }
    pub fn in_tree(&self, g: usize) -> Option<usize> {
        self.with_in.then(|| self.first + 2 * g + 1)
    }
    pub fn n_sides(&self) -> usize {
        self.groups * if self.with_in { 2 } else { 1 }
    }
}

/// How many glue boundary sides a statement's parameters imply (stream H's two,
/// or none).
fn glue_sides(p: &crate::compose::Params) -> usize {
    if p.bounds && p.glue.is_some() { 2 } else { 0 }
}

pub fn kv_layout(p: &crate::compose::Params) -> Option<KvLayout> {
    p.kv_bind.then(|| KvLayout {
        first: FIRST_TREE + glue_sides(p),
        groups: p.bands.len(),
        with_in: p.c > 0,
    })
}

/// The number of committed trees a statement claims, and whether it has a
/// boundary at all.
fn tree_count(f: &Fixed) -> usize {
    f.sizes.len()
}

/// The verifier-side chain check: "chunk `a`'s output is chunk `b`'s input",
/// from four byte strings and nothing else.
///
/// No witness, no prover state, and no re-verification: each proof is accepted
/// or rejected on its own by `verify_composed`, and this is the separate
/// question of whether two accepted proofs sit next to each other. The shape
/// comparison comes first because the boundary layout is a function of
/// `(tokens, d, a, blowup)`; a mismatch there makes the roots incomparable, so
/// it has to fail the check rather than be compared through.
pub fn chain_check(
    a_fixed: &[u8],
    a_proof: &[u8],
    b_fixed: &[u8],
    b_proof: &[u8],
) -> ChainVerdict {
    let Ok(fa) = serde_json::from_slice::<Fixed>(a_fixed) else {
        return ChainVerdict::rejected("unparseable first statement");
    };
    let Ok(fb) = serde_json::from_slice::<Fixed>(b_fixed) else {
        return ChainVerdict::rejected("unparseable second statement");
    };
    let Ok(pa) = serde_json::from_slice::<crate::common::Proof>(a_proof) else {
        return ChainVerdict::rejected("unparseable first proof");
    };
    let Ok(pb) = serde_json::from_slice::<crate::common::Proof>(b_proof) else {
        return ChainVerdict::rejected("unparseable second proof");
    };
    let roots = |p: &crate::common::Proof| -> Vec<String> {
        p.commitments.iter().map(|c| format!("{c:?}")).collect()
    };
    chain_of(&fa, &roots(&pa), &fb, &roots(&pb))
}

/// The pair check over parsed statements and the two proofs' root lists.
///
/// Split out of `chain_check` so it can be exercised without building a
/// `StarkProof`: the roots are the only thing the check reads out of a proof,
/// and everything else it reads is `Fixed`.  The tests at the bottom of this
/// file drive this function.
pub fn chain_of(
    fa: &Fixed,
    roots_a: &[String],
    fb: &Fixed,
    roots_b: &[String],
) -> ChainVerdict {
    let (na, nb) = (tree_count(fa), tree_count(fb));
    let (ka, kb) = (kv_layout(&fa.params), kv_layout(&fb.params));
    let (ga, gb) = (glue_sides(&fa.params), glue_sides(&fb.params));
    // Stream AA: the expected tree count is a function of the statement --
    // three, plus stream H's two when `--bounds` is on, plus one per K/V side.
    // With `--kv-bind` absent and `--bounds` on this is `FIRST_TREE + 2`
    // exactly as it was, so a pair of pre-AA proofs takes the pre-AA path.
    let want = |n: usize, g: usize, k: &Option<KvLayout>| {
        FIRST_TREE + g + k.as_ref().map(|k| k.n_sides()).unwrap_or(0) == n
    };
    if !want(na, ga, &ka) || !want(nb, gb, &kb) {
        return ChainVerdict::rejected(
            "a statement's tree count is not the one its parameters imply \
             (the boundary flags were off, or the statement is malformed)",
        );
    }
    if roots_a.len() < na || roots_b.len() < nb {
        return ChainVerdict::rejected("a proof carries fewer roots than its statement claims");
    }
    if ga + gb == 0 && (ka.is_none() || kb.is_none()) {
        return ChainVerdict::rejected(
            "neither chunk carries a boundary: there is nothing to compare",
        );
    }
    // The residual boundary exists only when *both* chunks committed stream H's
    // two trees; without them there is no `FIRST_TREE + 1` to read.
    let glue_present = ga == 2 && gb == 2;
    // everything the boundary layout depends on: the token count, the two
    // wiring shape numbers, the blowup, and the two trees' own column sizes
    let shape_ok = glue_present
        && fa.params.s == fb.params.s
        && fa.params.glue == fb.params.glue
        && fa.log_blowup == fb.log_blowup
        && fa.sizes[FIRST_TREE] == fb.sizes[FIRST_TREE]
        && fa.sizes[FIRST_TREE + 1] == fb.sizes[FIRST_TREE + 1];
    let out_root = glue_present.then(|| roots_a[FIRST_TREE + 1].clone());
    let in_root = glue_present.then(|| roots_b[FIRST_TREE].clone());
    // ---- stream AA: the K/V link ------------------------------------------
    //
    // The integer comparisons come first and they are not a formality.  Root
    // equality compares byte strings, so two roots are only *comparable* once
    // the two chunks agree on everything that fixes what a K/V column holds and
    // how tall it is: the head geometry, the band decomposition, the two grid
    // targets that choose the tilings, the quantisation widths that fix the
    // committed offsets, and -- the one that says the pair is a pair --
    // `c_a + S == c_b`, the consumer's context being the producer's plus one
    // block.  Without the adjacency comparison two honest but non-consecutive
    // chunks would be compared root to root and would simply differ, which is
    // the right answer for the wrong reason; with it the check says *why*.
    let (kv_present, kv_shape_ok, kv) = match (&ka, &kb) {
        (None, None) => (false, true, vec![]),
        (Some(ka), Some(kb)) => {
            let p = (&fa.params, &fb.params);
            let ok = p.0.hd == p.1.hd
                && p.0.nh == p.1.nh
                && p.0.bands == p.1.bands
                && p.0.corner
                && p.1.corner
                && p.0.variant == p.1.variant
                && p.0.q_bits == p.1.q_bits
                && p.0.k_bits == p.1.k_bits
                && p.0.chan_target == p.1.chan_target
                && p.0.grid_target == p.1.grid_target
                && ka.groups == kb.groups
                && kb.with_in
                && p.0.c + p.0.s == p.1.c;
            let mut links = vec![];
            let mut sizes_ok = true;
            if ok {
                for g in 0..ka.groups {
                    let (to, ti) = (ka.out_tree(g), kb.in_tree(g).expect("with_in"));
                    sizes_ok &= fa.sizes[to] == fb.sizes[ti];
                    let (ro, ri) = (roots_a[to].clone(), roots_b[ti].clone());
                    links.push(KvLink { group: g, out_tree: to, in_tree: ti,
                                        equal: ro == ri, out_root: ro, in_root: ri });
                }
            }
            (true, ok && sizes_ok, links)
        }
        // one chunk bound its K/V and the other did not: there is nothing to
        // compare, and "nothing to compare" must not read as "it chained"
        _ => (false, false, vec![]),
    };
    let axis = axis_of(&fa.params, &fb.params);
    let mut v = ChainVerdict {
        axis,
        present: glue_present,
        shape_ok,
        roots_equal: glue_present && out_root == in_root,
        out_root,
        in_root,
        kv_present,
        kv_shape_ok,
        kv_roots_equal: !kv.is_empty() && kv.iter().all(|l| l.equal),
        kv,
        reason: None,
        error: None,
    };
    v.reason = ChainVerdict::why(axis, &v);
    v
}

// -------------------------------------------------------------------- driver ---

/// One chunk proved, serialized and verified.  The bytes are on *disk*, and the
/// pair check below reads them back from there: the point of the feature is that
/// two proofs chain from bytes alone, so the probe should not be allowed to
/// smuggle anything through memory.
struct Chunk {
    name: String,
    dir: String,
    fixed_path: String,
    proof_path: String,
    accepts: bool,
    json: Value,
}

impl Chunk {
    fn fixed(&self) -> Vec<u8> {
        std::fs::read(&self.fixed_path).expect("the statement was not written")
    }
    fn proof(&self) -> Vec<u8> {
        std::fs::read(&self.proof_path).expect("the proof was not written")
    }
}

/// `chain_check` on two chunks, reading all four documents off disk.
fn chain_files(a: &Chunk, b: &Chunk) -> ChainVerdict {
    chain_check(&a.fixed(), &a.proof(), &b.fixed(), &b.proof())
}

/// `args` with `--witness` pointed at `dir`.  `arg_str` takes the first match,
/// so prepending is how an override works.
fn at(args: &[String], dir: &str) -> Vec<String> {
    let mut v = vec!["--witness".to_string(), dir.to_string()];
    v.extend(args.iter().cloned());
    v
}

/// Prove one chunk with the boundary on, serialize it to `out_dir`, and verify it
/// from those bytes.  Everything the probes need is in the returned `Chunk`.
fn prove_one(tag: &str, name: &str, dir: &str, out_dir: &str, args: &[String],
             tamper: &crate::compose::ProverTamper) -> Chunk {
    let a = at(args, dir);
    let (l, vname) = crate::layer_of(&a);
    let mut params = crate::params_of(&l, &a, &vname);
    params.bounds = true;
    assert!(params.glue.is_some(),
            "{name}: the boundary is the wiring family's two stream ties, and this \
             witness has no glue columns -- extract it with `extract.py --glue`");
    let out = crate::compose::prove_layer(&l, &params, tamper);
    let ser = crate::compose_ser(&out);
    std::fs::create_dir_all(out_dir).expect("the output directory");
    let fixed_path = format!("{out_dir}/{tag}.fixed.json");
    let proof_path = format!("{out_dir}/{tag}.proof.json");
    std::fs::write(&fixed_path, &ser.fixed_bytes).expect("writing the statement");
    std::fs::write(&proof_path, ser.proof_bytes.clone().unwrap_or_default())
        .expect("writing the proof");
    let (json, accepts) = crate::compose_finish(out, ser, &params);
    Chunk { name: name.into(), dir: dir.into(), fixed_path, proof_path, accepts, json }
}

fn chunk_json(c: &Chunk) -> Value {
    json!({
        "chunk": c.name,
        "witness": c.dir,
        "statement_file": c.fixed_path,
        "proof_file": c.proof_path,
        "accepts": c.accepts,
        "cells": c.json.get("cells"),
        "columns": c.json.get("columns"),
        "bounds": c.json.get("bounds"),
        "bound_roots": c.json.get("bound_roots"),
        "proof_bytes": c.json.get("proof_bytes"),
        "proof_ser_bytes": c.json.get("proof_ser_bytes"),
        "statement_bytes": c.json.get("statement_bytes"),
        "prove_secs": c.json.get("prove_secs"),
        "trace_secs": c.json.get("trace_secs"),
        "verify_secs": c.json.get("verify_secs"),
        "glue_ties_ok": c.json.get("glue_ties_ok"),
        "glue_vs_b1": c.json.pointer("/family_cells/glue_vs_b1"),
        "verdict": c.json.get("verdict"),
    })
}

/// The probe pair: two adjacent chunks proved on a shared boundary, the chain
/// check on the honest pair, and then each side tampered on its own.
///
/// The asymmetry is the whole point of the feature, so the probes assert it
/// rather than report it: a chunk whose boundary was moved by changing one of
/// its *inputs* is internally consistent and must still prove; only the pair
/// check may catch it.  The one probe that must take its own chunk down is the
/// boundary column tampered *alone*, because that is the equality constraint's
/// job.
pub fn cmd(args: &[String]) {
    let a_dir = crate::arg_str(args, "--a").expect("bounds needs --a <witness dir>");
    let b_dir = crate::arg_str(args, "--b").expect("bounds needs --b <witness dir>");
    let c_dir = crate::arg_str(args, "--c");
    let a_out = crate::arg_str(args, "--a-out-tampered");
    let b_in = crate::arg_str(args, "--b-in-tampered");
    let tie_cell: usize = crate::arg_scalar(args, "--tie-cell", 4321usize);
    let out_dir = crate::arg_str(args, "--out-dir").unwrap_or_else(|| "bounds_out".into());
    let none = crate::compose::ProverTamper::default();

    // the honest pair
    let a = prove_one("a", "N (the first chunk)", &a_dir, &out_dir, args, &none);
    let b = prove_one("b", "N+1 (the second chunk)", &b_dir, &out_dir, args, &none);
    assert!(a.accepts, "the first chunk's own proof was rejected");
    assert!(b.accepts, "the second chunk's own proof was rejected");
    let honest = chain_files(&a, &b);
    let mut probes = vec![];

    // 1. the boundary column alone.  Nothing else in the chunk moves, so the
    //    equality constraint is the only thing that can reject -- which is the
    //    statement that the root is bound to the proof and not merely appended.
    let t = prove_one(
        "p1_tie",
        "N, one boundary cell",
        &a_dir,
        &out_dir,
        args,
        &crate::compose::ProverTamper { bounds: Some((1, tie_cell)), ..Default::default() },
    );
    probes.push(json!({
        "probe": "one cell of the output boundary tree, the region left alone",
        "tampered_cell": tie_cell,
        "expects": "its own chunk's proof rejects (the equality constraint)",
        "accepts": t.accepts,
        "prove_error": t.json.get("prove_error"),
        "verdict": t.json.get("verdict"),
        "passed": !t.accepts,
    }));
    assert!(!t.accepts, "a boundary column that disagrees with its region accepted");

    // 2. the wrong neighbour: two chunks that are both entirely honest and both
    //    accept, and are simply not consecutive.  This is the asymmetry in its
    //    purest form -- nothing is tampered anywhere, and the pair check is the
    //    only thing in the system that can say so.
    if let Some(d) = &c_dir {
        let p = prove_one("p2_far", "a non-neighbour", d, &out_dir, args, &none);
        let ch = chain_files(&a, &p);
        probes.push(json!({
            "probe": "an honest chunk that is not N's successor",
            "expects": "both chunks accept; the pair check fails",
            "witness": d,
            "accepts": p.accepts,
            "chain": serde_json::to_value(&ch).unwrap(),
            "chains": ch.chains(),
            "passed": p.accepts && !ch.chains(),
            "chunk": chunk_json(&p),
        }));
        assert!(p.accepts, "an honest non-neighbour's own proof was rejected");
        assert!(!ch.chains(), "a non-neighbour chained");
    }

    // 3 and 4. each side's boundary moved by one bit of the witness it is
    //    extracted from.  What has to happen here depends on how much of the
    //    layer statement is closed, and the probe reports which:
    //
    //    * while the residual stream is a free committed input, the chunk stays
    //      internally consistent and proves, and only the pair check catches the
    //      move -- the case the feature was built for;
    //    * once the norm tails and the dequant output ties are in (stream E),
    //      the stream is pinned forward into the matmul inputs, so a one-bit
    //      change to it is caught *inside* the chunk and the proof is rejected.
    //
    //    Either outcome is a pass; what would be a failure is a chunk that
    //    accepts and still chains.  So that is the whole assertion.
    for (tag, name, dir, second) in [
        ("p3_out", "chunk N's output boundary, moved by one bit of its own witness",
         a_out.as_ref(), false),
        ("p4_in", "chunk N+1's input boundary, moved by one bit of its own witness",
         b_in.as_ref(), true),
    ] {
        let Some(d) = dir else { continue };
        let p = prove_one(tag, name, d, &out_dir, args, &none);
        let ch = if second { chain_files(&a, &p) } else { chain_files(&p, &b) };
        probes.push(json!({
            "probe": name,
            "expects": "either the chunk's own proof rejects it, or it accepts and the \
                        pair check fails -- never accepts and chains",
            "witness": d,
            "accepts": p.accepts,
            "caught_inside_the_chunk": !p.accepts,
            "prove_error": p.json.get("prove_error"),
            "verdict": p.json.get("verdict"),
            "chain": serde_json::to_value(&ch).unwrap(),
            "chains": ch.chains(),
            "passed": !(p.accepts && ch.chains()),
            "chunk": chunk_json(&p),
        }));
        assert!(!(p.accepts && ch.chains()), "a moved boundary accepted and still chained");
    }

    assert!(honest.chains(), "the honest pair did not chain");
    crate::util::emit(
        "bounds",
        json!({
            "chunks": [chunk_json(&a), chunk_json(&b)],
            "chain_honest": serde_json::to_value(&honest).unwrap(),
            "chains": honest.chains(),
            "read_from": out_dir,
            "probes": probes,
        }),
    );
}

/// Stream I (open item 3 of this stream's record): the chain check run along a
/// whole sequence rather than on one pair.
///
/// `--dir` holds one `<tag>.fixed.json` and one `<tag>.proof.json` per chunk,
/// written by `compose --ser-dir`; `--tags` names them in sequence order. Every
/// consecutive pair is checked, and the command reports how many links were run
/// and how many held. Nothing here touches a witness or a prover: it is the
/// verifier-side question, over bytes that were on disk before it started.
///
/// The two ends of the chain are not links and are not checked here -- the first
/// chunk's input and the last chunk's output are declared inputs under the v1
/// ground rule, so a sequence of `n` chunks has `n - 1` links.
pub fn chain_cmd(args: &[String]) {
    let dir = crate::arg_str(args, "--dir").expect("bounds-chain needs --dir <serialized dir>");
    let tags = crate::arg_str(args, "--tags")
        .expect("bounds-chain needs --tags <t0,t1,...> in sequence order");
    let tags: Vec<String> = tags.split(',').map(|t| t.trim().to_string()).collect();
    assert!(tags.len() >= 2, "a chain wants at least two chunks");

    let read = |tag: &str, what: &str| -> Vec<u8> {
        let path = format!("{dir}/{tag}.{what}.json");
        std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    };

    // `--axis layer|token` is the caller saying which edge family it believes
    // it handed over.  It is not a mode: the axis is still derived from the two
    // statements, and this only turns "the pair is on the other axis" from a
    // thing the reader has to notice into a failed link.  A gate that names its
    // axis cannot silently pass by having been given the wrong tags.
    let want_axis = crate::arg_str(args, "--axis").map(|a| {
        Axis::parse(&a).unwrap_or_else(|| panic!("--axis takes `layer` or `token`, got `{a}`"))
    });

    let mut links = vec![];
    let mut passed = 0usize;
    let mut by_axis = [0usize; 3];
    for w in tags.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let v = chain_check(&read(a, "fixed"), &read(a, "proof"),
                            &read(b, "fixed"), &read(b, "proof"));
        by_axis[match v.axis { Axis::Layer => 0, Axis::Token => 1, Axis::None => 2 }] += 1;
        let axis_ok = want_axis.is_none_or(|w| w == v.axis);
        let ok = v.chains() && axis_ok;
        if ok {
            passed += 1;
        }
        eprintln!("chain {a} -> {b} [{}]: {}{}", v.axis.name(),
                  if ok { "ok" } else { "FAILED" },
                  match (&v.reason, axis_ok) {
                      (_, false) => format!(" -- not on the requested axis `{}`",
                                            want_axis.unwrap().name()),
                      (Some(r), _) => format!(" -- {r}"),
                      _ => String::new(),
                  });
        links.push(json!({
            "from": a,
            "to": b,
            "axis": v.axis.name(),
            "axis_ok": axis_ok,
            "chains": ok,
            "reason": v.reason,
            "verdict": serde_json::to_value(&v).unwrap(),
        }));
    }
    let n = links.len();
    crate::util::emit(
        "bounds-chain",
        json!({
            "dir": dir,
            "chunks": tags,
            "axis_requested": want_axis.map(|a| a.name()),
            "links_run": n,
            "links_passed": passed,
            "links_layer": by_axis[0],
            "links_token": by_axis[1],
            "links_no_edge": by_axis[2],
            "all_chain": passed == n,
            "links": links,
        }),
    );
}

// ------------------------------------ stream EE: the chain over the whole grid ---
//
// A sequence proof is a grid of chunks indexed by (token block, layer), chained
// along both axes: the residual stream inside every block (`(b, k) -> (b, k+1)`,
// stream H's link) and the K/V cache across blocks at every layer
// (`(b, k) -> (b+1, k)`, stream AA's link).  `chain_cmd` above walks one list of
// tags in the order it was given and answers for each consecutive pair; this
// driver takes the whole grid, works out which links the grid *implies*, runs
// every one of them, and fails unless every implied link ran and chained.
//
// Why "implied" is the load-bearing word.  A driver that reports `links_passed
// / links_run` over whatever tags it was handed cannot tell a full chain from
// a short one: `z_box.sh`'s `grep '^L'` dropped every `B<b>L<k>` tag and
// reported a chain of nothing, without an error (stream DD's record).  So the
// count is derived from the grid's own block and layer sets, a missing file is
// a failure and not a shorter chain, a gap in either axis is a failure unless
// the caller says gaps are expected, and a caller that knows the shape can pin
// the counts from outside.

/// A tag's place on the grid, from its name: `B<b>L<k>` is `(Some(b), k)` and a
/// bare `L<k>` is `(None, k)` -- the block is then the one its statement
/// declares.  Anything else is not on the grid.  Numbers are parsed, not
/// compared as strings, so `B10` sorts after `B9` and `L10` after `L9`.
pub fn grid_tag(tag: &str) -> Option<(Option<usize>, usize)> {
    let num = |s: &str| (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse::<usize>().ok())
        .flatten();
    let rest = tag.strip_prefix('B');
    match rest {
        Some(r) => {
            let (b, l) = r.split_once('L')?;
            Some((Some(num(b)?), num(l)?))
        }
        None => Some((None, num(tag.strip_prefix('L')?)?)),
    }
}

/// The tags a `compose-chunks` plan names, in file order: the `--ser-tag` of
/// every non-comment line, or `chunk<i>` where a line has none (which is what
/// the loop itself calls such a chunk).
pub fn plan_tags(path: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .enumerate()
        .map(|(i, l)| {
            let a: Vec<String> = l.split_whitespace().map(str::to_string).collect();
            crate::arg_str(&a, "--ser-tag").unwrap_or_else(|| format!("chunk{i}"))
        })
        .collect()
}

/// The commitment roots of a serialized proof and nothing else.  `chain_of`
/// reads only the roots, so the 50 MB of decommitments behind them need not
/// be built into a `StarkProof` to answer a pair; the JSON is still read and
/// walked in full, so what is compared is what is on disk.
#[derive(serde::Deserialize)]
struct RootsOnly {
    commitments: Vec<
        <<stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel
            as stwo::core::channel::MerkleChannel>::H
            as stwo::core::vcs::MerkleHasher>::Hash,
    >,
}

/// One chunk as the grid driver holds it: its place, its statement and its
/// roots.  The bytes were read off `--dir` and are not kept.
struct GridChunk {
    tag: String,
    block: usize,
    layer: usize,
    fixed: Fixed,
    roots: Vec<String>,
}

/// `bounds-grid`: every link of the (block, layer) grid, on both axes, from
/// serialized bytes alone.
///
/// ```text
///   --dir DIR                   <tag>.fixed.json / <tag>.proof.json per chunk
///   --plan FILE                 the grid is the plan's --ser-tag set
///   --tags t0,t1,...            or an explicit list
///   (neither)                   or every *.fixed.json under --dir
///   --allow-gaps                a missing block or layer only shrinks the implied count
///   --expect-layer-links N      pin the implied count from outside
///   --expect-token-links N
/// ```
///
/// Exit status 0 only when every implied link ran and chained on its own
/// axis, every tag sits where its statement says it sits, and the pinned
/// counts (if given) are met; 1 otherwise, after the JSON is written.
pub fn grid_cmd(args: &[String]) {
    let dir = crate::arg_str(args, "--dir").expect("bounds-grid needs --dir <serialized dir>");
    let allow_gaps = crate::flag(args, "--allow-gaps");
    let expect_layer: Option<usize> = crate::arg_str(args, "--expect-layer-links")
        .map(|s| s.parse().expect("--expect-layer-links takes a count"));
    let expect_token: Option<usize> = crate::arg_str(args, "--expect-token-links")
        .map(|s| s.parse().expect("--expect-token-links takes a count"));

    // ---- which tags make up the grid ------------------------------------
    let (source, tags): (Value, Vec<String>) = if let Some(p) = crate::arg_str(args, "--plan") {
        (json!({"plan": p}), plan_tags(&p))
    } else if let Some(t) = crate::arg_str(args, "--tags") {
        (json!({"tags": t}), t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
    } else {
        let mut v: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter_map(|n| n.strip_suffix(".fixed.json").map(str::to_string))
            .collect();
        v.sort();
        (json!({"scan": dir}), v)
    };
    let mut failures: Vec<String> = vec![];
    let mut off_grid: Vec<String> = vec![];
    let mut missing: Vec<String> = vec![];
    let mut chunks: Vec<GridChunk> = vec![];
    let mut chunk_rows: Vec<Value> = vec![];

    // ---- read every chunk once: statement, and the proof's roots ---------
    for tag in &tags {
        let Some((tagged_block, layer)) = grid_tag(tag) else {
            off_grid.push(tag.clone());
            continue;
        };
        let fpath = format!("{dir}/{tag}.fixed.json");
        let ppath = format!("{dir}/{tag}.proof.json");
        let (Ok(fb), Ok(pb)) = (std::fs::read(&fpath), std::fs::read(&ppath)) else {
            missing.push(tag.clone());
            continue;
        };
        let fixed: Fixed = match serde_json::from_slice(&fb) {
            Ok(f) => f,
            Err(e) => { failures.push(format!("{tag}: unparseable statement: {e}")); continue; }
        };
        let roots: Vec<String> = match serde_json::from_slice::<RootsOnly>(&pb) {
            Ok(p) => p.commitments.iter().map(|c| format!("{c:?}")).collect(),
            Err(e) => { failures.push(format!("{tag}: unparseable proof: {e}")); continue; }
        };
        let (c, s) = (fixed.params.c, fixed.params.s);
        // The block a statement declares is its prefix width in blocks.  A tag
        // that names another block is a chunk filed on the wrong cell: that is
        // a failure in its own right, and the chunk is still placed where the
        // tag put it, so that the links into and out of that cell are run and
        // answered by the pair check -- the tag is the plan's claim, and the
        // grid checks the claim against the statement *and* the neighbours.
        let declared = (s > 0 && c % s == 0).then(|| c / s);
        let Some(declared) = declared else {
            failures.push(format!("{tag}: the statement declares c = {c}, s = {s}, which is not \
                                   a whole number of blocks"));
            continue;
        };
        if let Some(b) = tagged_block.filter(|b| *b != declared) {
            failures.push(format!("{tag}: the tag names block {b} but the statement declares \
                                   block {declared} (c = {c} = {declared} x {s})"));
        }
        let block = tagged_block.unwrap_or(declared);
        chunk_rows.push(json!({
            "tag": tag,
            "block": block,
            "block_tagged": tagged_block,
            "block_declared": declared,
            "block_tag_ok": block == declared,
            "layer": layer,
            "c": c, "s": s,
            "trees": fixed.sizes.len(),
            "bounds": fixed.params.bounds,
            "kv_bind": fixed.params.kv_bind,
            "statement_bytes": fb.len(),
            "proof_bytes": pb.len(),
        }));
        chunks.push(GridChunk { tag: tag.clone(), block, layer, fixed, roots });
    }
    for m in &missing {
        failures.push(format!("{m}: named by the plan, but {dir}/{m}.{{fixed,proof}}.json is not there"));
    }

    // ---- the grid, and the links it implies ------------------------------
    use std::collections::BTreeMap;
    let mut grid: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
    for (i, ch) in chunks.iter().enumerate() {
        if let Some(j) = grid.entry(ch.block).or_default().insert(ch.layer, i) {
            failures.push(format!("{} and {} both sit on block {}, layer {}",
                                  chunks[j].tag, ch.tag, ch.block, ch.layer));
        }
    }
    let mut gaps: Vec<Value> = vec![];
    // layer links: inside every block, consecutive layers
    let mut layer_links: Vec<(usize, usize)> = vec![];
    for (b, layers) in &grid {
        let ks: Vec<usize> = layers.keys().copied().collect();
        for w in ks.windows(2) {
            if w[1] == w[0] + 1 {
                layer_links.push((layers[&w[0]], layers[&w[1]]));
            } else {
                gaps.push(json!({"axis": "layer", "block": b, "from": w[0], "to": w[1]}));
            }
        }
    }
    // token links: at every layer, consecutive blocks
    let mut by_layer: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
    for (b, layers) in &grid {
        for (k, i) in layers {
            by_layer.entry(*k).or_default().insert(*b, *i);
        }
    }
    let mut token_links: Vec<(usize, usize)> = vec![];
    for (k, blocks) in &by_layer {
        let bs: Vec<usize> = blocks.keys().copied().collect();
        for w in bs.windows(2) {
            if w[1] == w[0] + 1 {
                token_links.push((blocks[&w[0]], blocks[&w[1]]));
            } else {
                gaps.push(json!({"axis": "token", "layer": k, "from": w[0], "to": w[1]}));
            }
        }
    }
    if !gaps.is_empty() && !allow_gaps {
        failures.push(format!("the grid has {} gap(s) and --allow-gaps was not given: a chain \
                               with a hole is shorter than the one the plan implies", gaps.len()));
    }
    let (want_layer, want_token) = (layer_links.len(), token_links.len());
    if let Some(n) = expect_layer.filter(|n| *n != want_layer) {
        failures.push(format!("the grid implies {want_layer} layer link(s); --expect-layer-links {n}"));
    }
    if let Some(n) = expect_token.filter(|n| *n != want_token) {
        failures.push(format!("the grid implies {want_token} token link(s); --expect-token-links {n}"));
    }
    if want_layer + want_token == 0 {
        failures.push("the grid implies no link at all: nothing to chain".to_string());
    }

    // ---- run them ---------------------------------------------------------
    let mut links: Vec<Value> = vec![];
    let mut run = [0usize; 2];
    let mut passed = [0usize; 2];
    for (ax, want, pairs) in [(0, Axis::Layer, &layer_links), (1, Axis::Token, &token_links)] {
        for (i, j) in pairs {
            let (a, b) = (&chunks[*i], &chunks[*j]);
            let v = chain_of(&a.fixed, &a.roots, &b.fixed, &b.roots);
            let axis_ok = v.axis == want;
            let ok = v.chains() && axis_ok;
            run[ax] += 1;
            if ok {
                passed[ax] += 1;
            }
            eprintln!("grid {} -> {} [{}]: {}{}", a.tag, b.tag, want.name(),
                      if ok { "ok" } else { "FAILED" },
                      match (&v.reason, axis_ok) {
                          (_, false) => format!(" -- the pair is on axis `{}`, not `{}`",
                                                v.axis.name(), want.name()),
                          (Some(r), _) => format!(" -- {r}"),
                          _ => String::new(),
                      });
            links.push(json!({
                "axis": want.name(),
                "from": a.tag, "to": b.tag,
                "block": [a.block, b.block], "layer": [a.layer, b.layer],
                "axis_ok": axis_ok,
                "chains": ok,
                "reason": v.reason,
                "verdict": serde_json::to_value(&v).unwrap(),
            }));
        }
    }
    let all_layer = run[0] == want_layer && passed[0] == run[0];
    let all_token = run[1] == want_token && passed[1] == run[1];
    let all_chain = failures.is_empty() && all_layer && all_token;
    let blocks: Vec<usize> = grid.keys().copied().collect();
    let layers: BTreeMap<String, Vec<usize>> = grid
        .iter()
        .map(|(b, l)| (b.to_string(), l.keys().copied().collect()))
        .collect();
    eprintln!("grid: {} block(s), {} chunk(s) on the grid, {} off it; layer {}/{} of {} implied, \
               token {}/{} of {} implied{}",
              blocks.len(), chunks.len(), off_grid.len(),
              passed[0], run[0], want_layer, passed[1], run[1], want_token,
              if all_chain { "" } else { " -- FAILED" });
    for f in &failures {
        eprintln!("grid: FAIL {f}");
    }
    crate::util::emit(
        "bounds-grid",
        json!({
            "dir": dir,
            "source": source,
            "tags": tags,
            "blocks": blocks,
            "layers": layers,
            "chunks": chunk_rows,
            "off_grid": off_grid,
            "missing": missing,
            "gaps": gaps,
            "allow_gaps": allow_gaps,
            "layer": {"expected": want_layer, "run": run[0], "passed": passed[0],
                      "pinned": expect_layer, "all_chain": all_layer},
            "token": {"expected": want_token, "run": run[1], "passed": passed[1],
                      "pinned": expect_token, "all_chain": all_token},
            "links_expected": want_layer + want_token,
            "links_run": run[0] + run[1],
            "links_passed": passed[0] + passed[1],
            "failures": failures,
            "all_chain": all_chain,
            "links": links,
        }),
    );
    if !all_chain {
        std::process::exit(1);
    }
}

// ------------------------------------------------- stream AA: the K/V sides ---
//
// The layer boundary above binds the residual stream along the *layer* axis.
// Stream AA binds the attention cache along the *token* axis, with the same
// instrument and for the same reason: today a chunk's K and V are ordinary
// committed base columns fed from the witness, tied to nothing outside the
// chunk, so two chunks of one sequence may have attended to two different
// prefixes and no verifier can tell.
//
// The three differences from the glue sides are all simplifications:
//
//  * K and V are single integers, not bf16 `(sm, ef)` pairs, so there is no
//    `PACK` and no degree-1 packing question -- one cell per value.
//  * a side is not one region's values but a *head band group's*, spread over
//    the group's band components; the tree's column order is therefore the
//    component-registration order, `[band's K][band's V]` band by band.
//  * a chunk commits an **out** side over its whole key set and an **in** side
//    over its declared prefix, and the chain is `out(L, b) == in(L, b+1)` -- so
//    every prefix key is committed twice.
//
// That duplicate is not an economy that was missed.  K and V feed LogUp range
// checks (`rel_k` in `ChanEval`, `rel_v` in `KeyEval`) whose challenge is drawn
// after tree 1 and before tree 2, while a boundary tree is committed after tree
// 2.  A value first committed *after* its own lookup challenge is not bound by
// that lookup, so the boundary copy cannot replace the base copy -- moving
// instead of copying would take the added cells from 264.2 M to 130.0 M at
// 8B/C=16K and would unbind K and V from their range checks.  It is the price
// of `FIRST_TREE = 3`, and the rule "add no commitment tree below index 3" is
// what makes it unavoidable.
//
// Why one pair per head band group rather than one global pair: the bands run
// all prefix bands first and then all corner bands (CC's frozen order), so a
// producer's global out tree would read
// `[g0 sub..][g1 sub..][g2 sub..][g0 corner][g1 corner][g2 corner]` while the
// consumer's global in tree reads `[g0 sub..][g1 sub..][g2 sub..]` with each
// group's last sub-band inline.  Per-group pairs put each group's corner
// immediately after its own sub-bands on both sides, and the two column lists
// coincide without moving CC's band order.

/// One K/V boundary tree as a band component sees it: which tree, and how many
/// of that component's columns go into it.
#[derive(Clone, Copy, Debug)]
pub struct KvBound {
    pub tree: usize,
    /// columns this component writes -- `lay.pk` in `ChanEval`, `lay.pv` in
    /// `KeyEval`
    pub n: usize,
}

/// One band's contribution to the K/V sides, as the plan knows it before any
/// column exists.
#[derive(Clone, Copy, Debug)]
pub struct KvBand {
    /// which head band group the band belongs to
    pub group: usize,
    /// the group's own block (`c = S` over the chunk's own decoded tokens)
    /// rather than one of its declared-prefix sub-bands
    pub corner: bool,
    pub pk: usize,
    pub pv: usize,
    pub chan_rows: usize,
    pub key_rows: usize,
}

/// One side of a chunk's K/V boundary.
#[derive(Clone, Debug)]
pub struct KvSide {
    pub tree: usize,
    /// which head band group's keys this side carries
    pub group: usize,
    /// the **out** side carries the group's whole key set, the chunk's own
    /// block included; the **in** side carries its declared prefix alone
    pub out: bool,
    /// the bands that write it, ascending -- which is component-registration
    /// order, and therefore the tree's column order
    pub bands: Vec<usize>,
    pub columns: usize,
    pub cells: usize,
}

/// The K/V sides, read off the per-band summary the plan already has.
///
/// `first_tree` is where they start: after the three that were always there and
/// after stream H's glue sides, so that turning either feature on does not move
/// the other's tree indices.
///
/// The **out** side comes first in each group.  Block 0 of a sequence has no
/// declared prefix and so no in side at all, and it must not commit an empty
/// tree -- `verify_ex` requires every committed tree to be read by at least one
/// component -- so ordering out first means block 0 drops a *trailing* side and
/// every other side keeps its index.
pub fn kv_sides(first_tree: usize, bands: &[KvBand]) -> Vec<KvSide> {
    let ngroups = bands.iter().map(|b| b.group + 1).max().unwrap_or(0);
    let mut sides: Vec<KvSide> = vec![];
    for g in 0..ngroups {
        for out in [true, false] {
            let idx: Vec<usize> = bands
                .iter()
                .enumerate()
                .filter(|(_, b)| b.group == g && (out || !b.corner))
                .map(|(i, _)| i)
                .collect();
            if idx.is_empty() {
                continue;
            }
            let columns = idx.iter().map(|i| bands[*i].pk + bands[*i].pv).sum();
            let cells = idx
                .iter()
                .map(|i| bands[*i].pk * bands[*i].chan_rows + bands[*i].pv * bands[*i].key_rows)
                .sum();
            sides.push(KvSide {
                tree: first_tree + sides.len(),
                group: g,
                out,
                bands: idx,
                columns,
                cells,
            });
        }
    }
    sides
}

/// What band `b` writes into the K/V trees, in the order the AIR must read the
/// masks: one `KvBound` per side the band participates in, in `plan.kv`'s order.
///
/// `n` is the component's own column count -- `lay.pk` for `ChanEval`, `lay.pv`
/// for `KeyEval` -- and the AIR asserts it against the mask count it reads.
pub fn kv_bounds_of(sides: &[KvSide], band: usize, n: usize) -> Vec<KvBound> {
    sides
        .iter()
        .filter(|s| s.bands.contains(&band))
        .map(|s| KvBound { tree: s.tree, n })
        .collect()
}

/// One K/V side's committed columns.
///
/// `cols` is handed in already assembled by the caller, which **clones the very
/// `Eval`s tree 1 committed** rather than re-deriving them from the witness.
/// That is deliberate, and it is the whole of why byte-for-byte with the region
/// holds: a second copy of `attn_trace`'s index map (`c = chunk * pk + p` for K,
/// `ch = chunk * pv + p` for V) would agree with the first on every shape
/// anyone tests and could disagree on the one that matters.  Here there is one
/// map, and the boundary tree is a literal copy of columns tree 1 already
/// holds.
///
/// `tamper` bumps one cell by 1, by flat index over `(column, row)` in the
/// side's own committed order.  It exists for the probe that has to show the
/// equality constraint bites: a boundary column that disagrees with the region
/// must take its **own chunk's** proof down, which is the statement that the
/// root is bound to the proof and not merely appended to it.
pub fn kv_cols(mut cols: Vec<Eval>, tamper: Option<usize>) -> Vec<Eval> {
    if let Some(t) = tamper {
        assert!(!cols.is_empty(), "a K/V side with no columns cannot be tampered");
        let rows = cols[0].domain.size();
        let (j, m) = ((t / rows) % cols.len(), t % rows);
        let v = cols[j].values.at(m);
        cols[j].values.set(m, v + M31::from_u32_unchecked(1));
    }
    cols
}

/// What the K/V boundary cost this chunk, for the record.
pub fn kv_report(sides: &[KvSide]) -> Value {
    let per: Vec<Value> = sides
        .iter()
        .map(|s| {
            json!({
                "tree": s.tree,
                "group": s.group,
                "side": if s.out { "out" } else { "in" },
                "bands": s.bands,
                "columns": s.columns,
                "cells": s.cells,
            })
        })
        .collect();
    json!({
        "sides": per,
        "columns": sides.iter().map(|s| s.columns).sum::<usize>(),
        "cells": sides.iter().map(|s| s.cells).sum::<usize>(),
    })
}

// ------------------------------------- stream EE: the sides, labelled ---------

/// One boundary side's root as the verifier reports it: which tree, which axis
/// it binds, whether it is the `in` or the `out` side, and -- for a K/V side --
/// which head band group.  Stream H named the residual pair `bound_in` and
/// `bound_out` by tree index (3 and 4), which was right while a statement
/// carried exactly one boundary pair; a statement now carries the sides its
/// plan lists, and a reader asks for a side by label.
#[derive(Debug, Clone, Serialize)]
pub struct SideRoot {
    pub tree: usize,
    pub axis: Axis,
    /// `"in"` or `"out"`
    pub role: &'static str,
    /// the head band group, on a K/V side; `None` on the residual pair
    pub group: Option<usize>,
    pub root: String,
}

/// Every side's root, off the plan's two side lists and a root-by-tree lookup.
/// The order is the prover's commit order, so it is also `bound_roots`' order.
pub fn side_roots(glue: &[Side], kv: &[KvSide], root: impl Fn(usize) -> String) -> Vec<SideRoot> {
    let mut v: Vec<SideRoot> = glue
        .iter()
        .map(|s| SideRoot {
            tree: s.bound.tree,
            axis: Axis::Layer,
            role: match s.bound.kind {
                TieKind::In(_) => "in",
                TieKind::Out(_) => "out",
            },
            group: None,
            root: root(s.bound.tree),
        })
        .collect();
    v.extend(kv.iter().map(|s| SideRoot {
        tree: s.tree,
        axis: Axis::Token,
        role: if s.out { "out" } else { "in" },
        group: Some(s.group),
        root: root(s.tree),
    }));
    v
}

// ------------------------------------------------------------------- tests ---
//
// The pair check reads exactly two things: the two `Fixed` statements, and the
// two proofs' root lists.  `chain_of` takes those directly, so the axis rule
// can be exercised on the coordinating box -- which has no GPU, no witnesses
// and not enough RAM to prove an 8B chunk -- rather than only on a card.
// These tests are about *which link is asked for*, not about whether a root is
// right; the roots here are strings chosen by hand.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::{Fixed, GlueParams, Params};

    /// A statement for one chunk: `c` keys of declared prefix, `s` decoded
    /// tokens, stream H's boundary always on, `--kv-bind` optional.
    fn fixed(c: usize, s: usize, kv: bool) -> Fixed {
        let params = Params {
            s,
            c,
            hd: 32,
            nh: 3,
            bands: vec![2, 1],
            variant: "intexp".into(),
            corner: kv,
            bounds: true,
            kv_bind: kv,
            glue: Some(GlueParams { d: 64, a: 4, ..Default::default() }),
            ..Default::default()
        };
        let n = 3 + 2 + kv_layout(&params).map(|k| k.n_sides()).unwrap_or(0);
        Fixed {
            params,
            log_blowup: 1,
            n_queries: 50,
            pow_bits: 0,
            sizes: vec![vec![4]; n],
            pre_root: "pre".into(),
        }
    }

    /// Root lists for a pair, with the boundaries the caller asks for made
    /// equal and every other root distinct.
    ///
    /// Tree layout for `--kv-bind` with two head band groups and `c > 0`:
    /// `[0 pre][1 base][2 inter][3 glue in][4 glue out][5 g0 out][6 g0 in]
    ///  [7 g1 out][8 g1 in]`.
    fn roots(fa: &Fixed, fb: &Fixed, kv_equal: bool, residual_equal: bool)
        -> (Vec<String>, Vec<String>) {
        let (na, nb) = (fa.sizes.len(), fb.sizes.len());
        let a: Vec<String> = (0..na).map(|i| format!("a{i}")).collect();
        let mut b: Vec<String> = (0..nb).map(|i| format!("b{i}")).collect();
        if residual_equal {
            b[FIRST_TREE] = a[FIRST_TREE + 1].clone();
        }
        if let (Some(ka), Some(kb)) = (kv_layout(&fa.params), kv_layout(&fb.params)) {
            if kv_equal {
                for g in 0..ka.groups.min(kb.groups) {
                    if let Some(ti) = kb.in_tree(g) {
                        b[ti] = a[ka.out_tree(g)].clone();
                    }
                }
            }
        }
        (a, b)
    }

    // ---- stream EE: the tag grammar ----------------------------------------

    #[test]
    fn grid_tags_parse_numerically() {
        assert_eq!(grid_tag("B0L8"), Some((Some(0), 8)));
        assert_eq!(grid_tag("B10L9"), Some((Some(10), 9)));
        assert_eq!(grid_tag("L31"), Some((None, 31)));
        assert_eq!(grid_tag("L0"), Some((None, 0)));
        for bad in ["head", "B8head", "chunk7", "B", "L", "BL1", "B1L", "b1l1", "B1L1x", ""] {
            assert_eq!(grid_tag(bad), None, "{bad:?} is not on the grid");
        }
        // sorted as numbers, so B10 follows B9 and L10 follows L9
        let mut tags: Vec<(Option<usize>, usize)> =
            ["B9L2", "B10L1", "B10L10", "B10L9", "B2L10", "B2L9"].iter().map(|t| grid_tag(t).unwrap()).collect();
        tags.sort();
        assert_eq!(tags, [(Some(2), 9), (Some(2), 10), (Some(9), 2), (Some(10), 1), (Some(10), 9), (Some(10), 10)]);
    }

    #[test]
    fn side_roots_are_labelled_from_the_plan() {
        // the root of tree `t` is the string `t`
        let root = |t: usize| t.to_string();
        let glue = [
            Side { bound: Bound { tree: 3, kind: TieKind::In(0), n: 1 }, region: 0, log_rows: 1 },
            Side { bound: Bound { tree: 4, kind: TieKind::Out(1), n: 1 }, region: 0, log_rows: 1 },
        ];
        let kv = [
            KvSide { tree: 5, group: 0, out: true, bands: vec![], columns: 0, cells: 0 },
            KvSide { tree: 6, group: 0, out: false, bands: vec![], columns: 0, cells: 0 },
            KvSide { tree: 7, group: 1, out: true, bands: vec![], columns: 0, cells: 0 },
        ];
        let v = side_roots(&glue, &kv, root);
        let lab: Vec<(usize, Axis, &str, Option<usize>, &str)> =
            v.iter().map(|s| (s.tree, s.axis, s.role, s.group, s.root.as_str())).collect();
        assert_eq!(lab, [
            (3, Axis::Layer, "in", None, "3"),
            (4, Axis::Layer, "out", None, "4"),
            (5, Axis::Token, "out", Some(0), "5"),
            (6, Axis::Token, "in", Some(0), "6"),
            (7, Axis::Token, "out", Some(1), "7"),
        ]);
        // no residual pair, only K/V sides: nothing is labelled by position
        let v = side_roots(&[], &kv, root);
        assert!(v.iter().all(|s| s.axis == Axis::Token));
        assert_eq!(v[0].tree, 5);
    }

    // ---- the axis is derived, and the two edge families are disjoint --------

    #[test]
    fn axis_layer_is_the_same_block() {
        let (a, b) = (fixed(1024, 512, true), fixed(1024, 512, true));
        assert_eq!(axis_of(&a.params, &b.params), Axis::Layer);
    }

    #[test]
    fn axis_token_is_the_next_block() {
        let (a, b) = (fixed(512, 512, true), fixed(1024, 512, true));
        assert_eq!(axis_of(&a.params, &b.params), Axis::Token);
    }

    #[test]
    fn axis_none_at_distance_two() {
        let (a, b) = (fixed(512, 512, true), fixed(1536, 512, true));
        assert_eq!(axis_of(&a.params, &b.params), Axis::None);
    }

    #[test]
    fn no_pair_is_on_both_axes() {
        // the whole reason the merged conjunction could not be satisfied
        for c in [0usize, 512, 1024, 4096] {
            for s in [1usize, 32, 512] {
                let (x, y) = (fixed(c, s, true), fixed(c + s, s, true));
                assert_ne!(axis_of(&x.params, &y.params), Axis::Layer);
                let (x, y) = (fixed(c, s, true), fixed(c, s, true));
                assert_ne!(axis_of(&x.params, &y.params), Axis::Token);
            }
        }
    }

    // ---- the K/V axis ------------------------------------------------------

    #[test]
    fn token_edge_chains_on_the_kv_link_alone() {
        let (a, b) = (fixed(512, 512, true), fixed(1024, 512, true));
        let (ra, rb) = roots(&a, &b, true, false);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Token);
        assert!(v.kv_roots_equal, "the K/V link was set up equal");
        assert!(!v.roots_equal, "two token blocks share no residual edge");
        assert!(v.chains(), "reason: {:?}", v.reason);
    }

    #[test]
    fn token_edge_does_not_chain_when_one_group_moves() {
        let (a, b) = (fixed(512, 512, true), fixed(1024, 512, true));
        let (ra, mut rb) = roots(&a, &b, true, false);
        // group 0's in tree: the shape of the H200 probe on one bit of `vvc.i8`
        let ti = kv_layout(&b.params).unwrap().in_tree(0).unwrap();
        rb[ti] = "moved".into();
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Token);
        assert!(!v.kv_roots_equal);
        assert!(!v.chains());
        assert!(!v.kv[0].equal && v.kv[1].equal);
    }

    #[test]
    fn token_edge_needs_the_flag_on_both_sides() {
        // the consumer proved without `--kv-bind`: nothing to compare, and
        // "nothing to compare" must not read as "it chained"
        let (a, b) = (fixed(512, 512, true), fixed(1024, 512, false));
        let (ra, rb) = roots(&a, &b, true, true);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Token);
        assert!(!v.kv_present);
        assert!(!v.chains(), "a token edge without the K/V boundary must not chain");
    }

    #[test]
    fn non_neighbour_does_not_chain_however_equal_its_roots() {
        let (a, b) = (fixed(512, 512, true), fixed(1536, 512, true));
        let (ra, rb) = roots(&a, &b, true, true);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::None);
        assert!(!v.chains());
    }

    // ---- the residual axis, unchanged from stream H ------------------------

    #[test]
    fn layer_edge_chains_on_the_residual_link_alone() {
        let (a, b) = (fixed(1024, 512, false), fixed(1024, 512, false));
        let (ra, rb) = roots(&a, &b, false, true);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Layer);
        assert!(v.chains(), "reason: {:?}", v.reason);
    }

    #[test]
    fn layer_edge_rejects_a_moved_residual_root() {
        let (a, b) = (fixed(1024, 512, false), fixed(1024, 512, false));
        let (ra, rb) = roots(&a, &b, false, false);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Layer);
        assert!(!v.roots_equal);
        assert!(!v.chains());
    }

    #[test]
    fn layer_edge_ignores_the_kv_trees() {
        // with `--kv-bind` on, two layers of one block have unrelated K/V
        // caches; the residual link is the whole question and the pair chains
        let (a, b) = (fixed(1024, 512, true), fixed(1024, 512, true));
        let (ra, rb) = roots(&a, &b, false, true);
        let v = chain_of(&a, &ra, &b, &rb);
        assert_eq!(v.axis, Axis::Layer);
        assert!(!v.kv_roots_equal, "the K/V roots differ, and that is expected here");
        assert!(v.chains(), "reason: {:?}", v.reason);
    }

    // ---- what the merged predicate did, kept as a regression ---------------

    /// Stream AA's merged predicate.
    fn merged(v: &ChainVerdict) -> bool {
        v.present && v.shape_ok && v.roots_equal && v.kv_shape_ok
            && (!v.kv_present || v.kv_roots_equal)
    }

    #[test]
    fn the_merged_predicate_failed_structurally_on_a_layer_edge() {
        // `kv_shape_ok` carries the adjacency `c_a + S == c_b`, which a layer
        // edge cannot satisfy: with `--kv-bind` on, *no* assignment of roots
        // makes the old predicate true.
        for kv_equal in [false, true] {
            for residual_equal in [false, true] {
                let (a, b) = (fixed(1024, 512, true), fixed(1024, 512, true));
                let (ra, rb) = roots(&a, &b, kv_equal, residual_equal);
                let v = chain_of(&a, &ra, &b, &rb);
                assert!(!v.kv_shape_ok);
                assert!(!merged(&v), "kv_equal={kv_equal} residual_equal={residual_equal}");
            }
        }
    }

    #[test]
    fn the_merged_predicate_failed_on_the_data_on_a_token_edge() {
        // `roots_equal` compares two different token blocks' residual
        // boundaries.  Nothing relates them, so the honest root table has them
        // unequal -- which is what the H200 measured.
        let (a, b) = (fixed(512, 512, true), fixed(1024, 512, true));
        let (ra, rb) = roots(&a, &b, true, false);
        let v = chain_of(&a, &ra, &b, &rb);
        assert!(v.kv_roots_equal && !v.roots_equal);
        assert!(!merged(&v));
        assert!(v.chains(), "the new predicate asks only for the K/V link here");
    }
}
