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
/// digest stand as the identity line.
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

/// What the pair check found. `chains` is the whole answer; the rest is so a
/// failure says which half of it failed.
#[derive(Debug, Clone, Serialize)]
pub struct ChainVerdict {
    /// both statements carry a boundary, i.e. the feature was on for both
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
    /// Stream AA: the K/V conjunct is added rather than substituted.  A pair
    /// proved with `--kv-bind` chains only if *both* boundaries hold; a pair
    /// proved without it chains exactly as stream H's did, which is what keeps
    /// stream I's recorded 62/62 meaningful.
    pub fn chains(&self) -> bool {
        self.present
            && self.shape_ok
            && self.roots_equal
            && self.kv_shape_ok
            && (!self.kv_present || self.kv_roots_equal)
    }
    fn rejected(msg: &str) -> Self {
        Self {
            present: false,
            shape_ok: false,
            roots_equal: false,
            out_root: None,
            in_root: None,
            kv_present: false,
            kv_shape_ok: false,
            kv_roots_equal: false,
            kv: vec![],
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
    let (na, nb) = (tree_count(&fa), tree_count(&fb));
    let (ka, kb) = (kv_layout(&fa.params), kv_layout(&fb.params));
    // Stream AA: the expected tree count is now a function of the statement --
    // three, plus stream H's two, plus one per K/V side.  With `--kv-bind`
    // absent this is `FIRST_TREE + 2` exactly as it was, so a pair of pre-AA
    // proofs takes the pre-AA path.
    let want = |n: usize, k: &Option<KvLayout>| {
        FIRST_TREE + 2 + k.as_ref().map(|k| k.n_sides()).unwrap_or(0) == n
    };
    if !want(na, &ka) || !want(nb, &kb) {
        return ChainVerdict::rejected(
            "a statement's tree count is not the one its parameters imply \
             (the boundary flag was off, or the statement is malformed)",
        );
    }
    if pa.commitments.len() < na || pb.commitments.len() < nb {
        return ChainVerdict::rejected("a proof carries fewer roots than its statement claims");
    }
    // everything the boundary layout depends on: the token count, the two
    // wiring shape numbers, the blowup, and the two trees' own column sizes
    let shape_ok = fa.params.s == fb.params.s
        && fa.params.glue == fb.params.glue
        && fa.log_blowup == fb.log_blowup
        && fa.sizes[FIRST_TREE] == fb.sizes[FIRST_TREE]
        && fa.sizes[FIRST_TREE + 1] == fb.sizes[FIRST_TREE + 1];
    let out_root = format!("{:?}", pa.commitments[FIRST_TREE + 1]);
    let in_root = format!("{:?}", pb.commitments[FIRST_TREE]);
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
                    let (ro, ri) = (format!("{:?}", pa.commitments[to]),
                                    format!("{:?}", pb.commitments[ti]));
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
    ChainVerdict {
        present: true,
        shape_ok,
        roots_equal: out_root == in_root,
        out_root: Some(out_root),
        in_root: Some(in_root),
        kv_present,
        kv_shape_ok,
        kv_roots_equal: !kv.is_empty() && kv.iter().all(|l| l.equal),
        kv,
        error: None,
    }
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

    let mut links = vec![];
    let mut passed = 0usize;
    for w in tags.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let v = chain_check(&read(a, "fixed"), &read(a, "proof"),
                            &read(b, "fixed"), &read(b, "proof"));
        let ok = v.chains();
        if ok {
            passed += 1;
        }
        eprintln!("chain {a} -> {b}: {}", if ok { "ok" } else { "FAILED" });
        links.push(json!({
            "from": a,
            "to": b,
            "chains": ok,
            "verdict": serde_json::to_value(&v).unwrap(),
        }));
    }
    let n = links.len();
    crate::util::emit(
        "bounds-chain",
        json!({
            "dir": dir,
            "chunks": tags,
            "links_run": n,
            "links_passed": passed,
            "all_chain": passed == n,
            "links": links,
        }),
    );
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
/// That is deliberate, and it is the whole of why byte-identity with the region
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
