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
    pub error: Option<String>,
}

impl ChainVerdict {
    pub fn chains(&self) -> bool {
        self.present && self.shape_ok && self.roots_equal
    }
    fn rejected(msg: &str) -> Self {
        Self {
            present: false,
            shape_ok: false,
            roots_equal: false,
            out_root: None,
            in_root: None,
            error: Some(msg.to_string()),
        }
    }
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
    let want = FIRST_TREE + 2;
    if na != want || nb != want {
        return ChainVerdict::rejected(
            "a statement carries no boundary commitment (the flag was off)",
        );
    }
    if pa.commitments.len() <= FIRST_TREE + 1 || pb.commitments.len() <= FIRST_TREE + 1 {
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
    ChainVerdict {
        present: true,
        shape_ok,
        roots_equal: out_root == in_root,
        out_root: Some(out_root),
        in_root: Some(in_root),
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
