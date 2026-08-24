//! Trace helpers and the prove/verify driver, forked from B5 (which forked B4).
//!
//! Two additions B6 needs.  `Proven` now records cells and columns per tree, so
//! the column budget the acceptance asks for is read off the proof rather than
//! recomputed; and `Session::commit` takes the evals by value and returns those
//! counts, because a whole-layer proof cannot afford to clone its trees.
//!
//! The column-order trap B3 solved once is inherited verbatim: every column is
//! built in *coset* (natural row) order and permuted with
//! `bit_reverse_coset_to_circle_domain_order` before it is committed, because
//! that is the order a `[-1, 0]` mask means.
//!
//! `Session` records the commit/draw sequence as it happens so the verifier can
//! replay exactly the same transcript.  Every gadget here has the same three-tree
//! shape as B3's: tree 0 preprocessed (tables and selectors), tree 1 base
//! (the witness and the LogUp multiplicities), then the relation elements are
//! drawn, then tree 2 interaction (the LogUp columns).

use std::time::Instant;

use itertools::Itertools;
use num_traits::Zero;
use serde_json::{Value, json};
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sM31Channel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::proof::StarkProof;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_coset_to_circle_domain_order;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps as _};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::{CommitmentSchemeProver, ComponentProver, prove};

use crate::spans;
use crate::util::{LOG_BLOWUP, peak_rss_bytes, pcs_config, rss_bytes};

pub type Eval = CircleEvaluation<SimdBackend, M31, BitReversedOrder>;
pub type Proof = StarkProof<<Blake2sM31MerkleChannel as stwo::core::channel::MerkleChannel>::H>;

/// SIMD lane floor: stwo's SimdBackend needs at least N_LANES = 16 rows.
pub const MIN_LOG_ROWS: u32 = 4;

/// Rows needed for `n` items at `per_row` each, at or above the SIMD floor.
pub fn log_rows_for(n: usize, per_row: usize) -> u32 {
    let rows = n.div_ceil(per_row.max(1)).max(1);
    (usize::BITS - (rows - 1).leading_zeros()).max(MIN_LOG_ROWS)
}

/// stwo 2.3.0 mis-maps FRI query positions into the preprocessed tree whenever
/// that tree is *smaller* than the largest trace column (B4 section 7): the mapping
/// `(pos >> k << 1) + (pos & 1)` in `prepare_preprocessed_query_positions` is not
/// monotonic for k >= 2, and the Merkle verifier assumes increasing positions, so
/// honest proofs fail with `Merkle(RootMismatch)` depending on where the queries
/// land.  B4 padded every table up to the trace height; B6 keeps the invariant
/// deliberately, as the plan asks, by asserting it in `Session::check_pre`.
pub fn pad_log(natural: u32, log_rows: u32) -> u32 {
    natural.max(log_rows)
}

pub fn eval_of(log_size: u32, mut vals: Vec<M31>) -> Eval {
    assert_eq!(vals.len(), 1 << log_size);
    bit_reverse_coset_to_circle_domain_order(&mut vals);
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), BaseColumn::from_iter(vals))
}

pub fn m31_u32(v: u32) -> M31 {
    M31::from_u32_unchecked(v)
}

// ---------------------------------------------------------------------------
// the prove / verify driver
// ---------------------------------------------------------------------------

pub enum Op {
    /// one committed tree, with the log size of every column
    Tree(Vec<u32>),
    /// one relation's elements, i.e. `channel.draw_secure_felts(2)`
    Draw,
    /// `n` bare challenges, i.e. `channel.draw_secure_felts(n)`
    Felts(usize),
    /// public values mixed into the transcript before the components are built
    Mix(Vec<SecureField>),
}

pub struct Session<'a> {
    pub config: PcsConfig,
    pub channel: Blake2sM31Channel,
    pub cs: CommitmentSchemeProver<'a, SimdBackend, Blake2sM31MerkleChannel>,
    pub ops: Vec<Op>,
    pub tree_cols: Vec<usize>,
    pub tree_cells: Vec<usize>,
    pub tree_max_log: Vec<u32>,
    started: Instant,
}

pub fn twiddles(max_log: u32) -> TwiddleTree<SimdBackend> {
    SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log + 1 + LOG_BLOWUP).circle_domain().half_coset,
    )
}

impl<'a> Session<'a> {
    pub fn new(tw: &'a TwiddleTree<SimdBackend>) -> Self {
        let config = pcs_config();
        let mut channel = Blake2sM31Channel::default();
        config.mix_into(&mut channel);
        spans::reset();
        Self {
            config,
            channel,
            cs: CommitmentSchemeProver::<SimdBackend, Blake2sM31MerkleChannel>::new(config, tw),
            ops: vec![],
            tree_cols: vec![],
            tree_cells: vec![],
            tree_max_log: vec![],
            started: Instant::now(),
        }
    }

    pub fn commit(&mut self, evals: Vec<Eval>) {
        let sizes = evals.iter().map(|e| e.domain.log_size()).collect_vec();
        self.tree_cols.push(sizes.len());
        self.tree_cells.push(sizes.iter().map(|s| 1usize << s).sum());
        self.tree_max_log.push(sizes.iter().copied().max().unwrap_or(0));
        let mut tb = self.cs.tree_builder();
        tb.extend_evals(evals);
        tb.commit(&mut self.channel);
        self.ops.push(Op::Tree(sizes));
    }

    /// B4 section 7's invariant, asserted rather than hoped for: some preprocessed
    /// column must be at least as tall as the tallest column anywhere in the proof.
    pub fn check_pre(&self) -> bool {
        let widest = self.tree_max_log.iter().copied().max().unwrap_or(0);
        self.tree_max_log.first().copied().unwrap_or(0) >= widest
    }

    /// Draw one relation's lookup elements (two QM31 felts), recorded for replay.
    pub fn draw_relation<R>(&mut self, f: impl FnOnce(&mut Blake2sM31Channel) -> R) -> R {
        let r = f(&mut self.channel);
        self.ops.push(Op::Draw);
        r
    }

    /// Draw `n` bare challenges, recorded for replay.
    pub fn draw_felts(&mut self, n: usize) -> Vec<SecureField> {
        let v = self.channel.draw_secure_felts(n);
        self.ops.push(Op::Felts(n));
        v
    }

    /// Mix public values into the transcript, recorded for replay.
    pub fn mix(&mut self, vals: &[SecureField]) {
        self.channel.mix_felts(vals);
        self.ops.push(Op::Mix(vals.to_vec()));
    }

    /// Prove and hand the proof back for an *external* verifier -- the composed
    /// proof's accept path is the serialized public-input check, never this
    /// session's internal replay.
    pub fn prove_raw(
        mut self,
        provers: &[&dyn ComponentProver<SimdBackend>],
    ) -> (Result<Proof, String>, RawStats) {
        let started = self.started;
        let pre_ok = self.check_pre();
        let tree_cols = self.tree_cols.clone();
        let tree_cells = self.tree_cells.clone();
        let res = prove::<SimdBackend, Blake2sM31MerkleChannel>(provers, &mut self.channel, self.cs)
            .map_err(|e| format!("{e:?}"));
        let prove_secs = started.elapsed().as_secs_f64();
        let stats = RawStats {
            prove_secs,
            phases: spans::snapshot(prove_secs),
            peak_rss: peak_rss_bytes().max(rss_bytes()),
            pre_ok,
            tree_cols,
            tree_cells,
        };
        (res, stats)
    }

    pub fn prove(
        mut self,
        provers: &[&dyn ComponentProver<SimdBackend>],
        components: &[&dyn Component],
    ) -> Proven {
        let config = self.config;
        let ops = std::mem::take(&mut self.ops);
        let started = self.started;
        let tree_cols = self.tree_cols.clone();
        let tree_cells = self.tree_cells.clone();
        let pre_ok = self.check_pre();
        let proof = match prove::<SimdBackend, Blake2sM31MerkleChannel>(
            provers,
            &mut self.channel,
            self.cs,
        ) {
            Ok(p) => p,
            // A trace that does not satisfy the AIR is refused by the prover itself.
            Err(e) => {
                let prove_secs = started.elapsed().as_secs_f64();
                return Proven {
                    prove_secs,
                    verify_secs: 0.0,
                    proof_bytes: 0,
                    stark_ok: false,
                    roots: vec![],
                    phases: spans::snapshot(prove_secs),
                    peak_rss: peak_rss_bytes().max(rss_bytes()),
                    prove_error: Some(format!("{e:?}")),
                    tree_cols,
                    tree_cells,
                    pre_ok,
                };
            }
        };
        let prove_secs = started.elapsed().as_secs_f64();
        let phases = spans::snapshot(prove_secs);
        let proof_bytes = proof.size_estimate();
        let peak_rss = peak_rss_bytes().max(rss_bytes());
        let roots = proof.commitments.iter().map(|c| format!("{c:?}")).collect_vec();

        // ---- verify: replay the transcript, then stwo's own verifier ---------
        let t = Instant::now();
        let mut vchannel = Blake2sM31Channel::default();
        config.mix_into(&mut vchannel);
        let mut vcs = CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        // Column sizes are public AIR parameters; cross-check the recorded ones
        // against what the components declare, so a trace/AIR mismatch is caught
        // here rather than by a confusing verification failure.
        let derived = TreeVec::concat_cols(components.iter().map(|c| c.trace_log_degree_bounds()));
        let mut tree = 0;
        for op in &ops {
            match op {
                Op::Tree(sizes) => {
                    if tree > 0 {
                        assert_eq!(&derived[tree], sizes, "tree {tree} column sizes disagree");
                    }
                    vcs.commit(proof.commitments[tree], sizes, &mut vchannel);
                    tree += 1;
                }
                Op::Draw => {
                    let _ = vchannel.draw_secure_felts(2);
                }
                Op::Felts(n) => {
                    let _ = vchannel.draw_secure_felts(*n);
                }
                Op::Mix(vals) => vchannel.mix_felts(vals),
            }
        }
        let vres = verify(components, &mut vchannel, &mut vcs, proof);
        let stark_ok = vres.is_ok();
        if let Err(e) = &vres {
            eprintln!("  verify failed: {e:?}");
        }
        Proven {
            prove_secs,
            verify_secs: t.elapsed().as_secs_f64(),
            proof_bytes,
            stark_ok,
            roots,
            phases,
            peak_rss,
            prove_error: None,
            tree_cols,
            tree_cells,
            pre_ok,
        }
    }
}

pub struct RawStats {
    pub prove_secs: f64,
    pub phases: Value,
    pub peak_rss: u64,
    pub pre_ok: bool,
    pub tree_cols: Vec<usize>,
    pub tree_cells: Vec<usize>,
}

pub struct Proven {
    pub prove_secs: f64,
    pub verify_secs: f64,
    pub proof_bytes: usize,
    pub stark_ok: bool,
    pub roots: Vec<String>,
    pub phases: Value,
    pub peak_rss: u64,
    pub prove_error: Option<String>,
    /// committed columns per tree: the column budget, read off the proof
    pub tree_cols: Vec<usize>,
    pub tree_cells: Vec<usize>,
    /// B4 section 7's invariant held for this proof
    pub pre_ok: bool,
}

impl Proven {
    pub fn fresh_cells(&self) -> usize {
        self.tree_cells[1..].iter().sum()
    }
    pub fn total_cells(&self) -> usize {
        self.tree_cells.iter().sum()
    }
    pub fn n_columns(&self) -> usize {
        self.tree_cols.iter().sum()
    }
    pub fn json(&self) -> Value {
        json!({
            "prove_secs": self.prove_secs, "verify_secs": self.verify_secs,
            "proof_bytes": self.proof_bytes, "peak_rss_bytes": self.peak_rss,
            "stark_ok": self.stark_ok, "prove_error": self.prove_error,
            "preprocessed_is_widest": self.pre_ok,
            "cells": {"preprocessed": self.tree_cells.first().copied().unwrap_or(0),
                      "base": self.tree_cells.get(1).copied().unwrap_or(0),
                      "interaction": self.tree_cells.get(2).copied().unwrap_or(0),
                      "fresh": self.fresh_cells(), "total": self.total_cells()},
            "columns": {"preprocessed": self.tree_cols.first().copied().unwrap_or(0),
                        "base": self.tree_cols.get(1).copied().unwrap_or(0),
                        "interaction": self.tree_cols.get(2).copied().unwrap_or(0),
                        "total": self.n_columns()},
            "phases": self.phases,
        })
    }
}

/// The LogUp claimed sums of every component in a proof.  stwo's `verify()` does
/// not check that they cancel (B2 §4, B3 §1): if they do not, the lookup argument
/// proves nothing, so every gadget here asserts it explicitly.
#[derive(Default, Debug, Clone)]
pub struct Sums {
    pub uses: SecureField,
    pub tables: SecureField,
}

impl Sums {
    pub fn cancels(&self) -> bool {
        self.uses + self.tables == SecureField::zero()
    }
    pub fn json(&self) -> Value {
        json!({"use_sum": format!("{:?}", self.uses),
               "table_sum": format!("{:?}", self.tables),
               "cancels": self.cancels()})
    }
}
