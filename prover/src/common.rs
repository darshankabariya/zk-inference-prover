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
use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::core::verifier::verify;
use stwo::prover::backend::simd::SimdBackend;
use rayon::prelude::*;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES, PackedM31};
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

/// The permutation `bit_reverse_coset_to_circle_domain_order` applies, in
/// closed form rather than as a swap loop.
///
/// stwo's version is `for i { j = f(i); if j > i { swap(i, j) } }`, which is
/// only a correct application of `f` because `f` is an involution -- so out of
/// place it is exactly `dst[i] = src[f(i)]`, with every element independent.
#[inline]
fn coset_perm(i: usize, log_n: u32) -> usize {
    let n = 1usize << log_n;
    let c = if i % 2 == 0 { i / 2 } else { (2 * n - i - 1) / 2 };
    c.reverse_bits() >> (usize::BITS - log_n)
}

/// Assert the closed form against stwo's own function, once. Applying the swap
/// loop to the identity vector yields `f` itself, so this is a direct
/// comparison and not a spot check.
fn coset_perm_selfcheck() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        for log_n in 1..=14u32 {
            let n = 1usize << log_n;
            let mut v: Vec<u32> = (0..n as u32).collect();
            bit_reverse_coset_to_circle_domain_order(&mut v);
            for (i, got) in v.iter().enumerate() {
                assert_eq!(*got as usize, coset_perm(i, log_n),
                           "coset_perm disagrees with stwo at log_n={log_n}, i={i}");
            }
        }
    });
}

pub fn eval_of(log_size: u32, mut vals: Vec<M31>) -> Eval {
    assert_eq!(vals.len(), 1 << log_size);
    // B7a milestone 4 (18.3): the shuffle and the repack as one parallel
    // out-of-place gather straight into the packed lanes. `ZK_PARALLEL_GATHER=0`
    // restores the serial in-place pair below.
    if crate::util::switch_on("ZK_PARALLEL_GATHER")
        && log_size >= LOG_N_LANES.max(crate::util::parallel_gather_min())
    {
        coset_perm_selfcheck();
        let t = std::time::Instant::now();
        let n_vec = 1usize << (log_size - LOG_N_LANES);
        let src = &vals[..];
        let data: Vec<PackedM31> = (0..n_vec)
            .into_par_iter()
            .map(|vr| {
                let mut lanes = [M31::zero(); N_LANES];
                for (l, x) in lanes.iter_mut().enumerate() {
                    *x = src[coset_perm((vr << LOG_N_LANES) | l, log_size)];
                }
                PackedM31::from_array(lanes)
            })
            .collect();
        crate::util::add_phase("evalof/fused", t.elapsed().as_secs_f64());
        return CircleEvaluation::new(
            CanonicCoset::new(log_size).circle_domain(),
            BaseColumn::from_simd(data),
        );
    }
    // B7a milestone 4 (18.2): a column's trip from a `Vec<M31>` to a committed
    // column is two passes, and they are timed apart.
    let t = std::time::Instant::now();
    bit_reverse_coset_to_circle_domain_order(&mut vals);
    crate::util::add_phase("evalof/bitrev", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let col = BaseColumn::from_iter(vals);
    crate::util::add_phase("evalof/pack", t.elapsed().as_secs_f64());
    CircleEvaluation::new(CanonicCoset::new(log_size).circle_domain(), col)
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

/// B7b (README 26): the crate's prover backend, for the tree cache and the
/// session. With the `cuda` feature this is the fork's CudaBackend, so
/// commitment and proving run on the device; without it the same code runs on
/// SimdBackend, which is the reference build every digest comparison is made
/// against.
#[cfg(feature = "cuda")]
pub type PcsB = stwo::prover::backend::cuda::CudaBackend;
#[cfg(not(feature = "cuda"))]
pub type PcsB = SimdBackend;

/// Stream PP (E11): the physical commitment tree indices this crate speaks.
/// Tree 0 is the shape tree and tree 1 the model tree -- both preprocessed,
/// both committed before the base trace, because their cells enter the LogUp
/// numerators.  `boundary::FIRST_TREE` is the first boundary tree.
pub const SHAPE_TREE: usize = 0;
pub const MODEL_TREE: usize = 1;
pub const BASE_TREE: usize = 2;
pub const INTER_TREE: usize = 3;

pub struct Session<'a> {
    pub config: PcsConfig,
    pub channel: Blake2sM31Channel,
    pub cs: CommitmentSchemeProver<'a, PcsB, Blake2sM31MerkleChannel>,
    pub ops: Vec<Op>,
    pub tree_cols: Vec<usize>,
    pub tree_cells: Vec<usize>,
    pub tree_max_log: Vec<u32>,
    started: Instant,
}

pub fn twiddles(max_log: u32) -> TwiddleTree<PcsB> {
    PcsB::precompute_twiddles(
        CanonicCoset::new(max_log + 1 + LOG_BLOWUP).circle_domain().half_coset,
    )
}

impl<'a> Session<'a> {
    pub fn new(tw: &'a TwiddleTree<PcsB>) -> Self {
        let config = pcs_config();
        let mut channel = Blake2sM31Channel::default();
        config.mix_into(&mut channel);
        spans::reset();
        // B7a milestone 3: keep the interpolated coefficients, which is what
        // routes OODS through the fork's batched `batch_eval_at_point` instead
        // of one barycentric kernel launch per (column, point). See
        // gpu/m3_patch.py; ZK_STORE_COEFFICIENTS=0 restores the old path.
        let mut cs = CommitmentSchemeProver::<PcsB, Blake2sM31MerkleChannel>::new(config, tw);
        if crate::util::store_coeffs() {
            cs.set_store_polynomials_coefficients();
        }
        // B12 stream HH: drop each tree's blown-up evaluations after its root
        // is mixed and rebuild them per phase. `lde_only` refuses when the
        // coefficients are off, so this is reached only when they are on.
        if crate::util::lde_only() {
            cs.set_lde_only();
        }
        Self {
            config,
            channel,
            cs,
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
        // The columns are built on the host. With the cuda feature they cross
        // to the device here; without it they are already where the prover
        // needs them and pass through unchanged.
        #[cfg(feature = "cuda")]
        let evals = crate::device::upload_all(evals);
        tb.extend_evals(evals);
        tb.commit(&mut self.channel);
        self.ops.push(Op::Tree(sizes));
        self.mark_tree();
    }

    /// B7a milestone 4 (18.5): commit a tree whose columns the caller still
    /// needs. The flat list is a list of references into the components, so
    /// building it costs nothing; see gpu/m4_refs.py.
    /// B7a milestone 4 (19.6): commit one tree whose columns may be on the
    /// host, on the device, or both. `Op::Tree`'s column order is part of the
    /// transcript, so the list is exactly the list the host path built.
    pub fn commit_any(&mut self, cols: Vec<crate::device::AnyCol>) {
        let sizes = cols.iter().map(|c| c.log_size()).collect_vec();
        self.tree_cols.push(sizes.len());
        self.tree_cells.push(sizes.iter().map(|s| 1usize << s).sum());
        self.tree_max_log.push(sizes.iter().copied().max().unwrap_or(0));
        let mut tb = self.cs.tree_builder();
        tb.extend_evals(crate::device::take_all(cols));
        tb.commit(&mut self.channel);
        self.ops.push(Op::Tree(sizes));
        self.mark_tree();
    }

    /// B7b (README 26): tree `i` as committed, cloned on the device -- what
    /// the preprocessed-tree cache keeps.
    pub fn tree_clone(&self, i: usize) -> stwo::prover::CommitmentTreeProver<PcsB, Blake2sM31MerkleChannel> {
        self.cs.trees[i].clone()
    }

    /// Tree `i`'s root, as a string, for the cache's check.
    pub fn tree_root(&self, i: usize) -> String {
        format!("{:?}", self.cs.trees[i].commitment.root())
    }

    /// Tree `i`'s column log sizes, off the recorded transcript.
    pub fn tree_sizes(&self, i: usize) -> Vec<u32> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                Op::Tree(sz) => Some(sz.clone()),
                _ => None,
            })
            .nth(i)
            .expect("no such tree")
    }

    /// B7b (README 26): commit a tree that was committed before -- the clone
    /// goes in, its root is mixed, and the transcript records exactly what
    /// `commit_any` would have. Nothing is interpolated, extended or hashed.
    pub fn commit_cached(
        &mut self,
        tree: stwo::prover::CommitmentTreeProver<PcsB, Blake2sM31MerkleChannel>,
        sizes: Vec<u32>,
    ) {
        self.tree_cols.push(sizes.len());
        self.tree_cells.push(sizes.iter().map(|s| 1usize << s).sum());
        self.tree_max_log.push(sizes.iter().copied().max().unwrap_or(0));
        self.cs.push_cached_tree(tree, &mut self.channel);
        self.ops.push(Op::Tree(sizes));
        self.mark_tree();
    }

    pub fn commit_ref(&mut self, evals: &[&Eval]) {
        let sizes = evals.iter().map(|e| e.domain.log_size()).collect_vec();
        self.tree_cols.push(sizes.len());
        self.tree_cells.push(sizes.iter().map(|s| 1usize << s).sum());
        self.tree_max_log.push(sizes.iter().copied().max().unwrap_or(0));
        let mut tb = self.cs.tree_builder();
        #[cfg(feature = "cuda")]
        tb.extend_evals(crate::device::upload_all_ref(evals));
        // Without the cuda feature `extend_evals` needs owned columns, so the
        // one copy `prove_layer` used to make is made here instead. Same
        // bytes, same count -- moved, not removed. See 18.5.
        #[cfg(not(feature = "cuda"))]
        tb.extend_evals(evals.iter().map(|e| (*e).clone()).collect());
        tb.commit(&mut self.channel);
        self.ops.push(Op::Tree(sizes));
        self.mark_tree();
    }

    /// B12 stream GG: one device-memory mark at the end of every commit, so a
    /// tree's bytes are read on their own.  Trees 0, 1 and 2 are marked here;
    /// the boundary trees from 3 up are marked once, together, by
    /// `mark_boundary` when the caller has committed the last of them.
    fn mark_tree(&self) {
        let i = self.tree_cols.len() - 1;
        // Stream PP (E11): tree 0 is the shape tree and tree 1 the model tree;
        // `commit_pre` keeps its name for the shape tree so the mark series is
        // comparable with the records made before the split.
        let label = match i {
            0 => "commit_pre",
            1 => "commit_model",
            2 => "commit_base",
            3 => "commit_inter",
            _ => return,
        };
        crate::device::mem_mark_tree(label, i, self.tree_cols[i], self.tree_cells[i],
                                     self.tree_max_log[i]);
    }

    /// The boundary trees (stream PP: 4 and up), as one mark: their columns and
    /// cells summed, their tallest column.  Nothing when there are none.
    pub fn mark_boundary(&self) {
        if self.tree_cols.len() <= 4 {
            return;
        }
        crate::device::mem_mark_tree(
            "commit_boundary",
            4,
            self.tree_cols[4..].iter().sum(),
            self.tree_cells[4..].iter().sum(),
            self.tree_max_log[4..].iter().copied().max().unwrap_or(0),
        );
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
        provers: &[&dyn ComponentProver<PcsB>],
    ) -> (Result<Proof, String>, RawStats) {
        let started = self.started;
        let pre_ok = self.check_pre();
        let tree_cols = self.tree_cols.clone();
        let tree_cells = self.tree_cells.clone();
        let res = prove::<PcsB, Blake2sM31MerkleChannel>(provers, &mut self.channel, self.cs)
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
        provers: &[&dyn ComponentProver<PcsB>],
        components: &[&dyn Component],
    ) -> Proven {
        let config = self.config;
        let ops = std::mem::take(&mut self.ops);
        let started = self.started;
        let tree_cols = self.tree_cols.clone();
        let tree_cells = self.tree_cells.clone();
        let pre_ok = self.check_pre();
        let proof = match prove::<PcsB, Blake2sM31MerkleChannel>(
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

impl RawStats {
    /// Stream PP (E11): the preprocessed trace, both trees.
    pub fn pre_cells(&self) -> usize {
        self.tree_cells[..BASE_TREE.min(self.tree_cells.len())].iter().sum()
    }
    /// Stream PP (E11): the model tree alone.
    pub fn model_cells(&self) -> usize {
        self.tree_cells.get(MODEL_TREE).copied().unwrap_or(0)
    }
    /// Everything a chunk pays for again.
    pub fn fresh_cells(&self) -> usize {
        self.tree_cells[BASE_TREE.min(self.tree_cells.len())..].iter().sum()
    }
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
    /// Everything a chunk pays for again: stream PP moved the preprocessed
    /// trace into two trees, so the fresh trees start at `BASE_TREE`.
    pub fn fresh_cells(&self) -> usize {
        self.tree_cells[BASE_TREE..].iter().sum()
    }
    /// Stream PP (E11): the preprocessed trace, both trees.
    pub fn pre_cells(&self) -> usize {
        self.tree_cells[..BASE_TREE.min(self.tree_cells.len())].iter().sum()
    }
    /// Stream PP (E11): the model tree alone.
    pub fn model_cells(&self) -> usize {
        self.tree_cells.get(MODEL_TREE).copied().unwrap_or(0)
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
            // stream PP: `preprocessed` is the two preprocessed trees together,
            // so it is comparable with the records made before the split;
            // `shape` and `model` are the halves.
            "cells": {"preprocessed": self.pre_cells(),
                      "shape": self.tree_cells.first().copied().unwrap_or(0),
                      "model": self.model_cells(),
                      "base": self.tree_cells.get(BASE_TREE).copied().unwrap_or(0),
                      "interaction": self.tree_cells.get(INTER_TREE).copied().unwrap_or(0),
                      "fresh": self.fresh_cells(), "total": self.total_cells()},
            "columns": {"preprocessed": self.tree_cols[..BASE_TREE.min(self.tree_cols.len())]
                            .iter().sum::<usize>(),
                        "shape": self.tree_cols.first().copied().unwrap_or(0),
                        "model": self.tree_cols.get(MODEL_TREE).copied().unwrap_or(0),
                        "base": self.tree_cols.get(BASE_TREE).copied().unwrap_or(0),
                        "interaction": self.tree_cols.get(INTER_TREE).copied().unwrap_or(0),
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
