//! B7b (README 26): the preprocessed tree, committed once per (layer, shape)
//! and kept on the device.
//!
//! Tree 0 is a function of the shape and of what the layer puts into it -- the
//! shared tables, the selectors, the weights and the scales -- and of nothing a
//! chunk changes. So a chunk whose layer and shape have been committed before
//! takes a device-to-device clone of that `CommitmentTreeProver` (the LDE'd
//! columns and the Merkle layers) and mixes its root, through the fork's own
//! `push_cached_tree`; the interpolation, the extension and the hash are not
//! run again.
//!
//! The key is `chunks::key_of`: the layer's source and the params, which is
//! what determines every cell of the tree in this binary. The check is the
//! root: under `B7_M5_PRECACHE=2` every chunk commits the tree itself and its
//! root is asserted equal to the cached one, which is a cell-for-cell
//! comparison by construction. Mode 1 takes the clone. `B7_M5_PREMUT=stalekey`
//! is the control: every layer maps to one key, so a loop over two different
//! layers must fail the root check (rule 5).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel;
use stwo::prover::CommitmentTreeProver;

// B7b: the crate's prover backend -- `CudaBackend` on the GPU tree,
// `SimdBackend` on the SIMD reference. `m5_patch.py` adds the alias to
// `common.rs`, so this module compiles on both and the tree cache is a no-op
// on the reference (mode 0, never taken).
use crate::common::PcsB;

pub type Tree = CommitmentTreeProver<PcsB, Blake2sM31MerkleChannel>;

/// The tree holds device pointers (on CUDA). The context is per process and
/// nothing here is bound to a thread, so handing it across threads is sound;
/// the compiler cannot know that, hence the wrapper.
struct SendTree(Tree);
unsafe impl Send for SendTree {}

struct Entry {
    tree: SendTree,
    sizes: Vec<u32>,
    root: String,
    /// Stream PP (E11): the model tree, committed beside the shape tree and
    /// cached with it -- one key covers both, because both are functions of the
    /// same (layer, shape) pair.
    model: SendTree,
    model_sizes: Vec<u32>,
    model_root: String,
}

static CACHE: OnceLock<Mutex<HashMap<u64, Entry>>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static CHECKS: AtomicU64 = AtomicU64::new(0);
static CLONE_NS: AtomicU64 = AtomicU64::new(0);
static FORGOTTEN: AtomicU64 = AtomicU64::new(0);
static PEAK_ENTRIES: AtomicU64 = AtomicU64::new(0);

fn cache() -> &'static Mutex<HashMap<u64, Entry>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn mode() -> u32 {
    crate::util::m4b_mode("B7_M5_PRECACHE")
}

fn key(k: Option<u64>) -> Option<u64> {
    match std::env::var("B7_M5_PREMUT").as_deref() {
        Ok("stalekey") => k.map(|_| 0),
        Ok("none") | Err(_) => k,
        Ok(other) => panic!("B7_M5_PREMUT={other}: unknown mutation"),
    }
}

pub fn report() -> serde_json::Value {
    serde_json::json!({
        "hits": HITS.load(Relaxed),
        "misses": MISSES.load(Relaxed),
        "checked": CHECKS.load(Relaxed),
        "clone_secs": CLONE_NS.load(Relaxed) as f64 / 1e9,
        "entries": cache().lock().unwrap().len(),
        // Stream O: how many trees the loop handed back, and the most it ever
        // held at once. On a sequence of distinct shapes the second number is
        // the one that decides whether the run fits on the card.
        "forgotten": FORGOTTEN.load(Relaxed),
        "peak_entries": PEAK_ENTRIES.load(Relaxed),
    })
}

/// Drop the tree cached under this key, and say whether there was one.
///
/// B11 stream O. B7b's loop proves one shape over and over: one key, one
/// entry, and chunk 0's tree is what every chunk after it clones. The
/// sequence is the opposite -- 33 chunks, 33 layers, 33 shapes, so 33
/// distinct keys. Every chunk misses, every chunk pays `tree_clone` for an
/// entry nothing will ever take, and every entry stays on the device until
/// the process exits. Across 33 processes that cost nothing, because each
/// process held one and then died; in one process it is 33 preprocessed trees
/// resident at the end, which is the run.
///
/// So the loop hands a key back once the chunk that owns it has been proved,
/// which is the point where the last borrowed view of that tree is dropped.
/// It only does so for a chunk whose shape no later chunk repeats -- if two
/// chunks share a key the second is a genuine hit and the entry must stay.
///
/// Forgetting is off while a mutation is set: `stalekey` maps every layer to
/// one key precisely so that the second chunk's root check fails against the
/// first chunk's entry (rule 5), and an entry handed back before that
/// comparison would make the control pass by losing its evidence.
pub fn forget(k: u64) -> bool {
    if mode() == 0 || mutating() {
        return false;
    }
    let Some(k) = key(Some(k)) else { return false };
    let gone = cache().lock().unwrap().remove(&k).is_some();
    if gone {
        FORGOTTEN.fetch_add(1, Relaxed);
    }
    gone
}

/// Whether `B7_M5_PREMUT` is asking for something other than the honest key.
fn mutating() -> bool {
    matches!(std::env::var("B7_M5_PREMUT").as_deref(), Ok(v) if v != "none")
}

// B7b: the view of a cached tree. The default is a full clone -- correct on
// every backend, and what the SIMD reference (which never takes the cache)
// gets. `m5_patch.py` replaces the body below with a borrowed-pointer view on
// the CUDA tree: the same coefficients, evaluations and Merkle layers, none
// copied and none freed with the session. A clone was measured first (26.3):
// 1,284 columns twice over and ~2,600 device copies of the layers, 0.023 s per
// chunk -- launch-bound, and most of what the cache saved.
// B7B-VIEW-BODY
#[cfg(feature = "cuda")]
fn view(t: &Tree) -> Tree {
    use stwo::prover::poly::circle::{CircleCoefficients, CircleEvaluation};
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    let polynomials = t
        .polynomials
        .iter()
        .map(|p| {
            let coeffs = p.coeffs.as_ref().map(|c| {
                CircleCoefficients::new(BaseFieldVec::from_borrowed_ptr(c.coeffs.device_ptr,
                                                                        c.coeffs.size))
            });
            // B12 stream HH: under `lde_only` the cached tree holds no
            // evaluations, so the view carries `None` and the domain comes off
            // `Poly.domain`, which outlives them.
            let evals = p.evals.as_ref().map(|e| {
                CircleEvaluation::new(
                    e.domain,
                    BaseFieldVec::from_borrowed_ptr(e.values.device_ptr, e.values.size),
                )
            });
            stwo::prover::Poly::from_parts(coeffs, evals, p.domain)
        })
        .collect();
    let layers = t.commitment.layers.iter().map(|l| l.borrowed_view()).collect();
    CommitmentTreeProver {
        polynomials,
        commitment: stwo::prover::vcs::prover::MerkleProver { layers },
    }
}

// Without the cuda feature the tree's columns are host memory and a plain
// clone is the view.
#[cfg(not(feature = "cuda"))]
fn view(t: &Tree) -> Tree {
    t.clone()
}

/// The cached tree for this key, as a view, when mode 1 has one. The column
/// log sizes are asserted: a key collision between two shapes would otherwise
/// be a wrong proof rather than a loud one.
pub fn take(k: Option<u64>, sizes: &[u32], model_sizes: &[u32]) -> Option<(Tree, Tree)> {
    if mode() != 1 {
        return None;
    }
    let k = key(k)?;
    let g = cache().lock().unwrap();
    let e = g.get(&k)?;
    assert_eq!(e.sizes, sizes, "precache: a cached tree with different column sizes under key {k:#x}");
    assert_eq!(e.model_sizes, model_sizes,
               "precache: a cached model tree with different column sizes under key {k:#x}");
    let t = std::time::Instant::now();
    let tree = view(&e.tree.0);
    let model = view(&e.model.0);
    CLONE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    HITS.fetch_add(1, Relaxed);
    Some((tree, model))
}

/// After a fresh commit of tree 0: store it under the key, or -- when the key
/// is already there -- assert the fresh root against the stored one.
pub fn fill(k: Option<u64>, sess: &crate::common::Session<'_>) {
    if mode() == 0 {
        return;
    }
    let Some(k) = key(k) else { return };
    let root = sess.tree_root(0);
    let model_root = sess.tree_root(1);
    let mut g = cache().lock().unwrap();
    match g.get(&k) {
        Some(e) => {
            CHECKS.fetch_add(1, Relaxed);
            assert_eq!(
                e.root, root,
                "precache: the preprocessed tree committed for key {k:#x} has a different root \
                 from the cached one -- the key does not determine the tree"
            );
            assert_eq!(
                e.model_root, model_root,
                "precache: the model tree committed for key {k:#x} has a different root from \
                 the cached one -- the key does not determine the tree"
            );
        }
        None => {
            MISSES.fetch_add(1, Relaxed);
            let t = std::time::Instant::now();
            let tree = sess.tree_clone(0);
            let model = sess.tree_clone(1);
            CLONE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            g.insert(k, Entry {
                tree: SendTree(tree), sizes: sess.tree_sizes(0), root,
                model: SendTree(model), model_sizes: sess.tree_sizes(1), model_root,
            });
            PEAK_ENTRIES.fetch_max(g.len() as u64, Relaxed);
        }
    }
}
