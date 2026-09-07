//! B11 stream R: the GKR sumcheck on the device.
//!
//! Before this stream `gen_eq_evals` was wired to a real kernel and the two
//! functions that *are* the sumcheck -- `next_layer` and
//! `sum_as_poly_in_first_variable` -- were `todo!()`. Both are kernels now,
//! semantically the CPU backend's loops (`backend/cpu/lookups/gkr.rs`), which
//! is what the CPU-match tests below assert. Every value is exact field
//! arithmetic, so the device proof is byte-identical to the host proof on the
//! same input layers -- the digest gate the composed arm relies on.
//!
//! Two costs are worth naming here rather than discovering in a profile:
//!
//! * `next_layer` launches once per call and never synchronises -- the whole
//!   circuit evaluation (`gen_layers`) is stream-ordered device work.
//! * `sum_as_poly_in_first_variable` must hand its two evaluations back to the
//!   host every round -- the round polynomial goes into the transcript, and
//!   the next challenge depends on it -- so each call is one kernel launch
//!   plus one small copy home (at most `2 * GKR_SUM_MAX_BLOCKS` felts). That
//!   per-round synchronisation is protocol-inherent: the only way to remove it
//!   is to batch *across instances of a round*, which lives above this trait.
//!
//! When `B11_GKR_PROF=1` each phase synchronises around its calls and feeds a
//! process-wide accumulator (`prof::snapshot`), so a profiled run splits the
//! sumcheck into eq evals / layer halvings / round sums / variable fixes. The
//! un-profiled path never synchronises beyond the copies it cannot avoid.

use num_traits::Zero;

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::cuda::CudaBackend;
use crate::prover::lookups::gkr_prover::{
    correct_sum_as_poly_in_first_variable, GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::prover::lookups::mle::Mle;
use crate::prover::lookups::sumcheck::MultivariatePolyOracle;
use crate::prover::lookups::utils::UnivariatePoly;
use crate::stwo_cuda::bindings::CudaSecureField;
use crate::stwo_cuda::{bindings, secure_field_vec::SecureFieldVec};

/// The most partial pairs a round-sum launch produces, so the copy home stays
/// one packet. 264 blocks of 256 threads covers an H100's 132 SMs twice over.
const GKR_SUM_MAX_BLOCKS: usize = 264;

/// A device buffer straight from the pool, with no zeroing pass: every cell of
/// a sumcheck buffer is written by the kernel that fills it, so the memset in
/// `cuda_malloc_uint32_t` would be an extra launch and an extra full write per
/// layer. Freed by the ordinary `Drop`, which is pool-aware.
pub(crate) fn alloc_secure(n: usize) -> SecureFieldVec {
    SecureFieldVec::new(unsafe { bindings::gkr_alloc_qm31(n as u32) }, n)
}

/// The round-sum partials land in one per-thread buffer, sized for the largest
/// launch and never freed: allocating it per round was measurable against
/// launches this small. Per-thread rather than process-wide because two
/// concurrent `sum_as_poly` calls (the test harness runs tests in parallel)
/// must not share a scratch buffer; the prover itself is sequential here.
fn partials_ptr() -> *const u32 {
    thread_local! {
        static PARTIALS: *const u32 = unsafe {
            bindings::gkr_alloc_qm31(2 * (GKR_SUM_MAX_BLOCKS * GKR_SUM_MAX_INSTANCES) as u32)
        };
    }
    PARTIALS.with(|p| *p)
}

/// The most instances one fused round writes partials for in a single launch.
///
/// B11 stream X, lever 1. Stream T wrote this as 64 with "the composed arm's
/// batch is fifteen" beside it, and a batch beyond it fell back to the
/// per-instance loop -- the exact round trip the fusion exists to remove.
/// Streams U and V then took the batch to 46, 76 and 86 instances, and from 76
/// on every round with more than 64 live oracles has been paying one launch and
/// one blocking copy home per instance: measured on the H100 at C=4096 with all
/// five flags, 12,092 round-sum calls against 347 rounds, an average of 35
/// per round where there should be one.
///
/// So the number is now a *chunk* size and not a cliff. A batch larger than it
/// is launched in groups of this many, which keeps the partials scratch bounded
/// and makes the size fallback unreachable; 256 is chosen so the composed arm's
/// 86 instances are one launch with room for the families after it, and costs
/// `2 * 264 * 256` field elements of scratch -- about 2 MiB, once per thread.
const GKR_SUM_MAX_INSTANCES: usize = 256;

unsafe extern "C" {
    /// B11 stream T: one round's sums for every live instance, in one launch.
    /// Declared here rather than in `bindings` because it is the GKR arm's own
    /// kernel and nothing else calls it.
    /// B11 stream T: one round's variable fixes, every column in one launch.
    fn gkr_fix_batch(
        src_host: *const *const std::ffi::c_void,
        is_base_host: *const u32,
        dst_host: *const *const u32,
        n_cols: u32,
        assignment: CudaSecureField,
        half: u32,
    );

    fn gkr_sum_batch(
        a_host: *const *const std::ffi::c_void,
        b_host: *const *const u32,
        kind_host: *const u32,
        n_inst: u32,
        eq: *const u32,
        lambda: CudaSecureField,
        n_terms: u32,
        partials: *const u32,
        n_blocks: u32,
    );
}

impl GkrOps for CudaBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let _t = prof::scope(prof::EQ);
        let y_size = y.len();
        let result_evals = alloc_secure(1 << y_size);

        unsafe {
            bindings::gen_eq_evals(
                v.into(),
                y.as_ptr() as *const CudaSecureField,
                y_size as u32,
                result_evals.device_ptr as *const CudaSecureField,
                result_evals.size as u32,
            );
        }

        Mle::new(result_evals)
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        let _t = prof::scope(prof::NEXT);
        let n_out = (1usize << layer.n_variables()) / 2;
        match layer {
            Layer::GrandProduct(mle) => {
                let out = alloc_secure(n_out);
                unsafe {
                    bindings::gkr_next_layer_grand_product(
                        mle.device_ptr,
                        out.device_ptr,
                        n_out as u32,
                    );
                }
                Layer::GrandProduct(Mle::new(out))
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                let num_out = alloc_secure(n_out);
                let den_out = alloc_secure(n_out);
                unsafe {
                    bindings::gkr_next_layer_logup_generic(
                        numerators.device_ptr,
                        denominators.device_ptr,
                        num_out.device_ptr,
                        den_out.device_ptr,
                        n_out as u32,
                    );
                }
                Layer::LogUpGeneric {
                    numerators: Mle::new(num_out),
                    denominators: Mle::new(den_out),
                }
            }
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => {
                let num_out = alloc_secure(n_out);
                let den_out = alloc_secure(n_out);
                unsafe {
                    bindings::gkr_next_layer_logup_multiplicities(
                        numerators.device_ptr,
                        denominators.device_ptr,
                        num_out.device_ptr,
                        den_out.device_ptr,
                        n_out as u32,
                    );
                }
                Layer::LogUpGeneric {
                    numerators: Mle::new(num_out),
                    denominators: Mle::new(den_out),
                }
            }
            Layer::LogUpSingles { denominators } => {
                let num_out = alloc_secure(n_out);
                let den_out = alloc_secure(n_out);
                unsafe {
                    bindings::gkr_next_layer_logup_singles(
                        denominators.device_ptr,
                        num_out.device_ptr,
                        den_out.device_ptr,
                        n_out as u32,
                    );
                }
                Layer::LogUpGeneric {
                    numerators: Mle::new(num_out),
                    denominators: Mle::new(den_out),
                }
            }
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        let _t = prof::scope(prof::SUM);
        let n_variables = h.n_variables();
        assert_ne!(n_variables, 0);
        let n_terms = 1usize << (n_variables - 1);
        let eq_evals = h.eq_evals.as_ref();
        let y = eq_evals.y();
        let lambda: CudaSecureField = h.lambda.into();

        let n_blocks = n_terms.div_ceil(256).min(GKR_SUM_MAX_BLOCKS).max(1);
        let partials = partials_ptr();

        unsafe {
            match &h.input_layer {
                Layer::GrandProduct(mle) => bindings::gkr_sum_grand_product(
                    mle.device_ptr,
                    eq_evals.device_ptr,
                    n_terms as u32,
                    partials,
                    n_blocks as u32,
                ),
                Layer::LogUpGeneric {
                    numerators,
                    denominators,
                } => bindings::gkr_sum_logup_generic(
                    numerators.device_ptr,
                    denominators.device_ptr,
                    eq_evals.device_ptr,
                    lambda,
                    n_terms as u32,
                    partials,
                    n_blocks as u32,
                ),
                Layer::LogUpMultiplicities {
                    numerators,
                    denominators,
                } => bindings::gkr_sum_logup_multiplicities(
                    numerators.device_ptr,
                    denominators.device_ptr,
                    eq_evals.device_ptr,
                    lambda,
                    n_terms as u32,
                    partials,
                    n_blocks as u32,
                ),
                Layer::LogUpSingles { denominators } => bindings::gkr_sum_logup_singles(
                    denominators.device_ptr,
                    eq_evals.device_ptr,
                    lambda,
                    n_terms as u32,
                    partials,
                    n_blocks as u32,
                ),
            }
        }

        // Field addition is exact and associative, so adding the block
        // partials on the host changes nothing about the values.
        let mut host: Vec<SecureField> = Vec::with_capacity(2 * n_blocks);
        unsafe {
            host.set_len(2 * n_blocks);
            bindings::gkr_read_partials(
                partials,
                n_blocks as u32,
                host.as_mut_ptr() as *mut u32,
            );
        }
        let mut eval_at_0 = SecureField::zero();
        let mut eval_at_2 = SecureField::zero();
        for pair in host.chunks_exact(2) {
            eval_at_0 += pair[0];
            eval_at_2 += pair[1];
        }

        eval_at_0 *= h.eq_fixed_var_correction;
        eval_at_2 *= h.eq_fixed_var_correction;
        correct_sum_as_poly_in_first_variable(eval_at_0, eval_at_2, claim, y, n_variables)
    }

    /// B11 stream T: the whole round in one launch and one copy home.
    ///
    /// Stream R measured the wall this removes: per instance per round, one
    /// launch plus one *blocking* copy home, because the next challenge depends
    /// on the answer. The oracles a round is live on all have the same number of
    /// variables and share one `eq` table and one `lambda` -- the conditions are
    /// checked here, not assumed, and a batch that fails any of them takes the
    /// per-instance loop. **Size is no longer one of those conditions**: B11
    /// stream X made the instance cap a chunk size, because from 76 instances on
    /// it was sending whole rounds back down the loop this exists to replace.
    /// The kernel's per-thread arithmetic is the
    /// single-instance kernels' arithmetic term for term, so every round
    /// polynomial is the field element it was before: this is a scheduling
    /// change, and the proof digest is what says so.
    fn sum_as_poly_in_first_variable_batch(
        hs: &[&GkrMultivariatePolyOracle<'_, Self>],
        claims: &[SecureField],
    ) -> Vec<UnivariatePoly<SecureField>> {
        let _t = prof::scope(prof::SUM);
        let fallback = || -> Vec<UnivariatePoly<SecureField>> {
            hs.iter()
                .zip(claims)
                .map(|(h, &claim)| Self::sum_as_poly_in_first_variable(h, claim))
                .collect()
        };
        // `B11_GKR_FUSE=0` pins the round back to the per-instance loop, so the
        // fused and unfused arms can be measured on one build and one chunk.
        if std::env::var("B11_GKR_FUSE").map(|v| v == "0").unwrap_or(false) {
            return fallback();
        }
        let Some(first) = hs.first() else { return vec![] };
        let n_variables = first.n_variables();
        let n_inst = hs.len();
        if n_inst < 2 || n_variables == 0 {
            return fallback();
        }
        let eq_ptr = first.eq_evals.as_ref().device_ptr;
        let lambda = first.lambda;
        let uniform = hs.iter().all(|h| {
            h.n_variables() == n_variables
                && h.eq_evals.as_ref().device_ptr == eq_ptr
                && h.lambda == lambda
        });
        if !uniform {
            return fallback();
        }

        let n_terms = 1usize << (n_variables - 1);
        let n_blocks = n_terms.div_ceil(256).min(GKR_SUM_MAX_BLOCKS).max(1);

        // One copy home per group of at most `GKR_SUM_MAX_INSTANCES`, which for
        // every batch the composed arm has ever built is one copy for the whole
        // round. The grouping is a scheduling decision and nothing else: each
        // instance's partials are the same field elements the same kernel body
        // produced before, summed on the host in the same order, so the round
        // polynomial is the polynomial it was.
        let partials = partials_ptr();
        let mut host: Vec<SecureField> = Vec::with_capacity(2 * n_blocks * n_inst);
        unsafe { host.set_len(2 * n_blocks * n_inst) };
        for (g, group) in hs.chunks(GKR_SUM_MAX_INSTANCES).enumerate() {
            let base = g * GKR_SUM_MAX_INSTANCES;
            let mut a: Vec<*const std::ffi::c_void> = Vec::with_capacity(group.len());
            let mut b: Vec<*const u32> = Vec::with_capacity(group.len());
            let mut kind: Vec<u32> = Vec::with_capacity(group.len());
            for h in group {
                let (ap, bp, k) = match &h.input_layer {
                    Layer::GrandProduct(mle) => {
                        (mle.device_ptr as *const _, std::ptr::null(), 0u32)
                    }
                    Layer::LogUpGeneric {
                        numerators,
                        denominators,
                    } => (numerators.device_ptr as *const _, denominators.device_ptr, 1),
                    Layer::LogUpMultiplicities {
                        numerators,
                        denominators,
                    } => (numerators.device_ptr as *const _, denominators.device_ptr, 2),
                    Layer::LogUpSingles { denominators } => {
                        (std::ptr::null(), denominators.device_ptr, 3)
                    }
                };
                a.push(ap);
                b.push(bp);
                kind.push(k);
            }
            let pairs = n_blocks * group.len();
            unsafe {
                gkr_sum_batch(
                    a.as_ptr(),
                    b.as_ptr(),
                    kind.as_ptr(),
                    group.len() as u32,
                    eq_ptr,
                    lambda.into(),
                    n_terms as u32,
                    partials,
                    n_blocks as u32,
                );
                bindings::gkr_read_partials(
                    partials,
                    pairs as u32,
                    host[2 * base * n_blocks..].as_mut_ptr() as *mut u32,
                );
            }
        }

        hs.iter()
            .zip(claims)
            .enumerate()
            .map(|(i, (h, &claim))| {
                let mut eval_at_0 = SecureField::zero();
                let mut eval_at_2 = SecureField::zero();
                for pair in host[2 * i * n_blocks..2 * (i + 1) * n_blocks].chunks_exact(2) {
                    eval_at_0 += pair[0];
                    eval_at_2 += pair[1];
                }
                eval_at_0 *= h.eq_fixed_var_correction;
                eval_at_2 *= h.eq_fixed_var_correction;
                correct_sum_as_poly_in_first_variable(
                    eval_at_0,
                    eval_at_2,
                    claim,
                    h.eq_evals.y(),
                    n_variables,
                )
            })
            .collect()
    }

    /// B11 stream T: one launch for every column of every layer the round folds.
    ///
    /// `Mle::fix_first_variable` allocates its output and launches once per
    /// column; a round folds two columns per LogUp instance, which stream R
    /// counted as 9,624 launches a chunk. The outputs are allocated first and
    /// their pointers handed to one kernel, so the round's folds cost one
    /// launch. The per-element arithmetic is `mle.cu`'s, unchanged.
    fn fix_first_variable_batch(
        layers: Vec<Layer<Self>>,
        challenge: SecureField,
    ) -> Vec<Layer<Self>> {
        let _t = prof::scope(prof::FIX);
        let unfused = std::env::var("B11_GKR_FUSE").map(|v| v == "0").unwrap_or(false);
        if unfused || layers.len() < 2 {
            return layers.into_iter().map(|l| l.fix_first_variable(challenge)).collect();
        }
        // Every layer here has the same number of variables, so every column has
        // the same length; a layer with nothing left to fold is handed back.
        let half = (1usize << layers[0].n_variables()) >> 1;
        if half == 0 || layers.iter().any(|l| l.n_variables() != layers[0].n_variables()) {
            return layers.into_iter().map(|l| l.fix_first_variable(challenge)).collect();
        }

        let mut src: Vec<*const std::ffi::c_void> = vec![];
        let mut is_base: Vec<u32> = vec![];
        let mut outs: Vec<SecureFieldVec> = vec![];
        for layer in &layers {
            let mut take = |p: *const u32, base: bool| {
                src.push(p as *const std::ffi::c_void);
                is_base.push(base as u32);
                outs.push(alloc_secure(half));
            };
            match layer {
                Layer::GrandProduct(mle) => take(mle.device_ptr, false),
                Layer::LogUpGeneric {
                    numerators,
                    denominators,
                } => {
                    take(numerators.device_ptr, false);
                    take(denominators.device_ptr, false);
                }
                Layer::LogUpMultiplicities {
                    numerators,
                    denominators,
                } => {
                    take(numerators.device_ptr, true);
                    take(denominators.device_ptr, false);
                }
                Layer::LogUpSingles { denominators } => take(denominators.device_ptr, false),
            }
        }
        let dst: Vec<*const u32> = outs.iter().map(|c| c.device_ptr).collect();
        unsafe {
            gkr_fix_batch(
                src.as_ptr(),
                is_base.as_ptr(),
                dst.as_ptr(),
                src.len() as u32,
                challenge.into(),
                half as u32,
            );
        }
        // The sources are freed as the old layers drop below; the pool frees are
        // stream-ordered behind the launch, exactly as the per-column path's are.
        let mut outs = outs.into_iter();
        let folded = layers
            .iter()
            .map(|layer| match layer {
                Layer::GrandProduct(_) => Layer::GrandProduct(Mle::new(outs.next().unwrap())),
                Layer::LogUpGeneric { .. } | Layer::LogUpMultiplicities { .. } => {
                    Layer::LogUpGeneric {
                        numerators: Mle::new(outs.next().unwrap()),
                        denominators: Mle::new(outs.next().unwrap()),
                    }
                }
                Layer::LogUpSingles { .. } => Layer::LogUpSingles {
                    denominators: Mle::new(outs.next().unwrap()),
                },
            })
            .collect();
        assert!(outs.next().is_none(), "more folded columns than layers asked for");
        drop(layers);
        folded
    }
}

/// The device sumcheck's phase accounting, on `B11_GKR_PROF=1`.
///
/// Each phase synchronises the device around its calls so wall time can be
/// attributed to it; the switch off, nothing here runs and nothing
/// synchronises. The accumulators are process-wide because the trait's
/// functions carry no state, and the prover is sequential where they run.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub const EQ: usize = 0;
    pub const NEXT: usize = 1;
    pub const SUM: usize = 2;
    pub const FIX: usize = 3;
    pub const NAMES: [&str; 4] = ["eq_evals", "next_layer", "round_sums", "fix_var"];

    static NANOS: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static CALLS: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];

    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("B11_GKR_PROF").map(|v| v != "0").unwrap_or(false))
    }

    /// `B11_GKR_PROF=2`: additionally print every call, for hunting a cost
    /// that an aggregate hides.
    fn verbose() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("B11_GKR_PROF").map(|v| v == "2").unwrap_or(false))
    }

    pub struct Scope {
        phase: usize,
        start: std::time::Instant,
    }

    pub fn scope(phase: usize) -> Option<Scope> {
        if !on() {
            return None;
        }
        unsafe { super::bindings::gkr_device_synchronize() };
        Some(Scope {
            phase,
            start: std::time::Instant::now(),
        })
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            unsafe { super::bindings::gkr_device_synchronize() };
            let nanos = self.start.elapsed().as_nanos() as u64;
            NANOS[self.phase].fetch_add(nanos, Ordering::Relaxed);
            CALLS[self.phase].fetch_add(1, Ordering::Relaxed);
            if verbose() {
                eprintln!("gkrprof: {} {} us", NAMES[self.phase], nanos / 1000);
            }
        }
    }

    pub fn reset() {
        for i in 0..NANOS.len() {
            NANOS[i].store(0, Ordering::Relaxed);
            CALLS[i].store(0, Ordering::Relaxed);
        }
    }

    /// `(phase, seconds, calls)` for every phase, in a fixed order.
    pub fn snapshot() -> Vec<(&'static str, f64, u64)> {
        (0..NAMES.len())
            .map(|i| {
                (
                    NAMES[i],
                    NANOS[i].load(Ordering::Relaxed) as f64 / 1e9,
                    CALLS[i].load(Ordering::Relaxed),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::iter::zip;

    use itertools::Itertools;
    use num_traits::One;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::test_utils::test_channel;
    use crate::prover::backend::{Column, CpuBackend};
    use crate::prover::backend::cuda::CudaBackend;
    use crate::prover::lookups::gkr_prover::{
        prove_batch, EqEvals, GkrMultivariatePolyOracle, GkrOps, Layer,
    };
    use crate::prover::lookups::gkr_verifier::{partially_verify_batch, Gate};
    use crate::prover::lookups::mle::Mle;
    use crate::stwo_cuda::base_field_vec::BaseFieldVec;
    use crate::stwo_cuda::secure_field_vec::SecureFieldVec;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(0)
    }

    fn secure(rng: &mut SmallRng, n: usize) -> Vec<SecureField> {
        (0..n).map(|_| rng.gen()).collect()
    }

    fn cuda_secure(values: &[SecureField]) -> Mle<CudaBackend, SecureField> {
        Mle::new(SecureFieldVec::from_vec(values.to_vec()))
    }

    fn cpu_secure(values: &[SecureField]) -> Mle<CpuBackend, SecureField> {
        Mle::new(values.to_vec())
    }

    /// The whole batched proof, value by value: output claims, every round
    /// polynomial of every layer's sumcheck, and every mask. Equality here is
    /// exactly what makes the serialised proofs -- and so the composed
    /// digests -- byte-identical.
    fn assert_proof_eq(
        a: &crate::prover::lookups::gkr_verifier::GkrBatchProof,
        b: &crate::prover::lookups::gkr_verifier::GkrBatchProof,
    ) {
        assert_eq!(
            a.output_claims_by_instance, b.output_claims_by_instance,
            "output claims differ"
        );
        assert_eq!(a.sumcheck_proofs.len(), b.sumcheck_proofs.len());
        for (i, (x, y)) in zip(&a.sumcheck_proofs, &b.sumcheck_proofs).enumerate() {
            assert_eq!(x.round_polys.len(), y.round_polys.len(), "layer {i}");
            for (r, (p, q)) in zip(&x.round_polys, &y.round_polys).enumerate() {
                assert_eq!(p.to_vec(), q.to_vec(), "layer {i} round {r}");
            }
        }
        assert_eq!(a.layer_masks_by_instance.len(), b.layer_masks_by_instance.len());
        for (i, (x, y)) in
            zip(&a.layer_masks_by_instance, &b.layer_masks_by_instance).enumerate()
        {
            assert_eq!(x.len(), y.len(), "instance {i}");
            for (l, (m, n)) in zip(x, y).enumerate() {
                assert_eq!(m.columns(), n.columns(), "instance {i} mask {l}");
            }
        }
    }

    /// Compare a CUDA layer against a CPU layer, column by column.
    fn assert_layer_eq(got: &Layer<CudaBackend>, want: &Layer<CpuBackend>) {
        let got = got.to_cpu();
        match (&got, want) {
            (Layer::GrandProduct(a), Layer::GrandProduct(b)) => assert_eq!(**a, **b),
            (
                Layer::LogUpGeneric {
                    numerators: an,
                    denominators: ad,
                },
                Layer::LogUpGeneric {
                    numerators: bn,
                    denominators: bd,
                },
            ) => {
                assert_eq!(**an, **bn);
                assert_eq!(**ad, **bd);
            }
            _ => panic!("the two backends produced different layer kinds"),
        }
    }

    #[test]
    fn gen_eq_evals_matches_cpu() {
        let two = BaseField::from(2).into();

        let from_raw = [7, 3, 5, 6, 1, 1, 9].repeat(4);
        let y = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let cpu_eq_evals = CpuBackend::gen_eq_evals(&y, two);
        let gpu_eq_evals = CudaBackend::gen_eq_evals(&y, two);

        assert_eq!(gpu_eq_evals.to_cpu(), *cpu_eq_evals);
    }

    #[test]
    fn next_layer_matches_cpu() {
        let mut rng = rng();
        // Sizes from the smallest halvable layer through the odd tails the
        // key tile plan produces (2-entry instances over tall rows are halved all
        // the way down), including sizes around the launch geometry.
        for log_n in [1usize, 2, 3, 5, 8, 9, 12] {
            let n = 1 << log_n;
            let nums = secure(&mut rng, n);
            let dens = secure(&mut rng, n);
            let base_nums: Vec<BaseField> = (0..n).map(|_| rng.gen()).collect();

            // GrandProduct
            let got = CudaBackend::next_layer(&Layer::GrandProduct(cuda_secure(&nums)));
            let want = CpuBackend::next_layer(&Layer::GrandProduct(cpu_secure(&nums)));
            assert_layer_eq(&got, &want);

            // LogUpGeneric
            let got = CudaBackend::next_layer(&Layer::LogUpGeneric {
                numerators: cuda_secure(&nums),
                denominators: cuda_secure(&dens),
            });
            let want = CpuBackend::next_layer(&Layer::LogUpGeneric {
                numerators: cpu_secure(&nums),
                denominators: cpu_secure(&dens),
            });
            assert_layer_eq(&got, &want);

            // LogUpMultiplicities
            let got = CudaBackend::next_layer(&Layer::LogUpMultiplicities {
                numerators: Mle::new(BaseFieldVec::from_vec(base_nums.clone())),
                denominators: cuda_secure(&dens),
            });
            let want = CpuBackend::next_layer(&Layer::LogUpMultiplicities {
                numerators: Mle::new(base_nums.clone()),
                denominators: cpu_secure(&dens),
            });
            assert_layer_eq(&got, &want);

            // LogUpSingles
            let got = CudaBackend::next_layer(&Layer::LogUpSingles {
                denominators: cuda_secure(&dens),
            });
            let want = CpuBackend::next_layer(&Layer::LogUpSingles {
                denominators: cpu_secure(&dens),
            });
            assert_layer_eq(&got, &want);
        }
    }

    #[test]
    fn sum_as_poly_in_first_variable_matches_cpu() {
        let mut rng = rng();
        // Both implementations use `claim` the same way (only in the O(m)
        // correction), so any claim exercises the match; the real claims are
        // covered by the whole-proof tests below.
        let claim = SecureField::from_u32_unchecked(9, 9, 9, 9);
        for log_n in [2usize, 3, 5, 8, 12] {
            let n = 1 << log_n;
            let y = secure(&mut rng, log_n);
            let lambda: SecureField = rng.gen();
            let nums = secure(&mut rng, n);
            let dens = secure(&mut rng, n);

            let cuda_eq = EqEvals::<CudaBackend>::generate(&y);
            let cpu_eq = EqEvals::<CpuBackend>::generate(&y);

            let cases: Vec<(Layer<CudaBackend>, Layer<CpuBackend>)> = vec![
                (
                    Layer::GrandProduct(cuda_secure(&nums)),
                    Layer::GrandProduct(cpu_secure(&nums)),
                ),
                (
                    Layer::LogUpGeneric {
                        numerators: cuda_secure(&nums),
                        denominators: cuda_secure(&dens),
                    },
                    Layer::LogUpGeneric {
                        numerators: cpu_secure(&nums),
                        denominators: cpu_secure(&dens),
                    },
                ),
                (
                    Layer::LogUpSingles {
                        denominators: cuda_secure(&dens),
                    },
                    Layer::LogUpSingles {
                        denominators: cpu_secure(&dens),
                    },
                ),
            ];

            for (cuda_layer, cpu_layer) in cases {
                let got = CudaBackend::sum_as_poly_in_first_variable(
                    &GkrMultivariatePolyOracle {
                        eq_evals: Cow::Borrowed(&cuda_eq),
                        input_layer: cuda_layer,
                        eq_fixed_var_correction: SecureField::one(),
                        lambda,
                    },
                    claim,
                );
                let want = CpuBackend::sum_as_poly_in_first_variable(
                    &GkrMultivariatePolyOracle {
                        eq_evals: Cow::Borrowed(&cpu_eq),
                        input_layer: cpu_layer,
                        eq_fixed_var_correction: SecureField::one(),
                        lambda,
                    },
                    claim,
                );
                assert_eq!(got.to_vec(), want.to_vec());
            }
        }
    }

    /// The port's oracle in miniature: the whole batched proof, byte for byte.
    /// Instances of differing sizes stagger into the batch exactly as the key tile
    /// plan's odd tails do.
    #[test]
    fn prove_batch_matches_cpu() {
        let mut rng = rng();
        let sizes = [7usize, 4, 2, 7, 5];
        let mut cuda_layers: Vec<Layer<CudaBackend>> = vec![];
        let mut cpu_layers: Vec<Layer<CpuBackend>> = vec![];
        let mut gates = vec![];
        for (i, &log_n) in sizes.iter().enumerate() {
            let n = 1 << log_n;
            let dens = secure(&mut rng, n);
            match i % 3 {
                0 => {
                    cuda_layers.push(Layer::LogUpSingles {
                        denominators: cuda_secure(&dens),
                    });
                    cpu_layers.push(Layer::LogUpSingles {
                        denominators: cpu_secure(&dens),
                    });
                    gates.push(Gate::LogUp);
                }
                1 => {
                    let nums = secure(&mut rng, n);
                    cuda_layers.push(Layer::LogUpGeneric {
                        numerators: cuda_secure(&nums),
                        denominators: cuda_secure(&dens),
                    });
                    cpu_layers.push(Layer::LogUpGeneric {
                        numerators: cpu_secure(&nums),
                        denominators: cpu_secure(&dens),
                    });
                    gates.push(Gate::LogUp);
                }
                _ => {
                    let nums: Vec<BaseField> = (0..n).map(|_| rng.gen()).collect();
                    cuda_layers.push(Layer::LogUpMultiplicities {
                        numerators: Mle::new(BaseFieldVec::from_vec(nums.clone())),
                        denominators: cuda_secure(&dens),
                    });
                    cpu_layers.push(Layer::LogUpMultiplicities {
                        numerators: Mle::new(nums),
                        denominators: cpu_secure(&dens),
                    });
                    gates.push(Gate::LogUp);
                }
            }
        }

        let (cuda_proof, cuda_artifact) = prove_batch(&mut test_channel(), cuda_layers);
        let (cpu_proof, cpu_artifact) = prove_batch(&mut test_channel(), cpu_layers);

        // Value equality of the whole proof is the digest gate in small.
        assert_proof_eq(&cuda_proof, &cpu_proof);
        assert_eq!(cuda_artifact.ood_point, cpu_artifact.ood_point);
        for (a, b) in zip(
            &cuda_artifact.claims_to_verify_by_instance,
            &cpu_artifact.claims_to_verify_by_instance,
        ) {
            assert_eq!(a, b);
        }

        partially_verify_batch(gates, &cuda_proof, &mut test_channel())
            .expect("the device proof verifies");
    }

    /// Grand product through the whole pipeline too -- the key tile arm never uses
    /// it, but the backend claims all four variants.
    /// B11 stream T: the fused round against the unfused path it replaces.
    ///
    /// The digest gate says the composed proof is unmoved; this says why, value
    /// by value, on the mixture the composed arm actually presents -- instances
    /// of every layer variant, all live in the same round, sharing one `eq`
    /// table and one `lambda`.
    #[test]
    fn sum_as_poly_batch_matches_unbatched() {
        let mut rng = rng();
        for log_n in [2usize, 3, 5, 8, 12] {
            let n = 1 << log_n;
            let y = secure(&mut rng, log_n);
            let lambda: SecureField = rng.gen();
            let eq = EqEvals::<CudaBackend>::generate(&y);

            // one oracle per variant, plus a second generic so the batch is not
            // a permutation of distinct kinds
            let mut layers: Vec<Layer<CudaBackend>> = vec![];
            for _ in 0..2 {
                layers.push(Layer::LogUpGeneric {
                    numerators: cuda_secure(&secure(&mut rng, n)),
                    denominators: cuda_secure(&secure(&mut rng, n)),
                });
            }
            layers.push(Layer::LogUpSingles {
                denominators: cuda_secure(&secure(&mut rng, n)),
            });
            layers.push(Layer::GrandProduct(cuda_secure(&secure(&mut rng, n))));
            layers.push(Layer::LogUpMultiplicities {
                numerators: Mle::new(BaseFieldVec::from_vec(
                    (0..n).map(|_| rng.gen::<BaseField>()).collect_vec(),
                )),
                denominators: cuda_secure(&secure(&mut rng, n)),
            });

            // a different correction and claim per oracle: both enter only the
            // host-side tail, which is exactly where a batch could mix them up
            let oracles = layers
                .into_iter()
                .enumerate()
                .map(|(i, input_layer)| GkrMultivariatePolyOracle {
                    eq_evals: Cow::Borrowed(&eq),
                    input_layer,
                    eq_fixed_var_correction: SecureField::from_u32_unchecked(
                        i as u32 + 1,
                        7,
                        0,
                        i as u32,
                    ),
                    lambda,
                })
                .collect_vec();
            let claims = (0..oracles.len())
                .map(|i| SecureField::from_u32_unchecked(i as u32 + 3, 1, 4, 1))
                .collect_vec();

            let refs = oracles.iter().collect_vec();
            let got = CudaBackend::sum_as_poly_in_first_variable_batch(&refs, &claims);
            assert_eq!(got.len(), oracles.len());
            for (i, (oracle, &claim)) in zip(&oracles, &claims).enumerate() {
                let want = CudaBackend::sum_as_poly_in_first_variable(oracle, claim);
                assert_eq!(got[i].to_vec(), want.to_vec(), "log_n={log_n} oracle={i}");
            }
        }
    }

    #[test]
    fn prove_batch_grand_product_matches_cpu() {
        let mut rng = rng();
        let values = secure(&mut rng, 1 << 6);
        let (cuda_proof, _) = prove_batch(
            &mut test_channel(),
            vec![Layer::<CudaBackend>::GrandProduct(cuda_secure(&values))],
        );
        let (cpu_proof, _) = prove_batch(
            &mut test_channel(),
            vec![Layer::<CpuBackend>::GrandProduct(cpu_secure(&values))],
        );
        assert_proof_eq(&cuda_proof, &cpu_proof);
        partially_verify_batch(vec![Gate::GrandProduct], &cuda_proof, &mut test_channel())
            .expect("the device grand-product proof verifies");
    }
}
