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
        static PARTIALS: *const u32 =
            unsafe { bindings::gkr_alloc_qm31(2 * GKR_SUM_MAX_BLOCKS as u32) };
    }
    PARTIALS.with(|p| *p)
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
        // band plan produces (2-entry instances over tall rows are halved all
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
    /// Instances of differing sizes stagger into the batch exactly as the band
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

    /// Grand product through the whole pipeline too -- the band arm never uses
    /// it, but the backend claims all four variants.
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
