//! Stream Q: the attention bands' lookups proved by LogUp-GKR.
//!
//! Today a band region's lookups are proved by committing running-sum columns:
//! `attn_trace::logup_cols` pairs the region's lookups, builds the pointwise
//! partial sums, and the AIR's `finalize_logup_in_pairs` asserts the recurrence
//! while a boundary constraint ties the last row to the region's entry in
//! `claims.use_sums`. Those columns are 88% of the bands' interaction cells and
//! the bands are two thirds of the interaction tree at long context, so they are
//! the largest single thing being committed that does not have to be.
//!
//! This module proves the same fractional sums with a sumcheck instead. It
//! commits nothing. What it produces is one `GkrBatchProof` per chunk and, from
//! that proof's output claims, exactly the same claimed sums the committed
//! columns produce today -- which is what `verify_composed`'s `sums_cancel`
//! consumes, so the global LogUp identity is unchanged.
//!
//! ## The shape of the argument
//!
//! A region's fractions are indexed by `(look, row)`. Laid out that way, a
//! region is a natural GKR instance: its input layer is an MLE over
//! `log2(n_looks) + log_rows` variables. Two details decide the layout.
//!
//! *The lookups are paired before they reach the circuit* (B11 stream R,
//! lever 1). The committed path already combines consecutive lookups into one
//! fraction per pair -- `plogup::paired`, the same batching
//! `finalize_logup_in_pairs` asserts -- and a LogUp GKR's own bottom layer
//! performs the identical combination. Feeding the circuit unpaired therefore
//! pays a whole extra input layer to recompute what `paired` already defines.
//! The input layer here is built from the *paired* fractions instead: one
//! entry per pair, the odd tail unpaired, which removes the bottom layer of
//! every tree and halves the sumcheck. The claimed sums are unchanged --
//! fraction addition is exact -- and `look_frac` through `paired` stays the
//! one definition of the fraction on both arms.
//!
//! *The entry axis wants a power of two.* Padding a 66-entry channel region up
//! to 128 would nearly double its sumcheck. Instead a region is split along the
//! binary decomposition of its entry count -- 66 becomes a 64-entry instance
//! and a 2-entry one -- which pads nothing and costs only instance bookkeeping.
//!
//! *The numerators come in two kinds.* A fully paired entry is generic even
//! when both its lookups have numerator one -- the combined numerator is
//! `d0 + d1` -- so paired entries live in `Layer::LogUpGeneric` instances. Only
//! an odd tail keeps its lookup's own kind: a unit tail becomes a
//! `Layer::LogUpSingles` instance and stores no numerators at all. A region
//! therefore contributes up to two groups of instances, and the split is by
//! numerator kind first, binary decomposition second.
//!
//! ## One `prove_batch` per chunk
//!
//! Every instance from every band goes into a single `prove_batch` call. This is
//! not an optimisation to add later: a sumcheck's domain halves every round, so
//! a per-region GKR would pay per-round launch overhead across dozens of small
//! trees, which stream C named as the likeliest way the sumcheck's cost model
//! turns out optimistic. `prove_batch` already draws one challenge pair per
//! layer and runs one batched sumcheck per layer across all instances, starting
//! each instance when the layer count reaches it, so the batching costs us
//! nothing but putting the instances in one vector.

use std::time::Instant;

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use rayon::prelude::*;
use stwo::core::channel::Channel;
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::column::SecureColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::lookups::gkr_prover::{Layer, prove_batch};
use stwo::prover::lookups::gkr_verifier::{
    Gate, GkrArtifact, GkrBatchProof, GkrError, partially_verify_batch,
};
use stwo::prover::lookups::mle::Mle;

use crate::attn_witness::Variant as AttnVariant;

/// One band region's lookup site, as `interaction_phase` reaches it.
///
/// `which` is the component index `devfrac::band_cols` uses -- 0 for the channel
/// region, 1 for the grid, 3 for the per-token -- so a site names the same thing
/// the existing code names.
#[derive(Clone, Debug)]
pub struct Site {
    pub band: usize,
    pub which: usize,
    pub log_rows: u32,
    /// numerator kind per lookup, in `add_to_relation` order: true when the
    /// numerator is one
    pub unit: Vec<bool>,
}

impl Site {
    pub fn tag(&self) -> String {
        format!("band{}/{}", self.band, ["chan", "key", "out", "row"][self.which])
    }
    pub fn n_looks(&self) -> usize {
        self.unit.len()
    }
    /// How many paired entries the site's lookups become: `plogup::paired`'s
    /// column count, the odd tail unpaired.
    pub fn n_ents(&self) -> usize {
        self.n_looks().div_ceil(2)
    }
    /// Whether entry `c` keeps a unit numerator. A fully paired entry never
    /// does -- combining `(1, d0)` and `(1, d1)` gives numerator `d0 + d1` --
    /// so only an odd tail whose lone lookup is unit stays unit.
    pub fn ent_unit(&self, c: usize) -> bool {
        2 * c + 1 == self.n_looks() && self.unit[2 * c]
    }
}

/// One GKR instance: a set of one site's paired entries whose count is a power
/// of two, over that site's rows.
#[derive(Clone, Debug)]
pub struct Inst {
    /// index into the site list
    pub site: usize,
    /// which of the site's paired entries this instance covers
    pub ents: Vec<usize>,
    pub log_rows: u32,
    /// every numerator in this instance is one
    pub unit: bool,
}

impl Inst {
    pub fn n_variables(&self) -> usize {
        self.ents.len().ilog2() as usize + self.log_rows as usize
    }
    /// Paired fractions: what the input layer holds and the sumcheck runs over.
    pub fn fractions(&self) -> usize {
        self.ents.len() << self.log_rows
    }
}

/// The instances the sites become, in site order.
///
/// Within a site the unit-numerator entries come first, then the rest; each
/// group is cut into power-of-two blocks largest first. The order is a pure
/// function of the site list, so the prover and the verifier derive the same
/// instance list from the plan and nothing about it travels in the proof.
pub fn plan(sites: &[Site]) -> Vec<Inst> {
    let mut out = vec![];
    for (i, s) in sites.iter().enumerate() {
        for unit in [true, false] {
            let group: Vec<usize> = (0..s.n_ents()).filter(|c| s.ent_unit(*c) == unit).collect();
            let mut at = 0usize;
            // largest block first: the highest set bit of what is left
            let mut left = group.len();
            while left > 0 {
                let take = 1usize << (usize::BITS - 1 - left.leading_zeros());
                out.push(Inst {
                    site: i,
                    ents: group[at..at + take].to_vec(),
                    log_rows: s.log_rows,
                    unit,
                });
                at += take;
                left -= take;
            }
        }
    }
    out
}

/// The gates the verifier needs, one per instance. Every band instance is a
/// LogUp fractional sum.
pub fn gates(insts: &[Inst]) -> Vec<Gate> {
    vec![Gate::LogUp; insts.len()]
}

/// Build one instance's input layer from a packed fraction source.
///
/// `frac(ent, vec_row)` is `plogup::paired`'s output -- the numerator and
/// denominator of one *paired* entry at one packed row, `look_frac` underneath
/// -- so the fractions proved here are the same function of the same committed
/// columns that the committed running-sum columns are built from. Nothing is
/// re-derived and nothing is fed unpaired.
fn input_layer<F>(inst: &Inst, frac: &F) -> Layer<SimdBackend>
where
    F: Fn(usize, usize) -> (PackedSecureField, PackedSecureField) + Sync,
{
    let n_vec_row = 1usize << (inst.log_rows - LOG_N_LANES);
    let n_vec = inst.ents.len() * n_vec_row;
    let length = inst.fractions();

    // `(entry, row)` in that order, so the leading `log2(n_ents)` variables are
    // the entry axis. The link in the memo reduces the input-layer claim over
    // exactly this split.
    let build = |pick: fn(&(PackedSecureField, PackedSecureField)) -> PackedSecureField| {
        let data: Vec<PackedSecureField> = (0..n_vec)
            .into_par_iter()
            .map(|k| {
                let (b, vr) = (k / n_vec_row, k % n_vec_row);
                pick(&frac(inst.ents[b], vr))
            })
            .collect();
        SecureColumn { data, length }
    };

    if inst.unit {
        Layer::LogUpSingles { denominators: Mle::new(build(|f| f.1)) }
    } else {
        Layer::LogUpGeneric {
            numerators: Mle::new(build(|f| f.0)),
            denominators: Mle::new(build(|f| f.1)),
        }
    }
}

/// The sites and input layers gathered from a pass over the bands.
///
/// `interaction_phase` builds each region's lookup list already, and that list
/// plus the column slice it indexes is only in scope right there. Rather than
/// reconstruct either outside -- two definitions of the same thing is how the
/// two arms drift apart -- the interaction pass hands them to this collector at
/// the same three sites where it calls `devfrac::band_cols`. Collecting is
/// purely additive: it reads the same columns and writes nothing the committed
/// path can see.
#[derive(Default)]
pub struct Collect {
    /// the band whose regions are arriving; the caller sets it before each
    /// band's interaction pass, because a corner band reaches
    /// `interaction_phase` with no `BandAir` to carry its index
    pub band: usize,
    pub sites: Vec<Site>,
    pub layers: Vec<Layer<SimdBackend>>,
    pub insts: Vec<Inst>,
    pub layers_secs: f64,
    /// the claimed sum of each site, summed directly from its fractions rather
    /// than through the sumcheck -- the self-test's reference
    pub direct: Vec<SecureField>,
    /// fill `direct` as sites arrive; off for a proving run, on for the test
    pub want_direct: bool,
}

impl Collect {
    pub fn with_direct(want_direct: bool) -> Self {
        Self { want_direct, ..Default::default() }
    }

    /// Take one region's lookups and build its instances' input layers.
    pub fn push(
        &mut self,
        which: usize,
        log_rows: u32,
        looks: &[crate::attn_trace::Look],
        cols: &[crate::common::Eval],
        pre: &[crate::common::Eval],
        exp: &crate::attn_air::ExpRel,
        ranges: &[crate::attn_air::RangeRel],
    ) {
        let t0 = Instant::now();
        let site = Site {
            band: self.band,
            which,
            log_rows,
            unit: looks.iter().map(crate::attn_trace::look_is_unit).collect(),
        };
        let at = self.sites.len();
        // `paired` batches consecutive lookups exactly as the committed path
        // does, so `frac(ent, vr)` is the paired entry's fraction and both arms
        // keep reading one definition -- `look_frac` through `paired`.
        let (n_ents, frac) = crate::plogup::paired(looks.len(), |look: usize, vr: usize| {
            crate::attn_trace::look_frac(&looks[look], vr, cols, pre, exp, ranges)
        });
        debug_assert_eq!(n_ents, site.n_ents());
        let mut direct = SecureField::zero();
        for inst in plan(std::slice::from_ref(&site)) {
            let inst = Inst { site: at, ..inst };
            if self.want_direct {
                direct += direct_sum(&inst, &frac);
            }
            self.layers.push(input_layer(&inst, &frac));
            self.insts.push(inst);
        }
        self.sites.push(site);
        self.direct.push(direct);
        self.layers_secs += t0.elapsed().as_secs_f64();
    }
}

/// B11 stream R: the device arm's layer plumbing.
///
/// The collected input layers are host (simd) columns; the sumcheck runs on
/// `CudaBackend`. The upload is timed on its own line because on the composed
/// cuda arm it is a stand-in: `gen_b65_dispatch` computes these very fractions
/// on the device already, so a resident arm pays a device-side *reshape* (one
/// pass moving the paired fractions into MLE layout), not a PCIe transfer. The
/// reshape's cost is measured directly -- one device-to-device copy of every
/// layer -- and reported beside the upload so neither is assumed.
#[cfg(feature = "cuda")]
pub mod dev {
    use stwo::prover::backend::Column as _;
    use stwo::prover::backend::cuda::CudaBackend;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::lookups::gkr_prover::Layer;
    use stwo::prover::lookups::mle::Mle;
    use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
    use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;

    pub fn sync() {
        unsafe { stwo::stwo_cuda::bindings::gkr_device_synchronize() };
    }

    pub fn upload_layer(l: &Layer<SimdBackend>) -> Layer<CudaBackend> {
        let up = |m: &Mle<SimdBackend, stwo::core::fields::qm31::SecureField>| {
            Mle::new(SecureFieldVec::from_vec(m.to_cpu()))
        };
        match l {
            Layer::GrandProduct(mle) => Layer::GrandProduct(up(mle)),
            Layer::LogUpGeneric { numerators, denominators } => Layer::LogUpGeneric {
                numerators: up(numerators),
                denominators: up(denominators),
            },
            Layer::LogUpMultiplicities { numerators, denominators } => {
                Layer::LogUpMultiplicities {
                    numerators: Mle::new(BaseFieldVec::from_vec(numerators.to_cpu())),
                    denominators: up(denominators),
                }
            }
            Layer::LogUpSingles { denominators } => Layer::LogUpSingles {
                denominators: up(denominators),
            },
        }
    }

    /// Grow the device memory pool once, ahead of the circuit evaluation.
    ///
    /// `gen_layers` materialises every GKR layer of every instance before the
    /// first sumcheck round, and nothing is freed until the rounds consume
    /// them -- so without this, every one of its allocations extends the
    /// pool's high-water mark and pays the driver's synchronous mapping cost,
    /// which profiling showed as ~0.5-1.5 ms per `next_layer` call regardless
    /// of size. One big allocation maps the same memory in one driver call;
    /// freeing it hands the range to the pool's free list.
    pub fn prewarm(layers: &[Layer<CudaBackend>]) -> f64 {
        use stwo::stwo_cuda::bindings;
        let input_qm31: usize = layers
            .iter()
            .map(|l| match l {
                Layer::GrandProduct(m) => m.len(),
                Layer::LogUpGeneric { numerators, denominators } => {
                    numerators.len() + denominators.len()
                }
                Layer::LogUpMultiplicities { denominators, .. } => 2 * denominators.len(),
                Layer::LogUpSingles { denominators } => denominators.len(),
            })
            .sum();
        // every level of every tree sums to under 2x the input layer; the
        // halved trees are generic (num + den) even where the input was
        // singles, so budget 4x, plus the eq tables and slack
        let budget = 4 * input_qm31 + (1 << 22);
        let t = std::time::Instant::now();
        unsafe {
            let p = bindings::gkr_alloc_qm31(budget as u32);
            bindings::cuda_free_memory(p as *const std::ffi::c_void);
        }
        // Launch every kernel the sumcheck uses once, on two elements. The
        // first launch of a kernel pays its module load -- measured at ~88 ms
        // for the first `next_layer` of a process, against ~13 us for every
        // call after it -- and the chunk loop pays that once per *process*,
        // not per chunk, exactly like `wall/cuda_init`.
        let two = SecureFieldVec::from_vec(vec![
            stwo::core::fields::qm31::SecureField::from_u32_unchecked(1, 0, 0, 0),
            stwo::core::fields::qm31::SecureField::from_u32_unchecked(2, 0, 0, 0),
        ]);
        let two_base = BaseFieldVec::from_vec(vec![
            stwo::core::fields::m31::M31::from_u32_unchecked(1),
            stwo::core::fields::m31::M31::from_u32_unchecked(2),
        ]);
        let out = |n: usize| SecureFieldVec::new_uninitialized(n);
        let (o1, o2, o3) = (out(1), out(1), out(2));
        let lambda = stwo::stwo_cuda::bindings::CudaSecureField::zero;
        unsafe {
            bindings::gkr_next_layer_grand_product(two.device_ptr, o1.device_ptr, 1);
            bindings::gkr_next_layer_logup_generic(
                two.device_ptr, two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_next_layer_logup_multiplicities(
                two_base.device_ptr, two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_next_layer_logup_singles(
                two.device_ptr, o1.device_ptr, o2.device_ptr, 1);
            bindings::gkr_sum_grand_product(two.device_ptr, two.device_ptr, 1, o3.device_ptr, 1);
            bindings::gkr_sum_logup_generic(
                two.device_ptr, two.device_ptr, two.device_ptr, lambda(), 1, o3.device_ptr, 1);
            bindings::gkr_sum_logup_multiplicities(
                two_base.device_ptr, two.device_ptr, two.device_ptr, lambda(), 1,
                o3.device_ptr, 1);
            bindings::gkr_sum_logup_singles(
                two.device_ptr, two.device_ptr, lambda(), 1, o3.device_ptr, 1);
            bindings::fix_first_variable_secure_field(
                two.device_ptr, 2, lambda(), o1.device_ptr);
            bindings::fix_first_variable_base_field(
                two_base.device_ptr, 2, lambda(), o1.device_ptr);
            bindings::gen_eq_evals(
                lambda(),
                two.device_ptr as *const stwo::stwo_cuda::bindings::CudaSecureField,
                0,
                o1.device_ptr as *const stwo::stwo_cuda::bindings::CudaSecureField,
                1,
            );
        }
        sync();
        t.elapsed().as_secs_f64()
    }

    /// One device-to-device pass over every layer column: what putting already
    /// resident fractions into MLE layout costs. The copies are dropped; only
    /// the time is kept.
    pub fn reshape_pass(layers: &[Layer<CudaBackend>]) {
        let copy = |src: &SecureFieldVec| {
            let mut dst = SecureFieldVec::new_uninitialized(src.len());
            dst.copy_from(src);
        };
        for l in layers {
            match l {
                Layer::GrandProduct(m) => copy(m),
                Layer::LogUpGeneric { numerators, denominators } => {
                    copy(numerators);
                    copy(denominators);
                }
                Layer::LogUpMultiplicities { denominators, .. } => copy(denominators),
                Layer::LogUpSingles { denominators } => copy(denominators),
            }
        }
    }
}

/// Prove everything a `Collect` gathered, in one batched GKR.
///
/// On the cuda build the sumcheck runs on the device -- that is the arm this
/// build exists to measure -- unless `B11_GKR_HOST=1` pins it to the host simd
/// backend for an on-box comparison. The proof bytes are the same either way,
/// which is the port's digest gate.
pub fn prove_collected(channel: &mut impl Channel, c: Collect) -> Proved {
    mix_shape(channel, &c.sites);
    let fractions = c.insts.iter().map(|i| i.fractions()).sum();
    let unpaired = c.sites.iter().map(|s| s.n_looks() << s.log_rows).sum();
    let n_instances = c.insts.len();

    #[cfg(feature = "cuda")]
    let (proof, arm, upload_secs, reshape_secs, prewarm_secs, prove_secs, phases) = {
        if std::env::var("B11_GKR_HOST").map(|v| v != "0").unwrap_or(false) {
            let t1 = Instant::now();
            let (proof, _artifact) = prove_batch(channel, c.layers);
            (proof, "host simd", 0.0, 0.0, 0.0, t1.elapsed().as_secs_f64(), None)
        } else {
            use stwo::prover::backend::cuda::lookups::gkr::prof;
            let t0 = Instant::now();
            let layers: Vec<_> = c.layers.iter().map(dev::upload_layer).collect();
            dev::sync();
            let upload_secs = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            dev::reshape_pass(&layers);
            dev::sync();
            let reshape_secs = t0.elapsed().as_secs_f64();
            // Once per process in production -- the pool keeps its high-water
            // mark across chunks -- so it is timed and reported, not counted
            // among the arm's per-chunk adds.
            let prewarm_secs = dev::prewarm(&layers);
            prof::reset();
            let t1 = Instant::now();
            let (proof, _artifact) = prove_batch(channel, layers);
            dev::sync();
            let prove_secs = t1.elapsed().as_secs_f64();
            let phases = prof::on().then(prof::snapshot);
            (proof, "device", upload_secs, reshape_secs, prewarm_secs, prove_secs, phases)
        }
    };
    #[cfg(not(feature = "cuda"))]
    let (proof, arm, upload_secs, reshape_secs, prewarm_secs, prove_secs, phases) = {
        let t1 = Instant::now();
        let (proof, _artifact) = prove_batch(channel, c.layers);
        let phases: Option<Vec<(&'static str, f64, u64)>> = None;
        (proof, "host simd", 0.0, 0.0, 0.0, t1.elapsed().as_secs_f64(), phases)
    };

    let sums = sums_of(c.sites.len(), &c.insts, &proof.output_claims_by_instance);
    let t2 = Instant::now();
    let proof_bytes = serde_json::to_vec(&proof).map(|b| b.len()).unwrap_or(0);
    let assembly_secs = t2.elapsed().as_secs_f64();
    Proved {
        proof,
        sums,
        cost: Cost {
            layers_secs: c.layers_secs,
            prove_secs,
            n_instances,
            fractions,
            unpaired,
            proof_bytes,
            arm,
            upload_secs,
            reshape_secs,
            prewarm_secs,
            assembly_secs,
            phases,
            // the caller adds what it measured around the whole call
            total_secs: c.layers_secs + prove_secs,
        },
        insts: c.insts,
    }
}

/// What proving the bands' lookups cost, apart from the proof itself.
#[derive(Default, Clone, Debug)]
pub struct Cost {
    /// building every instance's input layer
    pub layers_secs: f64,
    /// `prove_batch`: the circuit evaluation and every layer's sumcheck
    pub prove_secs: f64,
    pub n_instances: usize,
    /// paired fractions: the entries the input layers hold (B11 stream R)
    pub fractions: usize,
    /// one per lookup, before `paired` -- what stream Q's numbers counted
    pub unpaired: usize,
    /// the serialised `GkrBatchProof`
    pub proof_bytes: usize,
    /// where the sumcheck ran: "host simd" or "device"
    pub arm: &'static str,
    /// host-to-device transfer of the collected layers (device arm only). On a
    /// composed cuda arm the fractions are already resident, so this line is a
    /// stand-in and `reshape_secs` is the honest input cost.
    pub upload_secs: f64,
    /// one device-to-device pass over every layer: putting already resident
    /// fractions into MLE layout
    pub reshape_secs: f64,
    /// growing the device pool to the circuit's high-water mark, once per
    /// process in production (the pool persists across chunks)
    pub prewarm_secs: f64,
    /// serialising the proof
    pub assembly_secs: f64,
    /// the device sumcheck's phase split, when `B11_GKR_PROF=1`
    pub phases: Option<Vec<(&'static str, f64, u64)>>,
    /// everything the arm costs where the committed path costs its LogUp build:
    /// the input layers plus `prove_batch`
    pub total_secs: f64,
}

impl Cost {
    pub fn json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "instances": self.n_instances,
            "fractions": self.fractions,
            "unpaired": self.unpaired,
            "layers_secs": self.layers_secs,
            "prove_secs": self.prove_secs,
            "total_secs": self.total_secs,
            "proof_bytes": self.proof_bytes,
            "arm": self.arm,
            "upload_secs": self.upload_secs,
            "reshape_secs": self.reshape_secs,
            "prewarm_secs": self.prewarm_secs,
            "assembly_secs": self.assembly_secs,
        });
        if let Some(ph) = &self.phases {
            v["phases"] = ph
                .iter()
                .map(|(n, s, c)| serde_json::json!({"phase": n, "secs": s, "calls": c}))
                .collect();
        }
        v
    }
}

pub struct Proved {
    pub proof: GkrBatchProof,
    /// one claimed fractional sum per site, in site order -- what the committed
    /// running-sum columns claim today
    pub sums: Vec<SecureField>,
    pub insts: Vec<Inst>,
    pub cost: Cost,
}

/// Fold the batch's per-instance output claims into one claimed sum per site.
///
/// A LogUp circuit's output layer is the pair `[numerator, denominator]` of the
/// whole instance's fractional sum, so an instance contributes `p / q` and a
/// site is the sum of its instances. This is the only place the GKR's answer
/// crosses into the statement, and it is the same quantity
/// `interaction_phase` pushes onto `use_sums`.
fn sums_of(n_sites: usize, insts: &[Inst], claims: &[Vec<SecureField>]) -> Vec<SecureField> {
    let mut sums = vec![SecureField::zero(); n_sites];
    for (i, c) in insts.iter().zip(claims) {
        // an empty or malformed claim is a rejected proof, not a panic here
        if c.len() == 2 && !c[1].is_zero() {
            sums[i.site] += c[0] / c[1];
        }
    }
    sums
}

/// Verify a serialised batch GKR proof and return the claimed sum per site.
///
/// The instance list is derived from the plan by `plan`, never read from the
/// proof, so a prover cannot change the shape of what it proved. What comes back
/// is unchecked in one specific sense that the caller has to close: the
/// `GkrArtifact` carries claimed evaluations of the input layers at a random
/// point, and nothing here ties those to the committed base columns.
pub fn verify(
    channel: &mut impl Channel,
    sites: &[Site],
    bytes: &[u8],
) -> Result<(Vec<SecureField>, GkrArtifact, Vec<Inst>), String> {
    let proof: GkrBatchProof =
        serde_json::from_slice(bytes).map_err(|e| format!("unparseable GKR proof: {e}"))?;
    mix_shape(channel, sites);
    let insts = plan(sites);
    if proof.output_claims_by_instance.len() != insts.len() {
        return Err(format!(
            "the GKR proof carries {} instances, the plan says {}",
            proof.output_claims_by_instance.len(),
            insts.len()
        ));
    }
    // every instance's variable count is fixed by the plan
    let artifact = partially_verify_batch(gates(&insts), &proof, channel)
        .map_err(|e: GkrError| format!("{e}"))?;
    for (i, n) in insts.iter().zip(&artifact.n_variables_by_instance) {
        if i.n_variables() != *n {
            return Err(format!(
                "instance over {} variables, the plan says {}",
                n,
                i.n_variables()
            ));
        }
    }
    let sums = sums_of(sites.len(), &insts, &proof.output_claims_by_instance);
    Ok((sums, artifact, insts))
}

/// The claimed sum of one instance, read straight off its input layer.
///
/// This is the reference the GKR is checked against in the self-test: the plain
/// sum of `num / den` over every paired entry, by inversion, with no sumcheck
/// involved. It is `O(size)` and only for the test.
pub fn direct_sum<F>(inst: &Inst, frac: &F) -> SecureField
where
    F: Fn(usize, usize) -> (PackedSecureField, PackedSecureField) + Sync,
{
    let n_vec_row = 1usize << (inst.log_rows - LOG_N_LANES);
    let packed: PackedSecureField = (0..inst.ents.len() * n_vec_row)
        .into_par_iter()
        .map(|k| {
            let (b, vr) = (k / n_vec_row, k % n_vec_row);
            let (n, d) = frac(inst.ents[b], vr);
            n * d.inverse()
        })
        .reduce(|| PackedSecureField::zero(), |a, b| a + b);
    packed.to_array().iter().fold(SecureField::zero(), |a, b| a + *b)
}

/// A domain separator, so the GKR's transcript position cannot be confused with
/// anything else the channel carries.
pub fn mix_shape(channel: &mut impl Channel, sites: &[Site]) {
    let mut felts = vec![SecureField::from_u32_unchecked(0x51, 0x47, 0x4b, 0x52)];
    for s in sites {
        felts.push(SecureField::from_u32_unchecked(
            s.band as u32,
            s.which as u32,
            s.log_rows,
            s.n_looks() as u32,
        ));
    }
    channel.mix_felts(&felts);
}

#[allow(dead_code)]
const _: () = assert!(N_LANES == 1usize << LOG_N_LANES);

// ------------------------------------------------------------------ the test ---

/// `b65 gkr` -- the GKR arm's own self-test on a synthetic band.
///
/// What it establishes, in the order the record reports it:
///
/// 1. **The sums agree.** For every band region, the fractional sum the batched
///    GKR yields from its output claims equals, exactly, the claimed sum
///    `interaction_phase` pushes onto `use_sums` from the committed running-sum
///    columns. This is the strongest correctness statement available without the
///    `mle_eval` link: the two arms prove the same number about the same
///    fractions, bit for bit, and `sums_cancel` is what consumes that number.
/// 2. **A direct sum agrees too**, computed by plain inversion over every
///    fraction with no sumcheck in it, so a bug that moved the committed columns
///    and the GKR the same way would still be caught.
/// 3. **The proof verifies from bytes.** The `GkrBatchProof` is serialised, a
///    fresh channel is replayed from the same starting state, and
///    `partially_verify_batch` accepts.
/// 4. **A corrupted proof is rejected**, and the rejection is named.
/// 5. **What it cost**: instances, fractions, the input-layer build, the
///    sumcheck, and the proof bytes.
pub fn cmd(args: &[String]) {
    use crate::attn_witness::{Attn, Variant};
    use crate::layout::Shape;
    use stwo::core::channel::Blake2sM31Channel;

    // The first CUDA call in the process pays context creation and the fatbin
    // load; on the cuda build the sumcheck below must not be billed for it.
    crate::util::m4_touch_device();

    let src = crate::src_of(args);
    let sh = Shape {
        s: crate::arg_scalar(args, "--s", 64usize),
        c: crate::arg_scalar(args, "--ctx", 256usize),
        hd: crate::arg_scalar(args, "--hd", 64usize),
        insts: crate::arg_scalar(args, "--insts", 4usize),
    };
    let cfg = crate::attn_cfg(args, crate::arg_scalar(args, "--scores-per-row", 16usize));
    let a = Attn::get(&src, sh, Variant::IntExp);
    let l = crate::attn_trace::Layouts::choose(&sh, &cfg);
    let widths = crate::attn_trace::range_widths(&a, &cfg);
    let com = crate::attn_trace::commit_phase(&a, &l, &cfg);

    // The challenges have to be drawn the way the proof draws them, or the
    // relations are different relations and the sums are incomparable. This is
    // the order `attn_run::run` uses, without the commitment scheme: what the
    // fractions depend on is the drawn elements, not the Merkle roots.
    let mut ch = Blake2sM31Channel::default();
    crate::util::pcs_config().mix_into(&mut ch);
    let rel_exp = crate::attn_air::ExpRel::draw(&mut ch);
    let rels: Vec<crate::attn_air::RangeRel> =
        widths.iter().map(|_| crate::attn_air::RangeRel::draw(&mut ch)).collect();
    let f = ch.draw_secure_felts(3);
    let pow = crate::attn_air::Powers::new(f[0], f[1], f[2], &sh, &l.chan, &l.key, &l.out, l.rw);

    let mut collect = Collect::with_direct(true);
    collect.band = 0;
    let t0 = Instant::now();
    let inter = crate::attn_trace::interaction_phase(
        &a, &l, &cfg, &pow, &rel_exp, &rels, &com, &widths, true, !a.corner,
        None, Some(&mut collect));
    let committed_secs = t0.elapsed().as_secs_f64();

    let sites: Vec<Site> = collect.sites.clone();
    let direct = collect.direct.clone();
    assert_eq!(sites.len(), inter.use_sums.len(),
               "the collector saw {} band sites, `interaction_phase` pushed {} use sums",
               sites.len(), inter.use_sums.len());

    // The GKR runs on a channel replayed to the point the arm would reach it at,
    // and the verifier gets a channel in the same state.
    let mut pch = ch.clone();
    let proved = prove_collected(&mut pch, collect);
    let bytes = serde_json::to_vec(&proved.proof).expect("the GKR proof serialises");

    let mut vch = ch.clone();
    let verified = verify(&mut vch, &sites, &bytes);
    let vsums = match &verified {
        Ok((s, _, _)) => s.clone(),
        Err(_) => vec![],
    };
    let ood_len = match &verified {
        Ok((_, art, _)) => Some(art.ood_point.len()),
        Err(_) => None,
    };

    // the aimed corruption: one felt of the proof, which the sumcheck must catch
    let mut bad = proved.proof.clone();
    let tampered = tamper(&mut bad);
    let mut bch = ch.clone();
    let bad_bytes = serde_json::to_vec(&bad).unwrap();
    let bad_verdict = verify(&mut bch, &sites, &bad_bytes);
    let bad_rejected = bad_verdict.is_err();
    let bad_rejection = bad_verdict.err();

    let per_site: Vec<serde_json::Value> = sites
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let want = inter.use_sums[i];
            serde_json::json!({
                "site": s.tag(),
                "log_rows": s.log_rows,
                "looks": s.n_looks(),
                "unit_looks": s.unit.iter().filter(|u| **u).count(),
                "committed_sum": format!("{want:?}"),
                "gkr_sum": format!("{:?}", proved.sums[i]),
                "gkr_matches_committed": proved.sums[i] == want,
                "direct_matches_committed": direct[i] == want,
                "verified_sum_matches": vsums.get(i) == Some(&proved.sums[i]),
            })
        })
        .collect();

    let all_match = per_site.iter().all(|v| {
        v["gkr_matches_committed"] == true
            && v["direct_matches_committed"] == true
            && v["verified_sum_matches"] == true
    });
    let insts: Vec<serde_json::Value> = proved
        .insts
        .iter()
        .map(|i| serde_json::json!({
            "site": sites[i.site].tag(), "entries": i.ents.len(),
            "log_rows": i.log_rows, "n_variables": i.n_variables(),
            "unit": i.unit, "fractions": i.fractions(),
        }))
        .collect();

    crate::util::emit("gkr_selftest", serde_json::json!({
        "shape": sh.label(), "corner": a.corner, "variant": a.variant.name(),
        "scores_per_row": l.key.q, "keys_per_row": l.chan.pk, "rw": l.rw,
        "sites": per_site,
        "instances": insts,
        "sums_agree": all_match,
        "verifies_from_bytes": verified.is_ok(),
        "verify_error": verified.as_ref().err().cloned(),
        "ood_variables": ood_len,
        "control": {
            "what": tampered,
            "rejected": bad_rejected,
            "rejection": bad_rejection,
        },
        "cost": proved.cost.json(),
        "committed_interaction_secs": committed_secs,
    }));

    assert!(all_match, "the GKR arm's sums do not match the committed columns'");
    assert!(verified.is_ok(), "the GKR proof did not verify from bytes");
    assert!(bad_rejected, "a corrupted GKR proof was accepted");
}

/// Corrupt one felt of a batch GKR proof, and say what was corrupted.
///
/// The first round polynomial of the first layer's sumcheck is the earliest
/// thing the verifier checks, so a rejection there says the sumcheck is
/// load-bearing rather than that some later shape check tripped.
fn tamper(proof: &mut GkrBatchProof) -> String {
    if let Some(sc) = proof.sumcheck_proofs.first_mut() {
        if let Some(poly) = sc.round_polys.first_mut() {
            let mut c = poly.to_vec();
            if !c.is_empty() {
                c[0] += SecureField::one();
                *poly = stwo::prover::lookups::utils::UnivariatePoly::new(c);
                return "one coefficient of the first round polynomial, plus one".into();
            }
        }
    }
    if let Some(c) = proof.output_claims_by_instance.first_mut() {
        if let Some(v) = c.first_mut() {
            *v += SecureField::one();
            return "the first instance's output numerator claim, plus one".into();
        }
    }
    "nothing (the proof was empty)".into()
}

/// The band sites, derived from the plan alone.
///
/// This is the verifier's route to the instance list, and it has to agree with
/// what the prover's collector saw at the three `devfrac::band_cols` sites
/// exactly -- the same rows, the same lookup counts, the same numerator kinds,
/// in the same order. `Plan` is the object whose whole purpose is that the two
/// sides cannot disagree about layouts, so the derivation reads only from it.
///
/// The prover asserts this function against the collector's own sites before it
/// proves anything, so a drift between the two is a loud failure on the proving
/// side rather than a rejected proof on the verifying side.
pub fn sites_of(plan: &crate::compose::Plan) -> Vec<Site> {
    let mut out = vec![];
    for (b, corner) in plan.corner.iter().enumerate() {
        let l = &plan.lays[b];
        let (lc, lk) = (&l.chan, &l.key);
        let cfg = &plan.cfg;

        // ---- the channel region: the key and query range checks, all unit
        if cfg.range_check {
            out.push(Site {
                band: b,
                which: 0,
                log_rows: lc.g.log_rows,
                unit: vec![true; lc.pk + lc.pq],
            });
        }

        // ---- the grid region: one exp lookup per score slot, then the
        // variant's remainder checks, then the V range checks. On a corner band
        // the exp lookups carry `1 - mask` and are the only non-unit lookups in
        // the whole set.
        let mut unit = vec![!*corner; lk.q];
        if let AttnVariant::IntSplit(_) = plan.attn_variant {
            unit.extend(vec![true; lk.q]);
        }
        if cfg.range_check {
            unit.extend(vec![true; lk.pv]);
        }
        out.push(Site { band: b, which: 1, log_rows: lk.g.log_rows, unit });

        // ---- the per-token region: four denominator limbs per token. A corner
        // band has none -- the pair lives on its prefix band.
        if !*corner {
            out.push(Site {
                band: b,
                which: 3,
                log_rows: l.log_row,
                unit: vec![true; 4 * l.rw],
            });
        }
    }
    out
}

/// The two sides' site lists, compared field by field, with the first
/// disagreement named.
pub fn same_sites(a: &[Site], b: &[Site]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} sites against {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.band != y.band || x.which != y.which || x.log_rows != y.log_rows
            || x.unit != y.unit
        {
            return Err(format!(
                "site {i} differs: {} 2^{} {} looks ({} unit) against {} 2^{} {} looks ({} unit)",
                x.tag(), x.log_rows, x.n_looks(),
                x.unit.iter().filter(|u| **u).count(),
                y.tag(), y.log_rows, y.n_looks(),
                y.unit.iter().filter(|u| **u).count()));
        }
    }
    Ok(())
}
