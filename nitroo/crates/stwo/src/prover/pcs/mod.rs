use hashbrown::HashMap;
use itertools::Itertools;
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{info, span, Level};

use crate::core::channel::{Channel, MerkleChannel};
use crate::core::circle::CirclePoint;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{
    CommitmentSchemeProof, CommitmentSchemeProofAux, ExtendedCommitmentSchemeProof, PointSample,
};
use crate::core::pcs::utils::prepare_preprocessed_query_positions;
use crate::core::pcs::{PcsConfig, TreeSubspan, TreeVec};
use crate::core::poly::circle::CanonicCoset;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::vcs::MerkleHasher;
use crate::core::vcs::verifier::ExtendedMerkleDecommitment;
use crate::core::ColumnVec;
use crate::prover::air::component_prover::{Poly, Trace, WeightsHashMap};
use crate::prover::backend::{BackendForChannel, Col};
use crate::prover::fri::{FriDecommitResult, FriProver};
use crate::prover::pcs::quotient_ops::{compute_fri_quotients, compute_fri_quotients_by_group};
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::vcs::prover::MerkleProver;

pub mod quotient_ops;

/// The prover side of a FRI polynomial commitment scheme. See [super].
pub struct CommitmentSchemeProver<'a, B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub trees: TreeVec<CommitmentTreeProver<B, MC>>,
    pub config: PcsConfig,
    twiddles: &'a TwiddleTree<B>,
    pub store_polynomials_coefficients: bool,
    /// B12 stream HH: keep the coefficients only. A tree's blown-up
    /// evaluations are dropped once its root has been mixed into the channel,
    /// and every phase that needs them rebuilds them from the coefficients.
    pub lde_only: bool,
}
impl<'a, B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentSchemeProver<'a, B, MC> {
    /// Creates a new empty commitment scheme prover with the given configuration and twiddles. The
    /// commitment scheme does not store the polynomials coefficients by default.
    pub fn new(config: PcsConfig, twiddles: &'a TwiddleTree<B>) -> Self {
        CommitmentSchemeProver {
            trees: TreeVec::default(),
            config,
            twiddles,
            store_polynomials_coefficients: false,
            lde_only: false,
        }
    }

    /// Sets the commitment scheme to store the polynomials coefficients starting from the next
    /// commit.
    pub const fn set_store_polynomials_coefficients(&mut self) {
        self.store_polynomials_coefficients = true;
    }

    /// Sets the commitment scheme to drop each tree's blown-up evaluations
    /// after its root is mixed, starting from the next commit. Needs the
    /// coefficients, which are the only thing left to rebuild them from.
    pub const fn set_lde_only(&mut self) {
        assert!(
            self.store_polynomials_coefficients,
            "lde_only needs store_polynomials_coefficients"
        );
        self.lde_only = true;
    }

    fn commit(&mut self, polynomials: ColumnVec<CircleCoefficients<B>>, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        let tree = CommitmentTreeProver::new(
            polynomials,
            self.config.fri_config.log_blowup_factor,
            channel,
            self.twiddles,
            self.store_polynomials_coefficients,
            self.config.lifting_log_size,
            self.lde_only,
        );
        self.trees.push(tree);
    }

    /// Push a pre-built commitment tree (cloned from cache) and mix its root.
    pub fn push_cached_tree(&mut self, tree: CommitmentTreeProver<B, MC>, channel: &mut MC::C) {
        MC::mix_root(channel, tree.commitment.root());
        self.trees.push(tree);
    }

    pub fn tree_builder(&mut self) -> TreeBuilder<'_, 'a, B, MC> {
        TreeBuilder {
            tree_index: self.trees.len(),
            commitment_scheme: self,
            polys: Vec::default(),
        }
    }

    pub fn roots(&self) -> TreeVec<<MC::H as MerkleHasher>::Hash> {
        self.trees.as_ref().map(|tree| tree.commitment.root())
    }

    pub fn polynomials(&self) -> TreeVec<ColumnVec<&Poly<B>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().collect())
    }

    pub fn evaluations(
        &self,
    ) -> TreeVec<ColumnVec<&CircleEvaluation<B, BaseField, BitReversedOrder>>> {
        self.trees
            .as_ref()
            .map(|tree| tree.polynomials.iter().map(|poly| poly.evals()).collect())
    }

    pub fn trace(&self) -> Trace<'_, B> {
        let polys = self.polynomials();
        Trace { polys }
    }

    pub fn build_weights_hash_map(
        &self,
        sampled_points: &TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        max_log_size: u32,
    ) -> WeightsHashMap<B>
    where
        Col<B, SecureField>: Send + Sync,
    {
        let weights_dashmap = WeightsHashMap::<B>::new();

        self.polynomials()
            .zip_cols(sampled_points)
            .map_cols(|(poly, points)| {
                let compute_weights = |(log_size, point): (u32, CirclePoint<SecureField>)| {
                    weights_dashmap.entry((log_size, point)).or_insert_with(|| {
                        CircleEvaluation::<B, BaseField, BitReversedOrder>::barycentric_weights(
                            CanonicCoset::new(log_size),
                            point,
                        )
                    });
                };

                let log_size = poly.log_size();
                // For each sample point, compute the weights needed to evaluate the polynomial at
                // the folded sample point.
                // TODO(Leo): the computation `point.repeated_double(max_log_size - log_size)` is
                // likely repeated a bunch of times in a typical flat air. Consider moving it
                // outside the loop.
                #[cfg(not(feature = "parallel"))]
                points.iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });

                #[cfg(feature = "parallel")]
                points.par_iter().for_each(|&point| {
                    compute_weights((log_size, point.repeated_double(max_log_size - log_size)))
                });
            });

        weights_dashmap
    }

    pub fn prove_values(
        self,
        sampled_points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
        channel: &mut MC::C,
    ) -> ExtendedCommitmentSchemeProof<MC::H> {
        let span = span!(
            Level::INFO,
            "Evaluate columns out of domain",
            class = "EvaluateOutOfDomain"
        )
        .entered();

        let lifting_log_size = self.trees.last().unwrap().commitment.layers.len() as u32 - 1;
        let weights_hash_map = if self.store_polynomials_coefficients {
            None
        } else {
            Some(self.build_weights_hash_map(&sampled_points, lifting_log_size))
        };

        let samples: TreeVec<Vec<Vec<PointSample>>> = if self.store_polynomials_coefficients {
            // Batch OODS evaluation: group polynomials by coefficient log_size,
            // then batch-evaluate each group at each unique transformed point.
            // Reduces ~400 individual GPU kernel launches to ~15 batched calls.
            let polys = self.polynomials();

            // Initialize result structure with placeholder values.
            let mut samples = TreeVec::new(
                sampled_points
                    .iter()
                    .map(|tree_points| {
                        tree_points
                            .iter()
                            .map(|col_points| {
                                col_points
                                    .iter()
                                    .map(|&p| PointSample {
                                        point: p,
                                        value: SecureField::zero(),
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    })
                    .collect_vec(),
            );

            // Collect evaluation tasks grouped by (coeffs_log_size, transformed_point).
            let mut groups: HashMap<
                u32,
                Vec<(
                    CirclePoint<SecureField>,
                    Vec<(usize, usize, usize)>,
                    Vec<&CircleCoefficients<B>>,
                )>,
            > = HashMap::new();

            for (tree_idx, (tree_polys, tree_points)) in
                polys.iter().zip(sampled_points.iter()).enumerate()
            {
                for (col_idx, (poly, col_points)) in
                    tree_polys.iter().zip(tree_points.iter()).enumerate()
                {
                    let coeffs = poly.coeffs.as_ref().unwrap();
                    let log_size = coeffs.log_size();
                    let n_doubles = lifting_log_size - poly.log_size();

                    for (point_idx, &point) in col_points.iter().enumerate() {
                        let transformed = point.repeated_double(n_doubles);
                        let group = groups.entry(log_size).or_default();
                        if let Some(entry) =
                            group.iter_mut().find(|(p, _, _)| *p == transformed)
                        {
                            entry.1.push((tree_idx, col_idx, point_idx));
                            entry.2.push(coeffs);
                        } else {
                            group.push((
                                transformed,
                                vec![(tree_idx, col_idx, point_idx)],
                                vec![coeffs],
                            ));
                        }
                    }
                }
            }

            // Batch evaluate each group using B::batch_eval_at_point().
            for (_, point_groups) in groups {
                for (transformed_point, positions, coeffs_refs) in point_groups {
                    let results = B::batch_eval_at_point(&coeffs_refs, transformed_point);
                    for ((tree_idx, col_idx, point_idx), value) in
                        positions.into_iter().zip(results)
                    {
                        samples.0[tree_idx][col_idx][point_idx].value = value;
                    }
                }
            }

            samples
        } else {
            // Fallback: per-polynomial evaluation using barycentric weights.
            let eval_at_points =
                |(poly, points): (&Poly<B>, &Vec<CirclePoint<SecureField>>)| {
                    points
                        .iter()
                        .map(|&point| PointSample {
                            point,
                            value: poly.eval_at_point(
                                point.repeated_double(
                                    lifting_log_size - poly.log_size(),
                                ),
                                weights_hash_map.as_ref(),
                            ),
                        })
                        .collect_vec()
                };

            #[cfg(not(feature = "parallel"))]
            {
                self.polynomials()
                    .zip_cols(&sampled_points)
                    .map_cols(eval_at_points)
            }
            #[cfg(feature = "parallel")]
            {
                self.polynomials()
                    .zip_cols(&sampled_points)
                    .par_map_cols(eval_at_points)
            }
        };

        span.exit();

        let sampled_values = samples
            .as_cols_ref()
            .map_cols(|x| x.iter().map(|o| o.value).collect());
        channel.mix_felts(&sampled_values.clone().flatten_cols());

        // Compute oods quotients for boundary constraints on the sampled points.
        let random_coeff = channel.draw_secure_felt();
        let quotients = if self.lde_only {
            // B12 stream HH: the committed evaluations were dropped as each
            // root went into the channel. Rebuild them one log-size group at a
            // time, accumulate the group, and let it drop before the next
            // group is built. The grouping, the order of the groups and the
            // order inside a group are what they are below, so nothing about
            // the accumulation moves.
            let column_log_sizes = TreeVec::new(
                self.trees
                    .iter()
                    .map(|tree| tree.polynomials.iter().map(|p| p.log_size()).collect_vec())
                    .collect_vec(),
            );
            let flat_polys = self
                .trees
                .iter()
                .flat_map(|tree| tree.polynomials.iter())
                .collect_vec();
            let twiddles = self.twiddles;
            compute_fri_quotients_by_group(
                &column_log_sizes,
                &samples,
                random_coeff,
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
                |group, accumulate| {
                    let _span = span!(Level::INFO, "Quotient group extension").entered();
                    let rebuilt = group
                        .iter()
                        .map(|&i| flat_polys[i].rebuild_evals(twiddles))
                        .collect_vec();
                    accumulate(&rebuilt.iter().collect_vec());
                },
            )
        } else {
            let columns = self.evaluations();
            compute_fri_quotients(
                &columns,
                &samples,
                random_coeff,
                lifting_log_size,
                self.config.fri_config.log_blowup_factor,
            )
        };
        // Run FRI commitment phase on the oods quotients.
        let fri_prover =
            FriProver::<B, MC>::commit(channel, self.config.fri_config, &quotients, self.twiddles);

        // Proof of work.
        let span1 = span!(Level::INFO, "Grind", class = "Queries POW").entered();
        let proof_of_work = B::grind(channel, self.config.pow_bits);
        span1.exit();
        channel.mix_u64(proof_of_work);

        // FRI decommitment phase.
        let FriDecommitResult {
            fri_proof,
            query_positions,
            unsorted_query_locations,
        } = fri_prover.decommit(channel);
        // Build the query position tree.
        //
        // FRI samples its query positions in the lifted domain, which is as deep
        // as the deepest committed column anywhere in the proof.  Every tree then
        // needs those positions folded down to its *own* depth, because
        // `CommitmentTreeProver::decommit` measures a column's shift against the
        // tallest column of the tree it belongs to.  This used to be done for the
        // preprocessed tree alone, on the assumption that every other tree holds
        // at least one column at the full depth; a tree that does not -- an extra
        // tree committing one narrow, short family of columns, say -- was handed
        // positions from the larger domain and indexed off the end of its own
        // columns.  Folding per tree is the same computation for the preprocessed
        // tree and the identity for any tree that is already at full depth, so no
        // proof that verified before moves.
        let per_tree: Vec<Vec<usize>> = self
            .trees
            .iter()
            .map(|tree| {
                prepare_preprocessed_query_positions(
                    &query_positions,
                    lifting_log_size,
                    tree.commitment.layers.len() as u32 - 1,
                )
            })
            .collect();
        let query_positions_tree =
            TreeVec::new(per_tree.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
        let (queried_values, decommitments, aux): (Vec<_>, Vec<_>, Vec<_>) = self
            .trees
            .as_ref()
            .zip_eq(query_positions_tree)
            .map(|(tree, query_positions)| tree.decommit(query_positions, self.twiddles))
            .0
            .into_iter()
            .map(|(v, x)| (v, x.decommitment, x.aux))
            .multiunzip();
        ExtendedCommitmentSchemeProof {
            proof: CommitmentSchemeProof {
                commitments: self.roots(),
                sampled_values,
                decommitments: TreeVec(decommitments),
                queried_values: TreeVec(queried_values),
                proof_of_work,
                fri_proof: fri_proof.proof,
                config: self.config,
            },
            aux: CommitmentSchemeProofAux {
                unsorted_query_locations,
                trace_decommitment: TreeVec(aux),
                fri: fri_proof.aux,
            },
        }
    }
}

/// Helper struct for aggregating polynomials and evaluations for a commitment tree.
pub struct TreeBuilder<'a, 'b, B: BackendForChannel<MC>, MC: MerkleChannel> {
    tree_index: usize,
    commitment_scheme: &'a mut CommitmentSchemeProver<'b, B, MC>,
    polys: ColumnVec<CircleCoefficients<B>>,
}
impl<B: BackendForChannel<MC>, MC: MerkleChannel> TreeBuilder<'_, '_, B, MC> {
    pub fn extend_evals(
        &mut self,
        columns: Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>,
    ) -> TreeSubspan {
        let span = span!(Level::INFO, "Interpolation for commitment").entered();
        let polys = B::interpolate_columns(columns, self.commitment_scheme.twiddles);
        span.exit();

        self.extend_polys(polys)
    }

    pub fn extend_polys(
        &mut self,
        columns: impl IntoIterator<Item = CircleCoefficients<B>>,
    ) -> TreeSubspan {
        let col_start = self.polys.len();
        self.polys.extend(columns);
        let col_end = self.polys.len();
        TreeSubspan {
            tree_index: self.tree_index,
            col_start,
            col_end,
        }
    }

    pub fn commit(self, channel: &mut MC::C) {
        let _span = span!(Level::INFO, "Commitment").entered();
        self.commitment_scheme.commit(self.polys, channel);
    }
}

/// Prover data for a single commitment tree in a commitment scheme. The commitment scheme allows to
/// commit on a set of polynomials at a time. This corresponds to such a set.
pub struct CommitmentTreeProver<B: BackendForChannel<MC>, MC: MerkleChannel> {
    pub polynomials: ColumnVec<Poly<B>>,
    pub commitment: MerkleProver<B, MC::H>,
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> Clone for CommitmentTreeProver<B, MC>
where
    Poly<B>: Clone,
    MerkleProver<B, MC::H>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            polynomials: self.polynomials.clone(),
            commitment: self.commitment.clone(),
        }
    }
}

impl<B: BackendForChannel<MC>, MC: MerkleChannel> CommitmentTreeProver<B, MC> {
    pub fn new(
        polynomials: ColumnVec<CircleCoefficients<B>>,
        log_blowup_factor: u32,
        channel: &mut MC::C,
        twiddles: &TwiddleTree<B>,
        store_polynomials_coefficients: bool,
        lifting_log_size: Option<u32>,
        lde_only: bool,
    ) -> Self {
        let span = span!(Level::INFO, "Extension").entered();
        let mut polynomials = B::evaluate_polynomials(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
        );
        span.exit();

        let max_log_domain_size = polynomials
            .iter()
            .map(|poly| poly.log_size())
            .max()
            .unwrap_or_default();
        let _span = span!(Level::INFO, "Merkle",
            n_polys = polynomials.len(),
            max_log_domain_size = max_log_domain_size,
        ).entered();
        let tree = MerkleProver::commit(
            polynomials
                .iter()
                .map(|poly: &Poly<B>| &poly.evals().values)
                .collect(),
        );
        MC::mix_root(channel, tree.root());

        // B12 stream HH: the root is in the channel and the Merkle layers are
        // built, so nothing reads these cells again until the composition. Drop
        // them and keep the coefficients; the composition rebuilds per
        // component, the quotients per log-size group, the decommitment per
        // tree.
        if lde_only {
            let _span = span!(
                Level::INFO,
                "Drop extension",
                n_polys = polynomials.len(),
            )
            .entered();
            for poly in &mut polynomials {
                poly.drop_evals();
            }
        }

        CommitmentTreeProver {
            polynomials,
            commitment: tree,
        }
    }

    /// Decommits the merkle tree on the given query positions.
    /// Returns the flat values at the queried positions and the decommitment.
    fn decommit(
        &self,
        queries: &[usize],
        twiddles: &TwiddleTree<B>,
    ) -> (
        Vec<BaseField>,
        ExtendedMerkleDecommitment<MC::H>,
    ) {
        // Build queries_per_log_size from flat positions.
        let column_log_sizes: BTreeSet<u32> = self
            .polynomials
            .iter()
            .map(|p| p.log_size())
            .collect();
        let max_log_size = self.commitment.layers.len() as u32 - 1;
        let mut queries_per_log_size = BTreeMap::new();
        for &log_size in &column_log_sizes {
            let shift = max_log_size - log_size;
            let positions: Vec<usize> = queries
                .iter()
                .map(|&pos| (pos >> (shift + 1) << 1) + (pos & 1))
                .sorted()
                .dedup()
                .collect();
            queries_per_log_size.insert(log_size, positions);
        }

        // B12 stream HH: under `lde_only` this tree's evaluations went at
        // commit time. Rebuild the whole tree here -- the Merkle proof takes
        // whole columns, and the queried rows cannot be produced from the
        // coefficients on their own -- decommit, and let the rebuild drop
        // before the next tree is asked for. This phase runs after FRI, when
        // nothing else large is live, so one tree here does not raise the peak.
        let rebuilt: Option<Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>> =
            if self.polynomials.iter().any(|p| p.evals.is_none()) {
                let _span = span!(Level::INFO, "Decommit extension").entered();
                Some(
                    self.polynomials
                        .iter()
                        .map(|poly| poly.rebuild_evals(twiddles))
                        .collect_vec(),
                )
            } else {
                None
            };
        let eval_vec = match &rebuilt {
            Some(evals) => evals.iter().map(|e| &e.values).collect_vec(),
            None => self
                .polynomials
                .iter()
                .map(|poly| &poly.evals().values)
                .collect_vec(),
        };
        self.commitment.decommit(&queries_per_log_size, eval_vec)
    }
}
