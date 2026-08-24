//! The parallel LogUp column build — B6 §7's named remainder.
//!
//! `LogupTraceGenerator` builds one column at a time: per column it batch-inverts
//! the denominators (serial), multiplies and adds the previous column (parallel),
//! and only the frac *writes* could go wide.  B6 measured the LogUp build at
//! 38–47% of wall time after parallelising the Horner inner sums, and named this
//! as what keeps the commit+FRI+OODS share below the ≥85% bar.
//!
//! The dependency structure is friendlier than the generator's API: column `c` at
//! row `r` holds `sum_{c' <= c} frac_{c'}(r)` — a *pointwise* partial sum over the
//! column index — and only the last column is prefix-summed down the rows.  So:
//!
//!   phase A  every column's fracs `num * den^-1`, parallel over (column, row)
//!   phase B  the running pointwise sum, parallel over row chunks
//!   phase C  the last column's claimed sum, cumsum shift and prefix sum —
//!            exactly `finalize_last`, same operations in the same order
//!
//! The output is bit-identical to `LogupTraceGenerator` (same batch inversion,
//! same addition order, same shift and prefix sum), so every claimed sum and
//! every cell count elsewhere in the project stays comparable.

use num_traits::Zero;
use rayon::prelude::*;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{LOG_N_LANES, PackedM31};
use stwo::prover::backend::simd::prefix_sum::inclusive_prefix_sum;
use stwo::prover::backend::simd::qm31::{PackedQM31, batch_inverse_packed_qm31};

use crate::common::Eval;

/// Build `n_cols` LogUp columns in parallel.  `f(col, vec_row)` returns the
/// (numerator, denominator) of that column's fraction at that packed row —
/// for pairwise-batched lookups the caller passes the combined fraction
/// `(n0 d1 + n1 d0, d0 d1)`, exactly as it would feed `write_frac`.
///
/// Returns the columns (4 base columns each, in write order — the same
/// bit-reversed circle-domain order the committed trace is read in) and the
/// claimed sum, exactly as `LogupTraceGenerator::finalize_last` would.
pub fn build<F>(log_rows: u32, n_cols: usize, f: F) -> (Vec<Eval>, SecureField)
where
    F: Fn(usize, usize) -> (PackedQM31, PackedQM31) + Sync,
{
    assert!(n_cols > 0, "a LogUp trace needs at least one column");
    let n_vec = 1usize << (log_rows - LOG_N_LANES);

    // phase A: every column's fractions, parallel over columns; the batch
    // inversion inside a column is the same one the generator uses
    let fracs: Vec<Vec<PackedQM31>> = (0..n_cols)
        .into_par_iter()
        .map(|c| {
            let (nums, dens): (Vec<PackedQM31>, Vec<PackedQM31>) =
                (0..n_vec).into_par_iter().map(|vr| f(c, vr)).unzip();
            let mut inv = unsafe { stwo::core::utils::uninit_vec(n_vec) };
            batch_inverse_packed_qm31(&dens, &mut inv);
            nums.into_par_iter().zip(inv).map(|(n, d)| n * d).collect()
        })
        .collect();

    // phase B: pointwise running sums across columns, parallel over row chunks
    let mut cols: Vec<Vec<PackedQM31>> =
        (0..n_cols).map(|_| unsafe { stwo::core::utils::uninit_vec(n_vec) }).collect();
    {
        // split the destination columns so each row chunk owns a disjoint slice
        let mut slices: Vec<&mut [PackedQM31]> = cols.iter_mut().map(|c| c.as_mut_slice()).collect();
        const CHUNK: usize = 256;
        let mut chunked: Vec<Vec<&mut [PackedQM31]>> = vec![];
        while slices.first().map(|s| !s.is_empty()).unwrap_or(false) {
            let mut row = vec![];
            for s in slices.iter_mut() {
                let taken = std::mem::take(s);
                let (head, tail) = taken.split_at_mut(CHUNK.min(taken.len()));
                row.push(head);
                *s = tail;
            }
            chunked.push(row);
        }
        chunked.into_par_iter().enumerate().for_each(|(ci, mut dst)| {
            let base = ci * CHUNK;
            for i in 0..dst[0].len() {
                let mut run = PackedQM31::zero();
                for (c, d) in dst.iter_mut().enumerate() {
                    run += fracs[c][base + i];
                    d[i] = run;
                }
            }
        });
    }

    // phase C: the last column, exactly as `finalize_last` does it
    let last = cols.pop().unwrap();
    let mut coords: [Vec<PackedM31>; 4] = [vec![], vec![], vec![], vec![]];
    for (i, c) in coords.iter_mut().enumerate() {
        *c = last.par_iter().map(|v| v.into_packed_m31s()[i]).collect();
    }
    let coordinate_sums = coords
        .each_ref()
        .map(|c| c.iter().copied().sum::<PackedM31>().pointwise_sum());
    let claimed_sum = SecureField::from_m31_array(coordinate_sums);
    let cumsum_shift = claimed_sum / BaseField::from_u32_unchecked(1 << log_rows);
    let shift = PackedQM31::broadcast(cumsum_shift).into_packed_m31s();
    for (i, c) in coords.iter_mut().enumerate() {
        c.par_iter_mut().for_each(|x| *x -= shift[i]);
    }
    let domain = CanonicCoset::new(log_rows).circle_domain();
    let mut out: Vec<Eval> = Vec::with_capacity(4 * (cols.len() + 1));
    for col in &cols {
        let coord: Vec<Vec<PackedM31>> = (0..4usize)
            .into_par_iter()
            .map(|i| col.par_iter().map(|v| v.into_packed_m31s()[i]).collect())
            .collect();
        out.extend(coord.into_iter().map(|c| Eval::new(domain, BaseColumn::from_simd(c))));
    }
    out.extend(
        coords
            .into_iter()
            .map(|c| Eval::new(domain, inclusive_prefix_sum(BaseColumn::from_simd(c)))),
    );
    (out, claimed_sum)
}

/// The pairwise batching the constraint framework's `finalize_logup_in_pairs`
/// applies, as a column count and a per-column fraction: `n` lookups become
/// `ceil(n/2)` columns, the odd tail unpaired.
pub fn paired<G>(n_looks: usize, one: G) -> (usize, impl Fn(usize, usize) -> (PackedQM31, PackedQM31) + Sync)
where
    G: Fn(usize, usize) -> (PackedQM31, PackedQM31) + Sync,
{
    let n_cols = n_looks.div_ceil(2);
    let f = move |c: usize, vr: usize| {
        let i = 2 * c;
        let (n0, d0) = one(i, vr);
        if i + 1 < n_looks {
            let (n1, d1) = one(i + 1, vr);
            (n0 * d1 + n1 * d0, d0 * d1)
        } else {
            (n0, d0)
        }
    };
    (n_cols, f)
}
