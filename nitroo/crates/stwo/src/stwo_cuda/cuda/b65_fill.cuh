// B7a milestone 4 (19.6): the dequant witness fill, on the device.
//
// 18.7's row 1 -- "the four family fills, the only row with no shortcut". This
// is the first of the four, and it is one kernel per variant because the
// per-row work is `fill_mat_row` verbatim: an integer normalisation, two table
// reads that are closed forms, and the variant's scale arithmetic.
//
// Three things make it the whole of the fill rather than a piece of it:
//
//  * **the permutation is free.** The thread owns the *output* index `i` and
//    computes the coset row `j = m4_perm(i, log_rows)`, so the column is born
//    in committed order and `eval_of`'s shuffle and repack (1.19 s on the
//    host, 18.2) never happen.
//  * **the table lookups are closed forms.** `Tables::norm_at` and `frac_at`
//    are arithmetic on the host already; they port as written.
//  * **the multiplicities are a scatter-add.** Counting is commutative, so the
//    order the threads bump in changes nothing, and the counters come back to
//    the host as a few hundred kB -- which 4a permits: it forbids H2D of trace
//    columns, not D2H of a counter.
//
// The one part that is not integer arithmetic is `ref_bad`, `fill_mat_row`'s
// f64 cross-check of the table path against a single IEEE rounding. It is
// deterministic on both sides -- `exp2`/`ldexp` of an integer is exact, bf16
// rounding is bit manipulation, and the products are plain `double` multiplies
// with no add to contract into an FMA -- so it ports, and `B7_M4_FILL=2`
// compares the f64 result cell for cell rather than comparing the counts,
// which on an honest witness are both zero and would agree vacuously.
#ifndef B65_FILL_CUH
#define B65_FILL_CUH
#include "fields.cuh"

// Everything about the region's shape that `Map`, `MatLayout` and `Spec` fix.
// Mirrored by `#[repr(C)] struct FillCfg` in `devfill.rs`; the column indices
// are passed rather than recomputed so the kernel and `Map::at` agree by
// construction, exactly as the host builders do.
struct B65FillCfg {
    unsigned variant;  // 0 = pow2, 1 = chain
    unsigned sticky;   // Round::needs_sticky()
    unsigned kmax;
    unsigned do_ref;   // run the f64 cross-check (Round::Even and B7_M4_REFBAD)
    unsigned tokens;
    unsigned n_out;
    unsigned per_row;
    unsigned chunks;
    unsigned log_rows;
    unsigned slot0;    // Map::slot0
    unsigned per_slot; // Map::per_slot
    unsigned chan_w;   // preprocessed scale columns per slot: 1 pow2, 2 chain
    // Fields, as `Map::new` assigns them. `B65_FILL_NONE` = this variant has
    // no such field.
    unsigned f_acc, f_sg, f_m2, f_ph, f_st, f_r, f_m7, f_ef, f_nz, f_eo;
    unsigned f_mul[6]; // chain: [t][0..3] = (product mantissa, carry, exponent)
    unsigned sh0, sh1; // Map::shared
    unsigned n_norm;   // rows in the normalisation table (bounds the bump)
    unsigned n_frac;
    unsigned n_mul;
    unsigned n_r8;     // rows in the 8-bit range table
    unsigned mutate;   // B7_M4_FILLMUT: a deliberate defect, for the controls
};

#define B65_FILL_NONE 0xffffffffu

// The negative controls (`B7_M4_FILLMUT`). A check is only worth reporting if
// it fails when the fill is wrong in the ways this design could plausibly be
// wrong, and each of these is one of them: the permutation forgotten, a slot
// misaddressed, a multiplicity dropped, and two ways of getting the f64
// cross-check wrong -- through f32, which is the double rounding `bf16.rs`
// warns about in as many words, and with ties away from zero, which is the
// other rounding mode the same crate supports. `none` is the control's control
// and must change nothing.
#define B65_MUT_NONE 0u
#define B65_MUT_NOPERM 1u
#define B65_MUT_CHAN 2u
#define B65_MUT_NOBUMP 3u
#define B65_MUT_REF32 4u
#define B65_MUT_REFAWAY 5u

// One matmul dequant region: every committed and preprocessed column, in
// committed order, plus this region's table multiplicities.
//
// `acc` is the region's own witness -- `int32_t` because `kmax <= 21` bounds
// every magnitude the normalisation table can describe by `2^30`, which the
// caller asserts before it uploads. `norm3` is `(m7, expfield, nz)` per row and
// `mul2` is `(m7, carry)` per row, both shared across regions.
//
// `want_out`, when non-null, receives the f64 cross-check's bf16 pattern per
// (natural row, slot) -- the check's, never the prover's.
extern "C" void b65_fill_dequant(const int *acc,
                                 const int *jx, const int *jw,
                                 const unsigned *sx, const unsigned *sw,
                                 const unsigned *norm3, const unsigned *mul2,
                                 m31 *const *base_cols, unsigned n_base,
                                 m31 *const *pre_cols, unsigned n_pre,
                                 unsigned *m_norm, unsigned *m_frac,
                                 unsigned *m_mul, unsigned *m_r8,
                                 unsigned *counts, unsigned *want_out,
                                 B65FillCfg cfg);

#endif
