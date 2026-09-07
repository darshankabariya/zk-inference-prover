// B7a milestone 4 (section 3 row 1): the shared tables' own columns.
//
// Three columns at 2^23 rows -- `x`, `y` and `m_exp` -- are 0.101 GB of the
// 0.166 GB sub-bar 4a had left, two thirds of everything. They are the last
// committed columns in the composed proof that are still born on the host, and
// none of them is a witness:
//
//  * `x` is the row index. A closed form of the committed row and nothing else.
//  * `y` is `Tables::exp`, which is `exp_int_table(rows, nmax, frac)`: the
//    argument `-d 2^-frac` is rounded to bf16 *before* `exp`, so the whole
//    table is a gather out of a 2^16-entry map of bf16 patterns to numerators.
//    The map is built on the host -- 262 kB, and `exp()` never runs on the
//    device, so there is no libm to agree with.
//  * `m_exp` is a counter every key tile's witness generation already scatter-adds into (21.1).
//    It came back to the host only to be merged and re-uploaded.
//
// **Rule 8 is the reason for the patch.** `Layer::corrupt_exp_table` is rung
// 3's probe: one row of the table flipped and the witness rebuilt to match, so
// that only the pinned preprocessed root gives it away. A kernel that computed
// the closed form and stopped there would silently repair it. So the host
// diffs its actual table against the closed form -- one parallel pass, and it
// is the *measured* path, not a check mode -- and uploads the rows that differ.
// On an honest witness the list is empty; on the probe it has one entry.
#ifndef ZK_TABLE_WITNESS_GEN_CUH
#define ZK_TABLE_WITNESS_GEN_CUH
#include "fields.cuh"

// `Variant`, as `zk_attention_freivalds_witness_gen.cuh` numbers it.
#define ZK_TABLE_VAR_BF16X 0u
#define ZK_TABLE_VAR_INTEXP 1u
#define ZK_TABLE_VAR_INTSPLIT 2u

// The negative controls (`ZK_MUT_TABLES`), each a way this design could
// plausibly be wrong. `nopatch` is the one aimed at rule 8, and it is vacuous
// on an honest witness by construction -- which is a finding to write down and
// not a pass (20.2's `ref32` wearing another hat), so it is run against the
// corrupted-table witness `--corrupt-exp` builds.
#define ZK_TABLE_MUT_NONE 0u
#define ZK_TABLE_MUT_NOPERM 1u       // write at the coset row, not the committed one
#define ZK_TABLE_MUT_NOSCALE 2u      // index the map with `d`, not with bf16(-d 2^-frac)
#define ZK_TABLE_MUT_BF16TRUNC 3u    // truncate to bf16 instead of round-to-nearest-even
#define ZK_TABLE_MUT_NOPATCH 4u      // skip the rows where this witness's table differs
#define ZK_TABLE_MUT_ONECOUNTER 5u   // sum one key tile's exp counter instead of every key tile's

// Which column a defect is aimed at. Unaimed, every one of them would show at
// `x`, which is compared first, and the run would say nothing about the other
// two (section 5 rule 5).
#define ZK_TABLE_COL_X 0u
#define ZK_TABLE_COL_Y 1u
#define ZK_TABLE_COL_M 2u
#define ZK_TABLE_COL_ALL 3u

struct ZkTabCfg {
    unsigned log_rows;    // `p.log_table`
    unsigned rows;        // `1 << log_rows`
    unsigned variant;
    unsigned n_counters;  // one per key tile whose witness generation ran on the device
    // `frac.exp2()` as the host computed it, passed rather than recomputed so
    // the division is the same IEEE division on both sides.
    double two_pow;
    unsigned mutate;
    unsigned mut_col;
};

// `x_col`, `y_col` and `m_col` are `1 << log_rows` cells each, in committed
// order. `expmap` is 2^16 numerators. `counters` are the device counter arrays
// the key tile witness generations bumped, each `1 << log_rows` wide.
extern "C" void zk_witness_gen_tables(const unsigned *expmap,
                                const unsigned *const *counters,
                                const unsigned *patch_idx, const unsigned *patch_val,
                                unsigned n_patch,
                                m31 *x_col, m31 *y_col, m31 *m_col,
                                ZkTabCfg cfg);

#endif
