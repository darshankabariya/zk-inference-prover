// B7a milestone 4 (section 3 row 1): 4a's last 79 columns.
//
// 24.4 left sub-bar 4a at **0.016 GB in 79 columns** and named both halves.
// Neither is a witness and neither is a lookup:
//
//  * **31 shared-table columns, 0.012 GB.** The normalisation, fraction,
//    multiply, `q != 0` and range rows and their ten multiplicities. 23.1 wrote
//    that they "are data, not closed forms, so moving them would trade an
//    upload for an upload". Measured column by column that is wrong twice over:
//    the ten multiplicities are counters every witness generation already scatter-adds into
//    (23.1's trick, nine more times), and `norm_table`, the fraction table and
//    `mul_table` are all closed forms of the *row index* -- integer bit surgery
//    and one exact float conversion, no `exp`, no libm, nothing uploaded.
//  * **48 key tile selectors, 0.004 GB.** `commit_pre`'s per-key-tile ACTIVE / NOFIRST
//    / LAST / ISTART predicates, which `zk_acc.cu` has computed since 24.1 --
//    at the wrong time, because `commit_pre` runs before the challenges are
//    drawn and the accumulators run after. `zk_flags_key_tile` is that computation
//    lifted into a kernel of its own; this file turns its array into columns.
//
// **What each of the three tables is, as a function of its row.**
//
// | table | rows | row `d` |
// |---|---|---|
// | `norm` | `256 + (kmax+1) 256 (1+sticky)` | `d = 0` is the zero row; `d < 256` the exactly-representable arm, `bf16(d)` with `ph = 1`; else `k = (d-256)/per_k`, `M2 = 256 + ...`, and one rounding step |
// | `frac` | `2^(kmax+1)` | `d = 0` is the zero row; else `k = floor(log2 d)` and `r = d - 2^k`, giving `(2^k, r, sticky && r != 0)` |
// | `mul` | `2^14` | `a = d >> 7`, `b = d & 127`, and the correctly-rounded product of the two significands |
// | `qnz`, `range` | `2^q`, `2^w` | the row index, and for `qnz` its non-zero indicator |
//
// The only float in any of them is `from_f32(M2 as f32)` for `M2 < 256`, and
// that conversion is exact: eight significand bits fit bf16's eight, so the low
// sixteen bits of the f32 are zero and rounding is the identity. There is no
// libm on the device here either -- 23.1's rule, kept.
//
// **Section 5 rule 8, and the probe that makes it bite here.** 23.1 needed a
// host diff for the exp table because rung 3's *sixth* probe flips one of its
// rows and rebuilds the witness to match, so a kernel that computed the closed
// form and stopped would repair the tamper silently. Rung 3's **seventh** probe
// is `ProverTamper::norm_row`, and it does exactly the same thing to the
// normalisation table -- one row of `tables.norm` flipped inside the prover.
// So the same answer, generalised: the host diffs `tables.norm`, `tables.frac`
// and `tables.mul` against the closed form and uploads the rows that differ.
// The pass is in the **measured** path, not a check mode; it is empty on an
// honest run and one row on the probe, and it is what makes the device column
// equal to the host's table by construction on any witness rather than by
// argument on this one. `qnz` and the ranges have no host table to diff: they
// are `0..2^w` generated inline, so their closed form *is* their definition.
//
// The multiplicities are the other side of the same coin: they are counted from
// the witness, and they are counted *by the witness generations*, so a probe that corrupts a
// witness is carried into the counter it bumps before it reaches this file.
#ifndef ZK_PRE_WITNESS_GEN_CUH
#define ZK_PRE_WITNESS_GEN_CUH
#include "fields.cuh"
#include "zk_acc.cuh"

// Which shared table a `zk_witness_gen_pre_table` call is building.
#define ZK_PRE_KIND_NORM 0u
#define ZK_PRE_KIND_FRAC 1u
#define ZK_PRE_KIND_MUL 2u
#define ZK_PRE_KIND_QNZ 3u
#define ZK_PRE_KIND_RANGE 4u

// `bf16::Round`.
#define ZK_PRE_ROUND_EVEN 0u
#define ZK_PRE_ROUND_AWAY 1u
#define ZK_PRE_ROUND_ZERO 2u

// `[128, 256)` scaled by `2^a` has biased exponent `a + EXP_BIAS` (`bf16.rs`).
#define ZK_PRE_EXP_BIAS 134u

// The negative controls (`ZK_MUT_PREPROCESSED`), each a way this design could
// plausibly be wrong. Section 5 rule 5: each takes a `:target` suffix, because
// unaimed every one of them shows at the first column compared and the run then
// says nothing about the other seven targets.
#define ZK_PRE_MUT_NONE 0u
#define ZK_PRE_MUT_NOPERM 1u      // write at the coset row, not the committed one
#define ZK_PRE_MUT_NOSTICKY 2u    // norm, frac: drop the sticky bit
#define ZK_PRE_MUT_NOROUND 3u     // norm, mul: truncate instead of round-to-nearest-even
#define ZK_PRE_MUT_NORENORM 4u    // norm, mul: skip the `m + up == 256` renormalisation
#define ZK_PRE_MUT_NOPAD 5u       // norm, frac: give a row past the table's end a value
#define ZK_PRE_MUT_NOKWIN 6u      // frac: `k = log2(d)`, off by one window
#define ZK_PRE_MUT_ONECOUNTER 7u  // sum one component's counter instead of every component's
#define ZK_PRE_MUT_NOLAST 8u      // chan, key: LAST from `chunk + 1 == chunks`
#define ZK_PRE_MUT_NOFIRSTROW 9u  // chan, key, row: drop a selector's `j > 0` guard
#define ZK_PRE_MUT_NOACTIVE 10u   // chan, key: an inert lead row reads as in-group
#define ZK_PRE_MUT_NOISTART 11u   // chan, key, row: the instance start is never marked
#define ZK_PRE_MUT_NOINV 12u      // key: the scale column carries `delta`, not `1/delta`
#define ZK_PRE_MUT_NOIDENT 13u    // key: an inert row's scale is `1/delta[0]`, not 1
#define ZK_PRE_MUT_NOPATCH 14u    // skip the rows where this run's table is not the closed form

// Which family a defect is aimed at. `norm`/`frac`/`mul`/`qnz`/`range` are the
// shared tables (each covering that table's own multiplicity column too), and
// `chan`/`key`/`row` are the key tile's three sub-components.
#define ZK_PRE_AT_NORM 0u
#define ZK_PRE_AT_FRAC 1u
#define ZK_PRE_AT_MUL 2u
#define ZK_PRE_AT_QNZ 3u
#define ZK_PRE_AT_RANGE 4u
#define ZK_PRE_AT_CHAN 5u
#define ZK_PRE_AT_KEY 6u
#define ZK_PRE_AT_ROW 7u
#define ZK_PRE_AT_ALL 8u

struct ZkPreCfg {
    unsigned kind;
    unsigned log_rows;   // the committed height, a power of two
    unsigned rows;       // `1 << log_rows`
    unsigned real_rows;  // the table's own length; rows beyond it commit zero
    unsigned kmax;
    unsigned round;
    unsigned sticky;     // `round.needs_sticky()`
    unsigned n_cols;     // the preprocessed columns this table has
    unsigned n_counters; // one per component whose witness generation bumped this table
    unsigned mutate;
    unsigned at;         // which family the defect is aimed at
};

struct ZkSelCfg {
    unsigned shape;      // `ZK_ACC_SHAPE_CHAN` / `_KEY` / `_TOK`
    ZkAccGrid g;        // the sub-component's grid; `rows`/`log_rows` only for tok
    unsigned ctx;        // `C`, for `split(g, C).0` -- the key component only
    unsigned n_insts;
    unsigned with_scale; // the key component's scale column: 0 for `bf16x`
    unsigned n_cols;     // 5, 8 or 3
    unsigned fam;        // `ZK_PRE_AT_CHAN` / `_KEY` / `_ROW`: this call's family
    unsigned mutate;
    unsigned at;         // the family the defect is aimed at
};

// One shared table's preprocessed columns and its multiplicity column.
//
// `cols` are `cfg.n_cols` columns of `cfg.rows` cells in committed order.
// `m_col` is the multiplicity column, or null when no witness generation's counter is on the
// device and the host keeps it. `counters` are the device counter arrays those
// witness generations bumped, each `cfg.rows` wide. `patch_idx` / `patch_val` are the rows
// where this run's own table is not the closed form -- `n_patch` rows of
// `cfg.n_cols` values each, row-major, and empty on an honest run.
extern "C" void zk_witness_gen_pre_table(const unsigned *const *counters,
                                   const unsigned *patch_idx, const unsigned *patch_val,
                                   unsigned n_patch, m31 *const *cols, m31 *m_col,
                                   ZkPreCfg cfg);

// One key tile sub-component's selector columns, out of the flags `zk_flags_key_tile`
// wrote for exactly those rows. `inv_delta` is the master exp table's key
// scale, inverted, per instance -- null unless the key component wants it.
extern "C" void zk_witness_gen_sel(const unsigned *flags, m31 *const *cols,
                             const m31 *inv_delta, ZkSelCfg cfg);

#endif
