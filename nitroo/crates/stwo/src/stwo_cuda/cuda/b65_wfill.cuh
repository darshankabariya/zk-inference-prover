// B7a milestone 4 (section 3 row 1): the Freivalds check, attnout and key tile fills, on the
// device.
//
// 20.6's row 1 -- "the other three fills, 0.975 s, and none of them needs a new
// idea". The idea is 20.1's, three more times: the thread owns the *committed*
// row, so `eval_of` never runs; the layout maps are closed forms and port as
// written; the multiplicities are a warp-aggregated scatter-add whose counters
// come back as a D2H, which 4a permits.
//
// What is different from `b65_fill.cu`, and why this is its own file:
//
//  * **a region is several grids, not one.** The dequant region is one
//    `MatLayout`; a key tile is four sub-regions on three different `Grid`s plus a
//    per-token block that is not a `Grid` at all. So the unit here is the
//    sub-region and there are seven kernels, not one templated pair.
//  * **the counters are an array of arrays.** `commit_base` threads `mult_exp`
//    and `mult_r` through by `&mut` across every key tile, and `mult_r` is one
//    vector per range width. The kernels take the counter arrays they bump with
//    their lengths and one shared off-table counter, which is what makes the
//    key tiles' "awkward" part (20.6) ordinary.
//  * **nothing here is recomputed that the probes can corrupt away.** The grid
//    region's numerator column is `num[g]`, not `table[key(g)]`, even though an
//    honest witness makes those equal: `Corrupt::Numerator` replaces one
//    numerator with a different *legal* table output, and a kernel that
//    recomputed it from the table would silently repair the probe and turn a
//    rung-3 rejection into an acceptance.
#ifndef B65_WFILL_CUH
#define B65_WFILL_CUH
#include "fields.cuh"

#define B65_WFILL_NONE 0xffffffffu

// The negative controls (`B7_M4_WFILLMUT`). Each is a way one of these three
// fills could plausibly be wrong, and 19.6's caution is the reason `inst` is on
// the list: a control that cannot fire on the shape under test is a finding and
// not a pass, and the key tile's instance split is invisible on a one-instance
// key tile.
#define B65W_MUT_NONE 0u
#define B65W_MUT_NOPERM 1u   // write at the coset row, not the committed one
#define B65W_MUT_CHAN 2u     // the slot-to-channel map, off by one
#define B65W_MUT_NOBUMP 3u   // drop one multiplicity per 1,024 rows
#define B65W_MUT_DESCEND 4u  // the chunk order, not descending
#define B65W_MUT_INST 5u     // the (instance, local) split of a packed group
#define B65W_MUT_AOREF 6u    // attnout's f64 reference, ties to even
// B11 stream N, lever 2: the diagonal block key tile's own way of being wrong. A masked
// slot is not a lookup, so it bumps nothing; ignoring the mask bumps every
// slot and the multiplicity delta parts company with the host's. This is the
// defect the lever was opened on, made deliberate.
#define B65W_MUT_NOMASK 7u   // let a masked diagonal block slot bump the exp table

// `Grid` from `layout.rs`, field for field. `rows` is `1 << log_rows`.
struct B65Grid {
    unsigned log_rows;
    unsigned rows;
    unsigned groups;
    unsigned chunks;
    unsigned lead;
    unsigned descending;
};

// One counter array, as `Tables` holds it: `m_exp`, one `m_r[t]` per range
// width, or `m_qnz`. A null `m` means this fill does not bump that table at all
// (`range_check` off, or a variant with no auxiliary column).
struct B65Mult {
    unsigned *m;
    unsigned n;
};

// ------------------------------------------------------- Freivalds check --

// `freivalds_region`: activations committed with the +128 offset, weights
// preprocessed, one 8-bit range bump per committed cell.
struct B65FreivaldsCfg {
    B65Grid g;
    unsigned pq;       // activation slots per row
    unsigned pk;       // weight slots per row, per matmul
    unsigned n_in;
    unsigned tokens;
    unsigned n_mats;
    unsigned mutate;
};

extern "C" void b65_fill_freivalds(const signed char *qx,
                                   const signed char *const *qw, const unsigned *n_out,
                                   m31 *const *base_cols, unsigned n_base,
                                   m31 *const *pre_cols, unsigned n_pre,
                                   B65Mult m_r8, unsigned *counts,
                                   B65FreivaldsCfg cfg);

// --------------------------------------------------------------- attnout --

// `ao_region`: the rounding fold, its two range bumps and its `qm`
// multiplicity, plus the f64 reference the host computes alongside.
struct B65AoCfg {
    unsigned nh, s, hd;
    unsigned per_row;
    unsigned chunks;
    unsigned log_rows;
    unsigned lead;
    unsigned do_ref;
    unsigned mutate;
    // stream JJ (E11): the exponent window's two offsets and the accumulator
    // window's two limb widths
    int jv_min;
    int jo_max;
    unsigned acc_lo_bits;
    unsigned acc_hi_bits;
};

extern "C" void b65_fill_attnout(const unsigned *acc, const unsigned *den,
                                 const int *jo, const int *jv,
                                 m31 *const *base_cols, unsigned n_base,
                                 m31 *const *pre_cols, unsigned n_pre,
                                 B65Mult m_qnz, B65Mult m_rem,
                                 B65Mult m_acc_lo, B65Mult m_acc_hi,
                                 int *q_out, unsigned *want_out,
                                 unsigned *counts, B65AoCfg cfg);

// ----------------------------------------------------------------- key tiles --

// `attn_trace::commit_base`, whose four sub-regions are four kernels sharing
// one config. `variant` is `Variant`: 0 bf16x, 1 intexp, 2 intsplit(k).
#define B65W_VAR_BF16X 0u
#define B65W_VAR_INTEXP 1u
#define B65W_VAR_INTSPLIT 2u

struct B65KeyTileCfg {
    B65Grid gc, gk, go;   // LayoutChan, LayoutKey, LayoutOut
    unsigned pk, pq;      // LayoutChan: keys and queries per row
    unsigned kq, pv;      // LayoutKey: scores per row, values per row
    unsigned oq;          // LayoutOut: outputs per row
    unsigned log_row;     // the per-token region
    unsigned rows_tok;
    unsigned rw;
    unsigned s, c, hd, insts, toks;
    unsigned variant;
    unsigned split_k;     // IntSplit's k
    int off_k, off_q;
    unsigned range_check;
    unsigned z_lo_bits, z_hi_bits;
    long long z_min;
    // B11 stream N, lever 2: is this the diagonal block key tile? The grid kernel
    // needs it for one thing only -- a masked slot is not a lookup at all, so
    // it bumps no exp multiplicity (`Attn::visible`). The committed columns are
    // the same either way.
    unsigned diagonal_block;
    unsigned mutate;
};

// `counts[0]` is the off-table total the host adds to `Tables::off_table`.
extern "C" void b65_fill_key_tile_chan(const int *kk, const int *qq,
                                   m31 *const *cols,
                                   B65Mult m_k, B65Mult m_q,
                                   unsigned *counts, B65KeyTileCfg cfg);

extern "C" void b65_fill_key_tile_grid(const int *dd, const unsigned short *num,
                                   const unsigned short *xx, const unsigned *sc,
                                   const unsigned *delta, const int *vv,
                                   long long table0,
                                   m31 *const *cols, unsigned n_aux,
                                   B65Mult m_exp, B65Mult m_aux, B65Mult m_v,
                                   unsigned *counts, B65KeyTileCfg cfg);

extern "C" void b65_fill_key_tile_out(const unsigned *out, m31 *const *cols,
                                  unsigned *counts, B65KeyTileCfg cfg);

extern "C" void b65_fill_key_tile_tok(const unsigned *mm, const unsigned *zz,
                                  m31 *const *cols, unsigned with_max,
                                  B65Mult m_lo, B65Mult m_hi,
                                  unsigned *counts, B65KeyTileCfg cfg);

#endif
