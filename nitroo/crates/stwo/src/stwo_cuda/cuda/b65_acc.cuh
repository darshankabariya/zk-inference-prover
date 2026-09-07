// B7a milestone 4 (section 3 row 1): the Horner accumulators, on the device.
//
// 23.4 left 4a at 0.066 GB in 351 columns and named the three things in it.
// **272 of those columns are the interaction's accumulators** -- 0.050 GB, three
// quarters of everything -- and they are the last committed columns in the
// composed proof that are still born on the host. They are not lookups: each
// one is a first-order linear recurrence down the rows of a region,
//
//     x[j] = a[j] * x[j-1] + b[j],      x[-1] = 0,
//
// with `b[j]` an inner Horner sum over that row's own committed columns and
// `a[j]` a challenge power the row's position in the layout picks. Every
// accumulator in the proof has that shape:
//
// | region | stage 1 (`a` resets at a group start) | stage 2 (`b` lands on a group's last row) |
// |---|---|---|
// | band channel | `u` (r), `v` (s) | `acc` = sum of `u v`, folded by `r^C` per instance |
// | band grid | `zs`, `zn` (s), `wv` (rv) | `hs` (Horner in r), `av` = sum `wv zn`, `sn` = sum `zn` |
// | band per-token | `am`, `az` (s, `mu` at an instance start) | -- |
// | Freivalds group | `v` (s), `us[m]` (rv) | `accs[m]` = sum of `us[m] v` |
// | dequant | `z` (rv) | `h` (Horner in s) |
// | attnout | `z` (rv), `den` (raw) | `h`, `dz` (Horner in s, `mu` at a head start) |
//
// **Why one primitive covers all six.** The composition of two affine maps is
// affine -- `(a2, b2) . (a1, b1) = (a1 a2, a2 b1 + b2)` -- and it is associative,
// so a parallel scan over the pairs gives *exactly* the value the host's serial
// loop gives. M31 is a field: reassociating is not an approximation, it is the
// same element. A reset is `a[j] = 0`, so a segmented recurrence is not a
// special case either, and a raw (unscanned) lane is `a = 0` for every row.
// That leaves one scan kernel, one splitter and one stage-2 builder shared by
// every region, and six small kernels that do nothing but read the region's own
// committed columns and write `(a, b)`.
//
// **What is read, and why it is the columns and not the witness.** The host
// builds `b[j]` from the witness (`a.kk`, `m.acc`, `m0.qx`, ...); the kernels
// build it from the columns those witnesses were filled into by 20-22's fills,
// which are already on the device in committed order and already checked cell
// for cell. That is the AIR's own route -- the constraint reads the same
// columns -- and it means no witness is uploaded twice and no fill is
// recomputed here, so section 5 rule 8 has nothing to bite on: a probe that
// corrupts a witness corrupts the column, and the accumulator carries it.
//
// The committed column is not always the witness value, and the four places it
// is not are the four the controls are aimed at: the channel region commits
// `kk + off_k` (`nooffset`), the grid commits the shift *scaled* to the master
// exp table's exponent (`nodelta`) and, at `intsplit`'s design point, *split*
// across the key and auxiliary columns (`nosplit`), and the grid's numerator
// column commits `table[0]` on an inert lead row where the host's sum has no
// term at all (`nomask`).
#ifndef B65_ACC_CUH
#define B65_ACC_CUH
#include "fields.cuh"

// The negative controls (`B7_M4_ACCMUT`), each a way this design could
// plausibly be wrong. Section 5 rule 5: each takes a `:region` suffix, because
// unaimed every one of them panics at `chan band0`, which is checked first.
#define B65A_MUT_NONE 0u
#define B65A_MUT_NOPERM 1u     // read the coset row, not the committed one
#define B65A_MUT_NOCARRY 2u    // drop the scan's inter-block carry
#define B65A_MUT_NORESET 3u    // never reset a stage-1 lane at a group start
#define B65A_MUT_NOISTART 4u   // ignore the instance-start multiplier
#define B65A_MUT_NOOFFSET 5u   // do not subtract the column's committed offset
#define B65A_MUT_NODELTA 6u    // do not scale the grid's key column back down
#define B65A_MUT_NOMASK 7u     // let an inert lead row contribute a numerator
#define B65A_MUT_EVERYROW 8u   // land stage 2's term on every row, not the last
#define B65A_MUT_NOSPLIT 9u    // read the key column alone, not `idx 2^k + aux`
// B11 stream N: the residual stream's bindings, whose three ways of being wrong are its three
// factors -- the row-to-row step, the within-row weight, and `lam`, which is
// the only thing that keeps a leaf's two halves apart.
#define B65A_MUT_NOBINDSTEP 10u  // fold the rows with 1 instead of the binding's step
#define B65A_MUT_NOTIEW 11u     // weigh every member of a row's sequence alike
#define B65A_MUT_NOTIELAM 12u   // drop `ef`'s `lam`, collapsing (sm, ef) to sm

// Per-row layout facts, computed once by the region's own kernel so that the
// stage-2 builder and the scan need to know nothing about the layout.
#define B65A_ACTIVE 1u
#define B65A_NOFIRST 2u
#define B65A_LAST 4u
// The row where this lane's Horner picks up its per-instance weight. Each
// region defines it: an instance start for the bands and the Freivalds checks, a head
// start for the attnout, never for the dequant.
#define B65A_ISTART 8u

// `src1` of a stage-2 lane that has only one factor.
#define B65A_ONE 0xffffffffu

// `Grid`, as `b65_wfill.cuh` spells it.
struct B65AccGrid {
    unsigned log_rows;
    unsigned rows;
    unsigned groups;
    unsigned chunks;
    unsigned lead;
    unsigned descending;
};

// ---------------------------------------------------- the shared layout half --
//
// B7a milestone 4 (section 3 row 1, section 25): these four were `static` inside
// `b65_acc.cu` until the band selectors needed them. They are the *only* thing
// `commit_pre`'s 48 preprocessed columns say -- ACTIVE, NOFIRST, LAST and
// ISTART per coset row -- so they live in the header and `b65_pre_fill.cu`
// builds the columns from the same three lines the accumulators segment on.

// `common.rs`'s `m4_perm`: the closed form of
// `bit_reverse_coset_to_circle_domain_order`. An involution, so the thread that
// owns committed row `i` reads coset row `f(i)` and the coset row `j` a scan
// walks is held at committed index `f(j)`.
__device__ __forceinline__ unsigned b65a_perm(unsigned i, unsigned log_n) {
    const unsigned n = 1u << log_n;
    const unsigned c = (i & 1u) ? ((2u * n - i - 1u) >> 1) : (i >> 1);
    return __brev(c) >> (32u - log_n);
}

// `Grid::row`, as `b65_wfill.cu` has it.
__device__ __forceinline__ bool b65a_grid_row(const B65AccGrid &g, unsigned j,
                                              unsigned *grp, unsigned *chunk) {
    if (j < g.lead) {
        return false;
    }
    const unsigned idx = j - g.lead;
    unsigned gi = idx / g.chunks;
    if (g.descending != 0u) {
        gi = g.groups - 1u - gi;
    }
    *grp = gi;
    *chunk = g.chunks - 1u - (idx % g.chunks);
    return true;
}

// The three flags every grid-shaped region shares.
__device__ __forceinline__ unsigned b65a_grid_flags(bool act, unsigned chunk,
                                                    unsigned chunks) {
    unsigned f = 0u;
    if (!act) {
        return f;
    }
    f |= B65A_ACTIVE;
    if (chunk + 1u < chunks) {
        f |= B65A_NOFIRST;
    }
    if (chunk == 0u) {
        f |= B65A_LAST;
    }
    return f;
}

// `chan_istart` / `key_istart`: the first row of an instance in a grid-shaped
// region, where the outer sum picks up its per-instance weight. `modulus` is
// `hd` for the channel region and `C` for the key region -- `attn_trace.rs`
// writes them as one predicate each and this is that predicate.
__device__ __forceinline__ unsigned b65a_flags_of_grid(const B65AccGrid &g,
                                                       unsigned modulus, unsigned j) {
    unsigned grp = 0u, chunk = 0u;
    const bool act = b65a_grid_row(g, j, &grp, &chunk);
    unsigned f = b65a_grid_flags(act, chunk, g.chunks);
    if (j > 0u && act && chunk + 1u == g.chunks && (grp % modulus) == modulus - 1u) {
        f |= B65A_ISTART;
    }
    return f;
}

// `tok_istart`: the per-token region carries no group structure, so an instance
// starts wherever the row's highest token is `S - 1` mod `S`. Every row is
// active and there are no chunks, so LAST never fires here.
__device__ __forceinline__ unsigned b65a_flags_of_tok(unsigned rows, unsigned rw,
                                                      unsigned s_inst, unsigned j) {
    const unsigned long long hi = (unsigned long long)(rows - 1u - j) * rw + rw;
    unsigned f = B65A_ACTIVE;
    if (j > 0u) {
        f |= B65A_NOFIRST;
    }
    if (j > 0u && (hi % s_inst) == 0ull) {
        f |= B65A_ISTART;
    }
    return f;
}

// Which band shape a flags run is for.
#define B65A_SHAPE_CHAN 0u
#define B65A_SHAPE_KEY 1u
#define B65A_SHAPE_TOK 2u

// One band sub-region's layout, for the flags kernel alone.
struct B65AccFlags {
    unsigned shape;
    B65AccGrid g;      // the chan and key grids; `log_rows`/`rows` only for tok
    unsigned modulus;  // `hd`, `C`, or `S` -- the instance-start modulus
    unsigned rw;       // the per-token region's row width
};

// One stage-2 lane: `b = last ? V[src0] * V[src1] : 0` and
// `a = (flags & sel_i) ? m_i : (last ? m_l : m_n)`.
struct B65AccS2 {
    unsigned dst;
    unsigned src0;
    unsigned src1;
    unsigned sel_i;
    qm31 m_i;
    qm31 m_l;
    qm31 m_n;
};

struct B65AccChan {
    B65AccGrid g;
    unsigned pk;
    unsigned pq;
    unsigned hd;      // channels per instance: `chan_istart`'s modulus
    m31 off_k;
    m31 off_q;
    unsigned mutate;
};

struct B65AccKey {
    B65AccGrid g;
    unsigned q;       // scores (and numerators) per row
    unsigned pv;      // V values per row
    unsigned ctx;     // keys per instance: `key_istart`'s modulus, and `split`'s
    unsigned n_insts;
    m31 off_v;        // 128
    // `Attn::col_idx` per variant. `intsplit` commits the scaled shift split
    // across the key and auxiliary columns -- `d delta = idx 2^k + aux` -- so
    // the accumulator has to put it back together before dividing `delta` out,
    // exactly as the AIR does. `two_k` is 1 and `aux_cols` null for the
    // variants that do not split (section 5 rule 11).
    m31 two_k;
    unsigned mutate;
};

struct B65AccTok {
    unsigned log_rows;
    unsigned rows;
    unsigned rw;
    unsigned s_inst;  // tokens per instance: `tok_istart`'s modulus
    unsigned with_max;
    unsigned mutate;
};

struct B65AccFreivalds {
    B65AccGrid g;
    unsigned pq;      // activations per row (the committed side)
    unsigned pk;      // weights per row, per matmul (the preprocessed side)
    unsigned n_mats;
    m31 off_x;        // 128
    unsigned mutate;
};

struct B65AccDeq {
    unsigned log_rows;
    unsigned rows;
    unsigned per_row;
    unsigned chunks;
    unsigned mutate;
};

struct B65AccAo {
    unsigned log_rows;
    unsigned rows;
    unsigned per_row;
    unsigned chunks;
    unsigned lead;
    unsigned nh;
    unsigned s;
    unsigned mutate;
};

// B11 stream N: one of the residual-stream region's Horner bindings, which is the seventh
// region of the table at the top of this file and the plainest of them.
//
// A binding folds one of the region's *value sequences* -- a group's committed
// leaves, or the slots one output names -- into a single claim,
//
//     h[j] = step * h[j-1] + sum_k w[k] * (sm[j][k] + lam * ef[j][k]),
//
// which is the same affine recurrence every other region has, with no
// segmentation at all: no group starts, no instance starts, no last-row term.
// So a binding needs no flags and no stage 2 -- it is one stage-1 lane and the
// shared scan, and `a[j]` is the constant `step` everywhere but row 0.
//
// What it reads is not a committed column. `b65_residual_stream_fill.cu` leaves `(sm, ef)`
// per leaf and per output in `in_nat` / `out_nat`, in **coset row order**, for
// exactly this purpose; before this stream those two arrays were copied back
// across the link and folded on the host, which is all `gen/residual_stream_fill_d2h` was.
// `src` picks the array, `off` and `width` cut this binding's sequence out of a
// row of it, and `stride` is that array's own row width.
struct B65AccResidualStreamBind {
    unsigned src;     // 0 = `in_nat`, 1 = `out_nat`
    unsigned off;     // first member of this binding's sequence within a row
    unsigned width;   // members of the sequence
    unsigned stride;  // members per row of the source array
    unsigned w_off;   // where this binding's weights start in the packed array
    unsigned pad;
    qm31 step;
};

// ---------------------------------------------------------------- the shared half --

// The inclusive affine scan: `b[l][j] <- a[l][j] * b[l][j-1] + b[l][j]`, with
// `b[l][-1] = 0`, for `lanes` independent recurrences of `rows` rows each. `a`
// is clobbered (it becomes the within-block product prefix). Three phases, the
// shape `b65_logup_running`'s neighbours have: a block-local scan, a serial
// scan of the block totals, and the carry applied.
extern "C" void b65_acc_scan(qm31 *a, qm31 *b, unsigned rows, unsigned lanes,
                             unsigned mutate);

// Stage 2's `(a, b)`, from the flags one of the region kernels wrote and the
// stage-1 lanes the scan has already closed.
extern "C" void b65_acc_stage2(const unsigned *flags, qm31 *a, qm31 *b,
                               const B65AccS2 *lanes, unsigned n_lanes,
                               unsigned rows, unsigned mutate);

// The committed columns: `out[4 k + t][i] = v[lane[k]][f(i)].to_m31_array()[t]`,
// with `f` the closed form of `bit_reverse_coset_to_circle_domain_order`, so
// `eval_of` never runs on an accumulator column (20.1's design, for a QM31).
extern "C" void b65_acc_split(const qm31 *v, m31 **out, const unsigned *lane,
                              unsigned n_out, unsigned rows, unsigned log_rows,
                              unsigned mutate);

// The claimed sums: the last *coset* row of every lane, which is the value the
// region's Horner leaves behind. A few QM31 back to the host, and the only
// thing that comes back at all.
extern "C" void b65_acc_last(const qm31 *v, qm31 *host, unsigned rows,
                             unsigned lanes);

// ------------------------------------------------------------- the six regions --

// B7a milestone 4 (section 25): the band shapes' flags, written once at fill
// time. `commit_pre` runs before the challenges are drawn and the accumulators
// run after, so the array outlives both and the selectors and the segmentation
// are one computation rather than two that agree.
extern "C" void b65_flags_band(unsigned *flags, B65AccFlags cfg);


extern "C" void b65_acc_chan(const m31 *const *cols, const qm31 *r_pow,
                             const qm31 *sq_pow, qm31 r_step, qm31 sq_step,
                             const unsigned *flags, qm31 *a, qm31 *b,
                             B65AccChan cfg);

extern "C" void b65_acc_grid(const m31 *const *key_cols, const m31 *const *num_cols,
                             const m31 *const *aux_cols, const m31 *const *val_cols,
                             const qm31 *sg_pow, const qm31 *rv_pow, qm31 sg_step,
                             qm31 rv_step, const m31 *inv_delta,
                             const unsigned *flags, qm31 *a, qm31 *b,
                             B65AccKey cfg);

extern "C" void b65_acc_tok(const m31 *const *max_cols, const m31 *const *z_cols,
                            const qm31 *sr_pow, qm31 sr_step, qm31 mu,
                            const unsigned *flags, qm31 *a, qm31 *b,
                            B65AccTok cfg);

extern "C" void b65_acc_freivalds(const m31 *const *base_cols, const m31 *const *pre_cols,
                                  const qm31 *s_pow, const qm31 *rv_pow, qm31 s_step,
                                  qm31 rv_step, unsigned *flags, qm31 *a, qm31 *b,
                                  B65AccFreivalds cfg);

extern "C" void b65_acc_deq(const m31 *const *acc_cols, const qm31 *rv_pow,
                            qm31 rv_step, unsigned *flags, qm31 *a, qm31 *b,
                            B65AccDeq cfg);

extern "C" void b65_acc_ao(const m31 *den_col, const m31 *const *acc_cols,
                           const qm31 *rv_pow, qm31 rv_step, unsigned *flags,
                           qm31 *a, qm31 *b, B65AccAo cfg);

// B11 stream N: every binding of one residual-stream region, one lane each. `specs` and
// `weights` are host arrays and are cloned to the card here; `in_nat` and
// `out_nat` are already there, where `b65_residual_stream_fill.cu` left them.
extern "C" void b65_acc_residual_stream_bind(const int *in_nat, const int *out_nat,
                             const qm31 *weights, unsigned n_weights, qm31 lam,
                             const B65AccResidualStreamBind *specs, unsigned n_specs, qm31 *a,
                             qm31 *b, unsigned rows, unsigned mutate);

#endif
