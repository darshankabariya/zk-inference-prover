// B7a milestone 4 (section 3 row 1): 4a's last 79 columns.
#include "fields.cuh"
#include "utils.cuh"
#include "b65_pre_fill.cuh"

#define B65P_BLOCK 256u

// Is this defect aimed at this family?
__device__ __forceinline__ bool b65p_hit(unsigned mutate, unsigned at, unsigned mut,
                                         unsigned fam) {
    return mutate == mut && (at == fam || at == B65P_AT_ALL);
}

// ============================================================ the four tables ==

// `Round::carry`: given the 8-bit significand, the round bit and the sticky
// flag, does the significand step up? `bf16.rs`, arm for arm.
__device__ __forceinline__ unsigned b65p_carry(unsigned round, unsigned m, unsigned rb,
                                               unsigned st) {
    if (round == B65P_ROUND_AWAY) {
        return rb;
    }
    if (round == B65P_ROUND_ZERO) {
        return 0u;
    }
    return rb & (st | (m & 1u));
}

// `bf16::from_f32` for an integer below 256.
//
// Eight significand bits fit bf16's eight, so the f32 of such an integer has
// sixteen zero low bits and the round-to-nearest-even step is the identity --
// the conversion is exact and this is the same bit surgery the host runs, not
// an approximation of it.
__device__ __forceinline__ unsigned short b65p_bf16_small(unsigned m2) {
    const unsigned u = __float_as_uint((float)m2);
    const unsigned bias = 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)((u + bias) >> 16);
}

// `bf16::norm_table`'s row `d`, into `o[0..6]` = `(m2, ph, st, m7, expfield, nz)`
// -- the order `compose.rs`'s `get` array reads the struct in.
__device__ __forceinline__ void b65p_norm_row(const B65PreCfg &cfg, unsigned d,
                                              unsigned *o) {
    const bool nopad = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOPAD, B65P_AT_NORM);
    if (d == 0u || (d >= cfg.real_rows && !nopad)) {
        return;  // the zero row, and every row past the table's own length
    }
    if (d < 256u) {
        // the exactly-representable small magnitudes: `mag = M2 < 256`, `ph = 1`
        const unsigned short b = b65p_bf16_small(d);
        o[0] = d;
        o[1] = 1u;
        o[3] = b & 0x7fu;
        o[4] = (b >> 7) & 0xffu;
        o[5] = 1u;
        return;
    }
    const unsigned per_k = cfg.sticky ? 512u : 256u;
    const unsigned idx = d - 256u;
    unsigned k = idx / per_k;
    if (k > cfg.kmax) {
        k = cfg.kmax;  // only reachable under `nopad`
    }
    const unsigned rem = idx % per_k;
    const unsigned m2 = 256u + (cfg.sticky ? (rem >> 1) : rem);
    unsigned st = cfg.sticky ? (rem & 1u) : 0u;
    if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOSTICKY, B65P_AT_NORM)) {
        st = 0u;
    }
    // `m` in [128, 256); `m + up` reaches 256 only from `m = 255, rb = 1`, which
    // renormalises to significand 128 one binade up.
    const unsigned m = m2 >> 1, rb = m2 & 1u;
    const unsigned up = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOROUND, B65P_AT_NORM)
                            ? 0u
                            : b65p_carry(cfg.round, m, rb, st);
    unsigned msig = m + up, extra = 0u;
    if (msig == 256u && !b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NORENORM, B65P_AT_NORM)) {
        msig = 128u;
        extra = 1u;
    }
    o[0] = m2;
    o[1] = 1u << k;
    o[2] = st;
    o[3] = msig - 128u;
    o[4] = k + 1u + extra + B65P_EXP_BIAS;
    o[5] = 1u;
}

// `Tables::new`'s fraction table, row `d`, into `o[0..3]` = `(ph, r, st)`.
//
// The rows are `(2^k, r, ..)` for `k = 0..=kmax` and `r = 0..2^k`, one zero row
// first, so `d` and `k` are related by `2^k <= d < 2^(k+1)`: the window is a
// leading-zero count and `r` is what is left of `d` below it.
__device__ __forceinline__ void b65p_frac_row(const B65PreCfg &cfg, unsigned d,
                                              unsigned *o) {
    const bool nopad = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOPAD, B65P_AT_FRAC);
    if (d == 0u || (d >= cfg.real_rows && !nopad)) {
        return;
    }
    unsigned w = d;
    if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOKWIN, B65P_AT_FRAC)) {
        w = d - 1u;  // the window boundary, off by one
    }
    const unsigned k = w == 0u ? 0u : (31u - __clz(w));
    const unsigned r = w - (1u << k);
    o[0] = 1u << k;
    o[1] = r;
    o[2] = (cfg.sticky && r != 0u &&
            !b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOSTICKY, B65P_AT_FRAC))
               ? 1u
               : 0u;
}

// `bf16::mul_table`'s row `d`, into `o[0..4]` = `(a, b, m7, carry)`.
//
// `(128 + a)(128 + b)` is a 16-bit integer in `[2^14, 2^16)`, so the product's
// significand is the top 8 bits and the binade shift is its 16th bit. The table
// has exactly `2^14` rows, so there is no padded arm here.
__device__ __forceinline__ void b65p_mul_row(const B65PreCfg &cfg, unsigned d,
                                             unsigned *o) {
    if (d >= cfg.real_rows) {
        return;
    }
    const unsigned a = d >> 7, b = d & 127u;
    const unsigned p = (128u + a) * (128u + b);
    const unsigned carry = (p >> 15) & 1u;
    const unsigned shift = 7u + carry;
    const unsigned m = p >> shift;
    const unsigned rb = (p >> (shift - 1u)) & 1u;
    const unsigned st = (p & ((1u << (shift - 1u)) - 1u)) != 0u ? 1u : 0u;
    const unsigned up = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOROUND, B65P_AT_MUL)
                            ? 0u
                            : b65p_carry(cfg.round, m, rb, st);
    unsigned msig = m + up, extra = 0u;
    if (msig == 256u && !b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NORENORM, B65P_AT_MUL)) {
        msig = 128u;
        extra = 1u;
    }
    o[0] = a;
    o[1] = b;
    o[2] = msig - 128u;
    o[3] = carry + extra;
}

// One thread, one committed row, every column of one shared table.
//
// The cells are written raw, not reduced: `common.rs`'s `m31_u32` is
// `M31::from_u32_unchecked` and every value here is already a field element.
__global__ void b65p_table(const unsigned *const *__restrict__ counters,
                           m31 *const *__restrict__ cols, m31 *__restrict__ m_col,
                           const B65PreCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.rows) {
        return;
    }
    const unsigned fam = cfg.kind == B65P_KIND_NORM    ? B65P_AT_NORM
                         : cfg.kind == B65P_KIND_FRAC  ? B65P_AT_FRAC
                         : cfg.kind == B65P_KIND_MUL   ? B65P_AT_MUL
                         : cfg.kind == B65P_KIND_QNZ   ? B65P_AT_QNZ
                                                       : B65P_AT_RANGE;
    // the coset row this committed row holds; `noperm` is the control that says
    // the permutation is load-bearing, and `eval_of` is exactly what it replaces
    const unsigned d = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOPERM, fam)
                           ? i
                           : b65a_perm(i, cfg.log_rows);

    unsigned o[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    switch (cfg.kind) {
        case B65P_KIND_NORM:
            b65p_norm_row(cfg, d, o);
            break;
        case B65P_KIND_FRAC:
            b65p_frac_row(cfg, d, o);
            break;
        case B65P_KIND_MUL:
            b65p_mul_row(cfg, d, o);
            break;
        case B65P_KIND_QNZ:
            o[0] = d;
            o[1] = d != 0u ? 1u : 0u;
            break;
        default:
            o[0] = d;  // a range table is its own row index
            break;
    }
    for (unsigned c = 0; c < cfg.n_cols; ++c) {
        cols[c][i] = (m31)o[c];
    }

    // The multiplicity column: every fill scatter-added into a counter of its
    // own (20.1, 21.1) and the host's array was their sum. The sum happens here
    // instead and nothing comes back. A null column is "no fill that bumps this
    // table ran on the device", so the host keeps it and it stays a 4a upload.
    if (m_col != nullptr) {
        unsigned n = cfg.n_counters;
        if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_ONECOUNTER, fam) && n > 1u) {
            n = 1u;
        }
        unsigned s = 0u;
        for (unsigned c = 0; c < n; ++c) {
            s += counters[c][d];
        }
        m_col[i] = (m31)s;
    }
}

// The rows where this run's own table disagrees with the closed form.
//
// Empty for `frac` and `mul`, which nothing tampers with, and one entry for
// `ProverTamper::norm_row` -- rung 3's seventh probe, which flips one row of the
// normalisation table and rebuilds the witness to match so that only the pinned
// preprocessed root gives it away. Section 5 rule 8: a kernel that computed the
// closed form and stopped there would repair the probe silently.
__global__ void b65p_patch(const unsigned *__restrict__ idx,
                           const unsigned *__restrict__ val, const unsigned n,
                           m31 *const *__restrict__ cols, const B65PreCfg cfg) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const unsigned i = b65a_perm(idx[t], cfg.log_rows);
    for (unsigned c = 0; c < cfg.n_cols; ++c) {
        cols[c][i] = (m31)val[t * cfg.n_cols + c];
    }
}

// The family a table's `kind` belongs to, on the host side of the call.
static unsigned b65p_fam_of(unsigned kind) {
    switch (kind) {
        case B65P_KIND_NORM: return B65P_AT_NORM;
        case B65P_KIND_FRAC: return B65P_AT_FRAC;
        case B65P_KIND_MUL: return B65P_AT_MUL;
        case B65P_KIND_QNZ: return B65P_AT_QNZ;
        default: return B65P_AT_RANGE;
    }
}

extern "C" void b65_fill_pre_table(const unsigned *const *counters,
                                   const unsigned *patch_idx, const unsigned *patch_val,
                                   unsigned n_patch, m31 *const *cols, m31 *m_col,
                                   B65PreCfg cfg) {
    unsigned **d_cnt = nullptr;
    if (cfg.n_counters > 0u) {
        d_cnt = clone_to_device<unsigned *>(const_cast<unsigned **>(counters),
                                            cfg.n_counters);
    }
    m31 **d_cols = clone_to_device<m31 *>(const_cast<m31 **>(cols), cfg.n_cols);
    const unsigned gb = (cfg.rows + B65P_BLOCK - 1u) / B65P_BLOCK;
    b65p_table<<<gb, B65P_BLOCK>>>(d_cnt, d_cols, m_col, cfg);
    const unsigned fam = b65p_fam_of(cfg.kind);
    const bool nopatch =
        cfg.mutate == B65P_MUT_NOPATCH && (cfg.at == fam || cfg.at == B65P_AT_ALL);
    if (n_patch > 0u && !nopatch) {
        const unsigned pb = (n_patch + B65P_BLOCK - 1u) / B65P_BLOCK;
        b65p_patch<<<pb, B65P_BLOCK>>>(patch_idx, patch_val, n_patch, d_cols, cfg);
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_cols);
    if (d_cnt != nullptr) {
        cuda_free_memory(d_cnt);
    }
}

// ========================================================== the band selectors ==

// One thread, one committed row, one band sub-region's selector columns.
//
// Every one of them is a predicate of the flags `b65_flags_band` wrote for that
// coset row -- which is the point: `attn_trace.rs` spells the predicates once,
// the accumulators segment on them and the AIR reads them out of tree 0, and
// after this row all three are the same array rather than three transcriptions
// that agree.
//
// | column | chan | key | row |
// |---|---|---|---|
// | 0 | NOFIRST | NOFIRST | `j > 0` |
// | 1 | LAST | LAST | `j + 1 == rows` |
// | 2 | `j > 0` | `j > 0` | ISTART |
// | 3 | `j + 1 == rows` | `j + 1 == rows` | |
// | 4 | ISTART | `j > 0 && !LAST` | |
// | 5 | | `j > 0 && LAST` | |
//
// Stream CC (E11) adds a ninth key column, `ACTIVE` itself, when the grid leaves
// inert lead rows -- `cfg.n_cols == 9` is how this side is told.  It is the
// AIR's numerator-fold guard, and it is `act` unmodified, which is the whole
// point: the accumulator in `b65_acc.cu` has gated `nf` on the same bit since it
// was written, so what lands in tree 0 is the predicate the fold already used.
// | 6 | | ISTART | |
// | 7 | | the key scale, inverted, per instance | |
__global__ void b65p_sel(const unsigned *__restrict__ flags,
                         m31 *const *__restrict__ cols,
                         const m31 *__restrict__ scale, const B65SelCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.g.rows) {
        return;
    }
    const unsigned j = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOPERM, cfg.fam)
                           ? i
                           : b65a_perm(i, cfg.g.log_rows);
    const unsigned f = flags[j];
    const bool act = (f & B65A_ACTIVE) != 0u;
    const bool nofirst = (f & B65A_NOFIRST) != 0u;
    // `nolast`: the group's *first* row, not its last -- `chunk + 1 == chunks`
    // where the host writes `chunk == 0`.
    bool last = b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOLAST, cfg.fam)
                    ? (act && !nofirst)
                    : (f & B65A_LAST) != 0u;
    // `noactive`: an inert lead row reads as the end of a group -- the `None`
    // arm of `matches!(row(j), Some(0))`, forgotten.
    if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOACTIVE, cfg.fam) && !act) {
        last = true;
    }
    const bool istart =
        b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOISTART, cfg.fam) ? false
                                                                : (f & B65A_ISTART) != 0u;
    // every `j > 0` the host writes, in one place
    const bool pos =
        b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOFIRSTROW, cfg.fam) ? true : (j > 0u);
    const bool endrow = j + 1u == cfg.g.rows;

    if (cfg.shape == B65A_SHAPE_TOK) {
        cols[0][i] = (m31)(pos ? 1u : 0u);
        cols[1][i] = (m31)(endrow ? 1u : 0u);
        cols[2][i] = (m31)(istart ? 1u : 0u);
        return;
    }
    cols[0][i] = (m31)(nofirst ? 1u : 0u);
    cols[1][i] = (m31)(last ? 1u : 0u);
    cols[2][i] = (m31)(pos ? 1u : 0u);
    cols[3][i] = (m31)(endrow ? 1u : 0u);
    if (cfg.shape == B65A_SHAPE_CHAN) {
        cols[4][i] = (m31)(istart ? 1u : 0u);
        return;
    }
    cols[4][i] = (m31)((pos && !last) ? 1u : 0u);
    cols[5][i] = (m31)((pos && last) ? 1u : 0u);
    cols[6][i] = (m31)(istart ? 1u : 0u);

    // The master exp table's key scale, inverted, per instance: the grid commits
    // `d delta` and both the AIR and `b65_acc.cu`'s score accumulator divide it
    // back out. `scale` holds `1/delta` in its first `n_insts` entries and
    // `delta` itself in the next, which is what makes `noinv` a control rather
    // than an argument.
    m31 sc = 1u;
    if (cfg.with_scale != 0u) {
        unsigned grp = 0u, chunk = 0u;
        const bool a2 = b65a_grid_row(cfg.g, j, &grp, &chunk);
        unsigned inst = grp / cfg.ctx;
        if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOIDENT, cfg.fam)) {
            sc = scale[a2 ? inst : 0u];  // an inert row takes instance 0's scale
        } else if (!a2) {
            sc = 1u;
        } else {
            if (b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOINV, cfg.fam)) {
                inst += cfg.n_insts;  // `delta`, not `1/delta`
            }
            sc = scale[inst];
        }
    }
    cols[7][i] = sc;

    // Stream CC (E11): the inert-lead-row selector, present only when the grid
    // has such rows.  `noactive` is "the `None` arm forgotten", and forgetting it
    // here is writing 1 on an inert row -- the same defect the mutation already
    // injects into `last`, on the column that now carries the numerator fold's
    // guard, so one control covers both faces of it.
    if (cfg.n_cols > 8u) {
        const bool act_col =
            b65p_hit(cfg.mutate, cfg.at, B65P_MUT_NOACTIVE, cfg.fam) ? true : act;
        cols[8][i] = (m31)(act_col ? 1u : 0u);
    }
}

extern "C" void b65_fill_sel(const unsigned *flags, m31 *const *cols,
                             const m31 *scale, B65SelCfg cfg) {
    m31 **d_cols = clone_to_device<m31 *>(const_cast<m31 **>(cols), cfg.n_cols);
    const unsigned rows = cfg.g.rows;
    const unsigned bw = rows < B65P_BLOCK ? rows : B65P_BLOCK;
    b65p_sel<<<(rows + bw - 1u) / bw, bw>>>(flags, d_cols, scale, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_cols);
}
