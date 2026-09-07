// B11 stream M: the residual stream's generation, on the device. See `b65_residual_stream_fill.cuh` for
// why this is an interpreter and not an emit.
#include "fields.cuh"
#include "utils.cuh"
#include "b65_residual_stream_fill.cuh"

#define B65_RESIDUAL_STREAM_FILL_BLOCK 128u

// ---------------------------------------------------------------- the permutation --

// `common.rs`'s `m4_perm`, the same involution `b65_fill.cu` uses: the thread
// that owns committed output `i` computes the value of coset row `f(i)`.
__device__ __forceinline__ unsigned g_perm(unsigned i, unsigned log_n) {
    const unsigned n = 1u << log_n;
    const unsigned c = (i & 1u) ? ((2u * n - i - 1u) >> 1) : (i >> 1);
    return __brev(c) >> (32u - log_n);
}

// ------------------------------------------------------------------- field helpers --

#define B65_RESIDUAL_STREAM_P 2147483647LL

// `residual_stream::m31_of`: `(((v % p) + p) % p)`.
__device__ __forceinline__ m31 g_m31_of(long long v) {
    long long x = v % B65_RESIDUAL_STREAM_P;
    if (x < 0) x += B65_RESIDUAL_STREAM_P;
    return (m31)x;
}

// `residual_stream::sgn_of`: the field element read back as the signed value it encodes.
// This is what lets the kernel keep no per-row state at all -- a slot's
// operands are cells it already wrote.
__device__ __forceinline__ long long g_sgn_of(m31 v) {
    return ((long long)v > B65_RESIDUAL_STREAM_P / 2) ? (long long)v - B65_RESIDUAL_STREAM_P : (long long)v;
}

// ------------------------------------------------------------------ the multiplicities --

// The same warp-aggregated scatter-add `b65_fill.cu` uses. The residual stream's lookups
// concentrate hard -- every leaf of a padded row hits `val_at(0, 0) = 0` -- so
// without this one address serialises the kernel.
__device__ __forceinline__ void g_bump(unsigned *m, unsigned idx) {
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, idx);
    const unsigned leader = __ffs(peers) - 1u;
    if ((threadIdx.x & 31u) == leader) {
        atomicAdd(&m[idx], (unsigned)__popc(peers));
    }
}

// `residual_stream_region`'s `match ... { Some(i) => look.push(..), None => off += 1 }`,
// as one call: the index is `B65_RESIDUAL_STREAM_FILL_NONE` when the host's lookup is `None`.
__device__ __forceinline__ void g_bump_or_off(B65ResidualStreamMult t, unsigned idx,
                                              unsigned *off, unsigned drop) {
    if (idx == B65_RESIDUAL_STREAM_FILL_NONE || idx >= t.n) {
        *off += 1u;
        return;
    }
    if (drop) {
        return;
    }
    g_bump(t.m, idx);
}

// --------------------------------------------------------------- the closed forms --

// `residual_stream::split`: a bf16 pattern to `(sm, ef)`, with both bf16 zeros and both
// infinities/NaNs collapsing to the single `(0, 0)`.
__device__ __forceinline__ void g_split(unsigned pat, long long *sm, unsigned *ef) {
    const unsigned sg = (pat >> 15) & 1u;
    const unsigned e = (pat >> 7) & 0xffu;
    const unsigned m7 = pat & 0x7fu;
    if (e == 0u || e == 0xffu) {
        *sm = 0;
        *ef = 0u;
    } else {
        const long long m = (long long)(128u + m7);
        *sm = sg ? -m : m;
        *ef = e;
    }
}

// `residual_stream::join`.
__device__ __forceinline__ unsigned g_join(long long sm, unsigned ef) {
    if (sm == 0) {
        return 0u;
    }
    const unsigned sg = (sm < 0) ? 1u : 0u;
    const unsigned m7 = (unsigned)((sm < 0 ? -sm : sm) - 128);
    return ((sg << 15) | (ef << 7) | m7) & 0xffffu;
}

// `ResidualStreamTables::pow_at`: `[0, DMAX]` then `[-UMIN, -1]`.
__device__ __forceinline__ unsigned g_pow_at(long long u) {
    if (u >= 0 && u <= (long long)B65_RESIDUAL_STREAM_DMAX) {
        return (unsigned)u;
    }
    if (u >= -(long long)B65_RESIDUAL_STREAM_UMIN && u < 0) {
        return B65_RESIDUAL_STREAM_DMAX + 1u + (unsigned)(-u - 1);
    }
    return B65_RESIDUAL_STREAM_FILL_NONE;
}

// `residual_stream::val_at`: the zero row, then `(ef, m7, sign)`.
__device__ __forceinline__ unsigned g_val_at(long long sm, unsigned ef) {
    if (sm == 0 && ef == 0u) {
        return 0u;
    }
    const unsigned long long a = (unsigned long long)(sm < 0 ? -sm : sm);
    if (ef == 0u || ef >= 0xffu || a < 128ull || a >= 256ull) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    const unsigned m7 = (unsigned)(a - 128ull);
    return 1u + ((ef - 1u) * 128u + m7) * 2u + (sm < 0 ? 1u : 0u);
}

// `ResidualStreamTables::smul_at`.
__device__ __forceinline__ unsigned g_smul_idx(long long v) {
    if (v == 0) {
        return 0u;
    }
    const unsigned long long a = (unsigned long long)(v < 0 ? -v : v);
    if (a < 128ull || a >= 256ull) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    return 1u + (v < 0 ? 128u : 0u) + (unsigned)(a - 128ull);
}

__device__ __forceinline__ unsigned g_smul_at(long long a, long long b) {
    const unsigned ia = g_smul_idx(a);
    const unsigned ib = g_smul_idx(b);
    if (ia == B65_RESIDUAL_STREAM_FILL_NONE || ib == B65_RESIDUAL_STREAM_FILL_NONE) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    return ia * 257u + ib;
}

// `bf16::normalise`: `mag = m2 * ph + r`, `0 <= r < ph`.
__device__ __forceinline__ bool g_normalise(unsigned long long mag, unsigned kmax,
                                            unsigned *m2, unsigned *ph, unsigned *st,
                                            unsigned long long *r) {
    if (mag == 0ull) {
        *m2 = 0u;
        *ph = 0u;
        *st = 0u;
        *r = 0ull;
        return true;
    }
    const unsigned bits = 64u - (unsigned)__clzll((long long)mag);
    const unsigned k = bits > 9u ? bits - 9u : 0u;
    if (k > kmax) {
        return false;
    }
    const unsigned long long phv = 1ull << k;
    *ph = (unsigned)phv;
    *m2 = (unsigned)(mag >> k);
    *r = mag & (phv - 1ull);
    *st = (*r != 0ull) ? 1u : 0u;
    return true;
}

// `ResidualStreamTables::norm_at`. Note this is the *residual stream's* row order, which differs
// from `layer::Tables::norm_at` only in how it treats `st` on a non-sticky
// round: the residual stream masks `st` to zero in the key, the dequant rejects it. The
// caller masks before calling, exactly as `residual_stream_region` does.
__device__ __forceinline__ unsigned g_norm_at(unsigned m2, unsigned ph, unsigned st,
                                              unsigned kmax, unsigned sticky) {
    if (m2 == 0u && ph == 0u) {
        return 0u;
    }
    if (ph == 1u && m2 >= 1u && m2 < 256u) {
        return m2;
    }
    if (ph == 0u || (ph & (ph - 1u)) != 0u || m2 < 256u || m2 >= 512u) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    const unsigned k = (unsigned)(31 - __clz((int)ph));
    if (k > kmax) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    const unsigned per = 1u + sticky;
    return 256u + (k * 256u + (m2 - 256u)) * per + (sticky ? st : 0u);
}

// `ResidualStreamTables::frac_at`.
__device__ __forceinline__ unsigned g_frac_at(unsigned ph, unsigned long long r,
                                              unsigned kmax) {
    if (ph == 0u) {
        return (r == 0ull) ? 0u : B65_RESIDUAL_STREAM_FILL_NONE;
    }
    if ((ph & (ph - 1u)) != 0u) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    const unsigned k = (unsigned)(31 - __clz((int)ph));
    if (k > kmax || r >= (unsigned long long)ph) {
        return B65_RESIDUAL_STREAM_FILL_NONE;
    }
    return ph + (unsigned)r;
}

// ------------------------------------------------------------------- the kernel --

// One operand, read back out of the column it was written to. `NONE` is
// `Src::Zero`. A slot output's exponent is clamped on the way *out* of the slot
// that produced it (`outs.push((c_sm, c_ef.clamp(0, 0xfe)))`) while its cell
// keeps the unclamped value, so the clamp belongs here and only for a slot
// source -- which is what `from_slot` says.
__device__ __forceinline__ void g_operand(unsigned *const *cols, unsigned at,
                                          unsigned c_sm, unsigned c_ef, bool from_slot,
                                          unsigned mutate, long long *sm, unsigned *ef) {
    if (c_sm == B65_RESIDUAL_STREAM_FILL_NONE) {
        *sm = 0;
        *ef = 0u;
        return;
    }
    // `at` is the *committed* index this thread owns, which is where it wrote
    // the cells it is now reading back. `offby` reads the neighbouring row
    // instead: the whole "every intermediate is already a committed cell" idea
    // rests on this index, and it is the control that cannot be inert on an
    // honest witness the way the clamp is.
    const unsigned rd = (mutate == B65_RESIDUAL_STREAM_MUT_OFFBY) ? (at ^ 1u) : at;
    *sm = g_sgn_of(cols[c_sm][rd]);
    const long long e = g_sgn_of(cols[c_ef][rd]);
    if (from_slot && mutate != B65_RESIDUAL_STREAM_MUT_CLAMP) {
        *ef = (unsigned)(e < 0 ? 0 : (e > 0xfe ? 0xfe : e));
    } else {
        *ef = (unsigned)(e < 0 ? 0 : e);
    }
}

__global__ __launch_bounds__(B65_RESIDUAL_STREAM_FILL_BLOCK) void b65_residual_stream_fill_kernel(
    const unsigned *wit, const B65ResidualStreamGroup *groups, const B65ResidualStreamSlot *slots,
    const unsigned *norm3, unsigned n_norm, const int *smul3, unsigned n_smul,
    const int *const *un2, unsigned n_val, const unsigned *out_slot,
    unsigned *const *cols, unsigned *out_pat, int *out_nat, int *in_nat,
    B65ResidualStreamMult m_pow, B65ResidualStreamMult m_norm, B65ResidualStreamMult m_val, B65ResidualStreamMult m_frac,
    B65ResidualStreamMult m_smul, const B65ResidualStreamMult *m_un, unsigned *counts, B65ResidualStreamCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.rows) {
        return;
    }
    // The committed row this thread owns, and the coset row it holds. Under
    // `noperm` the two are the same, which is the control for the whole
    // "`eval_of` never runs" idea.
    const unsigned row = (cfg.mutate == B65_RESIDUAL_STREAM_MUT_NOPERM) ? i : g_perm(i, cfg.log_rows);
    const unsigned j = row;
    const bool live = j < cfg.live;
    unsigned off = 0u;
    unsigned bad_exp = 0u;
    unsigned un_off = 0u;

    // ---- leaves ----------------------------------------------------------
    for (unsigned g = 0; g < cfg.n_groups; ++g) {
        const B65ResidualStreamGroup gr = groups[g];
        for (unsigned k = 0; k < gr.n; ++k) {
            const unsigned flat = gr.flat + k;
            const unsigned pat = live ? wit[(size_t)j * cfg.n_leaves + flat] : 0u;
            long long sm;
            unsigned ef;
            g_split(pat, &sm, &ef);
            const unsigned b = gr.leaf_col + k * gr.width;
            cols[b][i] = g_m31_of(sm);
            cols[b + 1][i] = g_m31_of((long long)ef);
            // `j`, not `i`: the host builds `in_nat` by walking `filled` in
            // *coset* row order and never permutes it -- only the committed
            // columns go through `eval_of`. The bindings read this array by row, so
            // it stays in row order here too.
            in_nat[((size_t)j * cfg.n_leaves + flat) * 2] = (int)sm;
            in_nat[((size_t)j * cfg.n_leaves + flat) * 2 + 1] = (int)ef;
            const unsigned v = g_val_at(sm, ef);
            const unsigned drop =
                (cfg.mutate == B65_RESIDUAL_STREAM_MUT_NOBUMP && (j & 1023u) == 0u && k == 0u) ? 1u : 0u;
            if (gr.un == 0u) {
                g_bump_or_off(m_val, v, &off, drop);
            } else {
                const unsigned u = gr.un - 1u;
                g_bump_or_off(m_un[u], v, &off, drop);
                // the unary image, tabulated: `residual_stream::un_table` over `val_at`
                long long osm = 0;
                unsigned oef = 0u;
                if (v != B65_RESIDUAL_STREAM_FILL_NONE && v < n_val) {
                    osm = (long long)un2[u][(size_t)v * 2];
                    oef = (unsigned)un2[u][(size_t)v * 2 + 1];
                } else {
                    // the host still evaluates the unary here; the kernel has no
                    // table row to read, so it says so instead of guessing
                    un_off += 1u;
                }
                cols[b + 2][i] = g_m31_of(osm);
                cols[b + 3][i] = g_m31_of((long long)oef);
            }
        }
    }

    // ---- slots -----------------------------------------------------------
    for (unsigned s = 0; s < cfg.n_slots; ++s) {
        const B65ResidualStreamSlot sl = slots[s];
        long long a_sm, b_sm;
        unsigned a_ef, b_ef;
        g_operand(cols, i, sl.a_sm, sl.a_ef, (sl.from_slot & 1u) != 0u, cfg.mutate,
                  &a_sm, &a_ef);
        g_operand(cols, i, sl.b_sm, sl.b_ef, (sl.from_slot & 2u) != 0u, cfg.mutate,
                  &b_sm, &b_ef);
        const unsigned b = sl.base;
        const unsigned drop =
            (cfg.mutate == B65_RESIDUAL_STREAM_MUT_NOBUMP && (j & 1023u) == 0u && s == 0u) ? 1u : 0u;
        long long c_sm = 0;
        long long c_ef = 0;
        unsigned nz = 0u;
        if (cfg.kind == 0u) {
            long long etop = (long long)(a_ef > b_ef ? a_ef : b_ef);
            if (cfg.mutate == B65_RESIDUAL_STREAM_MUT_ANCHOR && (j & 1023u) == 0u && s == 0u) {
                etop += 1;
            }
            long long fa = 0, fb = 0;
            {
                const long long u = (long long)a_ef - etop + (long long)B65_RESIDUAL_STREAM_DMAX;
                const unsigned p = g_pow_at(u);
                g_bump_or_off(m_pow, p, &off, drop);
                fa = (p != B65_RESIDUAL_STREAM_FILL_NONE && u >= 0) ? (1LL << u) : 0LL;
            }
            {
                const long long u = (long long)b_ef - etop + (long long)B65_RESIDUAL_STREAM_DMAX;
                const unsigned p = g_pow_at(u);
                g_bump_or_off(m_pow, p, &off, drop);
                fb = (p != B65_RESIDUAL_STREAM_FILL_NONE && u >= 0) ? (1LL << u) : 0LL;
            }
            cols[b][i] = g_m31_of(fa);
            cols[b + 1][i] = g_m31_of(fb);
            const long long acc = a_sm * fa + b_sm * fb;
            const unsigned sg = (acc < 0) ? 1u : 0u;
            const unsigned long long mag =
                (unsigned long long)(acc < 0 ? -acc : acc);
            unsigned m2 = 0u, ph = 0u, st_raw = 0u;
            unsigned long long r = 0ull;
            if (!g_normalise(mag, cfg.kmax, &m2, &ph, &st_raw, &r)) {
                off += 1u;
                m2 = 0u;
                ph = 0u;
                st_raw = 0u;
                r = 0ull;
            }
            unsigned st = cfg.sticky ? st_raw : 0u;
            if (cfg.mutate == B65_RESIDUAL_STREAM_MUT_NOSTICKY) {
                st = 0u;
            }
            const unsigned ni = g_norm_at(m2, ph, st, cfg.kmax, cfg.sticky);
            unsigned m7n = 0u, efn = 0u, nzn = 0u;
            if (ni != B65_RESIDUAL_STREAM_FILL_NONE && ni < n_norm) {
                m7n = norm3[(size_t)ni * 3];
                efn = norm3[(size_t)ni * 3 + 1];
                nzn = norm3[(size_t)ni * 3 + 2];
            } else {
                off += 1u;
            }
            g_bump_or_off(m_norm, ni, &off, drop);
            g_bump_or_off(m_frac, g_frac_at(ph, r, cfg.kmax), &off, drop);
            c_sm = (long long)(128u * nzn + m7n) * (sg == 1u ? -1LL : 1LL);
            c_ef = (nzn == 1u)
                       ? (long long)efn + etop - (long long)B65_RESIDUAL_STREAM_DMAX
                             - (long long)B65_RESIDUAL_STREAM_EXP_BIAS
                       : 0LL;
            if (nzn == 1u && !(c_ef >= 1 && c_ef < 0xff)) {
                bad_exp += 1u;
            }
            cols[b + 2][i] = g_m31_of(etop);
            cols[b + 3][i] = g_m31_of((long long)mag);
            cols[b + 4][i] = g_m31_of((long long)sg);
            cols[b + 5][i] = g_m31_of((long long)m2);
            cols[b + 6][i] = g_m31_of((long long)ph);
            cols[b + 7][i] = g_m31_of((long long)r);
            unsigned kk = 8u;
            if (cfg.sticky) {
                cols[b + 8][i] = g_m31_of((long long)st);
                kk = 9u;
            }
            cols[b + kk][i] = g_m31_of((long long)m7n);
            cols[b + kk + 1][i] = g_m31_of((long long)efn);
            cols[b + kk + 2][i] = g_m31_of((long long)nzn);
            cols[b + kk + 3][i] = g_m31_of(c_sm);
            cols[b + kk + 4][i] = g_m31_of(c_ef);
            nz = nzn;
        } else {
            const unsigned si = g_smul_at(a_sm, b_sm);
            long long c_sm0 = 0;
            unsigned cy = 0u, nzc = 0u;
            if (si != B65_RESIDUAL_STREAM_FILL_NONE && si < n_smul) {
                c_sm0 = (long long)smul3[(size_t)si * 3];
                cy = (unsigned)smul3[(size_t)si * 3 + 1];
                nzc = (unsigned)smul3[(size_t)si * 3 + 2];
            } else {
                off += 1u;
            }
            g_bump_or_off(m_smul, si, &off, drop);
            if (cfg.mutate == B65_RESIDUAL_STREAM_MUT_NOMUL) {
                cy = 0u;
            }
            c_ef = (nzc == 1u)
                       ? (long long)a_ef + (long long)b_ef + (long long)cy - 127LL
                       : 0LL;
            if (nzc == 1u && !(c_ef >= 1 && c_ef < 0xff)) {
                bad_exp += 1u;
            }
            c_sm = c_sm0;
            cols[b][i] = g_m31_of(c_sm);
            cols[b + 1][i] = g_m31_of((long long)cy);
            cols[b + 2][i] = g_m31_of((long long)nzc);
            cols[b + 3][i] = g_m31_of(c_ef);
            nz = nzc;
        }
        (void)nz;
        // the slot's own output value, looked up like any other committed pair
        const unsigned vo = (c_ef >= 0 && c_ef < 0xff)
                                ? g_val_at(c_sm, (unsigned)(c_ef < 0 ? 0 : c_ef))
                                : B65_RESIDUAL_STREAM_FILL_NONE;
        g_bump_or_off(m_val, vo, &off, drop);
    }

    // ---- the outputs the bindings and the next region read --------------------
    for (unsigned k = 0; k < cfg.out_w; ++k) {
        const B65ResidualStreamSlot sl = slots[out_slot[k]];
        const long long sm = g_sgn_of(cols[sl.o_sm][i]);
        const long long e = g_sgn_of(cols[sl.o_ef][i]);
        const unsigned ef = (unsigned)(e < 0 ? 0 : (e > 0xfe ? 0xfe : e));
        // row order again, for the same reason `in_nat` is
        out_pat[(size_t)j * cfg.out_w + k] = g_join(sm, ef);
        out_nat[((size_t)j * cfg.out_w + k) * 2] = (int)sm;
        out_nat[((size_t)j * cfg.out_w + k) * 2 + 1] = (int)e;
    }

    if (off) {
        atomicAdd(&counts[0], off);
    }
    if (bad_exp) {
        atomicAdd(&counts[1], bad_exp);
    }
    if (un_off) {
        atomicAdd(&counts[2], un_off);
    }
}

extern "C" void b65_fill_residual_stream(
    const unsigned *wit, const B65ResidualStreamGroup *groups, const B65ResidualStreamSlot *slots,
    const unsigned *norm3, unsigned n_norm, const int *smul3, unsigned n_smul,
    const int *const *un2, unsigned n_un, unsigned n_val, const unsigned *out_slot,
    unsigned *const *cols, unsigned *out_pat, int *out_nat, int *in_nat,
    B65ResidualStreamMult m_pow, B65ResidualStreamMult m_norm, B65ResidualStreamMult m_val, B65ResidualStreamMult m_frac,
    B65ResidualStreamMult m_smul, const B65ResidualStreamMult *m_un, unsigned *counts, B65ResidualStreamCfg cfg) {
    // The two *arrays of pointers* the caller builds on the host have to be on
    // the card before a kernel can index them -- the same `clone_to_device`
    // `b65_fill_dequant` does for its column lists. Everything else the caller
    // hands over is already device memory.
    const unsigned blocks = (cfg.rows + B65_RESIDUAL_STREAM_FILL_BLOCK - 1u) / B65_RESIDUAL_STREAM_FILL_BLOCK;
    unsigned **d_cols =
        clone_to_device<unsigned *>(const_cast<unsigned **>(cols), cfg.n_cols);
    const int **d_un = nullptr;
    if (n_un > 0u) {
        d_un = clone_to_device<const int *>(const_cast<const int **>(un2), n_un);
    }
    b65_residual_stream_fill_kernel<<<blocks, B65_RESIDUAL_STREAM_FILL_BLOCK>>>(
        wit, groups, slots, norm3, n_norm, smul3, n_smul, d_un, n_val, out_slot,
        d_cols, out_pat, out_nat, in_nat, m_pow, m_norm, m_val, m_frac, m_smul,
        m_un, counts, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_cols);
    if (d_un != nullptr) {
        cuda_free_memory((void *)d_un);
    }
}
