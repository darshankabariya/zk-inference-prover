// B7a milestone 4 (19.6): the dequant witness generation, on the device.
#include "fields.cuh"
#include "utils.cuh"
#include "zk_dequant_witness_gen.cuh"

#define ZK_DEQUANT_WITNESS_GEN_BLOCK 128u
// Slots per block in the y direction. One thread owns a row and loops its
// slots, because consecutive threads then write consecutive cells of the same
// column; splitting the slots across blocks is what keeps a 2^12-row component
// from running on 32 blocks of a 132-SM device.
#define ZK_DEQUANT_WITNESS_GEN_SLOT_TILE 4u

// ---------------------------------------------------------------- the permutation --

// `common.rs`'s `coset_perm`: the closed form of
// `bit_reverse_coset_to_circle_domain_order`, which is an involution, so the
// thread that owns output `i` reads coset row `f(i)` and no swap is needed.
// Asserted against stwo's own function on the host by `coset_perm_selfcheck`.
__device__ __forceinline__ unsigned coset_perm(unsigned i, unsigned log_n) {
    const unsigned n = 1u << log_n;
    const unsigned c = (i & 1u) ? ((2u * n - i - 1u) >> 1) : (i >> 1);
    return __brev(c) >> (32u - log_n);
}

// ------------------------------------------------------------------- field helpers --

// `attn_trace::m31_of`: `(((v % p) + p) % p)`, for a value that may be negative.
__device__ __forceinline__ m31 m31_of_i64(long long v) {
    const long long p = 2147483647LL;
    long long x = v % p;
    if (x < 0) x += p;
    return (m31)x;
}

// ------------------------------------------------------------------ the multiplicities --

// A scatter-add, warp-aggregated: the lanes of a warp that hit the same row
// add once. The dequant's range-table index is `expfield + jx + jw`, which
// takes a handful of distinct values across millions of bumps, so without this
// one address serialises a whole kernel.
__device__ __forceinline__ void bump_at(unsigned *m, unsigned idx) {
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, idx);
    const unsigned leader = __ffs(peers) - 1u;
    if ((threadIdx.x & 31u) == leader) {
        atomicAdd(&m[idx], (unsigned)__popc(peers));
    }
}

// `layer::bump`: in range, count it; out of range, count it as off-table.
__device__ __forceinline__ void bump_range(unsigned *m, unsigned n, long long v,
                                           unsigned *off_table) {
    if (v >= 0 && v < (long long)n) {
        bump_at(m, (unsigned)v);
    } else {
        atomicAdd(off_table, 1u);
    }
}

// --------------------------------------------------------------------- bf16, on device --

// `bf16::pattern`, including its truncation to 16 bits: the host takes
// `(e + 127) as u32` before the shift, so a negative exponent wraps there and
// is masked here in exactly the same way.
__device__ __forceinline__ unsigned bf16_pattern(unsigned sg, unsigned expfield, unsigned m7) {
    return ((sg << 15) | (expfield << 7) | m7) & 0xffffu;
}

// `bf16::to_f32`: bf16 is the f32 prefix, so this is exact, and widening to
// double is exact too.
__device__ __forceinline__ double bf16_to_f64(unsigned bits) {
    return (double)__uint_as_float(bits << 16);
}

// `bf16::from_f64`: one IEEE rounding, as bit surgery. Normal numbers only,
// which is all the dequant produces.
__device__ __forceinline__ unsigned bf16_from_f64(double x, unsigned away = 0u) {
    if (x == 0.0) {
        return signbit(x) ? 0x8000u : 0u;
    }
    const unsigned long long u = (unsigned long long)__double_as_longlong(x);
    const unsigned sign = (unsigned)(u >> 63);
    int e = (int)((u >> 52) & 0x7ffull) - 1023;
    const unsigned long long man = u & ((1ull << 52) - 1ull);
    const unsigned SH = 52u - 7u;
    unsigned long long keep = man >> SH;
    const unsigned rb = (unsigned)((man >> (SH - 1u)) & 1ull);
    const bool st = (man & ((1ull << (SH - 1u)) - 1ull)) != 0ull;
    // `refaway` is the control: ties away from zero instead of to even, which
    // is the other mode `bf16::Round` carries and the plausible way to get
    // this line wrong. It differs on every tie, and an integer scaled by a
    // power of two produces those in quantity.
    const bool up = (rb == 1u) && (away != 0u || st || (keep & 1ull) == 1ull);
    unsigned long long m7 = keep + (up ? 1ull : 0ull);
    if (m7 == 128ull) {
        m7 = 0ull;
        e += 1;
    }
    return bf16_pattern(sign, (unsigned)(e + 127), (unsigned)m7);
}

// ------------------------------------------------------- the two closed-form lookups --

// `bf16::normalise`: `mag = M2 * ph + r`, `0 <= r < ph`. False means the
// magnitude needs a round-bit weight the table does not carry -- the host
// panics there, and the device counts it so the caller can panic with the
// component's name instead of proving something wrong.
__device__ __forceinline__ bool zk_normalise(unsigned long long mag, unsigned kmax,
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

// `Tables::norm_at`, the O(1) index B6.5 wrote to replace a linear scan.
// `ZK_DEQUANT_WITNESS_GEN_NONE` is the host's `None`.
__device__ __forceinline__ unsigned zk_norm_at(unsigned m2, unsigned ph, unsigned st,
                                                unsigned kmax, unsigned sticky) {
    if (m2 == 0u && ph == 0u && st == 0u) {
        return 0u;
    }
    if (m2 >= 1u && m2 < 256u && ph == 1u && st == 0u) {
        return m2;
    }
    if (m2 >= 256u && m2 < 512u && ph != 0u && (ph & (ph - 1u)) == 0u) {
        const unsigned k = (unsigned)(31 - __clz((int)ph));
        if (k > kmax || (st == 1u && sticky == 0u) || st > 1u) {
            return ZK_DEQUANT_WITNESS_GEN_NONE;
        }
        const unsigned per = 1u + sticky;
        return 256u + (k * 256u + (m2 - 256u)) * per + st;
    }
    return ZK_DEQUANT_WITNESS_GEN_NONE;
}

// `Tables::frac_at`: the `(ph, r, st)` table's rows are `ph + r`.
__device__ __forceinline__ unsigned zk_frac_at(unsigned ph, unsigned long long r) {
    if (ph == 0u) {
        return (r == 0ull) ? 0u : ZK_DEQUANT_WITNESS_GEN_NONE;
    }
    return (r < (unsigned long long)ph) ? (unsigned)((unsigned long long)ph + r)
                                        : ZK_DEQUANT_WITNESS_GEN_NONE;
}

// ------------------------------------------------------------------------ the witness generation --

// `counts`: [0] off-table range bumps, [1] `ref_bad`, [2] rows the tables
// cannot describe (the host's two `expect`s).
template <int VARIANT>
__global__ void zk_witness_gen_kernel(const int *__restrict__ acc,
                                const int *__restrict__ jx, const int *__restrict__ jw,
                                const unsigned *__restrict__ sx,
                                const unsigned *__restrict__ sw,
                                const unsigned *__restrict__ norm3,
                                const unsigned *__restrict__ mul2,
                                m31 *const *__restrict__ base_cols,
                                m31 *const *__restrict__ pre_cols,
                                unsigned *__restrict__ m_norm,
                                unsigned *__restrict__ m_frac,
                                unsigned *__restrict__ m_mul,
                                unsigned *__restrict__ m_r8,
                                unsigned *__restrict__ counts,
                                unsigned *__restrict__ want_out,
                                const ZkDequantWitnessGenCfg cfg) {
    const unsigned rows = 1u << cfg.log_rows;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    // the output index owns the row: `j` is where the coset order has it.
    // `noperm` is the control that says so -- without it the column is in
    // coset order and every cell but the fixed points of the involution moves.
    const unsigned j = (cfg.mutate == ZK_MUT_NOPERM) ? i : coset_perm(i, cfg.log_rows);
    const unsigned t = cfg.tokens - 1u - j / cfg.chunks;
    const unsigned b = cfg.chunks - 1u - j % cfg.chunks;

    const unsigned s_begin = blockIdx.y * ZK_DEQUANT_WITNESS_GEN_SLOT_TILE;
    const unsigned s_end = min(cfg.per_row, s_begin + ZK_DEQUANT_WITNESS_GEN_SLOT_TILE);

    if (blockIdx.y == 0u) {
        // the per-row shared columns: the activation's scale
        if (VARIANT == 0) {
            base_cols[cfg.sh0][i] = m31_of_i64((long long)jx[t]);
        } else {
            base_cols[cfg.sh0][i] = sx[2u * t];
            base_cols[cfg.sh1][i] = sx[2u * t + 1u];
        }
        // the five selectors, the last of the preprocessed columns
        const unsigned m = j % cfg.chunks;
        const unsigned last = (m == cfg.chunks - 1u) ? 1u : 0u;
        const unsigned sel0 = cfg.chan_w * cfg.per_row;
        pre_cols[sel0 + 0u][i] = (m != 0u) ? 1u : 0u;                  // nofirst
        pre_cols[sel0 + 1u][i] = last;                                  // last
        pre_cols[sel0 + 2u][i] = (j > 0u && last == 0u) ? 1u : 0u;      // m1
        pre_cols[sel0 + 3u][i] = (j > 0u && last == 1u) ? 1u : 0u;      // m2
        pre_cols[sel0 + 4u][i] = (j + 1u == rows) ? 1u : 0u;            // end
    }

    for (unsigned s = s_begin; s < s_end; ++s) {
        const unsigned c = b * cfg.per_row + s;
        const unsigned col0 = cfg.slot0 + s * cfg.per_slot;

        // the preprocessed per-channel scale columns. `chan` is the control
        // for the slot-to-channel map, which is the one thing the AIR cannot
        // catch on its own: a consistently misaddressed scale is a consistent
        // witness for the wrong matmul.
        const unsigned cs = (cfg.mutate == ZK_MUT_CHAN) ? (c + 1u) % cfg.n_out : c;
        if (VARIANT == 0) {
            pre_cols[s][i] = m31_of_i64((long long)jw[cs]);
        } else {
            pre_cols[2u * s][i] = sw[2u * cs];
            pre_cols[2u * s + 1u][i] = sw[2u * cs + 1u];
        }

        const long long a = (long long)acc[(size_t)t * cfg.n_out + c];
        base_cols[col0 + cfg.f_acc][i] = m31_of_i64(a);
        const unsigned sg = (a < 0) ? 1u : 0u;
        const unsigned long long mag = (unsigned long long)(a < 0 ? -a : a);

        unsigned m2, ph, st;
        unsigned long long r;
        if (!zk_normalise(mag, cfg.kmax, &m2, &ph, &st, &r)) {
            atomicAdd(&counts[2], 1u);
            continue;
        }
        if (cfg.sticky == 0u) {
            st = 0u;
        }
        const unsigned ri = zk_norm_at(m2, ph, st, cfg.kmax, cfg.sticky);
        const unsigned fi = zk_frac_at(ph, r);
        if (ri == ZK_DEQUANT_WITNESS_GEN_NONE || fi == ZK_DEQUANT_WITNESS_GEN_NONE || ri >= cfg.n_norm ||
            fi >= cfg.n_frac) {
            atomicAdd(&counts[2], 1u);
            continue;
        }
        const unsigned nr_m7 = norm3[3u * ri + 0u];
        const unsigned nr_ef = norm3[3u * ri + 1u];
        const unsigned nr_nz = norm3[3u * ri + 2u];
        bump_at(m_norm, ri);
        // `nobump` drops one frac multiplicity in every 1024 rows: the columns
        // still match and only the counter comparison can see it, which is
        // what gives that half of the check its own teeth.
        if (!(cfg.mutate == ZK_MUT_NOBUMP && (i & 1023u) == 0u)) {
            bump_at(m_frac, fi);
        }

        base_cols[col0 + cfg.f_sg][i] = sg;
        base_cols[col0 + cfg.f_m2][i] = m2;
        base_cols[col0 + cfg.f_ph][i] = ph;
        if (cfg.sticky != 0u) {
            base_cols[col0 + cfg.f_st][i] = st;
        }
        base_cols[col0 + cfg.f_r][i] = m31_of_i64((long long)r);
        base_cols[col0 + cfg.f_m7][i] = nr_m7;
        base_cols[col0 + cfg.f_ef][i] = nr_ef;
        base_cols[col0 + cfg.f_nz][i] = nr_nz;

        if (VARIANT == 0) {
            const long long jjx = (long long)jx[t] + (long long)jw[c];
            const long long eo = (nr_nz == 1u) ? ((long long)nr_ef + jjx) : 0LL;
            base_cols[col0 + cfg.f_eo][i] = m31_of_i64(eo);
            bump_range(m_r8, cfg.n_r8, eo, &counts[0]);
            if (cfg.do_ref != 0u) {
                // `a * 2^jjx`: `ldexp` of an exact integer is exact, which is
                // what makes this the same single rounding the host does with
                // `a as f64 * (jjx as f64).exp2()`.
                // `ref32` takes the same value through f32, which is the
                // double rounding `bf16.rs`'s own doc comment warns about --
                // the plausible wrong version of this line.
                const unsigned away = (cfg.mutate == ZK_MUT_REFAWAY) ? 1u : 0u;
                const unsigned want =
                    (cfg.mutate == ZK_MUT_REF32)
                        ? bf16_from_f64((double)ldexpf((float)a, (int)jjx), away)
                        : bf16_from_f64(ldexp((double)a, (int)jjx), away);
                const unsigned got =
                    bf16_pattern(sg, (unsigned)(eo > 0LL ? (unsigned long long)eo : 0ull),
                                 nr_m7);
                if (want != got) {
                    atomicAdd(&counts[1], 1u);
                }
                if (want_out != nullptr) {
                    want_out[(size_t)j * cfg.per_row + s] = want;
                }
            }
        } else {
            unsigned m_in = nr_m7;
            long long e_in = (long long)nr_ef;
            for (unsigned tt = 0u; tt < 2u; ++tt) {
                const unsigned ms = (tt == 0u) ? sx[2u * t] : sw[2u * c];
                const long long es = (tt == 0u) ? (long long)sx[2u * t + 1u]
                                                : (long long)sw[2u * c + 1u];
                const unsigned mi = m_in * 128u + ms;
                unsigned po = 0u, ca = 0u;
                if (mi >= cfg.n_mul) {
                    atomicAdd(&counts[2], 1u);
                    break;
                }
                if (nr_nz == 1u) {
                    po = mul2[2u * mi + 0u];
                    ca = mul2[2u * mi + 1u];
                }
                const long long eo =
                    (nr_nz == 1u) ? (e_in + es - 127LL + (long long)ca) : 0LL;
                base_cols[col0 + cfg.f_mul[3u * tt + 0u]][i] = po;
                base_cols[col0 + cfg.f_mul[3u * tt + 1u]][i] = ca;
                base_cols[col0 + cfg.f_mul[3u * tt + 2u]][i] = m31_of_i64(eo);
                if (nr_nz == 1u) {
                    bump_at(m_mul, mi);
                }
                bump_range(m_r8, cfg.n_r8, eo, &counts[0]);
                m_in = po;
                e_in = eo;
            }
            if (cfg.do_ref != 0u) {
                // Two `double` multiplies, each followed by a bf16 rounding.
                // There is no add for the compiler to contract into an FMA, and
                // `__dmul_rn` says so rather than relying on `-fmad`.
                const unsigned away = (cfg.mutate == ZK_MUT_REFAWAY) ? 1u : 0u;
                const double y0 =
                    bf16_to_f64(bf16_from_f64((double)(long long)mag, away));
                const double sxv = bf16_to_f64(bf16_pattern(0u, sx[2u * t + 1u], sx[2u * t]));
                const double swv = bf16_to_f64(bf16_pattern(0u, sw[2u * c + 1u], sw[2u * c]));
                const double t1 = bf16_to_f64(bf16_from_f64(__dmul_rn(y0, sxv), away));
                const unsigned want =
                    ((cfg.mutate == ZK_MUT_REF32)
                         ? bf16_from_f64((double)__fmul_rn((float)t1, (float)swv), away)
                         : bf16_from_f64(__dmul_rn(t1, swv), away))
                    | (sg << 15);
                const unsigned got = bf16_pattern(
                    sg, (unsigned)(e_in > 0LL ? (unsigned long long)e_in : 0ull), m_in);
                if (want != got) {
                    atomicAdd(&counts[1], 1u);
                }
                if (want_out != nullptr) {
                    want_out[(size_t)j * cfg.per_row + s] = want;
                }
            }
        }
    }
}

extern "C" void zk_witness_gen_dequant(const int *acc,
                                 const int *jx, const int *jw,
                                 const unsigned *sx, const unsigned *sw,
                                 const unsigned *norm3, const unsigned *mul2,
                                 m31 *const *base_cols, unsigned n_base,
                                 m31 *const *pre_cols, unsigned n_pre,
                                 unsigned *m_norm, unsigned *m_frac,
                                 unsigned *m_mul, unsigned *m_r8,
                                 unsigned *counts, unsigned *want_out,
                                 ZkDequantWitnessGenCfg cfg) {
    const unsigned rows = 1u << cfg.log_rows;
    m31 **d_base = clone_to_device<m31 *>(const_cast<m31 **>(base_cols), n_base);
    m31 **d_pre = clone_to_device<m31 *>(const_cast<m31 **>(pre_cols), n_pre);
    dim3 grid((rows + ZK_DEQUANT_WITNESS_GEN_BLOCK - 1u) / ZK_DEQUANT_WITNESS_GEN_BLOCK,
              (cfg.per_row + ZK_DEQUANT_WITNESS_GEN_SLOT_TILE - 1u) / ZK_DEQUANT_WITNESS_GEN_SLOT_TILE);
    if (cfg.variant == 0u) {
        zk_witness_gen_kernel<0><<<grid, ZK_DEQUANT_WITNESS_GEN_BLOCK>>>(
            acc, jx, jw, sx, sw, norm3, mul2, d_base, d_pre, m_norm, m_frac, m_mul,
            m_r8, counts, want_out, cfg);
    } else {
        zk_witness_gen_kernel<1><<<grid, ZK_DEQUANT_WITNESS_GEN_BLOCK>>>(
            acc, jx, jw, sx, sw, norm3, mul2, d_base, d_pre, m_norm, m_frac, m_mul,
            m_r8, counts, want_out, cfg);
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_base);
    cuda_free_memory(d_pre);
}
