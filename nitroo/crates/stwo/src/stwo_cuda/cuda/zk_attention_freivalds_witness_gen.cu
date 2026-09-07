// B7a milestone 4 (section 3 row 1): the Freivalds check, attnout and key tile witness generations, on the
// device.
#include "fields.cuh"
#include "utils.cuh"
#include "zk_attention_freivalds_witness_gen.cuh"

#define ZK_WITNESS_BLOCK 128u
// Slots per block in the y direction, as `ZK_DEQUANT_WITNESS_GEN_SLOT_TILE` is for the
// dequant: one thread owns a row and loops its slots, so consecutive threads
// write consecutive cells of the same column, and splitting the slots across
// blocks keeps a small component off 32 blocks of a 132-SM device.
#define ZK_WITNESS_TILE 4u

// ---------------------------------------------------------------- the permutation --

// `common.rs`'s `m4_perm`, verbatim from `zk_dequant_witness_gen.cu`: the closed form of
// `bit_reverse_coset_to_circle_domain_order`, an involution, so the thread that
// owns output `i` reads coset row `f(i)`.
__device__ __forceinline__ unsigned zk_witness_perm(unsigned i, unsigned log_n) {
    const unsigned n = 1u << log_n;
    const unsigned c = (i & 1u) ? ((2u * n - i - 1u) >> 1) : (i >> 1);
    return __brev(c) >> (32u - log_n);
}

// The coset row this thread's committed row holds. `noperm` is the control that
// says the permutation is load-bearing: without it every cell but the fixed
// points of the involution moves.
__device__ __forceinline__ unsigned zk_witness_row(unsigned i, unsigned log_n, unsigned mutate) {
    return (mutate == ZK_WITNESS_MUT_NOPERM) ? i : zk_witness_perm(i, log_n);
}

// ------------------------------------------------------------------- field helpers --

// `attn_trace::m31_of` / `layer::m31_of`: `(((v % p) + p) % p)`.
__device__ __forceinline__ m31 zk_witness_m31(long long v) {
    const long long p = 2147483647LL;
    long long x = v % p;
    if (x < 0) x += p;
    return (m31)x;
}

// A 64-bit witness value, as two 32-bit loads. The witness arrays reach the
// device through `Uint32Vec`, whose allocator promises 4-byte alignment and not
// 8, so a `long long*` cast would be undefined on a sub-allocated pointer. Two
// loads cost nothing here and are always correct.
__device__ __forceinline__ long long zk_witness_i64(const unsigned *p, size_t i) {
    const unsigned lo = p[2u * i];
    const unsigned hi = p[2u * i + 1u];
    return (long long)(((unsigned long long)hi << 32) | (unsigned long long)lo);
}

// ------------------------------------------------------------------- the layouts --

// `Grid::row`: `(group, chunk)` of a row, or "inert" for a lead row. Chunks
// count down so a Horner walking the rows in order leaves the group's total on
// its last row -- `descend` is the control for exactly that, and it is B3's
// trap inherited verbatim (`layout.rs`'s own words).
__device__ __forceinline__ bool zk_witness_grid_row(const ZkGrid &g, unsigned j,
                                              unsigned *grp, unsigned *chunk,
                                              unsigned mutate) {
    if (j < g.lead) {
        return false;
    }
    const unsigned idx = j - g.lead;
    unsigned gi = idx / g.chunks;
    if (g.descending != 0u) {
        gi = g.groups - 1u - gi;
    }
    *grp = gi;
    const unsigned m = idx % g.chunks;
    *chunk = (mutate == ZK_WITNESS_MUT_DESCEND) ? m : (g.chunks - 1u - m);
    return true;
}

// `attn_trace::split`: `(instance, local)` of a packed group index. `inst` is
// the control 19.6 asked for by name -- it is invisible on a one-instance key tile
// and bites on every other one, which is what makes it worth having.
__device__ __forceinline__ void zk_witness_split(unsigned g, unsigned n, unsigned insts,
                                           unsigned *inst, unsigned *local,
                                           unsigned mutate) {
    unsigned i = g / n;
    if (mutate == ZK_WITNESS_MUT_INST && insts > 1u) {
        i = (i + 1u) % insts;
    }
    *inst = i;
    *local = g % n;
}

// `AoLayout::at`: row -> (head, token, chunk), or inert.
__device__ __forceinline__ bool zk_witness_ao_at(const ZkAoCfg &cfg, unsigned j,
                                           unsigned *h, unsigned *t, unsigned *b) {
    if (j < cfg.lead) {
        return false;
    }
    const unsigned ja = j - cfg.lead;
    const unsigned per_head = cfg.s * cfg.chunks;
    *h = cfg.nh - 1u - ja / per_head;
    const unsigned rest = ja % per_head;
    *t = cfg.s - 1u - rest / cfg.chunks;
    const unsigned m = rest % cfg.chunks;
    *b = (cfg.mutate == ZK_WITNESS_MUT_DESCEND) ? m : (cfg.chunks - 1u - m);
    return true;
}

// `attn_trace::tok_of`.
__device__ __forceinline__ bool zk_witness_tok(const ZkKeyTileCfg &cfg, unsigned j, unsigned i,
                                         unsigned *t) {
    const unsigned v = (cfg.rows_tok - 1u - j) * cfg.rw + i;
    *t = v;
    return v < cfg.toks;
}

// ------------------------------------------------------------- the multiplicities --

// The warp-aggregated scatter-add of 20.1, unchanged: the lanes of a warp that
// hit the same row add once.
__device__ __forceinline__ void zk_witness_bump_at(unsigned *m, unsigned idx) {
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, idx);
    const unsigned leader = __ffs(peers) - 1u;
    if ((threadIdx.x & 31u) == leader) {
        atomicAdd(&m[idx], (unsigned)__popc(peers));
    }
}

// `attn_trace::bump_slice` and `layer::bump`, which are the same function: in
// range, count it; out of range, count it as off-table. A null counter array is
// the host's "this witness generation does not bump that table at all" -- `range_check` off,
// or a variant with no auxiliary column -- and skips the call entirely, which
// is what the host does too, so it must not touch `off_table` either.
__device__ __forceinline__ void zk_witness_bump(ZkMult mm, long long v, unsigned *off) {
    if (mm.m == nullptr) {
        return;
    }
    if (v >= 0 && v < (long long)mm.n) {
        zk_witness_bump_at(mm.m, (unsigned)v);
    } else {
        atomicAdd(off, 1u);
    }
}

// `nobump` drops one bump in every 1,024 committed rows. The columns still
// match and only the counter comparison can see it, which is what gives that
// half of the check its own teeth (20.2's shape).
__device__ __forceinline__ bool zk_witness_drop(unsigned mutate, unsigned i) {
    return mutate == ZK_WITNESS_MUT_NOBUMP && (i & 1023u) == 0u;
}

// ========================================================= the Freivalds witness generation ==

// `freivalds_component`'s committed half: `pq` activation columns, every cell bumped
// into the 8-bit range table (padding included -- the host bumps outside the
// match, and a padded slot commits the offset 128, which is a real table row).
__global__ void zk_witness_freivalds_base(const signed char *__restrict__ qx,
                                    m31 *const *__restrict__ base_cols,
                                    ZkMult m_r8, unsigned *__restrict__ counts,
                                    const ZkFreivaldsCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.g.rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.g.log_rows, cfg.mutate);
    unsigned k, chunk;
    const bool live = zk_witness_grid_row(cfg.g, j, &k, &chunk, cfg.mutate);

    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(cfg.pq, s0 + ZK_WITNESS_TILE);
    for (unsigned p = s0; p < s1; ++p) {
        int v = 128;
        if (live) {
            const unsigned t = chunk * cfg.pq + p;
            if (t < cfg.tokens) {
                v = (int)qx[(size_t)t * cfg.n_in + k] + 128;
            }
        }
        if (!zk_witness_drop(cfg.mutate, i)) {
            zk_witness_bump(m_r8, (long long)v, &counts[0]);
        }
        base_cols[p][i] = zk_witness_m31((long long)v);
    }
}

// `freivalds_component`'s preprocessed half: the weights of every matmul in the group,
// then the four selectors. `chan` is the control for the slot-to-channel map --
// a consistently misaddressed weight is a consistent witness for the wrong
// matmul, and the AIR cannot catch it on its own.
__global__ void zk_witness_freivalds_pre(const signed char *const *__restrict__ qw,
                                   const unsigned *__restrict__ n_out,
                                   m31 *const *__restrict__ pre_cols,
                                   const ZkFreivaldsCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.g.rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.g.log_rows, cfg.mutate);
    unsigned k, chunk;
    const bool live = zk_witness_grid_row(cfg.g, j, &k, &chunk, cfg.mutate);

    const unsigned n_w = cfg.n_mats * cfg.pk;
    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(n_w, s0 + ZK_WITNESS_TILE);
    for (unsigned q = s0; q < s1; ++q) {
        const unsigned mi = q / cfg.pk;
        unsigned p = q % cfg.pk;
        if (cfg.mutate == ZK_WITNESS_MUT_CHAN) {
            p = (p + 1u) % cfg.pk;
        }
        m31 v = 0u;
        if (live) {
            const unsigned n = n_out[mi];
            const unsigned c = chunk * cfg.pk + p;
            if (c < n) {
                v = zk_witness_m31((long long)qw[mi][(size_t)k * n + c]);
            }
        }
        pre_cols[q][i] = v;
    }

    if (blockIdx.y == 0u) {
        // the four selectors, in `freivalds_component`'s order
        pre_cols[n_w + 0u][i] = (live && chunk + 1u < cfg.g.chunks) ? 1u : 0u;
        pre_cols[n_w + 1u][i] = (live && chunk == 0u) ? 1u : 0u;
        pre_cols[n_w + 2u][i] = (j > 0u) ? 1u : 0u;
        pre_cols[n_w + 3u][i] = (j + 1u == cfg.g.rows) ? 1u : 0u;
    }
}

extern "C" void zk_witness_gen_freivalds(const signed char *qx,
                                   const signed char *const *qw, const unsigned *n_out,
                                   m31 *const *base_cols, unsigned n_base,
                                   m31 *const *pre_cols, unsigned n_pre,
                                   ZkMult m_r8, unsigned *counts,
                                   ZkFreivaldsCfg cfg) {
    m31 **d_base = clone_to_device<m31 *>(const_cast<m31 **>(base_cols), n_base);
    m31 **d_pre = clone_to_device<m31 *>(const_cast<m31 **>(pre_cols), n_pre);
    signed char **d_qw =
        clone_to_device<signed char *>(const_cast<signed char **>(qw), cfg.n_mats);
    unsigned *d_nout = clone_to_device<unsigned>(const_cast<unsigned *>(n_out), cfg.n_mats);
    const unsigned bx = (cfg.g.rows + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK;
    dim3 gb(bx, (cfg.pq + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_freivalds_base<<<gb, ZK_WITNESS_BLOCK>>>(qx, d_base, m_r8, counts, cfg);
    const unsigned n_w = cfg.n_mats * cfg.pk;
    dim3 gp(bx, (n_w + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_freivalds_pre<<<gp, ZK_WITNESS_BLOCK>>>(d_qw, d_nout, d_pre, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_base);
    cuda_free_memory(d_pre);
    cuda_free_memory(d_qw);
    cuda_free_memory(d_nout);
}

// =========================================================== the attnout witness generation ==

// `ao_component`, whose per-row body is the rounding fold: `qm` is the nearest
// integer to `|a| 2^e / den` with ties away from zero, written as the exact
// integer division the AIR checks, and `d` is the remainder that division
// leaves. `counts[2]` carries the host's two `assert!`s -- a non-positive
// denominator and the no-wrap window -- so the caller can panic with the
// component's name instead of proving something wrong.
__global__ void zk_witness_ao(const unsigned *__restrict__ acc,
                        const unsigned *__restrict__ den,
                        const int *__restrict__ jo, const int *__restrict__ jv,
                        m31 *const *__restrict__ base_cols,
                        m31 *const *__restrict__ pre_cols,
                        ZkMult m_qnz, ZkMult m_rem,
                        ZkMult m_acc_lo, ZkMult m_acc_hi,
                        int *__restrict__ q_out, unsigned *__restrict__ want_out,
                        unsigned *__restrict__ counts, const ZkAoCfg cfg) {
    const unsigned rows = 1u << cfg.log_rows;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.log_rows, cfg.mutate);
    const unsigned pr = cfg.per_row;
    unsigned h, t, b;
    const bool live = zk_witness_ao_at(cfg, j, &h, &t, &b);

    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(pr, s0 + ZK_WITNESS_TILE);

    // stream JJ (E11): the accumulator window's offset, and the exponent
    // window's lower end.  `e_lo = jv_min - jo_max` is the `e` the inert lead
    // rows commit, since they commit `jvp = 0` and `joff = 0`.
    const long long acc_off = 1LL << (cfg.acc_lo_bits + cfg.acc_hi_bits - 1u);
    const long long lo_mask = (1LL << cfg.acc_lo_bits) - 1LL;
    const int e_lo = cfg.jv_min - cfg.jo_max;

    if (blockIdx.y == 0u) {
        // the shared denominator column; an inert lead row commits 1, which is
        // what makes every constraint hold and every lookup land on a real row
        base_cols[0][i] = live ? zk_witness_m31(zk_witness_i64(den, (size_t)h * cfg.s + t)) : 1u;
        // stream JJ (E11): `joff = jo_max - jo`, the per-token output exponent,
        // committed because `jo` is prover data.  An inert lead row commits 0.
        base_cols[1][i] = live ? zk_witness_m31((long long)(cfg.jo_max - jo[t])) : 0u;
    }

    for (unsigned s = s0; s < s1; ++s) {
        const unsigned c0 = 2u + 9u * s;
        // the preprocessed `jvp = jv - jv_min`, a per-(head, channel) model
        // constant.  `chan` misaddresses it by one slot, which is this family's
        // version of 20.2's `chan`.
        unsigned sp = s;
        if (cfg.mutate == ZK_WITNESS_MUT_CHAN) {
            sp = (s + 1u) % pr;
        }
        pre_cols[s][i] = live
            ? zk_witness_m31((long long)(jv[(size_t)h * cfg.hd + b * pr + sp] - cfg.jv_min))
            : 0u;

        if (!live) {
            // the inert lead row: acc = 0, den = 1, joff = 0, so the `EPOW` key
            // is 0 and the pair is `(2^max(e_lo,0), 2^max(-e_lo,0))`; then
            // denp = den pj = pj, d = denp, dp = 2 pj - 1 - pj = pj - 1
            const long long num0 = 1LL << (e_lo > 0 ? e_lo : 0);
            const long long pj0 = 1LL << (e_lo < 0 ? -e_lo : 0);
            base_cols[c0 + 0u][i] = 0u;
            base_cols[c0 + 1u][i] = 0u;
            base_cols[c0 + 2u][i] = 0u;
            base_cols[c0 + 3u][i] = 0u;
            base_cols[c0 + 4u][i] = zk_witness_m31(pj0);
            base_cols[c0 + 5u][i] = zk_witness_m31(pj0);
            base_cols[c0 + 6u][i] = zk_witness_m31(num0);
            base_cols[c0 + 7u][i] = zk_witness_m31(pj0);
            base_cols[c0 + 8u][i] = zk_witness_m31(acc_off & lo_mask);
            if (!zk_witness_drop(cfg.mutate, i)) {
                zk_witness_bump(m_qnz, 0LL, &counts[0]);
            }
            zk_witness_bump(m_rem, pj0, &counts[0]);
            zk_witness_bump(m_rem, pj0 - 1LL, &counts[0]);
            zk_witness_bump(m_rem, pj0, &counts[0]);   // denp
            zk_witness_bump(m_acc_lo, acc_off & lo_mask, &counts[0]);
            zk_witness_bump(m_acc_hi, acc_off >> cfg.acc_lo_bits, &counts[0]);
            continue;
        }

        const unsigned ch = b * pr + s;
        const size_t idx = ((size_t)h * cfg.s + t) * cfg.hd + ch;
        const long long a = zk_witness_i64(acc, idx);
        const long long dv = zk_witness_i64(den, (size_t)h * cfg.s + t);
        if (dv <= 0) {
            atomicAdd(&counts[2], 1u);
            continue;
        }
        const int e = jv[(size_t)h * cfg.hd + ch] - jo[t];
        const long long num = 1LL << (e > 0 ? e : 0);
        const long long pj = 1LL << (e < 0 ? -e : 0);
        const long long denp = dv * pj;
        // stream JJ (E11): the offset accumulator and its low limb
        const long long av = a + acc_off;
        if (av < 0LL || av >= (1LL << (cfg.acc_lo_bits + cfg.acc_hi_bits))) {
            atomicAdd(&counts[2], 1u);
            continue;
        }
        const long long alo = av & lo_mask;
        const long long mag = a < 0 ? -a : a;
        const long long pm = mag * num;
        if (2LL * pm >= (1LL << 31) - 1LL || 2LL * denp >= (1LL << 31) - 1LL) {
            atomicAdd(&counts[2], 1u);
            continue;
        }
        const long long qm = (2LL * pm + denp) / (2LL * denp);
        const long long sg = (a < 0 && qm != 0) ? 1LL : 0LL;
        const long long d = (a < 0 && qm == 0) ? (denp - 2LL * pm)
                                               : (2LL * pm - (2LL * qm - 1LL) * denp);
        base_cols[c0 + 0u][i] = zk_witness_m31(a);
        base_cols[c0 + 1u][i] = zk_witness_m31(sg);
        base_cols[c0 + 2u][i] = zk_witness_m31(qm);
        base_cols[c0 + 3u][i] = zk_witness_m31(qm != 0 ? 1LL : 0LL);
        base_cols[c0 + 4u][i] = zk_witness_m31(denp);
        base_cols[c0 + 5u][i] = zk_witness_m31(d);
        base_cols[c0 + 6u][i] = zk_witness_m31(num);
        base_cols[c0 + 7u][i] = zk_witness_m31(pj);
        base_cols[c0 + 8u][i] = zk_witness_m31(alo);
        if (!zk_witness_drop(cfg.mutate, i)) {
            zk_witness_bump(m_qnz, qm, &counts[0]);
        }
        zk_witness_bump(m_rem, d, &counts[0]);
        zk_witness_bump(m_rem, 2LL * denp - 1LL - d, &counts[0]);
        // witness-slack section 7 item 1: `denp`'s own range check
        zk_witness_bump(m_rem, denp, &counts[0]);
        // stream JJ (E11): the other half of the no-wrap window.  The `EPOW`
        // window's own multiplicities are not bumped here -- `e = jv - jo` is a
        // function of the witness and the layout, so the host builds that whole
        // column beside the kernel and nothing has to come back off the card.
        zk_witness_bump(m_acc_lo, alo, &counts[0]);
        zk_witness_bump(m_acc_hi, (av - alo) >> cfg.acc_lo_bits, &counts[0]);
        q_out[idx] = (int)(a < 0 ? -qm : qm);

        if (cfg.do_ref != 0u) {
            // `(mag * 2^e) / den + 0.5`, floored: one f64 rounding per
            // operation and no add for the compiler to contract into an FMA.
            // `ldexp(1.0, e)` is `(e as f64).exp2()` exactly -- both are the
            // representable power of two -- and `aoref` is the control: bindings to
            // even instead of half away from zero, which an integer over an
            // integer denominator produces in quantity.
            const double sc = __dmul_rn((double)mag, ldexp(1.0, e));
            const double q = __ddiv_rn(sc, (double)dv);
            const long long want = (cfg.mutate == ZK_WITNESS_MUT_AOREF)
                                       ? (long long)rint(q)
                                       : (long long)floor(__dadd_rn(q, 0.5));
            if (want != qm) {
                atomicAdd(&counts[1], 1u);
            }
            if (want_out != nullptr) {
                if (want < 0LL || want > 0xffffffffLL) {
                    atomicAdd(&counts[2], 1u);
                }
                want_out[(size_t)j * pr + s] = (unsigned)want;
            }
        }
    }

    if (blockIdx.y == 0u) {
        // the six selectors, in `ao_component`'s order
        const bool last = live && b == 0u;
        const bool hfirst = last && t == cfg.s - 1u;
        m31 *const *sel = pre_cols + pr;
        sel[0][i] = (live && b + 1u < cfg.chunks) ? 1u : 0u;
        sel[1][i] = last ? 1u : 0u;
        sel[2][i] = (j > 0u && !last) ? 1u : 0u;
        sel[3][i] = (j > 0u && last && !hfirst) ? 1u : 0u;
        sel[4][i] = (j > 0u && last && hfirst) ? 1u : 0u;
        sel[5][i] = (j + 1u == rows) ? 1u : 0u;
    }
}

extern "C" void zk_witness_gen_attnout(const unsigned *acc, const unsigned *den,
                                 const int *jo, const int *jv,
                                 m31 *const *base_cols, unsigned n_base,
                                 m31 *const *pre_cols, unsigned n_pre,
                                 ZkMult m_qnz, ZkMult m_rem,
                                 ZkMult m_acc_lo, ZkMult m_acc_hi,
                                 int *q_out, unsigned *want_out,
                                 unsigned *counts, ZkAoCfg cfg) {
    m31 **d_base = clone_to_device<m31 *>(const_cast<m31 **>(base_cols), n_base);
    m31 **d_pre = clone_to_device<m31 *>(const_cast<m31 **>(pre_cols), n_pre);
    const unsigned rows = 1u << cfg.log_rows;
    dim3 g((rows + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK,
           (cfg.per_row + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_ao<<<g, ZK_WITNESS_BLOCK>>>(acc, den, jo, jv, d_base, d_pre, m_qnz, m_rem,
                               m_acc_lo, m_acc_hi, q_out, want_out, counts, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_base);
    cuda_free_memory(d_pre);
}

// ============================================================== the key tile witness generation ==

// The channel component: `pk` key columns then `pq` query columns, each cell
// offset into its range table so the committed value is non-negative.
__global__ void zk_witness_key_tile_chan(const int *__restrict__ kk, const int *__restrict__ qq,
                               m31 *const *__restrict__ cols,
                               ZkMult m_k, ZkMult m_q,
                               unsigned *__restrict__ counts, const ZkKeyTileCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.gc.rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.gc.log_rows, cfg.mutate);
    unsigned g, chunk;
    const bool live = zk_witness_grid_row(cfg.gc, j, &g, &chunk, cfg.mutate);
    unsigned inst = 0u, ch = 0u;
    if (live) {
        zk_witness_split(g, cfg.hd, cfg.insts, &inst, &ch, cfg.mutate);
    }

    const unsigned nslots = max(cfg.pk, cfg.pq);
    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(nslots, s0 + ZK_WITNESS_TILE);
    for (unsigned p = s0; p < s1; ++p) {
        if (p < cfg.pk) {
            long long v = (long long)cfg.off_k;
            if (live) {
                unsigned pp = (cfg.mutate == ZK_WITNESS_MUT_CHAN) ? (p + 1u) % cfg.pk : p;
                const unsigned c = chunk * cfg.pk + pp;
                if (c < cfg.c) {
                    v = (long long)kk[((size_t)inst * cfg.c + c) * cfg.hd + ch]
                        + (long long)cfg.off_k;
                }
            }
            if (cfg.range_check != 0u && !zk_witness_drop(cfg.mutate, i)) {
                zk_witness_bump(m_k, v, &counts[0]);
            }
            cols[p][i] = zk_witness_m31(v);
        }
        if (p < cfg.pq) {
            long long v = (long long)cfg.off_q;
            if (live) {
                const unsigned t = chunk * cfg.pq + p;
                if (t < cfg.s) {
                    v = (long long)qq[((size_t)inst * cfg.s + t) * cfg.hd + ch]
                        + (long long)cfg.off_q;
                }
            }
            if (cfg.range_check != 0u) {
                zk_witness_bump(m_q, v, &counts[0]);
            }
            cols[cfg.pk + p][i] = zk_witness_m31(v);
        }
    }
}

// `Attn::col_idx`, `aux_of` and `table_key` in one place, because they are three
// readings of the same scaled shift and getting them out of step is the bug the
// key tile packing exists to make visible.
__device__ __forceinline__ void zk_witness_grid_vals(const ZkKeyTileCfg &cfg,
                                               const int *dd, const unsigned short *xx,
                                               const unsigned *sc,
                                               const unsigned *delta,
                                               size_t gidx, unsigned inst,
                                               long long *idx, long long *aux,
                                               long long *key) {
    if (cfg.variant == ZK_WITNESS_VAR_BF16X) {
        *idx = zk_witness_i64(sc, gidx);
        *aux = (long long)xx[gidx];
        *key = (long long)xx[gidx];
        return;
    }
    const long long scaled = (long long)dd[gidx] * zk_witness_i64(delta, inst);
    if (cfg.variant == ZK_WITNESS_VAR_INTEXP) {
        *idx = scaled;
        *aux = 0LL;
    } else {
        *idx = scaled >> cfg.split_k;
        *aux = scaled & ((1LL << cfg.split_k) - 1LL);
    }
    *key = *idx;
}

// The grid component: the committed index, the numerator, the auxiliary limb the
// split variant needs, and the packed V values.
__global__ void zk_witness_key_tile_grid(const int *__restrict__ dd,
                               const unsigned short *__restrict__ num,
                               const unsigned short *__restrict__ xx,
                               const unsigned *__restrict__ sc,
                               const unsigned *__restrict__ delta,
                               const int *__restrict__ vv, long long table0,
                               m31 *const *__restrict__ cols, unsigned n_aux,
                               ZkMult m_exp, ZkMult m_aux, ZkMult m_v,
                               unsigned *__restrict__ counts, const ZkKeyTileCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.gk.rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.gk.log_rows, cfg.mutate);
    unsigned g, chunk;
    const bool live = zk_witness_grid_row(cfg.gk, j, &g, &chunk, cfg.mutate);
    unsigned inst = 0u, c = 0u;
    if (live) {
        zk_witness_split(g, cfg.c, cfg.insts, &inst, &c, cfg.mutate);
    }

    const unsigned nslots = max(cfg.kq, cfg.pv);
    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(nslots, s0 + ZK_WITNESS_TILE);
    for (unsigned p = s0; p < s1; ++p) {
        if (p < cfg.kq) {
            // `grid_at(j, p)`: the flat (instance, token, key) entry, or none
            bool have = false;
            bool vis = true;
            size_t gidx = 0;
            if (live) {
                const unsigned t = chunk * cfg.kq + p;
                if (t < cfg.s) {
                    gidx = ((size_t)inst * cfg.s + t) * cfg.c + c;
                    have = true;
                    // B11 stream N, lever 2: `Attn::visible`, which is
                    // `g % c <= (g / c) % s` and here is just `c <= t`. On the
                    // diagonal block a slot whose key runs past its token is
                    // masked, and a masked slot is not a lookup at all -- the
                    // AIR reads it at multiplicity zero -- so it bumps nothing
                    // and cannot be off-table. This is the *only* thing the
                    // mask changes for the witness generation: the committed value below is
                    // `col_idx(g)` either way, which is why a diagonal block key tile's
                    // columns matched cell for cell before this line existed
                    // and only its counters did not.
                    vis = (cfg.diagonal_block == 0u) || (cfg.mutate == ZK_WITNESS_MUT_NOMASK)
                          || (c <= t);
                }
            }
            long long idx = 0LL, aux = 0LL, key = 0LL;
            if (have) {
                zk_witness_grid_vals(cfg, dd, xx, sc, delta, gidx, inst, &idx, &aux, &key);
            }
            // padding takes index 0, a real table row, so multiplicities stay
            // exact -- and the bump is unconditional, `range_check` or not
            if (!zk_witness_drop(cfg.mutate, i) && vis) {
                zk_witness_bump(m_exp, have ? key : 0LL, &counts[0]);
            }
            cols[p][i] = zk_witness_m31(have ? idx : 0LL);
            // the numerator as the witness holds it, never as the table would
            // give it: `Corrupt::Numerator` is a legal table output in the
            // wrong place, and recomputing would repair it
            cols[cfg.kq + p][i] = zk_witness_m31(have ? (long long)num[gidx] : table0);
            if (p < n_aux) {
                if (cfg.variant == ZK_WITNESS_VAR_INTSPLIT) {
                    zk_witness_bump(m_aux, have ? aux : 0LL, &counts[0]);
                }
                cols[2u * cfg.kq + p][i] = zk_witness_m31(have ? aux : 0LL);
            }
        }
        if (p < cfg.pv) {
            long long v = 128LL;
            if (live) {
                const unsigned ch = chunk * cfg.pv + p;
                if (ch < cfg.hd) {
                    v = (long long)vv[((size_t)inst * cfg.c + c) * cfg.hd + ch] + 128LL;
                }
            }
            if (cfg.range_check != 0u) {
                zk_witness_bump(m_v, v, &counts[0]);
            }
            cols[2u * cfg.kq + n_aux + p][i] = zk_witness_m31(v);
        }
    }
}

extern "C" void zk_witness_gen_key_tile_chan(const int *kk, const int *qq,
                                   m31 *const *cols,
                                   ZkMult m_k, ZkMult m_q,
                                   unsigned *counts, ZkKeyTileCfg cfg) {
    const unsigned n = cfg.pk + cfg.pq;
    m31 **d = clone_to_device<m31 *>(const_cast<m31 **>(cols), n);
    dim3 g((cfg.gc.rows + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK,
           (max(cfg.pk, cfg.pq) + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_key_tile_chan<<<g, ZK_WITNESS_BLOCK>>>(kk, qq, d, m_k, m_q, counts, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d);
}

extern "C" void zk_witness_gen_key_tile_grid(const int *dd, const unsigned short *num,
                                   const unsigned short *xx, const unsigned *sc,
                                   const unsigned *delta, const int *vv,
                                   long long table0,
                                   m31 *const *cols, unsigned n_aux,
                                   ZkMult m_exp, ZkMult m_aux, ZkMult m_v,
                                   unsigned *counts, ZkKeyTileCfg cfg) {
    const unsigned n = 2u * cfg.kq + n_aux + cfg.pv;
    m31 **d = clone_to_device<m31 *>(const_cast<m31 **>(cols), n);
    dim3 g((cfg.gk.rows + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK,
           (max(cfg.kq, cfg.pv) + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_key_tile_grid<<<g, ZK_WITNESS_BLOCK>>>(dd, num, xx, sc, delta, vv, table0, d, n_aux,
                                      m_exp, m_aux, m_v, counts, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d);
}

// The A.V outputs. The composed proof moves these into the attnout component and
// passes `with_out = false`, so this kernel exists for the standalone path and
// costs nothing when it is not called.
__global__ void zk_witness_key_tile_out(const unsigned *__restrict__ out,
                              m31 *const *__restrict__ cols, const ZkKeyTileCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.go.rows) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.go.log_rows, cfg.mutate);
    unsigned g, chunk;
    const bool live = zk_witness_grid_row(cfg.go, j, &g, &chunk, cfg.mutate);
    unsigned inst = 0u, ch = 0u;
    if (live) {
        zk_witness_split(g, cfg.hd, cfg.insts, &inst, &ch, cfg.mutate);
    }
    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(cfg.oq, s0 + ZK_WITNESS_TILE);
    for (unsigned p = s0; p < s1; ++p) {
        long long v = 0LL;
        if (live) {
            const unsigned t = chunk * cfg.oq + p;
            if (t < cfg.s) {
                v = zk_witness_i64(out, ((size_t)inst * cfg.s + t) * cfg.hd + ch);
            }
        }
        cols[p][i] = zk_witness_m31(v);
    }
}

extern "C" void zk_witness_gen_key_tile_out(const unsigned *out, m31 *const *cols,
                                  unsigned *counts, ZkKeyTileCfg cfg) {
    (void)counts;
    m31 **d = clone_to_device<m31 *>(const_cast<m31 **>(cols), cfg.oq);
    dim3 g((cfg.go.rows + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK,
           (cfg.oq + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_key_tile_out<<<g, ZK_WITNESS_BLOCK>>>(out, d, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d);
}

// The per-token component: the row maxima, the denominators, and the denominator's
// low limb taken twice -- once against zero and once against `z_min`, which is
// the pair the AIR range-checks.
__global__ void zk_witness_key_tile_tok(const unsigned *__restrict__ mm,
                              const unsigned *__restrict__ zz,
                              m31 *const *__restrict__ cols, unsigned with_max,
                              ZkMult m_lo, ZkMult m_hi,
                              unsigned *__restrict__ counts, const ZkKeyTileCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.rows_tok) {
        return;
    }
    const unsigned j = zk_witness_row(i, cfg.log_row, cfg.mutate);
    const long long lo_mask = (1LL << cfg.z_lo_bits) - 1LL;
    const unsigned n_max = with_max != 0u ? cfg.rw : 0u;

    const unsigned s0 = blockIdx.y * ZK_WITNESS_TILE;
    const unsigned s1 = min(cfg.rw, s0 + ZK_WITNESS_TILE);
    for (unsigned p = s0; p < s1; ++p) {
        unsigned t;
        const bool have = zk_witness_tok(cfg, j, p, &t);
        if (with_max != 0u) {
            cols[p][i] = zk_witness_m31(have ? zk_witness_i64(mm, t) : 0LL);
        }
        const long long z0 = have ? zk_witness_i64(zz, t) : 0LL;
        cols[n_max + p][i] = zk_witness_m31(z0);
        for (unsigned k = 0u; k < 2u; ++k) {
            const long long z = z0 - (k == 0u ? 0LL : cfg.z_min);
            const long long lo = z & lo_mask;
            if (!zk_witness_drop(cfg.mutate, i)) {
                zk_witness_bump(m_lo, lo, &counts[0]);
            }
            zk_witness_bump(m_hi, (z - lo) >> cfg.z_lo_bits, &counts[0]);
            cols[n_max + cfg.rw + k * cfg.rw + p][i] = zk_witness_m31(lo);
        }
    }
}

extern "C" void zk_witness_gen_key_tile_tok(const unsigned *mm, const unsigned *zz,
                                  m31 *const *cols, unsigned with_max,
                                  ZkMult m_lo, ZkMult m_hi,
                                  unsigned *counts, ZkKeyTileCfg cfg) {
    const unsigned n = (with_max != 0u ? cfg.rw : 0u) + 3u * cfg.rw;
    m31 **d = clone_to_device<m31 *>(const_cast<m31 **>(cols), n);
    dim3 g((cfg.rows_tok + ZK_WITNESS_BLOCK - 1u) / ZK_WITNESS_BLOCK,
           (cfg.rw + ZK_WITNESS_TILE - 1u) / ZK_WITNESS_TILE);
    zk_witness_key_tile_tok<<<g, ZK_WITNESS_BLOCK>>>(mm, zz, d, with_max, m_lo, m_hi, counts, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d);
}
