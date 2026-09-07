// B7a milestone 4 (section 3 row 1): the shared tables' own columns.
#include "fields.cuh"
#include "utils.cuh"
#include "zk_table_witness_gen.cuh"

#define ZK_TABLE_BLOCK 256u

// ---------------------------------------------------------------- the permutation --

// `common.rs`'s `m4_perm`, verbatim from `zk_dequant_witness_gen.cu` and `zk_witness_gen.cu`: the
// closed form of `bit_reverse_coset_to_circle_domain_order`. It is an
// involution, so the thread that owns committed row `i` reads coset row `f(i)`
// -- and the committed row holding coset row `d` is `f(d)`, which is what the
// patch kernel needs.
__device__ __forceinline__ unsigned zk_table_perm(unsigned i, unsigned log_n) {
    const unsigned n = 1u << log_n;
    const unsigned c = (i & 1u) ? ((2u * n - i - 1u) >> 1) : (i >> 1);
    return __brev(c) >> (32u - log_n);
}

// Is this defect aimed at this column?
__device__ __forceinline__ bool zk_table_hit(const ZkTabCfg &cfg, unsigned mut, unsigned col) {
    return cfg.mutate == mut && (cfg.mut_col == col || cfg.mut_col == ZK_TABLE_COL_ALL);
}

// The coset row one column reads. `noperm` is the control that says the
// permutation is load-bearing: without it every cell but the involution's fixed
// points moves, and `eval_of` is exactly what it replaces.
__device__ __forceinline__ unsigned zk_table_row(const ZkTabCfg &cfg, unsigned i,
                                             unsigned jp, unsigned col) {
    return zk_table_hit(cfg, ZK_TABLE_MUT_NOPERM, col) ? i : jp;
}

// ------------------------------------------------------------------ bf16 --

// `bf16::from_f32`, round-to-nearest-even, without its NaN arm: the argument
// here is `-d 2^-frac`, finite and non-positive for every row of the table.
__device__ __forceinline__ unsigned short zk_table_from_f32(float x, bool trunc) {
    const unsigned u = __float_as_uint(x);
    if (trunc) {
        return (unsigned short)(u >> 16);
    }
    const unsigned bias = 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)((u + bias) >> 16);
}

// ------------------------------------------------------------- the kernel --

// One thread, one committed row, all three columns.
//
// The cells are written raw, not reduced: `common.rs`'s `m31_u32` is
// `M31::from_u32_unchecked`, and a row index, a numerator bounded by `nmax` and
// a lookup count are all already field elements. Reducing here would be a
// difference from the host, not a safety net.
__global__ void zk_table_tables(const unsigned *__restrict__ expmap,
                            const unsigned *const *__restrict__ counters,
                            m31 *__restrict__ x_col,
                            m31 *__restrict__ y_col,
                            m31 *__restrict__ m_col,
                            const ZkTabCfg cfg) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.rows) {
        return;
    }
    const unsigned jp = zk_table_perm(i, cfg.log_rows);

    // `pre_tables[0]`: the row index.
    x_col[i] = (m31)zk_table_row(cfg, i, jp, ZK_TABLE_COL_X);

    // `pre_tables[1]`: `exp_int_table(rows, nmax, frac)[d]`, which is the bf16
    // map read at the pattern of `-d 2^-frac`. The bf16x variant's table is the
    // map itself, so its pattern is the row.
    {
        const unsigned d = zk_table_row(cfg, i, jp, ZK_TABLE_COL_Y);
        unsigned short pat;
        if (cfg.variant == ZK_TABLE_VAR_BF16X || zk_table_hit(cfg, ZK_TABLE_MUT_NOSCALE, ZK_TABLE_COL_Y)) {
            pat = (unsigned short)d;
        } else {
            const double t = -(double)d / cfg.two_pow;
            pat = zk_table_from_f32((float)t, zk_table_hit(cfg, ZK_TABLE_MUT_BF16TRUNC, ZK_TABLE_COL_Y));
        }
        y_col[i] = (m31)expmap[pat];
    }

    // `base_mults[0]`: the exp table's multiplicities. Every key tile's witness generation
    // scatter-added into a counter of its own (21.1) and the host's array was
    // their sum; the sum happens here instead, and nothing comes back. A null
    // column is "no key tile built on the device", so there is no counter to sum
    // and the host keeps this one.
    if (m_col != nullptr) {
        const unsigned d = zk_table_row(cfg, i, jp, ZK_TABLE_COL_M);
        unsigned n = cfg.n_counters;
        if (zk_table_hit(cfg, ZK_TABLE_MUT_ONECOUNTER, ZK_TABLE_COL_M) && n > 1u) {
            n = 1u;
        }
        unsigned s = 0u;
        for (unsigned c = 0; c < n; ++c) {
            s += counters[c][d];
        }
        m_col[i] = (m31)s;
    }
}

// The rows where the witness's own table disagrees with the closed form. One
// entry for rung 3's corrupted-table probe and none otherwise, and it is the
// whole of section 5 rule 8 for this witness generation.
__global__ void zk_table_patch(const unsigned *__restrict__ idx,
                           const unsigned *__restrict__ val,
                           const unsigned n, m31 *__restrict__ y_col,
                           const ZkTabCfg cfg) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    y_col[zk_table_perm(idx[t], cfg.log_rows)] = (m31)val[t];
}

extern "C" void zk_witness_gen_tables(const unsigned *expmap,
                                const unsigned *const *counters,
                                const unsigned *patch_idx, const unsigned *patch_val,
                                unsigned n_patch,
                                m31 *x_col, m31 *y_col, m31 *m_col,
                                ZkTabCfg cfg) {
    unsigned **d_cnt = nullptr;
    if (cfg.n_counters > 0u) {
        d_cnt = clone_to_device<unsigned *>(const_cast<unsigned **>(counters),
                                            cfg.n_counters);
    }
    const unsigned gb = (cfg.rows + ZK_TABLE_BLOCK - 1u) / ZK_TABLE_BLOCK;
    zk_table_tables<<<gb, ZK_TABLE_BLOCK>>>(expmap, d_cnt, x_col, y_col, m_col, cfg);
    if (n_patch > 0u && cfg.mutate != ZK_TABLE_MUT_NOPATCH) {
        const unsigned pb = (n_patch + ZK_TABLE_BLOCK - 1u) / ZK_TABLE_BLOCK;
        zk_table_patch<<<pb, ZK_TABLE_BLOCK>>>(patch_idx, patch_val, n_patch, y_col, cfg);
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    if (d_cnt != nullptr) {
        cuda_free_memory(d_cnt);
    }
}
