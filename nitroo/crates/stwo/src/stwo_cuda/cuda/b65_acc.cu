// B7a milestone 4 (section 3 row 1): the Horner accumulators, on the device.
#include "fields.cuh"
#include "utils.cuh"
#include "b65_acc.cuh"

#define B65A_BLOCK 256u

static __device__ const qm31 B65A_QZERO = {{0u, 0u}, {0u, 0u}};

// The permutation, `Grid::row` and the flag closed forms are in `b65_acc.cuh`:
// `b65_pre_fill.cu` builds `commit_pre`'s 48 selector columns out of exactly
// the same three lines these kernels segment on (section 25).

// ------------------------------------------------------------------ field helpers --

__device__ __forceinline__ qm31 b65a_of_m31(m31 v) {
    qm31 x = B65A_QZERO;
    x.a.a = v;
    return x;
}

// One row's inner Horner sum over its own committed columns:
// `sum_p pow[p] * (cols[p][jp] - off)`.
//
// The host skips a zero term and this does not, which is the same element: the
// field's zero is additively neutral, and `m31_of` is a ring map, so committing
// `v + off` and subtracting `off` here is committing `v`.
__device__ __forceinline__ qm31 b65a_dot(const m31 *const *cols, const qm31 *pow,
                                         unsigned n, unsigned jp, m31 off) {
    qm31 acc = B65A_QZERO;
    for (unsigned p = 0; p < n; ++p) {
        acc = add(acc, mul(sub(cols[p][jp], off), pow[p]));
    }
    return acc;
}

// The grid's score Horner, which is the one place the committed column is not
// the value the accumulator wants twice over:
//
//   * `intexp` commits `d delta`, the shift scaled to the master exp table's
//     exponent, so the accumulator divides `delta` back out (`inv`);
//   * `intsplit(k)` commits `(d delta) >> k` in the key column and
//     `(d delta) & (2^k - 1)` in the auxiliary one, so the accumulator puts
//     `d delta = idx 2^k + aux` back together first -- `layout.rs`'s own
//     "the AIR reconstructs `d = (idx 2^k + aux) 2^-(F - frac_i)`", and rung 3
//     is the only design point where it is ever proved (section 5 rule 11);
//   * `bf16x` commits the score itself, and both constants are 1.
__device__ __forceinline__ qm31 b65a_dot_key(const m31 *const *key,
                                             const m31 *const *aux, const qm31 *pow,
                                             unsigned n, unsigned jp, m31 two_k,
                                             m31 inv) {
    qm31 acc = B65A_QZERO;
    for (unsigned p = 0; p < n; ++p) {
        m31 v = mul(key[p][jp], two_k);
        if (aux != nullptr) {
            v = add(v, aux[p][jp]);
        }
        acc = add(acc, mul(mul(v, inv), pow[p]));
    }
    return acc;
}

// A stage-1 lane's multiplier: the step inside a group, zero at its first row,
// which is how a segmented recurrence becomes an ordinary affine scan.
__device__ __forceinline__ qm31 b65a_step(bool nofirst, qm31 step, unsigned mutate) {
    return (nofirst || mutate == B65A_MUT_NORESET) ? step : B65A_QZERO;
}

// =============================================================== the flags ==

// One band sub-region's per-row layout facts, and nothing else.
//
// B7a milestone 4 (section 25): this used to be the first six lines of each of
// the three region kernels below. It is a kernel of its own because the 48
// preprocessed selector columns `commit_pre` builds are *these bits* -- and
// they are needed before the challenges are drawn, where the accumulators run
// after. One array, written once at fill time, read by both.
__global__ void b65a_flags_kernel(unsigned *__restrict__ flags,
                                  const B65AccFlags cfg) {
    const unsigned rows = cfg.g.rows;
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rows) {
        return;
    }
    flags[j] = cfg.shape == B65A_SHAPE_TOK
                   ? b65a_flags_of_tok(rows, cfg.rw, cfg.modulus, j)
                   : b65a_flags_of_grid(cfg.g, cfg.modulus, j);
}

extern "C" void b65_flags_band(unsigned *flags, B65AccFlags cfg) {
    const unsigned rows = cfg.g.rows;
    const unsigned bw = rows < B65A_BLOCK ? rows : B65A_BLOCK;
    b65a_flags_kernel<<<(rows + bw - 1u) / bw, bw>>>(flags, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// ================================================================== the scan ==

// Phase 1: the block-local affine scan, in shared memory.
//
// `(a2, b2) . (a1, b1) = (a1 a2, a2 b1 + b2)` is associative, so a
// Hillis-Steele sweep over the pairs gives the same field element the host's
// serial loop gives -- not an approximation of it.
__global__ void b65a_scan_local(qm31 *__restrict__ a, qm31 *__restrict__ b,
                                qm31 *__restrict__ blk_a, qm31 *__restrict__ blk_x,
                                unsigned rows, unsigned n_blocks) {
    extern __shared__ qm31 sh[];
    const unsigned tid = threadIdx.x;
    const unsigned bw = blockDim.x;
    const unsigned lane = blockIdx.y;
    const size_t off = (size_t)lane * rows + (size_t)blockIdx.x * bw;
    qm31 ca = a[off + tid];
    qm31 cx = b[off + tid];
    sh[tid] = ca;
    sh[bw + tid] = cx;
    __syncthreads();
    for (unsigned d = 1u; d < bw; d <<= 1) {
        qm31 na = ca, nx = cx;
        if (tid >= d) {
            const qm31 pa = sh[tid - d];
            const qm31 px = sh[bw + tid - d];
            nx = add(mul(ca, px), cx);
            na = mul(ca, pa);
        }
        __syncthreads();
        ca = na;
        cx = nx;
        sh[tid] = ca;
        sh[bw + tid] = cx;
        __syncthreads();
    }
    a[off + tid] = ca;
    b[off + tid] = cx;
    if (tid + 1u == bw) {
        blk_a[(size_t)lane * n_blocks + blockIdx.x] = ca;
        blk_x[(size_t)lane * n_blocks + blockIdx.x] = cx;
    }
}

// Phase 2: the block totals, scanned serially -- one thread per lane over at
// most `rows / 256` blocks, which at the widest region here is 512 steps.
__global__ void b65a_scan_carry(const qm31 *__restrict__ blk_a,
                                const qm31 *__restrict__ blk_x,
                                qm31 *__restrict__ carry, unsigned n_blocks,
                                unsigned lanes) {
    const unsigned l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= lanes) {
        return;
    }
    qm31 c = B65A_QZERO;
    for (unsigned k = 0; k < n_blocks; ++k) {
        const size_t i = (size_t)l * n_blocks + k;
        carry[i] = c;
        c = add(mul(blk_a[i], c), blk_x[i]);
    }
}

// Phase 3: the carry, applied. `nocarry` is the control that says the scan is
// not just `rows / 256` independent block scans.
__global__ void b65a_scan_apply(const qm31 *__restrict__ a, qm31 *__restrict__ b,
                                const qm31 *__restrict__ carry, unsigned rows,
                                unsigned n_blocks, unsigned mutate) {
    const unsigned tid = threadIdx.x;
    const unsigned lane = blockIdx.y;
    const size_t off = (size_t)lane * rows + (size_t)blockIdx.x * blockDim.x;
    const qm31 c = (mutate == B65A_MUT_NOCARRY)
                       ? B65A_QZERO
                       : carry[(size_t)lane * n_blocks + blockIdx.x];
    b[off + tid] = add(mul(a[off + tid], c), b[off + tid]);
}

extern "C" void b65_acc_scan(qm31 *a, qm31 *b, unsigned rows, unsigned lanes,
                             unsigned mutate) {
    if (lanes == 0u) {
        return;
    }
    const unsigned bw = rows < B65A_BLOCK ? rows : B65A_BLOCK;
    const unsigned n_blocks = rows / bw;  // rows is a power of two, so this is exact
    qm31 *blk_a = cuda_malloc<qm31>(lanes * n_blocks);
    qm31 *blk_x = cuda_malloc<qm31>(lanes * n_blocks);
    qm31 *carry = cuda_malloc<qm31>(lanes * n_blocks);
    const dim3 grid(n_blocks, lanes);
    b65a_scan_local<<<grid, bw, 2u * bw * sizeof(qm31)>>>(a, b, blk_a, blk_x, rows,
                                                          n_blocks);
    const unsigned cb = lanes < 64u ? lanes : 64u;
    b65a_scan_carry<<<(lanes + cb - 1u) / cb, cb>>>(blk_a, blk_x, carry, n_blocks, lanes);
    b65a_scan_apply<<<grid, bw>>>(a, b, carry, rows, n_blocks, mutate);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(blk_a);
    cuda_free_memory(blk_x);
    cuda_free_memory(carry);
}

// ================================================================ stage two ==

__global__ void b65a_stage2_kernel(const unsigned *__restrict__ flags,
                                   qm31 *__restrict__ a, qm31 *__restrict__ b,
                                   const B65AccS2 *__restrict__ lanes, unsigned rows,
                                   unsigned mutate) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rows) {
        return;
    }
    const B65AccS2 L = lanes[blockIdx.y];
    const unsigned f = flags[j];
    const bool last = (f & B65A_LAST) != 0u;
    const bool ist = mutate != B65A_MUT_NOISTART && L.sel_i != 0u && (f & L.sel_i) != 0u;
    qm31 bv = B65A_QZERO;
    if (last || mutate == B65A_MUT_EVERYROW) {
        bv = b[(size_t)L.src0 * rows + j];
        if (L.src1 != B65A_ONE) {
            bv = mul(bv, b[(size_t)L.src1 * rows + j]);
        }
    }
    a[(size_t)L.dst * rows + j] = ist ? L.m_i : (last ? L.m_l : L.m_n);
    b[(size_t)L.dst * rows + j] = bv;
}

extern "C" void b65_acc_stage2(const unsigned *flags, qm31 *a, qm31 *b,
                               const B65AccS2 *lanes, unsigned n_lanes, unsigned rows,
                               unsigned mutate) {
    if (n_lanes == 0u) {
        return;
    }
    B65AccS2 *d_lanes = clone_to_device<B65AccS2>(const_cast<B65AccS2 *>(lanes), n_lanes);
    const unsigned bw = rows < B65A_BLOCK ? rows : B65A_BLOCK;
    const dim3 grid((rows + bw - 1u) / bw, n_lanes);
    b65a_stage2_kernel<<<grid, bw>>>(flags, a, b, d_lanes, rows, mutate);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_lanes);
}

// ============================================================== the columns ==

__global__ void b65a_split_kernel(const qm31 *__restrict__ v, m31 **out,
                                  const unsigned *__restrict__ lane, unsigned rows,
                                  unsigned log_rows, unsigned mutate) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    const unsigned k = blockIdx.y;
    const unsigned src = (mutate == B65A_MUT_NOPERM) ? i : b65a_perm(i, log_rows);
    const qm31 x = v[(size_t)lane[k] * rows + src];
    // `SecureField::to_m31_array`'s order, which is `secure_cols`'s column order.
    out[4u * k + 0u][i] = x.a.a;
    out[4u * k + 1u][i] = x.a.b;
    out[4u * k + 2u][i] = x.b.a;
    out[4u * k + 3u][i] = x.b.b;
}

extern "C" void b65_acc_split(const qm31 *v, m31 **out, const unsigned *lane,
                              unsigned n_out, unsigned rows, unsigned log_rows,
                              unsigned mutate) {
    if (n_out == 0u) {
        return;
    }
    m31 **d_out = clone_to_device<m31 *>(out, 4u * n_out);
    unsigned *d_lane = clone_to_device<unsigned>(const_cast<unsigned *>(lane), n_out);
    const unsigned bw = rows < B65A_BLOCK ? rows : B65A_BLOCK;
    const dim3 grid((rows + bw - 1u) / bw, n_out);
    b65a_split_kernel<<<grid, bw>>>(v, d_out, d_lane, rows, log_rows, mutate);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_out);
    cuda_free_memory(d_lane);
}

extern "C" void b65_acc_last(const qm31 *v, qm31 *host, unsigned rows, unsigned lanes) {
    for (unsigned l = 0; l < lanes; ++l) {
        ASSERT_CUDA_SUCCESS(cudaMemcpy(host + l, v + (size_t)l * rows + (rows - 1u),
                                       sizeof(qm31), cudaMemcpyDeviceToHost));
    }
}

// ========================================================== the band channel ==

__global__ void b65a_chan_kernel(const m31 *const *__restrict__ cols,
                                 const qm31 *__restrict__ r_pow,
                                 const qm31 *__restrict__ sq_pow, qm31 r_step,
                                 qm31 sq_step, const unsigned *__restrict__ flags,
                                 qm31 *__restrict__ a, qm31 *__restrict__ b,
                                 const B65AccChan cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.g.rows) {
        return;
    }
    const unsigned f = flags[j];

    const unsigned jp = b65a_perm(j, cfg.g.log_rows);
    const bool raw = cfg.mutate == B65A_MUT_NOOFFSET;
    const qm31 kf = b65a_dot(cols, r_pow, cfg.pk, jp, raw ? 0u : cfg.off_k);
    const qm31 qf = b65a_dot(cols + cfg.pk, sq_pow, cfg.pq, jp, raw ? 0u : cfg.off_q);
    const bool nofirst = (f & B65A_NOFIRST) != 0u;
    const size_t r = cfg.g.rows;
    a[j] = b65a_step(nofirst, r_step, cfg.mutate);
    b[j] = kf;
    a[r + j] = b65a_step(nofirst, sq_step, cfg.mutate);
    b[r + j] = qf;
}

extern "C" void b65_acc_chan(const m31 *const *cols, const qm31 *r_pow,
                             const qm31 *sq_pow, qm31 r_step, qm31 sq_step,
                             const unsigned *flags, qm31 *a, qm31 *b,
                             B65AccChan cfg) {
    const m31 **d_cols =
        clone_to_device<const m31 *>(const_cast<const m31 **>(cols), cfg.pk + cfg.pq);
    qm31 *d_r = clone_to_device<qm31>(const_cast<qm31 *>(r_pow), cfg.pk);
    qm31 *d_s = clone_to_device<qm31>(const_cast<qm31 *>(sq_pow), cfg.pq);
    const unsigned bw = cfg.g.rows < B65A_BLOCK ? cfg.g.rows : B65A_BLOCK;
    b65a_chan_kernel<<<(cfg.g.rows + bw - 1u) / bw, bw>>>(d_cols, d_r, d_s, r_step,
                                                          sq_step, flags, a, b, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_cols);
    cuda_free_memory(d_r);
    cuda_free_memory(d_s);
}

// ============================================================= the band grid ==

__global__ void b65a_grid_kernel(const m31 *const *__restrict__ key_cols,
                                 const m31 *const *__restrict__ num_cols,
                                 const m31 *const *__restrict__ aux_cols,
                                 const m31 *const *__restrict__ val_cols,
                                 const qm31 *__restrict__ sg_pow,
                                 const qm31 *__restrict__ rv_pow, qm31 sg_step,
                                 qm31 rv_step, const m31 *__restrict__ inv_delta,
                                 const unsigned *__restrict__ flags,
                                 qm31 *__restrict__ a, qm31 *__restrict__ b,
                                 const B65AccKey cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.g.rows) {
        return;
    }
    const unsigned f = flags[j];
    const bool act = (f & B65A_ACTIVE) != 0u;
    // the group index, which is the one layout fact the flags do not carry:
    // `inv_delta` is per instance and `split(g, C).0` is where it comes from.
    unsigned grp = 0u, chunk = 0u;
    b65a_grid_row(cfg.g, j, &grp, &chunk);

    const unsigned jp = b65a_perm(j, cfg.g.log_rows);
    // `split(g, C).0`, and the scale that turns the committed `d delta` back
    // into the `d` the score accumulator sees. Off a lead row the key column is
    // zero, so the scale there cannot matter.
    const unsigned inst = act ? (grp / cfg.ctx) : 0u;
    const m31 scale = (cfg.mutate == B65A_MUT_NODELTA) ? 1u : inv_delta[inst];
    const bool split = aux_cols != nullptr && cfg.mutate != B65A_MUT_NOSPLIT;
    const m31 two_k = (cfg.mutate == B65A_MUT_NOSPLIT) ? 1u : cfg.two_k;
    const qm31 sf = b65a_dot_key(key_cols, split ? aux_cols : nullptr, sg_pow, cfg.q, jp,
                                 two_k, scale);
    // The numerator column commits `table[0]` on an inert lead row, where the
    // host's sum has no term at all -- `nomask` is that difference.
    qm31 nf = B65A_QZERO;
    if (act || cfg.mutate == B65A_MUT_NOMASK) {
        nf = b65a_dot(num_cols, sg_pow, cfg.q, jp, 0u);
    }
    const m31 ov = (cfg.mutate == B65A_MUT_NOOFFSET) ? 0u : cfg.off_v;
    const qm31 vf = b65a_dot(val_cols, rv_pow, cfg.pv, jp, ov);

    const bool nofirst = (f & B65A_NOFIRST) != 0u;
    const size_t r = cfg.g.rows;
    a[j] = b65a_step(nofirst, sg_step, cfg.mutate);
    b[j] = sf;
    a[r + j] = b65a_step(nofirst, sg_step, cfg.mutate);
    b[r + j] = nf;
    a[2u * r + j] = b65a_step(nofirst, rv_step, cfg.mutate);
    b[2u * r + j] = vf;
}

extern "C" void b65_acc_grid(const m31 *const *key_cols, const m31 *const *num_cols,
                             const m31 *const *aux_cols, const m31 *const *val_cols,
                             const qm31 *sg_pow, const qm31 *rv_pow, qm31 sg_step,
                             qm31 rv_step, const m31 *inv_delta,
                             const unsigned *flags, qm31 *a, qm31 *b,
                             B65AccKey cfg) {
    const m31 **d_key =
        clone_to_device<const m31 *>(const_cast<const m31 **>(key_cols), cfg.q);
    const m31 **d_num =
        clone_to_device<const m31 *>(const_cast<const m31 **>(num_cols), cfg.q);
    const m31 **d_aux =
        aux_cols == nullptr
            ? nullptr
            : clone_to_device<const m31 *>(const_cast<const m31 **>(aux_cols), cfg.q);
    const m31 **d_val =
        clone_to_device<const m31 *>(const_cast<const m31 **>(val_cols), cfg.pv);
    qm31 *d_sg = clone_to_device<qm31>(const_cast<qm31 *>(sg_pow), cfg.q);
    qm31 *d_rv = clone_to_device<qm31>(const_cast<qm31 *>(rv_pow), cfg.pv);
    m31 *d_inv = clone_to_device<m31>(const_cast<m31 *>(inv_delta), cfg.n_insts);
    const unsigned bw = cfg.g.rows < B65A_BLOCK ? cfg.g.rows : B65A_BLOCK;
    b65a_grid_kernel<<<(cfg.g.rows + bw - 1u) / bw, bw>>>(
        d_key, d_num, d_aux, d_val, d_sg, d_rv, sg_step, rv_step, d_inv, flags, a, b,
        cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_key);
    cuda_free_memory((void *)d_num);
    if (d_aux != nullptr) {
        cuda_free_memory((void *)d_aux);
    }
    cuda_free_memory((void *)d_val);
    cuda_free_memory(d_sg);
    cuda_free_memory(d_rv);
    cuda_free_memory(d_inv);
}

// ======================================================== the per-token block ==

__global__ void b65a_tok_kernel(const m31 *const *__restrict__ max_cols,
                                const m31 *const *__restrict__ z_cols,
                                const qm31 *__restrict__ sr_pow, qm31 sr_step, qm31 mu,
                                const unsigned *__restrict__ flags,
                                qm31 *__restrict__ a, qm31 *__restrict__ b,
                                const B65AccTok cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.rows) {
        return;
    }
    const bool istart = (flags[j] & B65A_ISTART) != 0u;

    const unsigned jp = b65a_perm(j, cfg.log_rows);
    const qm31 step =
        (istart && cfg.mutate != B65A_MUT_NOISTART) ? mu : sr_step;
    const qm31 mf = cfg.with_max ? b65a_dot(max_cols, sr_pow, cfg.rw, jp, 0u)
                                 : B65A_QZERO;
    const qm31 zf = b65a_dot(z_cols, sr_pow, cfg.rw, jp, 0u);
    const size_t r = cfg.rows;
    const qm31 av = (j > 0u) ? step : B65A_QZERO;
    a[j] = av;
    b[j] = mf;
    a[r + j] = av;
    b[r + j] = zf;
}

extern "C" void b65_acc_tok(const m31 *const *max_cols, const m31 *const *z_cols,
                            const qm31 *sr_pow, qm31 sr_step, qm31 mu,
                            const unsigned *flags, qm31 *a, qm31 *b,
                            B65AccTok cfg) {
    const m31 **d_max = clone_to_device<const m31 *>(
        const_cast<const m31 **>(max_cols), cfg.with_max ? cfg.rw : 1u);
    const m31 **d_z =
        clone_to_device<const m31 *>(const_cast<const m31 **>(z_cols), cfg.rw);
    qm31 *d_sr = clone_to_device<qm31>(const_cast<qm31 *>(sr_pow), cfg.rw);
    const unsigned bw = cfg.rows < B65A_BLOCK ? cfg.rows : B65A_BLOCK;
    b65a_tok_kernel<<<(cfg.rows + bw - 1u) / bw, bw>>>(d_max, d_z, d_sr, sr_step, mu,
                                                       flags, a, b, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_max);
    cuda_free_memory((void *)d_z);
    cuda_free_memory(d_sr);
}

// ================================================================= the walks ==

__global__ void b65a_walk_kernel(const m31 *const *__restrict__ base_cols,
                                 const m31 *const *__restrict__ pre_cols,
                                 const qm31 *__restrict__ s_pow,
                                 const qm31 *__restrict__ rv_pow, qm31 s_step,
                                 qm31 rv_step, unsigned *__restrict__ flags,
                                 qm31 *__restrict__ a, qm31 *__restrict__ b,
                                 const B65AccWalk cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.g.rows) {
        return;
    }
    unsigned grp = 0u, chunk = 0u;
    const bool act = b65a_grid_row(cfg.g, j, &grp, &chunk);
    const unsigned f = b65a_grid_flags(act, chunk, cfg.g.chunks);
    flags[j] = f;

    const unsigned jp = b65a_perm(j, cfg.g.log_rows);
    const m31 ox = (cfg.mutate == B65A_MUT_NOOFFSET) ? 0u : cfg.off_x;
    const qm31 vf = b65a_dot(base_cols, s_pow, cfg.pq, jp, ox);
    const bool nofirst = (f & B65A_NOFIRST) != 0u;
    const size_t r = cfg.g.rows;
    a[j] = b65a_step(nofirst, s_step, cfg.mutate);
    b[j] = vf;
    for (unsigned m = 0; m < cfg.n_mats; ++m) {
        const qm31 uf = b65a_dot(pre_cols + (size_t)m * cfg.pk, rv_pow, cfg.pk, jp, 0u);
        a[(size_t)(1u + m) * r + j] = b65a_step(nofirst, rv_step, cfg.mutate);
        b[(size_t)(1u + m) * r + j] = uf;
    }
}

extern "C" void b65_acc_walk(const m31 *const *base_cols, const m31 *const *pre_cols,
                             const qm31 *s_pow, const qm31 *rv_pow, qm31 s_step,
                             qm31 rv_step, unsigned *flags, qm31 *a, qm31 *b,
                             B65AccWalk cfg) {
    const m31 **d_base =
        clone_to_device<const m31 *>(const_cast<const m31 **>(base_cols), cfg.pq);
    const m31 **d_pre = clone_to_device<const m31 *>(
        const_cast<const m31 **>(pre_cols), cfg.pk * cfg.n_mats);
    qm31 *d_s = clone_to_device<qm31>(const_cast<qm31 *>(s_pow), cfg.pq);
    qm31 *d_rv = clone_to_device<qm31>(const_cast<qm31 *>(rv_pow), cfg.pk);
    const unsigned bw = cfg.g.rows < B65A_BLOCK ? cfg.g.rows : B65A_BLOCK;
    b65a_walk_kernel<<<(cfg.g.rows + bw - 1u) / bw, bw>>>(
        d_base, d_pre, d_s, d_rv, s_step, rv_step, flags, a, b, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_base);
    cuda_free_memory((void *)d_pre);
    cuda_free_memory(d_s);
    cuda_free_memory(d_rv);
}

// =============================================================== the dequant ==

__global__ void b65a_deq_kernel(const m31 *const *__restrict__ acc_cols,
                                const qm31 *__restrict__ rv_pow, qm31 rv_step,
                                unsigned *__restrict__ flags, qm31 *__restrict__ a,
                                qm31 *__restrict__ b, const B65AccDeq cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.rows) {
        return;
    }
    // `MatLayout::at` tiles exactly: every row is active, and the chunk index is
    // `chunks - 1 - j % chunks`, so the group's first row is `j % chunks == 0`.
    const unsigned m = j % cfg.chunks;
    unsigned f = B65A_ACTIVE;
    if (m != 0u) {
        f |= B65A_NOFIRST;
    }
    if (m + 1u == cfg.chunks) {
        f |= B65A_LAST;
    }
    flags[j] = f;

    const unsigned jp = b65a_perm(j, cfg.log_rows);
    a[j] = b65a_step((f & B65A_NOFIRST) != 0u, rv_step, cfg.mutate);
    b[j] = b65a_dot(acc_cols, rv_pow, cfg.per_row, jp, 0u);
}

extern "C" void b65_acc_deq(const m31 *const *acc_cols, const qm31 *rv_pow,
                            qm31 rv_step, unsigned *flags, qm31 *a, qm31 *b,
                            B65AccDeq cfg) {
    const m31 **d_cols =
        clone_to_device<const m31 *>(const_cast<const m31 **>(acc_cols), cfg.per_row);
    qm31 *d_rv = clone_to_device<qm31>(const_cast<qm31 *>(rv_pow), cfg.per_row);
    const unsigned bw = cfg.rows < B65A_BLOCK ? cfg.rows : B65A_BLOCK;
    b65a_deq_kernel<<<(cfg.rows + bw - 1u) / bw, bw>>>(d_cols, d_rv, rv_step, flags, a,
                                                       b, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_cols);
    cuda_free_memory(d_rv);
}

// =============================================================== the attnout ==

__global__ void b65a_ao_kernel(const m31 *__restrict__ den_col,
                               const m31 *const *__restrict__ acc_cols,
                               const qm31 *__restrict__ rv_pow, qm31 rv_step,
                               unsigned *__restrict__ flags, qm31 *__restrict__ a,
                               qm31 *__restrict__ b, const B65AccAo cfg) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cfg.rows) {
        return;
    }
    // `AoLayout::at`: (head, token, chunk), all descending, inert lead first.
    unsigned f = 0u;
    bool hfirst = false;
    if (j >= cfg.lead) {
        const unsigned ja = j - cfg.lead;
        const unsigned per_head = cfg.s * cfg.chunks;
        const unsigned rest = ja % per_head;
        const unsigned t = cfg.s - 1u - rest / cfg.chunks;
        const unsigned bb = cfg.chunks - 1u - rest % cfg.chunks;
        f = b65a_grid_flags(true, bb, cfg.chunks);
        hfirst = bb == 0u && t + 1u == cfg.s;
    }
    // The head start, where `h` and `dz` swap the token step for `mu`. It is a
    // last row by construction, so it is the flag stage 2 selects on.
    if (hfirst) {
        f |= B65A_ISTART;
    }
    flags[j] = f;

    const unsigned jp = b65a_perm(j, cfg.log_rows);
    const size_t r = cfg.rows;
    a[j] = b65a_step((f & B65A_NOFIRST) != 0u, rv_step, cfg.mutate);
    b[j] = b65a_dot(acc_cols, rv_pow, cfg.per_row, jp, 0u);
    // The denominator, as an unscanned lane: `a = 0` leaves the scan's own
    // recurrence holding the value itself, so stage 2 reads it like any other.
    a[r + j] = B65A_QZERO;
    b[r + j] = b65a_of_m31(den_col[jp]);
}

extern "C" void b65_acc_ao(const m31 *den_col, const m31 *const *acc_cols,
                           const qm31 *rv_pow, qm31 rv_step, unsigned *flags, qm31 *a,
                           qm31 *b, B65AccAo cfg) {
    const m31 **d_cols =
        clone_to_device<const m31 *>(const_cast<const m31 **>(acc_cols), cfg.per_row);
    qm31 *d_rv = clone_to_device<qm31>(const_cast<qm31 *>(rv_pow), cfg.per_row);
    const unsigned bw = cfg.rows < B65A_BLOCK ? cfg.rows : B65A_BLOCK;
    b65a_ao_kernel<<<(cfg.rows + bw - 1u) / bw, bw>>>(den_col, d_cols, d_rv, rv_step,
                                                      flags, a, b, cfg);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory((void *)d_cols);
    cuda_free_memory(d_rv);
}
