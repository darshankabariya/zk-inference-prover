#include <cstring>

#include "utils.cuh"
#include "gkr.cuh"

__global__ void gen_eq_evals_kernel(qm31 v, qm31 *factors, uint32_t y_size, qm31 *evals) {
    // Assumes `factors` holds 1 - y_i at position 2 * i and y_i at position 2 * i + 1
    // for all i = 0, .., y_size - 1.
    // TODO: See if shared memory speeds this up

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    qm31 eq_eval = v;
    unsigned int shifted_thread_index = thread_index;
    for (int i = 2 * y_size - 2; i >= 0; i -= 2) {
        eq_eval = mul(eq_eval, factors[i + (shifted_thread_index & 1)]);
        shifted_thread_index >>= 1;
    }
    evals[thread_index] = eq_eval;
}

void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size) {
    const unsigned int BLOCK_SIZE = 1024;
    const unsigned int NUMBER_OF_BLOCKS = (evals_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int factors_byte_length = sizeof(qm31) * y_size * 2;
    qm31 *factors = (qm31*)malloc(factors_byte_length);
    for(int i = 0; i < y_size; i++) {
        factors[2 * i] = sub(m31{1}, y[i]);
        factors[2 * i + 1] = y[i];
    }

    qm31 *factors_device = clone_to_device<qm31>(factors, y_size * 2);
    free(factors);

    gen_eq_evals_kernel<<<NUMBER_OF_BLOCKS, min(evals_size, BLOCK_SIZE)>>>(v, factors_device, y_size, evals);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // No sync: stream ordering ensures kernel completes before async free.
    cuda_free_memory(factors_device);
}

// ---- B11 stream R: the two ops that *are* the sumcheck, as kernels ----------
//
// `next_layer` halves a layer by fraction addition over adjacent pairs --
// indices 2i and 2i+1, the layer's last variable -- one launch per call and no
// synchronisation, because the output is consumed by later stream-ordered
// work. Every LogUp variant produces a generic layer, exactly as the CPU and
// SIMD backends do.
//
// `sum_as_poly_in_first_variable` reduces one oracle's terms to the round
// polynomial's evaluations at t = 0 and t = 2. One launch computes per-block
// partial pairs with a grid-stride loop; the host adds the few hundred
// partials, which changes nothing because field addition is exact and
// associative. The caller picks the block count so the copy home stays small.

#define GKR_SUM_BLOCK 256

__device__ __forceinline__ qm31 qm31_zero() {
    return qm31{cm31{0, 0}, cm31{0, 0}};
}

__device__ __forceinline__ qm31 dbl(qm31 x) {
    return add(x, x);
}

__global__ void gkr_next_layer_grand_product_kernel(
        const qm31 *inp, qm31 *out, uint32_t n_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        out[i] = mul(inp[2 * i], inp[2 * i + 1]);
    }
}

__global__ void gkr_next_layer_logup_generic_kernel(
        const qm31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        qm31 n0 = num[2 * i], n1 = num[2 * i + 1];
        qm31 d0 = den[2 * i], d1 = den[2 * i + 1];
        num_out[i] = add(mul(n0, d1), mul(n1, d0));
        den_out[i] = mul(d0, d1);
    }
}

__global__ void gkr_next_layer_logup_multiplicities_kernel(
        const m31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        m31 n0 = num[2 * i], n1 = num[2 * i + 1];
        qm31 d0 = den[2 * i], d1 = den[2 * i + 1];
        num_out[i] = add(mul(n0, d1), mul(n1, d0));
        den_out[i] = mul(d0, d1);
    }
}

__global__ void gkr_next_layer_logup_singles_kernel(
        const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_out) {
        qm31 d0 = den[2 * i], d1 = den[2 * i + 1];
        num_out[i] = add(d0, d1);
        den_out[i] = mul(d0, d1);
    }
}

// The shared block reduction of the (eval-at-0, eval-at-2) pair every sum
// kernel ends with. Thread accumulators arrive in registers; block partials
// leave through `partials[2b]` and `partials[2b + 1]`.
__device__ void gkr_block_reduce_pair(qm31 acc0, qm31 acc2, qm31 *partials) {
    __shared__ qm31 s0[GKR_SUM_BLOCK];
    __shared__ qm31 s2[GKR_SUM_BLOCK];
    s0[threadIdx.x] = acc0;
    s2[threadIdx.x] = acc2;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s0[threadIdx.x] = add(s0[threadIdx.x], s0[threadIdx.x + stride]);
            s2[threadIdx.x] = add(s2[threadIdx.x], s2[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[2 * blockIdx.x] = s0[0];
        partials[2 * blockIdx.x + 1] = s2[0];
    }
}

__global__ void gkr_sum_grand_product_kernel(
        const qm31 *inp, const qm31 *eq, uint32_t n_terms, qm31 *partials) {
    qm31 acc0 = qm31_zero(), acc2 = qm31_zero();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
         i += gridDim.x * blockDim.x) {
        qm31 a00 = inp[2 * i], a01 = inp[2 * i + 1];
        qm31 a10 = inp[2 * (n_terms + i)], a11 = inp[2 * (n_terms + i) + 1];
        qm31 a20 = sub(dbl(a10), a00), a21 = sub(dbl(a11), a01);
        qm31 e = eq[i];
        acc0 = add(acc0, mul(e, mul(a00, a01)));
        acc2 = add(acc2, mul(e, mul(a20, a21)));
    }
    gkr_block_reduce_pair(acc0, acc2, partials);
}

__global__ void gkr_sum_logup_generic_kernel(
        const qm31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials) {
    qm31 acc0 = qm31_zero(), acc2 = qm31_zero();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
         i += gridDim.x * blockDim.x) {
        qm31 n00 = num[2 * i], n01 = num[2 * i + 1];
        qm31 d00 = den[2 * i], d01 = den[2 * i + 1];
        qm31 n10 = num[2 * (n_terms + i)], n11 = num[2 * (n_terms + i) + 1];
        qm31 d10 = den[2 * (n_terms + i)], d11 = den[2 * (n_terms + i) + 1];
        qm31 n20 = sub(dbl(n10), n00), n21 = sub(dbl(n11), n01);
        qm31 d20 = sub(dbl(d10), d00), d21 = sub(dbl(d11), d01);
        qm31 numer0 = add(mul(n00, d01), mul(n01, d00));
        qm31 denom0 = mul(d00, d01);
        qm31 numer2 = add(mul(n20, d21), mul(n21, d20));
        qm31 denom2 = mul(d20, d21);
        qm31 e = eq[i];
        acc0 = add(acc0, mul(e, add(numer0, mul(lambda, denom0))));
        acc2 = add(acc2, mul(e, add(numer2, mul(lambda, denom2))));
    }
    gkr_block_reduce_pair(acc0, acc2, partials);
}

__global__ void gkr_sum_logup_multiplicities_kernel(
        const m31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials) {
    qm31 acc0 = qm31_zero(), acc2 = qm31_zero();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
         i += gridDim.x * blockDim.x) {
        m31 n00 = num[2 * i], n01 = num[2 * i + 1];
        qm31 d00 = den[2 * i], d01 = den[2 * i + 1];
        m31 n10 = num[2 * (n_terms + i)], n11 = num[2 * (n_terms + i) + 1];
        qm31 d10 = den[2 * (n_terms + i)], d11 = den[2 * (n_terms + i) + 1];
        // the numerators are base-field values; t = 2 keeps them in m31
        m31 n20 = sub(add(n10, n10), n00), n21 = sub(add(n11, n11), n01);
        qm31 d20 = sub(dbl(d10), d00), d21 = sub(dbl(d11), d01);
        qm31 numer0 = add(mul(n00, d01), mul(n01, d00));
        qm31 denom0 = mul(d00, d01);
        qm31 numer2 = add(mul(n20, d21), mul(n21, d20));
        qm31 denom2 = mul(d20, d21);
        qm31 e = eq[i];
        acc0 = add(acc0, mul(e, add(numer0, mul(lambda, denom0))));
        acc2 = add(acc2, mul(e, add(numer2, mul(lambda, denom2))));
    }
    gkr_block_reduce_pair(acc0, acc2, partials);
}

__global__ void gkr_sum_logup_singles_kernel(
        const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials) {
    qm31 acc0 = qm31_zero(), acc2 = qm31_zero();
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
         i += gridDim.x * blockDim.x) {
        qm31 d00 = den[2 * i], d01 = den[2 * i + 1];
        qm31 d10 = den[2 * (n_terms + i)], d11 = den[2 * (n_terms + i) + 1];
        qm31 d20 = sub(dbl(d10), d00), d21 = sub(dbl(d11), d01);
        qm31 numer0 = add(d00, d01);
        qm31 denom0 = mul(d00, d01);
        qm31 numer2 = add(d20, d21);
        qm31 denom2 = mul(d20, d21);
        qm31 e = eq[i];
        acc0 = add(acc0, mul(e, add(numer0, mul(lambda, denom0))));
        acc2 = add(acc2, mul(e, add(numer2, mul(lambda, denom2))));
    }
    gkr_block_reduce_pair(acc0, acc2, partials);
}

void gkr_next_layer_grand_product(const qm31 *inp, qm31 *out, uint32_t n_out) {
    uint32_t block = 256;
    uint32_t blocks = (n_out + block - 1) / block;
    gkr_next_layer_grand_product_kernel<<<blocks, block>>>(inp, out, n_out);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_next_layer_logup_generic(
        const qm31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t block = 256;
    uint32_t blocks = (n_out + block - 1) / block;
    gkr_next_layer_logup_generic_kernel<<<blocks, block>>>(num, den, num_out, den_out, n_out);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_next_layer_logup_multiplicities(
        const m31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t block = 256;
    uint32_t blocks = (n_out + block - 1) / block;
    gkr_next_layer_logup_multiplicities_kernel<<<blocks, block>>>(
        num, den, num_out, den_out, n_out);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_next_layer_logup_singles(
        const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out) {
    uint32_t block = 256;
    uint32_t blocks = (n_out + block - 1) / block;
    gkr_next_layer_logup_singles_kernel<<<blocks, block>>>(den, num_out, den_out, n_out);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_sum_grand_product(
        const qm31 *inp, const qm31 *eq, uint32_t n_terms, qm31 *partials, uint32_t n_blocks) {
    gkr_sum_grand_product_kernel<<<n_blocks, GKR_SUM_BLOCK>>>(inp, eq, n_terms, partials);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_sum_logup_generic(
        const qm31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials, uint32_t n_blocks) {
    gkr_sum_logup_generic_kernel<<<n_blocks, GKR_SUM_BLOCK>>>(
        num, den, eq, lambda, n_terms, partials);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_sum_logup_multiplicities(
        const m31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials, uint32_t n_blocks) {
    gkr_sum_logup_multiplicities_kernel<<<n_blocks, GKR_SUM_BLOCK>>>(
        num, den, eq, lambda, n_terms, partials);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

void gkr_sum_logup_singles(
        const qm31 *den, const qm31 *eq, qm31 lambda,
        uint32_t n_terms, qm31 *partials, uint32_t n_blocks) {
    gkr_sum_logup_singles_kernel<<<n_blocks, GKR_SUM_BLOCK>>>(
        den, eq, lambda, n_terms, partials);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// Profiling only: a plain device synchronisation the Rust side can call to
// attribute time to a phase. Never on the un-profiled path.
void gkr_device_synchronize() {
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
}

// Pass-2 allocation for the sumcheck's own buffers: straight from the pool,
// no zeroing. `cuda_malloc_uint32_t` memsets every allocation, which for the
// circuit evaluation means an extra full write pass (and an extra launch) per
// layer whose cells the kernel overwrites anyway. Freed by the ordinary
// `cuda_free_memory`, which is pool-aware.
qm31 *gkr_alloc_qm31(uint32_t n) {
    return cuda_mem_pool_allocate<qm31>(n);
}

// Pass-5 readback for the round sums: the per-round copy home lands in one
// pinned staging buffer (allocated once) instead of pageable memory, which
// saves the driver's staging on every one of the ~1,700 round trips a chunk
// makes. The caller hands a host destination; the pinned buffer is internal.
void gkr_read_partials(const qm31 *dev, uint32_t n_pairs, qm31 *out_host) {
    static qm31 *pinned = nullptr;
    if (pinned == nullptr) {
        ASSERT_CUDA_SUCCESS(cudaMallocHost((void **)&pinned, 2 * 264 * sizeof(qm31)));
    }
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(pinned, dev, 2 * n_pairs * sizeof(qm31),
                                        cudaMemcpyDeviceToHost, 0));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(0));
    memcpy(out_host, pinned, 2 * n_pairs * sizeof(qm31));
}

// B11 stream T: the link's coefficient column, folded where its columns are.
//
// `gkrlink::build_height_cols` needs `konst + sum_j w_j * col_j` pointwise over
// the base columns the fold references. Under a device fill those columns have
// no host mirror, so the fold happens here: the same affine combine the feed
// kernel computes per look, under the link's own weights. The accumulation
// order is the caller's weight order, which is the host builder's -- and field
// addition is exact either way, so the column is the same column.
__global__ void b65_link_coeff_kernel(m31 **__restrict__ cols,
                                      const qm31 *__restrict__ w,
                                      unsigned n_cols, qm31 k,
                                      qm31 *__restrict__ out, unsigned n_rows) {
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= n_rows) return;
    qm31 acc = k;
    for (unsigned j = 0; j < n_cols; ++j) {
        acc = add(acc, mul(cols[j][row], w[j]));
    }
    out[row] = acc;
}

void b65_link_coeff(m31 **cols, const qm31 *w_host, unsigned n_cols, qm31 k,
                    qm31 *out, unsigned n_rows) {
    cudaGetLastError();
    m31 **d_cols = n_cols ? clone_to_device<m31 *>(cols, n_cols) : nullptr;
    qm31 *d_w = n_cols ? clone_to_device<qm31>((qm31 *)w_host, n_cols) : nullptr;
    const unsigned block = n_rows < 256u ? n_rows : 256u;
    const unsigned blocks = (n_rows + block - 1u) / block;
    b65_link_coeff_kernel<<<blocks, block>>>(d_cols, d_w, n_cols, k, out, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    if (d_cols) cuda_free_memory(d_cols);
    if (d_w) cuda_free_memory(d_w);
}
