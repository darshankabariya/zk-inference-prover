// B7a milestone 4 (18.7): the LogUp phases that are not per-family.
#include "fields.cuh"
#include "utils.cuh"
#include "prefix_sum.cuh"
#include "gen_zk_logup.cuh"

#define ZK_LOGUP_BLOCK 256u

static __device__ const qm31 ZK_QZERO = {{0u, 0u}, {0u, 0u}};

__global__ void zk_scale_kernel(const qm31 *__restrict__ num,
                                 const qm31 *__restrict__ inv,
                                 qm31 *__restrict__ frac, unsigned n) {
    const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) frac[i] = mul(num[i], inv[i]);
}

extern "C" void zk_logup_scale(const qm31 *num, const qm31 *inv, qm31 *frac, unsigned n) {
    const unsigned block = n < ZK_LOGUP_BLOCK ? n : ZK_LOGUP_BLOCK;
    const unsigned blocks = (n + block - 1u) / block;
    zk_scale_kernel<<<blocks, block>>>(num, inv, frac, n);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// The coordinate order is `SecureField::to_m31_array`'s -- (a.a, a.b, b.a,
// b.b) -- which is the order `into_packed_m31s` gives the host builder and the
// order the generated constraint kernels write their quotient columns in.
__global__ void zk_running_kernel(const qm31 *__restrict__ frac, m31 **out,
                                   qm31 *__restrict__ last,
                                   unsigned n_rows, unsigned n_cols) {
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= n_rows) return;
    qm31 run = ZK_QZERO;
    for (unsigned c = 0; c + 1u < n_cols; ++c) {
        run = add(run, frac[(size_t)c * n_rows + row]);
        out[4u * c + 0u][row] = run.a.a;
        out[4u * c + 1u][row] = run.a.b;
        out[4u * c + 2u][row] = run.b.a;
        out[4u * c + 3u][row] = run.b.b;
    }
    run = add(run, frac[(size_t)(n_cols - 1u) * n_rows + row]);
    last[row] = run;
}

extern "C" void zk_logup_running(const qm31 *frac, m31 **out_cols, qm31 *last,
                                  unsigned n_rows, unsigned n_cols) {
    // `out_cols` holds the four m31 columns of each of the first `n_cols - 1`
    // running sums; the last column's four are phase C's and are written there.
    const unsigned n_out = 4u * (n_cols - 1u);
    m31 **d_out = n_out ? clone_to_device<m31 *>(out_cols, n_out) : nullptr;
    const unsigned block = n_rows < ZK_LOGUP_BLOCK ? n_rows : ZK_LOGUP_BLOCK;
    const unsigned blocks = (n_rows + block - 1u) / block;
    zk_running_kernel<<<blocks, block>>>(frac, d_out, last, n_rows, n_cols);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    if (d_out) cuda_free_memory(d_out);
}

__global__ void zk_partial_sum_kernel(const qm31 *__restrict__ last,
                                       qm31 *__restrict__ partials, unsigned n_rows) {
    __shared__ qm31 sh[ZK_LOGUP_BLOCK];
    const unsigned tid = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + tid;
    sh[tid] = i < n_rows ? last[i] : ZK_QZERO;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] = add(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0u) partials[blockIdx.x] = sh[0];
}

extern "C" void zk_logup_partial_sums(const qm31 *last, qm31 *partials_host,
                                       unsigned n_rows, unsigned *n_partials) {
    const unsigned block = ZK_LOGUP_BLOCK;
    const unsigned blocks = (n_rows + block - 1u) / block;
    qm31 *d_partials = cuda_malloc<qm31>(blocks);
    zk_partial_sum_kernel<<<blocks, block>>>(last, d_partials, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaMemcpy(partials_host, d_partials,
                                   (size_t)blocks * sizeof(qm31), cudaMemcpyDeviceToHost));
    cuda_free_memory(d_partials);
    *n_partials = blocks;
}

__global__ void zk_shift_split_kernel(const qm31 *__restrict__ last, m31 **out,
                                       qm31 shift, unsigned n_rows) {
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= n_rows) return;
    const qm31 v = last[row];
    out[0][row] = sub(v.a.a, shift.a.a);
    out[1][row] = sub(v.a.b, shift.a.b);
    out[2][row] = sub(v.b.a, shift.b.a);
    out[3][row] = sub(v.b.b, shift.b.b);
}

extern "C" void zk_logup_finish_last(const qm31 *last, m31 **out_cols, qm31 shift,
                                      unsigned n_rows) {
    m31 **d_out = clone_to_device<m31 *>(out_cols, 4u);
    const unsigned block = n_rows < ZK_LOGUP_BLOCK ? n_rows : ZK_LOGUP_BLOCK;
    const unsigned blocks = (n_rows + block - 1u) / block;
    zk_shift_split_kernel<<<blocks, block>>>(last, d_out, shift, n_rows);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_out);
    // `finalize_last`'s prefix sum, on the device: the fork's own
    // `inclusive_prefix_sum` takes and returns bit-reversed circle-domain
    // order, which is the order the committed column is in.
    for (unsigned j = 0u; j < 4u; ++j) {
        inclusive_prefix_sum(out_cols[j], n_rows);
    }
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}
