#include <cstring>
#include <cstdio>
#include <cstdlib>

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

// B11 stream T: one round's sums for every live instance, in one launch.
//
// Stream R named the binding wall: `sumcheck::prove_batch` asks each live
// oracle separately every round, and the next challenge depends on the answer,
// so a chunk pays one launch plus one blocking copy home per instance per round
// -- about 4,800 round trips at 30-36 us. Within a round every live oracle has
// the same number of variables and shares one `eq` table and one `lambda`, so
// the whole round is this kernel: `blockIdx.y` is the instance, `blockIdx.x`
// the chunk, and the partials land in that instance's own slice. The per-thread
// arithmetic is copied from the four single-instance kernels above, term for
// term, because it has to produce the same field elements they do.
//
// `kind`: 0 grand product, 1 LogUp generic, 2 LogUp multiplicities, 3 singles.
// `a` is the numerator side (qm31, or m31 for multiplicities, unused for
// singles), `b` the denominator side (unused for grand product).
__global__ void gkr_sum_batch_kernel(
        const void *const *a, const qm31 *const *b, const uint32_t *kind,
        const qm31 *eq, qm31 lambda, uint32_t n_terms, qm31 *partials,
        uint32_t n_blocks) {
    const uint32_t inst = blockIdx.y;
    qm31 *out = partials + 2u * (size_t)inst * n_blocks;
    const uint32_t k = kind[inst];
    qm31 acc0 = qm31_zero(), acc2 = qm31_zero();

    if (k == 0u) {
        const qm31 *inp = (const qm31 *)a[inst];
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
             i += gridDim.x * blockDim.x) {
            qm31 a00 = inp[2 * i], a01 = inp[2 * i + 1];
            qm31 a10 = inp[2 * (n_terms + i)], a11 = inp[2 * (n_terms + i) + 1];
            qm31 a20 = sub(dbl(a10), a00), a21 = sub(dbl(a11), a01);
            qm31 e = eq[i];
            acc0 = add(acc0, mul(e, mul(a00, a01)));
            acc2 = add(acc2, mul(e, mul(a20, a21)));
        }
    } else if (k == 1u) {
        const qm31 *num = (const qm31 *)a[inst];
        const qm31 *den = b[inst];
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
    } else if (k == 2u) {
        const m31 *num = (const m31 *)a[inst];
        const qm31 *den = b[inst];
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_terms;
             i += gridDim.x * blockDim.x) {
            m31 n00 = num[2 * i], n01 = num[2 * i + 1];
            qm31 d00 = den[2 * i], d01 = den[2 * i + 1];
            m31 n10 = num[2 * (n_terms + i)], n11 = num[2 * (n_terms + i) + 1];
            qm31 d10 = den[2 * (n_terms + i)], d11 = den[2 * (n_terms + i) + 1];
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
    } else {
        const qm31 *den = b[inst];
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
    }
    gkr_block_reduce_pair(acc0, acc2, out);
}

// The three small host arrays (two pointer tables and the kinds) are uploaded
// once per round; at fifteen instances that is under half a kilobyte against
// the fourteen launches and fourteen blocking copies it replaces.
void gkr_sum_batch(
        const void *const *a_host, const qm31 *const *b_host, const uint32_t *kind_host,
        uint32_t n_inst, const qm31 *eq, qm31 lambda, uint32_t n_terms,
        qm31 *partials, uint32_t n_blocks) {
    const void **d_a = clone_to_device<const void *>((const void **)a_host, n_inst);
    const qm31 **d_b = clone_to_device<const qm31 *>((const qm31 **)b_host, n_inst);
    uint32_t *d_kind = clone_to_device<uint32_t>((uint32_t *)kind_host, n_inst);
    dim3 grid(n_blocks, n_inst);
    gkr_sum_batch_kernel<<<grid, GKR_SUM_BLOCK>>>(
        d_a, d_b, d_kind, eq, lambda, n_terms, partials, n_blocks);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // Stream-ordered: the frees queue behind the kernel, and the caller's copy
    // home behind both.
    cuda_free_memory(d_a);
    cuda_free_memory(d_b);
    cuda_free_memory(d_kind);
}

// B11 stream T: one round's variable fixes, for every column of every folded
// instance, in one launch.
//
// The other half of the round's launch traffic: `Mle::fix_first_variable` is
// one launch per column, and a round folds two columns per LogUp instance. The
// arithmetic is `mle.cu`'s template body, term for term; `is_base` picks the
// source type, because a `LogUpMultiplicities` numerator column is base-field
// and folds into a secure-field one. Every column in a batch has the same
// length -- the layers a round folds all have the same number of variables.
__global__ void gkr_fix_batch_kernel(
        const void *const *src, const uint32_t *is_base, qm31 *const *dst,
        qm31 assignment, uint32_t half) {
    const uint32_t col = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    if (is_base[col]) {
        const m31 *e = (const m31 *)src[col];
        m31 l = e[i], r = e[i + half];
        dst[col][i] = add(l, mul(sub(r, l), assignment));
    } else {
        const qm31 *e = (const qm31 *)src[col];
        qm31 l = e[i], r = e[i + half];
        dst[col][i] = add(l, mul(sub(r, l), assignment));
    }
}

void gkr_fix_batch(
        const void *const *src_host, const uint32_t *is_base_host, qm31 *const *dst_host,
        uint32_t n_cols, qm31 assignment, uint32_t half) {
    const void **d_src = clone_to_device<const void *>((const void **)src_host, n_cols);
    uint32_t *d_base = clone_to_device<uint32_t>((uint32_t *)is_base_host, n_cols);
    qm31 **d_dst = clone_to_device<qm31 *>((qm31 **)dst_host, n_cols);
    const uint32_t block = half < 256u ? half : 256u;
    const uint32_t blocks = (half + block - 1u) / block;
    dim3 grid(blocks, n_cols);
    gkr_fix_batch_kernel<<<grid, block>>>(d_src, d_base, d_dst, assignment, half);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    // Stream-ordered, like every free on this path.
    cuda_free_memory(d_src);
    cuda_free_memory(d_base);
    cuda_free_memory(d_dst);
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
    // B11 stream T: one round's partials for every live instance now arrive in
    // one copy, so the staging buffer is sized for the whole round rather than
    // one instance.  Stream X made the launch group 256 instances
    // (`GKR_SUM_MAX_INSTANCES`, lookups/gkr.rs) and this stayed at T's 64: the
    // first round with more than 64 live instances -- an E11 block-2 chunk's,
    // 264 x 66 pairs -- hit the abort below (2026-09-02).  Sized to the group.
    static const uint32_t PINNED_PAIRS = 264u * 256u;
    // Per thread, like the partials buffer it reads: the test harness runs
    // tests in parallel and two concurrent readbacks must not share staging.
    // (Found by stream T: the batched copy is larger and takes longer, so the
    // window this always had became wide enough to fail a test.)
    static thread_local qm31 *pinned = nullptr;
    if (pinned == nullptr) {
        ASSERT_CUDA_SUCCESS(cudaMallocHost((void **)&pinned,
                                           2 * PINNED_PAIRS * sizeof(qm31)));
    }
    if (n_pairs > PINNED_PAIRS) {
        fprintf(stderr, "gkr_read_partials: %u pairs, staging holds %u\n",
                n_pairs, PINNED_PAIRS);
        abort();
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

// ---- B11 stream U: the link's committed columns, built where they are used --
//
// Stream T left the coefficient column folded on the card and then downloaded
// it, because the four column groups the univariate IOP commits -- the succinct
// carry quotients, the folded coefficient column, the eq column and the shifted
// prefix sum -- were built by five host loops. T itemised those loops at
// 0.190 s per C=16384 chunk and named them the single largest remaining item in
// the arm. Every one of them is pointwise or scan shaped, so they move here and
// the column never leaves the card: the commit path already takes device
// columns (`device::AnyCol::OwnedDev`).
//
// Each kernel reproduces its host loop value for value, which is what lets the
// wall digests stay byte identical. The arithmetic is exact field arithmetic
// over the same terms, and the two order sensitive pieces -- the carry
// quotients' scatter and the prefix sum's walk in coset order -- use the very
// index maps the host uses, `coset_index_to_circle_domain_index` composed with
// `bit_reverse`, both already device helpers in utils.cuh.

#define GKR_LINK_BLOCK 256

// One qm31 column split into the four M31 coordinate columns the tree commits.
// This is the host's `into_secure_column_by_coords`, whose packed layout is the
// same coordinate order (`to_m31_array`) that `qm31` has in memory here.
__global__ void b65_link_split_kernel(const qm31 *__restrict__ src,
                                      m31 *__restrict__ o0, m31 *__restrict__ o1,
                                      m31 *__restrict__ o2, m31 *__restrict__ o3,
                                      unsigned n_rows) {
    const unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= n_rows) return;
    const qm31 v = src[row];
    o0[row] = v.a.a;
    o1[row] = v.a.b;
    o2[row] = v.b.a;
    o3[row] = v.b.b;
}

void b65_link_split(const qm31 *src, m31 *o0, m31 *o1, m31 *o2, m31 *o3,
                    unsigned n_rows) {
    cudaGetLastError();
    const unsigned block = n_rows < GKR_LINK_BLOCK ? n_rows : GKR_LINK_BLOCK;
    const unsigned blocks = (n_rows + block - 1u) / block;
    b65_link_split_kernel<<<blocks, block>>>(src, o0, o1, o2, o3, n_rows);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

// The prefix sum's row differences and the claim's own total, in one pass.
//
// `out[row] = coeff[row] * eq[row] - shift` is what the prefix-sum constraint
// reads, and the unshifted products sum to the GKR claim -- the boundary the
// host builder asserts loudly so a weight bug fails the prover rather than only
// the verifier. The per-block partials come home and the host adds them, which
// changes nothing: field addition is exact and associative.
__global__ void b65_link_terms_kernel(const qm31 *__restrict__ coeff,
                                      const qm31 *__restrict__ eq, qm31 shift,
                                      qm31 *__restrict__ out,
                                      qm31 *__restrict__ partials,
                                      unsigned n_rows) {
    __shared__ qm31 red[GKR_LINK_BLOCK];
    qm31 acc = qm31_zero();
    for (unsigned row = threadIdx.x + blockDim.x * blockIdx.x; row < n_rows;
         row += blockDim.x * gridDim.x) {
        const qm31 p = mul(coeff[row], eq[row]);
        acc = add(acc, p);
        if (out) out[row] = sub(p, shift);
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2u; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = add(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = red[0];
}

void b65_link_terms(const qm31 *coeff, const qm31 *eq, qm31 shift, qm31 *out,
                    qm31 *partials_host, unsigned n_blocks, unsigned n_rows) {
    cudaGetLastError();
    qm31 *d_part = cuda_malloc<qm31>(n_blocks);
    b65_link_terms_kernel<<<n_blocks, GKR_LINK_BLOCK>>>(coeff, eq, shift, out,
                                                        d_part, n_rows);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_mem_copy_device_to_host<qm31>(d_part, partials_host, n_blocks);
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(0));
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d_part);
}

// The succinct carry-quotient column, straight into its coordinate columns.
//
// The column takes one of only `n_variables` values per half coset, indexed by
// the trailing-ones count of the half-coset row, so the whole column is two
// small tables and an index map. The host loop walks half-coset rows and
// scatters two rows each; so does this, thread for row.
__global__ void b65_link_carry_kernel(const qm31 *__restrict__ half0,
                                      const qm31 *__restrict__ half1,
                                      unsigned log_size,
                                      m31 *__restrict__ o0, m31 *__restrict__ o1,
                                      m31 *__restrict__ o2, m31 *__restrict__ o3,
                                      unsigned half_coset_size) {
    const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= half_coset_size) return;
    // trailing ones of `i`: the first zero bit's position
    const unsigned nt = (unsigned)__ffs((int)~i) - 1u;
    const unsigned r0 = bit_reverse(coset_index_to_circle_domain_index(i * 2u, log_size),
                                    log_size);
    const unsigned r1 = bit_reverse(coset_index_to_circle_domain_index(i * 2u + 1u, log_size),
                                    log_size);
    const qm31 v0 = half0[nt];
    const qm31 v1 = half1[nt];
    o0[r0] = v0.a.a; o1[r0] = v0.a.b; o2[r0] = v0.b.a; o3[r0] = v0.b.b;
    o0[r1] = v1.a.a; o1[r1] = v1.a.b; o2[r1] = v1.b.a; o3[r1] = v1.b.b;
}

void b65_link_carry(const qm31 *half0_host, const qm31 *half1_host, unsigned n_vars,
                    unsigned log_size, m31 *o0, m31 *o1, m31 *o2, m31 *o3) {
    cudaGetLastError();
    qm31 *d0 = clone_to_device<qm31>((qm31 *)half0_host, n_vars);
    qm31 *d1 = clone_to_device<qm31>((qm31 *)half1_host, n_vars);
    const unsigned half = (1u << log_size) / 2u;
    const unsigned block = half < GKR_LINK_BLOCK ? half : GKR_LINK_BLOCK;
    const unsigned blocks = (half + block - 1u) / block;
    b65_link_carry_kernel<<<blocks, block>>>(d0, d1, log_size, o0, o1, o2, o3, half);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(d0);
    cuda_free_memory(d1);
}

// ---- the prefix sum -------------------------------------------------------
//
// `simd::prefix_sum::inclusive_prefix_sum` takes evaluations in bit-reversed
// CircleDomain order and returns the inclusive prefix sum *in coset order*,
// written back in bit-reversed CircleDomain order. Its SIMD implementation
// folds the two orderings into the sweep itself; here the two permutations are
// explicit -- a gather into coset order, a plain scan, a scatter back -- which
// is the same function of the same values, and cheaper to be sure of.
//
// The composite maps are the ones utils.cuh already carries: coset row `c`
// lives at `bit_reverse(coset_index_to_circle_domain_index(c))`, and its
// inverse takes a stored row `i` to `circle_domain_index_to_coset_index(
// bit_reverse(i))`.

#define LINK_SCAN_THREADS 256
#define LINK_SCAN_ITEMS 4
#define LINK_SCAN_TILE (LINK_SCAN_THREADS * LINK_SCAN_ITEMS)

// The four coordinate arrays are scanned in one launch each pass: `blockIdx.y`
// picks the array and `stride` is its pitch.
__global__ void b65_link_scan_tile_kernel(m31 *data, m31 *bsums, unsigned n,
                                          unsigned stride, unsigned n_blocks) {
    __shared__ m31 sh[LINK_SCAN_THREADS];
    m31 *arr = data + (size_t)blockIdx.y * stride;
    const unsigned base = blockIdx.x * LINK_SCAN_TILE + threadIdx.x * LINK_SCAN_ITEMS;
    m31 v[LINK_SCAN_ITEMS];
    m31 sum = 0u;
    for (unsigned k = 0; k < LINK_SCAN_ITEMS; ++k) {
        const unsigned idx = base + k;
        sum = add(sum, idx < n ? arr[idx] : 0u);
        v[k] = sum;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned d = 1; d < LINK_SCAN_THREADS; d <<= 1) {
        const m31 t = threadIdx.x >= d ? sh[threadIdx.x - d] : 0u;
        __syncthreads();
        if (threadIdx.x >= d) sh[threadIdx.x] = add(sh[threadIdx.x], t);
        __syncthreads();
    }
    const m31 off = threadIdx.x ? sh[threadIdx.x - 1] : 0u;
    for (unsigned k = 0; k < LINK_SCAN_ITEMS; ++k) {
        const unsigned idx = base + k;
        if (idx < n) arr[idx] = add(v[k], off);
    }
    if (threadIdx.x == LINK_SCAN_THREADS - 1)
        bsums[(size_t)blockIdx.y * n_blocks + blockIdx.x] = sh[LINK_SCAN_THREADS - 1];
}

// Every tile but the first takes its predecessors' total.
__global__ void b65_link_scan_add_kernel(m31 *data, const m31 *bsums, unsigned n,
                                         unsigned stride, unsigned n_blocks) {
    if (blockIdx.x == 0) return;
    m31 *arr = data + (size_t)blockIdx.y * stride;
    const m31 off = bsums[(size_t)blockIdx.y * n_blocks + blockIdx.x - 1u];
    const unsigned base = blockIdx.x * LINK_SCAN_TILE + threadIdx.x * LINK_SCAN_ITEMS;
    for (unsigned k = 0; k < LINK_SCAN_ITEMS; ++k) {
        const unsigned idx = base + k;
        if (idx < n) arr[idx] = add(arr[idx], off);
    }
}

// An inclusive scan of `n_arrays` M31 arrays of length `n` at pitch `stride`.
// Three levels cover every link height a chunk has: a tile is 1,024 rows, so
// 2^22 rows reduce to 4,096 tile sums, then to 4, then to 1.
static void link_scan(m31 *data, unsigned n, unsigned stride, unsigned n_arrays) {
    if (n == 0) return;
    const unsigned n_blocks = (n + LINK_SCAN_TILE - 1u) / LINK_SCAN_TILE;
    m31 *bsums = cuda_malloc<m31>(n_blocks * n_arrays);
    const dim3 grid(n_blocks, n_arrays);
    b65_link_scan_tile_kernel<<<grid, LINK_SCAN_THREADS>>>(data, bsums, n, stride,
                                                           n_blocks);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    if (n_blocks > 1) {
        link_scan(bsums, n_blocks, n_blocks, n_arrays);
        b65_link_scan_add_kernel<<<grid, LINK_SCAN_THREADS>>>(data, bsums, n, stride,
                                                              n_blocks);
        ASSERT_CUDA_SUCCESS(cudaGetLastError());
    }
    cuda_free_memory(bsums);
}

__global__ void b65_link_gather_kernel(const qm31 *__restrict__ src,
                                       m31 *__restrict__ dst, unsigned log_size,
                                       unsigned n) {
    const unsigned c = threadIdx.x + blockDim.x * blockIdx.x;
    if (c >= n) return;
    const unsigned r = bit_reverse(coset_index_to_circle_domain_index(c, log_size),
                                  log_size);
    const qm31 v = src[r];
    dst[c] = v.a.a;
    dst[(size_t)n + c] = v.a.b;
    dst[(size_t)2 * n + c] = v.b.a;
    dst[(size_t)3 * n + c] = v.b.b;
}

__global__ void b65_link_scatter_kernel(const m31 *__restrict__ src,
                                        m31 *__restrict__ o0, m31 *__restrict__ o1,
                                        m31 *__restrict__ o2, m31 *__restrict__ o3,
                                        unsigned log_size, unsigned n) {
    const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    const unsigned c = circle_domain_index_to_coset_index(bit_reverse(i, log_size),
                                                          log_size);
    o0[i] = src[c];
    o1[i] = src[(size_t)n + c];
    o2[i] = src[(size_t)2 * n + c];
    o3[i] = src[(size_t)3 * n + c];
}

void b65_link_prefix(const qm31 *src, m31 *o0, m31 *o1, m31 *o2, m31 *o3,
                     unsigned log_size) {
    cudaGetLastError();
    const unsigned n = 1u << log_size;
    m31 *scratch = cuda_malloc<m31>(4u * n);
    const unsigned block = n < GKR_LINK_BLOCK ? n : GKR_LINK_BLOCK;
    const unsigned blocks = (n + block - 1u) / block;
    b65_link_gather_kernel<<<blocks, block>>>(src, scratch, log_size, n);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    link_scan(scratch, n, n, 4u);
    b65_link_scatter_kernel<<<blocks, block>>>(scratch, o0, o1, o2, o3, log_size, n);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    cuda_free_memory(scratch);
}
