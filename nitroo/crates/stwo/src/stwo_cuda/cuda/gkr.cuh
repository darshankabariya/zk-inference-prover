#ifndef GKR_H
#define GKR_H

#include "fields.cuh"

extern "C"
void gen_eq_evals(qm31 v, qm31 *y, uint32_t y_size, qm31 *evals, uint32_t evals_size);

// B11 stream R: the layer halving, one launch per call, every LogUp variant
// producing a generic layer exactly as the CPU backend does.
extern "C"
void gkr_next_layer_grand_product(const qm31 *inp, qm31 *out, uint32_t n_out);
extern "C"
void gkr_next_layer_logup_generic(
    const qm31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out);
extern "C"
void gkr_next_layer_logup_multiplicities(
    const m31 *num, const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out);
extern "C"
void gkr_next_layer_logup_singles(
    const qm31 *den, qm31 *num_out, qm31 *den_out, uint32_t n_out);

// B11 stream R: the round polynomial's evaluations at t = 0 and t = 2, as
// per-block partial pairs the host adds up. One launch per oracle per round.
extern "C"
void gkr_sum_grand_product(
    const qm31 *inp, const qm31 *eq, uint32_t n_terms, qm31 *partials, uint32_t n_blocks);
extern "C"
void gkr_sum_logup_generic(
    const qm31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
    uint32_t n_terms, qm31 *partials, uint32_t n_blocks);
extern "C"
void gkr_sum_logup_multiplicities(
    const m31 *num, const qm31 *den, const qm31 *eq, qm31 lambda,
    uint32_t n_terms, qm31 *partials, uint32_t n_blocks);
extern "C"
void gkr_sum_logup_singles(
    const qm31 *den, const qm31 *eq, qm31 lambda,
    uint32_t n_terms, qm31 *partials, uint32_t n_blocks);

// Profiling only.
extern "C"
void gkr_device_synchronize();

// Pool allocation without the zeroing pass; freed by `cuda_free_memory`.
extern "C"
qm31 *gkr_alloc_qm31(uint32_t n);

// The round-sum partials, read back through a pinned staging buffer.
extern "C"
void gkr_read_partials(const qm31 *dev, uint32_t n_pairs, qm31 *out_host);

// B11 stream T: the link's coefficient column, `konst + sum_j w_j * col_j`,
// folded over device-resident base columns.
extern "C"
void b65_link_coeff(m31 **cols, const qm31 *w_host, unsigned n_cols, qm31 k,
                    qm31 *out, unsigned n_rows);

#endif // GKR_H
