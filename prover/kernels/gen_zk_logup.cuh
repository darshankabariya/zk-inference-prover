// B7a milestone 4 (18.7): the LogUp phases that are not per-family.
//
// Written by hand and small, because none of it depends on the AIR: the
// per-family half is generated (`gen_zk_<family>.cu`) and stops at the
// numerators and denominators, and everything downstream of that is the same
// four steps for every component. They mirror `plogup`'s phases exactly --
// same batching, same inversion, same addition order, same shift, same prefix
// sum -- so the device columns are the host columns or the proof digest says
// otherwise.
#ifndef GEN_ZK_LOGUP_CUH
#define GEN_ZK_LOGUP_CUH
#include "fields.cuh"

// phase A, after the batch inverse: `frac = num * den^{-1}`, pointwise.
extern "C" void zk_logup_scale(const qm31 *num, const qm31 *inv, qm31 *frac, unsigned n);

// phase B: the pointwise running sum across columns. Column `c` of the output
// is `sum_{c' <= c} frac_{c'}`; the first `n_cols - 1` are final and are
// written straight out as four m31 columns each, and the last is handed to
// phase C as qm31.
extern "C" void zk_logup_running(const qm31 *frac, m31 **out_cols, qm31 *last,
                                  unsigned n_rows, unsigned n_cols);

// phase C, part 1: block partials of the last column, which the host adds up
// into the claimed sum. A field sum is associative and commutative, so the
// split changes nothing about the value.
extern "C" void zk_logup_partial_sums(const qm31 *last, qm31 *partials_host,
                                       unsigned n_rows, unsigned *n_partials);

// phase C, part 2: subtract the cumsum shift, split into four m31 columns and
// scan each in place -- `finalize_last`, on the device.
extern "C" void zk_logup_finish_last(const qm31 *last, m31 **out_cols, qm31 shift,
                                      unsigned n_rows);

#endif
