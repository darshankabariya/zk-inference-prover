// The declaration of the per-family generation dispatcher. The definition in
// this tree is the pre-emit stand-in (see gen_zk_dispatch.cu); running
// `zk-prover genemit` and installing its output
// (install_witness_gen_kernels.py) replaces both files with the generated
// dispatcher and its family kernels.
#ifndef GEN_ZK_DISPATCH_CUH
#define GEN_ZK_DISPATCH_CUH
#include "fields.cuh"
extern "C" void gen_zk_dispatch(
    const char *family, unsigned shape, unsigned fingerprint,
    m31 **pre_cols, unsigned n_pre,
    m31 **base_cols, unsigned n_base,
    const qm31 *sc_host, unsigned n_sc,
    qm31 *num, qm31 *den,
    unsigned n_rows, unsigned n_cols);
// The unpaired variant, added by B11 stream T's device feed. The generated
// header declares both (`cudagen.rs`, `emit_dispatch`), so the stand-in must
// too -- without it `prover/src/gkr_key_tile.rs`'s `FeedPlan::witness_gen` leaves
// `gen_zk_feed_dispatch` undefined and the cuda arm does not link at all on a
// fork with no installed store.
extern "C" void gen_zk_feed_dispatch(
    const char *family, unsigned shape, unsigned fingerprint,
    m31 **pre_cols, unsigned n_pre,
    m31 **base_cols, unsigned n_base,
    const qm31 *sc_host, unsigned n_sc,
    qm31 **num_at, qm31 **den_at,
    unsigned n_rows, unsigned n_looks);
#endif
