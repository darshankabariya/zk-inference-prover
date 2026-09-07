// The pre-emit stand-in for the generated per-family dispatcher, so the crate
// links on a fork that has not had `zk-prover genemit`'s output installed yet. It is
// what the generator emits for zero families: any call aborts loudly, which
// can only happen with a `ZK_DEVICE_*` device switch on and no kernels installed.
// `m4b_kernels.py` replaces this file with the real generated dispatcher.
#include <cstdio>
#include <cstdlib>
#include "gen_zk_dispatch.cuh"

extern "C" void gen_zk_dispatch(
    const char *family, unsigned, unsigned,
    m31 **, unsigned,
    m31 **, unsigned,
    const qm31 *, unsigned,
    qm31 *, qm31 *,
    unsigned, unsigned) {
    std::fprintf(stderr,
                 "gen_zk_dispatch: no generation kernel for family %s -- run "
                 "`zk-prover genemit` and install it (m4b_kernels.py) first\n",
                 family);
    std::abort();
}

// The unpaired variant, which B11 stream T added to the emitter and to
// `prover/src/gkr_key_tile.rs` without adding it here. That omission cost the
// stand-in the one property it exists for: from stream T until 2026-09-01 the
// cuda arm did not LINK on a fork with no installed store
// (`rust-lld: error: undefined symbol: gen_zk_feed_dispatch`), so every box
// needed a kernel-store install before it could build both arms at all. The
// generator emits exactly this fallthrough for zero families.
extern "C" void gen_zk_feed_dispatch(
    const char *family, unsigned, unsigned,
    m31 **, unsigned,
    m31 **, unsigned,
    const qm31 *, unsigned,
    qm31 **, qm31 **,
    unsigned, unsigned) {
    std::fprintf(stderr,
                 "gen_zk_feed_dispatch: no feed kernel for family %s -- run "
                 "`zk-prover genemit` and install it (m4b_kernels.py) first\n",
                 family);
    std::abort();
}
