// The pre-emit stand-in for the generated per-family dispatcher, so the crate
// links on a fork that has not had `b65 genemit`'s output installed yet. It is
// what the generator emits for zero families: any call aborts loudly, which
// can only happen with a `B7_M4_*` device switch on and no kernels installed.
// `m4b_kernels.py` replaces this file with the real generated dispatcher.
#include <cstdio>
#include <cstdlib>
#include "gen_b65_dispatch.cuh"

extern "C" void gen_b65_dispatch(
    const char *family, unsigned, unsigned,
    m31 **, unsigned,
    m31 **, unsigned,
    const qm31 *, unsigned,
    qm31 *, qm31 *,
    unsigned, unsigned) {
    std::fprintf(stderr,
                 "gen_b65_dispatch: no generation kernel for family %s -- run "
                 "`b65 genemit` and install it (m4b_kernels.py) first\n",
                 family);
    std::abort();
}
