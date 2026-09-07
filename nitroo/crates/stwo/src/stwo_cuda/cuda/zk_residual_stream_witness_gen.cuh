// B11 stream M: the residual stream's *generation*, on the device.
//
// `gen/residual_stream` was the one named host term with no device path at all -- 49.8 s
// of seq3's 190.9 s wall at C=4K, 97% of it inside four `residual_stream::residual_stream_component`
// calls per chunk. This is that witness generation.
//
// **Interpreted, not emitted, and the measurement is why.** The alternative was
// a new `genemit` target compiling each component's straight-line `Prog` to CUDA,
// the way the constraint emitter already compiles far bigger programs. Against
// that: a residual-stream component's program is *tiny and uniform* -- the sequence's `gn1`
// and `gn2` are 255 add slots over 256 leaves, `gn1b`/`gn2b` 31 over 32 -- and
// the rows are many (16,384 and 512). One thread owns a row and walks the
// program from a descriptor array that fits in L1; the per-slot dispatch is a
// handful of instructions against a normalisation and four table lookups, so
// the ratio interpretation is bad at never arises. And it needs no per-shape
// emit at all: nothing here can churn an installed family's bytes, and a new
// residual-stream shape costs no ptxas time -- which, on the family whose constraint
// kernel took 617.7 s to compile before stream K split it, is not a small
// thing.
//
// Three things make the interpreter cheap enough to be worth it:
//
//  * **every intermediate is already a committed cell.** The host keeps `lin`,
//    `lun` and `outs` per row; the kernel does not, because a slot's operands
//    are columns it (or a leaf) already wrote. `Src` resolves to a *column
//    index* on the host, so the kernel's slot descriptor is column indices and
//    the whole `Src` enum disappears. Per-thread state is a few registers.
//  * **the reads are coalesced.** One thread per committed row means
//    consecutive threads read consecutive cells of the same column, which is
//    the same property that makes `zk_dequant_witness_gen.cu` fast.
//  * **the unary is a table, not a function.** `un_image_parts` is
//    f64 -> f32 -> bf16 through `rsqrt` and `silu`; reproducing that bit for bit
//    on device transcendentals would be the one genuinely risky part of the
//    port. It is not needed: `residual_stream::un_table` is that function tabulated over
//    every legal `(sm, ef)`, indexed by `val_at`, and the AIR reads the table
//    too. So the kernel looks up what the host computes, and mode 2 is what
//    says they agree.
//
// The six index maps (`pow_at`, `val_at`, `smul_at`, `norm_at`, `frac_at` and
// the leaf/slot column maps) are all closed forms in `residual_stream.rs`, so they port as
// written -- 20.1's property again, on a sixth family.
#ifndef ZK_RESIDUAL_STREAM_WITNESS_GEN_CUH
#define ZK_RESIDUAL_STREAM_WITNESS_GEN_CUH
#include "fields.cuh"

#define ZK_RESIDUAL_STREAM_WITNESS_GEN_NONE 0xffffffffu

// `residual_stream.rs`'s constants, which the kernel needs as literals.
#define ZK_RESIDUAL_STREAM_DMAX 12u
#define ZK_RESIDUAL_STREAM_UMIN 260
#define ZK_RESIDUAL_STREAM_EXP_BIAS 134

// The negative controls (`ZK_MUT_WITNESS_GEN_RESIDUAL_STREAM`). Each is a way this witness generation could
// plausibly be wrong and mode 2 must catch. Section 5 rule 5: a control that
// cannot fire on the shape under test is a finding, not a pass, so `nomul` is
// on the list even though the sequence's four components are all `Add` -- it fires
// nowhere here, and that is recorded rather than hidden.
#define ZK_RESIDUAL_STREAM_MUT_NONE 0u
#define ZK_RESIDUAL_STREAM_MUT_NOPERM 1u    // write at the coset row, not the committed one
#define ZK_RESIDUAL_STREAM_MUT_ANCHOR 2u    // etop one too high on one slot per 1,024 rows
#define ZK_RESIDUAL_STREAM_MUT_NOBUMP 3u    // drop one multiplicity per 1,024 rows
#define ZK_RESIDUAL_STREAM_MUT_NOSTICKY 4u  // ignore the sticky bit in the normalisation key
#define ZK_RESIDUAL_STREAM_MUT_CLAMP 5u     // feed a slot's unclamped exponent downstream
#define ZK_RESIDUAL_STREAM_MUT_NOMUL 6u     // the multiply's carry, dropped
#define ZK_RESIDUAL_STREAM_MUT_OFFBY 7u     // read a slot operand from the neighbouring row

// One leaf group: `Prog::groups[g]` plus where its cells and its witness live.
struct ZkResidualStreamGroup {
    unsigned n;         // leaves in this group
    unsigned leaf_col;  // `Prog::leaf_at(g, 0)`
    unsigned width;     // `Prog::leaf_width(g)`: 2, or 4 with a unary
    unsigned un;        // 0 = `UnKind::None`, else 1 + the unary's table index
    unsigned flat;      // the group's first leaf in the row-flat leaf order
};

// One slot: `Prog::slots[s]` with every `Src` already resolved to the column
// its `(sm, ef)` lives in. `ZK_RESIDUAL_STREAM_WITNESS_GEN_NONE` is `Src::Zero`.
struct ZkResidualStreamSlot {
    unsigned a_sm, a_ef;
    unsigned b_sm, b_ef;
    unsigned base;        // `Prog::slot_at(s, round)`
    unsigned o_sm, o_ef;  // `Prog::out_cols(s, round)`
    unsigned from_slot;   // bit 0: operand a is a slot output; bit 1: b is
};

// One counter array, as `ResidualStreamTables` holds it.
struct ZkResidualStreamMult {
    unsigned *m;
    unsigned n;
};

struct ZkResidualStreamCfg {
    unsigned log_rows;
    unsigned rows;       // 1 << log_rows
    unsigned live;       // `ResidualStreamWitness::rows`; rows past it are inert
    unsigned n_groups;
    unsigned n_leaves;   // `Prog::n_leaves()`, the row-flat leaf count
    unsigned n_slots;
    unsigned n_cols;
    unsigned kind;       // 0 = `Kind::Add`, 1 = `Kind::Mul`
    unsigned sticky;
    unsigned kmax;
    unsigned out_w;      // total width of `Prog::outs`
    unsigned mutate;
};

extern "C" void zk_witness_gen_residual_stream(
    // the witness: `rows * n_leaves` bf16 patterns, widened, in row-flat order
    const unsigned *wit,
    const ZkResidualStreamGroup *groups,
    const ZkResidualStreamSlot *slots,
    // `norm[i]` as (m7, expfield, nz); `smul[i]` as (c_sm, cy, nz), the first
    // already signed the way `sgn_of` reads it
    const unsigned *norm3,
    unsigned n_norm,
    const int *smul3,
    unsigned n_smul,
    // one `(osm, oef)` pair per legal `val_at` index, per unary table
    const int *const *un2,
    unsigned n_un,
    unsigned n_val,
    // `Prog::outs`, flattened: `out_slot[k]` is the slot the k-th output holds
    const unsigned *out_slot,
    unsigned *const *cols,
    // `rows * out_w` each: the joined pattern, and the natural pair the bindings read
    unsigned *out_pat,
    int *out_nat,
    // `rows * n_leaves` pairs, likewise
    int *in_nat,
    ZkResidualStreamMult m_pow,
    ZkResidualStreamMult m_norm,
    ZkResidualStreamMult m_val,
    ZkResidualStreamMult m_frac,
    ZkResidualStreamMult m_smul,
    const ZkResidualStreamMult *m_un,
    // [0] off_table, [1] bad_exp, [2] leaves whose unary image is off the table
    unsigned *counts,
    ZkResidualStreamCfg cfg);

#endif
