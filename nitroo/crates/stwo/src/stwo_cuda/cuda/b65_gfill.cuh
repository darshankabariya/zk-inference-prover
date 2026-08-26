// B11 stream M: the glue's *generation*, on the device.
//
// `gen/glue` was the one named host term with no device path at all -- 49.8 s
// of seq3's 190.9 s wall at C=4K, 97% of it inside four `glue::glue_region`
// calls per chunk. This is that fill.
//
// **Interpreted, not emitted, and the measurement is why.** The alternative was
// a new `genemit` target compiling each region's straight-line `Prog` to CUDA,
// the way the constraint emitter already compiles far bigger programs. Against
// that: a glue region's program is *tiny and uniform* -- the sequence's `gn1`
// and `gn2` are 255 add slots over 256 leaves, `gn1b`/`gn2b` 31 over 32 -- and
// the rows are many (16,384 and 512). One thread owns a row and walks the
// program from a descriptor array that fits in L1; the per-slot dispatch is a
// handful of instructions against a normalisation and four table lookups, so
// the ratio interpretation is bad at never arises. And it needs no per-shape
// emit at all: nothing here can churn an installed family's bytes, and a new
// glue shape costs no ptxas time -- which, on the family whose constraint
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
//    the same property that makes `b65_fill.cu` fast.
//  * **the unary is a table, not a function.** `un_image_parts` is
//    f64 -> f32 -> bf16 through `rsqrt` and `silu`; reproducing that bit for bit
//    on device transcendentals would be the one genuinely risky part of the
//    port. It is not needed: `glue::un_table` is that function tabulated over
//    every legal `(sm, ef)`, indexed by `val_at`, and the AIR reads the table
//    too. So the kernel looks up what the host computes, and mode 2 is what
//    says they agree.
//
// The six index maps (`pow_at`, `val_at`, `smul_at`, `norm_at`, `frac_at` and
// the leaf/slot column maps) are all closed forms in `glue.rs`, so they port as
// written -- 20.1's property again, on a sixth family.
#ifndef B65_GFILL_CUH
#define B65_GFILL_CUH
#include "fields.cuh"

#define B65_GFILL_NONE 0xffffffffu

// `glue.rs`'s constants, which the kernel needs as literals.
#define B65G_DMAX 12u
#define B65G_UMIN 260
#define B65G_EXP_BIAS 134

// The negative controls (`B7_M4_GFILLMUT`). Each is a way this fill could
// plausibly be wrong and mode 2 must catch. Section 5 rule 5: a control that
// cannot fire on the shape under test is a finding, not a pass, so `nomul` is
// on the list even though the sequence's four regions are all `Add` -- it fires
// nowhere here, and that is recorded rather than hidden.
#define B65G_MUT_NONE 0u
#define B65G_MUT_NOPERM 1u    // write at the coset row, not the committed one
#define B65G_MUT_ANCHOR 2u    // etop one too high on one slot per 1,024 rows
#define B65G_MUT_NOBUMP 3u    // drop one multiplicity per 1,024 rows
#define B65G_MUT_NOSTICKY 4u  // ignore the sticky bit in the normalisation key
#define B65G_MUT_CLAMP 5u     // feed a slot's unclamped exponent downstream
#define B65G_MUT_NOMUL 6u     // the multiply's carry, dropped
#define B65G_MUT_OFFBY 7u     // read a slot operand from the neighbouring row

// One leaf group: `Prog::groups[g]` plus where its cells and its witness live.
struct B65GGroup {
    unsigned n;         // leaves in this group
    unsigned leaf_col;  // `Prog::leaf_at(g, 0)`
    unsigned width;     // `Prog::leaf_width(g)`: 2, or 4 with a unary
    unsigned un;        // 0 = `UnKind::None`, else 1 + the unary's table index
    unsigned flat;      // the group's first leaf in the row-flat leaf order
};

// One slot: `Prog::slots[s]` with every `Src` already resolved to the column
// its `(sm, ef)` lives in. `B65_GFILL_NONE` is `Src::Zero`.
struct B65GSlot {
    unsigned a_sm, a_ef;
    unsigned b_sm, b_ef;
    unsigned base;        // `Prog::slot_at(s, round)`
    unsigned o_sm, o_ef;  // `Prog::out_cols(s, round)`
    unsigned from_slot;   // bit 0: operand a is a slot output; bit 1: b is
};

// One counter array, as `GlueTables` holds it.
struct B65GMult {
    unsigned *m;
    unsigned n;
};

struct B65GlueCfg {
    unsigned log_rows;
    unsigned rows;       // 1 << log_rows
    unsigned live;       // `GlueWitness::rows`; rows past it are inert
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

extern "C" void b65_fill_glue(
    // the witness: `rows * n_leaves` bf16 patterns, widened, in row-flat order
    const unsigned *wit,
    const B65GGroup *groups,
    const B65GSlot *slots,
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
    // `rows * out_w` each: the joined pattern, and the natural pair the ties read
    unsigned *out_pat,
    int *out_nat,
    // `rows * n_leaves` pairs, likewise
    int *in_nat,
    B65GMult m_pow,
    B65GMult m_norm,
    B65GMult m_val,
    B65GMult m_frac,
    B65GMult m_smul,
    const B65GMult *m_un,
    // [0] off_table, [1] bad_exp, [2] leaves whose unary image is off the table
    unsigned *counts,
    B65GlueCfg cfg);

#endif
