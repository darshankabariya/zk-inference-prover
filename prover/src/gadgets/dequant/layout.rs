//! Which dequant variant each of the layer's seven matmuls is proved under.
//!
//! The row map is not here -- that is `crate::layout::MatLayout`, shared with
//! three other gadgets. What is here is the layer's matmul list and the two
//! variant assignments over it, which are a property of the spec the witness was
//! extracted under.

use crate::gadgets::dequant::Variant as DeqVariant;

/// The seven layer matmuls, in the order every component and every selector reads
/// them.  Which dequant gadget each takes is *not* fixed here: it is a property
/// of the spec the witness was extracted under, and the two differ.
pub const MAT_NAMES: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];

/// B6.5's mixed dequant layout, which the B6 review's per-matmul attribution
/// licensed: the power-of-two dequant where it is free (q/k/v/o, -0.43%) and the
/// bf16 chain where it is not.
pub const MAT_VARIANTS: [(&str, DeqVariant); 7] = [
    ("q", DeqVariant::Pow2), ("k", DeqVariant::Pow2), ("v", DeqVariant::Pow2),
    ("o", DeqVariant::Pow2),
    ("gate", DeqVariant::Chain), ("up", DeqVariant::Chain), ("down", DeqVariant::Chain),
];

/// B10's layout: the bf16 chain on all seven.  This is what B8 licensed
/// (`b65-design` costs +11.69% perplexity at 8B against `b6-design-p2o`'s
/// +8.81%) and what the denominator charges, so it is the layout of the spec
/// that is both scored and proved.
pub const MAT_VARIANTS_CHAIN: [(&str, DeqVariant); 7] = [
    ("q", DeqVariant::Chain), ("k", DeqVariant::Chain), ("v", DeqVariant::Chain),
    ("o", DeqVariant::Chain),
    ("gate", DeqVariant::Chain), ("up", DeqVariant::Chain), ("down", DeqVariant::Chain),
];

/// B10 step 0: the layout by name.  `zk-prover` is B6.5's mixed one, `b10` the chain
/// on all seven; a real witness carries its own (`deq_variants` in the manifest,
/// written by `extract.py` from the spec) and this is only the fallback and the
/// synthetic layer's choice.
pub fn mat_variants(layout: &str) -> [(&str, DeqVariant); 7] {
    match layout {
        "" | "b65" => MAT_VARIANTS,
        "b10" | "chain" => MAT_VARIANTS_CHAIN,
        // stream Z: the wired layout on a synthetic layer -- the chain
        // everywhere, the delayed requantize on v.  A smoke shape for the fold's
        // AIR, witness generation and LogUp before a real witness is anywhere near it.
        "zfold" => {
            let mut v = MAT_VARIANTS_CHAIN;
            v[2] = ("v", DeqVariant::Fold);
            v
        }
        other => panic!("unknown dequant layout {other} (b65 | b10 | zfold)"),
    }
}
