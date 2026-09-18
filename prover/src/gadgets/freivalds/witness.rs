//! The integers one Freivalds group is proved over, and the one place this
//! directory reads a `Layer`.
//!
//! The statement at the top of `mod.rs` has three matrices in it and this is
//! them: `A` the activations, `M` the weights of each matmul, `C` the product
//! the statement compares against.  Everything else a builder used to take --
//! which of the layer's matmuls the group is over, how wide they are, what the
//! group is called -- is here too, so the builders take one value and a shape
//! instead of a whole layer and a list of indices into it.
//!
//! It **borrows**.  A group's weights are megabytes on a real model (the head's
//! are gigabytes), and a witness that copied them would copy them once per
//! chunk; the composer holds the layer for the whole proof, so a view of it is
//! what the gadget wants.

use crate::layer::Layer;

/// One matmul of the group.
pub struct Mat<'a> {
    /// where the layer's own list holds it, which is the index the plan, the
    /// composed claims and the dequant components all name it by
    pub at: usize,
    /// the matmul's output width, the index `rv` weights
    pub n_out: usize,
    /// M: `(n_in, n_out)` row-major, the pinned weights
    pub w: &'a [i32],
    /// C: `(tokens, n_out)` row-major, the exact integer product.
    ///
    /// Nothing in a proof reads it -- this component never commits `C`, and in
    /// a composed proof the dequant component publishes the projection of it.
    /// The standalone run's self-check is what takes it.
    pub c: &'a [i64],
}

/// One Freivalds group's witness: the activations its matmuls share, and each
/// matmul's weights.
pub struct FreivaldsWitness<'a> {
    /// the group's name, which every preprocessed id of the component carries
    pub name: &'a str,
    /// A: `(tokens, n_in)` row-major, committed once for every matmul of the
    /// group -- which is what a group *is*, the matmuls that read one input
    pub x: &'a [i32],
    /// the hidden dimension, the index the matmuls sum over
    pub n_in: usize,
    pub tokens: usize,
    /// the group's matmuls, in the order it publishes their claims
    pub mats: Vec<Mat<'a>>,
}

impl<'a> FreivaldsWitness<'a> {
    /// One group's witness, off the layer the composer holds.
    ///
    /// This is the only place under `gadgets/freivalds/` that names a `Layer`:
    /// the gadget is over these integers and not over a decode step's witness
    /// format, and a builder that took the layer could reach anything in it.
    /// The two assertions are what a group means -- one committed input, one
    /// input width -- and they are checked here rather than by each builder.
    pub fn of_layer(l: &'a Layer, name: &'a str, mats: &[usize]) -> Self {
        let m0 = &l.mats[mats[0]];
        for i in mats {
            assert_eq!(l.mats[*i].n_in, m0.n_in, "Freivalds group mixes input widths");
            // q/k/v (and gate/up) must share one committed input
            assert_eq!(l.mats[*i].qx, m0.qx, "Freivalds group inputs differ");
        }
        FreivaldsWitness {
            name,
            x: &m0.qx,
            n_in: m0.n_in,
            tokens: m0.tokens,
            mats: mats
                .iter()
                .map(|i| {
                    let m = &l.mats[*i];
                    Mat { at: *i, n_out: m.n_out, w: &m.qw, c: &m.acc }
                })
                .collect(),
        }
    }

    /// The widest matmul of the group, which is what the row tiling is chosen
    /// over: every matmul of a group carries the same `pk` weights a row.
    pub fn n_out_max(&self) -> usize {
        self.mats.iter().map(|m| m.n_out).max().expect("a Freivalds group with no matmul")
    }

    /// How many weights the group pins, which is what its cost is per.
    pub fn n_weights(&self) -> usize {
        self.mats.iter().map(|m| m.w.len()).sum()
    }

    /// Where the layer's own list holds the group's matmuls.
    pub fn at(&self) -> Vec<usize> {
        self.mats.iter().map(|m| m.at).collect()
    }
}
