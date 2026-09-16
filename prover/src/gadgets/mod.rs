//! The gadgets: one directory each, and nothing here composes them.
//!
//! A gadget is one step of the decode the prover states on its own -- a
//! quantized matmul and its dequantization, the requantize seam in front of the
//! next one, the rotation, the attention block, the wiring family that threads
//! the residual stream, the embedding row, the argmax over the head's output.
//! Each one owns its column layout, its witness, its trace builder, its
//! constraints and the claims it publishes, and each one can be built, proved
//! and priced with no segment around it: that is what the `argmax`, `embed`,
//! `requant` and `residual-stream` subcommands do.
//!
//! What states a segment *over* the gadgets is `crate::compose`, and what binds
//! one gadget's output to the next one's input is the claims they publish --
//! `docs/statement.md` names every such edge.

pub mod argmax;
pub mod attention;
pub mod dequant;
pub mod embed;
pub mod freivalds;
pub mod requant;
pub mod residual_stream;
pub mod rope;
