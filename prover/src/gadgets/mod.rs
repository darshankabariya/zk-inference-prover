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

/// One step of the decode, proved with no segment around it.
///
/// The four associated types are what a gadget owes: the integers it is proved
/// over, everything about a run that is not those integers, what a negative
/// control corrupts, and what its own proof came to. They were already there in
/// every gadget, under seven different spellings; this is the spelling they
/// share.
///
/// The list of implementations is the census, and two things it says are worth
/// reading off it. `rope` is not on the list: the rotation and the q/k/v
/// quantize have no standalone proof, no subcommand and no control of their own,
/// and are exercised only inside a composed segment. `attention` and
/// `freivalds` are on it with `Tamper = ()`: what their controls corrupt is the
/// witness they are handed, not a value beside it, so there is nothing for this
/// to pass.
///
/// What the trait does **not** say is how a gadget composes. The columns it
/// commits, the lookups it draws and the claims it publishes to its neighbours
/// are still each gadget's own functions, called by name from `crate::compose`,
/// because their signatures do not yet agree: an interaction returns one, two,
/// four or five claimed sums, some in a named struct and some in a tuple, while
/// the statement already carries them flat in `Claims::use_sums`. Making that
/// return one shape is what a composer could then walk instead of naming each
/// gadget, and `docs/statement.md` is the list of edges such a walk would have
/// to bind.
pub trait Gadget {
    /// the integers this gadget is proved over
    type Witness;
    /// everything about a run that is not the witness
    type Cfg;
    /// what a negative control corrupts, or `()` where the gadget has none
    type Tamper;
    /// what its own proof came to
    type Outcome;

    /// Prove this gadget alone: honestly with `bad` absent, as a control with it
    /// present.
    fn run(w: &Self::Witness, cfg: &Self::Cfg, bad: Option<Self::Tamper>) -> Self::Outcome;

    /// The proof verified, the claimed sums cancelled, every binding held, and no
    /// lookup fell off its table.
    fn accepts(o: &Self::Outcome) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gadgets that prove alone are the ones that say so, and the census is
    /// this list: a gadget dropped from it, or one that stops owing a witness, a
    /// configuration, a control and an outcome, does not compile.
    #[test]
    fn every_gadget_that_proves_alone_owes_the_same_four_things() {
        fn owed<G: Gadget>() {}
        owed::<argmax::Argmax>();
        owed::<attention::Attention>();
        owed::<dequant::Dequant>();
        owed::<embed::Embed>();
        owed::<freivalds::Freivalds>();
        owed::<requant::Requant>();
        owed::<residual_stream::ResidualStream>();
    }
}
