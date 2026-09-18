//! What a gadget's AIR states about itself once (#221).
//!
//! Four of `FrameworkEval`'s methods are the same lines in every gadget of this
//! crate.  `log_size` is the table's height; `max_constraint_log_degree_bound`
//! is that height and one, because every set of constraints here is quadratic
//! in the trace; `cuda_eval_name` is the kernel store's key; and
//! `cuda_eval_blob` is `cudagen::blob_for(self)` in all ten of them.  Written
//! out, they are a dozen lines a component that say nothing about the component.
//!
//! So a gadget states the two facts they follow from -- where its rows are and
//! what its generated kernel is called -- as an [`Air`], and writes
//! [`framework_boilerplate!`] inside its `impl FrameworkEval`, which emits the
//! four.  What is left in the impl is `evaluate`.
//!
//! [`Ids`] is the other thing every gadget spells out: a preprocessed column's
//! id string.  They are all `zk_{family}{tag}_{column}` -- the family the
//! component belongs to, the tag that tells two components of one family apart,
//! and the column's own name -- and they are *pinned*, because they are the
//! keys a verifier resolves a preprocessed column by.  A gadget that builds
//! them through `Ids` says the family once and never formats a `zk_` again.

use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

/// What a component says about itself, so that the framework's own methods do
/// not have to be written out.
///
/// It is deliberately two facts and not four: the degree bound and the blob
/// follow from the rows and from the crate, and a gadget that needs another
/// degree says so with [`Air::DEGREE`] rather than by writing the method.
pub trait Air {
    /// The kernel store's key for this component's generated constraint kernel.
    ///
    /// Never rename one: a component whose name has no kernel installed under
    /// it falls back to the CPU accumulation path, silently and at full cost.
    const KERNEL: &'static str;

    /// How far above the table's height the constraints reach.  Every set of
    /// constraints in this crate is quadratic in the trace, which is one.
    const DEGREE: u32 = 1;

    /// The table's height: `2^log_rows` rows.
    fn log_rows(&self) -> u32;
}

/// The four `FrameworkEval` methods that follow from [`Air`], written inside a
/// gadget's `impl FrameworkEval for ...` block beside its `evaluate`.
///
/// It takes no arguments: everything it needs is on the type's own [`Air`]
/// impl, so there is no call-site token for a gadget to get wrong and no
/// `self` crossing the macro's hygiene.
#[macro_export]
macro_rules! framework_boilerplate {
    () => {
        /// The flat parameter block the generated constraint kernel is emitted
        /// against (`gpu/m2b/cudagen.rs`).
        fn cuda_eval_blob(&self) -> Option<Vec<u8>> {
            $crate::cudagen::blob_for(self)
        }
        /// The kernel store's key, which is [`Air::KERNEL`] and never renamed.
        fn cuda_eval_name(&self) -> &'static str {
            <Self as $crate::air::Air>::KERNEL
        }
        fn log_size(&self) -> u32 {
            $crate::air::Air::log_rows(self)
        }
        fn max_constraint_log_degree_bound(&self) -> u32 {
            $crate::air::Air::log_rows(self) + <Self as $crate::air::Air>::DEGREE
        }
    };
}

/// Where one component's preprocessed columns are named: `zk_{family}{tag}`.
///
/// The strings are pinned -- a verifier resolves a preprocessed column by its
/// id, so renaming one is not a refactor -- and this is what holds them still
/// while the Rust names move.  A gadget builds one per component and asks it
/// for a column; a primitive that owns columns of its own (`fold::Groups`)
/// takes the [`prefix`](Ids::prefix) and names them under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ids(String);

impl Ids {
    /// The namespace a component of `family` tagged `tag` names its columns in.
    /// The tag is the empty string on a family with one component.
    pub fn of(family: &str, tag: &str) -> Self {
        Ids(format!("zk_{family}{tag}"))
    }

    /// The prefix itself, for a primitive that names its own columns under it.
    pub fn prefix(&self) -> &str {
        &self.0
    }

    /// `{prefix}_{column}`, the id of one preprocessed column.
    pub fn at(&self, column: impl std::fmt::Display) -> PreProcessedColumnId {
        crate::relations::id(format!("{}_{column}", self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The id strings are the ones the gadgets wrote by hand, byte for byte.
    #[test]
    fn the_ids_are_the_strings_the_gadgets_spelled_out() {
        let ids = Ids::of("freivalds", "qkv");
        assert_eq!(ids.prefix(), "zk_freivaldsqkv");
        assert_eq!(ids.at("nofirst").id, "zk_freivaldsqkv_nofirst");
        assert_eq!(ids.at(format_args!("w{}_{:04}", 2, 17)).id, "zk_freivaldsqkv_w2_0017");
        // a family with one component carries no tag and no separator for it
        assert_eq!(Ids::of("residual_stream", "").prefix(), "zk_residual_stream");
    }
}
