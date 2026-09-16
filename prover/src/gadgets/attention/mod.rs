//! The attention block: three coupled components over one grid, not one.
//!
//! The layouts are `layout`, and they are the reason this gadget is a directory
//! and not a file: Q.K^T reduces over the channel and A.V over the key, so the
//! two inner products want different row axes and only one of them can carry
//! the (token, key) grid. `layout`'s module comment says which wins and what
//! that costs.
//!
//! * `layout` -- the three row axes and the grid they tile
//! * `witness` -- the integers one block of the attention is proved over
//! * `trace` -- the committed columns, per key tile
//! * `air` -- the constraints over them, one evaluator per row axis
//! * `run` -- the gadget on its own, without a segment

pub mod air;
pub mod layout;
pub mod run;
pub mod trace;
pub mod witness;
