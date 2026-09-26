//! The attention-output component: the A.V outputs of every head, requantized.
//!
//! `AttnOutEval` replaces the key tiles' A.V output component *and* rounds the output:
//! one row per (head, token, channel-chunk), the A.V outputs committed here as
//! `acc`, the int8 fold (`q = round(acc * 2^e / Z)`, B6's `int8` variant with the
//! review's two edge fixes) applied slot-wise, and two Horner accumulators that
//! reproduce the claims the key tiles' identities need:
//!
//!   h  = sum_h rv^(hd*h) sum_t s^t sum_j rv^j acc[h,t,j]   vs  sum_b rv^(hd*off_b) av_b
//!   dz = sum_h r^(C*h)   sum_t s^t den[h,t]                vs  sum_b r^(C*off_b)  az_b
//!
//! Per-slot the scale exponent `e = jv[h,j] - jo[t]` splits into `num = 2^max(e,0)`
//! (preprocessed) and a committed `denp = den * 2^max(-e,0)` pinned by a
//! preprocessed `pj` column, so the error identity stays degree 3.  `denp` is
//! range-checked in its own right (`ao_rem_bits`), which is what makes the
//! error identity an *integer* identity -- see the note beside the lookup.
//!
//! It is a gadget directory without a `run`: there is no standalone attnout
//! proof and no control of its own, so it is not on `Gadget`'s census -- the
//! same place `rope` is in.
//!
//! * `witness` -- the integers the fold is proved over
//! * `layout` -- the (head, token, channel-chunk) row map
//! * `air` -- the constraints, and the preprocessed ids they read
//! * `trace` -- the committed columns
//! * `claims` -- what the component publishes, and its LogUp columns
//! * `device` -- the same columns born on the card

pub mod air;
pub mod claims;
pub mod device;
pub mod layout;
pub mod trace;
pub mod witness;
