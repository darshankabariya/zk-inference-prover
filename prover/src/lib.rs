#![recursion_limit = "1024"]
//! The prover as a library: the gadgets, the composer that states one segment
//! over them, and the two device paths they are proved on.
//!
//! The binary is `cli`, and it is the only caller here that reads a command
//! line. Everything else is reachable on its own, which is what lets one gadget
//! be built, proved and priced without composing a segment around it -- the
//! `argmax`, `embed`, `requant` and `residual_stream` subcommands each do
//! exactly that.
//!
//! What the proof binds, edge by edge, is `docs/statement.md`; the recorded
//! digests are `prover/README.md`.

pub mod bf16;
pub mod boundary;
pub mod chunks;
pub mod cli;
pub mod common;
pub mod compose;
pub mod cudagen;
pub mod cudakernels;
pub mod dev_attention_freivalds_witness_gen;
pub mod dev_dequant_witness_gen;
pub mod dev_residual_stream_witness_gen;
pub mod devacc;
pub mod devfrac;
pub mod devlogup;
pub mod devpre;
pub mod devtab;
pub mod device;
pub mod exprprobe;
pub mod gadgets;
pub mod gencudakernels;
pub mod gkr_key_tile;
pub mod gkrlink;
pub mod layer;
pub mod logup_run;
pub mod plogup;
pub mod precache;
pub mod scalars;
pub mod shaperoot;
pub mod spans;
pub mod util;
pub mod witread;
