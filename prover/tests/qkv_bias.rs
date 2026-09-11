//! Issue #172: a layer whose q, k and v carry a projection bias cannot be stated.
//!
//! B1 adds the model's projection bias to the dequantized matmul output, before
//! the rotation and before the quantize, so the key tiles hold integers the
//! dequant component -- which states `acc s_x s_w` and nothing else -- cannot
//! derive.  `nn_out` is the one equality that compares the two, and it is the
//! only field that falsifies: every constraint holds, the proof verifies, and
//! the segment is rejected all the same.
//!
//! The shape is `sound_padded`, whose `S x nh` is 160 and therefore a power of
//! two at no block length.  With the switch off it proves and its digest is
//! pinned (`synthetic_digest.rs`), which is the other half of this test: the
//! padded rows are not what rejects a 40-head layer.

use std::process::Command;

const ARGS: &[&str] = &[
    "compose", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
    "--head-groups", "4,1", "--nin", "160", "--nout", "160",
    "--residual-stream-a", "32", "--kmax", "16", "--synth-residual-stream", "--diagonal-block",
];

#[test]
fn a_projection_bias_is_refused_and_named() {
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(ARGS)
        .env("RAYON_NUM_THREADS", "4")
        .env("ZK_SYNTHETIC_QKV_BIAS", "0.25")
        .output()
        .expect("run the prover binary");
    assert!(!out.status.success(),
            "a layer the statement cannot state was proved and accepted");
    let err = String::from_utf8_lossy(&out.stderr);
    // the count, on stderr, before the proof is built
    assert!(err.contains("are not the key tile's own"),
            "the run did not name the disagreement:\n{err}");
    // and the field that falsifies: the input side still binds, so the rejection
    // is the consumer side alone
    assert!(err.contains("\"nn_in_ok\":[true,true,true]"),
            "the producer side should still bind:\n{err}");
    assert!(err.contains("\"nn_out_ok\":[false,false,false]"),
            "nn_out_ok should be the field that falsifies:\n{err}");
    assert!(err.contains("the quantize components derived integers the key tiles do not hold"),
            "the rejection did not carry its diagnosis:\n{err}");
}
