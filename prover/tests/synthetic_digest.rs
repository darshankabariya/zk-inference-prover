//! The pinned synthetic proofs.  `compose` with no `--witness` proves one small
//! synthetic layer segment; its proof digest is the regression reference for
//! any change that must not alter the proof.  Both digests were recorded on
//! both backends; `prover/README.md` lists them with the rest.
//!
//! These run on the default (SIMD) backend.  With `--features cuda` the binary
//! needs a device, so CI runs this test only without the feature.

use std::process::Command;

const CASES: &[(&str, &[&str])] = &[
    (
        "34e30fa70297e2a9dc4705b54f5c29b76e0d5ee5d90744f1ddb8b21ec2b96b9d",
        &["compose", "--s", "32", "--ctx", "64", "--bands", "2,1", "--corner"],
    ),
    (
        "37421c358a7a0f2152e9ade8f4b8562d3613b04c06078c47f103fe336d09d320",
        &["compose", "--s", "128", "--ctx", "256", "--bands", "2,1", "--corner"],
    ),
];

fn prove(args: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_b65"))
        .args(args)
        .output()
        .expect("run the prover binary");
    assert!(
        out.status.success(),
        "b65 {} failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("one JSON document on stdout")
}

#[test]
fn synthetic_segment_digests_are_pinned() {
    for (digest, args) in CASES {
        let v = prove(args);
        let point = &v["result"]["point"];
        assert_eq!(point["accepts"], serde_json::Value::Bool(true), "{}", args.join(" "));
        assert_eq!(
            point["proof_sha256"].as_str().unwrap_or(""),
            *digest,
            "proof digest moved for `b65 {}`; if the change is meant to alter the proof, \
             record the new digest in prover/README.md and here in the same pull request",
            args.join(" ")
        );
    }
}
