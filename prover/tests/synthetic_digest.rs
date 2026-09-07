//! The pinned synthetic proofs.  `compose` with no `--witness` proves one small
//! synthetic layer segment; its proof digest is the regression reference for
//! any change that must not alter the proof.  The eleven shapes and their
//! digests are the ones `prover/README.md` lists; every one was recorded on the
//! SIMD backend on 2026-09-06.
//!
//! These run on the default (SIMD) backend.  With `--features cuda` the binary
//! needs a device, so CI runs this test only without the feature.

use std::process::Command;

const SYN: &[&str] = &[
    "compose", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--bands", "2,1", "--s", "32", "--ctx", "128", "--hd", "32",
];
const GLUE: &[&str] = &["--nout", "64", "--glue-a", "32", "--kmax", "16", "--synth-glue"];
const GKR5: &[&str] = &["--gkr-bands", "--gkr-dequant", "--gkr-glue", "--gkr-walks", "--gkr-attnout"];

/// (name, first 16 hex of the digest, whether the wiring family is on, the flags)
const CASES: &[(&str, &str, bool, &[&str])] = &[
    ("corner", "164079c6a045df22", false, &["--corner"]),
    ("flagoff", "156285f2f70eaa80", false, &[]),
    ("kvbind", "4d3835c942072f67", false, &["--corner", "--kv-bind"]),
    ("gkrbands", "c44ce8a832a9bec7", false, &["--corner", "--gkr-bands"]),
    ("gkrfive", "5d7aa189c702c2a5", false, &["--corner", "GKR5"]),
    ("kv_gkrbands", "b857c0b7a978642f", false, &["--corner", "--kv-bind", "--gkr-bands"]),
    ("kv_gkrfive", "a379fa16dc2b0e98", false, &["--corner", "--kv-bind", "GKR5"]),
    ("glue_corner", "5a4a05ea6274e7c6", true, &["--corner"]),
    ("glue_kv", "94c32161ecbbef84", true, &["--corner", "--kv-bind"]),
    ("glue_bounds", "775b1972badf31e4", true, &["--corner", "--bounds"]),
    ("glue_gkrfive", "d1aff3e6580569a4", true, &["--corner", "GKR5"]),
];

fn args_of(glue: bool, flags: &[&str]) -> Vec<String> {
    let mut a: Vec<String> = SYN.iter().map(|s| s.to_string()).collect();
    if glue {
        a.extend(GLUE.iter().map(|s| s.to_string()));
    }
    for f in flags {
        if *f == "GKR5" {
            a.extend(GKR5.iter().map(|s| s.to_string()));
        } else {
            a.push(f.to_string());
        }
    }
    a
}

fn prove(args: &[String]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_b65"))
        .args(args)
        .env("RAYON_NUM_THREADS", "4")
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
    for (name, digest, glue, flags) in CASES {
        let args = args_of(*glue, flags);
        let v = prove(&args);
        let point = &v["result"]["point"];
        assert_eq!(point["accepts"], serde_json::Value::Bool(true), "{name}: b65 {}", args.join(" "));
        let got = point["proof_sha256"].as_str().unwrap_or("");
        assert!(
            got.starts_with(digest),
            "{name}: proof digest moved for `b65 {}`: got {got}, pinned {digest}; if the change \
             is meant to alter the proof, record the new digest in prover/README.md and here in \
             the same pull request",
            args.join(" ")
        );
    }
}
