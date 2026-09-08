//! Issue #18: the shape root, derived once and then looked up.
//!
//! `zk-prover expect-root --shape-only` builds and commits the whole
//! preprocessed shape tree to read one root off it, and a grid of segments
//! derives roots that are identical between them. `--shape-cache DIR` keys the
//! root on what the shape tree is a function of -- the statement's own
//! parameters and the per-layer calibration -- and serves it from there.
//!
//! This pins the two halves the deliverable asks for on a synthetic segment:
//! the cache returns the root that was **committed** (not a root of its own),
//! and one parameter moved is a different key and a fresh derivation. It also
//! pins the control: `--shape-cache-verify` commits the tree anyway and
//! asserts the fresh root against the stored one, so a run that exits cleanly
//! under it is a run where the key determined the tree.
//!
//! These run on the default (SIMD) backend, like the digest test beside them.

use std::process::Command;

const SYN: &[&str] = &[
    "expect-root", "--shape-only", "--round", "away", "--variants", "intsplit6",
    "--table-bits", "16", "--s", "32", "--hd", "32",
];

/// The sound synthetic shape `prover/README.md` records a digest for, minus its
/// context, which the cases below move.
const SHAPE: &[&str] = &[
    "--head-groups", "2,1", "--nout", "64", "--residual-stream-a", "32", "--kmax", "16",
    "--synth-residual-stream", "--diagonal-block",
];

fn run(dir: &str, ctx: &str, extra: &[&str]) -> serde_json::Value {
    let mut args: Vec<String> = SYN.iter().map(|s| s.to_string()).collect();
    args.push("--ctx".into());
    args.push(ctx.into());
    args.extend(SHAPE.iter().map(|s| s.to_string()));
    args.push("--shape-cache".into());
    args.push(dir.into());
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(&args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    assert!(
        out.status.success(),
        "zk-prover {} failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    let doc: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    doc["result"].clone()
}

/// A directory of this test's own, so a second run of the suite starts empty.
fn cache_dir() -> String {
    let d = std::env::temp_dir().join(format!("zk-shape-root-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("cache directory");
    d.display().to_string()
}

#[test]
fn a_derived_root_is_served_back_and_a_moved_parameter_misses() {
    let dir = cache_dir();

    // first derivation: the tree is built and committed, and the root is stored
    let first = run(&dir, "128", &[]);
    assert_eq!(first["cached"], serde_json::json!(false),
               "the first derivation cannot be a hit");
    let root = first["pre_root"].as_str().expect("a shape root").to_string();
    assert!(!root.is_empty());
    let key = first["shape"]["key"].as_str().expect("a shape key").to_string();
    assert_eq!(key.len(), 64, "the key is a SHA-256, as hex");
    // what the key was taken over is published beside it
    assert!(first["shape"]["calibration"].is_array(),
            "the calibration the root is a function of is published");

    // second: the same statement and the same calibration, so the same key --
    // and the root that comes back is the one that was committed above
    let second = run(&dir, "128", &[]);
    assert_eq!(second["cached"], serde_json::json!(true), "the second run is a hit");
    assert_eq!(second["pre_root"].as_str(), Some(root.as_str()),
               "the cache served a root that is not the one the tree committed");
    assert_eq!(second["shape"]["key"].as_str(), Some(key.as_str()));
    // and it did no work: nothing was generated and nothing was committed
    assert_eq!(second["trace_secs"], serde_json::json!(0.0));
    assert_eq!(second["commit_shape_secs"], serde_json::json!(0.0));
    // the segment is the same one either way
    assert_eq!(second["columns"], first["columns"]);
    assert_eq!(second["cells"], first["cells"]);

    // the control: derive anyway and compare. A clean exit is the pass -- the
    // fresh root is asserted against the stored one inside the prover.
    let checked = run(&dir, "128", &["--shape-cache-verify"]);
    assert_eq!(checked["cached"], serde_json::json!(false),
               "the control derives rather than taking the stored root");
    assert_eq!(checked["pre_root"].as_str(), Some(root.as_str()));

    // one parameter moved -- the context -- is a different shape, so a
    // different key, a miss, and a root of its own
    let moved = run(&dir, "256", &[]);
    assert_eq!(moved["cached"], serde_json::json!(false),
               "a segment at another context must not be served the first one's root");
    assert_ne!(moved["shape"]["key"].as_str(), Some(key.as_str()),
               "the context moved and the shape key did not");
    assert_ne!(moved["pre_root"].as_str(), Some(root.as_str()),
               "two contexts cannot have one shape tree");

    // and it is now cached in its own right
    let moved_again = run(&dir, "256", &[]);
    assert_eq!(moved_again["cached"], serde_json::json!(true));
    assert_eq!(moved_again["pre_root"], moved["pre_root"]);

    let _ = std::fs::remove_dir_all(&dir);
}
