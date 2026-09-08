//! Issue #112: the shape root derived from the statement and the calibration,
//! with no dump on disk.
//!
//! `zk-prover expect-root --shape-only` opens no model file, but it opens the
//! dump's manifest and builds the exponent table and every stand-in array
//! before the derivation starts. `--calibration` replaces that load with
//! `Layer::from_params`: the statement gives every layout, every selector's
//! height and every table's width, the calibration gives the two shape columns
//! the statement does not, and nothing else is read.
//!
//! The claim is that the two derivations are the same derivation, so this pins
//! it on a segment whose whole shape tree is on the desk: the synthetic segment
//! `prover/README.md` records a digest for is composed once, which writes its
//! statement and publishes the calibration its key tiles were built at, and the
//! root that segment **committed** is then re-derived from those two alone.
//! The calibration is passed in each of the two forms the loader takes: the
//! per-key-tile block a run publishes, which is the exact one and the only one
//! a segment whose key tiles do not share one exponent per head has, and the
//! per-head list a model manifest carries, which `Layer::calibration_of` cycles
//! into the same shape.
//!
//! These run on the default (SIMD) backend, like the digest test beside them.

use std::process::Command;

/// The sound synthetic shape `prover/README.md` records a digest for -- the
/// same flags `synthetic_digest`'s first case composes.
const SHAPE: &[&str] = &[
    "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
    "--head-groups", "2,1", "--nout", "64", "--residual-stream-a", "32", "--kmax", "16",
    "--synth-residual-stream", "--diagonal-block",
];

fn run(cmd: &str, extra: &[String]) -> serde_json::Value {
    let mut args: Vec<String> = vec![cmd.to_string()];
    args.extend(SHAPE.iter().map(|s| s.to_string()));
    args.extend(extra.iter().cloned());
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

/// The same, for a run that has to be refused. Returns what it said.
fn refusal(cmd: &str, extra: &[String]) -> String {
    let mut args: Vec<String> = vec![cmd.to_string()];
    args.extend(SHAPE.iter().map(|s| s.to_string()));
    args.extend(extra.iter().cloned());
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(&args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    assert!(!out.status.success(), "zk-prover {} was not refused", args.join(" "));
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A directory of this test's own, so a second run of the suite starts empty.
fn work_dir() -> String {
    let d = std::env::temp_dir().join(format!("zk-shape-from-params-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("work directory");
    d.display().to_string()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn the_root_a_segment_commits_is_derived_from_its_parameters_alone() {
    let dir = work_dir();

    // the segment, composed: it commits a shape tree, writes its statement and
    // publishes the calibration its key tiles were built at. A `compose` run
    // reports its proof under `result.point`; `expect-root` stops before there
    // is one and reports the shape root itself.
    let composed = run("compose", &s(&["--ser-dir", &dir, "--ser-tag", "syn"]))["point"].clone();
    let root = composed["pre_root"].as_str().expect("a shape root").to_string();
    let key = composed["shape"]["key"].as_str().expect("a shape key").to_string();
    let calib = composed["shape"]["calibration"].clone();
    assert!(calib.is_array(), "the calibration the root is a function of is published");
    let statement = format!("{dir}/syn.fixed.json");
    assert!(std::path::Path::new(&statement).exists(), "the statement was written");

    // the calibration, as a document of its own -- what a verifier would hold
    // beside the statement
    let calib_file = format!("{dir}/calibration.json");
    std::fs::write(&calib_file, serde_json::to_vec_pretty(&calib).unwrap())
        .expect("write the calibration");

    // and the derivation from those two alone: no witness, no dump, no manifest
    let derived = run("expect-root",
                      &s(&["--shape-only", "--params", &statement,
                           "--calibration", &calib_file]));
    assert_eq!(derived["pre_root"].as_str(), Some(root.as_str()),
               "the root derived from the parameters is not the one the segment committed");
    assert_eq!(derived["shape"]["key"].as_str(), Some(key.as_str()),
               "the statement and the calibration are the same two, so the key is the same");
    assert_eq!(derived["shape"]["calibration"], calib,
               "the parameter-only load did not reproduce the calibration it was given");
    // and it is the same segment: the shape it reports is the statement's own
    assert_eq!(derived["c"], serde_json::json!(128));
    assert_eq!(derived["s"], serde_json::json!(32));

    // the same, from the run's whole `shape` block rather than the bare list:
    // it is what a run publishes, so it is what a reader has in hand
    let block = format!("{dir}/shape.json");
    std::fs::write(&block, serde_json::to_vec_pretty(&composed["shape"]).unwrap())
        .expect("write the shape block");
    let again = run("expect-root",
                    &s(&["--shape-only", "--params", &statement, "--calibration", &block]));
    assert_eq!(again["pre_root"].as_str(), Some(root.as_str()));

    // and from the per-head form, which is what a model manifest publishes.
    // This segment's heads happen to share one exponent -- asserted here rather
    // than assumed -- so the two forms name the same calibration and have to
    // give the same root.
    let one = calib.as_array().unwrap()[0]["frac_per_inst"][0].as_f64()
        .expect("a per-head exponent");
    assert!(calib.as_array().unwrap().iter().all(|k| {
        k["frac_per_inst"].as_array().unwrap().iter().all(|f| f.as_f64() == Some(one))
    }), "this segment's heads do not share one exponent, so the per-head form is not it");
    let heads = vec![one.to_string(); 3];                        // nh = 3
    let per_head = run("expect-root",
                       &s(&["--shape-only", "--params", &statement,
                            "--frac-per-head", &heads.join(",")]));
    assert_eq!(per_head["pre_root"].as_str(), Some(root.as_str()),
               "the per-head calibration is the same calibration");

    // a calibration that does not describe this statement's key tiles is a
    // refusal and not a root: the scale beside the master exponent is not the
    // one that exponent gives
    let mut bent = calib.clone();
    let was = bent[0]["delta"][0].as_i64().expect("a key tile's first scale");
    bent[0]["delta"][0] = serde_json::json!(was + 1);
    let bent_file = format!("{dir}/bent.json");
    std::fs::write(&bent_file, serde_json::to_vec_pretty(&bent).unwrap()).expect("write it");
    let why = refusal("expect-root",
                      &s(&["--shape-only", "--params", &statement, "--calibration", &bent_file]));
    assert!(why.contains("delta"), "the refusal names what disagreed:\n{why}");

    // and a calibration with the wrong number of key tiles likewise
    let mut short = calib.as_array().unwrap().clone();
    short.pop();
    let short_file = format!("{dir}/short.json");
    std::fs::write(&short_file, serde_json::to_vec_pretty(&short).unwrap()).expect("write it");
    let why = refusal("expect-root",
                      &s(&["--shape-only", "--params", &statement,
                           "--calibration", &short_file]));
    assert!(why.contains("key tiles"), "the refusal names what disagreed:\n{why}");

    // the parameter-only load derives the shape root and not the model root, so
    // it is refused without `--shape-only` rather than committing a tree of
    // stand-ins
    let why = refusal("expect-root",
                      &s(&["--params", &statement, "--calibration", &calib_file]));
    assert!(why.contains("--shape-only"), "the refusal says what is missing:\n{why}");

    let _ = std::fs::remove_dir_all(&dir);
}
