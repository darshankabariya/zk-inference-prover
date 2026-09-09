//! Issue #95: the verifier rebuilds the shape tree's root from the statement's
//! own parameters and the published calibration, instead of reading it off the
//! statement it verifies.
//!
//! Three things are pinned here, all on the synthetic segment
//! `prover/README.md` records a digest for, and all on the verifier's side of
//! the line: nothing below proves anything new and no recorded digest moves.
//!
//! * an honest proof passes, and the verdict says the root the derivation gave
//!   is the root the statement declares;
//! * a statement whose `pre_root` was edited after the fact is refused, and the
//!   refusal names both roots;
//! * the case that edit stands in for, and the one the issue is about: a proof
//!   whose committed shape tree really is a different tree. `--corrupt-exp`
//!   flips one row of the exponent table and rebuilds the witness around it, so
//!   every constraint holds, every lookup lands, and the statement declares the
//!   bent tree's own root -- `compose` accepts it, which is the control run
//!   here, and the derivation is what refuses it.
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

/// The grid's own tag for this segment's cell: a context of 128 over blocks of
/// 32 is block 4, and the synthetic segment is filed at layer 0.
const TAG: &str = "B4L0";

fn args_of(cmd: &str, shaped: bool, extra: &[String]) -> Vec<String> {
    let mut args: Vec<String> = vec![cmd.to_string()];
    if shaped {
        args.extend(SHAPE.iter().map(|s| s.to_string()));
    }
    args.extend(extra.iter().cloned());
    args
}

fn output_of(args: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary")
}

/// A run that has to succeed, and its `result`.
fn run(cmd: &str, shaped: bool, extra: &[String]) -> serde_json::Value {
    let args = args_of(cmd, shaped, extra);
    let out = output_of(&args);
    assert!(out.status.success(), "zk-prover {} failed:\n{}", args.join(" "),
            String::from_utf8_lossy(&out.stderr));
    let doc: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    doc["result"].clone()
}

/// A run that has to be refused. Returns what it said on stderr.
fn refusal(cmd: &str, shaped: bool, extra: &[String]) -> String {
    let args = args_of(cmd, shaped, extra);
    let out = output_of(&args);
    assert!(!out.status.success(), "zk-prover {} was not refused", args.join(" "));
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A grid run, which reports its JSON whether it passes or fails: the verdict
/// per chunk is in the document and the exit status is the answer.
fn grid(extra: &[String]) -> (bool, serde_json::Value) {
    let args = args_of("boundary-grid", false, extra);
    let out = output_of(&args);
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("boundary-grid wrote no JSON ({e}):\n{}",
                                   String::from_utf8_lossy(&out.stderr)));
    (out.status.success(), doc["result"].clone())
}

/// A directory of this test's own, so a second run of the suite starts empty.
fn work_dir(what: &str) -> String {
    let d = std::env::temp_dir()
        .join(format!("zk-verify-shape-root-{}-{what}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("work directory");
    d.display().to_string()
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn the_verifier_derives_the_shape_root_and_refuses_a_statement_that_names_another() {
    let honest_dir = work_dir("honest");
    let bent_dir = work_dir("bent");
    let bench = work_dir("bench");

    // ---- what a verifier holds -------------------------------------------
    //
    // The shape root this shape gives and the calibration it is a function of,
    // taken without proving anything: `expect-root --shape-only` derives tree 0
    // and stops, and publishes the calibration beside it. In a real setting
    // both come off the manifest `zk-prover model-roots` writes; here the
    // segment is synthetic, so its own published block is the calibration.
    let expected = run("expect-root", true, &s(&["--shape-only"]));
    let root = expected["pre_root"].as_str().expect("a shape root").to_string();
    let calib = expected["shape"]["calibration"].clone();
    assert!(calib.is_array(), "the calibration the root is a function of is published");
    let calib_file = format!("{bench}/calibration.json");
    std::fs::write(&calib_file, serde_json::to_vec_pretty(&calib).unwrap())
        .expect("write the calibration");
    // the prompt the grid demands beside the calibration; this segment carries
    // no embedding gadget, so it publishes no ids for the prompt to answer for
    let prompt_file = format!("{bench}/prompt.json");
    std::fs::write(&prompt_file, b"{\"ids\": []}").expect("write the prompt");

    // ---- an honest proof passes, and the verdict says what it derived -----
    let point = run("compose", true,
                    &s(&["--ser-dir", &honest_dir, "--ser-tag", TAG,
                         "--verify-calibration", &calib_file]))["point"].clone();
    assert_eq!(point["accepts"], serde_json::json!(true), "the honest proof was refused");
    let v = &point["verdict"];
    assert_eq!(v["shape_root_ok"], serde_json::json!(true),
               "the derived root is not the one the statement declares: {v}");
    assert_eq!(v["shape_root"].as_str(), Some(root.as_str()),
               "the verifier derived a root that is not this shape's");
    assert_eq!(point["pre_root"].as_str(), Some(root.as_str()),
               "the segment committed a shape tree that is not its parameters'");
    // the derivation is a cost, and the verdict says what it was
    assert!(v["shape_secs"].as_f64().expect("the derivation's seconds") > 0.0,
            "the first proof at a shape pays for the derivation");

    // ---- the grid verifies through the same function ----------------------
    let (ok, g) = grid(&s(&["--dir", &honest_dir, "--tags", TAG, "--verify",
                            "--prompt", &prompt_file, "--verify-calibration", &calib_file]));
    assert!(ok, "the honest chunk did not pass the grid: {g}");
    assert_eq!(g["shape_roots"]["matched"], serde_json::json!(1),
               "the grid did not derive this chunk's shape root: {}", g["shape_roots"]);
    assert_eq!(g["verified"]["accepted"], serde_json::json!(1));

    // ---- a statement whose pre_root was edited ---------------------------
    let edited = work_dir("edited");
    for part in ["claims", "proof"] {
        std::fs::copy(format!("{honest_dir}/{TAG}.{part}.json"),
                      format!("{edited}/{TAG}.{part}.json")).expect("copy the chunk");
    }
    let mut fixed: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{honest_dir}/{TAG}.fixed.json")).expect("the statement"))
        .expect("the statement is JSON");
    // one hex digit of the declared root, moved
    let bent_root = {
        let mut r: Vec<char> = root.chars().collect();
        let last = r.len() - 1;
        r[last] = if r[last] == '0' { '1' } else { '0' };
        r.into_iter().collect::<String>()
    };
    fixed["pre_root"] = serde_json::json!(bent_root);
    std::fs::write(format!("{edited}/{TAG}.fixed.json"),
                   serde_json::to_vec(&fixed).unwrap()).expect("write the statement");
    let (ok, g) = grid(&s(&["--dir", &edited, "--tags", TAG, "--verify",
                            "--prompt", &prompt_file, "--verify-calibration", &calib_file]));
    assert!(!ok, "a statement whose pre_root was edited was accepted: {g}");
    let why = g["verified"]["chunks"][0]["why"].as_str().expect("a reason").to_string();
    assert!(why.contains(&root) && why.contains(&bent_root),
            "the refusal names both roots:\n{why}");
    assert!(why.contains("\"shape_root\":false"), "the derived root is what refused it:\n{why}");

    // ---- and the shape tree that really is another tree -------------------
    //
    // The control first: one row of the exponent table flipped with the witness
    // rebuilt around it is internally consistent everywhere, so the run's own
    // verifier -- given no calibration -- accepts it and writes its statement.
    let bent = run("compose", true,
                   &s(&["--corrupt-exp", "4321", "--ser-dir", &bent_dir, "--ser-tag", TAG]))
        ["point"].clone();
    assert_eq!(bent["accepts"], serde_json::json!(true),
               "the bent shape tree was refused by something other than the derivation, so \
                this control no longer says what it is here to say");
    // and that run was given no calibration, so it derived nothing and says so
    // rather than reporting a check it did not make
    assert_eq!(bent["verdict"]["shape_root_ok"], serde_json::Value::Null,
               "a run given no calibration must not report a derived root");
    let declared = bent["pre_root"].as_str().expect("the bent root").to_string();
    assert_ne!(declared, root, "the bent exponent table did not move the shape root");

    // and the derivation refuses it: the statement's parameters and the
    // published calibration give the honest tree, and this proof committed
    // another one under a statement that declares its own
    let (ok, g) = grid(&s(&["--dir", &bent_dir, "--tags", TAG, "--verify",
                            "--prompt", &prompt_file, "--verify-calibration", &calib_file]));
    assert!(!ok, "a proof over a bent shape tree was accepted: {g}");
    let why = g["verified"]["chunks"][0]["why"].as_str().expect("a reason").to_string();
    assert!(why.contains(&root) && why.contains(&declared),
            "the refusal names the derived root and the declared one:\n{why}");
    // and it is the derivation and nothing else that refused it: the proof
    // committed the root its own statement declares, which is what `root_ok`
    // asks and all a verifier asked before this
    assert!(why.contains("\"root\":true"), "root_ok held on the bent proof:\n{why}");
    assert!(why.contains("\"shape_root\":false"), "the derived root is what refused it:\n{why}");

    // the same through the run's own verifier, which is the single-proof path
    // the command line exposes without a grid
    let why = refusal("compose", true,
                      &s(&["--corrupt-exp", "4321", "--verify-calibration", &calib_file]));
    assert!(why.contains(&root), "the refusal names the root the parameters give:\n{why}");
    assert!(why.contains("some other shape"), "the refusal says what it means:\n{why}");

    for d in [honest_dir, bent_dir, bench, edited] {
        let _ = std::fs::remove_dir_all(&d);
    }
}
