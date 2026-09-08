//! The negative controls over the model tree, at the two ends of the scale.
//!
//! The first bends one cell of a norm scale's preprocessed weight block and
//! rebuilds the segment around it.  The second (issue #17) swaps a whole
//! layer's worth: the `gate`, `up` and `down` weights, their weight scales and
//! the second norm's own weight, all from another layer's draw, with the
//! segment's activations rebuilt around them.  Either way every constraint of
//! the AIR component still holds and only a committed root moves; the weights
//! are the model's, so the root that moves is the model tree's, and each
//! control is a designed rejection through `model_root_ok` and through nothing
//! else.
//!
//! It is pinned here because the control was silently emptied once already --
//! when the preprocessed trace was split in two, the probe harness kept pinning
//! the shape root alone and the bent weight was accepted.  A run that ends in a
//! panic is not a pass either, so the test demands a clean exit and reads the
//! probe's own record.
//!
//! These run on the default (SIMD) backend, like the digest test beside them.

use std::process::Command;

const SYN: &[&str] = &[
    "compose-probes", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
];
const RESIDUAL_STREAM: &[&str] =
    &["--nout", "64", "--residual-stream-a", "32", "--kmax", "16", "--synth-residual-stream"];

/// The probe's name carries its norm scale's tag, so the suite is asked for it
/// by the part of the name that names the control.
const CONTROL: &str = "one norm-weight cell";

/// Issue #17: the whole-layer swap.  One cell is the smallest a model-tree
/// control can be; this is the largest -- the half of the model tree that hangs
/// off the segment's second norm, taken from another layer's draw with the
/// whole layer rebuilt around it.
const SWAP: &str = "model tree: this segment under another layer's weights";

const GKR5: &[&str] =
    &["--gkr-key-tiles", "--gkr-dequant", "--gkr-residual-stream", "--gkr-freivalds", "--gkr-attnout"];

/// The three sound shapes `prover/README.md` records digests for; the control
/// is the same probe at the same position on each.
const SHAPES: &[(&str, &[&str])] = &[
    ("sound", &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block"]),
    ("sound_gkrfive", &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", "GKR5"]),
    ("sound_aligned",
     &["--head-groups", "2,2", "--nin", "128", "--nout", "128", "--residual-stream-a", "32",
       "--kmax", "16", "--synth-residual-stream", "--diagonal-block"]),
];

fn args_of(flags: &[&str], controls: &str) -> Vec<String> {
    let mut a: Vec<String> = SYN.iter().map(|s| s.to_string()).collect();
    for f in flags {
        match *f {
            "RESIDUAL_STREAM" => a.extend(RESIDUAL_STREAM.iter().map(|s| s.to_string())),
            "GKR5" => a.extend(GKR5.iter().map(|s| s.to_string())),
            _ => a.push(f.to_string()),
        }
    }
    // `--probe` takes a comma-separated list, so both controls run in one
    // process: two provers of the aligned shape at once do not fit the memory
    // this suite is given, and one run is what the two of them need anyway.
    a.push("--probe".into());
    a.push(controls.into());
    a
}

/// One control, in one shape's run: it must be refused, the model root must be
/// the root that moved, the shape root must not have moved, and the rejection
/// must be by `model_root_ok` and nothing else.
fn refused_through_the_model_root(name: &str, probes: &serde_json::Value, control: &str) {
    let probes = probes.as_array().expect("result.probes");
    let p = probes
        .iter()
        .find(|p| p["probe"].as_str().is_some_and(|n| n.contains(control)))
        .unwrap_or_else(|| panic!("{name}: the suite ran no control named {control:?}"));
    assert!(p["skipped"].is_null(),
            "{name}: {control} was skipped: {}", p["skipped"]);
    assert_eq!(p["accepted"], serde_json::Value::Bool(false),
               "{name}: {control} was accepted");
    // the weights are model columns: the model tree's root moves and the shape
    // tree's does not
    assert_eq!(p["model_root_differs_from_honest"], serde_json::Value::Bool(true),
               "{name}: {control} did not move the model root, so the control aims at nothing");
    assert_eq!(p["root_differs_from_honest"], serde_json::Value::Bool(false),
               "{name}: {control} moved the shape root too, so the control no longer \
                isolates the model tree");
    // and by exactly one verdict field, which is what makes it a control over
    // the model tree rather than over anything else
    let flipped: Vec<&str> = p["falsified"].as_array().expect("falsified")
        .iter().map(|s| s.as_str().unwrap()).collect();
    assert_eq!(flipped, ["model_root_ok"],
               "{name}: {control} must reject by `model_root_ok` alone");
    assert_eq!(p["expect_reject_by"], serde_json::Value::String("the pinned model root".into()),
               "{name}: the record must name the root it rejects through");
}

/// The suite, on one shape, asked for both model-tree controls at once.
fn suite(name: &str, flags: &[&str]) -> serde_json::Value {
    let args = args_of(flags, &format!("{CONTROL},{SWAP}"));
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(&args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    // the suite asserts on its own controls, so a panic here is a control being
    // accepted -- a crash is never a pass
    assert!(
        out.status.success(),
        "{name}: zk-prover {} failed ({}):\n{}",
        args.join(" "),
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    v["result"]["probes"].clone()
}

/// Both model-tree controls, at the two ends of the scale: one norm-weight
/// cell bent, and a whole-layer swap -- the `gate`, `up` and `down` weights,
/// their weight scales and the second norm's own weight all from another
/// layer's draw, with the segment's activations rebuilt around them.  Each is
/// an honest segment under a model tree that is not the pinned one, and the
/// pinned model root is what refuses both.
///
/// One test and one process per shape: two provers of the aligned shape at once
/// do not fit the memory this suite is given.
#[test]
fn a_bent_and_a_swapped_model_tree_are_refused_through_the_model_root() {
    for (name, flags) in SHAPES {
        let probes = suite(name, flags);
        refused_through_the_model_root(name, &probes, CONTROL);
        refused_through_the_model_root(name, &probes, SWAP);
    }
}
