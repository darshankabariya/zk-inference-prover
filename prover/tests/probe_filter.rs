//! Issue #151: `--probe` is read before a control's segment is built.
//!
//! Six of the suite's controls bend the witness rather than the prover, so each
//! needs a segment of its own: the requantize seams' output side and their scale
//! binding, the `o` seam, the key tile and prefix ones, and the model swap.
//! Those segments used to be built before the filter was read, and on a dump a
//! segment is another read of the whole witness -- a run on the 8B chunk that
//! asked for one control took 3,144 s for a segment whose honest proof takes
//! 137 s, almost all of it reloads for controls that never ran.
//!
//! The suite counts the segments it builds and prints the count on its last line
//! of stderr. This pins the two ends of that: a run whose one control needs no
//! segment builds none, and a run whose one control needs one builds exactly
//! one. It also pins what must not change with the filter -- the record carries
//! the same controls, in the same order, whichever one was asked for.
//!
//! These run on the SIMD build alone, for the reason `attnout_jo_control.rs`
//! gives: a CUDA build has no generation kernel for the shapes the suite
//! restates.

use std::process::Command;

/// The `sound` shape of `prover/README.md`: the smallest one that states the
/// model swap, which is the control this test asks for a segment.
const SOUND: &[&str] = &[
    "compose-probes", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
    "--head-groups", "2,1", "--nout", "64", "--residual-stream-a", "32", "--kmax", "16",
    "--synth-residual-stream", "--diagonal-block",
];

/// A control that holds no segment of its own: it bends one accumulator of a
/// clone the suite already has.
const NO_SEGMENT: &str = "corrupted accumulator";

/// Issue #17's control, which rebuilds the segment around another layer's
/// weights -- one segment, and on this shape it is statable.
const ONE_SEGMENT: &str = "model tree: this segment under another layer's weights";

/// One run of the suite, asked for one control: its stderr and its record.
fn suite(control: &str) -> (String, Vec<serde_json::Value>) {
    let mut args: Vec<String> = SOUND.iter().map(|s| s.to_string()).collect();
    args.push("--probe".into());
    args.push(control.into());
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(&args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "zk-prover {} failed:\n{err}", args.join(" "));
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    let probes = v["result"]["probes"].as_array().expect("result.probes").clone();
    (err, probes)
}

/// The suite's own count of the segments it built, off its last line.
fn segments(err: &str) -> usize {
    err.lines()
        .rev()
        .find_map(|l| {
            l.strip_prefix("compose probes: ")
                .and_then(|r| r.strip_suffix(" segment rebuilds"))
        })
        .expect("the suite says how many segments it built")
        .parse()
        .expect("a count")
}

fn named<'a>(ps: &'a [serde_json::Value], control: &str) -> &'a serde_json::Value {
    ps.iter()
        .find(|p| p["probe"].as_str().is_some_and(|n| n.contains(control)))
        .unwrap_or_else(|| panic!("the suite has no record for {control:?}"))
}

fn names(ps: &[serde_json::Value]) -> Vec<String> {
    ps.iter().map(|p| p["probe"].as_str().expect("probe").to_string()).collect()
}

/// A control the filter left out builds no segment, and a control it selected
/// builds the one it needs -- and the record says the same thing either way.
#[test]
#[cfg_attr(feature = "cuda", ignore = "the control suite runs on the SIMD build: a CUDA build \
                                       has no generation kernel for the shapes the suite \
                                       restates")]
fn a_filtered_out_control_builds_no_segment() {
    let (err, ps) = suite(NO_SEGMENT);
    assert_eq!(segments(&err), 0,
               "a run whose one control needs no segment of its own built {} anyway",
               segments(&err));
    // the control that was asked for still ran, and is a rejection
    assert_eq!(named(&ps, NO_SEGMENT)["accepted"], serde_json::Value::Bool(false),
               "the selected control produced no rejection");
    // and the ones that were not are in the record, by name, saying why they
    // did not run -- which is what a filtered run's record said before
    assert_eq!(named(&ps, ONE_SEGMENT)["skipped"],
               serde_json::Value::String("not selected by --probe".into()),
               "a filtered-out control left the record");
    assert!(named(&ps, ONE_SEGMENT)["accepted"].is_null(),
            "a control that did not run carries a verdict");

    let (err2, ps2) = suite(ONE_SEGMENT);
    assert_eq!(segments(&err2), 1,
               "the model swap needs one segment of its own and the run built {}",
               segments(&err2));
    assert_eq!(named(&ps2, ONE_SEGMENT)["accepted"], serde_json::Value::Bool(false),
               "a segment proved under another layer's weights was accepted");
    // the filter decides which controls run, never which are recorded
    assert_eq!(names(&ps), names(&ps2),
               "the record's controls depend on which one the run asked for");
}
