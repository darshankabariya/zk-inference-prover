//! The attention output's two `jo` controls, and the suite that runs them.
//!
//! `jo` is the per-token output exponent and the prover's own choice inside the
//! statement's window; the committed `joff = jo_max - jo` is pinned across a
//! token's rows, and one control moves a single row's `jo` against that pin
//! while the other moves a whole token's and is caught -- when the chunk has an
//! `o` seam -- by the binding on the attention output's requantized integers.
//! Both moves have to stay inside the exponent window or the `EPOW` lookup
//! catches them first, and a shape whose windows admit one `jo` and no other
//! has no move to make: two of the three sound synthetic shapes are that shape,
//! and the control there used to assert and take the whole suite with it.
//!
//! So this pins three things: that the suite runs every control it can state and
//! reaches its own JSON on all three shapes, that a control the shape cannot
//! state is skipped by name with its reason in the record rather than crashing,
//! and (issue #24) that the `o` seam's control reaches the record on every shape
//! either way -- it used to reach it as neither, saying so on stderr alone.
//! A crash is never a pass.
//!
//! These run on the SIMD build alone. A binary built with `--features cuda`
//! takes the device path, where `devfrac` demands a generated kernel for the
//! lowering it is about to prove; the suite restates the claim for every
//! control it runs, so the fingerprints it asks for are not the ones `genemit`
//! emitted for the pinned shapes, and the run dies in "no generation kernel"
//! with no verdict at all -- on a host whose kernel store reproduces every
//! pinned digest beside it (issue #104). So both tests are ignored under the
//! feature, by name and with their reason in the run's own output, rather than
//! only saying here which backend they want. The pinned digests beside them
//! are the test that runs on both.

use std::process::Command;

const SYN: &[&str] = &[
    "compose-probes", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
];
const RESIDUAL_STREAM: &[&str] =
    &["--nout", "64", "--residual-stream-a", "32", "--kmax", "16", "--synth-residual-stream"];
const GKR5: &[&str] =
    &["--gkr-key-tiles", "--gkr-dequant", "--gkr-residual-stream", "--gkr-freivalds", "--gkr-attnout"];

const JO_ROW: &str = "attnout: one row's jo moved by one";
const JO_TOK: &str = "attnout: a whole token's jo moved by one";
/// Issue #24: the `o` seam's control, by the name the suite gives it.  Three
/// heads of 32 over 64-wide matmuls cannot state the seam and the first two
/// shapes are that shape; four heads over 128-wide ones can.
const O_SEAM: &str = "o seam: one activation the attention output did not produce";

/// The three sound shapes `prover/README.md` records digests for, and whether
/// each states the two `jo` controls. The first two draw every `jv` of the
/// exponent window and one `jo` for every token, so the window pins `jo` to
/// that one value and neither control has a move; the third's `jo` spans two
/// values and its `jv` leaves room, so both run.
const SHAPES: &[(&str, &[&str], bool)] = &[
    ("sound", &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block"], false),
    ("sound_gkrfive",
     &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", "GKR5"], false),
    ("sound_aligned",
     &["--head-groups", "2,2", "--nin", "128", "--nout", "128", "--residual-stream-a", "32",
       "--kmax", "16", "--synth-residual-stream", "--diagonal-block"], true),
];

fn args_of(flags: &[&str]) -> Vec<String> {
    let mut a: Vec<String> = SYN.iter().map(|s| s.to_string()).collect();
    for f in flags {
        match *f {
            "RESIDUAL_STREAM" => a.extend(RESIDUAL_STREAM.iter().map(|s| s.to_string())),
            "GKR5" => a.extend(GKR5.iter().map(|s| s.to_string())),
            _ => a.push(f.to_string()),
        }
    }
    a
}

/// Run the suite and return its probe records. A panic is a control the suite
/// asserted on or a shape it could not run: either way not a verdict, and the
/// record the run writes then says it was cut short rather than that it passed.
fn probes(name: &str, args: &[String]) -> Vec<serde_json::Value> {
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{name}: zk-prover {} failed:\n{err}{}",
        args.join(" "),
        if err.contains("no generation kernel") {
            "\nthat is a CUDA build with no kernel for this lowering, not a suite that ran \
             and disagreed: this test wants the SIMD build"
        } else {
            ""
        }
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("one JSON document on stdout");
    v["result"]["probes"].as_array().expect("result.probes").clone()
}

fn probe_named<'a>(ps: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
    ps.iter()
        .find(|p| p["probe"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("the suite has no record for {name:?}"))
}

#[test]
#[cfg_attr(feature = "cuda", ignore = "the jo controls run on the SIMD build: a CUDA build \
                                       has no generation kernel for the shapes the suite \
                                       restates")]
fn the_control_suite_reaches_its_own_json_on_every_sound_shape() {
    for (name, flags, jo_statable) in SHAPES {
        let ps = probes(name, &args_of(flags));
        // the suite's last record, so nothing stopped it on the way
        assert_eq!(ps.last().expect("some probes")["probe"],
                   serde_json::Value::String("tampered claim in the statement".into()),
                   "{name}: the suite did not reach its last control");
        // and every control it ran is a rejection or a designed acceptance, not
        // an accepted forgery
        for p in &ps {
            if p["accepted"] == serde_json::Value::Bool(true) {
                assert_eq!(p["expect_reject_by"],
                           serde_json::Value::String(
                               "nothing -- this chunk has no consumer for it".into()),
                           "{name}: {} was accepted", p["probe"]);
            }
        }
        for control in [JO_ROW, JO_TOK] {
            let p = probe_named(&ps, control);
            if *jo_statable {
                assert!(p["skipped"].is_null(),
                        "{name}: {control} was skipped on a shape whose windows leave a step");
                assert!(p["accepted"].is_boolean(),
                        "{name}: {control} produced no verdict");
            } else {
                // skipped by name, and the record says why -- never silently,
                // and never as an assert that ends the run
                let why = p["skipped"].as_str()
                    .unwrap_or_else(|| panic!("{name}: {control} ran on a shape whose \
                                               windows admit one jo and no other"));
                assert!(why.contains("admit one jo"),
                        "{name}: {control} was skipped without saying why: {why:?}");
                assert!(p["accepted"].is_null(),
                        "{name}: a skipped control must carry no verdict");
            }
        }
        // Issue #24: the seam's control is in the record on every shape -- a
        // rejection where the shape can state it, and a skip that says why where
        // it cannot.  On a dump it is the second, because the rebuild draws the
        // residual stream a second time and no window of the honest statement
        // admits that draw; here it is the shape's widths that decide.
        let seam = probe_named(&ps, O_SEAM);
        match seam["skipped"].as_str() {
            Some(why) => {
                assert!(why.contains("ao_q_ok is not statable here"),
                        "{name}: the seam control was skipped without saying why: {why:?}");
                assert!(seam["accepted"].is_null(),
                        "{name}: a skipped control must carry no verdict");
            }
            None => assert_eq!(seam["accepted"], serde_json::Value::Bool(false),
                               "{name}: the seam control produced no rejection"),
        }
        // the per-row control is the one the `joff` pin exists for: where it can
        // be stated it is a rejection and nothing else
        if *jo_statable {
            assert_eq!(probe_named(&ps, JO_ROW)["accepted"], serde_json::Value::Bool(false),
                       "{name}: a per-row output exponent was accepted");
        }
    }
}

/// And the window is what decides it, not the control: the shape the issue was
/// reported on states both moves as soon as the statement admits a `jo` above
/// the witness's own, which is what the skip's own record says to do. The `o`
/// seam is not live on this shape, so a whole token's `jo` is the one control
/// whose pass is an acceptance.
#[test]
#[cfg_attr(feature = "cuda", ignore = "the jo controls run on the SIMD build: a CUDA build \
                                       has no generation kernel for the shapes the suite \
                                       restates")]
fn a_jo_window_with_room_states_both_controls() {
    let mut args = args_of(SHAPES[0].1);
    args.extend(["--ao-jo-max", "-9", "--probe", "jo moved by one"].iter().map(|s| s.to_string()));
    let ps = probes("sound with room above jo", &args);
    let row = probe_named(&ps, JO_ROW);
    assert!(row["skipped"].is_null(), "the per-row control was still skipped");
    assert_eq!(row["accepted"], serde_json::Value::Bool(false),
               "a per-row output exponent was accepted");
    let tok = probe_named(&ps, JO_TOK);
    assert!(tok["skipped"].is_null(), "the whole-token control was still skipped");
    assert_eq!(tok["expect_reject_by"],
               serde_json::Value::String("nothing -- this chunk has no consumer for it".into()),
               "this shape has no o seam, so the whole-token move has no consumer to catch it");
}
