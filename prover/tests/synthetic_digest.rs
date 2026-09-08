//! The pinned synthetic proofs.  `compose` with no `--witness` proves one small
//! synthetic layer segment; its proof digest is the regression reference for
//! any change that must not alter the proof, and its binding table is the
//! regression reference for what the statement binds.  The shapes and their
//! digests are the ones `prover/README.md` lists.
//!
//! These run on the default (SIMD) backend.  With `--features cuda` the binary
//! needs a device, so CI runs this test only without the feature.

use std::process::Command;

const SYN: &[&str] = &[
    "compose", "--round", "away", "--variants", "intsplit6", "--table-bits", "16",
    "--s", "32", "--ctx", "128", "--hd", "32",
];
const RESIDUAL_STREAM: &[&str] =
    &["--nout", "64", "--residual-stream-a", "32", "--kmax", "16", "--synth-residual-stream"];
const GKR5: &[&str] =
    &["--gkr-key-tiles", "--gkr-dequant", "--gkr-residual-stream", "--gkr-freivalds", "--gkr-attnout"];
const SKIP_RS: &str = "--unsound-skip-residual-stream";
const SKIP_LB: &str = "--unsound-skip-layer-boundary";
const SKIP_KV: &str = "--unsound-skip-kv-boundary";

struct Case {
    name: &'static str,
    /// the first 16 hex of `proof_sha256`
    digest: &'static str,
    /// the flags after `SYN`; `RESIDUAL_STREAM` and `GKR5` expand
    flags: &'static [&'static str],
    /// the edge numbers the binding table must leave unbound, in order
    unbound: &'static [u8],
}

/// The sound defaults first, then the unbound configuration: the same shapes
/// with every gadget the skips name turned off, which is how the digests
/// recorded before the sound defaults reproduce.
const CASES: &[Case] = &[
    // the residual-stream family with both boundaries on; the 64-wide matmuls
    // under three heads of 32 leave the rotation (3) and the o seam (5) open
    Case { name: "sound", digest: "82085896a8408f36",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block"], unbound: &[3, 5] },
    Case { name: "sound_gkrfive", digest: "6bee5972d6f616aa",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", "GKR5"], unbound: &[3, 5] },
    // four heads of 32 over 128-wide matmuls: every layer edge bound
    Case { name: "sound_aligned", digest: "9c3cdebd97a7d57b",
           flags: &["--head-groups", "2,2", "--nin", "128", "--nout", "128", "--residual-stream-a", "32",
                    "--kmax", "16", "--synth-residual-stream", "--diagonal-block"],
           unbound: &[] },
    // the unbound configuration (recorded 2026-09-06 on the SIMD backend; the sound
    // shapes above on 2026-09-07)
    Case { name: "diagonal_block", digest: "164079c6a045df22",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS, SKIP_KV],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14] },
    Case { name: "flagoff", digest: "156285f2f70eaa80",
           flags: &["--head-groups", "2,1", SKIP_RS, SKIP_KV],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14] },
    Case { name: "kvbind", digest: "4d3835c942072f67",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13] },
    Case { name: "gkr_key_tiles", digest: "c44ce8a832a9bec7",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS, SKIP_KV, "--gkr-key-tiles"],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14] },
    Case { name: "gkrfive", digest: "5d7aa189c702c2a5",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS, SKIP_KV, "GKR5"],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14] },
    Case { name: "kv_gkr_key_tiles", digest: "b857c0b7a978642f",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS, "--gkr-key-tiles"],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13] },
    Case { name: "kv_gkrfive", digest: "a379fa16dc2b0e98",
           flags: &["--head-groups", "2,1", "--diagonal-block", SKIP_RS, "GKR5"],
           unbound: &[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13] },
    Case { name: "residual_stream_diagonal_block", digest: "5a4a05ea6274e7c6",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", SKIP_LB, SKIP_KV],
           unbound: &[3, 5, 13, 14] },
    Case { name: "residual_stream_kv", digest: "94c32161ecbbef84",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", SKIP_LB], unbound: &[3, 5, 13] },
    Case { name: "residual_stream_bounds", digest: "775b1972badf31e4",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", SKIP_KV], unbound: &[3, 5, 14] },
    Case { name: "residual_stream_gkrfive", digest: "d1aff3e6580569a4",
           flags: &["--head-groups", "2,1", "RESIDUAL_STREAM", "--diagonal-block", SKIP_LB, SKIP_KV, "GKR5"],
           unbound: &[3, 5, 13, 14] },
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

fn prove(args: &[String]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    assert!(
        out.status.success(),
        "zk-prover {} failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("one JSON document on stdout")
}

#[test]
fn synthetic_segment_digests_and_binding_tables_are_pinned() {
    for c in CASES {
        let args = args_of(c.flags);
        let v = prove(&args);
        let point = &v["result"]["point"];
        assert_eq!(point["accepts"], serde_json::Value::Bool(true), "{}: zk-prover {}", c.name, args.join(" "));
        let got = point["proof_sha256"].as_str().unwrap_or("");
        assert!(
            got.starts_with(c.digest),
            "{}: proof digest moved for `zk-prover {}`: got {got}, pinned {}; if the change is meant \
             to alter the proof, record the new digest in prover/README.md and here in the \
             same pull request",
            c.name, args.join(" "), c.digest
        );
        let unbound: Vec<u8> = point["binding"]["unbound"].as_array().expect("binding.unbound")
            .iter().map(|n| n.as_u64().unwrap() as u8).collect();
        assert_eq!(unbound, c.unbound, "{}: the binding table's unbound edges", c.name);
        // the same table reaches the provenance block, folded over the one proof
        assert_eq!(v["provenance"]["binding"]["unbound"], point["binding"]["unbound"], "{}", c.name);
        assert_eq!(v["provenance"]["proof_sha256"].as_str(), Some(got), "{}", c.name);
        let skips: Vec<&str> = args.iter().filter_map(|a| a.strip_prefix("--unsound-skip-")).collect();
        let prov_skips: Vec<&str> = v["provenance"]["unsound_skips"].as_array().unwrap()
            .iter().map(|s| s.as_str().unwrap()).collect();
        assert_eq!(prov_skips, skips, "{}: provenance.unsound_skips", c.name);
    }
}

/// The head shape, pinned here for the first time (issue #58).
///
/// A head chunk's family is the final norm, which builds no `o` seam -- and the
/// dump a head chunk is loaded from carries its layer's `o` matmul all the same,
/// so both `ao_q` and the `o` Freivalds check's `qX` are published and the
/// verifier used to compare two folds this statement never binds to each other.
/// An honest head chunk was rejected by that comparison and nothing else. Here
/// the seam is *absent*: `ao_q_open` is false beside a vacuous `ao_q_ok`, and the
/// binding table says the same.
///
/// A synthetic head carries no `key_head.u16`, so the argmax has to be skipped to
/// prove it at all and edge 16 is unbound here; on a head dump it is bound and
/// nothing else about the table moves. It takes no `--diagonal-block` either: a
/// head chunk commits no K/V boundary, so its plan holds the prefix key tiles
/// alone and a synthetic head built with the diagonal block would disagree with
/// it. Edge 14 is absent here for the same reason it is on a head dump.
#[test]
fn the_synthetic_head_chunk_proves_with_the_o_seam_absent() {
    let args = args_of(&["--head-groups", "2,1", "--nout", "64", "--residual-stream-a", "32",
                         "--kmax", "16", "--synth-head", "--synth-vocab", "256",
                         "--unsound-skip-argmax"]);
    let v = prove(&args);
    let point = &v["result"]["point"];
    assert_eq!(point["accepts"], serde_json::Value::Bool(true),
               "the honest head chunk must verify: zk-prover {}", args.join(" "));
    let got = point["proof_sha256"].as_str().unwrap_or("");
    assert!(got.starts_with("e3f31c1d4aeae77e"),
            "head: proof digest moved for `zk-prover {}`: got {got}; if the change is meant to \
             alter the proof, record the new digest in prover/README.md and here in the same \
             pull request", args.join(" "));
    // the seam is absent, not open: nothing about this chunk's `o` matmul is
    // compared, and the record says absent rather than a checked pass
    assert_eq!(point["verdict"]["ao_q_ok"], serde_json::Value::Bool(true), "head: ao_q_ok");
    assert_eq!(point["verdict"]["ao_q_open"], serde_json::Value::Bool(false), "head: ao_q_open");
    assert_eq!(point["binding"]["unbound"], serde_json::json!([16]), "head: unbound edges");
    assert_eq!(point["binding"]["absent"], serde_json::json!([3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]),
               "head: absent edges");
    assert_eq!(v["provenance"]["unsound_skips"], serde_json::json!(["argmax"]));
}

/// Issue #30: `compose` and `compose-chunks` resolve a repeated flag alike, so
/// a plan line rerun on its own reproduces the digest the loop gave it.
///
/// The rule is that the last occurrence wins. The shared prefix below names
/// `--table-bits 17` and the plan line names 16, which makes the line the
/// `sound` case above, so both commands reproduce its recorded digest. Were the
/// first occurrence to win, the loop would take 16 (it wrote the line first)
/// and the one-shot 17, which is the disagreement the issue reports.
#[test]
fn a_repeated_flag_resolves_the_same_way_in_compose_and_in_the_loop() {
    const SOUND: &str = "82085896a8408f36";
    let line = "--head-groups 2,1 --nout 64 --residual-stream-a 32 --kmax 16 \
                --synth-residual-stream --diagonal-block --table-bits 16";
    // the shared prefix: `SYN` with the table width the line overrides
    let mut shared: Vec<String> = SYN.iter().map(|s| s.to_string()).collect();
    let i = shared.iter().position(|w| w == "--table-bits").expect("SYN names --table-bits");
    shared[i + 1] = "17".to_string();

    let mut one_shot = shared.clone();
    one_shot.extend(line.split_whitespace().map(str::to_string));
    let got = prove(&one_shot);
    let one = got["result"]["point"]["proof_sha256"].as_str().unwrap_or("").to_string();
    assert!(one.starts_with(SOUND),
            "compose read the shared --table-bits over the line's: `zk-prover {}` gave {one}, \
             the line's own width gives {SOUND}", one_shot.join(" "));

    let plan = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("repeated-flag.plan.txt");
    std::fs::write(&plan, format!("# issue #30\n{line}\n")).expect("write the plan");
    let mut loop_args = vec!["compose-chunks".to_string()];
    loop_args.extend(shared[1..].iter().cloned());
    loop_args.push("--plan".to_string());
    loop_args.push(plan.to_string_lossy().to_string());
    let got = prove(&loop_args);
    let many = got["result"]["summary"]["digests"][0].as_str().unwrap_or("").to_string();
    assert_eq!(one, many,
               "the loop and the one-shot disagree on the same plan line: `zk-prover {}`",
               loop_args.join(" "));
}
