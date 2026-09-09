//! Issue #130: a run JSON names the dump its proof was made from.
//!
//! The 8B reference digest moved with no prover commit behind it, and neither
//! the page's row nor the run that produced it could say so: the run recorded
//! the commit, the command and the proof, and about the dump it recorded
//! nothing at all. A dump's `manifest.json` has carried a digest per array for
//! a while; what was missing was one number over them, in the record.
//!
//! Three things are pinned here, all on the reader's side of the line -- no
//! statement, no proof and no recorded digest moves for any of them:
//!
//! * a run that read a dump carries that dump's digest, its directory, and what
//!   the manifest says the dump was made from (the ids and the attention
//!   calibration);
//! * the digest is the roll-up the extractor computes, to the letter: the map
//!   and the number below are `harness/witness/extract.py --selftest`'s own, so
//!   the harness and the prover cannot drift apart without one of the two
//!   tests failing;
//! * a manifest that states a roll-up disagreeing with the arrays it lists is
//!   reported and not believed, and a run that read no dump says so rather than
//!   leaving the field out.
//!
//! The dump is the least any subcommand needs: `argmax --witness` reads the
//! manifest, the emitted tokens and the keys they are the maximum of, and
//! nothing else, so eight tokens over a vocabulary of 512 -- the shape
//! `cell_cost_subcommands` proves beside this -- is a whole dump here.
//!
//! These run on the default (SIMD) backend, like the digest test beside them.

use std::process::Command;

/// The two arrays the dump below holds, as its manifest lists them, and the
/// digest the roll-up gives over that list.
///
/// `harness/witness/extract.py`'s `GOLDEN_FILES` and `GOLDEN_DUMP` are the same
/// map and the same number. The two sha256s are the arrays' own, taken when
/// this test was written; the prover rolls up what a manifest claims and
/// re-reads no array, which is what lets it name a 4.2 GB dump for free.
const FILES: &str = r#"{
  "key_head.u16": {"values": 4096, "dtype": "uint16", "shape": [8, 512],
                   "sha256": "132ce9b6a2a54e71"},
  "tok_head.i32": {"values": 8, "dtype": "int32", "shape": [8],
                   "sha256": "e5e9f2e7cf0c271d"}
 }"#;
const GOLDEN: &str = "badad7e68e76e9c83289dc3f18abcab667fb3a4039677edaf9ff371607a71f26";

/// The ids file and the attention calibration the manifest names, in the shape
/// `extract.py` writes them: the second is the per-host quantization input this
/// issue is about (`jk.i32`, the per-head K scale exponents, is its K half).
const IDS_SHA: &str = "6c1b3d2a5f4e7089c1b3d2a5f4e7089c1b3d2a5f4e7089c1b3d2a5f4e7089c1b";
const CALIB_SHA: &str = "9a8b7c6d5e4f30219a8b7c6d5e4f30219a8b7c6d5e4f30219a8b7c6d5e4f3021";

/// The head dump, written where the test's own temporary directory is.
/// `stated` is the manifest's own `dump_sha256`, which the extractor writes and
/// this test bends in the second case.
fn write_dump(name: &str, stated: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).expect("a directory to hold the dump");
    let (tokens, vocab) = (8usize, 512usize);
    let mut key = vec![0u8; tokens * vocab * 2];
    let mut tok = vec![0u8; tokens * 4];
    for t in 0..tokens {
        for c in 0..vocab {
            let v = 1 + ((c * 7 + t * 13) % 41) as u16;
            key[(t * vocab + c) * 2..(t * vocab + c) * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        // one entry no other attains, so the emitted token is the row's first
        // maximum and `from_dump` accepts the dump
        let i = (t * 5 + 3) % vocab;
        key[(t * vocab + i) * 2..(t * vocab + i) * 2 + 2].copy_from_slice(&100u16.to_le_bytes());
        tok[t * 4..t * 4 + 4].copy_from_slice(&(i as i32).to_le_bytes());
    }
    std::fs::write(dir.join("key_head.u16"), &key).expect("write key_head.u16");
    std::fs::write(dir.join("tok_head.i32"), &tok).expect("write tok_head.i32");
    let man = format!(
        r#"{{
 "layer": 31,
 "block": 4,
 "head": {{"tokens": 8, "vocab": 512}},
 "ids_from": {{"file": "ids.npy", "sha256": "{IDS_SHA}", "bytes": 4096,
               "path": "out/ids.npy", "tokens": 512}},
 "attn_ranges": {{"file": "attn_ranges.json", "sha256": "{CALIB_SHA}", "bytes": 271360,
                  "path": "out/attn_ranges.json", "given": true}},
 "dump_sha256": "{stated}",
 "files": {FILES}
}}"#
    );
    std::fs::write(dir.join("manifest.json"), man).expect("write the manifest");
    dir.to_string_lossy().into_owned()
}

fn run(args: &[String]) -> serde_json::Value {
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

fn argmax_on(dir: Option<&str>) -> serde_json::Value {
    let mut args: Vec<String> = ["argmax", "--tokens", "8", "--vocab", "512", "--per-row", "64"]
        .iter().map(|s| s.to_string()).collect();
    if let Some(d) = dir {
        args.push("--witness".to_string());
        args.push(d.to_string());
    }
    run(&args)
}

#[test]
fn the_provenance_block_names_the_dump_the_run_read() {
    let dir = write_dump("dump-identity", GOLDEN);
    let v = argmax_on(Some(&dir));
    let prov = &v["provenance"];
    assert_eq!(prov["dump_sha256"].as_str(), Some(GOLDEN),
               "the run must name the dump it proved on: {prov}");
    let dumps = prov["dumps"].as_array().expect("provenance.dumps");
    assert_eq!(dumps.len(), 1, "one dump was read: {prov}");
    let d = &dumps[0];
    assert_eq!(d["dir"].as_str(), Some(dir.as_str()));
    assert_eq!(d["sha256"].as_str(), Some(GOLDEN));
    assert_eq!(d["layer"], 31, "the manifest's own layer");
    assert_eq!(d["block"], 4, "the manifest's own block");
    // beside the dump's digest, what the dump itself was made from
    assert_eq!(d["ids_from"]["sha256"].as_str(), Some(IDS_SHA),
               "the ids the dump was extracted from");
    assert_eq!(d["attn_ranges"]["sha256"].as_str(), Some(CALIB_SHA),
               "the attention calibration the dump was quantized under");
    assert_eq!(d["attn_ranges"]["given"], true,
               "and whether that calibration was given or was the host's own");
    assert!(d.get("dump_sha256_stated").is_none(),
            "the manifest states the roll-up its own arrays give: {d}");
}

#[test]
fn a_manifest_that_disagrees_with_its_own_arrays_is_reported() {
    let bent = "0000000000000000000000000000000000000000000000000000000000000000";
    let dir = write_dump("dump-identity-bent", bent);
    let v = argmax_on(Some(&dir));
    let d = &v["provenance"]["dumps"][0];
    assert_eq!(d["sha256"].as_str(), Some(GOLDEN),
               "the digest is the arrays the manifest lists, not the number it states");
    assert_eq!(d["dump_sha256_stated"].as_str(), Some(bent),
               "and a manifest edited since it was written is said to be: {d}");
}

#[test]
fn a_run_that_read_no_dump_says_so() {
    let v = argmax_on(None);
    let prov = &v["provenance"];
    assert_eq!(prov["dumps"], serde_json::json!([]),
               "a synthetic segment reads no dump: {prov}");
    assert_eq!(prov["dump_sha256"], serde_json::Value::Null,
               "and there is no dump digest to quote, stated as null: {prov}");
}
