//! The eight cell-cost subcommands, run through the binary.
//!
//! `dequant`, `width`, `kmax`, `pack`, `requant`, `residual-stream`, `argmax`
//! and `freivalds` each build one component on a synthetic shape and report what
//! it costs in committed cells; the cost model reads its cells-per-operation
//! constants off their JSON. None of them makes a proof whose digest is pinned,
//! so before this test nothing in the repository ran any of them, and seven of
//! them panicked in the vendored library on their own defaults (issue #70).
//!
//! Each case below is the command the micro-benchmark driver runs, with the
//! shape cut down so the whole test stays under a minute. Which trees a proof
//! commits does not depend on the shape, so a small witness catches the break
//! the large one had: the base trace at tree 2 and the interaction trace at
//! tree 3, which is where every component's own mask says they are.
//!
//! One default is not reproduced here and cannot be: `pack`'s own sweep runs 1,
//! 2, 4, 8 and 16 instances, and at 8 and 16 the per-instance exponent spread
//! puts the shared exponent table past the 2^22 rows `attn_witness` caps it at,
//! so nine of its keys fall off the table and the honest proof is refused. That
//! is a separate defect from the tree layout this test pins, it is older than
//! the split, and it is not fixed here; the case below runs 1 and 2.
//!
//! These run on the SIMD build alone. The eight subcommands are host-only by
//! design -- the micro-benchmark campaign runs them untimed on the host -- and
//! their synthetic shapes are in no kernel store, so a binary built with
//! `--features cuda` takes the device path and ends every one of them in
//! `devfrac`'s "no generation kernel" before a cell is counted. So the test is
//! ignored under the feature, by name and with its reason in the run's own
//! output, rather than only saying here which backend it wants.

use std::process::Command;

struct Case {
    /// the subcommand and the flags after it
    args: &'static [&'static str],
    /// the name the run prints as `bench`
    bench: &'static str,
}

/// The eight, in the order the driver runs them. The flags that differ from the
/// driver's are the ones that only set the size of the witness or the number of
/// points measured: `--kmax` (the dequant tables are `2^(kmax+1)` rows), `--n`,
/// `--rows`, `--tokens`, `--vocab`, `--widths`, `--kmax-sweep`, `--insts` and
/// the context of the packed attention.
const CASES: &[Case] = &[
    Case { bench: "dequant",
           args: &["dequant", "--n", "4096", "--per-row", "16", "--kmax", "12",
                   "--variants", "limbs,round,pow2,chain,int8", "--round", "even"] },
    Case { bench: "dequant_width",
           args: &["width", "--n", "4096", "--kmax", "12", "--widths", "4,16"] },
    Case { bench: "dequant_kmax",
           args: &["kmax", "--n", "4096", "--per-row", "16", "--kmax-sweep", "10,12"] },
    Case { bench: "pack",
           args: &["pack", "--s", "32", "--ctx", "64", "--hd", "64", "--insts", "1,2",
                   "--variants", "intexp"] },
    Case { bench: "requant", args: &["requant", "--round", "even"] },
    Case { bench: "residual_stream",
           args: &["residual-stream", "--rows", "256", "--leaves", "32", "--per-row", "16",
                   "--kmax", "18", "--round", "even"] },
    Case { bench: "argmax",
           args: &["argmax", "--tokens", "8", "--vocab", "512", "--per-row", "64"] },
    // `freivalds` proves one weight block twice, at `--nout` and at twice it, and
    // reports the difference; the flags below only make the block small.
    Case { bench: "freivalds",
           args: &["freivalds", "--s", "16", "--ctx", "32", "--hd", "16", "--nin", "32",
                   "--nout", "32", "--widths", "8,16"] },
];

fn run(args: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_zk-prover"))
        .args(args)
        .env("RAYON_NUM_THREADS", "4")
        .output()
        .expect("run the prover binary");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "zk-prover {} failed:\n{err}{}",
        args.join(" "),
        if err.contains("no generation kernel") {
            "\nthat is a CUDA build with no kernel for this shape, not a subcommand that \
             ran and disagreed: this test wants the SIMD build"
        } else {
            ""
        }
    );
    serde_json::from_slice(&out.stdout).expect("one JSON document on stdout")
}

/// Every `cells` object anywhere in the document, deepest last.
fn cell_counts(v: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, val) in m {
                if k == "cells" && val.is_object() {
                    out.push(val.clone());
                }
                cell_counts(val, out);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| cell_counts(x, out)),
        _ => {}
    }
}

fn u64_at(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

#[test]
#[cfg_attr(feature = "cuda", ignore = "the cell-cost subcommands are host-only: a CUDA build \
                                       has no generation kernel for their synthetic shapes")]
fn the_cell_cost_subcommands_run_and_report_their_cells() {
    for c in CASES {
        let v = run(c.args);
        assert_eq!(v["bench"].as_str(), Some(c.bench),
                   "zk-prover {}: the JSON names another measurement", c.args.join(" "));

        let mut cells = vec![];
        cell_counts(&v["result"], &mut cells);
        assert!(!cells.is_empty(),
                "zk-prover {}: no cell count in the JSON", c.args.join(" "));

        for cs in &cells {
            // The counts a gadget reads off its own committed trees. `base` is
            // tree 2 and `interaction` tree 3: with the model tree left out
            // they were the interaction tree's count and zero, which is the
            // same mistake that made `TreeVec::sub_tree` panic.
            if let (Some(base), Some(inter)) = (u64_at(cs, "base"), u64_at(cs, "interaction")) {
                assert!(base > 0 && inter > 0,
                        "zk-prover {}: base {base} and interaction {inter} cells; the base \
                         trace is tree 2 and the interaction trace tree 3, so a zero here \
                         means the proof committed fewer trees than its components name",
                        c.args.join(" "));
            }
            if let Some(fresh) = u64_at(cs, "fresh") {
                assert!(fresh > 0, "zk-prover {}: no fresh cells", c.args.join(" "));
            }
        }
    }
}
