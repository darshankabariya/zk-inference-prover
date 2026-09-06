#![recursion_limit = "1024"]
//! B6 — the end-to-end mini pipeline, and the dequant gadget it needs first.
//!
//! Subcommands: tables | dequant | selftest.  JSON on stdout, progress on stderr.

mod attn_air;
mod attn_run;
mod attn_trace;
mod argmax;
mod attn_witness;
mod bf16;
mod bounds;
mod common;
mod device;
mod compose;
mod cudagen;
mod devlogup;
mod devfill;
mod devfrac;
mod devpre;
mod chunks;
mod devtab;
mod precache;
mod scalars;
mod devacc;
mod devwfill;
mod devgfill;
mod gkrband;
mod gkrlink;
mod gencudakernels;
mod cudakernels;
mod exprprobe;
mod dequant;
mod embed;
mod glue;
mod layer;
mod layout;
mod plogup;
mod requant;
mod rope;
mod walk_air;
mod witread;
mod spans;
mod util;

use serde_json::{Value, json};

use attn_witness::{Attn, Corrupt, Src};
use bf16::Round;
use dequant::{Spec, Tamper, Variant, Witness};
use layout::Shape;
use util::emit;

pub fn arg_scalar<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.windows(2).find(|w| w[0] == key).and_then(|w| w[1].parse().ok()).unwrap_or(default)
}

pub fn arg_list<T: std::str::FromStr + Copy>(args: &[String], key: &str, default: &[T]) -> Vec<T> {
    match args.windows(2).find(|w| w[0] == key) {
        Some(w) => w[1].split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        None => default.to_vec(),
    }
}

pub fn arg_strs(args: &[String], key: &str, default: &[&str]) -> Vec<String> {
    match args.windows(2).find(|w| w[0] == key) {
        Some(w) => w[1].split(',').map(|s| s.trim().to_string()).collect(),
        None => default.iter().map(|s| s.to_string()).collect(),
    }
}

pub fn arg_str(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn spec_of(args: &[String], variant: &str, round: &str, per_row: usize) -> Spec {
    Spec {
        variant: Variant::parse(variant),
        round: Round::parse(round),
        // 21 is the largest sound value: 512 * 2^21 = 2^30 < p (bf16::KMAX_SOUND)
        kmax: arg_scalar(args, "--kmax", 21u32),
        per_row,
        limb_bits: (arg_scalar(args, "--limb-lo", 16u32), arg_scalar(args, "--limb-hi", 14u32)),
        // 7 bits on the quotient is B1's clamp at 127; 19 on the error covers
        // `2 den` for the denominators B5 measured (Z max 38,848, blind bound 2^22)
        q_bits: arg_scalar(args, "--q-bits", 7u32),
        rem_bits: arg_scalar(args, "--rem-bits", 19u32),
        fold_alpha: 0,
    }
}

pub fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

pub fn src_of(args: &[String]) -> Src {
    Src {
        dir: arg_str(args, "--witness"),
        head: arg_scalar(args, "--head", 0usize),
        q_bits: arg_scalar(args, "--qbits", 12u32),
        k_bits: arg_scalar(args, "--kbits", 11u32),
        nmax: (1u32 << arg_scalar(args, "--numbits", 8u32)) - 1,
        // 0 = as many rows as the witness needs, i.e. a complete table
        log_table: arg_scalar(args, "--table-bits", 0u32),
        seed: arg_scalar(args, "--seed", 7u64),
        // B10 step 0: empty = the real witness's own `deq_variants`, B6.5's
        // mixed layout for a synthetic one.  `--layout b10` is the chain on all
        // seven, which is the shape the emit chain has to generate for B10.
        layout: arg_strs(args, "--layout", &[""])[0].clone(),
        // B10 step 3: prove the band as the causal corner instead of the prefix
        // rectangle.  `--ctx` must equal `--s`; the mask is `key <= token`.
        corner: args.iter().any(|a| a == "--corner"),
        // Stream AA (E11): `--kv-bind` also tiles the declared prefix into
        // blocks of S keys, which is a witness-side decision as much as a plan
        // one -- see `briefs/aa-kvbind.md`.
        tile: args.iter().any(|a| a == "--kv-bind"),
    }
}

pub fn attn_cfg(args: &[String], grid_target: usize) -> attn_trace::Cfg {
    let nmax = (1u32 << arg_scalar(args, "--numbits", 8u32)) - 1;
    attn_trace::Cfg {
        chan_target: arg_scalar(args, "--keys-per-row", 128usize),
        grid_target,
        rw: arg_scalar(args, "--tokens-per-row", 16usize),
        range_check: !flag(args, "--no-range"),
        // 16 + 6 = 22 bits on the denominator: B5's parameter, which is exactly
        // what puts |A.V| inside the dequant's 2^30 window
        z_lo_bits: arg_scalar(args, "--z-lo-bits", 16u32),
        z_hi_bits: arg_scalar(args, "--z-hi-bits", 6u32),
        z_min: arg_scalar(args, "--z-min", nmax),
    }
}

/// Median of `reps` timings, with the spread recorded.
fn median(mut runs: Vec<(f64, Value)>) -> Value {
    let n = runs.len();
    runs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (lo, hi) = (runs[0].0, runs[n - 1].0);
    let mut j = runs.swap_remove(n / 2).1;
    j["reps"] = json!(n);
    j["prove_secs_min"] = json!(lo);
    j["prove_secs_max"] = json!(hi);
    j
}

// ------------------------------------------------------------------ tables ----
/// The two tables, checked against an independent single f64 rounding.
fn tables_cmd(args: &[String]) {
    let mut out = vec![];
    for k in arg_list(args, "--kmax", &[12u32, 16, 21]) {
        for r in arg_strs(args, "--round", &["even", "away", "zero"]) {
            eprintln!("tables kmax={k} round={r}");
            out.push(dequant::check_tables(k, Round::parse(&r),
                                           arg_scalar(args, "--samples", 200_000usize)));
        }
    }
    emit("tables", json!({"checks": out}));
}

// ----------------------------------------------------------------- dequant ----
/// The ladder: what each way of scaling and rounding an accumulator costs.
fn dequant_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 20);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let k: usize = arg_scalar(args, "--k", 896usize);
    let mut points = vec![];
    for vname in arg_strs(args, "--variants", &dequant::LADDER) {
        for rname in arg_strs(args, "--round", &["even"]) {
            let sp = spec_of(args, &vname, &rname, per_row);
            let w = Witness::get(&dir, &mat, n, per_row, k);
            let mut runs = vec![];
            for rep in 0..reps.max(1) {
                eprintln!("dequant {} n={n} per_row={per_row} (rep {})", sp.label(), rep + 1);
                let o = dequant::run(&w, &sp, None);
                if !o.accepts() {
                    eprintln!("  REJECTED: stark={} sums={} off_table={} err={:?}",
                              o.proven.stark_ok, o.sums.cancels(), o.off_table,
                              o.proven.prove_error);
                }
                assert!(o.accepts(), "honest dequant proof rejected ({})", sp.label());
                assert_eq!(o.ref_n - o.ref_ok, 0,
                           "gadget disagrees with the f64 reference ({})", sp.label());
                runs.push((o.proven.prove_secs, o.json(&w.source)));
            }
            points.push(median(runs));
        }
    }
    emit("dequant", json!({"points": points, "n": n, "per_row": per_row}));
}

// ------------------------------------------------------------------- width ----
/// What a wider row buys, B3's trade-off on a third region.
fn width_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 20);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let v = arg_strs(args, "--variants", &["pow2"])[0].clone();
    let mut points = vec![];
    for pr in arg_list(args, "--widths", &[4usize, 8, 16, 32, 64, 128]) {
        let sp = spec_of(args, &v, &arg_strs(args, "--round", &["even"])[0], pr);
        let w = Witness::get(&dir, &mat, n, pr, arg_scalar(args, "--k", 896usize));
        let mut runs = vec![];
        for _ in 0..reps.max(1) {
            eprintln!("width {} per_row={pr}", sp.label());
            let o = dequant::run(&w, &sp, None);
            assert!(o.accepts(), "honest proof rejected at per_row={pr}");
            runs.push((o.proven.prove_secs, o.json(&w.source)));
        }
        points.push(median(runs));
    }
    emit("dequant_width", json!({"points": points, "n": n, "variant": v}));
}

// ------------------------------------------------------------------- kmax -----
/// `kmax` is a completeness knob: how many table rows buy how much accumulator.
fn kmax_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 18);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let mut points = vec![];
    for k in arg_list(args, "--kmax-sweep", &[10u32, 12, 14, 16, 18, 21]) {
        let mut sp = spec_of(args, "pow2", "even", per_row);
        sp.kmax = k;
        let w = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
        eprintln!("kmax {k}");
        let o = dequant::run(&w, &sp, None);
        let mut j = o.json(&w.source);
        j["accepts"] = json!(o.accepts());
        points.push(j);
    }
    emit("dequant_kmax", json!({"points": points, "n": n, "per_row": per_row}));
}

// ------------------------------------------------------------------ probes ----
/// Everything that has to be rejected, and the one thing that must not be.
fn probes_cmd(args: &[String]) {
    let dir = arg_str(args, "--witness");
    let n: usize = arg_scalar(args, "--n", 1 << 12);
    let per_row: usize = arg_scalar(args, "--per-row", 8usize);
    let mat = arg_str(args, "--mat").unwrap_or_else(|| "gate".into());
    let sp = spec_of(args, "pow2", "even", per_row);
    let w = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
    let mut out = vec![];

    let honest = dequant::run(&w, &sp, None);
    assert!(honest.accepts(), "honest dequant proof rejected");
    let honest_root = honest.proven.roots.first().cloned().unwrap_or_default();
    out.push(json!({"probe": "honest", "note": "the reference run",
                    "result": honest.json(&w.source)}));

    let mut probe = |name: &str, note: &str, t: Tamper| {
        eprintln!("probe {name}");
        let o = dequant::run(&w, &sp, Some(t));
        let root = o.proven.roots.first().cloned().unwrap_or_default();
        out.push(json!({"probe": name, "note": note, "result": o.json(&w.source),
                        "root_differs_from_honest": root != honest_root}));
        (o.accepts(), o.proven.stark_ok, o.sums.cancels(), o.off_table)
    };

    // The wrap probe is the one that matters: it is B4 §1's negative control, and
    // it passes here *because* the normalisation's own bound is the integer reading.
    let (ok, ..) = probe(
        "wrap: decompose acc + p",
        "the same field element, a different integer.  B4's two 2^16 limbs accept \
         this and decode an accumulator off by exactly p; the normalisation cannot, \
         because M2 < 512 and ph <= 2^kmax bound the reconstruction below 2^30",
        Tamper::Wrap(11));
    assert!(!ok, "a wrap-around decomposition must be rejected");

    let (ok, stark, cancels, _) = probe(
        "wrong mantissa",
        "the committed bf16 mantissa is a different, perfectly legal value: every \
         constraint still holds and only the claimed sums object",
        Tamper::Mantissa(13));
    assert!(!ok && stark && !cancels,
            "a wrong mantissa must be caught by the claimed sums, not by verify()");

    let (ok, ..) = probe(
        "non-normalised decomposition",
        "M2 halved and ph doubled, which keeps mag = M2 ph + r satisfied: the \
         normalisation table's domain is what rejects it, since rows with M2 < 256 \
         exist only at ph = 1",
        Tamper::Denorm(17));
    assert!(!ok, "a non-normalised decomposition must be rejected");

    let (ok, ..) = probe(
        "remainder above its bound",
        "r >= ph with M2 lowered to compensate, so the reconstruction is still \
         satisfied: the (ph, r) pair table is the only thing that can see it",
        Tamper::Borrow(19));
    assert!(!ok, "r >= ph must be rejected");

    let (ok, ..) = probe(
        "sticky flag flipped",
        "the flag round-to-nearest-even reads, and the (ph, r, st) table pins it \
         from r -- so flipping it leaves no valid row",
        Tamper::Sticky(23));
    assert!(!ok, "a flipped sticky flag must be rejected");

    let (ok, _, cancels, _) = probe(
        "corrupted table",
        "one row of the normalisation table changed and the whole witness rebuilt \
         to match: internally consistent, claimed sums cancel, and only the \
         preprocessed root differs (B4 section 3, third gadget)",
        Tamper::Table(4321));
    assert!(ok && cancels, "a corrupted table with a matching witness is consistent");
    out.last_mut().unwrap()["extra"] =
        json!({"preprocessed_root_honest": honest_root,
               "note": "the root is the only difference, so every table root is \
                        part of the statement"});

    // the int8 fold's two closed edges (B6 review, open gap 7): both must reject
    // now, and both verified under B6's gadget
    let sp8 = spec_of(args, "int8", "even", per_row);
    let w8 = Witness::get(&dir, &mat, n, per_row, arg_scalar(args, "--k", 896usize));
    let honest8 = dequant::run(&w8, &sp8, None);
    assert!(honest8.accepts(), "honest int8 proof rejected");
    // an element with qm >= 1 for the tie shift, one with qm = 0, acc < 0 for the sign
    let pick = |want_zero: bool| -> Option<usize> {
        (0..w8.acc.len()).find(|i| {
            let den = w8.den[(i / per_row) % w8.den.len()];
            let num = w8.num[(i % per_row) % w8.num.len()];
            let qm = (2 * w8.acc[*i].abs() * num + den) / (2 * den);
            if want_zero { qm == 0 && w8.acc[*i] < 0 } else { qm >= 1 }
        })
    };
    let mut out8 = vec![];
    if let Some(i) = pick(false) {
        eprintln!("probe int8 tie shift");
        let o = dequant::run(&w8, &sp8, Some(Tamper::TieShift(i)));
        assert!(!o.accepts(), "the tie's second witness must be rejected");
        out8.push(json!({"probe": "int8: the tie's other witness (qm - 1, d + 2 den)",
                         "note": "B6's `2 den - d` check accepted this whenever d = 0; \
                                  the -1 excludes it for every d",
                         "accepted": o.accepts(), "stark_ok": o.proven.stark_ok,
                         "sums_cancel": o.sums.cancels(), "off_table": o.off_table}));
    }
    if let Some(i) = pick(true) {
        eprintln!("probe int8 sign flip");
        let o = dequant::run(&w8, &sp8, Some(Tamper::SignFlip(i)));
        assert!(!o.accepts(), "a flipped sub-half sign must be rejected");
        out8.push(json!({"probe": "int8: flipped sign on a sub-half quotient",
                         "note": "accepted before the (q, nz) pin; the constraint \
                                  sg (1 - nz) now refuses it in the prover",
                         "accepted": o.accepts(), "stark_ok": o.proven.stark_ok,
                         "sums_cancel": o.sums.cancels(), "off_table": o.off_table,
                         "prove_error": o.proven.prove_error}));
    } else {
        out8.push(json!({"probe": "int8: flipped sub-half sign",
                         "note": "no qm = 0, acc < 0 element in this witness"}));
    }
    emit("dequant_probes", json!({"n": n, "spec": sp.json(), "probes": out,
                                  "int8_edges": out8}));
}

// -------------------------------------------------------------------- pack ----
/// The column budget: what packing `insts` heads onto one column set buys.
///
/// This is the question B6 exists to answer.  B5 proved one head at ~1,200 columns
/// and 600 KiB-1 MiB of proof; B1's model has 24 x 14 = 336 head-instances, so
/// per-head instantiation extrapolates to a 200-340 MB proof, two orders over
/// a single-digit MiB proof.  Packing shares one column set across the band, and
/// the sweep below is what that is worth.
fn pack_cmd(args: &[String]) {
    let src = src_of(args);
    let s: usize = arg_scalar(args, "--s", 512usize);
    let c: usize = arg_scalar(args, "--ctx", 1024usize);
    let hd: usize = arg_scalar(args, "--hd", 64usize);
    let reps: usize = arg_scalar(args, "--reps", 1usize);
    let cfg = attn_cfg(args, arg_scalar(args, "--scores-per-row", 64usize));
    let v = attn_witness::Variant::parse(&arg_strs(args, "--variants", &["intexp"])[0]);
    let mut points = vec![];
    for insts in arg_list(args, "--insts", &[1usize, 2, 4, 8, 16]) {
        assert!(insts.is_power_of_two(), "insts must be a power of two (see layout.rs)");
        let sh = Shape { s, c, hd, insts };
        let a = Attn::get(&src, sh, v);
        let mut runs = vec![];
        for rep in 0..reps.max(1) {
            eprintln!("pack {} {} (rep {})", sh.label(), v.name(), rep + 1);
            let o = attn_run::run(&a, &cfg);
            if !o.accepts() {
                eprintln!("  REJECTED: stark={} qk={} av={} den={} sums={} off={}",
                          o.proven.stark_ok, o.qk_ok, o.av_ok, o.den_ok, o.sums.cancels(),
                          o.off_table);
            }
            assert!(o.accepts(), "honest packed proof rejected at insts={insts}");
            runs.push((o.proven.prove_secs, o.json(&a, &cfg)));
        }
        let mut p = median(runs);
        p["insts"] = json!(insts);
        points.push(p);
    }
    emit("pack", json!({"points": points, "s": s, "ctx": c, "hd": hd}));
}

/// Everything the packed attention has to reject, at a band of instances.
fn attn_probes_cmd(args: &[String]) {
    let src = src_of(args);
    let sh = Shape {
        s: arg_scalar(args, "--s", 64usize),
        c: arg_scalar(args, "--ctx", 256usize),
        hd: arg_scalar(args, "--hd", 64usize),
        insts: arg_scalar(args, "--insts", 4usize),
    };
    let cfg = attn_cfg(args, arg_scalar(args, "--scores-per-row", 16usize));
    let v = attn_witness::Variant::IntExp;
    let honest = attn_run::run(&Attn::get(&src, sh, v), &cfg);
    assert!(honest.accepts(), "honest packed proof rejected");
    let mut out = vec![json!({"probe": "honest",
                              "result": honest.json(&Attn::get(&src, sh, v), &cfg)})];
    let mut cases: Vec<(&str, Corrupt)> = vec![
        ("corrupted key in one instance", Corrupt::Key(1234, 3)),
        ("corrupted value in one instance", Corrupt::Value(999, 5)),
        ("wrong numerator in one instance", Corrupt::Numerator(555)),
        ("shift below the maximum in one instance", Corrupt::LowMax(3)),
        ("corrupted denominator in one instance", Corrupt::Denominator(7, 1)),
    ];
    if src.corner {
        // the corner's own control: a token that attends to a future key
        cases.push(("attends to one future key", Corrupt::Unmask(11)));
        cases.push(("attends to a second future key", Corrupt::Unmask(4242)));
    }
    for (name, bad) in cases {
        eprintln!("attn probe {name}");
        let a = Attn::corrupted(&src, sh, v, bad);
        let o = attn_run::run(&a, &cfg);
        assert!(!o.accepts(), "{name} must be rejected");
        out.push(json!({"probe": name, "result": o.json(&a, &cfg)}));
    }
    emit("attn_probes", json!({"shape": sh.label(), "probes": out}));
}

// ------------------------------------------------------------------ argmax ----
/// B10 step 4: the token tie, at both strengths, so the difference is measured.
fn argmax_cmd(args: &[String]) {
    let tokens: usize = arg_scalar(args, "--tokens", 8usize);
    let vocab: usize = arg_scalar(args, "--vocab", 4096usize);
    let ties: f64 = arg_scalar(args, "--ties", 0.25f64);
    let pr: usize = arg_scalar(args, "--per-row", 64usize);
    let seed: u64 = arg_scalar(args, "--seed", 7u64);
    // `--witness DIR` proves the argmax on the head's own keys: `key_head.u16`
    // and `tok_head.i32` out of an `extract.py --head` dump, with the shape read
    // from that dump's manifest rather than from the flags.
    let w = match arg_str(args, "--witness") {
        Some(dir) => {
            let man: serde_json::Value = serde_json::from_slice(
                &std::fs::read(format!("{dir}/manifest.json")).expect("manifest.json"))
                .expect("manifest.json parses");
            let tk = man["head"]["tokens"].as_u64().expect("manifest: head.tokens") as usize;
            let vc = man["head"]["vocab"].as_u64().expect("manifest: head.vocab") as usize;
            eprintln!("argmax on the dump's own keys: {tk} rows of {vc}");
            argmax::AWitness::from_dump(&dir, tk, vc).expect("the head dump's argmax columns")
        }
        None => argmax::AWitness::synthetic(tokens, vocab, ties, seed),
    };
    let (tokens, vocab) = (w.tokens, w.vocab);
    let mut points = vec![];
    for lax in [false, true] {
        eprintln!("argmax {}x{} lax={lax}", tokens, vocab);
        let (o, lay) = argmax::run(&w, lax, pr, None);
        if !o.accepts() {
            eprintln!("  REJECTED: stark={} sums={} off={}", o.proven.stark_ok,
                      o.sums.cancels(), o.off_table);
        }
        assert!(o.accepts(), "honest argmax proof rejected (lax={lax})");
        points.push(o.json(&w, &lay));
    }
    emit("argmax", json!({"points": points, "ties_in_witness": w.ties, "tokens": tokens,
                          "vocab": vocab}));
}

/// What each statement rejects -- and, for the tie, what only the strict one does.
fn argmax_probes_cmd(args: &[String]) {
    let tokens: usize = arg_scalar(args, "--tokens", 8usize);
    let vocab: usize = arg_scalar(args, "--vocab", 4096usize);
    let pr: usize = arg_scalar(args, "--per-row", 64usize);
    // every row tied, so the tie probe always has a second maximum to move to
    let w = argmax::AWitness::synthetic(tokens, vocab, 1.0, arg_scalar(args, "--seed", 7u64));
    let cases: Vec<(&str, argmax::Tamper)> = vec![
        ("the token is a *later* co-maximal entry", argmax::Tamper::Tie(3)),
        ("the token does not attain the maximum", argmax::Tamper::NotMax(5)),
        ("the claimed maximum is above every entry", argmax::Tamper::PmHigh(2)),
        ("the claimed maximum is below the true one", argmax::Tamper::PmLow(4)),
        // the strict close's own probe: `tok = vocab` is the all-ones indicator,
        // killed only by the final-slot constraint; the lax sum rejects it too
        ("the token equals the vocabulary size", argmax::Tamper::TokVocab(1)),
    ];
    let mut out = vec![];
    for lax in [false, true] {
        for (name, bad) in &cases {
            eprintln!("argmax probe [{}] {name}", if lax { "lax" } else { "strict" });
            let (o, lay) = argmax::run(&w, lax, pr, Some(*bad));
            out.push(json!({"probe": name, "lax": lax, "rejected": !o.accepts(),
                            "result": o.json(&w, &lay)}));
            // The tie probe is exactly the difference between the two statements:
            // the lax one cannot see it, and saying so is the point of running it.
            let is_tie = name.contains("co-maximal");
            if !lax {
                assert!(!o.accepts(), "strict: {name} must be rejected");
            } else if !is_tie {
                assert!(!o.accepts(), "lax: {name} must be rejected");
            }
        }
    }
    emit("argmax_probes", json!({"probes": out, "tokens": tokens, "vocab": vocab,
                                 "ties_in_witness": w.ties}));
}


// ------------------------------------------------------------------ embed ----

/// Stream BB: the embedding gadget on its own, and its cells price.
///
/// `--witness DIR` proves it on the dump's own `embed.u16`, `tok_in.i32` and
/// the wiring family's `xin`; without one it runs a synthetic block, which is
/// a shape check and nothing more -- the record says so wherever a synthetic
/// number appears.
fn embed_cmd(args: &[String]) {
    let target: usize = arg_scalar(args, "--embed-per-row", 256usize);
    let w = match arg_str(args, "--witness") {
        Some(dir) => {
            let (l, _) = {
                let mut a = args.to_vec();
                a.push("--witness".into());
                a.push(dir.clone());
                layer_of(&a)
            };
            let g = l.glue.as_ref().expect("embed needs a witness extracted with --glue");
            let tab = l.embed_tab.as_ref().expect("embed needs embed.u16 in the dump");
            let tok = l.tok_in.as_ref().expect("embed needs tok_in.i32 in the dump");
            let vocab = tab.len() / g.d;
            eprintln!("embed on the dump's own table: {} positions of {} channels, \
                       vocabulary {vocab}", g.tokens, g.d);
            embed::EWitness::from_cols(g.tokens, g.d, vocab, tab.clone(), tok.clone(),
                                       g.xin.clone())
                .expect("the dump's embedding columns")
        }
        None => embed::EWitness::synthetic(
            arg_scalar(args, "--tokens", 16usize),
            arg_scalar(args, "--d", 128usize),
            arg_scalar(args, "--vocab", 512usize),
            arg_scalar(args, "--seed", 11u64)),
    };
    let (o, lay) = embed::run(&w, target, None);
    if !o.accepts() {
        eprintln!("  REJECTED: stark={} sums={} off={} tie={}", o.proven.stark_ok,
                  o.sums.cancels(), o.off_table, o.tie_ok);
    }
    assert!(o.accepts(), "honest embedding proof rejected");
    // the forecast at the two real vocabularies, beside the measured point, so
    // the record never has to re-derive the arithmetic
    let price = json!({
        "llama8b": embed::price(4096, 512, 128_256, target),
        "qwen05b": embed::price(896, 512, 151_936, target),
    });
    // What the emit chain will see, at the *real* shapes: an emit is a box slot,
    // so the two kernels' sizes are worth knowing before one is booked.  This
    // lowers the two evaluators and reports the IR; it proves nothing and runs
    // no prover.
    let shapes: Vec<serde_json::Value> = [4096usize, 896]
        .iter()
        .flat_map(|d| {
            let vocab = if *d == 4096 { 128_256usize } else { 151_936 };
            let l = embed::EmbLayout::choose(*d, 512, vocab, target);
            let rel = {
                let mut ch = stwo::core::channel::Blake2sM31Channel::default();
                stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_BB01);
                embed::EmbRel::draw(&mut ch, l.arity())
            };
            let rk = {
                let mut ch = stwo::core::channel::Blake2sM31Channel::default();
                stwo::core::channel::Channel::mix_u64(&mut ch, 0xB65_BB02);
                attn_air::RangeRel::draw(&mut ch)
            };
            let felt = |a: u32| stwo::core::fields::qm31::SecureField::from_m31_array(
                [common::m31_u32(a), common::m31_u32(a + 1),
                 common::m31_u32(a + 2), common::m31_u32(a + 3)]);
            let mut tie = embed::tie_w(felt(7), felt(101), *d, 512, &l);
            tie.claimed = felt(1_000_003);
            tie.claimed_tok = felt(2_000_003);
            let use_eval = embed::EmbedEval { log_rows: l.log_use, per_row: l.pe,
                                              rel: rel.clone(), rel_ef: rk, tie };
            let tab_eval = embed::EmbTableEval { log_rows: l.log_tab, per_row: l.pe, rel };
            let one = |name: &str, ir: cudagen::Ir| json!({
                "family": name, "d": d, "vocab": vocab,
                "layout": l.json(512, vocab),
                "fingerprint": format!("{:#010x}", ir.fingerprint()),
                "instructions": ir.ops.len(), "constraints": ir.constraints.len(),
                "scalars": ir.scalars.len(), "m31_muls_per_row": ir.m31_muls(),
                "loads_per_row": ir.loads(),
            });
            vec![
                one("b65_embed_eval",
                    cudagen::lower(&use_eval, stwo::core::fields::qm31::SecureField::default(),
                                   l.log_use, l.log_use + 1)),
                one("b65_emb_table_eval",
                    cudagen::lower(&tab_eval, stwo::core::fields::qm31::SecureField::default(),
                                   l.log_tab, l.log_tab + 1)),
            ]
        })
        .collect();
    emit("embed", json!({"point": o.json(&w, &lay), "price_forecast": price,
                         "emit_shapes": shapes}));
}

/// What the gadget rejects, and which half of the argument each probe aims at.
fn embed_probes_cmd(args: &[String]) {
    let target: usize = arg_scalar(args, "--embed-per-row", 256usize);
    let w = embed::EWitness::synthetic(
        arg_scalar(args, "--tokens", 16usize),
        arg_scalar(args, "--d", 128usize),
        arg_scalar(args, "--vocab", 512usize),
        arg_scalar(args, "--seed", 11u64));
    let cases: Vec<(&str, &str, embed::Tamper)> = vec![
        ("one token id moved, the committed row left alone", "the embedding lookup",
         embed::Tamper::Tok(3)),
        ("one token id moved *and* its row recomputed from the table",
         "the tie to the layer's residual-stream input", embed::Tamper::TokRow(5)),
        ("one committed channel of the stream changed", "the embedding lookup",
         embed::Tamper::Val(777)),
    ];
    let mut out = vec![];
    for (name, by, bad) in &cases {
        eprintln!("embed probe {name}");
        let (o, lay) = embed::run(&w, target, Some(*bad));
        out.push(json!({"probe": name, "expect_reject_by": by, "rejected": !o.accepts(),
                        "result": o.json(&w, &lay)}));
        assert!(!o.accepts(), "{name} must be rejected");
    }
    // The table probe is the pinned-root probe and the standalone harness has no
    // pinned root to check against, so it is *expected to accept here* and the
    // composed run is where it bites. Running it anyway is what says so.
    // Stream KK: the cell is chosen in a row *no token uses*, so the rebuilt
    // witness is the honest one value for value and the pinned root is the only
    // thing that can see the change.  A cell in a used row makes the stream
    // differ from the layer's, which the tie catches first -- a rejection for
    // the wrong reason, and one that came and went with the witness's draw.
    let unused = (0..w.vocab as u32).find(|v| !w.tok.contains(v)).unwrap_or(0) as usize;
    let (o, lay) = embed::run(&w, target, Some(embed::Tamper::Table(unused * w.d + 1)));
    out.push(json!({"probe": "one cell of the committed table changed, witness rebuilt",
                    "expect_reject_by": "the pinned preprocessed root (composed only)",
                    "rejected": !o.accepts(), "accepts_standalone": o.accepts(),
                    "result": o.json(&w, &lay)}));
    assert!(o.accepts(),
            "the table probe must be internally consistent -- only the pinned root sees it");
    // Stream KK, `notes/witness-slack.md` section 1a, both arms of the same
    // witness: a position whose committed id is a *padded* one, outside the
    // vocabulary and inside the table, with the all-zero stream row that padded
    // table row held.  Nothing here is corrupted -- every constraint the region
    // carries holds -- so before the sentinel this accepted, and that is the
    // whole item.  Both runs are made, because "it rejects now" is only half
    // the claim.
    let lay0 = embed::EmbLayout::choose(w.d, w.tokens, w.vocab, target);
    if lay0.vpad == w.vocab {
        out.push(json!({"probe": "a token id outside the vocabulary",
                        "expect_reject_by": "the embedding lookup",
                        "skipped": "this vocabulary is a power of two, so the table \
                                    has no padded row and section 1a is vacuous here"}));
        emit("embed_probes", json!({"probes": out, "tokens": w.tokens, "d": w.d,
                                    "vocab": w.vocab, "vocab_padded": lay0.vpad}));
        return;
    }
    let wp = embed::EWitness::padded(w.tokens, w.d, w.vocab, target, 23, 3);
    let (o_open, lay) = embed::run(&wp, target, Some(embed::Tamper::LegacyPad));
    out.push(json!({"probe": "a token id outside the vocabulary, with a zero stream \
                              row, against the pre-KK table",
                    "expect_reject_by": "nothing -- this is the open statement, and it accepts",
                    "rejected": !o_open.accepts(), "accepts_before_the_fix": o_open.accepts(),
                    "result": o_open.json(&wp, &lay)}));
    let (o_shut, lay) = embed::run(&wp, target, None);
    out.push(json!({"probe": "the same witness against the table this crate builds",
                    "expect_reject_by": "the embedding lookup: a padded row carries the \
                                         out-of-range block index k, is the tuple of no use \
                                         row, and its multiplicity cannot cancel",
                    "rejected": !o_shut.accepts(), "result": o_shut.json(&wp, &lay)}));
    assert!(o_open.accepts(),
            "the pre-KK table must accept the padded id -- that is what section 1a records");
    assert!(!o_shut.accepts(), "the padded id must be rejected now");
    emit("embed_probes", json!({"probes": out, "tokens": w.tokens, "d": w.d,
                                "vocab": w.vocab, "vocab_padded": lay0.vpad}));
}

// ----------------------------------------------------------------- compose ----

/// Build the layer witness: real from `--witness`, else a small synthetic one.
///
/// A `--witness` that does not load is fatal (exit 2), never a fallback.  It
/// used to print one stderr line and prove a synthetic layer instead, and a
/// synthetic layer has `glue: None`, so a gate pointed at a dump the box did
/// not hold came back `rc=0` with the whole glue battery skipped -- a pass that
/// measured nothing (open item 17).
/// Stream KK: the chunk's declared coordinate, `--layer N` / `--block N` over
/// the manifest's own.
///
/// Not a switch on a check: the coordinate goes into the statement either way
/// and `bounds-grid` compares it either way.  It is an override because one
/// chunk's manifest cannot know where the grid puts it -- the `lm_head` chunk
/// is extracted at some layer and proved *after* the last one, so its plan
/// line says `--layer 32` and its block is the block it reads.
fn coord_of(l: &mut layer::Layer, args: &[String]) {
    if let Some(v) = arg_str(args, "--layer") {
        l.layer = Some(v.parse().expect("--layer takes a layer index"));
    }
    if let Some(v) = arg_str(args, "--block") {
        l.block = Some(v.parse().expect("--block takes a block index"));
    }
}

pub fn layer_of(args: &[String]) -> (layer::Layer, String) {
    let vname = arg_strs(args, "--variants", &["intsplit6"])[0].clone();
    let variant = attn_witness::Variant::parse(&vname);
    let bands = arg_list(args, "--bands", &[8usize, 4, 2]);
    let src = src_of(args);
    if let Some(dir) = &src.dir {
        // Stream LL (E11): the verifier's load.  Only the model half of the dump
        // is read; see `layer::Layer::load_weights`.  `--tight-table` sizes the
        // exp table off the witness's own shifts, so it is refused here rather
        // than silently sizing a table from stand-in zeros.
        let weights_only = flag(args, "--weights-only");
        assert!(!weights_only || !args.iter().any(|a| a == "--tight-table"),
                "--weights-only and --tight-table are exclusive: --tight-table \
                 sizes the exp table off the witness's own shifts, and the \
                 model-only load has no witness -- the size comes from the \
                 statement's `log_table` instead");
        // Under `--weights-only` **every** loader knob comes off the statement,
        // not off the command line: the variant, the prefix band list, whether
        // the dump is read with its corner and whether the prefix is tiled, the
        // master table's size and `nmax`.  A verifier holds `Fixed`, not the
        // plan line that produced it -- and a chunk proved with `--tight-table`
        // carries a `log_table` that neither the manifest nor any flag names.
        // So `b65 expect-root --weights-only --params <tag>.fixed.json --witness
        // DIR` is the whole command, and there is no line to get wrong.
        let (variant, vname, bands, src) = if weights_only {
            let f = arg_str(args, "--params").expect(
                "--weights-only needs --params FILE (the chunk's own `Fixed.params`)");
            let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--params {f}: {e}"));
            let v: serde_json::Value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("--params {f}: not JSON: {e}"));
            let p: compose::Params =
                serde_json::from_value(v.get("params").cloned().unwrap_or(v))
                    .unwrap_or_else(|e| panic!("--params {f}: not a statement's params: {e}"));
            eprintln!("expect-root: model-only load from the statement \
(variant={} bands={:?} corner={} kv_bind={} log_table={} nmax={})",
                      p.variant, p.bands, p.corner, p.kv_bind, p.log_table, p.nmax);
            (attn_witness::Variant::parse(&p.variant), p.variant.clone(), p.bands.clone(),
             attn_witness::Src { log_table: p.log_table, nmax: p.nmax,
                                 corner: p.corner, tile: p.kv_bind, ..src.clone() })
        } else {
            (variant, vname, bands, src.clone())
        };
        let loaded = if weights_only {
            layer::Layer::load_weights(dir, &src, &bands, variant)
        } else {
            layer::Layer::load(dir, &src, &bands, variant)
        };
        match loaded {
            Ok(mut l) => {
                // Stream LL: the one stand-in the statement has to choose.
                // `jo` is the attention output's per-token scale exponent --
                // committed data, not model data, so the model-only load has
                // none -- and the AO region admits it only inside the declared
                // window (`joff = jo_max - jo >= 0`, `layer.rs`'s "attnout: jo
                // above the window's top").  The window's *floor* is the safe
                // end: it makes `e = jv - jo` non-negative, so the fold's `pj`
                // is 1 and `denp` stays small, where the ceiling would put
                // `2 denp` within a factor of two of the field.  No
                // preprocessed column reads `jo`, so this moves no tree-0 cell.
                if weights_only {
                    let (_, p) = plan_params(args);
                    for x in l.ao.jo.iter_mut() {
                        *x = p.ao_jo_min;
                    }
                }
                tighten_of(&mut l, args);
                coord_of(&mut l, args);
                return (l, vname);
            }
            Err(e) => {
                eprintln!("--witness {dir}: the layer witness did not load: {e}");
                eprintln!("refusing to fall back to a synthetic layer; drop --witness to prove one");
                std::process::exit(2);
            }
        }
    }
    let bands = arg_list(args, "--bands", &[2usize, 1]);
    let mut l = layer::Layer::synthetic(
        arg_scalar(args, "--s", 32usize), arg_scalar(args, "--ctx", 128usize),
        arg_scalar(args, "--hd", 64usize), &bands, variant, &src,
        arg_scalar(args, "--nin", 64usize), arg_scalar(args, "--nout", 48usize), src.seed);
    // Stream Z: exercise the widened attention output on a synthetic layer --
    // a real dump declares its width in the manifest, a synthetic one takes it
    // here.  The int8-calibrated synthetic data stays complete at any wider
    // width, so this widens only the table domain and the o walk.
    l.attn_out_bits = arg_scalar(args, "--synth-ao-bits", l.attn_out_bits);
    // Stream II: the wiring family on a synthetic layer.  A witness-construction
    // switch and not a statement one -- without it a synthetic layer has no glue
    // columns and composes exactly as it always did, which is what keeps every
    // other stream's recorded synthetic digest where it is.
    if flag(args, "--synth-glue") {
        let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
        l.synth_glue(src.seed, arg_scalar(args, "--glue-a", 32usize), round);
    }
    // Stream II (E11): the head chunk on a synthetic layer -- an `lm_head`
    // matmul whose committed input is the final norm's output, and the wiring
    // family that proves it.  A witness-construction switch like
    // `--synth-glue`: without it a synthetic layer has no `lm_head` and no
    // family, and composes exactly as it always did.
    if flag(args, "--synth-head") {
        let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
        l.synth_head(src.seed, arg_scalar(args, "--glue-a", 32usize), round,
                     arg_scalar(args, "--synth-vocab", 256usize));
    }
    tighten_of(&mut l, args);
    coord_of(&mut l, args);
    (l, vname)
}

/// Stream J, off by default: `--tight-table [slack]` sizes the master exp table by
/// the largest key the bands actually look up rather than by the manifest's
/// `d_bits + spread`.  A negative slack is the tamper -- it takes the table *below*
/// what the witness needs, leaving off-table keys with the numerators they already
/// have, which no honest lookup argument can satisfy.
fn tighten_of(l: &mut layer::Layer, args: &[String]) {
    if !args.iter().any(|a| a == "--tight-table") {
        return;
    }
    let slack: i32 = arg_scalar(args, "--tight-table", 0i32);
    let need = l.key_bits() as i32;
    let (before, after) = l.set_master_table((need + slack).max(1) as u32);
    let off: usize = l.bands.iter().map(|b| b.off_table).sum();
    eprintln!("tight-table: 2^{before} -> 2^{after} (key max {}, {} bits, slack {slack}, \
off_table {off})",
              l.bands.iter().map(|b| b.key_max()).max().unwrap_or(0), need);
}

/// The design point a lowering command works on: a witness dump, or -- with
/// `--params FILE` -- the `params` object of a serialized statement.
///
/// Stream EE (open item 21).  `Fixed` carries every field `params_of` derives
/// from a dump, so a `<tag>.fixed.json` a run left on disk is a complete
/// description of that chunk's plan.  Reading it back is how a machine with no
/// witness -- the coordinating box -- lowers a real 8B plan line and compares
/// the fingerprints against what the run asked for.  It is a lowering path
/// only: nothing here can prove, because a statement carries no trace.
pub fn plan_params(args: &[String]) -> (String, compose::Params) {
    if let Some(f) = arg_str(args, "--params") {
        let bytes = std::fs::read(&f)
            .unwrap_or_else(|e| panic!("--params {f}: {e}"));
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("--params {f}: not JSON: {e}"));
        // a `Fixed` (what `--ser-dir` writes) or a bare `Params`
        let p = v.get("params").cloned().unwrap_or(v);
        let params: compose::Params = serde_json::from_value(p)
            .unwrap_or_else(|e| panic!("--params {f}: not a statement's params: {e}"));
        eprintln!("params: {f} (s={} c={} bands={:?} kv_bind={})",
                  params.s, params.c, params.bands, params.kv_bind);
        return (format!("--params {f}"), params);
    }
    let (l, vname) = layer_of(args);
    let params = params_of(&l, args, &vname);
    (l.source.clone(), params)
}

/// Stream LL (E11): where a derivation's `Params` come from.
///
/// `b65 compose` derives them from the witness, as it always has. `b65
/// expect-root` may instead be handed `--params FILE` -- a `Fixed` or a bare
/// `Params`, `plan_params`' own two shapes -- and then the plan is built from
/// *the statement's* declared shape rather than from a re-derivation of it.
/// That is the shape the check wants: the verifier holds the statement, and
/// four of the fields (`ao_jo_min` / `ao_jo_max` and `ao_acc_lo_bits` /
/// `ao_acc_hi_bits`) are read off activations a verifier does not have, so
/// `--weights-only` cannot derive them at all and requires the file.
///
/// Gated on `PRE_ONLY` so that `compose`'s own behaviour is untouched: nothing
/// but `expect-root` reads `--params` here.
fn expect_root_params(l: &layer::Layer, args: &[String], vname: &str) -> compose::Params {
    if !compose::PRE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        assert!(!flag(args, "--weights-only"),
                "--weights-only is `expect-root`'s: it loads no activations, so \
                 there is nothing to prove");
        return params_of(l, args, vname);
    }
    match arg_str(args, "--params") {
        Some(_) => {
            let (src, p) = plan_params(args);
            eprintln!("expect-root: params from the statement ({src})");
            p
        }
        None => {
            assert!(!flag(args, "--weights-only"),
                    "--weights-only needs --params FILE (the chunk's own \
                     `Fixed.params`): `ao_jo_min`/`ao_jo_max` and \
                     `ao_acc_lo_bits`/`ao_acc_hi_bits` are read off activations \
                     this load does not have");
            params_of(l, args, vname)
        }
    }
}

pub fn params_of(l: &layer::Layer, args: &[String], vname: &str) -> compose::Params {
    let b0 = &l.bands[0];
    let kmax = l.mats.iter().map(|m| m.kmax_blind()).max().unwrap()
        .min(bf16::KMAX_SOUND);
    // B10 step 3: `bands` names the *prefix* bands only.  The corner rides as one
    // more band per prefix band, with the same `insts` and head offset, and the
    // plan rebuilds them from the flag rather than from the list.
    let n_pre = l.bands.iter().filter(|b| !b.corner).count();
    // Stream AA (E11): with `--kv-bind` the declared prefix is `l.nsub` bands of
    // `S` keys per head band group, so the chunk's context is the sub-band's
    // times `nsub` and the head band list takes every `nsub`-th entry.  Both are
    // the identity when `nsub == 1`, which is every run without the flag.
    // Block 0 (open item 1) has no prefix band at all: `nsub == 0`, `c == 0`,
    // and the head band list is the corner bands', one per group.
    let (c, bands) = if l.nsub == 0 {
        (0, l.band_sizes[n_pre..].to_vec())
    } else {
        (b0.shape.c * l.nsub, l.band_sizes[..n_pre].iter().step_by(l.nsub).copied().collect())
    };
    // Stream JJ (E11): data-derived and overridable exactly as `ao_rem_bits`
    // is, and for the same reason -- a driver pins the model-wide worst case
    // and proves every layer as one shape.  Below the witness's own value an
    // `EPOW` lookup or an accumulator limb would land off its table, so that is
    // an error and not a smaller proof.
    //
    // Stream TT (E11), log row 73: these four come out of the struct literal
    // because the two limb widths now depend on the exponent window -- which
    // form of the magnitude bound fits is a question about `e_hi`.
    let ao_jv_min = {
        let want = arg_scalar(args, "--ao-jv-min", l.ao.jv_range().0);
        assert!(want <= l.ao.jv_range().0,
                "--ao-jv-min {want} is above this witness's {}", l.ao.jv_range().0);
        want
    };
    let ao_jv_max = {
        let want = arg_scalar(args, "--ao-jv-max", l.ao.jv_range().1);
        assert!(want >= l.ao.jv_range().1,
                "--ao-jv-max {want} is below this witness's {}", l.ao.jv_range().1);
        want
    };
    let ao_jo_min = {
        let want = arg_scalar(args, "--ao-jo-min", l.ao.jo_range().0);
        assert!(want <= l.ao.jo_range().0,
                "--ao-jo-min {want} is above this witness's {}", l.ao.jo_range().0);
        want
    };
    let ao_jo_max = {
        let want = arg_scalar(args, "--ao-jo-max", l.ao.jo_range().1);
        assert!(want >= l.ao.jo_range().1,
                "--ao-jo-max {want} is below this witness's {}", l.ao.jo_range().1);
        want
    };
    // Stream JJ: `|acc| <= 2^(lo + hi - 1)` under the offset form, so the two
    // limbs together carry one bit more than the accumulator needs.  16 is the
    // widest range table the layer already builds, which is why the low limb
    // stops there; neither limb ever goes below `MIN_LOG_ROWS`, since a range
    // table *is* its own `1 << bits` rows and the LogUp builder has no shorter
    // column.
    //
    // Stream TT (E11), log row 73: and the magnitude form is the same split one
    // bit narrower.  There the limbs range-check the committed `mag = |acc|`
    // rather than `acc + 2^(lo+hi-1)`, so they need `acc_bits()` and not
    // `acc_bits() + 1` -- the sign is read off the `sg` column the region
    // already commits.  The wider form is kept wherever it fits, so that every
    // chunk that composed before this existed keeps its columns and its digest;
    // the narrow one is taken only when `Plan::new`'s window would refuse.
    // `--ao-acc-mag` forces it, which is what the controls and the gate use.
    let e_hi = (ao_jv_max - ao_jo_min).max(0) as u32;
    let split = |need: u32| -> (u32, u32) {
        let lo = arg_scalar(args, "--ao-acc-lo-bits",
                            need.clamp(crate::common::MIN_LOG_ROWS, 16));
        let hi = arg_scalar(args, "--ao-acc-hi-bits",
                            need.saturating_sub(lo).max(crate::common::MIN_LOG_ROWS));
        (lo, hi)
    };
    let need_off = l.ao.acc_bits() + 1;
    let (lo_off, hi_off) = split(need_off);
    let (lo_mag, hi_mag) = split(l.ao.acc_bits());
    let ao_acc_mag =
        flag(args, "--ao-acc-mag") || 1 + lo_off + hi_off + e_hi > 31;
    let (ao_acc_lo_bits, ao_acc_hi_bits, need) = if ao_acc_mag {
        (lo_mag, hi_mag, l.ao.acc_bits())
    } else {
        (lo_off, hi_off, need_off)
    };
    assert!(ao_acc_lo_bits + ao_acc_hi_bits >= need,
            "--ao-acc-*-bits {ao_acc_lo_bits}+{ao_acc_hi_bits} is below this witness's {need}");
    assert!(ao_acc_lo_bits >= crate::common::MIN_LOG_ROWS
                && ao_acc_hi_bits >= crate::common::MIN_LOG_ROWS,
            "--ao-acc-*-bits {ao_acc_lo_bits}+{ao_acc_hi_bits}: a limb is below \
             MIN_LOG_ROWS");
    if ao_acc_mag && !flag(args, "--ao-acc-mag") {
        eprintln!("compose: the attention output takes the magnitude form -- the \
                   offset window is 1 + {lo_off} + {hi_off} + {e_hi} > 31 (log row 73)");
    }
    // Stream TT (E11), log row 73: the sibling bound, decided the same way.
    // `ao_q_bits` is `attn_out_bits - 1` and `ao_rem_bits` is the witness's own
    // (a driver may only widen it), so `11 + 20 = 31` is not a pin anything can
    // choose its way out of either -- the `denp2` form is what admits it.
    let ao_q_bits = l.attn_out_bits - 1;
    let ao_rem_bits = {
        let want = arg_scalar(args, "--ao-rem-bits", l.ao.rem_bits());
        assert!(want >= l.ao.rem_bits(),
                "--ao-rem-bits {want} is below this witness's {}", l.ao.rem_bits());
        want
    };
    let ao_denp2 = flag(args, "--ao-denp2") || ao_q_bits + ao_rem_bits > 30;
    if ao_denp2 && !flag(args, "--ao-denp2") {
        eprintln!("compose: the attention output takes the denp2 form -- the quotient \
                   term is {ao_q_bits} + {ao_rem_bits} > 30 (log row 73)");
    }
    compose::Params {
        s: l.ao.s, c, hd: l.ao.hd, nh: l.ao.nh,
        bands,
        corner: l.bands.len() > n_pre,
        // Stream E: the head's order-key claim.  Not under `--no-glue`'s gate,
        // because a head dump has no wiring columns to switch off.
        // `--head-argmax` implies the key claim: it is the counterparty
        head_key: flag(args, "--head-key") || flag(args, "--head-argmax"),
        head_argmax: flag(args, "--head-argmax"),
        head_argmax_lax: flag(args, "--head-argmax-lax"),
        variant: vname.to_string(),
        log_table: b0.log_table,
        q_bits: b0.q_bits, k_bits: b0.k_bits, nmax: b0.nmax,
        chan_target: arg_scalar(args, "--keys-per-row", 128usize),
        grid_target: arg_scalar(args, "--scores-per-row", 64usize),
        rw: arg_scalar(args, "--tokens-per-row", 16usize),
        z_lo_bits: arg_scalar(args, "--z-lo-bits", 16u32),
        z_hi_bits: arg_scalar(args, "--z-hi-bits", 6u32),
        z_min: b0.nmax,
        mats: l.mats.iter().map(|m| compose::MatParams {
            name: m.name.clone(), variant: m.variant.name().into(),
            n_in: m.n_in, n_out: m.n_out, tokens: m.tokens,
        }).collect(),
        deq_target: arg_scalar(args, "--deq-per-row", 16usize),
        kmax: arg_scalar(args, "--kmax", kmax),
        round: arg_strs(args, "--round", &["even"])[0].clone(),
        ao_target: arg_scalar(args, "--ao-per-row", 16usize),
        // Stream Z: the witness is the knob.  A dump extracted at a wider
        // attention output declares it in the manifest, and a dump that
        // predates the knob reads back 8 -- so every pre-change statement is
        // parameter-identical and the standing digests stand.
        ao_q_bits,
        // B9 step 1: `ao_rem_bits` and `kmax` are data-derived, so two real layers
        // of one model are two shapes and the chunk loop rejects the pair.  Both
        // are overridable now, so a driver can pin them at the model-wide worst
        // case and prove every layer of a model as one shape -- a few extra table
        // rows against a second emit chain.  Below the witness's own value the
        // range check would just push rows into `off_table`, so that is an error.
        // (Both are derived above, with the `denp2` decision that reads them.)
        ao_rem_bits,
        ao_jv_min, ao_jv_max, ao_jo_min, ao_jo_max,
        ao_acc_lo_bits, ao_acc_hi_bits, ao_acc_mag, ao_denp2,
        // Stream Z: the fold's two pins, data-derived exactly as `ao_rem_bits`
        // and overridable the same way so a driver can pin the model-wide worst
        // case.  Zero -- absent from the serialized statement -- on a witness
        // with no fold matmul.
        fold_rem_bits: match l.fold_pins() {
            Some((_, rem)) => {
                let want = arg_scalar(args, "--fold-rem-bits", rem);
                assert!(want >= rem,
                        "--fold-rem-bits {want} is below this witness's {rem}");
                want
            }
            None => 0,
        },
        fold_alpha: match l.fold_pins() {
            Some((alpha, _)) => arg_scalar(args, "--fold-alpha", alpha),
            None => 0,
        },
        walk_p: arg_scalar(args, "--p", 136usize),
        // B10 step 2: the wiring family rides on a witness `extract.py --glue`
        // wrote, and on nothing else -- a dump without those columns composes
        // exactly as it did before.  `--glue-a` is how many rows one token's
        // stream takes at the reduction's first level; it has to divide `d`.
        // Stream E's two regions are opt-in: `--norm-tails` wires `h = (x r) w`
        // and `--swiglu` the `silu(gate) * up` product.  Off, the composition is
        // the pre-change one cell for cell, which is what the bit-for-bit control
        // checks.
        glue: l.glue.as_ref().filter(|_| !flag(args, "--no-glue")).map(|g| {
            compose::GlueParams {
                d: g.d,
                a: arg_scalar(args, "--glue-a", 32usize),
                // carried only when the region that needs it is on, so that
                // with both flags off the serialized statement is not merely
                // equivalent to the pre-change crate's but identical to it
                // Stream II: the two additive regions are unconditional now, so
                // `dff` is always carried and `--norm-tails` / `--swiglu` are
                // accepted and ignored -- the drivers still pass them.
                dff: g.dff,
                eps: g.eps,
                inv_d: g.inv_d,
                // Stream JJ (E11): the dequant output ties are unconditional
                // too.  `--deq-ties` is accepted and ignored so drivers that
                // pass it keep working.
                deq_ties: true,
                // stream II: which family this chunk carries is the witness's
                // to say, not the command line's
                head_norm: g.head,
            }
        }),
        // Stream H: the boundary commitment, off unless asked for either way --
        // by the flag on the command line or by `B10_BOUNDS` in the environment.
        // Off is what makes the recorded digest the bit-for-bit line.
        bounds: flag(args, "--bounds") || bounds::enabled(),
        // Stream AA (E11): the K/V boundary trees, and the prefix tiling they
        // need.  Opt-in and skipped in the serialization, so a statement
        // written without it is byte identical to one from a crate that never
        // had the feature.
        kv_bind: flag(args, "--kv-bind"),
        // B11 stream Q: opt in to the LogUp-GKR arm for the bands. A flag and
        // not an env mode, because it changes the proof.
        gkr_bands: flag(args, "--gkr-bands"),
        gkr_dequant: flag(args, "--gkr-dequant"),
        gkr_glue: flag(args, "--gkr-glue"),
        gkr_walks: flag(args, "--gkr-walks"),
        gkr_attnout: flag(args, "--gkr-attnout"),
        // Stream BB (E11): the embedding gadget.  `--embed` needs the wiring
        // columns (its tie's counterparty is their `stream_in` claim) and the
        // dump's `embed.u16` / `tok_in.i32`; it panics on a dump without them
        // rather than standing anything in.  Zero -- absent from the serialized
        // statement -- everywhere else.
        embed_vocab: if flag(args, "--embed") {
            let g = l.glue.as_ref()
                .expect("--embed needs a witness extracted with --glue");
            let tab = l.embed_tab.as_ref()
                .expect("--embed needs embed.u16 in the dump");
            assert!(l.tok_in.is_some(), "--embed needs tok_in.i32 in the dump");
            assert_eq!(tab.len() % g.d, 0,
                       "embed.u16 holds {} values, which is not a multiple of d = {}",
                       tab.len(), g.d);
            tab.len() / g.d
        } else {
            0
        },
        embed_per_row: if flag(args, "--embed") {
            arg_scalar(args, "--embed-per-row", 256usize)
        } else {
            0
        },
        // Stream NN: the rotation and the q/k/v quantize, present exactly when
        // the witness carries what they need.  Not a flag: on a real dump the
        // columns are there or they are not, and on a synthetic layer
        // `--synth-glue` builds them exactly when the widths line up.  The
        // wiring family is a precondition -- the input side ties into it.
        rope: l.glue.as_ref().filter(|g| !g.head).and(l.rope.as_ref()).map(|r| {
            compose::RopeParams {
                nkv: r.nkv,
                pos0: r.pos0,
                theta: r.theta.to_bits(),
                rope_type: r.rope_type.clone(),
                rope_scale: r.rope_scale.iter().map(|v| v.to_bits()).collect(),
            }
        }),
    }
}

/// Run the whole serialized round trip: prove, write the statement and the
/// proof as bytes, verify from the bytes alone.
///
/// Stream I: `ser_to` is the sequence assembly's hook. The chain check that
/// joins two chunks reads four byte strings off disk and nothing else, so a
/// chunk that is going to be chained has to leave its statement and its proof
/// behind as files. Passing `None` is the old behaviour exactly -- the bytes
/// are still built, still verified from, and then dropped.
fn compose_once(l: &layer::Layer, params: &compose::Params, tamper: &compose::ProverTamper,
                ser_to: Option<(&str, &str)>) -> (serde_json::Value, bool) {
    let out = compose::prove_layer(l, params, tamper);
    let ser = compose_ser(&out);
    if let Some((dir, tag)) = ser_to {
        ser_write(dir, tag, &ser);
    }
    compose_finish(out, ser, params)
}

/// Leave one chunk's statement and proof on disk under `tag`.
///
/// Stream O: the chunk loop has to write exactly these two files, per chunk
/// and under the chunk's own tag, because `bounds-chain` joins two chunks by
/// reading them back off disk and nothing else. It was a branch inside
/// `compose_once` and is a function so that the loop can call it too.
pub fn ser_write(dir: &str, tag: &str, ser: &Ser) {
    std::fs::create_dir_all(dir).expect("the serialization directory");
    std::fs::write(format!("{dir}/{tag}.fixed.json"), &ser.fixed_bytes)
        .expect("writing the statement");
    // Stream KK: the claims, which `verify_composed` needs and which nothing
    // wrote before -- so `bounds-grid --verify` could not have verified a
    // serialized chunk however hard it tried.  They are small (tens of KB
    // beside a 50 MB proof) and they are a third of the document the verifier
    // reads, so they are written beside the other two.
    std::fs::write(format!("{dir}/{tag}.claims.json"), &ser.claims_bytes)
        .expect("writing the claims");
    if let Some(b) = &ser.proof_bytes {
        std::fs::write(format!("{dir}/{tag}.proof.json"), b).expect("writing the proof");
    }
    eprintln!("compose: statement, claims and proof written to \
               {dir}/{tag}.{{fixed,claims,proof}}.json");
}

/// The statement and the proof as bytes.
pub struct Ser {
    pub claims_bytes: Vec<u8>,
    pub fixed_bytes: Vec<u8>,
    pub proof_bytes: Option<Vec<u8>>,
}

/// B7a milestone 4 (18.2): section 5 puts serialization inside the whole wall
/// and no harness has ever timed it. B7b (26): on its own, so the chunk loop
/// serializes inside the wall and verifies after it.
pub fn compose_ser(out: &compose::Composed) -> Ser {
    let m4_t = std::time::Instant::now();
    let claims_bytes = serde_json::to_vec(&out.claims).unwrap();
    let fixed_bytes = serde_json::to_vec(&out.fixed).unwrap();
    crate::util::m4_add("wall/ser_statement", m4_t.elapsed().as_secs_f64());
    let proof_bytes = out.proof.as_ref().map(|pf| {
        let m4_t = std::time::Instant::now();
        let b = serde_json::to_vec(pf).unwrap();
        crate::util::m4_add("wall/ser_proof", m4_t.elapsed().as_secs_f64());
        b
    });
    Ser { claims_bytes, fixed_bytes, proof_bytes }
}

/// Verify from the bytes alone and assemble the report.
pub fn compose_finish(out: compose::Composed, ser: Ser, params: &compose::Params)
                      -> (serde_json::Value, bool) {
    let Ser { claims_bytes, fixed_bytes, proof_bytes } = ser;
    let mut proof_sha256 = String::new();
    let mut proof_ser_bytes = 0usize;
    let (verdict, proof_bytes_len, verify_json) = match (&out.proof, proof_bytes) {
        (Some(pf), Some(proof_bytes)) => {
            // section 5 rung 1: the GPU proof must be byte-identical to the
            // SIMD proof built with the same options. This is the digest.
            proof_sha256 = crate::device::sha256_hex(&proof_bytes);
            proof_ser_bytes = proof_bytes.len();
            let v = compose::verify_composed(&fixed_bytes, &claims_bytes, &proof_bytes);
            let j = serde_json::to_value(&v).unwrap();
            (Some(v), pf.size_estimate(), j)
        }
        _ => (None, 0, json!({"prove_error": out.prove_error})),
    };
    let accepts = verdict.as_ref().map(|v| v.accepts()).unwrap_or(false);
    let j = json!({
        "accepts": accepts,
        "verdict": verify_json,
        "prove_error": out.prove_error,
        "proof_bytes": proof_bytes_len,
        "statement_bytes": {"fixed": fixed_bytes.len(), "claims": claims_bytes.len()},
        "pre_root": out.fixed.pre_root,
        // stream PP: the per-layer model commitment this chunk proves against
        "model_root": out.fixed.model_root,
        "columns": {"preprocessed": out.n_columns[0], "base": out.n_columns[1],
                    "interaction": out.n_columns[2],
                    "total": out.n_columns.iter().sum::<usize>()},
        // stream PP: `preprocessed` is the shape tree and the model tree
        // together, so the number stays comparable with the pre-split records.
        "cells": {"preprocessed": out.stats.pre_cells(),
                  "shape": out.stats.tree_cells.first().copied().unwrap_or(0),
                  "model": out.stats.model_cells(),
                  "base": out.stats.tree_cells.get(crate::common::BASE_TREE)
                      .copied().unwrap_or(0),
                  "interaction": out.stats.tree_cells.get(crate::common::INTER_TREE)
                      .copied().unwrap_or(0),
                  "fresh": out.stats.fresh_cells()},
        "family_cells": out.family_cells,
        "ref_mismatch": out.ref_bad, "off_table": out.off_table,
        "glue_ties_ok": out.glue_ties_ok,
        // Stream H: the boundary trees' own cells and columns, and the two roots
        // the chain check compares.  Null on a chunk proved with the flag off.
        "bounds": out.bounds,
        // Stream AA: the same for the K/V boundary trees, and every boundary
        // root in the prover's order (stream H's two, then the K/V sides).
        "kv_bounds": out.kv_bounds,
        // Stream EE: the residual pair is found by its label, not by tree index,
        // so this JSON keeps H's shape while the statement may carry any number
        // of sides; `boundary_sides` is every side with its label.
        "bound_roots": verdict.as_ref().map(|v| json!({
            "in": v.side_root(bounds::Axis::Layer, "in", None),
            "out": v.side_root(bounds::Axis::Layer, "out", None)})),
        "boundary_roots": verdict.as_ref().map(|v| v.bound_roots.clone()),
        "boundary_sides": verdict.as_ref().map(|v| serde_json::to_value(&v.bound_sides).unwrap()),
        "trace_secs": out.trace_secs, "interaction_secs": out.interaction_secs,
        "prove_secs": out.stats.prove_secs,
        "verify_secs": verdict.as_ref().map(|v| v.verify_secs).unwrap_or(0.0),
        "peak_rss_bytes": out.stats.peak_rss,
        "preprocessed_is_widest": out.stats.pre_ok,
        "phases": out.stats.phases,
        "host_phases": crate::util::m4_report(),
        // B11 stream F: the bytes the named phases moved, so a phase's GB/s is
        // read off the same dump as its seconds
        "host_bytes": crate::util::m4_bytes_report(),
        "proof_sha256": proof_sha256,
        "proof_ser_bytes": proof_ser_bytes,
        // milestone 2a: what the host->device crossing cost (README 10.2)
        "upload": crate::device::report(),
        // milestone 3: what the constraint lowering cost, and where
        "blob": crate::cudagen::blob_report(),
        "params": serde_json::to_value(params).unwrap(),
    });
    (j, accepts)
}

/// Stream LL (E11), `notes/witness-slack.md` section 5 item 1, deliverable 1:
/// what tree 0 is made of, from `Params` alone.
///
///     b65 pre-split --params FILE
///
/// `FILE` is a JSON `Params` -- `result.point.params` of any `compose` run, so
/// the 8B reference chunk's tree 0 can be priced without the dump. The rollup
/// answers the two questions the design fork turns on: how much of tree 0 a
/// verifier could rebuild from the statement alone (`needs_params`), and how
/// much of it is the same on every token block of the layer (`varies_layer`),
/// which is what a published per-layer root would cover.
fn pre_split_cmd(args: &[String]) {
    let f = arg_str(args, "--params").expect("pre-split needs --params FILE");
    let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("--params {f}: {e}"));
    let p: compose::Params = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("--params {f}: not a Params document: {e}"));
    let plan = compose::Plan::new(&p);
    let mut report = plan.pre_split_report();
    report["params_file"] = json!(f);
    report["c"] = json!(p.c);
    report["s"] = json!(p.s);
    let cols: Vec<Value> = plan.pre_split().iter()
        .map(|c| json!({"id": c.id.id, "family": c.family, "log_size": c.log_size,
                        "needs": c.needs, "varies": c.varies}))
        .collect();
    report["column_list"] = json!(cols);
    eprintln!("pre-split: {} columns, {} cells; needs_model {:.2}%, varies_block {:.2}%",
              report["columns"], report["cells"],
              100.0 * report["needs_model"]["share"].as_f64().unwrap(),
              100.0 * report["varies_block"]["share"].as_f64().unwrap());
    emit("pre_split", report);
}

/// Stream LL (E11), `notes/witness-slack.md` section 5 item 1, deliverable 2:
/// the expected tree-0 root of one chunk, derived from the model.
///
///     b65 expect-root --witness DIR [--layer k] [--block b] <the chunk's flags>
///     b65 expect-root --weights-only --params <tag>.fixed.json --witness DIR ...
///
/// It takes exactly the arguments the chunk's `compose` line takes -- the plan
/// line, minus `--ser-tag` -- builds the chunk's `Plan` and its tree 0, and
/// stops there: the root, the model hash and the classification, and no proof.
/// The derived root is the prover's own by construction, because it is the
/// prover's own code up to `commit_pre` and there is no second implementation
/// of tree 0 to drift from it.
///
/// **`--weights-only` is the verifier's form.** Without it `Layer::load` reads
/// the whole dump, activations included, and recomputes `Q.K^T` on the way; tree
/// 0 needs only the model half (the weights, the weight scales, `jv`, the norm
/// weights, the embedding table, this stream's rope exponents, and the
/// manifest's `frac_per_head` / `master_table_bits`).  With it,
/// `Layer::load_weights` reads exactly those files and nothing else, and the
/// derived root must be the same -- which is what `ll_box.sh`'s `root` step
/// checks on a real dump, three ways: the full load's, the weights-only load's
/// and the recorded proof's own `pre_root`.
///
/// `--weights-only` requires `--params FILE`, the chunk's own `Fixed.params`:
/// `ao_jo_min`/`ao_jo_max` and `ao_acc_lo_bits`/`ao_acc_hi_bits` are read off
/// activations, so they cannot be re-derived here -- and taking them from the
/// statement is what the check wants anyway, since it is the statement's
/// declared shape a verifier holds.
///
/// **Stream PP (E11): `--shape-only` is the committed-model verifier's form.**
/// It stops one tree earlier -- after the shape tree and before the model tree
/// -- and emits `pre_root` alone.  The shape tree is a function of
/// `Fixed.params` and of the per-layer calibration the model manifest
/// publishes, so the root it derives is one no weight can move: the desk gate
/// derives it at two seeds, and the two agree while the model roots differ.
/// On a real dump the loader is still `--weights-only`'s, so the *files* are
/// still opened; a `Layer::from_params` that opens none is the box half's, and
/// this brief's Record names it.
fn expect_root_cmd(args: &[String]) {
    if flag(args, "--shape-only") {
        compose::SHAPE_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    compose::PRE_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    compose_cmd(args);
    unreachable!("expect-root: the prover did not stop after the preprocessed trees");
}

/// Stream LL (E11), deliverable 2: the model manifest KK's
/// `bounds-grid --expect-pre-roots` consumes.
///
///     b65 model-roots --plan PLAN [--pins "..."] [--out FILE] [--limit N]
///                     [--weights-only --params-dir DIR]
///
/// `PLAN` is an `o_plan.py` plan file: one `compose` argument line per chunk.
/// A plan line carries only the chunk's *own* arguments, so `--pins` supplies
/// the driver's shared prefix (`--round`, `--numbits`, `--variants`, `--bands`,
/// `--kmax` and the fold's two -- `ss_box.sh`'s `SEQPT`).
///
/// `--weights-only --params-dir DIR` is the verifier's form: each row is derived
/// from the model half of the dump and from `DIR/<ser-tag>.fixed.json`, the
/// statement `compose --ser-dir` wrote.  Without it the rows are derived the
/// prover's way, from the whole dump, which is what a manifest built before any
/// chunk has been proved has to do.
///
/// Three things it checks and one it reports: two chunks at one `(layer, c)`
/// must derive the same root and name the same model; one layer must name one
/// model; and **`master_table_bits` must be one value per layer across blocks**,
/// or the layer is a named rejection and no manifest is written (the desk
/// record's open item 3 -- `table/exp` is `1 << log_table` rows and `log_table`
/// comes off each extraction's own manifest, so this is a requirement on the
/// plan that nothing else states).
/// Each line is run through `expect-root` in its own process -- one chunk's
/// tree 0 at a time, which is what keeps the peak resident set to one chunk's
/// -- and the rows are collected into
///
///     [{"layer": k, "block": b, "c": n, "pre_root": "...", "model": "..."}]
///
/// The rows a grid needs are keyed by `(layer, c)`, which is what
/// `grid_cmd` looks them up by; `block` and `model` ride along so a reader can
/// see which dump each row came from and so the grid can compare model hashes.
fn model_roots_cmd(args: &[String]) {
    let plan = arg_str(args, "--plan").expect("model-roots needs --plan FILE");
    let text = std::fs::read_to_string(&plan)
        .unwrap_or_else(|e| panic!("--plan {plan}: {e}"));
    let limit = arg_scalar::<usize>(args, "--limit", usize::MAX);
    // Stream LL box half.  A plan file holds a chunk's *own* arguments; the
    // shared pins (`--round`, `--numbits`, `--variants`, `--bands`, `--kmax`,
    // the fold's two) live in the driver that runs it (`ss_box.sh`'s `SEQPT`),
    // so a plan line alone does not compose.  `--pins` is that prefix.
    let pins: Vec<String> = arg_str(args, "--pins").unwrap_or_default()
        .split_whitespace().map(|s| s.to_string()).collect();
    // The verifier's mode: derive each row from the model half of the dump and
    // the *statement's* own params, `<tag>.fixed.json` in `--params-dir` (what
    // `compose --ser-dir` writes).  Without it the rows are derived the
    // prover's way, from the whole dump, which is what a manifest built ahead
    // of any proof has to do.
    let weights_only = flag(args, "--weights-only");
    let params_dir = arg_str(args, "--params-dir");
    assert!(!weights_only || params_dir.is_some(),
            "model-roots --weights-only needs --params-dir DIR (the serialized \
             statements: DIR/<ser-tag>.fixed.json)");
    let exe = std::env::current_exe().expect("model-roots: no path to this binary");
    let mut rows: Vec<Value> = vec![];
    let mut seen: std::collections::BTreeSet<(i64, u64)> = Default::default();
    let mut skipped = 0usize;
    let mut missing: Vec<String> = vec![];
    let t0 = std::time::Instant::now();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if rows.len() >= limit {
            break;
        }
        // The plan line's own arguments **first**, then the shared pins, minus
        // the two that name the artifact rather than the chunk.  The order is
        // load-bearing: `arg_str` and `arg_scalar` take the *first* match, so a
        // line's own `--table-bits` or `--ao-rem-bits` has to precede the
        // shared one or the shared one silently wins (stream SS's rider 1,
        // `o_plan.py --ao-rem-per-line`).
        let mut a: Vec<String> = vec![];
        let mut tag = String::new();
        let mut it = line.split_whitespace();
        while let Some(w) = it.next() {
            if w == "--ser-dir" {
                it.next();
                continue;
            }
            if w == "--ser-tag" {
                tag = it.next().unwrap_or("").to_string();
                continue;
            }
            a.push(w.to_string());
        }
        a.extend(pins.iter().cloned());
        if weights_only {
            let d = params_dir.as_ref().unwrap();
            let f = format!("{d}/{tag}.fixed.json");
            if !std::path::Path::new(&f).exists() {
                missing.push(tag.clone());
                continue;
            }
            a.push("--weights-only".into());
            a.push("--params".into());
            a.push(f);
        }
        let out = std::process::Command::new(&exe)
            .arg("expect-root")
            .args(&a)
            .output()
            .unwrap_or_else(|e| panic!("model-roots: cannot run {exe:?}: {e}"));
        if !out.status.success() {
            eprintln!("model-roots: {line}\n{}", String::from_utf8_lossy(&out.stderr));
            panic!("model-roots: expect-root failed on a plan line");
        }
        let doc: Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("model-roots: expect-root wrote no JSON: {e}"));
        let r = &doc["result"];
        let layer = r["layer"].as_i64().unwrap_or(-1);
        let c = r["c"].as_u64().unwrap_or(0);
        // one row per (layer, c): every chunk of a grid at that cell has the
        // same tree 0, and a second run of the same cell must agree with the
        // first or the derivation is not a function of what it claims to be
        if let Some(prev) = rows.iter().find(|p| {
            p["layer"].as_i64() == Some(layer) && p["c"].as_u64() == Some(c)
        }) {
            assert_eq!(prev["pre_root"], r["pre_root"],
                       "model-roots: two chunks at (layer {layer}, c {c}) derive \
                        different shape roots");
            assert_eq!(prev["model_root"], r["model_root"],
                       "model-roots: two chunks at (layer {layer}, c {c}) derive \
                        different model roots");
            assert_eq!(prev["model"], r["model"],
                       "model-roots: two chunks at (layer {layer}, c {c}) name \
                        different models");
            skipped += 1;
            continue;
        }
        seen.insert((layer, c));
        rows.push(json!({
            "layer": layer,
            "block": r["block"],
            "c": c,
            "pre_root": r["pre_root"],
            // stream PP: the per-layer model commitment.  The shape root is
            // keyed by `(layer, c)` because the shape moves with the chunk;
            // this one is keyed by the layer alone, and the rejection below is
            // what says so.
            "model_root": r["model_root"],
            "model": r["model"],
            "source": r["source"],
            "tag": tag,
            "log_table": r["log_table"],
            "weights_only": r["weights_only"],
        }));
    }
    // the model hash is a per-layer constant: two rows at one layer that name
    // two models is exactly the case section 5 item 1's last sentence names
    let mut by_layer: std::collections::BTreeMap<i64, &Value> = Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        if let Some(prev) = by_layer.get(&l) {
            assert_eq!(prev["model"], r["model"],
                       "model-roots: layer {l} appears with two different model hashes");
        } else {
            by_layer.insert(l, r);
        }
    }
    // Stream LL box half, the desk record's open item 3: `master_table_bits` has
    // to be pinned per layer across blocks, or the model part of tree 0 is not a
    // per-layer constant -- `table/exp` is `1 << log_table` rows and `log_table`
    // comes off *each extraction's own manifest*, so two blocks of one layer can
    // legitimately carry two table sizes and neither is wrong.  A requirement on
    // the plan, checked here rather than assumed: a layer with two sizes is a
    // named rejection and no manifest is written, because a manifest whose rows
    // disagree about the layer's table is a manifest that pins nothing.
    let mut tb: std::collections::BTreeMap<i64, std::collections::BTreeMap<u64, Vec<String>>> =
        Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        let n = r["log_table"].as_u64().unwrap_or(0);
        tb.entry(l).or_default().entry(n).or_default()
            .push(r["tag"].as_str().unwrap_or("").to_string());
    }
    let mut rejections: Vec<Value> = tb.iter().filter(|(_, m)| m.len() > 1)
        .map(|(l, m)| json!({
            "layer": l,
            "why": "master_table_bits is not pinned across this layer's blocks",
            "table_bits": m.iter().map(|(n, tags)| json!({"log_table": n, "chunks": tags}))
                .collect::<Vec<Value>>(),
        }))
        .collect();
    // Stream PP (E11): the model root is the *published commitment* for a
    // layer, so it has to be one value across that layer's blocks.  The
    // classification is what should make that true -- every model-tree column
    // is `varies: Layer`, asserted in `Plan::pre_ids_split` -- and this is the
    // check that says it held on a real plan.  A layer with two model roots is
    // a named rejection and no manifest is written: a manifest whose rows
    // disagree about a layer's model commits nothing.
    let mut mr: std::collections::BTreeMap<i64, std::collections::BTreeMap<String, Vec<String>>> =
        Default::default();
    for r in &rows {
        let l = r["layer"].as_i64().unwrap_or(-1);
        let root = r["model_root"].as_str().unwrap_or("").to_string();
        mr.entry(l).or_default().entry(root).or_default()
            .push(r["tag"].as_str().unwrap_or("").to_string());
    }
    rejections.extend(mr.iter().filter(|(_, m)| m.len() > 1).map(|(l, m)| json!({
        "layer": l,
        "why": "the model root is not one value across this layer's blocks",
        "model_roots": m.iter().map(|(root, tags)| json!({"model_root": root, "chunks": tags}))
            .collect::<Vec<Value>>(),
    })));
    let body = json!({
        "plan": plan,
        "rows": rows,
        "distinct_cells": rows.len(),
        "repeat_cells": skipped,
        "layers": by_layer.len(),
        "models": by_layer.values()
            .map(|r| r["model"].as_str().unwrap_or("").to_string())
            .collect::<std::collections::BTreeSet<String>>().len(),
        "weights_only": weights_only,
        "no_statement": missing,
        "table_bits_by_layer": tb.iter()
            .map(|(l, m)| json!({"layer": l,
                                 "log_table": m.keys().collect::<Vec<&u64>>()}))
            .collect::<Vec<Value>>(),
        // stream PP: the published model commitment, one per layer
        "model_roots_by_layer": mr.iter()
            .map(|(l, m)| json!({"layer": l,
                                 "model_root": m.keys().collect::<Vec<&String>>()}))
            .collect::<Vec<Value>>(),
        "rejections": rejections,
        "secs": t0.elapsed().as_secs_f64(),
    });
    if !rejections.is_empty() {
        for r in &rejections {
            eprintln!("model-roots: REJECT layer {}: {}",
                      r["layer"], r["why"].as_str().unwrap_or(""));
        }
        emit("model_roots", body);
        eprintln!("model-roots: {} layer(s) rejected; no manifest written",
                  rejections.len());
        std::process::exit(1);
    }
    if let Some(f) = arg_str(args, "--out") {
        std::fs::write(&f, serde_json::to_string_pretty(&body["rows"]).unwrap())
            .unwrap_or_else(|e| panic!("--out {f}: {e}"));
        eprintln!("model-roots: {} rows -> {f}", rows.len());
    }
    emit("model_roots", body);
}

fn compose_cmd(args: &[String]) {
    // Stream EE (open item 21): before anything, because `prove_gen` reads it.
    cudagen::fp_out_set(args);
    let (mut l, vname) = layer_of(args);
    // B7a milestone 4 (section 3 row 1): rung 3's sixth probe on its own, so the
    // `nopatch` control has a witness whose exp table is *not* the closed form
    // to bite on (section 5 rules 5 and 8). The proof still accepts here -- the
    // table and the witness agree; it is the pinned preprocessed root, which
    // only `compose-probes` checks, that rejects it.
    // Stream PP (E11), the model-tree control: bend one cell of the first
    // matmul's quantized weight array.  It is a *weight*, so only the model
    // tree moves -- the shape root must not.  `--corrupt-exp` is the mirror
    // control: the exp table is a shape column now, so it moves the shape root
    // and leaves the model root alone.
    if let Some(i) = arg_str(args, "--corrupt-weight") {
        let idx: usize = i.parse().expect("--corrupt-weight wants a weight index");
        let m = l.mats.first_mut().expect("--corrupt-weight: the layer has no matmul");
        assert!(idx < m.qw.len(),
                "--corrupt-weight {idx}: the first matmul has {} weights", m.qw.len());
        m.qw[idx] += 1;
    }
    if let Some(i) = arg_str(args, "--corrupt-exp") {
        l.corrupt_exp_table(i.parse().expect("--corrupt-exp wants a table row"));
        eprintln!("compose: exp table row {i} flipped, witness rebuilt");
    }
    // B7a milestone 4 (section 3 row 1): rung 3's *seventh* probe on its own,
    // so the `nopatch` control has a run whose normalisation table is not the
    // closed form to bite on (section 5 rules 5 and 8). As with `--corrupt-exp`
    // the proof still accepts here -- the table and the witness agree; it is
    // the pinned preprocessed root, which only `compose-probes` checks, that
    // rejects it.
    let mut tamper = compose::ProverTamper::default();
    if let Some(i) = arg_str(args, "--corrupt-norm") {
        tamper.norm_row = Some(i.parse().expect("--corrupt-norm wants a table row"));
        eprintln!("compose: norm table row {i} flipped, witness rebuilt");
    }
    // Stream Z: the wired sites' bend controls.  Each bends one committed cell
    // after the honest derivation; a rejected proof (a non-zero exit through the
    // `accepts` assert below) is the pass.
    for (key, slot) in [("--corrupt-fold-qm", &mut tamper.fold_qm),
                        ("--corrupt-fold-dent", &mut tamper.fold_dent),
                        ("--corrupt-ao-clamp", &mut tamper.ao_clamp),
                        ("--corrupt-ao-den", &mut tamper.ao_den),
                        ("--corrupt-ao-jo", &mut tamper.ao_jo),
                        ("--corrupt-ao-num", &mut tamper.ao_num),
                        ("--corrupt-ao-acc", &mut tamper.ao_acc),
                        ("--corrupt-ao-jo-tok", &mut tamper.ao_jo_tok),
                        // stream TT (log row 73): the magnitude window's own
                        ("--corrupt-ao-mag", &mut tamper.ao_mag),
                        ("--corrupt-walk-act", &mut tamper.walk_over)] {
        if let Some(i) = arg_str(args, key) {
            *slot = Some(i.parse().unwrap_or_else(|_| panic!("{key} wants an index")));
            eprintln!("compose: bend {key} at {i} armed (a rejection is the pass)");
        }
    }
    let params = expect_root_params(&l, args, &vname);
    eprintln!("compose: {} bands={:?} variant={} kmax={} rem={} p={} jv=[{},{}] jo=[{},{}] \
acc={}+{}",
              l.source, l.band_sizes, vname, params.kmax, params.ao_rem_bits, params.walk_p,
              params.ao_jv_min, params.ao_jv_max, params.ao_jo_min, params.ao_jo_max,
              params.ao_acc_lo_bits, params.ao_acc_hi_bits);
    if params.ao_acc_mag {
        eprintln!("compose: attnout accumulator in the magnitude form (log row 73)");
    }
    // Stream I: `--ser-dir DIR [--ser-tag TAG]` leaves this chunk's statement and
    // proof on disk under that tag, which is what `bounds-chain` reads back.
    let ser_dir = arg_str(args, "--ser-dir");
    let ser_tag = arg_str(args, "--ser-tag").unwrap_or_else(|| "chunk".into());
    let (mut j, accepts) = compose_once(&l, &params, &tamper,
                                        ser_dir.as_deref().map(|d| (d, ser_tag.as_str())));
    // B12 stream GG: the `Composed` is dropped and the proof serialized by
    // now, so this is what the process still holds; then every mark of the
    // chunk goes into the point (the cuda arm only -- the reference arm has
    // none and the field stays absent).
    device::mem_mark("after_drop");
    if let Some(m) = device::mem_marks_json() {
        j["mem_marks"] = m;
    }
    assert!(accepts, "honest composed proof rejected: {j}");
    if tamper.norm_row.is_none() {
        assert_eq!(j["ref_mismatch"], 0,
                   "composed witness disagrees with the f64 reference");
    }
    assert_eq!(j["off_table"], 0, "honest composed witness has off-table values");
    emit("compose", json!({"source": l.source, "point": j}));
}

/// Stream J: the master exp table at three sizes on one witness.
///
/// `sized` is what the driver picks today (`d_bits + spread`, or `--table-bits`);
/// `tight` is the largest key the bands actually look up; `tamper` is one bit
/// below that, which leaves keys off the table holding the numerators they already
/// have -- an internally consistent witness that only the lookup can object to.
fn oversize_probes_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let mut points = vec![];
    let mut arm = |name: &str, l: &layer::Layer| -> bool {
        let params = params_of(l, args, &vname);
        let off: usize = l.bands.iter().map(|b| b.off_table).sum();
        eprintln!("oversize probe {name}: table 2^{} off_table {off}", params.log_table);
        let (j, accepts) = compose_once(l, &params, &Default::default(), None);
        points.push(json!({
            "arm": name, "log_table": params.log_table,
            "table_rows": 1usize << params.log_table,
            "witness_off_table": off,
            "accepts": accepts,
            "off_table": j["off_table"],
            "cells": j["cells"], "pre_root": j["pre_root"],
            "prove_error": j["prove_error"],
        }));
        accepts
    };
    assert!(arm("sized", &l), "the honest layer at the driver's size was rejected");
    let need = l.key_bits();

    let mut tight = layer_clone(&l);
    tight.set_master_table(need);
    assert!(arm("tight", &tight), "the honest layer at the tight size was rejected");

    let mut bad = layer_clone(&l);
    bad.set_master_table(need - 1);
    assert!(bad.bands.iter().map(|b| b.off_table).sum::<usize>() > 0,
            "the tamper arm has no off-table key -- nothing to reject");
    assert!(!arm("tamper", &bad),
            "a witness with keys past the table produced an accepting proof");

    emit("oversize-probes", json!({"source": l.source, "key_bits": need, "arms": points}));
}

fn psweep_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let mut points = vec![];
    for p in arg_list(args, "--p-sweep", &[32usize, 64, 136, 272, 541]) {
        let mut params = params_of(&l, args, &vname);
        params.walk_p = p;
        eprintln!("psweep p={p}");
        let (mut j, accepts) = compose_once(&l, &params, &Default::default(), None);
        assert!(accepts, "honest composed proof rejected at p={p}");
        j["p"] = json!(p);
        points.push(j);
    }
    emit("psweep", json!({"source": l.source, "points": points}));
}

/// Stream II, the requantize seams' output-side control.  One committed
/// activation of a walk group is moved to another legal value and the layer's
/// exact accumulators move with it, so the chunk proves an honest matmul of
/// the changed input: no tamper, no inconsistent region, just a prover that
/// multiplied activations of its own choosing -- which is what
/// `witness-slack.md` section 1d says the walk used to allow.
///
/// The element is the one whose bend disturbs the fewest *dequantized* outputs
/// of its token, because those are what the rest of the chunk consumes: one
/// activation's worth of accumulator is below a bf16 ulp on most channels, so
/// the search usually finds an element the consumers cannot see at all and the
/// seam's own output tie is then the only thing that can reject.  Returns the
/// element and how many dequantized outputs it did move.
fn bend_walk_qx(l: &mut layer::Layer, specs: &[dequant::Spec], mats: &[usize],
                start: usize) -> Option<(usize, usize)> {
    let m0 = &l.mats[mats[0]];
    let (n_in, tokens) = (m0.n_in, m0.tokens);
    if n_in * tokens == 0 {
        return None;
    }
    let noise = |l: &layer::Layer, idx: usize, d: i64| -> usize {
        let (t, k) = (idx / n_in, idx % n_in);
        mats.iter().map(|mi| {
            let m = &l.mats[*mi];
            (0..m.n_out).filter(|c| {
                let a0 = m.acc[t * m.n_out + c];
                let a1 = a0 + d * m.qw[k * m.n_out + c] as i64;
                // stream SS: a folded matmul's output is an integer, not a bf16
                // pattern -- `deq_ref_a` has no answer for it and used to panic
                // here the moment a `y-delay-v` witness reached a probe.
                if m.variant == dequant::Variant::Fold {
                    return layer::fold_out_a(a1, m, t, *c) != layer::fold_out_a(a0, m, t, *c);
                }
                layer::deq_ref_a(a1, m, &specs[*mi], t, *c)
                    != layer::deq_ref_a(a0, m, &specs[*mi], t, *c)
            }).count()
        }).sum()
    };
    // any other legal activation: the walks range check `qx + 2^(bits-1)`, and
    // both neighbours of an interior value are inside that window
    let dir = |l: &layer::Layer, idx: usize| -> i64 {
        if l.mats[mats[0]].qx[idx] < 126 { 1 } else { -1 }
    };
    let mut best = (usize::MAX, 0usize, 0i64);
    for step in 0..n_in * tokens {
        let idx = (start + step) % (n_in * tokens);
        let d = dir(l, idx);
        let s = noise(l, idx, d);
        if s < best.0 {
            best = (s, idx, d);
        }
        if s == 0 {
            break;
        }
    }
    let (moved, idx, d) = best;
    let (t, k) = (idx / n_in, idx % n_in);
    for mi in mats {
        let m = &mut l.mats[*mi];
        assert_eq!(m.n_in, n_in, "walk group mixes input widths");
        m.qx[idx] = (m.qx[idx] as i64 + d) as i32;
        let n_out = m.n_out;
        for c in 0..n_out {
            m.acc[t * n_out + c] += d * m.qw[k * n_out + c] as i64;
        }
    }
    Some((idx, moved))
}

fn compose_probes_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let params = params_of(&l, args, &vname);
    let (hj, ok) = compose_once(&l, &params, &Default::default(), None);
    assert!(ok, "honest composed proof rejected");
    let honest_root = hj["pre_root"].as_str().unwrap().to_string();
    let honest_v = hj["verdict"].clone();
    let mut out = vec![json!({"probe": "honest", "result": hj.clone()})];
    // Stream NN: which verdict fields a control actually turned off.  "The
    // control rejects" is not the whole claim a record should make -- "and by
    // exactly this one field" is, because a probe that trips three checks at
    // once has not isolated the tie it was written for.  Fields that are true
    // (or all-true) on the honest run and not on this one, and nothing else.
    let falsified = |got: &serde_json::Value| -> Vec<String> {
        let all_true = |v: &serde_json::Value| -> Option<bool> {
            match v {
                serde_json::Value::Bool(b) => Some(*b),
                serde_json::Value::Array(a)
                    if !a.is_empty() && a.iter().all(|x| x.is_boolean()) =>
                    Some(a.iter().all(|x| x.as_bool().unwrap())),
                _ => None,
            }
        };
        let mut f = vec![];
        if let Some(o) = honest_v.as_object() {
            for (k, hv) in o {
                if all_true(hv) == Some(true)
                    && all_true(got.get(k).unwrap_or(&serde_json::Value::Null)) == Some(false)
                {
                    f.push(k.clone());
                }
            }
        }
        f
    };

    // Stream NN-3: `--probe <substring>[,<substring>...]` runs only the controls
    // whose name contains one of them (`arg_strs` splits on commas and reads
    // one flag), so a control can have a process of its own.  Two reasons, both paid for on the H200's real 8B chunk: the suite
    // is 46 probes and about two hours there, and one probe that cannot fire on
    // a dump (`o seam`, whose rebuilt layer needs a bigger exp table than the
    // honest statement declares) panics and takes every probe after it with it.
    // With no `--probe` the list is what it always was.
    let only = arg_strs(args, "--probe", &[]);
    let mut probe = |name: &str, l2: &layer::Layer, t: &compose::ProverTamper,
                     expect: &str| -> bool {
        if !only.is_empty() && !only.iter().any(|w| name.contains(w.as_str())) {
            out.push(json!({"probe": name, "skipped": "not selected by --probe"}));
            return false;
        }
        eprintln!("compose probe {name}");
        // verify against the HONEST fixed statement: the trusted half comes from
        // setup, never from the prover under test
        let out2 = compose::prove_layer(l2, &params, t);
        // a tamper that changed nothing proved the honest witness, and calling
        // that "accepted" is how `drop-big` at layer 0 read as a soundness gap
        // (its aimed slot's operands shared an exponent); it is a broken probe
        assert_ne!(out2.glue_tamper_fired, Some(0),
                   "probe {name}: the glue tamper did not fire, so nothing was measured");
        let claims_b = serde_json::to_vec(&out2.claims).unwrap();
        let mut fixed = out2.fixed.clone();
        fixed.pre_root = honest_root.clone();
        let fixed_b = serde_json::to_vec(&fixed).unwrap();
        let (verdict, root_differs) = match &out2.proof {
            Some(pf) => {
                let proof_b = serde_json::to_vec(pf).unwrap();
                let v = compose::verify_composed(&fixed_b, &claims_b, &proof_b);
                let rd = format!("{:?}", pf.commitments[0]) != honest_root;
                (Some(v), rd)
            }
            None => (None, false),
        };
        let accepted = verdict.as_ref().map(|v| v.accepts()).unwrap_or(false);
        let flipped = verdict.as_ref()
            .map(|v| falsified(&serde_json::to_value(v).unwrap()))
            .unwrap_or_default();
        out.push(json!({
            "probe": name, "expect_reject_by": expect,
            "accepted": accepted,
            "falsified": flipped,
            "tamper_fired": out2.glue_tamper_fired,
            // Stream NN-3: the `s_v` tie is an equality between two tree-0
            // column families, so no verdict field can carry it -- a verifier
            // holds it through `pre_root`.  The probe record carries the
            // build's own reading, which is how a control on it is visible.
            "sv_tie": out2.family_cells.get("sv_tie"),
            "prove_error": out2.prove_error,
            "root_differs_from_honest": root_differs,
            "verdict": verdict.as_ref().map(|v| serde_json::to_value(v).unwrap()),
        }));
        accepted
    };

    // Stream OO: every `layer_clone` below is dropped as soon as its probe has
    // run.  A shadowed `let mut l2` is not freed until the function returns, so
    // the six of them were held for the whole run and the 8B suite was
    // OOM-killed on the box at probe 31 of 46 (log row 57, rc 137).  Nothing
    // about what a probe proves changes.
    // 1. a corrupted accumulator: the dequant region stays self-consistent and the
    //    Y = W X identity is what rejects
    let mut l2 = layer_clone(&l);
    let dn = l2.mats.iter().position(|m| m.name == "down").unwrap();
    let n = l2.mats[dn].acc.len();
    l2.mats[dn].acc[123 % n] += 1;
    assert!(!probe("corrupted accumulator (down)", &l2, &Default::default(),
                   "the Y = W X identity"), "corrupted acc accepted");
    drop(l2);

    // 2. a corrupted activation: the same identity, other side
    let mut l2 = layer_clone(&l);
    for name in ["q", "k", "v"] {
        let mi = l2.mats.iter().position(|m| m.name == name).unwrap();
        let n = l2.mats[mi].qx.len();
        let x = &mut l2.mats[mi].qx[77 % n];
        *x = if *x < 127 { *x + 1 } else { *x - 1 };
    }
    assert!(!probe("corrupted activation (qkv)", &l2, &Default::default(),
                   "the Y = W X identity"), "corrupted qx accepted");
    drop(l2);

    // 3. a corrupted key in band 0: the Q.K^T identity
    let mut l2 = layer_clone(&l);
    let n = l2.bands[0].kk.len();
    l2.bands[0].kk[1234 % n] += 1;
    assert!(!probe("corrupted key (band 0)", &l2, &Default::default(), "Q.K^T"),
            "corrupted key accepted");
    drop(l2);

    // 4. a corrupted attention output: the A.V link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.acc.len();
    l2.ao.acc[999 % n] += 1;
    assert!(!probe("corrupted attention output", &l2, &Default::default(), "A.V"),
            "corrupted attnout accepted");
    drop(l2);

    // 5. a corrupted attnout denominator: the denominator link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.den.len();
    l2.ao.den[7 % n] += 1;
    assert!(!probe("corrupted attnout denominator", &l2, &Default::default(),
                   "the denominator link"), "corrupted den accepted");
    drop(l2);

    // 5b. B10 step 3: a token that attends to a key the causal mask removes.  The
    //     corner's `sn` and the prefix's `sn` are what the per-token region's `az`
    //     is read against, and the AIR's fold drops the masked slot, so the two no
    //     longer add up to the denominator the witness claims.
    if l.bands.iter().any(|b| b.corner) {
        for (k, i) in [(0usize, 11usize), (1, 4242)] {
            let mut l2 = layer_clone(&l);
            l2.unmask_corner(i);
            let name: &'static str = if k == 0 {
                "attends to one future key"
            } else {
                "attends to a second future key"
            };
            assert!(!probe(name, &l2, &Default::default(), "the denominator link"),
                    "an unmasked future key accepted");
        }
    }

    // 6. the corrupted exp table, witness rebuilt to match: every internal check
    //    passes and the PINNED ROOT is what rejects -- B6.5's acceptance flip
    let mut l2 = layer_clone(&l);
    l2.corrupt_exp_table(4321);
    assert!(!probe("corrupted exp table, witness rebuilt", &l2, &Default::default(),
                   "the pinned preprocessed root"), "corrupted exp table accepted");
    drop(l2);

    // 7. the corrupted normalisation table, same discipline
    assert!(!probe("corrupted norm table, witness rebuilt", &l,
                   &compose::ProverTamper { norm_row: Some(1000), ..Default::default() },
                   "the pinned preprocessed root"), "corrupted norm table accepted");

    // 9-11. B10 step 2, rule 13: the wiring family's own probes, inside the
    //    composed path.  The first is the one the ties exist for -- a region's
    //    copy of the residual stream is changed and everything *inside* that
    //    region stays consistent, so only the equality between two regions'
    //    claims can catch it.  The other two are the add's own argument, run
    //    here rather than only in `glue-probes`, on the real layer's columns.
    if l.glue.is_some() {
        // stream II: the head chunk's family has one reduction and no add, so
        // there is no second region's copy of the stream to break -- that probe
        // is the layer's, and the head chunk's equivalent is the tail's own
        // copy, which probe 12 below runs on whichever family this chunk has.
        if !l.glue.as_ref().is_some_and(|g| g.head) {
            assert!(!probe("glue: gn2's copy of the residual stream", &l,
                           &compose::ProverTamper { glue_stream: Some((2, 4321)),
                                                    ..Default::default() },
                           "the tie gn1.xmid == gn2.in"),
                    "a broken residual thread accepted");
        }
        assert!(!probe("glue: one add's result, one step up", &l,
                       &compose::ProverTamper {
                           glue: Some((0, glue::Tamper::Result(97))), ..Default::default() },
                       "the add's output constraint"),
                "a corrupted glue result accepted");
        assert!(!probe("glue: the anchor at the smaller exponent", &l,
                       &compose::ProverTamper {
                           glue: Some((0, glue::Tamper::DropBig(1234))), ..Default::default() },
                       "the POW lookup (the dropped operand is the smaller one)"),
                "drop-big accepted");
        assert!(!probe("glue: a leaf that is not a bf16 value", &l,
                       &compose::ProverTamper {
                           glue: Some((1, glue::Tamper::NotBf16(55))), ..Default::default() },
                       "the VAL lookup"),
                "an illegal glue leaf accepted");
    }

    // 12-15. stream E's regions, on whichever of them this invocation built.
    //    The indices come from the plan rather than being counted by hand, so a
    //    later region cannot silently shift what these probes point at.
    {
        let gplan = compose::Plan::new(&params);
        if let Some(g) = gplan.glue.as_ref() {
            // `Tamper::Stream` counts leaves across every group of the row and
            // fires only inside group 0, so the index has to be derived from the
            // layout rather than picked: a tail row is `[nreal x, 1 ms]` since
            // stream II moved the weight into tree 0, and anything at or past
            // `nreal` corrupts nothing at all.
            let in0 = g.t.nreal / 2;
            // stream JJ: how far into the SwiGLU region's value list `up` starts
            let swiglu_n = g.tff.map(|t| t.tokens * t.d).unwrap_or(0);
            for (k, i) in g.tails.iter().copied().enumerate() {
                // stream II: the region's own tag, so a head chunk's single
                // tail is named `ntf` and the layer's two keep the names the
                // record already carries
                let n = g.regions[i].tag.clone();
                let _ = k;
                // the tie is the whole point of the tail: its copy of the
                // stream is changed and the region stays internally consistent,
                // so only `nt.In(0) == gn.In(0)` can catch it
                assert!(!probe(&format!("tail {n}: its own copy of the stream"), &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::Stream(in0))),
                                   ..Default::default() },
                               &format!("the tie {n}.In(0) == its reduction's In(0)")),
                        "a tail reading a different stream accepted");
                assert!(!probe(&format!("tail {n}: one multiply's result"), &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::Result(97))),
                                   ..Default::default() },
                               "the multiply's output constraint"),
                        "a corrupted tail result accepted");
                // stream II's three controls on the norm tail.  The first two
                // change what `r` is without breaking anything inside the
                // region: the `NormR` lookup still finds a row and the two
                // multiplies still hold, so the only thing that can reject is
                // the tie to the reduction's published sum of squares.
                assert!(!probe(&format!("tail {n}: a wrong r on one row"), &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::Ms(37))),
                                   ..Default::default() },
                               &format!("the tie {n}.In(1) == G the reduction's Out(0)")),
                        "a tail with a wrong r accepted");
                assert!(!probe(&format!("tail {n}: two rows of one token, two rs"), &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::MsRow(37))),
                                   ..Default::default() },
                               &format!("the tie {n}.In(1) == G the reduction's Out(0)")),
                        "a token whose rows disagree about r accepted");
            }
            // and the third: one norm-weight cell bent.  It is preprocessed
            // now, so every constraint still holds and the *root* is what
            // moves -- a different kind of rejection from every probe above,
            // and the one that says `w` is the model's and not the prover's.
            let wtag = g.regions[g.tails[0]].tag.clone();
            assert!(!probe(&format!("tail {wtag}: one norm-weight cell"), &l,
                           &compose::ProverTamper { glue_w: Some(4321),
                                                    ..Default::default() },
                           "the pinned preprocessed root"),
                    "a bent norm weight accepted");
            // the dequant output ties: the wiring family's copy of `o_out` (and
            // of `d_out`) is changed, the region's own add is recomputed from it,
            // and only the tie against the dequant region's published output can
            // tell.  This is the probe those ties exist for.
            // Stream JJ (E11): one per tie that has a counterparty.  For the
            // SwiGLU seam the bent value is `gate`'s copy or `up`'s -- the
            // region recomputes `silu(g) * u` from it either way, so nothing
            // inside the region can tell and only the tie against the gate (or
            // up) dequant region's published output rejects.
            for t in g.out_ties.iter() {
                let Some((ri, ti)) = t.at else { continue };
                let name = t.name;
                let vi = if name == "up" { 4321 + swiglu_n } else { 4321 };
                let _ = ti;
                assert!(!probe(&format!("deq tie: the wiring copy of {name}'s output"), &l,
                               &compose::ProverTamper {
                                   glue_add: Some((ri, vi)), ..Default::default() },
                               &format!("the tie {name}_out == the {name} region's output")),
                        "a corrupted dequant output copy accepted");
            }
            if let Some(i) = g.swiglu {
                assert!(!probe("swiglu: the product, one step up", &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::Result(4097))),
                                   ..Default::default() },
                               "the multiply's output constraint"),
                        "a corrupted swiglu product accepted");
                assert!(!probe("swiglu: a leaf that is not a bf16 value", &l,
                               &compose::ProverTamper {
                                   glue: Some((i, glue::Tamper::NotBf16(1234))),
                                   ..Default::default() },
                               "the UN / VAL lookup"),
                        "an illegal swiglu leaf accepted");
            }
        }
    }

    // stream II: the three requantize seams.  Each gets the three controls the
    //     brief names, and each has to reject by exactly one verdict field:
    //     the input probe by `rq_in_ok`, the output probe by `rq_out_ok`, and
    //     the scale probe by `sums_cancel` (the committed scale no longer
    //     divides the attained maximum, so its `(pm, scale)` lookup is off the
    //     table).  The indices are derived from the plan, never counted.
    {
        let rplan = compose::Plan::new(&params);
        for (k, rp) in rplan.rqs.iter().enumerate() {
            let n = rp.name.clone();
            assert!(!probe(&format!("requant {n}: one input value, replaced"), &l,
                           &compose::ProverTamper {
                               rq: Some((k, requant::RqTamper::In(37))),
                               ..Default::default() },
                           &format!("the tie rq_in[{k}] == the producer's Out(0)")),
                    "a requantize of values the producer did not write accepted");
            // the output side: no tamper at all.  The layer itself carries one
            // activation the requantize did not derive, and its own
            // accumulators move with it, so every constraint and every other
            // equality in the chunk still holds -- an honest matmul of the
            // prover's own choice of input, which is exactly what
            // `witness-slack.md` section 1d says the walk used to allow.
            let mut lb = layer_of(args).0;
            let bent = bend_walk_qx(&mut lb, &rplan.specs, &rplan.walks[rp.walk].mats, 41)
                .unwrap_or_else(|| panic!("requant {n}: the output-side control bent nothing"));
            eprintln!("  {n}: activation {} bent, {} dequantized outputs moved with it",
                      bent.0, bent.1);
            assert!(!probe(&format!("requant {n}: one activation the requantize did not derive"),
                           &lb, &Default::default(),
                           &format!("the tie rq_out[{k}] == walk_qx[{}]", rp.walk)),
                    "an activation the requantize did not derive accepted");
            assert!(!probe(&format!("requant {n}: the per-token scale, one step up"), &l,
                           &compose::ProverTamper {
                               rq: Some((k, requant::RqTamper::Scale(3))),
                               ..Default::default() },
                           "the (pm, scale) lookup"),
                    "a scale that is not the ceil-divide of the attained max accepted");
        }
        // Stream II (E11), the scale tie.  A *consumer's* per-token scale moved
        // by one, with its dequant region rederived under it, so every
        // constraint that region carries still holds -- an honest dequant by a
        // scale the requantize did not pin.  The bend has to be on a matmul
        // that is not its group's first, because the seam reads its scale off
        // that one and moving it would move both sides together; `down`'s group
        // is `[down]` alone, so its own pair is not separable by a witness bend
        // and would need a committed-column tamper instead.
        for (k, rp) in rplan.rqs.iter().enumerate() {
            let mats = &rplan.walks[rp.walk].mats;
            let Some(mi) = mats.get(1).copied() else { continue };
            let mut lb = layer_of(args).0;
            let name = lb.mats[mi].name.clone();
            match rp.v {
                requant::SVariant::Pow2 => lb.mats[mi].jx[0] += 1,
                requant::SVariant::Bf16 => lb.mats[mi].sx[0].1 += 1,
            }
            let at = mats.iter().position(|x| *x == mi).unwrap();
            let base: usize = rplan.rqs[..k].iter().map(|q| rplan.walks[q.walk].mats.len()).sum();
            assert!(!probe(&format!("requant {}: {name} dequantized by another scale",
                                    rp.name),
                           &lb, &Default::default(),
                           &format!("the tie rq_sc[{k}] == deq_sc[{name}]")),
                    "a dequant under a scale the requantize did not pin accepted");
            let _ = base + at;
        }

        // Stream II (E11), the `o` seam.  `o` has no requantize region --
        // `AttnOutEval` is its requantize -- so its control is the same shape as
        // the three above's output side and no tamper either: one `o`
        // activation the attention output did not produce, with the `o` matmul's
        // own accumulator moved to match, so the matmul is honestly proved and
        // only `claims.ao_q` against the walk's `qX` can see it.
        if let Some(wi) = rplan.walks.iter().position(|w| w.name == "o") {
            let mut lb = layer_of(args).0;
            let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
            let bent = lb.glue.is_some().then(|| {
                lb.synth_glue_bent(src_of(args).seed, arg_scalar(args, "--glue-a", 32usize),
                                   round, 41)
            }).flatten();
            match bent {
                Some(at) => {
                    eprintln!("  o seam: activation {at} bent, the whole layer rebuilt around it");
                    assert!(!probe("o seam: one activation the attention output did not produce",
                                   &lb, &Default::default(),
                                   &format!("the tie ao_q == walk_qx[{wi}]")),
                            "an o activation the attnout region did not requantize accepted");
                }
                None => eprintln!("  o seam: SKIP -- this shape has no wiring family, or its o \
walk is not nh*hd wide, so the two folds are over different sequences and `ao_q_ok` is not \
statable here"),
            }
        }

        // ---- stream NN: the rotation and the quantize -----------------------
        //
        // Two ties per side and one preprocessed pin each, and every one of
        // them has to reject by a *different* verdict field -- which is what
        // `falsified` in the probe record says, rather than prose:
        //
        //   producer side, q and k   one dequant output copy      deq_out_ok
        //   producer side, all three one committed input          nn_in_ok
        //   consumer side, all three one band cell                nn_out_ok
        //   the rotation's table     one angle cell               root_ok
        //   the quantize's table     one scale exponent           root_ok
        //   the division table       one quotient                 sums_cancel
        if !rplan.quants.is_empty() {
            let g = rplan.glue.as_ref().expect("the quantize regions need the wiring family");
            let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
            let (seed, ga) = (src_of(args).seed, arg_scalar(args, "--glue-a", 32usize));
            // the input side of each rotation: one copy of the matmul's dequant
            // output bent, the whole rotation and the quantize recomputed from
            // it, so every constraint in three regions still holds and only the
            // tie against the matmul's own published output can tell
            for rr in &g.ropes {
                assert!(!probe(&format!("rope {}: one dequant output copy", rr.name), &l,
                               &compose::ProverTamper {
                                   glue_add: Some((rr.mul, 4321)), ..Default::default() },
                               &format!("the tie deq_out[{}] == rp{}m.In(0)", rr.name, rr.name)),
                        "a rotation of values the matmul did not write accepted");
            }
            // one preprocessed angle: the products follow it, so every
            // constraint holds and the *root* is what moves.  This is the probe
            // that says `cos` and `sin` are the statement's and not the
            // prover's -- and, because they are a function of the block, that
            // tree 0 is pinned per block.
            assert!(!probe("rope q: one cos cell of the angle table", &l,
                           &compose::ProverTamper { rope_w: Some(97),
                                                    ..Default::default() },
                           "the pinned preprocessed root"),
                    "a bent rotation angle accepted");
            for (k, qp) in rplan.quants.iter().enumerate() {
                let n = qp.name;
                // the quantize's own input: another legal bf16 pattern, the
                // quotient rebuilt around it.  For q and k the counterparty is
                // the rotation's `Out(0)`; for v it is the matmul's `deq_out`.
                assert!(!probe(&format!("quant {n}: one input value, replaced"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::In(37))),
                                   ..Default::default() },
                               &format!("the tie nn_in[{k}] == the producer's output")),
                        "a quantize of values the producer did not write accepted");
                assert!(!probe(&format!("quant {n}: one quotient off its own table row"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::Out(41))),
                                   ..Default::default() },
                               "the (tt, m7, q) division table"),
                        "a quotient the division table does not carry accepted");
                assert!(!probe(&format!("quant {n}: one preprocessed scale exponent"), &l,
                               &compose::ProverTamper {
                                   quant: Some((k, rope::QTamper::Scale(5))),
                                   ..Default::default() },
                               "the pinned preprocessed root"),
                        "a bent quantize exponent accepted");
            }
            // the consumer side: one *band* cell moved and the bands rebuilt
            // around it, so the attention this chunk proves is an honest
            // attention over the bent column and nothing but the band's own
            // claim against the quantize region's output can see it.  The
            // generator is the witness's, so this needs a synthetic layer.
            for (which, name) in [(0usize, "Q"), (1, "K"), (2, "V")] {
                // the generator rebuilds the whole layer, so it only makes
                // sense on a synthetic one -- on a dump it would replace the
                // witness under test with a different layer and the probe would
                // measure nothing about this chunk
                if arg_str(args, "--witness").is_some() {
                    eprintln!("  band {name}: SKIP -- the consumer-side control needs the \
synthetic generator (Layer::synth_glue_band_bent); on a dump the bands are B1's");
                    continue;
                }
                let mut lb = layer_of(args).0;
                if lb.glue.is_none() {
                    eprintln!("  band {name}: SKIP -- no synthetic wiring family here");
                    continue;
                }
                lb.synth_glue_band_bent(seed, ga, round, which, 41);
                // stream SS: with a folded `v` there is no quantize region on
                // that side, so the falsified field is the dequant output tie
                // and not `nn_out_ok` -- the fold publishes the int8 cell
                // itself and the bands' V is what reads it
                let tie = if which == 2 && rplan.oside_band("v") {
                    "the tie deq_out[v] == the sum of the bands' own V claims"
                } else {
                    "the tie nn_out == the sum of the bands' own claims"
                };
                assert!(!probe(&format!("band {name}: one committed cell the matmuls did not produce"), &lb, &Default::default(),
                               tie),
                        "a band column the layer's matmuls did not produce accepted");
            }
            // stream NN (NN-2): and the prefix band's own copy of Q, whose
            // counterparty is its group's corner rather than a quantize region.
            // The bend moves one prefix cell and the band's whole attention is
            // rebuilt around it, so the chunk is an honest attention over the
            // cache with queries the `q` matmul did not produce -- and only
            // `nn_q_ok` can see that.
            if arg_str(args, "--witness").is_some() {
                eprintln!("  prefix Q: SKIP -- the consumer-side control needs the \
synthetic generator (Layer::synth_glue_prefix_q_bent); on a dump the bands are B1's");
            } else {
                let mut lp = layer_of(args).0;
                if lp.glue.is_none() {
                    eprintln!("  prefix Q: SKIP -- no synthetic wiring family here");
                } else if lp.bands.iter().all(|b| b.corner) {
                    eprintln!("  prefix Q: SKIP -- this shape has no prefix band");
                } else {
                    lp.synth_glue_prefix_q_bent(seed, ga, round, 41);
                    assert!(!probe("prefix band Q: one query the corner does not carry", &lp,
                                   &Default::default(),
                                   "the tie prefix.Q == corner.Q, per band group"),
                            "a prefix band's queries that are not the corner's accepted");
                }
            }
        }
    }

    // 16. stream E: the head tie.  One committed key of the argmax region is
    //     changed and the region is made consistent with it, so every constraint
    //     it carries still holds and only the equality with the head dequant's
    //     own key claim can reject.
    if compose::Plan::new(&params).argmax_mat().is_some() {
        assert!(!probe("head tie: one of the argmax's committed keys", &l,
                       &compose::ProverTamper {
                           argmax: Some(argmax::Tamper::Key(1234567)),
                           ..Default::default() },
                       "the tie am_key == head_key"),
                "an argmax on keys that are not the head's output accepted");
    }

    // 11c. Stream UU (E11): one cell of a *residual* boundary tree moved alone,
    //     the glue region it is tied to left exactly as it was.  Stream H's
    //     `bounds` probe made this assertion for the pair of sides a layer
    //     chunk carries, but only on real witness dirs; here it runs on every
    //     shape the probe command is given, which is what puts the head chunk's
    //     single input side (`bounds::sides`, one side, tree 3) under it.
    //     Nothing else in the chunk moves, so the only thing that can reject is
    //     the plain equality constraint that binds the tree to the region --
    //     which is the whole statement that the root is bound to the proof and
    //     not merely appended to it.
    if params.bounds {
        let n_glue = compose::Plan::new(&params).bounds.len();
        assert!(n_glue > 0,
                "--bounds produced no residual boundary side: the chunk has no wiring family");
        for k in 0..n_glue {
            assert!(!probe(&format!("residual boundary side {k}, its glue region untouched"),
                           &l,
                           &compose::ProverTamper { bounds: Some((k, 4321)),
                                                    ..Default::default() },
                           "the residual boundary equality constraint"),
                    "a residual boundary column that disagrees with its region accepted");
        }
    }

    // 12. Stream AA (E11): one cell of a K/V boundary tree moved *alone*, the
    //     region it is tied to left exactly as it was.  Nothing else in the
    //     chunk changes, so the only thing that can reject is `ChanEval`'s /
    //     `KeyEval`'s equality constraint -- and that is precisely the statement
    //     that the boundary root is *bound to the proof* and not merely appended
    //     to it.  A stream that cannot make this reject has not bound the root.
    //
    //     Every side is probed, not just the first: an out side and an in side
    //     are written by different sets of bands, and a binding that held on one
    //     and not the other would chain a chunk to a prefix it never proved.
    if params.kv_bind {
        let n_sides = compose::Plan::new(&params).kv.len();
        assert!(n_sides > 0, "--kv-bind produced no K/V boundary sides");
        for k in 0..n_sides {
            let side = compose::Plan::new(&params).bounds.len() + k;
            assert!(!probe(&format!("kv: one cell of K/V boundary side {k}, region untouched"),
                           &l,
                           &compose::ProverTamper { bounds: Some((side, 4321)),
                                                    ..Default::default() },
                           "the K/V boundary equality constraint"),
                    "a K/V boundary column that disagrees with its region accepted");
        }
    }
    // 7b. Stream BB: the embedding gadget's three probes, when the run asked for
    //     it.  The second is the aimed corruption -- the id moved to a different
    //     valid entry with the committed row recomputed to match, so the region
    //     is internally consistent and every constraint inside it holds; the tie
    //     to the layer's residual-stream input is the only thing that can see it.
    if params.embed_vocab > 0 {
        for (name, expect, t) in [
            ("embedding: one token id moved, the committed row left alone",
             "the embedding lookup (sums do not cancel)", embed::Tamper::Tok(3)),
            ("embedding: one token id moved to another valid entry, its row \
              recomputed to match",
             "the tie to the layer's residual-stream input", embed::Tamper::TokRow(5)),
            ("embedding: one committed channel of the stream changed",
             "the embedding lookup (sums do not cancel)", embed::Tamper::Val(777)),
        ] {
            assert!(!probe(name, &l,
                           &compose::ProverTamper { embed: Some(t), ..Default::default() },
                           expect),
                    "{name} accepted");
        }
        // and the table probe, which is the pinned-root probe: the region
        // rebuilds the stream from the corrupted table, so every internal check
        // passes and only tree 0's root disagrees
        assert!(!probe("embedding: one cell of the committed table changed, witness rebuilt",
                       &l,
                       &compose::ProverTamper { embed: Some(embed::Tamper::Table(4321)),
                                                ..Default::default() },
                       "the pinned preprocessed root"),
                "a corrupted embedding table accepted");
    }

    // 17. Stream JJ (E11): the attention output's denominator on a row the `dz`
    //     fold never publishes.  `den` is one committed cell per row and it
    //     entered exactly two places -- `denp = den pj` per slot, and `den *
    //     last` in the `dz` fold -- so on every row of a token but the last it
    //     was free, and with it `denp`.  This probe rounds one such row honestly
    //     against `den - 1`: `denp`, `qm`, `sg` and `d` are all rederived from
    //     it, every lookup lands on a real table row and every constraint of the
    //     pre-change AIR holds.  It is the control for `witness-slack.md`
    //     section 7 item 1 and for the row-`den` hole found beside it.
    {
        let lay = layer::AoLayout::choose(params.nh, params.s, params.hd, params.ao_target);
        if lay.chunks > 1 {
            // `b = 1` is a channel block the `dz` fold does not close on
            let idx = params.hd + lay.per_row;
            assert!(!probe("attnout: one non-last row rounded against den - 1", &l,
                           &compose::ProverTamper { ao_den: Some(idx),
                                                    ..Default::default() },
                           "the `den` pin across a token's rows"),
                    "a rounding by an unpinned denominator accepted");
        }
    }

    // 18. Stream JJ (E11), `witness-slack.md` section 1f: the three controls the
    //     exponent window and the accumulator window exist for.  Each bends one
    //     committed cell and rederives everything the honest builder would from
    //     it, so every other check of the chunk still passes and exactly one
    //     thing can reject:
    //
    //       * `--corrupt-ao-jo` moves one *row*'s `jo` by one and leaves the
    //         token's other rows alone.  Every `EPOW` lookup still lands on a
    //         real table row and the row's own arithmetic is self-consistent, so
    //         only the `nofirst` pin on `joff` can see it.  (A whole token's
    //         `jo` moved by one is a different matter and is *accepted*: `jo` is
    //         the prover's choice inside the window, and what pins it to the
    //         consumer's scale is the tie on `claims.ao_q`, which is stream
    //         II's.  See the record.)
    //       * `--corrupt-ao-num` doubles one slot's committed `num` and derives
    //         `qm`, `sg`, `d` and the reference from it.  The triple
    //         `(jvp + joff, num, pj)` is then not a row of the `EPOW` table, so
    //         only that lookup can reject it.
    //       * `--corrupt-ao-acc` moves one slot's committed low accumulator limb
    //         by one inside its own table, so both range lookups land on real
    //         rows and only the high limb's *value* -- no longer an integer --
    //         parts company with the multiplicity the table was bumped at.
    {
        let lay = layer::AoLayout::choose(params.nh, params.s, params.hd, params.ao_target);
        if lay.chunks > 1 {
            // a channel in the first block: `b = 0`, a row the `nofirst` pin
            // compares against its predecessor
            let idx = params.hd;
            assert!(!probe("attnout: one row's jo moved by one", &l,
                           &compose::ProverTamper { ao_jo: Some(idx),
                                                    ..Default::default() },
                           "the `joff` pin across a token's rows"),
                    "a per-row output exponent accepted");
        }
        assert!(!probe("attnout: one slot's num a doubling out of the window", &l,
                       &compose::ProverTamper { ao_num: Some(params.hd + 1),
                                                ..Default::default() },
                       "the `EPOW` window lookup"),
                "an exponent pair outside the window accepted");
        assert!(!probe("attnout: one accumulator limb off its own decomposition", &l,
                       &compose::ProverTamper { ao_acc: Some(params.hd + 2),
                                                ..Default::default() },
                       "the accumulator window's high limb"),
                "an unbounded accumulator accepted");
        // Stream TT (E11), log row 73: the magnitude form's own.  `amag` is
        // pinned by `(amag - acc)(amag + acc) = 0`, which has two roots; this
        // commits the other one.  Every constraint of the region holds and
        // every other lookup lands, and the committed magnitude is about
        // `2^31`, so `2 mag num` would run past the field -- which is the
        // forgery the window exists to refuse.  Only the range check on the
        // magnitude's limbs can see it.  Present only in the magnitude form,
        // because the offset form commits no such column.
        if params.ao_acc_mag {
            assert!(!probe("attnout: the magnitude committed as the negative root", &l,
                           &compose::ProverTamper { ao_mag: Some(params.hd + 3),
                                                    ..Default::default() },
                           "the magnitude window's limbs"),
                    "a magnitude past the no-wrap window accepted");
        } else {
            eprintln!("compose probe SKIP attnout magnitude root: this statement is in \
                       the offset form (--ao-acc-mag forces the other)");
        }
        // And the counterpart, whose verdict is now the *chunk's* to decide.  A
        // whole token's `jo` moved by one inside the window, with the region
        // rederived, changes every one of that token's requantized outputs.
        //
        // Stream JJ ran this and it accepted, and its record said what would
        // pin it: the consumer-side tie on `claims.ao_q`, which was stream II's
        // to add.  With that tie live it *does* pin it -- the `o` walk's
        // committed activations are the pre-bend integers and the attnout
        // region's fold is over the post-bend ones, so `ao_q_ok` alone rejects.
        //
        // On a chunk with no wiring family the `o` walk publishes no `qX` at
        // all, so there is no consumer and this still accepts.  That is not a
        // hole this control can close: `jo` is the prover's choice and a chunk
        // whose attention output nothing downstream reads has nothing to
        // compare it against.  Both cases are run, and the assert is the
        // chunk's own shape rather than a constant.
        let seam_live = compose::Plan::new(&params).glue.is_some()
            && l.mats.iter().any(|m| m.name == "o" && m.n_in == l.ao.nh * l.ao.hd);
        let jo_tok = probe("attnout: a whole token's jo moved by one", &l,
                           &compose::ProverTamper { ao_jo_tok: Some(1),
                                                    ..Default::default() },
                           if seam_live { "the o seam, `ao_q` against the walk's qX" }
                           else { "nothing -- this chunk has no consumer for it" });
        assert_eq!(!jo_tok, seam_live,
                   "a whole token's jo must reject exactly when the o seam is live: \
                    seam_live={seam_live}, accepted={jo_tok}");
    }

    // 8. a tampered claim in the serialized statement: the replayed transcript
    //    diverges and stwo's verify rejects
    {
        eprintln!("compose probe tampered claim");
        let out2 = compose::prove_layer(&l, &params, &Default::default());
        let mut claims = out2.claims.clone();
        claims.mat_h[0][0] ^= 1;
        let claims_b = serde_json::to_vec(&claims).unwrap();
        let fixed_b = serde_json::to_vec(&out2.fixed).unwrap();
        let proof_b = serde_json::to_vec(out2.proof.as_ref().unwrap()).unwrap();
        let v = compose::verify_composed(&fixed_b, &claims_b, &proof_b);
        assert!(!v.accepts(), "tampered claim accepted");
        out.push(json!({"probe": "tampered claim in the statement",
                        "expect_reject_by": "transcript divergence (stwo verify)",
                        "accepted": v.accepts(),
                        "verdict": serde_json::to_value(&v).unwrap()}));
    }
    emit("compose_probes", json!({"source": l.source, "probes": out}));
}

/// The Layer is plain data; clone by rebuilding the pieces (Attn is not Clone).
fn layer_clone(l: &layer::Layer) -> layer::Layer {
    layer::Layer {
        bands: l.bands.iter().map(attn_clone).collect(),
        band_sizes: l.band_sizes.clone(),
        head_offsets: l.head_offsets.clone(),
        nsub: l.nsub,
        mats: l.mats.iter().map(|m| layer::Mat {
            name: m.name.clone(), variant: m.variant, n_in: m.n_in, n_out: m.n_out,
            tokens: m.tokens, acc: m.acc.clone(), qx: m.qx.clone(), qw: m.qw.clone(),
            jx: m.jx.clone(), jw: m.jw.clone(), sx: m.sx.clone(), sw: m.sw.clone(),
            jv2: m.jv2.clone(),
        }).collect(),
        ao: layer::AttnOut { nh: l.ao.nh, s: l.ao.s, hd: l.ao.hd, acc: l.ao.acc.clone(),
                             den: l.ao.den.clone(), jo: l.ao.jo.clone(), jv: l.ao.jv.clone() },
        glue: l.glue.as_ref().map(|g| layer::GlueCols {
            d: g.d, dff: g.dff, tokens: g.tokens,
            xin: g.xin.clone(), o_out: g.o_out.clone(), xmid: g.xmid.clone(),
            d_out: g.d_out.clone(), xout: g.xout.clone(),
            r1: g.r1.clone(), r2: g.r2.clone(),
            w1: g.w1.clone(), w2: g.w2.clone(),
            h1: g.h1.clone(), h2: g.h2.clone(),
            silu_in: g.silu_in.clone(), silu_out: g.silu_out.clone(),
            x_down: g.x_down.clone(),
            eps: g.eps, inv_d: g.inv_d, reduce: g.reduce.clone(), head: g.head,
        }),
        head_keys: l.head_keys.clone(),
        head_tok: l.head_tok.clone(),
        attn_out_bits: l.attn_out_bits,
        embed_tab: l.embed_tab.clone(),
        tok_in: l.tok_in.clone(),
        layer: l.layer,
        block: l.block,
        rope: l.rope.clone(),
        source: l.source.clone(),
    }
}

fn attn_clone(a: &attn_witness::Attn) -> attn_witness::Attn {
    attn_witness::Attn {
        shape: a.shape, variant: a.variant, qq: a.qq.clone(), kk: a.kk.clone(),
        vv: a.vv.clone(), sc: a.sc.clone(), mm: a.mm.clone(), dd: a.dd.clone(),
        xx: a.xx.clone(), num: a.num.clone(), zz: a.zz.clone(), out: a.out.clone(),
        table: a.table.clone(), log_table: a.log_table, nmax: a.nmax, frac: a.frac,
        frac_per_inst: a.frac_per_inst.clone(), delta: a.delta.clone(),
        q_bits: a.q_bits, k_bits: a.k_bits, source: a.source.clone(),
        off_table: a.off_table, torch_mismatch: a.torch_mismatch, corner: a.corner,
        zz_total: a.zz_total.clone(),
    }
}

// ----------------------------------------------------------------- requant ----

fn rwitness_of(args: &[String], v: requant::SVariant) -> requant::RWitness {
    let default_mat = if v == requant::SVariant::Pow2 { "q" } else { "gate" };
    let mat = arg_str(args, "--mat").unwrap_or_else(|| default_mat.into());
    let n: usize = arg_scalar(args, "--n", 0usize);
    if let Some(dir) = arg_str(args, "--witness") {
        match requant::RWitness::load(&dir, &mat, n) {
            Ok(w) => return w,
            Err(e) => eprintln!("  (no real requant witness: {e})"),
        }
    }
    requant::RWitness::synthetic(arg_scalar(args, "--tokens", 64usize),
                                 arg_scalar(args, "--width", 64usize), v, 7)
}

/// The requantize + scale-argument gadget: cells per value, both scale shapes.
fn requant_cmd(args: &[String]) {
    let mut points = vec![];
    for vname in arg_strs(args, "--variants", &["pow2", "bf16"]) {
        let v = requant::SVariant::parse(&vname);
        let w = rwitness_of(args, v);
        for with_max in [false, true] {
            eprintln!("requant {} with_max={} ({})", vname, with_max, w.source);
            let o = requant::run(&w, v, with_max, arg_scalar(args, "--per-row", 16usize), None);
            if !o.accepts() {
                eprintln!("  REJECTED: stark={} sums={} off={} err={:?}", o.proven.stark_ok,
                          o.sums.cancels(), o.off_table, o.proven.prove_error);
            }
            assert!(o.accepts(), "honest requant proof rejected ({vname}, max={with_max})");
            assert_eq!(o.ref_n - o.ref_ok, 0,
                       "requant disagrees with B1's committed qx ({vname})");
            points.push(o.json(&w.source));
        }
    }
    let checks: Vec<_> = ["pow2", "bf16"].iter()
        .map(|v| requant::check_tables(requant::SVariant::parse(v), 20000)).collect();
    emit("requant", json!({"points": points, "tables": checks}));
}

fn requant_probes_cmd(args: &[String]) {
    let v = requant::SVariant::parse(&arg_strs(args, "--variants", &["bf16"])[0]);
    let w = rwitness_of(args, v);
    let honest = requant::run(&w, v, true, 8, None);
    assert!(honest.accepts(), "honest requant proof rejected");
    let mut out = vec![json!({"probe": "honest", "result": honest.json(&w.source)})];
    let cases: Vec<(&str, requant::Tamper)> = vec![
        ("wrong quotient", requant::Tamper::WrongQ(11)),
        ("nz flipped", requant::Tamper::FlipNz(13)),
        ("scale one step too big", requant::Tamper::ScaleUp(3)),
        ("max below an attained value", requant::Tamper::PmLow(5)),
        ("max above every value", requant::Tamper::PmHigh(5)),
    ];
    for (name, t) in cases {
        eprintln!("requant probe {name}");
        let o = requant::run(&w, v, true, 8, Some(t));
        assert!(!o.accepts(), "{name} must be rejected");
        out.push(json!({"probe": name, "accepted": o.accepts(),
                        "stark_ok": o.proven.stark_ok,
                        "sums_cancel": o.sums.cancels(), "off_table": o.off_table,
                        "prove_error": o.proven.prove_error}));
    }
    emit("requant_probes", json!({"variant": v.name(), "probes": out}));
}

// ------------------------------------------------------------- B10 glue ----
// The wiring between the composed layer's regions (B6.5 open gap 1), priced per
// gadget and checked against B1's own committed columns where there is one.
fn glue_cmd(args: &[String]) {
    use glue::{GlueWitness, Kind, Prog, Tiling, UnKind};
    let dir = arg_str(args, "--witness");
    let kmax: u32 = arg_scalar(args, "--kmax", 18u32);
    let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
    let leaves: usize = arg_scalar(args, "--leaves", 32usize);
    let per_row: usize = arg_scalar(args, "--per-row", 16usize);
    let rows: usize = arg_scalar(args, "--rows", 4096usize);
    let tokens: usize = arg_scalar(args, "--tokens", 512usize);
    let a: usize = arg_scalar(args, "--glue-a", 32usize);
    let mut points = vec![];

    let check = |o: &glue::Outcome, what: &str| {
        if !o.accepts() {
            eprintln!("  {what}: stark={} sums={} off={} tie={} err={:?}", o.proven.stark_ok,
                      o.sums.cancels(), o.off_table, o.tie_ok, o.proven.prove_error);
        }
        assert!(o.accepts(), "honest glue proof rejected ({what})");
        assert_eq!(o.ref_n - o.ref_ok, 0, "glue disagrees with its own reference ({what})");
        assert_eq!(o.bad_exp, 0, "{what}: {} results leave bf16's normal range", o.bad_exp);
    };

    // the three primitives, on synthetic values with a wide exponent spread so
    // the dropping rows and the exact zero are both exercised
    let cases: Vec<(&str, Prog, u64)> = vec![
        ("add", Prog::tree(leaves, leaves, UnKind::None), 7),
        ("unary+add", Prog::tree(leaves, leaves, UnKind::Square), 11),
        ("mul", Prog::pairs(per_row, Kind::Mul, UnKind::None, UnKind::None), 13),
        ("swiglu", Prog::pairs(per_row, Kind::Mul, UnKind::Silu, UnKind::None), 17),
        // Stream II: the norm tail as it is composed -- the reduction's answer
        // through `UnKind::NormR` and a preprocessed weight.  Here it runs on
        // synthetic values, which is the isolation the composed rung has not
        // got: a failure in this case is the region, never the wiring.
        ("normtail", Prog::norm_tail(per_row, UnKind::NormR {
            inv_d: bf16::from_f64(1.0 / per_row as f64), eps: bf16::from_f64(1e-5) }), 19),
    ];
    for (name, prog, seed) in &cases {
        let w = GlueWitness::synthetic(rows, prog, round, *seed);
        let o = glue::run(&w, prog, round, kmax, None);
        check(&o, name);
        points.push(o.json(&w.source, name));
    }

    // the composed layer's own four regions, on B1's own committed columns:
    // `xin + o_out` has to reproduce `xmid` and `xmid + d_out` has to reproduce
    // `xout`, and the two levels of each norm have to be `tree_sum`'s own order
    let mut drift = vec![];
    let mut b1 = vec![];
    if let Some(d) = &dir {
        let man: Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{d}/manifest.json")).expect("manifest"))
            .expect("manifest json");
        let dm = man["glue"]["d"].as_u64().expect("manifest: glue.d -- extract with --glue")
            as usize;
        let rd = |n: &str| -> Vec<u16> {
            glue::read_u16(d, n).unwrap_or_else(|e| panic!("{e}"))
        };
        let t = Tiling::new(tokens, dm, a);
        let l1 = Prog::norm_resid(t.nreal, t.b);
        let l2 = Prog::tree(t.a, t.a, UnKind::None);
        let mut stream = t.lay(&rd("xin")[..tokens * dm]);
        for (k, (add, want, wname)) in
            [("o_out", "xmid", "n1"), ("d_out", "xout", "n2")].iter().enumerate()
        {
            let w1 = GlueWitness::new(t.rows1(), vec![stream.clone(), t.lay(&rd(add))],
                                      &l1, round, format!("B1 {add} [{tokens} x {dm}]"));
            let o1 = glue::run(&w1, &l1, round, kmax, None);
            check(&o1, wname);
            let mut j = o1.json(&w1.source, &format!("norm{}+resid{}", k + 1, k + 1));
            // the region's own outputs, value by value, against B1's column
            let mut tab = glue::GlueTables::new(kmax, round, &l1.un_kinds());
            let r1 = glue::glue_region(&w1, &l1, round, "r", &mut tab, None);
            let got = t.unlay(&r1.outs[1]);
            let wv = rd(want);
            let bad = (0..tokens * dm)
                .filter(|i| glue::canon(got[*i]) != glue::canon(wv[*i]))
                .count();
            let neg0 = wv[..tokens * dm].iter().filter(|q| **q == 0x8000).count();
            j["b1_values_checked"] = json!(tokens * dm);
            j["b1_mismatch"] = json!(bad);
            j["b1_negative_zeros"] = json!(neg0);
            assert_eq!(bad, 0, "{wname}: the bf16 add disagrees with B1 on {bad}");
            points.push(j);
            b1.push(json!({"column": want, "values": tokens * dm, "mismatch": bad}));

            let w2 = GlueWitness::new(tokens, vec![t.roots_to_l2(&r1.outs[0])], &l2, round,
                                      format!("the level-1 roots [{tokens} x {}]", t.a));
            let o2 = glue::run(&w2, &l2, round, kmax, None);
            check(&o2, &format!("{wname} level 2"));
            points.push(o2.json(&w2.source, &format!("norm{}-l2", k + 1)));
            let mut tab2 = glue::GlueTables::new(kmax, round, &[]);
            let r2 = glue::glue_region(&w2, &l2, round, "r", &mut tab2, None);
            drift.push(glue::norm_drift(d, dm, tokens, round, wname, Some(&r2.outs[0]))
                       .expect("norm drift"));
            stream = t.lay(&got);
        }
    }
    // the three primitives, isolated -- the numbers `b2/cost_model.py` reads.
    // A bf16 value is two committed columns here (`sm`, `ef`), so a unary op's
    // marginal is its two output columns plus the lookup's paired 2 cells, and
    // the producer's own two columns are not its cost.
    let pick = |g: &str| -> f64 {
        points.iter().find(|p| p["gadget"] == g)
            .and_then(|p| p["cells_per_op_marginal"].as_f64()).unwrap_or(0.0)
    };
    let prim = json!({
        "add": pick("add"), "mul": pick("mul"), "unary": 4.0, "value": 2.0,
        "note": "add and mul measured on a power-of-two tiled region, each output range-checked against VAL; unary is its two output columns plus the lookup's 2 cells; a committed bf16 value is 2 cells",
    });
    emit("glue", json!({"points": points, "primitives": prim,
                        "tables": glue::check_tables(round, 20000),
                        "dmax": glue::DMAX, "against_b1": b1,
                        "norm_reduction_drift": drift}));
}

fn glue_probes_cmd(args: &[String]) {
    use glue::{GlueWitness, Kind, Prog, Tamper, UnKind};
    let kmax: u32 = arg_scalar(args, "--kmax", 18u32);
    let round = Round::parse(&arg_strs(args, "--round", &["even"])[0]);
    let rows: usize = arg_scalar(args, "--rows", 256usize);
    let mut out = vec![];
    let cases: Vec<(&str, Prog, u64)> = vec![
        ("add tree", Prog::tree(8, 8, UnKind::None), 7),
        ("square + add tree", Prog::tree(8, 8, UnKind::Square), 11),
        ("multiply", Prog::pairs(8, Kind::Mul, UnKind::None, UnKind::None), 13),
        ("norm + resid", Prog::norm_resid(8, 8), 19),
    ];
    for (label, prog, seed) in cases {
        let w = GlueWitness::synthetic(rows, &prog, round, seed);
        let honest = glue::run(&w, &prog, round, kmax, None);
        if !honest.accepts() {
            eprintln!("  {label}: stark={} sums={} off={} tie={} refbad={} err={:?}",
                      honest.proven.stark_ok, honest.sums.cancels(), honest.off_table,
                      honest.tie_ok, honest.ref_n - honest.ref_ok, honest.proven.prove_error);
        }
        assert!(honest.accepts(), "honest {label} rejected");
        assert_eq!(honest.ref_n - honest.ref_ok, 0, "{label}: reference mismatch");
        out.push(json!({"region": label, "probe": "honest",
                        "accepted": honest.accepts(),
                        "result": honest.json(&w.source, label)}));
        let tampers: Vec<Tamper> = if prog.kind == Kind::Mul {
            vec![Tamper::Result(3), Tamper::NotBf16(5)]
        } else {
            vec![Tamper::Result(3), Tamper::NormOut(5), Tamper::Anchor(7),
                 Tamper::DropBig(9), Tamper::Sign(11), Tamper::NotBf16(5)]
        };
        for t in tampers {
            eprintln!("glue probe {label}: {}", t.name());
            let o = glue::run(&w, &prog, round, kmax, Some(t));
            assert!(o.fired > 0, "{label}: {} did not fire, so nothing was measured", t.name());
            assert!(!o.accepts(), "{label}: {} must be rejected", t.name());
            out.push(json!({"region": label, "probe": t.name(),
                            "accepted": o.accepts(),
                            "tamper_fired": o.fired,
                            "stark_ok": o.proven.stark_ok,
                            "sums_cancel": o.sums.cancels(),
                            "off_table": o.off_table,
                            "tie_ok": o.tie_ok,
                            "prove_error": o.proven.prove_error}));
        }
    }
    emit("glue_probes", json!({"probes": out, "dmax": glue::DMAX}));
}

// ---------------------------------------------------------------- selftest ----
fn selftest(args: &[String]) {
    let mut runs = vec![];
    let n = arg_scalar(args, "--n", 1 << 10);
    for v in dequant::LADDER {
        for r in ["even", "away"] {
            if v == "limbs" && r == "away" {
                continue; // the baseline has no rounding mode
            }
            let sp = spec_of(args, v, r, 8);
            let w = Witness::get(&arg_str(args, "--witness"), "gate", n, 8, 896);
            let o = dequant::run(&w, &sp, None);
            if !o.accepts() {
                eprintln!("  {}: stark={} sums={} off={} err={:?}", sp.label(),
                          o.proven.stark_ok, o.sums.cancels(), o.off_table,
                          o.proven.prove_error);
            }
            assert!(o.accepts(), "selftest failed ({})", sp.label());
            assert!(o.proven.pre_ok, "preprocessed tree is not the widest ({})", sp.label());
            assert_eq!(o.ref_n - o.ref_ok, 0, "reference mismatch ({})", sp.label());
            runs.push(json!({"spec": sp.label(), "point": o.json(&w.source)}));
        }
    }
    // The packed attention, at a band of instances.  `intexp` is left out on
    // purpose: its table is indexed by the integer shift, so it needs 2^22-2^23 rows
    // on this witness and ~4.5 GiB per run -- nine of those in one process is how the
    // first version of this selftest got OOM-killed.  `intsplit6` exercises the same
    // packing at 2^16 rows, which is also what the sweep runs.
    let mut packed = vec![];
    for insts in [1usize, 2, 4] {
        let sh = Shape { s: 32, c: 128, hd: 64, insts };
        let cfg = attn_cfg(args, 8);
        for name in ["bf16x", "intsplit6"] {
            let v = attn_witness::Variant::parse(name);
            let a = Attn::get(&src_of(args), sh, v);
            let o = attn_run::run(&a, &cfg);
            if !o.accepts() {
                eprintln!("  attn {name} insts={insts}: stark={} qk={} av={} den={} sums={} \
                           off={} err={:?}", o.proven.stark_ok, o.qk_ok, o.av_ok, o.den_ok,
                          o.sums.cancels(), o.off_table, o.proven.prove_error);
            }
            assert!(o.accepts(), "packed attention selftest failed ({name}, insts={insts})");
            packed.push(json!({"variant": name, "insts": insts, "point": o.json(&a, &cfg)}));
        }
    }
    // the tables themselves, against a single f64 rounding
    let tbl = dequant::check_tables(16, Round::Even, 50_000);
    assert!(tbl["norm_mismatch"].as_array().unwrap().is_empty(), "norm table disagrees with f64");
    assert!(tbl["mul_mismatch"].as_array().unwrap().is_empty(), "mul table disagrees with f64");
    // the requantize / max-argument gadget, both scale shapes
    let mut rq = vec![];
    for v in ["pow2", "bf16"] {
        let sv = requant::SVariant::parse(v);
        let w = requant::RWitness::synthetic(32, 48, sv, 7);
        let o = requant::run(&w, sv, true, 8, None);
        assert!(o.accepts(), "requant selftest failed ({v})");
        assert_eq!(o.ref_n - o.ref_ok, 0, "requant reference mismatch ({v})");
        rq.push(o.json(&w.source));
    }
    // the composed proof, synthetic, through the serialized statement
    let (lc, vname) = layer_of(&["--table-bits".into(), "0".into()]);
    let pc = params_of(&lc, args, &vname);
    let (cj, ok) = compose_once(&lc, &pc, &Default::default(), None);
    assert!(ok, "composed selftest rejected: {cj}");
    emit("selftest", json!({"dequant": runs, "attention": packed, "tables": tbl,
                            "requant": rq, "compose": cj}));
}

/// Stream EE (open item 21): `fpcompare EMIT.json ASK.json` -- does the plan
/// line's emitted kernel set cover everything a chunk on that line looks up?
///
/// The rule is containment, not equality: one emit line installs the shapes of
/// every component of its design point, and a chunk asks for the subset its own
/// components use, so `ask ⊆ emit` is the property that keeps a run off the
/// scalar route. Any fingerprint on the ask side and not on the emit side is
/// the defect item 21 recorded, and it is named here with its family and its
/// instruction count.
fn fpcompare_cmd(args: &[String]) {
    let files: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    assert_eq!(files.len(), 2, "fpcompare wants EMIT.json and ASK.json");
    let read = |p: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))).unwrap()
    };
    let (emitted, asked) = (read(files[0]), read(files[1]));
    let mut missing = vec![];
    let mut n_ask = 0usize;
    for (family, fps) in asked["families"].as_object().expect("ask: no families") {
        for (fp, instr) in fps.as_object().unwrap() {
            n_ask += 1;
            if emitted["families"][family].get(fp).is_none() {
                missing.push(json!({"family": family, "fingerprint": fp,
                                    "instructions": instr}));
            }
        }
    }
    let n_emit: usize = emitted["families"].as_object().map(|o| {
        o.values().map(|v| v.as_object().unwrap().len()).sum()
    }).unwrap_or(0);
    for m in &missing {
        eprintln!("fpcompare: {} {} ({} instructions) is asked for and never emitted",
                  m["family"].as_str().unwrap(), m["fingerprint"].as_str().unwrap(),
                  m["instructions"]);
    }
    let covered = missing.is_empty();
    eprintln!("fpcompare: {n_ask} shape(s) asked, {n_emit} emitted, {} missing -- {}",
              missing.len(), if covered { "COVERED" } else { "NOT COVERED" });
    emit("fpcompare", json!({
        "emit": files[0], "ask": files[1],
        "emit_shapes": n_emit, "ask_shapes": n_ask,
        "missing": missing, "covered": covered,
    }));
    if !covered {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    spans::install();
    // B12 stream GG: the fork's three marks inside `prove_ex` (a no-op on
    // the SimdBackend arm).
    device::install_mem_hook();
    match cmd.as_str() {
        "tables" => tables_cmd(&args),
        "dequant" => dequant_cmd(&args),
        "width" => width_cmd(&args),
        "kmax" => kmax_cmd(&args),
        "probes" => probes_cmd(&args),
        "pack" => pack_cmd(&args),
        "attn-probes" => attn_probes_cmd(&args),
        "compose" => compose_cmd(&args),
        "pre-split" => pre_split_cmd(&args),
        "expect-root" => expect_root_cmd(&args),
        "model-roots" => model_roots_cmd(&args),
        "compose-chunks" => chunks::chunks_cmd(&args),
        "exprprobe" => exprprobe::cmd(&args),
        "cudacheck" => cudagen::cmd(&args),
        "cudaemit" => cudagen::emit_cmd(&args),
        "logupcheck" => cudagen::logup_cmd(&args),
        "genemit" => cudagen::gen_emit_cmd(&args),
        "compose-probes" => compose_probes_cmd(&args),
        "oversize-probes" => oversize_probes_cmd(&args),
        "psweep" => psweep_cmd(&args),
        "requant" => requant_cmd(&args),
        "requant-probes" => requant_probes_cmd(&args),
        "argmax" => argmax_cmd(&args),
        "argmax-probes" => argmax_probes_cmd(&args),
        "embed" => embed_cmd(&args),
        "embed-probes" => embed_probes_cmd(&args),
        "glue" => glue_cmd(&args),
        "bounds" => bounds::cmd(&args),
        "bounds-chain" => bounds::chain_cmd(&args),
        "bounds-grid" => bounds::grid_cmd(&args),
        "glue-probes" => glue_probes_cmd(&args),
        "selftest" => selftest(&args),
        "fpcompare" => fpcompare_cmd(&args),
        "gkr" => gkrband::cmd(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; \
                       expected tables|dequant|width|kmax|probes|pack|attn-probes|exprprobe|cudacheck|cudaemit|logupcheck|genemit|\
                       argmax|argmax-probes|embed|embed-probes|\
                       compose|pre-split|expect-root|model-roots|compose-probes|bounds-chain|bounds-grid|psweep|requant|requant-probes|glue|glue-probes|selftest");
            std::process::exit(2);
        }
    }
}
