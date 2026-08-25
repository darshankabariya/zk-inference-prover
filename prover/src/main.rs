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
mod gencudakernels;
mod cudakernels;
mod exprprobe;
mod dequant;
mod glue;
mod layer;
mod layout;
mod plogup;
mod requant;
mod walk_air;
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
    }
}

fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn src_of(args: &[String]) -> Src {
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
    }
}

fn attn_cfg(args: &[String], grid_target: usize) -> attn_trace::Cfg {
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
    let w = argmax::AWitness::synthetic(tokens, vocab, ties, seed);
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

// ----------------------------------------------------------------- compose ----

/// Build the layer witness: real from `--witness`, else a small synthetic one.
pub fn layer_of(args: &[String]) -> (layer::Layer, String) {
    let vname = arg_strs(args, "--variants", &["intsplit6"])[0].clone();
    let variant = attn_witness::Variant::parse(&vname);
    let bands = arg_list(args, "--bands", &[8usize, 4, 2]);
    let src = src_of(args);
    if let Some(dir) = &src.dir {
        match layer::Layer::load(dir, &src, &bands, variant) {
            Ok(l) => return (l, vname),
            Err(e) => eprintln!("  (no real layer witness: {e})"),
        }
    }
    let bands = arg_list(args, "--bands", &[2usize, 1]);
    (layer::Layer::synthetic(arg_scalar(args, "--s", 32usize), arg_scalar(args, "--ctx", 128usize),
                             arg_scalar(args, "--hd", 64usize), &bands, variant, &src,
                             arg_scalar(args, "--nin", 64usize), arg_scalar(args, "--nout", 48usize),
                             src.seed),
     vname)
}

pub fn params_of(l: &layer::Layer, args: &[String], vname: &str) -> compose::Params {
    let b0 = &l.bands[0];
    let kmax = l.mats.iter().map(|m| m.kmax_blind()).max().unwrap()
        .min(bf16::KMAX_SOUND);
    // B10 step 3: `bands` names the *prefix* bands only.  The corner rides as one
    // more band per prefix band, with the same `insts` and head offset, and the
    // plan rebuilds them from the flag rather than from the list.
    let n_pre = l.bands.iter().filter(|b| !b.corner).count();
    compose::Params {
        s: l.ao.s, c: b0.shape.c, hd: l.ao.hd, nh: l.ao.nh,
        bands: l.band_sizes[..n_pre].to_vec(),
        corner: l.bands.len() > n_pre,
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
        ao_q_bits: 7,
        // B9 step 1: `ao_rem_bits` and `kmax` are data-derived, so two real layers
        // of one model are two shapes and the chunk loop rejects the pair.  Both
        // are overridable now, so a driver can pin them at the model-wide worst
        // case and prove every layer of a model as one shape -- a few extra table
        // rows against a second emit chain.  Below the witness's own value the
        // range check would just push rows into `off_table`, so that is an error.
        ao_rem_bits: {
            let want = arg_scalar(args, "--ao-rem-bits", l.ao.rem_bits());
            assert!(want >= l.ao.rem_bits(),
                    "--ao-rem-bits {want} is below this witness's {}", l.ao.rem_bits());
            want
        },
        walk_p: arg_scalar(args, "--p", 136usize),
        // B10 step 2: the wiring family rides on a witness `extract.py --glue`
        // wrote, and on nothing else -- a dump without those columns composes
        // exactly as it did before.  `--glue-a` is how many rows one token's
        // stream takes at the reduction's first level; it has to divide `d`.
        glue: l.glue.as_ref().filter(|_| !flag(args, "--no-glue")).map(|g| {
            compose::GlueParams { d: g.d, a: arg_scalar(args, "--glue-a", 32usize) }
        }),
        // Stream H: the boundary commitment, off unless asked for either way --
        // by the flag on the command line or by `B10_BOUNDS` in the environment.
        // Off is what makes the recorded digest the identity line.
        bounds: flag(args, "--bounds") || bounds::enabled(),
    }
}

/// Run the whole serialized round trip: prove, write the statement and the
/// proof as bytes, verify from the bytes alone.
fn compose_once(l: &layer::Layer, params: &compose::Params, tamper: &compose::ProverTamper)
                -> (serde_json::Value, bool) {
    let out = compose::prove_layer(l, params, tamper);
    let ser = compose_ser(&out);
    compose_finish(out, ser, params)
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
        "columns": {"preprocessed": out.n_columns[0], "base": out.n_columns[1],
                    "interaction": out.n_columns[2],
                    "total": out.n_columns.iter().sum::<usize>()},
        "cells": {"preprocessed": out.stats.tree_cells.first().copied().unwrap_or(0),
                  "base": out.stats.tree_cells.get(1).copied().unwrap_or(0),
                  "interaction": out.stats.tree_cells.get(2).copied().unwrap_or(0),
                  "fresh": out.stats.tree_cells[1..].iter().sum::<usize>()},
        "family_cells": out.family_cells,
        "ref_mismatch": out.ref_bad, "off_table": out.off_table,
        "glue_ties_ok": out.glue_ties_ok,
        // Stream H: the boundary trees' own cells and columns, and the two roots
        // the chain check compares.  Null on a chunk proved with the flag off.
        "bounds": out.bounds,
        "bound_roots": verdict.as_ref().map(|v| json!({"in": v.bound_in, "out": v.bound_out})),
        "trace_secs": out.trace_secs, "interaction_secs": out.interaction_secs,
        "prove_secs": out.stats.prove_secs,
        "verify_secs": verdict.as_ref().map(|v| v.verify_secs).unwrap_or(0.0),
        "peak_rss_bytes": out.stats.peak_rss,
        "preprocessed_is_widest": out.stats.pre_ok,
        "phases": out.stats.phases,
        "host_phases": crate::util::m4_report(),
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

fn compose_cmd(args: &[String]) {
    let (mut l, vname) = layer_of(args);
    // B7a milestone 4 (section 3 row 1): rung 3's sixth probe on its own, so the
    // `nopatch` control has a witness whose exp table is *not* the closed form
    // to bite on (section 5 rules 5 and 8). The proof still accepts here -- the
    // table and the witness agree; it is the pinned preprocessed root, which
    // only `compose-probes` checks, that rejects it.
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
    let params = params_of(&l, args, &vname);
    eprintln!("compose: {} bands={:?} variant={} kmax={} rem={} p={}",
              l.source, l.band_sizes, vname, params.kmax, params.ao_rem_bits, params.walk_p);
    let (j, accepts) = compose_once(&l, &params, &tamper);
    assert!(accepts, "honest composed proof rejected: {j}");
    if tamper.norm_row.is_none() {
        assert_eq!(j["ref_mismatch"], 0,
                   "composed witness disagrees with the f64 reference");
    }
    assert_eq!(j["off_table"], 0, "honest composed witness has off-table values");
    emit("compose", json!({"source": l.source, "point": j}));
}

fn psweep_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let mut points = vec![];
    for p in arg_list(args, "--p-sweep", &[32usize, 64, 136, 272, 541]) {
        let mut params = params_of(&l, args, &vname);
        params.walk_p = p;
        eprintln!("psweep p={p}");
        let (mut j, accepts) = compose_once(&l, &params, &Default::default());
        assert!(accepts, "honest composed proof rejected at p={p}");
        j["p"] = json!(p);
        points.push(j);
    }
    emit("psweep", json!({"source": l.source, "points": points}));
}

fn compose_probes_cmd(args: &[String]) {
    let (l, vname) = layer_of(args);
    let params = params_of(&l, args, &vname);
    let (hj, ok) = compose_once(&l, &params, &Default::default());
    assert!(ok, "honest composed proof rejected");
    let honest_root = hj["pre_root"].as_str().unwrap().to_string();
    let mut out = vec![json!({"probe": "honest", "result": hj.clone()})];

    let mut probe = |name: &str, l2: &layer::Layer, t: &compose::ProverTamper,
                     expect: &str| -> bool {
        eprintln!("compose probe {name}");
        // verify against the HONEST fixed statement: the trusted half comes from
        // setup, never from the prover under test
        let out2 = compose::prove_layer(l2, &params, t);
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
        out.push(json!({
            "probe": name, "expect_reject_by": expect,
            "accepted": accepted,
            "prove_error": out2.prove_error,
            "root_differs_from_honest": root_differs,
            "verdict": verdict.as_ref().map(|v| serde_json::to_value(v).unwrap()),
        }));
        accepted
    };

    // 1. a corrupted accumulator: the dequant region stays self-consistent and the
    //    Y = W X identity is what rejects
    let mut l2 = layer_clone(&l);
    let dn = l2.mats.iter().position(|m| m.name == "down").unwrap();
    let n = l2.mats[dn].acc.len();
    l2.mats[dn].acc[123 % n] += 1;
    assert!(!probe("corrupted accumulator (down)", &l2, &Default::default(),
                   "the Y = W X identity"), "corrupted acc accepted");

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

    // 3. a corrupted key in band 0: the Q.K^T identity
    let mut l2 = layer_clone(&l);
    let n = l2.bands[0].kk.len();
    l2.bands[0].kk[1234 % n] += 1;
    assert!(!probe("corrupted key (band 0)", &l2, &Default::default(), "Q.K^T"),
            "corrupted key accepted");

    // 4. a corrupted attention output: the A.V link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.acc.len();
    l2.ao.acc[999 % n] += 1;
    assert!(!probe("corrupted attention output", &l2, &Default::default(), "A.V"),
            "corrupted attnout accepted");

    // 5. a corrupted attnout denominator: the denominator link
    let mut l2 = layer_clone(&l);
    let n = l2.ao.den.len();
    l2.ao.den[7 % n] += 1;
    assert!(!probe("corrupted attnout denominator", &l2, &Default::default(),
                   "the denominator link"), "corrupted den accepted");

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
        assert!(!probe("glue: gn2's copy of the residual stream", &l,
                       &compose::ProverTamper { glue_stream: Some((2, 4321)),
                                                ..Default::default() },
                       "the tie gn1.xmid == gn2.in"),
                "a broken residual thread accepted");
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
        mats: l.mats.iter().map(|m| layer::Mat {
            name: m.name.clone(), variant: m.variant, n_in: m.n_in, n_out: m.n_out,
            tokens: m.tokens, acc: m.acc.clone(), qx: m.qx.clone(), qw: m.qw.clone(),
            jx: m.jx.clone(), jw: m.jw.clone(), sx: m.sx.clone(), sw: m.sw.clone(),
        }).collect(),
        ao: layer::AttnOut { nh: l.ao.nh, s: l.ao.s, hd: l.ao.hd, acc: l.ao.acc.clone(),
                             den: l.ao.den.clone(), jo: l.ao.jo.clone(), jv: l.ao.jv.clone() },
        glue: l.glue.as_ref().map(|g| layer::GlueCols {
            d: g.d, dff: g.dff, tokens: g.tokens,
            xin: g.xin.clone(), o_out: g.o_out.clone(), xmid: g.xmid.clone(),
            d_out: g.d_out.clone(), xout: g.xout.clone(),
            w1: g.w1.clone(), w2: g.w2.clone(),
            eps: g.eps, inv_d: g.inv_d, reduce: g.reduce.clone(),
        }),
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
            assert!(!o.accepts(), "{label}: {} must be rejected", t.name());
            out.push(json!({"region": label, "probe": t.name(),
                            "accepted": o.accepts(),
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
    let (cj, ok) = compose_once(&lc, &pc, &Default::default());
    assert!(ok, "composed selftest rejected: {cj}");
    emit("selftest", json!({"dequant": runs, "attention": packed, "tables": tbl,
                            "requant": rq, "compose": cj}));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    spans::install();
    match cmd.as_str() {
        "tables" => tables_cmd(&args),
        "dequant" => dequant_cmd(&args),
        "width" => width_cmd(&args),
        "kmax" => kmax_cmd(&args),
        "probes" => probes_cmd(&args),
        "pack" => pack_cmd(&args),
        "attn-probes" => attn_probes_cmd(&args),
        "compose" => compose_cmd(&args),
        "compose-chunks" => chunks::chunks_cmd(&args),
        "exprprobe" => exprprobe::cmd(&args),
        "cudacheck" => cudagen::cmd(&args),
        "cudaemit" => cudagen::emit_cmd(&args),
        "logupcheck" => cudagen::logup_cmd(&args),
        "genemit" => cudagen::gen_emit_cmd(&args),
        "compose-probes" => compose_probes_cmd(&args),
        "psweep" => psweep_cmd(&args),
        "requant" => requant_cmd(&args),
        "requant-probes" => requant_probes_cmd(&args),
        "argmax" => argmax_cmd(&args),
        "argmax-probes" => argmax_probes_cmd(&args),
        "glue" => glue_cmd(&args),
        "bounds" => bounds::cmd(&args),
        "glue-probes" => glue_probes_cmd(&args),
        "selftest" => selftest(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; \
                       expected tables|dequant|width|kmax|probes|pack|attn-probes|exprprobe|cudacheck|cudaemit|logupcheck|genemit|\
                       argmax|argmax-probes|\
                       compose|compose-probes|psweep|requant|requant-probes|glue|glue-probes|selftest");
            std::process::exit(2);
        }
    }
}
