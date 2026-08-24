//! B7b (README 26): the chunk loop.
//!
//! `compose-chunks --chunks N` proves N chunk proofs of one shape in one
//! process: the CUDA context and the fatbin (0.52 s, per process, B7A.md 3)
//! are paid by chunk 0 alone, and every chunk after it is the per-chunk number
//! section 5's headline is built from. Each chunk is `compose::generate` then
//! `compose::prove_gen` -- the two halves of `prove_layer` -- and is reported
//! exactly as `compose` reports a proof, with its own `host_phases` (the
//! accumulator is keyed by chunk, `util::m4_set_chunk`) and its own digest.
//!
//! The layers a chunk proves: one (`--witness` or the synthetic shape), cycled;
//! or several, `--witnesses a,b,c` (real) or `--seeds 7,8,9` (synthetic), so a
//! loop can be made of chunks that share a shape and nothing else.
//!
//! `B7_M5_PIPE=1` is section 4's pipelining: a second thread generates chunk
//! k+1 while this one proves chunk k, one chunk ahead and no more (two chunks'
//! device memory is the price, 26.4). The serial and the overlapped wall are
//! both reported; the overlapped one is `loop_secs / N`, because a chunk's own
//! phases overlap another's and no longer sum to anything.

use std::time::Instant;

use serde_json::{Value, json};

use crate::compose::{self, Params, ProverTamper};
use crate::layer::Layer;
use crate::util::emit;
use crate::{arg_list, arg_scalar, arg_str};

/// What determines every cell of the preprocessed tree in this binary: the
/// layer (its source names its data, witness directory or synthetic seed) and
/// the parameters. FNV-1a over both.
pub fn key_of(l: &Layer, p: &Params, tamper: &ProverTamper) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut put = |b: &[u8]| {
        for x in b {
            h ^= *x as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    put(l.source.as_bytes());
    put(&serde_json::to_vec(p).unwrap());
    put(format!("{:?}", tamper.norm_row).as_bytes());
    h
}

struct SendGen(compose::Gen, usize);
// The generated chunk holds device pointers; the CUDA context is per process
// and nothing in it is bound to the thread that made it (precache.rs).
unsafe impl Send for SendGen {}

fn layers_of(args: &[String]) -> (Vec<Layer>, String) {
    if let Some(dirs) = arg_str(args, "--witnesses") {
        let mut out = vec![];
        let mut vname = String::new();
        for d in dirs.split(',') {
            let mut a: Vec<String> = args.to_vec();
            a.push("--witness".into());
            a.push(d.trim().to_string());
            let (l, v) = crate::layer_of(&a);
            assert!(l.source.contains(d.trim()), "no real layer witness at {d}");
            vname = v;
            out.push(l);
        }
        return (out, vname);
    }
    let seeds: Vec<u64> = arg_list(args, "--seeds", &[]);
    if !seeds.is_empty() {
        let mut out = vec![];
        let mut vname = String::new();
        for s in seeds {
            let mut a: Vec<String> = args.to_vec();
            a.push("--seed".into());
            a.push(s.to_string());
            let (l, v) = crate::layer_of(&a);
            vname = v;
            out.push(l);
        }
        return (out, vname);
    }
    let (l, v) = crate::layer_of(args);
    (vec![l], v)
}

struct Row {
    k: usize,
    li: usize,
    out: compose::Composed,
    ser: crate::Ser,
    /// the wall of this chunk on its own thread: generate + prove + serialize
    /// (serial), or prove + serialize (pipelined, where generation is elsewhere)
    secs: f64,
    gen_secs: f64,
    upload: Value,
}

pub fn chunks_cmd(args: &[String]) {
    let n: usize = arg_scalar(args, "--chunks", 4usize);
    let t_setup = Instant::now();
    let (layers, vname) = layers_of(args);
    let params = crate::params_of(&layers[0], args, &vname);
    for l in &layers[1..] {
        let p2 = crate::params_of(l, args, &vname);
        assert_eq!(serde_json::to_value(&p2).unwrap(), serde_json::to_value(&params).unwrap(),
                   "compose-chunks: the layers do not share one shape");
    }
    let tamper = ProverTamper::default();
    let pipe = crate::util::m4b_mode("B7_M5_PIPE") == 1;
    let keys: Vec<u64> = layers.iter().map(|l| key_of(l, &params, &tamper)).collect();
    let setup_secs = t_setup.elapsed().as_secs_f64();
    eprintln!("compose-chunks: {n} chunks over {} layer(s) [{}], variant={} pipe={pipe} \
               ircache={} precache={}; layers built in {setup_secs:.1} s",
              layers.len(), layers.iter().map(|l| l.source.as_str()).collect::<Vec<_>>().join("; "),
              vname, crate::scalars::mode(), crate::precache::mode());

    let mut rows: Vec<Row> = Vec::with_capacity(n);
    crate::device::reset();
    let t_loop = Instant::now();
    if !pipe {
        for k in 0..n {
            let li = k % layers.len();
            crate::util::m4_set_chunk(k as u64);
            if k > 0 {
                crate::device::reset();
            }
            let t = Instant::now();
            let g = compose::generate(&layers[li], &params, &tamper, Some(keys[li]));
            let gen_secs = t.elapsed().as_secs_f64();
            let out = compose::prove_gen(&layers[li], &params, g);
            let ser = crate::compose_ser(&out);
            let secs = t.elapsed().as_secs_f64();
            eprintln!("compose-chunks: chunk {k} (layer {li}) {secs:.3} s, generate {gen_secs:.3} s");
            rows.push(Row { k, li, out, ser, secs, gen_secs, upload: crate::device::report() });
        }
    } else {
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<SendGen>(1);
            let (layers, params, keys) = (&layers, &params, &keys);
            s.spawn(move || {
                for k in 0..n {
                    let li = k % layers.len();
                    crate::util::m4_set_chunk(k as u64);
                    let t = Instant::now();
                    let g = compose::generate(&layers[li], params, &tamper, Some(keys[li]));
                    eprintln!("compose-chunks: generated chunk {k} in {:.3} s",
                              t.elapsed().as_secs_f64());
                    if tx.send(SendGen(g, k)).is_err() {
                        return;
                    }
                }
            });
            for k in 0..n {
                let SendGen(g, kk) = rx.recv().expect("the generation thread died");
                assert_eq!(k, kk);
                let li = k % layers.len();
                crate::util::m4_set_chunk(k as u64);
                let t = Instant::now();
                let out = compose::prove_gen(&layers[li], params, g);
                let ser = crate::compose_ser(&out);
                let secs = t.elapsed().as_secs_f64();
                eprintln!("compose-chunks: proved chunk {k} (layer {li}) in {secs:.3} s");
                rows.push(Row { k, li, out, ser, secs, gen_secs: 0.0,
                                upload: crate::device::report() });
            }
        });
    }
    let loop_secs = t_loop.elapsed().as_secs_f64();

    // The verifier, after the loop: it is the CPU's and not the prover's wall.
    let mut chunks: Vec<Value> = vec![];
    let mut wholes = vec![];
    let mut stwos = vec![];
    let mut commits = vec![];
    let mut digests = vec![];
    for r in rows {
        crate::util::m4_set_chunk(r.k as u64);
        let (mut j, accepts) = crate::compose_finish(r.out, r.ser, &params);
        assert!(accepts, "compose-chunks: chunk {} rejected: {j}", r.k);
        assert_eq!(j["ref_mismatch"], 0, "chunk {}: witness disagrees with the f64 reference", r.k);
        assert_eq!(j["off_table"], 0, "chunk {}: off-table values", r.k);
        j["upload"] = r.upload;
        j["chunk"] = json!(r.k);
        j["layer"] = json!(r.li);
        j["layer_source"] = json!(layers[r.li].source);
        j["chunk_secs"] = json!(r.secs);
        j["generate_secs"] = json!(r.gen_secs);
        // section 5's whole wall, as m4_host.py reads it (rule 2)
        let hp = &j["host_phases"];
        let s = |k: &str| hp.get(k).and_then(|e| e["secs"].as_f64()).unwrap_or(0.0);
        let whole = j["trace_secs"].as_f64().unwrap() + s("wall/cuda_init") + s("wall/twiddles")
            + j["phases"]["wall_secs"].as_f64().unwrap() + s("wall/ser_statement")
            + s("wall/ser_proof");
        let stwo = j["phases"]["commit_secs"].as_f64().unwrap()
            + j["phases"]["prove_call_secs"].as_f64().unwrap();
        j["whole_wall"] = json!(whole);
        j["stwo_total"] = json!(stwo);
        wholes.push(whole);
        stwos.push(stwo);
        commits.push(j["phases"]["commit_secs"].as_f64().unwrap());
        digests.push(j["proof_sha256"].as_str().unwrap_or("").to_string());
        chunks.push(j);
    }
    let fresh = chunks[0]["cells"]["fresh"].as_f64().unwrap();
    let med = |v: &[f64]| {
        let mut w: Vec<f64> = v.to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if w.is_empty() { 0.0 } else { w[w.len() / 2] }
    };
    let rest = |v: &[f64]| if v.len() > 1 { v[1..].to_vec() } else { v.to_vec() };
    let summary = json!({
        "n": n, "layers": layers.len(), "pipe": pipe,
        "setup_secs": setup_secs,
        "loop_secs": loop_secs,
        "per_chunk_loop_secs": loop_secs / n as f64,
        "chunk0_whole": wholes[0],
        "whole_median_rest": med(&rest(&wholes)),
        "stwo_median_rest": med(&rest(&stwos)),
        "commit_median_rest": med(&rest(&commits)),
        "gen_median_rest": med(&rest(&wholes)) - med(&rest(&stwos)),
        "fresh_cells": fresh,
        "m_cells_per_s": {
            "whole_rest": fresh / med(&rest(&wholes)) / 1e6,
            "loop": fresh / (loop_secs / n as f64) / 1e6,
            "stwo_rest": fresh / med(&rest(&stwos)) / 1e6,
            "commit_rest": fresh / med(&rest(&commits)) / 1e6,
        },
        "digests": digests,
        "ircache": crate::scalars::report(),
        "precache": crate::precache::report(),
        "blob": crate::cudagen::blob_report(),
    });
    eprintln!("compose-chunks: {}", serde_json::to_string_pretty(&summary).unwrap());
    emit("compose_chunks", json!({
        "source": layers[0].source, "variant": vname,
        "params": serde_json::to_value(&params).unwrap(),
        "summary": summary, "chunks": chunks,
    }));
}
