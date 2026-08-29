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
//!
//! # B11 stream O: the loop takes the sequence
//!
//! Everything above proves one shape over and over. The sequence is 33 chunks
//! that share almost nothing: each layer has its own witness and its own
//! master table, and the head adds the argmax on top. `--plan FILE` is how the
//! loop is given that. Each line of the plan is one chunk's own arguments,
//! which are put in front of the ones on the command line, so a chunk's line
//! decides what the shared pins leave open and the loop no longer needs every
//! chunk to resolve the same parameters.
//!
//! Two things follow from the shapes being different, and both of them are
//! about what a chunk is allowed to leave behind.
//!
//! The first is the witness. Under a plan a chunk's layer is built when the
//! chunk is generated and dropped as soon as it has been proved, so what is in
//! host memory is the chunk being generated and the chunk being proved and
//! nothing else. Building all 33 up front, which is what the cycling loop does
//! with its handful of layers, would mean 33 real 8B witnesses at once.
//!
//! The second is the preprocessed tree. `precache` keeps the committed tree
//! under a key that stands for the layer and the shape, because in the cycling
//! loop every chunk after the first is the same key and takes a clone of it.
//! The sequence is the opposite: 33 layers at 33 shapes, so 33 keys, and
//! nothing is ever taken again. Cached anyway, each chunk would pay a device
//! clone of a tree no one will ask for and leave it there until the process
//! exits -- 33 resident by the end, where 33 processes held one at a time.
//!
//! So a chunk whose line appears once in the plan is generated with no key at
//! all: its tree is committed and dropped with its session, nothing is cloned
//! and nothing is kept. That is also exactly what the one-shot `compose` path
//! does (`prove_layer` passes no key), which is what makes the loop's chunks
//! comparable to the 33 processes they replace. A line that does repeat keeps
//! its key, because then the cache is doing what it was built for, and the
//! chunk that uses it last hands it back (`precache::forget`).
//!
//! `--ser-dir DIR` writes each chunk's statement and proof under the tag its
//! plan line gives it, which is what `bounds-chain` reads back to join two
//! chunks. It is the same pair of files the one-shot `compose` path writes.

//! # B11 stream P: what `build_secs` is made of
//!
//! `build_secs` is one number for four different kinds of work, and on the
//! sequence it is most of the wall (`briefs/o-overlap.md` section 5). The
//! loaders now charge their time to sites (`witread.rs`); this loop reads the
//! counters by difference around the region it already times, so every chunk
//! row carries `read_secs`, `decode_secs`, `check_secs` and `construct_secs`
//! beside `build_secs`, with the bytes read and the per-site map underneath.
//! The counters are process-wide, which is exactly right here: under a plan one
//! thread builds one chunk's layer at a time, and the difference across that
//! region is that chunk's.

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

/// One line of a plan: one chunk, and what the loop knows about it before it
/// has been built.
struct ChunkSpec {
    /// this chunk's own arguments, then the ones on the command line. The
    /// chunk's come first because `arg_str` takes the first match it finds, so
    /// putting them there is what lets a chunk decide its own table bits
    /// against shared pins that say nothing about them.
    args: Vec<String>,
    /// the line as it was written, which is what says whether two chunks are
    /// the same layer at the same shape.
    ///
    /// The comparison is on the text, and it is deliberately one-way: two
    /// identical lines resolve to the same layer and the same parameters and
    /// therefore to the same key, which is the direction the cache policy
    /// relies on. Two lines that differ only somewhere the key does not look
    /// -- their `--ser-tag`, say -- do share a key but are read here as
    /// different, and the pair simply goes uncached. That costs a hit nobody
    /// asked for and cannot cost correctness, which is the right way round.
    tail: String,
    /// what this chunk's statement and proof are called on disk
    tag: String,
    /// no later line repeats this one, so no later chunk can want this chunk's
    /// preprocessed tree
    last_use: bool,
    /// this line appears once in the whole plan, so its preprocessed tree is
    /// worth neither caching nor cloning
    unique: bool,
}

/// Read a plan. One chunk per line; `#` starts a comment and blank lines are
/// skipped, so a plan can say what it is.
fn plan_of(path: &str, args: &[String]) -> Vec<ChunkSpec> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("compose-chunks: the plan at {path}: {e}"));
    let tails: Vec<String> = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    assert!(!tails.is_empty(), "compose-chunks: the plan at {path} names no chunks");
    tails
        .iter()
        .enumerate()
        .map(|(i, tail)| {
            let mut a: Vec<String> = tail.split_whitespace().map(str::to_string).collect();
            let tag = arg_str(&a, "--ser-tag").unwrap_or_else(|| format!("chunk{i}"));
            // `--plan-end` closes the line. Nothing reads it; it is there so
            // that a bare flag written last on a line cannot pair with the
            // first shared argument and be mistaken for its value.
            a.push("--plan-end".into());
            a.extend_from_slice(args);
            ChunkSpec {
                args: a,
                tail: tail.clone(),
                tag,
                last_use: !tails[i + 1..].contains(tail),
                unique: tails.iter().filter(|t| *t == tail).count() == 1,
            }
        })
        .collect()
}

/// A chunk's layer, its parameters and its preprocessed-tree key: borrowed
/// from the set built in setup when the loop cycles one shape, owned by the
/// chunk when the loop is following a plan.
struct Shape {
    layer: Layer,
    params: Params,
    vname: String,
    key: u64,
}

enum Built<'a> {
    Cycled(&'a Layer, &'a Params, &'a str, u64),
    Own(Box<Shape>),
}

impl Built<'_> {
    fn layer(&self) -> &Layer {
        match self {
            Built::Cycled(l, ..) => *l,
            Built::Own(s) => &s.layer,
        }
    }
    fn params(&self) -> &Params {
        match self {
            Built::Cycled(_, p, ..) => *p,
            Built::Own(s) => &s.params,
        }
    }
    fn vname(&self) -> &str {
        match self {
            Built::Cycled(_, _, v, _) => *v,
            Built::Own(s) => &s.vname,
        }
    }
    fn key(&self) -> u64 {
        match self {
            Built::Cycled(.., k) => *k,
            Built::Own(s) => s.key,
        }
    }
}

/// Where the loop's chunks come from.
enum Source {
    /// B7b's loop: the layers are built once, in setup, and cycled. One shape,
    /// so one key, so every chunk after the first takes a cached tree -- which
    /// is the number section 5's headline is built from.
    Cycle { layers: Vec<Layer>, params: Params, keys: Vec<u64>, vname: String },
    /// Stream O's loop: one line of the plan per chunk, built when the chunk
    /// is generated.
    Plan(Vec<ChunkSpec>),
}

impl Source {
    fn build(&self, k: usize, tamper: &ProverTamper) -> Built<'_> {
        match self {
            Source::Cycle { layers, params, keys, vname } => {
                let li = k % layers.len();
                Built::Cycled(&layers[li], params, vname, keys[li])
            }
            Source::Plan(specs) => {
                let s = &specs[k];
                let (layer, vname) = crate::layer_of(&s.args);
                let params = crate::params_of(&layer, &s.args, &vname);
                let key = key_of(&layer, &params, tamper);
                Built::Own(Box::new(Shape { layer, params, vname, key }))
            }
        }
    }

    /// Which of the loop's layers this chunk proves. Under a plan every chunk
    /// is its own, so the index is the chunk's.
    fn li(&self, k: usize) -> usize {
        match self {
            Source::Cycle { layers, .. } => k % layers.len(),
            Source::Plan(_) => k,
        }
    }

    fn n_layers(&self) -> usize {
        match self {
            Source::Cycle { layers, .. } => layers.len(),
            Source::Plan(v) => v.len(),
        }
    }

    fn tag(&self, k: usize) -> String {
        match self {
            Source::Cycle { .. } => format!("chunk{k}"),
            Source::Plan(v) => v[k].tag.clone(),
        }
    }

    /// Whether this chunk's preprocessed tree can be handed back once it has
    /// been proved. Never in the cycling loop, where the whole point is that
    /// the next chunk takes it.
    fn last_use(&self, k: usize) -> bool {
        match self {
            Source::Cycle { .. } => false,
            Source::Plan(v) => v[k].last_use,
        }
    }

    /// The key this chunk should be generated under, and `None` when its shape
    /// occurs once and there is therefore nothing to cache for.
    fn pre_key(&self, k: usize, built: &Built<'_>) -> Option<u64> {
        match self {
            Source::Cycle { .. } => Some(built.key()),
            Source::Plan(v) if v[k].unique => None,
            Source::Plan(_) => Some(built.key()),
        }
    }

    /// How many of the plan's chunks share a shape with another and are
    /// therefore generated under a key. The cycling loop has no plan to count,
    /// and every one of its chunks is keyed by construction.
    fn n_keyed(&self) -> Option<usize> {
        match self {
            Source::Cycle { .. } => None,
            Source::Plan(v) => Some(v.iter().filter(|c| !c.unique).count()),
        }
    }

    fn describe(&self) -> String {
        match self {
            Source::Cycle { layers, .. } => {
                layers.iter().map(|l| l.source.as_str()).collect::<Vec<_>>().join("; ")
            }
            Source::Plan(v) => format!("{} chunks from a plan", v.len()),
        }
    }

    fn tails(&self) -> Vec<String> {
        match self {
            Source::Cycle { .. } => vec![],
            Source::Plan(v) => v.iter().map(|s| s.tail.clone()).collect(),
        }
    }
}

/// One generated chunk on its way from the generation thread to the one that
/// will prove it, with what its two halves cost.
struct SendGen<'a> {
    g: compose::Gen,
    k: usize,
    built: Built<'a>,
    build_secs: f64,
    split: crate::witread::Split,
    gen_secs: f64,
}

// The generated chunk holds device pointers; the CUDA context is per process
// and nothing in it is bound to the thread that made it (precache.rs). The
// layer that travels with it is host memory and moves for the same reason it
// was built on that thread: the chunk that proves it is the one that frees it.
unsafe impl Send for SendGen<'_> {}

struct Row {
    k: usize,
    li: usize,
    source: String,
    vname: String,
    params: Params,
    out: compose::Composed,
    ser: crate::Ser,
    /// the wall of this chunk on its own thread: generate + prove + serialize
    /// (serial), or prove + serialize (pipelined, where generation is elsewhere)
    secs: f64,
    /// building the layer: reading the witness off disk and resolving its
    /// parameters. Under a plan this is per chunk and it is not small; the
    /// cycling loop builds its layers once in setup and this is ~0 there.
    build_secs: f64,
    /// where those seconds went: read, decode, check, construct, and the
    /// per-site detail underneath them (stream P)
    split: crate::witread::Split,
    gen_secs: f64,
    upload: Value,
    /// what the card was holding the moment this chunk had been generated and
    /// nothing of it had been proved, and again once its prove had returned.
    /// The first is what a second chunk in flight would have to fit beside.
    mem_after_gen: u64,
    mem_after_prove: u64,
}

/// Write this chunk's statement and proof, and time the write.
///
/// The seconds are recorded under their own name rather than added to
/// `wall/ser_statement` or `wall/ser_proof`, because those two are what
/// `whole_wall` is built from and the one-shot `compose` path this loop is
/// compared against does its file writing outside that sum as well. Timing it
/// separately keeps the boundary the same and still says what the write cost.
fn write_ser(dir: &str, tag: &str, ser: &crate::Ser) {
    let t = Instant::now();
    crate::ser_write(dir, tag, ser);
    crate::util::m4_add("wall/ser_write", t.elapsed().as_secs_f64());
}

pub fn chunks_cmd(args: &[String]) {
    let t_setup = Instant::now();
    // Setup gets its own split. The cycling loop builds its layers here rather
    // than per chunk, and without this the work would be timed by nothing.
    let snap_setup = crate::witread::snapshot();
    let tamper = ProverTamper::default();
    let pipe = crate::util::m4b_mode("B7_M5_PIPE") == 1;
    let ser_dir = arg_str(args, "--ser-dir");

    let src = match arg_str(args, "--plan") {
        Some(path) => Source::Plan(plan_of(&path, args)),
        None => {
            let (layers, vname) = layers_of(args);
            let params = crate::params_of(&layers[0], args, &vname);
            for l in &layers[1..] {
                let p2 = crate::params_of(l, args, &vname);
                assert_eq!(
                    serde_json::to_value(&p2).unwrap(),
                    serde_json::to_value(&params).unwrap(),
                    "compose-chunks: the layers do not share one shape"
                );
            }
            let keys = layers.iter().map(|l| key_of(l, &params, &tamper)).collect();
            Source::Cycle { layers, params, keys, vname }
        }
    };
    // A plan says how many chunks there are; `--chunks` only sets the count
    // for the cycling loop, and saying both is a driver that has drifted.
    let n = match &src {
        Source::Plan(v) => {
            let asked: usize = arg_scalar(args, "--chunks", v.len());
            assert_eq!(asked, v.len(),
                       "compose-chunks: --chunks {asked} against a plan of {} chunks", v.len());
            v.len()
        }
        Source::Cycle { .. } => arg_scalar(args, "--chunks", 4usize),
    };
    let setup_secs = t_setup.elapsed().as_secs_f64();
    let setup_split = crate::witread::since(&snap_setup);
    eprintln!(
        "compose-chunks: {n} chunks over {} layer(s) [{}], pipe={pipe} \
         ircache={} precache={}; setup {setup_secs:.1} s",
        src.n_layers(),
        src.describe(),
        crate::scalars::mode(),
        crate::precache::mode()
    );

    let mut rows: Vec<Row> = Vec::with_capacity(n);
    crate::device::reset();
    let snap_loop = crate::witread::snapshot();
    let t_loop = Instant::now();
    if !pipe {
        for k in 0..n {
            crate::util::m4_set_chunk(k as u64);
            if k > 0 {
                crate::device::reset();
            }
            let t = Instant::now();
            let snap = crate::witread::snapshot();
            let built = src.build(k, &tamper);
            let build_secs = t.elapsed().as_secs_f64();
            let split = crate::witread::since(&snap);
            let pkey = src.pre_key(k, &built);
            let tg = Instant::now();
            let g = compose::generate(built.layer(), built.params(), &tamper, pkey);
            let gen_secs = tg.elapsed().as_secs_f64();
            let mem_after_gen = crate::device::mem_used_bytes();
            let out = compose::prove_gen(built.layer(), built.params(), g);
            let mem_after_prove = crate::device::mem_used_bytes();
            let ser = crate::compose_ser(&out);
            if let Some(dir) = &ser_dir {
                write_ser(dir, &src.tag(k), &ser);
            }
            let secs = t.elapsed().as_secs_f64();
            eprintln!(
                "compose-chunks: chunk {k} ({}) {secs:.3} s, generate {gen_secs:.3} s",
                built.layer().source
            );
            let row = Row {
                k,
                li: src.li(k),
                source: built.layer().source.clone(),
                vname: built.vname().to_string(),
                params: built.params().clone(),
                out,
                ser,
                secs,
                build_secs,
                split,
                gen_secs,
                upload: crate::device::report(),
                mem_after_gen,
                mem_after_prove,
            };
            // The witness goes here, before the next chunk loads its own, and
            // the preprocessed tree with it when no later chunk wants it.
            drop(built);
            if let Some(key) = pkey {
                if src.last_use(k) {
                    crate::precache::forget(key);
                }
            }
            rows.push(row);
        }
    } else {
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<SendGen>(1);
            let (src, tamper) = (&src, &tamper);
            s.spawn(move || {
                for k in 0..n {
                    crate::util::m4_set_chunk(k as u64);
                    let t = Instant::now();
                    let snap = crate::witread::snapshot();
                    let built = src.build(k, tamper);
                    let build_secs = t.elapsed().as_secs_f64();
                    let split = crate::witread::since(&snap);
                    let tg = Instant::now();
                    let g = compose::generate(built.layer(), built.params(), tamper,
                                              src.pre_key(k, &built));
                    let gen_secs = tg.elapsed().as_secs_f64();
                    eprintln!("compose-chunks: generated chunk {k} in {:.3} s \
                               (witness {build_secs:.3} s)", t.elapsed().as_secs_f64());
                    if tx.send(SendGen { g, k, built, build_secs, split, gen_secs }).is_err() {
                        return;
                    }
                }
            });
            for k in 0..n {
                let SendGen { g, k: kk, built, build_secs, split, gen_secs } =
                    rx.recv().expect("the generation thread died");
                assert_eq!(k, kk);
                crate::util::m4_set_chunk(k as u64);
                let pkey = src.pre_key(k, &built);
                let t = Instant::now();
                // Under overlap this is the pair in flight, not this chunk
                // alone: the next chunk is generating while this one proves.
                let mem_after_gen = crate::device::mem_used_bytes();
                let out = compose::prove_gen(built.layer(), built.params(), g);
                let mem_after_prove = crate::device::mem_used_bytes();
                let ser = crate::compose_ser(&out);
                if let Some(dir) = &ser_dir {
                    write_ser(dir, &src.tag(k), &ser);
                }
                let secs = t.elapsed().as_secs_f64();
                eprintln!("compose-chunks: proved chunk {k} ({}) in {secs:.3} s",
                          built.layer().source);
                let row = Row {
                    k,
                    li: src.li(k),
                    source: built.layer().source.clone(),
                    vname: built.vname().to_string(),
                    params: built.params().clone(),
                    out,
                    ser,
                    secs,
                    // All three measured on the generation thread, where they
                    // happened, and reported against the chunk they belong to.
                    build_secs,
                    split,
                    gen_secs,
                    upload: crate::device::report(),
                    mem_after_gen,
                    mem_after_prove,
                };
                drop(built);
                if let Some(key) = pkey {
                    if src.last_use(k) {
                        crate::precache::forget(key);
                    }
                }
                rows.push(row);
            }
        });
    }
    let loop_secs = t_loop.elapsed().as_secs_f64();
    // The loop's own split, taken from the counters rather than by adding the
    // chunks up: the two agree only if every loader call happened inside a
    // chunk's timed build, which is what this is here to show.
    let loop_split = crate::witread::since(&snap_loop);

    // The verifier, after the loop: it is the CPU's and not the prover's wall.
    let mut chunks: Vec<Value> = vec![];
    let mut wholes = vec![];
    let mut stwos = vec![];
    let mut commits = vec![];
    let mut digests = vec![];
    let mut gen_mem: Vec<u64> = vec![];
    let mut prove_mem: Vec<u64> = vec![];
    let mut builds: Vec<f64> = vec![];
    let mut gens: Vec<f64> = vec![];
    let head_source = rows[0].source.clone();
    let head_vname = rows[0].vname.clone();
    let head_params = rows[0].params.clone();
    for r in rows {
        crate::util::m4_set_chunk(r.k as u64);
        let (mut j, accepts) = crate::compose_finish(r.out, r.ser, &r.params);
        assert!(accepts, "compose-chunks: chunk {} rejected: {j}", r.k);
        assert_eq!(j["ref_mismatch"], 0, "chunk {}: witness disagrees with the f64 reference", r.k);
        assert_eq!(j["off_table"], 0, "chunk {}: off-table values", r.k);
        j["upload"] = r.upload;
        // Under overlap the device's byte counters are process-wide atomics
        // that two chunks are writing at once, so what a chunk reports is the
        // loop's running total and not its own. Saying which it is beats
        // publishing a number that looks per-chunk and is not.
        j["upload_scope"] = json!(if pipe { "loop-cumulative" } else { "chunk" });
        j["chunk"] = json!(r.k);
        j["layer"] = json!(r.li);
        j["layer_source"] = json!(r.source);
        j["chunk_secs"] = json!(r.secs);
        j["build_secs"] = json!(r.build_secs);
        r.split.merge_into(&mut j);
        j["generate_secs"] = json!(r.gen_secs);
        j["mem_after_gen"] = json!(r.mem_after_gen);
        j["mem_after_prove"] = json!(r.mem_after_prove);
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
        gen_mem.push(r.mem_after_gen);
        prove_mem.push(r.mem_after_prove);
        builds.push(r.build_secs);
        gens.push(r.gen_secs);
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
        "n": n, "layers": src.n_layers(), "pipe": pipe,
        "setup_secs": setup_secs,
        "loop_secs": loop_secs,
        "per_chunk_loop_secs": loop_secs / n as f64,
        "chunk0_whole": wholes[0],
        "whole_median_rest": med(&rest(&wholes)),
        "stwo_median_rest": med(&rest(&stwos)),
        "commit_median_rest": med(&rest(&commits)),
        "gen_median_rest": med(&rest(&wholes)) - med(&rest(&stwos)),
        "fresh_cells": fresh,
        // A plan's chunks have different shapes, so the sum is the sequence's
        // wall and the medians above are a mixture rather than a rate.
        "whole_sum": wholes.iter().sum::<f64>(),
        "stwo_sum": stwos.iter().sum::<f64>(),
        "commit_sum": commits.iter().sum::<f64>(),
        // Reading the witnesses off disk. `whole_wall` has never counted this
        // -- neither here nor in the one-shot path it is compared against --
        // so a loop whose wall is much larger than its `whole_sum` is holding
        // the difference here, and it is host work that overlap can hide.
        "build_sum": builds.iter().sum::<f64>(),
        // Stream P: the same seconds, split into the four kinds of work the
        // loaders do, with the bytes they read and the per-site detail. Taken
        // over the whole loop, so a term here is comparable to `build_sum` and
        // to nothing outside `loop_secs`.
        "read_sum": loop_split.kind(crate::witread::Kind::Read),
        "decode_sum": loop_split.kind(crate::witread::Kind::Decode),
        "check_sum": loop_split.kind(crate::witread::Kind::Check),
        "construct_sum": loop_split.kind(crate::witread::Kind::Construct),
        // Stream W: getting and giving back the arrays the four kinds above run
        // in. It is its own kind rather than a site under `construct` because
        // what the record has to be able to say is whether the seconds stream P
        // could not attribute are allocation, and a term folded into another
        // kind cannot answer that either way.
        "alloc_sum": loop_split.kind(crate::witread::Kind::Alloc),
        "build_split": loop_split.json(),
        "setup_build_split": setup_split.json(),
        "gen_sum": gens.iter().sum::<f64>(),
        "upload_scope": if pipe { "loop-cumulative" } else { "chunk" },
        "upload_total": crate::device::report(),
        // What the card was holding at each chunk's two boundaries. On the
        // serial arm `mem_after_gen` is one chunk's generated state with
        // nothing else in flight, which is the number that says whether a
        // second chunk would fit beside a prove.
        "mem_after_gen": gen_mem,
        "mem_after_prove": prove_mem,
        "mem_after_gen_max": gen_mem.iter().max().copied().unwrap_or(0),
        "mem_after_prove_max": prove_mem.iter().max().copied().unwrap_or(0),
        "m_cells_per_s": {
            "whole_rest": fresh / med(&rest(&wholes)) / 1e6,
            "loop": fresh / (loop_secs / n as f64) / 1e6,
            "stwo_rest": fresh / med(&rest(&stwos)) / 1e6,
            "commit_rest": fresh / med(&rest(&commits)) / 1e6,
        },
        "digests": digests,
        "plan": src.tails(),
        // How many of the plan's chunks share a shape with another one. On the
        // sequence this is 0: every layer is its own shape, so nothing is
        // cached and nothing is cloned, which is what the 33 processes did.
        "precache_keyed": src.n_keyed(),
        "ircache": crate::scalars::report(),
        "precache": crate::precache::report(),
        "blob": crate::cudagen::blob_report(),
    });
    eprintln!("compose-chunks: {}", serde_json::to_string_pretty(&summary).unwrap());
    emit("compose_chunks", json!({
        "source": head_source, "variant": head_vname,
        "params": serde_json::to_value(&head_params).unwrap(),
        "summary": summary, "chunks": chunks,
    }));
}
