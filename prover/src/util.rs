//! Shared config, timing and JSON helpers.

use serde_json::{Value, json};
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;

// ---------------------------------------------------------------------------
// Security parameters. Fixed for every measurement in B2 (see README §config).
// stwo's own conjectured-soundness accounting is
//     security_bits = pow_bits + log_blowup_factor * n_queries
// so blowup 2 (log_blowup_factor = 1) at 100 conjectured bits means exactly
// 100 FRI queries and no grinding.
// ---------------------------------------------------------------------------
pub const LOG_BLOWUP: u32 = 1;
pub const N_QUERIES: usize = 100;
pub const POW_BITS: u32 = 0;
pub const LOG_LAST_LAYER: u32 = 0;
pub const FOLD_STEP: u32 = 1;

pub fn pcs_config() -> PcsConfig {
    let cfg = PcsConfig {
        pow_bits: POW_BITS,
        fri_config: FriConfig::new(LOG_LAST_LAYER, LOG_BLOWUP, N_QUERIES, FOLD_STEP),
        lifting_log_size: None,
    };
    assert_eq!(cfg.security_bits(), 100, "security parameter drifted");
    cfg
}

/// An on-by-default switch, read once. `=0` turns it off; anything else, the
/// unset environment included, is on.
pub fn switch_on(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(true)
}

pub fn store_coeffs() -> bool {
    switch_on("ZK_STORE_COEFFICIENTS")
}

/// B12 stream HH: keep every committed column's coefficients and drop its
/// blown-up evaluations once the tree's root is mixed, rebuilding them per
/// phase. Opt-in, unlike the on-by-default switches, because it is a
/// residency trade and not a default.
///
/// It refuses loudly when the coefficients are off: they are the only thing a
/// dropped extension can be rebuilt from, and a run that proved anyway would be
/// reading a configuration it did not have.
pub fn lde_only() -> bool {
    let on = std::env::var("ZK_LDE_ONLY")
        .map(|v| v != "0")
        .unwrap_or(false);
    assert!(
        !on || store_coeffs(),
        "ZK_LDE_ONLY=1 needs ZK_STORE_COEFFICIENTS on: the coefficients are the only \
         thing the dropped evaluations can be rebuilt from"
    );
    on
}

/// The host-side timeline (B7a milestone 4, 18.2).
///
/// One named accumulator, reported as `host_phases` in every JSON, so where
/// the wall goes *outside* `prove_ex` is a measurement rather than an estimate.
/// It takes a lock, so it is called once per column or once per phase and
/// never per row; at the design point that is ~8k locks against a 5 s wall.
///
/// B7b (README 26): keyed by chunk. A thread names the chunk it is working on
/// (`set_chunk`) and every phase it adds lands under that chunk, so a
/// pipelined loop -- chunk k proving on one thread while chunk k+1 generates on
/// another -- still reports each chunk's phases as its own. A single proof is
/// chunk 0 and sees no difference.
type PhaseMap = std::collections::BTreeMap<&'static str, (u64, f64)>;
static PHASE_ACC: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<u64, PhaseMap>>,
> = std::sync::OnceLock::new();

thread_local! {
    static CHUNK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn set_chunk(k: u64) {
    CHUNK.with(|c| c.set(k));
}

pub fn current_chunk() -> u64 {
    CHUNK.with(|c| c.get())
}

fn phase_acc() -> &'static std::sync::Mutex<std::collections::BTreeMap<u64, PhaseMap>> {
    PHASE_ACC.get_or_init(|| std::sync::Mutex::new(Default::default()))
}

type ByteMap = std::collections::BTreeMap<&'static str, u64>;
static BYTE_ACC: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<u64, ByteMap>>,
> = std::sync::OnceLock::new();

fn byte_acc() -> &'static std::sync::Mutex<std::collections::BTreeMap<u64, ByteMap>> {
    BYTE_ACC.get_or_init(|| std::sync::Mutex::new(Default::default()))
}

pub fn add_phase(name: &'static str, secs: f64) {
    let k = current_chunk();
    // B11 stream V, lever 1: the same call that accumulates a phase's seconds
    // stamps its *end* on the memory timeline. `add_phase` already runs once per
    // phase and never per row, and it already carries the names the wall report
    // uses, so the timeline needs no second set of boundaries -- a phase's
    // bracket is this mark and the one before it. Off (no `ZK_MEMORY_TIMELINE`) this
    // is one `OnceLock` read and no allocation.
    if stwo::prover::lookups::gkr_prover::memmark::on() {
        stwo::prover::lookups::gkr_prover::memmark::mark(&format!("{name} {secs:.6}"));
    }
    let mut g = phase_acc().lock().unwrap();
    let e = g.entry(k).or_default().entry(name).or_insert((0, 0.0));
    e.0 += 1;
    e.1 += secs;
}

/// One boundary on the memory timeline that is not a phase's end.
///
/// `add_phase` covers every phase the wall report names; this is for the few
/// places where the interesting instant is a *start* -- the GKR's input layers
/// arriving on the card, say -- or where a phase is not timed at all.
pub fn mark_timeline(name: &str) {
    stwo::prover::lookups::gkr_prover::memmark::mark(name);
}

/// B11 stream F: the byte counter beside the timer.
///
/// A phase's achieved GB/s is only readable off one dump if the bytes it moved
/// are recorded where its seconds are. Same keying as `add_phase` -- per chunk,
/// per name -- reported as `host_bytes`.
pub fn add_bytes(name: &'static str, bytes: u64) {
    let k = current_chunk();
    let mut g = byte_acc().lock().unwrap();
    let e = g.entry(k).or_default().entry(name).or_insert(0);
    *e += bytes;
}

/// The current chunk's byte counters.
pub fn bytes_report() -> Value {
    let k = current_chunk();
    let g = byte_acc().lock().unwrap();
    let mut o = serde_json::Map::new();
    if let Some(m) = g.get(&k) {
        for (k, b) in m.iter() {
            o.insert((*k).to_string(), json!(b));
        }
    }
    Value::Object(o)
}

/// The current chunk's phases.
pub fn phases_report() -> Value {
    let k = current_chunk();
    let g = phase_acc().lock().unwrap();
    let mut o = serde_json::Map::new();
    if let Some(m) = g.get(&k) {
        for (k, (n, s)) in m.iter() {
            o.insert((*k).to_string(), json!({"calls": n, "secs": s}));
        }
    }
    Value::Object(o)
}

/// The first CUDA call in the process pays for context creation and the
/// fatbin load. Attributing that to the twiddles would make a per-process cost
/// look like a per-proof one, so it is drawn out on its own line.
pub fn touch_device() {
    #[cfg(feature = "cuda")]
    let _ = stwo::stwo_cuda::base_field_vec::BaseFieldVec::from_vec(
        vec![stwo::core::fields::m31::M31::from_u32_unchecked(0); 1],
    );
}

/// The smallest column the parallel gather is used for, `ZK_PARALLEL_GATHER_MIN`.
///
/// Default 16, swept in 18.4: below 2^16 rows rayon's per-column cost exceeds
/// what the gather saves, and at the design point 2,641 of the 4,612 columns
/// are that small. `=0` is every column, which is what 18.3 measured.
pub fn parallel_gather_min() -> u32 {
    std::env::var("ZK_PARALLEL_GATHER_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
}

/// The opposite default: a switch that is off unless the environment asks for
/// it. True means "not asked for", so a caller reads it as the early return.
pub fn switch_off(name: &str) -> bool {
    !std::env::var(name).map(|v| v != "0").unwrap_or(false)
}

/// A path's tri-state mode, read once: 0 the host path, 1 the device path, 2
/// both, with every cell of the second asserted against the first. A mode
/// selects a path that has still to prove itself equal to the one it replaces,
/// and the default has to be the proven one -- so a mode is born off, and
/// then, in the same commit that lands its digest gate, its default flips to
/// the value that gated. `default_path_mode` is that table.
pub fn path_mode(name: &str) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| default_path_mode(name))
}

/// The proven default for each tri-state mode, on the CUDA build.
///
/// B11 stream N, step 0. Every mode named here ran in stream M's seq4 and in
/// the three sequences before it, with all 66 proof digests byte-identical
/// back to stream I's, so on this build the proven value *is* the device
/// value and the default table should say so. Before this the values lived
/// only in a copy-pasted env string, which is how `ZK_DEVICE_ACCUMULATORS`, `ZK_DEVICE_PREPROCESSED`
/// and `ZK_GENERATED_LOGUP` came to sit dark for four streams (M's Record (1)):
/// b-port's one measured rejection dropped `ACC` and `PRE` alongside the
/// actually-broken `WITNESS_GEN`, and the reduced string then propagated verbatim
/// through four drivers.
///
/// Two kinds of value are deliberately absent. Mirror modes (`=2`, both paths
/// run with every cell asserted) and the `*MUT` mutations are debugging
/// instruments rather than paths, so they stay explicit-only. And a mode that
/// is not listed here is a path that has not yet gated: it is born off,
/// exactly as before, and graduates into this table in the commit that lands
/// its own gate.
#[cfg(feature = "cuda")]
fn default_path_mode(name: &str) -> u32 {
    match name {
        // The component witness generations, the shared tables, the pinned
        // staging pool, the preprocessed columns, the accumulators and the
        // residual stream's witness generation: all on.
        "ZK_DEVICE_WITNESS_GEN_DEQUANT"
        | "ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS"
        | "ZK_DEVICE_TABLES"
        | "ZK_DEVICE_STAGING_POOL"
        | "ZK_DEVICE_PREPROCESSED"
        | "ZK_DEVICE_ACCUMULATORS"
        | "ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM"
        // B11 stream N, lever 3: the residual stream's Horner bindings. Graduated here in the
        // commit that landed its gate -- the mode-2 mirror clean on all four
        // residual-stream components (40 columns, 401,408 cells, 10 claims), three aimed
        // controls each shown to bite, and the gate chunk's digest unmoved.
        | "ZK_DEVICE_RESIDUAL_STREAM_BINDING" => 1,
        // The two milestone-5 caches. They are in seq4's family env like
        // everything above, and they are what makes the table *seq4's* env
        // rather than its device subset: the lowering cache (`scalars.rs`) and
        // the preprocessed-tree clone (`precache.rs`). Neither is a device
        // path -- they are digest-neutral by construction, each with a mode-2
        // mirror that says so -- but leaving them off here would have made a
        // run with an empty env a *different configuration* from seq4, and a
        // slower one, which is the precise failure this table exists to stop.
        "ZK_LOWERING_CACHE" | "ZK_PREPROCESSED_TREE_CACHE" => 1,
        // B11 stream O, lever 3: the chunk loop generates chunk k+1 while it
        // proves chunk k. Graduated here in the commit that lands its gate,
        // which is the seq6 A/B at *both* contexts -- 33 chunks
        // each, the head included, all 66 digests byte-identical to seq5's and
        // both chains 31/31 from the loop's own serialized bytes, with the
        // pipelined arm winning at both (`loop_secs` 117.27 -> 71.58 s at
        // C=4096 and 200.18 -> 117.47 s at C=16384). The memory the second
        // chunk in flight costs is what gated this for a session: it is
        // +832 MiB at C=4096 and +2,560 MiB at C=16384 against a 143,771 MiB
        // card, measured, and the worst pair the sequence contains (L0 and L1
        // at C=16384, 76,823 MiB apiece on their own) peaked at 76,823 MiB
        // together. The reference backend is untouched -- `default_path_mode` is
        // all-zero without the `cuda` feature -- so the digest comparison
        // still has something to be a comparison against.
        "ZK_PIPELINE" => 1,
        // The generated fraction builders. `3` keeps 19.3's meaning: the
        // generated builder runs and is checked against the hand-written one.
        "ZK_GENERATED_LOGUP" => 3,
        _ => 0,
    }
}

/// Without the `cuda` feature every mode stays off. This is the one hard
/// constraint on step 0's table: the SimdBackend build is the reference that
/// every digest comparison is made against, and a device default on that backend
/// would call FFI stubs. The reference backend is untouched by construction.
#[cfg(not(feature = "cuda"))]
fn default_path_mode(_name: &str) -> u32 {
    0
}

/// Every mode this build knows, with the value it resolved to and where that
/// value came from.
///
/// The env string no longer names the configuration, so the report has to: a
/// recorded number must name its configuration even when the env is empty.
/// `source` is `"env"` when the run set the variable and `"default"` when it
/// took the table above.
pub fn modes_json() -> Value {
    let mut modes = serde_json::Map::new();
    let mut put = |name: &str, value: u32| {
        modes.insert(
            name.trim_start_matches("ZK_").to_string(),
            json!({
                "value": value,
                "source": if std::env::var(name).is_ok() { "env" } else { "default" },
            }),
        );
    };
    for name in [
        "ZK_DEVICE_WITNESS_GEN_DEQUANT",
        "ZK_DEVICE_WITNESS_GEN_ATTENTION_FREIVALDS",
        "ZK_DEVICE_TABLES",
        "ZK_DEVICE_STAGING_POOL",
        "ZK_DEVICE_PREPROCESSED",
        "ZK_DEVICE_ACCUMULATORS",
        "ZK_DEVICE_WITNESS_GEN_RESIDUAL_STREAM",
        "ZK_DEVICE_RESIDUAL_STREAM_BINDING",
        "ZK_GENERATED_LOGUP",
        "ZK_PIPELINE",
        "ZK_PREPROCESSED_TREE_CACHE",
        "ZK_LOWERING_CACHE",
    ] {
        put(name, path_mode(name));
    }
    // `ZK_DEVICE_WITNESS_GEN_KEY_TILE` is the one mode whose fallback is another mode
    // rather than the table, so it is asked for through its own reader. This
    // is the switch's global value; which witness generation a *particular* key tile takes also
    // depends on its layout (`dev_attention_freivalds_witness_gen::key_tile_witness_gen_mode`).
    put("ZK_DEVICE_WITNESS_GEN_KEY_TILE", crate::dev_attention_freivalds_witness_gen::key_tile_mode());
    // `ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK` has not gated and defaults off, but a run that
    // sets it must still name it here, or its JSON claims a configuration it
    // did not run.
    put("ZK_DEVICE_WITNESS_GEN_DIAGONAL_BLOCK", crate::dev_attention_freivalds_witness_gen::diagonal_block_mode());
    // The plain on/off switches resolve the same way and belong in the same
    // object. All of them default on and all of them predate this stream.
    for name in ["ZK_PARALLEL_GATHER", "ZK_COMMIT_BY_REFERENCE", "ZK_PARALLEL_SCAN",
                 "ZK_KERNEL_REFERENCE_CHECK"] {
        put(name, u32::from(switch_on(name)));
    }
    // B12 stream HH: an opt-in switch, off unless the env asks for it, so every
    // run JSON says which residency regime it ran under.
    put("ZK_LDE_ONLY", u32::from(lde_only()));
    Value::Object(modes)
}

pub fn config_json() -> Value {
    let cfg = pcs_config();
    json!({
        "log_blowup_factor": cfg.fri_config.log_blowup_factor,
        "blowup_factor": 1u32 << cfg.fri_config.log_blowup_factor,
        "n_queries": cfg.fri_config.n_queries,
        "pow_bits": cfg.pow_bits,
        "log_last_layer_degree_bound": cfg.fri_config.log_last_layer_degree_bound,
        "fold_step": cfg.fri_config.line_fold_step,
        "security_bits_conjectured": cfg.security_bits(),
        "hash": "Blake2s (Blake2sM31MerkleChannel)",
        // B7a milestone 3: which of the three changes this run had on.
        "m3": {
            "store_coeffs": switch_on("ZK_STORE_COEFFICIENTS"),
            "device_merkle_reduce": switch_on("ZK_DEVICE_MERKLE"),
            "fri_gather": switch_on("ZK_FRI_GATHER"),
            "prewarm": switch_on("ZK_PREWARM"),
            "lde_only": lde_only(),
            "merkle_mode": std::env::var("ZK_DEVICE_MERKLE").unwrap_or_else(|_| "1".into()),
            "launch_sync": std::env::var("ZK_LAUNCH_SYNC").is_ok()
        },
    })
}

/// Peak resident set size of this process, in bytes (Linux `VmHWM`).
pub fn peak_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Current resident set size, in bytes.
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

pub fn mem_total_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1).map(|v| v.to_string()))
        })
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// The card this process can see, as `nvidia-smi` names it (its model name and
/// total memory, `"<name>, <total> MiB"`), or null where there is none.  Open
/// item 9: every timing constant names its card and until this no artifact
/// recorded it -- the two boxes were separable only by host RAM and witness
/// path.  Asked once per process; `emit` runs per chunk in the loop.
/// What `nvidia-smi` reports, asked once: the first device's name and memory,
/// the driver version, and the CUDA version the driver supports.  `None` for
/// each when there is no `nvidia-smi` or it answers nothing, which is the
/// case on a machine without a device.
fn nvidia_smi() -> &'static (Option<String>, Option<String>, Option<String>) {
    static SMI: std::sync::OnceLock<(Option<String>, Option<String>, Option<String>)> =
        std::sync::OnceLock::new();
    SMI.get_or_init(|| {
        let query = |fields: &str| -> Option<String> {
            let out = std::process::Command::new("nvidia-smi")
                .args([&format!("--query-gpu={fields}"), "--format=csv,noheader"])
                .output()
                .ok()?;
            let s = String::from_utf8_lossy(&out.stdout);
            let first = s.lines().next()?.trim();
            (!first.is_empty()).then(|| first.to_string())
        };
        let gpu = query("name,memory.total");
        let driver = query("driver_version");
        // the CUDA version is only in the banner: `... CUDA Version: 12.9 ...`
        let cuda = std::process::Command::new("nvidia-smi").output().ok().and_then(|out| {
            let s = String::from_utf8_lossy(&out.stdout);
            let at = s.find("CUDA Version:")?;
            let rest = s[at + "CUDA Version:".len()..].trim_start();
            let v: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
            (!v.is_empty()).then_some(v)
        });
        (gpu, driver, cuda)
    })
}

fn gpu_name() -> Option<String> {
    nvidia_smi().0.clone()
}

pub fn env_json() -> Value {
    json!({
        "threads": rayon::current_num_threads(),
        "mem_total_bytes": mem_total_bytes(),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "simd_n_lanes": stwo::prover::backend::simd::m31::N_LANES,
        "gpu": gpu_name(),
    })
}

// ---------------------------------------------------------------------------
// Provenance: what a JSON document says about the run that produced it.  The
// block is assembled by `emit` from what the run noted along the way: every
// proof digest (`prov_note_proof`) with its binding table, and every
// `--unsound-skip-*` flag on the command line.  Nothing here enters a
// statement, a transcript or a proof, so the digest is unaffected.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Prov {
    digests: Vec<String>,
    bindings: Vec<Value>,
    dumps: Vec<Value>,
}

fn prov() -> &'static std::sync::Mutex<Prov> {
    static P: std::sync::OnceLock<std::sync::Mutex<Prov>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::Mutex::new(Prov::default()))
}

/// One proof was produced: its digest and its binding table (`Value::Null`
/// until the plan reports one).  A chunk loop calls this once per chunk.
pub fn prov_note_proof(digest: &str, binding: Value) {
    let mut p = prov().lock().unwrap();
    p.digests.push(digest.to_string());
    p.bindings.push(binding);
}

/// The identity of one dump: the per-array digests its manifest lists, rolled
/// into one number.
///
/// A dump's `manifest.json` already carries an entry per array under `files` --
/// the name, the element count, the dtype, the shape and the array's sha256 --
/// and nothing rolled them into a number a record could quote. So a recorded
/// proof digest named the commit that made it and not the dump it was made
/// from, and when the 8B reference digest moved with no prover commit behind it
/// (issue #130), neither record could say that what had moved was the dump.
///
/// The roll-up is over the arrays alone: the entries sorted by name, one line
/// each, `name values dtype shape sha256`, after a version tag. Not over the
/// manifest's own bytes, which carry the ids path that host read and the torch
/// version it ran, so two hosts that wrote the same arrays would get two
/// numbers out of them; and not over `written` or `link`, which say what is on
/// disk (`--slim`, `--share`) and not what the extraction computed.
///
/// It is what the manifest **claims** about its arrays, not a re-reading of
/// them: the extractor hashes each array as it writes it, and a 4.2 GB dump is
/// not re-hashed here. A reader who wants the bytes checked hashes the files.
///
/// `harness/witness/extract.py` rolls the same lines up the same way and writes
/// the result into the manifest as `dump_sha256`. This reads the `files` map
/// instead of that field, so a dump written before the field existed still has
/// an identity, and a manifest whose two disagree is reported rather than
/// believed. `None` for a manifest with no `files` map at all.
pub fn dump_digest(man: &Value) -> Option<String> {
    use sha2::{Digest, Sha256};
    let files = man.get("files")?.as_object()?;
    let mut names: Vec<&String> = files.keys().collect();
    names.sort();
    let mut h = Sha256::new();
    h.update(b"zk-dump-v1\n");
    for n in names {
        let f = &files[n];
        let shape: Vec<String> = f["shape"].as_array()
            .map(|a| a.iter().map(|v| v.to_string()).collect())
            .unwrap_or_default();
        h.update(format!("{n} {} {} {} {}\n", f["values"], f["dtype"].as_str().unwrap_or(""),
                         shape.join(","), f["sha256"].as_str().unwrap_or("")).as_bytes());
    }
    Some(format!("{:x}", h.finalize()))
}

/// This run read a dump: name it in the provenance block, by directory and by
/// digest, with what its manifest says about where it came from.
///
/// Called once per dump directory from `witread::read_manifest`, which is how
/// every reader of a dump opens one. A directory read twice -- the layer and
/// its key tiles both open the manifest -- is one dump and is noted once.
pub fn prov_note_dump(dir: &str, man: &Value) {
    let sha = dump_digest(man);
    let mut p = prov().lock().unwrap();
    if p.dumps.iter().any(|d| d["dir"] == json!(dir)) {
        return;
    }
    let mut e = json!({
        "dir": dir,
        "sha256": sha,
        "layer": man["layer"],
        "block": man["block"],
        // the ids the dump was extracted from and the attention calibration it
        // was quantized under, as the manifest names them: both null on a dump
        // written before the extractor recorded them
        "ids_from": man["ids_from"],
        "attn_ranges": man["attn_ranges"],
    });
    // a manifest that states its own roll-up and does not agree with the arrays
    // it lists has been edited since it was written, which is worth saying
    if man["dump_sha256"].is_string() && man["dump_sha256"].as_str().map(String::from) != sha {
        e["dump_sha256_stated"] = man["dump_sha256"].clone();
    }
    p.dumps.push(e);
}

/// The unsound switches on the command line, as their names: every
/// `--unsound-skip-<name>`, and `--unsound-argmax-lax` as `argmax-lax`.  The
/// lax argmax turns no gadget off -- it proves a weaker statement in its place
/// (`crate::ARGMAX_LAX`) -- so it is listed here for the same reason the skips
/// are: a run that used it is a control, and `all_bound` is false for it.
pub fn unsound_skips() -> Vec<String> {
    std::env::args()
        .filter_map(|a| {
            a.strip_prefix("--unsound-skip-")
                .map(|s| s.to_string())
                .or_else(|| (a == "--unsound-argmax-lax").then(|| "argmax-lax".to_string()))
        })
        .collect()
}

/// The binding tables of every proof this run made, folded into one: an edge
/// is `unbound` if any proof left it unbound, else `bound` if any proof bound
/// it, else `absent`.  `Null` when no proof reported a table.
fn fold_bindings(tables: &[Value]) -> Value {
    let tables: Vec<&Value> = tables.iter().filter(|t| !t.is_null()).collect();
    if tables.is_empty() {
        return Value::Null;
    }
    let mut edges: Vec<Value> = vec![];
    if let Some(first) = tables[0]["edges"].as_array() {
        for (i, e0) in first.iter().enumerate() {
            let states: Vec<&str> = tables.iter()
                .filter_map(|t| t["edges"].get(i).and_then(|e| e["state"].as_str()))
                .collect();
            let state = if states.contains(&"unbound") {
                "unbound"
            } else if states.contains(&"bound") {
                "bound"
            } else {
                "absent"
            };
            let why = tables.iter()
                .filter_map(|t| t["edges"].get(i))
                .find(|e| e["state"] == "unbound")
                .map(|e| e["why"].clone())
                .unwrap_or(Value::Null);
            edges.push(json!({"n": e0["n"], "edge": e0["edge"], "state": state, "why": why}));
        }
    }
    let unbound: Vec<Value> = edges.iter().filter(|e| e["state"] == "unbound")
        .map(|e| e["n"].clone()).collect();
    json!({"proofs": tables.len(), "edges": edges, "all_bound": unbound.is_empty(),
           "unbound": unbound})
}

pub fn provenance_json() -> Value {
    let p = prov().lock().unwrap();
    let mut args = std::env::args();
    let binary = args.next().and_then(|a| {
        std::path::Path::new(&a).file_name().map(|f| f.to_string_lossy().into_owned())
    });
    let (_, driver, cuda) = nvidia_smi();
    json!({
        "commit": env!("ZK_SOURCE_COMMIT"),
        "dirty": match env!("ZK_SOURCE_DIRTY") {
            "true" => json!(true),
            "false" => json!(false),
            _ => Value::Null,
        },
        "binary": binary,
        "args": args.collect::<Vec<String>>(),
        "backend": if cfg!(feature = "cuda") { "cuda" } else { "simd" },
        "gpu": gpu_name(),
        "driver_version": driver,
        "cuda_version": cuda,
        "proof_sha256": if p.digests.len() == 1 { json!(p.digests[0]) } else { Value::Null },
        "proof_sha256s": if p.digests.len() > 1 { json!(p.digests) } else { Value::Null },
        // issue #130: which dump this run was made from. `dump_sha256` is the
        // one a record quotes when there is one dump to quote, the way
        // `proof_sha256` is; `dumps` names every one a chunk loop read, and is
        // empty for a run on a synthetic segment, which reads none.
        "dump_sha256": if p.dumps.len() == 1 { p.dumps[0]["sha256"].clone() } else { Value::Null },
        "dumps": json!(p.dumps),
        "unsound_skips": unsound_skips(),
        "binding": fold_bindings(&p.bindings),
    })
}

pub fn emit(name: &str, body: Value) {
    let out = json!({
        "bench": name,
        "env": env_json(),
        "modes": modes_json(),
        "security": config_json(),
        "provenance": provenance_json(),
        "result": body,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
