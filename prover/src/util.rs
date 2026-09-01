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

/// B7a milestone 3's switches, read once. `=0` turns one off; the default is on.
pub fn m3_switch(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(true)
}

pub fn m3_store_coeffs() -> bool {
    m3_switch("B7_M3_COEFFS")
}

/// B7a milestone 4: the host-side timeline (18.2).
///
/// One named accumulator, reported as `host_phases` in every JSON, so where
/// the wall goes *outside* `prove_ex` is a measurement rather than an estimate.
/// It takes a lock, so it is called once per column or once per phase and
/// never per row; at the design point that is ~8k locks against a 5 s wall.
///
/// B7b (README 26): keyed by chunk. A thread names the chunk it is working on
/// (`m4_set_chunk`) and every phase it adds lands under that chunk, so a
/// pipelined loop -- chunk k proving on one thread while chunk k+1 generates on
/// another -- still reports each chunk's phases as its own. A single proof is
/// chunk 0 and sees no difference.
type M4Map = std::collections::BTreeMap<&'static str, (u64, f64)>;
static M4_ACC: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<u64, M4Map>>,
> = std::sync::OnceLock::new();

thread_local! {
    static M4_CHUNK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn m4_set_chunk(k: u64) {
    M4_CHUNK.with(|c| c.set(k));
}

pub fn m4_chunk() -> u64 {
    M4_CHUNK.with(|c| c.get())
}

fn m4_acc() -> &'static std::sync::Mutex<std::collections::BTreeMap<u64, M4Map>> {
    M4_ACC.get_or_init(|| std::sync::Mutex::new(Default::default()))
}

type M4BMap = std::collections::BTreeMap<&'static str, u64>;
static M4B_ACC: std::sync::OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<u64, M4BMap>>,
> = std::sync::OnceLock::new();

fn m4b_acc() -> &'static std::sync::Mutex<std::collections::BTreeMap<u64, M4BMap>> {
    M4B_ACC.get_or_init(|| std::sync::Mutex::new(Default::default()))
}

pub fn m4_add(name: &'static str, secs: f64) {
    let k = m4_chunk();
    // B11 stream V, lever 1: the same call that accumulates a phase's seconds
    // stamps its *end* on the memory timeline. `m4_add` already runs once per
    // phase and never per row, and it already carries the names the wall report
    // uses, so the timeline needs no second set of boundaries -- a phase's
    // bracket is this mark and the one before it. Off (no `B11_MEMMARK`) this
    // is one `OnceLock` read and no allocation.
    if stwo::prover::lookups::gkr_prover::memmark::on() {
        stwo::prover::lookups::gkr_prover::memmark::mark(&format!("{name} {secs:.6}"));
    }
    let mut g = m4_acc().lock().unwrap();
    let e = g.entry(k).or_default().entry(name).or_insert((0, 0.0));
    e.0 += 1;
    e.1 += secs;
}

/// One boundary on the memory timeline that is not a phase's end.
///
/// `m4_add` covers every phase the wall report names; this is for the few
/// places where the interesting instant is a *start* -- the GKR's input layers
/// arriving on the card, say -- or where a phase is not timed at all.
pub fn m4_mark(name: &str) {
    stwo::prover::lookups::gkr_prover::memmark::mark(name);
}

/// B11 stream F: the byte counter beside the timer.
///
/// A phase's achieved GB/s is only readable off one dump if the bytes it moved
/// are recorded where its seconds are. Same keying as `m4_add` -- per chunk,
/// per name -- reported as `host_bytes`.
pub fn m4_bytes(name: &'static str, bytes: u64) {
    let k = m4_chunk();
    let mut g = m4b_acc().lock().unwrap();
    let e = g.entry(k).or_default().entry(name).or_insert(0);
    *e += bytes;
}

/// The current chunk's byte counters.
pub fn m4_bytes_report() -> Value {
    let k = m4_chunk();
    let g = m4b_acc().lock().unwrap();
    let mut o = serde_json::Map::new();
    if let Some(m) = g.get(&k) {
        for (k, b) in m.iter() {
            o.insert((*k).to_string(), json!(b));
        }
    }
    Value::Object(o)
}

/// The current chunk's phases.
pub fn m4_report() -> Value {
    let k = m4_chunk();
    let g = m4_acc().lock().unwrap();
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
pub fn m4_touch_device() {
    #[cfg(feature = "cuda")]
    let _ = stwo::stwo_cuda::base_field_vec::BaseFieldVec::from_vec(
        vec![stwo::core::fields::m31::M31::from_u32_unchecked(0); 1],
    );
}

/// The smallest column the parallel gather is used for, `B7_M4_PARMIN`.
///
/// Default 16, swept in 18.4: below 2^16 rows rayon's per-column cost exceeds
/// what the gather saves, and at the design point 2,641 of the 4,612 columns
/// are that small. `=0` is every column, which is what 18.3 measured.
pub fn m4_parmin() -> u32 {
    std::env::var("B7_M4_PARMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
}

/// Milestone 4's switches, read once. `=0` turns one off; the default is on.
pub fn m4_switch(name: &str) -> bool {
    std::env::var(name).map(|v| v != "0").unwrap_or(true)
}

/// The opposite default: a switch that is off unless the environment asks for
/// it. True means "not asked for", so a caller reads it as the early return.
pub fn m4_switch_off(name: &str) -> bool {
    !std::env::var(name).map(|v| v != "0").unwrap_or(false)
}

/// Milestone 4's tri-state modes, read once. A mode selects a path that has
/// still to prove itself equal to the one it replaces, and the default has to
/// be the proven one -- so a mode is born off, and then, in the same commit
/// that lands its digest gate, its default flips to the value that gated.
/// `m4b_default` is that table.
pub fn m4b_mode(name: &str) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| m4b_default(name))
}

/// The proven default for each tri-state mode, on the CUDA build.
///
/// B11 stream N, step 0. Every mode named here ran in stream M's seq4 and in
/// the three sequences before it, with all 66 proof digests byte-identical
/// back to stream I's, so on this build the proven value *is* the device
/// value and the default table should say so. Before this the values lived
/// only in a copy-pasted env string, which is how `B7_M4_ACC`, `B7_M4_PRE`
/// and `B7_M4_LOGUPIR` came to sit dark for four streams (M's Record (1)):
/// b-port's one measured rejection dropped `ACC` and `PRE` alongside the
/// actually-broken `WFILL`, and the reduced string then propagated verbatim
/// through four drivers.
///
/// Two kinds of value are deliberately absent. Mirror modes (`=2`, both paths
/// run with every cell asserted) and the `*MUT` mutations are debugging
/// instruments rather than paths, so they stay explicit-only. And a mode that
/// is not listed here is a path that has not yet gated: it is born off,
/// exactly as before, and graduates into this table in the commit that lands
/// its own gate.
#[cfg(feature = "cuda")]
fn m4b_default(name: &str) -> u32 {
    match name {
        // The region fills, the witness fills, the shared tables, the pinned
        // staging pool, the preprocessed columns, the accumulators and the
        // glue's fill: all on.
        "B7_M4_FILL"
        | "B7_M4_WFILL"
        | "B7_M4_TABDEV"
        | "B7_M4_POOL"
        | "B7_M4_PRE"
        | "B7_M4_ACC"
        | "B7_M4_GFILL"
        // B11 stream N, lever 3: the glue's Horner ties. Graduated here in the
        // commit that landed its gate -- the mode-2 mirror clean on all four
        // glue regions (40 columns, 401,408 cells, 10 claims), three aimed
        // controls each shown to bite, and the gate chunk's digest unmoved.
        | "B7_M4_GTIE" => 1,
        // The two milestone-5 caches. They are in seq4's family env like
        // everything above, and they are what makes the table *seq4's* env
        // rather than its device subset: the lowering cache (`scalars.rs`) and
        // the preprocessed-tree clone (`precache.rs`). Neither is a device
        // path -- they are digest-neutral by construction, each with a mode-2
        // mirror that says so -- but leaving them off here would have made a
        // run with an empty env a *different configuration* from seq4, and a
        // slower one, which is the precise failure this table exists to stop.
        "B7_M5_IRCACHE" | "B7_M5_PRECACHE" => 1,
        // B11 stream O, lever 3: the chunk loop generates chunk k+1 while it
        // proves chunk k. Graduated here in the commit that lands its gate,
        // which is the seq6 A/B on the H200 at *both* contexts -- 33 chunks
        // each, the head included, all 66 digests byte-identical to seq5's and
        // both chains 31/31 from the loop's own serialized bytes, with the
        // pipelined arm winning at both (`loop_secs` 117.27 -> 71.58 s at
        // C=4096 and 200.18 -> 117.47 s at C=16384). The memory the second
        // chunk in flight costs is what gated this for a session: it is
        // +832 MiB at C=4096 and +2,560 MiB at C=16384 against a 143,771 MiB
        // card, measured, and the worst pair the sequence contains (L0 and L1
        // at C=16384, 76,823 MiB apiece on their own) peaked at 76,823 MiB
        // together. The reference arm is untouched -- `m4b_default` is
        // all-zero without the `cuda` feature -- so the digest comparison
        // still has something to be a comparison against.
        "B7_M5_PIPE" => 1,
        // The generated fraction builders. `3` keeps 19.3's meaning: the
        // generated builder runs and is checked against the hand-written one.
        "B7_M4_LOGUPIR" => 3,
        _ => 0,
    }
}

/// Without the `cuda` feature every mode stays off. This is the one hard
/// constraint on step 0's table: the SimdBackend build is the reference that
/// every digest comparison is made against, and a device default on that arm
/// would call FFI stubs. The reference arm is untouched by construction.
#[cfg(not(feature = "cuda"))]
fn m4b_default(_name: &str) -> u32 {
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
            name.trim_start_matches("B7_").to_string(),
            json!({
                "value": value,
                "source": if std::env::var(name).is_ok() { "env" } else { "default" },
            }),
        );
    };
    for name in [
        "B7_M4_FILL",
        "B7_M4_WFILL",
        "B7_M4_TABDEV",
        "B7_M4_POOL",
        "B7_M4_PRE",
        "B7_M4_ACC",
        "B7_M4_GFILL",
        "B7_M4_GTIE",
        "B7_M4_LOGUPIR",
        "B7_M5_PIPE",
        "B7_M5_PRECACHE",
        "B7_M5_IRCACHE",
    ] {
        put(name, m4b_mode(name));
    }
    // `B7_M4_WFILL_BAND` is the one mode whose fallback is another mode
    // rather than the table, so it is asked for through its own reader. This
    // is the switch's global value; which fill a *particular* band takes also
    // depends on its layout (`devwfill::band_fill_mode`).
    put("B7_M4_WFILL_BAND", crate::devwfill::band_mode());
    // `B7_M4_WFILL_CORNER` has not gated and defaults off, but a run that
    // sets it must still name it here, or its JSON claims a configuration it
    // did not run.
    put("B7_M4_WFILL_CORNER", crate::devwfill::corner_mode());
    // The plain on/off switches resolve the same way and belong in the same
    // object. All of them default on and all of them predate this stream.
    for name in ["B7_M4_PAREVAL", "B7_M4_NOCLONE", "B7_M4_SCAN", "B7_M4_REFBAD"] {
        put(name, u32::from(m4_switch(name)));
    }
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
            "store_coeffs": m3_switch("B7_M3_COEFFS"),
            "device_merkle_reduce": m3_switch("B7_M3_MERKLE"),
            "fri_gather": m3_switch("B7_M3_FRIGATHER"),
            "prewarm": m3_switch("B7_M3_PREWARM"),
            "merkle_mode": std::env::var("B7_M3_MERKLE").unwrap_or_else(|_| "1".into()),
            "launch_sync": std::env::var("B7_M3_LAUNCHSYNC").is_ok()
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

pub fn env_json() -> Value {
    json!({
        "threads": rayon::current_num_threads(),
        "mem_total_bytes": mem_total_bytes(),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "simd_n_lanes": stwo::prover::backend::simd::m31::N_LANES,
    })
}

pub fn emit(name: &str, body: Value) {
    let out = json!({
        "bench": name,
        "env": env_json(),
        "modes": modes_json(),
        "security": config_json(),
        "result": body,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
