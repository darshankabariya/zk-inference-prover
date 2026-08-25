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
    let mut g = m4_acc().lock().unwrap();
    let e = g.entry(k).or_default().entry(name).or_insert((0, 0.0));
    e.0 += 1;
    e.1 += secs;
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

/// Milestone 4's tri-state modes, read once. Unlike `m4_switch` these
/// default to **off**: a mode selects a path that has still to prove itself
/// equal to the one it replaces, and the default has to be the proven one.
pub fn m4b_mode(name: &str) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(0)
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
        "security": config_json(),
        "result": body,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
