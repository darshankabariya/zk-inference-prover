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
