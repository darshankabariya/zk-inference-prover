//! Phase breakdown of a stwo prove() call — B7a milestone 2a's version.
//!
//! This replaces B6.5's `spans.rs` for the port, for two reasons the fork
//! forces (README section 6 lists this as 2a's first patch):
//!
//! 1. **The CUDA `ComponentProver` uses different span names.** B6.5's leaf
//!    `"Constraint point-wise eval"` is emitted by the SIMD impl
//!    (`component_prover.rs:99,231`); the CudaBackend impl at `:263` emits
//!    `"evaluate_constraint_quotients_on_domain"` instead, with
//!    `"CPU fallback evaluation"` / `"GPU evaluate_constraint_quotients_on_domain"`
//!    / `"CPU auto-fallback (no CUDA kernel)"` inside it. Unpatched, the whole
//!    of constraint evaluation would land in `other` on the GPU path and the
//!    three boundaries of section 5 would be unreadable.
//!
//! 2. **The names collide once both backends are in play.** The CUDA
//!    constraint path opens a nested `"Extension"` span (`:293`) — the same
//!    name as the commit path's LDE leaf. B6.5's comment ("the spans listed in
//!    LEAVES do not nest inside one another at INFO level") stops being true,
//!    and a plain per-name sum double-counts.
//!
//! The fix for (2) is to stop relying on the names being non-nesting: a leaf's
//! time is recorded only when no *tracked* span is already open above it, so
//! the partition is disjoint by construction on any backend. Anything nested
//! inside a leaf is attributed to that leaf — which is the right answer for the
//! CUDA constraint path, where the nested `"Extension"` and denominator work
//! genuinely is constraint evaluation.
//!
//! `DETAIL` spans are reported alongside, not inside, the partition: they are
//! allowed to nest, and they are what section 5 means by "`constraint_eval`
//! printed on its own line, labelled CPU fallback, and kept out of the
//! headline".

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde_json::{Value, json};
use tracing::span::Id;
use tracing::{Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

/// The partition. Several span names may map to one label (the backends name
/// the same phase differently); time is taken from the outermost tracked span.
pub const LEAVES: &[(&str, &str)] = &[
    ("Interpolation for commitment", "interpolate"),
    ("Extension", "lde_extend"),
    ("Merkle", "merkle"),
    ("Constraint Extension", "constraint_extension"),
    // SimdBackend's ComponentProver
    ("Constraint point-wise eval", "constraint_eval"),
    // CudaBackend's ComponentProver: the whole of it, nested spans included
    ("evaluate_constraint_quotients_on_domain", "constraint_eval"),
    ("Evaluate columns out of domain", "oods_eval"),
    ("Compute FRI quotients", "fri_quotients"),
    ("commit", "fri_commit"),
    ("Grind", "grind"),
];

/// Reported next to the partition, never inside it. These nest.
pub const DETAIL: &[(&str, &str)] = &[
    ("CPU fallback evaluation", "constraint_eval_cpu_fallback"),
    ("CPU auto-fallback (no CUDA kernel)", "constraint_eval_cpu_autofallback"),
    ("GPU evaluate_constraint_quotients_on_domain", "constraint_eval_gpu"),
    ("Denom inverses generate", "constraint_eval_denom_inv"),
    ("component_polys generate", "constraint_eval_polys"),
];

pub const TOTAL_SPAN: &str = "prove_ex";

fn is_leaf(name: &str) -> bool {
    LEAVES.iter().any(|(s, _)| *s == name)
}

#[derive(Default)]
pub struct Timings(pub Mutex<HashMap<String, (f64, u64)>>);

struct Enter(Instant);
/// Set on a span that is nested inside an already-open tracked leaf; its time
/// belongs to that ancestor, so it is not recorded on its own.
struct Shadowed;

struct TimerLayer(Arc<Timings>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimerLayer {
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            if is_leaf(span.name()) {
                // Walk up with `parent()` rather than an iterator helper: the
                // helpers differ across tracing-subscriber versions, this does
                // not.
                let mut up = span.parent();
                while let Some(s) = up {
                    if is_leaf(s.name()) {
                        span.extensions_mut().replace(Shadowed);
                        break;
                    }
                    up = s.parent();
                }
            }
            span.extensions_mut().replace(Enter(Instant::now()));
        }
    }
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            let dt = ext
                .remove::<Enter>()
                .map(|e| e.0.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            let shadowed = ext.remove::<Shadowed>().is_some();
            drop(ext);
            if shadowed {
                return;
            }
            let name = span.name().to_string();
            let mut g = self.0.0.lock().unwrap();
            let e = g.entry(name).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
        }
    }
}

static TIMINGS: OnceLock<Arc<Timings>> = OnceLock::new();

/// Install the subscriber once per process. Safe to call repeatedly.
pub fn install() -> Arc<Timings> {
    TIMINGS
        .get_or_init(|| {
            let t = Arc::new(Timings::default());
            let subscriber = tracing_subscriber::registry()
                .with(TimerLayer(t.clone()))
                .with(tracing_subscriber::filter::LevelFilter::from_level(Level::INFO));
            tracing::subscriber::set_global_default(subscriber)
                .expect("failed to install span timer");
            t
        })
        .clone()
}

pub fn reset() {
    install().0.lock().unwrap().clear();
}

/// Snapshot the accumulated spans as a phase breakdown.
pub fn snapshot(wall_secs: f64) -> Value {
    let g = install();
    let map = g.0.lock().unwrap();
    let mut phases = serde_json::Map::new();
    let mut leaf_sum = 0.0;
    for (_, label) in LEAVES {
        phases.entry(label.to_string()).or_insert(json!({ "secs": 0.0, "calls": 0 }));
    }
    for (span, label) in LEAVES {
        let (secs, n) = map.get(*span).copied().unwrap_or((0.0, 0));
        if secs == 0.0 && n == 0 {
            continue;
        }
        leaf_sum += secs;
        let e = phases.get_mut(*label).unwrap();
        let prev_secs = e["secs"].as_f64().unwrap_or(0.0);
        let prev_calls = e["calls"].as_u64().unwrap_or(0);
        *e = json!({ "secs": prev_secs + secs, "calls": prev_calls + n });
    }
    let mut detail = serde_json::Map::new();
    for (span, label) in DETAIL {
        let (secs, n) = map.get(*span).copied().unwrap_or((0.0, 0));
        detail.insert((*label).to_string(), json!({ "secs": secs, "calls": n }));
    }
    // `prove_ex` covers only stwo's `prove()`; the trace commitments happen
    // before it, so wall time of the whole flow is the right denominator.
    let instrumented = map.get(TOTAL_SPAN).copied().unwrap_or((0.0, 0)).0;
    let total = wall_secs;
    phases.insert("other".to_string(), json!({ "secs": total - leaf_sum, "calls": 0 }));
    let commit_secs: f64 = ["interpolate", "lde_extend", "merkle", "constraint_extension"]
        .iter()
        .map(|k| phases[*k]["secs"].as_f64().unwrap())
        .sum();
    let fri_secs: f64 = ["fri_quotients", "fri_commit", "grind"]
        .iter()
        .map(|k| phases[*k]["secs"].as_f64().unwrap())
        .sum();
    json!({
        "wall_secs": wall_secs,
        "prove_call_secs": instrumented,
        "phases": phases,
        "detail": detail,
        "commit_secs": commit_secs,
        "fri_secs": fri_secs,
        "commit_plus_fri_frac": (commit_secs + fri_secs) / total,
    })
}
