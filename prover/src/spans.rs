//! Phase breakdown of a stwo prove() call.
//!
//! stwo instruments its prover with `tracing` spans. We install a subscriber
//! that accumulates busy time per span name at INFO level. The spans listed in
//! `LEAVES` do not nest inside one another at INFO level (the Merkle spans
//! inside FRI are emitted at TRACE and therefore filtered out), so their sum is
//! a genuine partition of prover time up to the "other" remainder.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use serde_json::{Value, json};
use tracing::span::Id;
use tracing::{Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

pub const LEAVES: &[(&str, &str)] = &[
    ("Interpolation for commitment", "interpolate"),
    ("Extension", "lde_extend"),
    ("Merkle", "merkle"),
    ("Constraint Extension", "constraint_extension"),
    ("Constraint point-wise eval", "constraint_eval"),
    ("Evaluate columns out of domain", "oods_eval"),
    ("Compute FRI quotients", "fri_quotients"),
    ("commit", "fri_commit"),
    ("Grind", "grind"),
];
pub const TOTAL_SPAN: &str = "prove_ex";

#[derive(Default)]
pub struct Timings(pub Mutex<HashMap<String, (f64, u64)>>);

struct Enter(Instant);

struct TimerLayer(Arc<Timings>);

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimerLayer {
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().replace(Enter(Instant::now()));
        }
    }
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let dt = span
                .extensions_mut()
                .remove::<Enter>()
                .map(|e| e.0.elapsed().as_secs_f64())
                .unwrap_or(0.0);
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
    for (span, label) in LEAVES {
        let (secs, n) = map.get(*span).copied().unwrap_or((0.0, 0));
        leaf_sum += secs;
        phases.insert((*label).to_string(), json!({ "secs": secs, "calls": n }));
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
        "commit_secs": commit_secs,
        "fri_secs": fri_secs,
        "commit_plus_fri_frac": (commit_secs + fri_secs) / total,
    })
}
