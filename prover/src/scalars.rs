//! B7b (README 26): the lowered IR cached by shape, and the scalars it cannot
//! cache -- read off the eval per chunk with a numeric probe.
//!
//! `Lower` (cudagen.rs) interns every scalar by *value*, in expression-visit
//! order. A scalar is one of two things: a relation element, which reaches the
//! lowering as an `ExtExpr::Param` and is resolved from `relation_entries`; or
//! a constant the eval itself handed to the evaluator, which reaches it as
//! `ExtExpr::SecureCol([Const; 4])` through one of the fork's `From` / `Add` /
//! `Sub` / `Mul` impls on `ExtExpr`. The fork's `expr` module never folds two
//! constants into one (26.2), so every such constant is a value the eval
//! *produced* -- and an `EvalAtRow` whose extension type records every value
//! that enters it collects all of them in microseconds, with no expression tree
//! at all.
//!
//! So the cache holds, per shape, the IR of the first lowering and a provenance
//! per scalar slot -- "the i-th recorded constant" or "entry k's z / alpha_j" --
//! established at fill time by matching values. A later chunk of the same shape
//! runs the two numeric probes, replays the provenance, and has the IR with its
//! own scalars in it. Values are 124-bit transcript draws, so two different
//! quantities matching by value is the same 2^-124 event `Lower::lit` already
//! accepts; two structurally equal quantities (the eval computing `z` twice)
//! match at fill time and are equal on every chunk.
//!
//! The *shape* is not the family name (section 5 rule 9: the four walk groups
//! share `b65_walk_eval` and differ in how many matmuls they carry). The probe
//! also hashes the eval's structure as it runs -- every mask, preprocessed,
//! relation and constraint call, every base-field constant's value, every
//! arithmetic step's kind, and each extension constant's 0 / 1 / other class,
//! which is the one thing about a scalar's value the lowering folds on -- and
//! that hash is part of the key. Two evals with the same name, sizes and
//! structure hash lower to the same ops.
//!
//! `B7_M5_IRCACHE`: 0 off (the default, the proven path), 1 on, 2 on with every
//! hit checked against the full lowering -- ops, widths, constraints, fractions,
//! scalars and refs, slot for slot. `B7_M5_IRMUT` (`shift` / `stale`, with an
//! optional `:family`) are the negative controls, section 5 rule 5.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use num_traits::{One, Zero};
use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::{SECURE_EXTENSION_DEGREE, SecureField};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval, Relation, RelationEntry};

use crate::cudagen::{Ir, RelEntry, relation_entries};

// ------------------------------------------------------------ the recorder ----

/// What one evaluation of an eval through the probe leaves behind: every
/// extension constant in order, and the hash of the eval's structure.
struct Probed {
    consts: Vec<SecureField>,
    h: u64,
}

thread_local! {
    static REC: RefCell<Probed> = const { RefCell::new(Probed { consts: Vec::new(), h: 0 }) };
}

/// fold bytes into the structure hash (FNV-1a)
fn tr(bytes: &[u8]) {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        for b in bytes {
            r.h ^= *b as u64;
            r.h = r.h.wrapping_mul(0x0100_0000_01b3);
        }
    });
}

fn tr_base(tag: u8, c: M31) {
    tr(&[tag]);
    tr(&c.0.to_le_bytes());
}

fn note(v: SecureField) {
    let l = v.to_m31_array();
    let class: u8 = if l[1].0 == 0 && l[2].0 == 0 && l[3].0 == 0 && l[0].0 <= 1 {
        l[0].0 as u8
    } else {
        2
    };
    tr(&[b'S', class]);
    REC.with(|r| r.borrow_mut().consts.push(v));
}

/// A base value. `Some` is a lone `BaseExpr::Const`, which becomes a scalar the
/// moment it is lifted into the extension field; `None` is anything else --
/// a column, or an expression, which the fork represents as a node and the
/// lowering never interns as a scalar.
#[derive(Clone, Debug)]
pub struct RecF(Option<M31>);

/// An extension value. Opaque: what matters is what entered it, and that was
/// recorded on the way in.
#[derive(Clone, Debug)]
pub struct RecE;

impl From<BaseField> for RecF {
    fn from(v: BaseField) -> Self {
        tr_base(b'K', v);
        RecF(Some(v))
    }
}
impl Zero for RecF {
    fn zero() -> Self {
        RecF::from(M31::zero())
    }
    fn is_zero(&self) -> bool {
        self.0 == Some(M31::zero())
    }
}
impl One for RecF {
    fn one() -> Self {
        RecF::from(M31::one())
    }
}
impl Add for RecF {
    type Output = RecF;
    fn add(self, _: RecF) -> RecF {
        tr(b"a");
        RecF(None)
    }
}
impl Sub for RecF {
    type Output = RecF;
    fn sub(self, _: RecF) -> RecF {
        tr(b"s");
        RecF(None)
    }
}
impl Mul for RecF {
    type Output = RecF;
    fn mul(self, _: RecF) -> RecF {
        tr(b"m");
        RecF(None)
    }
}
impl MulAssign for RecF {
    fn mul_assign(&mut self, _: RecF) {
        tr(b"m");
        self.0 = None;
    }
}
impl AddAssign for RecF {
    fn add_assign(&mut self, _: RecF) {
        tr(b"a");
        self.0 = None;
    }
}
impl AddAssign<BaseField> for RecF {
    fn add_assign(&mut self, rhs: BaseField) {
        tr_base(b'+', rhs);
        self.0 = None;
    }
}
impl Mul<BaseField> for RecF {
    type Output = RecF;
    fn mul(self, rhs: BaseField) -> RecF {
        tr_base(b'*', rhs);
        RecF(None)
    }
}
impl Neg for RecF {
    type Output = RecF;
    fn neg(self) -> RecF {
        tr(b"n");
        RecF(None)
    }
}
impl FieldExpOps for RecF {
    fn inverse(&self) -> Self {
        tr(b"i");
        RecF(None)
    }
}
// `BaseExpr op SecureField` is `ExtExpr::from(self) op ExtExpr::from(rhs)`:
// the base side becomes a scalar if it was a lone constant, the secure side
// always.
impl Add<SecureField> for RecF {
    type Output = RecE;
    fn add(self, rhs: SecureField) -> RecE {
        let _ = RecE::from(self);
        note(rhs);
        tr(b"A");
        RecE
    }
}
impl Mul<SecureField> for RecF {
    type Output = RecE;
    fn mul(self, rhs: SecureField) -> RecE {
        let _ = RecE::from(self);
        note(rhs);
        tr(b"M");
        RecE
    }
}

impl From<SecureField> for RecE {
    fn from(v: SecureField) -> Self {
        note(v);
        RecE
    }
}
impl From<RecF> for RecE {
    fn from(v: RecF) -> Self {
        match v.0 {
            Some(c) => note(SecureField::from(c)),
            None => tr(b"L"),
        }
        RecE
    }
}
impl Zero for RecE {
    fn zero() -> Self {
        // `ExtExpr::zero()` is `ExtExpr::from(BaseField::zero())`: a scalar
        // slot, folded at its uses but interned all the same.
        note(SecureField::zero());
        RecE
    }
    fn is_zero(&self) -> bool {
        false
    }
}
impl One for RecE {
    fn one() -> Self {
        note(SecureField::one());
        RecE
    }
}
impl Neg for RecE {
    type Output = RecE;
    fn neg(self) -> RecE {
        tr(b"N");
        RecE
    }
}
impl Add for RecE {
    type Output = RecE;
    fn add(self, _: RecE) -> RecE {
        tr(b"A");
        RecE
    }
}
impl Sub for RecE {
    type Output = RecE;
    fn sub(self, _: RecE) -> RecE {
        tr(b"B");
        RecE
    }
}
impl Mul for RecE {
    type Output = RecE;
    fn mul(self, _: RecE) -> RecE {
        tr(b"M");
        RecE
    }
}
impl AddAssign for RecE {
    fn add_assign(&mut self, _: RecE) {
        tr(b"A");
    }
}
impl MulAssign for RecE {
    fn mul_assign(&mut self, _: RecE) {
        tr(b"M");
    }
}
impl Add<BaseField> for RecE {
    type Output = RecE;
    fn add(self, rhs: BaseField) -> RecE {
        note(SecureField::from(rhs));
        tr(b"A");
        RecE
    }
}
impl Mul<BaseField> for RecE {
    type Output = RecE;
    fn mul(self, rhs: BaseField) -> RecE {
        note(SecureField::from(rhs));
        tr(b"M");
        RecE
    }
}
impl Add<SecureField> for RecE {
    type Output = RecE;
    fn add(self, rhs: SecureField) -> RecE {
        note(rhs);
        tr(b"A");
        RecE
    }
}
impl Sub<SecureField> for RecE {
    type Output = RecE;
    fn sub(self, rhs: SecureField) -> RecE {
        note(rhs);
        tr(b"B");
        RecE
    }
}
impl Mul<SecureField> for RecE {
    type Output = RecE;
    fn mul(self, rhs: SecureField) -> RecE {
        note(rhs);
        tr(b"M");
        RecE
    }
}
impl Add<RecF> for RecE {
    type Output = RecE;
    fn add(self, rhs: RecF) -> RecE {
        let _ = RecE::from(rhs);
        tr(b"A");
        RecE
    }
}
impl Mul<RecF> for RecE {
    type Output = RecE;
    fn mul(self, rhs: RecF) -> RecE {
        let _ = RecE::from(rhs);
        tr(b"M");
        RecE
    }
}

/// The evaluator: masks are opaque, constraints are counted, relation entries
/// are combined (which records the elements too -- a harmless superset), and
/// `combine_ef` of four lone constants is the one `SecureCol` case
/// `qm31_const` turns into a scalar.
struct ScalarProbe;

impl EvalAtRow for ScalarProbe {
    type F = RecF;
    type EF = RecE;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        tr(format!("I{interaction},{offsets:?};").as_bytes());
        offsets.map(|_| RecF(None))
    }

    fn get_preprocessed_column(&mut self, column: PreProcessedColumnId) -> Self::F {
        tr(format!("P{};", column.id).as_bytes());
        RecF(None)
    }

    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: From<G>,
    {
        tr(b"C;");
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        tr(b"E");
        if values.iter().all(|v| v.0.is_some()) {
            let a = values.map(|v| v.0.unwrap());
            note(SecureField::from_m31_array(a));
        }
        RecE
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        tr(format!("R{},{},{};", entry.relation().get_name(), entry.relation().get_size(),
                   entry.values().len()).as_bytes());
        let _ = entry.relation().combine(entry.values());
    }

    fn write_logup_frac(&mut self, _f: stwo::core::Fraction<Self::EF, Self::EF>) {}
    fn finalize_logup_batched(&mut self, _b: &Vec<usize>) {}
    fn finalize_logup(&mut self) {}
    fn finalize_logup_in_pairs(&mut self) {}
}

/// Every constant this eval hands its evaluator, in the order it hands them,
/// and the hash of its structure.
fn probe<E: FrameworkEval>(eval: &E) -> Probed {
    REC.with(|r| {
        let mut r = r.borrow_mut();
        r.consts.clear();
        r.h = 0xcbf2_9ce4_8422_2325;
    });
    let _ = eval.evaluate(ScalarProbe);
    REC.with(|r| std::mem::replace(&mut *r.borrow_mut(), Probed { consts: Vec::new(), h: 0 }))
}

// ---------------------------------------------------------------- the cache ----

/// Where a scalar slot's value comes from on a later chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Prov {
    /// the i-th recorded constant
    Const(usize),
    /// relation entry k's `z`
    Z(usize),
    /// relation entry k's `alpha_j`
    Alpha(usize, usize),
}

struct Entry {
    ir: Ir,
    prov: Vec<Prov>,
}

type Key = (&'static str, u32, u32, bool, u64);

static CACHE: OnceLock<Mutex<HashMap<Key, Entry>>> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static UNCACHEABLE: AtomicU64 = AtomicU64::new(0);
static CHECKS: AtomicU64 = AtomicU64::new(0);
static HIT_NS: AtomicU64 = AtomicU64::new(0);
static FILL_NS: AtomicU64 = AtomicU64::new(0);

fn cache() -> &'static Mutex<HashMap<Key, Entry>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn mode() -> u32 {
    crate::util::m4b_mode("B7_M5_IRCACHE")
}

/// `B7_M5_IRMUT`, aimed (rule 5): `shift` replays every slot's provenance one
/// slot over; `stale` replays nothing and keeps the fill-time scalars.
fn mutation(family: &str) -> &'static str {
    let Ok(spec) = std::env::var("B7_M5_IRMUT") else { return "none" };
    let (name, fam) = match spec.split_once(':') {
        Some((n, f)) => (n.to_string(), Some(f.to_string())),
        None => (spec.clone(), None),
    };
    let aimed = match fam {
        None => true,
        Some(f) => family == format!("b65_{f}_eval"),
    };
    if !aimed {
        return "none";
    }
    match name.as_str() {
        "shift" => "shift",
        "stale" => "stale",
        "none" => "none",
        other => panic!("B7_M5_IRMUT={other}: unknown mutation"),
    }
}

pub fn report() -> serde_json::Value {
    serde_json::json!({
        "hits": HITS.load(Relaxed),
        "misses": MISSES.load(Relaxed),
        "uncacheable": UNCACHEABLE.load(Relaxed),
        "checked": CHECKS.load(Relaxed),
        "hit_secs": HIT_NS.load(Relaxed) as f64 / 1e9,
        "fill_secs": FILL_NS.load(Relaxed) as f64 / 1e9,
        "entries": cache().lock().unwrap().len(),
    })
}

fn find(v: SecureField, consts: &[SecureField], rels: &[RelEntry]) -> Option<Prov> {
    if let Some(i) = consts.iter().position(|c| *c == v) {
        return Some(Prov::Const(i));
    }
    for (k, r) in rels.iter().enumerate() {
        if r.z == v {
            return Some(Prov::Z(k));
        }
        if let Some(j) = r.alphas.iter().position(|a| *a == v) {
            return Some(Prov::Alpha(k, j));
        }
    }
    None
}

fn replay(prov: &[Prov], consts: &[SecureField], rels: &[RelEntry], family: &str)
          -> Vec<SecureField> {
    prov.iter()
        .map(|p| match *p {
            Prov::Const(i) => *consts.get(i).unwrap_or_else(|| {
                panic!("{family}: the probe recorded {} constants, slot wants #{i}",
                       consts.len())
            }),
            Prov::Z(k) => rels[k].z,
            Prov::Alpha(k, j) => rels[k].alphas[j],
        })
        .collect()
}

fn same(a: &Ir, b: &Ir, family: &str) {
    assert_eq!(a.ops.len(), b.ops.len(), "{family}: cached IR has a different op count");
    assert!(a.ops == b.ops, "{family}: cached IR differs in its ops");
    assert_eq!(a.width, b.width, "{family}: cached IR differs in its widths");
    assert_eq!(a.constraints, b.constraints, "{family}: cached IR differs in its constraints");
    assert_eq!(a.fracs, b.fracs, "{family}: cached IR differs in its fractions");
    assert_eq!(a.cols, b.cols, "{family}: cached IR differs in its column counts");
    assert_eq!(a.pre_ids, b.pre_ids, "{family}: cached IR differs in its preprocessed ids");
    assert_eq!(a.scalar_refs, b.scalar_refs, "{family}: cached IR differs in its scalar refs");
    assert_eq!(a.log_size, b.log_size, "{family}: cached IR differs in log_size");
    assert_eq!(a.eval_log_size, b.eval_log_size, "{family}: cached IR differs in eval_log_size");
    assert_eq!(a.cumsum_shift, b.cumsum_shift, "{family}: cached IR differs in cumsum_shift");
    assert_eq!(a.scalars.len(), b.scalars.len(), "{family}: cached IR has a different slot count");
    if let Some(k) = (0..a.scalars.len()).find(|k| a.scalars[*k] != b.scalars[*k]) {
        panic!("{family}: replayed scalar slot {k} of {} differs from the lowering's: {:?} vs {:?}",
               a.scalars.len(), a.scalars[k], b.scalars[k]);
    }
}

/// The IR for this eval at this size -- lowered, or replayed from the cache.
///
/// `lower` is the full lowering, run on a miss, on every call with the cache
/// off, and on every hit under mode 2 as the reference the replay is checked
/// against. A `B7_M4_LOGUPMUT` run bypasses the cache: the mutation changes the
/// IR, and the cache would otherwise hide it (section 5 rule 10).
pub fn cached_ir<E: FrameworkEval>(
    eval: &E,
    log_size: u32,
    eval_log_size: u32,
    logup: bool,
    lower: impl FnOnce() -> Ir,
) -> Ir {
    let mode = mode();
    if mode == 0 || std::env::var("B7_M4_LOGUPMUT").is_ok() {
        return lower();
    }
    let family = eval.cuda_eval_name();
    let t = std::time::Instant::now();
    let Probed { consts, h } = probe(eval);
    let key: Key = (family, log_size, eval_log_size, logup, h);
    let hit = {
        let g = cache().lock().unwrap();
        g.get(&key).map(|e| (e.ir.clone(), e.prov.clone()))
    };
    if let Some((mut ir, mut prov)) = hit {
        let rels = relation_entries(eval);
        match mutation(family) {
            "shift" => prov.rotate_left(1),
            "stale" => prov.clear(),
            _ => {}
        }
        if !prov.is_empty() {
            ir.scalars = replay(&prov, &consts, &rels, family);
        }
        HIT_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        HITS.fetch_add(1, Relaxed);
        if mode >= 2 {
            let full = lower();
            same(&ir, &full, family);
            CHECKS.fetch_add(1, Relaxed);
        }
        return ir;
    }
    let ir = lower();
    let rels = relation_entries(eval);
    let prov: Option<Vec<Prov>> = ir.scalars.iter().map(|v| find(*v, &consts, &rels)).collect();
    FILL_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    match prov {
        Some(prov) => {
            MISSES.fetch_add(1, Relaxed);
            cache().lock().unwrap().insert(key, Entry { ir: ir.clone(), prov });
        }
        None => {
            // A slot no probe can account for: this family stays uncached, and
            // the count says so rather than a silently slower chunk.
            UNCACHEABLE.fetch_add(1, Relaxed);
            let k = ir.scalars.iter().position(|v| find(*v, &consts, &rels).is_none()).unwrap();
            eprintln!("ircache: {family} at 2^{log_size} is not cacheable: slot {k} of {} \
                       matches none of {} recorded constants or {} relation entries",
                      ir.scalars.len(), consts.len(), rels.len());
        }
    }
    ir
}
