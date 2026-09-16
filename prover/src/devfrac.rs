//! B7a milestone 4 (section 3, row 1): every family's LogUp fractions, from the AIR.
//!
//! 19.1 found that `ExprEvaluator::add_to_relation` names each lookup's
//! denominator, so 2b's lowering has emitted them since 2b and only the
//! numerator was missing; 19.4 turned that into a generated kernel and 19.6
//! wrote down what was left: "`genemit --families` takes the list; the kernels
//! are generated, not written". This file is the route that list travels.
//! There is no per-family code in it and none in the kernels either -- what a
//! family contributes is its `FrameworkEval`, which the verifier already has.
//!
//! Three things it has to get right, each asserted rather than assumed:
//!
//! * **the columns a shape reads.** 19.2 measured that no fraction IR reads a
//!   shifted row or the interaction trace, and that the key tiles read the base
//!   tree alone. A component still *declares* its preprocessed columns and the
//!   generated kernel still checks the count, so `Cols::Unread` supplies a null
//!   per named-but-unread column and `assert_supplied` checks, per shape and
//!   per witness, that no column the IR loads is one of them. That is what lets
//!   a key tile's interaction run without its preprocessed columns being uploaded
//!   -- which would put back exactly the 4a bytes this row removes.
//! * **which builder ran.** `ZK_GENERATED_LOGUP` keeps 19.3's meaning for every
//!   family: 0 the hand-written builder, 1 the IR on the host, 2 both asserted
//!   equal, 3 the generated kernel, 4 the kernel against the hand-written one.
//! * **where the columns end up.** A component whose witness generation left its columns on the
//!   device has no host columns to build from and no reason to bring its LogUp
//!   columns back, so it takes the kernel unconditionally and hands the
//!   commitment `AnyCol::OwnedDev` -- 19.4's "both transfers disappear
//!   together", for the families 20 and 21 put on the device.

use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Column as _;
use stwo::prover::backend::simd::m31::PackedM31;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo_constraint_framework::FrameworkEval;

use crate::common::Eval;
use crate::cudagen::{Ir, Op};
use crate::device::AnyCol;

/// A component's own columns of one tree, wherever they live.
pub enum Cols<'a> {
    Host(Vec<&'a Eval>),
    Dev(Vec<&'a BaseFieldVec>),
    /// The component declares columns of this tree and its fraction IR loads
    /// none of them.
    Unread,
}

/// A component's columns of one tree: on the device if its witness generation left them
/// there, on the host otherwise.
pub fn cols_of<'a>(host: &'a [Eval], dev: Option<&'a [BaseFieldVec]>) -> Cols<'a> {
    match dev {
        Some(d) => Cols::Dev(d.iter().collect()),
        None => Cols::Host(host.iter().collect()),
    }
}

/// A shared table's slice of the two flat lists: on the device when
/// `ZK_DEVICE_TABLES` uploaded them there, on the host otherwise. An empty `dev` is
/// the switch being off, which is also the state the mode-4 check runs in.
pub fn tab_cols<'a>(
    host: &'a [Eval],
    dev: &'a [BaseFieldVec],
    lo: usize,
    n: usize,
) -> Cols<'a> {
    if dev.is_empty() {
        Cols::Host(host[lo..lo + n].iter().collect())
    } else {
        Cols::Dev(dev[lo..lo + n].iter().collect())
    }
}

impl Cols<'_> {
    fn is_dev(&self) -> bool {
        matches!(self, Cols::Dev(_))
    }
    fn is_unread(&self) -> bool {
        matches!(self, Cols::Unread)
    }
    /// The packed host slices the interpreter reads. An unread tree gets empty
    /// slices, which a load would index straight out of -- `assert_supplied`
    /// gets there first and says which column and which tree.
    fn packed(&self, n: usize) -> Vec<&[PackedM31]> {
        match self {
            Cols::Host(v) => v.iter().map(|e| &e.data[..]).collect(),
            Cols::Unread => vec![&[][..]; n],
            Cols::Dev(_) => panic!("a device column has no host slice"),
        }
    }
    /// Device pointers, with a null per named-but-unread column.
    fn ptrs(&self, n: usize, up: &[BaseFieldVec]) -> Vec<*const u32> {
        match self {
            Cols::Dev(v) => v.iter().map(|c| c.device_ptr).collect(),
            Cols::Unread => vec![std::ptr::null(); n],
            Cols::Host(_) => up.iter().map(|c| c.device_ptr).collect(),
        }
    }
    /// The upload the *check* pays: modes 3 and 4 run the kernel over columns
    /// that are still on the host, which is 19.4's measurement and not the
    /// prover's path.
    fn upload(&self) -> Vec<BaseFieldVec> {
        match self {
            Cols::Host(v) => {
                v.iter().map(|e| BaseFieldVec::from_vec(e.values.to_cpu())).collect()
            }
            _ => vec![],
        }
    }
}

/// A key tile's three components, and its columns when its witness generation left them on the
/// device.
///
/// A key tile is one component with three components in it -- the channel grid, the
/// score grid and the per-token block -- so what a component owns is the slice
/// `Committed::at` names, and that slice is the same on both sides of the link
/// (mode 2 asserts `at == dev.at`).
pub struct KeyTileAir<'a> {
    pub tag: usize,
    pub chan: &'a crate::gadgets::attention::air::ChanEval,
    pub key: &'a crate::gadgets::attention::air::KeyEval,
    pub row: &'a crate::gadgets::attention::air::RowEval,
    pub dev: Option<&'a crate::dev_attention_freivalds_witness_gen::DevKeyTile>,
    /// B7a milestone 4 (section 3 row 1): this key tile's selector columns and the
    /// flags they came from. `Some` whenever `dev` is, because the accumulators
    /// read the flags.
    pub pre: Option<&'a crate::devpre::DevPreKeyTile>,
}

/// One key tile component's LogUp columns. `which` is its index in `Committed::at`.
///
/// The preprocessed columns are `Unread` for all three, and that is a
/// measurement and not a shortcut: 19.2 found the key tiles read the base tree
/// alone. It used to be a saving as well -- a key tile's selectors were host
/// columns and supplying them would have meant an upload -- and since section
/// 25 they are on the device, so supplying them would cost nothing either.
/// `assert_supplied` is what keeps the measurement true per shape.
pub fn key_tile_cols(
    air: Option<&KeyTileAir<'_>>,
    which: usize,
    log_rows: u32,
    at: &[usize],
    host_base: &[Eval],
    host: impl FnOnce() -> (Vec<Eval>, SecureField),
) -> (Vec<AnyCol<'static>>, SecureField) {
    let Some(air) = air else {
        let (cols, sum) = host();
        return (crate::device::host_cols(cols), sum);
    };
    let (lo, hi) = (at[which], at[which + 1]);
    let base = match air.dev {
        Some(d) => Cols::Dev(d.base[lo..hi].iter().collect()),
        None => Cols::Host(host_base[lo..hi].iter().collect()),
    };
    let tag = format!("{} key_tile{}", ["chan", "key", "out", "row"][which], air.tag);
    match which {
        0 => frac_cols(air.chan, &tag, log_rows, Cols::Unread, base, host),
        1 => frac_cols(air.key, &tag, log_rows, Cols::Unread, base, host),
        3 => frac_cols(air.row, &tag, log_rows, Cols::Unread, base, host),
        _ => panic!("key tile component {which} has no eval"),
    }
}

/// The fraction IR for one component, with `ZK_MUT_LOGUP` applied if it is
/// set and aimed at this family.
///
/// The suffix is section 5 rule 5. Unaimed, a defect in the lowering always
/// panics at whichever family is lowered first -- `chan key_tile0` -- and the run
/// then says nothing at all about whether the other families' checks would
/// have caught it. `ZK_MUT_LOGUP=relbyname:chan` is the case 19.6 asked for
/// by name: the defect is inert on a family with one instance per relation
/// name, so `dequant` was the wrong family to run it against.
pub fn ir_of<E: FrameworkEval>(eval: &E, log_rows: u32) -> Ir {
    let spec = std::env::var("ZK_MUT_LOGUP").ok();
    let mutate = match spec.as_deref() {
        None => crate::cudagen::Mutate::None,
        Some(spec) => {
            let (name, fam) = match spec.split_once(':') {
                Some((n, f)) => (n, Some(f)),
                None => (spec, None),
            };
            let aimed = match fam {
                None => true,
                Some(f) => eval.cuda_eval_name() == format!("zk_{f}_eval"),
            };
            if aimed { crate::cudagen::Mutate::parse(name) } else { crate::cudagen::Mutate::None }
        }
    };
    crate::cudagen::lower_logup_mut(eval, log_rows, mutate)
}

/// The compiled shape for a fraction IR, or a loud failure before a proof.
///
/// 19.4's fingerprint lesson: a stale `libstwo_cuda.a` must abort rather than
/// hand a component a kernel that computes something else.
fn shape_of(family: &str, ir: &Ir) -> usize {
    shape_opt(family, ir).unwrap_or_else(|| {
        let fp = ir.fingerprint();
        panic!(
            "no generation kernel for {family} fingerprint {fp:#010x} ({} lookups): \
             regenerate with `zk-prover genemit --families {family}` and rebuild libstwo_cuda.a",
            ir.fracs.len()
        )
    })
}

/// The same lookup, as a question rather than a demand.
///
/// B11 stream M: a *host-built* component has a hand-written builder to fall back
/// to, so a fingerprint the installed store does not carry can decline the
/// kernel instead of ending the proof -- announced per component, never
/// silent, on the same `ZK_KERNEL_FALLBACK` switch the constraint side has used
/// for the residual stream and diagonal block shapes since stream B (`cudagen.rs`). This is the
/// route the `lit` caveat needs: a lowering whose scalars fold differently from
/// the emitted one produces a fingerprint nothing compiled, and the residual stream's bindings
/// are the only place in the design that can do it.
///
/// A component whose columns are on the *device* has no host builder at all, so
/// `frac_cols` still demands `shape_of` there. 19.4's lesson is untouched
/// either way: this declines a kernel that does not exist; it never accepts one
/// that computes something else, because the fingerprint has to match.
fn shape_opt(family: &str, ir: &Ir) -> Option<usize> {
    let fp = ir.fingerprint();
    let (shape, looks) = crate::gencudakernels::gen_shape_of(family, fp)?;
    assert_eq!(looks, ir.fracs.len(), "{family}: lookup count disagrees with the kernel");
    Some(shape)
}

/// 19.2's three properties, per shape and per witness rather than once.
fn assert_supplied(family: &str, ir: &Ir, pre: &Cols, base: &Cols) {
    let n = |c: &Cols, t: usize| match c {
        Cols::Host(v) => v.len(),
        Cols::Dev(v) => v.len(),
        Cols::Unread => ir.cols[t],
    };
    assert_eq!(n(pre, 0), ir.cols[0], "{family}: preprocessed column count");
    assert_eq!(n(base, 1), ir.cols[1], "{family}: base column count");
    for o in &ir.ops {
        if let Op::Load { tree, col, off } = *o {
            assert_eq!(off, 0, "{family}: a fraction IR reading a shifted row");
            let unread = match tree {
                0 => pre.is_unread(),
                1 => base.is_unread(),
                _ => panic!("{family}: a fraction IR reading tree {tree}"),
            };
            assert!(
                !unread,
                "{family}: the IR loads tree {tree} column {col}, which was not supplied"
            );
        }
    }
}

/// One component's LogUp columns, by whichever route its columns allow.
///
/// `host` is the hand-written builder for this family: the default, and what
/// modes 2 and 4 compare against cell for cell. It is never called when the
/// component's columns are on the device, because then there are no host columns
/// to build from.
pub fn frac_cols<E: FrameworkEval>(
    eval: &E,
    tag: &str,
    log_rows: u32,
    pre: Cols<'_>,
    base: Cols<'_>,
    host: impl FnOnce() -> (Vec<Eval>, SecureField),
) -> (Vec<AnyCol<'static>>, SecureField) {
    let on_dev = base.is_dev() || pre.is_dev();
    let mode = crate::util::path_mode("ZK_GENERATED_LOGUP");
    if !on_dev && mode == 0 {
        let (cols, sum) = host();
        return (crate::device::host_cols(cols), sum);
    }
    let t = std::time::Instant::now();
    // B7b (README 26): by shape, with this chunk's scalars replayed.
    let ir = crate::scalars::cached_ir(eval, log_rows, log_rows, true, || ir_of(eval, log_rows));
    crate::util::add_phase("int/frac_lower", t.elapsed().as_secs_f64());
    let family = eval.cuda_eval_name();
    assert!(!ir.fracs.is_empty(), "{tag}: {family} has no lookups");
    assert_supplied(family, &ir, &pre, &base);

    // ---- the interpreter, on the host (modes 1 and 2) ----------------------
    if !on_dev && mode < 3 {
        let (p, b) = (pre.packed(ir.cols[0]), base.packed(ir.cols[1]));
        let (cols, sum) = crate::cudagen::logup_cols(&ir, log_rows, &p, &b);
        if mode == 2 {
            compare(tag, family, &cols, sum, host());
        }
        return (crate::device::host_cols(cols), sum);
    }

    // ---- the generated kernel ---------------------------------------------
    // B11 stream M: with host columns in hand a missing fingerprint can decline
    // instead of aborting, on `ZK_KERNEL_FALLBACK`. On the device there is
    // nothing to decline *to*, so the demand stands.
    let shape = if on_dev {
        shape_of(family, &ir)
    } else {
        match shape_opt(family, &ir) {
            Some(s) => s,
            None if crate::util::switch_on("ZK_KERNEL_FALLBACK") => {
                eprintln!(
                    "kernel fallback: {tag}: {family} fraction fingerprint {:#010x} \
                     ({} lookups) builds on the host",
                    ir.fingerprint(),
                    ir.fracs.len()
                );
                let (cols, sum) = host();
                return (crate::device::host_cols(cols), sum);
            }
            None => shape_of(family, &ir),
        }
    };
    let t = std::time::Instant::now();
    let (up_pre, up_base) = (pre.upload(), base.upload());
    let (pp, bp) = (pre.ptrs(ir.cols[0], &up_pre), base.ptrs(ir.cols[1], &up_base));
    crate::util::add_phase("int/frac_up", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let (dev, sum) =
        crate::devlogup::logup_cols_dev_ptrs(&ir, family, shape, log_rows, &pp, &bp);
    crate::util::add_phase("int/frac_kernel", t.elapsed().as_secs_f64());
    if on_dev {
        // The columns were computed where they will be committed, so they stay.
        assert!(
            mode < 4,
            "{tag}: mode 4 has no host columns to compare against -- \
             run the check with the host witness generations"
        );
        return (dev.into_iter().map(|c| AnyCol::OwnedDev(c, log_rows)).collect(), sum);
    }
    let t = std::time::Instant::now();
    let domain = stwo::core::poly::circle::CanonicCoset::new(log_rows).circle_domain();
    let cols: Vec<Eval> = dev
        .iter()
        .map(|c| {
            stwo::prover::poly::circle::CircleEvaluation::new(
                domain,
                stwo::prover::backend::simd::column::BaseColumn::from_iter(c.to_vec()),
            )
        })
        .collect();
    crate::util::add_phase("int/frac_down", t.elapsed().as_secs_f64());
    if mode == 4 {
        compare(tag, family, &cols, sum, host());
    }
    (crate::device::host_cols(cols), sum)
}

/// Every column, every cell, and the claimed sum -- not a sample, because a
/// lowering that is wrong on one lookup of one row is wrong.
fn compare(
    tag: &str,
    family: &str,
    got: &[Eval],
    sum: SecureField,
    want: (Vec<Eval>, SecureField),
) {
    assert_eq!(
        got.len(),
        want.0.len(),
        "{tag} ({family}): the IR built {} LogUp columns, the hand-written builder {}",
        got.len(),
        want.0.len()
    );
    let mut cells = 0usize;
    for (i, (a, b)) in got.iter().zip(&want.0).enumerate() {
        let (x, y) = (a.values.to_cpu(), b.values.to_cpu());
        cells += x.len();
        if let Some(k) = (0..x.len()).find(|k| x[*k] != y[*k]) {
            panic!(
                "{tag} ({family}): LogUp column {i} differs at cell {k}: \
                 the IR says {}, the hand-written builder {}",
                x[k].0, y[k].0
            );
        }
    }
    assert_eq!(sum, want.1, "{tag} ({family}): claimed sums differ");
    eprintln!(
        "logup-ir: {tag} ({family}) -- {} columns, {cells} cells and the claimed sum match",
        got.len()
    );
}
