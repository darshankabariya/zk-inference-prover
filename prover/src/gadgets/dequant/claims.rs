//! The matmul dequant component's claimed side and its LogUp columns.
//!
//! What the component publishes is two Horner accumulators -- the Freivalds
//! claimed side and, where a binding asks for one, the output and scale sides --
//! and the running sums of its lookups. `mat_logup` is the hand-written builder;
//! `mat_logup_ir` is the same columns lowered out of the evaluator the verifier
//! checks, and `ZK_GENERATED_LOGUP=2` asserts the two agree cell for cell.

use num_traits::{One, Zero};
use rayon::prelude::*;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use crate::common::{eval_of, m31_u32};
use crate::encoding::m31_of;
use crate::gadgets::dequant::Variant as DeqVariant;
use crate::gadgets::dequant::trace::{Mat, MatComponent, deq_ref, fold_out};

/// The matmul component's Freivalds claimed side and LogUp columns.
pub struct MatInteraction {
    pub cols: Vec<crate::device::AnyCol<'static>>,
    /// `rv^T C s`: the projection of the committed accumulator that this
    /// component publishes on the Freivalds gadget's behalf.  `C` is committed
    /// here and nowhere else, and `freivalds::claims::identity_holds` is what
    /// compares this against the Freivalds component's own side.
    pub claimed_h: SecureField,
    pub use_sum: SecureField,
    /// stream E: the output side's closing value, when this component carries one
    pub claimed_out: Option<SecureField>,
    /// stream II (E11): the x-side scale's, when a requantize seam feeds this
    /// matmul
    pub claimed_sc: Option<SecureField>,
}

/// The weights an output-side Horner claim runs on, as the composition derives
/// them from the binding challenge.  Same numbers on both sides of the binding: the
/// consumer's own claim is built from `w_residual_stream` too.
pub struct OutputClaimW {
    pub w: Vec<SecureField>,
    pub lam: SecureField,
    pub k_in: SecureField,
    pub k_bd: SecureField,
    pub kind: crate::gadgets::dequant::OKind,
    /// Stream SS: rows per kv head, on the one output side that folds under the
    /// key tile grid (a folded `v`).  `None` everywhere else, and then `k_bd` is
    /// taken at a token boundary exactly as stream E wrote it.  Whatever this
    /// says, `mat_component` marks the same rows -- the two have to agree or the
    /// interaction column and its AIR read different steps.
    pub head: Option<usize>,
}

/// B11 stream U, lever 3: `gkr` is `--gkr-dequant`, and it is the one thing in
/// this file the second GKR family changes.
///
/// The component's Horner accumulators (`z`, `h`, and stream E's output side) are
/// untouched -- they are what stays in the tree. What the flag removes is the
/// LogUp half: instead of `mat_logup`'s committed running-sum columns, the
/// component's fractions become GKR input layers (collected in `compose`, before
/// the proof's transcript position) and its claimed sum leaves `use_sums` for
/// `sums_cancel`. This is the same shape stream S gave the key tiles in
/// `attn_trace::interaction_phase`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn mat_interaction(m: &Mat, r: &MatComponent, rv: SecureField, s_ch: SecureField,
                       rels: &crate::compose::Rels, out: Option<&OutputClaimW>,
                       air: &crate::gadgets::dequant::DequantEval, gkr: bool,
                       sc: Option<(SecureField, SecureField)>)
                       -> MatInteraction {
    let lay = r.lay;
    let rows = 1usize << lay.log_rows;
    let (rv_pow, rv_step) = crate::relations::powers(rv, lay.per_row);
    // B7a milestone 4 (section 3 row 1): `z` and `h` where the component's own
    // accumulator columns are. `ZK_DEVICE_ACCUMULATORS=2` runs both and asserts every cell.
    let acc_dev = r.dev.as_ref().filter(|_| crate::devacc::mode() >= 1);
    let acc_host = acc_dev.is_none() || crate::devacc::mode() == 2;
    // B11 stream M: the component's real height, kept before `rows` is shadowed to
    // zero. `devacc` builds `z` and `h`; it does *not* build stream E's output
    // side, which is a third accumulator over the *witness* rather than over
    // this component's columns and so has nothing to do with where the witness generation went.
    // Suppressing it with the other two left `u[rows - 1]` indexing an empty
    // vector -- reachable only on an o-side matmul with `ZK_DEVICE_ACCUMULATORS` on, which
    // is `lm_head` and only `lm_head`, and `ACC` had never been on.
    let n_rows = rows;
    let rows = if acc_host { rows } else { 0 };
    // the Horner accumulators, inner sums in parallel, carries serial
    let phase_t = std::time::Instant::now();
    let inner: Vec<SecureField> = (0..rows)
        .into_par_iter()
        .map(|j| {
            let mut zf = SecureField::zero();
            if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                for s in 0..lay.per_row {
                    let a = m.acc[t * m.n_out + b * lay.per_row + s];
                    if a != 0 {
                        zf += rv_pow[s] * SecureField::from(m31_of(a));
                    }
                }
            }
            zf
        })
        .collect();
    crate::util::add_phase("int/deq_inner", phase_t.elapsed().as_secs_f64());
    let phase_t = std::time::Instant::now();
    let (mut z, mut h) = (vec![SecureField::zero(); rows], vec![SecureField::zero(); rows]);
    // B7a milestone 4 (18.4): the two accumulators, each with its own
    // dependency structure honoured rather than one serial pass over both.
    // The lead prefix keeps this exact: `inner` is zero on every lead row, so
    // the scan's extra `s_ch` steps there multiply zero — but only when the
    // lead is chunk-aligned, so the guard falls back to the serial loop if not.
    if crate::util::switch_on("ZK_PARALLEL_SCAN") && lay.chunks > 0 && rows % lay.chunks == 0
        && lay.lead % lay.chunks == 0 {
        z.par_chunks_mut(lay.chunks).enumerate().for_each(|(c, zc)| {
            let base = c * lay.chunks;
            let mut run = SecureField::zero();
            for (k, zk) in zc.iter_mut().enumerate() {
                run = if k > 0 { run * rv_step } else { SecureField::zero() } + inner[base + k];
                *zk = run;
            }
        });
        let n_last = rows / lay.chunks;  // `rows` is 0 on the device route
        let mut hh = vec![SecureField::zero(); n_last];
        let mut run = SecureField::zero();
        for (m, hm) in hh.iter_mut().enumerate() {
            run = run * s_ch + z[(m + 1) * lay.chunks - 1];
            *hm = run;
        }
        h.par_iter_mut().enumerate().for_each(|(j, hj)| {
            let m = (j + 1) / lay.chunks;
            *hj = if m == 0 { SecureField::zero() } else { hh[m - 1] };
        });
    } else {
        for j in 0..rows {
            let nofirst = matches!(lay.at_opt(m.tokens, j), Some((_, b)) if b + 1 < lay.chunks);
            let last = matches!(lay.at_opt(m.tokens, j), Some((_, 0)));
            z[j] = if nofirst { z[j - 1] * rv_step } else { SecureField::zero() } + inner[j];
            h[j] = if j == 0 {
                SecureField::zero()
            } else if last {
                h[j - 1] * s_ch
            } else {
                h[j - 1]
            };
            if last {
                h[j] += z[j];
            }
        }
    }
    crate::util::add_phase("int/deq_carry", phase_t.elapsed().as_secs_f64());
    let phase_t = std::time::Instant::now();
    let mut sink = crate::device::ColSink::new();
    let claimed_h = match acc_dev {
        Some(d) => {
            let got = crate::devacc::deq(d, &lay, &r.map, r.map.f.acc, rv, s_ch);
            let hh = acc_host.then(|| {
                let mut e = crate::encoding::secure_cols_pub(lay.log_rows, &z);
                e.extend(crate::encoding::secure_cols_pub(lay.log_rows, &h));
                (e, vec![h[rows - 1]])
            });
            crate::devacc::place(&format!("deq {} 2^{}", m.name, lay.log_rows), got, hh,
                                 &mut sink)[0]
        }
        None => {
            sink.extend(crate::encoding::secure_cols_pub(lay.log_rows, &z));
            sink.extend(crate::encoding::secure_cols_pub(lay.log_rows, &h));
            // What this component publishes for the Freivalds check is the
            // projection `rv^T C s` of the accumulator it commits: each row of
            // `C` folded against `rv` -- that is `z`, complete on the token's
            // last row -- and those folded against `s`.  `h` is the same Horner
            // carried row by row so the AIR can constrain it, and it closes on
            // this value; `project` is where the value is defined, in the gadget
            // that consumes it.
            let projected = crate::gadgets::freivalds::claims::project(
                (0..rows)
                    .filter(|j| matches!(lay.at_opt(m.tokens, *j), Some((_, 0))))
                    .map(|j| z[j]),
                s_ch);
            // A real assert, not a debug one: this is the only thing tying the
            // value published as `claimed_h` to the column the AIR constrains,
            // and the gate builds --release, where a debug assert is gone.  It
            // is one comparison per matmul, not per row.
            assert_eq!(projected, h[rows - 1],
                       "{}: the claimed side and the column that carries it disagree",
                       m.name);
            projected
        }
    };
    let mut cols = sink.into_inner();
    crate::util::add_phase("int/deq_cols", phase_t.elapsed().as_secs_f64());

    // stream E: the output side, between the claimed side's columns and the
    // LogUp columns -- the order the AIR reads them in.
    let claimed_out = out.map(|o| {
        // `n_rows`, not `rows`: this accumulator is the witness's, and it runs
        // whatever `ZK_DEVICE_ACCUMULATORS` did to the other two.
        let mut u = vec![SecureField::zero(); n_rows];
        for j in 0..n_rows {
            let mut f = SecureField::zero();
            if let Some((t, b)) = lay.at_opt(m.tokens, j) {
                for s in 0..lay.per_row {
                    let c = b * lay.per_row + s;
                    // stream SS: the fold has no bf16 image, so its reference is
                    // the integer itself and `deq_ref` is not asked for one.
                    if o.kind == crate::gadgets::dequant::OKind::Int {
                        f += SecureField::from(m31_of(fold_out(m, t, c))) * o.w[s];
                        continue;
                    }
                    let pat = crate::encoding::canon(deq_ref(m, &r.spec, t, c));
                    f += match o.kind {
                        crate::gadgets::dequant::OKind::Pair => {
                            let (sm, ef) = crate::encoding::split(pat);
                            (SecureField::from(m31_of(sm))
                             + SecureField::from(m31_u32(ef)) * o.lam) * o.w[s]
                        }
                        // the same key the argmax component compares, off the same
                        // host function the extractor's column was built with
                        // taken above, off `fold_out`, before `deq_ref` is asked
                        crate::gadgets::dequant::OKind::Int => unreachable!(),
                        crate::gadgets::dequant::OKind::Key => SecureField::from(
                            m31_u32(crate::gadgets::argmax::order_key(pat))) * o.w[s],
                    };
                }
            }
            // the multiplier is the step *into* this row: an ordinary chunk step
            // inside a token, the boundary jump on a row that begins one --
            // and, with `o.head`, the key tile grid's jump on the row after one that
            // begins a kv head.  The same predicate `mat_component` marked.
            let mult = if j == 0 {
                SecureField::zero()
            } else if let Some(mh) = o.head {
                if j % mh == 0 { o.k_bd } else { o.k_in }
            } else {
                match lay.at_opt(m.tokens, j) {
                    Some((_, b)) if b + 1 == lay.chunks => o.k_bd,
                    Some(_) => o.k_in,
                    None => SecureField::zero(),
                }
            };
            u[j] = if j == 0 { f } else { u[j - 1] * mult + f };
        }
        let claimed = u[n_rows - 1];
        cols.extend(crate::encoding::secure_cols_pub(lay.log_rows, &u)
            .into_iter().map(crate::device::AnyCol::OwnedHost));
        claimed
    });

    // Stream II (E11): the x-side scale, between the output side's column and
    // the LogUp columns -- the order the AIR reads them in.  One value per
    // token on the `last` rows, the same field elements this component's shared
    // columns commit, so the producing seam's own fold is comparable to it.
    let claimed_sc = sc.map(|(mu, lam)| {
        let mut v = vec![SecureField::zero(); n_rows];
        for j in 0..n_rows {
            let (f, last) = match lay.at_opt(m.tokens, j) {
                Some((t, 0)) => {
                    let x = match r.spec.variant {
                        DeqVariant::Pow2 => SecureField::from(m31_of(m.jx[t] as i64)),
                        _ => SecureField::from(m31_u32(m.sx[t].0))
                             + SecureField::from(m31_u32(m.sx[t].1)) * lam,
                    };
                    (x, true)
                }
                _ => (SecureField::zero(), false),
            };
            let mult = if j == 0 { SecureField::zero() }
                       else if last { mu } else { SecureField::one() };
            v[j] = if j == 0 { f } else { v[j - 1] * mult + f };
        }
        let claimed = v[n_rows - 1];
        cols.extend(crate::encoding::secure_cols_pub(lay.log_rows, &v)
            .into_iter().map(crate::device::AnyCol::OwnedHost));
        claimed
    });

    // the LogUp columns, in the exact order the AIR adds lookups -- or, under
    // the flag, the same fractions handed to the GKR collector instead
    let phase_t = std::time::Instant::now();
    let use_sum = if gkr {
        // the fractions were handed to the collector before the GKR proof ran,
        // in `compose`'s own pass -- see the comment there for why the two are
        // in different places
        SecureField::zero()
    } else {
        let (lcols, use_sum) = mat_logup(m, r, rels, air);
        crate::util::add_phase("int/deq_logup", phase_t.elapsed().as_secs_f64());
        cols.extend(lcols);
        use_sum
    };
    MatInteraction { cols, claimed_h, use_sum, claimed_out, claimed_sc }
}

/// The evaluator whose fraction IR *is* this component's lookups: the
/// component's own, with the parts that carry no lookup left off. The same
/// fractions the AIR states (B7a milestone 4, 18.7).
///
/// `DequantEval` is the component the verifier checks; lowering it with
/// `cudagen::lower_logup` gives every lookup's numerator and denominator as a
/// straight-line IR over this component's own columns, in `add_to_relation`
/// order -- which is the order `finalize_logup_in_pairs` batches them in and
/// therefore the order `paired` needs. Nothing below is dequant-specific:
/// swap the eval and it builds another family's LogUp columns.
///
/// Issue #43: the argument is the component `build_components` built, so the
/// prover, the GKR collector, the affinity report and `genemit` all lower one
/// evaluator derived one way. It used to be rebuilt from the plan here, which
/// left the emitter lowering the component and the prover lowering this, and
/// only the fingerprint said the two agreed. What the derivation drops:
///
///  * **the output side and the scale side.** Both are Horner accumulators
///    closed on a claim; neither adds a lookup, so neither can change this
///    eval's LogUp IR. Leaving them off keeps the lowering identical whether
///    the bindings are on or not, which is what `ZK_GENERATED_LOGUP=2`
///    compares.
///  * **the claimed side's claim.** `yside.claimed_h` enters constraints
///    alone -- never a lookup -- and at generation time there is no claimed
///    sum yet, so zero is correct and says so.
///
/// B11 stream U, lever 3: the GKR backend needs the same evaluator. `gkr` is
/// cleared -- the fractions the backend proves are the ones the committed
/// backend would have built, so the lowering is of the *un-ablated* evaluator
/// and asks for the flag-off shape, which is stream T's precedent for the key
/// tiles and is why the kernel store already carries this fingerprint.
pub fn deq_logup_eval(air: &crate::gadgets::dequant::DequantEval) -> crate::gadgets::dequant::DequantEval {
    let mut eval = air.clone();
    eval.output_claim = None;
    eval.sside = None;
    if let Some(y) = eval.yside.as_mut() {
        y.claimed_h = SecureField::zero();
    }
    eval.gkr = false;
    eval
}

fn mat_logup_ir(r: &MatComponent, air: &crate::gadgets::dequant::DequantEval, on_device: bool)
                -> (Vec<crate::device::AnyCol<'static>>, SecureField) {
    use stwo::prover::backend::simd::m31::PackedM31;
    let eval = deq_logup_eval(air);
    // `ZK_MUT_LOGUP` picks one of 2b's deliberate defects, so mode 2 has a
    // negative control. B7a milestone 4 (section 3 row 1): one place reads it
    // for every family, and it takes a `:family` suffix -- unaimed, a defect
    // panics at whichever family is lowered first and says nothing about the
    // rest (section 5 rule 5).
    let ir = crate::devfrac::ir_of(&eval, r.lay.log_rows);
    if on_device {
        // The same IR, run by the generated kernel instead of the interpreter.
        // The shape id comes from the compiled table by fingerprint, so a
        // stale library is a loud failure and never a wrong proof.
        let family = <crate::gadgets::dequant::DequantEval as
            stwo_constraint_framework::FrameworkEval>::cuda_eval_name(&eval);
        let fp = ir.fingerprint();
        let (shape, looks) = crate::gencudakernels::gen_shape_of(family, fp).unwrap_or_else(|| {
            panic!("no generation kernel for {family} fingerprint {fp:#010x}                     ({} lookups): regenerate with `zk-prover genemit` and rebuild libstwo_cuda.a",
                   ir.fracs.len())
        });
        assert_eq!(looks, ir.fracs.len(), "{family}: lookup count disagrees with the kernel");
        // B11 stream M: the component's preprocessed slice is wider than this
        // eval's on an o-side matmul.
        //
        // `mat_component` commits three extra selectors for a component that
        // publishes an output-side claim, and they are the *argmax*
        // component's -- it "shares the head dequant's step selectors"
        // (`compose.rs`, where it is built). `DequantEval` never reads them, so
        // its fraction IR declares the 506 columns it does read while the
        // component carries 509, and handing the kernel all of them fails
        // `logup_cols_dev_ptrs`' count check. The three sit *after* the ones
        // the IR loads, so the prefix is exactly what to supply -- and the
        // assert says so rather than trusting it.
        let np = ir.cols[0];
        let npre = |n: usize| {
            assert!(n >= np,
                    "{family}: the component has {n} preprocessed columns, fewer than \
                     the {np} its fraction IR loads");
            np
        };
        if let Some(d) = &r.dev {
            // B7a milestone 4 (19.6): the witness generation left this component's columns on
            // the device, so the fractions read them where they are and the
            // LogUp columns never come back. 19.4 measured both transfers and
            // said they disappear together; this is the together.
            let (cols, sum) = crate::devlogup::logup_cols_dev(
                &ir, family, shape, r.lay.log_rows, &d.pre[..npre(d.pre.len())], &d.base);
            return (cols.into_iter()
                        .map(|c| crate::device::AnyCol::OwnedDev(c, r.lay.log_rows))
                        .collect(),
                    sum);
        }
        let (cols, sum) = crate::devlogup::logup_cols_dev_to_host(
            &ir, family, shape, r.lay.log_rows, &r.pre[..npre(r.pre.len())], &r.base);
        return (crate::device::host_cols(cols), sum);
    }
    // the same prefix, for the interpreter: `logup_cols` indexes this list by
    // the IR's own column numbers, so a wider one is silently ignored rather
    // than loudly wrong -- but only the prefix is ever meant
    let pre: Vec<&[PackedM31]> =
        r.pre.iter().take(ir.cols[0]).map(|e| &e.data[..]).collect();
    let base: Vec<&[PackedM31]> = r.base.iter().map(|e| &e.data[..]).collect();
    let (cols, sum) = crate::cudagen::logup_cols(&ir, r.lay.log_rows, &pre, &base);
    (crate::device::host_cols(cols), sum)
}

fn mat_logup(m: &Mat, r: &MatComponent, rels: &crate::compose::Rels,
             air: &crate::gadgets::dequant::DequantEval)
             -> (Vec<crate::device::AnyCol<'static>>, SecureField) {
    // B7a milestone 4 (19.6): with the generate its witness on the device there is no host
    // column to build the fractions from, so the device path is the only one
    // -- and it is also the one that leaves 4a's bytes at zero.
    if r.dev.is_some() {
        return mat_logup_ir(r, air, true);
    }
    // Stream Z: the fold's witness generation is on the host and so is its LogUp -- the
    // hand-written builder below carries its looks, and no generated fraction
    // kernel is asked for.
    let logup_mode = if r.spec.variant == DeqVariant::Fold {
        0
    } else {
        crate::util::path_mode("ZK_GENERATED_LOGUP")
    };
    let from_ir = (logup_mode > 0).then(|| mat_logup_ir(r, air, logup_mode >= 3));
    // 2 checks the interpreter against the hand-written builder, 4 the kernel.
    if logup_mode == 1 || logup_mode == 3 {
        return from_ir.unwrap();
    }
    use stwo::prover::backend::simd::m31::PackedM31;
    use stwo::prover::backend::simd::qm31::PackedQM31;
    use stwo_constraint_framework::Relation;
    let map = &r.map;
    let f = map.f;
    let sp = &r.spec;
    let sticky = sp.round.needs_sticky();
    let one = PackedQM31::broadcast(SecureField::one());
    // pre layout: per-slot chan cols first (1 or 2 per slot), then selectors
    let chan_w = if matches!(sp.variant, DeqVariant::Chain | DeqVariant::Fold) { 2 } else { 1 };
    enum L {
        Norm([Src2; 6]),
        Frac([Src2; 3]),
        Mul([Src2; 4], Src2),
        Range(u32, Src2),
        /// stream Z, the fold: the derived complement `2 fdp - 1 - fe` and
        /// the clamp gap `qm - qc`, each into its range table
        Rem2(u32, Src2, Src2),
        Diff(u32, Src2, Src2),
    }
    #[derive(Clone, Copy)]
    enum Src2 {
        B(usize),
        P(usize),
        Zero,
    }
    let mut looks: Vec<L> = vec![];
    for s in 0..sp.per_row {
        // stream Z, the fold: no normalisation core -- its six looks stand alone,
        // in the AIR's `add_to_relation` order
        if sp.variant == DeqVariant::Fold {
            looks.push(L::Range(sp.q_bits, Src2::B(map.at(s, f.qm))));
            looks.push(L::Range(7, Src2::B(map.at(s, f.qc))));
            looks.push(L::Diff(sp.q_bits, Src2::B(map.at(s, f.qm)),
                               Src2::B(map.at(s, f.qc))));
            looks.push(L::Range(crate::gadgets::dequant::FOLD_LIMB, Src2::B(map.at(s, f.flo))));
            looks.push(L::Range(crate::gadgets::dequant::FOLD_LIMB, Src2::B(map.at(s, f.fb))));
            looks.push(L::Range(16, Src2::B(map.at(s, f.fa))));
            looks.push(L::Range(sp.rem_bits, Src2::B(map.at(s, f.fe))));
            looks.push(L::Rem2(sp.rem_bits, Src2::B(map.at(s, f.fdp)),
                               Src2::B(map.at(s, f.fe))));
            continue;
        }
        let st = if sticky { Src2::B(map.at(s, f.st)) } else { Src2::Zero };
        looks.push(L::Norm([Src2::B(map.at(s, f.m2)), Src2::B(map.at(s, f.ph)), st,
                            Src2::B(map.at(s, f.m7)), Src2::B(map.at(s, f.ef)),
                            Src2::B(map.at(s, f.nz))]));
        looks.push(L::Frac([Src2::B(map.at(s, f.ph)), Src2::B(map.at(s, f.r)), st]));
        match sp.variant {
            DeqVariant::Pow2 => looks.push(L::Range(8, Src2::B(map.at(s, f.eo)))),
            DeqVariant::Chain => {
                for t in 0..2 {
                    let m_in = if t == 0 { Src2::B(map.at(s, f.m7)) }
                               else { Src2::B(map.at(s, f.mul[0][0])) };
                    let ms = if t == 0 { Src2::B(map.shared[0]) } else { Src2::P(chan_w * s) };
                    looks.push(L::Mul([m_in, ms, Src2::B(map.at(s, f.mul[t][0])),
                                       Src2::B(map.at(s, f.mul[t][1]))],
                                      Src2::B(map.at(s, f.nz))));
                    looks.push(L::Range(8, Src2::B(map.at(s, f.mul[t][2]))));
                }
            }
            _ => unreachable!(),
        }
    }
    let zero = eval_of(r.lay.log_rows, vec![M31::zero(); 1 << r.lay.log_rows]);
    let get = |s: Src2, vr: usize| match s {
        Src2::B(i) => r.base[i].data[vr],
        Src2::P(i) => r.pre[i].data[vr],
        Src2::Zero => zero.data[vr],
    };
    let fr = |i: usize, vr: usize| -> (PackedQM31, PackedQM31) {
        match &looks[i] {
            L::Norm(c) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (one, rels.norm.combine(&v))
            }
            L::Frac(c) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (one, rels.frac.combine(&v))
            }
            L::Mul(c, m) => {
                let v: Vec<_> = c.iter().map(|s| get(*s, vr)).collect();
                (PackedQM31::from(get(*m, vr)), rels.mul.combine(&v))
            }
            L::Range(b, s) => (one, rels.range(*b).combine(&[get(*s, vr)])),
            L::Diff(b, x, y) => {
                let v = get(*x, vr) - get(*y, vr);
                (one, rels.range(*b).combine(&[v]))
            }
            L::Rem2(b, fdp, fe) => {
                let two = PackedM31::broadcast(M31::from_u32_unchecked(2));
                let onep = PackedM31::broadcast(M31::from_u32_unchecked(1));
                let v = get(*fdp, vr) * two - onep - get(*fe, vr);
                (one, rels.range(*b).combine(&[v]))
            }
        }
    };
    let (n_pairs, fpair) = crate::plogup::paired(looks.len(), fr);
    let host = crate::plogup::build(r.lay.log_rows, n_pairs, fpair);
    if let Some((cols, sum)) = from_ir {
        // `ZK_GENERATED_LOGUP=2`: the check. Every column, every cell, and the
        // claimed sum -- not a sample, because a lowering that is wrong on one
        // lookup of one row is wrong.
        use stwo::prover::backend::Column as _;
        assert_eq!(cols.len(), host.0.len(),
                   "{}: IR built {} LogUp columns, the hand-written builder {}",
                   m.name, cols.len(), host.0.len());
        for (i, (a, b)) in cols.iter().zip(&host.0).enumerate() {
            assert!(a.host().values.to_cpu() == b.values.to_cpu(),
                    "{}: LogUp column {i} differs between the IR and the hand-written builder",
                    m.name);
        }
        assert_eq!(sum, host.1, "{}: claimed sums differ", m.name);
        eprintln!("logup-ir: {} -- {} columns and the claimed sum match", m.name, cols.len());
    }
    (crate::device::host_cols(host.0), host.1)
}
