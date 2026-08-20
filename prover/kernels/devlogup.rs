//! B7a milestone 4 (18.7): the LogUp interaction, on the device.
//!
//! Section 6 lists "the LogUp prefix sums as kernels" as its own deliverable.
//! 18.4 found the prefix sums were never the cost -- `plogup::build`'s
//! *fractions* are, 89% of the dequant interaction -- and 18.7 found that the
//! fractions need no new generator: `ExprEvaluator::add_to_relation` makes each
//! lookup's denominator an extension intermediate, so 2b's lowering has been
//! emitting them since 2b and the only thing missing was the numerator. So the
//! per-family half of this file is *generated* (`gen_b65_<family>.cu`, from
//! `cudagen::emit_frac_family`), and what is written by hand is the four
//! phases that do not depend on the AIR at all.
//!
//! The pipeline, and which of `plogup`'s phases each step is:
//!
//! | step | what | `plogup` |
//! |---|---|---|
//! | `gen_b65_dispatch` | every lookup's numerator and denominator at every row, already batched in pairs | phase A, and `paired` |
//! | `batch_inverse_secure_field` | the denominators, one call over every column | phase A |
//! | `b65_logup_scale` | `frac = num * den^-1` | phase A |
//! | `b65_logup_running` | the pointwise running sum across columns | phase B |
//! | `b65_logup_partial_sums` + `b65_logup_finish_last` | the claimed sum, the shift, the prefix sum | phase C, `finalize_last` |
//!
//! Every one of those is exact field arithmetic, so the device columns equal
//! the host columns cell for cell or something is wrong -- which is what
//! `B7_M4_LOGUPIR=3` checks, and what the proof digest checks after it.

use std::ffi::{c_char, CString};

use stwo::core::fields::FieldExpOps;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::Column as _;
use stwo::stwo_cuda::base_field_vec::BaseFieldVec;
use stwo::stwo_cuda::secure_field_vec::SecureFieldVec;

use crate::common::Eval;
use crate::cudagen::Ir;

unsafe extern "C" {
    /// The generated per-family fraction kernel, behind one dispatcher keyed on
    /// `cuda_eval_name()` -- the same key the fork's constraint dispatch uses.
    fn gen_b65_dispatch(
        family: *const c_char,
        shape: u32,
        fingerprint: u32,
        pre_cols: *const *const u32,
        n_pre: u32,
        base_cols: *const *const u32,
        n_base: u32,
        sc_host: *const u32,
        n_sc: u32,
        num: *const u32,
        den: *const u32,
        n_rows: u32,
        n_cols: u32,
    );
    fn b65_logup_scale(num: *const u32, inv: *const u32, frac: *const u32, n: u32);
    fn b65_logup_running(
        frac: *const u32,
        out_cols: *const *const u32,
        last: *const u32,
        n_rows: u32,
        n_cols: u32,
    );
    fn b65_logup_partial_sums(
        last: *const u32,
        partials_host: *mut u32,
        n_rows: u32,
        n_partials: *mut u32,
    );
    fn b65_logup_finish_last(
        last: *const u32,
        out_cols: *const *const u32,
        shift: [u32; 4],
        n_rows: u32,
    );
    fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize);
}

/// One component's LogUp columns, computed on the device.
///
/// `pre` and `base` are the component's own columns, already resident. The
/// output is the committed columns and the claimed sum, in the same order and
/// with the same values `cudagen::logup_cols` produces on the host.
pub fn logup_cols_dev(
    ir: &Ir,
    family: &str,
    shape: usize,
    log_rows: u32,
    pre: &[BaseFieldVec],
    base: &[BaseFieldVec],
) -> (Vec<BaseFieldVec>, SecureField) {
    let n_rows = 1usize << log_rows;
    let n_looks = ir.fracs.len();
    let n_cols = n_looks.div_ceil(2);
    assert!(n_cols > 0, "a component with no lookups has no LogUp columns");
    assert_eq!(pre.len(), ir.cols[0], "{family}: preprocessed column count");
    assert_eq!(base.len(), ir.cols[1], "{family}: base column count");

    let pre_ptrs: Vec<*const u32> = pre.iter().map(|c| c.device_ptr).collect();
    let base_ptrs: Vec<*const u32> = base.iter().map(|c| c.device_ptr).collect();
    // the relation elements, flattened the way `qm31` is laid out on the device
    let sc: Vec<u32> = ir
        .scalars
        .iter()
        .flat_map(|s| s.to_m31_array().map(|x| x.0))
        .collect();

    let n = n_cols * n_rows;
    let num = SecureFieldVec::new_uninitialized(n);
    let den = SecureFieldVec::new_uninitialized(n);
    let cname = CString::new(family).unwrap();
    unsafe {
        gen_b65_dispatch(
            cname.as_ptr(),
            shape as u32,
            ir.fingerprint(),
            pre_ptrs.as_ptr(),
            pre_ptrs.len() as u32,
            base_ptrs.as_ptr(),
            base_ptrs.len() as u32,
            sc.as_ptr(),
            ir.scalars.len() as u32,
            num.device_ptr,
            den.device_ptr,
            n_rows as u32,
            n_cols as u32,
        );
    }

    // phase A's inversion: one call over every column, because inversion is
    // pointwise and the batch does not care where a column ends.
    let inv = SecureFieldVec::new_uninitialized(n);
    unsafe {
        batch_inverse_secure_field(den.device_ptr, inv.device_ptr, n);
        b65_logup_scale(num.device_ptr, inv.device_ptr, num.device_ptr, n as u32);
    }

    let out: Vec<BaseFieldVec> =
        (0..4 * n_cols).map(|_| BaseFieldVec::new_uninitialized(n_rows)).collect();
    let out_ptrs: Vec<*const u32> = out.iter().map(|c| c.device_ptr).collect();
    let last = SecureFieldVec::new_uninitialized(n_rows);
    unsafe {
        b65_logup_running(
            num.device_ptr,
            out_ptrs.as_ptr(),
            last.device_ptr,
            n_rows as u32,
            n_cols as u32,
        );
    }

    // phase C. The claimed sum comes back as block partials: a field sum is
    // associative and commutative, so where the split falls changes nothing,
    // and the host adding a few hundred QM31s costs nothing either.
    let blocks = n_rows.div_ceil(256);
    let mut partials = vec![0u32; 4 * blocks];
    let mut got = 0u32;
    unsafe {
        b65_logup_partial_sums(
            last.device_ptr,
            partials.as_mut_ptr(),
            n_rows as u32,
            &mut got,
        );
    }
    assert_eq!(got as usize, blocks, "{family}: partial-sum block count");
    let claimed_sum = partials
        .chunks_exact(4)
        .map(|c| {
            SecureField::from_m31_array([
                M31::from_u32_unchecked(c[0]),
                M31::from_u32_unchecked(c[1]),
                M31::from_u32_unchecked(c[2]),
                M31::from_u32_unchecked(c[3]),
            ])
        })
        .sum::<SecureField>();
    let cumsum_shift = claimed_sum * BaseField::from_u32_unchecked(1 << log_rows).inverse();
    let shift: [u32; 4] = cumsum_shift.to_m31_array().map(|x| x.0);
    unsafe {
        b65_logup_finish_last(
            last.device_ptr,
            out_ptrs[4 * (n_cols - 1)..].as_ptr(),
            shift,
            n_rows as u32,
        );
    }
    (out, claimed_sum)
}

/// The same, brought back to the host so it can be compared with the
/// hand-written builder cell for cell (`B7_M4_LOGUPIR=3`).
///
/// The download is the check's, not the prover's: nothing on the milestone's
/// own path moves a generated column back across the link.
pub fn logup_cols_dev_to_host(
    ir: &Ir,
    family: &str,
    shape: usize,
    log_rows: u32,
    pre: &[Eval],
    base: &[Eval],
) -> (Vec<Eval>, SecureField) {
    let up = |cols: &[Eval]| -> Vec<BaseFieldVec> {
        cols.iter().map(|e| BaseFieldVec::from_vec(e.values.to_cpu())).collect()
    };
    // The three costs, apart, because only the middle one is the milestone's:
    // the upload is here because the *fills* are still on the host (18.7's row
    // 1, still open) and the download is the check's alone.
    let t = std::time::Instant::now();
    let dpre = up(pre);
    let dbase = up(base);
    let t_up = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let (out, sum) = logup_cols_dev(ir, family, shape, log_rows, &dpre, &dbase);
    let t_dev = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let domain = stwo::core::poly::circle::CanonicCoset::new(log_rows).circle_domain();
    let cols = out
        .iter()
        .map(|c| {
            let v: Vec<M31> = c.to_vec();
            stwo::prover::poly::circle::CircleEvaluation::new(
                domain,
                stwo::prover::backend::simd::column::BaseColumn::from_iter(v),
            )
        })
        .collect();
    eprintln!(
        "logup-dev: {family} shape {shape} 2^{log_rows} x {} lookups -- \
         kernels {:.4} s, upload {:.4} s, download {:.4} s",
        ir.fracs.len(),
        t_dev,
        t_up,
        t.elapsed().as_secs_f64()
    );
    (cols, sum)
}
