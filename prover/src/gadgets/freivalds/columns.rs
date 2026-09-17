//! The Freivalds check's columns, declared once (#221).
//!
//! Three structs, one per tree, each generic over what a column is to whoever
//! holds the row.  The AIR instantiates them with the evaluator's field types
//! and reads named fields (`FreivaldsRow::read`); `trace.rs` instantiates the
//! preprocessed and base trees with committed columns and `claims.rs` the
//! interaction tree with the running columns' values, and all three hand their
//! tree back through `into_committed`, which is the one place a column's
//! position is written down.  `crate::columns` says what the scheme is and why
//! it is not a macro.
//!
//! Two orders live here and they are not the same one:
//!
//! - **the read order**, which is the field order of each struct.  It is what
//!   the AIR's masks are numbered by and therefore what `cudagen`'s lowering
//!   records, so moving a field moves a generated kernel's fingerprint.  The
//!   preprocessed tree is read selectors first, then the weights.
//! - **the committed order**, which is `into_committed`.  For the base and the
//!   interaction trees it is the read order.  For the preprocessed tree it is
//!   *not*: `compose.rs` and `run.rs` register the weights first and the four
//!   selectors after, and a preprocessed column is resolved by id rather than by
//!   position, so the two orders are free to differ and do.

use stwo_constraint_framework::EvalAtRow;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::columns::Run;
use crate::gadgets::freivalds::layout::FreivaldsLayout;
use crate::gadgets::freivalds::{FREIVALDS_SEL, FreivaldsEval, freivalds_sel, freivalds_w};

/// What the row's lengths come from: one Freivalds component's shape.
///
/// It borrows the group's name rather than owning it: the AIR builds one of
/// these on every `evaluate`, which is once per row of the evaluation domain.
#[derive(Clone, Copy, Debug)]
pub struct FreivaldsShape<'a> {
    /// the group's name, which every preprocessed id carries
    pub tag: &'a str,
    pub n_mats: usize,
    /// weights per row per matmul
    pub pk: usize,
    /// activation bytes per row
    pub pq: usize,
    /// whether the component carries stream II's activation claim
    pub qx: bool,
}

impl<'a> FreivaldsShape<'a> {
    /// The shape the AIR is over.
    pub fn of(air: &'a FreivaldsEval) -> Self {
        FreivaldsShape { tag: &air.tag, n_mats: air.n_mats, pk: air.pk, pq: air.pq,
                         qx: air.qx.is_some() }
    }
    /// The shape a builder is over, off the layout it tiles.
    pub fn new(tag: &'a str, lay: &FreivaldsLayout, n_mats: usize, qx: bool) -> Self {
        FreivaldsShape { tag, n_mats, pk: lay.pk, pq: lay.pq, qx }
    }

    /// How many columns each tree holds, off the same declaration the structs
    /// are: the weights and the four selectors, the activations, and the
    /// running accumulators.  The interaction count is in *running columns*;
    /// each is four M31 columns once encoded, and the LogUp columns that follow
    /// them belong to the relation rather than to the row.
    pub fn n_pre(&self) -> usize {
        self.n_mats * self.pk + FREIVALDS_SEL.len()
    }
    pub fn n_base(&self) -> usize {
        self.pq
    }
    pub fn n_inter(&self) -> usize {
        1 + 2 * self.n_mats + usize::from(self.qx)
    }
}

// ----------------------------------------------------------- preprocessed ----

/// The four step selectors and the pinned weights.
///
/// `nofirst` is "this row is not a channel's first chunk", `last` is "this row
/// is a channel's chunk 0", `notzero` is `j > 0` and `end` is the last row.
pub struct FreivaldsPre<C> {
    pub nofirst: C,
    pub last: C,
    pub notzero: C,
    pub end: C,
    /// `[matmul][weight in the row]`
    pub w: Vec<Vec<C>>,
}

impl<C> FreivaldsPre<C> {
    /// The tree's columns in *committed* order: every matmul's weights, then
    /// the four selectors.  `compose.rs` and `run.rs` build the id list in this
    /// order and the device witness generation writes its columns in it.
    pub fn into_committed(self) -> Vec<C> {
        let mut v: Vec<C> = self.w.into_iter().flatten().collect();
        v.push(self.nofirst);
        v.push(self.last);
        v.push(self.notzero);
        v.push(self.end);
        v
    }
}

impl FreivaldsPre<PreProcessedColumnId> {
    /// The component's preprocessed ids, one per field.
    pub fn ids(sh: &FreivaldsShape) -> Self {
        FreivaldsPre {
            nofirst: freivalds_sel(sh.tag, FREIVALDS_SEL[0]),
            last: freivalds_sel(sh.tag, FREIVALDS_SEL[1]),
            notzero: freivalds_sel(sh.tag, FREIVALDS_SEL[2]),
            end: freivalds_sel(sh.tag, FREIVALDS_SEL[3]),
            w: (0..sh.n_mats)
                .map(|m| (0..sh.pk).map(|p| freivalds_w(sh.tag, m, p)).collect())
                .collect(),
        }
    }
    /// The same ids in committed order -- the list the trace location allocator
    /// takes.
    pub fn committed_ids(sh: &FreivaldsShape) -> Vec<PreProcessedColumnId> {
        Self::ids(sh).into_committed()
    }
}

// ------------------------------------------------------------------- base ----

/// The committed activation bytes of one token chunk, offset by `act_off`.
pub struct FreivaldsBase<C> {
    pub act: Vec<C>,
}

impl<C> FreivaldsBase<C> {
    pub fn into_committed(self) -> Vec<C> {
        self.act
    }
}

// ------------------------------------------------------------ interaction ----

/// One matmul's two accumulators: `u_m = sum_i rv^i W_m[i, k]` down the row
/// group, and `acc_m = sum_k u_m v` across the channels.
pub struct FreivaldsMat<C> {
    pub u: C,
    pub acc: C,
}

/// The Horner accumulators, in the order the interaction tree holds them: the
/// shared `v`, then each matmul's `u` and `acc`, then stream II's activation
/// claim.  The LogUp columns follow these and are the relation's, not the row's.
pub struct FreivaldsInter<C> {
    /// `v = sum_t s^t qX[k, t]`, shared by every matmul of the group
    pub v: C,
    pub mats: Vec<FreivaldsMat<C>>,
    /// stream II's running claim over the committed activations, or `None` on a
    /// chunk with no wiring family
    pub qx: Option<C>,
}

impl<C> FreivaldsInter<C> {
    pub fn into_committed(self) -> Vec<C> {
        let mut v = Vec::with_capacity(1 + 2 * self.mats.len() + 1);
        v.push(self.v);
        for m in self.mats {
            v.push(m.u);
            v.push(m.acc);
        }
        v.extend(self.qx);
        v
    }
}

// -------------------------------------------------------------- the reader ---

/// One row of the Freivalds check, as the AIR sees it.
pub struct FreivaldsRow<F, EF> {
    pub pre: FreivaldsPre<F>,
    pub base: FreivaldsBase<F>,
    pub inter: FreivaldsInter<Run<EF>>,
}

impl<F, EF> FreivaldsRow<F, EF> {
    /// Pull the component's columns out of `eval` in declaration order.
    ///
    /// This is the order the masks are numbered in and so the order the
    /// generated kernel is emitted against; it is the same order the check has
    /// read its columns in since it was written, and
    /// `the_honest_trace_satisfies_every_constraint_on_every_row` is what says
    /// so rather than this comment.
    pub fn read<E: EvalAtRow<F = F, EF = EF>>(eval: &mut E, sh: &FreivaldsShape) -> Self {
        let ids = FreivaldsPre::ids(sh);
        FreivaldsRow {
            pre: FreivaldsPre {
                nofirst: eval.get_preprocessed_column(ids.nofirst),
                last: eval.get_preprocessed_column(ids.last),
                notzero: eval.get_preprocessed_column(ids.notzero),
                end: eval.get_preprocessed_column(ids.end),
                w: ids
                    .w
                    .into_iter()
                    .map(|m| m.into_iter().map(|id| eval.get_preprocessed_column(id)).collect())
                    .collect(),
            },
            base: FreivaldsBase { act: (0..sh.pq).map(|_| eval.next_trace_mask()).collect() },
            inter: FreivaldsInter {
                v: Run::read(eval),
                mats: (0..sh.n_mats)
                    .map(|_| FreivaldsMat { u: Run::read(eval), acc: Run::read(eval) })
                    .collect(),
                qx: sh.qx.then(|| Run::read(eval)),
            },
        }
    }
}

/// What each of `FreivaldsEval::evaluate`'s constraints states, in the order it
/// adds them, so a row-level test can say which one a control broke.
pub fn constraint_names(sh: &FreivaldsShape) -> Vec<String> {
    let mut names = vec!["v_horner".to_string()];
    for m in 0..sh.n_mats {
        names.push(format!("u_horner[{m}]"));
        names.push(format!("acc_step[{m}]"));
        names.push(format!("acc_claim[{m}]"));
    }
    if sh.qx {
        names.push("qx_horner".to_string());
        names.push("qx_claim".to_string());
    }
    names
}
