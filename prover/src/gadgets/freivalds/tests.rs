//! What the digests cannot say: that the circuit is the one the statement
//! describes.
//!
//! A pinned digest says the proof did not change, which is also what it says
//! when the circuit was wrong before and is wrong in the same way now.  The
//! three tests here say something about the circuit itself -- that an honest
//! trace satisfies every constraint on every row, that each control breaks the
//! one constraint it is written for, and that the identity accepts honest
//! claims and rejects a changed one.

use stwo::core::channel::Channel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo::prover::backend::Column;

use num_traits::{One, Zero};

use super::*;
use crate::bf16::Round;
use crate::columns::{ByteCells, Lookups, constraint_values};
use crate::common::Eval;
use crate::gadgets::attention::witness::{Src, Variant};
use crate::gadgets::freivalds::claims::{FreivaldsClaims, freivalds_interaction, identity_holds,
                                        project_product};
use crate::gadgets::freivalds::columns::{FreivaldsShape, Known, constraint_names};
use crate::gadgets::freivalds::layout::FREIVALDS_GROUPS;
use crate::gadgets::freivalds::trace::freivalds_component;
use crate::gadgets::freivalds::witness::FreivaldsWitness;
use crate::layer::{Layer, Tables};
use crate::relations::RangeRel;

const ACT_BITS: u32 = 8;
const P_TARGET: usize = 8;

/// A synthetic layer small enough that a whole Freivalds group builds,
/// closes its accumulators and is checked row by row in well under a second.
fn small_layer() -> Layer {
    let src = Src { dir: None, head: 0, q_bits: 12, k_bits: 11, nmax: 255, log_table: 8,
                    seed: 7, layout: String::new(), diagonal_block: false, tile: false };
    Layer::synthetic(8, 8, 8, &[1], Variant::parse("intsplit6"), &src, 16, 16, 0x221)
}

/// Which of a layer's matmuls a Freivalds group is over.
fn group_mats(l: &Layer, group: &str) -> Vec<usize> {
    let names = FREIVALDS_GROUPS.iter().find(|(g, _)| *g == group)
        .unwrap_or_else(|| panic!("no Freivalds group {group}")).1;
    l.mats.iter().enumerate()
        .filter(|(_, m)| names.contains(&m.name.as_str())).map(|(i, _)| i).collect()
}

/// One built component: the AIR, its three trees, what it published, and
/// what the controls left outside the constraints.
struct Built {
    air: FreivaldsEval,
    pre: Vec<Eval>,
    base: Vec<Eval>,
    inter: Vec<Eval>,
    use_sum: SecureField,
    off_table: usize,
    /// the layer's matmul indices this group is over
    mats: Vec<usize>,
    /// what the component published: one `(rv^T W_m)(X s)` per matmul
    claims: FreivaldsClaims,
    /// the two challenges the claims were built against
    rv: SecureField,
    s: SecureField,
}

/// Build one group exactly the way `run.rs` composes it, with the two
/// controls `trace.rs` takes.
fn build(l: &Layer, group: &str, with_qx: bool, corrupt_qx: Option<usize>,
         oversize_qx: Option<usize>) -> Built {
    let mats = group_mats(l, group);
    let w = FreivaldsWitness::of_layer(l, group, &mats);
    let mut tables = Tables::new(vec![0u32; 16], 4, 0, Round::Even, 4, vec![ACT_BITS], 0, 0);
    let r = freivalds_component(&w, P_TARGET, ACT_BITS, &mut tables, corrupt_qx, oversize_qx);
    let lay = r.lay;
    // fixed challenges: what is under test is the constraints, not the transcript
    let mut ch = stwo::core::channel::Blake2sM31Channel::default();
    ch.mix_u64(0x221);
    let rel_act = RangeRel::draw(&mut ch);
    let f = ch.draw_secure_felts(5);
    let (rv, s_ch) = (Powers::against(f[0], lay.pk), Powers::against(f[1], lay.pq));
    // the activation claim's weights and row steps are the wiring family's;
    // the constraints hold for any of them, so the test draws them
    let qx = with_qx.then(|| FreivaldsQx {
        w: Powers::against(f[2], lay.pq).cells, m_in: f[3], m_bd: f[4],
    });
    let sh = FreivaldsShape::new(group, &lay, w.mats.len(), with_qx);
    let air = |claims| FreivaldsEval {
        shape: sh.clone(), rv: rv.clone(), s: s_ch.clone(), claims,
        acts: ByteCells { bits: ACT_BITS, off: 1 << (ACT_BITS - 1),
                          rel: rel_act.clone(), by: Lookups::Own },
        qx: qx.clone(),
    };
    let open = air(FreivaldsClaims::open(w.mats.len(), with_qx));
    let it = freivalds_interaction(&w, &r, &open);
    // the claims the component publishes close the last two constraints
    let air = air(it.claims.clone());
    Built {
        pre: r.pre.clone(),
        base: r.base.clone(),
        inter: it.cols.into_iter().map(|c| c.into_host()).collect(),
        use_sum: it.use_sum,
        off_table: tables.off_table,
        mats: r.mats.clone(),
        claims: it.claims,
        rv: f[0],
        s: f[1],
        air,
    }
}

/// The (row, constraint name) pairs whose constraint does not hold.
///
/// The preprocessed tree is handed over in the order the AIR *reads* it,
/// which is the row schema's declaration order and is what a component's own
/// mask is numbered by; the committed order is the schema's too, and this is
/// the one place the two meet.
fn broken(b: &Built) -> Vec<(usize, String)> {
    let sh = &b.air.shape;
    let committed = Known::committed_ids(sh);
    assert_eq!(committed.len(), b.pre.len(), "the committed ids are not the tree");
    // A component's own preprocessed mask is numbered in the order it *reads*
    // its columns, and the tree is committed in another one, so the columns
    // are gathered here the way `FrameworkComponent` gathers them: by id.
    // The read order is the AIR's own, recorded rather than restated.
    let read = b.air
        .evaluate(stwo_constraint_framework::InfoEvaluator::new(
            b.air.log_size(), vec![], b.use_sum))
        .preprocessed_columns;
    let pre: Vec<Vec<M31>> = {
        let by_id: std::collections::HashMap<&str, &Eval> =
            committed.iter().map(|i| i.id.as_str()).zip(b.pre.iter()).collect();
        read.iter().map(|id| by_id[id.id.as_str()].values.to_cpu()).collect()
    };
    let base: Vec<Vec<M31>> = b.base.iter().map(|e| e.values.to_cpu()).collect();
    let inter: Vec<Vec<M31>> = b.inter.iter().map(|e| e.values.to_cpu()).collect();
    assert_eq!((pre.len(), base.len()), (sh.n_pre(), sh.n_base()),
               "the trees are not the ones the row declares");
    let trace: TreeVec<Vec<&Vec<M31>>> = TreeVec::new(vec![
        pre.iter().collect(),
        vec![],
        base.iter().collect(),
        inter.iter().collect(),
    ]);
    let names = constraint_names(sh);
    let mut bad = vec![];
    for (row, vals) in constraint_values(&b.air, &trace, b.use_sum).into_iter().enumerate() {
        assert_eq!(vals.len(), names.len(),
                   "the AIR adds {} constraints and the declaration names {}",
                   vals.len(), names.len());
        for (v, n) in vals.iter().zip(&names) {
            if !v.is_zero() {
                bad.push((row, n.clone()));
            }
        }
    }
    bad
}

/// A trace built through the declaration's writers satisfies every constraint
/// of the AIR that reads through it, on every row of the domain.
///
/// This is the half the pinned digests cannot cover: a digest says the proof
/// did not change, which is also what it says when the circuit was wrong
/// before and is wrong in the same way now.  What fails here is a reader and
/// a writer that disagree about a column's position, a selector true on the
/// wrong rows, a running sum carried across the wrong boundary, or a padded
/// row the constraints do not hold on -- by name, and at the row.
#[test]
fn the_honest_trace_satisfies_every_constraint_on_every_row() {
    let l = small_layer();
    for group in ["o", "qkv", "gateup"] {
        for qx in [false, true] {
            let b = build(&l, group, qx, None, None);
            assert_eq!(b.off_table, 0,
                       "{group}: an honest activation fell off the range table");
            let bad = broken(&b);
            assert!(bad.is_empty(), "{group} (qx={qx}): {} constraint(s) do not hold, \
                                     first {:?}", bad.len(), &bad[..bad.len().min(4)]);
        }
    }
}

/// And each control breaks the constraint it is written for and no other.
///
/// `corrupt_qx` moves one *committed* activation by one and leaves `x_fold`
/// on the honest witness, so the one equality that stops holding is
/// `x_fold`, on the one row that carries that cell -- and `x_claim_fold`
/// with it on a component that publishes a claim over the activations, which
/// folds the same committed cells against different weights.
///
/// `oversize_qx` is not a constraint control at all: it puts a committed
/// activation one past its range table, which is the *table's* business.  The
/// row sees it only as the same activation disagreeing with `x_fold`, and
/// what it really broke is the off-table counter.
#[test]
fn each_control_breaks_the_constraint_it_targets_and_nothing_else() {
    let l = small_layer();
    let idx = 5usize;
    for (control, off_table) in [("corrupt", 0usize), ("oversize", 1usize)] {
        for qx in [false, true] {
            let (c, o) = match control {
                "corrupt" => (Some(idx), None),
                _ => (None, Some(idx)),
            };
            let b = build(&l, "o", qx, c, o);
            let bad = broken(&b);
            let want: Vec<&str> = if qx { vec!["x_fold", "x_claim_fold"] }
                                  else { vec!["x_fold"] };
            assert_eq!(bad.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>(), want,
                       "{control}_qx (qx={qx}) broke {bad:?}");
            assert!(bad.windows(2).all(|w| w[0].0 == w[1].0),
                    "{control}_qx (qx={qx}) broke more than one row: {bad:?}");
            assert_eq!(b.off_table, off_table,
                       "{control}_qx (qx={qx}): off-table count");
        }
    }
}

/// The equality a composed verifier runs, on claims from honest builds.
///
/// One side is what each Freivalds component published for its matmuls; the
/// other is `project_product` over the layer's exact accumulators, which is
/// the value the dequant component publishes as `claimed_h`.  They have to
/// agree matmul for matmul, and moving one published product has to be seen
/// -- at that matmul and at no other.
#[test]
fn the_identity_accepts_honest_claims_and_rejects_a_changed_one() {
    let l = small_layer();
    let mut groups: Vec<FreivaldsClaims> = vec![];
    for group in ["o", "qkv", "gateup"] {
        let b = build(&l, group, false, None, None);
        // the consumed side: what the dequant component publishes for the same
        // matmuls, off the layer's exact integer accumulators
        let consumed = b.mats.iter()
            .map(|mi| {
                let m = &l.mats[*mi];
                project_product(&m.acc, m.tokens, m.n_out, b.rv, b.s)
            })
            .collect();
        groups.push(FreivaldsClaims { consumed, ..b.claims });
    }
    let honest = identity_holds(&groups);
    assert_eq!(honest.len(), groups.iter().map(|g| g.product.len()).sum::<usize>());
    assert!(honest.iter().all(|x| *x),
            "an honest run's claims do not satisfy the identity: {honest:?}");

    // and moving any one of the published products breaks that one alone
    for flat in 0..honest.len() {
        let mut bad: Vec<FreivaldsClaims> = groups.clone();
        let (mut gi, mut k) = (0usize, flat);
        while k >= bad[gi].product.len() {
            k -= bad[gi].product.len();
            gi += 1;
        }
        bad[gi].product[k] += SecureField::one();
        let got = identity_holds(&bad);
        let want: Vec<bool> = (0..honest.len()).map(|i| i != flat).collect();
        assert_eq!(got, want, "changing product {flat} was not seen where it was made");
    }
}
