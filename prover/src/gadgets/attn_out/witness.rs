//! The attention-output fold's witness.

use crate::gadgets::attn_out::layout::AoLayout;

/// The attention-output fold's witness: the A.V outputs of every head, their
/// denominators, and the power-of-two scale exponents.
pub struct AttnOut {
    pub nh: usize,
    pub s: usize,
    pub hd: usize,
    /// (nh, S, hd): the A.V outputs (the rectangle recomputation, as the key tiles)
    pub acc: Vec<i64>,
    /// (nh, S): the committed denominators Z
    pub den: Vec<i64>,
    /// per-token output scale exponent (pow2 s_o)
    pub jo: Vec<i32>,
    /// per-(head, channel) V scale exponent (pow2 s_v)
    pub jv: Vec<i32>,
}

impl AttnOut {
    pub fn e(&self, h: usize, t: usize, j: usize) -> i32 {
        self.jv[h * self.hd + j] - self.jo[t]
    }
    /// Stream JJ (E11): the per-token output exponent's window, from the data.
    /// `jo` is prover data; this is the window the statement admits it in.
    pub fn jo_range(&self) -> (i32, i32) {
        let lo = self.jo.iter().copied().min().unwrap_or(0);
        let hi = self.jo.iter().copied().max().unwrap_or(0);
        (lo, hi)
    }
    /// Which way a one-step `jo` move can go on slots whose exponents are `es`,
    /// or `None` when the statement's own windows leave no room.
    ///
    /// A `jo` control has to stay inside those windows, or the `EPOW` lookup
    /// would catch the move instead of the pin the control exists to test:
    /// `joff = jo_max - jo` must stay non-negative and every slot's
    /// `e = jv - jo` must stay in `[e_lo, e_hi]`.  Up first, because a larger
    /// `jo` is a smaller `num` and a smaller `num` cannot push `qm` past the
    /// `ao_q_bits` pair table; moving `jo` down can, which is itself worth
    /// knowing -- see the record.
    ///
    /// `None` is not a defect in this arithmetic. It says the two windows
    /// between them admit exactly one `jo` for these slots, so there is no `jo`
    /// a prover could commit that they do not already refuse, and the `joff`
    /// pin has nothing left to catch here. A witness whose `jv` spans the whole
    /// exponent window and whose every token carries the same `jo` -- the
    /// synthetic layer's -- is exactly that shape, and `compose-probes` skips
    /// the two controls on it by name rather than asserting on them.
    pub fn jo_step(es: &[i32], jo: i32, jo_max: i32, e_lo: i32, e_hi: i32) -> Option<i32> {
        if jo < jo_max && es.iter().all(|e| e - 1 >= e_lo) {
            Some(1)
        } else if es.iter().all(|e| e + 1 <= e_hi) {
            Some(-1)
        } else {
            None
        }
    }
    /// The exponents one row of the attention output covers: its token's, over
    /// the `per_row` channels of its channel block.
    pub fn row_es(&self, lay: &AoLayout, h: usize, t: usize, b: usize) -> Vec<i32> {
        (0..lay.per_row)
            .map(|s| self.jv[h * self.hd + b * lay.per_row + s] - self.jo[t])
            .collect()
    }
    /// The site `--corrupt-ao-jo` can be stated on: a channel index whose row
    /// has room for a one-step `jo` move, or `None` when no row of this witness
    /// has. The row must also not be its token's first, since the `nofirst` pin
    /// compares a row with its predecessor -- the same rule `ao_component`
    /// asserts on the index it is handed.
    pub fn jo_bend_site(&self, lay: &AoLayout, jo_max: i32, e_lo: i32, e_hi: i32)
                        -> Option<usize> {
        (0..1usize << lay.log_rows).find_map(|j| {
            let (h, t, b) = lay.at(self.nh, self.s, j)?;
            if b + 1 >= lay.chunks {
                return None;
            }
            Self::jo_step(&self.row_es(lay, h, t, b), self.jo[t], jo_max, e_lo, e_hi)?;
            Some((h * self.s + t) * self.hd + b * lay.per_row)
        })
    }
    /// The same for `--corrupt-ao-jo-tok`, whose move is over every row of one
    /// token: the token whose every slot has room, or `None`.
    pub fn jo_tok_site(&self, jo_max: i32, e_lo: i32, e_hi: i32) -> Option<usize> {
        (0..self.s).find(|t| {
            let es: Vec<i32> = (0..self.nh)
                .flat_map(|h| (0..self.hd).map(move |c| (h, c)))
                .map(|(h, c)| self.jv[h * self.hd + c] - self.jo[*t])
                .collect();
            Self::jo_step(&es, self.jo[*t], jo_max, e_lo, e_hi).is_some()
        })
    }
    /// Stream II (E11), at the JJ merge: the attention output's requantized
    /// integers, `sign(acc) * round(|acc| 2^(jv - jo) / den)` per
    /// `(head, token, channel)` in that order -- exactly what `ao_component`
    /// commits as `q_out` and what `AttnOutEval`'s `claimed_q` folds.
    ///
    /// It lives here, off `AttnOut` alone, so a witness builder can produce the
    /// `o` matmul's committed input without running the component: on a real dump
    /// B1 wrote that input and the component recomputes it, but a *synthetic*
    /// layer draws its `o` activations at random and the `o` seam could not be
    /// honest on any desk shape.  `ao_component` asserts its own `q_out` against
    /// this on every untampered run, so the two cannot drift apart.
    pub fn q_ref(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.nh * self.s * self.hd];
        for h in 0..self.nh {
            for t in 0..self.s {
                let den = self.den[h * self.s + t];
                for c in 0..self.hd {
                    let idx = (h * self.s + t) * self.hd + c;
                    let e = self.jv[h * self.hd + c] - self.jo[t];
                    let (num, pj) = (1i64 << e.max(0), 1i64 << (-e).max(0));
                    let (denp, pm) = (den * pj, self.acc[idx].abs() * num);
                    let qm = (2 * pm + denp) / (2 * denp);
                    out[idx] = if self.acc[idx] < 0 { -qm } else { qm };
                }
            }
        }
        out
    }

    /// The per-(head, channel) V exponent's window.  `jv` is calibrated offline
    /// and is a model constant, so this is a statement of the model.
    pub fn jv_range(&self) -> (i32, i32) {
        let lo = self.jv.iter().copied().min().unwrap_or(0);
        let hi = self.jv.iter().copied().max().unwrap_or(0);
        (lo, hi)
    }
    /// The bit length of `1 + max |acc|`: the width the two-limb range check on
    /// the accumulator has to cover, and with it the `2 mag num < p` half of
    /// the no-wrap window.
    pub fn acc_bits(&self) -> u32 {
        let m = self.acc.iter().map(|a| a.abs()).max().unwrap_or(0);
        64 - ((m as u64) + 1).leading_zeros()
    }

    /// The rounding-error range check's width: `d < 2 den 2^max(-e,0)` over every
    /// active site, from the data rather than a blind bound.
    pub fn rem_bits(&self) -> u32 {
        let mut m = 2i64;
        for h in 0..self.nh {
            for t in 0..self.s {
                for j in 0..self.hd {
                    let denp = self.den[h * self.s + t] << (-self.e(h, t, j)).max(0);
                    m = m.max(2 * denp);
                }
            }
        }
        64 - (m as u64).leading_zeros()
    }
}
