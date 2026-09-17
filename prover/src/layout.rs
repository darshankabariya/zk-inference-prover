//! The row map a matmul-shaped component tiles its rows over.
//!
//! `MatLayout` is the matmul dequant component's tiling, and it is crate-level
//! rather than the dequant gadget's own because four other places lay their rows
//! out the same way on purpose: the requantize seam and the rotation sit on the
//! row map of the dequant they feed, `argmax` tiles a vocabulary with it, and
//! `devacc` reads it to place a device column. A gadget borrowing another
//! gadget's layout is the edge #196 split `relations` and `encoding` out to
//! remove, so the shared tiling lives beside them instead.
//!
//! The layout, once, because every builder and every selector has to agree:
//! rows = (token, channel-chunk), `per_row` channels per row with
//! `per_row | n_out`, so the component tiles *exactly* (no padded slots, no
//! gating).  Tokens and chunks descend, slots ascend: row `j` holds token
//! `T-1 - j/chunks`, chunk `b = chunks-1 - j%chunks`, channel
//! `c = b*per_row + slot`.

/// SIMD lane floor, as `common.rs`'s `MIN_LOG_ROWS`.
pub const MIN_MAT_LOG_ROWS: u32 = 4;

/// The matmul dequant component's tiling: `per_row | n_out`, with `tokens * chunks`
/// a power of two exactly (`choose`) or padded up to one (`choose_padded`).
#[derive(Clone, Copy, Debug)]
pub struct MatLayout {
    pub per_row: usize,
    pub chunks: usize,
    pub log_rows: u32,
    /// B10 step 4: inert lead rows, as `AoLayout` and `Grid` have always had.
    /// Zero for every component that existed before the `lm_head` chunk, because a
    /// layer matmul's output width is a power of two and `tokens * chunks` lands
    /// on one exactly.  A *vocabulary* does not: 8B's 128,256 is `2^8 x 501` and
    /// Qwen's 151,936 is `2^7 x 1187`.
    pub lead: usize,
}

impl MatLayout {
    /// The padded chooser: a width that divides `n_out` and is near `target`,
    /// with the row count rounded *up* to a power of two and the slack carried as
    /// inert lead rows.
    ///
    /// The policy is not just "pad": how much padding costs depends entirely on
    /// the odd part of `tokens * n_out`.  At 8B that odd part is 501 against a
    /// 512 boundary, so any width pads **2.1%**; at Qwen it is 1187 against 2048,
    /// so any power-of-two width pads **42%** and the sane choice is the exact
    /// width 1187 instead.  So a candidate is rejected when its lead fraction
    /// exceeds `MAX_LEAD_FRAC`, and the search widens until an exact tiling
    /// appears -- which one always does at `pr = odd_part(tokens * n_out)`.
    pub fn choose_padded(n_out: usize, tokens: usize, target: usize) -> Self {
        const MAX_LEAD_FRAC: f64 = 0.25;
        let mk = |pr: usize| -> Option<Self> {
            if pr == 0 || n_out % pr != 0 {
                return None;
            }
            let chunks = n_out / pr;
            let active = tokens * chunks;
            let log_rows = (usize::BITS - (active - 1).leading_zeros()).max(MIN_MAT_LOG_ROWS);
            let rows = 1usize << log_rows;
            Some(Self { per_row: pr, chunks, log_rows, lead: rows - active })
        };
        let mut best: Option<Self> = None;
        // widen the search until something acceptable turns up; `n_out` itself is
        // always a legal width (chunks = 1, rows = tokens), so this terminates
        for pr in 1..=n_out {
            let Some(l) = mk(pr) else { continue };
            if (l.lead as f64) / ((1usize << l.log_rows) as f64) > MAX_LEAD_FRAC {
                continue;
            }
            let score = (pr as isize - target as isize).abs();
            let better = best.as_ref().map(|b| {
                let bs = (b.per_row as isize - target as isize).abs();
                score < bs
            }).unwrap_or(true);
            if better {
                best = Some(l);
            }
            // once past the target, a wider row is only ever a worse score unless
            // it is the exact tiling the odd part forces, so keep scanning but do
            // not let the loop run to `n_out` for no reason
            if pr > target && best.as_ref().is_some_and(|b| b.lead == 0) {
                break;
            }
        }
        best.expect("no layout, padded or otherwise")
    }

    /// `row -> (token, chunk)`, or `None` for an inert lead row.
    pub fn at_opt(&self, tokens: usize, j: usize) -> Option<(usize, usize)> {
        if j < self.lead {
            return None;
        }
        let ja = j - self.lead;
        Some((tokens - 1 - ja / self.chunks, self.chunks - 1 - ja % self.chunks))
    }

    /// Exact when one exists, padded otherwise.  Every shape the project proved
    /// before the `lm_head` chunk takes the exact branch, so this is a no-op for
    /// them; a vocabulary takes the other one.
    pub fn choose_or_padded(n_out: usize, tokens: usize, target: usize) -> Self {
        Self::try_choose(n_out, tokens, target)
            .unwrap_or_else(|| Self::choose_padded(n_out, tokens, target))
    }

    pub fn choose(n_out: usize, tokens: usize, target: usize) -> Self {
        Self::try_choose(n_out, tokens, target).expect("no exact tiling exists")
    }

    fn try_choose(n_out: usize, tokens: usize, target: usize) -> Option<Self> {
        // the largest divisor of n_out at most `target` whose row count is a
        // power of two (tokens is one in every real shape here)
        let mut best = None;
        for pr in 1..=n_out.min(4 * target.max(1)) {
            if n_out % pr != 0 {
                continue;
            }
            let chunks = n_out / pr;
            let rows = tokens * chunks;
            if !rows.is_power_of_two() || rows < 16 {
                continue;
            }
            let score = (pr as isize - target as isize).abs();
            if best.map(|(s, _)| score < s).unwrap_or(true) {
                best = Some((score, Self { per_row: pr, chunks, log_rows: rows.ilog2(),
                                           lead: 0 }));
            }
        }
        best.map(|(_, l)| l)
    }
    /// row -> (token, chunk); tokens and chunks descend
    pub fn at(&self, tokens: usize, j: usize) -> (usize, usize) {
        (tokens - 1 - j / self.chunks, self.chunks - 1 - j % self.chunks)
    }
}
