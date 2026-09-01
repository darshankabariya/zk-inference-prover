//! Trace layouts for `insts` attention heads sharing one column set.
//!
//! B5 proved one head and left the structural problem to B6: one head is
//! ~1,100-1,600 committed columns and 600 KiB-1 MiB of proof, and B1's model has
//! 24 x 14 = 336 head-instances, so per-head instantiation is a 200-340 MB proof --
//! two orders over a single-digit MiB proof.  Their proofs *require* head- and
//! layer-packed columns.
//!
//! The packing is cheaper than it looks, and that is B6's first result: it needs
//! **no extra committed or interaction column at all.**  Extend each region's
//! reduction axis over `(instance, group)` instead of `group` and the existing
//! Horner does the aggregation for free -- `hs = sum_{gk} r^{gk} zs_{gk}` with
//! `gk = i*C + c` is exactly `sum_i r^{C i} (sum_c r^c zs)`, i.e. every instance's
//! claim folded by a public power of the *same* challenge.  No extra challenge, no
//! accumulator column, and the verifier recovers per-instance soundness from
//! Schwartz-Zippel on a polynomial of degree `insts*C` instead of `C`.
//!
//! What the packing does need is that the regions whose outer accumulator is a
//! plain *sum* rather than a Horner (`acc` in the channel region, `av` and `sn` in
//! the grid) pick up the same per-instance weight, so the two sides of each
//! identity stay comparable.  That is one preprocessed selector pair per region and
//! zero fresh cells -- see `attn_air.rs`.
//!
//! `insts` must be a power of two, because the grid's token axis has to tile
//! exactly (B5's assertion, and the plan's warning): `chunks = rows / groups` is a
//! power of two only when `groups = insts * C` is.  A real `L*nh` therefore
//! decomposes into `popcount(L*nh)` bands -- 3 for B1's 336, 5 for the 27B proxy's
//! 1984 -- all sharing the same column widths, so the column count stays
//! proportional to `popcount(L*nh)` rather than to `L*nh`.
//!
//! The block is two Freivalds instances sharing one grid.  Writing `S` for the
//! query tokens a proof covers, `C` for the context and `hd` for the head width:
//!
//!   Q.K^T   score[t, c] = sum_j qQ[t, j] qK[c, j]     reduces over the channel j
//!   A.V     out[t, j]   = sum_c n[t, c] qV[c, j]      reduces over the key c
//!
//! so the two inner products want *different* row axes, and the (t, c) grid --
//! the only thing on the block's dominant multiplicity rung -- can only live in
//! one of them.  It lives in the key-major one, because that is the axis A.V
//! reduces over and the axis whose length is the context:
//!
//!   `LayoutChan`  row (channel j, chunk)   pk keys + pq queries    Q.K^T's inner sum
//!   `LayoutKey`   row (key c, chunk)       q scores + q numerators + pv values
//!   `LayoutOut`   row (channel j, chunk)   q A.V outputs           A.V's claimed side
//!
//! The price of that choice is that the softmax's row maximum `m_t` -- a
//! per-*token* value -- is not available in a key-major row.  B5's answer is to
//! commit `d[t, c] = m_t - score[t, c]` instead of the score and let the Q.K^T
//! identity absorb the shift:
//!
//!   sum_{t,c} r^c s^t score[t,c] = (sum_c r^c)(sum_t s^t m_t) - sum_{t,c} r^c s^t d[t,c]
//!
//! which costs one accumulator over `S` values, not one column over `S x C`.
//!
//! Every region is chunked over consecutive rows and closed by a Horner
//! recurrence, exactly as B3: a `[-1, 0]` mask, degree <= 3, blowup 2.

use serde_json::{Value, json};

pub const MIN_LOG_ROWS: u32 = 4; // SIMD lane floor (N_LANES = 16)
pub const MAX_LOG_ROWS: u32 = 26;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// query tokens covered by the proof
    pub s: usize,
    /// context keys (the committed KV cache)
    pub c: usize,
    /// head width
    pub hd: usize,
    /// head instances sharing one column set; a power of two (see the module doc)
    pub insts: usize,
}

impl Shape {
    pub fn label(&self) -> String {
        format!("S{}xC{}xhd{}xI{}", self.s, self.c, self.hd, self.insts)
    }
    /// entries of the (t, c) grid, summed over instances -- the dominant rung
    pub fn scores(&self) -> usize {
        self.s * self.c * self.insts
    }
    /// keys on the packed reduction axis
    pub fn keys(&self) -> usize {
        self.c * self.insts
    }
    /// channels on the packed reduction axis
    pub fn chans(&self) -> usize {
        self.hd * self.insts
    }
    /// query tokens on the packed per-token axis
    pub fn toks(&self) -> usize {
        self.s * self.insts
    }
    /// One head's shape, for the numbers that are per instance.
    pub fn one(&self) -> Self {
        Self { insts: 1, ..*self }
    }
}

/// Rows grouped by `groups`, `chunks` rows each, padded up to a power of two with
/// inert leading rows whose selectors are all zero.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub log_rows: u32,
    pub rows: usize,
    pub groups: usize,
    pub chunks: usize,
    pub lead: usize,
    /// group index counts down with the row index (needed when a Horner over
    /// groups has to assign `x^g` to group `g`)
    pub descending: bool,
}

impl Grid {
    pub fn new(groups: usize, log_rows: u32, cap: usize, descending: bool) -> Option<Self> {
        let rows = 1usize << log_rows;
        if rows < groups {
            return None;
        }
        // Open item 1: a `c = 0` band (block 0 has no prefix) reaches here with
        // `groups = c * insts = 0`, slips past the guard above and divides by
        // zero.  Block 0 is a solo corner and must never build a prefix band at
        // all; this says so rather than panicking in arithmetic.
        assert!(groups > 0, "a c = 0 band has no key grid: block 0 is the solo                              corner and builds no prefix band (open item 1)");
        let chunks = (rows / groups).min(cap.max(1));
        Some(Self { log_rows, rows, groups, chunks, lead: rows - groups * chunks, descending })
    }

    /// `(group, chunk)` of a row, or `None` for an inert row.  Chunks always
    /// count down so a Horner walking the rows in order leaves the group's total
    /// on its last row.
    pub fn row(&self, j: usize) -> Option<(usize, usize)> {
        if j < self.lead {
            return None;
        }
        let idx = j - self.lead;
        let g = idx / self.chunks;
        let g = if self.descending { self.groups - 1 - g } else { g };
        Some((g, self.chunks - 1 - (idx % self.chunks)))
    }

    pub fn json(&self) -> Value {
        json!({"log_rows": self.log_rows, "rows": self.rows, "groups": self.groups,
               "chunks": self.chunks, "lead_rows": self.lead,
               "row_padding_frac": self.lead as f64 / self.rows as f64})
    }
}

/// Q.K^T's inner-product region: one row group per channel.
#[derive(Clone, Copy, Debug)]
pub struct LayoutChan {
    pub g: Grid,
    /// keys per row (the committed cache, weighted by r^c)
    pub pk: usize,
    /// queries per row (weighted by s^t)
    pub pq: usize,
}

impl LayoutChan {
    pub fn new(sh: &Shape, log_rows: u32) -> Option<Self> {
        // one row group per (instance, channel); the key and query chunks stay
        // per-instance, because `u` and `v` are inner products of one head
        // descending, so an instance's own channels come first and the r^C fold
        // assigns instance i the same weight the grid's r-Horner gives it
        let g = Grid::new(sh.chans(), log_rows, sh.c.min(sh.s), true)?;
        Some(Self { g, pk: sh.c.div_ceil(g.chunks), pq: sh.s.div_ceil(g.chunks) })
    }
    /// the layout whose *keys per row* is closest to `target`
    pub fn choose(sh: &Shape, target: usize) -> Self {
        pick(MIN_LOG_ROWS..=MAX_LOG_ROWS, |lr| Self::new(sh, lr), |l| l.pk, target, |l| l.g.rows)
    }
    pub fn width(&self) -> usize {
        self.pk + self.pq
    }
    pub fn json(&self) -> Value {
        let mut v = self.g.json();
        v["keys_per_row"] = json!(self.pk);
        v["queries_per_row"] = json!(self.pq);
        v["width"] = json!(self.width());
        v
    }
}

/// The (t, c) grid: one row group per key.
#[derive(Clone, Copy, Debug)]
pub struct LayoutKey {
    pub g: Grid,
    /// scores (and numerators) per row, weighted by s^t
    pub q: usize,
    /// V values per row, weighted by rv^j
    pub pv: usize,
}

impl LayoutKey {
    pub fn new(sh: &Shape, log_rows: u32) -> Option<Self> {
        // one row group per (instance, key): the packed axis whose length is
        // `insts * C`, so the r-Horner folds instances for free
        let g = Grid::new(sh.keys(), log_rows, sh.s.min(sh.hd), true)?;
        // The token axis must tile exactly -- `attn_trace.rs`'s
        // "grid layout leaves padded token slots" assert is `q * chunks == S`,
        // and `q = S.div_ceil(chunks)`, so it is exactly `chunks | S`.  Before
        // open item 10's fix `pick` never checked this and returned layouts
        // that could not build: 22 of 32 blocks at C=16384, 4 of 8 at C=4096.
        if sh.s % g.chunks != 0 {
            return None;
        }
        Some(Self { g, q: sh.s.div_ceil(g.chunks), pv: sh.hd.div_ceil(g.chunks) })
    }
    /// the layout whose *scores per row* is closest to `target`
    pub fn choose(sh: &Shape, target: usize) -> Self {
        // Score on `chunks`, not on `q`.  With the tiling filter above, scoring
        // on `q` lands an odd-prefix block on `chunks = cap`, a 64x taller grid
        // (~2.8x the cells) -- the filter can only remove candidates, so the
        // nearest surviving `q` is far away.  `chunks ~ S / q`, so asking for
        // `S.div_ceil(target)` chunks asks for the same layout.
        //
        // Digest safety, checked by mirroring both rules over every shape that
        // builds today (S in {16,32,48,64,128,256,512,1024} x hd x insts x c x
        // target): the two rules agree everywhere EXCEPT `(S=48, target 8|16)`
        // and `(S=64, target=48)`, where `q` and `chunks` are reciprocal rather
        // than linear and the tie-break lands differently.  No invocation in
        // the tree uses those: the shapes actually run are S=512 (spr 64),
        // S=240, S=48 (spr 32), S=32 (spr 64) and S=16 (spr 8), and all of them
        // move nothing.  Re-check this before introducing a new `--s`/`--scores-
        // per-row` pair where `target` does not divide `S`.
        pick(MIN_LOG_ROWS..=MAX_LOG_ROWS, |lr| Self::new(sh, lr),
             |l| l.g.chunks, sh.s.div_ceil(target), |l| l.g.rows)
    }
    pub fn json(&self) -> Value {
        let mut v = self.g.json();
        v["scores_per_row"] = json!(self.q);
        v["values_per_row"] = json!(self.pv);
        v
    }
}

/// A.V's claimed side: one row group per output channel, in descending order so a
/// Horner assigns `rv^j`.
#[derive(Clone, Copy, Debug)]
pub struct LayoutOut {
    pub g: Grid,
    pub q: usize,
}

impl LayoutOut {
    pub fn new(sh: &Shape, log_rows: u32) -> Option<Self> {
        // one row group per (instance, output channel), descending, so the rv-Horner
        // assigns `rv^{j + hd i}` and folds instances by `rv^hd`
        let g = Grid::new(sh.chans(), log_rows, sh.s, true)?;
        // the out region's twin of the same assert (open item 10)
        if sh.s % g.chunks != 0 {
            return None;
        }
        Some(Self { g, q: sh.s.div_ceil(g.chunks) })
    }
    pub fn choose(sh: &Shape, target: usize) -> Self {
        pick(MIN_LOG_ROWS..=MAX_LOG_ROWS, |lr| Self::new(sh, lr),
             |l| l.g.chunks, sh.s.div_ceil(target), |l| l.g.rows)
    }
    pub fn json(&self) -> Value {
        let mut v = self.g.json();
        v["values_per_row"] = json!(self.q);
        v
    }
}

fn pick<T: Copy>(
    range: std::ops::RangeInclusive<u32>,
    make: impl Fn(u32) -> Option<T>,
    key: impl Fn(&T) -> usize,
    target: usize,
    rows: impl Fn(&T) -> usize,
) -> T {
    let mut best: Option<T> = None;
    for lr in range {
        let Some(l) = make(lr) else { continue };
        let score = (key(&l) as isize - target as isize).abs();
        let better = match &best {
            None => true,
            Some(b) => {
                let bs = (key(b) as isize - target as isize).abs();
                score < bs || (score == bs && rows(&l) < rows(b))
            }
        };
        if better {
            best = Some(l);
        }
    }
    best.expect("no valid layout")
}

/// Cell accounting for one head's proof, split the way the cost model needs it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cells {
    pub preprocessed: usize,
    pub base: usize,
    pub interaction: usize,
    /// everything on the (t, c) grid: the alpha term
    pub grid: usize,
    /// the committed KV cache (K and V), which is per proof, not per token
    pub kv: usize,
    /// LogUp interaction columns of the exp lookups
    pub exp: usize,
    /// LogUp interaction columns of the int range checks
    pub range: usize,
}

impl Cells {
    pub fn fresh(&self) -> usize {
        self.base + self.interaction
    }
    pub fn total(&self) -> usize {
        self.preprocessed + self.fresh()
    }
    pub fn json(&self, sh: &Shape) -> Value {
        json!({
            "preprocessed": self.preprocessed, "base": self.base,
            "interaction": self.interaction, "fresh": self.fresh(), "total": self.total(),
            "grid": self.grid, "kv_cache": self.kv, "exp_logup": self.exp,
            "range_logup": self.range,
            "fresh_per_token": self.fresh() as f64 / sh.s as f64,
            // the acceptance number: cells per (head, key) per decoded token
            "alpha": self.fresh() as f64 / sh.scores() as f64,
            "alpha_grid_only": self.grid as f64 / sh.scores() as f64,
            "instances": sh.insts,
        })
    }
}
