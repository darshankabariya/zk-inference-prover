//! Which matmuls share one check, and the grid it runs over.

use crate::layout::Grid;

/// The Freivalds groups: matmuls sharing one committed input.
/// B10 step 4: `lm_head` is its own group.  It shares no committed input with the
/// layer's matmuls -- its input is the *final* norm's output, not any layer's --
/// so it Freivalds checks alone, over `d` input channels like every other group.
pub const FREIVALDS_GROUPS: [(&str, &[&str]); 5] =
    [("qkv", &["q", "k", "v"]), ("o", &["o"]), ("gateup", &["gate", "up"]),
     ("down", &["down"]), ("head", &["lm_head"])];

/// One Freivalds component's layout: `Grid` over input channels.
#[derive(Clone, Copy, Debug)]
pub struct FreivaldsLayout {
    pub g: Grid,
    pub pk: usize,
    pub pq: usize,
}

impl FreivaldsLayout {
    pub fn choose(n_in: usize, n_out_max: usize, tokens: usize, p_target: usize) -> Self {
        // pick the chunk count whose weights-per-row is closest to the target,
        // preferring fewer padded rows on bindings
        let mut best: Option<(isize, usize, Self)> = None;
        for chunks in 1..=n_out_max {
            let pk = n_out_max.div_ceil(chunks);
            let pq = tokens.div_ceil(chunks);
            let rows = n_in * chunks;
            let log_rows = (usize::BITS - (rows - 1).leading_zeros()).max(4);
            let Some(g) = Grid::new(n_in, log_rows, chunks, true) else { continue };
            if g.chunks != chunks {
                continue;
            }
            let score = (pk as isize - p_target as isize).abs();
            let pad = (1usize << log_rows) - rows;
            let better = match &best {
                None => true,
                Some((s, p, _)) => score < *s || (score == *s && pad < *p),
            };
            if better {
                best = Some((score, pad, Self { g, pk, pq }));
            }
        }
        best.expect("no Freivalds layout").2
    }
}
