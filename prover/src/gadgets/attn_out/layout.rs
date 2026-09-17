//! The attnout component's row map.

/// The attnout component: rows = (head, token, chunk) descending, inert lead first.
#[derive(Clone, Copy, Debug)]
pub struct AoLayout {
    pub per_row: usize,
    pub chunks: usize,
    pub log_rows: u32,
    pub lead: usize,
}

impl AoLayout {
    pub fn choose(nh: usize, s: usize, hd: usize, target: usize) -> Self {
        let mut pr = target.min(hd).max(1);
        while hd % pr != 0 {
            pr -= 1;
        }
        let chunks = hd / pr;
        let active = nh * s * chunks;
        let log_rows = (usize::BITS - (active - 1).leading_zeros()).max(4);
        Self { per_row: pr, chunks, log_rows, lead: (1usize << log_rows) - active }
    }
    /// row -> (head, token, chunk), or None for an inert lead row
    pub fn at(&self, nh: usize, s: usize, j: usize) -> Option<(usize, usize, usize)> {
        if j < self.lead {
            return None;
        }
        let ja = j - self.lead;
        let per_head = s * self.chunks;
        let h = nh - 1 - ja / per_head;
        let rest = ja % per_head;
        Some((h, s - 1 - rest / self.chunks, self.chunks - 1 - rest % self.chunks))
    }
}
