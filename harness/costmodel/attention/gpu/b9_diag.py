"""B9 step 1: attribute `off_table` to the table that lost the row.

`off_table` is one counter over every lookup table a composed layer builds, so
"honest composed witness has off-table values" says a witness left *a* table and
nothing about which one -- and the two candidates (the dequant's bf16 output
exponent, the attention-output fold's quotient and remainder) want opposite
fixes.  This records, per table, how many rows were lost and the range of the
values that lost them, and prints it beside the count.

Runs after `b9_patch.py`.  Asserted and idempotent.

    python3 b9_diag.py <dst>
"""
import sys
from pathlib import Path

DST = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/b7/b65_gpu")
SRC = DST / "src"


def sub(path, old, new, what, count=1, marker=None):
    p = SRC / path
    s = p.read_text()
    probe = marker if marker is not None else new
    if probe in s:
        print(f"  {path}: {what} -- already applied")
        return
    n = s.count(old)
    assert n == count, f"{path}: {what}: expected {count} of the anchor, found {n}"
    p.write_text(s.replace(old, new))
    print(f"  {path}: {what}")


print(f"b9_diag.py -> {DST}")

sub("layer.rs",
    """    pub off_table: usize,
}""",
    """    pub off_table: usize,
    /// B9: which table lost a row, how many, and the range of the values that
    /// did -- `off_table` alone cannot tell the two failure modes apart
    pub off_kinds: std::collections::BTreeMap<String, (usize, i64, i64)>,
}""",
    "Tables carries the attribution")

sub("layer.rs",
    """            off_table: 0,
        }""",
    """            off_table: 0,
            off_kinds: Default::default(),
        }""",
    "Tables::new initialises it")

sub("layer.rs",
    """    pub fn bump_r(&mut self, bits: u32, v: i64) {
        let t = self.ridx(bits);
        bump(&mut self.m_r[t], v, &mut self.off_table);
    }""",
    """    pub fn bump_r(&mut self, bits: u32, v: i64) {
        let t = self.ridx(bits);
        bump(&mut self.m_r[t], v, &mut self.off_table);
    }
    pub fn note_off(&mut self, kind: &str, v: i64, n: usize) {
        let e = self.off_kinds.entry(kind.to_string()).or_insert((0, i64::MAX, i64::MIN));
        e.0 += n;
        e.1 = e.1.min(v);
        e.2 = e.2.max(v);
    }
    /// `bump_r` with attribution
    pub fn bump_rk(&mut self, bits: u32, v: i64, kind: &str) {
        let t = self.ridx(bits);
        let before = self.off_table;
        bump(&mut self.m_r[t], v, &mut self.off_table);
        if self.off_table != before {
            self.note_off(kind, v, 1);
        }
    }
    /// the (q, nz) pair table, with attribution
    pub fn bump_qnzk(&mut self, v: i64, kind: &str) {
        let before = self.off_table;
        bump(&mut self.m_qnz, v, &mut self.off_table);
        if self.off_table != before {
            self.note_off(kind, v, 1);
        }
    }""",
    "attributed bump helpers")

# the dequant region's bf16 output exponent, merged from the parallel blocks
sub("layer.rs",
    """            } else {
                tables.off_table += *n as usize;
            }
        }
        ref_bad += b.ref_bad;""",
    """            } else {
                tables.off_table += *n as usize;
                tables.note_off("deq_eo", *v, *n as usize);
            }
        }
        ref_bad += b.ref_bad;""",
    "the dequant's output exponent")

# the attention-output fold
sub("layer.rs",
    """                    bump(&mut tables.m_qnz, 0, &mut tables.off_table);
                    tables.bump_r(rem_bits, 1);              // d
                    tables.bump_r(rem_bits, 0);              // dp = 2*1 - 1 - 1""",
    """                    tables.bump_qnzk(0, "ao_qnz_lead");
                    tables.bump_rk(rem_bits, 1, "ao_rem_lead");    // d
                    tables.bump_rk(rem_bits, 0, "ao_rem_lead");    // dp = 2*1 - 1 - 1""",
    "the fold's inert lead row")

sub("layer.rs",
    """                    bump(&mut tables.m_qnz, qm, &mut tables.off_table);
                    tables.bump_r(rem_bits, d);
                    tables.bump_r(rem_bits, 2 * denp - 1 - d);""",
    """                    tables.bump_qnzk(qm, "ao_qnz");
                    tables.bump_rk(rem_bits, d, "ao_rem_d");
                    tables.bump_rk(rem_bits, 2 * denp - 1 - d, "ao_rem_dp");""",
    "the fold's quotient and remainder")


# ---------------------------------------------------------------------------
# the attention grid's own tables: `commit_base` takes a bare `&mut usize`, so
# the attribution rides on atomics rather than on a threaded map
# ---------------------------------------------------------------------------
sub("attn_trace.rs",
    """fn bump_slice(m: &mut [u32], v: i64, off: &mut usize) {""",
    """pub static EXP_OFF_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static EXP_OFF_MIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MAX);
pub static EXP_OFF_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
pub static EXP_KEY_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub static EXP_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn bump_slice(m: &mut [u32], v: i64, off: &mut usize) {""",
    "grid attribution statics")

sub("attn_trace.rs",
    """            let key = grid_at(j, i).map(|g| a.table_key(g)).unwrap_or(0);
            bump_slice(mult_exp, key, off_table);""",
    """            let key = grid_at(j, i).map(|g| a.table_key(g)).unwrap_or(0);
            {
                use std::sync::atomic::Ordering::Relaxed;
                EXP_KEY_MAX.fetch_max(key, Relaxed);
                EXP_ROWS.store(mult_exp.len(), Relaxed);
                let before = *off_table;
                bump_slice(mult_exp, key, off_table);
                if *off_table != before {
                    EXP_OFF_N.fetch_add(1, Relaxed);
                    EXP_OFF_MIN.fetch_min(key, Relaxed);
                    EXP_OFF_MAX.fetch_max(key, Relaxed);
                }
            }""",
    "the exp table's key")

sub("compose.rs",
    """        off_table: tables.off_table,""",
    """        off_table: {
            if tables.off_table > 0 {
                use std::sync::atomic::Ordering::Relaxed;
                eprintln!("off_table {} by table: {:?}", tables.off_table, tables.off_kinds);
                eprintln!("exp table: {} rows, max key {}, off {} in [{}, {}]",
                          crate::attn_trace::EXP_ROWS.load(Relaxed),
                          crate::attn_trace::EXP_KEY_MAX.load(Relaxed),
                          crate::attn_trace::EXP_OFF_N.load(Relaxed),
                          crate::attn_trace::EXP_OFF_MIN.load(Relaxed),
                          crate::attn_trace::EXP_OFF_MAX.load(Relaxed));
            }
            tables.off_table
        },""",
    "print the attribution")

print("b9_diag.py done")
