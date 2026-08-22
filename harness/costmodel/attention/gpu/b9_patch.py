"""B9 step 1: the two things that stop a real layer proving, and the two knobs
that stop a real *model*'s layers sharing one shape.

Runs after `m5_patch.py` on a tree the B7 chain has already built, so B7b's
binaries take B9's fix without a re-port. The same edits are in the source of
truth (`experiments/b65/bench/src`); this script is how they reach the derived
trees.

  1. `attn_witness.rs`: `Attn::build` takes a `master` exponent. Without it the
     master table's exponent is the largest `frac` over the band's *own* heads,
     so which table a band builds depends on how the heads were split into
     bands -- and `Layer::load`'s "every band's table is identical" then holds
     only where every band happens to contain a head at the layer-wide maximum.
     That is 4 of 24 Qwen-0.5B layers, and 0 of 32 Llama-3.1-8B ones.
  2. `layer.rs`: `Layer::load` reads the layer-wide maximum out of the manifest
     (`frac_per_head`, which the forward pass already writes) and hands it to
     every band. Each band then scales its shifts up to it and reads one table
     of `d_bits + spread` rows -- `master_table_bits` in the manifest.
  3. `main.rs`: `--ao-rem-bits`, so the attention-output fold's remainder range
     check can be pinned at a model-wide worst case. With `--kmax` (which
     already existed) and `--table-bits`, every layer of a model is then one
     shape and the chunk loop takes them all.

Asserted and idempotent, like every patch in this series.

    python3 b9_patch.py <dst>
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


print(f"b9_patch.py -> {DST}")

# ---------------------------------------------------------------------------
# 1. attn_witness.rs: the master exponent is an argument, not the band's own max
# ---------------------------------------------------------------------------
sub("attn_witness.rs",
    """        // one master table at the largest exponent; every other instance's shift is
        // scaled up to it by a public power of two, so the table is shared
        let frac = frac_per_inst.iter().cloned().fold(f64::MIN, f64::max);""",
    """        // one master table at the largest exponent; every other instance's shift is
        // scaled up to it by a public power of two, so the table is shared.
        //
        // B9 step 1a: that exponent has to be a *layer*-wide constant, not a
        // band-wide one.  Taking the largest `frac` over the band's own heads makes
        // the table a function of how the heads were split into bands, so
        // `Layer::load`'s "every band's table is identical" holds only where every
        // band happens to contain a head at the layer-wide maximum -- 4 of 24
        // Qwen-0.5B layers, and by B8's `frac_per_head`, 0 of 32 Llama-3.1-8B ones.
        // With `master` supplied every band scales its shifts up to the same
        // exponent and reads one table of `d_bits + spread` rows (B6 section 6).
        let natural = frac_per_inst.iter().cloned().fold(f64::MIN, f64::max);
        let frac = match master {
            Some(m) => {
                assert!(m >= natural - 1e-9,
                        "master exponent {m} is below this band's own maximum {natural}");
                m
            }
            None => natural,
        };""",
    "the master exponent is an argument")

sub("attn_witness.rs",
    """             nmax: u32, log_table: u32, frac_in: Option<Vec<f64>>, source: String,
             bad: Option<Corrupt>) -> Self {""",
    """             nmax: u32, log_table: u32, frac_in: Option<Vec<f64>>,
             master: Option<f64>, source: String,
             bad: Option<Corrupt>) -> Self {""",
    "build() takes `master`")

for old, new, what in [
    ("""        Self::build(sh, variant, qq, kk, vv, None, q_bits, k_bits, nmax, log_table, None,
                    format!("synthetic(seed={seed})"), None)""",
     """        Self::build(sh, variant, qq, kk, vv, None, q_bits, k_bits, nmax, log_table, None,
                    None, format!("synthetic(seed={seed})"), None)""",
     "synthetic()"),
    ("""        Ok(Self::build(sh, variant, qq, kk, vv, torch_num, q_bits, k_bits, nmax, log_table,
                       fracs,
                       format!(""",
     """        Ok(Self::build(sh, variant, qq, kk, vv, torch_num, q_bits, k_bits, nmax, log_table,
                       fracs, master,
                       format!(""",
     "load()"),
    ("""        Self::build(sh, variant, h.qq, h.kk, h.vv, None, q_bits, k_bits, nmax, log_table,
                    Some(vec![frac; sh.insts]), h.source, None)""",
     """        Self::build(sh, variant, h.qq, h.kk, h.vv, None, q_bits, k_bits, nmax, log_table,
                    Some(vec![frac; sh.insts]), None, h.source, None)""",
     "synthetic_shared()"),
    ("""        Self::build(sh, variant, h.qq, h.kk, h.vv, None, h.q_bits, h.k_bits, h.nmax,
                    h.log_table, Some(h.frac_per_inst), h.source, Some(bad))""",
     """        Self::build(sh, variant, h.qq, h.kk, h.vv, None, h.q_bits, h.k_bits, h.nmax,
                    h.log_table, Some(h.frac_per_inst), None, h.source, Some(bad))""",
     "corrupted()"),
]:
    sub("attn_witness.rs", old, new, f"{what} passes `master`")

sub("attn_witness.rs",
    """    pub fn load(dir: &str, head: usize, sh: Shape, variant: Variant, nmax: u32,
                log_table: u32) -> Result<Self, String> {""",
    """    /// `master`: the layer-wide exponent every band's table is pinned to (B9
    /// step 1a).  `None` keeps the band's own maximum, which is what B6.5 did and
    /// what makes a layer's bands disagree.
    #[allow(clippy::too_many_arguments)]
    pub fn load(dir: &str, head: usize, sh: Shape, variant: Variant, nmax: u32,
                log_table: u32, master: Option<f64>) -> Result<Self, String> {""",
    "load() takes `master`")

sub("attn_witness.rs",
    """            match Self::load(d, cfg.head, sh, variant, cfg.nmax, cfg.log_table) {""",
    """            match Self::load(d, cfg.head, sh, variant, cfg.nmax, cfg.log_table, None) {""",
    "get() keeps the band-wide exponent")

# ---------------------------------------------------------------------------
# 2. layer.rs: the layer-wide maximum, read out of the manifest
# ---------------------------------------------------------------------------
sub("layer.rs",
    """        let mut bands = vec![];
        let mut head_offsets = vec![];
        let mut off = 0usize;
        for insts in band_sizes {
            let sh = Shape { s, c, hd, insts: *insts };
            let src = Src { head: off, ..cfg.clone() };
            bands.push(Attn::load(dir, off, sh, variant, src.nmax, src.log_table)?);
            head_offsets.push(off);
            off += insts;
        }
        // one master table serves every band: same rows, same contents
        for b in &bands[1..] {
            assert_eq!(b.log_table, bands[0].log_table, "bands disagree on table rows");
            assert_eq!(b.table, bands[0].table, "bands disagree on the exp table");
            assert_eq!(b.frac, bands[0].frac, "bands disagree on the master exponent");
        }
""",
    """        // B9 step 1a: the master exponent is layer-wide.  `Attn::build` takes the
        // largest `frac` over the band's *own* heads, so which table a band builds
        // depends on how the heads were split -- and the three assertions below
        // then hold only where every band happens to contain a head at the
        // layer-wide maximum (4 of 24 Qwen-0.5B layers; 0 of 32 Llama-3.1-8B ones,
        // B8.md section 2 item 3).  The forward pass already writes every head's
        // exponent into the manifest, so the layer-wide maximum is read, not
        // derived, and handed to every band: each then scales its shifts up to it
        // and reads one table of `d_bits + spread` rows.  The rows are what the
        // sharing costs -- 2^`master_table_bits`, up to 8x a spread-0 layer's --
        // and `off_table` reports any key the chosen table cannot hold.
        let master: Option<f64> = man["frac_shared_per_head"].as_bool().unwrap_or(false)
            .then(|| man["frac_per_head"].as_array().map(|a| {
                a.iter().take(nh).filter_map(|v| v.as_f64()).fold(f64::MIN, f64::max)
            }))
            .flatten()
            .filter(|f| *f > f64::MIN);
        let mut bands = vec![];
        let mut head_offsets = vec![];
        let mut off = 0usize;
        for insts in band_sizes {
            let sh = Shape { s, c, hd, insts: *insts };
            let src = Src { head: off, ..cfg.clone() };
            bands.push(Attn::load(dir, off, sh, variant, src.nmax, src.log_table, master)?);
            head_offsets.push(off);
            off += insts;
        }
        // one master table serves every band: same rows, same contents.  With a
        // layer-wide `master` this holds by construction; without one (a dump whose
        // Q scale is not per head) it is the old precondition, and the message
        // names the fix.
        for b in &bands[1..] {
            assert_eq!(b.log_table, bands[0].log_table, "bands disagree on table rows");
            assert_eq!(b.table, bands[0].table,
                       "bands disagree on the exp table (no layer-wide master exponent: \\
                        the dump has frac_shared_per_head = false)");
            assert_eq!(b.frac, bands[0].frac, "bands disagree on the master exponent");
        }
""",
    "the layer-wide master exponent")

# ---------------------------------------------------------------------------
# 3. main.rs: `--ao-rem-bits`, so a model's layers can be pinned to one shape
# ---------------------------------------------------------------------------
sub("main.rs",
    """        ao_target: arg_scalar(args, "--ao-per-row", 16usize),
        ao_q_bits: 7,
        ao_rem_bits: l.ao.rem_bits(),""",
    """        ao_target: arg_scalar(args, "--ao-per-row", 16usize),
        ao_q_bits: 7,
        // B9 step 1: `ao_rem_bits` and `kmax` are data-derived, so two real layers
        // of one model are two shapes and the chunk loop rejects the pair.  Both
        // are overridable now, so a driver can pin them at the model-wide worst
        // case and prove every layer of a model as one shape -- a few extra table
        // rows against a second emit chain.  Below the witness's own value the
        // range check would just push rows into `off_table`, so that is an error.
        ao_rem_bits: {
            let want = arg_scalar(args, "--ao-rem-bits", l.ao.rem_bits());
            assert!(want >= l.ao.rem_bits(),
                    "--ao-rem-bits {want} is below this witness's {}", l.ao.rem_bits());
            want
        },""",
    "--ao-rem-bits")

print("b9_patch.py done")
